"""From person to identity, frame by frame: who the people in the picture are (the demo's ``N``).

:class:`PeopleRecognizer` runs the whole chain on a frame (on a worker thread in the demo; its :meth:`~PeopleRecognizer.
detect_frame` is what :class:`~bifold_demo.detection.worker.AsyncDetector` calls):

1. **People.** The person detector's boxes (YOLO), down to a low score, go to the person tracker
   (:class:`~bifold_demo.detection.tracker.PersonTracker`, ByteTrack): each person keeps a number while they move,
   turn away or are hidden for a moment.
2. **Faces in their head regions** (:mod:`.heads`). Each track's head region is looked at: in a high-resolution
   window of the virtual sensor when one came with the frame for it (the tracker asks for them, below), else in a
   crop of the frame. A frame's crops are limited (``searches``): tracks not decided yet first, then those looked at
   longest ago; faces too small to be found in the frame (:data:`MIN_FRAME_EYES`) wait for a window. Every face found
   is given to at most one track, the one whose head it fits.
3. **Evidence** (:mod:`.evidence`). Each face is aligned (from the window's pixels when it came from one), its
   quality measured and gated; the embedding budget of the profile's model (embeddings a second for the whole system,
   on the frames' clock) goes to the faces the scheduler picks; each embedding joins its track's evidence, after a
   check that it shows the same person as the evidence so far (else the track's old evidence and name go: the track
   jumped to someone else).
4. **Decisions** (:mod:`.decision`): a name, "unknown" or "not decided yet" per track, calibrated per model and face
   size against one target false-alarm rate (:mod:`.calibration`), one identity per track.
5. **Windows for the frames to come.** Tracks not decided yet (or due for a check), not known to be turned away, whose
   face is small in the frame, ask the :class:`~bifold_demo.sensor.WindowService` for a window on their head, tagged
   with the track's number. Its scheduler's ``priority`` hook puts undecided, frontal and large faces first and lets
   tracks take turns (:meth:`PeopleRecognizer.priority`). A webcam's windows are cut from its full-resolution picture;
   the code is the same.

Each frame gives a :class:`PeopleResult`: per track its box, decision and the face found (with landmarks), plus the
person detector's boxes, timings and the embedding rate. Runs are deterministic for the same frames: the tracker and
the decisions read only the frames' times (a scene's own time, or a camera's), and the budget fills on that clock.

Embeddings of people who are not known are kept only as long as their track lives (biometric data).
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from ..detection.tracker import PersonTracker, TrackerSettings
from .calibration import find_calibration, find_cohort
from .decision import NAMED, UNDECIDED, UNKNOWN, Decision, DecisionLog, DecisionSettings, IdentityPolicy
from .evidence import (
    Candidate,
    EmbeddingBudget,
    EmbeddingScheduler,
    FrameEvidence,
    QualityGate,
    TrackEvidence,
    rate_for,
)
from .heads import FaceSighting, HeadModel, assign_faces, face_size, find_faces
from .quality import assess

PERSON_SCORE = 0.1  # the person detector's lowest score the tracker sees (its second pass takes weak boxes)
SEARCHES = 6  # head regions looked at in a frame's own pixels (each about 12 ms with SCRFD here, 3 ms with YuNet)
MIN_FRAME_EYES = 3.0  # pixels between the eyes a face needs in the frame to be looked for there (else: a window)
WINDOW_FACE_PX = 64.0  # pixels between the eyes asked of a window (the embedders' crops have 35)
WINDOW_BELOW = 40.0  # a face this big in the frame needs no window
FRONTAL_FOR_WINDOW = 45.0  # degrees: a track whose last face was turned further asks for no window
RECHECK_WINDOW = 5.0  # seconds: a decided track asks for a window again after this
TURN = 0.05  # the window priority a frame of waiting adds (tracks take turns)

__all__ = ["PeopleRecognizer", "PeopleResult", "Person"]


@dataclass(frozen=True)
class Person:
    """A followed person in a frame: the track's ``id``, its smoothed ``box`` ``(x0, y0, x1, y1)``, the
    ``decision`` about them, the ``face`` found for them in this frame (a :class:`~.heads.FaceSighting`, else
    ``None``) and the ``quality`` of that face (else ``None``); ``embedded``: whether the face was embedded."""

    id: int
    box: tuple
    decision: Decision
    face: FaceSighting | None = None
    quality: object = None
    embedded: bool = False

    @property
    def state(self) -> str:
        return self.decision.state

    @property
    def name(self) -> str | None:
        return self.decision.name


@dataclass(frozen=True)
class PeopleResult:
    """One frame's people (the fields :class:`~bifold_demo.detection.worker.AsyncDetector` needs, and more).

    ``people``: the tracks found in the frame (:class:`Person`); ``detections``: the person detector's boxes (all
    classes, for the demo's ``D``); ``faces``: every face found (also those given to nobody); ``windows``: the windows
    the frame came with; ``timings``: milliseconds per stage; ``rate``: embeddings a second lately; ``model``,
    ``profile``: what embedded them; ``time``: the frame's time on the recognition's clock."""

    people: list
    frame_index: int
    source: str
    frame_size: tuple
    inference_ms: float
    timestamp: float
    time: float = 0.0
    detections: list = field(default_factory=list)
    faces: list = field(default_factory=list)
    windows: int = 0
    timings: dict = field(default_factory=dict)
    rate: float = 0.0
    model: str = ""
    profile: str = ""
    calibrated: bool = True

    def counts(self) -> dict[str, int]:
        """``{named, unknown, undecided}``: how many people in the frame are in each state."""
        out = {NAMED: 0, UNKNOWN: 0, UNDECIDED: 0}
        for person in self.people:
            out[person.state] += 1
        return out


@dataclass
class _Track:
    """What the recognizer keeps per track."""

    evidence: TrackEvidence
    searched: float = -math.inf  # when its head region was last looked at
    face_time: float | None = None  # when a face was last found for it
    yaw: float | None = None  # the last face's yaw
    eyes: float | None = None  # the last face's size in the frame (eye distance turned to the camera)
    embedded: float | None = None  # when it was last embedded
    waiting: str = "no face yet"  # why it has no evidence yet
    resets: int = 0


class PeopleRecognizer:
    """The chain from person to identity (see the module docs) for ``gallery`` (a :class:`~.gallery.Gallery`).

    Args:
        gallery: the known people; its engine (:class:`~.engine.FaceEngine`) finds and embeds the faces.
        detector: the person detector (``detect(image, conf=...)`` giving :class:`~bifold_demo.detection.Detection`
            items); ``None``: :meth:`process` must be given the boxes.
        calibration, cohort: the model's (default: found for it, :func:`~.calibration.find_calibration`).
        settings: the decisions' (:class:`~.decision.DecisionSettings`), e.g. the target false-alarm rate ``fpir``.
        windows: a :class:`~bifold_demo.sensor.WindowService` to ask for windows (``None``: none are asked for; the
            windows that come with frames are used all the same).
        rate: embeddings a second for the whole system (default: the model's, :func:`~.evidence.rate_for`).
        searches: head regions looked at in a frame's own pixels.
        log: a :class:`~.decision.DecisionLog` for every decision (the evaluation's; default: the last 2000 in
            memory).
    """

    def __init__(self, gallery, detector=None, engine=None, calibration=None, cohort=None,
                 settings: DecisionSettings | None = None, tracker: TrackerSettings | None = None, windows=None,
                 head: HeadModel | None = None, gate: QualityGate | None = None, rate: float | None = None,
                 searches: int = SEARCHES, window_face_px: float = WINDOW_FACE_PX, log: DecisionLog | None = None,
                 models_folder=None, scheduler: EmbeddingScheduler | None = None):
        self.gallery = gallery
        self.engine = engine or gallery.engine
        self.detector = detector
        self.model = str(getattr(self.engine, "model_id", "unknown"))
        if calibration is None:
            calibration = find_calibration(self.model, models_folder, getattr(self.engine, "threshold", None))
        if cohort is None and calibration.normalisation != "none":
            cohort = find_cohort(self.model, models_folder)
        self.calibration = calibration
        log = DecisionLog(keep=2000) if log is None else log
        self.policy = IdentityPolicy(gallery, calibration, cohort, settings, log)
        self.tracker = PersonTracker(tracker)
        self.head = head or HeadModel()
        self.gate = gate or QualityGate()
        self.budget = EmbeddingBudget(rate or rate_for(self.model))
        self.scheduler = scheduler or EmbeddingScheduler()
        self.searches = int(searches)
        self.window_face_px = float(window_face_px)
        self._windows = None
        self.windows = windows
        self.profile = getattr(getattr(self.engine, "profile", None), "name", "")
        self.tracks: dict[int, _Track] = {}
        self._source = None
        self._rate: list[float] = []  # times of the latest embeddings
        self._lock = threading.Lock()

    @property
    def log(self) -> DecisionLog | None:
        return self.policy.log

    @property
    def windows(self):
        """The :class:`~bifold_demo.sensor.WindowService` the tracks ask for windows (``None``: none asked for);
        setting one gives its scheduler the tracks' order (:meth:`priority`)."""
        return self._windows

    @windows.setter
    def windows(self, service) -> None:
        if service is self._windows:
            return
        if self._windows is not None:
            self._windows.request([])
        self._windows = service
        if service is not None:
            service.scheduler.priority = self.priority

    def reset(self) -> None:
        """Forget every track (another source, or the scene was reset)."""
        with self._lock:
            self.tracker.reset()
            self.tracks.clear()
            self.policy.tracks.clear()
            self._rate.clear()
            service = self.windows
            if service is not None:
                service.request([])

    # -- the windows' order -----------------------------------------------------------------------------------------

    @staticmethod
    def priority(request, waited) -> float:
        """The window scheduler's hook: the request's own priority (undecided, frontal and large faces higher), plus
        :data:`TURN` a frame waited, so that tracks take turns."""
        return float(request.priority) + TURN * float(waited or 0)

    # -- a frame ---------------------------------------------------------------------------------------------------

    def detect_frame(self, frame) -> PeopleResult:
        """The people of ``frame`` (person detection included; see :meth:`process`)."""
        return self.process(frame)

    def process(self, frame, detections=None) -> PeopleResult:
        """Run the chain on ``frame`` (a :class:`~bifold_demo.sources.base.Frame`, with its ``windows``);
        ``detections``: the person detector's for this frame (default: run :attr:`detector`)."""
        with self._lock:
            return self._process(frame, detections)

    def _process(self, frame, detections) -> PeopleResult:
        started = time.perf_counter()
        timings: dict[str, float] = {}
        if frame.source != self._source:
            if self._source is not None:
                self.tracker.reset()
                self.tracks.clear()
                self.policy.tracks.clear()
            self._source = frame.source
        now = float(frame.time if frame.time is not None else frame.timestamp)
        refresh = getattr(self.gallery, "refresh", None)
        if refresh is not None:
            refresh(wait=False)  # new photos are enrolled on a worker thread meanwhile

        # 1. people
        t0 = time.perf_counter()
        if detections is None:
            detections = self._detect(frame.image) if self.detector is not None else []
        timings["people"] = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        people = [d for d in detections if getattr(d, "label", "person") == "person"]
        boxes = np.array([d.box for d in people], dtype=np.float64).reshape(-1, 4)
        scores = np.array([d.confidence for d in people], dtype=np.float64)
        active = self.tracker.update(boxes, scores, now, frame.index)
        for gone in self.tracker.removed:
            self.tracks.pop(gone.id, None)
            self.policy.forget(gone.id)
        for track in active:
            self.tracks.setdefault(track.id, _Track(TrackEvidence(self.model)))
        timings["track"] = (time.perf_counter() - t0) * 1000.0

        # 2. faces in the head regions
        t0 = time.perf_counter()
        boxes_now = {t.id: t.box for t in active}
        windows = list(getattr(frame, "windows", None) or ())
        windowed = {w.tag for w in windows}
        regions = self._searches(active, windowed, now)
        found = find_faces(self.engine, frame.image, regions, windows) if (regions or windows) else []
        sightings = assign_faces(found, boxes_now, self.head)
        mine = {s.track: s for s in sightings if s.track is not None}
        timings["faces"] = (time.perf_counter() - t0) * 1000.0

        # 3. quality, the gate, the budget, embeddings
        t0 = time.perf_counter()
        measured = self._measure(mine, now)
        timings["quality"] = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        embedded = self._embed(measured, now, frame.index)
        timings["embed"] = (time.perf_counter() - t0) * 1000.0

        # 4. decisions
        t0 = time.perf_counter()
        aggregates, described, waiting = {}, {}, {}
        for track in active:
            info = self.tracks[track.id]
            aggregates[track.id] = info.evidence.aggregate(now)
            described[track.id] = self._describe(info)
            waiting[track.id] = info.waiting
        decisions = self.policy.decide(aggregates, now, frame.index, described, waiting)
        timings["decide"] = (time.perf_counter() - t0) * 1000.0

        # 5. windows for the frames to come
        service = self.windows  # (the demo may take it away meanwhile)
        if service is not None:
            self._request_windows(service, active, decisions, frame.index, frame.size)

        result_people = []
        for track in active:
            sighting = mine.get(track.id)
            quality = measured.get(track.id, (None, None))[1] if track.id in measured else None
            result_people.append(Person(track.id, track.box, decisions[track.id], sighting, quality,
                                        track.id in embedded))
        self._rate = [t for t in self._rate if now - t <= 5.0]
        span = max(1.0, min(5.0, now - (self._rate[0] if self._rate else now)))
        finished = time.perf_counter()
        return PeopleResult(result_people, frame.index, frame.source, frame.size, (finished - started) * 1000.0,
                            finished, now, list(detections), sightings, len(windows), timings, len(self._rate) / span,
                            self.model, self.profile, not self.calibration.is_provisional)

    def _detect(self, image) -> list:
        try:
            return list(self.detector.detect(image, conf=PERSON_SCORE))
        except TypeError:  # a detector without a score of its own to lower
            return list(self.detector.detect(image))

    def _searches(self, active, windowed, now) -> dict:
        """The head regions to look at in the frame's own pixels this frame (see the module docs)."""
        wanted = []
        for track in active:
            if track.id in windowed:
                continue
            info = self.tracks[track.id]
            if self.head.expected(track.box)[1] < MIN_FRAME_EYES:
                info.waiting = "too small to find"
                continue
            memory = self.policy.tracks.get(track.id)
            decided = memory is not None and memory.state in (NAMED, UNKNOWN)
            wanted.append((decided, info.searched, track.id, track.box))
        wanted.sort(key=lambda w: w[:3])
        chosen = {}
        for _decided, _last, track_id, box in wanted[:max(0, self.searches)]:
            self.tracks[track_id].searched = now
            chosen[track_id] = self.head.square(box)
        return chosen

    def _measure(self, mine: dict, now: float) -> dict:
        """``{track: (sighting, quality, aligned, why not)}`` for the faces given to tracks: aligned from the pixels
        they were found in, measured, gated (eDifFIQA-T, where the profile has it, only for the faces the geometry
        lets through)."""
        out = {}
        for track_id, sighting in mine.items():
            info = self.tracks.get(track_id)
            if info is None:
                continue
            aligned = self.engine.align(sighting.image, sighting.local)
            quality = assess(sighting.local, aligned)
            info.face_time, info.yaw = now, quality.yaw
            info.eyes = face_size(sighting.face)
            out[track_id] = (sighting, quality, aligned, self.gate.check(quality, sighting.size, sighting.face.score))
        fiqa = getattr(self.engine, "fiqa", None)
        passed = [k for k, v in out.items() if v[3] is None]
        if fiqa is not None and passed:
            from .align import crops

            scores = fiqa.score(crops([out[k][2] for k in passed]))
            for k, score in zip(passed, scores):
                sighting, quality, aligned, _ = out[k]
                quality = assess(sighting.local, aligned, float(score))
                out[k] = (sighting, quality, aligned, self.gate.check(quality, sighting.size, sighting.face.score))
        for k, (_s, _q, _a, why) in out.items():
            if not self.tracks[k].evidence.count:
                self.tracks[k].waiting = why or "waiting for its turn"  # (the embeddings' budget is shared)
        return out

    def _embed(self, measured: dict, now: float, frame: int) -> set:
        """Embed the faces the scheduler picks within the budget; returns the tracks embedded."""
        candidates = []
        for track_id, (_sighting, quality, _aligned, why) in measured.items():
            if why is not None:
                continue
            info = self.tracks[track_id]
            memory = self.policy.tracks.get(track_id)
            decided = memory is not None and memory.state in (NAMED, UNKNOWN)
            enough = info.evidence.aggregate(now)
            enough = enough is not None and enough.weight >= self.policy.settings.min_weight
            median = info.evidence.median_weight(self.scheduler.median_after) if enough else None
            candidates.append(Candidate(track_id, quality.weight, track_id, decided, info.embedded, median))
        if not candidates:
            return set()
        chosen = self.scheduler.choose(candidates, self.budget.available(now), now)
        if not chosen:
            return set()
        from .align import crops

        picked = [measured[c.track] for c in chosen]
        embedded = self.engine.embedder.embed(crops([p[2] for p in picked]))
        self.budget.take(len(chosen))
        done = set()
        spec = getattr(self.engine, "embedder_spec", None)
        for k, (candidate, (sighting, quality, _aligned, _why)) in enumerate(zip(chosen, picked)):
            info = self.tracks[candidate.track]
            vector = np.asarray(embedded.vectors[k], dtype=np.float64)
            norm = float(embedded.norms[k])
            if spec is not None and getattr(spec, "norm_quality", False):
                quality = assess(sighting.local, picked[k][2], quality.fiqa, norm)
            frame_evidence = FrameEvidence(vector, quality.weight, now, sighting.size, quality.yaw, self.model,
                                           quality, sighting.source, frame)
            before = info.evidence.aggregate(now)
            if before is not None:
                score = self.policy.self_score(vector, before.vector)
                if not self.policy.consistent(score, min(sighting.size, before.size), max(abs(quality.yaw),
                                                                                          before.yaw)):
                    info.evidence.reset(now)  # the track shows someone else now
                    self.policy.reset(candidate.track, now)
                    info.resets += 1
            info.evidence.add(frame_evidence)
            info.embedded = now
            info.waiting = ""
            self._rate.append(now)
            done.add(candidate.track)
        return done

    def _describe(self, info: _Track) -> dict:
        """A track's evidence for the decision log."""
        frames = info.evidence.frames
        return {"times": [round(f.time, 3) for f in frames], "weights": [round(f.weight, 4) for f in frames],
                "sizes": [round(f.size, 2) for f in frames], "yaws": [round(f.yaw, 1) for f in frames],
                "sources": [f.source for f in frames], "frames_seen": [f.frame for f in frames],
                "fiqa": [None if f.quality is None or f.quality.fiqa is None else round(f.quality.fiqa, 3)
                         for f in frames], "resets": info.resets}

    def _request_windows(self, service, active, decisions: dict, frame: int, size) -> None:
        """Ask ``service`` for windows on the heads that need them (see the module docs)."""
        from ..sensor import WindowRequest

        requests = []
        for track in active:
            info = self.tracks[track.id]
            decision = decisions.get(track.id)
            decided = decision is not None and decision.state in (NAMED, UNKNOWN)
            if decided and info.embedded is not None and decision.time - info.embedded < RECHECK_WINDOW:
                continue
            if info.yaw is not None and abs(info.yaw) > FRONTAL_FOR_WINDOW and info.face_time is not None \
                    and decision.time - info.face_time < 2.0:
                continue  # turned away a moment ago: a window would show the back of the head
            eyes = self.head.expected(track.box)[1]
            if info.eyes is not None and decision.time - (info.face_time or -math.inf) < 2.0:
                eyes = info.eyes
            if eyes >= WINDOW_BELOW:
                continue  # big enough in the frame itself
            x0, y0, x1, y1 = track.box
            if x1 <= 0 or y1 <= 0 or x0 >= size[0] or y0 >= size[1]:
                continue
            frontal = 1.0 if info.yaw is None else math.cos(math.radians(min(abs(info.yaw), 89.0)))
            large = min(1.0, eyes * 6.0 / self.window_face_px)  # (a window at level 1 of the 6x sensor)
            priority = (0.0 if decided else 2.0) + frontal + large
            requests.append(WindowRequest(track.id, box=track.box, face_px=self.window_face_px, priority=priority,
                                          frame=frame))
        try:
            service.request(requests)
        except ValueError:  # (a request the service cannot serve: none this frame)
            service.request([])
