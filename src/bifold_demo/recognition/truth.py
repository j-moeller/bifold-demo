"""What a scene knows about the people in its picture, to score person tracks and identity decisions against.

**For evaluation only**: nothing in the recognition reads it (a real camera has no such thing). It is what the tests
and the measurements of step F2 count with, and what the evaluation (E1) builds on:

* :func:`frame_truth`: for a scene at its current moment, every visible person whose box reaches the picture, with
  who they are (the scene object's ``name`` and its avatar, the identity), their box (the posed skeleton, as selection
  mode draws it), how far away they are, their eyes in the picture and the eye distance of their face turned to the
  camera (the sensor's definition, :func:`~bifold_demo.sensor.requests.frontal_eye_distance`), how their head is
  turned (:func:`~bifold_demo.sensor.service.head_pose`) and how much of their box nearer people cover.
* :func:`match_tracks`: which track follows which person in a frame (the most overlap, at least ``min_iou``).
* :class:`IdSwitches`: identity switches (CLEAR-MOT's IDSW): a person followed by one track and later by another.
* :class:`Outcomes`: what the decisions did per **segment** (one person followed by one track), the visitor's view:
  whether and when a known person's face was first *shown* (turned to the camera, eyes not hidden, big enough) and
  first named correctly, wrong names, "unknown" said to a known person; :meth:`Outcomes.summary` gives the shares
  (named correctly within 2 and 5 seconds of the face being shown, the false-alarm rate per track for strangers, ...).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..detection.assignment import match
from ..detection.tracker import iou

__all__ = ["IdSwitches", "Outcomes", "PersonTruth", "Segment", "frame_truth", "match_tracks", "simulated_detections"]


@dataclass(frozen=True)
class PersonTruth:
    """One person in the picture (see the module docs). ``box``: ``(x0, y0, x1, y1)`` view pixels; ``eyes``: their
    right and left eye in view pixels; ``eye_distance``: view pixels between the eyes of their face turned to the
    camera; ``facing``, ``yaw``, ``pitch``: degrees (:class:`~bifold_demo.sensor.service.Truth`'s definitions);
    ``covered``: the share of their box behind nearer people's boxes (0-1); ``eyes_covered``: whether a nearer
    person's box covers the point between their eyes."""

    name: str
    avatar: str
    box: tuple
    distance: float
    eyes: tuple
    eye_distance: float
    facing: float
    yaw: float
    pitch: float
    covered: float = 0.0
    eyes_covered: bool = False

    @property
    def frontal(self) -> bool:
        """Turned less than 45 degrees from the camera (the research's "yaw under 45")."""
        return abs(self.yaw) < 45.0 and self.facing < 60.0


def frame_truth(scene, size) -> list[PersonTruth]:
    """The people of ``scene`` (as it is now) in a picture of ``size`` = (width, height), nearest first (see the module
    docs). People without eye bones, invisible ones and those whose box misses the picture are left out."""
    from ..selection import person_box
    from ..sensor.requests import frontal_eye_distance
    from ..sensor.service import eye_points, head_pose

    w, h = int(size[0]), int(size[1])
    camera = scene.camera
    clip_from_world = camera.projection_matrix(w / h) @ camera.view_matrix()
    position = np.asarray(camera.current_position, dtype=np.float64)
    found = []
    for person in scene:
        if not getattr(person, "current_visible", False):
            continue
        eyes = eye_points(person)
        if eyes is None:
            continue
        box = person_box(person, clip_from_world, (w, h))
        if box is None or box[2] <= 0 or box[3] <= 0 or box[0] >= w or box[1] >= h:
            continue
        clip = np.c_[eyes, np.ones(2)] @ clip_from_world.T
        if (clip[:, 3] <= 1e-9).any():
            continue
        ndc = clip[:, :2] / clip[:, 3:]
        px = np.c_[(ndc[:, 0] + 1) / 2 * w, (1 - ndc[:, 1]) / 2 * h]
        middle = eyes.mean(axis=0)
        frontal = frontal_eye_distance(clip_from_world, (w, h), camera, middle,
                                       float(np.linalg.norm(eyes[0] - eyes[1])))
        pose = head_pose(person, camera, middle)
        model = getattr(person, "skinned_model", None)
        found.append((float(np.linalg.norm(middle - position)), person.name, str(getattr(model, "name", "")),
                      tuple(float(v) for v in box), tuple(map(tuple, px.tolist())), float(frontal or 0.0), pose))
    found.sort(key=lambda entry: entry[0])
    out = []
    for k, (distance, name, avatar, box, eyes, frontal, pose) in enumerate(found):
        nearer = [entry[3] for entry in found[:k]]
        middle = np.mean(eyes, axis=0)
        eyes_covered = any(b[0] <= middle[0] <= b[2] and b[1] <= middle[1] <= b[3] for b in nearer)
        out.append(PersonTruth(name, avatar, box, distance, eyes, frontal, pose["facing"], pose["yaw"],
                               pose["pitch"], _covered(box, nearer), eyes_covered))
    return out


def _covered(box, others, samples: int = 24) -> float:
    """The share of ``box`` inside any of ``others`` (a grid of ``samples`` x ``samples`` points)."""
    if not others:
        return 0.0
    x0, y0, x1, y1 = box
    xs = np.linspace(x0, x1, samples)
    ys = np.linspace(y0, y1, samples)
    gx, gy = np.meshgrid(xs, ys)
    inside = np.zeros_like(gx, dtype=bool)
    for a in others:
        inside |= (gx >= a[0]) & (gx <= a[2]) & (gy >= a[1]) & (gy <= a[3])
    return float(inside.mean())


def simulated_detections(truth: list[PersonTruth], rng: np.random.Generator, jitter: float = 0.02,
                         size=None) -> tuple[np.ndarray, np.ndarray]:
    """``(boxes, scores)`` a person detector might give for ``truth``: each box moved by ``jitter`` of its height,
    confident when the person is in the open, weak when half covered by nearer people, missed when mostly covered
    (more than 0.75) or smaller than 12 pixels. For tests of the tracker without rendering; cut to the picture
    ``size`` when given."""
    boxes, scores = [], []
    for person in truth:
        height = person.box[3] - person.box[1]
        if height < 12 or person.covered > 0.75:
            continue
        box = np.array(person.box) + rng.normal(0.0, jitter * height, 4)
        if size is not None:
            box = np.clip(box, 0, [size[0], size[1], size[0], size[1]])
        if box[2] - box[0] < 2 or box[3] - box[1] < 2:
            continue
        score = 0.85 - 0.9 * person.covered + rng.normal(0.0, 0.04)
        boxes.append(box)
        scores.append(float(np.clip(score, 0.12, 0.95)))
    return np.array(boxes).reshape(-1, 4), np.array(scores)


def match_tracks(truth: list[PersonTruth], tracks, min_iou: float = 0.5, boxes=None) -> dict[str, int]:
    """``{person name: track id}``: which track follows whom in a frame, by the most overlap of the track's box (or
    ``boxes``, one per track) with the person's, at least ``min_iou`` (each track and person at most once)."""
    tracks = list(tracks)
    if not truth or not tracks:
        return {}
    track_boxes = boxes if boxes is not None else [t.box for t in tracks]
    overlap = iou([p.box for p in truth], track_boxes)
    pairs, _, _ = match(1.0 - overlap, 1.0 - min_iou + 1e-9)
    return {truth[i].name: int(tracks[j].id) for i, j in pairs}


class IdSwitches:
    """CLEAR-MOT's identity switches: each time a person is followed by another track than the one that followed
    them last (gaps in between allowed). They are of two kinds: ``swaps``, where the person was followed by the other
    track at most ``gap`` seconds before (two tracks swapped people, or one jumped to someone else), and ``returns``,
    where the person was out of sight (unfollowed) for longer (a new track after the lost one was given up: by design
    for a tracker that follows boxes alone). ``switches`` = swaps + returns; ``followed``: person-frames matched to a
    track; ``seen``: person-frames in all; ``tracks``: ``{person: [track ids in order]}``."""

    def __init__(self, gap: float = 1.0):
        self.gap = float(gap)
        self.swaps = 0
        self.returns = 0
        self.followed = 0
        self.seen = 0
        self.tracks: dict[str, list[int]] = {}
        self._last: dict[str, tuple[int, float]] = {}

    @property
    def switches(self) -> int:
        return self.swaps + self.returns

    def update(self, truth: list[PersonTruth], matched: dict[str, int], now: float = 0.0) -> int:
        """Count a frame at time ``now``: ``truth`` (the people in it) and ``matched`` (:func:`match_tracks`). Returns
        the switches in this frame."""
        new = 0
        for person in truth:
            self.seen += 1
            track = matched.get(person.name)
            if track is None:
                continue
            self.followed += 1
            last = self._last.get(person.name)
            if last is not None and last[0] != track:
                new += 1
                if now - last[1] <= self.gap:
                    self.swaps += 1
                else:
                    self.returns += 1
            if last is None or last[0] != track:
                self.tracks.setdefault(person.name, []).append(track)
            self._last[person.name] = (track, float(now))
        return new

    @property
    def coverage(self) -> float:
        """The share of person-frames followed by a track."""
        return self.followed / self.seen if self.seen else 0.0


@dataclass
class Segment:
    """One person followed by one track (see :class:`Outcomes`). Times are the frames' (seconds)."""

    person: str
    avatar: str
    track: int
    known: bool
    start: float
    end: float
    shown: float = 0.0  # seconds the face was shown (turned to the camera, eyes not hidden, big enough)
    first_shown: float | None = None
    correct_at: float | None = None  # when it was first named correctly
    wrong: tuple = ()  # wrong names given
    unknown_at: float | None = None  # when it was first called unknown
    named_at: float | None = None  # when it was first named (rightly or not)
    frames: int = 0
    best_eyes: float = 0.0  # the biggest face shown (view pixels between the eyes, turned to the camera)

    def eligible(self, seconds: float) -> bool:
        """Whether the face was shown, at least ``seconds`` in all."""
        return self.first_shown is not None and self.shown >= seconds - 1e-9

    def named_within(self, seconds: float) -> bool:
        """Named correctly at most ``seconds`` after the face was first shown."""
        return self.correct_at is not None and self.first_shown is not None and \
            self.correct_at - self.first_shown <= seconds + 1e-9


class Outcomes:
    """The decisions per segment, against the truth (for evaluation only). ``known``: the identities (avatars) of
    the gallery; a face counts as *shown* when :attr:`PersonTruth.frontal`, its eyes are not covered by a nearer
    person and its eye distance in the view (turned to the camera) is at least ``min_eyes`` pixels."""

    def __init__(self, known, min_eyes: float = 3.0):
        self.known = set(known)
        self.min_eyes = float(min_eyes)
        self.segments: dict[tuple[str, int], Segment] = {}
        self._last: float | None = None

    def update(self, now: float, truth: list[PersonTruth], matched: dict, decisions: dict) -> None:
        """Count a frame at ``now``: ``truth``, ``matched`` (:func:`match_tracks`: person -> track) and
        ``decisions`` (track -> :class:`~.decision.Decision`)."""
        dt = 0.0 if self._last is None else max(0.0, now - self._last)
        self._last = now
        for person in truth:
            track = matched.get(person.name)
            if track is None:
                continue
            key = (person.name, track)
            seg = self.segments.get(key)
            if seg is None:
                seg = self.segments[key] = Segment(person.name, person.avatar, int(track),
                                                   person.avatar in self.known, now, now)
            seg.end, seg.frames = now, seg.frames + 1
            if person.frontal and not person.eyes_covered and person.eye_distance >= self.min_eyes:
                if seg.first_shown is None:
                    seg.first_shown = now
                else:
                    seg.shown += dt
                seg.best_eyes = max(seg.best_eyes, person.eye_distance)
            decision = decisions.get(track)
            if decision is None:
                continue
            if decision.state == "named":
                if seg.named_at is None:
                    seg.named_at = now
                if decision.name == person.avatar:
                    if seg.correct_at is None:
                        seg.correct_at = now
                elif decision.name not in seg.wrong:
                    seg.wrong = (*seg.wrong, decision.name)
            elif decision.state == "unknown" and seg.unknown_at is None:
                seg.unknown_at = now

    def summary(self, shown: float = 1.0, within=(2.0, 5.0)) -> dict:
        """The shares over the segments whose face was shown at least ``shown`` seconds (``eligible``) and over all:

        * known people: ``named_within`` (named correctly within each of ``within`` seconds of the face being first
          shown), ``named`` (at all), the median and 90th percentile ``time_to_name``, ``confused`` (another known
          person's name given), ``unknown`` ("unknown" said);
        * strangers: ``fpir`` (named at all: the false-alarm rate per track), ``called_unknown``; and over every
          stranger segment, eligible or not, ``fpir_all``.
        """
        segs = list(self.segments.values())
        known = [g for g in segs if g.known and g.eligible(shown)]
        strangers = [g for g in segs if not g.known and g.eligible(shown)]
        all_strangers = [g for g in segs if not g.known]
        times = sorted(g.correct_at - g.first_shown for g in known
                       if g.correct_at is not None and g.first_shown is not None and g.correct_at >= g.first_shown)

        def share(items, test):
            return (sum(1 for g in items if test(g)) / len(items)) if items else None

        return {
            "segments": len(segs), "known_eligible": len(known), "strangers_eligible": len(strangers),
            "named_within": {str(t): share(known, lambda g, t=t: g.named_within(t)) for t in within},
            "named": share(known, lambda g: g.correct_at is not None),
            "time_to_name": {"median": float(np.median(times)) if times else None,
                             "p90": float(np.quantile(times, 0.9)) if times else None, "n": len(times)},
            "confused": share(known, lambda g: bool(g.wrong)),
            "unknown": share(known, lambda g: g.unknown_at is not None),
            "fpir": share(strangers, lambda g: g.named_at is not None),
            "called_unknown": share(strangers, lambda g: g.unknown_at is not None),
            "fpir_all": share(all_strangers, lambda g: g.named_at is not None),
            "strangers_all": len(all_strangers),
        }
