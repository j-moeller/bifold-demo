"""Following people from frame to frame: person tracks with stable numbers, from the person detector's boxes.

The face recognition decides per *person track*, not per face box: a person keeps their number while their face
turns away, is too small or is hidden for a moment, so the evidence of who they are gathers on the person (see
:mod:`bifold_demo.recognition.people`). :class:`PersonTracker` is ByteTrack (Zhang et al., ECCV 2022), written here in
numpy so that no AGPL tracker comes with it (the commercial profile's person detector is not Ultralytics either):

* **Motion.** Every track has a Kalman filter on its box (:class:`BoxKalman`, as DeepSORT and ByteTrack have it): the
  box's centre, its width over its height and its height, and how fast each changes, with noise in proportion to the
  height. Frames that come at uneven times (a detector on a worker thread skips frames) are allowed for: a step of
  ``dt`` seconds counts as ``dt * fps`` frames of the motion model.
* **Two passes.** Confident boxes (score at least :attr:`TrackerSettings.high`) are matched first, to the tracks
  followed and to those lost a moment ago, by their overlap with the boxes the filters predict (weighted by the
  boxes' scores); then the weak boxes (down to :attr:`~TrackerSettings.low`) are matched to the followed tracks left
  over. A person half hidden behind someone gets a weak box, which keeps their track going instead of starting a new
  one. The matching is an assignment at the least cost (:func:`~.assignment.match`).
* **Births and deaths.** A confident box nobody claims (at least :attr:`~TrackerSettings.new`) starts a *tentative*
  track, confirmed when the next frame finds it again (else forgotten); a track nothing matches is *lost* and kept
  :attr:`~TrackerSettings.lost_for` seconds (about one) in case the person comes out from behind someone, then removed.
  Its number is never used again.

Deterministic: the same boxes at the same times give the same tracks. :meth:`PersonTracker.update` returns the
confirmed tracks found in the frame; :attr:`Track.box` is the filter's (smoothed) box, :attr:`Track.detection` the
detector's own.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .assignment import match

TENTATIVE, TRACKED, LOST, REMOVED = "tentative", "tracked", "lost", "removed"


@dataclass(frozen=True)
class TrackerSettings:
    """ByteTrack's settings (its defaults, with the lost time in seconds):

    high: boxes at least this confident take part in the first pass (``track_thresh``);
    low: weaker boxes down to this take part in the second (``0.1``); weaker ones are ignored;
    new: a box nobody claims starts a track from this score (``track_thresh + 0.1``);
    first, second, tentative: the most a pair may cost in the first pass, the second and for tentative tracks (one
        minus the overlap, intersection over union; the first pass's overlap multiplied by the box's score);
    lost_for: seconds a lost track is kept;
    fps: the frames a second the motion model counts in.
    """

    high: float = 0.5
    low: float = 0.1
    new: float = 0.6
    first: float = 0.8
    second: float = 0.5
    tentative: float = 0.7
    lost_for: float = 1.0
    fps: float = 30.0


class BoxKalman:
    """A constant-velocity Kalman filter on ``(cx, cy, aspect, height)`` and their rates (DeepSORT's): noise in
    proportion to the box's height, so near and far people are followed alike."""

    POSITION, VELOCITY = 1.0 / 20.0, 1.0 / 160.0  # DeepSORT's weights

    def initiate(self, measurement) -> tuple[np.ndarray, np.ndarray]:
        z = np.asarray(measurement, dtype=np.float64)
        mean = np.r_[z, np.zeros(4)]
        h = z[3]
        std = np.array([2 * self.POSITION * h, 2 * self.POSITION * h, 1e-2, 2 * self.POSITION * h,
                        10 * self.VELOCITY * h, 10 * self.VELOCITY * h, 1e-5, 10 * self.VELOCITY * h])
        return mean, np.diag(std ** 2)

    def predict(self, mean, covariance, steps: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        """The state ``steps`` frames on (a fraction or several frames: the noise grows in proportion)."""
        if steps <= 0:
            return mean.copy(), covariance.copy()
        h = max(mean[3], 1e-3)
        std = np.array([self.POSITION * h, self.POSITION * h, 1e-2, self.POSITION * h,
                        self.VELOCITY * h, self.VELOCITY * h, 1e-5, self.VELOCITY * h])
        motion = np.eye(8)
        motion[:4, 4:] = steps * np.eye(4)
        mean = motion @ mean
        covariance = motion @ covariance @ motion.T + np.diag(std ** 2) * steps
        return mean, covariance

    def update(self, mean, covariance, measurement) -> tuple[np.ndarray, np.ndarray]:
        h = max(mean[3], 1e-3)
        std = np.array([self.POSITION * h, self.POSITION * h, 1e-1, self.POSITION * h])
        projected = covariance[:4, :4] + np.diag(std ** 2)
        gain = np.linalg.solve(projected, covariance[:4, :]).T  # (P H^T) S^-1, S symmetric
        innovation = np.asarray(measurement, dtype=np.float64) - mean[:4]
        return mean + gain @ innovation, covariance - gain @ projected @ gain.T


def to_xyah(box) -> np.ndarray:
    """``(x0, y0, x1, y1)`` to ``(cx, cy, width / height, height)``."""
    x0, y0, x1, y1 = (float(v) for v in box)
    h = max(y1 - y0, 1e-3)
    return np.array([(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / h, h])


def to_box(xyah) -> tuple[float, float, float, float]:
    cx, cy, a, h = (float(v) for v in xyah[:4])
    w = a * h
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


def iou(a, b) -> np.ndarray:
    """Intersection over union of every box of ``a`` (``(n, 4)``, ``x0, y0, x1, y1``) with every box of ``b``."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    w = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    h = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = w * h
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


@dataclass(eq=False)
class Track:
    """A followed person. ``id``: their number (from 1, never reused); ``state``: ``tentative``, ``tracked``,
    ``lost`` or ``removed``; ``detection`` and ``score``: the box (``x0, y0, x1, y1``) and score of the detection that
    matched last; ``start``, ``seen``: the times of the first and the last match; ``hits``: matches so far;
    ``frame``: the frame of the last match; ``lost_since``: when it was last lost (``None``: never)."""

    id: int
    mean: np.ndarray
    covariance: np.ndarray
    detection: tuple
    score: float
    start: float
    seen: float
    frame: int | None = None
    hits: int = 1
    state: str = TENTATIVE
    lost_since: float | None = None
    found_again: int = 0  # times it was lost and found again
    confirmed: bool = False  # found in two frames (or in the first frame of all): never tentative again

    @property
    def box(self) -> tuple[float, float, float, float]:
        """The filter's box ``(x0, y0, x1, y1)``: the detection smoothed (and, while lost, where it should be)."""
        return to_box(self.mean)

    @property
    def velocity(self) -> tuple[float, float]:
        """Pixels a frame the box's centre moves (of the motion model)."""
        return float(self.mean[4]), float(self.mean[5])


class PersonTracker:
    """ByteTrack on person boxes (see the module docs). Not thread safe: one thread updates it."""

    def __init__(self, settings: TrackerSettings | None = None):
        self.settings = settings or TrackerSettings()
        self.kalman = BoxKalman()
        self.tracks: list[Track] = []  # tentative, followed and lost ones (removed ones leave)
        self.removed: list[Track] = []  # the tracks removed by the last update
        self.time: float | None = None
        self.frames = 0
        self._next_id = 1

    def reset(self) -> None:
        """Forget every track (another source); numbers go on from where they were."""
        self.tracks, self.removed, self.time, self.frames = [], [], None, 0

    @property
    def lost(self) -> list[Track]:
        return [t for t in self.tracks if t.state == LOST]

    def active(self) -> list[Track]:
        """The confirmed tracks the last frame found, by number."""
        return sorted((t for t in self.tracks if t.state == TRACKED and t.seen == self.time), key=lambda t: t.id)

    def update(self, boxes, scores, now: float | None = None, frame: int | None = None) -> list[Track]:
        """Follow the people into a frame with the person boxes ``boxes`` (``(n, 4)``, ``x0, y0, x1, y1``, pixels)
        and their ``scores``, at time ``now`` (seconds; default: one frame of :attr:`TrackerSettings.fps` after the
        last). Returns the confirmed tracks found in this frame (:meth:`active`)."""
        s = self.settings
        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        if len(boxes) != len(scores):
            raise ValueError(f"{len(boxes)} boxes and {len(scores)} scores")
        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1]) & np.isfinite(boxes).all(axis=1)
        boxes, scores = boxes[valid], scores[valid]
        if now is None:
            now = (self.time or 0.0) + (1.0 / s.fps if self.time is not None else 0.0)
        now = float(now)
        steps = 0.0 if self.time is None else max(0.0, now - self.time) * s.fps
        first_frame = self.time is None and not self.tracks
        self.time, self.frames = now, self.frames + 1
        for track in self.tracks:  # where everyone should be now
            mean = track.mean.copy()
            if track.state != TRACKED:
                mean[7] = 0.0  # (ByteTrack: a lost track's height stops changing)
            track.mean, track.covariance = self.kalman.predict(mean, track.covariance, steps)
        strong = scores >= s.high
        weak = (scores >= s.low) & ~strong
        high_idx, low_idx = np.flatnonzero(strong), np.flatnonzero(weak)
        followed = [t for t in self.tracks if t.state == TRACKED]
        tentative = [t for t in self.tracks if t.state == TENTATIVE]
        lost = [t for t in self.tracks if t.state == LOST]

        # 1. confident boxes with the followed and the lost tracks (overlap weighted by the box's score)
        pool = followed + lost
        cost = self._cost(pool, boxes[high_idx], scores[high_idx])
        pairs, left_tracks, left_high = match(cost, s.first)
        for k, d in pairs:
            self._found(pool[k], boxes[high_idx[d]], scores[high_idx[d]], now, frame)
        # 2. weak boxes with the followed tracks left over
        still = [pool[k] for k in left_tracks if pool[k].state == TRACKED]
        cost = self._cost(still, boxes[low_idx], None)
        pairs, left_still, _ = match(cost, s.second)
        for k, d in pairs:
            self._found(still[k], boxes[low_idx[d]], scores[low_idx[d]], now, frame)
        for k in left_still:
            still[k].state, still[k].lost_since = LOST, now
        # 3. tentative tracks with the confident boxes left over; the others are forgotten
        rest = high_idx[left_high]
        cost = self._cost(tentative, boxes[rest], scores[rest])
        pairs, left_tentative, left_rest = match(cost, s.tentative)
        for k, d in pairs:
            self._found(tentative[k], boxes[rest[d]], scores[rest[d]], now, frame)
        removed = [tentative[k] for k in left_tentative]
        # 4. new tracks from the confident boxes nobody claimed
        for d in rest[left_rest]:
            if scores[d] >= s.new:
                mean, covariance = self.kalman.initiate(to_xyah(boxes[d]))
                track = Track(self._next_id, mean, covariance, tuple(map(float, boxes[d])), float(scores[d]), now,
                              now, frame, state=TRACKED if first_frame else TENTATIVE, confirmed=first_frame)
                self._next_id += 1
                self.tracks.append(track)
        # 5. lost too long: removed
        for track in self.tracks:
            if track.state == LOST and now - track.seen > s.lost_for:
                removed.append(track)
        for track in removed:
            track.state = REMOVED
        self.removed = removed
        self.tracks = [t for t in self.tracks if t.state != REMOVED]
        self._drop_duplicates()
        return self.active()

    def _cost(self, tracks, boxes, scores) -> np.ndarray:
        """One minus the overlap of the tracks' predicted boxes with ``boxes`` (times the boxes' ``scores``)."""
        if not tracks or not len(boxes):
            return np.zeros((len(tracks), len(boxes)))
        overlap = iou([t.box for t in tracks], boxes)
        if scores is not None:
            overlap = overlap * np.asarray(scores)[None, :]
        return 1.0 - overlap

    def _found(self, track: Track, box, score: float, now: float, frame) -> None:
        track.mean, track.covariance = self.kalman.update(track.mean, track.covariance, to_xyah(box))
        if track.state == LOST:
            track.found_again += 1
        track.state, track.confirmed = TRACKED, True
        track.detection, track.score = tuple(float(v) for v in box), float(score)
        track.seen, track.frame = now, frame
        track.hits += 1

    def _drop_duplicates(self) -> None:
        """A followed track and a lost one on nearly the same box (overlap above 0.85): the younger goes (ByteTrack's
        ``remove_duplicate_stracks``)."""
        followed = [t for t in self.tracks if t.state == TRACKED]
        lost = [t for t in self.tracks if t.state == LOST]
        if not followed or not lost:
            return
        overlap = iou([t.box for t in followed], [t.box for t in lost])
        drop = set()
        for i, j in zip(*np.nonzero(overlap > 0.85)):
            a, b = followed[i], lost[j]
            drop.add(id(b) if a.seen - a.start >= b.seen - b.start else id(a))
        for track in self.tracks:
            if id(track) in drop:
                track.state = REMOVED
                self.removed.append(track)
        self.tracks = [t for t in self.tracks if t.state != REMOVED]
