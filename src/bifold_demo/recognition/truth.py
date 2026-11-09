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
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..detection.assignment import match
from ..detection.tracker import iou

__all__ = ["IdSwitches", "PersonTruth", "frame_truth", "match_tracks", "simulated_detections"]


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
