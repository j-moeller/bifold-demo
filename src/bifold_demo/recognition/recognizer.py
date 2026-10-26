"""Recognising the faces in the video frames: who is who, frame by frame (run on a worker thread by the demo).

A single frame of a small face (30-60 pixels across, as the people in the fair are) gives a noisy embedding. So the
faces are followed from frame to frame (a face is the one of the previous frame its box overlaps most), each followed
face averages the embeddings of its last ``TRACK_LENGTH`` frames, and it is named from that average. A face keeps its
name while the average stays within ``KEEP_MARGIN`` of the threshold for that person (hysteresis), so names do not
flicker; a face not seen for ``FORGET_AFTER`` seconds is forgotten.
"""

from __future__ import annotations

import itertools
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ..sources.base import Frame
from .engine import Face, FaceEngine
from .gallery import Gallery, Match

TRACK_LENGTH = 8  # frames of embeddings a followed face averages
KEEP_MARGIN = 0.06  # a named face keeps its name down to this much below the threshold
MIN_OVERLAP = 0.25  # intersection over union of the boxes for a face to be the one of the previous frame
FORGET_AFTER = 1.0  # seconds


@dataclass(frozen=True)
class Recognized:
    """A face in a frame and who it is (``match.name`` is ``None`` for a stranger); ``track``: the number of the
    followed face it belongs to."""

    face: Face
    match: Match
    track: int = 0

    @property
    def name(self) -> str | None:
        return self.match.name


@dataclass(frozen=True)
class FaceResult:
    """The faces of one frame (the fields :class:`~bifold_demo.detection.worker.AsyncDetector` needs)."""

    faces: list[Recognized]
    frame_index: int
    source: str
    frame_size: tuple[int, int]
    inference_ms: float
    timestamp: float


@dataclass
class _Track:
    number: int
    box: tuple
    seen: float
    vectors: deque = field(default_factory=lambda: deque(maxlen=TRACK_LENGTH))
    name: str | None = None

    def mean(self) -> np.ndarray:
        v = np.sum(self.vectors, axis=0)
        return v / max(np.linalg.norm(v), 1e-12)


def _overlap(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    w = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    h = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = w * h
    return inter / max(aw * ah + bw * bh - inter, 1e-9)


class FaceRecognizer:
    """Finds the faces in a frame and names them from ``gallery`` (which is looked at again for new photos at most
    once a second), following them over frames (module docs). ``detect_frame`` makes it usable with
    :class:`~bifold_demo.detection.worker.AsyncDetector`."""

    def __init__(self, gallery: Gallery, engine: FaceEngine | None = None):
        self.gallery = gallery
        self.engine = engine or gallery.engine
        if gallery._engine is None:
            gallery._engine = self.engine
        self._tracks: list[_Track] = []
        self._numbers = itertools.count(1)
        self._source: str | None = None

    def reset(self) -> None:
        """Forget the followed faces (e.g. the picture changed to another source)."""
        self._tracks = []

    def recognize(self, image, now: float | None = None) -> list[Recognized]:
        now = time.monotonic() if now is None else now
        self.gallery.refresh()
        self._tracks = [t for t in self._tracks if now - t.seen <= FORGET_AFTER]
        found = self.engine.faces(image)
        pairs = sorted(((_overlap(face.box, track.box), i, j) for i, (face, _) in enumerate(found)
                        for j, track in enumerate(self._tracks)), reverse=True)
        taken_faces, taken_tracks, owner = set(), set(), {}
        for overlap, i, j in pairs:
            if overlap < MIN_OVERLAP:
                break
            if i not in taken_faces and j not in taken_tracks:
                taken_faces.add(i)
                taken_tracks.add(j)
                owner[i] = self._tracks[j]
        results = []
        for i, (face, vector) in enumerate(found):
            track = owner.get(i)
            if track is None:
                track = _Track(next(self._numbers), face.box, now)
                self._tracks.append(track)
            track.box, track.seen = face.box, now
            track.vectors.append(vector)
            results.append(Recognized(face, self._name(track), track.number))
        return results

    def _name(self, track: _Track) -> Match:
        mean = track.mean()
        match = self.gallery.match(mean)
        if match.name is None and track.name is not None:
            kept = self.gallery.score(mean, track.name)
            if kept >= self.gallery.threshold - KEEP_MARGIN:
                return Match(track.name, kept, match.score)
        track.name = match.name
        return match

    def detect_frame(self, frame: Frame) -> FaceResult:
        if frame.source != self._source:
            self.reset()
            self._source = frame.source
        start = time.perf_counter()
        faces = self.recognize(frame.image, frame.timestamp)
        now = time.perf_counter()
        return FaceResult(faces, frame.index, frame.source, frame.size, (now - start) * 1000.0, now)
