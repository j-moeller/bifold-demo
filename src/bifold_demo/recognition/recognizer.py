"""Recognising the faces in the video frames: who is who, frame by frame (run on a worker thread by the demo)."""

from __future__ import annotations

import time
from dataclasses import dataclass

from ..sources.base import Frame
from .engine import Face, FaceEngine
from .gallery import Gallery, Match


@dataclass(frozen=True)
class Recognized:
    """A face in a frame and who it is (``match.name`` is ``None`` for a stranger)."""

    face: Face
    match: Match

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


class FaceRecognizer:
    """Finds the faces in a frame and names them from ``gallery`` (which is looked at again for new photos at most
    once a second). ``detect_frame`` makes it usable with :class:`~bifold_demo.detection.worker.AsyncDetector`."""

    def __init__(self, gallery: Gallery, engine: FaceEngine | None = None):
        self.gallery = gallery
        self.engine = engine or gallery.engine
        if gallery._engine is None:
            gallery._engine = self.engine

    def recognize(self, image) -> list[Recognized]:
        self.gallery.refresh()
        return [Recognized(face, self.gallery.match(vector)) for face, vector in self.engine.faces(image)]

    def detect_frame(self, frame: Frame) -> FaceResult:
        start = time.perf_counter()
        faces = self.recognize(frame.image)
        now = time.perf_counter()
        return FaceResult(faces, frame.index, frame.source, frame.size, (now - start) * 1000.0, now)
