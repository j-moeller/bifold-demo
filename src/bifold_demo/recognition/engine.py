"""Finding faces in an image and turning them into embeddings (OpenCV's FaceDetectorYN and FaceRecognizerSF)."""

from __future__ import annotations

import threading
from dataclasses import dataclass

import cv2
import numpy as np

from . import models

DETECT_SCORE = 0.7  # YuNet's confidence a face must reach
MIN_FACE = 24  # pixels: smaller faces are found but too small to recognise reliably
MAX_SIDE = 1600  # longer images are scaled down for detection (big photos to enrol)


@dataclass(frozen=True)
class Face:
    """A face in an image: ``box`` ``(x, y, w, h)`` in pixels, five ``landmarks`` ``(x, y)`` (the eyes as seen from
    the camera left to right, the nose tip, the mouth corners left to right) and the detector's ``score``."""

    box: tuple[float, float, float, float]
    landmarks: tuple[tuple[float, float], ...]
    score: float

    @property
    def size(self) -> float:
        return min(self.box[2], self.box[3])

    def row(self) -> np.ndarray:
        """The detector's own row (what ``FaceRecognizerSF.alignCrop`` takes)."""
        return np.array([*self.box, *np.ravel(self.landmarks), self.score], np.float32)


class FaceEngine:
    """Detects faces (:meth:`detect`) and embeds them (:meth:`embed`: a unit 128-vector; the dot product of two is
    their cosine similarity). The models are downloaded on first use (``models_folder``, see :mod:`.models`). Safe to
    share between threads (a lock: OpenCV's face objects keep state between calls)."""

    def __init__(self, models_folder=None, download: bool = True, score: float = DETECT_SCORE):
        paths = models.ensure_all(models_folder, download)
        self._detector = cv2.FaceDetectorYN.create(str(paths[models.DETECTOR.name]), "", (320, 320), score, 0.3, 5000)
        self._recognizer = cv2.FaceRecognizerSF.create(str(paths[models.RECOGNIZER.name]), "")
        self._lock = threading.Lock()

    def detect(self, image: np.ndarray, min_size: float = MIN_FACE) -> list[Face]:
        """Faces in ``image`` (BGR) at least ``min_size`` pixels across, biggest first."""
        h, w = image.shape[:2]
        scale = min(1.0, MAX_SIDE / max(h, w))
        small = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA) \
            if scale < 1.0 else image
        with self._lock:
            self._detector.setInputSize((small.shape[1], small.shape[0]))
            _, rows = self._detector.detect(np.ascontiguousarray(small))
        faces = []
        for row in rows if rows is not None else []:
            row = row.astype(np.float64) / np.r_[np.full(14, scale), 1.0]
            face = Face(tuple(row[:4]), tuple(map(tuple, row[4:14].reshape(5, 2))), float(row[14]))
            if face.size >= min_size:
                faces.append(face)
        faces.sort(key=lambda f: -f.box[2] * f.box[3])
        return faces

    def embed(self, image: np.ndarray, face: Face) -> np.ndarray:
        """The unit-length embedding of ``face`` in ``image`` (aligned on its landmarks to SFace's 112x112 crop)."""
        with self._lock:
            crop = self._recognizer.alignCrop(image, face.row())
            vector = self._recognizer.feature(crop).ravel().astype(np.float64)
        return vector / max(np.linalg.norm(vector), 1e-12)

    def faces(self, image: np.ndarray, min_size: float = MIN_FACE) -> list[tuple[Face, np.ndarray]]:
        """``(face, embedding)`` for every face in ``image``."""
        return [(face, self.embed(image, face)) for face in self.detect(image, min_size)]
