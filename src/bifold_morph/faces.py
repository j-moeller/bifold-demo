"""Finding faces (YuNet) and following them from frame to frame with steadied landmarks."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field

import cv2
import numpy as np

DETECT_SCORE = 0.7  # YuNet's confidence a face must reach
MIN_FACE = 40  # pixels: smaller faces are too small to edit
DETECT_SIDE = 640  # frames are scaled down to this longer side for detection (YuNet is fast, this is plenty)


@dataclass(frozen=True)
class Face:
    """A face in an image: ``box`` ``(x, y, w, h)`` in pixels, five ``landmarks`` (5, 2) (the eyes as seen in the
    image left to right, the nose tip, the mouth corners left to right) and the detector's ``score``."""

    box: tuple[float, float, float, float]
    landmarks: np.ndarray
    score: float

    @property
    def size(self) -> float:
        return min(self.box[2], self.box[3])

    @property
    def center(self) -> np.ndarray:
        x, y, w, h = self.box
        return np.array([x + w / 2, y + h / 2])


class FaceDetector:
    """OpenCV's YuNet (``cv2.FaceDetectorYN``). Safe to share between threads."""

    def __init__(self, model_path, score: float = DETECT_SCORE):
        self._detector = cv2.FaceDetectorYN.create(str(model_path), "", (320, 320), score, 0.3, 5000)
        self._lock = threading.Lock()

    def detect(self, image: np.ndarray, min_size: float = MIN_FACE) -> list[Face]:
        """Faces in ``image`` (BGR) at least ``min_size`` pixels across, biggest first."""
        h, w = image.shape[:2]
        scale = min(1.0, DETECT_SIDE / max(h, w))
        small = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA) \
            if scale < 1.0 else image
        with self._lock:
            self._detector.setInputSize((small.shape[1], small.shape[0]))
            _, rows = self._detector.detect(np.ascontiguousarray(small))
        faces = []
        for row in rows if rows is not None else []:
            row = row.astype(np.float64)
            # detections are at pixel centres of the small image; back to the frame's
            pts = (row[:14] + 0.5) / scale - 0.5
            box = (pts[0], pts[1], row[2] / scale, row[3] / scale)
            face = Face(box, pts[4:14].reshape(5, 2), float(row[14]))
            if face.size >= min_size:
                faces.append(face)
        faces.sort(key=lambda f: -f.box[2] * f.box[3])
        return faces


class OneEuro:
    """The 1€ filter (Casiez et al. 2012) for an array of values: steady while they hold still, quick to follow
    when they move. ``min_cutoff`` (Hz) sets the smoothing at rest, ``beta`` how much speed (units per second) opens
    it up."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.02, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.value: np.ndarray | None = None
        self._speed: np.ndarray | None = None
        self._t: float | None = None

    @staticmethod
    def _alpha(cutoff, dt: float):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        x = np.asarray(x, np.float64)
        if self.value is None or self._t is None or t <= self._t:
            self.value, self._speed, self._t = x.copy(), np.zeros_like(x), t
            return self.value.copy()
        dt = t - self._t
        speed = (x - self.value) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self._speed = a_d * speed + (1 - a_d) * self._speed
        cutoff = self.min_cutoff + self.beta * np.abs(self._speed)
        a = self._alpha(cutoff, dt)
        self.value = a * x + (1 - a) * self.value
        self._t = t
        return self.value.copy()


@dataclass
class Track:
    """One face followed over time: its newest detection, steadied landmarks and whatever the editor keeps for it
    (``state``, e.g. its smoothed StyleGAN code)."""

    id: int
    face: Face
    landmarks: np.ndarray
    last_seen: float
    filter: OneEuro = field(repr=False)
    state: dict = field(default_factory=dict, repr=False)


class FaceTracker:
    """Matches each frame's faces to the faces of the frames before (nearest centre, within ``reach`` face sizes)
    and steadies their landmarks; a face not seen for ``keep`` seconds is forgotten."""

    def __init__(self, reach: float = 0.6, keep: float = 0.5, min_cutoff: float = 1.5, beta: float = 0.01):
        self.reach, self.keep = reach, keep
        self.min_cutoff, self.beta = min_cutoff, beta
        self.tracks: list[Track] = []
        self._next_id = 1

    def update(self, faces: list[Face], t: float) -> list[Track]:
        """The tracks of ``faces`` (seen at time ``t``, seconds), in the order of ``faces``."""
        free = list(self.tracks)
        result = []
        for face in faces:
            best, best_d = None, self.reach * face.size
            for track in free:
                d = float(np.hypot(*(track.face.center - face.center)))
                if d < best_d:
                    best, best_d = track, d
            if best is None:
                best = Track(self._next_id, face, face.landmarks.copy(), t, OneEuro(self.min_cutoff, self.beta))
                self._next_id += 1
                self.tracks.append(best)
            else:
                free.remove(best)
            best.face, best.last_seen = face, t
            # speed counted in hundredths of the face's size, so near and far faces are steadied alike
            best.filter.beta = self.beta * 100.0 / max(face.size, 1.0)
            best.landmarks = best.filter(face.landmarks, t)
            result.append(best)
        self.tracks = [tr for tr in self.tracks if t - tr.last_seen <= self.keep]
        return result

    def reset(self) -> None:
        self.tracks = []
