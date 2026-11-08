"""How good a face is for recognition: one :class:`Quality` record per face.

Quality is relative to the recogniser (a study of BSI's OFIQ found a general quality score of little use), so this
keeps several plain measures side by side instead of one number, and leaves weighing them to the caller (the gallery
here, the decision per person track in F2):

* **Geometry** from the five landmarks (the measures of ISO/IEC 29794-5, own code):

  * ``ied``: pixels between the eye centres in the image (the standards ask for at least 90 in reference photos;
    the faces in the demo's fair have 10-22);
  * ``yaw``, ``pitch``, ``roll`` in degrees, as the head textures count them: yaw to the head's own left positive,
    pitch down positive, roll towards the left shoulder positive. They come from fitting a mean face's five points
    in 3D (:data:`FACE_3D`: the alignment template with depths calibrated on avatar renders at known angles) to the
    landmarks with a scaled orthographic camera (a linear least-squares fit, then the nearest rotation). The yaw is
    reliable (on real photos within about 2 degrees of MediaPipe's estimate); **the pitch is not**: five points
    cannot tell a nod from a face's own proportions (a long nose, a short chin), so on real photos it read 10-17
    degrees further down than MediaPipe's on 6 of 16 frontal photos. It is recorded only: nothing here decides on it
    (the pose bins, the weight and the enrolment checks use the yaw), and F2 should not either without a better
    estimate (MediaPipe's face landmarks, which the head textures use);
  * ``pose``: the pose bin, ``frontal`` (yaw within :data:`FRONTAL` degrees), ``left`` and ``right`` (turned to the
    head's own left or right), ``profile`` (beyond :data:`PROFILE` degrees either way).

* **Image measures** on the aligned crop's central face region (eyes to chin): ``sharpness`` (the variance of the
  Laplacian over the variance of the grey levels, so contrast does not count), ``brightness`` and ``contrast`` (mean
  and spread of the grey levels, 0-255), ``clipped`` (the share of pixels nearly black or white), and ``outside``
  (the share of the crop beyond the picture, from :mod:`.align`).
* **Learnt measures**: ``fiqa``, eDifFIQA-T's score (higher is better; profiles ``best`` and ``licensed``), and
  ``norm``, the embedder's raw embedding length where it is a quality signal (AdaFace).

:attr:`Quality.weight` combines them into a weight from 0 to 1 for averaging a person's embeddings (the gallery's
templates); a starting point, documented with it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .align import TEMPLATE

FRONTAL, PROFILE = 20.0, 55.0  # degrees of yaw: the pose bins' edges
POSES = ("frontal", "left", "right", "profile")
# the mean face in 3D: the template's points (crop pixels; x right, y down) and their depth, z away from the camera
# (the nose tip and the mouth corners stand out from the eyes' plane). The depths were fitted to SCRFD's landmarks on
# 1620 renders of 20 avatars at known angles (yaw -60 to 60, pitch -15 to 15): the yaw comes back within 2 degrees
# (median) with a spread of 1.6-3.6 degrees, the pose bin in 98.6 % of the renders; the pitch reads about 5 degrees
# further down than the avatars look, and 10-17 degrees on some real photos (see the module docs: it is used in no
# decision). YuNet's landmarks follow the pose less well (84 % of the bins right).
NOSE_DEPTH, MOUTH_DEPTH = 32.0, 14.0
FACE_3D = np.c_[TEMPLATE, [0.0, 0.0, -NOSE_DEPTH, -MOUTH_DEPTH, -MOUTH_DEPTH]]
REGION = (slice(36, 104), slice(28, 84))  # rows, columns of the aligned crop: eyes to chin, cheek to cheek
DARK, BRIGHT = 8, 247  # grey levels counted as clipped


@dataclass(frozen=True)
class Quality:
    ied: float
    yaw: float
    pitch: float
    roll: float
    sharpness: float
    brightness: float
    contrast: float
    clipped: float
    outside: float
    detector: float
    fiqa: float | None = None
    norm: float | None = None

    @property
    def pose(self) -> str:
        return pose_bin(self.yaw)

    @property
    def weight(self) -> float:
        """0-1: how much this face should count among a person's faces. The eye distance up to 40 pixels (from 8),
        a frontal pose (the cosine of the yaw, squared, nothing beyond 75 degrees), sharpness up to 0.15 and, where
        known, eDifFIQA-T's score up to 0.6, multiplied; a face mostly outside the picture counts less."""
        w = _ramp(self.ied, 8.0, 40.0) * _ramp(self.sharpness, 0.0, 0.15) * (1.0 - min(self.outside, 1.0))
        w *= math.cos(math.radians(min(abs(self.yaw), 75.0))) ** 2 if abs(self.yaw) < 75.0 else 0.0
        if self.fiqa is not None:
            w *= _ramp(self.fiqa, 0.0, 0.6)
        return float(w)


def _ramp(value: float, low: float, high: float) -> float:
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def pose_bin(yaw: float) -> str:
    """``frontal``, ``left`` (turned to the head's own left: yaw positive), ``right`` or ``profile``."""
    if abs(yaw) >= PROFILE:
        return "profile"
    if abs(yaw) < FRONTAL:
        return "frontal"
    return "left" if yaw > 0 else "right"


def head_pose(landmarks) -> tuple[float, float, float]:
    """``(yaw, pitch, roll)`` in degrees (see the module docs) of a face with these five landmarks (image pixels, in
    the detectors' order). The roll is the tilt of the line through the eyes (turning and nodding, the yaw applied
    after the pitch as the head textures count them, keep that line level); yaw and pitch come from the 3D fit of
    the landmarks with the roll taken out."""
    points = np.asarray(landmarks, np.float64).reshape(5, 2)
    eyes = points[1] - points[0]
    roll = math.atan2(eyes[1], eyes[0])
    c, s = math.cos(-roll), math.sin(-roll)
    image = (points - points.mean(axis=0)) @ np.array([[c, s], [-s, c]])  # rotated by -roll: the eyes level
    model = FACE_3D - FACE_3D.mean(axis=0)
    fit, *_ = np.linalg.lstsq(model, image, rcond=None)  # model @ fit ~ image: fit = (s R[:2]).T, 3 x 2
    u, _, vt = np.linalg.svd(fit.T, full_matrices=False)
    rows = u @ vt  # the nearest two orthonormal rows of a rotation (head -> camera)
    rotation = np.vstack([rows, np.cross(rows[0], rows[1])])
    forward = -rotation[:, 2]  # where the nose points, in the camera (x right, y down, z away)
    yaw = math.degrees(math.atan2(forward[0], -forward[2]))
    pitch = math.degrees(math.atan2(forward[1], math.hypot(forward[0], forward[2])))
    return yaw, pitch, math.degrees(roll)


def image_measures(crop: np.ndarray) -> tuple[float, float, float, float]:
    """``(sharpness, brightness, contrast, clipped)`` of an aligned crop (see the module docs)."""
    grey = crop.astype(np.float32).mean(axis=2) if crop.ndim == 3 else crop.astype(np.float32)
    laplacian = cv2.Laplacian(grey, cv2.CV_32F, ksize=3)[REGION]
    face = grey[REGION]
    contrast = float(face.std())
    sharpness = float(laplacian.var() / max(face.var(), 1.0))
    clipped = float(((face < DARK) | (face > BRIGHT)).mean())
    return sharpness, float(face.mean()), contrast, clipped


def assess(face, aligned, fiqa: float | None = None, norm: float | None = None) -> Quality:
    """The :class:`Quality` of ``face`` (a :class:`~.detect.Face`) from its :class:`~.align.Aligned` crop, with
    eDifFIQA's score and the embedder's raw norm when known."""
    yaw, pitch, roll = head_pose(face.landmarks)
    sharpness, brightness, contrast, clipped = image_measures(aligned.crop)
    return Quality(face.eye_distance, yaw, pitch, roll, sharpness, brightness, contrast, clipped, aligned.outside,
                   float(face.score), None if fiqa is None else float(fiqa), None if norm is None else float(norm))


class FiqaModel:
    """eDifFIQA (``spec``, a quality model of :mod:`.registry`) at ``path``: :meth:`score` gives a score per aligned
    crop, higher for faces more useful to a recogniser."""

    def __init__(self, path, spec, provider: str | None = None, bgr: bool | None = None):
        from .runtime import OnnxModel

        self.spec = spec
        self.bgr = spec.bgr if bgr is None else bool(bgr)
        width, height = spec.size
        self.model = OnnxModel(path, (1, 3, height, width), provider=provider)

    def score(self, crops: np.ndarray) -> np.ndarray:
        """``(N,)`` scores of aligned crops ``(N, 112, 112, 3)`` (BGR)."""
        from .runtime import batches

        crops = np.asarray(crops)
        out = np.zeros(len(crops), np.float32)
        for part in batches(len(crops), self.model.fixed):
            x = crops[part].astype(np.float32)
            if not self.bgr:
                x = x[..., ::-1]
            x = np.ascontiguousarray(((x - self.spec.mean) / self.spec.std).transpose(0, 3, 1, 2))
            out[part] = np.asarray(self.model.run(x)[0], np.float32).reshape(-1)
        return out
