"""Lining faces up for the embedders: the ArcFace five-point similarity transform to a 112 x 112 crop.

Almost every public face recogniser (AdaFace, InsightFace's ArcFace models, AuraFace, SFace) was trained on 112 x
112 crops made by one similarity transform (rotation, one scale, shift) that takes the detector's five landmarks as
close as possible (least squares) to fixed points, :data:`TEMPLATE`: the eyes at (38.29, 51.70) and (73.53, 51.50),
the nose tip at (56.03, 71.74), the mouth corners at (41.55, 92.37) and (70.73, 92.20). InsightFace's ``norm_crop``
and OpenCV's ``FaceRecognizerSF.alignCrop`` use exactly these points. :func:`similarity` is Umeyama's least-squares
estimate (what InsightFace uses, through scikit-image), without reflections; it keeps the face's proportions, so a
turned face looks turned, as the recognisers saw in training.

**Borders.** Where the crop reaches beyond the image (a face at the picture's edge), the crop is black, as the
training crops were (``borderValue=0``); :attr:`Aligned.outside` says how much of the crop that is, a quality
signal.

**Big faces.** A face much bigger than the crop (an enrolment photo with 200 pixels between the eyes, a scale of
0.18) would alias badly if only warped: bilinear sampling then skips most pixels. With ``antialias`` the part of the
image the crop shows is first shrunk with area averaging to the crop's scale, then warped (only when the crop is
smaller than :data:`ANTIALIAS_BELOW` of the face's size). The landmarks still go exactly onto the template. Measured
on avatar renders and a few photos, it makes no difference beyond noise to how well the recognisers separate people
(docs/research/face_recognition_f1_results.md); it is kept as the sound way to shrink an image several times.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

SIZE = 112
TEMPLATE = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655],
                     [70.7299, 92.2041]])
TEMPLATE_EYES = float(np.linalg.norm(TEMPLATE[1] - TEMPLATE[0]))  # 35.24 crop pixels between the eyes
ANTIALIAS_BELOW = 0.8  # shrink first when the crop has fewer pixels than this per image pixel


def similarity(src, dst) -> np.ndarray:
    """The 2 x 3 matrix of the similarity transform (rotation, uniform scale, shift; no reflection) taking the points
    ``src`` closest to ``dst`` in the least-squares sense (Umeyama, 1991)."""
    src, dst = np.asarray(src, np.float64), np.asarray(dst, np.float64)
    mean_src, mean_dst = src.mean(axis=0), dst.mean(axis=0)
    a, b = src - mean_src, dst - mean_dst
    covariance = b.T @ a / len(src)
    u, d, vt = np.linalg.svd(covariance)
    sign = np.eye(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:  # a reflection would fit better: take the best rotation instead
        sign[1, 1] = -1.0
    rotation = u @ sign @ vt
    variance = (a ** 2).sum() / len(src)
    scale = float(np.trace(np.diag(d) @ sign) / variance) if variance > 0 else 1.0
    shift = mean_dst - scale * rotation @ mean_src
    return np.c_[scale * rotation, shift]


@dataclass(frozen=True)
class Aligned:
    """A face lined up for the embedders: the 112 x 112 ``crop`` (the image's colour order), the ``matrix`` taking
    image pixels to crop pixels, its ``scale`` (crop pixels per image pixel) and the share of the crop ``outside``
    the image (filled black)."""

    crop: np.ndarray
    matrix: np.ndarray
    scale: float
    outside: float


def _homogeneous(matrix: np.ndarray) -> np.ndarray:
    return np.vstack([matrix, [0.0, 0.0, 1.0]])


def align(image: np.ndarray, landmarks, size: int = SIZE, antialias: bool = True) -> Aligned:
    """The face with these five ``landmarks`` (pixels of ``image``, in the detectors' order) lined up on the template
    (see the module docs)."""
    template = TEMPLATE * (size / SIZE)
    matrix = similarity(np.asarray(landmarks, np.float64).reshape(5, 2), template)
    scale = float(np.hypot(matrix[0, 0], matrix[1, 0]))
    h, w = image.shape[:2]
    inverse = np.linalg.inv(_homogeneous(matrix))
    source, warp = image, matrix
    if antialias and scale < ANTIALIAS_BELOW:
        # the part of the image the crop shows (a margin of two crop pixels), shrunk to the crop's scale first
        corners = np.array([[-2.0, -2.0, 1.0], [size + 2.0, -2.0, 1.0], [-2.0, size + 2.0, 1.0],
                            [size + 2.0, size + 2.0, 1.0]]) @ inverse.T
        x0, y0 = np.floor(corners[:, :2].min(axis=0)).astype(int)
        x1, y1 = np.ceil(corners[:, :2].max(axis=0)).astype(int)
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
        if x1 - x0 >= 2 and y1 - y0 >= 2:
            part = image[y0:y1, x0:x1]
            small_w, small_h = max(1, round((x1 - x0) * scale)), max(1, round((y1 - y0) * scale))
            source = cv2.resize(part, (small_w, small_h), interpolation=cv2.INTER_AREA)
            sx, sy = small_w / (x1 - x0), small_h / (y1 - y0)
            # image pixel -> pixel of the shrunk part (pixel centres kept): s = (x - x0 + 0.5) * f - 0.5
            shrink = np.array([[sx, 0.0, (0.5 - x0) * sx - 0.5], [0.0, sy, (0.5 - y0) * sy - 0.5], [0, 0, 1]])
            warp = (_homogeneous(matrix) @ np.linalg.inv(shrink))[:2]
    crop = cv2.warpAffine(source, warp, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                          borderValue=0)
    return Aligned(crop, matrix, scale, _outside(inverse, (h, w), size))


def _outside(inverse: np.ndarray, shape, size: int, step: int = 4) -> float:
    """The share of the crop (sampled every ``step`` pixels) whose image pixel lies outside the image."""
    grid = np.arange(step / 2, size, step)
    xs, ys = np.meshgrid(grid, grid)
    points = np.c_[xs.ravel(), ys.ravel(), np.ones(xs.size)] @ inverse.T
    h, w = shape[:2]
    out = (points[:, 0] < -0.5) | (points[:, 0] > w - 0.5) | (points[:, 1] < -0.5) | (points[:, 1] > h - 0.5)
    return float(out.mean())


def crops(aligned) -> np.ndarray:
    """The crops of a list of :class:`Aligned`, stacked: ``(N, size, size, 3)``."""
    aligned = list(aligned)
    if not aligned:
        return np.zeros((0, SIZE, SIZE, 3), np.uint8)
    return np.stack([a.crop for a in aligned])
