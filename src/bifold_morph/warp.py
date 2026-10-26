"""Cutting faces out of a frame the way each network was trained, and putting results back.

Every network here was trained on faces aligned to a *template*: where the eyes, the nose tip and the mouth corners
sit in its square input, as shares of the side. :func:`matrix` finds the similarity transform (rotation, uniform
scale, shift) that moves a face's five landmarks closest to a template in the least-squares sense (Umeyama 1991),
:func:`warp` cuts the square out and :func:`paste` puts a result back along the same transform.

Coordinates are OpenCV's: a pixel's centre is at integer coordinates.
"""

from __future__ import annotations

import cv2
import numpy as np

# five points: the eyes as seen in the image left to right, the nose tip, the mouth corners left to right
TEMPLATES = {
    # InsightFace's ArcFace crop (112 x 112): face recognition, FairFace
    "arcface_112_v2": np.array([[0.34191607, 0.46157411], [0.65653393, 0.45983393], [0.50022500, 0.64050536],
                                [0.37097589, 0.82469196], [0.63151696, 0.82325089]]),
    # the ArcFace crop with a little more room round it (inswapper, HyperSwap)
    "arcface_128": np.array([[0.36167656, 0.40387734], [0.63696719, 0.40235469], [0.50019687, 0.56044219],
                             [0.38710391, 0.72160547], [0.61507734, 0.72034453]]),
    # the FFHQ crop: the whole head with hair and some background (FRAN)
    "ffhq_512": np.array([[0.37691676, 0.46864664], [0.62285697, 0.46912813], [0.50123859, 0.61331904],
                          [0.39308822, 0.72541100], [0.61150205, 0.72490465]]),
    # StyleGANEX's wider crop with background (used with the landmarks drawn in to 0.875 round the nose)
    "styleganex_384": np.array([[0.42353745, 0.52289879], [0.57725008, 0.52319972], [0.50123859, 0.61331904],
                                [0.43364461, 0.68337652], [0.57015325, 0.68306005]]),
}


def similarity(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """The similarity transform (2, 3) taking points ``src`` (n, 2) closest to ``dst`` (n, 2): Umeyama's
    least-squares solution, no reflection."""
    src, dst = np.asarray(src, np.float64), np.asarray(dst, np.float64)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s, d = src - mu_s, dst - mu_d
    cov = d.T @ s / len(src)
    u, sv, vt = np.linalg.svd(cov)
    sign = np.diag([1.0, np.sign(np.linalg.det(u) * np.linalg.det(vt))])
    rot = u @ sign @ vt
    scale = np.trace(np.diag(sv) @ sign) / max((s ** 2).sum() / len(src), 1e-12)
    m = np.zeros((2, 3))
    m[:, :2] = scale * rot
    m[:, 2] = mu_d - scale * rot @ mu_s
    return m


def matrix(landmarks, template: str, size: int) -> np.ndarray:
    """The affine map (2, 3) from frame pixels to the pixels of a ``size`` x ``size`` crop on ``template``."""
    return similarity(np.asarray(landmarks, np.float64).reshape(5, 2), TEMPLATES[template] * size)


def warp(image: np.ndarray, m: np.ndarray, size: int) -> np.ndarray:
    """The ``size`` x ``size`` crop of ``image`` under ``m``. A face much bigger than the crop is first shrunk with
    area averaging, so the crop does not alias."""
    scale = float(np.sqrt(abs(np.linalg.det(m[:, :2]))))
    if scale < 0.5:
        f = 2 * scale  # shrink to twice the crop's resolution, then warp
        small = cv2.resize(image, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        fx, fy = small.shape[1] / image.shape[1], small.shape[0] / image.shape[0]
        to_frame = np.array([[1 / fx, 0, 0.5 / fx - 0.5], [0, 1 / fy, 0.5 / fy - 0.5], [0, 0, 1]])
        m, image = m @ to_frame, small
    return cv2.warpAffine(image, m, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def paste(frame: np.ndarray, crop: np.ndarray, mask: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Blend ``crop`` (S, S, 3), cut out of ``frame`` along ``m``, back into ``frame`` (uint8, changed in place and
    returned), weighted by ``mask`` (S, S) in [0, 1]."""
    size = crop.shape[0]
    h, w = frame.shape[:2]
    inv = cv2.invertAffineTransform(m)
    corners = np.array([[0, 0], [size, 0], [size, size], [0, size]], np.float64) - 0.5
    pts = corners @ inv[:, :2].T + inv[:, 2]
    x0, y0 = np.clip(np.floor(pts.min(0)).astype(int), 0, [w, h])
    x1, y1 = np.clip(np.ceil(pts.max(0)).astype(int) + 1, 0, [w, h])
    if x1 <= x0 or y1 <= y0:
        return frame
    to_crop = m.copy()
    to_crop[:, 2] += m[:, :2] @ [x0, y0]  # from the box's pixels instead of the frame's
    stack = np.dstack([crop.astype(np.float32) * mask[..., None], mask]).astype(np.float32)
    warped = cv2.warpAffine(stack, to_crop, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    region = frame[y0:y1, x0:x1].astype(np.float32)
    alpha = warped[..., 3:]
    frame[y0:y1, x0:x1] = np.clip(region * (1 - alpha) + warped[..., :3] + 0.5, 0, 255).astype(np.uint8)
    return frame


def box_mask(size: int, blur: float = 0.3, padding: float = 0.0) -> np.ndarray:
    """1 in the middle of a square crop, fading to 0 at its edges over ``blur`` of half the side (FaceFusion's
    box mask): hides the crop's border when it goes back into the frame."""
    fade = max(1, int(size * 0.5 * blur))
    pad = int(size * padding)
    mask = np.ones((size, size), np.float32)
    mask[:max(fade // 2, pad)] = 0
    mask[-max(fade // 2, pad):] = 0
    mask[:, :max(fade // 2, pad)] = 0
    mask[:, -max(fade // 2, pad):] = 0
    return cv2.GaussianBlur(mask, (0, 0), fade * 0.25) if fade > 1 else mask


def center_crop(image: np.ndarray, center, side: float, size: int) -> tuple[np.ndarray, np.ndarray]:
    """An upright ``size`` x ``size`` crop of ``side`` frame pixels round ``center`` (InsightFace's attribute crop),
    with its matrix."""
    s = size / side
    m = np.array([[s, 0, size / 2 - s * center[0]], [0, s, size / 2 - s * center[1]]])
    return cv2.warpAffine(image, m, (size, size), flags=cv2.INTER_LINEAR, borderValue=0), m
