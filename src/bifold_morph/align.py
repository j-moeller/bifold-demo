"""FFHQ face alignment from five landmarks, and putting an edited face back into the frame.

StyleGAN2 and e4e know faces only as they appear in FFHQ: upright, the eyes on one line in the upper middle, the
mouth below. :func:`ffhq_quad` finds that square in a frame the way the FFHQ dataset was made (Karras et al.,
``align_images.py``, from the eye centres and the mouth corners), :func:`crop` cuts it out as a ``size`` x ``size``
image and :func:`paste` warps an image of the same square back into the frame.

Coordinates are OpenCV's: a pixel's centre is at integer coordinates.
"""

from __future__ import annotations

import math

import cv2
import numpy as np


def ffhq_quad(landmarks) -> np.ndarray:
    """The FFHQ crop square (4, 2) for five landmarks (the eyes as seen in the image left to right, the nose tip,
    the mouth corners left to right): its top-left, bottom-left, bottom-right and top-right corners."""
    lm = np.asarray(landmarks, np.float64).reshape(5, 2)
    eye_left, eye_right, _, mouth_left, mouth_right = lm
    eye_avg = (eye_left + eye_right) / 2
    eye_to_eye = eye_right - eye_left
    eye_to_mouth = (mouth_left + mouth_right) / 2 - eye_avg
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= max(np.hypot(*x), 1e-9)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    return np.stack([c - x - y, c - x + y, c + x + y, c + x - y])


def quad_matrix(quad: np.ndarray, size: int) -> np.ndarray:
    """The affine map (2, 3) from frame pixels to the pixels of the ``size`` x ``size`` crop of ``quad``."""
    src = np.float32(quad[:3])
    # the quad's corners are the outer edges of the crop's corner pixels
    dst = np.float32([[-0.5, -0.5], [-0.5, size - 0.5], [size - 0.5, size - 0.5]])
    return cv2.getAffineTransform(src, dst).astype(np.float64)


def quad_side(quad: np.ndarray) -> float:
    return float(np.hypot(*(quad[3] - quad[0])))


def crop(image: np.ndarray, quad: np.ndarray, size: int) -> np.ndarray:
    """The ``size`` x ``size`` crop of ``quad`` from ``image``; parts outside the image are mirrored in. A face much
    bigger than ``size`` is first shrunk with area averaging (as FFHQ was), so the crop does not alias."""
    matrix = quad_matrix(quad, size)
    shrink = quad_side(quad) / size
    if shrink >= 2.0:
        h, w = image.shape[:2]
        x0, y0 = np.floor(quad.min(0)).astype(int) - 2
        x1, y1 = np.ceil(quad.max(0)).astype(int) + 3
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
        if x1 - x0 >= 2 and y1 - y0 >= 2:
            r = 1.5 / shrink
            roi = image[y0:y1, x0:x1]
            small = cv2.resize(roi, (max(1, round((x1 - x0) * r)), max(1, round((y1 - y0) * r))),
                               interpolation=cv2.INTER_AREA)
            rx, ry = small.shape[1] / (x1 - x0), small.shape[0] / (y1 - y0)
            # frame = (small + 0.5) / r - 0.5 + offset, per axis
            to_frame = np.array([[1 / rx, 0, 0.5 / rx - 0.5 + x0], [0, 1 / ry, 0.5 / ry - 0.5 + y0], [0, 0, 1]])
            matrix = matrix @ to_frame
            image = small
    return cv2.warpAffine(image, matrix, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def frame_box(quad: np.ndarray, width: int, height: int) -> tuple[int, int, int, int] | None:
    """The pixels ``(x0, y0, x1, y1)`` of the frame that ``quad`` covers (end exclusive), or ``None`` if none."""
    x0, y0 = np.floor(quad.min(0)).astype(int)
    x1, y1 = np.ceil(quad.max(0)).astype(int) + 1
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, width), min(y1, height)
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def paste(frame: np.ndarray, layer: np.ndarray, mask: np.ndarray, quad: np.ndarray, replace: bool) -> np.ndarray:
    """Warp ``layer`` (S, S, 3), an image of the square ``quad`` in the frame, into ``frame`` (uint8, changed in
    place and returned), weighted by ``mask`` (S, S) in [0, 1].

    ``replace=False``: ``layer`` is a change, added to the frame's pixels (``frame + mask * layer``);
    ``replace=True``: ``layer`` is an image that replaces them (``frame + mask * (layer - frame)``)."""
    size = layer.shape[0]
    h, w = frame.shape[:2]
    box = frame_box(quad, w, h)
    if box is None:
        return frame
    x0, y0, x1, y1 = box
    matrix = quad_matrix(quad, size)
    matrix[:, 2] += matrix[:, :2] @ [x0, y0]  # from the box's pixels instead of the frame's
    stack = np.dstack([layer.astype(np.float32) * mask[..., None], mask]).astype(np.float32)
    warped = cv2.warpAffine(stack, matrix, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    region = frame[y0:y1, x0:x1].astype(np.float32)
    alpha = warped[..., 3:]
    out = region * (1 - alpha) + warped[..., :3] if replace else region + warped[..., :3]
    frame[y0:y1, x0:x1] = np.clip(out + 0.5, 0, 255).astype(np.uint8)
    return frame


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def border_mask(size: int, inner: float = 0.12, outer: float = 0.02) -> np.ndarray:
    """1 over the crop, fading to 0 towards its edges: full weight ``inner`` (a share of the side) from the edge,
    none ``outer`` from it. For changes, which are mostly 0 away from the face anyway."""
    c = (np.arange(size, dtype=np.float32) + 0.5) / size
    edge = np.minimum(c, 1 - c)
    d = np.minimum(edge[:, None], edge[None, :])
    return _smoothstep(outer, inner, d).astype(np.float32)


def face_mask(size: int) -> np.ndarray:
    """A soft ellipse round the face, forehead and chin of an FFHQ crop (eyes at about 47 %, mouth at about 74 %
    of the height): where a generated face replaces the real one."""
    c = (np.arange(size, dtype=np.float32) + 0.5) / size
    dx = (c[None, :] - 0.5) / 0.36
    dy = (c[:, None] - 0.54) / 0.43
    r = np.sqrt(dx * dx + dy * dy)
    return (1.0 - _smoothstep(0.78, 1.0, r)).astype(np.float32)


def box_in_crop(box, quad: np.ndarray, size: int) -> tuple[float, float, float, float]:
    """A frame box ``(x, y, w, h)`` as the axis-aligned box ``(x0, y0, x1, y1)`` round it in the crop's pixels."""
    x, y, bw, bh = box
    corners = np.array([[x, y, 1], [x + bw, y, 1], [x, y + bh, 1], [x + bw, y + bh, 1]], np.float64)
    pts = corners @ quad_matrix(quad, size).T
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    return float(x0), float(y0), float(x1), float(y1)


def rotation_deg(quad: np.ndarray) -> float:
    """How far the crop is turned in the frame, in degrees (0 = upright)."""
    top = quad[3] - quad[0]
    return math.degrees(math.atan2(top[1], top[0]))
