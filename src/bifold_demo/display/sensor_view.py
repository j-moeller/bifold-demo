"""The debug view of the virtual high-resolution sensor (``W`` in the demo): where the frame's windows are and what
they show, and stable tags for the person boxes they are asked for until the person tracker gives tracks."""

from __future__ import annotations

import cv2
import numpy as np

from .overlay import FONT, _text_color, _ui_scale, class_color

INSET_SHARE = 0.2  # of the picture's width: the column of windows on the right


class BoxTags:
    """Stable numbers for the person boxes of successive detections: a box keeps the number of the box of the
    detection before that it overlaps most (intersection over union at least ``overlap``), else gets a new one. A
    stand-in for the person tracker's tracks."""

    def __init__(self, overlap: float = 0.3):
        self.overlap = overlap
        self._boxes: list = []
        self._tags: list[int] = []
        self._next = 1

    def assign(self, boxes) -> list[int]:
        """The numbers of ``boxes`` (``(x0, y0, x1, y1)`` each), in their order."""
        boxes = [tuple(float(v) for v in box) for box in boxes]
        tags: list[int | None] = [None] * len(boxes)
        taken = set()
        pairs = sorted(((_iou(box, old), i, j) for i, box in enumerate(boxes) for j, old in enumerate(self._boxes)),
                       reverse=True)
        for score, i, j in pairs:
            if score < self.overlap:
                break
            if tags[i] is None and j not in taken:
                tags[i] = self._tags[j]
                taken.add(j)
        for i, tag in enumerate(tags):
            if tag is None:
                tags[i] = self._next
                self._next += 1
        self._boxes, self._tags = boxes, tags
        return tags


def _iou(a, b) -> float:
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _colour(tag):
    return class_color(tag if isinstance(tag, int) else sum(map(ord, str(tag))))


def draw_windows(image: np.ndarray, windows, from_size: tuple[int, int] | None = None) -> None:
    """Draw in place where each window (:class:`bifold_demo.sensor.WindowResult`) lies in the picture, and the windows
    themselves in a column on the right, each with its tag, level, eye distance and render time. ``from_size``: the
    (width, height) the windows' view pixels refer to, if not the picture's."""
    if not windows:
        return
    h, w = image.shape[:2]
    sx, sy = (w / from_size[0], h / from_size[1]) if from_size else (1.0, 1.0)
    s = _ui_scale(image)
    thickness, font_scale, text_thickness = max(1, round(1.5 * s)), 0.45 * s, max(1, round(1.1 * s))
    gap = max(4, round(8 * s))
    side_w = round(INSET_SHARE * w)
    side_h = (h - gap * (len(windows) + 1)) // len(windows)
    y = gap
    for result in windows:
        colour = _colour(result.tag)
        x0, y0, x1, y1 = result.view_rect
        cv2.rectangle(image, (round(x0 * sx), round(y0 * sy)), (round(x1 * sx), round(y1 * sy)), colour, thickness,
                      cv2.LINE_AA)
        thumb = result.image
        scale = min(side_w / thumb.shape[1], side_h / thumb.shape[0])
        tw, th = max(1, round(thumb.shape[1] * scale)), max(1, round(thumb.shape[0] * scale))
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        thumb = cv2.resize(thumb, (tw, th), interpolation=interpolation)
        left = w - gap - tw
        image[y:y + th, left:left + tw] = thumb
        cv2.rectangle(image, (left - 1, y - 1), (left + tw, y + th), colour, thickness)
        eyes = result.truth.eye_distance if result.truth is not None else result.eye_distance
        label = f"{result.tag} L{result.window.level}" + (f" {eyes:.0f}px" if eyes else "") + \
            f" {result.render_ms + result.camera_ms:.0f}ms"
        (lw, lh), base = cv2.getTextSize(label, FONT, font_scale, text_thickness)
        pad = max(2, round(3 * s))
        cv2.rectangle(image, (left, y + th - lh - base - 2 * pad), (left + lw + 2 * pad, y + th), colour, -1)
        cv2.putText(image, label, (left + pad, y + th - base - pad), FONT, font_scale, _text_color(colour),
                    text_thickness, cv2.LINE_AA)
        y += th + gap
