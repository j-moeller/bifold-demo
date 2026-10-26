"""Drawing detections and status information on top of video frames."""

from __future__ import annotations

import cv2
import numpy as np

from ..detection.results import Detection

FONT = cv2.FONT_HERSHEY_SIMPLEX

# distinct BGR colours, indexed by class id
PALETTE = [
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255), (49, 210, 207), (10, 249, 72),
    (23, 204, 146), (134, 219, 61), (52, 147, 26), (187, 212, 0), (168, 153, 44), (255, 194, 0),
    (147, 69, 52), (255, 115, 100), (236, 24, 0), (255, 56, 132), (133, 0, 82), (255, 56, 203),
    (200, 149, 255), (199, 55, 255),
]


def class_color(class_id: int) -> tuple[int, int, int]:
    return PALETTE[class_id % len(PALETTE)]


def _ui_scale(image: np.ndarray) -> float:
    return max(image.shape[0], image.shape[1] * 9 / 16) / 720.0


def _text_color(bg) -> tuple[int, int, int]:
    b, g, r = bg
    return (0, 0, 0) if 0.299 * r + 0.587 * g + 0.114 * b > 150 else (255, 255, 255)


def draw_detections(image: np.ndarray, detections: list[Detection], from_size: tuple[int, int] | None = None) -> None:
    """Draw labelled boxes in place. ``from_size`` is the (width, height) the boxes refer to, if different."""
    h, w = image.shape[:2]
    sx, sy = (w / from_size[0], h / from_size[1]) if from_size else (1.0, 1.0)
    s = _ui_scale(image)
    thickness = max(2, round(2 * s))
    font_scale = 0.6 * s
    text_thickness = max(1, round(1.5 * s))
    for det in detections:
        x1, y1, x2, y2 = (round(v) for v in (det.box[0] * sx, det.box[1] * sy, det.box[2] * sx, det.box[3] * sy))
        color = class_color(det.class_id)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
        label = f"{det.label} {det.confidence:.0%}"
        (tw, th), baseline = cv2.getTextSize(label, FONT, font_scale, text_thickness)
        pad = max(3, round(4 * s))
        top = y1 - th - baseline - 2 * pad
        if top < 0:  # no room above the box: put the label inside it
            top = y1
        cv2.rectangle(image, (x1, top), (x1 + tw + 2 * pad, top + th + baseline + 2 * pad), color, -1)
        cv2.putText(image, label, (x1 + pad, top + pad + th), FONT, font_scale, _text_color(color),
                    text_thickness, cv2.LINE_AA)


def draw_panel(image: np.ndarray, lines: list[str], corner: str = "top-left", alpha: float = 0.55) -> None:
    """Draw lines of text on a translucent dark panel in a corner of the image."""
    if not lines:
        return
    s = _ui_scale(image)
    font_scale, thickness = 0.55 * s, max(1, round(1.2 * s))
    pad, gap = round(10 * s), round(8 * s)
    sizes = [cv2.getTextSize(line, FONT, font_scale, thickness)[0] for line in lines]
    line_h = max(h for _, h in sizes)
    width = max(w for w, _ in sizes) + 2 * pad
    height = len(lines) * line_h + (len(lines) - 1) * gap + 2 * pad
    ih, iw = image.shape[:2]
    margin = round(12 * s)
    x = margin if "left" in corner else iw - width - margin
    y = margin if "top" in corner else ih - height - margin
    x, y = max(0, x), max(0, y)
    x2, y2 = min(iw, x + width), min(ih, y + height)
    roi = image[y:y2, x:x2]
    roi[:] = (roi * (1 - alpha)).astype(np.uint8)
    for i, line in enumerate(lines):
        baseline_y = y + pad + line_h + i * (line_h + gap)
        cv2.putText(image, line, (x + pad, baseline_y), FONT, font_scale, (240, 240, 240), thickness, cv2.LINE_AA)


def draw_banner(image: np.ndarray, message: str, color=(40, 40, 200)) -> None:
    """A coloured message bar across the top centre (errors, notifications)."""
    s = _ui_scale(image)
    font_scale, thickness = 0.6 * s, max(1, round(1.5 * s))
    (tw, th), baseline = cv2.getTextSize(message, FONT, font_scale, thickness)
    pad = round(10 * s)
    iw = image.shape[1]
    x = max(0, (iw - tw) // 2 - pad)
    y = round(12 * s)
    cv2.rectangle(image, (x, y), (min(iw - 1, x + tw + 2 * pad), y + th + baseline + 2 * pad), color, -1)
    cv2.putText(image, message, (x + pad, y + pad + th), FONT, font_scale, _text_color(color), thickness, cv2.LINE_AA)


def draw_selection(image: np.ndarray, box, label: str, detail: str = "", color=(0, 200, 255)) -> None:
    """Mark a selected person: corner brackets around ``box`` (x0, y0, x1, y1) and a tag with ``label`` (and
    ``detail`` on a second line) above it, or below it when there is no room above."""
    s = _ui_scale(image)
    ih, iw = image.shape[:2]
    x0, y0, x1, y1 = (round(v) for v in box)
    thickness = max(2, round(2.5 * s))
    arm = max(6, round(0.25 * min(x1 - x0, y1 - y0)))
    for cx, cy, dx, dy in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
        cv2.line(image, (cx, cy), (cx + dx * arm, cy), color, thickness, cv2.LINE_AA)
        cv2.line(image, (cx, cy), (cx, cy + dy * arm), color, thickness, cv2.LINE_AA)
    lines = [label] + ([detail] if detail else [])
    font_scale, text_thickness = 0.55 * s, max(1, round(1.4 * s))
    sizes = [cv2.getTextSize(line, FONT, font_scale, text_thickness) for line in lines]
    pad, gap = max(3, round(5 * s)), max(2, round(4 * s))
    line_h = max(th + baseline for (_tw, th), baseline in sizes)
    width = max(tw for (tw, _th), _b in sizes) + 2 * pad
    height = len(lines) * line_h + (len(lines) - 1) * gap + 2 * pad
    top = y0 - height - gap
    if top < 0:
        top = min(y1 + gap, ih - height)
    left = min(max(0, x0), max(0, iw - width))
    cv2.rectangle(image, (left, top), (left + width, top + height), color, -1)
    for k, (line, ((_tw, th), _baseline)) in enumerate(zip(lines, sizes)):
        y = top + pad + k * (line_h + gap) + th
        cv2.putText(image, line, (left + pad, y), FONT, font_scale, _text_color(color), text_thickness, cv2.LINE_AA)
