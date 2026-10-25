"""Cutting people out of images: YOLO instance segmentation, refined with GrabCut."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

SEGMENTATION_MODEL = "yolo26n-seg.pt"


@dataclass
class Cutout:
    """One person cut out of an image.

    Attributes:
        rgba: the person cropped to their outline, RGBA (transparent around them), row 0 = top.
        box: ``(x1, y1, x2, y2)`` of the crop in the source image.
        confidence: detector confidence that this is a person.
        touches: edges of the source image the person is cut off by (``"top"``, ``"bottom"``, ...).
    """

    rgba: np.ndarray
    box: tuple[int, int, int, int]
    confidence: float
    touches: set[str] = field(default_factory=set)

    @property
    def full_body(self) -> bool:
        """Head and feet are inside the image."""
        return not ({"top", "bottom"} & self.touches)

    @property
    def area(self) -> int:
        return int((self.rgba[..., 3] > 127).sum())


class PersonSegmenter:
    """Finds people in BGR images and returns them as transparent cut-outs."""

    def __init__(self, model: str = SEGMENTATION_MODEL, device: str = "cpu", models_dir="models"):
        from ultralytics import YOLO

        from ..detection.yolo import resolve_model_path

        self.model = YOLO(resolve_model_path(model, models_dir))
        self.device = device

    def cutouts(self, image: np.ndarray, conf: float = 0.4, refine: bool = True) -> list[Cutout]:
        """All people in ``image``, largest first."""
        result = self.model.predict(image, classes=[0], conf=conf, retina_masks=True, device=self.device,
                                    verbose=False)[0]
        if result.masks is None:
            return []
        masks = result.masks.data.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        people = []
        for mask, confidence in zip(masks, confidences):
            mask = cv2.resize(mask, (image.shape[1], image.shape[0])) if mask.shape != image.shape[:2] else mask
            binary = (mask > 0.5).astype(np.uint8)
            if refine:
                binary = refine_mask(image, binary)
            cutout = cut_out(image, binary, float(confidence))
            if cutout is not None:
                people.append(cutout)
        return sorted(people, key=lambda c: c.area, reverse=True)


def refine_mask(image: np.ndarray, mask: np.ndarray, iterations: int = 3) -> np.ndarray:
    """Snap a coarse 0/1 person mask to the image edges with GrabCut (works best on plain backgrounds)."""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return mask
    h, w = mask.shape
    size = max(ys.max() - ys.min(), xs.max() - xs.min())
    band = max(2, int(size * 0.015))
    pad = 3 * band
    x1, y1 = max(0, xs.min() - pad), max(0, ys.min() - pad)
    x2, y2 = min(w, xs.max() + pad + 1), min(h, ys.max() + pad + 1)
    roi_mask, roi_image = mask[y1:y2, x1:x2], image[y1:y2, x1:x2]

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band + 1, 2 * band + 1))
    labels = np.full(roi_mask.shape, cv2.GC_BGD, np.uint8)
    labels[cv2.dilate(roi_mask, kernel, iterations=2) > 0] = cv2.GC_PR_BGD
    labels[roi_mask > 0] = cv2.GC_PR_FGD
    labels[cv2.erode(roi_mask, kernel, iterations=2) > 0] = cv2.GC_FGD

    scale = min(1.0, 700 / max(roi_mask.shape))  # GrabCut is slow on large images
    small_image = cv2.resize(roi_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else roi_image
    small_labels = (cv2.resize(labels, (small_image.shape[1], small_image.shape[0]), interpolation=cv2.INTER_NEAREST)
                    if scale < 1 else labels)
    if not (small_labels == cv2.GC_FGD).any() or not (small_labels == cv2.GC_BGD).any():
        return mask
    try:
        cv2.grabCut(small_image, small_labels, None, np.zeros((1, 65)), np.zeros((1, 65)), iterations,
                    cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return mask
    refined = np.isin(small_labels, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
    if scale < 1:
        refined = (cv2.resize(refined.astype(np.float32), (roi_mask.shape[1], roi_mask.shape[0])) > 0.5).astype(np.uint8)
    refined = _largest_component(refined)
    out = np.zeros_like(mask)
    out[y1:y2, x1:x2] = refined
    return out


def _largest_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 2:
        return mask
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest).astype(np.uint8)


def cut_out(image: np.ndarray, mask: np.ndarray, confidence: float = 1.0, margin: int = 2) -> Cutout | None:
    """Crop ``image`` (BGR) to the 0/1 ``mask`` and make everything outside it transparent."""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    h, w = mask.shape
    tol = max(2, round(0.01 * max(h, w)))  # masks often stop a few pixels short of the border
    touches = {edge for edge, hit in (("top", ys.min() < tol), ("bottom", ys.max() >= h - tol),
                                      ("left", xs.min() < tol), ("right", xs.max() >= w - tol)) if hit}
    x1, y1 = max(0, xs.min() - margin), max(0, ys.min() - margin)
    x2, y2 = min(w, xs.max() + margin + 1), min(h, ys.max() + margin + 1)
    alpha = cv2.GaussianBlur(mask[y1:y2, x1:x2].astype(np.float32), (3, 3), 0)
    rgb = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
    rgba = np.dstack([rgb, (alpha * 255).astype(np.uint8)])
    return Cutout(rgba=rgba, box=(int(x1), int(y1), int(x2), int(y2)), confidence=confidence, touches=touches)
