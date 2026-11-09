"""YOLOX (Megvii, Apache-2.0): the commercial profile's person detector, without AGPL code.

The Ultralytics YOLO models and code the demo uses by default are AGPL-3.0; the commercial profile
(:mod:`bifold_demo.recognition.registry`) finds people with YOLOX instead: its official ONNX files from the YOLOX
GitHub release 0.1.1rc0 (``yolox_s.onnx``, 640 x 640, COCO mAP 40.5; ``yolox_tiny.onnx``, 416 x 416, 32.8), listed in
the face recognition's registry with their SHA-256 (downloaded and checked like the face models: ``bifold-faces
fetch --profile commercial``, ``bifold-faces check``), run by ONNX Runtime.

:class:`YoloxDetector` has :class:`~.yolo.YoloDetector`'s interface (``detect``, ``detect_frame``, ``conf``,
``model_name``, ``device``, ``names``, ``class_ids``), so the demo and the person tracks use either. The image is
letterboxed into the model's square input (scaled to fit with its aspect kept, at the top left, the rest grey 114),
in BGR order and 0-255, as YOLOX's own ``preproc`` does; the output, per anchor of the strides 8, 16 and 32, is the
box's centre offset and log size in strides, the objectness and the 80 COCO classes' scores (both already through a
sigmoid in the graph), decoded as YOLOX's ``demo_postprocess`` does; a box's score is its objectness times its class's
score, and overlapping boxes of a class are suppressed (non-maximum suppression at an overlap of ``iou``).

Trained on COCO 2017 (its annotations CC BY 4.0, its images under Flickr's terms): "commercial" is about the
weights' and the code's licences, as for the face models (see the registry).
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from ..sources.base import Frame
from .results import Detection, DetectionResult

COCO = ("person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
        "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush")
STRIDES = (8, 16, 32)
PAD = 114  # the grey YOLOX letterboxes with

__all__ = ["COCO", "YoloxDetector", "decode", "letterbox"]


def letterbox(image: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, float]:
    """``image`` scaled to fit ``size`` = (width, height) with its aspect kept, at the top left of a grey canvas;
    and the scale."""
    width, height = size
    h, w = image.shape[:2]
    scale = min(width / w, height / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    canvas = np.full((height, width, 3), PAD, np.uint8)
    canvas[:new_h, :new_w] = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return canvas, scale


def decode(output: np.ndarray, size: tuple[int, int], strides=STRIDES) -> np.ndarray:
    """YOLOX's raw ``output`` (anchors x 85) for an input of ``size`` = (width, height): ``(anchors, 4 + 1 + 80)``
    with the boxes as ``x0, y0, x1, y1`` input pixels (``demo_postprocess``, then corners)."""
    width, height = size
    grids, expanded = [], []
    for stride in strides:
        h, w = height // stride, width // stride
        xv, yv = np.meshgrid(np.arange(w), np.arange(h))
        grids.append(np.stack((xv, yv), axis=2).reshape(-1, 2))
        expanded.append(np.full((h * w, 1), stride))
    grid = np.concatenate(grids).astype(np.float32)
    stride = np.concatenate(expanded).astype(np.float32)
    out = np.asarray(output, np.float32).reshape(len(grid), -1).copy()
    centre = (out[:, :2] + grid) * stride
    half = np.exp(out[:, 2:4]) * stride / 2.0
    out[:, :2], out[:, 2:4] = centre - half, centre + half
    return out


def _nms(boxes: np.ndarray, scores: np.ndarray, overlap: float) -> list[int]:
    order = np.argsort(-scores, kind="stable")
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        rest = order[1:]
        w = np.clip(np.minimum(boxes[i, 2], boxes[rest, 2]) - np.maximum(boxes[i, 0], boxes[rest, 0]), 0, None)
        h = np.clip(np.minimum(boxes[i, 3], boxes[rest, 3]) - np.maximum(boxes[i, 1], boxes[rest, 1]), 0, None)
        inter = w * h
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        order = rest[inter / np.maximum(area[i] + area[rest] - inter, 1e-9) <= overlap]
    return keep


class YoloxDetector:
    """YOLOX (see the module docs): ``model`` is a registry id (``yolox_s`` or ``yolox_tiny``), its file found in the
    face models' folder (``models_folder``) and downloaded when missing if ``download``; ``conf``, ``iou`` and
    ``classes`` as for :class:`~.yolo.YoloDetector`; ``provider``: ONNX Runtime's (default: automatic)."""

    def __init__(self, model: str = "yolox_s", conf: float = 0.35, iou: float = 0.45, classes=None,
                 models_folder=None, download: bool = True, provider: str | None = None, progress=print):
        from .. import downloads
        from ..recognition import registry
        from ..recognition.runtime import OnnxModel, provider_choice

        self.spec = registry.model(model)
        if self.spec.task != "people":
            raise ValueError(f"{model} is not a person detector")
        path = registry.path(self.spec, models_folder, download, progress)
        if not downloads.verified(path, self.spec.file.sha256):
            raise OSError(f"{path} is damaged: its SHA-256 is not the registered one (fetch it again: bifold-faces "
                          f"fetch --model {model})")
        width, height = self.spec.size
        self.model = OnnxModel(path, (1, 3, height, width), provider=provider)
        self.model_name = self.spec.file.name
        self.device = provider_choice(provider)
        self.names = dict(enumerate(COCO))
        self.conf, self.iou = float(conf), float(iou)
        self.class_ids = self._class_ids(classes)

    def _class_ids(self, classes) -> list[int] | None:
        if not classes:
            return None
        by_name = {name: k for k, name in self.names.items()}
        ids = []
        for item in classes:
            if isinstance(item, int) or str(item).isdigit():
                ids.append(int(item))
            elif str(item).lower() in by_name:
                ids.append(by_name[str(item).lower()])
            else:
                raise ValueError(f"unknown class {item!r}; the model knows: {', '.join(self.names.values())}")
        return ids

    def detect(self, image: np.ndarray, conf: float | None = None) -> list[Detection]:
        """Objects in a BGR image, at least ``conf`` confident (default: :attr:`conf`)."""
        size = self.spec.size
        board, scale = letterbox(image, size)
        blob = np.ascontiguousarray(board.transpose(2, 0, 1)[None].astype(np.float32))
        out = decode(self.model.run(blob)[0], size)
        scores = out[:, 4:5] * out[:, 5:]
        classes = scores.argmax(axis=1)
        best = scores[np.arange(len(scores)), classes]
        keep = best >= (self.conf if conf is None else conf)
        if self.class_ids is not None:
            keep &= np.isin(classes, self.class_ids)
        boxes, best, classes = out[keep, :4] / scale, best[keep], classes[keep]
        found = []
        for k in sorted(set(classes.tolist())):
            mine = np.flatnonzero(classes == k)
            for i in _nms(boxes[mine], best[mine], self.iou):
                j = mine[i]
                box = tuple(float(v) for v in np.clip(boxes[j], 0, [image.shape[1], image.shape[0]] * 2))
                found.append(Detection(box, float(best[j]), int(k), self.names[int(k)]))
        found.sort(key=lambda d: -d.confidence)
        return found

    def detect_frame(self, frame: Frame) -> DetectionResult:
        start = time.perf_counter()
        detections = self.detect(frame.image)
        end = time.perf_counter()
        return DetectionResult(detections=detections, frame_index=frame.index, source=frame.source,
                               frame_size=frame.size, inference_ms=(end - start) * 1000.0, timestamp=end)
