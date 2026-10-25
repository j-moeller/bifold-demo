"""YOLO object detection via Ultralytics (runs on Apple-silicon GPUs through PyTorch MPS)."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..sources.base import Frame
from .results import Detection, DetectionResult

DEFAULT_MODEL = "yolo26n.pt"


def pick_device(device: str = "auto") -> str:
    """``"auto"`` picks Apple's GPU (``"mps"``), then CUDA, then the CPU."""
    if device != "auto":
        return device
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_model_path(model: str, models_dir: str | Path = "models") -> str:
    """Store bare official weight names (``yolo26n.pt``) in ``models_dir`` instead of the working directory."""
    path = Path(model)
    if path.exists() or path.parent != Path(".") or path.suffix != ".pt":
        return model
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    return str(Path(models_dir) / path.name)


class YoloDetector:
    """Wraps an Ultralytics YOLO model.

    Args:
        model: weights name or path (downloaded automatically for official models,
            e.g. ``yolo26n.pt``, ``yolo26s.pt``, ``yolo11n.pt``).
        device: ``"auto"``, ``"mps"``, ``"cpu"`` or ``"cuda"``.
        conf: minimum confidence for a detection.
        iou: overlap threshold for non-maximum suppression.
        imgsz: inference resolution (frames are letterboxed to it; boxes come back in frame pixels).
        classes: only report these classes (names like ``"person"`` or COCO ids).
    """

    def __init__(self, model: str = DEFAULT_MODEL, device: str = "auto", conf: float = 0.35, iou: float = 0.5,
                 imgsz: int = 640, classes=None, models_dir: str | Path = "models"):
        from ultralytics import YOLO

        self.model_name = Path(model).name
        self.model = YOLO(resolve_model_path(model, models_dir))
        self.names: dict[int, str] = dict(self.model.names)
        self.device = pick_device(device)
        self.conf, self.iou, self.imgsz = conf, iou, imgsz
        self.class_ids = self._class_ids(classes)
        self.warmup()

    def _class_ids(self, classes) -> list[int] | None:
        if not classes:
            return None
        by_name = {name.lower(): idx for idx, name in self.names.items()}
        ids = []
        for item in classes:
            if isinstance(item, int) or str(item).isdigit():
                ids.append(int(item))
            elif str(item).lower() in by_name:
                ids.append(by_name[str(item).lower()])
            else:
                raise ValueError(f"unknown class {item!r}; the model knows: {', '.join(self.names.values())}")
        return ids

    def warmup(self) -> None:
        """Run once on a blank image so the first real frame is fast; falls back to the CPU if the GPU fails."""
        blank = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        try:
            self.detect(blank)
        except Exception as exc:  # noqa: BLE001 - any accelerator failure
            if self.device == "cpu":
                raise
            print(f"[detector] {self.device} failed ({exc}); falling back to the CPU")
            self.device = "cpu"
            self.detect(blank)

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Detect objects in a BGR image."""
        result = self.model.predict(
            image, conf=self.conf, iou=self.iou, imgsz=self.imgsz, device=self.device,
            classes=self.class_ids, verbose=False,
        )[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        confidence = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)
        return [
            Detection(box=tuple(float(v) for v in box), confidence=float(c), class_id=int(k), label=self.names[int(k)])
            for box, c, k in zip(xyxy, confidence, class_ids)
        ]

    def detect_frame(self, frame: Frame) -> DetectionResult:
        start = time.perf_counter()
        detections = self.detect(frame.image)
        end = time.perf_counter()
        return DetectionResult(
            detections=detections, frame_index=frame.index, source=frame.source, frame_size=frame.size,
            inference_ms=(end - start) * 1000.0, timestamp=end,
        )
