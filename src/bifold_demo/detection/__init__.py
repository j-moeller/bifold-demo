"""Object detection on video frames."""

from .results import Detection, DetectionResult
from .worker import AsyncDetector
from .yolo import DEFAULT_MODEL, YoloDetector, pick_device

__all__ = ["DEFAULT_MODEL", "AsyncDetector", "Detection", "DetectionResult", "YoloDetector", "pick_device"]
