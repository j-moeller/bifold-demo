"""Detector output types."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    """One detected object; ``box`` is ``(x1, y1, x2, y2)`` in pixels of the analysed frame."""

    box: tuple[float, float, float, float]
    confidence: float
    class_id: int
    label: str


@dataclass(frozen=True)
class DetectionResult:
    """All detections for one frame."""

    detections: list[Detection]
    frame_index: int
    source: str
    frame_size: tuple[int, int]  # (width, height) of the analysed frame
    inference_ms: float
    timestamp: float  # time.perf_counter() when the result was produced
