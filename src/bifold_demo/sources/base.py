"""Common interface for everything that produces video frames."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Frame:
    """One video frame.

    Attributes:
        image: HxWx3 ``uint8`` BGR pixels (OpenCV convention).
        index: increasing frame number within its source.
        timestamp: ``time.perf_counter()`` when the frame was produced.
        source: name of the source that produced it.
        time: the scene's time the frame shows (``None`` for a camera).
        windows: high-resolution windows delivered with the frame, of the same moment
            (:class:`bifold_demo.sensor.WindowResult`; empty when none were asked for).
    """

    image: np.ndarray
    index: int
    timestamp: float
    source: str
    time: float | None = None
    windows: list = field(default_factory=list)

    @property
    def size(self) -> tuple[int, int]:
        """``(width, height)`` in pixels."""
        return self.image.shape[1], self.image.shape[0]


class FrameSource(ABC):
    """A stream of images. ``start`` acquires resources, ``read`` returns the next frame."""

    name: str = "source"

    def start(self) -> None:
        """Open the device / create resources. Called once before the first ``read``."""

    @abstractmethod
    def read(self, timeout: float = 1.0) -> Frame | None:
        """Return the next frame, waiting up to ``timeout`` seconds; ``None`` if none is available."""

    def stop(self) -> None:
        """Release resources."""

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
