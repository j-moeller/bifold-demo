"""Background detection so the video keeps its full frame rate while the model runs."""

from __future__ import annotations

import threading
import traceback

from ..sources.base import Frame
from .results import DetectionResult


class AsyncDetector:
    """Runs ``detector.detect_frame`` on a worker thread.

    Only the most recent submitted frame is kept: if the model is slower than the
    video, intermediate frames are skipped rather than queued (no growing lag).
    """

    def __init__(self, detector):
        self.detector = detector
        self.fps = 0.0  # completed inferences per second (smoothed)
        self.error: str | None = None
        self._cond = threading.Condition()
        self._pending: Frame | None = None
        self._result: DetectionResult | None = None
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> "AsyncDetector":
        if self._thread is None:
            self._running = True
            self._thread = threading.Thread(target=self._loop, name="detector", daemon=True)
            self._thread.start()
        return self

    def submit(self, frame: Frame) -> None:
        """Queue ``frame`` for detection, replacing any frame still waiting."""
        with self._cond:
            self._pending = frame
            self._cond.notify_all()

    def latest(self) -> DetectionResult | None:
        return self._result

    def wait_for(self, frame: Frame, timeout: float = 2.0) -> DetectionResult | None:
        """Block until the result for ``frame`` (or a newer frame) is available."""
        def ready():
            r = self._result
            return r is not None and r.source == frame.source and r.frame_index >= frame.index

        with self._cond:
            self._cond.wait_for(lambda: ready() or not self._running, timeout)
        return self._result

    def _loop(self) -> None:
        last_done = None
        while True:
            with self._cond:
                self._cond.wait_for(lambda: self._pending is not None or not self._running)
                if not self._running:
                    return
                frame, self._pending = self._pending, None
            try:
                result = self.detector.detect_frame(frame)
                self.error = None
            except Exception as exc:  # noqa: BLE001 - keep the demo alive, show the error on screen
                if self.error is None:
                    traceback.print_exc()
                self.error = f"{type(exc).__name__}: {exc}"
                continue
            if last_done is not None:
                interval = max(result.timestamp - last_done, 1e-6)
                self.fps = 1.0 / interval if self.fps == 0 else 0.9 * self.fps + 0.1 / interval
            last_done = result.timestamp
            with self._cond:
                self._result = result
                self._cond.notify_all()

    def stop(self) -> None:
        with self._cond:
            self._running = False
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
