"""Webcam frame source (a background thread keeps only the newest frame)."""

from __future__ import annotations

import sys
import threading
import time

import cv2

from .base import Frame, FrameSource


class WebcamSource(FrameSource):
    """Captures from camera ``index`` (0 = built-in FaceTime camera on a MacBook).

    Args:
        width, height: requested resolution; the camera may pick the closest it supports.
        fps: requested frame rate.
        mirror: flip horizontally so the preview behaves like a mirror.
    """

    name = "webcam"

    def __init__(self, index: int = 0, width: int = 1280, height: int = 720, fps: float = 30.0, mirror: bool = True):
        self.index, self.width, self.height, self.fps, self.mirror = index, width, height, fps, mirror
        self._cap = None
        self._thread = None
        self._running = False
        self._cond = threading.Condition()
        self._frame: Frame | None = None
        self._last_returned = 0
        self.error: str | None = None

    def start(self) -> None:
        if self._cap is not None:
            return
        backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
        cap = cv2.VideoCapture(self.index, backend)
        if not cap.isOpened():
            cap.release()
            hint = (
                " On macOS, allow camera access for your terminal app in "
                "System Settings > Privacy & Security > Camera, then restart the terminal."
                if sys.platform == "darwin"
                else ""
            )
            raise RuntimeError(f"cannot open camera {self.index}.{hint}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cap = cap
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, name="webcam", daemon=True)
        self._thread.start()

    def _capture_loop(self) -> None:
        index = 0
        failures = 0
        while self._running:
            ok, image = self._cap.read()
            if not ok:
                failures += 1
                if failures > 50:
                    self.error = "camera stopped delivering frames"
                time.sleep(0.01)
                continue
            failures = 0
            self.error = None
            if self.mirror:
                image = cv2.flip(image, 1)
            index += 1
            with self._cond:
                self._frame = Frame(image=image, index=index, timestamp=time.perf_counter(), source=self.name)
                self._cond.notify_all()

    def read(self, timeout: float = 1.0) -> Frame | None:
        """Wait for a frame newer than the previous one returned (up to ``timeout`` seconds)."""
        if self._cap is None:
            self.start()
        with self._cond:
            self._cond.wait_for(lambda: self._frame is not None and self._frame.index > self._last_returned, timeout)
            frame = self._frame
        if frame is not None:
            self._last_returned = frame.index
        return frame

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._frame = None
        self._last_returned = 0
