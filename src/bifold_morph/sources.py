"""Where frames come from: the webcam, a video file or a still image. Each keeps only its newest frame, so a slow
consumer always gets the present, not a backlog."""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass(frozen=True)
class Frame:
    image: np.ndarray  # BGR uint8
    index: int  # counts up from 1
    timestamp: float  # time.perf_counter() when it was taken


class Source:
    """A background thread reading frames; :meth:`read` waits for one newer than the last it returned."""

    name = "source"

    def __init__(self, mirror: bool = False):
        self.mirror = mirror
        self.error: str | None = None
        self._cond = threading.Condition()
        self._frame: Frame | None = None
        self._last = 0
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is None:
            self._open()
            self._running = True
            self._thread = threading.Thread(target=self._loop, name=self.name, daemon=True)
            self._thread.start()

    def _open(self) -> None:
        pass

    def _grab(self) -> np.ndarray | None:
        raise NotImplementedError

    def _loop(self) -> None:
        index = 0
        while self._running:
            image = self._grab()
            if image is None:
                continue
            if self.mirror:
                image = cv2.flip(image, 1)
            index += 1
            with self._cond:
                self._frame = Frame(image, index, time.perf_counter())
                self._cond.notify_all()

    def read(self, timeout: float = 1.0) -> Frame | None:
        self.start()
        with self._cond:
            self._cond.wait_for(lambda: self._frame is not None and self._frame.index > self._last, timeout)
            frame = self._frame
        if frame is not None and frame.index > self._last:
            self._last = frame.index
            return frame
        return None

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._close()

    def _close(self) -> None:
        pass


class Webcam(Source):
    """Camera ``index`` (0: the built-in FaceTime camera of a MacBook), mirrored by default like a mirror."""

    name = "webcam"

    def __init__(self, index: int = 0, width: int = 1280, height: int = 720, fps: float = 30.0, mirror: bool = True):
        super().__init__(mirror)
        self.index, self.width, self.height, self.fps = index, width, height, fps
        self._cap = None
        self._failures = 0

    def _open(self) -> None:
        backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
        cap = cv2.VideoCapture(self.index, backend)
        if not cap.isOpened():
            cap.release()
            hint = (" On macOS, allow camera access for your terminal app in System Settings > Privacy & Security > "
                    "Camera, then restart the terminal." if sys.platform == "darwin" else "")
            raise RuntimeError(f"cannot open camera {self.index}.{hint}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cap = cap

    def _grab(self) -> np.ndarray | None:
        ok, image = self._cap.read()
        if not ok:
            self._failures += 1
            if self._failures > 50:
                self.error = "the camera stopped delivering frames"
            time.sleep(0.01)
            return None
        self._failures, self.error = 0, None
        return image

    def _close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class VideoFile(Source):
    """A video file played in a loop at its own frame rate."""

    name = "video"

    def __init__(self, path, mirror: bool = False):
        super().__init__(mirror)
        self.path = Path(path)
        self._cap = None
        self._next = 0.0

    def _open(self) -> None:
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot open video {self.path}")
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._period = 1.0 / fps if fps and fps > 0 else 1 / 30
        self._next = time.perf_counter()

    def _grab(self) -> np.ndarray | None:
        delay = self._next - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        self._next = max(self._next + self._period, time.perf_counter() - self._period)
        ok, image = self._cap.read()
        if not ok:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, image = self._cap.read()
        return image if ok else None

    def _close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class StillImage(Source):
    """One picture, offered again and again (10 times a second) so the sliders can be tried on it."""

    name = "image"

    def __init__(self, path, mirror: bool = False):
        super().__init__(mirror)
        self.path = Path(path)
        self._image = None

    def _open(self) -> None:
        self._image = cv2.imread(str(self.path), cv2.IMREAD_COLOR)
        if self._image is None:
            raise RuntimeError(f"cannot read image {self.path}")

    def _grab(self) -> np.ndarray | None:
        time.sleep(0.1)
        return self._image.copy()


def open_source(spec: str, width: int = 1280, height: int = 720, mirror: bool | None = None) -> Source:
    """``spec``: a camera number ("0"), a video file or an image file. Cameras are mirrored unless ``mirror`` is
    False; files only if it is True."""
    if spec.isdigit():
        return Webcam(int(spec), width, height, mirror=True if mirror is None else mirror)
    path = Path(spec).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"no camera number and no file: {spec}")
    cls = StillImage if path.suffix.lower() in IMAGE_SUFFIXES else VideoFile
    return cls(path, mirror=bool(mirror))
