"""On-screen window (OpenCV HighGUI). Must be used from the main thread on macOS."""

from __future__ import annotations

import cv2
import numpy as np


class Window:
    """Shows frames at their native resolution; can switch to fullscreen."""

    def __init__(self, title: str = "bifold demo", fullscreen: bool = False):
        self.title = title
        cv2.namedWindow(title, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
        self.fullscreen = False
        self._size: tuple[int, int] | None = None
        self._shown = False
        if fullscreen:
            self.toggle_fullscreen()

    def show(self, image: np.ndarray) -> None:
        h, w = image.shape[:2]
        if not self.fullscreen and self._size != (w, h):
            cv2.resizeWindow(self.title, w, h)  # 1:1 pixels, i.e. normal resolution
            self._size = (w, h)
        cv2.imshow(self.title, image)
        self._shown = True

    def poll_key(self, delay_ms: int = 1) -> int | None:
        """Process window events; return the pressed key code (lower-cased letters) or ``None``."""
        key = cv2.waitKey(delay_ms)
        if key < 0:
            return None
        key &= 0xFF
        return ord(chr(key).lower()) if chr(key).isalpha() else key

    def is_open(self) -> bool:
        if not self._shown:
            return True
        try:
            # 0 means the user closed the window; -1 means the backend cannot tell
            return cv2.getWindowProperty(self.title, cv2.WND_PROP_VISIBLE) != 0
        except cv2.error:
            return False

    def toggle_fullscreen(self) -> None:
        self.fullscreen = not self.fullscreen
        mode = cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL
        cv2.setWindowProperty(self.title, cv2.WND_PROP_FULLSCREEN, mode)
        if not self.fullscreen:
            self._size = None  # restore native size on the next frame

    def close(self) -> None:
        try:
            cv2.destroyWindow(self.title)
            cv2.waitKey(1)
        except cv2.error:
            pass
