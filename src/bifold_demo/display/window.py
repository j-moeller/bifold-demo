"""On-screen window (OpenCV HighGUI). Must be used from the main thread on macOS."""

from __future__ import annotations

import cv2
import numpy as np

# Keys without a character of their own, as :meth:`Window.poll_key` returns them (beyond Unicode: never a character)
TAB, BACKSPACE = 9, 8
BACKTAB, LEFT, RIGHT, UP, DOWN = 0x110001, 0x110002, 0x110003, 0x110004, 0x110005  # Shift+Tab, arrow keys

# what the HighGUI backends report for them (cv2.waitKeyEx)
_SPECIAL = {
    25: BACKTAB, 127: BACKSPACE,  # macOS: Shift+Tab types a back-tab, the delete key a DEL
    0xF702: LEFT, 0xF703: RIGHT, 0xF700: UP, 0xF701: DOWN,  # macOS function keys
    11: BACKTAB,  # Qt (the Linux wheels): Shift+Tab as Qt's ASCII for it; other keys as X11 keysyms (below)
    0x250000: LEFT, 0x270000: RIGHT, 0x260000: UP, 0x280000: DOWN,  # Windows (cannot tell Shift+Tab from Tab)
}
# X11 keysyms (GTK and Qt); GTK adds the modifiers above bit 16
_GTK = {0xFE20: BACKTAB, 0xFF51: LEFT, 0xFF53: RIGHT, 0xFF52: UP, 0xFF54: DOWN}


def key_code(raw: int) -> int | None:
    """A ``cv2.waitKeyEx`` code as :meth:`Window.poll_key` returns it: characters as their code (letters
    lower-cased), Escape 27, Return 13, Tab ``TAB``; Shift+Tab, the arrows and Backspace as ``BACKTAB``, ``LEFT``,
    ``RIGHT``, ``UP``, ``DOWN`` and ``BACKSPACE``; ``None`` for other keys (function keys, ...) or no key."""
    if raw < 0:
        return None
    if raw in _SPECIAL:
        return _SPECIAL[raw]
    key = raw & 0xFFFF  # GTK puts the modifier keys above bit 16
    if key in _GTK:
        return _GTK[key]
    if 0xFF00 <= key <= 0xFF1F:  # GTK BackSpace, Tab, Return, Escape: their ASCII codes
        key &= 0xFF
    if key == 0 or key > 0xFF:  # Windows special keys have the low bits clear; other function keys
        return None
    return ord(chr(key).lower()) if chr(key).isalpha() else key


class Window:
    """Shows frames at their native resolution; can switch to fullscreen. Left clicks are collected for
    :meth:`poll_clicks`, in the pixels of the shown image (HighGUI maps them, also when the window is scaled)."""

    def __init__(self, title: str = "bifold demo", fullscreen: bool = False):
        self.title = title
        cv2.namedWindow(title, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
        self.fullscreen = False
        self._size: tuple[int, int] | None = None
        self._shown = False
        self._clicks: list[tuple[int, int]] = []
        try:
            cv2.setMouseCallback(title, self._on_mouse)
        except cv2.error:  # a backend without mouse events: no clicks
            pass
        if fullscreen:
            self.toggle_fullscreen()

    def _on_mouse(self, event: int, x: int, y: int, _flags: int, _param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self._clicks.append((int(x), int(y)))

    def poll_clicks(self) -> list[tuple[int, int]]:
        """The left clicks ``(x, y)`` since the last call (events arrive while :meth:`poll_key` runs)."""
        clicks, self._clicks = self._clicks, []
        return clicks

    def show(self, image: np.ndarray) -> None:
        h, w = image.shape[:2]
        if not self.fullscreen and self._size != (w, h):
            cv2.resizeWindow(self.title, w, h)  # 1:1 pixels, i.e. normal resolution
            self._size = (w, h)
        cv2.imshow(self.title, image)
        self._shown = True

    def poll_key(self, delay_ms: int = 1) -> int | None:
        """Process window events; return the pressed key (see :func:`key_code`) or ``None``."""
        return key_code(cv2.waitKeyEx(delay_ms))

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
