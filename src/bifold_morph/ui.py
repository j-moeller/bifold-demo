"""The window: the video with the faces' labels, and a strip below it with the age and gender sliders.

Everything is drawn with OpenCV into the picture itself (no toolkit widgets), so it looks and works the same with
every HighGUI backend: drag a slider's knob or click anywhere on its track; double-click or right-click it to set it
back to 0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np

from .state import AgeGender

FONT = cv2.FONT_HERSHEY_SIMPLEX
BG, TEXT, DIM, TRACK = (32, 32, 32), (235, 235, 235), (150, 150, 150), (85, 85, 85)
ACCENT = (60, 170, 255)  # BGR: orange
FACE_BOX = (120, 220, 120)

# Keys without a character of their own, as Window.poll_key returns them (beyond Unicode: never a character)
TAB, BACKSPACE = 9, 8
BACKTAB, LEFT, RIGHT, UP, DOWN = 0x110001, 0x110002, 0x110003, 0x110004, 0x110005
_SPECIAL = {
    25: BACKTAB, 127: BACKSPACE,  # macOS: Shift+Tab types a back-tab, the delete key a DEL
    0xF702: LEFT, 0xF703: RIGHT, 0xF700: UP, 0xF701: DOWN,  # macOS function keys
    11: BACKTAB,  # Qt (the Linux wheels): Shift+Tab; other keys as X11 keysyms (below)
    0x250000: LEFT, 0x270000: RIGHT, 0x260000: UP, 0x280000: DOWN,  # Windows
}
_GTK = {0xFE20: BACKTAB, 0xFF51: LEFT, 0xFF53: RIGHT, 0xFF52: UP, 0xFF54: DOWN}  # X11 keysyms


def key_code(raw: int) -> int | None:
    """A ``cv2.waitKeyEx`` code as :meth:`Window.poll_key` returns it: characters as their code (letters
    lower-cased), Escape 27, Return 13; the arrows, Shift+Tab and Backspace as ``LEFT``, ``RIGHT``, ``UP``,
    ``DOWN``, ``BACKTAB`` and ``BACKSPACE``; ``None`` for other keys or no key."""
    if raw < 0:
        return None
    if raw in _SPECIAL:
        return _SPECIAL[raw]
    key = raw & 0xFFFF  # GTK puts the modifier keys above bit 16
    if key in _GTK:
        return _GTK[key]
    if 0xFF00 <= key <= 0xFF1F:  # GTK BackSpace, Tab, Return, Escape: their ASCII codes
        key &= 0xFF
    if key == 0 or key > 0xFF:
        return None
    return ord(chr(key).lower()) if chr(key).isalpha() else key


def ui_scale(width: int) -> float:
    return float(np.clip(width / 1280.0, 0.55, 2.5))


def text_size(text: str, scale: float, thickness: int) -> tuple[int, int]:
    (w, h), _ = cv2.getTextSize(text, FONT, scale, thickness)
    return w, h


@dataclass
class Slider:
    """A value from ``minimum`` to ``maximum`` (0 in the middle) with names for its two ends; ``describe`` turns
    the value into the text shown above it."""

    title: str
    minimum: float
    maximum: float
    left: str
    right: str
    describe: Callable[[float], str]
    step: float  # for the keyboard
    value: float = 0.0
    track: tuple[int, int, int] = field(default=(0, 0, 0), repr=False)  # x0, x1, y in the strip, set by draw()

    def set(self, value: float) -> None:
        self.value = float(np.clip(value, self.minimum, self.maximum))

    def nudge(self, steps: float) -> None:
        v = self.value + steps * self.step
        self.set(0.0 if abs(v) < self.step / 2 else v)

    def value_at(self, x: float) -> float:
        x0, x1, _ = self.track
        t = (x - x0) / max(x1 - x0, 1)
        value = self.minimum + float(np.clip(t, 0, 1)) * (self.maximum - self.minimum)
        snap = 0.03 * (self.maximum - self.minimum)  # the middle is easy to hit
        return 0.0 if abs(value) < snap else value

    def x_of(self, value: float) -> int:
        x0, x1, _ = self.track
        return round(x0 + (value - self.minimum) / (self.maximum - self.minimum) * (x1 - x0))


class Panel:
    """The strip with the sliders and a status line. :meth:`mouse` takes events in the strip's own pixels."""

    def __init__(self, sliders: list[Slider]):
        self.sliders = sliders
        self.selected = 0  # the slider the arrow keys move
        self._dragging: Slider | None = None
        self._reach = 14  # pixels round a track that still count as on it, set by draw()

    def height(self, width: int) -> int:
        return round((70 * len(self.sliders) + 44) * ui_scale(width))

    def draw(self, width: int, status: str, help_text: str) -> np.ndarray:
        s = ui_scale(width)
        strip = np.full((self.height(width), width, 3), BG, np.uint8)
        fs, th = 0.6 * s, max(1, round(1.4 * s))
        margin = round(24 * s)
        self._reach = max(8, round(20 * s))
        label_w = max(text_size(t, fs, th)[0] for sl in self.sliders for t in (sl.left, sl.right))
        for i, sl in enumerate(self.sliders):
            y = round((70 * i + 48) * s)
            x0, x1 = margin + label_w + round(18 * s), width - margin - label_w - round(18 * s)
            sl.track = (x0, x1, y)
            title = f"{sl.title}: {sl.describe(sl.value)}"
            tw, tht = text_size(title, fs, th)
            color = TEXT if i == self.selected else DIM
            cv2.putText(strip, title, ((x0 + x1 - tw) // 2, y - round(16 * s)), FONT, fs, color, th, cv2.LINE_AA)
            for text, x in ((sl.left, margin), (sl.right, x1 + round(18 * s))):
                cv2.putText(strip, text, (x, y + tht // 2), FONT, fs, DIM, th, cv2.LINE_AA)
            cv2.line(strip, (x0, y), (x1, y), TRACK, max(2, round(4 * s)), cv2.LINE_AA)
            mid, knob = sl.x_of(0.0), sl.x_of(sl.value)
            cv2.line(strip, (mid, y - round(8 * s)), (mid, y + round(8 * s)), DIM, max(1, round(2 * s)), cv2.LINE_AA)
            if knob != mid:
                cv2.line(strip, (mid, y), (knob, y), ACCENT, max(2, round(4 * s)), cv2.LINE_AA)
            cv2.circle(strip, (knob, y), round(10 * s), ACCENT, -1, cv2.LINE_AA)
            cv2.circle(strip, (knob, y), round(10 * s), TEXT, max(1, round(1.5 * s)), cv2.LINE_AA)
        y = strip.shape[0] - round(14 * s)
        fs2 = 0.5 * s
        cv2.putText(strip, status, (margin, y), FONT, fs2, TEXT, th, cv2.LINE_AA)
        hw, _ = text_size(help_text, fs2, th)
        if margin + text_size(status, fs2, th)[0] + round(30 * s) + hw < width:
            cv2.putText(strip, help_text, (width - margin - hw, y), FONT, fs2, DIM, th, cv2.LINE_AA)
        return strip

    def _hit(self, x: int, y: int) -> Slider | None:
        for sl in self.sliders:
            x0, x1, ty = sl.track
            if x0 - self._reach <= x <= x1 + self._reach and abs(y - ty) <= self._reach:
                return sl
        return None

    def mouse(self, event: int, x: int, y: int, flags: int) -> bool:
        """Handle a HighGUI mouse event at ``(x, y)`` in the strip; ``True`` if a slider changed."""
        if event in (cv2.EVENT_LBUTTONDBLCLK, cv2.EVENT_RBUTTONDOWN):
            sl = self._hit(x, y)
            if sl is not None:
                sl.set(0.0)
                self.selected = self.sliders.index(sl)
                return True
        elif event == cv2.EVENT_LBUTTONDOWN:
            self._dragging = self._hit(x, y)
            if self._dragging is not None:
                self.selected = self.sliders.index(self._dragging)
                self._dragging.set(self._dragging.value_at(x))
                return True
        elif event == cv2.EVENT_MOUSEMOVE and self._dragging is not None:
            if flags & cv2.EVENT_FLAG_LBUTTON:
                self._dragging.set(self._dragging.value_at(x))
                return True
            self._dragging = None
        elif event == cv2.EVENT_LBUTTONUP:
            self._dragging = None
        return False

    def key(self, key: int) -> bool:
        """Arrow keys: up/down pick a slider, left/right move it. ``True`` if the key was one of them."""
        if key in (UP, DOWN):
            self.selected = (self.selected + (1 if key == DOWN else -1)) % len(self.sliders)
        elif key in (LEFT, RIGHT):
            self.sliders[self.selected].nudge(1 if key == RIGHT else -1)
        else:
            return False
        return True


def describe_label(before: AgeGender | None, after: AgeGender | None, edited: bool) -> str:
    """MiVOLO's reading of a face, e.g. "31 -> 52 y, male 98% -> female 81%"."""
    if before is None and after is None:
        return ""
    def one(r: AgeGender) -> tuple[str, str]:
        return f"{r.age:.0f}", f"{r.gender} {r.confidence:.0%}"
    if not edited or after is None or before is None:
        age, gender = one(after or before)
        return f"{age} y, {gender}"
    (a0, g0), (a1, g1) = one(before), one(after)
    return f"{a0} -> {a1} y, {g0} -> {g1}"


def draw_face(image: np.ndarray, box, label: str, with_box: bool) -> None:
    """A thin box round a face and ``label`` under it (above it at the bottom edge)."""
    h, w = image.shape[:2]
    s = ui_scale(w)
    x, y, bw, bh = box
    x0, y0, x1, y1 = round(x), round(y), round(x + bw), round(y + bh)
    if with_box:
        cv2.rectangle(image, (x0, y0), (x1, y1), FACE_BOX, max(1, round(1.5 * s)), cv2.LINE_AA)
    if not label:
        return
    fs, th = 0.55 * s, max(1, round(1.3 * s))
    tw, tht = text_size(label, fs, th)
    pad = max(3, round(5 * s))
    left = int(np.clip((x0 + x1 - tw) // 2 - pad, 0, max(0, w - tw - 2 * pad)))
    top = y1 + 4 if y1 + tht + 3 * pad + 4 <= h else max(0, y0 - tht - 3 * pad - 4)
    cv2.rectangle(image, (left, top), (left + tw + 2 * pad, top + tht + 2 * pad), BG, -1)
    cv2.putText(image, label, (left + pad, top + pad + tht), FONT, fs, TEXT, th, cv2.LINE_AA)


class Window:
    """An OpenCV HighGUI window (main thread only on macOS) showing images at their own size, or fullscreen.
    Mouse events go to ``on_mouse(event, x, y, flags)`` in the pixels of the shown image (HighGUI maps them, also
    when the window is scaled)."""

    def __init__(self, title: str, on_mouse=None, fullscreen: bool = False):
        self.title = title
        cv2.namedWindow(title, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
        self.fullscreen = False
        self._size: tuple[int, int] | None = None
        self._shown = False
        if on_mouse is not None:
            try:
                cv2.setMouseCallback(title, lambda e, x, y, f, _p: on_mouse(e, x, y, f))
            except cv2.error:  # a backend without mouse events: keyboard only
                pass
        if fullscreen:
            self.toggle_fullscreen()

    def show(self, image: np.ndarray) -> None:
        h, w = image.shape[:2]
        if not self.fullscreen and self._size != (w, h):
            cv2.resizeWindow(self.title, w, h)
            self._size = (w, h)
        cv2.imshow(self.title, image)
        self._shown = True

    def poll_key(self, delay_ms: int = 1) -> int | None:
        return key_code(cv2.waitKeyEx(delay_ms))

    def is_open(self) -> bool:
        if not self._shown:
            return True
        try:
            return cv2.getWindowProperty(self.title, cv2.WND_PROP_VISIBLE) != 0
        except cv2.error:
            return False

    def toggle_fullscreen(self) -> None:
        self.fullscreen = not self.fullscreen
        mode = cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL
        cv2.setWindowProperty(self.title, cv2.WND_PROP_FULLSCREEN, mode)
        if not self.fullscreen:
            self._size = None

    def close(self) -> None:
        try:
            cv2.destroyWindow(self.title)
            cv2.waitKey(1)
        except cv2.error:
            pass
