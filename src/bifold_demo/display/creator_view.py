"""The character creator's screen: the avatar on a turntable, rendered off-screen, and the creator's rows beside it.

:class:`CreatorView` draws the screen of a :class:`~bifold_demo.creator.Creator` over a frame of the demo (the fair,
dimmed, stays behind it) and turns clicks into choices: every chip, text box and button it draws is a place to click
(:attr:`CreatorView.hits`). The rows are those of :data:`~bifold_demo.creator.ROWS`, one line each: text chips (the
avatars, the faces, the presets, the items, the saved characters), colour chips, pattern thumbnails (in each
pattern's own colours) and the fabric variations as small tiles. A row whose chips do not fit scrolls to keep the
one in use in sight (``<`` and ``>`` mark that there are more). Under the rows: what the row in focus is set to,
what is going on (with a spinner) and the keys.

The turntable has a renderer of its own, made on the first draw on the thread that draws (OpenGL contexts belong to
their thread: the demo's main thread); without OpenGL the screen still works, with the reason where the avatar would
be.
"""

from __future__ import annotations

import time
import traceback

import cv2
import numpy as np

from .overlay import FONT, _ui_scale

TURNTABLE = 0.34  # share of the picture's width the turntable takes
BACKDROP = (52, 54, 62)  # RGB behind the avatar
DIM = 0.3  # how bright the demo's picture stays behind the screen
ACCENT = (0, 200, 255)  # BGR: the row in focus, the choice in use
TEXT, GREY, DARK = (235, 235, 235), (150, 150, 150), (40, 40, 44)
CHIP = (70, 72, 80)
PLACEHOLDERS = {"name": "Enter, then type a name", "fabric": "Enter, then describe a fabric: blue paisley silk"}
EMPTY = {"saved": "none saved yet", "variation": "describe a fabric above", "avatar": "no avatars prepared"}


def _text(image, text: str, x: int, y: int, scale: float, colour=TEXT, thickness: int = 1) -> None:
    cv2.putText(image, text, (int(x), int(y)), FONT, scale, colour, thickness, cv2.LINE_AA)


def _width(text: str, scale: float, thickness: int = 1) -> int:
    return cv2.getTextSize(text, FONT, scale, thickness)[0][0]


def _fit(text: str, width: int, scale: float, thickness: int = 1) -> str:
    """``text`` cut (with ``...``) to fit ``width`` pixels."""
    if _width(text, scale, thickness) <= width:
        return text
    while text and _width(text + "...", scale, thickness) > width:
        text = text[:-1]
    return text + "..."


class CreatorView:
    """Draws the creator's screen and maps clicks on it to the creator (see the module docs)."""

    def __init__(self, creator):
        self.creator = creator
        self.hits: list[tuple] = []  # (x0, y0, x1, y1, row or action, index) of what was drawn last
        self._renderer = None
        self._error: str | None = None  # why the turntable cannot be drawn
        self._screen_error: str | None = None
        self._scene = None
        self._person = None
        self._t0 = time.perf_counter()
        self._last = None
        self._thumbs: dict = {}

    def release(self) -> None:
        """Free the turntable's renderer (on the thread that drew)."""
        if self._renderer is not None:
            self._renderer.release()
            self._renderer = None
        self._scene = self._person = None

    # -- drawing ------------------------------------------------------------------------------------------------

    def draw(self, image: np.ndarray) -> np.ndarray:
        """The screen over ``image`` (a BGR frame of the demo), as a new image of its size."""
        canvas = (image.astype(np.float32) * DIM).astype(np.uint8)
        h, w = canvas.shape[:2]
        s = _ui_scale(canvas)
        self.hits = []
        tw = max(120, round(w * TURNTABLE))
        try:
            canvas[:, :tw] = self._turntable(tw, h, s)
            self._buttons(canvas, tw, h, s)
            self._rows(canvas, tw, w, h, s)
        except Exception as exc:  # noqa: BLE001 - a bug in the screen never stops the demo (Esc still closes it)
            if self._screen_error != str(exc):
                traceback.print_exc()
                self._screen_error = str(exc)
            _text(canvas, f"the creator's screen failed: {type(exc).__name__}: {exc}", round(12 * s), h // 2,
                  0.5 * s, (80, 80, 255))
        return canvas

    def _turntable(self, width: int, height: int, s: float) -> np.ndarray:
        backdrop = np.empty((height, width, 3), np.uint8)
        backdrop[:] = BACKDROP[::-1]
        person = self.creator.person
        message = None
        if person is None:
            message = "preparing the avatar ..." if self.creator.avatar else "no avatar"
        elif self._error is None:
            try:
                return self._render(person, width, height)
            except Exception as exc:  # noqa: BLE001 - no OpenGL here: the rest of the screen still works
                self._error = f"cannot draw the avatar: {type(exc).__name__}: {exc}"
                print(f"[creator] {self._error}")
        message = message or self._error
        y = height // 2
        for line in _wrap(message, width - round(24 * s), 0.5 * s):
            _text(backdrop, line, round(12 * s), y, 0.5 * s, GREY)
            y += round(22 * s)
        return backdrop

    def _render(self, person, width: int, height: int) -> np.ndarray:
        from ..scene import Camera, Scene
        from ..scene.renderer import Renderer

        if self._renderer is None or (self._renderer.width, self._renderer.height) != (width, height):
            if self._renderer is not None:
                self._renderer.release()
            self._renderer = Renderer(width, height)
        if self._person is not person:
            self._scene = Scene(background=BACKDROP, ambient=0.55, light_direction=(-0.3, -0.6, -1.0))
            self._scene.add(person)
            self._person = person
            model = person.skinned_model
            tall = float(model.metadata.get("height") or getattr(model, "height", 0) or 1.75)
            # the whole body and a hat, the feet above the buttons
            self._scene.camera = Camera(position=(0.0, 0.58 * tall, 2.6 * tall), target=(0.0, 0.5 * tall, 0.0),
                                        fov=30.0)
        t = time.perf_counter() - self._t0
        dt = 0.0 if self._last is None else max(0.0, min(t - self._last, 0.25))
        self._last = t
        self._scene.update(t, dt)
        return self._renderer.render(self._scene)

    def _buttons(self, canvas, tw: int, h: int, s: float) -> None:
        """The turntable's buttons along its foot."""
        labels = [("<", ("turn", -45.0)), ("front", ("front", None)), ("back", ("back", None)), (">", ("turn", 45.0)),
                  ("stop" if self.creator.turning else "turn", ("spin", None))]
        scale, pad, gap = 0.45 * s, round(8 * s), round(6 * s)
        widths = [_width(text, scale) + 2 * pad for text, _ in labels]
        x = max(round(6 * s), (tw - sum(widths) - gap * (len(widths) - 1)) // 2)
        y0, y1 = h - round(40 * s), h - round(14 * s)
        for (text, (action, value)), width in zip(labels, widths):
            cv2.rectangle(canvas, (x, y0), (x + width, y1), CHIP, -1)
            _text(canvas, text, x + pad, y1 - round(8 * s), scale)
            self.hits.append((x, y0, x + width, y1, action, value))
            x += width + gap
        name = self.creator.avatar or ""
        _text(canvas, _fit(name, tw - round(20 * s), 0.5 * s), round(10 * s), round(24 * s), 0.5 * s, GREY)

    def _rows(self, canvas, tw: int, w: int, h: int, s: float) -> None:
        from ..creator import ROWS, TITLES

        c = self.creator
        x0, x1 = tw + round(18 * s), w - round(14 * s)
        panel = canvas[:, tw:]  # the demo's picture shows faintly through
        panel[:] = (panel * 0.4 + np.array(DARK, np.float32) * 0.6).astype(np.uint8)
        _text(canvas, "Character creator", x0, round(30 * s), 0.7 * s, TEXT, 2)
        state = f"{c.name or 'no name yet'} ({c.state()})"
        _text(canvas, _fit(state, x1 - x0 - round(230 * s), 0.5 * s), x0 + round(230 * s), round(30 * s), 0.5 * s,
              GREY)
        top, bottom = round(44 * s), h - round(78 * s)
        rh = max(14, min(round(30 * s), (bottom - top) // len(ROWS)))
        label_w = round(96 * s)
        for k, row in enumerate(ROWS):
            y = top + k * rh
            focused = k == c.row
            if focused:
                cv2.rectangle(canvas, (x0 - round(8 * s), y + 1), (x1 + round(6 * s), y + rh - 1), (64, 66, 74), -1)
                cv2.rectangle(canvas, (x0 - round(8 * s), y + 1), (x0 - round(5 * s), y + rh - 1), ACCENT, -1)
            _text(canvas, TITLES[row], x0, y + round(rh * 0.66), 0.45 * s, ACCENT if focused else GREY)
            self._row(canvas, row, x0 + label_w, x1, y, rh, s, focused)
        row = ROWS[c.row]
        lines = [(f"{TITLES[row]}: {c.value(row) if row not in ('name', 'fabric') else c.text(row)}", TEXT),
                 (self._status(), ACCENT if c.busy() else GREY), (c.help(), GREY)]
        y = h - round(58 * s)
        for text, colour in lines:
            _text(canvas, _fit(text, x1 - x0, 0.45 * s), x0, y, 0.45 * s, colour)
            y += round(21 * s)

    def _status(self) -> str:
        busy = self.creator.busy()
        if busy:
            spinner = "|/-\\"[int(time.perf_counter() * 8) % 4]
            return f"{spinner} " + "; ".join(busy)
        return self.creator.last_fabric_note or "ready"

    def _row(self, canvas, row: str, x0: int, x1: int, y: int, rh: int, s: float, focused: bool) -> None:
        from ..creator import TEXT_ROWS

        if row in TEXT_ROWS:
            self._text_box(canvas, row, x0, x1, y, rh, s)
        elif row == "scale":
            self._chips(canvas, row, ["smaller", "larger"], None, x0, x1, y, rh, s, focused,
                        value=self.creator.value("scale"))
        else:
            c = self.creator
            self._chips(canvas, row, c.choices(row), c.current(row), x0, x1, y, rh, s, focused)

    def _text_box(self, canvas, row: str, x0: int, x1: int, y: int, rh: int, s: float) -> None:
        c = self.creator
        editing = c.editing == row
        right = x0 + round((x1 - x0) * (0.55 if row == "fabric" else 0.6))
        top, bottom = y + round(4 * s), y + rh - round(4 * s)
        cv2.rectangle(canvas, (x0, top), (right, bottom), (30, 30, 34), -1)
        cv2.rectangle(canvas, (x0, top), (right, bottom), ACCENT if editing else CHIP, 1)
        self.hits.append((x0, top, right, bottom, row, None))
        text = c.text(row)
        scale = 0.45 * s
        if editing:
            caret = "|" if int(time.perf_counter() * 2) % 2 == 0 else " "
            shown, colour = text + caret, TEXT
        else:
            shown, colour = (text, TEXT) if text else (PLACEHOLDERS[row], GREY)
        inner = right - x0 - round(12 * s)
        while _width(shown, scale) > inner and len(shown) > 1:  # the end of a long text stays in sight
            shown = shown[1:]
        _text(canvas, shown, x0 + round(6 * s), bottom - round(7 * s), scale, colour)
        if row == "fabric":
            x = right + round(8 * s)
            width = _width("make", scale) + round(16 * s)
            cv2.rectangle(canvas, (x, top), (x + width, bottom), CHIP, -1)
            _text(canvas, "make", x + round(8 * s), bottom - round(7 * s), scale)
            self.hits.append((x, top, x + width, bottom, "make", None))
            note = c.fabric_status()
            if note:
                left = x + width + round(8 * s)
                _text(canvas, _fit(note, x1 - left, 0.4 * s), left, bottom - round(7 * s), 0.4 * s, GREY)

    def _chips(self, canvas, row: str, choices, current, x0: int, x1: int, y: int, rh: int, s: float,
               focused: bool, value: str = "") -> None:
        from ..creator import ENTER_ROWS

        c = self.creator
        scale = 0.42 * s
        top, bottom = y + round(4 * s), y + rh - round(4 * s)
        side = bottom - top
        if not choices:
            _text(canvas, EMPTY.get(row, "-"), x0, bottom - round(6 * s), scale, GREY)
            return
        kinds = [self._kind(row, choice) for choice in choices]
        widths = [side if kind != "text" else _width(choice, scale) + round(14 * s)
                  for kind, choice in zip(kinds, choices)]
        gap = round(5 * s)
        cursor = c.cursor.get(row) if row in ENTER_ROWS else None
        focus = cursor if cursor is not None else (current if current is not None else 0)
        focus = min(max(focus, 0), len(choices) - 1)
        if value:
            x1 -= _width(value, scale) + round(10 * s)
            _text(canvas, value, x1 + round(10 * s), bottom - round(6 * s), scale, GREY)
        arrow = round(12 * s)
        start = 0  # the first chip shown: so that the one in focus fits
        while start < focus and x0 + arrow + sum(widths[start:focus + 1]) + gap * (focus - start) > x1 - arrow:
            start += 1
        x = x0 + (arrow if start > 0 else 0)
        if start > 0:
            _text(canvas, "<", x0, bottom - round(6 * s), scale, GREY)
            self.hits.append((x0, top, x0 + arrow, bottom, row, start - 1))
        for index in range(start, len(choices)):
            width = widths[index]
            if x + width > x1 - (arrow if index < len(choices) - 1 else 0):
                _text(canvas, ">", x1 - arrow + round(2 * s), bottom - round(6 * s), scale, GREY)
                self.hits.append((x1 - arrow, top, x1, bottom, row, index))
                break
            self._chip(canvas, row, index, choices[index], kinds[index], (x, top, x + width, bottom), scale, s,
                       chosen=index == current, marked=focused and index == cursor,
                       enabled=c.enabled(row, index))
            self.hits.append((x, top, x + width, bottom, row, index))
            x += width + gap

    def _kind(self, row: str, choice: str) -> str:
        if row in ("colour", "colour 2", "variation"):
            return "swatch"
        if row == "pattern" and choice not in ("own", "fabric"):
            return "swatch"
        return "text"

    def _chip(self, canvas, row, index, choice, kind, box, scale, s, chosen, marked, enabled) -> None:
        x0, y0, x1, y1 = box
        if kind == "text":
            fill = ACCENT if chosen else CHIP
            cv2.rectangle(canvas, (x0, y0), (x1, y1), fill, -1)
            colour = (20, 20, 20) if chosen else (TEXT if enabled else (110, 110, 110))
            _text(canvas, choice, x0 + round(7 * s), y1 - round(6 * s), scale, colour)
        else:
            tile = self._swatch(row, index, choice, x1 - x0)
            if tile is not None:
                canvas[y0:y1, x0:x1] = tile
            if not enabled:
                canvas[y0:y1, x0:x1] //= 3
            if chosen:
                cv2.rectangle(canvas, (x0 - 2, y0 - 2), (x1 + 1, y1 + 1), (255, 255, 255), 2)
        if marked:
            cv2.rectangle(canvas, (x0 - 3, y0 - 3), (x1 + 2, y1 + 2), ACCENT, 1)

    def _swatch(self, row: str, index: int, choice: str, side: int) -> np.ndarray | None:
        """A ``side`` x ``side`` BGR picture of a colour, a pattern (in its own colours; plain in the garment's) or a
        fabric variation (kept once made)."""
        from ..characters.outfit import Pattern, color
        from ..creator import primary_of

        if row in ("colour", "colour 2"):
            key = ("colour", choice, side)
        elif row == "pattern" and choice == "plain":
            key = ("colour", tuple(primary_of(self.creator.fills.get(self.creator.garment)) or color("navy")), side)
        elif row == "pattern":
            key = ("pattern", choice, side)
        else:
            variations = self.creator.variations
            if index >= len(variations):
                return None
            key = ("variation", id(variations[index]), side)
        tile = self._thumbs.get(key)
        if tile is None:
            if key[0] == "colour":
                tile = np.empty((side, side, 3), np.uint8)
                tile[:] = color(key[1])[::-1]
            else:
                rgb = Pattern(choice).image if key[0] == "pattern" else self.creator.variations[index].image
                rgb = np.clip(np.asarray(rgb, np.float32), 0, 255).astype(np.uint8)
                rgb = np.tile(rgb, (2, 2, 1))
                tile = cv2.cvtColor(cv2.resize(rgb, (side, side), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR)
            if len(self._thumbs) > 400:
                self._thumbs.clear()
            self._thumbs[key] = tile
        return tile

    # -- clicks -------------------------------------------------------------------------------------------------

    def click(self, point) -> str | None:
        """A left click at ``point`` (pixels of the last screen drawn): what it chose, or ``None``."""
        x, y = point
        for x0, y0, x1, y1, what, index in reversed(self.hits):
            if x0 <= x <= x1 and y0 <= y <= y1:
                return self._clicked(what, index)
        return None

    def _clicked(self, what: str, value):
        c = self.creator
        if what == "turn":
            c.turn(value)
        elif what in ("front", "back"):
            c.turn(to=0.0 if what == "front" else 180.0)
        elif what == "spin":
            c.turn(turning=not c.turning)
        elif what == "make":
            return c.describe_fabric(c.fabric_text) if c.fabric_text.strip() else c.click("fabric")
        else:
            return c.click(what, value)
        return None


def _wrap(text: str, width: int, scale: float) -> list[str]:
    lines, line = [], ""
    for word in str(text).split():
        candidate = f"{line} {word}".strip()
        if line and _width(candidate, scale) > width:
            lines.append(line)
            line = word
        else:
            line = candidate
    return lines + ([line] if line else [])
