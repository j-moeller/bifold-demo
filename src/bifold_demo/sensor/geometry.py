"""Where windows are: a declared sensor whose full frame is the scene camera's view, its binning levels, windows
snapped to its pixel grid, and the maps between view, sensor and window pixels.

**Coordinates.** Pixel coordinates are continuous and image-style: x to the right, y down, ``(0, 0)`` the top-left
corner of the top-left pixel; pixel ``(i, j)`` is the square ``[i, i + 1) x [j, j + 1)`` (its centre at
``(i + 0.5, j + 0.5)``). In these coordinates every map between the main view, the sensor and a window is a scale and
a shift, exactly, so boxes found in a window map back to the main frame without rounding.

**Levels.** A window at level ``k`` has one pixel for every ``k x k`` sensor pixels: a binned read-out of the sensor.
Only binnings that divide both sides of the sensor are levels, so that the whole frame at level ``k`` is a sensor of
``width / k`` x ``height / k`` pixels and a window at that level is exactly a crop of it. Level 1 is the sensor itself.
:meth:`Sensor.plan` picks the coarsest level whose face is still as large as asked for (or level 1, when no level is),
then coarser ones while the window has more pixels than allowed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from ..scene.scene import SubCamera
from ..scene.transforms import crop_matrix

ASPECT_TOLERANCE = 0.005  # how much (relative) a sensor's width / height may differ from the view's
DEFAULT_SCALE = 6  # the default sensor: six times the view in each direction (7680 x 4320 for a 1280 x 720 view)
_EPS = 1e-6  # in level pixels: a region edge this close to a pixel edge counts as on it


@dataclass(frozen=True)
class Sensor:
    """A declared sensor of ``width`` x ``height`` pixels whose full frame is the main view's: the scene camera's
    picture, which the main view shows at ``view`` = ``(width, height)`` pixels.

    The sensor and the view must have the same shape (width / height within ``ASPECT_TOLERANCE``); :meth:`with_lines`
    and :meth:`scaled` make one that has. :attr:`levels` are its binnings (see the module docs).
    """

    width: int
    height: int
    view: tuple[int, int]

    def __post_init__(self):
        width, height = int(self.width), int(self.height)
        view = (int(self.view[0]), int(self.view[1]))
        if min(width, height, *view) <= 0:
            raise ValueError(f"a sensor and a view need pixels: sensor {width}x{height}, view {view[0]}x{view[1]}")
        if abs((width / height) / (view[0] / view[1]) - 1.0) > ASPECT_TOLERANCE:
            raise ValueError(f"a {width}x{height} sensor does not have the {view[0]}x{view[1]} view's shape; use "
                             f"Sensor.with_lines({height}, {view})")
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "height", height)
        object.__setattr__(self, "view", view)

    @classmethod
    def with_lines(cls, lines: int, view) -> Sensor:
        """A sensor ``lines`` pixels high with the view's shape (``with_lines(2160, (1280, 720))``: 3840 x 2160)."""
        w, h = int(view[0]), int(view[1])
        return cls(max(1, round(int(lines) * w / h)), int(lines), (w, h))

    @classmethod
    def scaled(cls, scale: float, view) -> Sensor:
        """A sensor ``scale`` times the view in each direction (rounded to whole lines)."""
        return cls.with_lines(max(1, round(int(view[1]) * float(scale))), view)

    def __str__(self) -> str:
        return f"{self.width}x{self.height}"

    @property
    def size(self) -> tuple[int, int]:
        return self.width, self.height

    @property
    def view_aspect(self) -> float:
        """Width / height of the whole picture (the view's: the projection is the view's)."""
        return self.view[0] / self.view[1]

    @property
    def scale_xy(self) -> tuple[float, float]:
        """Sensor pixels per view pixel, across and down (the same but for rounding)."""
        return self.width / self.view[0], self.height / self.view[1]

    @property
    def scale(self) -> float:
        """Sensor pixels per view pixel."""
        sx, sy = self.scale_xy
        return (sx + sy) / 2

    @cached_property
    def levels(self) -> tuple[int, ...]:
        """The binnings ``k`` that divide both sides, finest first (1 is always one)."""
        g = math.gcd(self.width, self.height)
        small = [k for k in range(1, math.isqrt(g) + 1) if g % k == 0]
        return tuple(sorted({*small, *(g // k for k in small)}))

    def frame(self, level: int = 1) -> tuple[int, int]:
        """The whole frame's size at ``level``."""
        self._check_level(level)
        return self.width // level, self.height // level

    def _check_level(self, level: int) -> None:
        if level not in self.levels:
            raise ValueError(f"{level} is no level of a {self} sensor (its levels: {', '.join(map(str, self.levels))})")

    def from_view(self, points) -> np.ndarray:
        """View pixels ``(..., 2)`` to sensor pixels."""
        return np.asarray(points, dtype=np.float64) * np.array(self.scale_xy)

    def to_view(self, points) -> np.ndarray:
        """Sensor pixels ``(..., 2)`` to view pixels."""
        return np.asarray(points, dtype=np.float64) / np.array(self.scale_xy)

    def window(self, region, level: int = 1, multiple: int = 1) -> Window:
        """The smallest window at ``level`` that covers ``region`` = ``(x0, y0, x1, y1)`` (view pixels), its sides
        grown to a multiple of ``multiple`` pixels (about the region's centre), and moved (or cut) to lie within the
        frame: a sensor has no pixels beyond its edges, so a region at an edge is off-centre in its window.

        Its bottom-left corner lies on an even pixel of the frame at that level, as OpenGL counts them (from the
        bottom): a renderer's 2x2 pixel quads, and with them its texture filtering, then line up with those of a
        render of the whole frame. The window moves by a pixel for that, or grows by ``multiple`` more when it has
        no room; a region at the frame's edge may leave no way (an odd frame or ``multiple``)."""
        self._check_level(level)
        x0, y0, x1, y1 = (float(v) for v in region)
        if not (x1 > x0 and y1 > y0):
            raise ValueError(f"empty region {tuple(region)}")
        sx, sy = self.scale_xy
        fw, fh = self.frame(level)
        left, width = _span(x0 * sx / level, x1 * sx / level, fw, multiple, lambda _length: 0)
        top, height = _span(y0 * sy / level, y1 * sy / level, fh, multiple, lambda length: (fh - length) % 2)
        return Window(self, level, left, top, width, height)

    def plan(self, region, ied: float | None = None, face_px: float | None = None, max_pixels: int | None = None,
             multiple: int = 1, max_side: int | None = None) -> Window:
        """The window for ``region`` (view pixels): at the coarsest level at which a face with ``ied`` view pixels
        between the eyes still has ``face_px`` (at level 1 when no level gets there, or when ``ied`` or ``face_px`` is
        not given), then coarser while it has more than ``max_pixels`` pixels or a side longer than ``max_side`` (what
        a renderer can draw). The face then has ``ied * window.scale`` pixels between the eyes."""
        levels = self.levels
        start = 0
        if ied is not None and ied > 0 and face_px is not None and face_px > 0:
            reaching = [i for i, k in enumerate(levels) if ied * self.scale / k >= face_px]
            start = reaching[-1] if reaching else 0
        window = None
        for k in levels[start:]:
            window = self.window(region, k, multiple)
            if (max_pixels is None or window.pixels <= max_pixels) and \
                    (max_side is None or max(window.size) <= max_side):
                break
        return window


def _span(lo: float, hi: float, frame: int, multiple: int, parity) -> tuple[int, int]:
    """``(start, length)`` of the pixels of a frame ``frame`` pixels long that cover ``[lo, hi)`` (what of it lies in
    the frame): the length grown to a multiple of ``multiple`` about the middle and cut to the frame, the start moved
    into the frame and, where there is room, to the parity ``parity(length)``, growing the length by ``multiple``
    once more for it if need be (else the start nearest the middle)."""
    a = math.floor(lo + _EPS)
    b = max(a + 1, math.ceil(hi - _EPS))
    need, step = b - a, max(1, int(multiple))
    first = min(-(-need // step) * step, frame)
    inside_a, inside_b = min(max(a, 0), frame), max(min(b, frame), 0)
    nearest = None
    for length in (first, min(first + step, frame)):
        lowest, highest = max(inside_b - length, 0), min(inside_a, frame - length)  # the starts that cover it
        centred = min(max(a - (length - need) // 2, lowest), highest)
        nearest = nearest or (centred, length)
        middle = a - (length - need) / 2
        for start in sorted((centred, centred - 1, centred + 1), key=lambda s: (abs(s - middle), s)):
            if lowest <= start <= highest and start % 2 == parity(length):
                return start, length
    return nearest


@dataclass(frozen=True)
class Window:
    """A rectangle of a :class:`Sensor`'s frame at a binning ``level``: ``x``, ``y`` (its top-left pixel) and
    ``width`` x ``height`` in pixels of that level. Exactly a crop of the whole frame read out at that level, which a
    renderer draws through an off-axis part of the camera's frustum (:attr:`ndc`, :meth:`crop_matrix`)."""

    sensor: Sensor
    level: int
    x: int
    y: int
    width: int
    height: int

    @property
    def size(self) -> tuple[int, int]:
        return self.width, self.height

    @property
    def pixels(self) -> int:
        return self.width * self.height

    @property
    def sensor_rect(self) -> tuple[int, int, int, int]:
        """``(x0, y0, x1, y1)`` in sensor pixels."""
        k = self.level
        return self.x * k, self.y * k, (self.x + self.width) * k, (self.y + self.height) * k

    @property
    def view_rect(self) -> tuple[float, float, float, float]:
        """``(x0, y0, x1, y1)`` in view pixels."""
        x0, y0, x1, y1 = self.sensor_rect
        sx, sy = self.sensor.scale_xy
        return x0 / sx, y0 / sy, x1 / sx, y1 / sy

    @property
    def scale_xy(self) -> tuple[float, float]:
        """Window pixels per view pixel, across and down."""
        sx, sy = self.sensor.scale_xy
        return sx / self.level, sy / self.level

    @property
    def scale(self) -> float:
        """Window pixels per view pixel: a face with ``d`` view pixels between the eyes has ``d * scale`` here."""
        return self.sensor.scale / self.level

    @property
    def ndc(self) -> tuple[float, float, float, float]:
        """``(left, bottom, right, top)`` in normalised device coordinates of the whole picture."""
        x0, y0, x1, y1 = self.sensor_rect
        w, h = self.sensor.size
        return 2 * x0 / w - 1, 1 - 2 * y1 / h, 2 * x1 / w - 1, 1 - 2 * y0 / h

    def crop_matrix(self) -> np.ndarray:
        """The 4x4 matrix that, in front of the camera's projection, shows just this window."""
        return crop_matrix(*self.ndc)

    def camera(self, camera) -> SubCamera:
        """``camera`` seen through this window (for a renderer of the window's size)."""
        return SubCamera(camera, self.ndc, self.sensor.view_aspect)

    def at_level(self, level: int) -> Window:
        """The same rectangle of the sensor at ``level``, a level that divides this window's (finer: ``self.level /
        level`` times the pixels across)."""
        level = int(level)
        if level == self.level:
            return self
        if level <= 0 or self.level % level:
            raise ValueError(f"level {level} does not divide the window's level {self.level}")
        ratio = self.level // level
        return Window(self.sensor, level, self.x * ratio, self.y * ratio, self.width * ratio, self.height * ratio)

    def to_sensor(self, points) -> np.ndarray:
        """Window pixels ``(..., 2)`` to sensor pixels."""
        return (np.asarray(points, dtype=np.float64) + (self.x, self.y)) * self.level

    def from_sensor(self, points) -> np.ndarray:
        """Sensor pixels ``(..., 2)`` to window pixels."""
        return np.asarray(points, dtype=np.float64) / self.level - (self.x, self.y)

    def to_view(self, points) -> np.ndarray:
        """Window pixels ``(..., 2)`` to view pixels (where a face found in the window is in the main frame)."""
        return self.sensor.to_view(self.to_sensor(points))

    def from_view(self, points) -> np.ndarray:
        """View pixels ``(..., 2)`` to window pixels."""
        return self.from_sensor(self.sensor.from_view(points))

    def box_to_view(self, box) -> tuple[float, float, float, float]:
        """A box ``(x0, y0, x1, y1)`` in window pixels, in view pixels."""
        (x0, y0), (x1, y1) = self.to_view(np.reshape(np.asarray(box, dtype=np.float64), (2, 2)))
        return float(x0), float(y0), float(x1), float(y1)

    def box_from_view(self, box) -> tuple[float, float, float, float]:
        """A box ``(x0, y0, x1, y1)`` in view pixels, in window pixels."""
        (x0, y0), (x1, y1) = self.from_view(np.reshape(np.asarray(box, dtype=np.float64), (2, 2)))
        return float(x0), float(y0), float(x1), float(y1)

    def __str__(self) -> str:
        return (f"{self.width}x{self.height} at level {self.level} of {self.sensor} "
                f"(sensor {self.sensor_rect}, {self.scale:.2f} px per view px)")
