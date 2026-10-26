"""What a garment region is filled with: a colour, a procedural pattern or an image swatch, laid on the cloth.

Every fill answers one question, :meth:`Fill.colors`: the base colour (RGB 0-255, float32) at a set of texels, given
where they are on the cloth (:class:`Where`: the cloth coordinates of :mod:`.surface` in metres, the size of each
texel on the body, and the texels' place in the atlas). The compositor (:mod:`.tailor`) then puts the shading and
detail of the avatar's own texture back on top, so it does not matter to a fill what the garment looked like.

* :class:`Solid`: one colour (a name such as ``"navy"``, ``"#1f2a44"`` or ``(31, 42, 68)``; see :func:`color`).
* :class:`Swatch`: an image of a piece of fabric, tiled over the cloth at ``scale`` metres per repeat and turned by
  ``rotation`` degrees. Seamless swatches tile without seams; the generative "describe a fabric" step makes these.
* :class:`Pattern`: one of the procedural :data:`.patterns.PATTERNS` in a primary and a secondary colour; it is a
  swatch whose tile is made by :func:`.patterns.tile`.
* :class:`Atlas`: colours already laid out in the texture atlas (the avatar's own albedo, or a painted texture).
* :class:`Keep`: leaves the region as the avatar has it.

A tiled fill is filtered like a mipmap: each texel reads the tile at the level where one tile pixel is about one
texel on the body, so pinstripes finer than a texel blend into their mixed colour instead of flickering.
Fills turn into small JSON-friendly specs and back (:meth:`Fill.spec`, :func:`from_spec`), for outfit recipes.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import patterns

# colour names (sRGB) a prompt or a recipe may use; "light", "dark", "pale" and "deep" shade them (see color())
COLORS = {
    "black": (22, 22, 24), "white": (240, 240, 236), "off white": (232, 228, 214), "ivory": (238, 232, 210),
    "cream": (236, 226, 196), "grey": (128, 128, 130), "gray": (128, 128, 130), "silver": (192, 194, 198),
    "charcoal": (54, 56, 60), "graphite": (70, 72, 76), "slate": (92, 102, 112),
    "navy": (31, 42, 74), "navy blue": (31, 42, 74), "midnight blue": (24, 30, 56), "blue": (38, 84, 172),
    "royal blue": (40, 72, 190), "cobalt": (0, 71, 171), "sky blue": (120, 176, 222), "baby blue": (160, 200, 232),
    "light blue": (150, 190, 226), "denim blue": (64, 92, 140), "indigo": (44, 52, 110), "teal": (18, 112, 116),
    "turquoise": (40, 186, 180), "aqua": (90, 200, 210), "cyan": (40, 190, 220), "petrol": (20, 80, 96),
    "green": (46, 122, 60), "dark green": (24, 66, 44), "forest green": (34, 80, 46), "bottle green": (20, 70, 48),
    "emerald": (20, 130, 90), "lime": (150, 200, 40), "lime green": (150, 200, 40), "mint": (160, 220, 190),
    "sage": (150, 168, 130), "olive": (96, 100, 46), "olive green": (96, 100, 46), "khaki": (176, 160, 112),
    "army green": (76, 84, 52), "moss": (98, 110, 54),
    "yellow": (236, 200, 40), "mustard": (204, 160, 40), "gold": (212, 172, 60), "lemon": (244, 228, 90),
    "orange": (232, 112, 30), "burnt orange": (190, 84, 30), "rust": (164, 70, 36), "coral": (242, 112, 96),
    "peach": (246, 180, 148), "salmon": (240, 128, 112),
    "red": (192, 32, 40), "scarlet": (210, 30, 30), "crimson": (170, 20, 50), "cherry": (180, 20, 40),
    "burgundy": (110, 22, 40), "maroon": (110, 30, 34), "wine": (104, 28, 48), "brick": (160, 64, 48),
    "pink": (236, 140, 172), "hot pink": (232, 60, 140), "rose": (220, 120, 140), "magenta": (200, 30, 150),
    "fuchsia": (210, 40, 160), "blush": (232, 180, 180),
    "purple": (110, 50, 140), "violet": (128, 70, 180), "lavender": (180, 160, 216), "lilac": (196, 162, 210),
    "plum": (100, 40, 80), "aubergine": (70, 30, 60),
    "brown": (100, 64, 40), "chocolate": (76, 46, 30), "tan": (196, 156, 108), "camel": (186, 146, 98),
    "beige": (214, 196, 160), "sand": (206, 186, 146), "taupe": (140, 124, 110), "chestnut": (136, 70, 44),
    "cognac": (150, 84, 40), "stone": (190, 182, 166),
}
SHADES = {"light": ("mix", (255, 255, 255), 0.35), "pale": ("mix", (255, 255, 255), 0.55),
          "dark": ("scale", None, 0.6), "deep": ("scale", None, 0.78), "bright": ("bright", None, 0.0)}


def color(spec) -> tuple[int, int, int]:
    """RGB of a colour name (``"navy"``, ``"light grey"``), ``"#rrggbb"`` / ``"#rgb"`` or ``(r, g, b)``."""
    if isinstance(spec, (tuple, list, np.ndarray)):
        values = [float(v) for v in spec][:3]
        if len(values) != 3:
            raise ValueError(f"a colour needs three components, got {spec!r}")
        return tuple(round(min(255.0, max(0.0, v))) for v in values)
    text = str(spec).strip().lower()
    if text.startswith("#"):
        hexa = text[1:]
        if len(hexa) == 3:
            hexa = "".join(c * 2 for c in hexa)
        if not re.fullmatch(r"[0-9a-f]{6}", hexa):
            raise ValueError(f"invalid hex colour {spec!r}")
        return tuple(int(hexa[i:i + 2], 16) for i in (0, 2, 4))
    text = re.sub(r"[-_]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    if text in COLORS:
        return COLORS[text]
    first, _, rest = text.partition(" ")
    if first in SHADES and rest:
        return shade(color(rest), first)
    raise ValueError(f"unknown colour {spec!r}")


def hex_color(rgb) -> str:
    """``"#rrggbb"`` of an RGB colour."""
    r, g, b = (int(v) for v in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def shade(rgb, how: str) -> tuple[int, int, int]:
    """``rgb`` made lighter, darker, paler, deeper or brighter (a :data:`SHADES` word)."""
    kind, target, amount = SHADES[how]
    c = np.asarray(rgb, np.float64)
    if kind == "mix":
        c = c * (1 - amount) + np.asarray(target) * amount
    elif kind == "scale":
        c = c * amount
    else:  # brighter and more saturated
        c = np.clip((c - c.mean()) * 1.3 + c.mean() * 1.12, 0, 255)
    return tuple(round(float(v)) for v in np.clip(c, 0, 255))


@dataclass
class Where:
    """The texels a fill is asked about (``N`` of them)."""

    uv: np.ndarray  # (N, 2) float32 cloth coordinates in metres (see .surface)
    part: np.ndarray  # (N,) int8 body part (see .surface.PARTS)
    footprint: np.ndarray  # (N,) float32 metres per texel
    rows: np.ndarray  # (N,) the texels' rows in the atlas
    cols: np.ndarray  # (N,) and columns
    size: int  # the atlas' size


class Fill:
    """A region's base colour (see the module docs)."""

    def colors(self, where: Where) -> np.ndarray:
        raise NotImplementedError

    def spec(self):
        """A JSON-friendly description (see :func:`from_spec`)."""
        raise NotImplementedError

    def keeps(self) -> bool:
        """True when the region stays as the avatar has it."""
        return False


class Keep(Fill):
    """Leaves the region as it is."""

    def colors(self, where):
        raise TypeError("Keep has no colours")

    def keeps(self) -> bool:
        return True

    def spec(self):
        return "keep"

    def __repr__(self) -> str:
        return "Keep()"


class Solid(Fill):
    """One colour."""

    def __init__(self, rgb):
        self.rgb = color(rgb)

    def colors(self, where: Where) -> np.ndarray:
        return np.broadcast_to(np.asarray(self.rgb, np.float32), (len(where.rows), 3))

    def spec(self):
        return hex_color(self.rgb)

    def __repr__(self) -> str:
        return f"Solid({self.spec()!r})"


class Atlas(Fill):
    """Colours already in the atlas' layout: ``rgb`` is ``(H, W, 3)`` (resized to the atlas when it differs)."""

    def __init__(self, rgb: np.ndarray, name: str = "atlas"):
        self.rgb = np.asarray(rgb, np.float32)
        self.name = name

    def colors(self, where: Where) -> np.ndarray:
        rgb = self.rgb
        if rgb.shape[:2] != (where.size, where.size):
            rgb = cv2.resize(rgb, (where.size, where.size), interpolation=cv2.INTER_AREA)
        return rgb[where.rows, where.cols, :3]

    def spec(self):
        return {"atlas": self.name}


class Swatch(Fill):
    """An image of fabric (``image``: a path, or an RGB ``uint8``/float array), tiled over the cloth at ``scale`` metres
    per repeat (its width), turned by ``rotation`` degrees. ``seamless=False`` mirrors the image at its edges so a
    swatch that does not tile shows no hard seams (at the price of a mirror symmetry)."""

    def __init__(self, image, scale: float = 0.25, rotation: float = 0.0, seamless: bool = True):
        self.source = image if isinstance(image, (str, Path)) else None
        if self.source is not None:
            data = cv2.imread(str(image), cv2.IMREAD_COLOR)
            if data is None:
                raise FileNotFoundError(f"cannot read the swatch {image}")
            image = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        rgb = np.asarray(image, np.float32)[..., :3]
        if not seamless:
            rgb = np.concatenate([rgb, rgb[:, ::-1]], axis=1)
            rgb = np.concatenate([rgb, rgb[::-1]], axis=0)
        self.scale = float(scale)
        self.rotation = float(rotation)
        self.seamless = seamless
        self._levels = _pyramid(rgb)

    def colors(self, where: Where) -> np.ndarray:
        return _sample(self._levels, where.uv, where.footprint, self.scale, self.rotation)

    def spec(self):
        out = {"swatch": str(self.source) if self.source is not None else "image", "scale": self.scale}
        if self.rotation:
            out["rotation"] = self.rotation
        if not self.seamless:
            out["seamless"] = False
        return out


class Pattern(Swatch):
    """A procedural pattern (see :data:`.patterns.PATTERNS`) in ``primary`` and ``secondary`` (a default contrast
    when ``None``), ``scale`` metres per repeat (the pattern's own default when ``None``) and ``rotation`` degrees."""

    def __init__(self, kind: str, primary=None, secondary=None, scale: float | None = None, rotation: float = 0.0):
        if kind not in patterns.PATTERNS:
            raise KeyError(f"unknown pattern {kind!r}; known: {', '.join(patterns.PATTERNS)}")
        self.kind = kind
        self.primary = color(primary) if primary is not None else patterns.DEFAULT_PRIMARY.get(kind, COLORS["navy"])
        self.secondary = color(secondary) if secondary is not None else tuple(
            round(float(v)) for v in patterns.secondary_for(kind, self.primary))
        self.scale = float(scale) if scale is not None else patterns.PATTERNS[kind][1]
        self.rotation = float(rotation)
        self.source, self.seamless = None, True
        self._levels = _pyramid(patterns.tile(kind, self.primary, self.secondary))

    def spec(self):
        out = {"pattern": self.kind, "colors": [hex_color(self.primary), hex_color(self.secondary)],
               "scale": round(self.scale, 5)}
        if self.rotation:
            out["rotation"] = self.rotation
        return out

    def __repr__(self) -> str:
        return f"Pattern({self.kind!r}, {self.primary}, {self.secondary}, scale={self.scale:g}, " \
               f"rotation={self.rotation:g})"


def from_spec(spec) -> Fill:
    """A fill from a spec: a :class:`Fill`, ``"keep"``, a colour (name, hex, ``[r, g, b]``), a short description
    (``"navy pinstripe"``, see :func:`.prompts.parse_fill`), or a dict: ``{"color": ...}``,
    ``{"pattern": "tartan", "colors": ["red", "navy"], "scale": 0.15, "rotation": 45}`` or
    ``{"swatch": "fabric.png", "scale": 0.2}``."""
    if isinstance(spec, Fill):
        return spec
    if isinstance(spec, dict):
        if "color" in spec:
            return Solid(spec["color"])
        if "pattern" in spec:
            colours = list(spec.get("colors", []))
            return Pattern(spec["pattern"], colours[0] if colours else None, colours[1] if len(colours) > 1 else None,
                           spec.get("scale"), spec.get("rotation", 0.0))
        if "swatch" in spec:
            return Swatch(spec["swatch"], spec.get("scale", 0.25), spec.get("rotation", 0.0),
                          spec.get("seamless", True))
        if "prompt" in spec:
            from .prompts import parse_fill

            return parse_fill(spec["prompt"])
        raise ValueError(f"cannot make a fill of {spec!r}")
    if isinstance(spec, (tuple, list)):
        return Solid(spec)
    text = str(spec).strip()
    if text.lower() == "keep":
        return Keep()
    try:
        return Solid(text)
    except ValueError:
        from .prompts import parse_fill

        return parse_fill(text)


# -- tiled sampling --------------------------------------------------------------------------------------------


def _pyramid(rgb: np.ndarray) -> list[np.ndarray]:
    """The tile at full size and halved down to one pixel (box filtered), for mipmapped sampling. The tile is first
    resized to a square power of two (at most 1024)."""
    h, w = rgb.shape[:2]
    side = 1 << int(min(10, max(2, round(math.log2(max(h, w))))))
    base = cv2.resize(np.ascontiguousarray(rgb, np.float32), (side, side),
                      interpolation=cv2.INTER_AREA if max(h, w) > side else cv2.INTER_LINEAR)
    levels = [base]
    while levels[-1].shape[0] > 1:
        a = levels[-1]
        levels.append(0.25 * (a[0::2, 0::2] + a[1::2, 0::2] + a[0::2, 1::2] + a[1::2, 1::2]))
    return levels


def _sample(levels: list[np.ndarray], uv: np.ndarray, footprint: np.ndarray, scale: float,
            rotation: float) -> np.ndarray:
    """The tile ``levels`` laid over cloth coordinates ``uv`` (metres): one repeat per ``scale`` metres, turned by
    ``rotation`` degrees, read at the mip level that suits each texel's ``footprint`` (trilinear). Texels sorted by
    footprint (as the compositor passes them) read each level from one slice."""
    side = levels[0].shape[0]
    a = math.radians(rotation)
    c, s = np.float32(math.cos(a) / scale), np.float32(math.sin(a) / scale)
    u, v = uv[:, 0].astype(np.float32), uv[:, 1].astype(np.float32)
    px = np.mod(c * u + s * v, 1.0) * side  # tile pixels (+ 0.5)
    py = np.mod(s * u - c * v, 1.0) * side  # image rows run down, v runs up
    lod = np.log2(np.maximum(np.nan_to_num(footprint, nan=0.0) * np.float32(side / scale), 1e-6), dtype=np.float32)
    np.clip(lod, 0.0, len(levels) - 1.0, out=lod)
    out = np.empty((len(u), 3), np.float32)
    if len(lod) > 1 and np.all(lod[1:] >= lod[:-1]):
        cuts = np.searchsorted(lod, np.arange(1, len(levels)), side="left")
        parts = [slice(a, b) for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(lod)]) if b > a]
    else:
        low = np.floor(lod)
        parts = [low == k for k in np.unique(low)]
    for part in parts:
        k = int(np.floor(lod[part][0]))
        frac = (lod[part] - k)[:, None]
        here = _bilinear(levels[k], px[part], py[part], 2.0 ** -k)
        if k + 1 < len(levels) and frac.max() > 0:
            there = _bilinear(levels[k + 1], px[part], py[part], 2.0 ** -(k + 1))
            here += (there - here) * frac
        out[part] = here
    return out


def _bilinear(level: np.ndarray, px: np.ndarray, py: np.ndarray, factor: float) -> np.ndarray:
    """``level`` read at tile pixels ``(px, py)`` (centres at +0.5) scaled by ``factor``, bilinear, wrapping."""
    side, n = level.shape[0], len(px)
    if side == 1:
        return np.repeat(level.reshape(1, 3), n, axis=0)
    width = 4096  # cv2.remap wants maps narrower than 32767: lay the points out in rows
    shape = (-(-n // width), width) if n > width else (1, n)
    count = shape[0] * shape[1]
    mx = np.zeros(count, np.float32)
    my = np.zeros(count, np.float32)
    mx[:n] = px
    my[:n] = py
    mx *= np.float32(factor)
    my *= np.float32(factor)
    mx -= 0.5
    my -= 0.5
    out = cv2.remap(level, mx.reshape(shape), my.reshape(shape), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    return out.reshape(-1, 3)[:n]
