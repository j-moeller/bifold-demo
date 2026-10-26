"""Procedural fabrics in numpy: one seamless repeat (a tile) of each pattern, in a primary and a secondary colour.

Woven patterns are woven: a thread grid where at each crossing either the warp (vertical) or the weft (horizontal)
thread is on top, by the weave (plain, 2/2 twill, 3/1 twill, herringbone), and each thread has a colour. Gingham is a
plain weave of white and coloured bands, tartan a 2/2 twill of a symmetric sett, houndstooth a 2/2 twill of four dark
and four light threads, denim a 3/1 twill of an indigo warp (a little uneven, like slub yarn) and a white weft, so their
mixed tones come out the way real cloth gives them. Printed patterns (stripes, pinstripes, checks, windowpane, polka
dots, camouflage) are drawn, anti-aliased. Camouflage is periodic noise (random phases, low-pass filtered in the
Fourier domain, so it tiles) cut into four colours.

:func:`tile` makes ``(T, T, 3)`` float32 RGB (0-255) for one repeat; the fills (:mod:`.fills`) lay it on the cloth
coordinates of :mod:`.surface` at a size in metres. :data:`PATTERNS` lists the patterns with their default size and
the default secondary colour when only one colour is given.
"""

from __future__ import annotations

import cv2
import numpy as np

TILE = 256  # pixels per repeat


def _mix(a, b, t: float) -> np.ndarray:
    return np.asarray(a, np.float32) * (1.0 - t) + np.asarray(b, np.float32) * t


def luminance(rgb) -> float:
    r, g, b = (float(v) for v in rgb)
    return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0


def contrast(rgb, light=(240, 240, 236), dark=(28, 28, 30)) -> np.ndarray:
    """A light colour for a dark ``rgb``, a dark one for a light ``rgb``."""
    return np.asarray(light if luminance(rgb) < 0.5 else dark, np.float32)


def _weave(warp: np.ndarray, weft: np.ndarray, up) -> np.ndarray:
    """``(N, N, 3)``: threads coloured ``warp[i]`` (columns) and ``weft[j]`` (rows); ``up(i, j)`` is where the warp is
    on top. ``N`` is the length of ``warp`` (and of ``weft``)."""
    n = len(warp)
    j, i = np.mgrid[0:n, 0:n]
    top = up(i, j)
    return np.where(top[..., None], warp[i], weft[j]).astype(np.float32)


def _twill(over: int = 2, under: int = 2):
    period = over + under
    return lambda i, j: ((i + j) % period) < over


def _plain(i, j):
    return ((i + j) % 2) == 0


def _upscale(threads: np.ndarray, size: int) -> np.ndarray:
    n = threads.shape[0]
    if size % n == 0:
        return np.repeat(np.repeat(threads, size // n, axis=0), size // n, axis=1)
    return cv2.resize(threads, (size, size), interpolation=cv2.INTER_AREA)


def _bands(n: int, sett, colours: dict) -> np.ndarray:
    """``(n, 3)``: thread colours of a sett ``[(key, threads), ...]`` repeated to ``n`` threads."""
    seq = np.concatenate([np.repeat(np.asarray(colours[k], np.float32)[None], c, axis=0) for k, c in sett])
    reps = int(np.ceil(n / len(seq)))
    return np.tile(seq, (reps, 1))[:n]


def _coverage(x: np.ndarray, width: float, pixel: float) -> np.ndarray:
    """How much of each pixel centred at ``x`` (repeats, period 1) the band ``[0, width)`` covers (box filtered)."""
    def ramp(t):  # the integral of the band over one period, periodic part
        f = np.floor(t)
        r = t - f
        return f * width + np.minimum(r, width)

    return (ramp(x + pixel / 2) - ramp(x - pixel / 2)) / pixel


def stripes(primary, secondary, size: int = TILE, width: float = 0.5) -> np.ndarray:
    """Vertical stripes of ``secondary`` (``width`` of the repeat, in the middle of the tile) on ``primary``."""
    x = (np.arange(size) + 0.5) / size - (0.5 - width / 2)
    c = _coverage(x, width, 1.0 / size)[None, :, None]
    row = np.asarray(secondary, np.float32) * c + np.asarray(primary, np.float32) * (1 - c)
    return np.repeat(row, size, axis=0).astype(np.float32)


def pinstripe(primary, secondary, size: int = TILE) -> np.ndarray:
    return stripes(primary, secondary, size, width=0.07)


def checks(primary, secondary, size: int = TILE) -> np.ndarray:
    """A checkerboard: two squares of each colour per repeat."""
    x = (np.arange(size) + 0.5) / size
    a = _coverage(x, 0.5, 1.0 / size)
    c = (a[None, :] * (1 - a[:, None]) + a[:, None] * (1 - a[None, :]))[..., None]
    return (np.asarray(primary, np.float32) * (1 - c) + np.asarray(secondary, np.float32) * c).astype(np.float32)


def windowpane(primary, secondary, size: int = TILE) -> np.ndarray:
    """Thin lines of ``secondary`` both ways on ``primary`` (through the middle of the tile)."""
    x = (np.arange(size) + 0.5) / size - 0.475
    a = _coverage(x, 0.05, 1.0 / size)
    c = np.maximum(a[None, :], a[:, None])[..., None]
    return (np.asarray(primary, np.float32) * (1 - c) + np.asarray(secondary, np.float32) * c).astype(np.float32)


def polka_dots(primary, secondary, size: int = TILE, radius: float = 0.16) -> np.ndarray:
    """Dots of ``secondary`` on ``primary``, two per repeat (one offset by half a repeat both ways)."""
    y, x = (np.mgrid[0:size, 0:size] + 0.5) / size
    d = np.minimum(np.hypot(x - 0.25, y - 0.25), np.hypot(x - 0.75, y - 0.75))
    for dx in (-1.0, 1.0):  # the dots near the edges, from the next tiles
        for dy in (-1.0, 0.0, 1.0):
            d = np.minimum(d, np.hypot(x - 0.25 - dx, y - 0.25 - dy))
            d = np.minimum(d, np.hypot(x - 0.75 - dx, y - 0.75 - dy))
    c = np.clip((radius - d) * size + 0.5, 0.0, 1.0)[..., None]
    return (np.asarray(primary, np.float32) * (1 - c) + np.asarray(secondary, np.float32) * c).astype(np.float32)


def gingham(primary, secondary, size: int = TILE) -> np.ndarray:
    """A plain weave of bands of 8 ``primary`` and 8 ``secondary`` threads both ways (three tones)."""
    colours = {"p": primary, "s": secondary}
    band = _bands(16, [("p", 8), ("s", 8)], colours)
    return _upscale(_weave(band, band, _plain), size)


def tartan(primary, secondary, size: int = 512, dark=None, accent=None) -> np.ndarray:
    """A 2/2 twill of a symmetric sett: ``primary`` ground, broad ``secondary`` and dark bands, thin accent lines."""
    dark = np.asarray(dark if dark is not None else _mix(secondary, (10, 12, 20), 0.6), np.float32)
    accent = np.asarray(accent if accent is not None else contrast(primary, (236, 214, 110), (240, 238, 228)),
                        np.float32)
    colours = {"p": primary, "s": secondary, "d": dark, "a": accent}
    half = [("p", 24), ("d", 4), ("p", 4), ("a", 2), ("p", 4), ("s", 14), ("d", 2), ("s", 2), ("a", 2)]
    sett = half + half[::-1]
    n = sum(c for _, c in sett)
    band = _bands(n, sett, colours)
    return _upscale(_weave(band, band, _twill(2, 2)), size)


def houndstooth(primary, secondary, size: int = TILE) -> np.ndarray:
    """A 2/2 twill of 4 ``primary`` and 4 ``secondary`` threads both ways: the broken check."""
    colours = {"p": primary, "s": secondary}
    band = _bands(8, [("p", 4), ("s", 4)], colours)
    return _upscale(_weave(band, band, _twill(2, 2)), size)


def twill(primary, secondary, size: int = TILE) -> np.ndarray:
    """Diagonal ribs: a 2/2 twill of a ``primary`` warp and a ``secondary`` weft."""
    warp = np.repeat(np.asarray(primary, np.float32)[None], 16, axis=0)
    weft = np.repeat(np.asarray(secondary, np.float32)[None], 16, axis=0)
    return _upscale(_weave(warp, weft, _twill(2, 2)), size)


def herringbone(primary, secondary, size: int = TILE) -> np.ndarray:
    """A twill whose diagonal turns every 8 threads: columns of zigzags."""
    warp = np.repeat(np.asarray(primary, np.float32)[None], 16, axis=0)
    weft = np.repeat(np.asarray(secondary, np.float32)[None], 16, axis=0)

    def up(i, j):
        right = ((i + j) % 4) < 2
        left = ((i - j) % 4) < 2
        return np.where((i // 8) % 2 == 0, right, left)

    return _upscale(_weave(warp, weft, up), size)


def denim(primary, secondary, size: int = TILE, seed: int = 7) -> np.ndarray:
    """A 3/1 twill of an uneven ``primary`` warp (slub yarn: each thread a little lighter or darker) and a
    ``secondary`` weft."""
    rng = np.random.default_rng(seed)
    n = 64
    shade = 1.0 + 0.16 * (rng.random(n) - 0.5) + 0.08 * np.sin(np.arange(n) / n * 2 * np.pi * 3 + rng.random() * 6)
    warp = np.asarray(primary, np.float32)[None] * shade[:, None].astype(np.float32)
    weft = np.repeat(_mix(secondary, primary, 0.25)[None], n, axis=0)
    return np.clip(_upscale(_weave(warp, weft, _twill(3, 1)), size), 0, 255)


def _periodic_noise(size: int, cycles: float, rng) -> np.ndarray:
    """Smooth noise that tiles: random phases low-pass filtered to about ``cycles`` blobs across, in [0, 1]."""
    spectrum = np.fft.fft2(rng.standard_normal((size, size)))
    f = np.fft.fftfreq(size) * size
    radius = np.hypot(f[:, None], f[None, :])
    field = np.real(np.fft.ifft2(spectrum * np.exp(-(radius / cycles) ** 2)))
    field -= field.min()
    return field / max(field.max(), 1e-9)


def camo(primary, secondary, size: int = TILE, seed: int = 3, dark=None, light=None) -> np.ndarray:
    """Woodland-style camouflage: blobs of ``primary``, ``secondary``, a dark and a light colour."""
    rng = np.random.default_rng(seed)
    dark = np.asarray(dark if dark is not None else _mix(secondary, (20, 18, 14), 0.65), np.float32)
    light = np.asarray(light if light is not None else _mix(primary, (205, 190, 150), 0.55), np.float32)
    a = _periodic_noise(size, 4.0, rng)
    b = _periodic_noise(size, 6.0, rng)
    out = np.empty((size, size, 3), np.float32)
    out[:] = np.asarray(primary, np.float32)
    low, high = np.quantile(a, [0.38, 0.72])
    for colour, mask in ((light, a < low), (secondary, a > high), (dark, b > np.quantile(b, 0.8))):
        w = _soften(mask.astype(np.float32))[..., None]
        out = out * (1 - w) + colour * w
    return out


def _soften(mask: np.ndarray, sigma: float = 0.8) -> np.ndarray:
    """``mask`` blurred by about a pixel, wrapping round the tile's edges (anti-aliased blob edges)."""
    pad = 4
    wide = np.pad(mask, pad, mode="wrap")
    return cv2.GaussianBlur(wide, (0, 0), sigma)[pad:-pad, pad:-pad]


PATTERNS = {
    # name: (tile function, default repeat in metres, default secondary colour for a single colour)
    "stripes": (stripes, 0.03, "contrast"),
    "pinstripe": (pinstripe, 0.012, "contrast"),
    "checks": (checks, 0.05, "contrast"),
    "windowpane": (windowpane, 0.07, "contrast"),
    "gingham": (gingham, 0.016, "white"),
    "tartan": (tartan, 0.15, "tartan"),
    "houndstooth": (houndstooth, 0.024, "contrast"),
    "polka dots": (polka_dots, 0.035, "contrast"),
    "twill": (twill, 0.008, "tone"),
    "herringbone": (herringbone, 0.024, "tone"),
    "denim": (denim, 0.02, "white"),
    "camo": (camo, 0.5, "camo"),
}
DEFAULT_PRIMARY = {"denim": (44, 62, 102), "tartan": (168, 28, 36), "camo": (86, 92, 52)}


def secondary_for(pattern: str, primary) -> np.ndarray:
    """The second colour of ``pattern`` when only ``primary`` is given."""
    rule = PATTERNS[pattern][2]
    p = np.asarray(primary, np.float32)
    if rule == "white":
        return np.asarray((236, 234, 226), np.float32)
    if rule == "tone":  # a lighter or darker shade of the same colour
        return _mix(p, (255, 255, 255), 0.3) if luminance(p) < 0.5 else _mix(p, (0, 0, 0), 0.3)
    if rule == "tartan":
        greenish = p[1] > p[0] and p[1] > p[2]
        return np.asarray((24, 36, 78) if greenish else (24, 66, 44), np.float32)
    if rule == "camo":
        return _mix(p, (92, 64, 38), 0.7)
    if pattern == "pinstripe":
        return contrast(p, (205, 205, 205), (60, 60, 64))
    return contrast(p)


def tile(pattern: str, primary, secondary=None) -> np.ndarray:
    """One repeat of ``pattern`` (a :data:`PATTERNS` name) as ``(T, T, 3)`` float32 RGB."""
    if pattern not in PATTERNS:
        raise KeyError(f"unknown pattern {pattern!r}; known: {', '.join(PATTERNS)}")
    primary = np.asarray(primary, np.float32)
    secondary = secondary_for(pattern, primary) if secondary is None else np.asarray(secondary, np.float32)
    return PATTERNS[pattern][0](primary, secondary)
