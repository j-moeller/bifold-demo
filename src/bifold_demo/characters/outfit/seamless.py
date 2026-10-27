"""Seamless fabric swatches without a model: how well an image tiles, a repair that makes any image tile, even light.

A swatch is one repeat of a fabric: :class:`~.fills.Swatch` lays it side by side over the clothes, so its right edge
meets its own left edge and its bottom its top. Three tools work on any image, a generated fabric or a photo:

* :func:`seam_score`: how much the wrap (the right edge against the left one, the bottom against the top) stands
  out from the rest of the image. Three signs of a seam are measured in CIE Lab, on the image at about 256 pixels:

  - a step: the colour difference between the last and the first column (row), averaged along the seam, at four
    scales (1, 2, 4 and 8 pixels), against the strongest such difference between two neighbouring columns (rows)
    inside the image. A seam makes the wrap the strongest edge of the image; a checked fabric whose checks happen to
    meet at the edge has edges as strong inside, so it is not taken for one;
  - a band: a line along the wrap that is lighter, darker or of another colour than the cloth on either side of it
    (the dark rim a decoder with zero padding paints), against the strongest such line elsewhere, 2-8 pixels wide;
  - a broken rhythm: a regular pattern repeats, so a window of 16 columns (rows) inside it has a twin a repeat
    away; the window across a seam that breaks the rhythm (stripes, checks or dots that do not line up) has none.
    Its distance to the most alike window elsewhere is set against the same for the windows inside (their 99th
    percentile). A repeat longer than half the swatch cannot be checked this way: its windows have no twin inside.

  The score is the largest ratio: about 1 or less for an image that tiles, and more where a seam shows. Measured on
  synthetic textures at 512 pixels: of 150 that tile (smooth and crisp blobs, procedural patterns rolled to random
  places, weaves), 2 score above 1.25 (crisp two-colour blobs, up to 1.29), and a seam starts to show at about
  1.15-1.2. All 24 textures cut out of larger ones score 2.1 or more; of 36 patterns whose rhythm breaks at the wrap,
  35 score above 1.25 (most above 9); of 6 dark rims of 8-15 %, 4 do. :data:`SEAM_THRESHOLD` (1.25) is where a seed
  is rejected. A single repeat of a pattern that is mirrored rather than repeated (a tartan sett) can score up to
  about 1.5 though it tiles.

* :func:`make_seamless`: the repair. It rolls the image by half, so the seams cross in its middle and its new edges
  are the old middle (which joins up by itself), then quilts the cross, first left to right, then top to bottom: the
  strips on either side of the seam are laid over each other and cut along the path where they agree best (the
  minimum-error boundary cut of Efros and Freeman's image quilting; in the second pass a path whose ends meet, so
  the first pass's tiling stays). The overlap is chosen among 6-25 % of the width (height) as the one whose strips
  differ least, which for a regular pattern is the one that keeps its rhythm. Fine detail switches at the cut
  (feathered over two pixels, so a weave is not ghosted) and the coarse colour cross-fades round it. Where the
  result still shows a seam (smooth fabric whose strips have nothing alike to cut along: blobs a third of the swatch
  across), a third overlaps instead and the coarse colour cross-fades over all of it. The result is smaller by the
  overlaps and is resampled back to the original size (as a repeat, so the resampling wraps round its edges). Before
  any of it the light is evened out (a quadratic surface, since the image does not tile yet), so a lighting gradient
  does not turn into a step at the cut. It needs no model and gives the same image every time: about 0.13 s at 512
  pixels, 0.25 s at 768 (twice that when the cross-fade is tried too). On the synthetic textures above it brings
  46 of 48 cut-out ones to 1.25 or less; the two left are the smoothest (three blobs across).

* :func:`equalise_light`: even light. A flat-lay photo of fabric, or a generated one, has light falling off towards
  a side or a corner; the tailor puts the garment's own folds and shading on the swatch, so the swatch itself should
  be evenly lit. The light is the part of the log-luminance that varies once across the swatch (its Fourier modes of
  one cycle, so a swatch that tiles still tiles), or a quadratic surface for an image that does not tile; it is
  divided out of all three channels alike, so the colours (hue and saturation) and the weave stay, and the mean
  brightness is kept. A pattern of a few repeats across the swatch is left alone; one large motif is evened too.

:func:`resize_periodic` resamples a repeat with its edges wrapping; :func:`tiled` lays an image out 2 x 2 to look at
its seams.
"""

from __future__ import annotations

import cv2
import numpy as np

from .fills import image_rgb

SEAM_THRESHOLD = 1.25  # seam_score above this: a visible seam (see the module docs)
ANALYSIS = 256  # the side the seam score looks at the image at
STEP_SCALES = (1, 2, 4, 8)  # pixels at the analysis size
BAND_RADII = (1, 2, 4)  # half widths of the bands looked for, at the analysis size
RHYTHM_HALF = 8  # the rhythm test compares windows of twice this many columns (rows), at the analysis size
FLOOR = 1.0  # CIE Lab units added to both sides of a ratio, so flat images score 1, not noise over noise
OVERLAP = (0.06, 0.25)  # the repair's overlap, as a share of the width (height)
WIDE_OVERLAP = 0.33  # the overlap of the repair's wide cross-fade, for smooth fabric


def _lab(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.clip(np.asarray(rgb, np.float32), 0.0, 255.0) / 255.0, cv2.COLOR_RGB2Lab)


def _shrink(a: np.ndarray, k: int) -> np.ndarray:
    """``a`` box filtered by ``k`` both ways (its sides are multiples of ``k``)."""
    if k == 1:
        return a
    h, w = a.shape[:2]
    return a.reshape(h // k, k, w // k, k, -1).mean(axis=(1, 3))


def _remap_periodic(rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    xs = (np.arange(width, dtype=np.float32) + 0.5) * np.float32(w / width) - 0.5
    ys = (np.arange(height, dtype=np.float32) + 0.5) * np.float32(h / height) - 0.5
    mx, my = np.meshgrid(xs, ys)
    out = cv2.remap(np.ascontiguousarray(rgb), mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)
    return out.reshape(height, width, *rgb.shape[2:])


def resize_periodic(rgb, width: int, height: int | None = None) -> np.ndarray:
    """``rgb`` (float32, ``(H, W)`` or ``(H, W, C)``) resampled to ``width`` x ``height`` as one repeat: what leaves
    one edge comes back in at the other, so a seamless image stays seamless (``cv2.resize`` would repeat the edge
    pixels instead). Bicubic; a shrink by two or more ends in a box filter, against aliasing."""
    rgb = np.asarray(rgb, np.float32)
    height = width if height is None else height
    h, w = rgb.shape[:2]
    if (h, w) == (height, width):
        return rgb.copy()
    f = max(1, min(h // height, w // width))
    if f == 1:
        return _remap_periodic(rgb, width, height)
    if (h, w) != (height * f, width * f):
        rgb = _remap_periodic(rgb, width * f, height * f)
    return rgb.reshape(height, f, width, f, *rgb.shape[2:]).mean(axis=(1, 3))


def _analysed(image) -> np.ndarray:
    """The image in Lab at about :data:`ANALYSIS` pixels, sides multiples of 8 (resampled as a repeat)."""
    rgb = image_rgb(image)
    h, w = rgb.shape[:2]
    scale = min(1.0, ANALYSIS / max(h, w))
    size = (max(16, round(w * scale / 8) * 8), max(16, round(h * scale / 8) * 8))
    if (w, h) != size:
        rgb = resize_periodic(rgb, *size)
    return _lab(rgb)


def _step(lab: np.ndarray) -> float:
    """The step ratio of the vertical seam (the columns' wrap) of ``lab``."""
    worst = 0.0
    for k in STEP_SCALES:
        if lab.shape[1] // k < 4:
            break
        b = _shrink(lab, k)
        col = np.linalg.norm(np.roll(b, -1, axis=1) - b, axis=2).mean(axis=0)  # col[j]: columns j and j + 1
        worst = max(worst, float((col[-1] + FLOOR) / (col[:-1].max() + FLOOR)))
    return worst


def _band(lab: np.ndarray) -> float:
    """The band ratio of the vertical seam of ``lab``: a line along the wrap unlike the cloth beside it."""
    m = lab.mean(axis=0).astype(np.float64)  # the columns' mean colours
    w = len(m)
    total = np.concatenate([np.zeros((1, 3)), np.cumsum(np.concatenate([m, m, m]), axis=0)])
    j = np.arange(w)

    def mean(lo, hi):  # the mean colour of columns lo .. hi - 1, wrapping round
        return (total[hi + w] - total[lo + w]) / (hi - lo)[:, None]

    worst = 0.0
    for r in BAND_RADII:
        if 6 * r >= w // 2:
            break
        line = np.linalg.norm(mean(j - r, j + r) - (mean(j - 3 * r, j - r) + mean(j + r, j + 3 * r)) / 2, axis=1)
        far = np.abs((j + w // 2) % w - w // 2) >= 3 * r  # lines that do not touch the wrap
        worst = max(worst, float((line[0] + FLOOR / 4) / (line[far].max() + FLOOR / 4)))
    return worst


def _rhythm(lab: np.ndarray) -> float:
    """The rhythm ratio of the vertical seam of ``lab``: how far the window of columns across the wrap is from the
    most alike window elsewhere, against the same for the windows inside (their 99th percentile). A regular pattern
    repeats, so a window inside it has a twin a repeat away; one across a seam that breaks the rhythm has none."""
    h, w = lab.shape[:2]
    m = RHYTHM_HALF
    if w < 8 * m:
        return 0.0
    x = lab.transpose(1, 0, 2).reshape(w, -1).astype(np.float64)
    sq = (x ** 2).sum(axis=1)
    d = np.maximum(sq[:, None] + sq[None, :] - 2.0 * (x @ x.T), 0.0) / h  # mean squared distance of two columns
    window = np.zeros_like(d)
    for t in range(1 - m, m + 1):  # columns j - m + 1 .. j + m against i - m + 1 .. i + m
        window += np.roll(d, (-t, -t), axis=(0, 1))
    j = np.arange(w)
    window[np.abs((j[None, :] - j[:, None] + w // 2) % w - w // 2) <= m] = np.inf  # not itself, nor overlapping
    straddle = (j >= w - m) | (j < m - 1)  # windows (but the wrap's own) that reach across the wrap
    window[:, straddle] = np.inf
    distance = np.sqrt(window.min(axis=1) / (2 * m))
    inner = distance[~straddle & (j != w - 1)]
    return float((distance[-1] + FLOOR) / (np.percentile(inner, 99) + FLOOR))


def seam_scores(image) -> dict[str, float]:
    """The parts of :func:`seam_score`: ``{"step x", "band x", "rhythm x", "step y", "band y", "rhythm y"}`` (x: the
    seam where the right edge meets the left one; y: the bottom against the top)."""
    lab = _analysed(image)
    out = {}
    for axis, a in (("x", lab), ("y", np.ascontiguousarray(lab.transpose(1, 0, 2)))):
        out.update({f"step {axis}": _step(a), f"band {axis}": _band(a), f"rhythm {axis}": _rhythm(a)})
    return out


def seam_score(image) -> float:
    """How much the wrap of ``image`` (a path or an array, see :func:`.fills.image_rgb`) stands out: about 1 or less
    when it tiles, above :data:`SEAM_THRESHOLD` when a seam shows (see the module docs)."""
    return max(seam_scores(image).values())


# -- repair ---------------------------------------------------------------------------------------------------------


def _path(cost: np.ndarray, start: int | None = None) -> tuple[float, np.ndarray]:
    """The top-to-bottom path through ``cost`` (H, w) of least total cost that moves at most one column per row
    (starting at column ``start`` when given): ``(its cost, its column in each row)``."""
    h, w = cost.shape
    total = cost[0].astype(np.float64).copy()
    if start is not None:
        total[np.arange(w) != start] = np.inf
    back = np.zeros((h, w), np.int8)
    for y in range(1, h):
        left = np.r_[np.inf, total[:-1]]
        right = np.r_[total[1:], np.inf]
        choice = np.argmin(np.stack([left, total, right]), axis=0)
        back[y] = choice - 1
        total = cost[y] + np.choose(choice, [left, total, right])
    if start is not None:  # it ends next to where it started: a closed path round a cylinder
        total[np.abs(np.arange(w) - start) > 1] = np.inf
    cut = np.empty(h, np.int64)
    cut[-1] = int(np.argmin(total))
    for y in range(h - 1, 0, -1):
        cut[y - 1] = cut[y] + back[y, cut[y]]
    return float(total[cut[-1]]), cut


def _min_cut(cost: np.ndarray, closed: bool = False) -> np.ndarray:
    """The column, row by row, of the cheapest top-to-bottom path through ``cost`` (H, w); ``closed``: its two ends
    meet (at most a column apart), so that the image it cuts still tiles top to bottom."""
    _, cut = _path(cost)
    if not closed:
        return cut
    found = None
    for start in dict.fromkeys([int(cut[0]), int(cut[-1]), int(np.argmin(cost[0]))]):
        candidate = _path(cost, start)
        if found is None or candidate[0] < found[0]:
            found = candidate
    return found[1]


def _quilt_columns(rgb: np.ndarray, guide: np.ndarray, closed: bool, wide: bool) -> np.ndarray:
    """``rgb`` made to tile left to right (narrower by the overlap): rolled by half, the strips either side of the
    seam in its middle overlaid and cut where they agree best. ``guide``: the image the costs are measured on;
    ``closed``: keep the image tiling top to bottom (the cut's ends meet); ``wide``: a third of the width overlaps
    and the coarse colour cross-fades over all of it (else the overlap whose strips differ least, and a cross-fade
    round the cut)."""
    h, w = rgb.shape[:2]
    half = w // 2
    rgb, guide = np.roll(rgb, half, axis=1), np.roll(guide, half, axis=1)
    low, high = max(4, round(w * OVERLAP[0])), max(5, min(half - 1, round(w * OVERLAP[1])))
    if wide:
        k = max(5, min(half - 1, round(w * WIDE_OVERLAP)))
    else:
        costs = []
        for k in range(low, high + 1):  # strip a: the k columns left of the seam; strip b: the k right of it
            costs.append(float(np.mean((guide[:, half - k:half] - guide[:, half:half + k]) ** 2)))
        k = low + int(np.argmin(costs))
    a, b = rgb[:, half - k:half], rgb[:, half:half + k]
    cost = cv2.GaussianBlur(np.sum((guide[:, half - k:half] - guide[:, half:half + k]) ** 2, axis=2), (0, 0), 1.0)
    margin = k // 3  # the cut keeps to the middle third, so the strips' outer edges stay as they were
    cost[:, :margin] += 1e6
    cost[:, k - margin:] += 1e6
    cut = _min_cut(cost, closed)
    take_a = (np.arange(k)[None, :] < cut[:, None]).astype(np.float32)  # a left of the cut, b from it on
    # two bands: the fine detail switches at the cut (feathered over two pixels), the coarse colour cross-fades round it
    sigma = max(1.5, k / 8)
    if wide:  # from all a at its outer edge to all b at its: smooth fabric has nothing to cut along
        t = (np.arange(k, dtype=np.float32) + 0.5) / k
        coarse = np.broadcast_to((1.0 - t * t * (3.0 - 2.0 * t))[None, :, None], (h, k, 1))
        fine = cv2.GaussianBlur(take_a, (0, 0), sigmaX=max(2.0, k / 16), sigmaY=1.0)[..., None]
    else:
        coarse = cv2.GaussianBlur(take_a, (0, 0), sigmaX=sigma, sigmaY=sigma)[..., None]
        fine = cv2.GaussianBlur(take_a, (0, 0), sigmaX=2.0, sigmaY=1.0)[..., None]
    a_low = cv2.GaussianBlur(a, (0, 0), sigma)
    b_low = cv2.GaussianBlur(b, (0, 0), sigma)
    joined = a_low * coarse + b_low * (1.0 - coarse) + (a - a_low) * fine + (b - b_low) * (1.0 - fine)
    out = np.concatenate([rgb[:, :half - k], joined, rgb[:, half + k:]], axis=1)
    return np.roll(out, -(out.shape[1] // 2), axis=1)  # back about where it was


def _quilted(rgb: np.ndarray, wide: bool) -> np.ndarray:
    h, w = rgb.shape[:2]
    for second in (False, True):  # left to right, then top to bottom (keeping the first)
        if second:
            rgb = np.ascontiguousarray(rgb.transpose(1, 0, 2))
        guide = cv2.GaussianBlur(_lab(rgb), (0, 0), 1.0)
        rgb = _quilt_columns(rgb, guide, closed=second, wide=wide)
        if second:
            rgb = np.ascontiguousarray(rgb.transpose(1, 0, 2))
    return np.clip(resize_periodic(rgb, w, h), 0.0, 255.0)


def make_seamless(image, even_light: bool = True) -> np.ndarray:
    """``image`` (a path or an array, see :func:`.fills.image_rgb`) made to tile, as float32 RGB 0-255 of the same
    size (see the module docs). ``even_light``: even out the light first (no wrap: the image does not tile yet)."""
    rgb = image_rgb(image)
    if even_light:
        rgb = equalise_light(rgb, periodic=False)
    quilted = _quilted(rgb, wide=False)  # crisp: the detail is cut, not blended (no ghosts of a pattern)
    score = seam_score(quilted)
    if score <= SEAM_THRESHOLD:
        return quilted
    faded = _quilted(rgb, wide=True)  # smooth fabric with nothing alike to cut along: a wide cross-fade
    return faded if seam_score(faded) < score else quilted


# -- light ----------------------------------------------------------------------------------------------------------


def _light_periodic(log: np.ndarray) -> np.ndarray:
    """The light of a swatch that tiles: the part of ``log`` (H, W) that varies once across it (the Fourier modes of
    one cycle, either way and diagonally), with its mean taken out."""
    spectrum = np.fft.rfft2(log)
    keep = np.zeros(spectrum.shape, bool)
    keep[[0, 1, -1], :2] = True
    keep[0, 0] = False
    return np.fft.irfft2(np.where(keep, spectrum, 0), s=log.shape)


def _light_open(log: np.ndarray) -> np.ndarray:
    """The light of an image that does not tile: a quadratic surface fitted to ``log`` (H, W) by least squares (a
    gradient across it, a vignette), with its mean taken out."""
    h, w = log.shape

    def basis(rows, cols):
        v, u = np.meshgrid(rows / max(1, h - 1) - 0.5, cols / max(1, w - 1) - 0.5, indexing="ij")
        return np.stack([np.ones_like(u), u, v, u * u, u * v, v * v], axis=-1)

    step = max(1, max(h, w) // 128)  # (fitted on every step-th pixel)
    rows, cols = np.arange(0, h, step), np.arange(0, w, step)
    coef = np.linalg.lstsq(basis(rows, cols).reshape(-1, 6), log[::step, ::step].ravel(), rcond=None)[0]
    surface = basis(np.arange(h), np.arange(w)) @ coef
    return surface - surface.mean()


def equalise_light(image, periodic: bool = True, strength: float = 1.0,
                   limits: tuple[float, float] = (0.5, 2.0)) -> np.ndarray:
    """``image`` with even light, as float32 RGB 0-255 (see the module docs): the light that varies across the whole
    image is divided out of all three channels alike, and the mean brightness stays. ``periodic``: the image tiles,
    and its light is the part of its log-luminance that varies once across it (Fourier modes, so it keeps tiling);
    otherwise a quadratic surface (a gradient, a vignette). ``strength``: how much of it to take out; ``limits``: the
    range of the gain."""
    rgb = image_rgb(image)
    lum = (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]) / 255.0
    log = np.log(lum.astype(np.float64) + 0.02)
    light = _light_periodic(log) if periodic else _light_open(log)
    gain = np.exp(-strength * light).astype(np.float32)
    np.clip(gain, *limits, out=gain)
    gain *= np.float32(lum.mean() / max(float((lum * gain).mean()), 1e-6))  # the same mean luminance as before
    return np.clip(rgb * gain[..., None], 0.0, 255.0)


def tiled(image, repeats: int = 2) -> np.ndarray:
    """``image`` laid out ``repeats`` x ``repeats`` times (float32 RGB), to see its seams."""
    rgb = image_rgb(image)
    return np.tile(rgb, (repeats, repeats, 1))


__all__ = ["SEAM_THRESHOLD", "equalise_light", "make_seamless", "resize_periodic", "seam_score", "seam_scores",
           "tiled"]
