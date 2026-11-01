"""Seamless fabric swatches without a model: how well an image tiles, a repair that makes any image tile, even light.

A swatch is one repeat of a fabric: :class:`~.fills.Swatch` lays it side by side over the clothes, so its right edge
meets its own left edge and its bottom its top. Three tools work on any image, a generated fabric or a photo:

* :func:`seam_score`: how much the wrap (the right edge against the left one, the bottom against the top) stands
  out from the rest of the image. Three signs of a seam are measured in CIE Lab, on the image at about 256 pixels:

  - a step: the colour difference between the last and the first column (row), averaged along the seam, at four
    scales (1, 2, 4 and 8 pixels), against the strongest such difference between two neighbouring columns (rows)
    inside the image; and the same along each eighth of the seam (one motif cut by the edge). A seam makes the wrap
    the strongest edge of the image; a checked fabric whose checks happen to meet at the edge has edges as strong
    inside, so it is not taken for one;
  - a band: a line along the wrap that is lighter, darker or of another colour than the cloth on either side of it
    (the dark rim a decoder with zero padding paints), against the strongest such line elsewhere, 2-8 pixels wide;
  - a broken rhythm (:func:`_rhythm`): a regular pattern repeats, so a window of 16 columns (rows) inside it has a
    twin a repeat away, and the window across a seam that breaks the rhythm has none; and with wider windows (32-128
    columns), for bands wider than that, the window across the wrap against the windows one repeat away on either
    side, when every window inside matches its own. A repeat longer than about half the swatch cannot be checked.

  The score is the largest ratio: about 1 or less for an image that tiles, and more where a seam shows. Measured on
  synthetic textures at 512 pixels: of 310 that tile (smooth and crisp blobs, grain, procedural patterns rolled to
  random places, weaves), 4 score above 1.25 (up to 1.29), as do none of 30 prints and 24 wide stripes and checks
  that fit; a seam starts to show at about 1.15-1.2. Caught: all 24 smooth textures cut out of larger ones (2.0 or
  more), all 36 patterns and all 102 wide stripes, checks and gingham whose rhythm breaks at the wrap (12 or more),
  11 of 12 seams along a tenth to a third of the edge, 4 of 6 dark rims of 8-15 %, and 98 of 100 other cut-outs (two
  of a fine twill over blotches pass at 1.02-1.09, its jog at the wrap faint). :data:`SEAM_THRESHOLD` (1.25) is
  where a seed is rejected. A single repeat of a pattern that is mirrored rather than repeated (a tartan sett) can
  score up to about 1.5 though it tiles.

* :func:`make_seamless`: the repair. It rolls the image by half, so the seams cross in its middle and its new edges
  are the old middle (which joins up by itself), then quilts the cross, first left to right, then top to bottom: the
  strips on either side of the seam are laid over each other and cut along the path where they agree best (the
  minimum-error boundary cut of Efros and Freeman's image quilting; in the second pass a path whose ends meet, so
  the first pass's tiling stays). The overlap is chosen among 6-25 % of the width (height): among those whose strips
  differ least (for a regular pattern, those that keep its rhythm), the one whose cut costs least. Each side keeps
  its own pixels up to the cut (feathered over a pixel), so motifs are cut, never blended into ghosts; a slow
  difference across the cut (a shade, light) is evened by an offset that fades out either side, measured only where
  the cut runs through cloth alike on both sides. Where the result still shows a seam and the image has no crisp
  edges (smooth fabric whose strips have nothing alike to cut along: blobs a third of the swatch across), a third
  overlaps instead and the coarse colour cross-fades over all of it. The result is smaller by the overlaps and is
  resampled back to the original size (as a repeat, so the resampling wraps round its edges). The light is evened
  before (no wrap yet) and after. An image that tiles already comes back as it is. It needs no model and gives the
  same image every time: 0.2-0.3 s at 512 pixels, about 0.5 s at 768. Of the 48 smooth cut-outs above it brings 45
  to 1.25 or less (the three left are the smoothest, three blobs across), and 23 of 24 cut-out prints of motifs on a
  plain ground (one scores 1.35 with no seam to be seen: a cluster of motifs near the wrap reads as a band).

* :func:`equalise_light`: even light. A flat-lay photo of fabric, or a generated one, has light falling off towards
  a side or a corner; the tailor puts the garment's own folds and shading on the swatch, so the swatch itself should
  be evenly lit. The light is the slow variation of the log-luminance (once across a swatch that tiles, so it still
  tiles; a gradient or a vignette across an image that does not). It is fitted to how the brightness changes from
  place to place, robustly, so the edges of motifs count for nothing and the layout of a print is not taken for
  light (:func:`_light`). It is taken out only when it is light: it changes the brightness by more than 5 %, it is
  2.5 times stronger than the image's own next slower variations, and it leaves the colours alone (a slow change
  that comes with one is the pattern's: a tartan's sett, two-colour blotches). Then it is divided out of all three
  channels alike, so the colours and the weave stay, and the mean brightness is kept; otherwise the image comes back
  as it is. Measured: evenly lit random prints stay exactly as they are; on prints under a light of 1.5 x from dark
  to light, the ground comes out within 1.02-1.12 x; blotches of 3-6 across, and a pattern whose own repeat is one or
  two across the swatch, keep their light (it cannot be told from the pattern). 15-30 ms at 512 pixels.

:func:`resize_periodic` resamples a repeat with its edges wrapping; :func:`tiled` lays an image out 2 x 2 to look at
its seams.
"""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np

from .fills import image_rgb

SEAM_THRESHOLD = 1.25  # seam_score above this: a visible seam (see the module docs)
ANALYSIS = 256  # the side the seam score looks at the image at
STEP_SCALES = (1, 2, 4, 8)  # pixels at the analysis size
BAND_RADII = (1, 2, 4)  # half widths of the bands looked for, at the analysis size
RHYTHM_HALVES = (8, 16, 32, 64)  # the rhythm test: windows of twice these many columns (rows), at the analysis size
SEGMENT = 32  # the step test also looks along each part of the seam this many rows (columns) long, at the analysis size
FLOOR = 1.0  # CIE Lab units added to both sides of a ratio, so flat images score 1, not noise over noise
OVERLAP = (0.06, 0.25)  # the repair's overlap, as a share of the width (height)
WIDE_OVERLAP = 0.33  # the overlap of the repair's wide cross-fade, for smooth fabric
CRISP = 0.015  # an image with strong edges in more than this share of its pixels is never cross-faded
LIGHT_GRID = 64  # the light is estimated on the image box-filtered to this many pixels across
LIGHT_MIN = 0.05  # light that changes the brightness by less than this (5 %) across the image is left alone
LIGHT_PROMINENCE = 2.5  # ... and light not this much stronger than the image's own next slower variations
LIGHT_COLOUR = 0.5  # ... and light that changes the colours by more than this share of how it changes the brightness


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
    """The step ratio of the vertical seam (the columns' wrap) of ``lab``: along the whole seam, and along each
    part of :data:`SEGMENT` rows (a seam along part of the edge, one motif cut)."""
    worst = 0.0
    for k in STEP_SCALES:
        if lab.shape[1] // k < 4:
            break
        b = _shrink(lab, k)
        diff = np.linalg.norm(np.roll(b, -1, axis=1) - b, axis=2)  # diff[:, j]: columns j and j + 1
        col = diff.mean(axis=0)
        worst = max(worst, float((col[-1] + FLOOR) / (col[:-1].max() + FLOOR)))
        parts = diff.shape[0] * k // SEGMENT
        if k <= 2 and parts >= 2:
            rows = diff.shape[0] // parts
            seg = diff[:rows * parts].reshape(parts, rows, -1).mean(axis=1)  # seg[s, j]: part s of columns j, j + 1
            worst = max(worst, float((seg[:, -1].max() + FLOOR) / (seg[:, :-1].max() + FLOOR)))
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


def _rhythm(lab: np.ndarray, halves: tuple[int, ...] = RHYTHM_HALVES) -> float:
    """The rhythm ratio of the vertical seam of ``lab``. A regular pattern repeats, so a window of columns inside it
    has a twin a repeat away; one across a seam that breaks the rhythm has none. Two tests:

    - the nearest twin, with windows of ``2 * halves[0]`` columns (a jog of a fine pattern): how far the window
      across the wrap is from the most alike window elsewhere, against the same for the windows inside (their 99th
      percentile);
    - the repeat, with wider windows (``halves[1:]``: a band of the wrong width in a pattern of wide stripes or
      checks, inside which a narrow window has twins everywhere): the shift at which every window inside matches
      the one that far to its right (when there is one: a pattern, not a texture), and the window across the wrap
      against the windows that far on either side, against the worst match inside."""
    h, w = lab.shape[:2]
    x = lab.transpose(1, 0, 2).reshape(w, -1).astype(np.float64)
    sq = (x ** 2).sum(axis=1)
    d = np.maximum(sq[:, None] + sq[None, :] - 2.0 * (x @ x.T), 0.0) / h  # mean squared distance of two columns
    j = np.arange(w)
    twin = (j[:, None] + j[None, :]) % w
    near = d[j[:, None], twin]  # near[i, g]: column i against the column g to its right
    total = np.concatenate([np.zeros((1, w)), np.cumsum(np.concatenate([near, near]), axis=0)])
    worst = 0.0
    for m in halves:
        if w < 4 * m:
            break
        start = (j - m + 1) % w
        raw = total[start + 2 * m] - total[start]  # [i, g]: columns i - m + 1 .. i + m against g further on
        straddle = (j >= w - m) | (j < m - 1)  # windows that reach across the wrap (the wrap's own is w - 1)
        apart = (j > m) & (j < w - m)  # (not itself, nor overlapping)
        if m == halves[0]:  # the nearest twin of each window
            window = np.where(apart[None, :] & ~straddle[twin], raw, np.inf)
            distance = np.sqrt(window.min(axis=1) / (2 * m))
            worst = max(worst, float((distance[-1] + FLOOR) / (np.percentile(distance[~straddle], 99) + FLOOR)))
            continue
        # the repeat of the pattern, from the inside (the shift at which pairs of windows inside, the second to the
        # right of the first without passing the wrap, match best), and the window across the wrap against those a
        # repeat away on either side (inside, when the repeat is long)
        inside = ~straddle[:, None] & ~straddle[twin] & apart[None, :] & (j[:, None] + j[None, :] < w)
        farthest = np.where(inside, raw, -np.inf).max(axis=0)  # [g]: the worst match of a pair g apart
        farthest[inside.sum(axis=0) < np.maximum(w // 8, j // 2)] = np.inf  # (pairs over half a repeat at least)
        g = int(np.argmin(farthest))
        known = farthest[np.isfinite(farthest)]
        if not known.size or farthest[g] > 0.1 * np.median(known):
            continue  # (no shift at which every window matches: a texture, not a repeating pattern)
        across = max(raw[w - 1, g], raw[w - 1, w - g])  # (a repeat away on either side)
        worst = max(worst, float((np.sqrt(across / (2 * m)) + FLOOR) / (np.sqrt(farthest[g] / (2 * m)) + FLOOR)))
    return worst


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


def _path(cost: np.ndarray, start: int | None = None, reach: int = 1) -> tuple[float, np.ndarray]:
    """The top-to-bottom path through ``cost`` (H, w) of least total cost that moves at most ``reach`` columns per row
    (starting at column ``start`` when given): ``(its cost, its column in each row)``."""
    h, w = cost.shape
    total = cost[0].astype(np.float64).copy()
    if start is not None:
        total[np.arange(w) != start] = np.inf
    shifts = np.arange(-reach, reach + 1)
    options = np.full((len(shifts), w), np.inf)
    columns = np.arange(w)
    back = np.zeros((h, w), np.int8)
    for y in range(1, h):
        for k, s in enumerate(shifts):  # option k: from column x + s of the row before
            if s < 0:
                options[k, -s:] = total[:s]
            elif s > 0:
                options[k, :-s] = total[s:]
            else:
                options[k] = total
        choice = np.argmin(options, axis=0)
        back[y] = shifts[choice]
        total = cost[y] + options[choice, columns]
    if start is not None:  # it ends next to where it started: a closed path round a cylinder
        total[np.abs(np.arange(w) - start) > 1] = np.inf
    cut = np.empty(h, np.int64)
    cut[-1] = int(np.argmin(total))
    for y in range(h - 1, 0, -1):
        cut[y - 1] = cut[y] + back[y, cut[y]]
    return float(total[cut[-1]]), cut


def _min_cut(cost: np.ndarray, closed: bool = False) -> tuple[float, np.ndarray]:
    """The cheapest top-to-bottom path through ``cost`` (H, w): ``(its cost per row, its column row by row)``;
    ``closed``: its two ends meet (at most a column apart), so that the image it cuts still tiles top to bottom. The
    path is found on pairs of rows (moving up to two columns a pair), so the loop over rows, Python's part of the
    work, is half as long; a closed path is the cheapest one that starts where the cheapest open one does."""
    h = cost.shape[0]
    pairs = h // 2
    if pairs < 32:
        total, cut = _path(cost)
        if closed:
            total, cut = _path(cost, int(cut[0]))
        return total / h, cut
    small = cost[:2 * pairs:2] + cost[1:2 * pairs:2]
    if h % 2:
        small[-1] += cost[-1]
    total, cut = _path(small, reach=2)
    if closed:
        total, cut = _path(small, int(cut[0]), reach=2)
    cut = np.repeat(cut, 2)
    return total / h, (np.r_[cut, cut[-1]] if h % 2 else cut)


def _smooth_rows(values: np.ndarray, weights: np.ndarray, closed: bool) -> np.ndarray:
    """``values`` (h, c) smoothed down the rows by a Gaussian that counts each row by its weight (h,), wrapping round
    when ``closed``; where no row nearby counts, it fades to nothing."""
    h = values.shape[0]
    sigma = max(2.0, h / 64)
    pad = int(3 * sigma) + 1
    mode = "wrap" if closed else "reflect"
    both = np.concatenate([values * weights[:, None], weights[:, None]], axis=1).astype(np.float32)
    padded = np.ascontiguousarray(np.pad(both, ((pad, pad), (0, 0)), mode=mode))
    smooth = cv2.GaussianBlur(padded, (1, 0), sigmaX=0.1, sigmaY=sigma)[pad:pad + h]
    return smooth[:, :-1] / np.maximum(smooth[:, -1:], 0.5)


def _overlap(guide: np.ndarray, half: int, low: int, high: int) -> tuple[int, np.ndarray]:
    """The overlap of the strips either side of column ``half`` of ``guide``, ``low`` to ``high`` columns, and its
    costs: among the overlaps whose strips differ least (those keep a regular pattern's rhythm), the one whose
    cheapest cut costs least (it has room to go round motifs)."""
    ks = np.arange(low, high + 1)  # strip a: the k columns left of the seam; strip b: the k right of it
    diffs = np.array([np.mean((guide[:, half - k:half] - guide[:, half:half + k]) ** 2) for k in ks])
    minima = (diffs <= np.r_[diffs[1:], np.inf]) & (diffs <= np.r_[np.inf, diffs[:-1]])
    candidates = ks[minima & (diffs <= 1.25 * diffs.min() + 1e-6)]
    chosen, chosen_cost, chosen_total = None, None, np.inf
    for k in candidates[np.argsort(diffs[candidates - low])][:4]:
        cost = _cut_costs(guide, half, int(k))
        total = _min_cut(cost)[0]
        if total < chosen_total:
            chosen, chosen_cost, chosen_total = int(k), cost, total
    return chosen, chosen_cost


def _cut_costs(guide: np.ndarray, half: int, k: int) -> np.ndarray:
    """What a cut through each pixel of the overlap of ``k`` columns costs: the strips' difference there, blurred a
    little; its first and last columns are kept out of reach."""
    cost = cv2.GaussianBlur(np.sum((guide[:, half - k:half] - guide[:, half:half + k]) ** 2, axis=2), (0, 0), 1.0)
    margin = max(2, k // 12)
    cost[:, :margin] += 1e6
    cost[:, k - margin:] += 1e6
    return cost


def _quilt_columns(rgb: np.ndarray, guide: np.ndarray, closed: bool, wide: bool) -> np.ndarray:
    """``rgb`` made to tile left to right (narrower by the overlap): rolled by half, the strips either side of the
    seam in its middle overlaid and cut where they agree best. ``guide``: the image the costs are measured on;
    ``closed``: keep the image tiling top to bottom (the cut's ends meet); ``wide``: a third of the width overlaps
    and the coarse colour cross-fades over all of it, for smooth fabric (else the overlap of :func:`_overlap`).

    Each side keeps its own pixels up to the cut (feathered over a pixel or two), so motifs are cut, never blended
    into ghosts. What differs slowly across the cut (a little more light on one side, a shade) is evened out by an
    offset: half the difference at the cut, taken from one side and given to the other, fading out over the
    overlap's width either way. Only the rows where the cut runs through cloth alike on both sides measure that
    difference (where it has to cross a motif, the motif would be smeared along the cut)."""
    h, w = rgb.shape[:2]
    half = w // 2
    rgb, guide = np.roll(rgb, half, axis=1), np.roll(guide, half, axis=1)
    low, high = max(4, round(w * OVERLAP[0])), max(5, min(half - 1, round(w * OVERLAP[1])))
    if wide:
        k = max(5, min(half - 1, round(w * WIDE_OVERLAP)))
        cost = _cut_costs(guide, half, k)
    else:
        k, cost = _overlap(guide, half, low, high)
    a, b = rgb[:, half - k:half], rgb[:, half:half + k]
    cut = _min_cut(cost, closed)[1]
    take_a = (np.arange(k)[None, :] < cut[:, None]).astype(np.float32)  # a left of the cut, b from it on
    if wide:  # from all a at its outer edge to all b at its: smooth fabric has nothing to cut along
        t = (np.arange(k, dtype=np.float32) + 0.5) / k
        coarse = np.broadcast_to((1.0 - t * t * (3.0 - 2.0 * t))[None, :, None], (h, k, 1))
        fine = cv2.GaussianBlur(take_a, (0, 0), sigmaX=max(2.0, k / 16), sigmaY=1.0)[..., None]
        sigma = max(1.5, k / 8)
        a_low, b_low = cv2.GaussianBlur(a, (0, 0), sigma), cv2.GaussianBlur(b, (0, 0), sigma)
        joined = a_low * coarse + b_low * (1.0 - coarse) + (a - a_low) * fine + (b - b_low) * (1.0 - fine)
        out = np.concatenate([rgb[:, :half - k], joined, rgb[:, half + k:]], axis=1)
        return np.roll(out, -(out.shape[1] // 2), axis=1)  # back about where it was
    fine = cv2.GaussianBlur(take_a, (0, 0), sigmaX=0.8, sigmaY=0.5)[..., None]
    joined = a * fine + b * (1.0 - fine)
    out = np.concatenate([rgb[:, :half - k], joined, rgb[:, half + k:]], axis=1)
    # the slow difference across the cut, row by row: a and b next to the cut, on both sides of it; measured where
    # they are alike (a colour difference of a few units: a shade, light), not where the cut crosses a motif
    rows = np.arange(h)
    near = np.clip(cut[:, None] + np.arange(-2, 2)[None, :], 0, k - 1)
    difference = (a[rows[:, None], near] - b[rows[:, None], near]).mean(axis=1)
    apart = np.linalg.norm(guide[rows[:, None], half - k + near] - guide[rows[:, None], half + near], axis=2).mean(
        axis=1)
    alike = (apart <= max(6.0, 2.5 * float(np.median(apart)))).astype(np.float32)
    difference = _smooth_rows(difference, alike, closed)
    reach = max(8.0, float(k))
    x = np.arange(out.shape[1], dtype=np.float32)[None, :] - (half - k + cut)[:, None].astype(np.float32)
    side = np.clip(1.0 - np.abs(x + 0.5) / reach, 0.0, 1.0)
    side = side * side * (3.0 - 2.0 * side) * np.where(x < 0, -0.5, 0.5)  # -1/2 at the cut on a's side, +1/2 on b's
    out += side[..., None] * difference[:, None, :]
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
    size (see the module docs). An image that tiles already (its :func:`seam_score` is :data:`SEAM_THRESHOLD` or
    less) is returned as it is, so the repair never makes a seamless image worse. ``even_light``: even out the light
    first (no wrap: the image does not tile yet) and once more when it tiles (:func:`equalise_light`)."""
    rgb = image_rgb(image)
    if seam_score(rgb) <= SEAM_THRESHOLD:
        return equalise_light(rgb) if even_light else rgb.copy()
    if even_light:
        rgb = equalise_light(rgb, periodic=False)
    quilted = _quilted(rgb, wide=False)  # crisp: the detail is cut, not blended (no ghosts of a pattern)
    score = seam_score(quilted)
    if score > SEAM_THRESHOLD and not _crisp(rgb):
        faded = _quilted(rgb, wide=True)  # smooth fabric with nothing alike to cut along: a wide cross-fade
        if seam_score(faded) < score:
            quilted = faded
    return equalise_light(quilted) if even_light else quilted


def _crisp(rgb: np.ndarray) -> bool:
    """Whether ``rgb`` has crisp edges (motifs on a ground, stripes) in more than :data:`CRISP` of its pixels:
    a cross-fade would ghost them, so the repair never fades it (smooth blotches, grain or a fine weave it may)."""
    lab = cv2.GaussianBlur(_lab(rgb), (0, 0), 1.5)
    across = np.linalg.norm(np.roll(lab, -1, axis=1) - lab, axis=2)
    down = np.linalg.norm(np.roll(lab, -1, axis=0) - lab, axis=2)
    return float((np.maximum(across, down) > 6.0).mean()) > CRISP


# -- light ----------------------------------------------------------------------------------------------------------


def _modes(periodic: bool, light: bool) -> list[tuple[int, int]]:
    """The slow variations (ky, kx) the light is made of (``light``), or the next slower ones a fabric has of its own,
    to compare it with. A swatch that tiles: Fourier modes of one cycle across it (light: either way and diagonally),
    resp. two, one of each conjugate pair. An image that does not tile: cosines of up to one cycle, i.e. half cycles
    across it (light: a gradient, a vignette), resp. one and a half to two cycles."""
    if periodic:
        band = 1 if light else 2
        return [(ky, kx) for ky in range(-band, band + 1) for kx in range(band + 1)
                if max(abs(ky), kx) == band and (kx > 0 or ky > 0)]
    sizes = (1, 2) if light else (3, 4)
    return [(ky, kx) for ky in range(5) for kx in range(5) if max(ky, kx) in sizes]


def _basis(modes, periodic: bool, h: int, w: int) -> np.ndarray:
    """The functions of ``modes`` on an (h, w) grid (pixel centres): (h, w, n); a periodic mode gives a cosine and a
    sine."""
    u = (np.arange(w) + 0.5) / w
    v = (np.arange(h) + 0.5) / h
    out = []
    for ky, kx in modes:
        if periodic:
            cu, su = np.cos(2 * np.pi * kx * u), np.sin(2 * np.pi * kx * u)
            cv, sv = np.cos(2 * np.pi * ky * v), np.sin(2 * np.pi * ky * v)
            out += [np.outer(cv, cu) - np.outer(sv, su), np.outer(sv, cu) + np.outer(cv, su)]  # cos, sin of the sum
        else:
            out.append(np.outer(np.cos(np.pi * ky * v), np.cos(np.pi * kx * u)))
    return np.stack(out, axis=-1)


def _changes(a: np.ndarray, periodic: bool) -> np.ndarray:
    """How ``a`` (h, w, ...) changes from each pixel to the next one right and the next one down (wrapping round when
    ``periodic``), all in one column: (n, ...)."""
    if periodic:
        steps = (np.roll(a, -1, axis=1) - a, np.roll(a, -1, axis=0) - a)
    else:
        steps = (a[:, 1:] - a[:, :-1], a[1:] - a[:-1])
    return np.concatenate([s.reshape(-1, *a.shape[2:]) for s in steps])


@lru_cache(maxsize=8)
def _light_design(periodic: bool, h: int, w: int) -> tuple[np.ndarray, np.ndarray, int]:
    """The functions the light is fitted with on an (h, w) grid, the light's first and then the next slower
    variations: ``(the functions (h, w, k), how they change (see :func:`_changes`), how many are the light's)``."""
    light, other = _modes(periodic, True), _modes(periodic, False)
    basis = np.concatenate([_basis(light, periodic, h, w), _basis(other, periodic, h, w)], axis=-1)
    design = _changes(basis, periodic)
    basis.flags.writeable = design.flags.writeable = False  # (shared by every call)
    return basis, design, len(light) * (2 if periodic else 1)


def _light(log: np.ndarray, periodic: bool) -> tuple[np.ndarray, float, float]:
    """The light of ``log`` (the log-luminance, (H, W)): ``(its field, with the mean taken out, on the grid of
    :data:`LIGHT_GRID`; how much it changes the brightness across the image, as a factor; how much stronger it is
    than the image's own next slower variations)``.

    It is fitted to how the brightness changes from one place to the next, on the image box-filtered to
    :data:`LIGHT_GRID` across, so a weave averages out: light changes the brightness a little everywhere, the edge
    of a motif a lot in one place. The fit is robust (iteratively reweighted least squares with Tukey's biweight),
    so the edges of motifs, large changes, count for nothing, and the layout of a print (more motifs on one side) is
    not taken for light; the plain ground and the inside of the motifs, where light is all that changes, decide."""
    h, w = log.shape
    gh, gw = (LIGHT_GRID, max(8, round(LIGHT_GRID * w / h))) if h >= w else (max(8, round(LIGHT_GRID * h / w)),
                                                                            LIGHT_GRID)
    if periodic:
        grid = resize_periodic(log.astype(np.float32), gw, gh).astype(np.float64)
    else:
        grid = cv2.resize(log.astype(np.float32), (gw, gh), interpolation=cv2.INTER_AREA).astype(np.float64)
    basis, design, n = _light_design(periodic, gh, gw)
    target = _changes(grid, periodic)
    coef = np.zeros(basis.shape[-1])
    for _ in range(12):
        residual = target - design @ coef
        scale = max(1.4826 * float(np.median(np.abs(residual))), 0.003)  # (a floor: a flat image has no noise)
        z = residual / (4.685 * scale)
        weight = np.where(np.abs(z) < 1.0, (1.0 - z * z) ** 2, 0.0)
        weighted = design * weight[:, None]
        coef = np.linalg.solve(weighted.T @ design + 1e-9 * np.eye(len(coef)), weighted.T @ target)
    size = np.sqrt((basis ** 2).mean(axis=(0, 1))) * np.abs(coef)  # each function's share of the variation
    prominence = float(np.sqrt((size[:n] ** 2).mean()) / max(np.sqrt((size[n:] ** 2).mean()), 1e-9))
    field = basis[..., :n] @ coef[:n]
    return field - field.mean(), float(np.exp(np.ptp(field))), prominence


def _full_size(field: np.ndarray, h: int, w: int, periodic: bool) -> np.ndarray:
    """A light field on its grid, at (h, w): it is smooth, so resampling it is as good as evaluating it there."""
    field = np.asarray(field, np.float32)
    big = resize_periodic(field, w, h) if periodic else cv2.resize(field, (w, h), interpolation=cv2.INTER_CUBIC)
    return big - big.mean()


def _judged(rgb: np.ndarray, periodic: bool) -> tuple[np.ndarray, np.ndarray, dict]:
    """``(the luminance, the light's field on its grid, the verdict of :func:`light_of`)`` of ``rgb``."""
    lum = (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]) / 255.0
    field, change, prominence = _light(np.log(lum.astype(np.float64) + 0.02), periodic)
    verdict = {"change": change, "prominence": prominence, "colour": None, "light": False}
    if change >= 1.0 + LIGHT_MIN and prominence >= LIGHT_PROMINENCE:
        shifted = rgb.astype(np.float64) + 8.0  # (so that dark pixels' colours are not noise)
        colour = [_light(np.log(shifted[..., c] / shifted[..., 1]), periodic)[0] for c in (0, 2)]
        size = [float(np.sqrt((f ** 2).mean())) for f in (field, *colour)]
        verdict["colour"] = float(np.hypot(size[1], size[2]) / max(size[0], 1e-9))
        verdict["light"] = verdict["colour"] <= LIGHT_COLOUR
    return lum, field, verdict


def light_of(image, periodic: bool = True) -> dict:
    """What :func:`equalise_light` makes of the light of ``image``: ``{"change": how much it changes the brightness
    across the image (a factor, 1: none), "prominence": how much stronger it is than the image's own next slower
    variations, "colour": how much the colours change with it, against the brightness (only worked out when the
    other two pass), "light": whether it is taken out}``."""
    return _judged(image_rgb(image), periodic)[2]


def equalise_light(image, periodic: bool = True, strength: float = 1.0,
                   limits: tuple[float, float] = (0.5, 2.0)) -> np.ndarray:
    """``image`` with even light, as float32 RGB 0-255 (see the module docs). The light is estimated from how the
    brightness changes from place to place, robustly, so motifs do not count (:func:`light_of`). It is divided out of
    all three channels alike, keeping the colours and the mean brightness, only when it is light: when it changes
    the brightness by more than :data:`LIGHT_MIN` across the image, is :data:`LIGHT_PROMINENCE` times stronger than
    the image's own next slower variations, and leaves the colours alone (light changes the brightness, not the
    colour: a slow change that comes with one is the pattern's, a tartan's sett, two-colour blotches). Otherwise the
    image is returned as it is. ``periodic``: the image tiles (the light varies once across it, so it keeps tiling);
    else a gradient or a vignette. ``strength``: how much of it to take out; ``limits``: the range of the gain.
    Should the result clip, it is darkened a little instead."""
    rgb = image_rgb(image)
    lum, light, verdict = _judged(rgb, periodic)
    if not verdict["light"]:
        return rgb.copy()
    gain = np.exp(-strength * _full_size(light, *rgb.shape[:2], periodic)).astype(np.float32)
    np.clip(gain, *limits, out=gain)
    gain *= np.float32(lum.mean() / max(float((lum * gain).mean()), 1e-6))  # the same mean luminance as before
    out = rgb * gain[..., None]
    top = float(np.percentile(out.max(axis=2), 99.9))
    if top > 255.0:  # (darker, rather than light grounds clipping to white)
        out *= np.float32(255.0 / top)
    return np.clip(out, 0.0, 255.0)


def tiled(image, repeats: int = 2) -> np.ndarray:
    """``image`` laid out ``repeats`` x ``repeats`` times (float32 RGB), to see its seams."""
    rgb = image_rgb(image)
    return np.tile(rgb, (repeats, repeats, 1))


__all__ = ["SEAM_THRESHOLD", "equalise_light", "light_of", "make_seamless", "resize_periodic", "seam_score",
           "seam_scores", "tiled"]
