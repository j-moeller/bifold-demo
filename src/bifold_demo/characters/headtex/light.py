"""Stage 4: the photos' lighting taken out, the avatar's own soft light put back, and the person's skin tone.

**The light in a photo.** Skin is close to a diffuse surface, and the light falling on a diffuse surface from far
away depends, to within a few per cent, only on which way the surface faces: a smooth function on the sphere of
normals that the nine spherical harmonics of orders 0 to 2 describe (Ramamoorthi and Hanrahan 2001; the lighting model
of DECA and Deep3DFaceRecon). Stage 3 knows the normal of every texel and the colour a photo shows there, so each
photo's light is fitted to its face-skin texels (cheeks, forehead, nose, chin; the brows, eyes, lips and nostrils,
which have colours of their own, are left out, and so are texels whose chroma is far from the rest's, such as a
beard or make-up: :func:`like_skin`): the luminance as nine coefficients times the harmonics of the normal, by
weighted least squares with a little ridge on orders 1 and 2 (a photo sees only half of the sphere of normals),
re-weighted three times (Tukey's biweight) so that cast shadows, highlights, stray hair and misregistered texels drop
out. :class:`Shading` is such a fit, scaled to 1 at the 99th percentile of the fitted texels' shading: the brightest
light the skin shows. Dividing the photo's colours by it takes the photo's low-frequency light out and leaves the
skin as bright as where the light fell squarely: a face lit from one side comes out evenly lit, as if all of it
faced the light (and the avatar's own shading, below, is scaled the same way, so the levels agree where photos and
the avatar's texture meet). Only the shading the fitted texels showed is trusted (:meth:`Shading.clamped`): away
from the normals it was fitted to (under the chin, for a photo from the front) the harmonics are an extrapolation,
and dividing by a value darker than any the skin showed makes bright patches.

Only the luminance is fitted, the same shading divides all three channels. Fitting each channel on its own would also
take out the skin's own colour where it follows the shape of the face (redder cheeks, a paler forehead), which matters
more for a likeness than a coloured light does; a photo's overall colour cast is what the gain compensation of stage 3
evens out between the views, and stage 6 matches to the body. The photos' levels differ after de-lighting (each is
scaled to the brightest light it saw: in a photo of the shadowed side of a face, that is shadow); the gains of stage
3, anchored on the most frontal photo, put them on its level.

**What it cannot take out.** Shadows that one part of the face casts on another (the nose's on the cheek under a
spotlight), sharp highlights, and light so harsh that the shadowed side is lost in the dark or in JPEG noise: the
harmonics are a function of the normal only, and nine of them cannot follow the sharp edge between a lit and an
unlit side (they ring by about a tenth there). Texels a photo shows in deep shadow therefore count less in stage 3's
blend (:func:`light_weight`, after the gains), so another photo that shows them lit takes over. Measured on four
renders of an avatar lit from one side (see the README): with the renderer's ambient light at 0.55 (the fitted light
5 times brighter on the lit side) the difference between the two halves of the face falls from 21 to 3 (Lab
lightness); at 0.35 (18 times) from 30 to 9; at 0.2 and 0.1 (51 times) from 37 and 41 to 10 and 12, and the lit half
comes out 5 to 10 darker than it should. A photo whose fitted light is that harsh (:attr:`Shading.contrast` over 8)
gets a warning.

**The avatar's own soft light.** A Rocketbox head texture has soft light and ambient occlusion painted in (lighter
on top, darker under the chin), and the body texture has the same; a de-lit face on that body would look pasted on.
The same fit on the avatar's own texture (its face and neck skin) gives its painted shading as a function of the
normal (:func:`own_shading`; scaled to 1 at the brightest of the face skin a photo from the front sees, and kept
within :data:`OWN_RANGE`, as the fit is extrapolated over the back of the head). Stage 3 blends in "albedo" (each
photo divided by its shading, the avatar's own texture divided by its own) and the result is multiplied by the
avatar's shading again, so the new head is lit as the old one was. Its texels no photo shows come back exactly as they
were (times the tone change below).

**The skin tone.** The person's skin target is a trimmed median, in Lab, of the cheeks and the forehead (where the
landmarks put them on the head) over all the photos, after de-lighting and the gain compensation, as the finished
texture will show them (:func:`tone`, a :class:`Tone`): the lightest and darkest fifth of the samples (highlights,
shadows, stray hair) and those far from the median colour are dropped first. Stage 6 matches the body's skin to it.
The same median of the avatar's own texture, on the same texels, gives the change that tone-matches the texels no
photo shows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

SH_ORDER_SCALE = (0.282095, 0.488603, 1.092548, 0.315392, 0.546274)  # the real spherical harmonics' constants
RIDGE = (0.0, 0.002, 0.01)  # ridge per order (0, 1, 2), relative to the data's weight
ROUNDS = 3  # re-weighting rounds of the robust fit
TUKEY = 4.685  # Tukey's biweight constant, in robust standard deviations of the relative residuals
MIN_TEXELS = 200  # fewer face-skin texels than this: no fit
SHADING_RANGE = (0.15, 4.0)  # the fitted shading a colour is divided by is clamped to this
RANGE_WIDEN = (0.8, 1.25)  # ... and to the range the fitted texels showed (1st and 99th percentiles), widened so
DARK, LIT = 0.12, 0.45  # light_weight: 0 at this shading or less, 1 from this one up
OWN_RANGE = (0.6, 1.4)  # the avatar's own shading is clamped to this (its fit is extrapolated over the back)
TRIM = (20.0, 80.0)  # skin_tone: percentiles of the lightness kept
TRIM_CHROMA = 2.5  # ... and samples within this many robust deviations of the median a and b
MIN_CHROMA_SPREAD = 6.0  # like_skin: within this of the median a and b at least
NORMALISE_AT = 99.0  # percentile of the fitted shading over the fitted texels that is scaled to 1
CONTRAST_FLOOR = 0.02  # Shading.contrast: the darkest shading counts as at least this (a fit can dip to 0 or below)

LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)  # Rec. 709 luminance of linear RGB


# -- colour spaces ---------------------------------------------------------------------------------------------


def _srgb_table() -> np.ndarray:
    c = np.arange(256, dtype=np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)


SRGB_TO_LINEAR = _srgb_table()


def to_linear(srgb) -> np.ndarray:
    """Linear RGB (float32 0-1) of sRGB values: ``uint8`` 0-255 (through a table), or floats 0-1."""
    srgb = np.asarray(srgb)
    if srgb.dtype == np.uint8:
        return SRGB_TO_LINEAR[srgb]
    c = np.clip(srgb.astype(np.float32), 0.0, 1.0)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)


def to_srgb(linear, as_bytes: bool = True) -> np.ndarray:
    """sRGB of linear RGB (floats): ``uint8`` 0-255, or floats 0-1 with ``as_bytes=False``."""
    c = np.clip(np.asarray(linear, np.float32), 0.0, 1.0)
    s = np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1.0 / 2.4) - 0.055)
    return np.clip(s * 255.0 + 0.5, 0, 255).astype(np.uint8) if as_bytes else s.astype(np.float32)


def lab(linear) -> np.ndarray:
    """CIE Lab (L 0-100) of linear RGB (``(..., 3)`` floats)."""
    a = np.ascontiguousarray(np.clip(np.asarray(linear, np.float32), 0.0, 1.0))
    shape = a.shape
    return cv2.cvtColor(a.reshape(-1, 1, 3), cv2.COLOR_LRGB2Lab).reshape(shape)


def from_lab(values) -> np.ndarray:
    """Linear RGB of CIE Lab values (``(..., 3)``)."""
    a = np.ascontiguousarray(np.asarray(values, np.float32))
    shape = a.shape
    return np.clip(cv2.cvtColor(a.reshape(-1, 1, 3), cv2.COLOR_Lab2LRGB).reshape(shape), 0.0, None)


def luminance(linear) -> np.ndarray:
    return np.asarray(linear, np.float32) @ LUMA


def delta_e(a, b) -> np.ndarray:
    """CIE76 colour difference of Lab values (about 2: just noticeable side by side; 10: plainly different)."""
    return np.linalg.norm(np.asarray(a, np.float64) - np.asarray(b, np.float64), axis=-1)


# -- spherical harmonics ---------------------------------------------------------------------------------------


def sh_basis(normals) -> np.ndarray:
    """``(N, 9)`` real spherical harmonics of orders 0-2 at unit ``normals`` (``(N, 3)``)."""
    n = np.asarray(normals, np.float64).reshape(-1, 3)
    x, y, z = n[:, 0], n[:, 1], n[:, 2]
    c0, c1, c2, c3, c4 = SH_ORDER_SCALE
    return np.stack([np.full_like(x, c0), c1 * y, c1 * z, c1 * x, c2 * x * y, c2 * y * z, c3 * (3.0 * z * z - 1.0),
                     c2 * x * z, c4 * (x * x - y * y)], axis=1)


ORDERS = np.array([0, 1, 1, 1, 2, 2, 2, 2, 2])


@dataclass
class Shading:
    """Low-frequency light as a function of the surface's normal: ``coefficients`` of :func:`sh_basis`, scaled so
    the shading is 1 where the texels it was fitted to were lit most (its 99th percentile there, see the module
    docs). ``residual``: the robust spread of the relative residuals of the fit (0.05: the light explains the colours
    to within 5 %)."""

    coefficients: np.ndarray  # (9,)
    residual: float = math.nan
    texels: int = 0
    contrast: float = 1.0  # the brightest over the darkest shading of the fitted texels (5th and 95th percentiles)
    inliers: float = 1.0  # the share of the fitted texels the robust fit kept
    range: tuple = SHADING_RANGE  # the shading the fitted texels showed (widened): what dividing by it trusts

    def __call__(self, normals) -> np.ndarray:
        """``(N,)`` shading at ``normals`` (``(N, 3)``)."""
        return (sh_basis(normals) @ self.coefficients).astype(np.float32)

    def clamped(self, normals) -> np.ndarray:
        """The shading at ``normals`` within :attr:`range`: away from the normals it was fitted to (under the
        chin, say, for a photo from the front) the harmonics are an extrapolation, and dividing by a value darker
        than any the skin showed would make bright patches."""
        return np.clip(self(normals), *self.range)

    @property
    def direction(self) -> np.ndarray:
        """Where most of the light comes from (character space, unit; zero for even light): the order-1 part."""
        c = self.coefficients
        v = np.array([c[3], c[1], c[2]])
        length = np.linalg.norm(v)
        return v / length if length > 1e-9 else np.zeros(3)

    @property
    def strength(self) -> float:
        """How directional the light is: the order-1 part relative to the even (order-0) part (0: even light)."""
        c = self.coefficients
        return float(np.linalg.norm(c[1:4]) * SH_ORDER_SCALE[1] / max(abs(c[0]) * SH_ORDER_SCALE[0], 1e-9))

    @classmethod
    def flat(cls) -> Shading:
        """Even light: shading 1 everywhere."""
        c = np.zeros(9)
        c[0] = 1.0 / SH_ORDER_SCALE[0]
        return cls(c)


def fit_shading(values, normals, weights=None, *, level=None, ridge=RIDGE, rounds: int = ROUNDS) -> Shading | None:
    """The :class:`Shading` that best explains ``values`` (``(N,)`` luminances of one material, e.g. skin, under
    one light) at ``normals`` (``(N, 3)``), with ``weights`` (``(N,)``); ``None`` with fewer than
    :data:`MIN_TEXELS` usable values. Robust (see the module docs): the fit is repeated ``rounds`` times with Tukey's
    biweight of the relative residuals. The shading is scaled to 1 at the :data:`NORMALISE_AT` percentile of the
    fitted values (of those ``level``, ``(N,)`` bool, picks): the brightest light the skin shows."""
    values = np.asarray(values, np.float64).ravel()
    normals = np.asarray(normals, np.float64).reshape(-1, 3)
    w = np.ones(len(values)) if weights is None else np.asarray(weights, np.float64).ravel().copy()
    w[~np.isfinite(values) | (values <= 0)] = 0.0
    if (w > 0).sum() < MIN_TEXELS:
        return None
    scale = float(np.average(values, weights=w))
    if not scale > 0:
        return None
    y = values / scale
    basis = sh_basis(normals)
    penalty = np.diag(np.asarray(ridge, np.float64)[ORDERS])
    robust = np.ones(len(y))
    coefficients = np.zeros(9)
    sigma = math.nan
    for k in range(rounds + 1):
        weight = w * robust
        total = weight.sum()
        if total <= 0:
            return None
        a = (basis * weight[:, None]).T @ basis / total + penalty
        b = (basis * weight[:, None]).T @ y / total
        coefficients = np.linalg.solve(a, b)
        predicted = basis @ coefficients
        relative = (y - predicted) / np.maximum(np.abs(predicted), 0.05)
        used = w > 0
        sigma = 1.4826 * float(np.median(np.abs(relative[used]))) + 1e-4
        if k < rounds:
            u = relative / (TUKEY * sigma)
            robust = np.where(np.abs(u) < 1.0, (1.0 - u * u) ** 2, 0.0)
    shading = basis @ coefficients
    kept = (w > 0) & (robust > 0)
    if not kept.any():
        return None
    scaled = kept if level is None else kept & np.asarray(level, bool).ravel()
    if not scaled.any():
        scaled = kept
    peak = weighted_percentile(shading[scaled], (w * robust)[scaled], NORMALISE_AT)
    if not peak > 0:
        return None
    coefficients /= peak
    seen = shading[kept] / peak
    low, high = np.percentile(seen, (1.0, 99.0))
    contrast = float(np.percentile(seen, 95) / max(np.percentile(seen, 5), CONTRAST_FLOOR))
    trusted = (max(SHADING_RANGE[0], RANGE_WIDEN[0] * float(low)), min(SHADING_RANGE[1], RANGE_WIDEN[1] * float(high)))
    return Shading(coefficients, sigma, int(kept.sum()), contrast, float(kept.sum() / max((w > 0).sum(), 1)),
                   trusted)


def weighted_percentile(values, weights, q: float) -> float:
    """The ``q``-th percentile of ``values`` with ``weights``."""
    values, weights = np.asarray(values, np.float64).ravel(), np.asarray(weights, np.float64).ravel()
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    if cumulative[-1] <= 0:
        return float(np.percentile(values, q))
    return float(values[order][min(len(values) - 1, np.searchsorted(cumulative, q / 100.0 * cumulative[-1]))])


def like_skin(colours, weights=None) -> np.ndarray:
    """bool per colour (``(N, 3)`` linear RGB): its chroma (Lab a and b, which a white light hardly changes) is
    near the (weighted) median of all: a beard, stubble, make-up or a shadowed patch of another colour is not."""
    values = lab(colours)
    w = np.ones(len(values)) if weights is None else np.asarray(weights, np.float64)
    keep = np.ones(len(values), bool)
    for c in (1, 2):
        m = _weighted_median(values[:, c].astype(np.float64), w)
        spread = 1.4826 * _weighted_median(np.abs(values[:, c] - m).astype(np.float64), w)
        keep &= np.abs(values[:, c] - m) <= max(TRIM_CHROMA * spread, MIN_CHROMA_SPREAD)
    return keep


def delight(colours, normals, shading: Shading) -> np.ndarray:
    """``colours`` (``(N, 3)`` linear RGB) with ``shading`` taken out (divided by it, :meth:`Shading.clamped`)."""
    return (np.asarray(colours, np.float32) / shading.clamped(normals)[:, None]).astype(np.float32)


def light_weight(shading_values) -> np.ndarray:
    """How much a sample counts by the light that fell on it (``shading_values``: the fitted shading there): 0 in
    deep shadow (at :data:`DARK` and less, where the de-lit colour is mostly noise), 1 when well lit (:data:`LIT`)."""
    u = np.clip((np.asarray(shading_values, np.float32) - DARK) / (LIT - DARK), 0.0, 1.0)
    return (u * u * (3.0 - 2.0 * u)).astype(np.float32)


def own_shading(texture_linear, normals, mask, face=None) -> tuple[Shading, np.ndarray]:
    """The avatar's own painted shading (see the module docs): ``(fit, shading per texel)`` from its texture (linear
    RGB, ``(H, W, 3)``) and the texels' ``normals`` (``(H, W, 3)``), fitted on ``mask`` (its face and neck skin),
    scaled to 1 at the brightest of the ``face`` texels (as a photo's light is: the levels match where they meet) and
    clamped to :data:`OWN_RANGE` (the fit is extrapolated over the back of the head). Flat without enough skin."""
    h, w = mask.shape
    covered = np.isfinite(normals).all(axis=-1)
    fit = None
    if mask.sum() >= MIN_TEXELS:
        colours = texture_linear[mask]
        skin = like_skin(colours)
        fit = fit_shading(luminance(colours), normals[mask], skin.astype(np.float64),
                          level=None if face is None else face[mask])
    fit = fit or Shading.flat()
    out = np.ones((h, w), np.float32)
    out[covered] = np.clip(fit(normals[covered]), *OWN_RANGE)
    return fit, out


# -- the skin tone ---------------------------------------------------------------------------------------------


@dataclass
class Tone:
    """A robust colour of one kind of texel (the person's skin, the avatar's skin, hair): the trimmed Lab median of
    ``samples`` texels, their robust ``spread`` (Lab, per channel), and each photo's own median (``per_view``)."""

    lab: np.ndarray  # (3,) L 0-100
    samples: int = 0
    spread: np.ndarray = field(default_factory=lambda: np.zeros(3))
    per_view: dict = field(default_factory=dict)  # view name -> (3,) Lab

    @property
    def linear(self) -> np.ndarray:
        """Linear RGB (0-1)."""
        return from_lab(self.lab.reshape(1, 3))[0]

    @property
    def rgb(self) -> tuple[int, int, int]:
        """sRGB 0-255."""
        return tuple(int(v) for v in to_srgb(self.linear))

    def __str__(self) -> str:
        r, g, b = self.rgb
        return f"#{r:02x}{g:02x}{b:02x} (Lab {self.lab[0]:.0f} {self.lab[1]:+.0f} {self.lab[2]:+.0f})"


def trimmed_median(values_lab, weights=None) -> tuple[np.ndarray, np.ndarray, int]:
    """``(median, spread, kept)`` of Lab samples (``(N, 3)``): the lightness trimmed to :data:`TRIM` percentiles and
    a and b to :data:`TRIM_CHROMA` robust deviations around their median, then the (weighted) median of the rest."""
    v = np.asarray(values_lab, np.float64).reshape(-1, 3)
    w = np.ones(len(v)) if weights is None else np.asarray(weights, np.float64).ravel()
    keep = (w > 0) & np.isfinite(v).all(axis=1)
    if keep.sum() < 5:
        return np.full(3, np.nan), np.full(3, np.nan), 0
    v, w = v[keep], w[keep]
    lo, hi = np.percentile(v[:, 0], TRIM)
    keep = (v[:, 0] >= lo) & (v[:, 0] <= hi)
    for c in (1, 2):
        m = np.median(v[keep, c])
        mad = 1.4826 * np.median(np.abs(v[keep, c] - m)) + 0.5
        keep &= np.abs(v[:, c] - m) <= TRIM_CHROMA * mad
    if keep.sum() < 3:
        keep[:] = True
    v, w = v[keep], w[keep]
    median = np.array([_weighted_median(v[:, c], w) for c in range(3)])
    spread = 1.4826 * np.array([_weighted_median(np.abs(v[:, c] - median[c]), w) for c in range(3)])
    return median, spread, len(v)


def _weighted_median(x: np.ndarray, w: np.ndarray) -> float:
    order = np.argsort(x, kind="stable")
    cumulative = np.cumsum(w[order])
    return float(x[order][np.searchsorted(cumulative, 0.5 * cumulative[-1])])


def tone(samples: dict, weights: dict | None = None) -> Tone | None:
    """The :class:`Tone` of Lab samples pooled over the photos (``{view name: (N, 3) Lab}``, with ``weights`` per
    sample); each photo weighs in by its samples' total weight. ``None`` without enough samples."""
    names = [n for n, s in samples.items() if len(s)]
    if not names:
        return None
    values = np.concatenate([samples[n] for n in names])
    w = np.concatenate([weights[n] if weights else np.ones(len(samples[n])) for n in names])
    median, spread, kept = trimmed_median(values, w)
    if kept == 0:
        return None
    per_view = {}
    for n in names:
        m, _s, k = trimmed_median(samples[n], weights[n] if weights else None)
        if k:
            per_view[n] = m
    return Tone(median, kept, spread, per_view)
