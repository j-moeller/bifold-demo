"""Stage 0 of the outfits: which garment each texel of an avatar's body texture belongs to.

A Rocketbox body texture is one atlas for the clothes, the shoes and the bare skin (hands, neck, arms or legs). Each
covered texel gets one of the :data:`REGIONS`:

* ``skin``: never painted by an outfit. It belongs to the face component, which matches the skin to a person's
  photos; :attr:`BodyRegions.skin` is its mask and :attr:`BodyRegions.skin_color` the avatar's own skin colour.
* ``top``: the jacket, coat or other outer layer of the upper body (a shirt worn on its own is the top);
* ``shirt``: an inner layer seen under the top: the shirt front and collar under a jacket, a hood, long sleeves under
  a t-shirt;
* ``bottoms``: trousers, skirt or shorts; ``shoes``;
* ``other``: belts, prints, trims and buttons with a colour of their own. Outfits leave them as they are.

How :func:`label` works:

1. The :mod:`..uvbake` bake gives each texel its bind-pose point and normal, its UV island and the bone that moves it
   most; the bones fall into :data:`GROUPS` (torso, arms, hands, legs, feet, ...).
2. Skin: the avatar's skin colour is sampled from its head texture (below the eyes, on the nose and the forehead, seen
   from the front; the most common colour there, so a beard or a fringe does not count). The hand texels that look
   like it give a colour model of the body's skin (:class:`SkinModel`, a Gaussian in Lab that knows how the painted
   skin darkens in its creases); texels close to it are skin, and so are shaded patches inside the skin (a knee).
3. In the Rocketbox atlases each UV island is one piece of cloth (or of skin): an island is a shoe, part of the
   bottoms or part of the upper body by the bone groups of its texels. So the hem of a trouser leg, which follows the
   foot bone, stays with the trousers, and a coat's skirt, which follows the thighs, with the coat.
4. Colours split the garments: k-means of a slightly blurred atlas (so weaves and small prints average out) in a
   colour space where a fabric's painted shading (darker and duller in the folds) stays one colour, merged by hue.
   In the upper body the colour that covers most of the middle of the back is the ``top``, and colours with a fair
   share of the back belong to it, unless they only show below it (an inner layer's hem). Thin shapes (stripes,
   piping, zips, printed numbers) are ``other``; large colours, and colours at the collar, are the ``shirt`` (an
   inner layer shows at the front, the collar and the cuffs, not on the back); a tie (a narrow island hanging in
   front of the chest) is ``other``. In the bottoms, another colour below the knees is boots (``shoes``), and a small
   one at the waist a belt (``other``), or the hem of a top or a shirt when it has its colour.
5. Overrides (:func:`load_overrides`): a small JSON file per avatar for the ones the automatic split gets wrong:
   rules that relabel texels by their label, island, bone group, side, height or a box in the atlas. The package
   ships overrides for the avatars it knows (``overrides/<avatar>.json``); a file ``<avatar>.json`` in the outfit
   cache folder (:func:`cache_folder`) is applied after them.

Flags (:attr:`BodyRegions.flags`): ``mirrored`` lists the islands whose UV layout runs the other way round on the body
(a print on them would read as in a mirror), ``overlapping`` the islands that share texels with another part of the
body (one texel, two places: a 3D pattern cannot be right on both). The Rocketbox bodies have no mirrored islands; the
only overlapping ones are small parts stacked on each other (earrings, a necklace), which are labelled ``other``.

The labels are made once per avatar (about 3 s for a 1024 texture on a CPU): :func:`body_regions` keeps them while
the model lives, and on disk in the outfit cache folder (the bake's triangle map and the automatic labels, 0.1-0.2 MB
per avatar), so the next start only rebuilds the per-texel maps from the triangle map (under a second).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ...scene.textures import Texture, as_texture
from .. import uvbake

REGIONS = ("skin", "top", "shirt", "bottoms", "shoes", "other")
SKIN, TOP, SHIRT, BOTTOMS, SHOES, OTHER = range(len(REGIONS))
CLOTHES = REGIONS[1:]
OUTSIDE = 255  # the label of texels no triangle covers
GROUPS = ("none", "head", "neck", "torso", "pelvis", "arms", "hands", "legs", "feet")
UPPER_GROUPS = ("head", "neck", "torso", "arms")
LOWER_GROUPS = ("pelvis", "legs")
FEATHER = 1.2  # pixels (at 1024 texels): the Gaussian that softens the region masks (a 2-3 px ramp)
VERSION = 3  # of the labelling: a cache made by another version is made again
OVERRIDES = Path(__file__).with_name("overrides")
_SKIN_DEFAULT = np.array([65.0, 14.0, 18.0], np.float32)  # Lab of a mid skin tone, when an avatar has no face or hands


def bone_group(name: str) -> str:
    """The :data:`GROUPS` entry of a bone, from its name (Biped ``Bip01 L Thigh`` or Mixamo ``LeftUpLeg`` style)."""
    n = name.lower()
    rules = (("pelvis", "pelvis"), ("hips", "pelvis"), ("clavicle", "torso"), ("shoulder", "torso"),
             ("spine", "torso"), ("chest", "torso"), ("neck", "neck"), ("forearm", "arms"), ("upperarm", "arms"),
             ("finger", "hands"), ("thumb", "hands"), ("hand", "hands"), ("thigh", "legs"), ("upleg", "legs"),
             ("calf", "legs"), ("toe", "feet"), ("foot", "feet"), ("leg", "legs"), ("arm", "arms"))
    for word, group in rules:
        if word in n:
            return group
    return "torso" if n in ("bip01", "root", "") else "head"


def texture_of(spec):
    """The :class:`~...scene.textures.Texture` of a character primitive's texture: arrays are RGB(A) in RGB order
    there (like the renderer reads them), anything else goes through ``as_texture``."""
    if isinstance(spec, np.ndarray) and spec.ndim == 3 and spec.shape[2] in (3, 4):
        return Texture.from_array(spec, order="rgb", name="character")
    return as_texture(spec)


def body_primitive(model) -> int | None:
    """Index of the primitive that holds the clothes: named ``*_body`` (Rocketbox), else the largest one that is not a
    head, hair or opacity layer."""
    best, most = None, -1
    for k, prim in enumerate(model.primitives):
        name = prim.name.lower()
        if name.endswith("body"):
            return k
        if any(w in name for w in ("head", "hair", "opacity", "glasses", "lash", "eye")):
            continue
        if len(prim.indices) > most:
            best, most = k, len(prim.indices)
    return best


@dataclass
class BodyRegions:
    """The regions of one avatar's body texture (``size`` x ``size`` texels, row 0 at the top of the image)."""

    avatar: str
    primitive: int  # index of the body primitive in the model
    bake: uvbake.UVBake
    labels: np.ndarray  # uint8 (size, size): index into REGIONS, OUTSIDE where no triangle is
    groups: np.ndarray  # uint8 (size, size): index into GROUPS (0 outside)
    skin_lab: np.ndarray  # (3,) Lab of the avatar's skin (as found on its hands, else its face)
    flags: dict = field(default_factory=dict)  # "mirrored", "overlapping": island ids; "overrides": files applied
    automatic: np.ndarray | None = None  # the labels before the overrides
    _masks: dict = field(default_factory=dict, repr=False)

    @property
    def size(self) -> int:
        return self.bake.size

    @property
    def skin(self) -> np.ndarray:
        """bool mask of the skin texels (outfits never write them)."""
        return self.labels == SKIN

    @property
    def skin_color(self) -> tuple[int, int, int]:
        """The avatar's skin colour (RGB 0-255)."""
        rgb = cv2.cvtColor(self.skin_lab.reshape(1, 1, 3).astype(np.float32), cv2.COLOR_Lab2RGB)[0, 0]
        return tuple(round(float(v)) for v in np.clip(rgb * 255.0, 0, 255))

    def hard(self, region: str) -> np.ndarray:
        """bool mask of ``region``."""
        return self.labels == REGIONS.index(region)

    def mask(self, region: str) -> np.ndarray:
        """float32 mask of ``region``, feathered over 2-3 texels where it meets another garment. It is 0 on the skin
        and outside the triangles, and the masks of the garments add up to 1 on every clothes texel."""
        if region not in self._masks:
            self._masks.update(feather(self.labels))
        return self._masks[region]

    def counts(self) -> dict[str, int]:
        """Texels per region."""
        found = np.bincount(self.labels[self.labels != OUTSIDE].ravel(), minlength=len(REGIONS))
        return {name: int(found[k]) for k, name in enumerate(REGIONS)}

    def image(self, atlas=None, alpha: float = 0.55) -> np.ndarray:
        """A BGR picture of the labels (over the atlas, RGB, when given): for checking them and writing overrides."""
        return overlay(self.labels, atlas, alpha)


COLORS_BGR = np.array([(140, 170, 235), (60, 60, 220), (60, 200, 240), (200, 120, 40), (60, 160, 60),
                       (200, 60, 200)], np.uint8)  # skin, top, shirt, bottoms, shoes, other


def overlay(labels: np.ndarray, atlas=None, alpha: float = 0.55) -> np.ndarray:
    """``labels`` as colours (see :data:`COLORS_BGR`), blended over ``atlas`` (RGB) when given."""
    image = np.zeros((*labels.shape, 3), np.uint8)
    inside = labels != OUTSIDE
    image[inside] = COLORS_BGR[labels[inside]]
    if atlas is None:
        return image
    base = cv2.cvtColor(np.asarray(atlas)[..., :3], cv2.COLOR_RGB2BGR)
    if base.shape[:2] != labels.shape:
        base = cv2.resize(base, labels.shape[::-1], interpolation=cv2.INTER_AREA)
    out = base.copy()
    out[inside] = (base[inside] * (1.0 - alpha) + image[inside] * alpha).astype(np.uint8)
    return out


def feather(labels: np.ndarray) -> dict[str, np.ndarray]:
    """Soft masks of every region of ``labels`` (see :meth:`BodyRegions.mask`)."""
    size = labels.shape[0]
    sigma = FEATHER * size / 1024.0
    clothes = (labels != OUTSIDE) & (labels != SKIN)
    soft = {}
    for k, name in enumerate(REGIONS):
        if name == "skin":
            continue
        soft[name] = cv2.GaussianBlur((labels == k).astype(np.float32), (0, 0), sigma)
    total = np.maximum(sum(soft.values()), 1e-6)
    masks = {"skin": (labels == SKIN).astype(np.float32)}
    for name, m in soft.items():
        masks[name] = np.where(clothes, m / total, 0.0).astype(np.float32)
    return masks


# -- the avatar's skin colour ----------------------------------------------------------------------------------


def _lab(rgb) -> np.ndarray:
    """float32 Lab (L 0-100) of an RGB ``uint8`` image."""
    return cv2.cvtColor(np.asarray(rgb, np.float32) / 255.0, cv2.COLOR_RGB2Lab)


def _distance(lab: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Colour distance to ``ref`` with the lightness counting half (painted shading changes it)."""
    d = lab - ref
    return np.sqrt((0.5 * d[..., 0]) ** 2 + d[..., 1] ** 2 + d[..., 2] ** 2)


def _mode(samples: np.ndarray, radius: float = 12.0, rounds: int = 4) -> np.ndarray:
    """The centre of the biggest group of Lab ``samples``: the median, moved to the median of the samples near it."""
    centre = np.median(samples, axis=0)
    for _ in range(rounds):
        near = samples[_distance(samples, centre) < radius]
        if len(near) < max(5, len(samples) // 5):
            break
        centre = np.median(near, axis=0)
    return centre.astype(np.float32)


def face_skin(model) -> np.ndarray | None:
    """Lab of the skin of ``model``'s face (below the eyes, on the nose and the forehead), or ``None`` without a head
    texture and face bones."""
    from ..face_photo import head_layout

    layout = head_layout(model)
    if layout is None:
        return None
    head = model.primitives[layout.primitive]
    texture = texture_of(head.texture)
    if texture is None:
        return None
    rgb = cv2.GaussianBlur(texture.rgba[..., :3], (0, 0), 1.5)
    lab = _lab(rgb)
    x, y = layout.front[:, 0], layout.front[:, 1]  # y down
    marks, ed = layout.landmarks, layout.eye_distance
    eyes_y, mid_x, nose_y = marks[:2, 1].mean(), marks[:2, 0].mean(), marks[2, 1]
    zones = [(np.abs(x - ex) < 0.25 * ed) & (y > eyes_y + 0.3 * ed) & (y < eyes_y + 0.65 * ed) for ex in marks[:2, 0]]
    zones.append((np.abs(x - mid_x) < 0.12 * ed) & (y > eyes_y + 0.1 * ed) & (y < nose_y - 0.15 * ed))
    zones.append((np.abs(x - mid_x) < 0.45 * ed) & (y > eyes_y - 1.0 * ed) & (y < eyes_y - 0.5 * ed))
    normals = np.asarray(head.normals, np.float64)
    forward = normals[:, 2] / np.maximum(np.linalg.norm(normals, axis=1), 1e-9)
    triangles = np.asarray(head.indices, np.int64).reshape(-1, 3)
    seen = np.zeros(len(x), bool)
    seen[triangles[layout.seen].ravel()] = True
    pick = np.any(zones, axis=0) & (forward > 0.5) & seen
    if pick.sum() < 5:
        return None
    h, w = lab.shape[:2]
    uv = np.asarray(head.uvs, np.float64)[pick]
    rows = np.clip(((1.0 - uv[:, 1]) * h).astype(int), 0, h - 1)
    cols = np.clip((uv[:, 0] * w).astype(int), 0, w - 1)
    return _mode(lab[rows, cols])


# -- labelling ------------------------------------------------------------------------------------------------


def masked_blur(image: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    """``image`` blurred within ``mask`` only (normalised convolution), so neighbouring colours do not bleed in."""
    m = mask.astype(np.float32)
    weight = cv2.GaussianBlur(m, (0, 0), sigma)
    if image.ndim == 3:
        out = cv2.GaussianBlur(image * m[..., None], (0, 0), sigma) / np.maximum(weight, 1e-6)[..., None]
    else:
        out = cv2.GaussianBlur(image * m, (0, 0), sigma) / np.maximum(weight, 1e-6)
    return out.astype(np.float32)


def _kmeans(x: np.ndarray, k: int, seed: int = 0, iterations: int = 25) -> np.ndarray:
    """Centres of ``k`` clusters of the rows of ``x`` (k-means++ start, deterministic)."""
    rng = np.random.default_rng(seed)
    if len(x) > 40000:
        x = x[rng.choice(len(x), 40000, replace=False)]
    k = max(1, min(k, len(x)))
    centres = [x[rng.integers(len(x))]]
    for _ in range(1, k):
        d = np.min(((x[:, None, :] - np.array(centres)[None]) ** 2).sum(-1), axis=1)
        if d.sum() <= 0:
            break
        centres.append(x[rng.choice(len(x), p=d / d.sum())])
    centres = np.array(centres, np.float64)
    for _ in range(iterations):
        nearest = ((x[:, None, :] - centres[None]) ** 2).sum(-1).argmin(axis=1)
        moved = np.array([x[nearest == j].mean(axis=0) if (nearest == j).any() else centres[j]
                          for j in range(len(centres))])
        if np.allclose(moved, centres, atol=1e-3):
            break
        centres = moved
    return centres


def _colour_distance(a: np.ndarray, b: np.ndarray) -> float:
    """How different two :func:`_features` colours are: a change of hue counts fully, a change of chroma between two
    colourful ones (a fabric duller in the shade) less than one between a colour and a grey."""
    ca, cb = np.hypot(a[1], a[2]), np.hypot(b[1], b[2])
    hue = abs(np.arctan2(a[2], a[1]) - np.arctan2(b[2], b[1]))
    hue = min(hue, 2 * np.pi - hue)
    dh = 2.0 * np.sqrt(ca * cb) * np.sin(hue / 2.0)
    dc = (1.0 - 0.3 * min(ca, cb) / max(ca, cb, 1e-6)) * (ca - cb)
    return float(np.sqrt((a[0] - b[0]) ** 2 + dc ** 2 + dh ** 2))


def _same_colour(a: np.ndarray, b: np.ndarray) -> bool:
    """Whether two :func:`_features` colours look like one fabric: colourful ones of about the same hue, or greys
    of about the same lightness."""
    colourful = min(np.hypot(a[1], a[2]), np.hypot(b[1], b[2])) > 12.0
    return _colour_distance(a, b) < (MERGE if colourful else 0.5 * MERGE)


def _merge(centres: np.ndarray, counts: np.ndarray, closer: float) -> np.ndarray:
    """Which merged colour each of the k-means ``centres`` belongs to: the closest two (see :func:`_colour_distance`)
    are merged (their centre weighted by ``counts``) until all are at least ``closer`` apart."""
    groups = [[k] for k in range(len(centres))]
    centre = [np.asarray(c, np.float64) for c in centres]
    weight = [float(n) for n in counts]
    while len(groups) > 1:
        best = None
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                d = _colour_distance(centre[i], centre[j])
                if best is None or d < best[0]:
                    best = (d, i, j)
        d, i, j = best
        if d >= closer:
            break
        total = max(weight[i] + weight[j], 1e-9)
        centre[i] = (centre[i] * weight[i] + centre[j] * weight[j]) / total
        weight[i] = total
        groups[i] += groups[j]
        del groups[j], centre[j], weight[j]
    merged = np.zeros(len(centres), np.int32)
    for g, members in enumerate(groups):
        merged[members] = g
    return merged


def _clusters(features: np.ndarray, mask: np.ndarray, k: int, closer: float, sigma: float) -> np.ndarray:
    """int map: the colour cluster of each texel of ``mask`` (-1 elsewhere), spatially smoothed."""
    x = features[mask].astype(np.float64)
    out = np.full(mask.shape, -1, np.int32)
    if len(x) == 0:
        return out
    centres = _kmeans(x, k)
    nearest = np.empty(len(x), np.int64)
    for start in range(0, len(x), 200000):  # in chunks: (texels, k, 3) at once is large
        part = x[start:start + 200000]
        nearest[start:start + 200000] = ((part[:, None, :] - centres[None]) ** 2).sum(-1).argmin(axis=1)
    merged = _merge(centres, np.bincount(nearest, minlength=len(centres)), closer)
    nearest = merged[nearest]
    n = int(merged.max()) + 1
    if n > 1:  # a majority vote over a few texels removes speckles
        votes = []
        for j in range(n):
            one = np.zeros(mask.shape, np.float32)
            one[mask] = nearest == j
            votes.append(cv2.GaussianBlur(one, (0, 0), sigma)[mask])
        nearest = np.argmax(votes, axis=0)
    out[mask] = nearest
    return out


def _components(mask: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    n, comp, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return n, comp, stats[:, cv2.CC_STAT_AREA]


class SkinModel:
    """The colours of an avatar's skin: a Gaussian in Lab (mean and covariance) fitted to its hands."""

    FLOOR = np.array([8.0, 2.5, 2.5])  # Lab standard deviations added to the hands' own: the body is shaded more
    CAP = np.array([15.0, 5.0, 6.0])  # and at most these, so a sleeve over the hand bones cannot widen it much
    START = np.array([12.0, 4.0, 5.0])  # around the face's colour, to pick the hand texels to learn from

    def __init__(self, mean, covariance):
        self.mean = np.asarray(mean, np.float64)
        std = np.sqrt(np.maximum(np.diag(covariance), 1e-9))
        shrink = np.minimum(1.0, self.CAP / std)
        self.covariance = np.asarray(covariance, np.float64) * np.outer(shrink, shrink)
        self._inverse = np.linalg.inv(self.covariance)

    @classmethod
    def fit(cls, samples: np.ndarray, start) -> SkinModel:
        """Fitted to the Lab rows ``samples`` that look like ``start`` (the face's skin), a few rounds leaving out
        the ones far from the bulk; just ``start`` when too few of them do."""
        model = cls(start, np.diag(cls.START ** 2))
        x = np.asarray(samples, np.float64)
        for _ in range(4):
            keep = x[model.distance(x) < 3.0]
            if len(keep) < 100:
                break
            model = cls(keep.mean(axis=0), np.cov(keep.T) + np.diag(cls.FLOOR ** 2))
        return model

    def distance(self, lab: np.ndarray) -> np.ndarray:
        """Mahalanobis distance of Lab colours (``(..., 3)``) to the skin."""
        d = np.asarray(lab, np.float64) - self.mean
        return np.sqrt(np.maximum(np.einsum("...i,ij,...j->...", d, self._inverse, d), 0.0))


SKIN_DISTANCE = 3.0  # Mahalanobis distance within which a texel is skin (twice that on the hands)
MERGE = 14.0  # garment colours closer than this (in :func:`_features` units) are one colour


def _skin(lab_fine, covered, groups, island, ref, scale) -> tuple[np.ndarray, SkinModel, np.ndarray]:
    """bool skin mask, the skin model and which texels are on islands that are mostly hand."""
    hands = covered & (groups == GROUPS.index("hands"))
    model = SkinModel.fit(lab_fine[hands], ref)
    d = np.full(covered.shape, np.inf, np.float32)
    d[covered] = model.distance(lab_fine[covered])
    skin = d < SKIN_DISTANCE
    # islands that are mostly hand are hands: their creases and nails are skin too
    n_islands = int(island.max()) + 1 if covered.any() else 0
    in_island = np.bincount(island[covered], minlength=n_islands)
    on_hands = np.bincount(island[hands], minlength=n_islands)
    hand_island = on_hands > 0.6 * np.maximum(in_island, 1)
    hand_texels = covered & hand_island[np.maximum(island, 0)]
    skin |= hand_texels & (d < 2.0 * SKIN_DISTANCE)
    kernel = np.ones((3, 3), np.uint8)
    skin = cv2.morphologyEx(skin.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)
    n, comp, area = _components(skin)
    keep = area >= 600 * scale ** 2
    keep[0] = False
    keep |= np.bincount(comp[hand_texels & skin], minlength=n) > 0
    keep[0] = False
    skin = keep[comp]
    # small holes in the skin (a crease, a nail) are skin
    n, comp, area = _components(covered & ~skin)
    small = area < 120 * scale ** 2
    small[0] = False
    skin |= covered & small[comp]
    return skin, model, hand_texels


def _features(lab: np.ndarray) -> np.ndarray:
    """Colour features for the garment clusters: some lightness, and the chroma relative to the lightness, so the
    painted shading of one fabric (darker and duller in the folds) stays one colour while black, grey and white,
    and dark and light fabrics, stay apart."""
    light = lab[..., 0:1]
    return np.concatenate([0.45 * light, 50.0 * lab[..., 1:] / (light + 15.0)], axis=-1).astype(np.float32)


def _thickness(mask: np.ndarray) -> float:
    """How thick the shapes in ``mask`` are: the 90th percentile of the distance to their edge (pixels)."""
    if not mask.any():
        return 0.0
    distance = cv2.distanceTransform(np.pad(mask, 1).astype(np.uint8), cv2.DIST_L2, 3)[1:-1, 1:-1]
    return float(np.percentile(distance[mask], 90))


def _enclaves(labels: np.ndarray, family: np.ndarray, small: float) -> None:
    """Small patches of one garment label inside another one take the label around them (in place)."""
    kernel = np.ones((5, 5), np.uint8)
    for value in (TOP, SHIRT, BOTTOMS, OTHER):
        n, comp, stats, _ = cv2.connectedComponentsWithStats(((labels == value) & family).astype(np.uint8), 8)
        for k in range(1, n):
            if stats[k, cv2.CC_STAT_AREA] >= small:
                continue
            x, y, w, h = stats[k, :4]
            x0, y0 = max(0, x - 3), max(0, y - 3)
            box = comp[y0:y + h + 3, x0:x + w + 3] == k
            ring = cv2.dilate(box.astype(np.uint8), kernel).astype(bool) & ~box
            around = labels[y0:y + h + 3, x0:x + w + 3][ring]
            around = around[(around != OUTSIDE) & (around != SKIN)]
            if len(around) == 0:
                continue
            values, counts = np.unique(around, return_counts=True)
            if counts.max() >= 0.85 * len(around) and values[counts.argmax()] != value:
                labels[y0:y + h + 3, x0:x + w + 3][box] = values[counts.argmax()]


def label(model, primitive: int, bake: uvbake.UVBake, rgb: np.ndarray, skin_ref=None) -> tuple:
    """Automatic labels of ``model.primitives[primitive]`` (``rgb``: its texture at the bake's size).

    Returns ``(labels, groups, skin_lab)`` (see :class:`BodyRegions`)."""
    size = bake.size
    scale = size / 1024.0
    covered = bake.covered
    names = model.skeleton.names
    groups = _groups(names, bake)
    lab = _lab(rgb)
    lab_fine = masked_blur(lab, covered, 1.0 * scale)
    ref = skin_ref if skin_ref is not None else _SKIN_DEFAULT
    skin, skin_model, _ = _skin(lab_fine, covered, groups, bake.island, ref, scale)

    labels = np.full((size, size), OUTSIDE, np.uint8)
    labels[skin] = SKIN
    cloth = covered & ~skin
    # each island: shoes, bottoms or upper body by the bone groups of its cloth texels
    n_islands = int(bake.island.max()) + 1 if covered.any() else 0
    per = np.zeros((n_islands, len(GROUPS)), np.int64)
    np.add.at(per, (bake.island[cloth], groups[cloth]), 1)
    idx = {g: GROUPS.index(g) for g in GROUPS}
    upper = per[:, [idx[g] for g in UPPER_GROUPS]].sum(1) + per[:, idx["none"]] + per[:, idx["hands"]]
    lower = per[:, [idx[g] for g in LOWER_GROUPS]].sum(1)
    kind = np.argmax(np.c_[upper, lower, per[:, idx["feet"]]], axis=1)  # 0 upper body, 1 bottoms, 2 shoes
    texel_kind = np.full((size, size), -1, np.int8)
    texel_kind[cloth] = kind[bake.island[cloth]]
    labels[texel_kind == 2] = SHOES

    features = _features(masked_blur(lab, cloth, 3.0 * scale))
    position, normal = bake.position, bake.normal
    rest = model.skeleton.rest_global[:, :3, 3]
    neck = model.skeleton.find("Neck")
    with np.errstate(invalid="ignore"):
        heights = position[..., 1][covered]
    neck_y = rest[neck, 1] if neck is not None else float(np.percentile(heights, 85))
    hips = model.skeleton.find("Pelvis", "Hips")
    hip_y = rest[hips, 1] if hips is not None else float(np.percentile(heights, 50))
    calf = model.skeleton.find("L Calf", "LeftLeg", "Calf")
    knee_y = rest[calf, 1] if calf is not None else 0.55 * hip_y
    neck_point = rest[neck] if neck is not None else np.array([0.0, neck_y, 0.0])
    spine = np.array([("spine" in n.lower() or "chest" in n.lower()) for n in names] + [False])

    family = texel_kind == 0
    cluster = _clusters(features, family, 5, MERGE, 1.5 * scale)
    upper_colours = []  # (feature centre, region) of the upper body's colours
    if (cluster >= 0).any():
        with np.errstate(invalid="ignore"):
            back = family & spine[bake.bone] & (normal[..., 2] < -0.85) & (np.abs(position[..., 0]) < 0.1) & \
                (position[..., 1] > hip_y + 0.15) & (position[..., 1] < neck_y - 0.08)
            near_neck = family & (np.linalg.norm(position - neck_point, axis=-1) < 0.12)
        n = int(cluster.max()) + 1
        count = np.bincount(cluster[family], minlength=n)
        size_of = count / max(family.sum(), 1)
        back_of = np.bincount(cluster[back], minlength=n) / max(back.sum(), 1)
        neck_of = np.bincount(cluster[near_neck], minlength=n) / np.maximum(count, 1)
        top = int(np.argmax(back_of if back.sum() > 100 * scale ** 2 else size_of))
        back_y = np.where(back, position[..., 1], np.nan)
        top_low = np.nanpercentile(back_y[cluster == top], 20) if back_of[top] > 0 else -np.inf
        for j in range(n):
            # another colour on the back is part of the top too, unless it only shows below it (an inner hem)
            hem = back_of[j] > 0 and np.nanmedian(back_y[cluster == j]) < top_low
            if j == top or (back_of[j] >= 0.12 and not hem):
                region = TOP
            elif _thickness(cluster == j) < 5.0 * scale:
                region = OTHER  # stripes, piping, zips, printed letters
            elif size_of[j] >= 0.12 or (neck_of[j] >= 0.1 and size_of[j] >= 0.003):
                region = SHIRT  # a big inner layer (long sleeves under a t-shirt), or one showing at the collar
            else:
                region = OTHER
            labels[cluster == j] = region
            upper_colours.append((features[cluster == j].mean(axis=0), region))

    family = texel_kind == 1
    cluster = _clusters(features, family, 4, MERGE, 1.5 * scale)
    if (cluster >= 0).any():
        n = int(cluster.max()) + 1
        count = np.bincount(cluster[family], minlength=n)
        main = int(np.argmax(count))
        mean = np.array([features[cluster == j].mean(axis=0) if count[j] else np.zeros(3) for j in range(n)])
        labels[family] = BOTTOMS
        y = np.where(family, position[..., 1], np.nan)
        for j in range(n):
            if j == main or np.linalg.norm(mean[j] - mean[main]) < 12.0:
                continue
            parts, comp, area = _components(cluster == j)
            for c in range(1, parts):
                if area[c] < 50 * scale ** 2:
                    continue
                part = comp == c
                with np.errstate(invalid="ignore"):
                    low, waist = np.mean(y[part] < knee_y), np.mean(y[part] > hip_y - 0.05)
                region = BOTTOMS
                if low >= 0.7 and area[c] >= 0.01 * family.sum():
                    region = SHOES  # boots
                elif waist >= 0.5 and area[c] < 0.15 * family.sum():
                    region = OTHER  # a belt, or a shirt's hem
                    colour = features[part].mean(axis=0)  # ... unless it is the colour of a garment above
                    same = [(_colour_distance(colour, centre), above) for centre, above in upper_colours
                            if _same_colour(colour, centre)]
                    if same:
                        region = min(same)[1]  # the hem of a top or a shirt
                labels[part] = region
    # a tie: a narrow island of its own hanging in front of the middle of the chest
    for k in np.unique(bake.island[texel_kind == 0]):
        part = (bake.island == k) & (texel_kind == 0)
        p = position[part]
        if len(p) and np.ptp(p[:, 0]) <= 0.09 and abs(p[:, 0].mean()) <= 0.02 and np.ptp(p[:, 1]) >= 0.15 \
                and p[:, 2].min() >= 0.02 and p[:, 1].max() < neck_y:
            labels[part] = OTHER
    # small parts stacked in the atlas (earrings, a necklace: one texel in several places) are trinkets
    for k in island_flags(model.primitives[primitive], bake)["overlapping"]:
        part = (bake.island == k) & cloth
        if part.sum() < 0.005 * covered.sum():
            labels[part] = OTHER
    _enclaves(labels, cloth, 400 * scale ** 2)
    # patches of cloth inside the skin that look like shaded skin (a knee, a knuckle) are skin
    n, comp, area = _components(cloth & (labels != OUTSIDE))
    kernel = np.ones((5, 5), np.uint8)
    for k in np.nonzero(area < 4000 * scale ** 2)[0]:
        if k == 0:
            continue
        part = comp == k
        ring = cv2.dilate(part.astype(np.uint8), kernel).astype(bool) & ~part & covered
        if ring.any() and np.mean(labels[ring] == SKIN) > 0.9 and \
                np.median(skin_model.distance(lab_fine[part])) < 2.0 * SKIN_DISTANCE:
            labels[part] = SKIN
    return labels, groups, skin_model.mean.astype(np.float32)


def _groups(names, bake) -> np.ndarray:
    lut = np.array([GROUPS.index(bone_group(n)) for n in names] + [0], np.uint8)
    groups = np.zeros((bake.size, bake.size), np.uint8)
    groups[bake.covered] = lut[bake.bone[bake.covered]]
    return groups


# -- flags -------------------------------------------------------------------------------------------------------


def island_flags(prim, bake: uvbake.UVBake) -> dict[str, list[int]]:
    """``{"mirrored": [...], "overlapping": [...]}``: island ids (see the module docs)."""
    tri = np.asarray(prim.indices, np.int64).reshape(-1, 3)
    uv = np.asarray(prim.uvs, np.float64)
    p = np.asarray(prim.positions, np.float64)
    nrm = np.asarray(prim.normals, np.float64)
    e1, e2 = uv[tri[:, 1]] - uv[tri[:, 0]], uv[tri[:, 2]] - uv[tri[:, 0]]
    uv_area = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    face = np.cross(p[tri[:, 1]] - p[tri[:, 0]], p[tri[:, 2]] - p[tri[:, 0]])
    outward = np.einsum("nd,nd->n", face, nrm[tri].sum(axis=1))
    way = np.sign(uv_area) * np.sign(outward)  # +1 or -1: how the UV layout runs against the surface
    weight = np.abs(uv_area)
    island = uvbake.islands(prim)
    n = int(island.max()) + 1 if len(island) else 0
    per = np.bincount(island, weights=way * weight, minlength=n)
    usual = 1.0 if (way * weight).sum() >= 0 else -1.0
    mirrored = [int(k) for k in np.nonzero(np.sign(per) == -usual)[0]]
    texels = np.bincount(bake.island[bake.covered], minlength=n)
    shared = np.bincount(bake.island[bake.covered & bake.overlap], minlength=n)
    overlapping = [int(k) for k in np.nonzero(shared > np.maximum(20, 0.01 * texels))[0]]
    # an island drawn under another one shows no texels of its own: it shares all of its texture
    drawn = np.zeros(n, bool)
    drawn[np.unique(bake.island[bake.covered])] = True
    area = np.bincount(island, weights=weight, minlength=n) * 0.5 * bake.size ** 2  # in texels
    overlapping = sorted(set(overlapping) | {int(k) for k in np.nonzero(~drawn & (area > 20))[0]})
    return {"mirrored": mirrored, "overlapping": overlapping}


# -- overrides ---------------------------------------------------------------------------------------------------


def load_overrides(avatar: str, folder=None) -> list[tuple[Path, dict]]:
    """The override files of ``avatar``: the package's own, then ``<folder>/<avatar>.json`` (``folder``: the outfit
    cache folder), as ``(path, content)``.

    A file holds ``{"rules": [...]}``; each rule names a ``region`` and the texels it takes, all of the conditions in
    ``where`` together: ``label`` (a region name or a list of them), ``island`` (ids), ``group`` (bone groups, see
    :data:`GROUPS`), ``bone`` (bone names), ``side`` (``"left"`` or ``"right"`` of the body: +X or -X), ``facing``
    (``"front"`` or ``"back"``), ``below`` / ``above`` (bind-pose height in metres) and ``uv`` (a box
    ``[u0, v0, u1, v1]`` in texture coordinates, v up). For example a long robe whose lower part the bones call the
    bottoms: ``{"rules": [{"region": "top", "where": {"label": "bottoms"}}]}``."""
    found = []
    for base in (OVERRIDES, None if folder is None else Path(folder)):
        if base is None:
            continue
        path = base / f"{_safe(avatar)}.json"
        if path.is_file():
            found.append((path, json.loads(path.read_text(encoding="utf-8"))))
    return found


def apply_overrides(regions: BodyRegions, rules, model) -> np.ndarray:
    """The labels of ``regions`` with the override ``rules`` applied in order (a new array)."""
    labels = regions.labels.copy()
    bake = regions.bake
    names = model.skeleton.names
    for rule in rules:
        target = rule.get("region")
        if target not in REGIONS:
            raise ValueError(f"override rule {rule!r}: region must be one of {', '.join(REGIONS)}")
        where = rule.get("where", {})
        take = bake.covered.copy()
        for key, value in where.items():
            values = value if isinstance(value, list) and key != "uv" else [value]
            if key == "label":
                take &= np.isin(labels, [REGIONS.index(v) for v in values])
            elif key == "island":
                take &= np.isin(bake.island, [int(v) for v in values])
            elif key == "group":
                take &= np.isin(regions.groups, [GROUPS.index(v) for v in values])
            elif key == "bone":
                take &= np.isin(bake.bone, [names.index(v) for v in values])
            elif key == "side":
                with np.errstate(invalid="ignore"):
                    take &= (bake.position[..., 0] > 0) if value == "left" else (bake.position[..., 0] < 0)
            elif key == "facing":
                with np.errstate(invalid="ignore"):
                    take &= (bake.normal[..., 2] >= 0) if value == "front" else (bake.normal[..., 2] < 0)
            elif key in ("below", "above"):
                with np.errstate(invalid="ignore"):
                    y = bake.position[..., 1]
                    take &= (y < float(value)) if key == "below" else (y > float(value))
            elif key == "uv":
                u0, v0, u1, v1 = (float(v) for v in value)
                rows = (1.0 - np.arange(bake.size) / bake.size)[:, None]
                cols = (np.arange(bake.size) / bake.size)[None, :]
                take &= (cols >= u0) & (cols <= u1) & (rows >= v0) & (rows <= v1)
            else:
                raise ValueError(f"override rule {rule!r}: unknown condition {key!r}")
        labels[take] = REGIONS.index(target)
    return labels


# -- making and caching ------------------------------------------------------------------------------------------


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


CACHE_ENV = "BIFOLD_OUTFIT_CACHE"


def cache_folder(model) -> Path | None:
    """Where the labels of ``model`` are kept: ``$BIFOLD_OUTFIT_CACHE`` if set, else ``<characters folder>/outfit``
    for an avatar loaded from a prepared library (its body texture is a file in ``<characters folder>/avatars``),
    else ``None``."""
    if os.environ.get(CACHE_ENV):
        return Path(os.environ[CACHE_ENV]).expanduser()
    index = body_primitive(model)
    texture = model.primitives[index].texture if index is not None else None
    if isinstance(texture, (str, os.PathLike)):
        path = Path(texture)
        if path.parent.name == "avatars":
            return path.parent.parent / "outfit"
    return None


def _texture_rgb(prim, size: int) -> np.ndarray:
    texture = texture_of(prim.texture)
    if texture is None:
        return np.full((size, size, 3), 128, np.uint8)
    rgb = texture.rgba[..., :3]
    if rgb.shape[:2] != (size, size):
        rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(rgb)


def texture_size(prim) -> int:
    """The bake size for a primitive: its texture's larger side (1024 when it has none)."""
    texture = texture_of(prim.texture)
    return 1024 if texture is None else int(max(texture.width, texture.height))


def _key(model, primitive: int, rgb: np.ndarray, skin_ref) -> str:
    prim = model.primitives[primitive]
    h = hashlib.blake2b(f"regions{VERSION}:{prim.name}".encode(), digest_size=12)
    for array in (prim.positions, prim.uvs, prim.indices, prim.joints, prim.weights, rgb,
                  np.zeros(3) if skin_ref is None else skin_ref):
        a = np.ascontiguousarray(array)
        h.update(f"{a.dtype}{a.shape}".encode())
        h.update(a.tobytes())
    h.update("\n".join(model.skeleton.names).encode())
    return h.hexdigest()


_memory: dict[int, tuple] = {}


def body_regions(model, cache: bool | os.PathLike | str = True, overrides: bool = True) -> BodyRegions:
    """The :class:`BodyRegions` of ``model`` (kept while the model lives).

    ``cache``: ``True`` keeps them on disk in :func:`cache_folder` (when the avatar comes from a prepared library and
    the folder can be written), a folder keeps them there, ``False`` makes them afresh. ``overrides``: apply the
    override files (:func:`load_overrides`)."""
    hit = _memory.get(id(model))
    if hit is not None and hit[0] is model and hit[1] == (str(cache), overrides):
        return hit[2]
    regions = _make(model, cache, overrides)
    _memory[id(model)] = (model, (str(cache), overrides), regions)
    return regions


def _make(model, cache, overrides: bool) -> BodyRegions:
    primitive = body_primitive(model)
    if primitive is None:
        raise ValueError(f"{model.name} has no body to dress")
    prim = model.primitives[primitive]
    size = texture_size(prim)
    rgb = _texture_rgb(prim, size)
    skin_ref = face_skin(model)
    folder = cache_folder(model) if cache is True else (Path(cache) if cache else None)
    path = folder / f"{_safe(model.name)}.npz" if folder is not None else None
    key = _key(model, primitive, rgb, skin_ref)
    stored = _read(path, key)
    if stored is not None:
        triangle, overlap, labels, skin_lab = stored
        bake = uvbake.from_triangles(model, primitive, triangle, overlap)
        groups = _groups(model.skeleton.names, bake)
    else:
        triangle, overlap = uvbake.rasterise(prim, size)
        bake = uvbake.from_triangles(model, primitive, triangle, overlap)
        labels, groups, skin_lab = label(model, primitive, bake, rgb, skin_ref)
        _write(path, key, triangle, overlap, labels, skin_lab)
    regions = BodyRegions(model.name, primitive, bake, labels, groups, np.asarray(skin_lab, np.float32),
                          island_flags(prim, bake), automatic=labels)
    if overrides:
        files = load_overrides(model.name, folder if folder is not None else cache_folder(model))
        rules = [rule for _, content in files for rule in content.get("rules", [])]
        if rules:
            regions.labels = apply_overrides(regions, rules, model)
        regions.flags["overrides"] = [str(p) for p, _ in files]
    return regions


def _read(path: Path | None, key: str):
    if path is None or not path.is_file():
        return None
    try:
        with np.load(path) as z:
            if str(z["key"]) != key:
                return None
            size = int(z["size"])
            overlap = np.unpackbits(z["overlap"])[: size * size].reshape(size, size).astype(bool)
            return z["triangle"].astype(np.int32), overlap, z["labels"], z["skin_lab"]
    except (OSError, ValueError, KeyError):
        return None  # a damaged or old file: made again


def _write(path: Path | None, key: str, triangle, overlap, labels, skin_lab) -> None:
    if path is None:
        return
    small = np.int16 if triangle.max() < 32767 else np.int32
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, key=np.array(key), size=np.array(triangle.shape[0]),
                            triangle=triangle.astype(small), overlap=np.packbits(overlap), labels=labels,
                            skin_lab=np.asarray(skin_lab, np.float32))
        os.replace(temporary, path)
    except OSError:
        pass  # a read-only library: keep them in memory only
