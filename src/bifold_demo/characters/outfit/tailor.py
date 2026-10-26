"""Stage 1 of the outfits: new fabrics on an avatar's clothes that keep the look of its own texture.

The texture of a Rocketbox garment has its folds, seams, stitching, buttons and ambient occlusion painted in. A
:class:`Tailor` splits it, region by region, into a base colour and two luminance layers (:class:`ShadingLayers`)::

    L = luminance of the texture,  B = L blurred within the region (about 12 texels at 1024)
    D = L / B                (detail: weave, stitching, seams, small wrinkles; clipped to 0.35-1.8)
    S = B / ref              (shading: folds and ambient occlusion; ref = the region's 75th percentile of B)

and makes a new texture as ``fill x D x S``: a navy fill gets the jacket's folds and stitching, a tartan bends round
them. A garment sewn from blocks of different colours (a hoodie with a grey body, a black yoke and navy sleeves)
would keep its dark blocks dark under any fill, so each colour block of a region (:func:`.regions.colour_blocks`)
has a ref of its own: L is first divided by the ref of its texel's block (cross-faded over a texel or two where two
blocks meet, which leaves a thin dark seam there), and D and S are made of that. A region of one colour (most) is
one block. The blur stays inside each region (normalised convolution), so a white shirt does not light up the edge of a
black jacket. Where two garments meet, their feathered masks (:meth:`.regions.BodyRegions.mask`) cross-fade over 2-3
texels; skin texels are never written (their mask is 0), and the texture is extended a few texels past the edges of
the islands so mipmapping does not bleed the old colours in. ``fill = texture / (D x S)`` (the region's own albedo,
:attr:`ShadingLayers.albedo`) gives the texture back.

An :class:`Outfit` is a recipe: region -> fill (see :mod:`.fills`), with a name; :func:`presets` are the ones in
``presets.json``. :func:`dress` puts one on a :class:`~..character.Character` as its own body texture
(``character.texture_overrides``; the avatar's other users keep theirs) and :func:`undress` takes it off. The
Tailor of an avatar keeps the labels, cloth coordinates and layers (:func:`tailor_for`; those of the last
:data:`TAILORS` avatars stay in memory), so dressing again costs only the fills: 60-200 ms for a 1024 texture,
0.3-0.7 s for a 2048 one (on a CPU; the first outfit also sorts out each garment's texels, about 50 ms a garment at
1024). An outfit goes on the body texture the character wears (:func:`dress`), so a skin matched to a person's photos
stays.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ...scene.textures import Texture
from . import regions as R
from .fills import Fill, Where, from_spec
from .surface import surface

DETAIL_SIGMA = 12.0  # texels at 1024: what is finer than this is detail, coarser is shading
DETAIL_RANGE = (0.35, 1.8)
SHADING_REFERENCE = 75  # percentile of the blurred luminance that a fill's colour stands for
EPSILON = 0.02  # keeps the ratios of very dark textures from amplifying noise
DILATE = 8  # texels the new texture reaches past the islands (at 1024)
PRESET_FILE = Path(__file__).with_name("presets.json")


def luminance(rgb: np.ndarray) -> np.ndarray:
    """Rec. 709 luminance (0-1) of RGB 0-255."""
    rgb = np.asarray(rgb, np.float32)
    return (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]) / 255.0


@dataclass
class ShadingLayers:
    """The texture of one avatar's body split into detail and shading, per garment region (see the module docs)."""

    detail: np.ndarray  # D, float32 (size, size); 1 on skin and outside
    shading: dict  # region -> S, float32 (size, size), meaningful on and next to the region
    reference: dict  # region -> the luminance a fill's colour stands for (on the region's largest colour block)
    albedo: np.ndarray  # float32 (size, size, 3): the texture divided by D x S (its own base colour)
    blocks: dict = field(default_factory=dict)  # colour block id -> its reference luminance (see colour_blocks)


def _block_references(lum: np.ndarray, hard: np.ndarray, blocks: np.ndarray, sigma: float) -> tuple[dict, np.ndarray]:
    """``{block: reference luminance}`` of the colour blocks of a region (``hard``), largest first, and the luminance
    of the region blurred within it (used when it is one block)."""
    ids, counts = np.unique(blocks[hard], return_counts=True)
    if len(ids) == 1:
        blurred = R.masked_blur(lum, hard, sigma)
        return {int(ids[0]): float(np.percentile(blurred[hard], SHADING_REFERENCE))}, blurred
    refs = {}
    pad = int(4 * sigma) + 1
    for b in ids[np.argsort(-counts, kind="stable")]:
        mask = hard & (blocks == b)
        rows, cols = np.nonzero(mask.any(axis=1))[0], np.nonzero(mask.any(axis=0))[0]
        box = (slice(max(0, rows[0] - pad), rows[-1] + pad + 1), slice(max(0, cols[0] - pad), cols[-1] + pad + 1))
        refs[int(b)] = float(np.percentile(R.masked_blur(lum[box], mask[box], sigma)[mask[box]], SHADING_REFERENCE))
    return refs, None


def shading_layers(rgb: np.ndarray, labels: np.ndarray, blocks: np.ndarray | None = None) -> ShadingLayers:
    """:class:`ShadingLayers` of a texture (RGB, ``(size, size, 3)``) with region ``labels`` and colour ``blocks``
    (:func:`.regions.colour_blocks`; without them each region is one block)."""
    size = labels.shape[0]
    sigma = DETAIL_SIGMA * size / 1024.0
    lum = luminance(rgb)
    if blocks is None:
        blocks = R.colour_blocks(labels, None, rgb)
    detail = np.ones((size, size), np.float32)
    own = np.ones((size, size), np.float32)  # each texel's own region's S
    shading, reference, references = {}, {}, {}
    for k, name in enumerate(R.REGIONS):
        if name == "skin":
            continue
        hard = labels == k
        if not hard.any():
            continue
        refs, blurred = _block_references(lum, hard, blocks, sigma)
        reference[name] = next(iter(refs.values()))
        references.update(refs)
        if blurred is not None:  # one colour: S = B / ref and D = L / B
            ref = reference[name]
            s = ((blurred + EPSILON) / (ref + EPSILON)).astype(np.float32)
            detail[hard] = np.clip((lum[hard] + EPSILON) / (blurred[hard] + EPSILON), *DETAIL_RANGE)
        else:  # colour blocks: the luminance relative to each block's own reference, then split the same way
            soft = R.FEATHER * size / 1024.0
            total = np.zeros(int(hard.sum()), np.float64)
            weight = np.zeros(int(hard.sum()), np.float64)
            for b, ref in refs.items():  # cross-faded over a texel or two where two blocks meet
                w = cv2.GaussianBlur((hard & (blocks == b)).astype(np.float32), (0, 0), soft)[hard]
                total += w * ref
                weight += w
            relative = np.zeros((size, size), np.float32)
            relative[hard] = (lum[hard] + EPSILON) / (total / np.maximum(weight, 1e-9) + EPSILON)
            s = R.masked_blur(relative, hard, sigma)
            detail[hard] = np.clip(relative[hard] / np.maximum(s[hard], 1e-6), *DETAIL_RANGE)
        shading[name] = s
        own[hard] = s[hard]
    albedo = np.asarray(rgb, np.float32) / (detail * own)[..., None]
    return ShadingLayers(detail, shading, reference, albedo, references)


class Outfit:
    """A recipe: ``fills`` maps regions (``"top"``, ``"shirt"``, ``"bottoms"``, ``"shoes"``, ``"other"``) to fills, or
    to anything :func:`.fills.from_spec` takes (``"navy"``, ``"red tartan"``, ``{"pattern": ...}``). Regions it leaves
    out keep the avatar's own clothes."""

    def __init__(self, fills: dict | None = None, name: str = "custom"):
        self.name = name
        self.fills: dict[str, Fill] = {}
        for region, spec in (fills or {}).items():
            if region not in R.CLOTHES:
                raise ValueError(f"an outfit fills {', '.join(R.CLOTHES)}, not {region!r}")
            self.fills[region] = from_spec(spec)

    @classmethod
    def parse(cls, text: str, name: str | None = None) -> Outfit:
        """An outfit from a description (see :func:`.prompts.parse_outfit`), or a preset's name."""
        key = text.strip().lower()
        if key in presets():
            return preset(key)
        from .prompts import parse_outfit

        return cls(parse_outfit(text), name=name or text)

    def spec(self) -> dict:
        """JSON-friendly: ``{"name": ..., "fills": {region: fill spec}}``."""
        return {"name": self.name, "fills": {region: fill.spec() for region, fill in self.fills.items()}}

    @classmethod
    def from_spec(cls, spec: dict) -> Outfit:
        return cls(spec.get("fills", {}), name=spec.get("name", "custom"))

    def __repr__(self) -> str:
        return f"Outfit({self.name!r}, {self.fills})"


_presets: dict | None = None


def presets() -> dict[str, dict]:
    """The outfit recipes shipped in ``presets.json``: ``{name: {region: fill spec}}``."""
    global _presets
    if _presets is None:
        data = json.loads(PRESET_FILE.read_text(encoding="utf-8"))
        _presets = {name.lower(): entry["fills"] for name, entry in data["presets"].items()}
    return _presets


def preset(name: str) -> Outfit:
    """The preset ``name`` (see :func:`presets`) as an :class:`Outfit`."""
    key = name.strip().lower()
    if key not in presets():
        raise KeyError(f"no outfit preset {name!r}; presets: {', '.join(presets())}")
    return Outfit(presets()[key], name=key)


class Tailor:
    """Makes body textures for one avatar (``model``): its regions, cloth coordinates and shading layers, kept so that
    each outfit only costs its fills."""

    def __init__(self, model, regions: R.BodyRegions | None = None):
        self.model = model
        self.regions = regions if regions is not None else R.body_regions(model)
        self.primitive = model.primitives[self.regions.primitive]
        size = self.regions.size
        texture = R.texture_of(self.primitive.texture)
        rgba = texture.rgba if texture is not None else np.full((size, size, 4), 128, np.uint8)
        if rgba.shape[:2] != (size, size):
            rgba = cv2.resize(rgba, (size, size), interpolation=cv2.INTER_AREA)
        self.original = np.ascontiguousarray(rgba)
        self._surface = None
        self._layers = None
        self._texels: dict[str, _Texels] = {}
        self._bleed = None

    @property
    def size(self) -> int:
        return self.regions.size

    @property
    def surface(self):
        if self._surface is None:
            self._surface = surface(self.model, self.regions)
        return self._surface

    @property
    def layers(self) -> ShadingLayers:
        if self._layers is None:
            rgb = self.original[..., :3]
            blocks = R.colour_blocks(self.regions.labels, self.regions.colours, rgb)
            self._layers = shading_layers(rgb, self.regions.labels, blocks)
        return self._layers

    def where(self, region: str) -> _Texels:
        """The texels a fill of ``region`` writes (its feathered mask above 0), sorted by their size on the body."""
        if region not in self._texels:
            mask = self.regions.mask(region)
            rows, cols = (a.astype(np.int32) for a in np.nonzero(mask > 0))
            surf = self.surface
            foot = surf.footprint[rows, cols]
            foot = np.nan_to_num(foot, nan=float(np.nanmedian(foot)) if len(foot) else 0.002)
            order = np.argsort(foot, kind="stable")  # so a fill reads each mip level from one slice
            rows, cols, foot = rows[order], cols[order], foot[order]
            uv = np.nan_to_num(surf.uv[rows, cols], nan=0.0)
            where = Where(uv.astype(np.float32), surf.part[rows, cols], foot.astype(np.float32), rows, cols,
                          self.size)
            layers = self.layers
            factor = layers.detail[rows, cols] * layers.shading[region][rows, cols] if region in layers.shading \
                else np.ones(len(rows), np.float32)
            index = rows * np.int32(self.size) + cols
            self._texels[region] = _Texels(where, index, mask[rows, cols].astype(np.float32),
                                           factor.astype(np.float32))
        return self._texels[region]

    def texture(self, outfit, base=None) -> np.ndarray:
        """The body texture (RGBA ``uint8``, the original's size) with ``outfit`` (an :class:`Outfit`, a dict of
        fills or a description) on it.

        ``base``: the texture to put it on instead of the avatar's own (a :class:`~...scene.textures.Texture`, an RGB(A)
        array in RGB order or a path; resized to the original's size), such as one whose skin the face component
        matched to a person's photos. Its texels outside the garments the outfit fills stay as they are; the folds and
        stitching always come from the avatar's own texture."""
        outfit = _outfit(outfit)
        start = self.original if base is None else _base_rgba(base, self.size)
        flat = start[..., :3].reshape(-1, 3)
        out = flat.astype(np.float32)
        changed = np.zeros(self.size * self.size, bool)
        for region, fill in outfit.fills.items():
            if fill.keeps() or region not in self.layers.shading:
                continue
            texels = self.where(region)
            if len(texels.index) == 0:
                continue
            new = np.asarray(fill.colors(texels.where), np.float32) * texels.factor[:, None]
            np.clip(new, 0.0, 255.0, out=new)
            new -= flat[texels.index]
            new *= texels.weight[:, None]
            out[texels.index] += new  # the weights of the garments add up to at most 1: it stays in 0-255
            changed[texels.index] = True
        self._extend(out, changed)
        rgba = start.copy()
        rgba[..., :3] = (out + 0.5).astype(np.uint8).reshape(self.size, self.size, 3)
        return rgba

    def _extend(self, out: np.ndarray, changed: np.ndarray) -> None:
        """Copy the new colours a few texels past the islands' edges (in place), where they were changed."""
        if self._bleed is None:
            covered = self.regions.bake.covered
            distance, nearest = cv2.distanceTransformWithLabels((~covered).astype(np.uint8), cv2.DIST_L2, 3,
                                                                labelType=cv2.DIST_LABEL_PIXEL)
            ys, xs = np.nonzero(covered)
            index = np.zeros(int(nearest.max()) + 1, np.int32)
            index[nearest[ys, xs]] = ys * self.size + xs  # label -> the covered texel it names
            near = (~covered) & (distance <= DILATE * self.size / 1024.0)
            rows, cols = np.nonzero(near)
            self._bleed = ((rows * self.size + cols).astype(np.int32), index[nearest[rows, cols]])
        target, source = self._bleed
        take = changed[source]
        out[target[take]] = out[source[take]]


@dataclass
class _Texels:
    """What a fill of one region needs, worked out once: where the texels are, and how the original looks there."""

    where: Where
    index: np.ndarray  # int32 flat texel indices
    weight: np.ndarray  # float32: the region's feathered mask
    factor: np.ndarray  # float32: detail x shading


def _base_rgba(base, size: int) -> np.ndarray:
    """RGBA ``uint8`` ``(size, size, 4)`` of a base texture (see :meth:`Tailor.texture`)."""
    texture = R.texture_of(base)
    if texture is None:
        raise ValueError(f"cannot dress over {base!r}")
    rgba = texture.rgba
    if rgba.shape[:2] != (size, size):
        shrink = rgba.shape[0] > size
        rgba = cv2.resize(rgba, (size, size), interpolation=cv2.INTER_AREA if shrink else cv2.INTER_LINEAR)
    return np.ascontiguousarray(rgba)


def _outfit(outfit) -> Outfit:
    if isinstance(outfit, Outfit):
        return outfit
    if isinstance(outfit, dict):
        return Outfit(outfit)
    if isinstance(outfit, str):
        return Outfit.parse(outfit)
    raise TypeError(f"cannot dress in {outfit!r}")


TAILORS = 3  # avatars whose Tailor stays in memory (the last ones dressed; 120-140 MB each at 1024 texels)
_tailors = R.Recent(TAILORS)


def tailor_for(model) -> Tailor:
    """The :class:`Tailor` of ``model``, made once; those of the last :data:`TAILORS` avatars stay in memory."""
    tailor = _tailors.get(model)
    return tailor if tailor is not None else _tailors.put(model, Tailor(model))


def _dressed(texture) -> bool:
    return isinstance(texture, Texture) and getattr(texture, "outfit", None) is not None


def dress(character, outfit, base=None) -> Texture:
    """Put ``outfit`` (an :class:`Outfit`, ``{region: fill}``, a preset's name or a description such as
    ``"red tartan jacket, black trousers"``) on ``character``: its body texture becomes its own
    (``character.texture_overrides``). Dressing again reuses that texture (the renderer uploads the new pixels).

    The outfit goes on the body texture the character wears: the avatar's own, or one somebody else gave it (the face
    component matching its skin to a person's photos), whose skin and other texels the outfit leaves alone stay as
    they are; ``base`` gives another one (see :meth:`Tailor.texture`). The texture keeps it (``texture.base``), so
    dressing again starts from it too and :func:`undress` gives it back."""
    tailor = tailor_for(character.skinned_model)
    outfit = _outfit(outfit)
    name = tailor.primitive.name
    current = character.texture_overrides.get(name)
    if base is None:
        base = getattr(current, "base", None) if _dressed(current) else current
    rgba = tailor.texture(outfit, base=base)
    if _dressed(current):
        current.update(rgba, order="rgb")
        texture = current
    else:
        texture = Texture(rgba, name=f"{character.name} outfit")
        character.texture_overrides[name] = texture
    texture.outfit = outfit
    texture.base = base
    return texture


def undress(character) -> None:
    """Take the outfit off ``character``: the body texture it was put on comes back (the avatar's own, or the one it
    wore before :func:`dress`)."""
    model = character.skinned_model
    index = R.body_primitive(model)
    if index is None:
        return
    name = model.primitives[index].name
    current = character.texture_overrides.get(name)
    if _dressed(current):
        base = getattr(current, "base", None)
        if base is None:
            del character.texture_overrides[name]
        else:
            character.texture_overrides[name] = base


def outfit_of(character) -> Outfit | None:
    """The outfit ``character`` wears (put on with :func:`dress`), or ``None``."""
    model = character.skinned_model
    index = R.body_primitive(model)
    if index is None:
        return None
    return getattr(character.texture_overrides.get(model.primitives[index].name), "outfit", None)


__all__ = ["Outfit", "ShadingLayers", "Tailor", "dress", "outfit_of", "preset", "presets", "shading_layers",
           "tailor_for", "undress"]
