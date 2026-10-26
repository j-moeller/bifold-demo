"""Stage 1 of the outfits: new fabrics on an avatar's clothes that keep the look of its own texture.

The texture of a Rocketbox garment has its folds, seams, stitching, buttons and ambient occlusion painted in. A
:class:`Tailor` splits it, region by region, into a base colour and two luminance layers (:class:`ShadingLayers`)::

    L = luminance of the texture,  B = L blurred within the region (about 12 texels at 1024)
    D = L / B                (detail: weave, stitching, seams, small wrinkles; clipped to 0.35-1.8)
    S = B / ref              (shading: folds and ambient occlusion; ref = the region's 75th percentile of B)

and makes a new texture as ``fill x D x S``: a navy fill gets the jacket's folds and stitching, a tartan bends round
them. The blur stays inside each region (normalised convolution), so a white shirt does not light up the edge of a
black jacket. Where two garments meet, their feathered masks (:meth:`.regions.BodyRegions.mask`) cross-fade over 2-3
texels; skin texels are never written (their mask is 0), and the texture is extended a few texels past the edges of
the islands so mipmapping does not bleed the old colours in. ``fill = texture / (D x S)`` (the region's own albedo,
:attr:`ShadingLayers.albedo`) gives the texture back.

An :class:`Outfit` is a recipe: region -> fill (see :mod:`.fills`), with a name; :func:`presets` are the ones in
``presets.json``. :func:`dress` puts one on a :class:`~..character.Character` as its own body texture
(``character.texture_overrides``; the avatar's other users keep theirs) and :func:`undress` takes it off. The
Tailor of an avatar keeps the labels, cloth coordinates and layers (:func:`tailor_for`), so dressing again costs only
the fills: 50-150 ms for a 1024 texture, 0.2-0.5 s for a 2048 one (on a CPU; the first outfit also sorts out each
garment's texels, about 50 ms a garment).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
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
    reference: dict  # region -> the luminance a fill's colour stands for
    albedo: np.ndarray  # float32 (size, size, 3): the texture divided by D x S (its own base colour)


def shading_layers(rgb: np.ndarray, labels: np.ndarray) -> ShadingLayers:
    """:class:`ShadingLayers` of a texture (RGB, ``(size, size, 3)``) with region ``labels``."""
    size = labels.shape[0]
    sigma = DETAIL_SIGMA * size / 1024.0
    lum = luminance(rgb)
    detail = np.ones((size, size), np.float32)
    own = np.ones((size, size), np.float32)  # each texel's own region's S
    shading, reference = {}, {}
    for k, name in enumerate(R.REGIONS):
        if name == "skin":
            continue
        hard = labels == k
        if not hard.any():
            continue
        blurred = R.masked_blur(lum, hard, sigma)
        ref = float(np.percentile(blurred[hard], SHADING_REFERENCE))
        reference[name] = ref
        s = ((blurred + EPSILON) / (ref + EPSILON)).astype(np.float32)
        shading[name] = s
        detail[hard] = np.clip((lum[hard] + EPSILON) / (blurred[hard] + EPSILON), *DETAIL_RANGE)
        own[hard] = s[hard]
    albedo = np.asarray(rgb, np.float32) / (detail * own)[..., None]
    return ShadingLayers(detail, shading, reference, albedo)


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
        self._flat = self.original[..., :3].reshape(-1, 3).astype(np.float32)
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
            self._layers = shading_layers(self.original[..., :3], self.regions.labels)
        return self._layers

    def where(self, region: str) -> _Texels:
        """The texels a fill of ``region`` writes (its feathered mask above 0), sorted by their size on the body."""
        if region not in self._texels:
            mask = self.regions.mask(region)
            rows, cols = np.nonzero(mask > 0)
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
            index = rows.astype(np.int64) * self.size + cols
            self._texels[region] = _Texels(where, index, mask[rows, cols].astype(np.float32),
                                           factor.astype(np.float32), self._flat[index])
        return self._texels[region]

    def texture(self, outfit) -> np.ndarray:
        """The body texture (RGBA ``uint8``, the original's size) with ``outfit`` (an :class:`Outfit`, a dict of
        fills or a description) on it."""
        outfit = _outfit(outfit)
        out = self._flat.copy()
        changed = np.zeros(self.size * self.size, bool)
        for region, fill in outfit.fills.items():
            if fill.keeps() or region not in self.layers.shading:
                continue
            texels = self.where(region)
            if len(texels.index) == 0:
                continue
            new = np.asarray(fill.colors(texels.where), np.float32) * texels.factor[:, None]
            np.clip(new, 0.0, 255.0, out=new)
            new -= texels.base
            new *= texels.weight[:, None]
            out[texels.index] += new  # the weights of the garments add up to at most 1: it stays in 0-255
            changed[texels.index] = True
        self._extend(out, changed)
        rgba = self.original.copy()
        rgba[..., :3] = (out + 0.5).astype(np.uint8).reshape(self.size, self.size, 3)
        return rgba

    def _extend(self, out: np.ndarray, changed: np.ndarray) -> None:
        """Copy the new colours a few texels past the islands' edges (in place), where they were changed."""
        if self._bleed is None:
            covered = self.regions.bake.covered
            distance, nearest = cv2.distanceTransformWithLabels((~covered).astype(np.uint8), cv2.DIST_L2, 3,
                                                                labelType=cv2.DIST_LABEL_PIXEL)
            ys, xs = np.nonzero(covered)
            index = np.zeros(int(nearest.max()) + 1, np.int64)
            index[nearest[ys, xs]] = ys * self.size + xs  # label -> the covered texel it names
            near = (~covered) & (distance <= DILATE * self.size / 1024.0)
            rows, cols = np.nonzero(near)
            self._bleed = (rows.astype(np.int64) * self.size + cols, index[nearest[rows, cols]])
        target, source = self._bleed
        take = changed[source]
        out[target[take]] = out[source[take]]


@dataclass
class _Texels:
    """What a fill of one region needs, worked out once: where the texels are, and how the original looks there."""

    where: Where
    index: np.ndarray  # flat texel indices
    weight: np.ndarray  # the region's feathered mask
    factor: np.ndarray  # detail x shading
    base: np.ndarray  # (N, 3) the original colours


def _outfit(outfit) -> Outfit:
    if isinstance(outfit, Outfit):
        return outfit
    if isinstance(outfit, dict):
        return Outfit(outfit)
    if isinstance(outfit, str):
        return Outfit.parse(outfit)
    raise TypeError(f"cannot dress in {outfit!r}")


_tailors: dict[int, tuple] = {}


def tailor_for(model) -> Tailor:
    """The :class:`Tailor` of ``model`` (made once and kept while the model lives)."""
    hit = _tailors.get(id(model))
    if hit is not None and hit[0] is model:
        return hit[1]
    tailor = Tailor(model)
    _tailors[id(model)] = (model, tailor)
    return tailor


def dress(character, outfit) -> Texture:
    """Put ``outfit`` (an :class:`Outfit`, ``{region: fill}``, a preset's name or a description such as
    ``"red tartan jacket, black trousers"``) on ``character``: its body texture becomes its own
    (``character.texture_overrides``). Dressing again reuses that texture (the renderer uploads the new pixels)."""
    tailor = tailor_for(character.skinned_model)
    outfit = _outfit(outfit)
    rgba = tailor.texture(outfit)
    name = tailor.primitive.name
    current = character.texture_overrides.get(name)
    if isinstance(current, Texture) and getattr(current, "outfit", None) is not None:
        current.update(rgba, order="rgb")
        texture = current
    else:
        texture = Texture(rgba, name=f"{character.name} outfit")
        character.texture_overrides[name] = texture
    texture.outfit = outfit
    return texture


def undress(character) -> None:
    """Give ``character`` the avatar's own clothes back."""
    model = character.skinned_model
    index = R.body_primitive(model)
    if index is None:
        return
    name = model.primitives[index].name
    current = character.texture_overrides.get(name)
    if current is not None and getattr(current, "outfit", None) is not None:
        del character.texture_overrides[name]


def outfit_of(character) -> Outfit | None:
    """The outfit ``character`` wears (put on with :func:`dress`), or ``None``."""
    model = character.skinned_model
    index = R.body_primitive(model)
    if index is None:
        return None
    return getattr(character.texture_overrides.get(model.primitives[index].name), "outfit", None)


__all__ = ["Outfit", "ShadingLayers", "Tailor", "dress", "outfit_of", "preset", "presets", "shading_layers",
           "tailor_for", "undress"]
