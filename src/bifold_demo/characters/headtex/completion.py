"""Stage 5: what no photo shows, completed: the other half of the face mirrored in, and an optional generative layer.

After stages 3 and 4 every texel no photo shows well (the trimap's ``generate`` and the weak end of ``refine``) holds
the avatar's own texture, tone-matched to the person's skin. :func:`complete` improves on that in two steps:

1. **The mirror** (deterministic). A face is nearly symmetric, and so are the Rocketbox heads' meshes (every texel's
   point, mirrored across the face's plane, lands within a millimetre of another texel's on the median; within 3 mm
   for 90 to 99 %). A texel the photos show poorly whose mirror image they show well takes the mirror's colour: a
   photo of one side of the face completes the other. It is copied in "albedo" (the avatar's painted soft light
   divided out and put back for the texel's own normal), with a weight that rises as the texel's own confidence falls
   and as its mirror's rises, and blended into the rest by the same multi-band blend, island by island, as stage 3's
   (:func:`.fuse.blend`), so no seam shows where the mirrored part meets the photos or the avatar's texture. Where
   neither side was seen (the back of the head, under the chin) the tone-matched avatar texture stays. The mirror is
   found once per head atlas (:func:`mirror_map`: a grid of 1 mm cells, then the nearest texel round the found one).
2. **A generative layer** (optional, :class:`Inpainter`). A model such as FLUX.2 [klein] (:class:`FluxInpainter`,
   through ``diffusers``; any pipeline with its call signature can be passed in) gets the completed texture with the
   texels still to generate painted grey, and a prompt naming the person's skin and hair colours. One call. Its
   result counts only on those texels, second-class: tone-matched to the deterministic result (its own detail kept,
   not its tone), feathered at the edges and mixed in at :data:`GENERATED_WEIGHT` at most; photo texels are never
   overwritten. A model that cannot load or fails leaves the deterministic result, with a warning (stage 7 can also
   drop the layer when it lowers the identity check).

The result is a :class:`Completion`: the texture, the confidence with the mirrored texels in it (second-hand: they
count :data:`MIRROR_CONFIDENCE` of their mirror's), the trimap made from it, how much each texel took from the mirror
and from the model, and warnings.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import fuse, light

MIRROR_CELL = 0.001  # metres: the grid the mirror images are looked up in
MIRROR_REACH = 0.003  # metres: a mirror image further than this from every texel of the other side has no mirror
MIRROR_NORMAL = 0.5  # cosine: the mirror texel's normal must face the mirrored normal's way (not the back of an ear)
MIRROR_STEPS = 6  # steps of a texel from the grid's find towards the nearest texel, at most
MIRROR_CONFIDENCE = 0.6  # a mirrored texel's confidence: this share of its mirror's
BETTER_BY = 0.2  # confidence: a mirror counts fully when it is this much surer than the texel itself
GENERATED_WEIGHT = 0.4  # the most a generated texel counts against the deterministic result
GENERATED_FEATHER = 4.0  # texels (at 1024): the generated layer's edge
GREY = (128, 128, 128)  # what the texels to generate are painted for the model
PROMPT = ("Fill the flat grey areas of this human head texture atlas (a UV layout: the face in the middle, the scalp "
          "and the back of the head round it) with matching {skin} skin and {hair} hair, seamless with the rest, the "
          "same soft even light, no shadows; keep every other part exactly as it is.")


# -- the mirror -------------------------------------------------------------------------------------------------


def symmetry_plane(model) -> tuple[np.ndarray, np.ndarray]:
    """``(point, normal)``: the face's plane of symmetry in the bind pose (through the middle of the eyes and the
    mouth corners, square to the line between the two sides)."""
    from .landmarks import face_frame

    frame = face_frame(model)
    across = (frame.eyes[1] - frame.eyes[0]) + (frame.mouth[1] - frame.mouth[0])
    return frame.centre.astype(np.float64), across / np.linalg.norm(across)


_mirrors: dict[int, tuple] = {}


def mirror_map(atlas: fuse.HeadAtlas, model) -> np.ndarray:
    """int32 ``(size, size)``: for each texel of the face, scalp and neck the flat index of the texel at its mirror
    image across the face's plane (:func:`symmetry_plane`), or -1 (no texel of the other side within
    :data:`MIRROR_REACH`, facing the same way, of the same kind: hair for hair, skin for skin). Kept in memory for
    the atlas."""
    hit = _mirrors.get(id(atlas))
    if hit is not None and hit[0] is atlas:
        return hit[1]
    point, normal = symmetry_plane(model)
    size = atlas.size
    photo = atlas.photo_parts
    flat = np.flatnonzero(photo)
    positions = atlas.bake.position.reshape(-1, 3)  # float32, NaN outside the islands
    normals = atlas.bake.normal.reshape(-1, 3)
    p = positions[flat].astype(np.float64)
    mirrored = p - 2.0 * ((p - point) @ normal)[:, None] * normal
    n = normals[flat].astype(np.float64)
    turned = n - 2.0 * (n @ normal)[:, None] * normal
    found = _grid_lookup(p, mirrored, flat)  # the texel in the mirror image's cell (or a neighbouring one), or -1
    scalp = (atlas.parts == fuse.SCALP).ravel()
    best = np.full(len(flat), -1, np.int64)
    for start in range(0, len(flat), 65536):
        part = slice(start, start + 65536)
        best[part] = _nearest_round(found[part], mirrored[part], positions, photo.ravel(), size)
    ok = best >= 0
    distance = np.full(len(flat), np.inf)
    distance[ok] = np.linalg.norm(positions[best[ok]] - mirrored[ok], axis=1)
    agree = np.zeros(len(flat), bool)
    agree[ok] = np.einsum("nk,nk->n", normals[best[ok]].astype(np.float64), turned[ok]) > MIRROR_NORMAL
    same = np.zeros(len(flat), bool)
    same[ok] = scalp[best[ok]] == scalp[flat[ok]]
    out = np.full(size * size, -1, np.int32)
    keep = ok & (distance < MIRROR_REACH) & agree & same
    out[flat[keep]] = best[keep]
    out = out.reshape(size, size)
    _mirrors[id(atlas)] = (atlas, out)
    while len(_mirrors) > fuse.ATLASES:
        _mirrors.pop(next(iter(_mirrors)))
    return out


def _cells(points: np.ndarray) -> np.ndarray:
    """int64 keys of the :data:`MIRROR_CELL` grid cells of ``points`` (metres, within 4 m of the origin)."""
    c = np.floor(points / MIRROR_CELL).astype(np.int64) + 4096
    return (c[:, 0] * 8192 + c[:, 1]) * 8192 + c[:, 2]


def _grid_lookup(points: np.ndarray, queries: np.ndarray, flat: np.ndarray) -> np.ndarray:
    """For each query point, the flat texel index of a point of ``points`` (texels ``flat``) in its grid cell, else in
    one of the 26 cells round it (nearer ones first), else -1."""
    keys = _cells(points)
    cells, first = np.unique(keys, return_index=True)
    out = np.full(len(queries), -1, np.int64)
    offsets = sorted(((dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)),
                     key=lambda o: abs(o[0]) + abs(o[1]) + abs(o[2]))
    base = np.floor(queries / MIRROR_CELL).astype(np.int64) + 4096
    for dx, dy, dz in offsets:
        todo = np.flatnonzero(out < 0)
        if not len(todo):
            break
        c = base[todo] + (dx, dy, dz)
        key = (c[:, 0] * 8192 + c[:, 1]) * 8192 + c[:, 2]
        at = np.clip(np.searchsorted(cells, key), 0, len(cells) - 1)
        there = cells[at] == key
        out[todo[there]] = flat[first[at[there]]]
    return out


def _nearest_round(start: np.ndarray, targets: np.ndarray, positions: np.ndarray, allowed: np.ndarray,
                   size: int) -> np.ndarray:
    """From the texels ``start`` (flat indices, -1: none), walking downhill in the layout: the texel nearest each
    target point among the ``allowed`` ones within one texel, :data:`MIRROR_STEPS` times (a step of a texel or
    less; the grid's find is within a cell, a few texels)."""
    best = start.copy()
    ok = np.flatnonzero(best >= 0)
    dy, dx = (a.ravel() for a in np.meshgrid((-1, 0, 1), (-1, 0, 1), indexing="ij"))
    targets = targets.astype(np.float32)
    for _ in range(MIRROR_STEPS):
        if not len(ok):
            break
        rows, cols = np.divmod(best[ok], size)
        candidates = np.clip(rows[:, None] + dy, 0, size - 1) * size + np.clip(cols[:, None] + dx, 0, size - 1)
        diff = positions[candidates] - targets[ok][:, None, :]
        d = np.einsum("nkd,nkd->nk", diff, diff)
        d[~allowed[candidates]] = np.inf
        pick = candidates[np.arange(len(ok)), np.nan_to_num(d, nan=np.inf).argmin(axis=1)]
        moved = pick != best[ok]
        best[ok] = pick
        ok = ok[moved]  # those that stopped have found it
    return best


# -- the completion ---------------------------------------------------------------------------------------------


@dataclass
class Completion:
    """A head texture completed by stage 5 (see the module docs)."""

    rgba: np.ndarray  # uint8 (size, size, 4), like HeadTexture.rgba
    confidence: np.ndarray  # float32 0-1: the photos', the mirrored texels' (second-hand) in it
    trimap: np.ndarray  # uint8: fuse.GENERATE, REFINE, KEEP from that confidence (TRIMAP_OUTSIDE outside)
    mirrored: np.ndarray  # float32 0-1: how much of each texel came from its mirror image
    generated: np.ndarray  # float32 0-1: how much from the generative layer (0 without one)
    method: str = "mirror"  # "mirror", or "mirror + <the model's name>"
    warnings: list = field(default_factory=list)
    timings: dict = field(default_factory=dict)
    deterministic: np.ndarray | None = None  # with a generative layer: the texture without it (stage 7 may drop it)

    def without_generated(self) -> Completion:
        """This completion without its generative layer (itself when it has none)."""
        if self.deterministic is None:
            return self
        return Completion(self.deterministic, self.confidence, self.trimap, self.mirrored,
                          np.zeros_like(self.generated), "mirror", list(self.warnings), dict(self.timings))

    def share(self) -> dict:
        """The share of the texels the photos reach (face, scalp, neck) that the mirror and the model filled (counting
        each texel by how much it took)."""
        return {"mirrored": float(self.mirrored[self.trimap != fuse.TRIMAP_OUTSIDE].mean()),
                "generated": float(self.generated[self.trimap != fuse.TRIMAP_OUTSIDE].mean())}


def _smooth(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def complete(result, model, *, inpainter=None, references=None, prompt: str | None = None) -> Completion:
    """Stage 5 on ``result`` (a :class:`~.texture.HeadTexture` of ``model``'s head): the mirror, then, with an
    ``inpainter`` (an :class:`Inpainter`), the generative layer (see the module docs). ``references``: photos (BGR)
    the model may look at (default: the first two registered photos); ``prompt``: instead of :data:`PROMPT`
    (``{skin}`` and ``{hair}`` are filled in)."""
    atlas = result.atlas
    timings = {}
    start = time.perf_counter()
    mirror = mirror_map(atlas, model)
    timings["mirror map (once per avatar)"] = time.perf_counter() - start
    start = time.perf_counter()
    conf = result.confidence.astype(np.float32)
    shading = atlas.shading[..., None]
    albedo = light.to_linear(result.rgba[..., :3]) / shading
    has = mirror >= 0
    index = np.where(has, mirror, 0).ravel()
    conf_m = np.where(has, conf.ravel()[index].reshape(conf.shape), 0.0)
    need = _smooth((fuse.KEEP_AT - conf) / (fuse.KEEP_AT - fuse.REFINE_AT))  # 1 where no photo shows it
    sure = _smooth((conf_m - fuse.REFINE_AT) / (fuse.KEEP_AT - fuse.REFINE_AT))  # 1 where the mirror is well seen
    better = _smooth((conf_m - conf) / BETTER_BY)
    take = (need * sure * better * has * atlas.photo_parts).astype(np.float32)
    out = albedo
    if take.max() > 0.01:
        mirrored = albedo.reshape(-1, 3)[index].reshape(albedo.shape)
        out, _share = fuse.blend([albedo, mirrored], [1.0 - take, take], atlas.bake.island, fuse.bands_for(atlas.size))
        # the blend's coarse bands reach a little round what was mirrored: the texels that took nothing (the photos'
        # sure ones) stay exactly as they were, the change fading in over the first tenth of a mirror's weight
        reach = _smooth(take / 0.1)[..., None]
        out = albedo + (out - albedo) * reach
    confidence = np.maximum(conf, take * MIRROR_CONFIDENCE * conf_m).astype(np.float32)
    trimap = fuse.trimap(confidence, atlas)
    timings["mirror"] = time.perf_counter() - start
    linear = np.clip(out, 0.0, None) * shading
    generated = np.zeros(conf.shape, np.float32)
    warnings, method, deterministic = [], "mirror", None
    if inpainter is not None:
        start = time.perf_counter()
        to_fill = (trimap == fuse.GENERATE) & atlas.photo_parts
        try:
            if to_fill.sum() < 50:
                raise _Nothing
            mirrored_only = linear
            linear, generated = _generative(linear, to_fill, atlas, result, inpainter, references, prompt)
            method = f"mirror + {getattr(inpainter, 'name', type(inpainter).__name__)}"
            deterministic = _finished(mirrored_only, result, atlas)
        except _Nothing:
            pass
        except Exception as exc:  # noqa: BLE001 - a model that cannot load or fails: the deterministic result stays
            warnings.append(f"the generative completion failed ({type(exc).__name__}: {exc}): the mirror and the "
                            "avatar's own texture, tone-matched, are left")
        timings["generative layer"] = time.perf_counter() - start
    return Completion(_finished(linear, result, atlas), confidence, trimap, take, generated, method, warnings, timings,
                      deterministic)


def _finished(linear: np.ndarray, result, atlas) -> np.ndarray:
    """The texture (RGBA) of ``linear`` colours: sRGB, the eyes exactly as they were, the gutters padded."""
    rgba = np.dstack([light.to_srgb(linear), result.rgba[..., 3]])
    eyes = atlas.parts == fuse.EYE
    rgba[eyes] = result.rgba[eyes]
    return fuse.pad(rgba, atlas.covered)


class _Nothing(Exception):
    """Nothing left to generate."""


def _generative(linear, to_fill, atlas, result, inpainter, references, prompt):
    """The generative layer on ``linear`` (the completed texture, linear RGB): ``(linear, weight)``."""
    size = atlas.size
    image = light.to_srgb(linear)
    image[to_fill] = GREY
    mask = (to_fill * 255).astype(np.uint8)
    words = {"skin": skin_words(result.skin), "hair": hair_words(result.hair)}
    text = (prompt or PROMPT).format(**words)
    if references is None:
        references = [v.image for v in result.views if v.usable and getattr(v, "image", None) is not None]
    refs = [cv2.cvtColor(_fit(r, 768), cv2.COLOR_BGR2RGB) for r in references][:2]
    made = np.asarray(inpainter(image, mask, text, refs))
    if made.shape[:2] != image.shape[:2]:
        made = cv2.resize(made, (size, size), interpolation=cv2.INTER_CUBIC)
    made = light.to_linear(made[..., :3].astype(np.uint8))
    # its own detail, the deterministic result's tone: a gain per channel over the texels it fills
    inside = to_fill & (made.sum(axis=2) > 1e-4)
    if inside.sum() < 20:
        raise ValueError("the model gave back nothing where it was to fill")
    gain = np.exp(np.mean(np.log(np.maximum(linear[inside], 1e-4)), axis=0)
                  - np.mean(np.log(np.maximum(made[inside], 1e-4)), axis=0))  # (geometric means)
    made *= np.clip(gain, 0.25, 4.0)
    # second-class: only the texels to generate, feathered inside them, at GENERATED_WEIGHT at most
    sigma = GENERATED_FEATHER * size / 1024
    inner = cv2.erode(to_fill.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=max(1, round(sigma)))
    weight = cv2.GaussianBlur(inner.astype(np.float32), (0, 0), sigma) * to_fill * GENERATED_WEIGHT
    weight = weight.astype(np.float32)
    return linear * (1.0 - weight[..., None]) + made * weight[..., None], weight


def skin_words(tone) -> str:
    """The skin tone (a :class:`~.light.Tone`) in words, for a prompt."""
    if tone is None:
        return "natural"
    lightness = float(tone.lab[0])
    for limit, words in ((75, "very fair"), (65, "fair"), (55, "light-medium"), (45, "medium"), (35, "brown")):
        if lightness > limit:
            return words
    return "deep brown"


def hair_words(tone) -> str:
    """The hair colour (a :class:`~.light.Tone`) in words, for a prompt."""
    if tone is None:
        return "matching"
    lightness, a, b = (float(v) for v in tone.lab)
    chroma = float(np.hypot(a, b))
    if lightness < 18:
        return "black"
    if chroma < 8 and lightness > 55:
        return "grey"
    if a > 12 and chroma > 22:
        return "auburn" if lightness < 45 else "red"
    for limit, words in ((30, "dark brown"), (45, "brown"), (60, "light brown")):
        if lightness < limit:
            return words
    return "blond"


def _fit(image: np.ndarray, longest: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(1.0, longest / max(h, w))
    return cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1 \
        else image


# -- the generative layer ---------------------------------------------------------------------------------------


class Inpainter:
    """What stage 5's generative layer needs: ``inpainter(image, mask, prompt, references) -> image`` (RGB ``uint8``
    images of the same size; ``mask`` 255 where it is to fill, ``references`` RGB images it may look at). ``name``
    says which model it is (in :attr:`Completion.method`)."""

    name = "inpainter"

    def __call__(self, image: np.ndarray, mask: np.ndarray, prompt: str, references=()) -> np.ndarray:
        raise NotImplementedError


class FluxInpainter(Inpainter):
    """FLUX.2 [klein] 4B (image editing with references, through ``diffusers``; see
    :data:`bifold_demo.people.generator.PRESETS`) as stage 5's :class:`Inpainter`: one call at ``size`` x ``size``
    with the texture (its texels to fill painted grey) as the first reference image. ``pipeline``: a loaded pipeline,
    or any callable with its call signature (tests pass a stand-in); else the preset's model is loaded on first use,
    shared with the rest of the demo (:func:`~bifold_demo.people.generator.shared_generator`; from Hugging Face:
    about 16 GB of memory)."""

    def __init__(self, preset: str = "flux2-klein", pipeline=None, device: str = "auto", size: int = 768,
                 steps: int | None = None, seed: int = 0):
        from ...people.generator import get_preset

        self.preset = get_preset(preset) if isinstance(preset, str) else preset
        self.name = self.preset.name
        self.device, self.size, self.seed = device, int(size) // 16 * 16, int(seed)
        self.steps = steps or self.preset.steps
        self._pipe = pipeline

    @property
    def pipe(self):
        if self._pipe is None:
            from ...people.generator import shared_generator

            self._pipe = shared_generator(self.preset, device=self.device).pipe
        return self._pipe

    def __call__(self, image: np.ndarray, mask: np.ndarray, prompt: str, references=()) -> np.ndarray:
        from PIL import Image

        from ...people.generator import pipeline_lock

        h, w = image.shape[:2]
        small = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_AREA)
        images = [Image.fromarray(np.ascontiguousarray(small))] + [Image.fromarray(np.ascontiguousarray(r))
                                                                   for r in references]
        pipe = self.pipe
        with pipeline_lock(pipe):  # (one call at a time: the fabrics may use the same pipeline)
            out = pipe(image=images, prompt=prompt, height=self.size, width=self.size,
                       num_inference_steps=self.steps, guidance_scale=self.preset.guidance,
                       generator=_generator(self.seed))
        made = np.asarray(out.images[0].convert("RGB"))
        return cv2.resize(made, (w, h), interpolation=cv2.INTER_CUBIC)

    def release(self) -> None:
        """Let go of the model (its memory is given back once nothing else holds it)."""
        self._pipe = None


def _generator(seed: int):
    try:
        import torch
    except ImportError:  # a stand-in pipeline without torch
        return None
    return torch.Generator("cpu").manual_seed(seed)


__all__ = ["Completion", "FluxInpainter", "Inpainter", "complete", "mirror_map", "symmetry_plane"]
