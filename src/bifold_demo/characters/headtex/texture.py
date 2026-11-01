"""Stages 0 to 4 end to end: photos of a person's head in, a texture for an avatar's own head mesh out.

::

    from bifold_demo.characters.headtex import head_texture, wear

    result = head_texture(["front.jpg", "left.jpg", "right.jpg"], model, views=[None, "left", "right"])
    print(result.summary())        # the photos' poses and shares, the skin tone, warnings, timings
    wear(person, result)           # a scene Character of that model gets the head texture

:func:`head_texture` finds the landmarks on the avatar's head (stage 0, cached), registers the photos (stage 1),
projects them into the head texture's layout and fuses them (stage 3, :mod:`.fuse`), taking each photo's light out
and putting the avatar's own soft light back (stage 4, :mod:`.light`). :func:`fuse_views` does stages 3 and 4 on
photos already registered. The result, a :class:`HeadTexture`, holds the texture (RGBA), the confidence of every texel
and the keep / refine / generate trimap for stage 5, the person's skin tone for stage 6, each photo's share, the
warnings and the timings.

The order of the work, per photo: project (the depth test, the photo read through the face warp, the weights), fit
and divide out its light; then the gains between the photos (anchored on the most frontal one) and the weights scaled
by how well lit each texel was; then the skin tone, the avatar's own texture tone-matched to it, the multi-band blend
in "albedo" (light divided out), the eyes and the inside of the mouth back as they were, the avatar's soft light
multiplied back in, sRGB, padding.

To check a result: :func:`preview` renders the avatar wearing it from the front, three-quarter and the side;
:func:`debug_images` shows the parts, the confidence, the trimap and each photo's colours and weights in the layout;
:func:`neck_seam` and :func:`skin_gap` measure how far the head's skin is from the body's (what stage 6 closes).

The work takes about five seconds for four 1280x960 photos on four cores (and about ten more once per avatar, for
its landmarks, regions and atlas, which are then cached on disk). Most of it is numpy and OpenCV, which let go of
Python's interpreter lock while they compute: :func:`head_texture_async` runs it on a worker thread and returns a
:class:`~concurrent.futures.Future`. Beside a render loop paced to 30 frames a second the frames stay as they were
(16 ms against 14) and the texture takes twice as long; a render loop that never waits (one slower than 30 frames a
second, on the software renderer) keeps the lock while the renderer draws, and the worker then takes ten times as
long.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import fuse, light
from .fuse import EYE, FACE, INNER, NECK, SCALP, HeadAtlas, Projection

SKIN_LIKE = (22.0, 40.0)  # Lab distance from the avatar's skin (lightness counting half): fully, then not at all skin
SCALP_SKIN_LIKE = (8.0, 16.0)  # ... on the scalp, where a texel must look just like the skin to be toned with it
FIT_FACE = 0.6  # the segmenter's face-skin confidence a texel needs to be fitted for the light, and for the skin tone
HARSH = 8.0  # a photo whose fitted light varies more than this over the face (linear: 3 stops) gets a warning
POOR_FIT = 0.7  # ... one whose robust light fit keeps less than this share of the face skin
TONE_LIMITS = (0.2, 5.0)  # the tone change of the avatar's own texture, per channel


@dataclass
class HeadTexture:
    """A head texture made from photos (see the module docs). ``rgba`` is an image (row 0 at the top) in RGB
    order, like the avatars' own textures."""

    rgba: np.ndarray  # uint8 (size, size, 4)
    confidence: np.ndarray  # float32 (size, size) 0-1
    trimap: np.ndarray  # uint8 (size, size): fuse.GENERATE, REFINE, KEEP; fuse.TRIMAP_OUTSIDE outside the islands
    skin: light.Tone | None  # the person's skin as the texture shows it (cheeks and forehead), for stage 6
    own_skin: light.Tone | None  # the avatar's own skin on the same texels
    hair: light.Tone | None  # the person's hair where the photos show it on the scalp
    shares: dict  # photo name -> its share of the texels the photos can reach; "avatar": what the avatar's own keeps
    views: list  # the registered views (stage 1), usable or not
    warnings: list = field(default_factory=list)
    timings: dict = field(default_factory=dict)  # stage -> seconds
    atlas: HeadAtlas | None = None
    projections: list = field(default_factory=list)  # each usable photo in the layout (with debug: their factors)
    model: str = ""
    primitive: str = ""  # the head primitive's name (a Character's texture_overrides key)

    @property
    def size(self) -> int:
        return self.rgba.shape[0]

    def texture(self, name: str | None = None):
        """The texture as a scene :class:`~bifold_demo.scene.textures.Texture`."""
        from ...scene.textures import Texture

        return Texture(self.rgba, name=name or f"{self.model} head from photos")

    def save(self, path) -> Path:
        """Write the texture as a PNG (or any image file OpenCV writes)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), cv2.cvtColor(self.rgba, cv2.COLOR_RGBA2BGRA)):
            raise OSError(f"cannot write {path}")
        return path

    def coverage(self) -> dict:
        """The share of each part's texels (face, scalp, neck) the photos show surely (trimap ``keep``) and at all
        (``refine`` or better)."""
        out = {}
        for k in (FACE, SCALP, NECK):
            mine = self.atlas.parts == k
            if mine.any():
                out[fuse.PARTS[k]] = (float((self.trimap[mine] == fuse.KEEP).mean()),
                                      float((self.trimap[mine] >= fuse.REFINE).mean()))
        return out

    def summary(self) -> str:
        lines = [f"{self.model}: a {self.size}x{self.size} head texture from {len(self.projections)} of "
                 + f"{len(self.views)} photos"]
        for view in self.views:
            lines.append(f"  {view.summary()}")
        lines.append("  shares: " + ", ".join(f"{name} {100 * share:.0f} %" for name, share in self.shares.items()))
        for name, (sure, seen) in self.coverage().items():
            lines.append(f"  {name}: {100 * sure:.0f} % shown surely, {100 * seen:.0f} % at all")
        if self.skin is not None:
            own = f" (the avatar's: {self.own_skin})" if self.own_skin is not None else ""
            lines.append(f"  skin {self.skin}{own}")
        if self.hair is not None:
            lines.append(f"  hair {self.hair}")
        for p in self.projections:
            if p.shading is not None:
                lines.append(f"  {p.name}: light {describe_light(p.shading)}, fit within "
                             f"{100 * p.shading.residual:.0f} %, gains " + " ".join(f"{g:.2f}" for g in p.gain))
        lines += [f"  warning: {w}" for w in self.warnings]
        lines.append("  timings: " + ", ".join(f"{k} {v:.2f} s" for k, v in self.timings.items()))
        return "\n".join(lines)


def head_texture(photos, model, *, views=None, focals=None, landmarker=None, segmenter=None, head=None,
                 size: int | None = None, cache_dir=None, debug: bool = False) -> HeadTexture:
    """Photos of a person's head (paths or BGR images) turned into a texture for ``model``'s head (see the module
    docs). ``views``: what each photo shows (``"front"``, ``"left"``, ``"right"``, ``"back"`` or ``None``: found from
    its face); ``focals``: focal lengths in pixels, when known; ``landmarker`` and ``segmenter``: MediaPipe's (made
    when not given; the segmenter is what keeps hair, clothes and background out, so it is made too); ``head``: the
    avatar's stage-0 landmarks (found and cached when not given); ``size``: the texture's (default its own).
    ``debug``: keep each weight's factors on the projections."""
    from .landmarks import head_landmarks
    from .register import head_regions, register_photos

    timings = {}
    own_tasks = []
    try:
        if landmarker is None:
            from .detect import Landmarker

            landmarker = Landmarker()
            own_tasks.append(landmarker)
        if segmenter is None:
            from .detect import Segmenter

            segmenter = Segmenter()
            own_tasks.append(segmenter)
        start = time.perf_counter()
        if head is None:
            head = head_landmarks(model, landmarker, cache_dir=cache_dir)
        labels = head_regions(model, head.primitive, segmenter, cache_dir=cache_dir)
        timings["landmarks and regions (once per avatar)"] = time.perf_counter() - start
        start = time.perf_counter()
        registered = register_photos(list(photos), head, model, views=views, focals=focals, landmarker=landmarker,
                                     segmenter=segmenter)
        timings["registration"] = time.perf_counter() - start
    finally:
        for task in own_tasks:
            task.close()
    result = fuse_views(registered, model, head, size=size, labels=labels, cache_dir=cache_dir, debug=debug)
    result.timings = {**timings, **result.timings}
    return result


def fuse_views(views, model, head, *, size: int | None = None, labels="auto", cache_dir=None,
               debug: bool = False, delight: bool = True) -> HeadTexture:
    """Stages 3 and 4 on registered ``views`` (:func:`~.register.register_photos`) of ``model``'s head (``head``: its
    stage-0 landmarks): a :class:`HeadTexture`. ``labels``: the segmenter's labels of the head (see
    :func:`.fuse.head_atlas`); ``delight=False`` leaves the photos' light in (for comparison)."""
    timings = {}
    start = time.perf_counter()
    atlas = fuse.head_atlas(model, head, size, labels=labels, cache_dir=cache_dir)
    timings["atlas (once per avatar)"] = time.perf_counter() - start
    warnings = [f"{v.name}: not used: {v.reason}" for v in views if not v.usable]
    usable = [v for v in views if v.usable and v.camera is not None]
    if not usable:
        raise ValueError("no usable photo: " + "; ".join(warnings) if warnings else "no photos")
    start = time.perf_counter()
    projections = [fuse.project(v, atlas, debug=debug) for v in usable]
    _unique_names(projections)
    timings["projection"] = time.perf_counter() - start
    start = time.perf_counter()
    for p in projections:
        warnings += p.warnings
        if delight:
            warnings += _delight(p, atlas)
    gains = fuse.gains(projections, atlas.skin_fit, _anchor(usable))
    for p, g in zip(projections, gains):
        p.colour *= g
        p.gain = g
        if delight and p.shading is not None:
            _light_weights(p, atlas)
    spread = gains.max(axis=0) / gains.min(axis=0)
    if len(projections) > 1 and spread.max() > 1.5:
        warnings.append(f"the photos' exposure or white balance differ (gains up to {spread.max():.1f} times "
                        "apart): evened out")
    skin, own_skin, hair = _tones(projections, atlas)
    base = _base(atlas, skin, own_skin)
    timings["light and gains"] = time.perf_counter() - start
    start = time.perf_counter()
    sources = [p.colour for p in projections] + [base]
    weights = [p.weight for p in projections] + [fuse.BASE_WEIGHT * atlas.covered.astype(np.float32)]
    albedo, share = fuse.blend(sources, weights, atlas.bake.island, fuse.bands_for(atlas.size))
    keep = (atlas.parts == EYE) | (atlas.parts == INNER)
    albedo[keep] = base[keep]
    timings["blend"] = time.perf_counter() - start
    start = time.perf_counter()
    linear = np.clip(albedo, 0.0, None) * atlas.shading[..., None]
    rgba = np.dstack([light.to_srgb(linear), atlas.own_rgba[..., 3]])
    eyes = atlas.parts == EYE
    rgba[eyes] = atlas.own_rgba[eyes]  # (exactly as they were)
    rgba = fuse.pad(rgba, atlas.covered)
    conf = fuse.confidence(projections, atlas)
    tri = fuse.trimap(conf, atlas)
    reach = atlas.photo_parts
    shares = {p.name: float(share[k][reach].mean()) for k, p in enumerate(projections)}
    shares["avatar"] = float(share[-1][reach].mean())
    warnings += _coverage_warnings(tri, atlas)
    if skin is not None and len(skin.per_view) > 1:
        values = np.array(list(skin.per_view.values()))
        apart = float(np.max(light.delta_e(values[:, None], values[None])))
        if apart > 8.0:
            warnings.append(f"the photos disagree on the skin tone by up to {apart:.0f} (Lab delta E) after the light "
                            "and gains: mixed light or a strong colour cast")
    timings["finish"] = time.perf_counter() - start
    if not debug:
        for p in projections:
            p.factors = {}
    prim = model.primitives[atlas.primitive]
    return HeadTexture(rgba, conf, tri, skin, own_skin, hair, shares, list(views), warnings, timings, atlas,
                       projections, model.name, prim.name)


def _anchor(views) -> int | None:
    """The photo whose level the others' gains follow: the most frontal one (it sees both sides of the face, so the
    brightest light on it), if it is turned less than :data:`~.fuse.ANCHOR_YAW`. A photo of the shadowed side of a
    face sees only shadow and would take it for the light (see :mod:`.light`)."""
    frontal = [k for k, v in enumerate(views) if v.method == "landmarks" and abs(v.angles[0]) < fuse.ANCHOR_YAW]
    return min(frontal, key=lambda k: abs(views[k].angles[0])) if frontal else None


def _unique_names(projections) -> None:
    seen = {}
    for p in projections:
        seen[p.name] = seen.get(p.name, 0) + 1
        if seen[p.name] > 1:
            p.name = f"{p.name} {seen[p.name]}"


def _delight(p: Projection, atlas: HeadAtlas) -> list[str]:
    """Stage 4 on one photo: its light fitted to its face skin and divided out; the weights scaled by how well lit
    each texel was. Returns warnings."""
    fit = atlas.skin_fit & (p.weight > 0.02) & (p.face >= round(FIT_FACE * 255))
    colours, weights = p.colour[fit], p.weight[fit]
    weights = weights * light.like_skin(colours, weights)  # not a beard, stubble or make-up
    shading = light.fit_shading(light.luminance(colours), atlas.bake.normal[fit], weights)
    if shading is None:
        return [f"{p.name}: too little face skin to fit its light to: left in"]
    p.shading = shading
    seen = p.weight > 0
    p.colour[seen] = light.delight(p.colour[seen], atlas.bake.normal[seen], shading)
    out = []
    if shading.contrast > HARSH:
        out.append(f"{p.name}: harsh light ({shading.contrast:.1f} times brighter on the lit side): taken out as far "
                   "as it follows the shape of the head; shadows the face casts on itself stay")
    if shading.inliers < POOR_FIT:
        out.append(f"{p.name}: the light fit kept only {100 * shading.inliers:.0f} % of the face skin (shadows the "
                   "face casts, hair over it, or a poor registration)")
    return out


def _light_weights(p: Projection, atlas: HeadAtlas) -> None:
    """A photo's weights scaled by how well lit each texel was (:func:`.light.light_weight`), by the light of the
    anchor photo: a photo's own shading is relative to the brightest light it saw, which in a photo of the shadowed
    side of a face is shadow; its gain (by luminance) says how much less light that was."""
    seen = p.weight > 0
    level = float(light.luminance(np.asarray(p.gain, np.float32)))
    p.weight[seen] *= light.light_weight(p.shading(atlas.bake.normal[seen]) / max(level, 1e-3))


def _tones(projections, atlas: HeadAtlas):
    """``(skin, own_skin, hair)``: the person's skin on the cheeks and forehead and hair on the scalp, as the
    texture will show them (de-lit, gain-compensated, the avatar's own shading back), and the avatar's own skin on
    the same texels."""
    patch = atlas.patch > 0
    skin_samples, skin_weights, hair_samples, hair_weights = {}, {}, {}, {}
    for p in projections:
        on = patch & (p.weight > 0.05) & (p.face >= round(FIT_FACE * 255))
        if on.sum() >= 20:
            skin_samples[p.name] = light.lab(p.colour[on] * atlas.shading[on][:, None])
            skin_weights[p.name] = p.weight[on]
        scalp = (atlas.parts == SCALP) & (p.weight > 0.05) & (p.hair >= round(FIT_FACE * 255))
        if scalp.sum() >= 20:
            hair_samples[p.name] = light.lab(p.colour[scalp] * atlas.shading[scalp][:, None])
            hair_weights[p.name] = p.weight[scalp]
    skin = light.tone(skin_samples, skin_weights)
    hair = light.tone(hair_samples, hair_weights)
    own = light.tone({"avatar": light.lab(atlas.own[patch])}) if patch.sum() >= 20 else None
    return skin, own, hair


def _base(atlas: HeadAtlas, skin, own_skin) -> np.ndarray:
    """The avatar's own texture in albedo (its shading divided out), its skin tone-matched to the person's: each
    channel scaled by the ratio of the two skin tones, on the texels whose colour is like the avatar's skin (hair,
    the eyes and the teeth keep theirs)."""
    base = atlas.own_albedo.copy()
    if skin is None or own_skin is None:
        return base
    from ..outfit.regions import _distance

    ratio = np.clip(skin.linear / np.maximum(own_skin.linear, 1e-4), *TONE_LIMITS).astype(np.float32)
    lab = light.lab(cv2.GaussianBlur(atlas.own, (0, 0), 1.5))
    d = _distance(lab, own_skin.lab.astype(np.float32))
    scalp = atlas.parts == SCALP  # hair there unless it looks just like skin (a bald scalp)
    near, far = np.where(scalp, SCALP_SKIN_LIKE[0], SKIN_LIKE[0]), np.where(scalp, SCALP_SKIN_LIKE[1], SKIN_LIKE[1])
    like = np.clip((far - d) / (far - near), 0.0, 1.0).astype(np.float32)
    like[(atlas.parts == EYE) | ~atlas.covered] = 0.0
    like = fuse.masked_soften(like, atlas.covered, 2.0 * atlas.size / 1024)
    return base * (1.0 + (ratio - 1.0) * like[..., None])


def _coverage_warnings(tri: np.ndarray, atlas: HeadAtlas) -> list[str]:
    """What the photos leave to the avatar's own texture: the face's sides, the back of the head."""
    out = []
    face = atlas.parts == FACE
    if not face.any():
        return out
    x = np.nan_to_num(atlas.bake.position[..., 0])
    z = np.nan_to_num(atlas.bake.position[..., 2])
    corners_z = atlas.corners[..., 2]
    middle = 0.5 * float(corners_z.min() + corners_z.max())  # half way between the face and the back of the head
    for name, region in (("the right side of the head", face & (x < -0.03)), ("the left side of the head",
                                                                             face & (x > 0.03)),
                         ("the back of the head", atlas.photo_parts & (z < middle - 0.02))):
        if region.sum() > 500 and (tri[region] >= fuse.REFINE).mean() < 0.25:
            out.append(f"no photo shows {name}: the avatar's own texture, tone-matched, is left there")
    return out


# -- on a character, in the background --------------------------------------------------------------------------


def wear(character, result: HeadTexture):
    """Put the head texture on ``character`` (a scene :class:`~bifold_demo.characters.character.Character` of the
    model it was made for): its own head texture (``texture_overrides``), the avatar's other users keep theirs.
    Returns the :class:`~bifold_demo.scene.textures.Texture`."""
    texture = result.texture(f"{character.name} head from photos")
    character.texture_overrides[result.primitive] = texture
    return texture


def take_off(character, result: HeadTexture | None = None) -> None:
    """Give ``character`` its avatar's own head texture back."""
    from ..face_photo import head_layout

    if result is not None:
        character.texture_overrides.pop(result.primitive, None)
        return
    layout = head_layout(character.skinned_model)
    if layout is not None:
        character.texture_overrides.pop(character.skinned_model.primitives[layout.primitive].name, None)


SEAM_REACH = 0.01  # metres: a body vertex this near the head mesh's open neck is across the seam from it
NECK_BAND = 0.12  # metres: the body's skin this near the seam and the head's visible neck count for the tone step


def neck_seam(model, head_rgba=None, atlas: HeadAtlas | None = None) -> dict:
    """How the head texture meets the body texture at the neck (Lab delta E; ``nan`` where there is nothing to
    compare): ``"seam"``, the median difference across the seam itself (the head mesh's open edge at the neck and
    the body's points there, within :data:`SEAM_REACH`), where the body texture is skin
    (:func:`~..outfit.regions.body_regions`; on most Rocketbox avatars a collar covers the seam), with ``"pairs"``
    such points; and ``"step"``, the difference between the median colours of the head's visible neck (the atlas's
    neck part) and the body's skin within :data:`NECK_BAND` of the seam (the throat and the neckline), what shows
    where an open collar leaves the join visible. ``head_rgba``: a head texture (RGB(A) image, default the avatar's
    own); ``atlas``: the head's :class:`~.fuse.HeadAtlas` (else the step is not measured)."""
    from ..outfit.regions import body_regions, texture_of
    from .landmarks import face_frame

    regions = body_regions(model)
    frame = face_frame(model)
    head, body = model.primitives[frame.primitive], model.primitives[regions.primitive]
    positions = np.asarray(head.positions, np.float64)
    tri = np.asarray(head.indices, np.int64).reshape(-1, 3)
    _, merged = np.unique(np.round(positions / 1e-5).astype(np.int64), axis=0, return_inverse=True)
    edges = np.sort(merged[tri][:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    open_vertices = np.unique(unique[counts == 1])  # merged ids on the mesh's open edges
    seam = np.nonzero(np.isin(merged, open_vertices) & (positions[:, 1] < frame.chin[1] - 0.02))[0]
    out = {"seam": float("nan"), "pairs": 0, "step": float("nan")}
    if not len(seam):
        return out
    head_image = np.asarray(texture_of(head.texture).rgba if head_rgba is None else head_rgba)
    body_image = texture_of(body.texture).rgba
    size = regions.size
    if body_image.shape[0] != size:
        body_image = cv2.resize(body_image, (size, size), interpolation=cv2.INTER_AREA)
    body_lab = light.lab(light.to_linear(cv2.blur(np.ascontiguousarray(body_image[..., :3]), (3, 3))))
    head_lab = light.lab(light.to_linear(cv2.blur(np.ascontiguousarray(head_image[..., :3]), (3, 3))))

    def texels(uvs, h, w):
        return (np.clip(((1.0 - uvs[:, 1]) * h).astype(int), 0, h - 1),
                np.clip((uvs[:, 0] * w).astype(int), 0, w - 1))

    body_positions = np.asarray(body.positions, np.float64)
    distance = np.linalg.norm(positions[seam, None] - body_positions[None], axis=2)
    nearest = distance.argmin(axis=1)
    close = distance[np.arange(len(seam)), nearest] < SEAM_REACH
    hr, hc = texels(np.asarray(head.uvs, np.float64)[seam[close]], *head_lab.shape[:2])
    br, bc = texels(np.asarray(body.uvs, np.float64)[nearest[close]], size, size)
    skin = regions.skin[br, bc]
    if skin.any():
        out["seam"] = float(np.median(light.delta_e(head_lab[hr, hc][skin], body_lab[br, bc][skin])))
        out["pairs"] = int(skin.sum())
    if atlas is not None:
        seam_points = positions[seam]

        def near_seam(points):
            return np.linalg.norm(points[:, None] - seam_points[None], axis=2).min(axis=1) < NECK_BAND

        body_skin = np.nonzero(regions.skin.ravel())[0]
        body_skin = body_skin[near_seam(regions.bake.position.reshape(-1, 3)[body_skin].astype(np.float64))]
        neck = np.nonzero((atlas.parts == fuse.NECK).ravel())[0]
        if len(body_skin) > 20 and len(neck) > 20:
            scale = head_lab.shape[0] / atlas.size
            rows, cols = np.unravel_index(neck, atlas.parts.shape)
            head_neck = head_lab[(rows * scale).astype(int), (cols * scale).astype(int)]
            out["step"] = float(light.delta_e(np.median(head_neck, axis=0),
                                              np.median(body_lab.reshape(-1, 3)[body_skin], axis=0)))
    return out


def skin_gap(result: HeadTexture, model) -> float:
    """The colour difference (Lab delta E) between the person's skin on the head texture (:attr:`HeadTexture.skin`)
    and the skin of the avatar's body (its hands, :attr:`~..outfit.regions.BodyRegions.skin_lab`): what stage 6
    closes."""
    from ..outfit.regions import body_regions

    if result.skin is None:
        return float("nan")
    return float(light.delta_e(result.skin.lab, body_regions(model).skin_lab))


PREVIEW_YAWS = (0.0, 35.0, 80.0)  # front, three-quarter, side


def preview(model, result: HeadTexture | None, size: int = 480, renderer=None, yaws=PREVIEW_YAWS) -> np.ndarray:
    """Renders of ``model`` wearing ``result`` (``None``: its own head texture) from the front, three-quarter and
    the side, lit like the fair, side by side (a BGR image; needs OpenGL)."""
    from ...scene import Scene
    from ...scene.renderer import Renderer
    from ..character import Character
    from .camera import PhotoCamera
    from .landmarks import face_frame

    own = renderer is None
    renderer = renderer or Renderer(size, size)
    try:
        scene = Scene(background=(150, 160, 172), ambient=0.55, light_direction=(-0.4, -0.6, -0.7))
        person = Character(model, name="preview", shadow=False)
        if result is not None:
            wear(person, result)
        scene.add(person)
        scene.update(0.0, 0.0)
        pivot = face_frame(model).pivot - np.array([0.0, 0.04, 0.0])
        tiles = []
        for yaw in yaws:
            camera = PhotoCamera.around(pivot, 0.62, yaw, 0.0, fov=35.0, size=(renderer.width, renderer.height))
            scene.camera = camera.scene_camera()
            tiles.append(renderer.render(scene))
        return np.hstack(tiles)
    finally:
        if own:
            renderer.release()


TRIMAP_COLOURS = {fuse.GENERATE: (60, 60, 200), fuse.REFINE: (40, 190, 230), fuse.KEEP: (80, 200, 80),
                  fuse.TRIMAP_OUTSIDE: (30, 30, 30)}  # BGR


def debug_images(result: HeadTexture) -> dict[str, np.ndarray]:
    """BGR pictures of what the fusion did, by file name: the texture, the atlas's parts, the confidence, the
    trimap (green keep, yellow refine, red generate), the avatar's own shading, and per photo its colours in the
    layout (light taken out) and its weights."""
    out = {"texture.png": cv2.cvtColor(result.rgba[..., :3], cv2.COLOR_RGB2BGR)}
    atlas = result.atlas
    if atlas is not None:
        out["parts.png"] = atlas.image()
        out["own_shading.png"] = np.clip(atlas.shading * 170.0, 0, 255).astype(np.uint8)
    out["confidence.png"] = np.clip(result.confidence * 255.0 + 0.5, 0, 255).astype(np.uint8)
    tri = np.zeros((*result.trimap.shape, 3), np.uint8)
    for value, colour in TRIMAP_COLOURS.items():
        tri[result.trimap == value] = colour
    out["trimap.png"] = tri
    for k, p in enumerate(result.projections):
        stem = f"photo{k + 1}_{p.name.replace(' ', '_')}"
        out[f"{stem}_colour.png"] = cv2.cvtColor(light.to_srgb(p.colour), cv2.COLOR_RGB2BGR)
        out[f"{stem}_weight.png"] = np.clip(p.weight * 255.0 + 0.5, 0, 255).astype(np.uint8)
        for name, factor in p.factors.items():
            out[f"{stem}_{name}.png"] = np.clip(factor * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return out


_pool: ThreadPoolExecutor | None = None
_pool_lock = threading.Lock()


def head_texture_async(photos, model, **options) -> Future:
    """:func:`head_texture` on a worker thread (one at a time): a :class:`~concurrent.futures.Future` of the
    :class:`HeadTexture` (its ``result()`` raises what the work raised)."""
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = ThreadPoolExecutor(1, thread_name_prefix="head-texture")
    return _pool.submit(head_texture, photos, model, **options)


def describe_light(shading: light.Shading | None) -> str:
    """Where a photo's light came from, in words (for the debug output)."""
    if shading is None:
        return "not fitted"
    x, y, z = shading.direction
    parts = []
    if abs(x) > 0.35:
        parts.append("the person's left" if x > 0 else "the person's right")
    if abs(y) > 0.35:
        parts.append("above" if y > 0 else "below")
    if abs(z) > 0.35:
        parts.append("the front" if z > 0 else "behind")
    side = " and ".join(parts) or "all round"
    return f"from {side}, {shading.strength:.2f} directional, contrast {shading.contrast:.1f}"


__all__ = ["HeadTexture", "debug_images", "describe_light", "fuse_views", "head_texture", "head_texture_async",
           "neck_seam", "preview", "skin_gap", "take_off", "wear"]
