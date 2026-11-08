"""Stage 3: the photos projected into the head texture's own layout and fused into one texture.

**What each texel is** (:class:`HeadAtlas`, once per head model and texture size). The UV bake
(:mod:`..uvbake`) gives every texel of the head texture its point and normal on the head in the bind pose. Each
texel is also given a part (:data:`PARTS`):

* ``eye``: the eyeballs (the eye bones' texels and the bake's overlap: both eyes share one patch of the texture) and
  the eye openings (inside the lids' contour seen from the front: the eyeballs and the lids' inner rims). A photo's
  eyes look wherever the person looked: projected, they would paint a second, misplaced iris on both eyeballs, and
  the white of the eye on the rims. The avatar's own eyes stay, exactly (stage 6 recolours the irises).
* ``inner``: what no camera outside the head ever sees: the inside of the mouth (teeth, gums, tongue, behind closed
  lips in the bind pose), the inside of the nostrils and of the neck. Found by z-buffers (:mod:`.raster`) from 33
  directions around the head, with the body's triangles in them (they close the head mesh's open neck, through which
  a camera from below would see the mouth from inside), and none from straight below; the avatar's own texture stays
  there, tone-matched where it looks like skin.
* ``neck``: texels the neck and spine bones move (the head mesh reaches down to the collar), and what the segmenter
  calls body skin on renders of the avatar below the mouth (under the jaw). The segmenter calls a person's neck body
  skin, so body skin counts there.
* ``scalp``: what the segmenter calls hair on renders of the avatar (:func:`~.register.head_regions`, per triangle;
  specks voted away by the neighbouring triangles, and spread into the strands of long hair outside the head's
  outline), or, without those labels, the texels whose colour is far from the avatar's skin. A UV island that is
  mostly scalp is all scalp (a strand of hair the segmenter partly took for skin). The *hairline* is the scalp and
  the face skin within :data:`HAIRLINE` of it.
* ``face``: the rest: the face, the ears, the skin of the head.

Features (:data:`FEATURES`: brows, eyes, lips, nostrils) and the cheeks and forehead (the skin tone's texels) come
from the stage-0 landmarks, seen from the front. On renders and on real photos the segmenter calls brows, a light
beard, lips and ears face skin (all 20 brow landmarks in each of 4 renders and 4 photos), so the photos' brows are
fused like the rest of the face; the features are only left out of the light fit, the gain compensation and the skin
tone, whose colours they would skew.

**One photo in the layout** (:func:`project`, a :class:`Projection`). For every texel of the face, scalp and neck:
its point projected by the view's camera is depth-tested against the head mesh as that camera sees it (a z-buffer of
the head, about 400 pixels tall; the nearest depth within a pixel counts, so samples never straddle the edge of the
nose, an ear or the chin), and the photo's colour is read where :meth:`~.register.View.photo_pixels` says (through
the face warp), from a mipmapped copy of the photo cropped to the head, at the level that matches the texel's size in
the photo (no aliasing in a 12-megapixel photo). Its weight is the product of:

* the viewing angle: the cosine between the normal and the direction to the camera, to the power
  :data:`ANGLE_POWER` (0 below :data:`MIN_FACING`);
* the segmentation class at the pixel: the segmenter's confidence of face skin on face texels (and of hair in the
  hairline: the only face texels a photo's hair may go to, as hairlines differ), of hair and, at half weight,
  "others" (blond hair seen from behind) on the scalp (never face skin: a person's forehead under a receding
  hairline must not paint skin over the avatar's hair), of face and body skin on the neck; never background or
  clothes, and only the person whose head the camera was fitted to (the connected parts of the masks that the
  avatar's head covers): other people in the photo do not count;
* the distance from the edge of those masks, from the photo's border and from the head's outlines as the camera
  sees them (where the z-buffer's depth jumps: just behind the jaw a texel passes the depth test while its pixel,
  a little misregistered, shows the neck below): rising from 0 to 1 over :data:`EDGE_RAMP`, :data:`BORDER_RAMP` and
  :data:`OUTLINE_RAMP` of the face's size, so a slightly misregistered edge does not bleed hair or background in;
* the view's quality (stage 1) and the photo pixels per texel (:data:`GOOD_FOOTPRINT` for full marks, seen squarely;
  smoothed over a few texels, as it is constant per triangle), counted in texels of the head's ordinary texture
  (:attr:`HeadAtlas.marks_size`: 1024 for the Rocketbox heads), so a texture made at their 2048 detail size marks a
  photo as one made at 1024 does, and the confidence and the trimap below do not change with the size;
* a tenth on the lips of a photo with an open mouth.

Stage 4 (:mod:`.light`) then divides each projection by its photo's fitted light and, after the gains, multiplies the
weight by how well lit each texel was.

**Gain compensation** (:func:`gains`). Exposure and white balance differ between photos. Each photo gets a gain per
channel from a least-squares fit on the overlaps of the face skin (Brown and Lowe's, as in OpenCV's stitcher): the
mean colours two photos show in their overlap (relative to the overlaps' mean) are made to agree. The most frontal
photo (turned less than :data:`ANCHOR_YAW`) is the anchor, held near a gain of 1, and the others follow it: de-lit,
a photo of the shadowed side of a face is scaled to the brightest light it saw, which is shadow, and its gain is
what puts it back at the level of the light. Without an anchor every gain is pulled towards 1 alike.

**The blend** (:func:`blend`). A multi-band blend (Burt and Adelson; :data:`BANDS` bands at 1024 texels) island by
island: in each UV island's box every source is filled outside where it has colours by push-pull (:func:`push_pull`),
so no colour of another island or of the empty gutter ever enters a blur, and its Laplacian pyramid is blended with
the Gaussian pyramid of its soft winner-takes-all weight (the weights to the power :data:`WINNER_POWER`, normalised).
The finest band thus comes from the best photo at each texel (no ghosted double edges where the registrations differ
by a texel or two), the coarse ones mix the photos over wide areas (no visible seams in tone). The avatar's own
texture is one more source with a small constant weight (:data:`BASE_WEIGHT`: what a good photo gives a texel turned
65 degrees away), tone-matched to the person's skin, so texels no photo shows keep it, the photos' tone feathers
into it, and the last, grazing texels of a photo's reach, where its colours are least sure, lean on it.

**Padding** (:func:`pad`). Around the islands the texture is filled by push-pull from the islands' own texels, so
bilinear filtering and the renderer's mipmaps never pull in other colours (the avatars' own textures are black there,
which shows as dark seams on far-away heads).

**Confidence and trimap** (:func:`confidence`, :func:`trimap`). A texel's confidence is ``1 - exp(-W / 0.25)`` of the
total photo weight ``W`` (a good photo that sees a texel squarely gives it 0.9 or more, one turned 60 degrees away
about 0.4); from it the trimap for stage 5 (the generative completion): ``keep`` (confidence :data:`KEEP_AT` and up,
and the eyes and the inside of the mouth, which stay the avatar's own), ``refine`` (from :data:`REFINE_AT`, about
where the avatar's own texture starts to outweigh the photos in the blend), ``generate`` (the rest).
"""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass, field

import cv2
import numpy as np

from .. import uvbake
from . import canonical as cf
from . import light, raster, raycast
from .camera import PhotoCamera

PARTS = ("outside", "face", "scalp", "neck", "eye", "inner")
OUTSIDE, FACE, SCALP, NECK, EYE, INNER = range(len(PARTS))
FEATURES = ("brows", "eyes", "lips", "nostrils")
BROWS, EYES, LIPS, NOSTRILS = 1, 2, 4, 8  # bits of HeadAtlas.features
CHEEKS, FOREHEAD = 1, 2  # HeadAtlas.patch
TRIMAP = ("generate", "refine", "keep")
GENERATE, REFINE, KEEP = range(len(TRIMAP))
TRIMAP_OUTSIDE = 255
ATLAS_VERSION = 4

HAIRLINE = 0.02  # metres: the face skin this near the scalp may take a photo's hair
FEATURE_MARGIN = 0.003  # metres around the features' landmarks
HAIR_DISTANCE = 22.0  # Lab distance from the skin (lightness counting half) beyond which a texel is hair (fallback)
INNER_VIEWS = tuple((float(yaw), float(pitch)) for pitch in (-50.0, -20.0, 15.0, 50.0) for yaw in range(0, 360, 45)) + (
    (0.0, 85.0),)  # (yaw, pitch): the cameras that find what is never seen from outside (none from straight below:
# it would sit inside the body and look up through the open neck)
INNER_SIZE = 320  # pixels: their z-buffers
RASTER_SIZE = 400  # pixels: the longer side of a view's z-buffer of the head
DEPTH_TOLERANCE = 0.003  # metres: a texel this much behind the nearest surface still counts as seen
MIN_FACING = 0.1  # cosine: texels turned further away from the camera are not taken from the photo
ANGLE_POWER = 3.0
EDGE_RAMP = 0.05  # of the face's size: the weight rises from 0 to 1 over this distance from a mask's edge
BORDER_RAMP = 0.02  # ... and over this from the photo's border
OUTLINE_RAMP = 0.03  # ... and over this from an outline of the head (where its depth jumps by DEPTH_EDGE or more)
DEPTH_EDGE = 0.01  # metres between neighbouring pixels of a view's z-buffer: an outline
GOOD_FOOTPRINT = 1.0  # photo pixels per texel (seen squarely; texels at HeadAtlas.marks_size) for full marks
RESOLUTION_SMOOTHING = 4.0  # texels (at 1024): the Gaussian that smooths the photo pixels per texel
MOUTH_OPEN = 0.35  # MediaPipe's jawOpen score above which the lips count a tenth
MASK_LEVEL = 0.5  # segmenter confidence: the edge of a mask
OTHERS_ON_SCALP = 0.5
BANDS = 5  # multi-band levels at 1024 texels (one more per doubling)
WINNER_POWER = 6.0
BASE_WEIGHT = 0.08  # the avatar's own texture in the blend (a confidence of about 0.27: where "generate" ends)
CONFIDENCE_SCALE = 0.25  # confidence = 1 - exp(-W / this), W the total photo weight
KEEP_AT, REFINE_AT = 0.7, 0.25  # confidence: the trimap's thresholds
SIGMA_N, SIGMA_G = 0.01, 0.1  # gain compensation: noise of the overlaps' mean colours (relative), spread of the gains
SIGMA_G_ANCHOR, SIGMA_G_FREE = 0.02, 1.0  # ... of the anchor photo's gain, and of the others' when there is one
ANCHOR_YAW = 30.0  # degrees: the anchor is the most frontal photo, if it is turned less than this
MIN_OVERLAP = 20.0  # ... and the least overlap that counts (the sum of the smaller weight over the face skin)
PAD_DILATE = 4  # texels the islands are first extended by their own border colours, at 1024 (then push-pull)


# -- the atlas: what each texel is -------------------------------------------------------------------------------


@dataclass
class HeadAtlas:
    """What each texel of an avatar's head texture is (see the module docs). Arrays are ``(size, size)`` like the
    bake's (row 0 at the top of the texture image)."""

    model: str
    primitive: int
    bake: uvbake.UVBake
    parts: np.ndarray  # uint8, PARTS
    hairline: np.ndarray  # bool: scalp, and face skin within HAIRLINE of it
    features: np.ndarray  # uint8 bits: BROWS, EYES, LIPS, NOSTRILS
    patch: np.ndarray  # uint8: CHEEKS, FOREHEAD (the skin tone's texels)
    scale: np.ndarray  # float32 metres per texel (0 outside)
    jacobian: np.ndarray  # (T, 3, 2) the change of a triangle's point per texel (column, row)
    corners: np.ndarray  # (T, 3, 3) the head primitive's triangles (bind pose)
    own: np.ndarray  # float32 (size, size, 3): the avatar's own texture, linear RGB
    own_rgba: np.ndarray  # uint8 (size, size, 4): the same as it is (RGBA)
    shading: np.ndarray  # float32: the avatar's own painted shading (light.own_shading), 1 outside
    own_fit: light.Shading = field(default_factory=light.Shading.flat)
    scalp_from: str = "segmenter"  # where the scalp came from: "segmenter" (head_regions) or "colour"
    marks_size: int = 0  # the texture size the photos' resolution is marked at (:func:`marks_size`; 0: size)

    @property
    def size(self) -> int:
        return self.bake.size

    @property
    def covered(self) -> np.ndarray:
        return self.parts != OUTSIDE

    @property
    def photo_parts(self) -> np.ndarray:
        """bool: the texels photos are projected to (face, scalp, neck)."""
        return (self.parts == FACE) | (self.parts == SCALP) | (self.parts == NECK)

    @property
    def skin_fit(self) -> np.ndarray:
        """bool: the face skin a photo's light is fitted to and the photos' gains are compared on: the face part,
        not a feature, not the hairline."""
        return (self.parts == FACE) & (self.features == 0) & ~self.hairline

    @property
    def own_albedo(self) -> np.ndarray:
        """The avatar's own texture with its painted shading divided out."""
        return self.own / self.shading[..., None]

    def image(self) -> np.ndarray:
        """A BGR picture of the parts over the avatar's texture (for checking them): face green, scalp red, neck
        yellow, eyes blue, inner magenta; features darker, the hairline striped, cheeks and forehead lighter."""
        colours = np.array([(0, 0, 0), (90, 200, 90), (40, 40, 210), (60, 200, 230), (220, 120, 40), (200, 60, 200)],
                           np.float32)
        base = cv2.cvtColor(light.to_srgb(self.own), cv2.COLOR_RGB2BGR).astype(np.float32)
        tint = colours[self.parts]
        tint[self.features > 0] *= 0.55
        tint[self.patch > 0] = 0.5 * tint[self.patch > 0] + 127
        stripes = (np.add.outer(np.arange(self.size), np.arange(self.size)) // 6) % 2 == 0
        tint[self.hairline & (self.parts == FACE) & stripes] = (40, 40, 210)
        out = np.where(self.covered[..., None], 0.45 * base + 0.55 * tint, base)
        return np.clip(out, 0, 255).astype(np.uint8)


_atlases: dict[tuple, tuple] = {}
ATLASES = 2  # head atlases kept in memory (about 66 MiB each at 1024 texels, 265 MiB at 2048)


def head_atlas(model, head, size: int | None = None, *, labels="auto", cache_dir=None) -> HeadAtlas:
    """The :class:`HeadAtlas` of ``model``'s head (``head``: its stage-0 :class:`~.landmarks.HeadLandmarks`) at
    ``size`` texels (default: the head texture's own size). ``labels``: the segmenter's labels of the head's outline
    (:func:`~.register.head_regions`); ``"auto"`` takes them from that function's cache when they are there, ``None``
    finds the scalp by colour. Kept in memory for the last few models and on disk next to the landmarks
    (``cache_dir`` as for :func:`~.landmarks.head_landmarks`; ``False``: memory only)."""
    from .register import Outline, head_regions

    primitive = head.primitive
    prim = model.primitives[primitive]
    if size is None:
        size = texture_size(prim)
    if isinstance(labels, str) and labels == "auto":
        labels = head_regions(model, primitive, cache_dir=cache_dir)
    outline = Outline.of(model, primitive) if labels is not None else None
    key = _atlas_key(model, head, size, labels)
    hit = _atlases.get((id(model), key))
    if hit is not None and hit[0] is model:
        return hit[1]
    from .landmarks import cache_folder

    path = None if cache_dir is False else cache_folder(cache_dir) / f"atlas-{key}.npz"
    stored = _read_atlas(path, key)
    if stored is not None:
        bake = uvbake.from_triangles(model, primitive, stored["triangle"], stored["overlap"])
        parts, hairline, features, patch = (stored[k] for k in ("parts", "hairline", "features", "patch"))
        scalp_from = str(stored["scalp_from"])
    else:
        triangle, overlap = uvbake.rasterise(prim, size)
        bake = uvbake.from_triangles(model, primitive, triangle, overlap)
        parts, hairline, features, patch, scalp_from = _label(model, head, bake, labels, outline)
        _write_atlas(path, key, triangle, overlap, parts, hairline, features, patch, scalp_from)
    atlas = _finish_atlas(model, primitive, bake, parts, hairline, features, patch, scalp_from)
    _atlases[(id(model), key)] = (model, atlas)
    while len(_atlases) > ATLASES:
        _atlases.pop(next(iter(_atlases)))
    return atlas


def texture_size(prim) -> int:
    """The head texture's larger side (1024 without one): of its largest version (``full_texture``: 2048 for the
    heads of a library prepared in format 2), so that a head from photos is made at the size the head can show."""
    from ..outfit.regions import texture_of

    texture = texture_of(getattr(prim, "full_texture", prim.texture))
    return 1024 if texture is None else int(max(texture.width, texture.height))


def marks_size(prim) -> int:
    """The texture size a photo's resolution is marked at (:data:`GOOD_FOOTPRINT` photo pixels per texel): the head's
    ordinary texture's larger side (1024 for the Rocketbox heads, whose 2048 version is their ``detail``; 1024
    without one). A head texture made at another size marks a photo as one at this size does, so how much the photos
    count, against each other and against the avatar's own texture, how sure the result is and what stage 5 fills in
    do not change with the size: at 2048 a photo with half a pixel per texel has full marks."""
    from ..outfit.regions import texture_of

    texture = texture_of(prim.texture)
    return 1024 if texture is None else int(max(texture.width, texture.height))


def _atlas_key(model, head, size, labels) -> str:
    prim = model.primitives[head.primitive]
    h = hashlib.blake2b(f"headtex-atlas-{ATLAS_VERSION}-{size}".encode(), digest_size=12)
    arrays = [prim.positions, prim.normals, prim.uvs, prim.indices, prim.joints, prim.weights, head.points,
              np.zeros(1) if labels is None else labels]
    for array in map(np.asarray, arrays):
        h.update(f"{array.dtype}{array.shape}".encode())
        h.update(np.ascontiguousarray(array).tobytes())
    h.update("\n".join(model.skeleton.names).encode())
    return h.hexdigest()


def _read_atlas(path, key):
    if path is None or not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            if str(z["key"]) != key:
                return None
            size = int(z["size"])
            out = {name: z[name] for name in ("parts", "features", "patch", "scalp_from")}
            out["triangle"] = z["triangle"].astype(np.int32)
            out["overlap"] = np.unpackbits(z["overlap"])[:size * size].reshape(size, size).astype(bool)
            out["hairline"] = np.unpackbits(z["hairline"])[:size * size].reshape(size, size).astype(bool)
            return out
    except Exception:  # noqa: BLE001 - a damaged or foreign file: made again
        return None


def _write_atlas(path, key, triangle, overlap, parts, hairline, features, patch, scalp_from) -> None:
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp.npz")
        small = np.int16 if triangle.max() < 32767 else np.int32
        np.savez_compressed(tmp, key=np.array(key), size=np.array(triangle.shape[0]), triangle=triangle.astype(small),
                            overlap=np.packbits(overlap), hairline=np.packbits(hairline), parts=parts,
                            features=features, patch=patch, scalp_from=np.array(scalp_from))
        os.replace(tmp, path)
    except OSError:
        pass  # a read-only folder: kept in memory only


def _finish_atlas(model, primitive, bake, parts, hairline, features, patch, scalp_from) -> HeadAtlas:
    """The parts of the atlas that are quick to make again: the scales, the avatar's own texture and shading."""
    from ..outfit.regions import texture_of

    prim = model.primitives[primitive]
    size = bake.size
    corners = raycast.corners_of(prim.positions, prim.indices)
    jacobian, area_scale = _uv_jacobian(prim, corners, size)
    scale = np.zeros((size, size), np.float32)
    scale[bake.covered] = area_scale[bake.triangle[bake.covered]]
    texture = texture_of(getattr(prim, "full_texture", prim.texture))
    if texture is None:
        rgba = np.full((size, size, 4), (190, 150, 130, 255), np.uint8)
    else:
        rgba = texture.rgba
        if rgba.shape[:2] != (size, size):
            shrink = rgba.shape[0] > size
            rgba = cv2.resize(rgba, (size, size), interpolation=cv2.INTER_AREA if shrink else cv2.INTER_LINEAR)
    own = light.to_linear(rgba[..., :3])
    face = (parts == FACE) & (features == 0) & ~hairline
    front = face & (np.nan_to_num(bake.normal[..., 2]) > 0.3)  # the face skin a photo from the front sees
    fit, shading = light.own_shading(own, bake.normal, face | (parts == NECK), front)
    return HeadAtlas(model.name, primitive, bake, parts, hairline, features, patch, scale, jacobian, corners, own,
                     np.ascontiguousarray(rgba), shading, fit, scalp_from, marks_size(prim))


def _uv_jacobian(prim, corners, size) -> tuple[np.ndarray, np.ndarray]:
    """``(jacobian, scale)`` per triangle: ``(T, 3, 2)`` the change of the point per texel (column, row), and
    ``(T,)`` metres per texel (the square root of the area ratio)."""
    tri = np.asarray(prim.indices, np.int64).reshape(-1, 3)
    uv = uvbake.uv_pixels(prim.uvs, size)[tri]  # (T, 3, 2) texel coordinates (x = column, y = row)
    du = np.stack([uv[:, 1] - uv[:, 0], uv[:, 2] - uv[:, 0]], axis=2)  # (T, 2, 2) columns: the two edges
    dp = np.stack([corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]], axis=2)  # (T, 3, 2)
    det = du[:, 0, 0] * du[:, 1, 1] - du[:, 0, 1] * du[:, 1, 0]
    ok = np.abs(det) > 1e-12
    inverse = np.zeros_like(du)
    inverse[ok, 0, 0], inverse[ok, 1, 1] = du[ok, 1, 1] / det[ok], du[ok, 0, 0] / det[ok]
    inverse[ok, 0, 1], inverse[ok, 1, 0] = -du[ok, 0, 1] / det[ok], -du[ok, 1, 0] / det[ok]
    jacobian = dp @ inverse
    area3 = 0.5 * np.linalg.norm(np.cross(dp[:, :, 0], dp[:, :, 1]), axis=1)
    scale = np.where(ok, np.sqrt(area3 / np.maximum(0.5 * np.abs(det), 1e-12)), 0.0).astype(np.float32)
    return jacobian, scale


def _label(model, head, bake, labels, outline):
    """``(parts, hairline, features, patch, scalp_from)`` of a bake (see the module docs)."""
    from ..outfit.regions import bone_group
    from .landmarks import face_frame

    size = bake.size
    covered = bake.covered
    prim = model.primitives[head.primitive]
    corners = raycast.corners_of(prim.positions, prim.indices)
    names = model.skeleton.names
    group = np.array([bone_group(n) for n in names] + ["none"])
    bones = np.where(bake.bone >= 0, bake.bone, len(names))
    eye_bones = [i for i, n in enumerate(names) if n.endswith(("REye", "LEye"))]
    parts = np.zeros((size, size), np.uint8)
    parts[covered] = FACE
    parts[covered & (group[bones] != "head")] = NECK
    frame = face_frame(model)
    outer = _outer(corners, bake, frame.pivot, _occluders(model, head.primitive))
    parts[covered & ~outer] = INNER
    parts[covered & (np.isin(bake.bone, eye_bones) | bake.overlap)] = EYE
    features, patch, openings = _features(head, bake, corners, frame)
    parts[openings & (parts == FACE)] = EYE
    scalp_from = "segmenter"
    scalp = np.zeros((size, size), bool)
    if labels is not None and outline is not None and outline.sources is not None:
        from .register import HAIR, OTHER

        mine = outline.sources[:, 0] == head.primitive
        label = np.full(len(corners), -1, np.int8)
        label[outline.sources[mine, 1]] = np.asarray(labels)[mine]
        label = _smooth_labels(label, corners)
        scalp[covered] = label[bake.triangle[covered]] == HAIR
        # what the segmenter calls body skin on the avatar below the mouth: under the jaw
        body = np.zeros((size, size), bool)
        body[covered] = label[bake.triangle[covered]] == OTHER
        low = np.nan_to_num(bake.position[..., 1], nan=np.inf) < frame.mouth[:, 1].mean()
        parts[body & low & (parts == FACE) & (features == 0)] = NECK
    else:
        scalp_from = "colour"
        scalp = _hair_by_colour(model, head, bake, patch)
    scalp &= ((parts == FACE) | (parts == NECK)) & (features == 0)
    parts[scalp] = SCALP
    _hair_islands(parts, bake.island)
    scalp = parts == SCALP
    hairline = scalp | ((parts == FACE) & _near(scalp, bake, corners, prim, HAIRLINE))
    return parts, hairline, features, patch, scalp_from


HAIR_ISLAND = 0.6  # an island this much scalp is all scalp (a strand of an avatar's long hair)


def _hair_islands(parts: np.ndarray, island: np.ndarray) -> None:
    """Islands mostly scalp become all scalp (in place): the strands of long hair that is part of the head mesh are
    islands of their own, and a patch of them the segmenter took for skin must never take a photo's skin."""
    photo = (parts == FACE) | (parts == SCALP) | (parts == NECK)
    ids = island[photo]
    total = np.bincount(ids, minlength=island.max() + 1)
    hair = np.bincount(island[parts == SCALP], minlength=island.max() + 1)
    mostly = np.nonzero(hair >= HAIR_ISLAND * np.maximum(total, 1))[0]
    parts[photo & np.isin(island, mostly)] = SCALP


def _smooth_labels(label: np.ndarray, corners: np.ndarray, rounds: int = 2, spread: int = 4) -> np.ndarray:
    """The segmenter's labels per triangle (``-1``: none) with specks removed: each labelled triangle takes the
    label covering most area round its corners (triangles sharing a point, across the UV seams), ``rounds`` times;
    then triangles without a label (outside the head's outline: strands of long hair the neck moves) take their
    neighbours' for ``spread`` rings."""
    from .register import HAIR, OTHER, SKIN

    classes = np.array([SKIN, HAIR, OTHER])
    points = corners.reshape(-1, 3)
    _, vertex = np.unique(np.round(points / 1e-5).astype(np.int64), axis=0, return_inverse=True)
    vertex = vertex.reshape(-1, 3)
    area = 0.5 * np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1)
    label = np.where(np.isin(label, classes), label, -1).astype(np.int8)

    def around():
        votes = np.zeros((vertex.max() + 1, len(classes)))
        for k, c in enumerate(classes):
            mine = label == c
            np.add.at(votes[:, k], vertex[mine].ravel(), np.repeat(area[mine], 3))
        return votes[vertex].sum(axis=1)

    for _ in range(rounds):
        votes = around()
        labelled = (label >= 0) & (votes.sum(axis=1) > 0)
        label[labelled] = classes[votes[labelled].argmax(axis=1)]
    for _ in range(spread):
        votes = around()
        empty = (label < 0) & (votes.sum(axis=1) > 0)
        if not empty.any():
            break
        label[empty] = classes[votes[empty].argmax(axis=1)]
    return label


def _occluders(model, primitive: int) -> np.ndarray:
    """``(T, 3, 3)``: the body's triangles, which close the head mesh's open neck (seen from below, the inside of
    the mouth would show through it)."""
    from ..outfit.regions import body_primitive

    body = body_primitive(model)
    if body is None or body == primitive:
        return np.zeros((0, 3, 3))
    prim = model.primitives[body]
    return raycast.corners_of(prim.positions, prim.indices)


def _outer(corners, bake, pivot, occluders=None) -> np.ndarray:
    """bool per texel: seen by at least one camera around the head (:data:`INNER_VIEWS`): first by the triangles
    that are the nearest surface at some pixel of a z-buffer, then, for the texels of the others, by a depth test
    (small triangles may own no pixel). ``occluders``: more triangles that hide the head (the body)."""
    size = bake.size
    seen_triangle = np.zeros(len(corners), bool)
    cameras = [PhotoCamera.around(pivot, 0.8, yaw, pitch, fov=32.0, size=(INNER_SIZE, INNER_SIZE))
               for yaw, pitch in INNER_VIEWS]
    everything = corners if occluders is None or not len(occluders) else np.concatenate([corners, occluders])
    rasters = []
    for camera in cameras:
        r = raster.rasterise(camera, everything, cull=True)
        shown = np.unique(r.triangle[r.triangle >= 0])
        seen_triangle[shown[shown < len(corners)]] = True
        rasters.append((camera, r.nearest(1)))
    outer = np.zeros((size, size), bool)
    covered = bake.covered
    outer[covered] = seen_triangle[bake.triangle[covered]]
    rows, cols = np.nonzero(covered & ~outer)
    if len(rows):
        points = bake.position[rows, cols].astype(np.float64)
        normals = bake.normal[rows, cols].astype(np.float64)
        found = np.zeros(len(rows), bool)
        for camera, near in rasters:  # (a tight tolerance: the teeth are only millimetres behind the lips)
            to_camera = camera.position - points
            facing = np.einsum("nk,nk->n", normals, to_camera) / np.linalg.norm(to_camera, axis=1)
            found |= (facing > 0) & _depth_test(camera, near, points, facing, INNER_TOLERANCE, 0.5)
        outer[rows[found], cols[found]] = True
    return outer


INNER_TOLERANCE = 0.0005  # metres: the depth test's tolerance when looking for the parts never seen from outside


def _depth_test(camera: PhotoCamera, near: np.ndarray, points, facing, tolerance: float = DEPTH_TOLERANCE,
                slope_pixels: float = 2.5) -> np.ndarray:
    """bool per point: no nearer surface in ``near`` (a z-buffer's :meth:`~.raster.Raster.nearest` depths, at
    ``camera``'s size) than the point, give or take ``tolerance`` metres and, at grazing angles, the change of depth
    over ``slope_pixels`` pixels (it changes fast across a pixel there, and the nearest depth comes from up to one
    and a half pixels away)."""
    h, w = near.shape
    pixels = camera.project(points)
    depth = camera.depth(points)
    ix = np.clip(np.round(pixels[:, 0]).astype(np.int64), 0, w - 1)
    iy = np.clip(np.round(pixels[:, 1]).astype(np.int64), 0, h - 1)
    inside = (pixels[:, 0] > -0.5) & (pixels[:, 0] < w - 0.5) & (pixels[:, 1] > -0.5) & (pixels[:, 1] < h - 0.5)
    pixel = depth / camera.focal  # metres per pixel at the point's depth
    slope = np.sqrt(np.clip(1.0 - facing * facing, 0.0, 1.0)) / np.maximum(facing, 0.15)
    return inside & (depth > 0) & (depth <= near[iy, ix] + tolerance + slope_pixels * pixel * slope)


def _features(head, bake, corners, frame):
    """``(features, patch, front)``: the feature bits and the cheeks and forehead (from the landmarks, seen from the
    front), and which texels the front view sees."""
    size = bake.size
    eye = frame.eye_distance
    camera = PhotoCamera.around(frame.centre + np.array([0.0, 0.3 * eye, 0.0]), 2.0, 0.0, 0.0, fov=9.0,
                                size=(800, 800))
    near = raster.rasterise(camera, corners).nearest(1)
    covered = bake.covered
    rows, cols = np.nonzero(covered)
    points = bake.position[rows, cols].astype(np.float64)
    normals = bake.normal[rows, cols].astype(np.float64)
    to_camera = camera.position - points
    facing = np.einsum("nk,nk->n", normals, to_camera) / np.linalg.norm(to_camera, axis=1)
    seen = (facing > 0.05) & _depth_test(camera, near, points, facing)
    pixels = camera.project(points)
    marks = camera.project(head.points)
    px_per_m = camera.focal / 2.0
    margin = max(1, round(FEATURE_MARGIN * px_per_m))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1))

    def polygon(ids, hull=False, grow=True):
        pts = marks[list(ids)]
        if hull:
            pts = cv2.convexHull(pts.astype(np.float32)).reshape(-1, 2)
        layer = np.zeros((800, 800), np.uint8)
        cv2.fillPoly(layer, [np.round(pts).astype(np.int32)], 1)
        return cv2.dilate(layer, kernel) if grow else layer

    nostrils = (98, 64, 240, 75, 79, 166, 60, 20, 242, 141, 94, 370, 462, 250, 290, 392, 309, 305, 460, 294, 327, 2)
    layers = {BROWS: np.maximum(polygon(cf.RIGHT_BROW, True), polygon(cf.LEFT_BROW, True)),
              EYES: np.maximum(polygon(cf.RIGHT_EYE), polygon(cf.LEFT_EYE)),
              LIPS: polygon(cf.LIPS_OUTER), NOSTRILS: polygon(nostrils, True)}
    ix = np.clip(np.round(pixels[:, 0]).astype(int), 0, 799)
    iy = np.clip(np.round(pixels[:, 1]).astype(int), 0, 799)
    features = np.zeros((size, size), np.uint8)
    values = np.zeros(len(rows), np.uint8)
    for bit, layer in layers.items():
        values |= np.where(seen & (layer[iy, ix] > 0), bit, 0).astype(np.uint8)
    features[rows, cols] = values
    # the eye openings (inside the lids' margins, seen from the front): the eyeballs and the lids' inner rims
    openings = np.maximum(polygon(cf.RIGHT_EYE, grow=False), polygon(cf.LEFT_EYE, grow=False))
    front = np.zeros((size, size), bool)
    front[rows, cols] = seen & (openings[iy, ix] > 0)
    # the skin tone's texels: round the cheeks (below the eyes, beside the nose) and on the forehead (above the brows)
    patch = np.zeros((size, size), np.uint8)
    brow_top = min(marks[list(cf.RIGHT_BROW + cf.LEFT_BROW), 1])
    radius = 0.3 * eye * px_per_m
    cheek = np.zeros(len(rows), bool)
    for k in (50, 280):
        cheek |= np.hypot(pixels[:, 0] - marks[k, 0], pixels[:, 1] - marks[k, 1]) < radius
    forehead = (np.hypot(pixels[:, 0] - marks[151, 0], (pixels[:, 1] - marks[151, 1]) * 1.6) < 1.3 * radius) & (
        pixels[:, 1] < brow_top - 0.05 * eye * px_per_m)
    patch[rows, cols] = np.where(seen & (values == 0) & cheek, CHEEKS, np.where(seen & (values == 0) & forehead,
                                                                               FOREHEAD, 0))
    return features, patch, front


def _hair_by_colour(model, head, bake, patch) -> np.ndarray:
    """The fallback for the scalp without the segmenter's labels: texels moved by the head whose colour is far from
    the avatar's skin (its cheeks and forehead), cleaned of specks."""
    from ..outfit.regions import _distance, texture_of

    prim = model.primitives[head.primitive]
    texture = texture_of(getattr(prim, "full_texture", prim.texture))
    size = bake.size
    if texture is None:
        return np.zeros((size, size), bool)
    rgb = texture.rgba[..., :3]
    if rgb.shape[:2] != (size, size):
        rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    lab = light.lab(light.to_linear(cv2.GaussianBlur(rgb, (0, 0), 1.5)))
    skin = patch > 0
    if skin.sum() < 20:
        return np.zeros((size, size), bool)
    reference, _spread, _kept = light.trimmed_median(lab[skin])
    hair = (_distance(lab, reference.astype(np.float32)) > HAIR_DISTANCE) & bake.covered
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    hair = cv2.morphologyEx(hair.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(hair, cv2.MORPH_CLOSE, kernel).astype(bool)


def _near(mask, bake, corners, prim, distance) -> np.ndarray:
    """bool per texel: within ``distance`` metres of ``mask``'s texels on the head (by the vertices: each texel's
    nearest corner)."""
    size = bake.size
    if not mask.any():
        return np.zeros((size, size), bool)
    tri = np.asarray(prim.indices, np.int64).reshape(-1, 3)
    positions = np.asarray(prim.positions, np.float64)
    marked = np.zeros(len(tri), bool)
    marked[np.unique(bake.triangle[mask])] = True
    sources = positions[np.unique(tri[marked])]
    # each vertex's distance to the nearest vertex of the marked triangles (a few thousand by a few thousand)
    d = np.full(len(positions), np.inf)
    for s in range(0, len(positions), 512):
        chunk = positions[s:s + 512]
        d[s:s + 512] = np.sqrt(((chunk[:, None] - sources[None]) ** 2).sum(-1)).min(axis=1)
    near_triangle = (d[tri] < distance).any(axis=1)
    out = np.zeros((size, size), bool)
    covered = bake.covered
    out[covered] = near_triangle[bake.triangle[covered]]
    return out | mask


# -- one photo in the layout -------------------------------------------------------------------------------------


@dataclass
class Projection:
    """One photo seen in the head texture's layout (see the module docs)."""

    name: str
    colour: np.ndarray  # float32 (size, size, 3) linear RGB; 0 where the photo shows nothing
    weight: np.ndarray  # float32 (size, size) 0-1
    face: np.ndarray  # uint8 (size, size): the segmenter's face-skin confidence at each texel's pixel (0-255)
    hair: np.ndarray  # uint8: its hair confidence
    footprint: np.ndarray  # float16 (size, size): photo pixels per texel
    quality: float
    factors: dict = field(default_factory=dict)  # name -> float32 (size, size), with debug=True: the weight's factors
    shading: light.Shading | None = None  # stage 4: the photo's fitted light
    gain: np.ndarray = field(default_factory=lambda: np.ones(3, np.float32))
    warnings: list = field(default_factory=list)

    @property
    def seen(self) -> np.ndarray:
        return self.weight > 0


def sample(image: np.ndarray, points) -> np.ndarray:
    """Bilinear samples of ``image`` (``(H, W)`` or ``(H, W, C)``) at ``points`` (``(N, 2)`` pixels, centres at whole
    numbers; 0 outside)."""
    points = np.asarray(points, np.float32).reshape(-1, 2)
    n = len(points)
    width = 4096
    rows = max(1, math.ceil(n / width))
    padded = np.full((rows * width, 2), -10.0, np.float32)
    padded[:n] = points
    maps = padded.reshape(rows, width, 2)
    out = cv2.remap(image, maps[..., 0].copy(), maps[..., 1].copy(), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0)
    return out.reshape(rows * width, *image.shape[2:])[:n]


def _projection_jacobian(camera: PhotoCamera, points: np.ndarray) -> np.ndarray:
    """``(N, 2, 3)``: how the photo pixel of each point changes as it moves (pixels per metre)."""
    c = camera.to_camera(points)
    z = c[:, 2]
    fx, fy = camera.K[0, 0], camera.K[1, 1]
    j = np.zeros((len(points), 2, 3))
    j[:, 0, 0] = fx / z
    j[:, 0, 2] = -fx * c[:, 0] / (z * z)
    j[:, 1, 1] = fy / z
    j[:, 1, 2] = -fy * c[:, 1] / (z * z)
    return j @ camera.R


def _smooth(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def project(view, atlas: HeadAtlas, *, debug: bool = False) -> Projection:
    """``view`` (a usable :class:`~.register.View`) in ``atlas``'s layout: a :class:`Projection` (see the module
    docs)."""
    size = atlas.size
    camera = view.camera
    target = atlas.photo_parts
    rows, cols = np.nonzero(target)
    points = atlas.bake.position[rows, cols].astype(np.float64)
    normals = atlas.bake.normal[rows, cols].astype(np.float64)
    to_camera = camera.position - points
    facing = np.einsum("nk,nk->n", normals, to_camera) / np.linalg.norm(to_camera, axis=1)
    use = facing > MIN_FACING
    # the depth test: a z-buffer of the head as the camera sees it, cropped to the head
    part = raster.crop(camera, atlas.corners.reshape(-1, 3), 0.05, RASTER_SIZE)
    zbuffer = raster.rasterise(part.camera, atlas.corners)
    near = zbuffer.nearest(1)
    idx = np.nonzero(use)[0]
    use[idx] = _depth_test(part.camera, near, points[idx], facing[idx])
    idx = np.nonzero(use)[0]
    rows, cols, points, facing = rows[idx], cols[idx], points[idx], facing[idx]
    pixels = view.photo_pixels(points)
    w_img, h_img = view.size
    inside = (pixels[:, 0] >= 0) & (pixels[:, 0] <= w_img - 1) & (pixels[:, 1] >= 0) & (pixels[:, 1] <= h_img - 1)
    rows, cols, points, facing, pixels = rows[inside], cols[inside], points[inside], facing[inside], pixels[inside]
    outline = _away_from_outlines(zbuffer, part.camera, points, _face_size(view) * part.factor)
    # the texels' size in the photo
    tri = atlas.bake.triangle[rows, cols]
    j = _projection_jacobian(camera, points) @ atlas.jacobian[tri]  # (n, 2, 2) photo pixels per texel
    a = (j[:, 0, 0] ** 2 + j[:, 1, 0] ** 2)
    b = (j[:, 0, 0] * j[:, 0, 1] + j[:, 1, 0] * j[:, 1, 1])
    c = (j[:, 0, 1] ** 2 + j[:, 1, 1] ** 2)
    largest = np.sqrt(0.5 * (a + c) + np.sqrt(0.25 * (a - c) ** 2 + b * b))  # the longer axis of the footprint
    area = np.abs(j[:, 0, 0] * j[:, 1, 1] - j[:, 0, 1] * j[:, 1, 0])
    colour = _read_photo(view, pixels, largest)
    # the segmentation: class confidences (of this person only), distances from the masks' edges
    classes, edges = _classes(view, atlas, pixels, rows, cols)
    part_of = atlas.parts[rows, cols]
    hairline = atlas.hairline[rows, cols]
    allowed = np.where(part_of == NECK, classes["face"] + classes["body"],
                       np.where(part_of == SCALP, classes["hair"] + OTHERS_ON_SCALP * classes["others"],
                                classes["face"] + np.where(hairline, classes["hair"], 0.0)))
    edge = np.where(part_of == NECK, edges["neck"], np.where(part_of == SCALP, edges["scalp"],
                                                             np.where(hairline, edges["hairline"], edges["face"])))
    face_px = _face_size(view)
    border = np.minimum.reduce([pixels[:, 0], w_img - 1 - pixels[:, 0], pixels[:, 1], h_img - 1 - pixels[:, 1]])
    edge = edge * _smooth(border / max(BORDER_RAMP * face_px, 1.0)) * outline
    angle = np.power(np.clip(facing, 0.0, 1.0), ANGLE_POWER)
    squarely = np.sqrt(area / np.maximum(facing, 0.05))  # photo pixels per texel, as if seen squarely
    # (constant per triangle, as the layout's scale is: smoothed, so the weights show no facets)
    grid = np.zeros((size, size), np.float32)
    grid[rows, cols] = squarely
    seen = np.zeros((size, size), bool)
    seen[rows, cols] = True
    squarely = masked_soften(grid, seen, RESOLUTION_SMOOTHING * size / 1024)[rows, cols]
    marked = squarely * (size / (atlas.marks_size or size))  # per texel of the head's ordinary texture
    resolution = np.sqrt(np.clip(marked / GOOD_FOOTPRINT, 0.0, 1.0))
    weight = angle * np.clip(allowed, 0.0, 1.0) * edge * resolution * max(view.quality, 0.05)
    shapes = view.marks.blendshapes if view.marks is not None else {}
    warnings = []
    if (shapes or {}).get("jawOpen", 0.0) > MOUTH_OPEN:
        lips = (atlas.features[rows, cols] & LIPS) > 0
        weight[lips] *= 0.1
        warnings.append(f"{view.name}: the mouth is open: its lips count a tenth")
    out = Projection(view.name, np.zeros((size, size, 3), np.float32), np.zeros((size, size), np.float32),
                     np.zeros((size, size), np.uint8), np.zeros((size, size), np.uint8),
                     np.zeros((size, size), np.float16), float(view.quality), warnings=warnings)
    out.colour[rows, cols] = colour
    out.weight[rows, cols] = weight
    out.face[rows, cols] = np.clip(classes["face"] * 255.0 + 0.5, 0, 255).astype(np.uint8)
    out.hair[rows, cols] = np.clip(classes["hair"] * 255.0 + 0.5, 0, 255).astype(np.uint8)
    out.footprint[rows, cols] = squarely.astype(np.float16)
    if debug:
        for name, values in (("angle", angle), ("class", allowed), ("edge", edge), ("resolution", resolution)):
            grid = np.zeros((size, size), np.float32)
            grid[rows, cols] = values
            out.factors[name] = grid
    return out


def _away_from_outlines(zbuffer, camera: PhotoCamera, points, face_px: float) -> np.ndarray:
    """0-1 per point: how far its pixel is from an outline of the head as ``camera`` sees it (the silhouette and
    where one part hides another: the jaw over the neck, an ear, the side of the nose), rising over
    :data:`OUTLINE_RAMP` of the face (``face_px``: its size in ``camera``'s pixels). Just behind an outline a texel
    can pass the depth test and still read what lies beyond the outline in the photo (the neck below the jaw)."""
    depth = zbuffer.depth
    finite = np.isfinite(depth)
    d = np.where(finite, depth, np.float32(1e6)).astype(np.float32)
    kernel = np.ones((3, 3), np.uint8)
    jump = (cv2.dilate(d, kernel) - cv2.erode(d, kernel)) > DEPTH_EDGE
    distance = cv2.distanceTransform((~jump).astype(np.uint8), cv2.DIST_L2, 3)
    h, w = depth.shape
    pixels = camera.project(points)
    ix = np.clip(np.round(pixels[:, 0]).astype(np.int64), 0, w - 1)
    iy = np.clip(np.round(pixels[:, 1]).astype(np.int64), 0, h - 1)
    return _smooth(distance[iy, ix] / max(OUTLINE_RAMP * face_px, 1.0)).astype(np.float64)


def _face_size(view) -> float:
    """The face's size in the photo (pixels): from its landmarks, else from the avatar's projected ones."""
    from .register import face_size

    if view.marks is not None:
        return face_size(view.marks.points)
    if view.projected is not None:
        return face_size(view.projected)
    return 0.25 * max(view.size)


def _read_photo(view, pixels, largest) -> np.ndarray:
    """The photo's colours (linear RGB) at ``pixels``, from a mipmapped copy cropped to them, at the level where a
    texel (``largest``: its longer axis, photo pixels) is about one pixel."""
    image = view.image
    h, w = image.shape[:2]
    margin = 8
    x0, y0 = max(0, int(pixels[:, 0].min()) - margin), max(0, int(pixels[:, 1].min()) - margin)
    x1, y1 = min(w, int(pixels[:, 0].max()) + margin + 2), min(h, int(pixels[:, 1].max()) + margin + 2)
    crop = image[y0:y1, x0:x1]
    if crop.ndim == 2:
        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    scale = min(1.0, 1.25 / max(float(np.median(largest)), 1e-6))  # no more detail than the texels hold
    if scale < 0.95:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        scale = 1.0
    level0 = light.to_linear(crop[..., ::-1])  # RGB, linear
    points = (pixels - (x0, y0) + 0.5) * scale - 0.5
    size = np.maximum(largest * scale, 1.0)
    levels = int(np.clip(np.ceil(np.log2(size.max())), 0, 5))
    pyramid = [level0]
    for _ in range(levels):
        if min(pyramid[-1].shape[:2]) < 4:
            break
        pyramid.append(cv2.pyrDown(pyramid[-1]))
    level = np.clip(np.log2(size), 0.0, len(pyramid) - 1)
    low = np.floor(level).astype(np.int64)
    frac = (level - low).astype(np.float32)
    out = np.zeros((len(points), 3), np.float32)
    for k, img in enumerate(pyramid):
        at = (low == k) | ((low + 1 == k) & (frac > 0))
        if not at.any():
            continue
        values = sample(img, points[at] / (2 ** k))  # pyrDown: pixel i of a level is pixel 2i of the one below
        f = np.where(low[at] == k, 1.0 - frac[at], frac[at])[:, None]
        out[at] += values * f
    return out


def _classes(view, atlas, pixels, rows, cols) -> tuple[dict, dict]:
    """The segmenter's confidences at ``pixels`` (``"face"``, ``"hair"``, ``"body"``, ``"others"``), of the person
    the camera was fitted to only, and the distances from the edges of the masks each part uses (``"face"``,
    ``"scalp"``, ``"neck"``), as 0-1 ramps over :data:`EDGE_RAMP` of the face."""
    from .detect import mask_at

    names = ("face", "hair", "body", "others")
    w_img, h_img = view.size
    if view.masks and "face" in view.masks:
        shape = view.masks["face"].shape[:2]
        mh, mw = shape
        masks = {n: (mask_at(view.masks[n], (mw, mh)) if n in view.masks else np.zeros(shape, np.float32))
                 for n in names}
    else:  # no segmentation: the face oval of the landmarks is the face skin, nothing else counts
        scale = min(1.0, 1024 / max(w_img, h_img))
        mw, mh = max(1, round(w_img * scale)), max(1, round(h_img * scale))
        oval = np.zeros((mh, mw), np.float32)
        points = view.marks.points if view.marks is not None else view.projected
        if points is not None:
            polygon = (points[list(cf.FACE_OVAL)] + 0.5) * scale - 0.5
            cv2.fillPoly(oval, [np.round(polygon).astype(np.int32)], 1.0)
        masks = {n: np.zeros((mh, mw), np.float32) for n in names}
        masks["face"] = oval
    person = _person(view, masks, (mw, mh))
    at = (pixels + 0.5) * np.array([mw / w_img, mh / h_img]) - 0.5
    classes = {n: sample(masks[n] * person, at).astype(np.float64) for n in names}
    ramp = max(EDGE_RAMP * _face_size(view) * mw / w_img, 1.0)
    edges = {}
    for name, members in (("face", ("face",)), ("hairline", ("face", "hair")), ("scalp", ("hair", "others")),
                          ("neck", ("face", "body"))):
        region = (sum(masks[n] for n in members) * person > MASK_LEVEL).astype(np.uint8)
        distance = cv2.distanceTransform(region, cv2.DIST_L2, 3)
        edges[name] = _smooth(sample(distance, at) / ramp).astype(np.float64)
    return classes, edges


def _person(view, masks, size) -> np.ndarray:
    """1 on the connected parts of the photo's head and body masks that the avatar's head covers (the person the
    camera was fitted to), 0 elsewhere (other people), at the masks' ``size``."""
    mw, mh = size
    w_img, h_img = view.size
    union = (masks["face"] + masks["hair"] + masks["body"] + masks["others"]) > MASK_LEVEL
    count, labels = cv2.connectedComponents(union.astype(np.uint8), connectivity=8)
    if count <= 2:
        return union.astype(np.float32) if count == 2 else np.zeros((mh, mw), np.float32)
    points = view.marks.points if view.marks is not None else view.projected
    if points is None:
        return union.astype(np.float32)
    at = np.round((points + 0.5) * np.array([mw / w_img, mh / h_img]) - 0.5).astype(int)
    at[:, 0] = np.clip(at[:, 0], 0, mw - 1)
    at[:, 1] = np.clip(at[:, 1], 0, mh - 1)
    found = np.unique(labels[at[:, 1], at[:, 0]])
    found = found[found > 0]
    return np.isin(labels, found).astype(np.float32) if len(found) else union.astype(np.float32)


# -- gains, the blend, padding -------------------------------------------------------------------------------------


def gains(projections, mask: np.ndarray, anchor: int | None = None) -> np.ndarray:
    """``(V, 3)`` gains per photo and channel that make the photos agree on their overlaps on ``mask`` (the face
    skin): Brown and Lowe's least squares (see the module docs). ``anchor``: the photo whose level the others
    follow (held near a gain of 1, the others nearly free); without one every gain is pulled towards 1 alike."""
    v = len(projections)
    spread = np.full(v, SIGMA_G)
    if anchor is not None:
        spread[:] = SIGMA_G_FREE
        spread[anchor] = SIGMA_G_ANCHOR
    out = np.ones((v, 3))
    if v < 2:
        return out.astype(np.float32)
    n = np.zeros((v, v))
    mean = np.zeros((v, v, 3))  # mean[i, j]: photo i's mean colour where it overlaps photo j
    texels = np.flatnonzero(mask)
    weights = np.stack([p.weight.ravel()[texels] for p in projections]).astype(np.float64)
    colours = np.stack([p.colour.reshape(-1, 3)[texels] for p in projections]).astype(np.float64)
    for i in range(v):
        for j in range(i + 1, v):
            m = np.minimum(weights[i], weights[j])
            total = float(m.sum())
            if total < MIN_OVERLAP:
                continue
            n[i, j] = n[j, i] = total
            mean[i, j] = m @ colours[i] / total
            mean[j, i] = m @ colours[j] / total
    if not n.any():
        return out.astype(np.float32)
    mean /= max(float(mean[n > 0].mean()), 1e-6)  # relative to the overlaps' mean colour: SIGMA_N is relative
    for ch in range(3):
        a = np.zeros((v, v))
        b = np.zeros(v)
        for i in range(v):
            for j in range(v):
                if i == j or n[i, j] == 0:
                    continue
                a[i, i] += n[i, j] * (mean[i, j, ch] ** 2 / SIGMA_N ** 2 + 1.0 / spread[i] ** 2)
                a[i, j] -= n[i, j] * mean[i, j, ch] * mean[j, i, ch] / SIGMA_N ** 2
                b[i] += n[i, j] / spread[i] ** 2
            a[i, i] += 1e-6
            b[i] += 1e-6
        out[:, ch] = np.linalg.solve(a, b)
    return np.clip(out, 0.25, 4.0).astype(np.float32)


def push_pull(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """``image`` (``(H, W)`` or ``(H, W, C)`` floats) with the pixels outside ``mask`` filled smoothly from those
    inside: averaged down a pyramid ("pull"), then filled back up from the coarser levels ("push")."""
    image = np.asarray(image, np.float32)
    inside = np.asarray(mask, bool)
    if inside.all() or not inside.any():
        return image.copy()
    expand = image.ndim == 2
    values = image[..., None] if expand else image
    weight = inside.astype(np.float32)
    levels = [(np.where(inside[..., None], values, np.float32(0.0)), weight)]
    while min(levels[-1][1].shape) > 1:
        v, w = levels[-1]
        h2, w2 = max(1, (w.shape[0] + 1) // 2), max(1, (w.shape[1] + 1) // 2)
        levels.append((_resize(v, (w2, h2), cv2.INTER_AREA), cv2.resize(w, (w2, h2), interpolation=cv2.INTER_AREA)))
    v, w = levels[-1]
    filled = v / np.maximum(w, 1e-12)[..., None]
    for v, w in reversed(levels[1:-1]):  # the coarser levels: partly covered cells mix their own and the push
        up = _resize(filled, (w.shape[1], w.shape[0]), cv2.INTER_LINEAR)
        a = np.minimum(w, 1.0)[..., None]
        filled = v / np.maximum(w, 1e-12)[..., None] * a + up * (1.0 - a)
    # the finest level: its own pixels inside the mask, the push outside
    up = _resize(filled, (inside.shape[1], inside.shape[0]), cv2.INTER_LINEAR)
    out = np.where(inside[..., None], values, up)
    return out[..., 0] if expand else out


def _resize(values: np.ndarray, size, interpolation) -> np.ndarray:
    """``cv2.resize`` of an ``(H, W, C)`` array with any number of channels (OpenCV's area averaging takes at most
    four at a time), keeping the channel axis."""
    channels = values.shape[2]
    parts = [cv2.resize(np.ascontiguousarray(values[..., k:k + 4]), size, interpolation=interpolation)
             for k in range(0, channels, 4)]
    parts = [p[..., None] if p.ndim == 2 else p for p in parts]
    return parts[0] if len(parts) == 1 else np.concatenate(parts, axis=2)


def bands_for(size: int) -> int:
    """The multi-band blend's levels for a texture of ``size`` texels: :data:`BANDS` at 1024, one more per
    doubling."""
    return max(2, BANDS + round(math.log2(max(size, 16) / 1024.0)))


def blend(sources, weights, islands: np.ndarray, levels: int) -> tuple[np.ndarray, np.ndarray]:
    """The multi-band blend of ``sources`` (``(H, W, 3)`` each) with ``weights`` (``(H, W)`` each, 0 where a source
    has no colour) island by island (``islands``: ``(H, W)`` ids, -1 outside): ``(blended, share)``, where
    ``share`` (``(S, H, W)``) is each source's weight in the finest band. See the module docs."""
    h, w = islands.shape
    out = np.zeros((h, w, 3), np.float32)
    share = np.zeros((len(sources), h, w), np.float32)
    margin = 2 ** levels
    for island in np.unique(islands[islands >= 0]):
        inside = islands == island
        ys, xs = np.nonzero(inside)
        y0, y1 = max(0, ys.min() - margin), min(h, ys.max() + margin + 1)
        x0, x1 = max(0, xs.min() - margin), min(w, xs.max() + margin + 1)
        box = (slice(y0, y1), slice(x0, x1))
        m = inside[box]
        active = [k for k in range(len(sources)) if (weights[k][box][m] > 0).any()]
        if not active:
            continue
        region = out[box]
        if len(active) == 1:  # one source (the avatar's own, on the eyes or a strand of hair no photo shows)
            region[m] = sources[active[0]][box][m]
            share[active[0]][box][m] = 1.0
            continue
        stack = np.stack([np.where(m, weights[k][box], 0.0) for k in active]).astype(np.float32)
        top = stack.max(axis=0)
        powered = (stack / np.maximum(top, 1e-12)) ** WINNER_POWER
        total = powered.sum(axis=0)
        winners = powered / np.maximum(total, 1e-12)
        covered = total > 0
        # every source filled from its own texels, the weights from the island's: no other island's colour
        filled = [push_pull(sources[k][box], stack[i] > 0) for i, k in enumerate(active)]
        win = push_pull(np.moveaxis(winners, 0, -1), covered)
        win /= np.maximum(win.sum(axis=-1, keepdims=True), 1e-6)
        depth = max(1, min(levels, int(math.log2(max(2, min(y1 - y0, x1 - x0))))))
        result = _multiband(filled, [win[..., i] for i in range(len(active))], depth)
        region[m] = result[m]
        for i, k in enumerate(active):
            share[k][box][m] = winners[i][m]
    return out, share


def _multiband(images, weights, levels: int) -> np.ndarray:
    """Burt and Adelson's blend of ``images`` with ``weights`` (summing to 1 everywhere) over ``levels`` bands."""
    gaussians = []
    laplacians = []
    for image in images:
        g = [image]
        for _ in range(levels - 1):
            g.append(cv2.pyrDown(g[-1]))
        lap = [g[k] - cv2.pyrUp(g[k + 1], dstsize=(g[k].shape[1], g[k].shape[0])) for k in range(levels - 1)]
        lap.append(g[-1])
        laplacians.append(lap)
    for weight in weights:
        g = [weight]
        for _ in range(levels - 1):
            g.append(cv2.pyrDown(g[-1]))
        gaussians.append(g)
    result = None
    for k in reversed(range(levels)):
        total = sum(gaussians[i][k] for i in range(len(images)))
        band = sum(laplacians[i][k] * gaussians[i][k][..., None] for i in range(len(images))) / np.maximum(
            total, 1e-6)[..., None]
        if result is None:
            result = band
        else:
            result = cv2.pyrUp(result, dstsize=(band.shape[1], band.shape[0])) + band
    return result


def masked_soften(values: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    """``values`` blurred by a Gaussian (``sigma`` texels) over ``mask`` only (normalised convolution), so nothing
    from outside the islands blurs in; unchanged outside ``mask``."""
    if sigma <= 0:
        return values
    m = mask.astype(np.float32)
    norm = cv2.GaussianBlur(m, (0, 0), sigma)
    v = values * (m[..., None] if values.ndim == 3 else m)
    blurred = cv2.GaussianBlur(v, (0, 0), sigma)
    blurred = blurred / np.maximum(norm, 1e-6)[..., None] if values.ndim == 3 else blurred / np.maximum(norm, 1e-6)
    return np.where(mask[..., None] if values.ndim == 3 else mask, blurred, values).astype(np.float32)


def pad(rgba: np.ndarray, covered: np.ndarray) -> np.ndarray:
    """``rgba`` (``uint8``) with the texels outside ``covered`` filled from the islands: first each island grows by
    its own border colours (:data:`PAD_DILATE` texels at 1024), then push-pull fills the rest."""
    size = covered.shape[0]
    values = rgba.astype(np.float32)
    mask = covered.copy()
    grow = max(1, round(PAD_DILATE * size / 1024))
    kernel = np.ones((3, 3), np.float32)
    for _ in range(grow):  # each texel next to the islands: the mean of its covered neighbours
        count = cv2.filter2D(mask.astype(np.float32), -1, kernel, borderType=cv2.BORDER_CONSTANT)
        rim = (count > 0) & ~mask
        if not rim.any():
            break
        total = cv2.filter2D(values * mask[..., None], -1, kernel, borderType=cv2.BORDER_CONSTANT)
        values[rim] = total[rim] / count[rim, None]
        mask |= rim
    values = push_pull(values, mask)
    out = np.clip(values + 0.5, 0, 255).astype(np.uint8)
    out[covered] = rgba[covered]
    return out


def confidence(projections, atlas: HeadAtlas) -> np.ndarray:
    """float32 0-1 per texel: how surely the photos show it (see the module docs); 0 on the eyes and inner parts."""
    total = sum(p.weight for p in projections) if projections else np.zeros(atlas.parts.shape, np.float32)
    out = (1.0 - np.exp(-np.asarray(total, np.float32) / CONFIDENCE_SCALE)).astype(np.float32)
    out[~atlas.photo_parts] = 0.0
    return out


def trimap(conf: np.ndarray, atlas: HeadAtlas) -> np.ndarray:
    """uint8 per texel: :data:`KEEP`, :data:`REFINE` or :data:`GENERATE` for stage 5 (see the module docs),
    :data:`TRIMAP_OUTSIDE` outside the islands."""
    out = np.full(conf.shape, TRIMAP_OUTSIDE, np.uint8)
    covered = atlas.covered
    out[covered] = np.where(conf[covered] >= KEEP_AT, KEEP, np.where(conf[covered] >= REFINE_AT, REFINE, GENERATE))
    out[(atlas.parts == EYE) | (atlas.parts == INNER)] = KEEP
    return out
