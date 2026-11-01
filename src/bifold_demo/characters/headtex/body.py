"""Stage 6: the body's skin matched to the face, the neck seam levelled, the hair recoloured.

The head texture now shows the person; the body texture (a separate atlas: hands, neck, arms or legs, and the clothes)
still shows the avatar's skin, and the two meet at the neck. :func:`match_body` makes them one person:

1. **The body's skin** (the outfits' skin mask, :func:`~..outfit.regions.body_regions`: the hands, the neck and
   the neckline, bare arms and legs) is given the person's skin tone, the face's skin target of stage 4
   (:attr:`~.texture.HeadTexture.skin`). The transfer is a mean transfer in linear light: one gain per channel takes
   the robust median of the body's skin (its hands where it has enough of them: they are what is seen next to the
   face) to the target. A gain keeps every ratio between texels, so the body's painted detail and its soft light
   (the creases of the knuckles, the shadow under the collar) stay as they were, also for a target much darker or
   lighter than the avatar. (A full covariance transfer, MKL, would also map the spread of the body's colours onto the
   spread of the cheeks and forehead, which is far smaller, and flatten the painted shading.) The mask is softened
   by a texel or two; the clothes stay as they are.
2. **The neck seam.** Where the head mesh's open edge meets the body (:func:`~.texture.seam_pairs`) and the body
   texture is skin there, the two textures are compared pair by pair (the logarithm of each channel, in linear
   light). A smooth offset round the neck is fitted to those differences by least squares (:data:`SECTORS` sectors,
   neighbours held together, sectors without pairs led to the median offset) and spread into a band on both sides,
   half each: the head's face and neck texels within :data:`HEAD_BAND` of the seam, the body's skin within
   :data:`BODY_BAND`, fading to nothing at the band's end. On most Rocketbox avatars a collar covers most of the seam;
   there the offset only touches what the collar hides.
3. **The hair** (optional): when the photos showed the person's hair on the scalp (:attr:`~.texture.HeadTexture.hair`)
   and it differs from the avatar's, the hair cards (the ``*_opacity`` or ``*hair*`` primitives: strands, buns,
   fringes) and the scalp texels no photo showed are recoloured towards it in Lab: its lightness and chroma, the
   strands' own lightness detail (scaled with the lightness), on the texels whose colour is like the avatar's hair
   (the eyelashes on the same texture stay).

The result, a :class:`BodyMatch`, holds the head texture (levelled at the neck), the body texture (the base for
outfits: :meth:`~..outfit.tailor.Tailor.texture` builds on any body texture), the recoloured hair textures, and what
was done.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from . import fuse, light

GAIN_LIMITS = (0.2, 5.0)  # the body skin's gain per channel (linear light)
MIN_HANDS = 200  # texels: with fewer skin texels on the hands, all the body's skin gives its tone
NECK_FADE = 0.1  # metres beyond texture.NECK_BAND over which the neck's gain gives way to the face's
SKIN_FEATHER = 1.2  # texels (at 1024): the skin mask's softening
SECTORS = 24  # round the neck: the seam's offset is fitted per sector
SMOOTHNESS = 1.0  # how strongly neighbouring sectors are held together (relative to a sector's mean pair count)
TO_MEDIAN = 0.05  # ... and sectors to the median offset (where they have no pairs)
SEAM_OUTLIER = 12.0  # Lab delta E: seam pairs further apart than this are not compared (a painted line, a shadow)
HEAD_BAND = 0.04  # metres from the seam: the head's texels the offset fades out over
BODY_BAND = 0.06  # ... and the body's skin
HAIR_PRIMITIVES = ("opacity", "hair")  # name parts of the primitives with hair cards
HAIR_DIFFERENT = 6.0  # Lab delta E: hair closer to the avatar's than this is left alone
HAIR_LIKE = (20.0, 40.0)  # Lab distance from the avatar's hair colour (lightness counting half): fully, then not hair
HAIR_NOT_SKIN = 8.0  # ... and how much nearer to the hair than to the avatar's skin a texel must be to count fully
HAIR_LIGHTNESS = (0.4, 1.3)  # the scaling of the strands' lightness detail


@dataclass
class BodyMatch:
    """What stage 6 made (see the module docs). Textures are RGBA ``uint8`` images (row 0 at the top), by primitive
    name in :meth:`textures`."""

    head: np.ndarray  # the head texture, levelled at the neck
    body: np.ndarray | None  # the body texture with the person's skin (None: the avatar has no body texture)
    hair: dict = field(default_factory=dict)  # primitive name -> recoloured hair texture
    head_primitive: str = ""
    body_primitive: str = ""
    gain: np.ndarray = field(default_factory=lambda: np.ones(3, np.float32))  # the body skin's, per channel
    neck_gain: np.ndarray | None = None  # ... by the neck (to the head's neck colour), when there is skin there
    seam: dict = field(default_factory=dict)  # "pairs" (skin pairs levelled), "offset" (median, Lab delta E)
    warnings: list = field(default_factory=list)

    def textures(self) -> dict[str, np.ndarray]:
        out = {self.head_primitive: self.head}
        if self.body is not None:
            out[self.body_primitive] = self.body
        out.update(self.hair)
        return out


def match_body(result, model, head_rgba=None, *, hair: bool = True) -> BodyMatch:
    """Stage 6 for ``result`` (a :class:`~.texture.HeadTexture` of ``model``'s head): see the module docs.
    ``head_rgba``: the head texture to level (default ``result.rgba``; stage 5's completion); ``hair``: recolour
    the hair too."""
    from ..outfit.regions import GROUPS, body_regions, texture_of

    head = np.array(result.rgba if head_rgba is None else head_rgba, np.uint8, copy=True)
    match = BodyMatch(head, None, head_primitive=result.primitive)
    try:
        regions = body_regions(model)
    except ValueError as exc:  # no body primitive
        match.warnings.append(f"the body's skin is not matched: {exc}")
        return match
    prim = model.primitives[regions.primitive]
    match.body_primitive = prim.name
    texture = texture_of(prim.texture)
    size = regions.size
    if texture is None:
        match.warnings.append("the body has no texture: its skin is not matched")
        return match
    body = texture.rgba
    if body.shape[:2] != (size, size):
        body = cv2.resize(body, (size, size), interpolation=cv2.INTER_AREA if body.shape[0] > size
                          else cv2.INTER_LINEAR)
    body = np.array(body, np.uint8, copy=True)
    skin = regions.skin
    linear = light.to_linear(body[..., :3])
    if result.skin is None or skin.sum() < 20:
        match.warnings.append("no skin tone from the photos (or no skin on the body): the body keeps its own")
    else:
        hands = skin & (regions.groups == GROUPS.index("hands"))
        source_texels = hands if hands.sum() >= MIN_HANDS else skin
        source = _median(linear[source_texels])
        gain = np.clip(result.skin.linear / np.maximum(source, 1e-4), *GAIN_LIMITS).astype(np.float32)
        match.gain = gain
        log_gain = np.broadcast_to(np.log(gain), (size, size, 3)).copy()
        near = _near_neck(match.head, linear, model, regions, result.atlas)
        if near is not None:  # by the neck: the head's neck colour
            weight, neck_gain = near
            log_gain += weight[..., None] * (np.log(neck_gain) - np.log(gain))
            match.neck_gain = neck_gain
        soft = cv2.GaussianBlur(skin.astype(np.float32), (0, 0), SKIN_FEATHER * size / 1024)
        linear = linear * np.exp(log_gain * soft[..., None])
    body_rgb = light.to_srgb(linear)
    match.body = np.dstack([body_rgb, body[..., 3]])
    _level_seam(match, model, regions, result.atlas)
    if hair and result.hair is not None:
        _recolour_hair(match, model, result)
    if result.atlas is not None and match.head.shape[0] == result.atlas.size:
        match.head = fuse.pad(match.head, result.atlas.covered)  # the gutters again, from the changed texels
    return match


def hands_tone(model, body_rgba=None) -> np.ndarray | None:
    """Lab of the skin of ``model``'s hands on ``body_rgba`` (default: its own body texture), a trimmed median (all
    the body's skin when the hands show little); ``None`` without a body texture or skin."""
    from ..outfit.regions import GROUPS, body_regions, texture_of

    try:
        regions = body_regions(model)
    except ValueError:
        return None
    if body_rgba is None:
        texture = texture_of(model.primitives[regions.primitive].texture)
        if texture is None:
            return None
        body_rgba = texture.rgba
    size = regions.size
    if body_rgba.shape[0] != size:
        body_rgba = cv2.resize(np.ascontiguousarray(body_rgba), (size, size), interpolation=cv2.INTER_AREA)
    hands = regions.skin & (regions.groups == GROUPS.index("hands"))
    texels = hands if hands.sum() >= MIN_HANDS else regions.skin
    if texels.sum() < 20:
        return None
    return light.trimmed_median(light.lab(light.to_linear(body_rgba[..., :3][texels])))[0]


def _median(linear: np.ndarray) -> np.ndarray:
    """The robust (trimmed Lab) median of linear RGB colours ``(N, 3)``, as linear RGB."""
    lab, _spread, _kept = light.trimmed_median(light.lab(linear))
    return light.from_lab(lab[None])[0]


def _near_neck(head_rgba, body_linear, model, regions, atlas):
    """``(weight, gain)``: how much each body texel is "by the neck" (1 within :data:`~.texture.NECK_BAND` of the
    seam, fading to 0 over :data:`NECK_FADE` more) and the gain that gives the body's skin there (``body_linear``:
    the body texture, linear RGB) the median colour of the head's neck; ``None`` without a seam, a neck part or skin
    by the neck."""
    from .texture import NECK_BAND, distance_to, seam_pairs

    if atlas is None or head_rgba.shape[0] != atlas.size:
        return None
    pairs = seam_pairs(model)
    if pairs is None:
        return None
    neck = atlas.parts == fuse.NECK
    rows, cols = np.nonzero(regions.skin)
    d = distance_to(regions.bake.position[rows, cols], pairs.points, NECK_BAND + NECK_FADE)
    close = d < NECK_BAND
    if neck.sum() < 20 or close.sum() < 20:
        return None
    gain = np.clip(_median(light.to_linear(head_rgba[..., :3][neck])) / np.maximum(
        _median(body_linear[rows[close], cols[close]]), 1e-4), *GAIN_LIMITS).astype(np.float32)
    weight = np.zeros(regions.skin.shape, np.float32)
    fade = np.clip((NECK_BAND + NECK_FADE - d) / NECK_FADE, 0.0, 1.0)
    weight[rows, cols] = fade * fade * (3.0 - 2.0 * fade)
    return weight, gain


# -- the neck seam ----------------------------------------------------------------------------------------------


def sector_offsets(angles: np.ndarray, offsets: np.ndarray, sectors: int = SECTORS) -> np.ndarray:
    """``(sectors, C)``: a smooth offset per sector round a ring fitted to ``offsets`` (``(N, C)``) at ``angles``
    (radians): least squares with neighbouring sectors held together (:data:`SMOOTHNESS`) and every sector held a
    little to the median offset (:data:`TO_MEDIAN`), which is all a sector without data gets."""
    angles, offsets = np.asarray(angles, np.float64), np.asarray(offsets, np.float64).reshape(len(angles), -1)
    sector = np.floor((angles % (2.0 * np.pi)) / (2.0 * np.pi) * sectors).astype(int) % sectors
    count = np.bincount(sector, minlength=sectors).astype(np.float64)
    total = np.zeros((sectors, offsets.shape[1]))
    np.add.at(total, sector, offsets)
    median = np.median(offsets, axis=0)
    scale = max(float(count[count > 0].mean()), 1.0) if count.any() else 1.0
    a = np.diag(count + TO_MEDIAN * scale)
    for k in range(sectors):
        for j in ((k - 1) % sectors, (k + 1) % sectors):
            a[k, k] += SMOOTHNESS * scale
            a[k, j] -= SMOOTHNESS * scale
    b = total + TO_MEDIAN * scale * median
    return np.linalg.solve(a, b)


def at_angles(table: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """``(N, C)``: the per-sector ``table`` (``(K, C)``, sector ``k`` centred at ``(k + 0.5) / K`` of a turn) read
    at ``angles``, linearly between the sectors' centres, round the ring."""
    k = len(table)
    position = (np.asarray(angles, np.float64) % (2.0 * np.pi)) / (2.0 * np.pi) * k - 0.5
    low = np.floor(position).astype(int)
    frac = (position - low)[:, None]
    return table[low % k] * (1.0 - frac) + table[(low + 1) % k] * frac


def _level_seam(match: BodyMatch, model, regions, atlas) -> None:
    """Level the neck seam between ``match.head`` and ``match.body`` (in place; see the module docs)."""
    from .texture import distance_to, seam_pairs

    pairs = seam_pairs(model)
    match.seam = {"pairs": 0, "offset": 0.0}
    if pairs is None or not len(pairs.head_uv) or atlas is None:
        return
    size = regions.size
    br, bc = pairs.texels("body", size)
    skin = regions.skin[br, bc]
    if skin.sum() < 3:
        return  # the collar covers the seam: nothing of it shows
    hr, hc = pairs.texels("head", match.head.shape[0])
    head_lin = light.to_linear(cv2.blur(np.ascontiguousarray(match.head[..., :3]), (3, 3)))[hr, hc][skin]
    body_lin = light.to_linear(cv2.blur(np.ascontiguousarray(match.body[..., :3]), (3, 3)))[br, bc][skin]
    apart = light.delta_e(light.lab(head_lin), light.lab(body_lin))
    like = apart < SEAM_OUTLIER  # (not a dark painted line at the neckline, which the skin mask may include)
    if like.sum() < 3:
        return
    difference = np.log(np.maximum(head_lin[like], 1e-3)) - np.log(np.maximum(body_lin[like], 1e-3))
    centre = pairs.points.mean(axis=0)
    close_points = pairs.points[pairs.close][skin][like]
    angles = np.arctan2(close_points[:, 0] - centre[0], close_points[:, 2] - centre[2])
    table = sector_offsets(angles, difference)
    match.seam = {"pairs": int(like.sum()), "offset": float(np.median(apart[like]))}
    ring = pairs.points

    def spread(image, mask, positions, band, sign):
        rows, cols = np.nonzero(mask)
        points = positions[rows, cols].astype(np.float64)
        d = distance_to(points, ring, band)
        near = d < band
        rows, cols, points, d = rows[near], cols[near], points[near], d[near]
        if not len(rows):
            return
        fade = 1.0 - d / band
        fade = (fade * fade * (3.0 - 2.0 * fade))[:, None]
        theta = np.arctan2(points[:, 0] - centre[0], points[:, 2] - centre[2])
        factor = np.exp(sign * 0.5 * at_angles(table, theta) * fade)
        image[rows, cols, :3] = light.to_srgb(light.to_linear(image[rows, cols, :3]) * factor)

    head_mask = (atlas.parts == fuse.FACE) | (atlas.parts == fuse.NECK)
    if match.head.shape[0] == atlas.size:
        spread(match.head, head_mask, atlas.bake.position, HEAD_BAND, -1.0)
    spread(match.body, regions.skin, regions.bake.position, BODY_BAND, 1.0)


# -- the hair ---------------------------------------------------------------------------------------------------


def recolour(rgb: np.ndarray, source_lab, target_lab, weight: np.ndarray) -> np.ndarray:
    """``rgb`` (``uint8`` sRGB, ``(N, 3)``) moved from the colour ``source_lab`` towards ``target_lab`` (Lab), by
    ``weight`` (``(N,)`` 0-1): the target's lightness and chroma, each texel's lightness difference from the source
    scaled with the lightness (within :data:`HAIR_LIGHTNESS`), its chroma difference kept."""
    lab = light.lab(light.to_linear(rgb))
    source, target = np.asarray(source_lab, np.float32), np.asarray(target_lab, np.float32)
    scale = float(np.clip(target[0] / max(float(source[0]), 1.0), *HAIR_LIGHTNESS))
    new = lab.copy()
    new[:, 0] = target[0] + (lab[:, 0] - source[0]) * scale
    new[:, 1:] = lab[:, 1:] + (target[1:] - source[1:])
    new[:, 0] = np.clip(new[:, 0], 0.0, 100.0)
    out = lab + (new - lab) * np.asarray(weight, np.float32)[:, None]
    return light.to_srgb(light.from_lab(out))


def hair_like(lab: np.ndarray, hair, skin=None) -> np.ndarray:
    """0-1 per Lab colour (``(N, 3)``): how much it is the avatar's hair, of colour ``hair`` (Lab): near it
    (:data:`HAIR_LIKE`) and, with the avatar's ``skin`` colour given, nearer to the hair than to the skin by
    :data:`HAIR_NOT_SKIN` (blond hair and skin are close in Lab; skin must never be recoloured as hair)."""
    from ..outfit.regions import _distance

    d = _distance(lab, np.asarray(hair, np.float32))
    out = np.clip((HAIR_LIKE[1] - d) / (HAIR_LIKE[1] - HAIR_LIKE[0]), 0.0, 1.0)
    if skin is not None:
        out *= np.clip((_distance(lab, np.asarray(skin, np.float32)) - d) / HAIR_NOT_SKIN, 0.0, 1.0)
    return out


def _recolour_hair(match: BodyMatch, model, result) -> None:
    """The hair cards and the scalp texels no photo showed, towards the photos' hair colour (in place)."""
    from ..outfit.regions import texture_of

    target = result.hair.lab
    atlas = result.atlas
    skin = result.own_skin.lab if result.own_skin is not None else None
    scalp = (atlas.parts == fuse.SCALP) if atlas is not None else None
    if scalp is not None and scalp.sum() > 50 and match.head.shape[0] == atlas.size:
        own = light.lab(atlas.own[scalp])
        source, _spread, kept = light.trimmed_median(own)
        if kept and light.delta_e(source, target) > HAIR_DIFFERENT:
            unseen = np.clip(1.0 - result.confidence[scalp] / fuse.KEEP_AT, 0.0, 1.0)
            weight = unseen * hair_like(own, source, skin)
            rgb = match.head[..., :3]
            rgb[scalp] = recolour(rgb[scalp], source, target, weight)
    for prim in model.primitives:
        name = prim.name.lower()
        if not any(part in name for part in HAIR_PRIMITIVES) or "head" in name or "body" in name:
            continue
        texture = texture_of(prim.texture)
        if texture is None:
            continue
        rgba = np.array(texture.rgba, np.uint8, copy=True)
        opaque = rgba[..., 3] > 128
        if opaque.sum() < 50:
            continue
        lab = light.lab(light.to_linear(rgba[..., :3][opaque]))
        source, _spread, kept = light.trimmed_median(lab)
        if not kept or light.delta_e(source, target) <= HAIR_DIFFERENT:
            continue
        drawn = rgba[..., 3] > 0
        lab_all = light.lab(light.to_linear(rgba[..., :3][drawn]))
        rgba[..., :3][drawn] = recolour(rgba[..., :3][drawn], source, target, hair_like(lab_all, source, skin))
        match.hair[prim.name] = rgba


__all__ = ["BodyMatch", "at_angles", "hair_like", "hands_tone", "match_body", "recolour", "sector_offsets"]
