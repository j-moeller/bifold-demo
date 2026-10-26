"""Outfits: garment regions of a body texture, cloth coordinates, procedural fabrics, prompts and the compositor that
keeps the texture's shading; on a small synthetic person, and on the real avatars when they are prepared."""

import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from test_yoga import synthetic_biped

from bifold_demo.characters.character import Character
from bifold_demo.characters.model import CharacterModel, CharacterState, SkinnedPrimitive
from bifold_demo.characters.outfit import (
    PATTERNS,
    REGIONS,
    Atlas,
    Outfit,
    Pattern,
    Solid,
    color,
    dress,
    outfit_of,
    parse_fill,
    parse_outfit,
    presets,
    undress,
)
from bifold_demo.characters.outfit import patterns as P
from bifold_demo.characters.outfit import regions as R
from bifold_demo.characters.outfit.fills import Where, from_spec
from bifold_demo.characters.outfit.surface import PARTS
from bifold_demo.characters.outfit.tailor import Tailor, shading_layers

CHARACTERS = Path(os.environ.get("BIFOLD_CHARACTERS_TEST_DIR", "/nonexistent")).expanduser()
SIZE = 256  # texels of the synthetic person's body texture
SKIN_RGB = tuple(int(v) for v in np.clip(cv2.cvtColor(np.float32([[R._SKIN_DEFAULT]]), cv2.COLOR_Lab2RGB)[0, 0] * 255,
                                         0, 255))


# -- a small person: tubes for the torso, arms, hands, legs and feet, each its own UV island ---------------------


def _tube(a, b, radius, bones, box, n=24, m=12, angles=(0.0, 2 * np.pi)):
    """A tube from ``a`` to ``b`` (the first half skinned to ``bones[0]``, the rest to ``bones[1]``), unrolled onto
    the atlas box ``(u0, v0, u1, v1)`` (around: u, along: v); ``angles``: the part of the way round it covers (angle 0
    points to the person's left, pi / 2 to their back)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    axis = (b - a) / np.linalg.norm(b - a)
    e1 = np.cross(axis, (0, 0, 1)) if abs(axis[2]) < 0.9 else np.cross(axis, (1, 0, 0))
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    u0, v0, u1, v1 = box
    pos, nrm, uv, joints = [], [], [], []
    for i in range(m + 1):
        for j in range(n + 1):
            angle = angles[0] + (angles[1] - angles[0]) * j / n
            radial = np.cos(angle) * e1 + np.sin(angle) * e2
            pos.append(a + (b - a) * i / m + radius * radial)
            nrm.append(radial)
            uv.append((u0 + (u1 - u0) * j / n, v0 + (v1 - v0) * i / m))
            joints.append(bones[0] if i < m / 2 else bones[1])
    tri = []
    for i in range(m):
        for j in range(n):
            k = i * (n + 1) + j
            tri += [(k, k + n + 1, k + 1), (k + 1, k + n + 1, k + n + 2)]
    return np.array(pos), np.array(nrm), np.array(uv), np.array(tri), np.array(joints)


def _person(texture: np.ndarray | None = None, cut: bool = False) -> CharacterModel:
    """The small person; ``cut``: its torso is two UV islands, the back half and the front half, cut at its sides
    and laid out apart in the atlas."""
    skeleton = synthetic_biped().skeleton
    bone = skeleton.index
    torso = [((0, 0.93, 0), (0, 1.45, 0), 0.15, ("Bip01 Spine", "Bip01 Spine1"), (0.30, 0.40, 0.70, 0.95), "top", None)]
    if cut:
        torso = [((0, 0.93, 0), (0, 1.45, 0), 0.15, ("Bip01 Spine", "Bip01 Spine1"), (0.30, 0.40, 0.48, 0.95), "top",
                  (0.0, np.pi)),
                 ((0, 0.93, 0), (0, 1.45, 0), 0.15, ("Bip01 Spine", "Bip01 Spine1"), (0.52, 0.40, 0.70, 0.95), "top",
                  (np.pi, 2 * np.pi))]
    parts = torso + [(*part, None) for part in [  # (a, b, radius, bones, uv box, colour, angles)
        ((0.2, 1.38, 0), (0.52, 1.07, 0), 0.05, ("Bip01 L UpperArm", "Bip01 L Forearm"), (0.72, 0.40, 0.84, 0.95),
         "top"),
        ((-0.2, 1.38, 0), (-0.52, 1.07, 0), 0.05, ("Bip01 R UpperArm", "Bip01 R Forearm"), (0.16, 0.40, 0.28, 0.95),
         "top"),
        ((0.52, 1.07, 0), (0.6, 0.95, 0), 0.04, ("Bip01 L Hand", "Bip01 L Hand"), (0.86, 0.40, 0.98, 0.60), "skin"),
        ((-0.52, 1.07, 0), (-0.6, 0.95, 0), 0.04, ("Bip01 R Hand", "Bip01 R Hand"), (0.02, 0.40, 0.14, 0.60), "skin"),
        ((0.1, 0.93, 0), (0.11, 0.1, 0), 0.07, ("Bip01 L Thigh", "Bip01 L Calf"), (0.52, 0.02, 0.70, 0.38),
         "bottoms"),
        ((-0.1, 0.93, 0), (-0.11, 0.1, 0), 0.07, ("Bip01 R Thigh", "Bip01 R Calf"), (0.30, 0.02, 0.48, 0.38),
         "bottoms"),
        ((0.11, 0.08, -0.05), (0.13, 0.03, 0.16), 0.045, ("Bip01 L Foot", "Bip01 L Toe0"), (0.76, 0.02, 0.90, 0.20),
         "shoes"),
        ((-0.11, 0.08, -0.05), (-0.13, 0.03, 0.16), 0.045, ("Bip01 R Foot", "Bip01 R Toe0"), (0.06, 0.02, 0.20, 0.20),
         "shoes"),
    ]]
    paint = {"top": (36, 52, 110), "bottoms": (120, 120, 116), "shoes": (24, 22, 20), "skin": SKIN_RGB}
    if texture is None:
        texture = np.zeros((SIZE, SIZE, 4), np.uint8)
        rng = np.random.default_rng(1)
        for *_, (u0, v0, u1, v1), what, _ in parts:
            r0, r1 = int((1 - v1) * SIZE) - 2, int((1 - v0) * SIZE) + 2
            c0, c1 = int(u0 * SIZE) - 2, int(u1 * SIZE) + 2
            block = np.full((r1 - r0, c1 - c0, 3), paint[what], np.float32)
            if what != "skin":  # weave and a little noise
                block *= (1.0 + 0.12 * rng.standard_normal(block.shape[:2]))[..., None]
            texture[r0:r1, c0:c1, :3] = np.clip(block, 0, 255)
            texture[r0:r1, c0:c1, 3] = 255
    arrays = [_tube(a, b, r, tuple(bone(n) for n in bones), box, n=12 if angles else 24,
                    angles=angles or (0.0, 2 * np.pi)) for a, b, r, bones, box, _, angles in parts]
    offsets = np.cumsum([0] + [len(x[0]) for x in arrays[:-1]])
    pos = np.concatenate([x[0] for x in arrays]).astype(np.float32)
    nrm = np.concatenate([x[1] for x in arrays]).astype(np.float32)
    uv = np.concatenate([x[2] for x in arrays]).astype(np.float32)
    tri = np.concatenate([x[3] + o for x, o in zip(arrays, offsets)]).astype(np.uint32)
    joints = np.zeros((len(pos), 4), np.int64)
    joints[:, 0] = np.concatenate([x[4] for x in arrays])
    weights = np.zeros((len(pos), 4), np.float32)
    weights[:, 0] = 1.0
    body = SkinnedPrimitive("p000_body", pos, nrm, uv, tri, joints, weights, texture)
    return CharacterModel("tube_person", skeleton, [body], np.linalg.inv(skeleton.rest_global))


@pytest.fixture(scope="module")
def person():
    return _person()


@pytest.fixture(scope="module")
def regions(person):
    return R.body_regions(person, cache=False)


def _share(mask, labels, region):
    return float(np.mean(labels[mask] == REGIONS.index(region)))


def test_regions_follow_bones_islands_and_skin(person, regions):
    groups, labels = regions.groups, regions.labels
    assert regions.size == SIZE and regions.primitive == 0
    assert _share(groups == R.GROUPS.index("hands"), labels, "skin") > 0.95
    assert _share(groups == R.GROUPS.index("feet"), labels, "shoes") > 0.95
    assert _share(groups == R.GROUPS.index("legs"), labels, "bottoms") > 0.95
    assert _share(groups == R.GROUPS.index("torso"), labels, "top") > 0.95
    assert _share(groups == R.GROUPS.index("arms"), labels, "top") > 0.95
    assert np.all(labels[~regions.bake.covered] == R.OUTSIDE)
    assert np.array_equal(regions.skin, labels == R.SKIN)
    assert all(abs(a - b) < 12 for a, b in zip(regions.skin_color, SKIN_RGB))
    assert regions.flags["mirrored"] == [] and regions.flags["overlapping"] == []


def test_masks_are_feathered_by_a_few_texels_and_never_cover_skin():
    labels = np.full((1024, 1024), R.TOP, np.uint8)
    labels[:, 512:] = R.SHIRT
    labels[:100] = R.SKIN
    masks = R.feather(labels)
    top, shirt = masks["top"], masks["shirt"]
    row = top[500]
    ramp = np.nonzero((row > 0.02) & (row < 0.98))[0]
    assert 2 <= len(ramp) <= 6 and abs(ramp.mean() - 511.5) < 1.5  # a 2-3 texel ramp each side of the seam
    clothes = labels != R.SKIN
    assert np.allclose((top + shirt + masks["other"])[clothes], 1.0, atol=1e-5)
    assert np.all(top[:100] == 0) and np.all(shirt[:100] == 0)
    assert top[300, 100] == pytest.approx(1.0) and top[101, 100] == pytest.approx(1.0)  # no fade towards the skin


def test_mirrored_and_overlapping_islands_are_flagged(person):
    from bifold_demo.characters.uvbake import islands

    prim = person.primitives[0]
    tube = 13 * 25  # vertices per tube: the torso, the left arm, the right arm, the left hand, the right hand, ...
    uv = prim.uvs.copy()
    uv[:tube, 0] = 1.0 - uv[:tube, 0]  # the torso's island mirrored (onto itself)
    uv[3 * tube:4 * tube] = uv[4 * tube:5 * tube]  # the left hand drawn over the right one
    changed = SkinnedPrimitive("p000_body", prim.positions, prim.normals, uv, prim.indices, prim.joints, prim.weights,
                               prim.texture)
    model = CharacterModel("changed", person.skeleton, [changed], person.inverse_bind)
    flags = R.body_regions(model, cache=False).flags
    island = islands(changed)
    triangles = 2 * 12 * 24  # per tube
    assert flags["mirrored"] == [int(island[0])]
    assert flags["overlapping"] == [int(island[4 * triangles])]


def test_overrides_relabel_texels_and_can_live_next_to_the_cache(person, regions, tmp_path):
    rules = [{"region": "other", "where": {"group": "feet", "side": "left"}},
             {"region": "shirt", "where": {"label": "top", "above": 1.3, "facing": "front"}}]
    labels = R.apply_overrides(regions, rules, person)
    feet = regions.groups == R.GROUPS.index("feet")
    with np.errstate(invalid="ignore"):
        left = regions.bake.position[..., 0] > 0
        high = regions.bake.position[..., 1] > 1.3
    assert np.all(labels[feet & left] == R.OTHER) and np.all(labels[feet & ~left] == R.SHOES)
    assert np.any(labels == R.SHIRT) and np.all(labels[labels == R.SHIRT] == R.SHIRT) and \
        np.all(high[labels == R.SHIRT])
    with pytest.raises(ValueError):
        R.apply_overrides(regions, [{"region": "hat", "where": {}}], person)
    with pytest.raises(ValueError):
        R.apply_overrides(regions, [{"region": "top", "where": {"colour": "red"}}], person)
    (tmp_path / "tube_person.json").write_text(json.dumps({"rules": rules[:1]}))
    again = R.body_regions(person, cache=tmp_path)
    assert np.all(again.labels[feet & left] == R.OTHER) and np.all(again.automatic[feet & left] == R.SHOES)
    assert again.flags["overrides"] == [str(tmp_path / "tube_person.json")]


def test_the_shipped_overrides_are_valid():
    files = sorted(R.OVERRIDES.glob("*.json"))
    assert files
    for path in files:
        for rule in json.loads(path.read_text())["rules"]:
            assert rule["region"] in REGIONS
            assert set(rule.get("where", {})) <= {"label", "island", "group", "bone", "side", "facing", "below",
                                                  "above", "uv"}


def test_labels_are_cached_on_disk(person, tmp_path):
    first = R.body_regions(person, cache=tmp_path, overrides=False)
    path = tmp_path / "tube_person.npz"
    assert path.is_file() and path.stat().st_size < 200_000
    R._memory.clear()
    again = R.body_regions(person, cache=tmp_path, overrides=False)
    assert again is not first and np.array_equal(again.labels, first.labels)
    assert np.allclose(again.bake.position, first.bake.position, equal_nan=True)
    assert np.array_equal(again.colours, first.colours)


def test_damaged_cache_files_are_made_again_and_bad_overrides_left_out(person, tmp_path, monkeypatch):
    first = R.body_regions(person, cache=tmp_path, overrides=False)
    path = tmp_path / "tube_person.npz"
    good = path.read_bytes()
    third = len(good) // 3
    for damaged in (good[:5000], good[:third] + bytes(third) + good[2 * third:]):  # cut short; a member broken
        path.write_bytes(damaged)
        R._memory.clear()
        again = R.body_regions(person, cache=tmp_path, overrides=False)
        assert np.array_equal(again.labels, first.labels) and path.read_bytes() != damaged  # made again
    # a user's override file that is no JSON, or holds a bad rule, is left out with a warning
    for content in ("{not json", json.dumps({"rules": [{"region": "hat"}]}), json.dumps(["top"])):
        (tmp_path / "tube_person.json").write_text(content)
        R._memory.clear()
        with pytest.warns(UserWarning, match="is not used"):
            regions = R.body_regions(person, cache=tmp_path)
        assert regions.flags["overrides"] == [] and np.array_equal(regions.labels, first.labels)
    # a write that fails (a read-only folder) leaves no temporary file behind
    def refuse(*_):
        raise OSError("read-only")

    monkeypatch.setattr(R.os, "replace", refuse)
    target = tmp_path / "elsewhere" / "tube_person.npz"
    R._write(target, "key", first.bake.triangle, first.bake.overlap, first.labels, first.skin_lab, first.colours)
    assert not target.exists() and not list(target.parent.glob("*"))


def test_only_the_last_few_avatars_stay_in_memory(person):
    import gc
    import weakref

    from bifold_demo.characters.outfit import tailor as T

    models = [CharacterModel(f"p{k}", person.skeleton, person.primitives, person.inverse_bind)
              for k in range(T.TAILORS + 2)]
    tailors = [weakref.ref(T.tailor_for(m)) for m in models]
    assert T.tailor_for(models[-1]) is tailors[-1]() and len(T._tailors) == T.TAILORS
    gc.collect()
    assert tailors[0]() is None and tailors[-1]() is not None  # the first ones dressed are let go
    assert len(R._memory) <= R.KEEP
    # the regions' cache does not keep a model alive
    from bifold_demo.characters import face_photo

    model = CharacterModel("gone", person.skeleton, person.primitives, person.inverse_bind)
    R.body_regions(model, cache=False)
    face_photo._layouts.pop(id(model), None)  # (the face's layouts, which the skin colour comes from, keep theirs)
    gone = weakref.ref(model)
    del model
    gc.collect()
    assert gone() is None


# -- colours, patterns, fills ------------------------------------------------------------------------------------


def test_colours_by_name_hex_and_shade():
    assert color("navy") == (31, 42, 74) and color("#ff8000") == (255, 128, 0) and color("#f80") == (255, 136, 0)
    assert color((300, -5, 10.4)) == (255, 0, 10)
    light, dark = color("light grey"), color("dark grey")
    assert light[0] > color("grey")[0] > dark[0]
    with pytest.raises(ValueError):
        color("blurple")


@pytest.mark.parametrize("kind", sorted(PATTERNS))
def test_pattern_tiles_are_seamless_and_use_both_colours(kind):
    primary, secondary = (200, 40, 40), (240, 240, 240)
    tile = P.tile(kind, primary, secondary)
    assert tile.dtype == np.float32 and tile.shape[0] == tile.shape[1] and tile.shape[2] == 3
    assert tile.min() >= 0 and tile.max() <= 255
    # across the wrap the tile changes no more than between neighbouring texels inside it
    for axis in (0, 1):
        inside = np.abs(np.diff(tile, axis=axis)).max()
        wrap = np.abs(np.take(tile, 0, axis=axis) - np.take(tile, -1, axis=axis)).max()
        assert wrap <= inside + 1.0, (kind, axis)
    def near(c):
        return np.mean(np.linalg.norm(tile - np.float32(c), axis=2) < 40)

    if kind not in ("denim", "camo", "tartan", "twill", "herringbone"):
        assert near(primary) > 0.3 and near(secondary) > 0.03


def test_woven_patterns_mix_like_cloth():
    dark, light = np.float32((20, 20, 20)), np.float32((240, 240, 240))
    hounds = P.houndstooth(dark, light)
    assert np.mean(hounds[..., 0] < 128) == pytest.approx(0.5, abs=0.02)
    tones = np.unique(np.round(P.gingham(dark, light, size=16)[..., 0]))
    assert len(tones) == 2  # at thread level two colours, ...
    ging = P.gingham(dark, light)
    mean = cv2.resize(ging, (2, 2), interpolation=cv2.INTER_AREA)[..., 0]
    assert sorted(np.round(mean.ravel() / 10)) == [2, 13, 13, 24]  # ... seen from afar dark, two mixed, light
    stripes = P.stripes(dark, light)
    assert np.mean(stripes[..., 0] > 128) == pytest.approx(0.5, abs=0.02)
    assert np.mean(P.pinstripe(dark, light)[..., 0] > 128) == pytest.approx(0.07, abs=0.02)


def _where(uv, footprint=0.0005):
    uv = np.asarray(uv, np.float32)
    n = len(uv)
    return Where(uv, np.zeros(n, np.int8), np.full(n, footprint, np.float32), np.zeros(n, int), np.zeros(n, int), 8)


def test_tiled_fills_follow_scale_rotation_and_blur_when_finer_than_a_texel():
    stripes = Pattern("stripes", "black", "white", scale=0.04)
    across = _where([(x, 0.3) for x in np.linspace(0, 0.08, 81)])
    along = _where([(0.011, y) for y in np.linspace(0, 0.08, 81)])
    a, b = stripes.colors(across)[:, 0], stripes.colors(along)[:, 0]
    assert a.max() - a.min() > 150 and b.max() - b.min() < 5  # vertical stripes: they change across, not along
    assert np.allclose(a[:41], a[40:], atol=3)  # one repeat per 4 cm
    turned = Pattern("stripes", "black", "white", scale=0.04, rotation=90)
    assert np.ptp(turned.colors(along)[:, 0]) > 150 and np.ptp(turned.colors(across)[:, 0]) < 5
    pin = Pattern("pinstripe", "black", "white", scale=0.01)
    far = pin.colors(_where(np.random.default_rng(0).random((200, 2)), footprint=0.05))
    tile_mean = P.pinstripe(np.float32(color("black")), np.float32(color("white"))).mean(axis=(0, 1))
    assert np.allclose(far, tile_mean, atol=3)  # each texel covers many lines: their mixed colour


def test_a_swatch_is_a_fill_like_the_patterns(tmp_path):
    from bifold_demo.characters.outfit import Swatch

    image = np.zeros((64, 64, 3), np.uint8)
    image[:, 32:] = (0, 200, 0)
    path = tmp_path / "fabric.png"
    cv2.imwrite(str(path), image[..., ::-1])
    swatch = Swatch(path, scale=0.1)
    c = swatch.colors(_where([(0.02, 0.0), (0.07, 0.0)]))
    assert c[0, 1] < 30 and c[1, 1] > 170
    again = from_spec(swatch.spec())
    assert np.allclose(again.colors(_where([(0.07, 0.0)])), c[1:])


def test_swatches_take_images_as_pipelines_give_them_and_keep_them_in_their_spec(tmp_path):
    from bifold_demo.characters.outfit import Swatch

    probe = _where([(0.02, 0.0), (0.07, 0.0)])
    half = np.zeros((64, 64, 3), np.float32)
    half[:, 32:] = (0.0, 0.8, 0.0)  # floats in 0-1
    swatch = Swatch(half, scale=0.1)
    c = swatch.colors(probe)
    assert c[0].max() < 5 and c[1, 1] == pytest.approx(204, abs=2)
    again = from_spec(json.loads(json.dumps(swatch.spec())))  # the image is in the spec
    assert np.allclose(again.colors(probe), c, atol=1.0)
    grey = Swatch(np.tile(np.linspace(0, 255, 64).astype(np.uint8), (64, 1)), scale=0.1).colors(probe)
    assert np.allclose(grey[:, 0], grey[:, 2]) and grey[1, 0] > grey[0, 0] + 100  # a grey image is grey
    rgba = np.dstack([np.full((8, 8, 3), 65535, np.uint16), np.zeros((8, 8), np.uint16)])
    assert np.allclose(Swatch(rgba).colors(probe), 255)  # 16 bits, alpha dropped
    painted = (np.random.default_rng(0).random((8, 8, 3)) * 255).astype(np.uint8)
    path = tmp_path / "painted.png"
    cv2.imwrite(str(path), painted[..., ::-1])
    rows, cols = np.mgrid[0:8, 0:8]
    where = Where(np.zeros((64, 2), np.float32), np.zeros(64, np.int8), np.ones(64, np.float32), rows.ravel(),
                  cols.ravel(), 8)
    for atlas in (Atlas(painted), Atlas(path)):
        assert np.allclose(atlas.colors(where), painted.reshape(-1, 3))
        assert np.allclose(from_spec(json.loads(json.dumps(atlas.spec()))).colors(where), painted.reshape(-1, 3))
    assert Atlas(path).spec() == {"atlas": str(path)}


def test_fill_specs_round_trip():
    for spec in ("navy", "keep", [10, 20, 30], {"color": "red"}, "red tartan", "white polka dots on black",
                 {"pattern": "camo", "colors": ["olive"], "scale": 0.4, "rotation": 30}):
        fill = from_spec(spec)
        assert from_spec(fill.spec()).spec() == fill.spec()


def test_keywords_make_fills():
    f = parse_fill("navy pinstripe")
    assert isinstance(f, Pattern) and f.kind == "pinstripe" and f.primary == color("navy")
    assert parse_fill("red tartan").kind == "tartan" and parse_fill("red tartan").primary == color("red")
    camo = parse_fill("olive camo")
    assert camo.kind == "camo" and camo.primary == color("olive")
    dots = parse_fill("white polka dots on black")
    assert dots.kind == "polka dots" and dots.primary == color("black") and dots.secondary == color("white")
    two = parse_fill("red and black checks")
    assert two.primary == color("red") and two.secondary == color("black")
    assert isinstance(parse_fill("navy"), Solid)
    assert parse_fill("dark green").spec() == Solid(color("dark green")).spec()
    big, thin = parse_fill("large red stripes"), parse_fill("thin horizontal red stripes")
    assert big.scale > PATTERNS["stripes"][1] > thin.scale and thin.rotation == 90
    assert parse_fill("plaid").kind == "tartan" and parse_fill("dogtooth").kind == "houndstooth"
    assert parse_fill("light denim").primary[2] > P.DEFAULT_PRIMARY["denim"][2]
    assert parse_fill("navy pinstripe").spec() == parse_fill("navy pinstripe").spec()  # the same words, the same fill
    with pytest.raises(ValueError):
        parse_fill("something nice")


def test_keywords_make_outfits():
    outfit = parse_outfit("red tartan jacket, white shirt and black trousers")
    assert set(outfit) == {"top", "shirt", "bottoms"}
    assert outfit["top"].kind == "tartan" and outfit["bottoms"].spec() == Solid("black").spec()
    jeans = parse_outfit("blue jeans with white trainers")
    assert jeans["bottoms"].kind == "denim" and jeans["shoes"].spec() == Solid("white").spec()
    suit = parse_outfit("charcoal suit")
    assert set(suit) == {"top", "bottoms"}
    assert set(parse_outfit("black and white stripes")) == {"top"}
    # parts it understands nothing of are left out: it fails only when it understands no part
    nice = parse_outfit("a nice navy pinstripe suit, please")
    assert set(nice) == {"top", "bottoms"} and nice["top"].kind == "pinstripe"
    assert set(parse_outfit("red tartan jacket with pockets")) == {"top"}
    assert parse_outfit("red tartan jacket with gold buttons")["top"].kind == "tartan"  # details: not the jacket
    with pytest.raises(ValueError):
        parse_outfit("please")
    # layers, and the last garment word names the garment
    for text in ("white shirt under a navy blazer", "a navy blazer over a white shirt"):
        layered = parse_outfit(text)
        assert layered["shirt"].spec() == Solid("white").spec() and layered["top"].spec() == Solid("navy").spec()
    assert set(parse_outfit("white dress shirt")) == {"shirt"} and set(parse_outfit("white t shirt")) == {"top"}
    assert parse_outfit("a jeans jacket")["top"].kind == "denim"
    striped = parse_outfit("white shirt with navy stripes")  # "with" and a fabric: more about the garment before
    assert set(striped) == {"shirt"} and striped["shirt"].kind == "stripes"
    assert striped["shirt"].primary == color("white") and striped["shirt"].secondary == color("navy")
    late = parse_outfit("black trousers, red tartan")  # a fabric on its own is the top's
    assert late["top"].kind == "tartan" and late["bottoms"].spec() == Solid("black").spec()
    assert set(parse_outfit("a red jacket, navy stripes")) == {"top"}  # ... unless a part names the top
    assert parse_outfit("a red jacket, navy stripes")["top"].spec() == Solid("red").spec()


def test_presets_are_valid_outfits():
    assert {"business navy", "casual denim", "sporty", "trade-fair staff", "festival", "all black"} <= set(presets())
    for name in presets():
        outfit = Outfit.parse(name)
        assert outfit.name == name and set(outfit.fills) <= set(R.CLOTHES) and outfit.fills


# -- the compositor ------------------------------------------------------------------------------------------------


def test_detail_times_shading_gives_the_texture_back():
    size = 256
    yy, xx = np.mgrid[0:size, 0:size]
    shading = 0.6 + 0.4 * np.sin(yy / size * np.pi)  # low frequency, like folds and occlusion
    detail = np.where(xx % 4 < 2, 1.15, 0.85)  # a fine weave
    base = np.float32((60, 90, 160))
    rgb = np.clip(base * (shading * detail)[..., None], 0, 255).astype(np.float32)
    labels = np.full((size, size), R.TOP, np.uint8)
    labels[:, :20] = R.SKIN
    layers = shading_layers(rgb, labels)
    inner = (slice(40, 216), slice(60, 200))
    rebuilt = layers.albedo * (layers.detail * layers.shading["top"])[..., None]
    assert np.allclose(rebuilt[labels == R.TOP], rgb[labels == R.TOP], atol=0.5)
    assert np.corrcoef(layers.detail[inner].ravel(), detail[inner].ravel())[0, 1] > 0.9
    assert np.corrcoef(layers.shading["top"][inner].ravel(), shading[inner].ravel())[0, 1] > 0.95
    assert np.all(layers.detail[:, :20] == 1.0)


def test_colour_blocks_give_each_colour_of_a_garment_its_own_shading_reference():
    size = 256
    yy, xx = np.mgrid[0:size, 0:size]
    shading = 0.7 + 0.3 * np.sin(yy / size * 2 * np.pi)  # folds running across both blocks
    detail = np.where((xx + yy) % 4 < 2, 1.1, 0.9)
    grey, black = np.float32((170, 170, 170)), np.float32((24, 24, 26))
    base = np.where((xx < 128)[..., None], grey, black)  # a grey body and a black yoke, sewn together at x = 128
    rgb = np.clip(base * (shading * detail)[..., None], 0, 255).astype(np.float32)
    labels = np.full((size, size), R.TOP, np.uint8)
    colours = np.where(xx < 128, 0, 3).astype(np.int16)
    blocks = R.colour_blocks(labels, colours, rgb)
    assert len(np.unique(blocks)) == 2
    layers = shading_layers(rgb, labels, blocks)
    flat = shading_layers(rgb, labels)  # one reference for the whole region (each region one block)
    left, right = (slice(20, 236), slice(10, 110)), (slice(20, 236), slice(146, 246))

    def lit(layers, part):  # how bright a fill comes out there
        return float(np.mean((layers.detail * layers.shading["top"])[part]))

    assert lit(flat, right) < 0.25 * lit(flat, left)  # one reference: the yoke stays dark under any fill
    assert lit(layers, right) == pytest.approx(lit(layers, left), rel=0.1)  # its own: it takes the fill too
    for part in (left, right):  # the folds stay in both
        assert np.corrcoef(layers.shading["top"][part].ravel(), shading[part].ravel())[0, 1] > 0.95
    rebuilt = layers.albedo * (layers.detail * layers.shading["top"])[..., None]
    assert np.allclose(rebuilt, rgb, atol=0.5)  # the albedo still gives the texture back
    assert layers.blocks[int(blocks[0, 0])] > 4 * layers.blocks[int(blocks[0, -1])]
    # one colour found in two clusters (the two halves of a robe) is one block; a speck of another colour joins it
    halves = np.where(yy < 128, 0, 32).astype(np.int16)
    halves[:2, :2] = 5
    speck = rgb.copy()
    speck[:, :] = grey
    speck[:2, :2] = (200, 30, 30)
    assert len(np.unique(R.colour_blocks(labels, halves, speck))) == 1
    labels[:, :20] = R.SKIN
    assert np.all(R.colour_blocks(labels, colours, rgb)[:, :20] == -1)


def test_a_fill_of_a_garments_own_colour_gives_its_texture_back(person, regions):
    tailor = Tailor(person, regions)
    original = tailor.original[..., :3].astype(np.float32)
    rgba = tailor.texture({"top": (36, 52, 110)}).astype(np.float32)  # the colour the top was painted in (+ noise)
    inner = cv2.erode(regions.hard("top").astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    ratio = (rgba[..., :3][inner] + 1.0) / (original[inner] + 1.0)
    assert np.median(ratio) == pytest.approx(1.0, abs=0.12)  # the colour is its colour at the shading reference
    assert np.std(ratio / np.median(ratio, axis=0)) < 0.04  # and the same texture: weave, noise and all


def test_a_fill_on_a_small_person_keeps_skin_and_shading(person, regions):
    tailor = Tailor(person, regions)
    original = tailor.original
    same = tailor.texture(Outfit({"top": Atlas(tailor.layers.albedo), "bottoms": Atlas(tailor.layers.albedo)}))
    inner = cv2.erode((regions.labels == R.TOP).astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    assert np.abs(same[inner].astype(int) - original[inner].astype(int)).max() <= 1  # the albedo gives it back
    red = tailor.texture({"top": "red", "shoes": "white"})
    skin = regions.skin
    assert np.array_equal(red[skin], original[skin])  # skin is never written
    top = regions.hard("top")
    assert np.all(red[top][:, 0] > red[top][:, 2])  # red now
    legs = cv2.erode(regions.hard("bottoms").astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    assert np.array_equal(red[legs], original[legs])  # regions the outfit leaves out keep their clothes
    lum_old = original[..., :3].astype(np.float32).mean(axis=2)[inner]
    lum_new = red[..., :3].astype(np.float32).mean(axis=2)[inner]
    assert np.corrcoef(lum_old, lum_new)[0, 1] > 0.9  # the weave and the noise stay
    assert np.all(red[regions.hard("shoes")][:, :3].mean(axis=1) > original[regions.hard("shoes")][:, :3].mean(axis=1))


def test_cloth_coordinates_make_stripes_run_down_the_body(person, regions):
    tailor = Tailor(person, regions)
    surf = tailor.surface
    torso = (surf.part == PARTS.index("torso"))
    assert torso.sum() > 1000 and np.all(surf.part[regions.skin] == -1)
    assert (surf.part == PARTS.index("left arm")).sum() > 100 and (surf.part == PARTS.index("right leg")).sum() > 100
    # around the torso tube (radius 0.15 m) the unrolled width is its girth
    u = surf.uv[..., 0][torso]
    assert np.ptp(u) == pytest.approx(2 * np.pi * 0.15, rel=0.1)
    # vertical stripes: one colour along the body (same u), changing round it
    rgba = tailor.texture({"top": Pattern("stripes", "black", "white", scale=0.1)})
    lum = rgba[..., :3].astype(np.float32).mean(axis=2)
    cols = np.nonzero(torso.any(axis=0))[0]
    col = cols[len(cols) // 3]
    rows = np.nonzero(torso[:, col])[0][5:-5]
    assert np.ptp(lum[rows, col]) < 80  # down a column of the torso island (along the body): the same stripe
    row = rows[len(rows) // 2]
    assert np.ptp(lum[row, cols[5:-5]]) > 120  # across it: the stripes


def test_patterns_run_on_across_uv_cuts():
    model = _person(cut=True)
    regions = R.body_regions(model, cache=False)
    tailor = Tailor(model, regions)
    surf, bake = tailor.surface, regions.bake
    torso = surf.part == PARTS.index("torso")
    ids, counts = np.unique(bake.island[torso], return_counts=True)
    halves = [torso & (bake.island == k) for k in ids[np.argsort(-counts)[:2]]]  # (the rest: the arms' tops)
    assert min(h.sum() for h in halves) > 0.4 * torso.sum()  # the back and the front: apart in the atlas
    rgba = tailor.texture({"top": Pattern("stripes", "black", "white", scale=0.08, rotation=45)})
    lum = rgba[..., :3].astype(np.float32).mean(axis=2)
    a, b = (np.argwhere(h) for h in halves)
    pa, pb = bake.position[halves[0]], bake.position[halves[1]]
    near = np.linalg.norm(pa[:, None, :] - pb[None, :, :], axis=2)  # texel to texel, on the body
    i, j = np.nonzero(near < 0.006)  # where the halves meet, at the person's sides
    assert len(i) > 20 and np.ptp(pa[i, 1]) > 0.3  # (all along the cut)
    ua, ub = surf.uv[tuple(a[i].T)], surf.uv[tuple(b[j].T)]
    assert np.abs(ua - ub).max() < 0.015  # the cloth coordinates run on across the cut ...
    rows = np.random.default_rng(0).integers(0, len(a), len(i))
    across = np.abs(lum[tuple(a[i].T)] - lum[tuple(b[j].T)]).mean()
    anywhere = np.abs(lum[tuple(a[rows].T)] - lum[tuple(b[j].T)]).mean()
    assert across < 0.35 * anywhere  # ... and so do the diagonal stripes


def test_an_outfit_goes_on_the_texture_the_character_wears(person, regions):
    from bifold_demo.scene import Texture

    one = Character(person, name="one")
    matched = person.primitives[0].texture.copy()
    matched[regions.skin, :3] = (205, 150, 120)  # say, the face component matched the skin to a photo
    base = Texture(matched, name="matched skin")
    one.texture_overrides["p000_body"] = base
    texture = dress(one, {"top": "red"})
    rgba = texture.rgba
    assert np.array_equal(rgba[regions.skin], matched[regions.skin])  # the skin it wore stays
    top = regions.hard("top")
    assert np.all(rgba[top][:, 0] > rgba[top][:, 2]) and texture.base is base
    again = dress(one, "all black")  # dressing again starts from it too
    assert again is texture and np.array_equal(texture.rgba[regions.skin], matched[regions.skin])
    undress(one)
    assert one.texture_overrides["p000_body"] is base  # and taking the outfit off gives it back


def test_dress_and_undress_a_character(person):
    one = Character(person, behaviour=lambda t: CharacterState((0, 0, 0), 0.0, []), name="one")
    two = Character(person, name="two")
    texture = dress(one, "trade-fair staff")
    assert one.texture_overrides == {"p000_body": texture} and two.texture_overrides == {}
    assert outfit_of(one).name == "trade-fair staff"
    version = texture.version
    again = dress(one, {"top": "olive camo"})
    assert again is texture and texture.version > version  # the same texture, new pixels
    one.texture_overrides["other_part"] = "#ffffff"
    undress(one)
    assert one.texture_overrides == {"other_part": "#ffffff"} and outfit_of(one) is None


def test_the_cli_writes_a_texture_and_the_regions(tmp_path, capsys, monkeypatch, person):
    from bifold_demo.characters import cli

    monkeypatch.setattr(cli, "CharacterLibrary", lambda dest: _Library(person))
    out, regions = tmp_path / "out.png", tmp_path / "regions.png"
    assert cli.main(["outfit", "tube_person", "navy pinstripe", "-o", str(out), "--regions", str(regions)]) == 0
    assert cv2.imread(str(out)).shape == (SIZE, SIZE, 3) and cv2.imread(str(regions)) is not None
    assert "pinstripe" in capsys.readouterr().out
    assert cli.main(["outfit", "tube_person", "a nice navy pinstripe suit, please"]) == 0
    assert "bottoms: pinstripe" in capsys.readouterr().out
    assert cli.main(["outfit", "--presets"]) == 0 and "business navy" in capsys.readouterr().out


class _Library:
    def __init__(self, model):
        self.model = model

    def available(self):
        return {self.model.name: {}}

    def load(self, name):
        return self.model

    def explain(self, name):
        return f"no {name}"


# -- the real avatars -------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    if not CHARACTERS.is_dir():
        pytest.skip("set BIFOLD_CHARACTERS_TEST_DIR to a prepared character library")
    from bifold_demo.characters.library import CharacterLibrary

    with pytest.MonkeyPatch.context() as patch:  # the labels' cache goes to a temporary folder, not the library
        patch.setenv(R.CACHE_ENV, str(tmp_path_factory.mktemp("outfit_cache")))
        yield CharacterLibrary(CHARACTERS)


def _avatar(library, name):
    if name not in library.available():
        pytest.skip(f"{name} is not prepared")
    return library.load(name)


@pytest.mark.parametrize("name", ["Business_Male_02", "Female_Adult_01", "Male_Adult_05", "Male_Adult_14"])
def test_real_avatars_get_plausible_regions(library, name):
    model = _avatar(library, name)
    regions = R.body_regions(model)
    labels, bake = regions.labels, regions.bake
    fingers = np.isin(bake.bone, [k for k, n in enumerate(model.skeleton.names) if "Finger" in n])
    assert _share(fingers, labels, "skin") > 0.95  # (the hand bones also move the cuffs of the sleeves)
    toes = np.isin(bake.bone, [k for k, n in enumerate(model.skeleton.names) if "Toe" in n])
    assert _share(toes, labels, "shoes") > 0.9  # (the foot bones also move the hems of the trousers)
    thighs = np.isin(bake.bone, [model.skeleton.index(f"Bip01 {s} Thigh") for s in "LR"])
    assert _share(thighs, labels, "bottoms") > 0.85
    spine = np.isin(bake.bone, [k for k, n in enumerate(model.skeleton.names) if "Spine" in n])
    with np.errstate(invalid="ignore"):
        back = spine & (bake.normal[..., 2] < -0.85)
    assert _share(back, labels, "top") > 0.9
    counts = regions.counts()
    assert counts["top"] > counts["shirt"] and counts["bottoms"] > counts["shoes"] > 0
    assert 0.03 < counts["skin"] / sum(counts.values()) < 0.25
    assert regions.flags["mirrored"] == [] and regions.flags["overlapping"] == []


def test_a_long_robe_is_all_top_by_its_override(library):
    regions = R.body_regions(_avatar(library, "Male_Adult_21"))
    assert regions.counts()["bottoms"] == 0 and (regions.automatic == R.BOTTOMS).sum() > 100_000


def test_a_blouse_over_a_skirt_is_the_top_front_and_back(library):
    model = _avatar(library, "Female_Party_02")
    regions = R.body_regions(model)
    labels, bake = regions.labels, regions.bake
    hips = np.isin(bake.bone, [k for k, n in enumerate(model.skeleton.names) if "Pelvis" in n or "Thigh" in n])
    with np.errstate(invalid="ignore"):
        y, facing = bake.position[..., 1], bake.normal[..., 2]
        below_belt = hips & (labels != R.SKIN) & (y > 0.95) & (y < 1.0)  # the blouse, which the hip bones move
        assert _share(below_belt & (facing > 0.5), labels, "top") > 0.95
        assert _share(below_belt & (facing < -0.5), labels, "top") > 0.95
    assert regions.counts()["other"] > 1000 and regions.flags["overrides"]  # the belt's loose end (an override)


def test_a_t_shirt_has_no_stray_hems(library):
    regions = R.body_regions(_avatar(library, "Female_Child_01"))
    upper = np.isin(regions.groups, [R.GROUPS.index(g) for g in ("torso", "arms", "neck")])
    other = (regions.labels == R.OTHER) & upper
    n, _, stats, _ = cv2.connectedComponentsWithStats(other.astype(np.uint8), connectivity=8)
    strips = [stats[k, :4] for k in range(1, n) if max(stats[k, 2:4]) > 60 and min(stats[k, 2:4]) <= 12]
    assert not strips  # (the blur along the t-shirt's edges was a colour of its own: pink lines in every outfit)
    assert regions.counts()["shirt"] > 0.3 * regions.counts()["top"]  # the long sleeves under it


def _frontmost(bake, select, step=0.01, depth=0.003):
    """The texels of ``select`` at most ``depth`` behind the frontmost one at their height: what is seen."""
    y, z = bake.position[..., 1][select], bake.position[..., 2][select]
    bins = np.floor(y / step).astype(int)
    bins -= bins.min()
    front = np.full(bins.max() + 1, -np.inf)
    np.maximum.at(front, bins, z)
    out = np.zeros_like(select)
    out[select] = z >= front[bins] - depth
    return out


@pytest.mark.parametrize("name", ["Business_Male_02", "Business_Male_05", "Business_Male_07"])
def test_a_tie_and_its_knot_are_left_alone(library, name):
    model = _avatar(library, name)
    regions = R.body_regions(model)
    labels, p = regions.labels, regions.bake.position
    neck = model.skeleton.rest_global[model.skeleton.find("Neck"), 1, 3]
    with np.errstate(invalid="ignore"):
        middle = regions.bake.covered & (labels != R.SKIN) & (np.abs(p[..., 0]) < 0.01) & (p[..., 2] > 0.0)
        knot = _frontmost(regions.bake, middle & (p[..., 1] > neck - 0.06) & (p[..., 1] < neck - 0.02))
        tie = _frontmost(regions.bake, middle & (p[..., 1] > neck - 0.15) & (p[..., 1] < neck - 0.10))
    assert knot.sum() > 100 and _share(knot, labels, "other") > 0.9  # (the knot took the jacket's colour)
    assert _share(tie, labels, "other") > 0.9


@pytest.mark.parametrize("name", ["Business_Male_07", "Male_Adult_12", "Male_Adult_14"])
def test_belts_are_left_alone(library, name):
    from bifold_demo.characters.outfit.tailor import luminance

    model = _avatar(library, name)
    regions = R.body_regions(model)
    labels, y = regions.labels, regions.bake.position[..., 1]
    hip = model.skeleton.rest_global[model.skeleton.find("Pelvis"), 1, 3]
    lum = luminance(R._texture_rgb(model.primitives[regions.primitive], regions.size))
    with np.errstate(invalid="ignore"):
        waist = np.isin(labels, [R.BOTTOMS, R.OTHER]) & (y > hip - 0.03) & (y < hip + 0.2)
    leather = waist & (lum < 0.2)  # the belts are dark leather on lighter trousers
    assert leather.sum() > 500 and _share(leather, labels, "other") > 0.8


def test_a_colour_blocked_hoodie_takes_a_fill_on_every_block(library):
    from bifold_demo.characters.outfit.tailor import luminance

    tailor = Tailor(_avatar(library, "Male_Adult_17"))
    regions = tailor.regions
    blocks = R.colour_blocks(regions.labels, regions.colours, tailor.original[..., :3])
    top = cv2.erode(regions.hard("top").astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    ids, counts = np.unique(blocks[top], return_counts=True)
    big = ids[counts > 0.05 * top.sum()]
    assert len(big) == 3  # grey, navy and black
    white = luminance(tailor.texture({"top": "white"}))
    bright = [float(np.median(white[top & (blocks == b)])) for b in big]
    assert min(bright) > 0.75 * max(bright) and min(bright) > 0.6  # each block white, not just the grey one


def test_a_preset_on_a_real_avatar_keeps_its_detail_and_skin(library):
    model = _avatar(library, "Business_Male_02")
    tailor = Tailor(model)
    tailor.layers, tailor.surface  # noqa: B018
    start = time.perf_counter()
    rgba = tailor.texture("charcoal suit")
    seconds = time.perf_counter() - start
    original, regions = tailor.original, tailor.regions
    assert np.array_equal(rgba[regions.skin], original[regions.skin])
    top = cv2.erode(regions.hard("top").astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)

    def high(image):
        lum = image[..., :3].astype(np.float32).mean(axis=2)
        return (lum - cv2.GaussianBlur(lum, (0, 0), 3.0))[top]

    assert np.corrcoef(high(original), high(rgba))[0, 1] > 0.6  # the stitching, seams and weave stay
    assert abs(rgba[top][:, :3].mean() - np.mean(color("charcoal"))) < 25
    assert seconds < 1.0


def test_a_dressed_avatar_renders_in_its_new_clothes(gl, library):
    from bifold_demo.scene import Camera, Scene
    from bifold_demo.scene.renderer import Renderer

    model = _avatar(library, "Business_Male_02")
    images = []
    for outfit in (None, "trade-fair staff"):
        person = Character(model, behaviour=lambda t: CharacterState((0, 0, 0), 0.0, []), name="p")
        if outfit:
            dress(person, outfit)
        scene = Scene(background=(255, 255, 255), ambient=0.6)
        scene.add(person)
        scene.update(0.0, 0.0)
        scene.camera = Camera(position=(0, 1.2, 1.6), target=(0, 1.2, 0), fov=30)
        images.append(Renderer(120, 120, samples=0).render(scene).astype(int))
    before, after = (image[50:70, 52:68].reshape(-1, 3).mean(axis=0) for image in images)  # the chest, BGR
    assert after[1] > after[2] + 20 and abs(before[1] - before[2]) < 25  # teal now, dark blue-grey before
