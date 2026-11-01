"""Head textures from photos, stages 5 to 7, on synthetic data (no MediaPipe, no OpenGL, no avatars, no models).

The synthetic head of ``test_headtex`` and its "photos" of ``test_headtex_fuse`` (a z-buffer of the head from known
cameras) for the completion; the small tube person of ``test_outfit`` (its hands are skin) for the body's skin; a
stand-in for the diffusers pipeline; stand-ins for the face recognition and the renders of the identity check.
"""

import types

import cv2
import numpy as np
import pytest
import test_headtex_fuse as fuse_tests
from test_headtex import CENTRE, synthetic_head_model
from test_headtex_fuse import SKIN, TEX, cameras_for, lab_of, session

from bifold_demo.characters import uvbake
from bifold_demo.characters.headtex import body, fuse, light
from bifold_demo.characters.headtex import completion as stage5
from bifold_demo.characters.headtex import person as stages
from bifold_demo.characters.headtex import texture as stage4

isolated, truth, avatar = fuse_tests.isolated, fuse_tests.truth, fuse_tests.avatar  # (their fixtures, used here too)


def symmetric_person(model) -> np.ndarray:
    """A person's texture for the synthetic head (RGB): skin a little darker than the avatar's, with stripes across
    the head (by the height) and rings round the face's middle (by the distance from it): the same on both sides."""
    bake = uvbake.bake(model, 0, TEX)
    p = np.nan_to_num(bake.position)
    pattern = (1.0 + 0.22 * np.sin(2.0 * np.pi * (p[..., 1] - CENTRE[1]) / 0.025)) * (
        1.0 + 0.15 * np.cos(2.0 * np.pi * np.abs(p[..., 0] - CENTRE[0]) / 0.04))
    return np.clip(SKIN * 0.85 * pattern[..., None], 0, 255).astype(np.uint8)


class FakePipeline:
    """Stands in for a diffusers image-editing pipeline (FLUX.2 [klein]'s call signature): gives back a darker skin
    with a checker of its own, or raises ``fail``; remembers its calls."""

    def __init__(self, fail=None):
        self.calls, self.fail = [], fail

    def __call__(self, image, prompt, height, width, num_inference_steps, guidance_scale, generator):
        from PIL import Image

        self.calls.append({"image": image, "prompt": prompt, "size": (width, height), "steps": num_inference_steps})
        if self.fail is not None:
            raise self.fail
        yy, xx = np.mgrid[0:height, 0:width]
        checker = np.where(((xx // 4) + (yy // 4)) % 2 == 0, 1.15, 0.85)[..., None]
        rgb = np.clip(np.array([150.0, 105.0, 85.0]) * checker, 0, 255).astype(np.uint8)
        return types.SimpleNamespace(images=[Image.fromarray(rgb)])


# -- stage 5 ----------------------------------------------------------------------------------------------------


def test_the_mirror_map_pairs_the_two_sides_of_the_face(avatar):
    model, _head, atlas = avatar
    mirror = stage5.mirror_map(atlas, model)
    assert stage5.mirror_map(atlas, model) is mirror  # once per atlas
    has = mirror >= 0
    assert has[atlas.photo_parts].mean() > 0.9 and not has[~atlas.photo_parts].any()
    positions = atlas.bake.position.reshape(-1, 3)
    rows, cols = np.nonzero(has)
    mine = positions[rows * atlas.size + cols]
    theirs = positions[mirror[rows, cols]]
    flipped = mine.copy()
    flipped[:, 0] = 2.0 * CENTRE[0] - mine[:, 0]  # the synthetic face's plane: x = 0
    assert np.percentile(np.linalg.norm(theirs - flipped, axis=1), 95) < 0.002
    twice = mirror.ravel()[mirror[rows, cols]]  # the mirror's mirror: the texel itself, or one next to it
    back = positions[np.where(twice >= 0, twice, 0)]
    assert np.median(np.linalg.norm(back - mine, axis=1)[twice >= 0]) < 0.001


def test_the_mirror_completes_the_side_no_photo_shows(avatar, truth):
    """Photos from the front and of the person's right side: their left side, which no photo shows well, takes the
    right side's colours mirrored (the person's stripes and rings, not the avatar's checker); what the photos show
    surely stays as it was."""
    model, head, atlas = avatar
    person = symmetric_person(model)
    result = session(model, head, truth, cameras_for([0.0, 45.0]), texture=person)
    done = stage5.complete(result, model)
    x = np.nan_to_num(atlas.bake.position[..., 0] - CENTRE[0])
    filled = done.mirrored > 0.5
    assert filled.sum() > 1000 and (x[filled] > 0.0).mean() > 0.9  # the person's left side
    source = result.rgba[..., :3].reshape(-1, 3)[stage5.mirror_map(atlas, model)[filled]]
    assert np.median(light.delta_e(lab_of(done.rgba[..., :3][filled]), lab_of(source))) < 1.5  # the mirror's
    want = lab_of(person[filled])[:, 0]  # the person's pattern (its level is the photos', known up to exposure)
    before = np.corrcoef(lab_of(result.rgba[..., :3][filled])[:, 0], want)[0, 1]
    after = np.corrcoef(lab_of(done.rgba[..., :3][filled])[:, 0], want)[0, 1]
    assert after > 0.8 and after > before + 0.3, (before, after)
    untouched = atlas.covered & (done.mirrored == 0)
    assert np.abs(done.rgba[untouched].astype(int) - result.rgba[untouched]).max() <= 1
    assert (done.confidence >= result.confidence - 1e-6).all()
    assert (done.trimap[filled] != fuse.GENERATE).mean() > 0.8  # second-hand, but no longer "generate"
    assert done.method == "mirror" and done.share()["mirrored"] > 0.02 and done.deterministic is None


def test_the_generative_layer_fills_only_what_is_left_and_counts_less(avatar, truth, monkeypatch):
    """One photo from the front: the back of the head is left to generate. A stand-in pipeline's darker, checkered
    skin goes only there, tone-matched to the deterministic result and at 40 % at most; a pipeline that fails or a
    model that cannot load leaves the deterministic result, with a warning."""
    model, head, atlas = avatar
    result = session(model, head, truth, cameras_for([0.0]))
    plain = stage5.complete(result, model)
    to_fill = (plain.trimap == fuse.GENERATE) & atlas.photo_parts
    assert to_fill.sum() > 0.2 * atlas.covered.sum()
    fake = FakePipeline()
    done = stage5.complete(result, model, inpainter=stage5.FluxInpainter(pipeline=fake, size=TEX))
    assert len(fake.calls) == 1 and done.method == "mirror + flux2-klein" and not done.warnings
    call = fake.calls[0]
    assert call["size"] == (TEX, TEX) and len(call["image"]) == 2  # the texture and the photo as a reference
    painted = np.asarray(call["image"][0])
    assert (painted[to_fill] == stage5.GREY).all() and "skin" in call["prompt"]
    assert np.array_equal(done.deterministic, plain.rgba)
    changed = np.abs(done.rgba[..., :3].astype(int) - plain.rgba[..., :3]).max(axis=2) > 1
    assert changed.sum() > 1000 and not changed[atlas.covered & ~to_fill].any()  # never the photos' texels
    assert done.generated.max() <= stage5.GENERATED_WEIGHT + 1e-6 and not done.generated[~to_fill].any()
    core = to_fill & (done.generated > 0.3)

    def tone(rgb):  # the geometric mean colour, in Lab
        return light.lab(np.exp(np.log(np.maximum(light.to_linear(rgb), 1e-4)).mean(axis=0)))

    old, new = plain.rgba[..., :3][core], done.rgba[..., :3][core]
    assert light.delta_e(tone(old), tone(new)) < 2.0  # its detail, not its tone
    model_checker = ((np.mgrid[0:TEX, 0:TEX] // 4).sum(axis=0) % 2 == 0)[core]  # where the model's was lighter
    lift = lab_of(new)[:, 0] - lab_of(old)[:, 0]
    assert lift[model_checker].mean() > lift[~model_checker].mean() + 2.0  # (its checker shows, faintly)
    assert done.without_generated().rgba is done.deterministic
    failing = stage5.complete(result, model, inpainter=stage5.FluxInpainter(
        pipeline=FakePipeline(RuntimeError("out of memory")), size=TEX))
    assert failing.method == "mirror" and np.array_equal(failing.rgba, plain.rgba)
    assert any("out of memory" in w for w in failing.warnings)
    from bifold_demo.people import generator

    def unreachable(self, *args, **kwargs):
        raise OSError("Hugging Face is unreachable")

    monkeypatch.setattr(generator.ImageGenerator, "__init__", unreachable)
    unloaded = stage5.complete(result, model, inpainter=stage5.FluxInpainter(size=TEX))
    assert np.array_equal(unloaded.rgba, plain.rgba) and any("unreachable" in w for w in unloaded.warnings)


def test_prompt_words_for_skin_and_hair():
    def tone(lab):
        return light.Tone(np.array(lab, float))

    assert stage5.skin_words(tone([70, 15, 20])) == "fair" and stage5.skin_words(tone([30, 12, 18])) == "deep brown"
    assert stage5.hair_words(tone([12, 2, 3])) == "black" and stage5.hair_words(tone([70, 4, 28])) == "blond"
    assert stage5.hair_words(tone([60, 1, 2])) == "grey" and stage5.hair_words(None) == "matching"


# -- stage 6 ----------------------------------------------------------------------------------------------------


def _tube_person():
    """``test_outfit``'s small person, its hands (the skin) painted with creases."""
    from test_outfit import _person

    texture = _person().primitives[0].texture.copy()
    from bifold_demo.characters.outfit import regions as R

    skin = R.body_regions(_person(texture), cache=False).skin
    rows = np.nonzero(skin)[0]
    texture[..., :3][skin] = np.clip(texture[..., :3][skin] * (1.0 + 0.2 * np.sin(rows / 2.0))[:, None], 0, 255)
    return _person(texture), texture


def test_the_body_takes_the_faces_skin_and_keeps_its_detail():
    from bifold_demo.characters.outfit import regions as R

    model, texture = _tube_person()
    regions = R.body_regions(model, cache=False)
    skin = regions.skin
    target = light.Tone(np.array([45.0, 17.0, 24.0]))  # a darker person than the avatar (Lab 65)
    result = types.SimpleNamespace(rgba=np.zeros((8, 8, 4), np.uint8), skin=target, own_skin=None, hair=None,
                                   atlas=None, primitive="head", confidence=None)
    match = body.match_body(result, model)
    assert set(match.textures()) == {"head", "p000_body"} and not match.warnings
    got = light.trimmed_median(light.lab(light.to_linear(match.body[..., :3][skin])))[0]
    assert light.delta_e(got, target.lab) < 1.5
    inner = cv2.erode(skin.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    old, new = light.to_linear(texture[..., :3][inner]), light.to_linear(match.body[..., :3][inner])
    bright = (old > 0.05).all(axis=1)
    ratio = np.log(new[bright] / old[bright])
    assert (np.std(ratio, axis=0) < 0.03).all()  # one gain per channel: the creases stay as they were
    assert np.allclose(np.exp(ratio.mean(axis=0)), match.gain, rtol=0.05)
    clothes = regions.bake.covered & (cv2.dilate(skin.astype(np.uint8), np.ones((7, 7), np.uint8)) == 0)
    assert np.array_equal(match.body[clothes], texture[clothes])
    assert np.array_equal(match.body[..., 3], texture[..., 3])


def test_an_outfit_stays_on_over_the_persons_skin():
    from bifold_demo.characters.character import Character
    from bifold_demo.characters.outfit import dress, outfit_of
    from bifold_demo.characters.outfit import regions as R

    model, texture = _tube_person()
    skin = R.body_regions(model, cache=False).skin
    matched = texture.copy()
    matched[..., :3][skin] = (120, 80, 60)
    person = stages.PersonTextures(model.name, {"p000_body": matched}, "head", "p000_body")
    one = Character(model, name="one", shadow=False)
    dress(one, "all black")
    worn = person.wear(one)
    current = one.texture_overrides["p000_body"]
    assert outfit_of(one).name == "all black" and current.base is worn["p000_body"]
    assert np.array_equal(current.rgba[skin], matched[skin])
    stages.take_off_textures(one, worn)
    current = one.texture_overrides["p000_body"]
    assert outfit_of(one).name == "all black" and np.array_equal(current.rgba[skin], texture[skin])
    two = Character(model, name="two", shadow=False)
    worn = person.wear(two)
    assert two.texture_overrides["p000_body"] is worn["p000_body"]
    stages.take_off_textures(two, worn)
    assert not two.texture_overrides


def test_seam_offsets_follow_the_differences_round_the_neck():
    rng = np.random.default_rng(3)
    angles = rng.uniform(0.0, 1.5 * np.pi, 80)  # no pairs in the last quarter (a collar)
    true = 0.2 + 0.1 * np.cos(angles)
    table = body.sector_offsets(angles, (true + rng.normal(0.0, 0.01, len(angles)))[:, None])
    assert table.shape == (body.SECTORS, 1)
    assert np.abs(body.at_angles(table, angles)[:, 0] - true).max() < 0.05
    gap = body.at_angles(table, np.array([1.75 * np.pi]))[0, 0]
    assert 0.1 < gap < 0.35  # led by its neighbours and the median
    ends = body.at_angles(table, np.array([2.0 * np.pi - 1e-6, 1e-6]))[:, 0]
    assert abs(ends[0] - ends[1]) < 0.01  # round the ring, no step


def test_hair_is_recoloured_and_skin_is_not():
    blond = np.array([[215, 180, 120]] * 5 + [[180, 150, 95]] * 5, np.uint8)
    skin = np.array([[220, 170, 140]], np.uint8)  # close to blond hair in Lab
    lab = light.lab(light.to_linear(np.vstack([blond, skin])))
    hair_lab = np.median(lab[:10], axis=0)
    weight = body.hair_like(lab, hair_lab, lab[10])
    assert (weight[:10] > 0.9).all() and weight[10] == 0.0
    assert body.hair_like(lab, hair_lab)[10] > 0.5  # (without the skin's colour it would have counted as hair)
    dark = np.array([18.0, 4.0, 6.0])
    out = light.lab(light.to_linear(body.recolour(np.vstack([blond, skin]), hair_lab, dark, weight)))
    assert abs(np.median(out[:10, 0]) - 18.0) < 4.0 and np.abs(np.median(out[:10, 1:], axis=0) - dark[1:]).max() < 4
    assert out[0, 0] > out[5, 0]  # the lighter strands stay lighter
    assert light.delta_e(out[10], lab[10]) < 0.5


# -- stage 7 and the whole way ----------------------------------------------------------------------------------


class FaceEngine:
    """Stands in for the face recognition: an image with anything in it is one face; its embedding is its mean
    colour (with a common part)."""

    def detect(self, image, min_size=0):
        return [object()] if image.any() else []

    def embed(self, image, face):
        vector = np.r_[image.reshape(-1, 3).mean(axis=0), 30.0]
        return vector / np.linalg.norm(vector)


def test_the_identity_check_compares_renders_with_the_photos(monkeypatch):
    model = synthetic_head_model()
    red = np.zeros((20, 20, 3), np.uint8)
    red[..., 2] = 200
    renders = [red, np.zeros_like(red)]  # the front looks like the photo, the other shows no face
    monkeypatch.setattr(stage4, "preview", lambda m, textures, renderer=None, yaws=(): np.hstack(renders))
    out = stages.identity(model, {}, [red, np.zeros_like(red)], engine=FaceEngine(), yaws=(0.0, 30.0))
    assert out["front"] > 0.99 and np.isnan(out["turned 30"])
    with pytest.raises(ValueError):
        stages.identity(model, {}, [np.zeros_like(red)], engine=FaceEngine())


def test_finishing_checks_and_drops_a_generated_layer_that_lowers_the_identity(avatar, truth, monkeypatch):
    model, head, _atlas = avatar
    result = session(model, head, truth, cameras_for([0.0, 40.0]))
    scores = iter([{"front": 0.2}, {"front": 0.40}, {"front": 0.55}, {"front": 0.55}])  # own; with, without; final
    monkeypatch.setattr(stages, "identity", lambda *args, **kwargs: next(scores))
    inpainter = stage5.FluxInpainter(pipeline=FakePipeline(), size=TEX)
    person = stages.finish(result, model, inpainter=inpainter, check=True, engine=object())
    assert any("lowered the identity check (0.55 -> 0.40): dropped" in w for w in person.warnings)
    assert person.completion.method == "mirror" and np.array_equal(person.head, person.completion.rgba)
    assert person.checks["identity"] == {"front": 0.55}
    assert person.checks["identity of the avatar's own head"] == {"front": 0.2}
    assert any("not matched" in w for w in person.warnings)  # the synthetic head has no body
    assert set(person.textures) == {"s_head"} and "identity (SFace" in person.summary()
    scores = iter([{"front": 0.2}, {"front": 0.56}, {"front": 0.55}, {"front": 0.56}])
    kept = stages.finish(result, model, inpainter=inpainter, check=True, engine=object())
    assert kept.completion.method == "mirror + flux2-klein"
    plain = stages.finish(result, model)  # no model, no check
    assert "identity" not in plain.checks and plain.completion.method == "mirror"


def test_a_persons_textures_are_made_once_for_the_same_photos(tmp_path, monkeypatch):
    model = synthetic_head_model()
    rng = np.random.default_rng(5)
    photos = [tmp_path / "a.png", tmp_path / "b.png"]
    for photo in photos:
        cv2.imwrite(str(photo), rng.integers(0, 255, (24, 24, 3), np.uint8))
    calls = []

    def head_texture(photos, model, **options):
        calls.append(list(photos))
        return "stages 0-4"

    def finish(result, model, **options):
        return stages.PersonTextures(model.name, {"s_head": np.full((8, 8, 4), len(calls), np.uint8)}, "s_head",
                                     checks={"neck seam before": 3.0, "neck seam after": float("nan")})

    monkeypatch.setattr(stage4, "head_texture", head_texture)
    monkeypatch.setattr(stages, "finish", finish)
    cache = tmp_path / "cache"
    first = stages.person_textures(photos, model, cache=cache)
    again = stages.person_textures([str(p) for p in photos], model, cache=cache)
    assert len(calls) == 1 and again.cached and not first.cached
    assert np.array_equal(again.head, first.head) and again.checks["neck seam before"] == 3.0
    assert again.checks["neck seam after"] is None  # (NaN is kept as JSON's null)
    cv2.imwrite(str(photos[1]), rng.integers(0, 255, (24, 24, 3), np.uint8))  # a photo changed: made again
    stages.person_textures(photos, model, cache=cache)
    assert len(calls) == 2
    for png in cache.glob("*/s_head.png"):
        png.write_bytes(b"damaged")
    stages.person_textures(photos, model, cache=cache)
    assert len(calls) == 3
    store = stages.PersonCache(tmp_path / "small", keep=1)
    store.put("one", first)
    store.put("two", first)
    assert [p.name for p in (tmp_path / "small").iterdir()] == ["two"]


def test_person_textures_are_saved_worn_and_kept_by_a_created_character(tmp_path):
    from bifold_demo.characters.character import Character
    from bifold_demo.characters.created import CharacterBook, load

    model = synthetic_head_model()
    image = np.random.default_rng(6).integers(0, 255, (16, 16, 4), np.uint8)
    person = stages.PersonTextures(model.name, {"s_head": image}, "s_head",
                                   checks={"face to hands before": 7.0, "face to hands after": 0.5})
    back = stages.PersonTextures.load(person.save(tmp_path / "saved"))
    assert np.array_equal(back.head, image) and back.checks == person.checks and back.cached
    one = Character(model, name="ada", shadow=False)
    worn = person.wear(one)
    assert np.array_equal(one.texture_overrides["s_head"].rgba, image)
    stages.take_off_textures(one, worn)
    assert "s_head" not in one.texture_overrides
    path = CharacterBook(tmp_path / "created").save(person.created("Ada at the fair", face="Ada"))
    loaded = load(path)
    assert loaded.face == "Ada" and loaded.avatar == model.name and np.array_equal(loaded.textures["s_head"], image)
    assert "the face's skin against the hands': delta E 7.0 before stage 6, 0.5 after" in person.summary()
