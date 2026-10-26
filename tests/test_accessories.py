"""Items for the 3D people: fitting them, wearing them, the arm holding a sign, the looks and drawing them."""

import numpy as np
import pytest
from test_behaviour import synthetic_model
from test_yoga import synthetic_biped

from bifold_demo.characters import accessories
from bifold_demo.characters.accessories import Wardrobe, dress, wearing
from bifold_demo.characters.character import Character, skin_vertices
from bifold_demo.characters.model import AnimationClip, CharacterState
from bifold_demo.characters.performer import Performer


@pytest.fixture(scope="module")
def biped():
    return synthetic_biped()


def _person(model, behaviour=None):
    return Character(model, behaviour or (lambda t: CharacterState((0.0, 0.0, 0.0), 0.0, [])), name="p")


def _top(model, bones):
    sk = model.skeleton
    keep = [sk.index(b) for b in bones]
    points = [p.positions[np.isin(p.joints[:, 0], keep)] for p in model.primitives]
    return np.concatenate([q for q in points if len(q)])


def test_items_fit_the_head_and_the_hand(biped):
    fit = accessories.fit(biped)
    head = _top(biped, ["Bip01 Head"])
    hat = (fit.rest[fit.head] @ fit.hat_offset @ np.c_[fit.hat.positions, np.ones(len(fit.hat.positions))].T).T
    assert hat[:, 1].max() > head[:, 1].max()  # over the head ...
    assert hat[:, 1].min() < head[:, 1].max() - 0.005  # ... and down around it
    glasses = (fit.rest[fit.head] @ fit.glasses_offset
               @ np.c_[fit.glasses.positions, np.ones(len(fit.glasses.positions))].T).T
    assert glasses[:, 2].max() > head[:, 2].max()  # in front of the face
    assert np.ptp(glasses[:, 0]) < 0.25 and abs(glasses[:, 0].mean()) < 0.01  # a pair, centred
    assert fit.shirt is not None and fit.shirt.texture is None and len(fit.shirt.indices) > 50


def test_dress_puts_items_on_and_takes_them_off(biped, tmp_path):
    person = _person(biped)
    assert dress(person, "Hat", "navy").name == "navy hat"  # a built-in look by name
    dress(person, "t-shirt", "#ff8800")  # any texture value
    dress(person, "glasses", True)  # the first look
    image = tmp_path / "poster.png"
    import cv2

    cv2.imwrite(str(image), np.full((40, 60, 3), 200, np.uint8))
    dress(person, "sign", image)
    assert sorted(wearing(person)) == ["glasses", "hat", "sign", "tshirt"]
    assert [p.name for p in person.props] == ["p.hat", "p.glasses", "p.sign", "p.sign_stick"]
    assert len(person.worn) == 1 and person.worn[0].texture is not None
    dress(person, "hat", None)
    dress(person, "tshirt", False)
    assert sorted(wearing(person)) == ["glasses", "sign"] and person.worn == []
    with pytest.raises(KeyError):
        dress(person, "scarf", True)
    with pytest.raises(FileNotFoundError):
        dress(person, "sign", tmp_path / "missing.png")
    with pytest.raises(ValueError):
        dress(_person(synthetic_model()), "hat", True)  # not a Biped


def test_the_t_shirt_moves_with_the_body(biped):
    person = _person(biped)
    dress(person, "tshirt", "white")
    shirt = person.worn[0]
    model = accessories.fit(biped).shirt
    bent = biped.skeleton.rest_local.copy()
    spine = biped.skeleton.index("Bip01 Spine1")
    c, s = np.cos(0.6), np.sin(0.6)
    bent[spine, :3, :3] = np.array([[1, 0, 0], [0, c, -s], [0, s, c]]) @ bent[spine, :3, :3]
    palette = biped.skinning_matrices(biped.skeleton.global_transforms(bent))
    worn = type(biped)(biped.name, biped.skeleton, [shirt], biped.inverse_bind)
    moved = skin_vertices(worn, palette)[0]
    assert np.abs(moved - model.positions).max() > 0.05  # bending the spine moves the shirt like the body


def test_the_sign_is_held_up_and_put_away_for_yoga(biped):
    frames = 21
    clip = AnimationClip("yoga_test", 10.0, ["Bip01"], np.tile((0.0, 0.0, 0.0, 1.0), (frames, 1, 1)),
                         np.zeros((frames, 1, 3)), metadata={"pose": "cow", "hold": (0.5, 1.5), "view": 0.0})
    performer = Performer(lambda t: CharacterState((0.0, 0.0, 0.0), 0.0, []))
    person = _person(biped, performer)
    hand = biped.skeleton.index("Bip01 R Hand")
    person.update(0.0)
    hanging = person.pose[hand, :3, 3].copy()
    dress(person, "sign", "welcome", t=0.0)
    person.update(0.25)
    halfway = person.pose[hand, :3, 3].copy()
    person.update(1.0)
    holding = person.pose[hand, :3, 3].copy()
    assert holding[2] > hanging[2] + 0.1 and holding[1] > hanging[1] + 0.1  # the hand up in front
    assert 0.0 < np.linalg.norm(halfway - hanging) < np.linalg.norm(holding - hanging)  # lifted gradually
    board = person.props[0]
    assert board.current_visible and board.matrices()[0][1, 3] > holding[1] + 0.2  # the board above the fist
    act = performer.perform(clip, 2.0, heading=0.0)
    person.update(act.clip_start + 1.0)
    assert not board.current_visible  # put away while doing the pose
    assert np.linalg.norm(person.pose[hand, :3, 3] - hanging) < 0.02
    person.update(act.end + 2.0)
    assert board.current_visible and np.allclose(person.pose[hand, :3, 3], holding, atol=1e-6)
    dress(person, "sign", None, t=act.end + 2.0)
    person.update(act.end + 4.0)
    assert np.allclose(person.pose[hand, :3, 3], hanging, atol=1e-6) and person.props == []


def test_the_wardrobe_offers_your_files_first(tmp_path):
    import cv2

    for name in ("hat.png", "hat_red.jpg", "shirt-blue.png", "notes.txt"):
        cv2.imwrite(str(tmp_path / name), np.zeros((8, 8, 3), np.uint8)) if name.endswith(("png", "jpg")) else \
            (tmp_path / name).write_text("x")
    (tmp_path / "sign").mkdir()
    cv2.imwrite(str(tmp_path / "sign" / "poster.png"), np.zeros((8, 8, 3), np.uint8))
    wardrobe = Wardrobe(tmp_path, textures={"glasses": ["#ff0000"]})
    assert [name for name, _ in wardrobe.looks("hat", now=0.0)][:2] == ["hat", "hat_red"]
    assert wardrobe.looks("tshirt", now=0.0)[0][0] == "shirt-blue"
    assert wardrobe.looks("sign", now=0.0)[0][0] == "poster"
    assert wardrobe.looks("glasses", now=0.0)[0] == ("#ff0000", "#ff0000")
    assert [name for name, _ in wardrobe.looks("hat", now=0.0)][2:] == list(accessories.built_in()["hat"])
    cv2.imwrite(str(tmp_path / "hat_new.png"), np.zeros((8, 8, 3), np.uint8))
    assert "hat_new" not in [name for name, _ in wardrobe.looks("hat", now=0.5)]  # checked once a second
    assert "hat_new" in [name for name, _ in wardrobe.looks("hat", now=2.0)]
    assert wardrobe.look("sign", "Poster") == tmp_path / "sign" / "poster.png"
    assert wardrobe.look("sign", "#00ff00") == "#00ff00"
    assert Wardrobe(None, built_in=False).looks("hat") == []


def test_dressed_people_are_drawn_with_their_items(gl, biped):
    from bifold_demo.scene import Camera, Scene
    from bifold_demo.scene.renderer import Renderer

    scene = Scene(background=(255, 255, 255), camera=Camera(position=(0.0, 1.2, 3.0), target=(0.0, 1.1, 0.0)))
    person = scene.add(_person(biped))
    scene.update(0.0, 0.0)
    renderer = Renderer(160, 160, samples=0)
    plain = renderer.render(scene)
    for item, look in (("hat", "#ff0000"), ("glasses", "sunglasses"), ("tshirt", "#00ff00"), ("sign", "#0000ff")):
        dress(person, item, look, t=0.0)
    scene.update(2.0, 2.0)
    dressed = renderer.render(scene)
    b, g, r = (dressed[..., k].astype(int) for k in range(3))
    green, red, blue = (g > r + 60) & (g > b + 60), (r > g + 60) & (r > b + 60), (b > r + 60) & (b > g + 60)
    assert green.sum() > 20 and red.sum() > 5 and blue.sum() > 20 and (dressed != plain).any()
