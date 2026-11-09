"""The character creator: its logic without a window (an avatar, a preset, a garment's colour and pattern, a fabric from a
description, items, a face, saving, loading and deleting), its keys, the demo's creator mode with a saved character
joining a small fair, and its screen (OpenGL and the prepared avatars).

Synthetic avatars, the keyword fabric (no model) and a stand-in face painter: no real faces. The screen test skips
without OpenGL or the prepared avatars (``BIFOLD_CHARACTERS_TEST_DIR``).
"""

import os
from pathlib import Path

import cv2
import numpy as np
import pytest
from test_outfit import _person
from test_yoga import synthetic_biped

from bifold_demo import creator as C
from bifold_demo.characters import accessories
from bifold_demo.characters.created import CharacterBook
from bifold_demo.characters.model import CharacterModel
from bifold_demo.characters.outfit import FabricGenerator, Pattern, Solid, presets
from bifold_demo.characters.outfit.fabric import Fabric
from bifold_demo.creator import ROWS, Creator, FaceTextures
from bifold_demo.display.window import BACKSPACE, BACKTAB, DOWN, LEFT, RIGHT, TAB, UP, key_code, typed_char

CHARACTERS = Path(os.environ.get("BIFOLD_CHARACTERS_TEST_DIR", "/nonexistent")).expanduser()


def _tube_woman() -> CharacterModel:
    """The outfit tests' tube person (a body to dress) with the box limbs and head of the synthetic Biped (items)."""
    tube, boxes = _person(), synthetic_biped()
    return CharacterModel("Tube_Woman", tube.skeleton, [*tube.primitives, *boxes.primitives], tube.inverse_bind)


def _box_man() -> CharacterModel:
    """The synthetic Biped, its boxes named like hair: it has no body to dress (but it wears items)."""
    model = synthetic_biped()
    model.name = "Box_Man"
    for primitive in model.primitives:
        primitive.name = primitive.name.replace("box", "hair")
    return model


def _head(model) -> str:
    """The box on the head bone: the "head" the stand-in painter paints."""
    return next(p.name for p in model.primitives if p.joints[0, 0] == model.skeleton.index("Bip01 Head"))


MODELS = {"Tube_Woman": _tube_woman(), "Box_Man": _box_man()}
HEAD = _head(MODELS["Tube_Woman"])


class Library:
    """Stands in for the prepared avatars: the tube woman (she can be dressed) and the box man (he cannot)."""

    def available(self):
        return {"Box_Man": {"sex": "male", "age_group": "adult"}, "Tube_Woman": {"sex": "female", "age_group": "adult"}}

    def load(self, name):
        if name not in MODELS:
            raise KeyError(f"the avatar {name!r} is not prepared")
        return MODELS[name]

    def motions(self):
        raise RuntimeError("no clips here")


class Faces:
    def __init__(self, people=("Ada", "Grace")):
        self.known = list(people)

    def people(self):
        return self.known

    def photos(self, name):
        return [Path(f"{name.lower()}_{k}.png") for k in range(2)]  # never read: the painter is a stand-in


def painter(calls=None, colour=(200, 30, 160)):
    """A face painter that paints the head box in ``colour`` (and notes what it was asked for)."""

    def paint(photos, model):
        if calls is not None:
            calls.append((list(photos), model.name))
        rgba = np.zeros((8, 8, 4), np.uint8)
        rgba[..., :3], rgba[..., 3] = colour, 255
        return FaceTextures({_head(model): rgba}, "a stand-in")

    return paint


def _creator(tmp_path, **options) -> Creator:
    options.setdefault("painter", painter())
    options.setdefault("faces", Faces())
    options.setdefault("fabrics", FabricGenerator(None, cache=False))  # no model: the keyword fabric
    creator = Creator(CharacterBook(tmp_path / "created"), library=Library(), threads=False, **options)
    creator.open()
    return creator


def _body(creator):
    return creator.person.texture_overrides[creator.prepared.body].rgba


# -- the logic ------------------------------------------------------------------------------------------------------


def test_an_avatar_is_prepared_and_put_on_the_turntable(tmp_path):
    creator = _creator(tmp_path)
    assert creator.avatar == "Box_Man" and creator.busy() == ["preparing Box_Man (the first time a few seconds)"]
    assert creator.wait()
    assert creator.prepared.name == "Box_Man" and creator.prepared.tailor is None and creator.prepared.items
    assert creator.person.skinned_model is MODELS["Box_Man"]
    assert not creator.enabled("garment", 0) and not creator.enabled("pattern", 1)  # (a box man has no clothes)
    assert creator.pick("pattern", 1) == creator.prepared.why and creator.prepared.why.startswith("Box_Man")
    assert creator.set_avatar("Tube_Woman") == "avatar: Tube_Woman (preparing it ...)"
    assert creator.wait() and creator.prepared.garments == ("top", "bottoms", "shoes")
    assert not creator.enabled("garment", 1) and creator.select_garment("shirt") == "Tube_Woman has no shirt"
    assert creator.set_avatar("Nobody") == "avatar: Nobody (preparing it ...)"
    messages = creator.update() + creator.update()
    assert messages and messages[0].startswith("cannot prepare Nobody: KeyError") and creator.wait()


def test_presets_garments_patterns_colours_and_scale(tmp_path):
    creator = _creator(tmp_path)
    creator.set_avatar("Tube_Woman")
    assert creator.wait()
    assert creator.prepared.body not in creator.person.texture_overrides  # its own clothes until an outfit is chosen
    assert creator.apply_preset("trade-fair staff") == "outfit: trade-fair staff"
    assert creator.wait()
    staff = _body(creator).copy()
    assert creator.outfit_name == "trade-fair staff" and set(creator.fills) == {"top", "shirt", "bottoms", "shoes"}
    assert creator.current("preset") == list(presets()).index("trade-fair staff")
    assert creator.select_garment("top") == "garment: top"
    assert creator.set_pattern("tartan").startswith("top: tartan, teal and")
    assert creator.set_colour("red").startswith("top: tartan, red and")
    assert creator.set_colour("navy", second=True) == "top: tartan, red and navy"
    top = creator.fills["top"]
    assert isinstance(top, Pattern) and (top.primary, top.secondary) == ((192, 32, 40), (31, 42, 74))
    assert creator.outfit_name == "custom" and creator.current("pattern") == C.PATTERN_CHOICES.index("tartan")
    assert C.COLOURS[creator.current("colour")] == "red" and C.COLOURS[creator.current("colour 2")] == "navy"
    assert creator.scale_by(2.0) == "top: 0.3 m per repeat" and creator.fills["top"].scale == pytest.approx(0.3)
    assert creator.wait()
    tartan = _body(creator)
    assert not np.array_equal(tartan, staff)
    assert creator.set_pattern("plain") == "top: red" and isinstance(creator.fills["top"], Solid)
    assert creator.scale_by(2.0) == "the top has no pattern to scale"
    assert creator.set_pattern("own") == "top: the avatar's own" and "top" not in creator.fills
    assert creator.set_pattern("fabric").startswith("describe a fabric first")
    spec = creator.outfit().spec()
    assert spec["name"] == "custom" and set(spec["fills"]) == {"shirt", "bottoms", "shoes"}


def test_a_fabric_from_a_description_comes_as_variations(tmp_path):
    creator = _creator(tmp_path)
    creator.set_avatar("Tube_Woman")
    creator.select_garment("bottoms")
    assert creator.describe_fabric("red tartan wool") == "making a fabric of 'red tartan wool' for the bottoms ..."
    assert creator.busy() and creator.wait()
    assert creator.variations and creator.variations[0].preview  # the keyword preview first, at once
    fill = creator.fills["bottoms"]
    assert isinstance(fill, Fabric) and fill.text == "red tartan wool" and fill.method == "procedural"
    assert creator.last_fabric_note.startswith("fabric 'red tartan wool': procedural: tartan")
    assert "no model asked for" in creator.last_fabric_note  # why: it fell back to the keyword fabric
    assert creator.current("pattern") == C.PATTERN_CHOICES.index("fabric")
    assert creator.pick_variation(0).startswith("bottoms: procedural") and creator.variation == 0
    assert creator.scale_by(0.5).startswith("bottoms: ") and isinstance(creator.fills["bottoms"], Fabric)
    assert creator.describe_fabric("a gucci logo").startswith("Brand names can't be made into a fabric here")
    assert creator.describe_fabric("   ") == "describe a fabric, e.g. blue paisley silk"


def test_the_fabric_never_replaces_a_choice_made_meanwhile(tmp_path):
    from bifold_demo.characters.outfit.fabric import FabricJob, procedural

    jobs = []

    class Generator:  # its jobs are driven by hand here
        def start(self, text):
            jobs.append(FabricJob(text, text, [0], 64))
            return jobs[-1]

    creator = _creator(tmp_path, fabrics=Generator())
    creator.set_avatar("Tube_Woman")
    creator.describe_fabric("red tartan")
    job = jobs[-1]
    job._emit(procedural("red tartan", preview=True))
    creator.update()
    assert isinstance(creator.fills["top"], Fabric) and creator.variation == 0  # the preview, at once
    assert creator.busy()[-1] == "making the fabric: starting"
    job.progress = {"model": "FLUX.2-klein-4B", "seed": 1, "step": 2, "steps": 4}
    assert creator.fabric_status() == "FLUX.2-klein-4B, seed 1: step 2/4"
    creator.set_colour("navy")  # chosen by hand meanwhile
    job._emit(procedural("red tartan"))
    job._finish("done")
    creator.update()
    assert creator.fills["top"].rgb == (31, 42, 74) and len(creator.variations) == 2  # it stays
    creator.describe_fabric("white polka dots on black")
    jobs[-1]._emit(procedural("white polka dots on black", preview=True))
    creator.update()
    creator.set_name("Ada")
    creator.save()  # what is saved stays on: the best is not put on when the job ends
    jobs[-1]._emit(procedural("white polka dots on black"))
    jobs[-1]._finish("done")
    creator.update()
    assert creator.variation == 0 and creator.fills["top"] is not None


def test_items_and_a_face_go_on_the_person(tmp_path):
    calls = []
    creator = _creator(tmp_path, painter=painter(calls))
    creator.set_avatar("Tube_Woman")
    assert creator.set_item("hat", "navy") == "hat: navy"
    assert creator.set_item("sign", "welcome") == "sign: welcome"
    assert creator.wait()
    assert set(accessories.wearing(creator.person)) == {"hat", "sign"}
    assert creator.set_item("hat", None) == "hat: none" and set(accessories.wearing(creator.person)) == {"sign"}
    assert creator.current("sign") == creator.choices("sign").index("welcome")
    assert creator.set_face("Ada") == "making Ada's face (a few seconds) ..."
    messages = []
    while creator._jobs or creator._due():
        messages += creator.update()
    assert messages == ["Ada's face is on (a stand-in)"]
    assert calls == [([Path("ada_0.png"), Path("ada_1.png")], "Tube_Woman")]
    assert tuple(creator.person.texture_overrides[HEAD].rgba[0, 0, :3]) == (200, 30, 160)
    creator.set_avatar("Box_Man")  # another avatar: the face is made again for it, the items stay
    assert creator.wait() and calls[-1][1] == "Box_Man" and set(accessories.wearing(creator.person)) == {"sign"}
    head = _head(MODELS["Box_Man"])
    assert tuple(creator.person.texture_overrides[head].rgba[0, 0, :3]) == (200, 30, 160)
    assert creator.set_face(None) == "face: the avatar's own" and head not in creator.person.texture_overrides


def test_a_face_that_cannot_be_made_is_said_and_left_off(tmp_path):
    def broken(photos, model):
        raise ValueError("no face found in the photos")

    creator = _creator(tmp_path, painter=broken)
    creator.set_face("Grace")
    messages = []
    for _ in range(4):
        messages += creator.update()
    assert messages == ["cannot make Grace's face for Box_Man: ValueError: no face found in the photos"]
    assert creator.face is None and creator.wait()


def test_save_load_and_delete(tmp_path):
    creator = _creator(tmp_path)
    creator.set_avatar("Tube_Woman")
    creator.apply_preset("highland")
    creator.select_garment("bottoms")
    creator.describe_fabric("navy pinstripe")
    creator.set_item("glasses", "sunglasses")
    creator.set_face("Ada")
    assert creator.wait()  # (the fabric too: else the one in use when saving is saved)
    assert creator.save() == "type a name first (the Name row: Enter, type, Enter)"
    creator.set_name("Ada Lovelace")
    assert creator.save() == "saving Ada Lovelace ..."
    assert creator.wait()
    messages = creator.update()
    assert any(m.startswith("saved Ada Lovelace in ") for m in messages), messages
    assert creator.state() == "Ada Lovelace: saved"
    saved = creator.book.get("Ada Lovelace")
    expected = creator.character()
    assert (saved.avatar, saved.face, saved.items) == ("Tube_Woman", "Ada", {"glasses": "sunglasses"})
    assert saved.outfit == expected.outfit and saved.outfit["fills"]["bottoms"]["fabric"] == "navy pinstripe"
    assert sorted(saved.textures) == sorted([HEAD, "p000_body"])
    for name in saved.textures:
        assert np.array_equal(np.asarray(saved.textures[name]), expected.textures[name])

    other = _creator(tmp_path)  # another creator (the demo started again) loads it for editing
    assert other.load("Ada Lovelace") == "editing Ada Lovelace (Tube_Woman, Ada's face, custom)"
    assert other.wait()
    again = other.character()
    assert (again.name, again.avatar, again.face, again.items) == (saved.name, saved.avatar, saved.face, saved.items)
    assert again.outfit == saved.outfit and again.created == saved.created and other.fabric_text == "navy pinstripe"
    for name in saved.textures:  # as saved: nothing was made again
        assert np.array_equal(again.textures[name], np.asarray(saved.textures[name]))
    assert set(accessories.wearing(other.person)) == {"glasses"} and other.state() == "Ada Lovelace: saved"
    other.set_colour("navy")
    assert other.state() == "Ada Lovelace: changed" and other.wait()
    assert not np.array_equal(other.character().textures["p000_body"], np.asarray(saved.textures["p000_body"]))

    creator.new()  # a new character of the same name asks before it replaces the saved one
    creator.set_name("Ada Lovelace")
    assert creator.save() == "Ada Lovelace is saved already: save again to replace it"
    assert other.delete() == "delete Ada Lovelace? Press delete again (it leaves the fair)"
    assert other.delete().startswith("deleted Ada Lovelace") and not (tmp_path / "created" / "Ada Lovelace").exists()
    assert other.delete() == "nothing saved under that name to delete" and other.state() == "not saved yet"
    assert Creator(None, library=Library(), threads=False).save().startswith("characters cannot be saved")


def test_a_loaded_characters_outfit_goes_on_its_faces_skin(tmp_path):
    """A face that comes with a body texture (its skin matched to the face) is what the outfit goes on; it is saved
    with the character, so when the character is loaded and its outfit changed, the skin stays the face's."""
    skin = (90, 60, 40)

    def with_body(photos, model):
        head = np.zeros((8, 8, 4), np.uint8)
        head[..., :3], head[..., 3] = (200, 30, 160), 255
        body = np.array(model.primitives[0].texture, np.uint8, copy=True)  # (the tube woman's body)
        body[..., :3][body[..., 3] > 0] = skin
        return FaceTextures({_head(model): head, "p000_body": body}, "a stand-in with a body")

    creator = _creator(tmp_path, painter=with_body)
    creator.set_avatar("Tube_Woman")
    creator.set_face("Ada")
    creator.apply_preset("highland")
    creator.set_name("Ada")
    assert creator.wait()
    creator.save()
    assert creator.wait()
    creator.update()
    assert (tmp_path / "created" / "Ada" / "p000_body.base.png").is_file()
    from bifold_demo.characters.outfit import regions as R

    hands = R.body_regions(MODELS["Tube_Woman"]).skin
    other = _creator(tmp_path, painter=painter())  # (its painter would give no body: the saved one must do)
    other.load("Ada")
    assert other.wait() and other.state() == "Ada: saved"
    other.set_colour("navy")  # a new outfit, over the face's skin
    assert other.wait()
    body = other.character().textures["p000_body"]
    assert np.all(body[..., :3][hands] == skin)


def test_edit_a_person_of_the_scene_as_a_new_character(tmp_path):
    from bifold_demo.characters.character import Character
    from bifold_demo.characters.outfit import dress

    person = Character(MODELS["Tube_Woman"], name="walker3")
    dress(person, "sporty")
    accessories.dress(person, "hat", "red")
    creator = _creator(tmp_path)
    assert creator.open(person, face="Grace").startswith("a new character from walker3 (Tube_Woman)")
    assert creator.wait()
    assert (creator.avatar, creator.name, creator.face, creator.items) == ("Tube_Woman", "", "Grace", {"hat": "red"})
    assert creator.outfit_name == "sporty" and set(creator.fills) == set(presets()["sporty"])


# -- keys -----------------------------------------------------------------------------------------------------------


def test_keys_move_between_rows_choose_and_type(tmp_path):
    creator = _creator(tmp_path)
    assert creator.wait()

    def press(key, typed=None):
        return creator.handle_key(key, typed)

    assert press(UP) == (True, None) and ROWS[creator.row] == "saved"  # round to the last row
    press(TAB)
    press(TAB)
    assert ROWS[creator.row] == "name" and press(BACKTAB) == (True, None) and creator.row == 0
    assert press(RIGHT) == (True, "avatar: Tube_Woman (preparing it ...)") and creator.wait()
    assert press(LEFT)[1].startswith("avatar: Box_Man") and press(RIGHT)[1].startswith("avatar: Tube_Woman")
    press(DOWN)
    assert press(13) == (True, None) and creator.editing == "name"
    for char in "Ada q,c":  # while typing, letters are text (not quit, not close, not turn)
        assert press(key_code(ord(char)), char) == (True, None)
    assert creator.name == "Ada q,c" and creator.active
    press(BACKSPACE)
    press(BACKSPACE)
    press(BACKSPACE)
    press(BACKSPACE)
    assert press(13) == (True, "name: Ada") and creator.editing is None
    press(13)
    press(key_code(ord("X")), "X")
    assert press(27) == (True, None) and creator.name == "Ada"  # Esc: as it was
    creator.row = ROWS.index("colour")
    creator.select_garment("top")
    assert press(RIGHT) == (True, "top: black")
    assert press(RIGHT) == (True, "top: white") and isinstance(creator.fills["top"], Solid)
    creator.row = ROWS.index("hat")
    assert press(RIGHT) == (True, "hat: navy")
    creator.row = ROWS.index("scale")
    assert press(RIGHT) == (True, "the top has no pattern to scale")
    creator.turn(to=10.0)
    assert press(ord(".")) == (True, None) and creator.heading_at() == pytest.approx(55.0)
    assert press(ord("b")) == (True, None) and creator.heading_at() == 180.0 and not creator.turning
    press(ord(" "))
    assert creator.turning
    assert press(ord("p")) == (False, None)  # the demo's screenshot
    assert press(ord("q")) == (True, None) and creator.active  # nothing here (Esc closes, then Q quits)
    creator.row = ROWS.index("save")
    assert press(RIGHT) == (True, None) and creator.cursor["save"] == 1  # marked: Enter does it
    assert press(13) == (True, "nothing saved under that name to delete")
    assert press(27) == (True, "character creator closed") and not creator.active
    assert press(DOWN) == (False, None)


def test_typed_characters_keep_their_case():
    assert key_code(ord("A")) == ord("a") and typed_char(ord("A")) == "A" and typed_char(ord(",")) == ","
    for raw in (13, 27, 8, 9, 127, 0xFF0D, 0xFF51, 0x250000, -1):
        assert typed_char(raw) is None


def test_clicks_on_the_screen_choose(tmp_path):
    from bifold_demo.display.creator_view import CreatorView

    creator = _creator(tmp_path)
    assert creator.wait()
    view = CreatorView(creator)
    view._error = "no OpenGL in this test"  # (no turntable)
    view.draw(np.zeros((720, 1280, 3), np.uint8))

    def hit(what, index):
        x0, y0, x1, y1, *_ = next(h for h in view.hits if h[4:] == (what, index))
        return (x0 + x1) // 2, (y0 + y1) // 2

    assert view.click(hit("avatar", 1)) == "avatar: Tube_Woman (preparing it ...)"
    assert creator.wait()
    view.draw(np.zeros((720, 1280, 3), np.uint8))
    assert view.click(hit("preset", 0)) == "outfit: business navy" and creator.row == ROWS.index("preset")
    assert view.click(hit("name", None)) is None and creator.editing == "name"
    assert view.click(hit("front", None)) is None and creator.heading_at() == 0.0 and not creator.turning
    assert view.click((5, 5)) is None


# -- the demo -------------------------------------------------------------------------------------------------------


def test_the_creator_in_the_demo_saves_a_character_who_joins_the_fair(tmp_path, monkeypatch):
    from test_created_cast import LIBRARY, small_fair

    from bifold_demo.app import HELP, DemoApp, DemoConfig

    monkeypatch.chdir(tmp_path)
    (tmp_path / "faces" / "Ada").mkdir(parents=True)
    cv2.imwrite(str(tmp_path / "faces" / "Ada" / "one.png"), np.full((32, 32, 3), 128, np.uint8))  # (not a face)
    scene = small_fair()
    app = DemoApp(DemoConfig(headless=True, frames=1, width=160, height=90, detect=False, uploads=None,
                             created=str(tmp_path / "created")), scene=scene)
    app.created._library = LIBRARY
    messages = []
    app.notify = lambda message, *args, **kwargs: messages.append(message)
    window = type("W", (), {"toggle_fullscreen": lambda self: None, "typed": None})()

    def press(key, typed=None):
        window.typed = typed
        app._handle_key(key, window)

    try:
        assert "C creator" in HELP
        press(ord("c"))
        creator = app.creator
        assert creator.active and messages[-1].startswith("character creator:")
        assert creator.library is LIBRARY and creator.people() == ["Ada"]
        creator.painter = painter()
        creator.set_avatar("Woman_2")
        press(DOWN)
        press(13)
        for char in "Ada":
            press(ord(char.lower()), char)
        press(13)
        press(DOWN)
        press(RIGHT)  # the face: Ada
        for _ in range(ROWS.index("hat") - ROWS.index("face")):
            press(DOWN)
        press(RIGHT)  # a hat
        while ROWS[creator.row] != "save":
            press(DOWN)
        press(13)
        assert messages[-1] == "saving Ada ..." and creator.wait()
        for note in creator.update():
            app.notify(note)
        assert any(m.startswith("saved Ada in ") for m in messages), messages
        app._update_created(wait=True)
        assert "Ada joined the fair" in messages
        person = scene["approach1"]
        assert person.created.name == "Ada" and person.created.face == "Ada"
        assert list(accessories.wearing(person)) == ["hat"]
        assert any(tuple(t.rgba[0, 0, :3]) == (200, 30, 160) for t in person.texture_overrides.values())
        press(27)
        assert not creator.active and app._running
        # C in selection mode: a created character is opened to edit it, anyone else is a new character
        from test_selection import _clips

        app.selection.clips = _clips()
        scene.update(12.0, 0.1)
        for created in (True, False):
            app.selection.enter(scene, (1280, 720))
            chosen = next(p.character for p in app.selection.people if (p.character.created is not None) == created)
            app.selection._select(next(p for p in app.selection.people if p.character is chosen))
            press(ord("c"))
            said = "editing Ada (Woman_2, Ada's face)" if created else \
                f"a new character from {chosen.name} ({chosen.skinned_model.name}): type a name, then save"
            assert creator.active and not app.selection.active and messages[-1] == said
            press(27)
        press(ord("q"))
        assert not app._running
    finally:
        if app.creator is not None:
            app.creator.close()
        app.created.close()
        app.selection.close()


# -- the face hook --------------------------------------------------------------------------------------------------


def test_the_face_comes_from_the_head_texture_pipeline_or_the_simple_painting(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from bifold_demo.characters import face_photo, headshape

    model = MODELS["Tube_Woman"]
    photos = []
    for k in range(2):
        photos.append(tmp_path / f"p{k}.png")
        cv2.imwrite(str(photos[-1]), np.full((40, 40, 3), 90 + k, np.uint8))
    monkeypatch.setattr(C, "_headtex_missing", lambda: None)
    textures = {HEAD: np.zeros((4, 4, 4), np.uint8), "p000_body": np.ones((4, 4, 4), np.uint8)}
    made = SimpleNamespace(textures=textures, body=textures["p000_body"], result=SimpleNamespace(projections=[1]),
                           warnings=["dim"])
    monkeypatch.setattr(headshape, "fitted_person", lambda photos, model: SimpleNamespace(person=made, shape=None))
    result = C.face_texture(photos, model)  # (the shape and all the stages; no shape: the avatar's own head)
    assert result.textures == textures and result.method == "head and body textures from 1 of 2 photos"
    assert result.notes == ["dim"] and result.shape is None
    shape = object()
    monkeypatch.setattr(headshape, "fitted_person", lambda photos, model: SimpleNamespace(person=made, shape=shape))
    result = C.face_texture(photos, model)
    assert result.shape is shape and result.method == "the head's shape and head and body textures from 1 of 2 photos"

    def fails(photos, model):
        raise RuntimeError("no usable photo")

    class Engine:  # the face detector: the second photo has the bigger face
        def detect(self, image, min_size=0):
            size = 10 + int(image[0, 0, 0]) - 90
            return [SimpleNamespace(size=size, landmarks=[(1, 1)] * 5, box=(0, 0, size, size))]

    from bifold_demo.recognition import engine

    monkeypatch.setattr(headshape, "fitted_person", fails)
    monkeypatch.setattr(engine, "FaceEngine", Engine)
    monkeypatch.setattr(face_photo, "head_layout", lambda model: SimpleNamespace(primitive=1))
    painted = []
    monkeypatch.setattr(face_photo, "paint_face",
                        lambda model, image, marks, box: painted.append(int(image[0, 0, 0])) or np.ones((2, 2, 4)))
    result = C.face_texture(photos, model)
    assert painted == [91] and list(result.textures) == [model.primitives[1].name]
    assert result.method == "p1.png painted on" and "no usable photo" in result.notes[0]
    monkeypatch.setattr(C, "_headtex_missing", lambda: "head textures from photos need MediaPipe")
    assert "need MediaPipe" in C.face_texture(photos, model).notes[0]
    with pytest.raises(ValueError):
        C.face_texture([], model)


def test_known_faces_are_the_folders_people(tmp_path):
    (tmp_path / "Ada").mkdir()
    for name in ("Ada/a.png", "Ada/b.jpg", "Grace_Hopper.png"):
        (tmp_path / name).write_bytes(b"")
    faces = C.KnownFaces(tmp_path)
    assert faces.people() == ["Ada", "Grace Hopper"] and [p.name for p in faces.photos("Ada")] == ["a.png", "b.jpg"]
    gallery = type("G", (), {"photos": lambda self, name: [tmp_path / "Ada" / "b.jpg"]})()
    assert [p.name for p in C.KnownFaces(tmp_path, lambda: gallery).photos("Ada")] == ["b.jpg", "a.png"]


# -- the screen -----------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_library(tmp_path_factory):
    if not (CHARACTERS / "avatars").is_dir():
        pytest.skip(f"no prepared characters in {CHARACTERS} (set BIFOLD_CHARACTERS_TEST_DIR)")
    from bifold_demo.characters.library import CharacterLibrary
    from bifold_demo.characters.outfit import regions as R

    with pytest.MonkeyPatch.context() as patch:  # the labels' cache goes to a temporary folder, not the library
        patch.setenv(R.CACHE_ENV, str(tmp_path_factory.mktemp("outfit_cache")))
        yield CharacterLibrary(CHARACTERS)


def test_the_screen_shows_the_avatar_on_a_turntable_and_the_rows(gl, real_library, tmp_path):
    from bifold_demo.display.creator_view import BACKDROP, CreatorView

    avatar = "Business_Male_02" if "Business_Male_02" in real_library.available() else min(real_library.available())
    creator = Creator(CharacterBook(tmp_path / "created"), library=real_library, faces=Faces(), threads=False,
                      painter=painter(), fabrics=FabricGenerator(None, cache=False))
    creator.open()
    creator.set_avatar(avatar)
    creator.apply_preset("all black")
    creator.set_item("hat", "red")
    assert creator.wait(120)
    creator.turn(to=0.0)
    view = CreatorView(creator)
    try:
        frame = np.full((720, 1280, 3), 200, np.uint8)
        screen = view.draw(frame)
        tw = round(1280 * 0.34)
        turntable = screen[:, :tw].reshape(-1, 3).astype(int)
        backdrop = np.array(BACKDROP[::-1])
        person = np.abs(turntable - backdrop).sum(axis=1) > 40
        assert 0.08 < person.mean() < 0.6  # the avatar stands on the turntable
        red = (turntable[:, 2] > 120) & (turntable[:, 1] < 70) & (turntable[:, 0] < 70)
        assert red.sum() > 200  # the red hat
        panel = screen[:, tw:]
        assert (panel > 200).any() and screen.shape == frame.shape  # the rows' text
        out = os.environ.get("BIFOLD_CREATOR_PREVIEW")
        cv2.imwrite(out or str(tmp_path / "creator.png"), screen)
        preset = next(h for h in view.hits if h[4] == "preset" and h[5] == 0)  # a click on a preset
        assert view.click(((preset[0] + preset[2]) // 2, (preset[1] + preset[3]) // 2)) == "outfit: business navy"
        assert creator.wait(60)
        again = view.draw(frame)
        assert not np.array_equal(again[:, :tw], screen[:, :tw])
    finally:
        view.release()
        creator.close()
