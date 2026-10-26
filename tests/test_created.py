"""Created characters: saved with their textures, found again, deleted, and their looks applied to a 3D person."""

import json

import numpy as np
import pytest
from test_yoga import synthetic_biped

from bifold_demo.characters import created as created_module
from bifold_demo.characters.character import Character
from bifold_demo.characters.created import CharacterBook, CreatedCharacter, apply_looks, load, save, slug, textures_of
from bifold_demo.characters.model import CharacterState


def _image(colour, size=8):
    image = np.zeros((size, size, 4), np.uint8)
    image[..., :3], image[..., 3] = colour, 255
    return image


def test_a_saved_character_loads_back_the_same(tmp_path):
    ada = CreatedCharacter("Ada at the fair", "Female_Adult_01", face="Ada", outfit={"name": "business navy"},
                           items={"hat": "navy"}, textures={"f001_head": _image((200, 150, 120)),
                                                             "f001_body": _image((10, 20, 30))})
    path = save(ada, tmp_path)
    assert path == tmp_path / "Ada at the fair" and ada.folder == path and ada.created > 0
    data = json.loads((path / "character.json").read_text())
    assert data["textures"] == {"f001_head": "f001_head.png", "f001_body": "f001_body.png"} and data["format"] == 1
    again = load(path)
    assert (again.name, again.avatar, again.face, again.outfit, again.items) == \
        ("Ada at the fair", "Female_Adult_01", "Ada", {"name": "business navy"}, {"hat": "navy"})
    assert np.array_equal(again.textures["f001_head"], ada.textures["f001_head"])
    assert again.describe() == "Female_Adult_01, Ada's face, business navy"
    first = again.created
    again.outfit = {"name": "all black"}
    save(again, tmp_path)  # saved again under the same name: replaced, created kept
    assert load(path).outfit == {"name": "all black"} and load(path).created == first
    assert sorted(p.name for p in tmp_path.iterdir()) == ["Ada at the fair"]  # no temporary folders left


def test_names_become_safe_folder_names():
    assert slug(" Ada / Lovelace ") == "Ada  Lovelace" and slug("Zoë-2.0") == "Zoë-2.0"
    for bad in ("", "...", "///", ".hidden"):
        if bad == ".hidden":
            assert slug(bad) == "hidden"
        else:
            with pytest.raises(ValueError):
                slug(bad)


def test_the_book_sees_new_changed_broken_and_deleted_characters(tmp_path):
    book = CharacterBook(tmp_path / "created")
    assert book.refresh(now=0.0) is False and book.characters() == []
    book.save(CreatedCharacter("Ben", "Male_Adult_05"))
    other = CharacterBook(tmp_path / "created")  # e.g. the demo, while the creator saved
    assert other.refresh(now=0.0) and [c.name for c in other.characters()] == ["Ben"]
    save(CreatedCharacter("Cleo", "Female_Adult_04"), tmp_path / "created")
    assert not other.refresh(now=0.5)  # looked at once a second
    assert other.refresh(now=1.5) and [c.name for c in other.characters()] == ["Ben", "Cleo"]
    broken = tmp_path / "created" / "Dan"
    broken.mkdir()
    (broken / "character.json").write_text("{not json")
    other.refresh(now=3.0)
    assert "Dan" in other.problems and other.get("Dan") is None
    assert other.delete("Ben") and not (tmp_path / "created" / "Ben").exists() and not other.delete("Ben")
    assert [c.name for c in other.characters()] == ["Cleo"]
    newer = tmp_path / "created" / "Cleo" / "character.json"
    data = json.loads(newer.read_text())
    data["format"] = created_module.FORMAT + 1
    newer.write_text(json.dumps(data))
    other.refresh(now=5.0)
    assert "Cleo" in other.problems and "newer version" in other.problems["Cleo"]


def test_looks_are_applied_to_a_3d_person_and_read_back():
    model = synthetic_biped()
    person = Character(model, lambda t: CharacterState((0.0, 0.0, 0.0), 0.0, []), name="p")
    first = model.primitives[0].name
    look = CreatedCharacter("Eve", model.name, textures={first: _image((255, 0, 0)), "not_a_part": _image((0, 0, 255))},
                            items={"hat": "red", "scarf": "blue"})  # no scarf item: skipped
    apply_looks(person, look)
    assert set(person.texture_overrides) == {first}
    assert tuple(person.texture_overrides[first].rgba[0, 0, :3]) == (255, 0, 0)
    from bifold_demo.characters import accessories

    assert list(accessories.wearing(person)) == ["hat"]
    back = textures_of(person)
    assert set(back) == {first} and tuple(back[first][0, 0, :3]) == (255, 0, 0)


def test_new_looks_replace_the_old_ones(tmp_path):
    import cv2

    from bifold_demo.characters import accessories

    model = synthetic_biped()
    person = Character(model, lambda t: CharacterState((0.0, 0.0, 0.0), 0.0, []), name="p")
    apply_looks(person, CreatedCharacter("Eve", model.name, items={"hat": "red", "glasses": "sunglasses"}))
    assert sorted(accessories.wearing(person)) == ["glasses", "hat"]
    folder = tmp_path / "Eve"
    folder.mkdir()
    cv2.imwrite(str(folder / "my_sign.png"), np.full((20, 30, 3), (0, 200, 0), np.uint8))
    again = CreatedCharacter("Eve", model.name, items={"sign": "my_sign.png"}, folder=folder)  # a file of its own
    apply_looks(person, again)
    worn = accessories.wearing(person)
    assert list(worn) == ["sign"] and worn["sign"].look == str(folder / "my_sign.png")


def test_an_item_that_cannot_be_worn_is_left_out():
    from bifold_demo.characters import accessories

    model = synthetic_biped()
    person = Character(model, lambda t: CharacterState((0.0, 0.0, 0.0), 0.0, []), name="p")
    left_out = apply_looks(person, CreatedCharacter("Eve", model.name, items={"hat": 5, "glasses": "sunglasses",
                                                                             "scarf": "blue"}))
    assert list(accessories.wearing(person)) == ["glasses"]
    assert [p.split(":")[0] for p in left_out] == ["hat", "scarf"] and "unknown item 'scarf'" in left_out[1]


def test_a_saved_look_of_the_wrong_kind_is_a_problem_of_the_book(tmp_path):
    for look in ("navy", True, False, None, [20, 40, 200], [20, 40, 200, 255]):
        assert created_module._a_look(look), look
    for look in (5, {"colour": "navy"}, [1, 2], [0, 0, 300], ["a", "b", "c"], [True, 0, 0]):
        assert not created_module._a_look(look), look
    path = save(CreatedCharacter("Ada", "Female_Adult_01", items={"hat": "navy"}), tmp_path)
    data = json.loads((path / "character.json").read_text())
    data["items"] = {"hat": 5}  # written by hand, or by a creator with a bug
    (path / "character.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="the look of the hat"):
        load(path)
    book = CharacterBook(tmp_path)
    book.refresh(force=True)
    assert book.characters() == [] and "the look of the hat" in book.problems["Ada"]
    # saving such a look fails at once (the creator hears of it), and what was saved stays
    with pytest.raises(ValueError, match="the look of the glasses"):
        save(CreatedCharacter("Ben", "Male_Adult_01", items={"glasses": {"colour": "navy"}}), tmp_path)
    with pytest.raises(ValueError, match="the look of the hat"):
        book.save(CreatedCharacter("Ada", "Female_Adult_01", items={"hat": 5}))
    assert not (tmp_path / "Ben").exists() and json.loads((path / "character.json").read_text())["items"] == {"hat": 5}


def test_the_book_reads_images_only_when_they_are_used(tmp_path):
    from bifold_demo.characters.created import Textures

    save(CreatedCharacter("Ada", "Female_Adult_01", textures={"f001_body": _image((1, 2, 3))}), tmp_path)
    book = CharacterBook(tmp_path)
    book.refresh(force=True)
    ada = book.get("Ada")
    assert isinstance(ada.textures, Textures) and list(ada.textures) == ["f001_body"] and not ada.textures._images
    assert tuple(ada.textures["f001_body"][0, 0, :3]) == (1, 2, 3) and len(ada.textures._images) == 1
    ada.textures["f001_head"] = _image((4, 5, 6))  # edited and saved again, by the creator
    book.save(ada)
    again = load(tmp_path / "Ada")
    assert isinstance(again.textures, dict) and tuple(again.textures["f001_head"][0, 0, :3]) == (4, 5, 6)
    assert tuple(again.textures["f001_body"][0, 0, :3]) == (1, 2, 3)
    (tmp_path / "Ada" / "f001_head.png").unlink()
    with pytest.raises(OSError, match="cannot read"):
        load(tmp_path / "Ada", lazy=True)  # a missing image is found at once, a broken one when it is read
