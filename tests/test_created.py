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
