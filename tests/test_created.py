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
    left_out = apply_looks(person, look)
    assert left_out[0] == f"textures of not_a_part: {model.name} has no such part"  # painted for another avatar
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
    assert list(worn) == ["sign"] and worn["sign"].name == "my_sign"
    assert tuple(worn["sign"].look.rgba[0, 0]) == (0, 200, 0, 255)  # read from the character's folder
    (tmp_path / "elsewhere.png").write_bytes((folder / "my_sign.png").read_bytes())
    for look in ("../elsewhere.png", str(tmp_path / "elsewhere.png"), "gone.png"):  # never a file elsewhere
        left_out = apply_looks(person, CreatedCharacter("Eve", model.name, items={"sign": look}, folder=folder))
        assert len(left_out) == 1 and left_out[0].startswith("sign: ") and not accessories.wearing(person), look


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


def test_item_pictures_are_read_where_the_looks_are_prepared(tmp_path, monkeypatch):
    """prepare_looks (on the worker) reads everything; apply_looks (on the main thread) then reads no file."""
    import cv2

    from bifold_demo.characters import accessories
    from bifold_demo.scene import textures

    model = synthetic_biped()
    folder = tmp_path / "Eve"
    folder.mkdir()
    cv2.imwrite(str(folder / "poster.png"), np.full((40, 60, 3), (0, 0, 220), np.uint8))
    (tmp_path / "items").mkdir()  # the demo's items folder: a look of the wardrobe that is a file
    cv2.imwrite(str(tmp_path / "items" / "glasses_blue.png"), np.full((8, 16, 3), (200, 0, 0), np.uint8))
    wardrobe = accessories.Wardrobe(tmp_path / "items")
    eve = CreatedCharacter("Eve", model.name, items={"sign": "poster.png", "hat": "red", "glasses": "glasses_blue"},
                           textures={model.primitives[0].name: _image((1, 2, 3))}, folder=folder)
    looks = created_module.prepare_looks(eve, model, wardrobe)
    poster = looks.items["sign"]
    assert tuple(poster.rgba[0, 0, :3]) == (220, 0, 0) and poster._upload[0] == poster.version  # laid out for the GPU
    hat, glasses = looks.shared  # the wardrobe's (shared with everyone wearing them), read already
    assert hat is accessories.built_in()["hat"]["red"] and tuple(glasses.rgba[0, 0, :3]) == (0, 0, 200)
    assert set(map(id, looks.gpu())) == {id(poster), id(looks.textures[model.primitives[0].name]), id(hat),
                                         id(glasses)}

    def no_reading(*args, **kwargs):
        raise AssertionError("a file read on the main thread")

    monkeypatch.setattr(textures, "_read_image", no_reading)
    monkeypatch.setattr(created_module, "_read", no_reading)
    person = Character(model, lambda t: CharacterState((0.0, 0.0, 0.0), 0.0, []), name="p")
    assert apply_looks(person, eve, wardrobe, looks=looks) == []
    worn = accessories.wearing(person)
    assert worn["sign"].look is poster and worn["hat"] and textures.as_texture(worn["glasses"].look) is glasses


def test_images_of_any_kind_are_read_right_and_cut_to_size(tmp_path):
    import cv2
    from PIL import Image

    folder = save(CreatedCharacter("Ada", "Female_Adult_01", textures={"a": _image((1, 1, 1))}), tmp_path)
    red = (200, 40, 40)
    files = {
        "rgba16": lambda f: cv2.imwrite(str(f), (np.dstack([np.full((8, 8), c) for c in (*red[::-1], 255)]) * 257)
                                        .astype(np.uint16)),
        "grey": lambda f: Image.fromarray(np.full((8, 8), 90, np.uint8), "L").save(f),
        "grey_alpha": lambda f: Image.fromarray(np.dstack([np.full((8, 8), 90, np.uint8),
                                                           np.full((8, 8), 200, np.uint8)]), "LA").save(f),
        "palette": lambda f: Image.fromarray(np.full((8, 8, 3), red, np.uint8)).convert("P", palette=Image.ADAPTIVE)
        .save(f),
        "big": lambda f: cv2.imwrite(str(f), np.full((3000, 1500, 3), red[::-1], np.uint8)),
    }
    for name, write in files.items():
        write(folder / f"{name}.png")
    data = json.loads((folder / "character.json").read_text())
    data["textures"] = {name: f"{name}.png" for name in files}
    (folder / "character.json").write_text(json.dumps(data))
    images = load(folder).textures
    assert images["rgba16"].dtype == np.uint8 and tuple(images["rgba16"][0, 0]) == (*red, 255)  # not all white
    assert tuple(images["grey"][0, 0]) == (90, 90, 90, 255) and tuple(images["grey_alpha"][0, 0]) == (90, 90, 90, 200)
    assert tuple(images["palette"][0, 0]) == (*red, 255)
    assert images["big"].shape == (created_module.MAX_TEXTURE, created_module.MAX_TEXTURE // 2, 4)


def test_the_book_keeps_a_character_while_its_file_cannot_be_read(tmp_path):
    """An editor or a sync tool writes character.json in place: for a moment it is empty or half written. The
    character stays as it was meanwhile (and the book says why), and is read anew once the file reads well."""
    book = CharacterBook(tmp_path)
    book.save(CreatedCharacter("Ada", "Female_Adult_01", items={"hat": "navy"}))
    demo = CharacterBook(tmp_path)
    demo.refresh(now=0.0)
    ada = demo.get("Ada")
    file = tmp_path / "Ada" / "character.json"
    good = file.read_text()
    for now, broken in ((1.5, ""), (3.0, good[: len(good) // 2]), (4.5, '{"name": "Ada", "avatar": 5}')):
        file.write_text(broken)
        demo.refresh(now=now)
        assert demo.get("Ada") is ada and demo.problems["Ada"].endswith("(shown as it was)"), broken
    file.write_text(good)  # written again as it was: nothing changed
    assert not demo.refresh(now=6.0) and demo.get("Ada") is ada and demo.problems == {}
    file.write_text(good.replace("navy", "red"))
    assert demo.refresh(now=7.5) and demo.get("Ada") is not ada and demo.get("Ada").items == {"hat": "red"}
    assert demo.problems == {}
    (tmp_path / "Ben").mkdir()  # never readable: left out
    (tmp_path / "Ben" / "character.json").write_text("[]")
    demo.refresh(now=9.0)
    assert demo.get("Ben") is None
    assert demo.problems["Ben"] == "ValueError: not a character file (an array, not an object)"


def test_a_change_to_any_file_of_a_character_is_seen(tmp_path):
    book = CharacterBook(tmp_path)
    book.save(CreatedCharacter("Ada", "Female_Adult_01", textures={"f001_body": _image((1, 2, 3))}))
    demo = CharacterBook(tmp_path)
    demo.refresh(now=0.0)
    ada = demo.get("Ada")
    png = tmp_path / "Ada" / "f001_body.png"
    png.write_bytes(png.read_bytes())  # rewritten (a copy finishing, say): its time changes
    import os
    import time

    os.utime(png, ns=(time.time_ns(), time.time_ns() + 10**9))
    assert demo.refresh(now=1.5) and demo.get("Ada") is not ada
    assert not demo.refresh(now=3.0)  # nothing changed since


def test_saving_again_keeps_the_other_files_of_the_folder(tmp_path):
    ada = CreatedCharacter("Ada", "Female_Adult_01", items={"sign": "poster.png"},
                           textures={"f001_body": _image((1, 2, 3)), "f001_head": _image((4, 5, 6))})
    folder = save(ada, tmp_path)
    (folder / "poster.png").write_bytes(b"the user's picture")  # copied in by hand: the sign's picture
    (folder / "notes.txt").write_text("hers")
    del ada.textures["f001_head"]
    ada.textures["f001_body"] = _image((7, 8, 9))
    save(ada, tmp_path)
    assert sorted(p.name for p in folder.iterdir()) == ["character.json", "f001_body.png", "notes.txt", "poster.png"]
    assert (folder / "poster.png").read_bytes() == b"the user's picture"  # kept; the old head texture is gone
    assert tuple(load(folder).textures["f001_body"][0, 0, :3]) == (7, 8, 9)


def test_a_name_that_goes_to_another_characters_folder_is_refused(tmp_path):
    save(CreatedCharacter("Ada", "Female_Adult_01", textures={"f001_body": _image((1, 2, 3))}), tmp_path)
    for other in ("Ada?", "Ada/"):  # both go to the folder Ada/
        with pytest.raises(ValueError, match="holds another character, 'Ada'"):
            save(CreatedCharacter(other, "Male_Adult_05"), tmp_path)
    assert load(tmp_path / "Ada").name == "Ada" and load(tmp_path / "Ada").avatar == "Female_Adult_01"
    save(CreatedCharacter("Ada", "Male_Adult_05"), tmp_path)  # the same one: replaced
    assert load(tmp_path / "Ada").avatar == "Male_Adult_05"
    book = CharacterBook(tmp_path)
    assert not book.delete("Ada?") and (tmp_path / "Ada").is_dir()  # nor is another one deleted for it
    assert book.delete("Ada") and not (tmp_path / "Ada").exists()


def test_a_copied_folder_is_a_problem_and_the_own_one_counts(tmp_path):
    import shutil

    folder = save(CreatedCharacter("Ada", "Female_Adult_01", items={"hat": "navy"}), tmp_path)
    shutil.copytree(folder, tmp_path / "Ada copy")  # copied by hand ("Ada copy" comes first, sorted)
    book = CharacterBook(tmp_path)
    book.refresh(force=True)
    assert [c.folder.name for c in book.characters()] == ["Ada"]
    assert book.problems == {"Ada copy": "another folder (Ada) holds a character called 'Ada'"}
    data = json.loads((folder / "character.json").read_text())
    data["items"] = {"hat": "red"}  # edits to the character's own folder show
    (folder / "character.json").write_text(json.dumps(data))
    book.refresh(force=True)
    assert book.get("Ada").items == {"hat": "red"}


def test_the_book_never_raises_for_what_is_in_its_folder(tmp_path, monkeypatch):
    from pathlib import Path

    book = CharacterBook(tmp_path / "created")
    book.save(CreatedCharacter("Ada", "Female_Adult_01"))
    (tmp_path / "created" / "a file").write_text("not a folder")
    assert book.refresh(force=True) is False and [c.name for c in book.characters()] == ["Ada"]
    real = Path.iterdir
    monkeypatch.setattr(Path, "iterdir", lambda path: (_ for _ in ()).throw(FileNotFoundError(2, "gone", str(path))))
    assert book.refresh(force=True) is False and book.trouble.startswith("cannot look into")  # moved away, say
    assert [c.name for c in book.characters()] == ["Ada"]  # as it was
    monkeypatch.setattr(Path, "iterdir", real)
    import bifold_demo.characters.created as module

    monkeypatch.setattr(module, "_stamp", lambda path: (_ for _ in ()).throw(OSError("changing")))
    book.refresh(force=True)
    assert book.trouble is None and [c.name for c in book.characters()] == ["Ada"]  # its files changing just now
