"""Who wears fitted heads: created characters (``shape.npz`` beside ``character.json``), the known faces of the fair
(``recognition.cast``), the character creator's turntable; and a scene character changing its model. Synthetic: the
avatar and the tiny identity space of ``test_headshape``."""

import numpy as np
import pytest
from test_headshape import synthetic_avatar, tiny_space
from test_recognition import _People, _settle, cast_scene  # noqa: F401 - (the cast's fixture)

from bifold_demo.characters import created as created_module
from bifold_demo.characters.character import Character
from bifold_demo.characters.headshape import head as s0
from bifold_demo.characters.headshape import rig
from bifold_demo.characters.headshape import space as identity
from bifold_demo.characters.model import CharacterState


@pytest.fixture(scope="module")
def fitted():
    """The synthetic avatar, a shape for it and the model wearing it."""
    model = synthetic_avatar()
    space, landmarks, marks = tiny_space(model)
    keep = identity.space_landmarks
    identity.space_landmarks = lambda: landmarks
    try:
        hs = s0.make_head_space(model, marks, space)
    finally:
        identity.space_landmarks = keep
    offsets = np.einsum("k,knd->nd", np.array([1.0, -1.0, 1.0]) - hs.alpha, hs.modes(space).astype(np.float64))
    shape = rig.make_shape(model, hs, offsets)
    return model, shape, rig.apply_shape(model, shape)


def test_a_character_changes_its_model_and_keeps_the_rest(fitted):
    model, _shape, wearing = fitted
    person = Character(model, lambda t: CharacterState((1.0, 0.0, 2.0), 90.0, []), name="p")
    person.texture_overrides["s_head"] = "a face"
    person.update(1.0)
    person.use_model(wearing)
    assert person.skinned_model is wearing and person.texture_overrides == {"s_head": "a face"}
    assert np.allclose(person.current_position, (1.0, 0.0, 2.0))
    eye = model.skeleton.index("Bip01 REye")
    assert np.allclose(person.pose[eye], wearing.skeleton.rest_global[eye])  # the new rest pose at once
    person.use_model(model)
    assert np.allclose(person.pose[eye], model.skeleton.rest_global[eye])
    other = synthetic_avatar()
    other.skeleton.names[3] = "Bip01 Something"
    with pytest.raises(ValueError, match="skeleton"):
        person.use_model(other)


def test_a_created_character_keeps_its_shape_beside_its_file(fitted, tmp_path):
    model, shape, _wearing = fitted
    character = created_module.CreatedCharacter("Ada", model.name, face="Ada Lovelace",
                                                textures={"s_head": np.zeros((4, 4, 4), np.uint8)}, shape=shape)
    folder = created_module.save(character, tmp_path)
    assert (folder / created_module.SHAPE_FILE).is_file()
    again = created_module.load(folder)
    assert again.shape is not None and np.array_equal(again.shape.offsets, shape.offsets)
    worn, note = created_module.wearing(again, model)
    assert note == "" and worn is not model
    assert np.allclose(worn.primitives[1].positions, model.primitives[1].positions + shape.offsets, atol=1e-6)
    character.shape = None  # saved again without one: the file goes
    created_module.save(character, tmp_path)
    assert not (folder / created_module.SHAPE_FILE).exists() and created_module.load(folder).shape is None
    assert created_module.wearing(created_module.load(folder), model) == (model, "")


def test_a_shape_made_for_another_version_of_the_avatar_is_left_off(fitted, tmp_path):
    model, shape, _wearing = fitted
    character = created_module.CreatedCharacter("Ben", model.name, shape=shape)
    other = synthetic_avatar()
    other.primitives[1].positions = other.primitives[1].positions * np.float32(1.01)  # prepared again since
    worn, note = created_module.wearing(character, other)
    assert worn is other and "keeps Synthetic's own head" in note and "another version" in note


def test_a_damaged_shape_file_is_a_problem_of_the_book(fitted, tmp_path):
    model, shape, _wearing = fitted
    character = created_module.CreatedCharacter("Cleo", model.name, shape=shape)
    folder = created_module.save(character, tmp_path)
    (folder / created_module.SHAPE_FILE).write_bytes(b"not a shape")
    book = created_module.CharacterBook(tmp_path)
    book.refresh(force=True)
    assert "Cleo" in book.problems and book.get("Cleo") is None
    assert book.delete("Cleo") and not folder.exists()  # deleted with the character


def test_known_faces_wear_their_fitted_heads_and_give_them_back(cast_scene, monkeypatch, tmp_path):  # noqa: F811
    """The cast's known people get the fitted head made from their photos with their textures; when they leave the
    faces folder, the 3D person has its own head again."""
    from bifold_demo.recognition import cast as cast_module
    from bifold_demo.recognition.cast import FaceCast

    person = cast_scene["audience1"]
    own = person.skinned_model
    head = synthetic_like(own)

    def from_photos(character, photos, cache, model=None):
        assert model is None or model is own  # (made on the avatar, never on someone's fitted head)
        return {"box1": np.full((4, 4, 4), 7, np.uint8)}, head

    monkeypatch.setattr(cast_module, "_from_photos", from_photos)
    monkeypatch.setattr(cast_module, "_painted", lambda *a: (_ for _ in ()).throw(AssertionError("not painted")))
    gallery = _People({"Ada": ["ada.jpg", "ada2.jpg"]})
    gallery.folder = tmp_path
    cast = FaceCast(gallery)
    try:
        _settle(cast, cast_scene, now=0.0)
        assert cast.wears(person) == "Ada" and person.skinned_model is head
        gallery.people_["Ada"] = ["ada.jpg", "ada2.jpg", "ada3.jpg"]  # made again, on the avatar's own model
        _settle(cast, cast_scene, now=1.5)
        assert person.skinned_model is head
        del gallery.people_["Ada"]
        _settle(cast, cast_scene, now=3.0)
        assert person.skinned_model is own and not person.texture_overrides
    finally:
        cast.close()


def synthetic_like(model):
    """A copy of ``model`` (the same skeleton) with its first primitive moved a little: a stand-in fitted head."""
    from dataclasses import replace

    from bifold_demo.characters.model import CharacterModel

    moved = replace(model.primitives[0], positions=model.primitives[0].positions + np.float32(0.001))
    return CharacterModel(model.name, model.skeleton, [moved, *model.primitives[1:]], model.inverse_bind)


def test_the_creators_person_wears_the_fitted_head_and_saves_it(tmp_path, monkeypatch):
    """A face made with a fitted head puts the head on the person on the turntable; without a face the avatar's own
    head is back; a saved character keeps the shape."""
    from test_creator import MODELS, _creator, _head

    from bifold_demo.characters import headshape
    from bifold_demo.creator import FaceTextures

    shape = object()
    heads = {}

    def apply(model, given):
        assert given is shape
        return heads.setdefault(model.name, synthetic_like(model))

    def paint(photos, model):
        return FaceTextures({_head(model): np.full((8, 8, 4), 200, np.uint8)}, "a stand-in", shape=shape)

    monkeypatch.setattr(headshape, "apply_shape", apply)
    creator = _creator(tmp_path, painter=paint)
    creator.set_avatar("Tube_Woman")
    assert creator.wait()
    assert creator.person.skinned_model is MODELS["Tube_Woman"]
    creator.set_face("Ada")
    while creator._jobs or creator._due():
        creator.update()
    assert creator.person.skinned_model is heads["Tube_Woman"]
    assert creator.character().shape is shape
    creator.set_face(None)
    creator.update()
    assert creator.person.skinned_model is MODELS["Tube_Woman"] and creator.character().shape is None
