"""A photo's face painted on a 3D person: lined up by the landmarks, only on the front of the head, for one person."""

import os
from pathlib import Path

import cv2
import numpy as np
import pytest
from test_yoga import synthetic_biped

from bifold_demo.characters import face_photo
from bifold_demo.characters.character import Character
from bifold_demo.characters.model import CharacterState

CHARACTERS = Path(os.environ.get("BIFOLD_CHARACTERS_TEST_DIR", "/nonexistent")).expanduser()


def _still(model, heading=0.0):
    return Character(model, behaviour=lambda t: CharacterState((0.0, 0.0, 0.0), heading, []), name="p")


def test_a_model_without_face_bones_has_no_head_to_paint():
    model = synthetic_biped()
    assert face_photo.head_layout(model) is None
    with pytest.raises(ValueError):
        face_photo.paint_face(model, np.zeros((10, 10, 3), np.uint8), [(0, 0)] * 5)


@pytest.fixture(scope="module")
def avatar():
    if not CHARACTERS.is_dir():
        pytest.skip("set BIFOLD_CHARACTERS_TEST_DIR to a prepared character library")
    from bifold_demo.characters.library import CharacterLibrary

    library = CharacterLibrary(CHARACTERS)
    names = sorted(library.available())
    return library.load("Male_Adult_05" if "Male_Adult_05" in names else names[0])


def test_the_photo_goes_on_the_front_of_the_head_only(avatar):
    layout = face_photo.head_layout(avatar)
    assert layout is not None and 0.05 < layout.eye_distance < 0.08
    head = avatar.primitives[layout.primitive]
    # a "photo" of a face with its landmarks where a frontal face has them, painted green
    photo = np.full((400, 400, 3), (40, 200, 40), np.uint8)
    marks = [(150, 170), (250, 170), (200, 230), (160, 290), (240, 290)]
    painted = face_photo.paint_face(avatar, photo, marks, box=(110, 90, 180, 260))
    from bifold_demo.scene.textures import as_texture

    before = as_texture(head.texture).rgba
    changed = np.abs(painted.astype(int) - before.astype(int)).sum(axis=2) > 30
    h, w = changed.shape
    uv = np.asarray(head.uvs)

    def texel(k):
        return min(h - 1, int((1 - uv[k, 1]) * h)), min(w - 1, int(uv[k, 0] * w))

    points = np.asarray(head.positions)
    nose = int(np.argmax(np.where(np.abs(points[:, 0]) < 0.01, points[:, 2], -np.inf)))
    back = int(np.argmin(points[:, 2]))
    assert changed[texel(nose)] and not changed[texel(back)]  # the nose tip is painted, the back of the head is not
    assert painted[texel(nose)][1] > painted[texel(nose)][0] + 40  # green, from the photo
    assert 0.01 < changed.mean() < 0.5


def test_put_face_paints_one_person_and_comes_off_again(avatar):
    one, two = _still(avatar), _still(avatar)
    photo = np.full((400, 400, 3), (40, 40, 200), np.uint8)
    marks = [(150, 170), (250, 170), (200, 230), (160, 290), (240, 290)]
    face_photo.put_face(one, photo, landmarks=marks, box=(110, 90, 180, 260))
    head = avatar.primitives[face_photo.head_layout(avatar).primitive].name
    assert set(one.texture_overrides) == {head} and two.texture_overrides == {}
    face_photo.take_face_off(one)
    assert one.texture_overrides == {}


def test_a_painted_face_is_recognised_as_the_person_in_the_photo(gl, avatar, tmp_path):
    from bifold_demo.recognition import FaceEngine, Gallery
    from bifold_demo.scene import Camera, Scene
    from bifold_demo.scene.renderer import Renderer
    from bifold_demo.scene.textures import sample_image

    path = sample_image("zidane.jpg")
    if not path:
        pytest.skip("the Ultralytics sample photo is not installed")
    try:
        engine = FaceEngine(download=os.environ.get("BIFOLD_NO_DOWNLOADS") is None)
    except (OSError, ValueError) as exc:
        pytest.skip(f"face models unavailable: {exc}")
    photo = cv2.imread(str(path))
    left, right = sorted(engine.detect(photo), key=lambda f: f.box[0])
    for name, face in (("Left Man", left), ("Right Man", right)):
        x, y, w, h = (int(v) for v in face.box)
        (tmp_path / name).mkdir()
        crop = photo[max(0, y - h // 2):y + 2 * h, x - w // 2:x + w * 3 // 2]
        cv2.imwrite(str(tmp_path / name / "photo.jpg"), crop)
    gallery = Gallery(tmp_path, engine)
    gallery.refresh(force=True)
    scene = Scene(background=(225, 228, 232), ambient=0.6, light_direction=(-0.3, -1.0, -0.4))
    person = scene.add(_still(avatar))
    face_photo.put_face(person, photo, landmarks=right.landmarks, box=right.box)
    scene.update(0.0, 0.0)
    head = person.pose[avatar.skeleton.index("Bip01 Head"), :3, 3]
    scene.camera = Camera(position=(0.0, head[1] + 0.1, 0.9), target=(0.0, head[1] + 0.07, 0.0), fov=28)
    image = Renderer(300, 300).render(scene)
    found = engine.faces(image, min_size=0)
    assert found and gallery.match(found[0][1]).name == "Right Man"
    face_photo.take_face_off(person)
    found = engine.faces(Renderer(300, 300).render(scene), min_size=0)
    assert not found or gallery.match(found[0][1]).name is None  # the avatar's own face is nobody known
