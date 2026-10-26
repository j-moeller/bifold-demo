"""Face recognition: the gallery of known people (with a stand-in engine) and the real models on a sample photo."""

import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from bifold_demo.recognition import gallery as gallery_module
from bifold_demo.recognition.engine import Face
from bifold_demo.recognition.gallery import THRESHOLD, Gallery


class FakeEngine:
    """Faces are drawn as solid colours: one face per image, its embedding from the colour (so photos of the same
    colour are the same person); black images have no face."""

    def __init__(self):
        self.embedded = 0

    def detect(self, image, min_size=0):
        if not image.any():
            return []
        return [Face((0.0, 0.0, float(image.shape[1]), float(image.shape[0])), ((0.0, 0.0),) * 5, 0.9)]

    def embed(self, image, face):
        self.embedded += 1
        colour = image.reshape(-1, 3).mean(axis=0)
        vector = np.zeros(128)
        vector[:3] = colour
        vector[3] = 40.0  # a common part: different colours still score a little alike
        return vector / np.linalg.norm(vector)


def _photo(path: Path, colour, size=(20, 20)):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((*size, 3), colour, np.uint8))
    return path


def _vector(colour):
    return FakeEngine().embed(np.full((2, 2, 3), colour, np.uint8), None)


def test_the_gallery_knows_a_person_per_folder_and_single_photos(tmp_path):
    folder = tmp_path / "faces"
    _photo(folder / "Ada Lovelace" / "one.png", (200, 10, 10))
    _photo(folder / "Ada Lovelace" / "two.jpg", (190, 20, 10))
    _photo(folder / "Grace_Hopper.png", (10, 10, 200))
    _photo(folder / "Ada Lovelace" / "dark.png", (0, 0, 0))  # no face in it
    (folder / "notes.txt").write_text("not a photo")
    gallery = Gallery(folder, FakeEngine())
    assert gallery.refresh(now=0.0)
    assert gallery.people() == {"Ada Lovelace": 2, "Grace Hopper": 1}
    assert gallery.problems() == [folder / "Ada Lovelace" / "dark.png"]
    ada = gallery.match(_vector((195, 15, 10)))
    assert ada.name == "Ada Lovelace" and ada.score > 0.99 and ada.runner_up < ada.score
    assert gallery.match(_vector((10, 10, 200))).name == "Grace Hopper"
    stranger = gallery.match(_vector((10, 200, 10)))  # green: nobody known
    assert stranger.name is None and stranger.score < THRESHOLD
    assert Gallery(tmp_path / "missing", FakeEngine()).match(_vector((1, 2, 3))).name is None


def test_the_gallery_caches_embeddings_and_sees_new_photos(tmp_path, monkeypatch):
    folder = tmp_path / "faces"
    _photo(folder / "Ada" / "one.png", (200, 10, 10))
    first = FakeEngine()
    Gallery(folder, first).refresh(now=0.0)
    assert first.embedded == 1 and (folder / gallery_module.CACHE).is_file()
    again = FakeEngine()
    gallery = Gallery(folder, again)
    gallery.refresh(now=0.0)
    assert again.embedded == 0 and gallery.people() == {"Ada": 1}  # from the cache
    assert gallery.match(_vector((200, 10, 10))).name == "Ada"
    _photo(folder / "Bob" / "bob.png", (10, 10, 200))
    assert not gallery.refresh(now=0.5)  # looked at once a second
    assert gallery.refresh(now=1.5) and gallery.people() == {"Ada": 1, "Bob": 1} and again.embedded == 1
    (folder / "Bob" / "bob.png").unlink()
    assert gallery.refresh(now=3.0) and gallery.people() == {"Ada": 1}
    assert gallery.match(_vector((10, 10, 200))).name is None


def test_adding_photos_copies_them_into_the_persons_folder(tmp_path):
    folder = tmp_path / "faces"
    a = _photo(tmp_path / "in" / "me.png", (200, 10, 10))
    b = _photo(tmp_path / "in2" / "me.png", (205, 12, 10))  # the same file name: kept apart
    dark = _photo(tmp_path / "in" / "dark.png", (0, 0, 0))
    gallery = Gallery(folder, FakeEngine())
    problems = gallery.add(" Ada Lovelace ", [a, b, dark, tmp_path / "nothing.png"])
    assert len(problems) == 2 and "no face" in problems[0] and "not an image" in problems[1]
    assert sorted(p.name for p in (folder / "Ada Lovelace").iterdir()) == ["me.png", "me_2.png"]
    assert gallery.people() == {"Ada Lovelace": 2}
    assert gallery.photos("Ada Lovelace")[0].parent == folder / "Ada Lovelace"
    with pytest.raises(ValueError):
        gallery.add("../escape", [a])


# -- the real models ------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def engine():
    from bifold_demo.recognition.engine import FaceEngine

    try:
        return FaceEngine(download=os.environ.get("BIFOLD_NO_DOWNLOADS") is None)
    except (OSError, ValueError) as exc:  # offline: no models
        pytest.skip(f"face models unavailable: {exc}")


@pytest.fixture(scope="module")
def zidane():
    from bifold_demo.scene.textures import sample_image

    path = sample_image("zidane.jpg")
    if not path:
        pytest.skip("the Ultralytics sample photo is not installed")
    return cv2.imread(str(path))


def test_the_engine_finds_the_faces_and_tells_people_apart(engine, zidane):
    faces = engine.faces(zidane)
    assert len(faces) == 2 and all(f.score > 0.8 for f, _ in faces)
    (_, a), (_, b) = faces
    assert abs(np.linalg.norm(a) - 1.0) < 1e-6 and a @ b < THRESHOLD  # two different men
    # the same men in other "photos": brighter, smaller, mirrored
    brighter = cv2.convertScaleAbs(zidane, alpha=1.2, beta=25)
    smaller = cv2.resize(zidane, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    mirrored = zidane[:, ::-1].copy()
    for other in (brighter, smaller, mirrored):
        found = engine.faces(other)  # a part of a face at the picture's edge may come along
        scores = np.array([[x @ y for _, y in found] for x in (a, b)])
        assert (scores.max(axis=1) > 0.6).all()  # each man is found again ...
        assert scores[0].argmax() != scores[1].argmax()  # ... in a face of his own
        assert scores[0, scores[1].argmax()] < THRESHOLD and scores[1, scores[0].argmax()] < THRESHOLD


def test_a_gallery_of_real_photos_names_the_faces(engine, zidane, tmp_path):
    faces = engine.detect(zidane)
    folder = tmp_path / "faces"
    for name, face in zip(("Left Man", "Right Man"), sorted(faces, key=lambda f: f.box[0])):
        x, y, w, h = (int(v) for v in face.box)
        crop = zidane[max(0, y - h // 2):y + 2 * h, max(0, x - w // 2):x + w + w // 2]
        (folder / name).mkdir(parents=True)
        cv2.imwrite(str(folder / name / "crop.jpg"), crop)
    gallery = Gallery(folder, engine)
    gallery.refresh(force=True)
    assert gallery.people() == {"Left Man": 1, "Right Man": 1}
    named = sorted(((f.box[0], gallery.match(v).name) for f, v in engine.faces(cv2.convertScaleAbs(zidane, alpha=0.8))))
    assert [name for _, name in named] == ["Left Man", "Right Man"]
