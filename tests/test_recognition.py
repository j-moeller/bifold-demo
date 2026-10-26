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


def test_the_demo_names_the_faces_in_the_rendered_scene(gl, engine, zidane, tmp_path, monkeypatch):
    from bifold_demo.app import DemoApp, DemoConfig
    from bifold_demo.scene import Camera, Quad, Scene
    from bifold_demo.scene.textures import sample_image

    monkeypatch.chdir(tmp_path)
    faces = sorted(engine.detect(zidane), key=lambda f: f.box[0])
    for name, face in zip(("Left Man", "Right Man"), faces):
        x, y, w, h = (int(v) for v in face.box)
        (tmp_path / "faces" / name).mkdir(parents=True)
        crop = zidane[max(0, y - h // 2):y + 2 * h, x - w // 2:x + w * 3 // 2]
        cv2.imwrite(str(tmp_path / "faces" / name / "crop.jpg"), crop)
    scene = Scene(background=(40, 40, 40), camera=Camera(position=(0, 0, 2.2), target=(0, 0, 0), fov=50))
    scene.add(Quad(texture=sample_image("zidane.jpg"), height=1.6, lit=False), name="photo")
    app = DemoApp(DemoConfig(headless=True, frames=1, width=960, height=540, detect=False, recognize=True,
                             uploads=None, faces=str(tmp_path / "faces")), scene=scene)
    app.run()  # headless runs are in step: the faces of each frame are found before it is drawn
    assert app.recognize_enabled and app.recognizer is not None and app.face_worker is None
    frame = app.scene_source.read()
    result = app._faces_for(frame)
    assert sorted(f.name for f in result.faces) == ["Left Man", "Right Man"]
    lines = app._status_lines(frame, None)
    canvas = app._compose(frame, None, result)
    assert (canvas != frame.image).any()
    assert "faces: 2 known, in the picture: Left Man, Right Man" in lines


def test_n_switches_the_face_recognition(gl, tmp_path, monkeypatch):
    from bifold_demo.app import DemoApp, DemoConfig
    from bifold_demo.recognition import gallery, recognizer
    from bifold_demo.recognition.engine import FaceEngine

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(FaceEngine, "__init__", lambda self, *a, **k: None)
    monkeypatch.setattr(FaceEngine, "faces", lambda self, image, min_size=0: [])
    app = DemoApp(DemoConfig(headless=True, frames=1, width=160, height=90, detect=False, uploads=None))
    app.run()
    window = type("W", (), {"toggle_fullscreen": lambda self: None})()
    messages = []
    app.notify = lambda message, *args, **kwargs: messages.append(message)
    app._handle_key(ord("n"), window)
    assert app.recognize_enabled and isinstance(app.recognizer, recognizer.FaceRecognizer)
    assert isinstance(app.recognizer.gallery, gallery.Gallery) and messages[-1] == "face recognition on"
    frame = app.scene_source.read()
    assert app._faces_for(frame).faces == []
    app._handle_key(ord("n"), window)
    assert not app.recognize_enabled and app._faces_for(frame) is None and messages[-1] == "face recognition off"


def test_followed_faces_average_their_frames_and_keep_their_names(tmp_path):
    from bifold_demo.recognition.recognizer import FaceRecognizer

    gallery = Gallery(tmp_path / "faces", FakeEngine())
    _photo(tmp_path / "faces" / "Ada" / "a.png", (200, 10, 10))
    gallery.refresh(force=True)
    ada = _vector((200, 10, 10))
    noise = np.zeros(128)
    noise[10] = 1.0

    class Frames:
        """Each frame: (box, embedding) of the faces in it."""

        def __init__(self):
            self.frames = []

        def faces(self, image, min_size=0):
            return [(Face(box, ((0.0, 0.0),) * 5, 0.9), v / np.linalg.norm(v)) for box, v in self.frames.pop(0)]

    engine = Frames()
    recognizer = FaceRecognizer(gallery, engine)
    weak = ada * 0.3 + noise * 0.95  # one frame alone is below the threshold
    assert ada @ (weak / np.linalg.norm(weak)) < THRESHOLD
    engine.frames = [[((10, 10, 40, 40), ada)], [((12, 11, 40, 40), weak)], [((14, 12, 40, 40), weak)],
                     [((300, 10, 40, 40), weak)]]
    first = recognizer.recognize(None, now=0.0)[0]
    assert first.name == "Ada"
    assert [recognizer.recognize(None, now=t)[0].name for t in (0.1, 0.2)] == ["Ada", "Ada"]  # followed: averaged
    elsewhere = recognizer.recognize(None, now=0.3)[0]
    assert elsewhere.name is None and elsewhere.track != first.track  # another face, on its own
    engine.frames = [[((16, 12, 40, 40), weak)]]
    assert recognizer.recognize(None, now=2.0)[0].track not in (first.track, elsewhere.track)  # forgotten after 1 s


class _People:
    """A stand-in gallery: ``people`` maps names to their photos."""

    engine = None

    def __init__(self, people):
        self.people_ = dict(people)

    def refresh(self, *args, **kwargs):
        return False

    def people(self):
        return {name: len(photos) for name, photos in sorted(self.people_.items())}

    def photos(self, name):
        return list(self.people_.get(name, []))


@pytest.fixture
def cast_scene(monkeypatch):
    from test_yoga import synthetic_biped

    from bifold_demo.characters import face_photo
    from bifold_demo.characters.character import Character
    from bifold_demo.characters.model import CharacterState
    from bifold_demo.recognition import cast as cast_module
    from bifold_demo.scene import Scene
    from bifold_demo.scene.textures import Texture

    model = synthetic_biped()
    layout = type("Layout", (), {"primitive": 0})()
    monkeypatch.setattr(face_photo, "head_layout", lambda m: layout)
    monkeypatch.setattr(cast_module, "_painted", lambda character, photo, engine: Texture.solid("#ff0000", str(photo)))
    scene = Scene()
    for name in ("walker1", "approach1", "audience2", "audience1"):
        scene.add(Character(model, lambda t: CharacterState((0.0, 0.0, 0.0), 0.0, []), name=name), name=name)
    return scene


def _settle(cast, scene, now):
    import time

    messages = cast.update(scene, now=now)
    deadline = time.monotonic() + 5.0
    while cast._painting and time.monotonic() < deadline:
        time.sleep(0.01)
    return messages + cast.update(scene, now=now)


def test_the_known_people_get_3d_people_in_front_of_the_camera_first(cast_scene):
    from bifold_demo.recognition.cast import FaceCast

    gallery = _People({"Ada": ["ada.jpg"], "Ben": ["ben.jpg", "ben2.jpg"]})
    cast = FaceCast(gallery)
    try:
        messages = _settle(cast, cast_scene, now=0.0)
        head = cast_scene["audience1"].skinned_model.primitives[0].name
        assert cast.wears(cast_scene["audience1"]) == "Ada" and cast.wears(cast_scene["audience2"]) == "Ben"
        assert head in cast_scene["audience1"].texture_overrides and not cast_scene["walker1"].texture_overrides
        assert "audience1 has Ada's face" in messages
        gallery.people_["Cleo"] = ["cleo.jpg"]  # a new person a moment later: the next one in front of the camera
        _settle(cast, cast_scene, now=0.5)
        assert cast.wears(cast_scene["approach1"]) is None  # looked at once a second
        _settle(cast, cast_scene, now=1.5)
        assert cast.wears(cast_scene["approach1"]) == "Cleo"
        del gallery.people_["Ada"]  # gone from the folder: the face comes off
        messages = _settle(cast, cast_scene, now=3.0)
        assert cast.wears(cast_scene["audience1"]) is None and not cast_scene["audience1"].texture_overrides
        assert any("Ada is no longer known" in m for m in messages)
        assert cast.put(cast_scene["walker1"], "Ben").startswith("putting Ben's face")  # by hand: Ben moves
        _settle(cast, cast_scene, now=3.1)
        assert cast.wears(cast_scene["walker1"]) == "Ben" and cast.wears(cast_scene["audience2"]) is None
        assert cast.put(cast_scene["walker1"], None) == "walker1 has their own face again"
        assert cast.put(cast_scene["walker1"], "Nobody") == "no photo of Nobody with a face"
    finally:
        cast.close()


def test_the_face_key_of_selection_mode_steps_through_the_known_people(cast_scene):
    from bifold_demo.recognition.cast import FaceCast
    from bifold_demo.selection import Selection

    cast = FaceCast(_People({"Ada": ["ada.jpg"], "Ben": ["ben.jpg"]}), auto=False)
    selection = Selection.__new__(Selection)
    selection.cast, selection.selected = cast, cast_scene["walker1"]
    try:
        assert selection.wear("face", 0.0).startswith("putting Ada's face on walker1 (1 of 2; again: the next)")
        _settle(cast, cast_scene, now=0.0)
        assert selection.wears(cast_scene["walker1"]) == "Ada's face"
        assert selection.wear("face", 0.0).endswith("(2 of 2; again: their own)")
        _settle(cast, cast_scene, now=0.1)
        assert selection.wear("face", 0.0) == "walker1 has their own face again"
        selection.cast = None
        assert selection.wear("face", 0.0).startswith("nobody known yet")
    finally:
        cast.close()
