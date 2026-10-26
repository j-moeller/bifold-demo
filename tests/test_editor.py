import numpy as np
import torch

from bifold_morph import align
from bifold_morph.directions import Directions
from bifold_morph.editor import FaceEditor, Settings
from bifold_morph.faces import Face
from bifold_morph.mivolo import AgeGender

LANDMARKS = np.array([[270.0, 200.0], [330.0, 200.0], [300.0, 235.0], [275.0, 270.0], [325.0, 270.0]])


class FakeDetector:
    def __init__(self, faces):
        self.faces = faces

    def detect(self, image):
        return list(self.faces)


class FakeGenerator(torch.nn.Module):
    n_latent = 18

    def synthesis(self, codes, size):
        # a flat image whose grey level is the mean of the first code entry: moving the code changes it
        level = codes[:, :, 0].mean(1).clamp(-1, 1)
        return level[:, None, None, None].expand(-1, 3, size, size).clone()


class FakeE4E(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.generator = FakeGenerator()
        self.calls = 0

    def encode(self, faces):
        self.calls += 1
        assert faces.shape[1:] == (3, 256, 256)
        return torch.zeros(len(faces), 18, 512, dtype=faces.dtype)


class FakeMiVOLO:
    def predict(self, crops):
        return [AgeGender(float(c.mean()), 0.9) for c in crops]


def make_editor(faces=None, mivolo=None, size=64):
    face = Face((260.0, 170.0, 80.0, 110.0), LANDMARKS, 0.95)
    first = np.zeros(512)
    first[0] = 1.0
    directions = Directions(age=first, gender=-first, years_per_unit=10.0, gender_gap=0.4)
    return FaceEditor(FakeE4E(), directions, FakeDetector([face] if faces is None else faces), "cpu", size=size,
                      mivolo=mivolo)


def frame():
    return np.full((480, 640, 3), 100, np.uint8)


def test_sliders_at_zero_leave_the_picture_alone():
    editor = make_editor()
    image = frame()
    result = editor.process(image, Settings())
    assert np.array_equal(result.image, image) and result.image is not image
    assert len(result.faces) == 1 and editor.e4e.calls == 0  # nothing to edit: e4e does not even run


def test_age_adds_the_generators_change():
    editor = make_editor()
    result = editor.process(frame(), Settings(age=5.0), t=0.0)
    # +5 years = +0.5 units along the first entry: the fake generator's grey goes 0 -> 0.5, i.e. +63.75 levels
    quad = align.ffhq_quad(LANDMARKS)
    cx, cy = np.round(quad.mean(0)).astype(int)
    assert result.image[cy, cx, 0] == 164
    assert result.image[5, 5, 0] == 100  # away from the face
    assert set(result.timings) >= {"detect", "encode", "generate", "paste"}


def test_replace_mode_shows_the_generated_face():
    editor = make_editor()
    result = editor.process(frame(), Settings(gender=-1.0, replace=True), t=0.0)
    # towards female = +GENDER_END * gap * 1 along the first entry here (gender points the other way)
    quad = align.ffhq_quad(LANDMARKS)
    cx, cy = np.round(quad.mean(0)).astype(int)
    level = (min(1.25 * 0.4, 1.0) + 1) * 127.5
    assert abs(int(result.image[cy, cx, 0]) - round(level)) <= 1


def test_switched_off_means_untouched():
    editor = make_editor()
    image = frame()
    assert np.array_equal(editor.process(image, Settings(age=30, enabled=False)).image, image)


def test_no_faces():
    editor = make_editor(faces=[])
    result = editor.process(frame(), Settings(age=10))
    assert result.faces == [] and editor.e4e.calls == 0


def test_readout_before_and_after():
    editor = make_editor(mivolo=FakeMiVOLO())
    result = editor.process(frame(), Settings(age=5.0), t=0.0)
    face = result.faces[0]
    assert face.before.age == 100.0 and face.after.age > face.before.age
    plain = editor.process(frame(), Settings(), t=0.1).faces[0]
    assert plain.after is plain.before


def test_codes_are_smoothed_per_face():
    editor = make_editor()
    editor.process(frame(), Settings(age=5.0), t=0.0)
    track = editor.tracker.tracks[0]
    assert "codes" in track.state and track.state["t"] == 0.0
    editor.process(frame(), Settings(age=5.0), t=0.05)
    assert track.state["t"] == 0.05
