import numpy as np
import torch

from bifold_morph import warp
from bifold_morph.distill import GENDER_RANGE
from bifold_morph.editor import FaceEditor
from bifold_morph.faces import Face
from bifold_morph.state import Settings

LANDMARKS = np.array([[270.0, 200.0], [330.0, 200.0], [300.0, 235.0], [275.0, 270.0], [325.0, 270.0]])


class FakeDetector:
    def __init__(self, faces):
        self.faces = faces

    def detect(self, image):
        return list(self.faces)


class Brighten(torch.nn.Module):
    """A stand-in student: brightens the face by the gender amount, and records what it was given."""

    def forward(self, x, gender, years):
        self.seen = (x.shape, gender.tolist(), years.tolist())
        return (x + 0.1 * gender[:, None, None, None]).clamp(-1, 1)


def editor(faces=None):
    face = Face((260.0, 170.0, 80.0, 110.0), LANDMARKS, 0.95)
    return FaceEditor(Brighten(), FakeDetector([face] if faces is None else faces), "cpu")


def frame():
    return np.full((480, 640, 3), 100, np.uint8)


def test_sliders_at_zero_leave_the_picture_alone():
    e = editor()
    image = frame()
    result = e.process(image, Settings())
    assert np.array_equal(result.image, image) and result.image is not image
    assert len(result.faces) == 1 and not hasattr(e.student, "seen")  # the student did not even run


def test_the_gender_slider_reaches_the_student():
    e = editor()
    result = e.process(frame(), Settings(gender=-1.0, age=12.0), t=0.0)  # all the way towards female
    shape, gender, years = e.student.seen
    assert shape == (1, 3, 256, 256) and gender == [GENDER_RANGE] and years == [12.0]
    cx, cy = LANDMARKS.mean(0).astype(int)
    assert result.image[cy, cx, 0] > 100  # brightened where the face is
    assert result.image[5, 5, 0] == 100  # untouched far away
    assert {"detect", "student", "paste"} <= set(result.timings)


def test_switched_off_or_no_faces():
    image = frame()
    assert np.array_equal(editor().process(image, Settings(gender=1, enabled=False)).image, image)
    assert editor(faces=[]).process(image, Settings(gender=1)).faces == []


def test_amounts():
    assert FaceEditor.amounts(Settings(gender=1.0, age=-20)) == (-GENDER_RANGE, -20)
