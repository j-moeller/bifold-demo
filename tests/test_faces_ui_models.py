import cv2
import numpy as np
import pytest

from bifold_morph import models
from bifold_morph.faces import Face, FaceTracker, OneEuro
from bifold_morph.state import AgeGender
from bifold_morph.ui import LEFT, RIGHT, UP, Panel, Slider, describe_label, key_code


def face_at(x, y, size=100.0):
    lm = np.array([[x - 15, y - 10], [x + 15, y - 10], [x, y + 5], [x - 12, y + 25], [x + 12, y + 25]], float)
    return Face((x - size / 2, y - size / 2, size, size), lm, 0.9)


def test_one_euro_holds_still_and_follows():
    f = OneEuro(min_cutoff=1.0, beta=0.0)
    assert f(np.array([0.0]), 0.0) == pytest.approx([0.0])
    noisy = [f(np.array([1.0 if i % 2 else -1.0]), i / 30) for i in range(1, 60)]
    assert abs(noisy[-1][0]) < 0.3  # jitter is damped
    for i in range(60, 200):
        out = f(np.array([10.0]), i / 30)
    assert out[0] == pytest.approx(10.0, abs=0.01)  # a held value is reached


def test_tracker_keeps_ids_and_forgets():
    tracker = FaceTracker(keep=0.5)
    a, b = tracker.update([face_at(100, 100), face_at(400, 100)], 0.0)
    a2, b2 = tracker.update([face_at(405, 102), face_at(103, 99)], 0.1)
    assert (a2.id, b2.id) == (b.id, a.id)  # matched by position, not by order
    assert tracker.update([face_at(700, 300)], 0.2)[0].id not in (a.id, b.id)
    tracker.update([], 1.0)
    assert tracker.tracks == []


def test_slider_values():
    s = Slider("Age", -40, 40, "younger", "older", str, step=5.0)
    s.track = (100, 500, 20)
    assert s.value_at(100) == -40 and s.value_at(500) == 40
    assert s.value_at(302) == 0.0  # snaps to the middle
    assert s.value_at(1000) == 40
    s.set(100)
    assert s.value == 40
    s.set(2.0)
    s.nudge(-1)
    assert s.value == -3.0
    s.set(1.0)
    s.nudge(-1)
    assert s.value == -4.0
    s.set(2.0)
    s.nudge(-0.5)  # ends within half a step of 0: snaps to it
    assert s.value == 0.0


def test_panel_mouse_and_keys():
    age = Slider("Age", -40, 40, "younger", "older", lambda v: f"{v:+.0f}", step=5.0)
    gender = Slider("Gender", -1, 1, "female", "male", lambda v: f"{v:.2f}", step=0.1)
    panel = Panel([age, gender])
    strip = panel.draw(1280, "status", "help")
    assert strip.shape == (panel.height(1280), 1280, 3)
    x0, x1, y = gender.track
    assert panel.mouse(cv2.EVENT_LBUTTONDOWN, x1, y, 0) and gender.value == 1.0 and panel.selected == 1
    assert panel.mouse(cv2.EVENT_MOUSEMOVE, x0, y, cv2.EVENT_FLAG_LBUTTON) and gender.value == -1.0
    panel.mouse(cv2.EVENT_LBUTTONUP, x0, y, 0)
    assert not panel.mouse(cv2.EVENT_MOUSEMOVE, x1, y, 0) and gender.value == -1.0  # not dragging any more
    assert panel.mouse(cv2.EVENT_RBUTTONDOWN, (x0 + x1) // 2, y, 0) and gender.value == 0.0
    assert not panel.mouse(cv2.EVENT_LBUTTONDOWN, x0, 2, 0)  # above the tracks: no slider
    assert panel.key(UP) and panel.selected == 0
    assert panel.key(RIGHT) and age.value == 5.0
    assert panel.key(LEFT) and age.value == 0.0
    assert not panel.key(ord("x"))


def test_key_codes():
    assert key_code(-1) is None
    assert key_code(ord("A")) == ord("a")
    assert key_code(0xF702) == LEFT and key_code(0xFF53) == RIGHT and key_code(0x260000) == UP
    assert key_code(27) == 27


def test_labels():
    young, old = AgeGender(31.2, 0.98), AgeGender(52.4, 0.19)
    assert describe_label(young, old, edited=True) == "31 -> 52 y, male 98% -> female 81%"
    assert describe_label(young, young, edited=False) == "31 y, male 98%"
    assert describe_label(None, None, edited=True) == ""


def test_models_folder_and_checks(tmp_path, monkeypatch):
    monkeypatch.setenv(models.MODELS_ENV, str(tmp_path))
    assert models.models_dir() == tmp_path
    with pytest.raises(FileNotFoundError):
        models.ensure(models.FAIRFACE, download=False)
    fake = models.ModelFile("x.bin", "test", url="http://unused", sha256="0" * 64, size=24)
    (tmp_path / "x.bin").write_bytes(b"PK\x03\x04" + b"\0" * 20)
    assert models.ensure(fake, download=False) == tmp_path / "x.bin"  # the right size: taken as it is
    (tmp_path / "x.bin").write_bytes(b"short")
    with pytest.raises(FileNotFoundError):
        models.ensure(fake, download=False)  # damaged and not allowed to fetch
    assert all(m.url.startswith("https://") and len(m.sha256) == 64 for m in models.ALL)
