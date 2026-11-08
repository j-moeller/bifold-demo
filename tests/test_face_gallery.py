"""The gallery of known people: enrolment checks, duplicates, pose-bin templates, near-twins, the decision hook and the
per-model caches, with a scripted stand-in engine (no model needed)."""

import dataclasses
import math
import os

import cv2
import numpy as np
import pytest

from bifold_demo.recognition import gallery as G
from bifold_demo.recognition.align import Aligned
from bifold_demo.recognition.detect import Face
from bifold_demo.recognition.engine import FaceRecord
from bifold_demo.recognition.quality import Quality

DIM = 64


def _unit(seed, noise=0.0, base=None):
    rng = np.random.default_rng(seed)
    v = (base if base is not None else rng.normal(size=DIM)) + noise * rng.normal(size=DIM)
    return v / np.linalg.norm(v)


@dataclasses.dataclass
class Shot:
    """What the scripted engine finds in a photo of one colour."""

    vector: np.ndarray
    ied: float = 100.0
    yaw: float = 0.0
    sharpness: float = 0.2
    brightness: float = 120.0
    faces: int = 1
    second: float = 0.3  # the second face's size, relative to the first


class ScriptedEngine:
    """Photos are solid colours; ``shots`` says what each colour shows."""

    def __init__(self, shots, model_id="tiny", threshold=0.5, provider="cpu"):
        self.shots, self.model_id, self.threshold, self.provider_name = shots, model_id, threshold, provider
        self.analysed = 0

    def _shot(self, image):
        return self.shots[tuple(int(c) for c in image[0, 0])]

    def detect(self, image, min_size=0):
        if not image.any():
            return []
        shot = self._shot(image)
        marks = ((40.0, 50.0), (40.0 + shot.ied, 50.0), (60.0, 70.0), (45.0, 90.0), (75.0, 90.0))
        faces = [Face((10.0, 10.0, 200.0, 240.0), marks, 0.9)]
        if shot.faces > 1:
            faces.append(Face((300.0, 10.0, 200.0 * shot.second, 240.0 * shot.second), marks, 0.8))
        return faces

    def analyse(self, image, faces):
        self.analysed += 1
        shot = self._shot(image)
        q = Quality(shot.ied, shot.yaw, 0.0, 0.0, shot.sharpness, shot.brightness, 40.0, 0.0, 0.0, 0.9)
        crop = Aligned(np.zeros((112, 112, 3), np.uint8), np.eye(2, 3), 1.0, 0.0)
        return [FaceRecord(faces[0], crop, shot.vector, 20.0, q)]


def _photo(path, colour):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((24, 24, 3), colour, np.uint8))
    return path


def test_enrolment_checks_each_photo(tmp_path):
    shots = {(10, 0, 0): Shot(_unit(1)), (20, 0, 0): Shot(_unit(2), faces=2, second=0.8),
             (30, 0, 0): Shot(_unit(3), ied=30.0), (40, 0, 0): Shot(_unit(4), ied=60.0),
             (50, 0, 0): Shot(_unit(5), sharpness=0.01), (60, 0, 0): Shot(_unit(6), sharpness=0.03),
             (70, 0, 0): Shot(_unit(7), brightness=30.0), (80, 0, 0): Shot(_unit(8), yaw=70.0),
             (90, 0, 0): Shot(_unit(9), faces=2, second=0.4)}
    folder = tmp_path / "faces"
    for k, colour in enumerate(shots, 1):
        _photo(folder / f"P{k}" / "a.png", colour)
    _photo(folder / "P0" / "dark.png", (0, 0, 0))
    gallery = G.Gallery(folder, ScriptedEngine(shots))
    gallery.refresh(force=True)
    report = {str(p.relative_to(folder).parent): (status, problems) for p, status, problems in gallery.report()}
    assert report["P0"] == ("rejected", ("no face found",))
    assert report["P2"][0] == "rejected" and "2 faces of about the same size" in report["P2"][1][0]
    assert report["P3"][0] == "rejected" and "too small: 30 pixels" in report["P3"][1][0]
    assert report["P4"][0] == "ok" and "a small face: 60 pixels" in report["P4"][1][0]
    assert report["P5"][0] == "rejected" and "blurred" in report["P5"][1][0]
    assert report["P6"][0] == "ok" and "a little blurred" in report["P6"][1][0]
    assert report["P7"][0] == "ok" and "badly lit" in report["P7"][1][0]
    assert report["P8"][0] == "ok" and "profile view (+70 degrees)" in report["P8"][1][0]
    assert "P1" not in report and "P9" not in report  # a small second face is fine
    assert gallery.people() == {f"P{k}": 1 for k in (1, 4, 6, 7, 8, 9)}
    assert sorted(p.parent.name for p in gallery.problems()) == ["P0", "P2", "P3", "P5"]
    assert gallery.match(_unit(3)).name is None  # a rejected photo is no one's


def test_duplicates_templates_by_pose_and_near_twins(tmp_path):
    ada = _unit(1)
    ada_left = _unit(11, base=ada, noise=0.9)  # Ada turned: alike, not the same
    shots = {(10, 0, 0): Shot(ada, sharpness=0.3), (11, 0, 0): Shot(_unit(1, base=ada, noise=0.01), sharpness=0.1),
             (12, 0, 0): Shot(ada_left, yaw=35.0), (20, 0, 0): Shot(_unit(2)),
             (30, 0, 0): Shot(_unit(3, base=_unit(2), noise=0.3))}  # Cy looks like Bo
    folder = tmp_path / "faces"
    _photo(folder / "Ada" / "front.png", (10, 0, 0))
    _photo(folder / "Ada" / "again.png", (11, 0, 0))
    _photo(folder / "Ada" / "turned.png", (12, 0, 0))
    _photo(folder / "Bo.png", (20, 0, 0))
    _photo(folder / "Cy.png", (30, 0, 0))
    gallery = G.Gallery(folder, ScriptedEngine(shots))
    gallery.refresh(force=True)
    assert gallery.people() == {"Ada": 2, "Bo": 1, "Cy": 1}
    duplicate = [(p.name, problems) for p, status, problems in gallery.report() if status == "duplicate"]
    assert duplicate == [("again.png", ("nearly the same as front.png",))]  # the sharper one stays
    assert gallery.poses("Ada") == {"frontal": 1, "left": 1}
    best = gallery.scores(ada_left)[0]
    assert best.name == "Ada" and best.pose == "left" and best.score == pytest.approx(1.0)
    assert best.template == pytest.approx(1.0) and best.photo == pytest.approx(1.0)
    twins = gallery.near_twins()
    assert [name for name, _ in twins["Bo"]] == ["Cy"] and [name for name, _ in twins["Cy"]] == ["Bo"]
    assert "Ada" not in twins
    assert ("near-twin" in [status for _, status, _ in gallery.report()])
    assert gallery.photos("Ada") == [folder / "Ada" / "front.png", folder / "Ada" / "turned.png"]


def test_matching_gives_a_score_per_person_and_the_decision_can_be_replaced(tmp_path):
    shots = {(10, 0, 0): Shot(_unit(1)), (20, 0, 0): Shot(_unit(2))}
    folder = tmp_path / "faces"
    _photo(folder / "Ada.png", (10, 0, 0))
    _photo(folder / "Bo.png", (20, 0, 0))
    engine = ScriptedEngine(shots, threshold=0.6)
    gallery = G.Gallery(folder, engine)
    gallery.refresh(force=True)
    assert gallery.threshold == 0.6  # the engine's model's own
    probe = _unit(1, base=_unit(1), noise=0.6)
    scores = gallery.scores(probe)
    assert [s.name for s in scores] == ["Ada", "Bo"] and scores[0].score > scores[1].score
    match = gallery.match(probe)
    assert match.score == scores[0].score and match.runner_up == scores[1].score
    assert match.name == ("Ada" if scores[0].score >= 0.6 else None)
    gallery.threshold = -1.0
    assert gallery.match(probe).name == "Ada" and gallery.score(probe, "Bo") == scores[1].score
    seen = []
    gallery.decide = lambda s, q: seen.append((s, q)) or G.Match("Bo", s[1].score, s[0].score)  # e.g. F2's
    assert gallery.match(probe, quality="good").name == "Bo" and seen[0] == (scores, "good")
    with pytest.raises(ValueError, match="another model"):
        gallery.scores(np.ones(DIM + 1))


def test_caches_are_kept_per_model_and_forget_removed_photos(tmp_path):
    shots = {(10, 0, 0): Shot(_unit(1)), (20, 0, 0): Shot(_unit(2))}
    folder = tmp_path / "faces"
    _photo(folder / "Ada.png", (10, 0, 0))
    bo = _photo(folder / "Bo.png", (20, 0, 0))
    one, two = ScriptedEngine(shots, "model_one"), ScriptedEngine(shots, "model_two")
    G.Gallery(folder, one).refresh(force=True)
    G.Gallery(folder, two).refresh(force=True)
    assert (folder / G.cache_name("model_one")).is_file() and (folder / G.cache_name("model_two")).is_file()
    again = ScriptedEngine(shots, "model_one")
    first = G.Gallery(folder, again)
    first.refresh(force=True)
    assert again.analysed == 0 and first.people() == {"Ada": 1, "Bo": 1}  # from its own cache
    bo.unlink()  # Bo leaves: his embedding leaves every model's cache, also the one not in use
    first.refresh(force=True)
    for model in ("model_one", "model_two"):
        with np.load(folder / G.cache_name(model)) as data:
            assert [str(p) for p in data["paths"]] == ["Ada.png"]
    coreml = ScriptedEngine(shots, "model_one", provider="coreml")
    G.Gallery(folder, coreml).refresh(force=True)
    assert coreml.analysed == 1  # made by another provider: looked at again


def test_a_photo_replaced_under_its_name_is_enrolled_again_and_leaves_the_other_caches(tmp_path):
    """Review L3 and L4: a photo changed in place is looked at again (its size and time changed), and the other
    models' caches lose the old photo's embedding (it is no longer the person's photo)."""
    old, new = _unit(1), _unit(2)
    shots = {(10, 0, 0): Shot(old), (20, 0, 0): Shot(new)}
    folder = tmp_path / "faces"
    photo = _photo(folder / "Ada" / "a.png", (10, 0, 0))
    one = ScriptedEngine(shots, "model_one")
    gallery = G.Gallery(folder, one)
    gallery.refresh(force=True)
    G.Gallery(folder, ScriptedEngine(shots, "model_two")).refresh(force=True)
    _photo(photo, (20, 0, 0))  # another photo under the same name
    os.utime(photo, ns=(photo.stat().st_atime_ns, photo.stat().st_mtime_ns + 10**9))
    assert gallery.refresh(force=True) and one.analysed == 2
    assert gallery.match(new).name == "Ada" and gallery.scores(old)[0].score < 0.5
    with np.load(folder / G.cache_name("model_one")) as data:
        assert np.allclose(data["vectors"][0], new)
    with np.load(folder / G.cache_name("model_two")) as data:
        assert len(data["paths"]) == 0  # the replaced photo's embedding is gone


def test_a_photo_without_a_quality_record_is_still_enrolled():
    class Plain:  # an engine without analyse (the demo's old interface)
        def detect(self, image, min_size=0):
            return [Face((0.0, 0.0, 50.0, 60.0), ((0.0, 0.0),) * 5, 0.9)]

        def embed(self, image, face):
            return np.ones(4) / 2.0

    found = G.enrol(Plain(), np.zeros((8, 8, 3), np.uint8))
    assert found["vector"].tolist() == [0.5] * 4 and "status" not in found and found["face_size"] == 50.0
    assert G.enrol(Plain(), None)["problems"] == ("not an image",)
    assert math.isnan(G.Photo(None, "x", (0, 0), None).ied)
