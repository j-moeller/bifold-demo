import numpy as np
import pytest

from helpers import silhouette

from bifold_demo.people import PeopleLibrary
from bifold_demo.scenes import fair


@pytest.fixture
def library(tmp_path):
    lib = PeopleLibrary(tmp_path / "people")
    for i in range(3):
        lib.add(silhouette(color=(40 * i, 80, 120)), name=f"stand{i}", height=1.7 + 0.05 * i)
    for i in range(2):
        lib.add(silhouette(), name=f"walk{i}", pose="walking", facing="left")
    lib.add(silhouette(120, 100), name="bust", full_body=False, height=0.95)
    return lib


def _build(monkeypatch, people):
    monkeypatch.setattr(fair, "people_for_scene", lambda folder: people)
    return fair.build_scene()


def test_fair_scene_layout(monkeypatch, library):
    scene = _build(monkeypatch, library.people())
    names = set(scene.objects)
    assert {f"visitor{i}" for i in range(1, fair.VISITORS + 1)} <= names
    assert {f"walker{i}" for i in range(1, fair.WALKERS + 1)} <= names
    assert {"floor", "stage_screen", "banner", "speaker", "booth1_wall", "booth8_sign", "staff1"} <= names

    for i in range(90):  # 9 seconds
        scene.update(i / 10, 0.1 if i else 0.0)
    camera = np.array(fair.CAMERA_POSITION)
    for i in range(1, fair.VISITORS + 1):
        offset = scene[f"visitor{i}"].current_position - camera
        assert 1.5 < np.hypot(offset[0], offset[2]) < 7  # in front of the camera ...
        assert offset[2] < 0
    for i in range(1, fair.WALKERS + 1):
        walker = scene[f"walker{i}"]
        assert abs(walker.current_position[0]) <= fair.HALL_WIDTH / 2  # ... walkers stay inside the hall
        assert min(abs(walker.current_position[2] - z) for z in fair.WALK_AISLES) < 0.6


def test_walkers_face_where_they_go(monkeypatch, library):
    scene = _build(monkeypatch, library.people())
    walkers = [scene[f"walker{i}"] for i in (1, 2)]  # walker1 goes right, walker2 left
    scene.update(0.0, 0.0)
    start = [w.current_position[0] for w in walkers]
    scene.update(0.5, 0.5)
    moved = [w.current_position[0] - s for w, s in zip(walkers, start)]
    assert moved[0] > 0 > moved[1]
    left_facing = library.people()[-1].texture().rgba  # walk1 faces left like walk0
    textures = [w.current_texture.rgba for w in walkers]
    assert any(not np.array_equal(t, left_facing) for t in textures)  # the one going right is mirrored


def test_fair_scene_without_people(monkeypatch):
    scene = _build(monkeypatch, [])
    assert "visitor1" not in scene and "booth1_wall" in scene
    scene.update(1.0, 0.1)


def test_staff_stand_behind_counters(monkeypatch, library):
    scene = _build(monkeypatch, library.people())
    scene.update(0.0, 0.0)
    staff = scene["staff1"]
    top = staff.current_position[1] + 0.95 / 2  # busts are 0.95 m tall
    assert 1.6 < top < 1.85  # head at standing height, the counter hides the cut-off waist


def test_fair_scene_renders(gl, monkeypatch, library):
    from bifold_demo.scene.renderer import Renderer

    scene = _build(monkeypatch, library.people())
    renderer = Renderer(320, 180, samples=0)
    try:
        scene.update(2.0, 0.1)
        image = renderer.render(scene)
    finally:
        renderer.release()
    assert image.shape == (180, 320, 3) and image.std() > 20  # a detailed picture, not a blank frame


@pytest.mark.yolo
def test_yolo_sees_the_default_people_in_the_hall(gl, tmp_path):
    pytest.importorskip("ultralytics")
    from bifold_demo.detection import YoloDetector
    from bifold_demo.people import default_people
    from bifold_demo.scene.renderer import Renderer

    try:
        people = default_people(cache=tmp_path / "cache", models_dir=tmp_path / "models")
        detector = YoloDetector("yolo26n.pt", device="cpu", models_dir=tmp_path / "models")
    except Exception as exc:  # noqa: BLE001 - offline machine
        pytest.skip(f"cannot load YOLO weights: {exc}")
    if not people:
        pytest.skip("no default people available")
    fair_people = fair.people_for_scene
    try:
        fair.people_for_scene = lambda folder: people
        scene = fair.build_scene()
    finally:
        fair.people_for_scene = fair_people
    renderer = Renderer(1280, 720)
    try:
        scene.update(3.0, 0.1)
        persons = [d for d in detector.detect(renderer.render(scene)) if d.label == "person"]
    finally:
        renderer.release()
    assert len(persons) >= fair.VISITORS
