"""The virtual sensor's window API (bifold_demo.sensor): requests, head regions, the scheduler, windows delivered
with the frames of a scene or a real camera, and what the scene knows about who is in them."""

import math
import time

import cv2
import numpy as np
import pytest
from test_sensor import VIEW, sample_scene

from bifold_demo.characters.model import CharacterState
from bifold_demo.scene import Camera
from bifold_demo.sensor import (
    HeadRegion,
    Scheduler,
    Sensor,
    WindowRequest,
    WindowService,
    crop_windows,
    render_windows,
)
from bifold_demo.sensor.requests import head_point_region
from bifold_demo.sensor.service import Motion, find_truth, plan_request
from bifold_demo.sources import FixedStepClock, Frame, SceneSource

# -- requests ---------------------------------------------------------------------------------------------------------


def test_the_head_region_is_a_square_around_the_top_of_the_box():
    head = HeadRegion()
    x0, y0, x1, y1 = head.region((100, 50, 160, 350))  # a box 300 pixels tall
    assert x1 - x0 == pytest.approx(0.30 * 1.30 * 300) and y1 - y0 == pytest.approx(x1 - x0)
    assert (x0 + x1) / 2 == pytest.approx(130) and (y0 + y1) / 2 == pytest.approx(50 + 0.12 * 300)
    assert head.eye_distance((100, 50, 160, 350)) == pytest.approx(0.036 * 300)
    tight = HeadRegion(centre=0.07, side=0.15, margin=0.0)
    assert tight.region((0, 0, 10, 100)) == pytest.approx((-2.5, -0.5, 12.5, 14.5))


def test_a_request_names_exactly_one_region():
    assert WindowRequest("a", box=(1, 2, 30, 80)).box == (1.0, 2.0, 30.0, 80.0)
    with pytest.raises(ValueError, match="exactly one"):
        WindowRequest("a")
    with pytest.raises(ValueError, match="exactly one"):
        WindowRequest("a", box=(0, 0, 1, 1), region=(0, 0, 1, 1))
    with pytest.raises(ValueError, match="x1 > x0"):
        WindowRequest("a", region=(10, 0, 5, 20))


def test_a_head_point_is_seen_as_a_pinhole_camera_sees_it():
    camera = Camera(position=(0, 1.6, 6), target=(0, 1.6, -6), fov=55)
    view = (1280, 720)
    focal = 360 / math.tan(math.radians(55 / 2))  # pixels
    region, eyes = head_point_region(camera, view, (0, 1.6, 2.0), size=0.5)  # 4 m straight ahead
    assert eyes == pytest.approx(focal * 0.063 / 4.0, rel=1e-3)
    assert ((region[0] + region[2]) / 2, (region[1] + region[3]) / 2) == pytest.approx((640, 360), abs=1e-6)
    assert region[2] - region[0] == pytest.approx(focal * 0.5 / 4.0, rel=1e-3)
    _region, side_eyes = head_point_region(camera, view, (2.5, 1.6, 2.0))  # off to the side: stretched
    assert side_eyes > eyes
    assert head_point_region(camera, view, (0, 1.6, 9.0)) is None  # behind the camera


def test_requests_plan_their_windows():
    sensor = Sensor(7680, 4320, (1280, 720))
    camera = Camera(position=(0, 1.6, 6), target=(0, 1.6, -6), fov=55)
    box = (600, 200, 680, 500)  # a person 300 pixels tall: about 11 pixels between the eyes
    window, eyes = plan_request(WindowRequest("p", box=box), sensor)
    assert window.level == 1 and eyes == pytest.approx(0.036 * 300 * 6)
    x0, y0, x1, y1 = window.view_rect
    hx0, hy0, hx1, hy1 = HeadRegion().region(box)
    assert x0 <= hx0 and y0 <= hy0 and x1 >= hx1 and y1 >= hy1
    close, close_eyes = plan_request(WindowRequest("p", box=box, eye_distance=40, face_px=112), sensor, multiple=16)
    assert close.level == 2 and close_eyes == pytest.approx(120) and close.width % 16 == 0
    assert plan_request(WindowRequest("p", region=(1300, 10, 1400, 90)), sensor) is None  # not in the view
    point, point_eyes = plan_request(WindowRequest("p", head=(0, 1.6, 2.0), face_px=None), sensor, camera)
    focal = 360 / math.tan(math.radians(27.5))
    assert point.level == 1 and point_eyes == pytest.approx(6 * focal * 0.063 / 4, rel=1e-3)
    with pytest.raises(ValueError, match="camera"):
        plan_request(WindowRequest("p", head=(0, 1.6, 2.0)), sensor)


def test_the_scheduler_serves_at_most_n_by_priority_and_in_turn():
    requests = [WindowRequest(tag, region=(0, 0, 10, 10)) for tag in "abcdef"]
    scheduler = Scheduler(max_windows=4)
    served = []
    for frame in range(1, 7):
        chosen = scheduler.choose(requests, frame)
        assert len(chosen) == 4
        served += [r.tag for r in chosen]
    assert {tag: served.count(tag) for tag in "abcdef"} == {tag: 4 for tag in "abcdef"}  # in turn
    assert [r.tag for r in Scheduler(2).choose(requests, 1)] == ["a", "b"]  # the asked order among equals
    urgent = requests + [WindowRequest("z", region=(0, 0, 10, 10), priority=5.0)]
    assert scheduler.choose(urgent, 7)[0].tag == "z"
    assert scheduler.waited("z", 9) == 2 and scheduler.waited("never", 9) is None
    scheduler.choose(requests[:2], 10)  # c-f and z gone from the requests: forgotten
    assert scheduler.waited("c", 10) is None
    late = Scheduler(max_windows=1)
    late.choose([WindowRequest("x", region=(0, 0, 9, 9), priority=1)], 1)
    late.choose([WindowRequest("x", region=(0, 0, 9, 9), priority=1), WindowRequest("y", region=(0, 0, 9, 9))], 2)
    assert late.waited("y", 5) == 3  # never served: counted from its first request


def test_the_priority_hook_orders_the_windows():
    calls = []

    def hook(request, waited):
        calls.append((request.tag, waited))
        return {"slow": 0.0, "fast": 1.0}[request.tag] + waited * 0.4

    scheduler = Scheduler(max_windows=1, priority=hook)
    requests = [WindowRequest("slow", region=(0, 0, 9, 9)), WindowRequest("fast", region=(0, 0, 9, 9))]
    order = [scheduler.choose(requests, frame)[0].tag for frame in range(1, 8)]
    assert order[:4] == ["fast"] * 4 and "slow" in order  # the hook's aging lets the low priority in
    assert ("slow", 0) in calls and ("slow", 3) in calls


# -- windows of a scene -----------------------------------------------------------------------------------------------


def _person_box(scene, person, view=VIEW):
    """The view-pixel box of ``person``'s posed skeleton, as a detector would give it (a little padded)."""
    from bifold_demo.selection import person_box

    clip_from_world = scene.camera.projection_matrix(view[0] / view[1]) @ scene.camera.view_matrix()
    return person_box(person, clip_from_world, view)


def test_windows_of_a_person_box_hold_their_head_and_know_who_it_is(gl):
    from bifold_demo.scene.renderer import Renderer

    scene = sample_scene()
    renderer = Renderer(*VIEW)
    try:
        renderer.render(scene)
        box = _person_box(scene, scene["near"])
        requests = [WindowRequest("near", box=box, face_px=None), WindowRequest("gone", region=(400, 0, 500, 50))]
        sensor = Sensor.scaled(4, VIEW)
        results = render_windows(scene, renderer, requests, sensor, frame=7)
        assert [r.tag for r in results] == ["near"]  # the region outside the view is left out
        result = results[0]
        assert result.image.shape == (result.window.height, result.window.width, 3)
        assert result.time == scene.time and result.frame == 7 and result.preset == "ideal"
        assert result.window.level == 1 and result.window.width % 16 == 0
        assert result.render_ms > 0
        truth = result.truth  # the boxy people have no eye bones: nobody known
        assert truth is None
        eyes = result.eye_distance
        assert eyes == pytest.approx(HeadRegion().eye_distance(box) * 4)
        assert np.allclose(result.to_view([[0, 0]]), [result.view_rect[:2]])
    finally:
        renderer.release()


def test_the_service_delivers_a_few_windows_with_every_frame_of_its_moment(gl):
    scene = sample_scene()
    service = WindowService(Sensor.scaled(4, VIEW), max_windows=2, truth=False)
    source = SceneSource(scene, *VIEW, clock=FixedStepClock(10.0), windows=service)
    try:
        first = source.read()
        assert first.windows == [] and first.time == 0.0  # nothing asked yet
        service.request([WindowRequest(tag, region=(20 + 30 * k, 40, 50 + 30 * k, 70)) for k, tag in enumerate("abc")])
        service.request_once([WindowRequest("once", region=(200, 100, 230, 130), priority=1.0)])
        frames = [source.read() for _ in range(3)]
        assert [[w.tag for w in f.windows] for f in frames] == [["once", "a"], ["b", "c"], ["a", "b"]]
        for frame in frames:
            for window in frame.windows:
                assert window.time == frame.time and window.frame == frame.index
        assert [f.time for f in frames] == pytest.approx([0.1, 0.2, 0.3])
        assert service.last is frames[-1].windows and len(service.requests) == 3
        service.clear()
        assert source.read().windows == []
    finally:
        source.stop()


def test_the_truth_finds_the_person_and_their_motion():
    """With eye bones: who is in a window, where their eyes are and how fast they move (window pixels)."""
    from bifold_demo.characters.character import Character
    from bifold_demo.characters.model import CharacterModel, Skeleton, SkinnedPrimitive
    from bifold_demo.scene import Scene
    from bifold_demo.scene.geometry import box_mesh

    rest = np.repeat(np.eye(4)[None], 3, axis=0)
    rest[1, :3, 3], rest[2, :3, 3] = (-0.032, 1.6, 0.1), (0.032, 1.6, 0.1)
    skeleton = Skeleton(["Root", "Bip01 REye", "Bip01 LEye"], [-1, 0, 0], rest)
    body = box_mesh(0.4, 1.7, 0.2)
    joints, weights = np.zeros((len(body.positions), 4), np.int64), np.zeros((len(body.positions), 4), np.float32)
    weights[:, 0] = 1.0
    part = SkinnedPrimitive("body", body.positions + np.float32([0, 0.85, 0]), body.normals, body.uvs, body.indices,
                            joints, weights, np.full((2, 2, 4), 200, np.uint8))
    model = CharacterModel("eyes", skeleton, [part], np.linalg.inv(skeleton.rest_global))
    scene = Scene()
    scene.camera = Camera(position=(0, 1.6, 6), target=(0, 1.6, -6), fov=55)
    # facing the camera (+Z): the eyes 0.1 m in front of the root, the person's left (+X) the camera's right
    walker = scene.add(Character(model, lambda t: CharacterState((0.5 * t, 0.0, 2.0), 0.0, [])), name="walker")
    scene.add(Character(model, lambda t: CharacterState((-1.5, 0.0, 2.0), 0.0, [])), name="other")
    motion = Motion()
    for t in (0.0, 0.1):
        scene.update(t, 0.1)
        motion.observe(scene)
    assert np.allclose(motion.velocity(walker), [0.5, 0, 0])
    sensor = Sensor(5120, 2880, (1280, 720))
    window, _eyes = plan_request(WindowRequest("w", head=(0.05, 1.6, 2.1), face_px=None), sensor, scene.camera)
    truth = find_truth(scene, window, motion)
    assert truth.name == "walker" and truth.avatar == "eyes" and truth.inside
    assert truth.distance == pytest.approx(math.dist((0.05, 1.6, 2.1), (0, 1.6, 6)), abs=1e-6)
    focal = 360 / math.tan(math.radians(27.5)) * 4  # sensor pixels
    assert truth.eye_distance == pytest.approx(focal * 0.064 / 3.9, rel=0.01)
    # the person walks to their left = the camera's right (+x), at 0.5 m/s: about focal * 0.5 / 3.9 pixels a second
    assert truth.velocity[0] == pytest.approx(focal * 0.5 / 3.9, rel=0.02) and abs(truth.velocity[1]) < 1
    assert find_truth(scene, window, motion, subject=scene["other"]).name == "other"
    assert find_truth(scene, sensor.window((0, 0, 50, 50)), motion) is None  # nobody's eyes there


# -- windows of a real camera -----------------------------------------------------------------------------------------


def _pattern(w, h):
    rng = np.random.default_rng(3)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def test_windows_cut_from_a_full_resolution_picture():
    full = _pattern(1920, 1080)
    requests = [WindowRequest("a", region=(100, 50, 160, 120), face_px=None),
                WindowRequest("b", region=(500, 300, 700, 500), eye_distance=60, face_px=40),
                WindowRequest("h", head=(0, 1, 0))]
    results = crop_windows(full, (1280, 720), requests, frame=3, multiple=1)
    assert [r.tag for r in results] == ["a", "b"]  # a camera knows no head points
    a, b = results
    x0, y0, x1, y1 = a.window.sensor_rect
    assert a.window.level == 1 and np.array_equal(a.image, full[y0:y1, x0:x1])
    assert a.time is None and a.frame == 3 and a.truth is None
    assert b.window.level == 2  # 90 sensor pixels between the eyes, 45 binned
    x0, y0, x1, y1 = b.window.sensor_rect
    binned = full[y0:y1, x0:x1].reshape(b.window.height, 2, b.window.width, 2, 3).mean(axis=(1, 3))
    assert np.abs(b.image.astype(float) - binned).max() <= 1.0
    assert b.box_to_view((0, 0, b.window.width, b.window.height)) == pytest.approx(b.view_rect)


class _FakeCapture:
    """Stands in for cv2.VideoCapture: delivers one big picture over and over."""

    picture = None
    asked: dict | None = None  # what the source set: property -> value

    def __init__(self, *args):
        pass

    def isOpened(self):  # OpenCV's name
        return True

    def set(self, prop, value):
        _FakeCapture.asked[prop] = value
        return True

    def read(self):
        time.sleep(0.005)
        return True, _FakeCapture.picture.copy()

    def release(self):
        pass


def test_a_webcam_captures_high_detects_on_a_copy_and_cuts_windows_from_the_full_picture(monkeypatch):
    from bifold_demo.sources import webcam

    _FakeCapture.picture, _FakeCapture.asked = _pattern(1920, 1080), {}
    monkeypatch.setattr(webcam.cv2, "VideoCapture", _FakeCapture)
    service = WindowService(max_windows=1)
    source = webcam.WebcamSource(width=1280, height=720, mirror=False, capture=(1920, 1080), windows=service)
    try:
        service.request([WindowRequest("face", region=(400, 200, 440, 260), face_px=None)])
        frame = source.read(timeout=2.0)
        assert _FakeCapture.asked[cv2.CAP_PROP_FRAME_WIDTH] == 1920
        assert frame.image.shape == (720, 1280, 3)  # the detector's copy
        assert np.array_equal(frame.image, cv2.resize(_FakeCapture.picture, (1280, 720), interpolation=cv2.INTER_AREA))
        (window,) = frame.windows
        x0, y0, x1, y1 = window.window.sensor_rect
        assert window.window.sensor.size == (1920, 1080) and window.window.scale == 1.5
        assert np.array_equal(window.image, _FakeCapture.picture[y0:y1, x0:x1])
        later = source.read(timeout=2.0)
        assert later.index > frame.index and len(later.windows) == 1
    finally:
        source.stop()


def test_frames_carry_their_time_and_windows():
    frame = Frame(np.zeros((2, 2, 3), np.uint8), 1, 0.0, "scene")
    assert frame.time is None and frame.windows == []
