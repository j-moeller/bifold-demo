"""The virtual high-resolution sensor (bifold_demo.sensor): its geometry, and windows that are exactly crops of a
render of the whole frame at the sensor's resolution. The tests with real avatars need ``BIFOLD_CHARACTERS_TEST_DIR``.
"""

import math
import os
from pathlib import Path

import numpy as np
import pytest

from bifold_demo.characters.character import Character
from bifold_demo.characters.model import CharacterModel, CharacterState, Skeleton, SkinnedPrimitive
from bifold_demo.scene import Box, Camera, Fog, Plane, Quad, Scene, Sphere, Texture, checkerboard
from bifold_demo.scene.geometry import box_mesh
from bifold_demo.scene.scene import SubCamera
from bifold_demo.scene.transforms import crop_matrix
from bifold_demo.sensor import Sensor

CHARACTERS = Path(os.environ.get("BIFOLD_CHARACTERS_TEST_DIR", "/nonexistent")).expanduser()
VIEW = (320, 180)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return math.inf if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)


# -- geometry ---------------------------------------------------------------------------------------------------------


def test_a_sensor_has_the_views_shape_and_its_binnings_as_levels():
    sensor = Sensor(7680, 4320, (1280, 720))
    assert sensor.scale == 6 and sensor.view_aspect == pytest.approx(16 / 9)
    assert sensor.levels[:8] == (1, 2, 3, 4, 5, 6, 8, 10) and sensor.levels[-1] == 480
    assert all(7680 % k == 0 and 4320 % k == 0 for k in sensor.levels)
    assert sensor.frame(6) == (1280, 720)
    assert Sensor.with_lines(2160, (1280, 720)).size == (3840, 2160)
    assert Sensor.scaled(1.5, (1280, 720)).size == (1920, 1080)
    assert Sensor.with_lines(4320, (1366, 768)).size == (7684, 4320)  # rounded, within the tolerance
    with pytest.raises(ValueError, match="shape"):
        Sensor(4096, 3072, (1280, 720))
    with pytest.raises(ValueError, match="level"):
        sensor.frame(7)


def test_windows_are_snapped_to_the_level_grid_and_cover_the_region():
    sensor = Sensor(7680, 4320, (1280, 720))
    region = (100.3, 50.6, 140.2, 91.1)
    for level in (1, 2, 3, 5):
        w = sensor.window(region, level)
        x0, y0, x1, y1 = w.sensor_rect
        assert all(v % level == 0 for v in (x0, y0, x1, y1))  # on the level's grid
        vx0, vy0, vx1, vy1 = w.view_rect
        assert vx0 <= region[0] and vy0 <= region[1] and vx1 >= region[2] and vy1 >= region[3]
        assert vx0 > region[0] - level / 6 and vx1 < region[2] + level / 6  # the smallest such window
        assert w.size == (w.width, w.height) and w.pixels == w.width * w.height
    exact = sensor.window((10, 20, 30, 40), 1)  # on the grid already: no extra pixel from rounding
    assert exact.sensor_rect == (60, 120, 180, 240)


def test_windows_grow_to_a_multiple_and_stay_inside_the_frame():
    sensor = Sensor(7680, 4320, (1280, 720))
    w = sensor.window((100, 100, 110.5, 107), 1, multiple=16)
    assert w.width % 16 == 0 and w.height % 16 == 0
    vx0, _, vx1, _ = w.view_rect
    assert vx0 <= 100 and vx1 >= 110.5 and abs((vx0 + vx1) / 2 - 105.25) < 2 / 6  # grown about the centre
    corner = sensor.window((-30, -20, 40, 50), 1)  # over the top-left corner: moved inside
    assert corner.sensor_rect[:2] == (0, 0) and corner.size == (420, 420)
    edge = sensor.window((1250, 700, 1300, 760), 2)  # over the bottom-right corner
    assert edge.sensor_rect[2:] == (7680, 4320)
    huge = sensor.window((-500, -500, 2000, 2000), 4)  # larger than the frame: cut to it
    assert huge.sensor_rect == (0, 0, 7680, 4320) and huge.size == sensor.frame(4)
    with pytest.raises(ValueError, match="empty"):
        sensor.window((5, 5, 5, 9))


def test_the_plan_picks_the_coarsest_level_that_still_gives_the_face_its_pixels():
    sensor = Sensor(7680, 4320, (1280, 720))
    region = (600, 100, 700, 200)  # 100 view pixels square
    far = sensor.plan(region, ied=11, face_px=112)  # 66 px at most: the sensor itself
    assert far.level == 1 and 11 * far.scale == pytest.approx(66)
    near = sensor.plan(region, ied=44, face_px=112)  # 264 px on the sensor: binned by 2 still gives 132
    assert near.level == 2 and 44 * near.scale == pytest.approx(132)
    assert sensor.plan(region, ied=44, face_px=60).level == 4  # 66 px
    limited = sensor.plan(region, ied=11, face_px=112, max_pixels=300 * 300)  # 600 x 600 at level 1 is too many
    assert limited.level == 2 and limited.pixels <= 300 * 300
    assert sensor.plan(region).level == 1  # nothing asked: the finest
    assert sensor.plan(region, max_pixels=100 * 100).pixels <= 100 * 100


def test_pixels_map_between_view_sensor_and_window_both_ways():
    sensor = Sensor(5120, 2880, (1280, 720))
    w = sensor.window((300.2, 100.7, 380.9, 190.1), 2)
    points = np.array([[0.0, 0.0], [w.width, w.height], [12.5, 30.25]])
    view = w.to_view(points)
    assert np.allclose(w.from_view(view), points)
    assert np.allclose(view[0], w.view_rect[:2]) and np.allclose(view[1], w.view_rect[2:])
    assert np.allclose(w.from_sensor(w.to_sensor(points)), points)
    assert np.allclose(sensor.to_view(sensor.from_view(view)), view)
    assert np.allclose(view[2], (np.array([12.5, 30.25]) * 2 + (w.x * 2, w.y * 2)) / 4)  # 4 sensor px per view px
    box = (10.0, 20.0, 50.0, 70.0)
    assert np.allclose(w.box_from_view(w.box_to_view(box)), box)
    assert w.scale == 2 and str(w).startswith(f"{w.width}x{w.height} at level 2")


def test_the_window_projection_puts_points_where_the_view_has_them():
    """A point projected through the window camera lands at the window pixel that the view pixel maps to."""
    camera = Camera(position=(0.3, 1.6, 6.0), target=(0.0, 1.2, -5.0), fov=50)
    sensor = Sensor(3840, 2160, (1280, 720))
    w = sensor.window((500, 200, 700, 330), 1)
    sub = w.camera(camera)
    assert isinstance(sub, SubCamera) and np.allclose(sub.current_position, camera.current_position)
    assert np.allclose(sub.projection_matrix(1.0), w.crop_matrix() @ camera.projection_matrix(16 / 9))
    def pixel(point, proj, size):
        clip = proj @ camera.view_matrix() @ np.append(point, 1.0)
        ndc = clip[:2] / clip[3]
        return np.array([(ndc[0] + 1) / 2 * size[0], (1 - ndc[1]) / 2 * size[1]])

    for point in ([0.2, 1.4, 0.0], [-0.4, 1.0, 1.5]):
        in_view = pixel(point, camera.projection_matrix(16 / 9), (1280, 720))
        assert np.allclose(pixel(point, sub.projection_matrix(None), w.size), w.from_view(in_view))
    with pytest.raises(ValueError, match="empty"):
        crop_matrix(0.5, 0.0, 0.5, 1.0)


# -- windows drawn ----------------------------------------------------------------------------------------------------


def _boxy_model() -> CharacterModel:
    """A small skinned person: a red body, a blue head with a green nose (bones Hips, Neck, Head)."""
    rest = np.repeat(np.eye(4)[None], 3, axis=0)
    rest[1, 1, 3], rest[2, 1, 3] = 1.2, 0.1
    skeleton = Skeleton(["Hips", "Neck", "Head"], [-1, 0, 1], rest)

    def box(lo, hi, bone, color):
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        mesh = box_mesh(*(hi - lo))
        positions = mesh.positions + ((lo + hi) / 2).astype(np.float32)
        joints = np.zeros((len(positions), 4), np.int64)
        weights = np.zeros((len(positions), 4), np.float32)
        joints[:, 0], weights[:, 0] = bone, 1.0
        rgba = np.zeros((4, 4, 4), np.uint8)
        rgba[...] = (*color, 255)
        return SkinnedPrimitive(f"box{bone}", positions, mesh.normals, mesh.uvs, mesh.indices, joints, weights, rgba)

    primitives = [box((-0.2, 0.0, -0.12), (0.2, 1.2, 0.12), 0, (200, 40, 40)),
                  box((-0.12, 1.3, -0.12), (0.12, 1.6, 0.12), 2, (40, 60, 200)),
                  box((-0.04, 1.4, 0.12), (0.04, 1.48, 0.4), 2, (40, 200, 60))]
    return CharacterModel("boxy", skeleton, primitives, np.linalg.inv(skeleton.rest_global))


def _standing(x, z, heading):
    state = CharacterState(position=(x, 0.0, z), heading=heading, layers=[])
    return lambda t: state


def sample_scene(background=(40, 60, 90)) -> Scene:
    """Textured boxes, a sphere, a poster, two skinned people with blob shadows, a floor and fog."""
    scene = Scene(background=background, fog=Fog(start=3, end=30, color=(200, 200, 210), strength=0.6))
    scene.camera = Camera(position=(0, 1.6, 6), target=(0, 1.2, -6), fov=55)
    scene.add(Plane(size=(20, 40), texture=checkerboard(8, 2), uv_repeat=10), name="floor")
    for i in range(6):
        scene.add(Box(size=(0.6, 1.2, 0.4), texture=checkerboard(4, 1), color=(200, 120 + 20 * i, 90),
                      position=(-3 + i * 1.3, 0.6, -1 - 2 * i), rotation=(0, 17 * i, 0)), name=f"box{i}")
    scene.add(Sphere(radius=0.4, texture=checkerboard(8, 4), position=(0.5, 1.5, 1.0)), name="ball")
    scene.add(Quad(size=(1.0, 0.7), texture=checkerboard(16, 1), position=(-1.2, 1.4, 2.0), lit=False), name="poster")
    model = _boxy_model()
    scene.add(Character(model, _standing(-0.6, 2.5, 20.0)), name="near")
    scene.add(Character(model, _standing(1.4, -4.0, -30.0)), name="far")
    scene.update(0.0, 0.0)
    return scene


def _windows_against_full_renders(scene, sensor, cases, min_psnr=50.0):
    from bifold_demo.scene.renderer import Renderer

    view = Renderer(*sensor.view)
    fulls = {}
    try:
        view.render(scene)
        for region, level in cases:
            w = sensor.window(region, level)
            image = view.render_window(scene, w.ndc, w.size)
            assert image.shape == (w.height, w.width, 3) and image.dtype == np.uint8
            if level not in fulls:
                full = Renderer(*sensor.frame(level))
                try:
                    fulls[level] = full.render(scene)
                finally:
                    full.release()
            crop = fulls[level][w.y:w.y + w.height, w.x:w.x + w.width]
            diff = np.abs(crop.astype(np.int16) - image.astype(np.int16)).max(axis=2)
            assert psnr(crop, image) >= min_psnr, (w, psnr(crop, image))
            assert (diff > 8).mean() < 0.002, w  # only a few silhouette pixels, where MSAA samples fall differently
    finally:
        view.release()


def test_a_window_is_the_crop_of_a_render_of_the_whole_sensor(gl):
    scene = sample_scene()
    sensor = Sensor.scaled(4, VIEW)  # 1280 x 720, levels 1, 2, 4, 5, ...
    cases = [((100, 40, 140, 80), 1), ((0, 0, 30, 30), 1), ((300, 150, 320, 180), 1), ((120, 60, 190, 130), 1),
             ((150, 60, 200, 110), 2), ((10, 100, 90, 170), 4), ((140, 80, 160, 100), 5)]
    _windows_against_full_renders(scene, sensor, cases)


def test_a_window_shows_its_part_of_a_background_image(gl):
    rgb = np.zeros((90, 300, 3), np.uint8)  # wider than the view: "cover" crops its sides
    rgb[:, :, 0] = np.linspace(0, 255, 300)[None, :]
    rgb[:, :, 1] = np.linspace(0, 255, 90)[:, None]
    scene = sample_scene(background=Texture.from_array(rgb, order="rgb"))
    sensor = Sensor.scaled(3, VIEW)
    _windows_against_full_renders(scene, sensor, [((0, 0, 60, 40), 1), ((250, 10, 320, 60), 1), ((60, 20, 200, 90), 3)])


def test_a_sub_camera_draws_the_same_window_with_its_own_renderer(gl):
    from bifold_demo.scene.renderer import Renderer

    scene = sample_scene()
    w = Sensor.scaled(4, VIEW).window((110, 50, 170, 110), 1)
    view = Renderer(*VIEW)
    try:
        view.render(scene)
        drawn = view.render_window(scene, w.ndc, w.size)
    finally:
        view.release()
    own = Renderer(*w.size)
    camera, scene.camera = scene.camera, w.camera(scene.camera)
    try:
        assert np.array_equal(own.render(scene), drawn)
    finally:
        scene.camera = camera
        own.release()


def test_windows_draw_in_the_renderers_own_context(gl):
    """With another renderer made after it (its context current), a renderer's windows are still its own."""
    from bifold_demo.scene.renderer import WINDOW_TARGETS, Renderer

    scene = sample_scene()
    sensor = Sensor.scaled(4, VIEW)
    view = Renderer(*VIEW)
    other = None
    try:
        view.render(scene)
        w = sensor.window((100, 40, 160, 100), 1)
        before = view.render_window(scene, w.ndc, w.size)
        other = Renderer(64, 64)  # current now
        assert np.array_equal(view.render_window(scene, w.ndc, w.size), before)
        for k in range(WINDOW_TARGETS + 2):  # more sizes than are kept: the oldest framebuffers are freed
            v = sensor.window((10 + k, 10, 40 + 3 * k, 50), 1)
            assert view.render_window(scene, v.ndc, v.size).shape == (v.height, v.width, 3)
        assert len(view._window_targets) == WINDOW_TARGETS
        assert np.array_equal(view.render_window(scene, w.ndc, w.size), before)
        with pytest.raises(ValueError, match="cannot be drawn"):
            view.render_window(scene, w.ndc, (view.max_window + 1, 10))
    finally:
        view.release()
        if other is not None:
            other.release()


# -- real avatars -----------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def avatar_scene():
    from bifold_demo.characters.library import CharacterLibrary

    if not (CHARACTERS / "avatars").is_dir():
        pytest.skip(f"no prepared characters in {CHARACTERS} (set BIFOLD_CHARACTERS_TEST_DIR)")
    library = CharacterLibrary(CHARACTERS)
    names = sorted(library.available())[:3]
    if not names:
        pytest.skip("the prepared library has no avatars")
    scene = Scene(background=(205, 207, 212), ambient=0.62, light_direction=(-0.25, -1.0, -0.35),
                  fog=Fog(start=18, end=95, color=(214, 216, 222), strength=0.55))
    scene.camera = Camera(position=(0.0, 1.6, 6.0), target=(0.0, 1.5, -12.0), fov=55.0, near=0.1)
    scene.add(Plane(size=(20, 40), texture=checkerboard(8, 2), uv_repeat=10), name="floor")
    people = []
    for k, (x, z, heading) in enumerate([(-0.9, 2.2, 160.0), (0.6, 1.6, 200.0), (1.5, -3.0, 180.0)]):
        person = Character(library.load(names[k % len(names)]), _standing(x, z, heading), name=f"p{k}")
        scene.add(person)
        people.append(person)
    scene.update(0.0, 0.0)
    for person in people:
        for job in person.texture_jobs:
            job.result()
    return scene, people


def _eyes_in_view(person, camera, view):
    skeleton = person.skinned_model.skeleton
    model = person.matrices()[0]
    points = []
    for name in ("Bip01 REye", "Bip01 LEye"):
        local = person.pose[skeleton.find(name), :3, 3]
        world = model[:3, :3] @ local + model[:3, 3]
        clip = camera.projection_matrix(view[0] / view[1]) @ camera.view_matrix() @ np.append(world, 1.0)
        ndc = clip[:2] / clip[3]
        points.append(((ndc[0] + 1) / 2 * view[0], (1 - ndc[1]) / 2 * view[1]))
    return np.array(points)


def test_real_avatar_head_windows_are_crops_of_the_whole_sensor(gl, avatar_scene):
    scene, people = avatar_scene
    view = (640, 360)
    sensor = Sensor.scaled(4, view)  # 2560 x 1440
    cases = []
    for person in people:
        eyes = _eyes_in_view(person, scene.camera, view)
        cx, cy = eyes.mean(axis=0)
        half = 2.5 * np.linalg.norm(eyes[0] - eyes[1])
        cases += [((cx - half, cy - half, cx + half, cy + half), level) for level in (1, 2)]
    _windows_against_full_renders(scene, sensor, cases, min_psnr=45.0)
