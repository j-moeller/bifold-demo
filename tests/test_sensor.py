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
    huge = Sensor.scaled(24, (1280, 720))  # 30720 x 17280: the whole view is wider than a renderer can draw
    whole = huge.plan((0, 0, 1280, 720), max_side=16384)
    assert whole.level == 2 and max(whole.size) <= 16384 and huge.plan((0, 0, 1280, 720)).level == 1


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
            # only a few pixels of silhouettes and alpha-tested hair cards, where MSAA samples fall differently,
            # and where a hair card and the scalp fight for depth (the two projections round differently)
            assert (diff > 8).sum() <= max(6, 0.004 * diff.size), w
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


# -- culling ----------------------------------------------------------------------------------------------------------


def test_frustum_planes_tell_boxes_and_spheres_outside():
    from bifold_demo.scene.frustum import boxes_outside, frustum_planes, spheres_outside, world_boxes

    camera = Camera(position=(0, 0, 0), target=(0, 0, -1), fov=60, near=0.1, far=50)
    planes = frustum_planes(camera.projection_matrix(1.0) @ camera.view_matrix())
    centres = np.array([[0, 0, -5], [0, 0, 5], [10, 0, -5], [0, 0, -80], [3.2, 0, -5]], float)
    halves = np.array([[0.1, 0.1, 0.1]] * 4 + [[0.5, 0.5, 0.5]])
    # in front; behind the camera; far to the side; beyond the far plane; overlapping the right side (half inside)
    assert boxes_outside(planes, centres, halves).tolist() == [False, True, True, True, False]
    assert spheres_outside(planes, centres, np.array([0.1] * 4 + [0.5])).tolist() == [False, True, True, True, False]
    assert spheres_outside(planes, centres[2:3], [8.0]).tolist() == [False]  # a big sphere reaches in
    turned = np.eye(4)
    turned[:3, :3] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]  # a quarter turn about y
    turned[:3, 3] = (1, 2, 3)
    centre, half = world_boxes([turned], [[1.0, 0, 0]], [[2.0, 0.5, 0.25]])
    assert np.allclose(centre, [[1, 2, 2]]) and np.allclose(half, [[0.25, 0.5, 2.0]])


def test_bone_spheres_hold_the_skinned_mesh():
    from bifold_demo.characters.character import skin_vertices
    from bifold_demo.characters.model import AnimationClip, Layer, quat_from_axis_angle
    from bifold_demo.scene.frustum import Bounds

    model = _boxy_model()
    bounds = Bounds()
    for degrees in (0.0, 35.0, 90.0, -120.0):
        clip = AnimationClip(f"neck{degrees}", 30.0, ["Neck"], quat_from_axis_angle([1, 0, 0], [[degrees]]),
                             np.zeros((1, 1, 3)))
        person = Character(model, lambda t, c=clip: CharacterState((0.3, 0.0, -1.0), 40.0, [Layer(c, 0.0, 1.0)]),
                           scale=1.3)
        person.update(0.0)
        centres, radii, owners = bounds.spheres([person], None)
        assert (owners == 0).all() and len(radii) == 3
        placed = person.matrices()[0]
        for points in skin_vertices(model, person.skin_palette):
            world = points @ placed[:3, :3].T + placed[:3, 3]
            d = np.linalg.norm(world[:, None, :] - centres[None], axis=2) - radii[None]
            assert (d.min(axis=1) <= 1e-6).all(), degrees  # every vertex in some bone's sphere


def test_culled_windows_are_the_same_and_leave_out_what_is_not_in_them(gl):
    from bifold_demo.scene.renderer import Renderer

    scene = sample_scene()
    sensor = Sensor.scaled(4, VIEW)
    view = Renderer(*VIEW)
    try:
        view.render(scene)
        for region in [(100, 40, 140, 80), (0, 0, 30, 30), (120, 60, 190, 130), (300, 150, 320, 180),
                       (140, 75, 175, 110), (60, 20, 110, 160), (150, 0, 170, 20)]:
            w = sensor.window(region, 1)
            culled = view.render_window(scene, w.ndc, w.size)
            stats = dict(view.window_stats)
            assert np.array_equal(culled, view.render_window(scene, w.ndc, w.size, cull=False)), region
            assert stats["objects_drawn"] <= stats["objects"] and stats["people_drawn"] <= stats["people"]
        sky = sensor.window((150, 0, 170, 20), 1)  # above everything but the back of the hall
        view.render_window(scene, sky.ndc, sky.size)
        assert view.window_stats["people_drawn"] == 0 and view.window_stats["objects_drawn"] < 3
        eyes = _project(scene, view, scene["near"], [0.0, 1.45, 0.4])  # the near person's nose
        head = sensor.window((eyes[0] - 6, eyes[1] - 6, eyes[0] + 6, eyes[1] + 6), 1)
        view.render_window(scene, head.ndc, head.size)
        assert view.window_stats["people_drawn"] == 1 and view.window_stats["shadows_drawn"] == 0
    finally:
        view.release()


def _project(scene, renderer, person, local):
    """``local`` (character space of ``person``) in view pixels."""
    world = person.matrices()[0] @ np.append(local, 1.0)
    clip = scene.camera.projection_matrix(renderer.aspect) @ scene.camera.view_matrix() @ world
    ndc = clip[:2] / clip[3]
    return (ndc[0] + 1) / 2 * renderer.width, (1 - ndc[1]) / 2 * renderer.height


def test_windows_follow_people_posed_again_after_the_frame(gl):
    """Windows reuse the joint palettes the frame uploaded, but not once the people were posed again."""
    from bifold_demo.characters.model import AnimationClip, Layer, quat_from_axis_angle
    from bifold_demo.scene.renderer import Renderer

    scene = sample_scene()
    sensor = Sensor.scaled(4, VIEW)
    nod = AnimationClip("nod", 30.0, ["Neck"], quat_from_axis_angle([1, 0, 0], [[50.0]]), np.zeros((1, 1, 3)))
    view = Renderer(*VIEW)
    try:
        view.render(scene)
        x, y = _project(scene, view, scene["near"], [0.0, 1.45, 0.2])
        w = sensor.window((x - 25, y - 25, x + 25, y + 25), 1)
        before = view.render_window(scene, w.ndc, w.size)
        scene["near"].behaviour = lambda t: CharacterState((-0.6, 0.0, 2.5), 20.0, [Layer(nod, 0.0, 1.0)])
        scene.update(0.0, 0.0)  # posed again, no frame drawn
        after = view.render_window(scene, w.ndc, w.size)
        assert not np.array_equal(after, before)
        own = Renderer(*w.size)
        camera, scene.camera = scene.camera, w.camera(scene.camera)
        try:
            assert np.array_equal(own.render(scene), after)
        finally:
            scene.camera = camera
            own.release()
    finally:
        view.release()


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


def test_real_avatars_stay_in_their_bone_spheres_when_animated():
    from bifold_demo.characters.character import skin_vertices
    from bifold_demo.characters.library import CharacterLibrary
    from bifold_demo.characters.model import Layer
    from bifold_demo.scene.frustum import Bounds

    if not (CHARACTERS / "avatars").is_dir():
        pytest.skip(f"no prepared characters in {CHARACTERS} (set BIFOLD_CHARACTERS_TEST_DIR)")
    library = CharacterLibrary(CHARACTERS)
    clips, roles = library.clips(), library.roles()
    chosen = [clips[name] for role in ("walk", "talk") for name in roles.get(role, [])[:1] if name in clips]
    names = sorted(library.available())[:2]
    if not chosen or not names:
        pytest.skip("the prepared library lacks avatars or walk and talk clips")
    bounds = Bounds()
    for name in names:
        model = library.load(name, textures="none")
        for clip in chosen:
            for t in np.linspace(0.0, clip.duration, 4):
                person = Character(model, lambda _t, c=clip, at=t: CharacterState((1.0, 0.0, -2.0), 30.0,
                                                                                   [Layer(c, at, 1.0)]))
                person.update(0.0)
                centres, radii, _owners = bounds.spheres([person], None)
                placed = person.matrices()[0]
                for points in skin_vertices(model, person.skin_palette):
                    world = points @ placed[:3, :3].T + placed[:3, 3]
                    gap = np.linalg.norm(world[:, None, :] - centres[None], axis=2) - radii[None]
                    assert (gap.min(axis=1) <= 1e-6).all(), (name, clip.name, t)


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
