"""The renderer's shading of mesh parts with normal and specular maps, the hemisphere light and the quality setting."""

import numpy as np
import pytest

from bifold_demo.characters.character import Character
from bifold_demo.characters.model import CharacterModel, Skeleton, SkinnedPrimitive
from bifold_demo.characters.tangents import vertex_tangents
from bifold_demo.scene import Camera, Plane, Scene

pytestmark = pytest.mark.gl


def _wall(tilt=(0.0, 0.0), specular=0, mirrored=False, maps=True):
    """A one-bone character: a 1 m square wall facing +z (u along +x, or -x when ``mirrored``; v up) whose packed
    map tilts every normal by ``tilt`` = (towards +u, towards +v) in tangent space, with a specular level."""
    positions = np.array([[-0.5, 0.5, 0], [0.5, 0.5, 0], [0.5, 1.5, 0], [-0.5, 1.5, 0]], np.float32)
    normals = np.tile(np.array([0, 0, 1], np.float32), (4, 1))
    uvs = np.array([[1, 0], [0, 0], [0, 1], [1, 1]] if mirrored else [[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
    indices = np.array([[0, 1, 2], [0, 2, 3]], np.uint32)
    n = np.array([np.sin(tilt[0]), np.sin(tilt[1]), 0.0])
    n[2] = np.sqrt(1 - n[0] ** 2 - n[1] ** 2)
    packed = np.zeros((8, 8, 4), np.uint8)
    packed[..., :2] = np.round((n[:2] + 1) * 127.5).astype(np.uint8)
    packed[..., 2], packed[..., 3] = specular, 255
    colour = np.full((8, 8, 4), 200, np.uint8)
    colour[..., 3] = 255
    primitive = SkinnedPrimitive("wall_head", positions, normals, uvs, indices, np.zeros((4, 4), np.int32),
                                 np.tile(np.array([1, 0, 0, 0], np.float32), (4, 1)), colour)
    if maps:
        primitive.tangents = vertex_tangents(positions, normals, uvs, indices)
        primitive.maps = packed
    skeleton = Skeleton(["Root"], [-1], np.eye(4)[None])
    return Character(CharacterModel("wall", skeleton, [primitive], np.eye(4)[None]), shadow=False)


def _render(person, light, quality="high", ambient=0.0, renderer=None, **settings):
    from bifold_demo.scene.renderer import Renderer

    scene = Scene(background=(0, 0, 0), ambient=ambient, light_direction=light)
    scene.camera = Camera(position=(0, 1, 2), target=(0, 1, 0), fov=40)
    scene.add(person, name="wall")
    scene.update(0.0, 0.0)
    own = renderer is None
    renderer = renderer or Renderer(64, 64, samples=0)
    try:
        renderer.quality = quality
        for key, value in settings.items():
            setattr(renderer, key, value)
        return renderer.render(scene)[32, 32].astype(int)  # BGR
    finally:
        if own:
            renderer.release()


TILT = np.radians(30)


def test_the_mapped_program_compiles_and_draws_a_flat_map_like_no_map(gl):
    from bifold_demo.scene.renderer import Renderer

    renderer = Renderer(64, 64, samples=0)
    try:
        assert renderer._mapped() is renderer._mapped()  # compiled once
        light = (-1.0, 0.0, -1.0)
        flat = _render(_wall(), light, renderer=renderer)
        plain = _render(_wall(maps=False), light, renderer=renderer)
        assert np.abs(flat - plain).max() <= 2 and flat[1] > 100
    finally:
        renderer.release()


@pytest.mark.parametrize("mirrored", [False, True])
def test_a_normal_map_turns_the_surface_the_way_its_texture_says(gl, mirrored):
    """A map tilting the normal towards +u brightens the wall under a light from the side u grows to and darkens it
    under a light from the other side; with mirrored UVs (+u to the left) the other way round."""
    toward_u = _wall(tilt=(TILT, 0.0), mirrored=mirrored)
    flat = _wall(mirrored=mirrored)
    from_right, from_left = (-1.0, 0.0, -1.0), (1.0, 0.0, -1.0)  # the light travels towards -x: it comes from +x
    brighter, darker = (from_left, from_right) if mirrored else (from_right, from_left)
    assert _render(toward_u, brighter)[1] > _render(flat, brighter)[1] + 15
    assert _render(toward_u, darker)[1] < _render(flat, darker)[1] - 15


def test_green_points_up_the_texture(gl):
    toward_v = _wall(tilt=(0.0, TILT))
    flat = _wall()
    from_above, from_below = (0.0, -1.0, -1.0), (0.0, 1.0, -1.0)
    assert _render(toward_v, from_above)[1] > _render(flat, from_above)[1] + 15
    assert _render(toward_v, from_below)[1] < _render(flat, from_below)[1] - 15


def test_the_specular_map_scales_a_highlight(gl):
    """Lit from the camera's direction, the wall reflects towards the camera: the higher its specular level, the
    brighter; level 0 adds nothing, and the renderer's ``specular`` scales it."""
    head_on = (0.0, 0.0, -1.0)
    levels = [_render(_wall(specular=s), head_on, ambient=0.2)[1] for s in (0, 18, 120)]
    assert levels[0] < levels[1] < levels[2]
    assert levels[0] == _render(_wall(maps=False), head_on, ambient=0.2)[1]
    assert _render(_wall(specular=120), head_on, ambient=0.2, specular=0.0)[1] == levels[0]


def test_the_low_quality_draws_without_maps(gl):
    light = (-1.0, 0.0, -1.0)
    tilted = _wall(tilt=(TILT, 0.0), specular=120)
    assert np.abs(_render(tilted, light, quality="low") - _render(_wall(maps=False), light, quality="low")).max() <= 1
    with pytest.raises(ValueError, match="quality"):
        _render(tilted, light, quality="best")


def test_windows_have_their_own_quality(gl):
    from bifold_demo.scene.renderer import Renderer

    scene = Scene(background=(0, 0, 0), ambient=0.0, light_direction=(-1.0, 0.0, -1.0))
    scene.camera = Camera(position=(0, 1, 2), target=(0, 1, 0), fov=40)
    scene.add(_wall(tilt=(TILT, 0.0)), name="wall")
    scene.update(0.0, 0.0)
    renderer = Renderer(64, 64, samples=0)
    try:
        renderer.quality, renderer.window_quality = "low", "high"
        view = renderer.render(scene)[32, 32].astype(int)
        window = renderer.render_window(scene, (-0.25, -0.25, 0.25, 0.25), (64, 64))[32, 32].astype(int)
        assert window[1] > view[1] + 15  # the window shows the map, the cheap view does not
    finally:
        renderer.release()


def test_ambient_light_comes_more_from_above_with_the_hemisphere(gl):
    from bifold_demo.scene.renderer import Renderer

    def floor(hemisphere, quality):
        scene = Scene(background=(0, 0, 0), ambient=0.5, light_direction=(0.0, 1.0, 0.0),  # the light from below
                      hemisphere=hemisphere)
        scene.camera = Camera(position=(0, 2, 2), target=(0, 0, 0), fov=40)
        scene.add(Plane(size=(4, 4), texture="#ffffff"), name="floor")
        scene.update(0.0, 0.0)
        renderer = Renderer(64, 64, samples=0)
        try:
            renderer.quality = quality
            return int(renderer.render(scene)[32, 32, 1])
        finally:
            renderer.release()

    assert floor(0.0, "high") == pytest.approx(128, abs=2)  # ambient 0.5 of white, as it looks
    assert floor(0.5, "high") > floor(0.0, "high") + 20  # a floor faces up: half as much again
    assert floor(0.5, "low") == floor(0.0, "high")  # the low quality keeps the ambient light the same everywhere


# -- detail levels: the 2048 textures only where a picture needs them ------------------------------------------------


def _headed(colour, detail=None, override=None):
    """A one-bone character with a "Head" bone whose wall (``_wall``) has ``colour`` (RGB), a ``Detail`` with
    ``detail`` (RGB) when given, and wears ``override`` (a Texture) instead when given."""
    from bifold_demo.characters.model import Detail

    person = _wall(maps=False)
    model = person.skinned_model
    model.skeleton = Skeleton(["Head"], [-1], np.eye(4)[None])
    part = model.primitives[0]
    part.texture = np.full((8, 8, 4), (*colour, 255), np.uint8)
    if detail is not None:
        part.detail = Detail(np.full((16, 16, 4), (*detail, 255), np.uint8))
    person = Character(model, shadow=False)
    if override is not None:
        person.texture_overrides[part.name] = override
    return person


def _scene(person, distance):
    scene = Scene(background=(0, 0, 0), ambient=1.0)
    scene.camera = Camera(position=(0, 1, distance), target=(0, 1, 0), fov=40)
    scene.add(person, name="p")
    scene.update(0.0, 0.0)
    return scene


def test_heads_get_their_detail_textures_only_where_their_eyes_are_far_enough_apart(gl):
    from bifold_demo.scene.renderer import Renderer

    person = _headed((200, 0, 0), detail=(0, 200, 0))
    renderer = Renderer(64, 64, samples=0)
    renderer.detail_from = 1.0  # (a face in a 64-pixel picture is small)
    try:
        far, near = _scene(person, 40.0), _scene(person, 2.0)  # 0.14 and 2.5 pixels between the eyes
        assert renderer.render(far)[32, 32].tolist() == [0, 0, 200]  # BGR: the 1024 texture
        assert renderer.memory()["detail_heads"] == 0
        assert renderer.render(near)[32, 32].tolist() == [0, 200, 0]  # the detail texture
        memory = renderer.memory()
        assert memory["detail_heads"] == 1 and memory["detail_loads"] == 1 and 0 < memory["detail"] < memory["textures"]
        renderer.detail_from = float("inf")  # never
        assert renderer.render(near)[32, 32].tolist() == [0, 0, 200]
        renderer.detail_from, renderer.detail_heads = 1.0, 0  # keep none: freed after the frame
        assert renderer.render(near)[32, 32].tolist() == [0, 200, 0]
        assert renderer.memory()["detail_heads"] == 0 and renderer.memory()["detail"] == 0
        renderer.detail_heads = 6
        window = renderer.render_window(far, (-0.05, -0.05, 0.05, 0.05), (64, 64))  # 20x the picture: 2.8 pixels
        assert window[32, 32].tolist() == [0, 200, 0]
    finally:
        renderer.release()


def test_detail_textures_can_be_read_in_the_background(gl):
    import time

    from bifold_demo.scene.renderer import Renderer

    person = _headed((200, 0, 0), detail=(0, 200, 0))
    renderer = Renderer(64, 64, samples=0)
    renderer.detail_wait, renderer.detail_from = False, 1.0
    try:
        near = _scene(person, 2.0)
        shown = [renderer.render(near)[32, 32].tolist()]
        deadline = time.monotonic() + 10
        while shown[-1] != [0, 200, 0] and time.monotonic() < deadline:
            time.sleep(0.02)
            shown.append(renderer.render(near)[32, 32].tolist())
        assert shown[0] in ([0, 0, 200], [0, 200, 0]) and shown[-1] == [0, 200, 0]  # the 1024 until it is there
    finally:
        renderer.release()


def test_a_large_texture_a_person_wears_is_drawn_from_its_1024_level_unless_near(gl):
    from bifold_demo.scene import Texture
    from bifold_demo.scene.detail import base_copy
    from bifold_demo.scene.renderer import Renderer

    rgb = np.zeros((2048, 2048, 3), np.uint8)
    rgb[:, :, 1] = 180
    override = Texture.from_array(rgb, order="rgb")
    person = _headed((200, 0, 0), override=override)
    renderer = Renderer(64, 64, samples=0)
    renderer.detail_from = 1.0
    try:
        assert renderer.render(_scene(person, 40.0))[32, 32].tolist() == [0, 180, 0]
        base = base_copy(override)
        assert (base.width, base.height) == (1024, 1024) and base_copy(override) is base  # made once
        assert base.id in renderer._textures and override.id not in renderer._textures
        assert renderer.render(_scene(person, 2.0))[32, 32].tolist() == [0, 180, 0]
        assert override.id in renderer._textures  # near: the whole texture
    finally:
        renderer.release()


def test_the_demo_source_sets_the_quality_and_reads_detail_in_the_background_when_live(gl):
    from bifold_demo.sources.clock import FixedStepClock
    from bifold_demo.sources.scene_source import SceneSource

    live = SceneSource(Scene(), 32, 18)
    live.quality = "low"
    recorded = SceneSource(Scene(), 32, 18, clock=FixedStepClock(30.0))
    try:
        live.start()
        recorded.start()
        assert live.renderer.quality == "low" and live.renderer.window_quality == "high"
        assert not live.renderer.detail_wait and recorded.renderer.detail_wait  # a recording waits for them
    finally:
        live.stop()
        recorded.stop()
