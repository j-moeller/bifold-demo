"""Close-ups in pictures (step S4): windows of the close quality draw subdivided heads where faces are large, and
skin, eye and lash shading; the view and the other qualities draw as before. Synthetic characters (no avatars)."""

import math
import time

import numpy as np
import pytest

from bifold_demo.characters import subdivide as sd
from bifold_demo.characters.character import Character
from bifold_demo.characters.model import CharacterModel, CharacterState, Skeleton, SkinnedPrimitive
from bifold_demo.characters.tangents import vertex_tangents
from bifold_demo.scene import Camera, Scene

pytestmark = pytest.mark.gl


@pytest.fixture(scope="module")
def avatar():
    from test_headshape import synthetic_avatar

    return synthetic_avatar()


@pytest.fixture
def renderer(gl, tmp_path, monkeypatch):
    from bifold_demo.scene.renderer import Renderer

    monkeypatch.setenv("BIFOLD_HEADTEX_CACHE", str(tmp_path))
    made = Renderer(320, 180, samples=4)
    yield made
    made.release()


def _scene(person, distance=3.0):
    scene = Scene(background=(40, 40, 40), ambient=0.5, light_direction=(-0.3, -0.5, -0.8))
    scene.camera = Camera(position=(0.0, 1.6, distance), target=(0.0, 1.6, 0.0), fov=40.0)
    scene.add(person, name="p")
    scene.update(0.0, 0.0)
    return scene


def _window(renderer, scene, eye_px):
    """A window round the head with its eyes ``eye_px`` pixels apart (a frontal face's, 63 mm)."""
    from bifold_demo.scene.detail import EYE_DISTANCE

    camera = scene.camera
    view_proj = camera.projection_matrix(renderer.aspect) @ camera.view_matrix()
    head = np.array([0.0, 1.6, 0.0, 1.0])
    p = view_proj @ head
    q = view_proj @ (head + np.array([EYE_DISTANCE, 0.0, 0.0, 0.0]))
    span = (q[0] / q[3] - p[0] / p[3]) * renderer.width / 2  # eye pixels in the view
    half = 0.25  # of the view's NDC: the window is 0.5 x 0.5 of it
    size = round(eye_px / span * half * renderer.width)
    x, y = p[:2] / p[3]
    return renderer.render_window(scene, (x - half, y - half * renderer.aspect, x + half, y + half * renderer.aspect),
                                  (size, size))


def _drawn_level(renderer, person):
    finer = renderer._window_finer.get(id(person))
    return 0 if finer is None else finer[1].metadata["subdivided"]["level"]


def test_windows_of_the_close_quality_draw_a_finer_head_where_the_face_is_large(renderer, avatar):
    person = Character(avatar, shadow=False)
    scene = _scene(person)
    renderer.render(scene)
    assert renderer.memory()["subdivided"] == 0  # the view: as it was
    renderer.subdivide_from = (60.0, 140.0)
    _window(renderer, scene, 40)
    assert _drawn_level(renderer, person) == 0 and renderer.memory()["subdivided"] == 0
    _window(renderer, scene, 100)
    assert _drawn_level(renderer, person) == 1
    x4 = renderer.memory()["subdivided"]
    _window(renderer, scene, 200)
    assert _drawn_level(renderer, person) == 2
    assert renderer.memory()["subdivided"] > 3 * x4 > 0  # (16 against 4 times the triangles)
    finer = person.detailed(2)
    assert len(finer.primitives[1].indices) == 16 * len(avatar.primitives[1].indices)
    renderer.window_quality = "high"
    _window(renderer, scene, 200)
    assert _drawn_level(renderer, person) == 0  # only the close quality
    renderer.window_quality, renderer.subdivide_from = "close", ()
    _window(renderer, scene, 200)
    assert _drawn_level(renderer, person) == 0


def test_small_faces_in_close_pictures_are_drawn_as_at_the_high_quality(renderer, avatar):
    """Below ``close_from`` and ``subdivide_from`` a close picture draws a face exactly as the high quality does
    (bit for bit: the faces a calibration measures at 8-44 pixels between the eyes do not change); at and above, it
    does not."""
    person = Character(avatar, shadow=False)
    scene = _scene(person)
    renderer.render(scene)
    for eye_px, same in ((40, True), (100, False)):
        renderer.window_quality = "high"
        high = _window(renderer, scene, eye_px)
        renderer.window_quality = "close"
        close = _window(renderer, scene, eye_px)
        assert np.array_equal(high, close) == same, eye_px


def test_a_window_draws_the_finer_head_where_the_coarse_one_was(renderer, avatar):
    """The finer head covers the coarse head's pixels but at its outline (where the curved surface bulges a little)."""
    person = Character(avatar, shadow=False)
    scene = _scene(person)
    renderer.render(scene)
    renderer.subdivide_from = (float("inf"),)
    coarse = _window(renderer, scene, 200).astype(int)
    renderer.subdivide_from = (1.0, 2.0)
    fine = _window(renderer, scene, 200).astype(int)
    assert _drawn_level(renderer, person) == 2
    changed = np.abs(fine - coarse).max(axis=2) > 24
    assert 0 < changed.mean() < 0.03
    mask_fine = renderer.window_mask(person)  # (the mask of the window drawn last: the finer head's)
    assert mask_fine.shape == fine.shape[:2] and mask_fine.max() == 1.0


def test_the_finer_head_follows_the_model_the_person_wears(renderer, avatar):
    from bifold_demo.characters.headshape import rig

    person = Character(avatar, shadow=False)
    first = person.detailed(1)
    assert first is person.detailed(1) and first.skeleton is avatar.skeleton
    head = avatar.primitives[1]
    offsets = np.zeros_like(head.positions)
    offsets[:, 2] = 0.002 * (head.positions[:, 1] > 1.6)  # the upper head forward by 2 mm: a shape of its own
    shape = rig.Shape(avatar.name, rig.base_key(avatar), head.name, offsets)
    person.use_model(rig.apply_shape(avatar, shape))
    second = person.detailed(1)
    assert second is not first and second.skeleton is person.skinned_model.skeleton
    n = len(head.positions)
    assert np.allclose(second.primitives[1].positions[:n], person.skinned_model.primitives[1].positions)


def test_the_finer_head_is_made_in_the_background_when_asked(renderer, avatar, monkeypatch):
    from bifold_demo.characters.model import CharacterModel as Model

    copy = Model(avatar.name, avatar.skeleton, list(avatar.primitives), avatar.inverse_bind, dict(avatar.metadata))
    person = Character(copy, shadow=False)
    scene = _scene(person)
    renderer.render(scene)
    renderer.detail_wait, renderer.subdivide_from = False, (1.0,)
    real = sd.subdivide
    monkeypatch.setattr(sd, "subdivide", lambda *a, **k: (time.sleep(0.3), real(*a, **k))[1])
    _window(renderer, scene, 120)
    levels = [_drawn_level(renderer, person)]
    deadline = time.monotonic() + 30
    while levels[-1] == 0 and time.monotonic() < deadline:
        time.sleep(0.05)
        _window(renderer, scene, 120)
        levels.append(_drawn_level(renderer, person))
    assert levels[0] == 0 and levels[-1] == 1  # the coarse head until the finer one is there
    no_head = CharacterModel("plain", Skeleton(["Root"], [-1], np.eye(4)[None]), [_sphere_part("plain_body")],
                             np.eye(4)[None])
    assert sd.detail_model(no_head, 1) is None


# -- shading -----------------------------------------------------------------------------------------------------------


def _sphere_part(name, radius=0.05, rows=12, cols=18, centre=(0.0, 1.6, 0.0), joint=0, maps=True):
    v, u = np.meshgrid(np.linspace(0.0, 1.0, rows + 1), np.linspace(0.0, 1.0, cols + 1), indexing="ij")
    theta, phi = v * math.pi, (u - 0.5) * 2 * math.pi
    d = np.stack([np.sin(theta) * np.sin(phi), np.cos(theta), np.sin(theta) * np.cos(phi)], axis=-1).reshape(-1, 3)
    tri = []
    for r in range(rows):
        for c in range(cols):
            a, b = r * (cols + 1) + c, r * (cols + 1) + c + 1
            tri += [(a, a + cols + 1, b), (b, a + cols + 1, b + cols + 1)]
    tri = np.array(tri)
    tri = tri[np.linalg.norm(np.cross(d[tri[:, 1]] - d[tri[:, 0]], d[tri[:, 2]] - d[tri[:, 0]]), axis=1) > 1e-9]
    p = (d * radius + centre).astype(np.float32)
    uv = np.c_[u.ravel(), 1.0 - v.ravel()].astype(np.float32)
    colour = np.full((8, 8, 4), 180, np.uint8)
    colour[..., 3] = 255
    weights = np.tile([1.0, 0, 0, 0], (len(p), 1)).astype(np.float32)
    part = SkinnedPrimitive(name, p, d.astype(np.float32), uv, tri.astype(np.uint32),
                            np.full((len(p), 4), joint, np.int32), weights, colour)
    if maps:
        flat = np.zeros((4, 4, 4), np.uint8)
        flat[..., :2], flat[..., 3] = 128, 255
        part.tangents, part.maps = vertex_tangents(p, d, uv, tri), flat
    return part


def _ball(name="b_head", bones=("Bip01 Head",), joint=0):
    skeleton = Skeleton(list(bones), [-1] + [0] * (len(bones) - 1), np.repeat(np.eye(4)[None], len(bones), 0))
    model = CharacterModel("ball", skeleton, [_sphere_part(name, joint=joint)], np.repeat(np.eye(4)[None],
                                                                                       len(bones), 0))
    return Character(model, lambda t: CharacterState((0.0, 0.0, 0.0), 0.0, []), shadow=False)


def _shot(renderer, person, quality, light, size=64, distance=0.4):
    scene = Scene(background=(0, 0, 0), ambient=0.1, light_direction=light)
    scene.camera = Camera(position=(0.0, 1.6, distance), target=(0.0, 1.6, 0.0), fov=20.0)
    scene.add(person, name="p")
    scene.update(0.0, 0.0)
    renderer.quality = quality
    return renderer.render(scene).astype(int)


def test_skin_is_lit_past_where_the_light_grazes_it_up_close(gl):
    """The close quality wraps the key light round a head's skin, red furthest: where the light grazes the ball's
    edge the close picture is lighter and redder; where it falls straight on, the same."""
    from bifold_demo.scene.renderer import Renderer

    renderer = Renderer(64, 64, samples=0)
    renderer.close_from = 0.0  # (the ball's "eyes" are 29 pixels apart here)
    try:
        person = _ball()
        light = (-1.0, 0.0, 0.0)  # from +x: the ball's +x half lit, its middle (facing the camera) grazed
        high, close = _shot(renderer, person, "high", light), _shot(renderer, person, "close", light)
        middle = (32, 31)  # the ball's middle column: its normal at right angles to the light
        assert close[middle][2] > high[middle][2] + 8  # (BGR: red)
        assert close[middle][2] - high[middle][2] > close[middle][0] - high[middle][0]  # red wraps furthest
        lit = (32, 52)
        assert abs(close[lit] - high[lit]).max() <= 6
        renderer.skin_wrap = 0.0
        assert np.abs(_shot(renderer, person, "close", light) - high).max() <= 1  # (no wrap: as the high quality)
        renderer.skin_wrap, renderer.close_from = 1.0, 64.0
        assert np.array_equal(_shot(renderer, person, "close", light), high)  # (a face too small to be a close-up)
    finally:
        renderer.release()


def test_an_eyeball_gets_a_corneas_highlight_up_close(gl):
    """A ball moved by an eye bone (an eyeball) lit from the camera's side shows a small bright highlight in the
    middle at the close quality; the same ball on the head bone does not."""
    from bifold_demo.scene.renderer import Renderer

    renderer = Renderer(256, 256, samples=0)  # (the highlight is a few pixels across here)
    renderer.close_from = 0.0
    try:
        light = (0.0, 0.0, -1.0)  # from the camera
        eye = _ball("b_head", ("Bip01 Head", "Bip01 REye"), joint=1)
        high, close = _shot(renderer, eye, "high", light), _shot(renderer, eye, "close", light)
        middle = (slice(120, 137), slice(120, 137))
        assert close[middle].min(axis=2).max() > 250 > high[middle].max() + 40  # white, over the iris
        assert (close[middle].min(axis=2) > 250).sum() < 40  # small
        assert np.abs(close[128, 64] - high[128, 64]).max() <= 3  # (away from it: as before)
        skin = _ball("b_head", ("Bip01 Head", "Bip01 REye"), joint=0)
        renderer.skin_wrap = 0.0
        assert np.abs(_shot(renderer, skin, "close", light)[middle] - _shot(renderer, skin, "high", light)[middle]
                      ).max() <= 1
    finally:
        renderer.release()


def test_lashes_and_hair_cards_are_antialiased_up_close(gl):
    """A hair card (a mesh part named like one) with a cut-out: at the high quality its edge is a hard step, at the
    close quality (alpha to coverage, with the picture's MSAA) its edge pixels are part covered."""
    from bifold_demo.scene.renderer import Renderer

    positions = np.array([[-0.05, 1.55, 0], [0.05, 1.55, 0], [0.05, 1.65, 0], [-0.05, 1.65, 0]], np.float32)
    texture = np.zeros((64, 64, 4), np.uint8)
    texture[..., :3] = 255
    yy, xx = np.mgrid[:64, :64]
    texture[..., 3] = np.where((xx - 32) ** 2 + (yy - 32) ** 2 < 24 ** 2, 255, 0)  # a disc
    card = SkinnedPrimitive("c_opacity", positions, np.tile([0, 0, 1], (4, 1)).astype(np.float32),
                            np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32),
                            np.array([[0, 1, 2], [0, 2, 3]], np.uint32), np.zeros((4, 4), np.int32),
                            np.tile([1.0, 0, 0, 0], (4, 1)).astype(np.float32), texture)
    model = CharacterModel("card", Skeleton(["Root"], [-1], np.eye(4)[None]), [card], np.eye(4)[None])
    person = Character(model, shadow=False)
    renderer = Renderer(64, 64, samples=4)
    renderer.close_from = 0.0
    try:
        edges = {}
        for quality in ("high", "close"):
            image = _shot(renderer, person, quality, (0.0, 0.0, -1.0), distance=0.3)
            grey = image.max(axis=2)
            edges[quality] = int(((grey > 10) & (grey < grey.max() - 10)).sum())
        assert edges["close"] > 3 * max(edges["high"], 1)
        renderer.coverage = False
        image = _shot(renderer, person, "close", (0.0, 0.0, -1.0), distance=0.3).max(axis=2)
        assert int(((image > 10) & (image < image.max() - 10)).sum()) == edges["high"]  # (cut out as at high)
    finally:
        renderer.release()
