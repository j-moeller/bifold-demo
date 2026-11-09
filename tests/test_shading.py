"""The renderer's shading of mesh parts with normal and specular maps, the hemisphere light and the quality setting."""

import numpy as np
import pytest

from bifold_demo.characters.character import Character
from bifold_demo.characters.model import CharacterModel, Skeleton, SkinnedPrimitive
from bifold_demo.characters.tangents import vertex_tangents
from bifold_demo.scene import Camera, Plane, Scene, Texture

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


def _floor_head(heading=0.0, head_yaw=0.0):
    """A one-bone ("Head") character whose part is a 1 m square facing up round the bone (u along +x), with a
    checker of normals tilted 30 degrees towards +u and +v by turns, as a head with a "Head" bone turns it."""
    from bifold_demo.characters.model import CharacterState

    positions = np.array([[-0.5, 0, 0.5], [0.5, 0, 0.5], [0.5, 0, -0.5], [-0.5, 0, -0.5]], np.float32)
    normals = np.tile(np.array([0, 1, 0], np.float32), (4, 1))
    uvs = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
    indices = np.array([[0, 1, 2], [0, 2, 3]], np.uint32)
    packed = np.full((8, 8, 4), 255, np.uint8)
    tilt = np.round((np.sin(TILT) + 1) * 127.5)
    cells = (np.add.outer(np.arange(8), np.arange(8)) // 2) % 2 == 0
    packed[..., 0] = np.where(cells, tilt, 128)
    packed[..., 1] = np.where(cells, 128, tilt)
    packed[..., 2] = 0
    colour = np.full((8, 8, 4), 200, np.uint8)
    primitive = SkinnedPrimitive("floor_head", positions, normals, uvs, indices, np.zeros((4, 4), np.int32),
                                 np.tile(np.array([1, 0, 0, 0], np.float32), (4, 1)), colour)
    primitive.tangents = vertex_tangents(positions, normals, uvs, indices)
    primitive.maps = packed
    skeleton = Skeleton(["Head"], [-1], np.eye(4)[None])
    person = Character(CharacterModel("floor", skeleton, [primitive], np.eye(4)[None]), shadow=False)
    person.behaviour = lambda _t: CharacterState((0.0, 0.0, 0.0), heading, [], head_yaw=head_yaw)
    return person


def _turned_with(person, turn):
    """The picture of ``person`` from above under a light from the side, the camera and the light turned by ``turn``
    (4x4) about the origin: what a turned head looks like to a camera and a light that turn with it."""
    from bifold_demo.scene.renderer import Renderer

    light, up = turn[:3, :3] @ np.array([-1.0, -0.6, -0.3]), turn[:3, :3] @ np.array([0.0, 0.0, -1.0])
    scene = Scene(background=(0, 0, 0), ambient=0.0, light_direction=tuple(light))
    scene.add(person, name="floor")
    scene.update(0.0, 0.0)
    scene.camera = Camera(position=(0, 1.2, 0), target=(0, 0, 0), fov=40, up=tuple(up))
    renderer = Renderer(64, 64, samples=0)
    try:
        return renderer.render(scene)[20:44, 20:44].astype(int)  # (the middle: the square's turned edges left out)
    finally:
        renderer.release()


def test_the_tangent_frame_turns_with_the_head_bone_and_the_person(gl):
    """A normal-mapped part turned by its bone (the head turning) or with the whole person, and the camera and the
    light turned with it, looks the same: the tangents are skinned and turned like the normals (else the map's
    slopes would point the old way and the light would fall differently)."""
    straight = _turned_with(_floor_head(), np.eye(4))
    assert np.ptp(straight) > 20  # (the checker of slopes shows)
    bone = _floor_head(head_yaw=40.0)
    bone.update(0.0)
    assert np.abs(_turned_with(bone, bone.pose[0]) - straight).mean() < 1.0
    person = _floor_head(heading=90.0)
    person.update(0.0)
    assert np.abs(_turned_with(person, person.matrices()[0]) - straight).mean() < 1.0


def test_maps_read_from_files_are_data_as_arrays_are(gl, tmp_path):
    """A packed map in a PNG file, as the prepared avatars have them (decoded on the preloading thread with its
    mipmaps), draws as the same map given as an array: near, and far away where only its mipmaps show (a checker of
    opposite slopes averages to a flat normal; averaged as colours it would tilt)."""
    import cv2

    from bifold_demo.scene.renderer import Renderer

    packed = np.zeros((64, 64, 4), np.uint8)
    packed[..., 0] = np.where((np.add.outer(np.arange(64), np.arange(64)) % 2) == 0, 255, 0)
    packed[..., 1], packed[..., 2], packed[..., 3] = 128, 40, 255
    path = tmp_path / "wall_maps.png"
    cv2.imwrite(str(path), cv2.cvtColor(packed, cv2.COLOR_RGBA2BGRA))
    from_array = _wall()
    from_array.skinned_model.primitives[0].maps = packed
    model = _wall().skinned_model
    model.primitives[0].maps = str(path)
    from_file = Character(model, shadow=False)  # (decodes the file on the preloading thread)
    for job in from_file.texture_jobs:
        job.result()

    def pixel(person, distance):
        scene = Scene(background=(0, 0, 0), ambient=0.1, light_direction=(-1.0, 0.0, -0.4))
        scene.camera = Camera(position=(0, 1, distance), target=(0, 1, 0), fov=40)
        scene.add(person, name="wall")
        scene.update(0.0, 0.0)
        renderer = Renderer(64, 64, samples=0)
        try:
            return renderer.render(scene)[32, 32].astype(int)
        finally:
            renderer.release()

    for distance in (2.0, 25.0):  # the wall fills the picture, then covers about four pixels
        assert np.abs(pixel(from_file, distance) - pixel(from_array, distance)).max() <= 2, distance
    flat = _wall()
    assert np.abs(pixel(from_array, 25.0) - pixel(flat, 25.0)).max() <= 3  # (the slopes average out far away)


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


# -- faces from photos: the person's own relief, shine and light -----------------------------------------------------


def _mapped_head():
    """The synthetic head of test_headtex (a sphere with a nose and a Rocketbox avatar's face bones), skin-coloured,
    with tangents and a packed map that tilts every normal 30 degrees towards +u and gives it a shine."""
    from test_headtex import synthetic_head_model

    model = synthetic_head_model()
    head = model.primitives[0]
    head.texture = np.full((64, 64, 4), (200, 160, 140, 255), np.uint8)
    head.tangents = vertex_tangents(head.positions, head.normals, head.uvs, head.indices)
    packed = np.full((64, 64, 4), 255, np.uint8)
    packed[..., 0], packed[..., 1], packed[..., 2] = round((np.sin(TILT) + 1) * 127.5), 128, 60
    head.maps = packed
    return model


def _face_pixel(person, quality="high"):
    """The middle of ``person``'s face (the synthetic head's) from the front, in light from above to the side and
    an ambient light that comes from above (hemisphere 0.5)."""
    from test_headtex import CENTRE

    from bifold_demo.characters.headtex.camera import PhotoCamera

    scene = Scene(background=(0, 0, 0), ambient=0.4, light_direction=(-0.6, -0.5, -0.6), hemisphere=0.5)
    scene.add(person, name="p")
    scene.update(0.0, 0.0)
    scene.camera = PhotoCamera.around(CENTRE, 0.5, 0.0, fov=30.0, size=(64, 64)).scene_camera()
    return _render_scene(scene, quality)[30:34, 30:34].astype(float).mean(axis=(0, 1))


def _render_scene(scene, quality):
    from bifold_demo.scene.renderer import Renderer

    renderer = Renderer(64, 64, samples=0)
    try:
        renderer.quality = quality
        return renderer.render(scene)
    finally:
        renderer.release()


def test_a_face_from_photos_is_drawn_without_the_avatars_maps_and_hemisphere_light(gl):
    """A head wearing a texture of its own (a face from photos) is drawn, where the face is, as at the low quality:
    the avatar's normal and specular maps would put the avatar's folds and shine on the person's face, and the
    hemisphere light would add soft light from above to a face that has it already. Given maps of its own
    (``maps_overrides``), it is drawn with them and the scene's light."""
    from bifold_demo.scene.textures import as_data_texture

    model = _mapped_head()
    own = _face_pixel(Character(model, shadow=False))
    low = _face_pixel(Character(model, shadow=False), quality="low")
    assert np.abs(own - low).max() > 10  # (the maps and the hemisphere light show on the avatar's own face)
    photo = Character(model, shadow=False)
    name = model.primitives[0].name
    photo.texture_overrides[name] = Texture(model.primitives[0].texture.copy())  # (the same colours)
    assert photo.photo_mask(model.primitives[0]) is not None
    assert np.abs(_face_pixel(photo) - low).max() <= 1.5
    flat = as_data_texture(np.full((8, 8, 4), (128, 128, 0, 255), np.uint8))
    photo.maps_overrides[name] = flat
    assert photo.photo_mask(model.primitives[0]) is None
    plain = Character(_mapped_head(), shadow=False)
    plain.skinned_model.primitives[0].maps = flat.rgba
    assert np.abs(_face_pixel(photo) - _face_pixel(plain)).max() <= 1.5  # its own maps, in the scene's light
    older = _mapped_head()  # a head of a cache without maps: the hemisphere light is left off the whole head
    older.primitives[0].maps = older.primitives[0].tangents = None
    photo = Character(older, shadow=False)
    photo.texture_overrides[name] = Texture(older.primitives[0].texture.copy())
    assert np.abs(_face_pixel(photo) - low).max() <= 1.5


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
        renderer.detail_from = 2.0  # (the eyes are 2.5 pixels apart: still near)
        assert renderer.render(near)[32, 32].tolist() == [0, 200, 0]
        renderer.detail_from = 1.0
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


def test_a_near_head_is_drawn_with_its_detail_maps(gl):
    """Near, a head is drawn with its detail maps too (here tilted the other way than its 1024 maps, under a light
    from the side the 1024 maps' slopes face)."""
    from bifold_demo.characters.model import Detail
    from bifold_demo.scene.renderer import Renderer

    model = _wall(tilt=(TILT, 0.0)).skinned_model
    model.skeleton = Skeleton(["Head"], [-1], np.eye(4)[None])
    part = model.primitives[0]
    part.positions = part.positions - np.array([0, 1, 0], np.float32)  # the wall round the head bone
    part.detail = Detail(part.texture, _wall(tilt=(-TILT, 0.0)).skinned_model.primitives[0].maps)
    scene = Scene(background=(0, 0, 0), ambient=0.0, light_direction=(-1.0, 0.0, -1.0))
    scene.camera = Camera(position=(0, 0, 2), target=(0, 0, 0), fov=40)
    scene.add(Character(model, shadow=False), name="p")
    scene.update(0.0, 0.0)
    renderer = Renderer(64, 64, samples=0)
    try:
        renderer.detail_from = float("inf")
        with_1024 = int(renderer.render(scene)[32, 32, 1])
        renderer.detail_from = 1.0
        with_detail = int(renderer.render(scene)[32, 32, 1])
    finally:
        renderer.release()
    assert with_detail < with_1024 - 15


def _crowd(xs, detail=(0, 200, 0)):
    """People at ``xs`` = [(x, z)] (metres) in front of a camera at (0, 0, 4) looking at the origin, each a wall
    centred on its "Head" bone with a red 1024 texture and a detail texture of ``detail``."""
    from bifold_demo.characters.model import CharacterState

    scene = Scene(background=(0, 0, 0), ambient=1.0)
    scene.camera = Camera(position=(0, 0, 4), target=(0, 0, 0), fov=40)
    people = []
    for k, (x, z) in enumerate(xs):
        person = _headed((200, 0, 0), detail=detail)
        part = person.skinned_model.primitives[0]
        part.positions = part.positions - np.array([0, 1, 0], np.float32)  # the wall round the head bone
        person.behaviour = (lambda _t, p=(x, 0.0, z): CharacterState(p, 0.0, []))
        scene.add(person, name=f"p{k}")
        people.append(person)
    scene.update(0.0, 0.0)
    return scene, people


def _window_on(renderer, scene, person, half=0.15):
    """A 64x64 window round ``person``'s head bone (the middle pixel's BGR)."""
    p = scene.camera.projection_matrix(1.0) @ scene.camera.view_matrix() @ np.append(person.current_position, 1.0)
    x, y = p[:2] / p[3]
    return renderer.render_window(scene, (x - half, y - half, x + half, y + half), (64, 64))[32, 32].tolist()


def test_more_heads_near_than_are_kept_keep_their_places(gl):
    """Three people near the camera, room for two heads' detail textures, a window on each head every frame (the
    sensor's windows, round robin): the two read first keep theirs and are read once, the third is drawn at 1024
    without being read; a head that comes much nearer takes the place of the weakest one no picture of the frame drew
    yet, whose textures leave the GPU at once."""
    from bifold_demo.characters.model import CharacterState
    from bifold_demo.scene.renderer import Renderer

    scene, people = _crowd([(-1.2, -0.5), (0.0, 0.0), (1.2, 0.0)])  # the first a little further away
    renderer = Renderer(64, 64, samples=0)
    renderer.detail_heads, renderer.detail_from = 2, 3.0  # (the view's faces are 1.4 pixels: the windows' 9)
    green, red = [0, 200, 0], [0, 0, 200]
    try:
        for _frame in range(5):
            renderer.render(scene)
            assert [_window_on(renderer, scene, p) for p in people] == [green, green, red]
        memory = renderer.memory()
        assert memory["detail_loads"] == 2 and memory["detail_heads"] == 2 and memory["detail_refused"] == 5
        assert memory["detail_cpu"] == 0  # the pixels given up once on the GPU
        textures = memory["textures"]
        # the third comes nearer (twice the eye distance) and is drawn first: it takes the first one's place
        people[2].behaviour = (lambda _t: CharacterState((1.0, 0.0, 2.0), 0.0, []))
        scene.update(0.0, 0.0)
        for _frame in range(3):
            renderer.render(scene)
            assert [_window_on(renderer, scene, p) for p in people[::-1]] == [green, green, red]
        memory = renderer.memory()
        assert memory["detail_loads"] == 3 and memory["detail_heads"] == 2
        assert memory["textures"] == textures  # the first's detail textures left the GPU
    finally:
        renderer.release()


def test_a_head_read_in_the_background_and_not_drawn_since_gives_its_place_up(gl):
    """Read in the background, a head's detail textures finish while no picture asks for them any more (the person
    left the windows): another head can still take their place once the first one's claim has faded."""
    import time

    from bifold_demo.scene.renderer import Renderer

    scene, people = _crowd([(-1.2, 0.0), (1.2, 0.0)])
    renderer = Renderer(64, 64, samples=0)
    renderer.detail_heads, renderer.detail_from, renderer.detail_wait = 1, 3.0, False
    try:
        renderer.render(scene)
        _window_on(renderer, scene, people[0])  # asks for the first head's: read in the background
        deadline = time.monotonic() + 10
        while renderer.memory()["detail_loads"] < 1 and time.monotonic() < deadline:
            time.sleep(0.02)
        shown = []
        for _frame in range(12):  # only the second person from now on
            renderer.render(scene)
            shown.append(_window_on(renderer, scene, people[1]))
            time.sleep(0.05)
        assert renderer.memory()["detail_loads"] == 2 and shown[-1] == [0, 200, 0]
    finally:
        renderer.release()


def test_a_head_outside_the_picture_is_not_read(gl):
    from bifold_demo.scene.detail import eye_pixels
    from bifold_demo.scene.renderer import Renderer

    view_proj = np.eye(4)  # (normalised device coordinates are the scene's)
    assert eye_pixels((0.0, 0.0, 0.5), view_proj, (100, 100), (1.0, 0.0, 0.0)) == pytest.approx(3.15)
    assert eye_pixels((1.05, 0.0, 0.5), view_proj, (100, 100), (1.0, 0.0, 0.0)) > 0  # its face may reach in
    assert eye_pixels((1.5, 0.0, 0.5), view_proj, (100, 100), (1.0, 0.0, 0.0)) == 0.0
    scene, _people = _crowd([(-1.5, 3.0)])  # beside the camera, outside its 40-degree picture
    renderer = Renderer(64, 64, samples=0)
    renderer.detail_from = 1.0
    try:
        renderer.render(scene)
        assert renderer.memory()["detail_loads"] == 0
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
