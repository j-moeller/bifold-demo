import gc

import numpy as np
import pytest

from bifold_demo.characters.character import Character, skin_vertices
from bifold_demo.characters.model import (
    AnimationClip,
    CharacterModel,
    CharacterState,
    Layer,
    Poser,
    Skeleton,
    SkinnedPrimitive,
    compose,
    quat_from_axis_angle,
)
from bifold_demo.scene import Camera, Plane, Scene, SceneObject, Texture
from bifold_demo.scene.geometry import Mesh, box_mesh
from bifold_demo.scene.transforms import rotation_y, scaling, translation

RED, BLUE, GREEN = (255, 0, 0), (0, 0, 255), (0, 255, 0)


def _box(lo, hi, bone, color, top_bone=None):
    """A box skinned to ``bone``; with ``top_bone`` its top vertices are shared 50/50 with that bone."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    mesh = box_mesh(*(hi - lo))
    positions = mesh.positions + ((lo + hi) / 2).astype(np.float32)
    joints = np.zeros((len(positions), 4), np.int64)
    weights = np.zeros((len(positions), 4), np.float32)
    joints[:, 0], weights[:, 0] = bone, 1.0
    if top_bone is not None:
        top = positions[:, 1] > hi[1] - 1e-4
        joints[top, 1], weights[top, :2] = top_bone, 0.5
    rgba = np.zeros((4, 4, 4), np.uint8)
    rgba[...] = (*color, 255)  # RGB order, like the FBX loader produces
    return SkinnedPrimitive(f"box{bone}", positions, mesh.normals, mesh.uvs, mesh.indices, joints, weights, rgba)


def _model(blend=False):
    """A red body (Hips), a blue head (Head, child of Neck) and a green nose sticking out towards +Z.

    With ``blend`` the top of the body is skinned half to the neck.
    """
    rest = np.repeat(np.eye(4)[None], 3, axis=0)
    rest[1, 1, 3], rest[2, 1, 3] = 1.2, 0.1
    skeleton = Skeleton(["Hips", "Neck", "Head"], [-1, 0, 1], rest)
    primitives = [
        _box((-0.2, 0.0, -0.12), (0.2, 1.2, 0.12), 0, RED, top_bone=1 if blend else None),
        _box((-0.12, 1.3, -0.12), (0.12, 1.6, 0.12), 2, BLUE),
        _box((-0.04, 1.4, 0.12), (0.04, 1.48, 0.4), 2, GREEN),
    ]
    return CharacterModel("boxy", skeleton, primitives, np.linalg.inv(skeleton.rest_global))


def _clip(bone, axis, degrees):
    return AnimationClip(f"{bone}{degrees}", 30.0, [bone], quat_from_axis_angle(axis, [[degrees]]), np.zeros((1, 1, 3)))


def _behaviour(heading=0.0, layers=(), **look):
    return lambda t: CharacterState((0.0, 0.0, 0.0), heading, list(layers), **look)


def _scene(character, floor=False, camera=None):
    scene = Scene(background=(0, 0, 0))
    scene.camera = camera or Camera(position=(0, 1.0, 4.0), target=(0, 0.9, 0), fov=45)
    if floor:
        scene.add(Plane(size=(8, 8), texture="#c8c8c8", lit=False), name="floor")
    character.lit = False
    scene.add(character, name="person")
    return scene


@pytest.fixture
def render(gl):
    from bifold_demo.scene.renderer import Renderer

    renderer = Renderer(160, 120, samples=0)

    def _render(scene):
        scene.update(0.0, 0.0)
        return renderer.render(scene)

    _render.renderer = renderer
    yield _render
    renderer.release()


def _mask(image, rgb):
    return np.all(image == np.array(rgb[::-1]), axis=-1)


def _centre(mask):
    ys, xs = np.nonzero(mask)
    assert len(xs), "colour not found in the image"
    return xs.mean(), ys.mean()


def test_rest_pose_keeps_texture_colours(render):
    image = render(_scene(Character(_model())))
    assert image[80, 80].tolist() == [0, 0, 255]  # red body, still red (RGBA texture in RGB order)
    assert _mask(image, BLUE).sum() > 50 and _mask(image, GREEN).sum() > 5
    assert image[5, 5].tolist() == [0, 0, 0]


def test_rotating_a_bone_moves_its_pixels(render):
    rest = render(_scene(Character(_model())))
    bend = [Layer(_clip("Neck", [0, 0, 1], 60), 0.0)]
    bent = render(_scene(Character(_model(), _behaviour(layers=bend))))
    (x0, y0), (x1, y1) = _centre(_mask(rest, BLUE)), _centre(_mask(bent, BLUE))
    assert x1 < x0 - 5 and y1 > y0 + 1  # +Z roll tips the head over towards -X (left in the image)
    assert np.array_equal(_mask(rest, RED)[60:], _mask(bent, RED)[60:])  # the lower body stays put


def test_gpu_skinning_matches_cpu_reference(render):
    layers = [Layer(_clip("Neck", [1, 0, 1], 80), 0.0), Layer(_clip("Head", [0, 1, 0], -70), 0.0)]
    character = Character(_model(blend=True), _behaviour(heading=25.0, layers=layers, head_pitch=15))
    camera = Camera(position=(1.0, 1.5, 2.2), target=(0, 1.1, 0), fov=45)
    gpu = render(_scene(character, camera=camera))

    reference = Scene(background=(0, 0, 0))
    reference.camera = camera
    points = skin_vertices(character.skinned_model, character.skin_palette)
    for i, (prim, p) in enumerate(zip(character.skinned_model.primitives, points)):
        mesh = Mesh(p, prim.indices, uvs=prim.uvs, normals=prim.normals)
        reference.add(SceneObject(mesh, texture=Texture.from_array(prim.texture, order="rgb"), rotation=(0, 25.0, 0),
                                  lit=False), name=f"part{i}")
    cpu = render(reference)
    different = np.any(np.abs(gpu.astype(int) - cpu.astype(int)) > 8, axis=-1)
    assert different.mean() < 0.001  # ignoring the second joint weight already differs in ~0.4 % of the pixels
    assert _mask(gpu, GREEN).sum() > 5
    assert not np.array_equal(_mask(gpu, RED), _mask(render(_scene(Character(_model(), _behaviour(heading=25.0)),
                                                                    camera=camera)), RED))  # the body bends


def test_heading_turns_the_character(render):
    front = render(_scene(Character(_model())))
    turned = render(_scene(Character(_model(), _behaviour(heading=90.0))))
    head_x = _centre(_mask(front, BLUE))[0]
    assert abs(_centre(_mask(front, GREEN))[0] - head_x) < 2  # nose points at the camera
    assert _centre(_mask(turned, GREEN))[0] > head_x + 4  # heading 90 faces +X: right in the image


def test_head_yaw_turns_only_the_head(render):
    rest = Character(_model())
    looking = Character(_model(), _behaviour(head_yaw=80.0))
    a, b = render(_scene(rest)), render(_scene(looking))
    assert _centre(_mask(b, GREEN))[0] > _centre(_mask(a, GREEN))[0] + 4  # positive yaw = to the character's left
    assert np.array_equal(_mask(a, RED)[60:], _mask(b, RED)[60:])
    np.testing.assert_allclose(looking.skin_palette[0], np.eye(4), atol=1e-6)  # body bone untouched
    assert not np.allclose(looking.skin_palette[2], np.eye(4), atol=1e-3)


def test_head_look_matches_rotate_in_character_space():
    model = _model(blend=True)
    layers = [Layer(_clip("Neck", [1, 0, 0.3], 30), 0.0), Layer(_clip("Hips", [0, 1, 0], 20), 0.0, 0.5)]
    character = Character(model, _behaviour(layers=layers, head_yaw=50.0, head_pitch=-20.0))

    def adjust(local, poser):
        for bone, share in ((1, 0.4), (2, 0.6)):
            yaw = rotation_y(50.0 * share)[:3, :3]
            pitch = compose(quat_from_axis_angle([[1.0, 0, 0]], [-20.0 * share]), np.zeros((1, 3)))[0, :3, :3]
            poser.rotate_in_character_space(local, bone, yaw @ pitch)

    expected = Poser(model).global_pose(layers, adjust=adjust)
    np.testing.assert_allclose(character.pose, expected, atol=1e-9)
    assert character.skin_palette.dtype == np.float32 and character.skin_palette.shape == (3, 4, 4)


def test_fast_poser_matches_poser():
    rng = np.random.default_rng(3)
    parents = [-1] + [int(rng.integers(0, i)) for i in range(1, 30)]
    rest = compose(quat_from_axis_angle(rng.normal(size=(30, 3)), rng.uniform(-40, 40, 30)), rng.normal(size=(30, 3)))
    rest[0, :3, 3] = (0, 0.9, 0)
    skeleton = Skeleton([f"b{i}" for i in range(30)], parents, rest)
    model = CharacterModel("random", skeleton, [_box((-0.1, 0, -0.1), (0.1, 1, 0.1), 0, RED)],
                           np.linalg.inv(skeleton.rest_global))
    clips = []
    for k in range(2):
        names = [f"b{i}" for i in rng.choice(30, 20, replace=False)] + ["b0"]
        q = quat_from_axis_angle(rng.normal(size=(5, 21, 3)), rng.uniform(-90, 90, (5, 21)))
        clips.append(AnimationClip(f"c{k}", 30.0, names, q, rng.normal(size=(5, 21, 3)) + [0, 1.8, 0], root="b0"))
    layers = [Layer(clips[0], 0.05, 0.7), Layer(clips[1], 0.11, 0.3)]
    character = Character(model, _behaviour(layers=layers))
    np.testing.assert_allclose(character.pose, Poser(model).global_pose(layers), atol=1e-9)


def test_model_matrix_follows_the_contract():
    character = Character(_model(), lambda t: CharacterState((1.0, 0.0, -2.0), 30.0, []), scale=(1.1, 0.9, 1.0))
    character.update(0.5)
    model, normal = character.matrices()
    expected = translation((1.0, 0.0, -2.0)) @ rotation_y(30.0) @ scaling((1.1, 0.9, 1.0))
    np.testing.assert_allclose(model, expected, atol=1e-12)
    np.testing.assert_allclose(normal, np.linalg.inv(expected[:3, :3]).T, atol=1e-12)
    assert character.current_rotation.tolist() == [0.0, 30.0, 0.0]
    still = Character(_model())
    assert still.current_position.tolist() == [0, 0, 0] and still.current_visible
    np.testing.assert_allclose(still.skin_palette, np.repeat(np.eye(4)[None], 3, axis=0), atol=1e-6)


def test_invisible_state_is_not_drawn(render):
    hidden = Character(_model(), lambda t: CharacterState((0, 0, 0), 0.0, [], visible=False))
    image = render(_scene(hidden))
    assert not hidden.current_visible and image.max() == 0


def test_blob_shadow_darkens_the_floor_under_the_feet(render):
    camera = Camera(position=(0, 2.5, 2.5), target=(0, 0, 0), fov=50)
    with_shadow = render(_scene(Character(_model()), floor=True, camera=camera)).astype(int)
    without = render(_scene(Character(_model(), shadow=False), floor=True, camera=camera)).astype(int)
    floor = without[100, 10]
    assert np.all(without[100:, :].reshape(-1, 3).max(axis=0) <= floor)  # uniform floor in front of the body
    darker = (without - with_shadow).sum(axis=-1)
    beside_feet = darker[60:90, 20:140].max()
    assert 20 < beside_feet < 0.7 * floor.sum()  # visible but soft
    assert np.abs(darker[:, :8]).max() == 0  # far from the character the floor is untouched
    assert np.all(darker >= 0)


def test_gpu_resources_are_shared_and_released(render):
    model = _model()
    scene = _scene(Character(model))
    scene.add(Character(model, lambda t: CharacterState((0.8, 0.0, 0.0), 0.0, [])), name="twin")
    render(scene)
    renderer = render.renderer
    assert len(renderer._primitives) == 3 and len(renderer._joints) == 2
    render(scene)
    assert len(renderer._primitives) == 3
    del scene, model
    gc.collect()
    renderer._collect_garbage()
    assert not renderer._primitives and not renderer._joints
