import numpy as np
import pytest

from bifold_demo.characters.model import (
    AnimationClip,
    CharacterModel,
    Layer,
    Poser,
    Skeleton,
    SkinnedPrimitive,
    compose,
    decompose,
    quat_from_axis_angle,
    quat_from_matrix,
    quat_multiply,
    quat_nlerp,
    quat_to_matrix,
    root_scale,
)
from bifold_demo.scene.transforms import euler_rotation


def test_quaternion_round_trips():
    rng = np.random.default_rng(0)
    q = rng.normal(size=(50, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q *= np.sign(q[:, 3:])
    np.testing.assert_allclose(quat_from_matrix(quat_to_matrix(q)), q, atol=1e-9)
    m = euler_rotation((30, 40, 50))[:3, :3]
    np.testing.assert_allclose(quat_to_matrix(quat_from_matrix(m)), m, atol=1e-12)


def test_quaternion_product_matches_matrices():
    a, b = quat_from_axis_angle([0, 1, 0], 90), quat_from_axis_angle([1, 0, 0], 30)
    np.testing.assert_allclose(quat_to_matrix(quat_multiply(a, b)), quat_to_matrix(a) @ quat_to_matrix(b), atol=1e-12)
    np.testing.assert_allclose(quat_to_matrix(a) @ [1, 0, 0], [0, 0, -1], atol=1e-12)


def test_nlerp_takes_the_short_way():
    a = quat_from_axis_angle([0, 0, 1], 10)
    b = -quat_from_axis_angle([0, 0, 1], 30)  # same rotation as +30 degrees, opposite hemisphere
    mid = quat_nlerp(a, b, 0.5)
    np.testing.assert_allclose(quat_to_matrix(mid), quat_to_matrix(quat_from_axis_angle([0, 0, 1], 20)), atol=1e-9)


def _arm_skeleton():
    """root (1 m up, like a pelvis) -> upper (+0.5 m) -> lower (+0.5 m)."""
    rest = np.repeat(np.eye(4)[None], 3, axis=0)
    rest[0, 1, 3] = 1.0
    rest[1, 1, 3] = 0.5
    rest[2, 1, 3] = 0.5
    return Skeleton(["root", "upper", "lower"], [-1, 0, 1], rest)


def test_forward_kinematics_matches_a_naive_loop():
    rng = np.random.default_rng(1)
    names = [f"b{i}" for i in range(12)]
    parents = [-1] + [int(rng.integers(0, i)) for i in range(1, 12)]
    local = compose(quat_from_axis_angle(rng.normal(size=(12, 3)), rng.uniform(-90, 90, 12)), rng.normal(size=(12, 3)))
    skeleton = Skeleton(names, parents, local)
    naive = np.empty_like(local)
    for i, p in enumerate(parents):
        naive[i] = local[i] if p < 0 else naive[p] @ local[i]
    np.testing.assert_allclose(skeleton.global_transforms(local), naive, atol=1e-12)
    assert list(skeleton.descendants(0)) == list(range(12))


def test_skeleton_rejects_unsorted_bones():
    with pytest.raises(ValueError):
        Skeleton(["a", "b"], [1, -1], np.repeat(np.eye(4)[None], 2, axis=0))


def test_find_bones_by_name():
    skeleton = Skeleton(["Bip01", "Bip01 Pelvis", "Bip01 Head"], [-1, 0, 1], np.repeat(np.eye(4)[None], 3, axis=0))
    assert skeleton.find("bip01 head") == 2
    assert skeleton.find("Neck", "Head") == 2
    assert skeleton.find("Tail") is None


def _clip(degrees, root_heights=(1.0, 1.0)):
    frames = len(degrees)
    rot = np.zeros((frames, 2, 4))
    rot[:, 0, 3] = 1
    rot[:, 1] = quat_from_axis_angle(np.tile([0, 0, 1], (frames, 1)), degrees)
    tr = np.zeros((frames, 2, 3))
    tr[:, 0, 1] = np.linspace(*root_heights, frames)
    return AnimationClip("bend", fps=10, bone_names=["root", "lower"], rotations=rot, translations=tr, root="root")


def test_clip_quaternions_are_made_continuous_on_construction():
    q = quat_from_axis_angle(np.tile([0, 0, 1.0], (6, 1)), np.linspace(0, 100, 6))[:, None]
    flipped = q * np.array([1, -1, -1, 1, -1, 1.0])[:, None, None]
    clip = AnimationClip("signs", 10, ["b"], flipped, np.zeros((6, 1, 3)))
    assert np.all(np.sum(clip.rotations[1:] * clip.rotations[:-1], axis=-1) >= 0)  # neighbours blend as they are
    np.testing.assert_allclose(quat_to_matrix(clip.rotations), quat_to_matrix(q), atol=1e-12)  # same rotations
    assert flipped[1, 0, 3] < 0  # the caller's array is left alone


def test_clip_sampling_interpolates_and_loops():
    clip = _clip([0, 90, 0])
    assert clip.duration == pytest.approx(0.2)
    q, _ = clip.sample(0.05)
    np.testing.assert_allclose(quat_to_matrix(q[1]), quat_to_matrix(quat_from_axis_angle([0, 0, 1], 45)), atol=1e-9)
    q_loop, _ = clip.sample(0.25)
    np.testing.assert_allclose(q_loop, clip.sample(0.05)[0], atol=1e-12)
    clip.loop = False
    np.testing.assert_allclose(clip.sample(5.0)[0], clip.rotations[-1], atol=1e-12)


def _model():
    skeleton = _arm_skeleton()
    prim = SkinnedPrimitive("arm", positions=np.array([[0, 2.5, 0]], np.float32), normals=np.array([[1, 0, 0]], np.float32),
                            uvs=np.zeros((1, 2), np.float32), indices=np.zeros((0, 3), np.uint32),
                            joints=np.array([[2, 0, 0, 0]]), weights=np.array([[1, 0, 0, 0]], np.float32))
    return CharacterModel("arm", skeleton, [prim], np.linalg.inv(skeleton.rest_global))


def test_poser_drives_rotations_and_keeps_the_models_bone_lengths():
    model = _model()
    poser = Poser(model)
    clip = _clip([90, 90], root_heights=(2.0, 2.0))  # the clip's skeleton stands twice as tall
    world = poser.global_pose([Layer(clip, 0.0)])
    np.testing.assert_allclose(world[0, :3, 3], [0, 1.0, 0], atol=1e-12)  # root height rescaled to the model's
    np.testing.assert_allclose(world[2, :3, 3], [0, 2.0, 0], atol=1e-12)  # bones keep the model's lengths
    tip = model.skinning_matrices(world)[2] @ [0, 2.5, 0, 1]  # the vertex 0.5 m past the lower joint
    np.testing.assert_allclose(tip[:3], [-0.5, 2.0, 0.0], atol=1e-9)  # swung 90 degrees about z


LEG = ["Bip01", "Bip01 Pelvis", "Bip01 L Thigh", "Bip01 L Calf", "Bip01 L Foot", "Bip01 L Toe0"]


def _leg_offsets(thigh=0.45, shin=0.45, ankle=0.08):
    """Bone offsets of a one-legged Biped whose toe joint touches the floor in the rest pose."""
    return np.array([(0, thigh + shin + ankle, 0), (0, 0, 0), (0.1, 0, 0), (0, -thigh, 0), (0, -shin, 0),
                     (0, -ankle, 0.12)], dtype=np.float64)


def _leg_model(**sizes):
    rest = np.repeat(np.eye(4)[None], len(LEG), axis=0)
    rest[:, :3, 3] = _leg_offsets(**sizes)
    skeleton = Skeleton(LEG, list(range(-1, len(LEG) - 1)), rest)
    return CharacterModel("leg", skeleton, [], np.linalg.inv(skeleton.rest_global))


def _bent_knee_walk(frames=21):
    """Knees bending up to 40 degrees with the foot kept flat; the root sinks so the clip's own toe stays on y = 0."""
    bend = 40.0 * np.sin(np.linspace(0, np.pi, frames)) ** 2
    rotations = np.zeros((frames, len(LEG), 4))
    rotations[..., 3] = 1
    axis = np.tile([1.0, 0, 0], (frames, 1))
    rotations[:, 2], rotations[:, 3], rotations[:, 4] = (quat_from_axis_angle(axis, d) for d in (bend, -2 * bend, bend))
    translations = np.repeat(_leg_offsets()[None], frames, axis=0)
    translations[:, 0, 1] = 0.08 + 0.9 * np.cos(np.radians(bend))
    return AnimationClip("knees", 30, LEG, rotations, translations, root="Bip01", metadata={"kind": "walk"})


def test_root_scale_comes_from_the_leg_not_the_animated_root_height():
    walk = _bent_knee_walk()
    assert walk.translations[:, 0, 1].mean() < 0.96  # the pelvis sinks while the knees bend
    assert root_scale(_leg_model().skeleton, walk) == pytest.approx(1.0, abs=1e-12)
    assert root_scale(_leg_model(thigh=0.225, shin=0.225, ankle=0.04).skeleton, walk) == pytest.approx(0.5)
    child = _leg_model(thigh=0.36, shin=0.36, ankle=0.05)  # shorter legs and a proportionally lower ankle
    assert root_scale(child.skeleton, walk) == pytest.approx(0.77 / 0.98)  # hip heights, not the leg lengths
    no_legs = _clip([0, 0], root_heights=(1.6, 2.0))
    assert root_scale(_arm_skeleton(), no_legs) == pytest.approx(0.5)  # falls back to the clip's highest root


@pytest.mark.parametrize("sizes, tolerance", [({}, 1e-9), ({"thigh": 0.225, "shin": 0.225, "ankle": 0.04}, 1e-9),
                                              ({"thigh": 0.36, "shin": 0.36, "ankle": 0.05}, 4e-3)])
def test_poser_keeps_the_feet_on_the_floor_when_the_knees_bend(sizes, tolerance):
    model, walk = _leg_model(**sizes), _bent_knee_walk()
    poser = Poser(model)
    toe = [poser.global_pose([Layer(walk, f / walk.fps)])[-1, 1, 3] for f in range(walk.frames)]
    np.testing.assert_allclose(toe, 0.0, atol=tolerance)


def test_poser_blends_layers():
    poser = Poser(_model())
    a, b = _clip([0, 0]), _clip([90, 90])
    local = poser.local_pose([Layer(a, 0, 1.0), Layer(b, 0, 1.0)])
    expected = quat_to_matrix(quat_from_axis_angle([0, 0, 1], 45))
    np.testing.assert_allclose(local[2, :3, :3], expected, atol=1e-9)


def test_rotate_in_character_space_turns_a_bone_about_world_up():
    skeleton = _arm_skeleton()
    local = skeleton.rest_local.copy()
    local[1, :3, :3] = quat_to_matrix(quat_from_axis_angle([1, 0, 0], 90))  # upper bone now points along +Z
    poser = Poser(CharacterModel("arm", skeleton, [], np.linalg.inv(skeleton.rest_global)))
    yaw = quat_to_matrix(quat_from_axis_angle([0, 1, 0], 90))
    before = skeleton.global_transforms(local)
    poser.rotate_in_character_space(local, 2, yaw)
    after = skeleton.global_transforms(local)
    np.testing.assert_allclose(after[2, :3, 3], before[2, :3, 3], atol=1e-12)  # the joint stays put
    np.testing.assert_allclose(after[2, :3, :3], yaw @ before[2, :3, :3], atol=1e-12)  # and turns about world up
    np.testing.assert_allclose(after[1], before[1], atol=1e-12)  # the parent is untouched


def test_decompose_inverts_compose():
    q = quat_from_axis_angle(np.eye(3), [10, 20, 30])
    t = np.arange(9.0).reshape(3, 3)
    q2, t2 = decompose(compose(q, t))
    np.testing.assert_allclose(q2, q, atol=1e-12)
    np.testing.assert_allclose(t2, t)
