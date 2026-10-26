"""Yoga pose clips: the solver on a synthetic Biped, and every pose on real avatars when they are prepared."""

import math
import os
from pathlib import Path

import numpy as np
import pytest

from bifold_demo.characters import yoga
from bifold_demo.characters.model import CharacterModel, Layer, Poser, Skeleton, SkinnedPrimitive
from bifold_demo.scene.geometry import box_mesh

# a prepared character library (bifold-characters fetch --dest DIR) for the real-avatar tests; they skip without
CHARACTERS = Path(os.environ.get("BIFOLD_CHARACTERS_TEST_DIR", "/nonexistent")).expanduser()

# rest positions (character space, metres) of a Rocketbox adult, rounded; the rest rotations are left as identity
CORE = [("Bip01", None, (0, 0.923, 0)), ("Bip01 Pelvis", "Bip01", (0, 0.923, 0)),
        ("Bip01 Spine", "Bip01 Pelvis", (0, 1.044, -0.002)), ("Bip01 Spine1", "Bip01 Spine", (0, 1.168, -0.004)),
        ("Bip01 Spine2", "Bip01 Spine1", (0, 1.289, -0.018)), ("Bip01 Neck", "Bip01 Spine2", (0, 1.454, -0.072)),
        ("Bip01 Head", "Bip01 Neck", (0, 1.522, -0.056))]
SIDE = [("Clavicle", "Neck", (0.077, 1.403, -0.056)), ("UpperArm", "Clavicle", (0.172, 1.380, -0.061)),
        ("Forearm", "UpperArm", (0.359, 1.210, -0.063)), ("Hand", "Forearm", (0.524, 1.063, 0.031)),
        ("Finger0", "Hand", (0.526, 1.044, 0.07)), ("Finger01", "Finger0", (0.524, 1.024, 0.088)),
        ("Finger02", "Finger01", (0.527, 0.991, 0.106)), ("Finger1", "Hand", (0.567, 0.997, 0.084)),
        ("Finger11", "Finger1", (0.586, 0.967, 0.097)), ("Finger12", "Finger11", (0.593, 0.947, 0.104)),
        ("Finger2", "Hand", (0.577, 0.994, 0.066)), ("Finger21", "Finger2", (0.595, 0.956, 0.077)),
        ("Finger22", "Finger21", (0.603, 0.93, 0.084)), ("Finger3", "Hand", (0.578, 0.99, 0.044)),
        ("Finger31", "Finger3", (0.593, 0.955, 0.054)), ("Finger32", "Finger31", (0.597, 0.935, 0.06)),
        ("Finger4", "Hand", (0.572, 0.987, 0.024)), ("Finger41", "Finger4", (0.577, 0.958, 0.031)),
        ("Finger42", "Finger41", (0.579, 0.943, 0.035)), ("Thigh", "Spine", (0.088, 0.923, 0.0)),
        ("Calf", "Thigh", (0.104, 0.499, -0.032)), ("Foot", "Calf", (0.118, 0.100, -0.063)),
        ("Toe0", "Foot", (0.152, 0.002, 0.069))]


def synthetic_biped() -> CharacterModel:
    """A box-limbed person with the Biped bone names (each box skinned to one bone)."""
    bones = list(CORE)
    for s, sign in (("L", 1), ("R", -1)):
        for name, parent, (x, y, z) in SIDE:
            parent = f"Bip01 {s} {parent}" if parent not in ("Neck", "Spine") else f"Bip01 {parent}"
            bones.append((f"Bip01 {s} {name}", parent, (sign * x, y, z)))
    names = [b[0] for b in bones]
    parents = [names.index(p) if p else -1 for _, p, _ in bones]
    position = np.array([b[2] for b in bones], float)
    rest = np.repeat(np.eye(4)[None], len(bones), axis=0)
    for k, p in enumerate(parents):
        rest[k, :3, 3] = position[k] - (position[p] if p >= 0 else 0)
    skeleton = Skeleton(names, parents, rest)
    primitives = []

    def box(bone, lo, hi):
        lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
        mesh = box_mesh(*np.maximum(hi - lo, 0.02))
        positions = (mesh.positions + (lo + hi) / 2).astype(np.float32)
        joints = np.zeros((len(positions), 4), np.int64)
        weights = np.zeros((len(positions), 4), np.float32)
        joints[:, 0], weights[:, 0] = bone, 1.0
        primitives.append(SkinnedPrimitive(f"box{bone}", positions, mesh.normals, mesh.uvs, mesh.indices, joints,
                                           weights, np.full((2, 2, 4), 180, np.uint8)))

    children = {k: [c for c, p in enumerate(parents) if p == k] for k in range(len(bones))}
    for k, name in enumerate(names):
        if name.endswith("Foot"):
            x = position[k, 0]
            box(k, (x - 0.045, 0.0, -0.12), (x + 0.045, 0.09, 0.07))  # the sole at y = 0, the heel behind
        elif name.endswith("Toe0"):
            x = position[k, 0]
            box(k, (x - 0.04, 0.0, 0.07), (x + 0.04, 0.03, 0.13))
        elif children[k]:
            end = position[children[k][0]]
            box(k, position[k] - 0.03, end + 0.03)
        else:
            box(k, position[k] - 0.015, position[k] + 0.015)
    return CharacterModel("synthetic_biped", skeleton, primitives, np.linalg.inv(skeleton.rest_global))


@pytest.fixture(scope="module")
def body():
    return yoga.Body(synthetic_biped())


def test_turn_follows_the_conventions():
    np.testing.assert_allclose(yoga.turn(pitch=90) @ (0, 1, 0), (0, 0, 1), atol=1e-12)  # bends forward
    np.testing.assert_allclose(yoga.turn(yaw=90) @ (0, 0, 1), (1, 0, 0), atol=1e-12)  # turns to the left (+X)
    np.testing.assert_allclose(yoga.turn(roll=90) @ (0, 1, 0), (-1, 0, 0), atol=1e-12)  # tilts to the right


def test_two_bone_ik_reaches_and_bends_towards_the_pole():
    start, target = np.array([0.0, 1.0, 0.0]), np.array([0.1, 0.3, 0.2])
    middle, end, _ = yoga._two_bone(start, 0.42, 0.40, target, (0, 0, 1))
    np.testing.assert_allclose(end, target, atol=1e-9)
    assert np.linalg.norm(middle - start) == pytest.approx(0.42) and np.linalg.norm(end - middle) == pytest.approx(0.40)
    assert middle[2] > 0.2  # the knee points forward
    middle, end, _ = yoga._two_bone(start, 0.42, 0.40, (0, -5, 0), (0, 0, 1))  # out of reach: stops, straight
    assert np.linalg.norm(end - start) == pytest.approx(0.82, rel=1e-3)


def test_solve_reaches_the_targets_and_matches_forward_kinematics(body):
    for pose in (yoga.mountain(body), yoga.POSES["yogi_squat"].steps(body)[1].pose):
        local, glob = yoga.solve(body, pose)
        np.testing.assert_allclose(body.skeleton.global_transforms(local), glob, atol=1e-9)  # a valid clip frame
        np.testing.assert_allclose(glob[body.root, :3, 3], pose.root, atol=1e-12)
        for side, s in body.sides.items():
            np.testing.assert_allclose(glob[s["hand"], :3, 3], getattr(pose, f"{side}_arm").target, atol=1e-6)
            np.testing.assert_allclose(glob[s["foot"], :3, 3], getattr(pose, f"{side}_leg").target, atol=1e-6)
            foot = glob[s["foot"], :3, :3] @ s["foot_forward"]
            np.testing.assert_allclose(foot, getattr(pose, f"{side}_leg").forward, atol=1e-6)  # toes where told
        rotations = glob[:, :3, :3]
        np.testing.assert_allclose(rotations @ rotations.transpose(0, 2, 1), np.repeat(np.eye(3)[None], len(glob), 0),
                                   atol=1e-9)


def test_settle_puts_the_contacts_on_the_floor(body):
    pose = yoga.POSES["yogi_squat"].steps(body)[1].pose
    lifted = yoga.replace(pose, left_leg=yoga.replace(pose.left_leg, target=(*pose.left_leg.target[:1],
                                                                              pose.left_leg.target[1] + 0.05,
                                                                              pose.left_leg.target[2])))
    settled = yoga.settle(body, lifted)
    verts = yoga.skin(body, yoga.solve(body, settled)[1])
    for side in body.sides:
        assert verts[body.foot_group(side), 1].min() == pytest.approx(0.0, abs=1e-4)


def test_clips_loop_and_hold_with_whole_breaths(body):
    clip = yoga.yoga_clip(body.model, "yogi_squat")
    assert clip.loop and clip.root == "Bip01" and clip.metadata["kind"] == "yoga"
    np.testing.assert_allclose(clip.rotations[0], clip.rotations[-1], atol=1e-12)  # seamless loop
    np.testing.assert_allclose(clip.translations[0], clip.translations[-1], atol=1e-12)
    start, end = clip.metadata["hold"]
    assert end - start >= 4.0
    steps = [yoga.replace(s, pose=yoga.settle(body, s.pose)) for s in yoga.POSES["yogi_squat"].steps(body)]
    pieces, duration = yoga.timeline(steps)
    assert duration == pytest.approx(clip.duration)
    at_start, at_end = yoga.pose_at(pieces, start + 1e-9), yoga.pose_at(pieces, end - 1e-9)
    assert at_start.spine == pytest.approx(at_end.spine, abs=1e-6)  # the breaths end where they began
    middle = yoga.pose_at(pieces, start + 0.5 * (end - start) / round((end - start) / yoga.BREATH))  # breathed in
    assert middle.spine[0] < at_start.spine[0] - 1.5  # the chest lifts


def test_clips_play_through_the_poser_as_solved(body):
    clip = yoga.yoga_clip(body.model, "yogi_squat")
    poser = Poser(body.model)
    frame = clip.frames // 2
    played = poser.global_pose([Layer(clip, frame / clip.fps)])
    expected = body.skeleton.global_transforms(
        np.concatenate([np.concatenate([_quat_matrix(clip.rotations[frame]), clip.translations[frame][..., None]], 2),
                        np.tile([0, 0, 0, 1.0], (len(body.names), 1, 1))], 1))
    np.testing.assert_allclose(played, expected, atol=1e-6)  # the root is not rescaled: the clip is the avatar's own


def _quat_matrix(q):
    from bifold_demo.characters.model import quat_to_matrix

    return quat_to_matrix(q)


def test_unknown_poses_and_skeletons_are_refused(body):
    with pytest.raises(KeyError):
        yoga.yoga_clip(body.model, "headstand")
    names = [n.replace("Bip01", "mixamorig") for n in body.names]
    other = CharacterModel("other", Skeleton(names, body.skeleton.parents, body.skeleton.rest_local),
                           body.model.primitives, body.model.inverse_bind)
    with pytest.raises(ValueError):
        yoga.Body(other)


def test_cow_hollows_the_back_on_hands_and_knees(body):
    spec = yoga.POSES["cow"]
    steps = spec.steps(body)
    table = yoga.settle(body, steps[spec.main - 1].pose)
    cow = yoga.settle(body, steps[spec.main].pose)
    shapes = []
    for pose in (table, cow):
        _, glob = yoga.solve(body, pose)
        left = body.sides["left"]
        hips, chest = glob[body.root, :3, 3], glob[body.spine[2], :3, 3]
        shoulder, wrist = glob[left["upper"], :3, 3], glob[left["hand"], :3, 3]
        knee, hip_joint = glob[left["calf"], :3, 3], glob[left["thigh"], :3, 3]
        assert abs(wrist[2] - shoulder[2]) < 0.12 and shoulder[1] > wrist[1] + 0.8 * body.arm  # arms under the shoulders
        assert abs(knee[2] - hip_joint[2]) < 0.15 and hip_joint[1] > knee[1] + 0.8 * left["thigh_length"]  # thighs up
        middle = glob[body.spine[1], :3, 3]  # how far the middle of the back sags below the hips-chest line
        line = hips + (chest - hips) * np.dot(middle - hips, chest - hips) / np.dot(chest - hips, chest - hips)
        shapes.append((line[1] - middle[1], glob[body.head, :3, :3] @ (0, 1, 0)))
    assert shapes[1][0] > shapes[0][0] + 0.01  # the belly sinks
    assert np.allclose(table.left_arm.target, cow.left_arm.target) and table.knees_down and cow.knees_down
    assert np.isclose(cow.left_arm.target[1], table.left_arm.target[1])


def _bend(glob, a, b, c) -> float:
    """How far the joint at ``b`` is bent (0 = straight), degrees."""
    u, w = glob[a, :3, 3] - glob[b, :3, 3], glob[c, :3, 3] - glob[b, :3, 3]
    return 180.0 - math.degrees(math.acos(np.clip(u @ w / np.linalg.norm(u) / np.linalg.norm(w), -1, 1)))


def test_warrior_1_lunges_with_the_arms_overhead(body):
    spec = yoga.POSES["warrior_1"]
    pose = yoga.settle(body, spec.steps(body)[spec.main].pose)
    _, glob = yoga.solve(body, pose)
    left, right = body.sides["left"], body.sides["right"]
    assert 75.0 < _bend(glob, left["thigh"], left["calf"], left["foot"]) < 105.0  # the front knee at a right angle
    assert _bend(glob, right["thigh"], right["calf"], right["foot"]) < 25.0  # the back leg straight
    front, back = glob[left["foot"], :3, 3], glob[right["foot"], :3, 3]
    assert front[2] - back[2] > 1.1 * body.leg  # a long stance
    assert abs(glob[left["calf"], 2, 3] - front[2]) < 0.1  # the knee over the ankle
    toes = glob[right["foot"], :3, :3] @ right["foot_forward"]
    assert 30.0 < math.degrees(math.atan2(-toes[0], toes[2])) < 60.0  # the back foot turned out
    head = glob[body.head, 1, 3]
    for s in (left, right):
        assert glob[s["hand"], 1, 3] > head + 0.25  # the hands high overhead ...
        assert _bend(glob, s["upper"], s["fore"], s["hand"]) < 15.0  # ... on straight arms
    assert pose.pelvis[1] == 0.0  # the hips square to the front


def test_downward_dog_is_an_inverted_v(body):
    spec = yoga.POSES["downward_dog"]
    pose = yoga.settle(body, spec.steps(body)[spec.main].pose)
    _, glob = yoga.solve(body, pose)
    left = body.sides["left"]
    hips, head = glob[body.root, :3, 3], glob[body.head, :3, 3]
    hand, foot, shoulder = glob[left["hand"], :3, 3], glob[left["foot"], :3, 3], glob[left["upper"], :3, 3]
    assert hips[1] > 0.6 * body.leg  # the hips are the highest point of the body
    assert hips[1] > shoulder[1] > head[1] - 0.05  # the head hangs between the arms
    assert hand[2] - foot[2] > 0.9 * body.leg  # hands and feet far apart
    for s in body.sides.values():
        assert _bend(glob, s["upper"], s["fore"], s["hand"]) < 15.0  # straight arms ...
        assert _bend(glob, s["thigh"], s["calf"], s["foot"]) < 15.0  # ... and legs
    trunk = math.degrees(math.atan2(hips[1] - hand[1], hips[2] - hand[2]))
    assert 120.0 < trunk < 145.0  # arms and back slope up from the hands, the legs down to the feet
    assert all(getattr(pose, limb).contact for limb in yoga.LIMBS)  # both hands and both feet on the floor


def test_warrior_2_opens_to_the_front_with_the_arms_level(body):
    spec = yoga.POSES["warrior_2"]
    pose = yoga.settle(body, spec.steps(body)[spec.main].pose)
    _, glob = yoga.solve(body, pose)
    left, right = body.sides["left"], body.sides["right"]
    front, back = glob[left["foot"], :3, 3], glob[right["foot"], :3, 3]
    assert front[0] - back[0] > 1.2 * body.leg and abs(front[2] - back[2]) < 0.05  # wide, across the body
    assert 75.0 < _bend(glob, left["thigh"], left["calf"], left["foot"]) < 105.0  # the front knee bent ...
    assert abs(glob[left["calf"], 0, 3] - front[0]) < 0.1  # ... over the ankle
    assert _bend(glob, right["thigh"], right["calf"], right["foot"]) < 15.0  # the back leg straight
    toes = glob[left["foot"], :3, :3] @ left["foot_forward"]
    assert toes[0] > 0.95  # the front foot points along the stance
    for s in (left, right):
        shoulder, wrist = glob[s["upper"], :3, 3], glob[s["hand"], :3, 3]
        assert abs(wrist[1] - shoulder[1]) < 0.06 and abs(wrist[0] - shoulder[0]) > 0.9 * body.arm  # level arms
    gaze = glob[body.head, :3, :3] @ body.rest_rot[body.head].T @ (0.0, 0.0, 1.0)
    assert gaze[0] > 0.8  # looking over the front hand


def test_yogi_squat_is_deep_with_wide_knees_and_palms_together(body):
    pose = yoga.settle(body, yoga.POSES["yogi_squat"].steps(body)[1].pose)
    _, glob = yoga.solve(body, pose)
    left, right = body.sides["left"], body.sides["right"]
    hips = glob[body.root, 1, 3]
    knees = [glob[s["calf"], :3, 3] for s in (left, right)]
    feet = [glob[s["foot"], :3, 3] for s in (left, right)]
    assert hips < min(k[1] for k in knees)  # the hips below the knees
    assert hips < 0.4 * body.leg + 0.12
    assert all(abs(k[0]) > abs(f[0]) for k, f in zip(knees, feet))  # knees wider than the feet
    hands = [glob[s["hand"], :3, 3] for s in (left, right)]
    assert np.linalg.norm(hands[0] - hands[1]) < 0.1  # palms together ...
    assert hands[0][1] > glob[body.root, 1, 3] + 0.3 * body.torso  # ... at the chest
    elbows = [glob[s["fore"], :3, 3] for s in (left, right)]
    assert all(np.linalg.norm(e - k) < 0.2 for e, k in zip(elbows, knees))  # elbows at the knees


# -- every pose on real avatars --------------------------------------------------------------------------------


def _real_avatars():
    from bifold_demo.characters.library import CharacterLibrary

    if not (CHARACTERS / "avatars").is_dir():
        pytest.skip("set BIFOLD_CHARACTERS_TEST_DIR to a prepared character library")
    library = CharacterLibrary(CHARACTERS)
    available = library.available()
    picks = [next((n for n in sorted(available) if available[n].get("age_group") == group
                   and available[n].get("sex") == sex), None)
             for group, sex in (("adult", "female"), ("adult", "male"), ("child", "female"), ("child", "male"))]
    picks = [p for p in picks if p]
    if not picks:
        pytest.skip("the prepared library has no avatars")
    return library, picks


@pytest.mark.parametrize("name", list(yoga.POSES))
def test_real_avatars_do_every_pose_on_the_floor(name):
    library, avatars = _real_avatars()
    for avatar in avatars:
        model = library.load(avatar, textures="none")
        body = yoga.Body(model)
        clip = yoga.yoga_clip(model, name)
        poser = Poser(model)
        hold = 0.5 * sum(clip.metadata["hold"])
        lowest = min(float(yoga.skin(body, poser.global_pose([Layer(clip, t)]))[:, 1].min())
                     for t in np.linspace(0.0, clip.duration, 24))
        assert lowest > -0.015, (avatar, lowest)  # nothing sinks into the floor
        spec = yoga.POSES[name]
        pose = yoga.settle(body, spec.steps(body)[spec.main].pose)
        verts = yoga.skin(body, poser.global_pose([Layer(clip, hold)]))
        for limb in yoga.LIMBS:
            if getattr(pose, limb).contact:
                side = limb.split("_")[0]
                group = body.hand_group(side) if limb.endswith("arm") else body.foot_group(side)
                kneeling = pose.knees_down and limb.endswith("leg") and yoga._dorsum_down(getattr(pose, limb))
                limit = 0.025 if kneeling else 0.01  # the top of a kneeling foot can only point down so far
                assert abs(verts[group, 1].min()) < limit, (avatar, limb)  # resting on the floor
        if pose.knees_down:
            for side in body.sides:
                assert abs(verts[body.knee_group(side), 1].min()) < 0.015, (avatar, side)  # knees on the floor
        assert math.isfinite(clip.metadata["root_height"])
