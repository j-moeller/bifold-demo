import math
import os
import time
from pathlib import Path

import numpy as np
import pytest

from bifold_demo.characters import behaviour as B
from bifold_demo.characters.behaviour import Behaviour, Hide, Idle, MotionLibrary, Turn, Walk
from bifold_demo.characters.character import Character
from bifold_demo.characters.model import (
    AnimationClip,
    CharacterModel,
    Poser,
    Skeleton,
    SkinnedPrimitive,
    quat_from_axis_angle,
    quat_multiply,
    quat_to_matrix,
)

CAMERA = (0.0, 1.6, 6.0)
SCRATCH = "/tmp/claude-0/-home-user-bifold-demo/44ad5b06-773f-5ffd-8306-515788fc8497/scratchpad"
CHARACTERS = Path(os.environ.get("BIFOLD_CHARACTERS_TEST_DIR", f"{SCRATCH}/charcache"))  # a prepared library
BONES = ["Bip01", "Bip01 Pelvis", "Bip01 Spine", "Bip01 Neck", "Bip01 Head", "Bip01 L Thigh", "Bip01 R Thigh",
         "Bip01 L Calf", "Bip01 L Foot", "Bip01 L Toe0", "Bip01 R Calf", "Bip01 R Foot", "Bip01 R Toe0"]
PARENTS = [-1, 0, 1, 2, 3, 1, 1, 5, 7, 8, 6, 10, 11]
ROOT_HEIGHT = 0.95
# bone offsets from the parent in character axes (the root's is its height); straight legs put the toes on the floor
OFFSETS = [(0, ROOT_HEIGHT, 0), (0, 0, 0), (0, 0.15, 0), (0, 0.45, 0), (0, 0.1, 0), (0.1, -0.05, 0), (-0.1, -0.05, 0),
           (0, -0.45, 0), (0, -0.4, 0), (0, -0.05, 0.12), (0, -0.45, 0), (0, -0.4, 0), (0, -0.05, 0.12)]
TOES = [BONES.index("Bip01 L Toe0"), BONES.index("Bip01 R Toe0")]
# Like a Biped, the root bone's own axes are not the character's: every clip and the model carry this rotation.
FIX = quat_from_axis_angle([1.0, 1.0, 1.0], 120.0)
FIX_M = quat_to_matrix(FIX)


def _local_axis(axis):
    """``axis`` (character space) in the frame of the bones below the root, which inherit ``FIX``."""
    return FIX_M.T @ np.asarray(axis, dtype=np.float64)


def synthetic_clip(name, kind, frames, speed=0.0, turn=0.0, head=None, pitch=None, root_yaw=None):
    """A clip with the contract's metadata: in place, facing +Z at frame 0, root yaw kept inside turn clips."""
    u = np.linspace(0.0, 1.0, frames)
    rotations = np.zeros((frames, len(BONES), 4))
    rotations[..., 3] = 1.0
    yaw = root_yaw(u) if root_yaw is not None else np.zeros(frames)
    rotations[:, 0] = quat_multiply(quat_from_axis_angle([0.0, 1.0, 0.0], yaw), FIX)
    head_q = quat_from_axis_angle(_local_axis([0, 1, 0]), head(u) if head is not None else np.zeros(frames))
    pitch_q = quat_from_axis_angle(_local_axis([1, 0, 0]), pitch(u) if pitch is not None else np.zeros(frames))
    rotations[:, 4] = quat_multiply(head_q, pitch_q)
    swing = 30.0 * np.sin(2 * np.pi * u) if kind == "walk" else np.zeros(frames)
    rotations[:, 5] = quat_from_axis_angle(_local_axis([1, 0, 0]), swing)
    rotations[:, 6] = quat_from_axis_angle(_local_axis([1, 0, 0]), -swing)
    translations = np.zeros((frames, len(BONES), 3))
    translations[:, 1:] = [_local_axis(offset) for offset in OFFSETS[1:]]  # the rig the clip was made for
    translations[:, 0, 1] = ROOT_HEIGHT + (0.02 * np.cos(4 * np.pi * u) if kind == "walk" else 0.0)
    return AnimationClip(name, 30.0, list(BONES), rotations, translations, loop=kind != "turn", root="Bip01",
                         metadata={"kind": kind, "speed": speed, "turn": turn})


def _smooth(u):
    return u * u * (3 - 2 * u)


def synthetic_clips(full=False):
    """Clips named like the Rocketbox ones; ``full=False`` lacks the same roles as the prepared real library."""
    clips = [
        synthetic_clip("m_walk_neutral_01", "walk", 37, speed=1.0),
        synthetic_clip("f_walk_neutral_01", "walk", 37, speed=1.2),
        synthetic_clip("m_walk_slow_01", "walk", 45, speed=0.8),
        synthetic_clip("m_idle_neutral_01", "idle", 121, head=lambda u: -20 * np.sin(np.pi * u) ** 2),
        synthetic_clip("f_idle_neutral_01", "idle", 151, head=lambda u: 35 * np.sin(2 * np.pi * u) ** 2,
                       pitch=lambda u: 8 * np.sin(2 * np.pi * u) ** 2),
        synthetic_clip("m_idle_look_around_01", "gesture", 140, head=lambda u: -115 * np.sin(np.pi * u) ** 2),
        synthetic_clip("m_gestic_talk_neutral_01", "gesture", 300,
                       head=lambda u: -45 * np.sin(np.pi * u) ** 2 + 15 * np.sin(6 * np.pi * u)),
        synthetic_clip("f_gestic_listen_neutral_01", "gesture", 300, head=lambda u: 9 * np.sin(4 * np.pi * u)),
        synthetic_clip("m_turn_left_180", "turn", 74, turn=180.0, root_yaw=lambda u: 180 * _smooth(u)),
        synthetic_clip("f_turn_right_90", "turn", 71, turn=-90.0, root_yaw=lambda u: -90 * _smooth(u)),
    ]
    if full:
        clips += [
            synthetic_clip("m_turn_right_180", "turn", 74, turn=-180.0, root_yaw=lambda u: -180 * _smooth(u)),
            synthetic_clip("m_turn_left_90", "turn", 60, turn=90.0, root_yaw=lambda u: 90 * _smooth(u)),
            synthetic_clip("m_cell_phone_textmessage", "gesture", 150, pitch=lambda u: 20 + 0 * u),
        ]
    return {clip.name: clip for clip in clips}


def synthetic_model(size=0.9):
    """A two-box person on the clip skeleton, ``size`` times the clips' proportions (so ``stride_scale = size``)."""
    from bifold_demo.scene.geometry import box_mesh

    rest = np.repeat(np.eye(4)[None], len(BONES), axis=0)
    rest[0, :3, :3] = FIX_M
    rest[0, :3, 3] = np.multiply(OFFSETS[0], size)
    for i in range(1, len(BONES)):
        rest[i, :3, 3] = _local_axis(OFFSETS[i]) * (size if i > 2 else 1.0)
    skeleton = Skeleton(BONES, PARENTS, rest)
    primitives = []
    for bone, lo, hi in ((1, (-0.2, 0.0, -0.12), (0.2, 1.4, 0.12)), (4, (-0.1, 1.45, -0.1), (0.1, 1.7, 0.1))):
        lo, hi = np.asarray(lo) * size, np.asarray(hi) * size
        mesh = box_mesh(*(hi - lo))
        positions = (mesh.positions + (lo + hi) / 2).astype(np.float32)
        joints = np.zeros((len(positions), 4), np.int64)
        weights = np.zeros((len(positions), 4), np.float32)
        joints[:, 0], weights[:, 0] = bone, 1.0
        primitives.append(SkinnedPrimitive(f"part{bone}", positions, mesh.normals, mesh.uvs, mesh.indices, joints,
                                           weights, np.full((2, 2, 4), 200, np.uint8)))
    return CharacterModel("synthetic", skeleton, primitives, np.linalg.inv(skeleton.rest_global))


@pytest.fixture(scope="module")
def sparse():
    return MotionLibrary(synthetic_clips())


@pytest.fixture(scope="module")
def full():
    return MotionLibrary(synthetic_clips(full=True))


@pytest.fixture(scope="module")
def behaviours(sparse, full):
    """Every factory in typical and tricky places, for each library."""
    return {name: list(_factories(lib)) for name, lib in (("sparse", sparse), ("full", full), (None, None))}


def _yaw(direction):
    return math.degrees(math.atan2(direction[0], direction[2]))


def _angle_to_camera(behaviour, t, camera=CAMERA):
    state = behaviour(t)
    x, _, z = state.position
    cx, cz = (camera(t)[0], camera(t)[2]) if callable(camera) else (camera[0], camera[2])
    if not state.visible or math.hypot(cx - x, cz - z) < B.MIN_CAMERA_DISTANCE:
        return None
    return abs(B._wrap(behaviour.face_heading(t) - math.degrees(math.atan2(cx - x, cz - z))))


_ROOT_YAW = {}


def _root_yaw(clip):
    """Per-frame yaw of the clip's root bone (the synthetic root rotation is a yaw times ``FIX``)."""
    if clip.name not in _ROOT_YAW or _ROOT_YAW[clip.name][0] is not clip:
        q = quat_multiply(clip.rotations[:, 0], FIX * [-1, -1, -1, 1])
        _ROOT_YAW[clip.name] = (clip, np.degrees(np.unwrap(2 * np.arctan2(q[:, 1], q[:, 3]))).tolist())
    return _ROOT_YAW[clip.name][1]


def _body_yaw(state):
    """``heading`` plus the (blended) yaw of the root bone: where the body faces."""
    body = None
    for layer in state.layers:
        clip, yaw = layer.clip, _root_yaw(layer.clip)
        last = clip.frames - 1
        f = layer.time * clip.fps
        f = f % last if clip.loop else min(max(f, 0.0), last)
        i = min(int(f), last - 1)
        value = yaw[i] + (yaw[i + 1] - yaw[i]) * (f - i)
        body = value if body is None else body + B._wrap(value - body) * layer.weight
    return state.heading + (body or 0.0)


def _factories(lib, camera=CAMERA):
    yield "approach", B.approach((-1, 0, -14), camera=camera, library=lib, rng=1)
    yield "approach_straight", B.approach((0, 0, -14), camera=camera, library=lib, rng=2, bow=0.0,
                                          style={"sex": "female", "age_group": "older"})
    yield "walk_by", B.walk_by([(-9, 0, -7), (9, 0, -7)], camera=camera, library=lib, rng=3)
    yield "walk_at_camera", B.walk_by([(0.2, 0, -12), (0.2, 0, 3.5)], camera=camera, library=lib, rng=4)
    yield "walk_past_camera", B.walk_by([(1.0, 0, -8), (1.0, 0, 11)], camera=camera, library=lib, rng=5)
    yield "teleport", B.walk_by([(-9, 0, -3), (9, 0, -3)], camera=camera, library=lib, rng=6, return_mode="teleport")
    yield "wander", B.wander(((-3, 0), (3, 4)), 7, camera=camera, library=lib, style={"sex": "female"})
    yield "chat", B.chat((-2, 0, -6), (-1.4, 0, -6.6), partners=[(-1.4, 0, -6.6)], camera=camera, library=lib, rng=8)
    yield "stand_facing_camera", B.stand((0, 0, 2), 0.0, camera=camera, library=lib, rng=9, style={"sex": "female"})
    yield "stand_next_to_camera", B.stand((0.9, 0, 6.2), 90.0, camera=camera, library=lib, rng=10)


# -- motion library ------------------------------------------------------------------------------------


def test_roles_from_clip_names_and_fallbacks(sparse):
    assert B.classify_clip("m_walk_neutral_01") == "walk" and B.classify_clip("f_turn_right_90") == "turn_right_90"
    assert B.classify_clip("m_idle_look_around_01") == "look_around" and B.classify_clip("m_run_fast") is None
    assert sparse.names("walk") == ["f_walk_neutral_01", "m_walk_neutral_01"]
    assert sparse.role("walk", "female").name == "f_walk_neutral_01"
    assert sparse.role("walk", "male").name == "m_walk_neutral_01"
    assert sparse.role("stroll", "female").name == "m_walk_slow_01"  # related role, other sex variant
    assert sparse.role("phone", "male").name == "m_idle_neutral_01"  # no phone clip: idle instead
    assert sparse.role("turn_left_180", "female").name == "m_turn_left_180"
    with pytest.raises(KeyError):
        sparse.role("turn_left_90")
    library = MotionLibrary(synthetic_clips())
    library.register("phone", "m_gestic_talk_neutral_01")
    assert library.role("phone").name == "m_gestic_talk_neutral_01"
    assert library.role_of(library.clips["m_gestic_talk_neutral_01"]) == "talk"


def test_head_yaw_matches_forward_kinematics(sparse):
    model = synthetic_model()
    poser = Poser(model)
    head = model.skeleton.index("Bip01 Head")
    forward = model.skeleton.rest_global[head, :3, :3].T @ np.array([0.0, 0.0, 1.0])
    for name in ("f_idle_neutral_01", "m_idle_look_around_01", "m_turn_left_180", "f_turn_right_90"):
        clip = sparse.clips[name]
        yaw, elevation = sparse.head_yaw(clip), sparse.head_elevation(clip)
        for frame in range(0, clip.frames, 7):
            pose = poser.global_pose([B.Layer(clip, frame / clip.fps)])
            d = pose[head, :3, :3] @ forward
            assert B._wrap(_yaw(d) - yaw[frame]) == pytest.approx(0.0, abs=1e-6)
            assert math.degrees(math.asin(-d[1] / np.linalg.norm(d))) == pytest.approx(elevation[frame], abs=1e-6)
    after = sparse.after_turn(sparse.clips["m_turn_left_180"])
    assert sparse.head_yaw(after)[-1] == pytest.approx(0.0, abs=1e-6)  # yawed back by its turn


def test_stride_scale_is_the_posers_size_ratio(sparse):
    assert B.stride_scale(synthetic_model(0.8), sparse.clips["m_idle_neutral_01"]) == pytest.approx(0.8)


def test_face_heading_is_where_the_rendered_head_points(full):
    """``face_heading`` includes the clip's own head motion and Character's way of applying yaw and pitch."""
    model = synthetic_model()
    head = model.skeleton.index("Bip01 Head")
    forward = model.skeleton.rest_global[head, :3, :3].T @ np.array([0.0, 0.0, 1.0])
    worst = 0.0
    for behaviour in (B.stand((0, 0, 2), 0.0, camera=CAMERA, library=full, rng=3, style={"sex": "female"}),
                      B.approach((2, 0, -12), camera=CAMERA, library=full, rng=4)):
        character = Character(model, behaviour)
        for t in np.linspace(0.0, behaviour.duration, 150):
            character.update(t)
            d = character.pose[head, :3, :3] @ forward
            worst = max(worst, abs(B._wrap(character.state.heading + _yaw(d) - behaviour.face_heading(t))))
    assert worst < 0.5


def test_pitch_shift_matches_character_head_turn():
    model = synthetic_model()
    head = model.skeleton.index("Bip01 Head")
    forward = model.skeleton.rest_global[head, :3, :3].T @ np.array([0.0, 0.0, 1.0])
    clip = synthetic_clip("look", "gesture", 2, head=lambda u: 50 + 0 * u, pitch=lambda u: 10 + 0 * u)
    for yaw, pitch in ((0.0, 30.0), (-40.0, 28.0), (60.0, -15.0)):
        character = Character(model, lambda t: B.CharacterState((0, 0, 0), 0.0, [B.Layer(clip, 0.0)], yaw, pitch))
        character.update(0.0)
        face = _yaw(character.pose[head, :3, :3] @ forward)
        assert B._wrap(face - (50.0 + yaw + B.pitch_shift(50.0, 10.0, yaw, pitch))) == pytest.approx(0.0, abs=0.05)
    assert abs(B.pitch_shift(50.0, 10.0, -40.0, 28.0)) > 1.0  # the effect is real


# -- motion ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("library", ["sparse", "full", None])
def test_position_and_body_heading_are_continuous(library, behaviours):
    dt = 1 / 30
    for name, behaviour in behaviours[library]:
        hidden = behaviour.teleports()
        assert all(behaviour(t).visible for t in np.arange(0, behaviour.duration, 0.1)) == (not hidden)
        previous = None
        for t in np.arange(0.0, 2.0 * behaviour.duration, dt):
            state = behaviour(t)
            body = _body_yaw(state)
            tl = (t + behaviour.offset) % behaviour.duration
            if previous is not None and not any(a - dt <= tl <= b + dt for a, b in hidden):
                (x0, _, z0), h0, b0 = previous
                x, _, z = state.position
                assert math.hypot(x - x0, z - z0) < 2.0 * dt, (name, t)  # walking speed, no jumps
                assert abs(B._wrap(body - b0)) < 12.0, (name, t)  # turning at most 360 deg/s
                jump = abs(B._wrap(state.heading - h0))
                assert jump < 12.0 or abs(jump - 90.0) < 3.0 or abs(jump - 180.0) < 3.0, (name, t)  # turn clips
            previous = (state.position, state.heading, body)


def test_feet_do_not_slide(sparse):
    style = {"sex": "male", "stride_scale": 0.85}
    behaviour = B.walk_by([(-6, 0, -5), (0, 0, -9), (6, 0, -5)], camera=CAMERA, library=sparse, style=style, rng=2)
    clip = sparse.clips["m_walk_neutral_01"]
    checked = 0
    for t in np.arange(0.0, behaviour.duration, 0.05):
        a, b = behaviour(t), behaviour(t + 0.01)
        if len(a.layers) == 1 and len(b.layers) == 1 and a.layers[0].clip is clip and b.layers[0].clip is clip:
            moved = math.dist(a.position, b.position)
            stepped = (b.layers[0].time - a.layers[0].time) * clip.metadata["speed"] * style["stride_scale"]
            assert stepped == pytest.approx(moved, rel=B.STRIDE_STRETCH + 1e-3, abs=1e-5)
            checked += 1
    assert checked > 100


def test_feet_are_the_clips_toe_positions(sparse):
    """``MotionLibrary.feet`` is the forward kinematics of the clip's own skeleton, found from the bone names."""
    poser = Poser(synthetic_model(1.0))  # the proportions the clips were made for
    for name in ("m_walk_neutral_01", "m_idle_neutral_01", "m_turn_left_180"):
        clip = sparse.clips[name]
        times = np.arange(0, clip.frames, 5) / clip.fps
        expected = np.array([poser.global_pose([B.Layer(clip, t)])[TOES, :3, 3] for t in times])
        assert np.allclose(sparse.feet(clip, times), expected, atol=1e-9)
        between = poser.global_pose([B.Layer(clip, times[3] + 0.4 / clip.fps)])[TOES, :3, 3]
        assert np.allclose(sparse.feet(clip, times[3] + 0.4 / clip.fps), between, atol=1e-3)  # interpolated
    # in a Biped the thighs hang from the spine: the bone that puts the feet on the floor
    parents, offsets = list(PARENTS), list(OFFSETS)
    parents[5] = parents[6] = 2
    offsets[5], offsets[6] = (0.1, -0.2, 0.0), (-0.1, -0.2, 0.0)
    rest = np.repeat(np.eye(4)[None], len(BONES), axis=0)
    rest[0, :3, :3], rest[0, :3, 3] = FIX_M, offsets[0]
    rest[1:, :3, 3] = [_local_axis(offset) for offset in offsets[1:]]
    skeleton = Skeleton(BONES, parents, rest)
    biped = Poser(CharacterModel("biped", skeleton, [], np.linalg.inv(skeleton.rest_global)))
    clip = synthetic_clip("m_walk_neutral_01", "walk", 37, speed=1.0)
    clip.translations[:, 1:] = [_local_axis(offset) for offset in offsets[1:]]
    clip.rotations[:, 2] = quat_from_axis_angle(_local_axis([1, 0, 0]), 8.0 * np.sin(np.linspace(0, 2 * np.pi, 37)))
    times = np.arange(0, clip.frames, 4) / clip.fps
    expected = np.array([biped.global_pose([B.Layer(clip, t)])[TOES, :3, 3] for t in times])
    assert np.allclose(MotionLibrary({}).feet(clip, times), expected, atol=1e-9)
    legless = AnimationClip("legless", 30.0, BONES[:5], np.tile([0.0, 0.0, 0.0, 1.0], (3, 5, 1)), np.zeros((3, 5, 3)))
    assert sparse.feet(legless, 0.0) is None
    mixamo = AnimationClip("mixamo", 30.0, ["mixamorig:Hips", "mixamorig:LeftUpLeg", "mixamorig:LeftLeg",
                                            "mixamorig:LeftFoot", "mixamorig:LeftToeBase", "mixamorig:LeftHand"],
                           np.zeros((1, 6, 4)), np.zeros((1, 6, 3)))
    assert B._leg(mixamo, "l") == [1, 2, 3, 4] and B._leg(mixamo, "r") == []


def _toe_mismatch(behaviour, poser, toes=TOES):
    """``(walk length, "start" or "stop", (less, more))`` for each crossfade between walking and standing: how far
    each toe is (on the floor plane, in character space) between the two poses the crossfade blends, in metres."""
    out, entries = [], behaviour._entries
    for j, entry in enumerate(entries):
        before = entries[j - 1]
        if not entry.fade or entry.clip is None or before.clip is None or (entry.walk is None) == (before.walk is None):
            continue
        a, b = (poser.global_pose([B.Layer(*e.sample(t)[:2])])[toes][:, [0, 2], 3]
                for e, t in ((before, entry.t0 if j else entry.t0 + behaviour.duration), (entry, entry.t0)))
        walk = entry if entry.walk is not None else before
        out.append((walk.walk.length, "start" if walk is entry else "stop", sorted(np.linalg.norm(b - a, axis=1))))
    return out


def test_walks_start_and_stop_in_the_pose_they_blend_with(sparse):
    """Walks start where the walk clip's legs are together, like standing, and are stretched (by at most
    ``STRIDE_STRETCH``) to stop there too, so the crossfades do not drag the feet over the floor."""
    poser = Poser(synthetic_model())
    found = [m for _, behaviour in _factories(sparse) for m in _toe_mismatch(behaviour, poser)]
    assert len(found) > 20 and {kind for _, kind, _ in found} == {"start", "stop"}
    assert max(m[1] for length, _, m in found if length > 6.5) < 0.03  # a random phase: up to 0.4 m
    assert np.mean([m[1] for _, _, m in found]) < 0.05  # short walks may not stretch enough to stop in step
    for _, behaviour in _factories(sparse):
        for entry in behaviour._entries:
            if entry.walk is not None and entry.clip is not None:
                stretch = entry.scale * entry.clip.metadata["speed"] * behaviour.style.stride - 1.0
                assert abs(stretch) <= B.STRIDE_STRETCH + 1e-9
    on = Behaviour([Walk([(0, 4)]), Walk([(0, 9)])], sparse, rng=1)  # straight on: one stop and go, in step
    first, second = [e for e in on._entries if e.walk is not None][:2]
    assert second.t0 == first.t1 and second.sample(second.t0)[1] == pytest.approx(first.sample(first.t1)[1])


@pytest.mark.skipif(not (CHARACTERS / "clips").is_dir(), reason="no prepared Rocketbox characters")
def test_real_walks_start_and_stop_with_the_feet_in_place(monkeypatch):
    """With the Rocketbox clips ``feet`` finds the Biped's legs (toes on the floor in every clip), and the
    crossfades between standing and walking move the toes about half as far as random walk phases did (the rest
    is the clips' wider stance when standing)."""
    from bifold_demo.characters.crowd import avatar_style
    from bifold_demo.characters.library import CharacterLibrary

    library = CharacterLibrary(CHARACTERS)
    motions, available = library.motions(), library.available()
    avatars = [next((n for n, e in sorted(available.items()) if e.get("sex") == sex and e.get("age_group") == "adult"),
                    None) for sex in ("male", "female")]
    if not library.is_ready() or None in avatars:
        pytest.skip("the prepared library lacks clips or adult avatars")
    for clip in motions.clips.values():
        feet = motions.feet(clip, np.arange(clip.frames) / clip.fps)
        assert feet is not None and np.abs(feet[..., 1].min(axis=1)).mean() < 0.03, clip.name

    def mismatch():
        found = []
        for name in avatars:
            model = library.load(name, textures="none")
            toes = [model.skeleton.find("Bip01 L Toe0"), model.skeleton.find("Bip01 R Toe0")]
            style = avatar_style(available[name], model, motions)
            for seed in range(3):
                for behaviour in (B.approach((-1, 0, -14), camera=CAMERA, library=motions, rng=seed, style=style),
                                  B.walk_by([(-9, 0, -7), (9, 0, -7)], camera=CAMERA, library=motions, rng=seed,
                                            style=style)):
                    found += [m for _, _, m in _toe_mismatch(behaviour, Poser(model), toes)]
        return np.array(found)

    aligned = mismatch()
    monkeypatch.setattr(B.Behaviour, "_time_walks", lambda self: None)  # the walk clips at random phases
    random_phase = mismatch()
    assert len(aligned) >= 20
    assert aligned[:, 0].mean() < 0.5 * random_phase[:, 0].mean() and aligned[:, 0].mean() < 0.09  # planted toe
    assert aligned[:, 1].mean() < 0.7 * random_phase[:, 1].mean()


def test_default_walking_speed_follows_the_clip(sparse):
    behaviour = B.walk_by([(-9, 0, -7), (9, 0, -7)], camera=CAMERA, library=sparse, rng=1, style={"sex": "female"})
    piece = next(p for p in behaviour._pieces if p.kind == "walk")
    assert 1.2 * 0.92 <= piece.speed <= 1.2 * 1.1  # f_walk_neutral_01 speed, pace +-10 %
    assert B.walk_by([(-9, 0), (9, 0)], 1.5, library=sparse)._pieces[0].speed == 1.5


def test_approach_comes_close_and_turns_around(sparse):
    for seed in range(3):
        behaviour = B.approach((1, 0, -14), camera=CAMERA, library=sparse, rng=seed, stop_distance=(2.5, 3.5),
                               offset=0.0)
        states = [behaviour(t) for t in np.arange(0.0, behaviour.duration, 0.05)]
        distance = np.array([math.hypot(s.position[0] - CAMERA[0], s.position[2] - CAMERA[2]) for s in states])
        nearest = int(np.argmin(distance))  # where it stops
        assert 2.5 - 1e-6 <= distance[nearest] <= 3.5 + 1e-6
        assert distance[0] > 19.0 and np.all(np.diff(distance[:nearest]) <= 1e-9)  # coming closer all the way
        arriving = _body_yaw(states[nearest - 40])
        leaving = next(i for i in range(nearest, len(states)) if distance[i] > distance[nearest] + 1.0)
        assert abs(abs(B._wrap(_body_yaw(states[leaving]) - arriving)) - 180.0) < 25.0  # turned around
        assert any(s.layers and s.layers[-1].clip.metadata.get("kind") == "turn" for s in states)


def test_turns_fall_back_without_fitting_clips(sparse):
    def turn_clips(behaviour):
        return {layer.clip.name.split("@")[0] for t in np.arange(0, behaviour.duration, 0.1)
                for layer in behaviour(t).layers if layer.clip.metadata.get("kind") == "turn"}

    left90 = Behaviour([Idle(2.0), Turn(90.0), Idle(2.0), Turn(-90.0)], sparse, rng=1)
    assert turn_clips(left90) == {"f_turn_right_90"}  # no 90-degree left clip, and the 180 one does not fit
    around = Behaviour([Idle(2.0), Turn(-180.0), Idle(2.0), Turn(180.0)], sparse, rng=1)
    assert turn_clips(around) == {"f_turn_right_90", "m_turn_left_180"}  # the 90 clip plus 90 degrees by hand
    assert around(around.duration * 0.5).heading == pytest.approx(-180.0, abs=1e-6)
    walks_only = MotionLibrary({k: v for k, v in synthetic_clips().items() if "walk" in k or "idle_neutral" in k})
    behaviour = B.approach((0, 0, -10), camera=CAMERA, library=walks_only, rng=3)
    assert behaviour.look_feasible and not turn_clips(behaviour)
    headings = [behaviour(t).heading for t in np.arange(0, behaviour.duration, 1 / 30)]
    assert max(abs(B._wrap(b - a)) for a, b in zip(headings, headings[1:])) < 12.0
    bare = B.walk_by([(-5, 0), (5, 0)], camera=CAMERA, library=None, rng=1)
    assert all(not bare(t).layers for t in np.arange(0, bare.duration, 0.5))


def test_crossfades_blend_two_layers_with_weights_summing_to_one(behaviours):
    fades = 0
    for _, behaviour in behaviours["sparse"]:
        for t in np.arange(0.0, behaviour.duration, 0.02):
            layers = behaviour(t).layers
            assert len(layers) <= 2
            if layers:
                assert sum(layer.weight for layer in layers) == pytest.approx(1.0)
                assert all(0.0 <= layer.weight <= 1.0 for layer in layers)
            fades += len(layers) == 2
    assert fades > 50


def test_behaviours_are_deterministic(sparse):
    def snapshot(behaviour):
        return [(s.position, s.heading, s.head_yaw, s.head_pitch, s.visible,
                 [(layer.clip.name, layer.time, layer.weight) for layer in s.layers])
                for s in (behaviour(t) for t in np.linspace(0, 200, 97))]

    for make in (lambda r: B.approach((-1, 0, -14), camera=CAMERA, library=sparse, rng=r),
                 lambda r: B.wander(((-5, -8), (5, 0)), r, camera=CAMERA, library=sparse),
                 lambda r: B.chat((0, 0, -4), (1, 0, -4), camera=CAMERA, library=sparse, rng=r)):
        assert snapshot(make(5)) == snapshot(make(5))
        assert snapshot(make(5)) != snapshot(make(6))
    a = B.walk_by([(-9, 0, -7), (9, 0, -7)], camera=CAMERA, library=sparse, rng=np.random.default_rng(3))
    b = B.walk_by([(-9, 0, -7), (9, 0, -7)], camera=CAMERA, library=sparse, rng=np.random.default_rng(3))
    assert snapshot(a) == snapshot(b)
    times = np.random.default_rng(1).uniform(0, 300, 50)
    first = [a(t) for t in times]
    assert [a(t) for t in times[::-1]][::-1] == first  # a pure function of t: order and history do not matter


# -- never looking at the camera -------------------------------------------------------------------------


@pytest.mark.parametrize("library", ["sparse", "full", None])
def test_never_looks_at_the_camera(library, behaviours):
    for name, behaviour in behaviours[library]:
        assert behaviour.look_feasible, name
        worst = 180.0
        for t in np.arange(0.0, 3.0 * behaviour.duration, 1 / 30):
            assert abs(behaviour(t).head_yaw) <= B.HEAD_YAW_LIMIT + 1e-9
            angle = _angle_to_camera(behaviour, t)
            if angle is not None:
                worst = min(worst, angle)
        assert worst >= B.MIN_ANGLE - 1e-6, (name, worst)


def test_never_looks_at_a_moving_camera(sparse):
    def camera(t):
        return 0.4 * math.sin(2 * math.pi * t / 23.0), 1.6, 6.0 + 0.3 * math.cos(2 * math.pi * t / 31.0)

    for name, behaviour in list(_factories(sparse, camera))[:5]:
        assert behaviour.look_feasible, name
        angles = [_angle_to_camera(behaviour, t, camera) for t in np.arange(0.0, 2.0 * behaviour.duration, 1 / 30)]
        assert min(a for a in angles if a is not None) >= B.MIN_ANGLE - 1e-6, name


def test_min_angle_and_head_limit_are_parameters(sparse):
    behaviour = B.approach((0, 0, -12), camera=CAMERA, library=sparse, rng=2, bow=0.0, min_angle=50.0)
    assert behaviour.look_feasible
    angles = [_angle_to_camera(behaviour, t) for t in np.arange(0.0, behaviour.duration, 1 / 30)]
    assert min(a for a in angles if a is not None) >= 50.0 - 1e-6


def test_turn_directions_keep_the_face_off_the_camera(sparse):
    """Turning around twice the same way on a line towards the camera would sweep the face across it."""
    behaviour = Behaviour([Walk([(0, -2)]), Idle(3.0), Turn(180.0, either_way=True), Walk([(0, -12)]), Idle(2.0),
                           Turn(180.0, either_way=True)], sparse, start=(0, -12), camera=CAMERA, rng=3)
    turns = [c[2] for c in behaviour._choices]
    assert sorted(turns) == [-180.0, 180.0] and behaviour.look_feasible
    fixed = Behaviour([Walk([(0, -2)]), Idle(3.0), Turn(180.0), Walk([(0, -12)]), Idle(2.0), Turn(180.0)], sparse,
                      start=(0, -12), camera=CAMERA, rng=3)
    assert not fixed.look_feasible  # without the choice there is no way


# -- cost ---------------------------------------------------------------------------------------------------


def test_build_and_evaluation_are_cheap(sparse):
    start = time.perf_counter()
    behaviours = [B.approach((-1, 0, -14), camera=CAMERA, library=sparse, rng=s) for s in range(4)] + \
        [B.walk_by([(-9, 0, -7), (9, 0, -7)], camera=CAMERA, library=sparse, rng=s) for s in range(4)]
    build = (time.perf_counter() - start) / len(behaviours)
    times = np.random.default_rng(0).uniform(0.0, 300.0, 2000)
    start = time.perf_counter()
    for behaviour in behaviours:
        for t in times:
            behaviour(t)
    evaluation = (time.perf_counter() - start) / (len(times) * len(behaviours))
    assert build < 0.15  # about 20 ms
    assert evaluation < 50e-6  # about 10 us


def test_segments_validate():
    with pytest.raises(TypeError):
        Behaviour([object()])
    with pytest.raises(ValueError):
        B.walk_by([(0, 0)], library=None)
    with pytest.raises(ValueError):
        B.approach((0, 0), library=None)
    hide = Behaviour([Walk([(5, 0)]), Hide(2.0)], None, start=(0, 0), heading=90.0)
    assert hide.teleports() and not hide(hide.duration - 1.0).visible
