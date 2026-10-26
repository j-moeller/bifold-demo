"""The audience playing with the camera: the moves, the plan, posing them, and the fair's audience doing them."""

import itertools
import math
import os
import time
from pathlib import Path

import numpy as np
import pytest
from test_yoga import synthetic_biped

from bifold_demo.characters import accessories, faces
from bifold_demo.characters import camera_play as cp
from bifold_demo.characters.behaviour import MotionLibrary, approach, classify_clip, walk_by, watch
from bifold_demo.characters.character import Character
from bifold_demo.characters.model import AnimationClip
from bifold_demo.characters.performer import Performer

CHARACTERS = Path(os.environ.get("BIFOLD_CHARACTERS_TEST_DIR", "/nonexistent")).expanduser()
CAMERA = (0.0, 1.6, 6.0)
SPOT = (0.0, 0.0, 2.1)
BODY = ("side_steps", "lean_sideways", "lean_forward", "closer", "wave")  # moves that keep the face on the camera
BODY_METRES = [cp.U, cp.V, cp.HIPS_U, cp.HIPS_V, cp.HEAD_X, cp.HEAD_Y, cp.HEAD_Z]
FEET = [cp.LEFT_U, cp.LEFT_V, cp.LEFT_LIFT, cp.RIGHT_U, cp.RIGHT_V, cp.RIGHT_LIFT]
DEGREES = [cp.PITCH, cp.ROLL, cp.HEAD_ROLL, cp.HEAD_YAW, cp.HEAD_PITCH]
WEIGHTS = [cp.WAVE, cp.GESTURE, cp.FACE0, cp.FACE1, cp.FACE2, cp.FACE3]


@pytest.fixture(scope="module")
def biped():
    return synthetic_biped()


def _playing(model, kind, seed=1, library=None, behaviour=None, start=10.0, **options):
    """A person at SPOT watching the camera who does one move of ``kind`` from ``start``."""
    behaviour = behaviour or watch(SPOT, CAMERA, library=library, rng=seed)
    move = cp.make_move(kind, model, seed, library=library, **options)
    move.start, move.heading = start, behaviour(start).heading
    play = cp.CameraPlay(behaviour, [move], CAMERA, 360.0)
    person = Character(model, behaviour=play, name="p")
    person.pose_adjusters.append(cp.PlayPose(play, model))
    return person, play, move


def _world(person):
    return person.matrices()[0] @ person.pose


def _face(person, model):
    """Where the face points (world) and the direction from the eyes to the camera."""
    world = _world(person)
    head = model.skeleton.index("Bip01 Head")
    forward = world[head, :3, :3] @ (model.skeleton.rest_global[head, :3, :3].T @ (0.0, 0.0, 1.0))
    eye = world[head, :3, :3] @ (cp.figure(model).eye - model.skeleton.rest_global[head, :3, 3]) + world[head, :3, 3]
    return forward, np.asarray(CAMERA) - eye


def _angles(a, b):
    """(yaw, pitch) of direction ``a`` minus those of ``b``, degrees."""
    yaw = math.degrees(math.atan2(a[0], a[2]) - math.atan2(b[0], b[2]))
    pitch = math.degrees(math.asin(a[1] / np.linalg.norm(a)) - math.asin(b[1] / np.linalg.norm(b)))
    return (yaw + 180.0) % 360.0 - 180.0, pitch


# -- the moves -----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", cp.MOVES)
def test_every_move_starts_and_ends_in_the_watching_pose(biped, kind):
    weights = [k for k in range(len(cp.CHANNELS)) if k not in cp._PHASES]
    for seed in range(6):
        for playful in (0.35, 1.0, 1.45):
            move = cp.make_move(kind, biped, seed, playful=playful)
            table = move.table
            assert move.kind == kind and 1.5 < move.duration < 14.0
            assert np.all(table[0, weights] == 0.0) and np.all(table[-1, weights] == 0.0)
            assert np.all(np.abs(table[:3, weights]) < 0.02) and np.all(np.abs(table[-3:, weights]) < 0.02)  # eased
            speed = np.abs(np.diff(table, axis=0)).max(axis=0) * cp.RATE  # no jumps: per second, at most
            assert np.all(speed[BODY_METRES] < 1.2) and np.all(speed[FEET] < 3.0)  # a foot swings faster
            assert np.all(speed[DEGREES] < 150.0) and np.all(speed[WEIGHTS] < 10.0)  # a face changes in 0.2 s
            assert set(move.faces) <= set(faces.EXPRESSIONS) and move.gesture in (None, *cp.GESTURES)


def test_the_moves_have_natural_sizes(biped):
    s = cp.figure(biped).scale
    for seed in range(8):
        side = cp.make_move("side_steps", biped, seed).table
        assert 0.2 * s < np.abs(side[:, cp.U]).max() < 0.55 * s and np.abs(side[:, cp.V]).max() == 0.0
        closer = cp.make_move("closer", biped, seed).table
        assert 0.35 * s <= closer[:, cp.V].max() <= 0.9 * s + 1e-9 and np.abs(closer[:, cp.U]).max() == 0.0
        lean = cp.make_move("lean_forward", biped, seed).table
        assert 14.0 <= lean[:, cp.PITCH].max() <= 37.0 and lean[:, cp.HIPS_V].min() < -0.02  # the hips go back
        sideways = cp.make_move("lean_sideways", biped, seed).table
        assert 9.0 < np.abs(sideways[:, cp.ROLL]).max() < 26.0
        assert sideways[:, cp.ROLL].max() > 5.0 and sideways[:, cp.ROLL].min() < -5.0  # one way and the other
        assert np.all(sideways[:, cp.HIPS_U] * sideways[:, cp.ROLL] >= 0.0)  # the hips shift away from the tilt
        for table in (side, closer):  # one foot at a time
            assert not np.any((table[:, cp.LEFT_LIFT] > 1e-6) & (table[:, cp.RIGHT_LIFT] > 1e-6))
            assert 0.02 < table[:, [cp.LEFT_LIFT, cp.RIGHT_LIFT]].max() < 0.08  # a low arc
    shy = [cp.make_move("side_steps", biped, k, playful=0.3).table[:, cp.U] for k in range(8)]
    playful = [cp.make_move("side_steps", biped, k, playful=1.5).table[:, cp.U] for k in range(8)]
    assert np.mean([np.ptp(u) for u in shy]) < np.mean([np.ptp(u) for u in playful])


def test_a_foot_moves_only_while_it_is_lifted(biped):
    for kind in ("side_steps", "closer"):
        for seed in range(5):
            table = cp.make_move(kind, biped, seed).table
            for u, v, lift in cp._FEET:
                moving = np.abs(np.diff(table[:, [u, v]], axis=0)).max(axis=1) > 1e-9
                lifted = (table[1:, lift] > 0.0) | (table[:-1, lift] > 0.0)
                assert not np.any(moving & ~lifted)


# -- posing them -----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["side_steps", "closer", "lean_sideways", "lean_forward"])
def test_planted_feet_stay_where_they_are(biped, kind):
    sk = biped.skeleton
    feet = [sk.index("Bip01 L Foot"), sk.index("Bip01 R Foot")]
    for seed in (1, 2):
        person, play, move = _playing(biped, kind, seed)
        person.update(move.start - 1.0)
        rest = _world(person)[feet, :3, 3].copy()
        planted = [None, None]
        for t in np.arange(move.start, move.end + 0.5, 1 / 60):
            person.update(float(t))
            found = play.sample(float(t))
            v = found[1] if found else [0.0] * len(cp.CHANNELS)
            for j, (cu, cv, cl) in enumerate(cp._FEET):
                where = _world(person)[feet[j], :3, 3]
                if v[cl] > 0.0:
                    planted[j] = None
                    continue
                if planted[j] is None:
                    planted[j] = where.copy()
                assert np.linalg.norm(where - planted[j]) < 0.01  # within 1 cm while planted
                expected = rest[j] + np.array([v[cu], 0.0, v[cv]]) @ _turn(move.heading)  # where the plan puts it
                assert np.linalg.norm((where - expected)[[0, 2]]) < 0.01 and abs(where[1] - rest[j][1]) < 0.01
        person.update(move.end + 1.0)
        np.testing.assert_allclose(_world(person)[feet, :3, 3], rest, atol=1e-9)


def _turn(heading):
    """Rows: the person's left and forward on the floor (x, y, z) for ``heading``."""
    h = math.radians(heading)
    return np.array([[math.cos(h), 0.0, -math.sin(h)], [0.0, 1.0, 0.0], [math.sin(h), 0.0, math.cos(h)]])


@pytest.mark.parametrize("kind", BODY)
def test_the_face_keeps_on_the_camera(biped, kind):
    for seed in (1, 4):
        person, play, move = _playing(biped, kind, seed)
        twin = Character(biped, behaviour=play.inner, name="twin")
        for t in np.arange(move.start, move.end, 1 / 15):
            v = play.sample(float(t))[1]
            if v[cp.HEAD_YAW] or v[cp.HEAD_PITCH]:  # a face is made, wobbling the head on purpose
                continue
            person.update(float(t))
            twin.update(float(t))
            yaw, pitch = _angles(*_face(person, biped))
            yaw0, pitch0 = _angles(*_face(twin, biped))
            assert abs(yaw) < 3.0 and abs(yaw - yaw0) < 2.5 and abs(pitch - pitch0) < 3.0, (t, yaw, pitch, pitch0)


def _hand_jumps(model, kind, seed, library=None, behaviour=None, **options):
    """``(move, metres)``: the largest jump of a hand from one 60 fps frame to the next during a move of ``kind``,
    beyond the jump of the same hand of the twin who does not play."""
    person, play, move = _playing(model, kind, seed, library=library, behaviour=behaviour, **options)
    twin = Character(model, behaviour=play.inner, name="twin")
    hands = [model.skeleton.index(f"Bip01 {s} Hand") for s in "LR"]
    worst, last = 0.0, None
    for t in np.arange(move.start - 0.1, move.end + 0.1, 1 / 60):
        person.update(float(t))
        twin.update(float(t))
        now = _world(person)[hands, :3, 3], _world(twin)[hands, :3, 3]
        if last is not None:
            jump = np.linalg.norm(now[0] - last[0], axis=1) - np.linalg.norm(now[1] - last[1], axis=1)
            worst = max(worst, float(jump.max()))
        last = now
    return move, worst


def test_the_arms_move_smoothly(biped):
    """The hands come up to the face and go back without the arms swinging out or flipping over, and they do not
    jump when a gesture starts or ends during a lean (the arms keep hanging from the bent torso all along)."""
    for gesture in cp.GESTURES:
        assert _hand_jumps(biped, "faces", 0, gesture=gesture)[1] < 0.045  # 2.7 m/s at most
    leans = [_hand_jumps(biped, "lean_forward", seed, playful=1.4) for seed in range(4)]
    assert any(move.gesture for move, _ in leans) and max(jump for _, jump in leans) < 0.045
    assert _hand_jumps(biped, "wave", 1)[1] < 0.045


def test_moves_are_a_pure_function_of_time_and_repeat_every_period(biped):
    person, play, move = _playing(biped, "closer", 3)
    times = np.linspace(move.start - 1, move.end + 1, 40)
    states = [play(float(t)) for t in times]
    for t, state in zip(times[::-1], states[::-1]):  # backwards, then a period later: the same
        assert play(float(t)) == state and play(float(t) + 360.0).position == pytest.approx(state.position)
    track = play.positions(times)
    np.testing.assert_allclose(track, [(s.position[0], s.position[2]) for s in states], atol=1e-9)
    assert max(np.hypot(*(track - track[0]).T)) > 0.3  # it does go closer ...
    assert play(move.start - 0.5) == play.inner(move.start - 0.5)  # ... and outside the moves nothing changes
    person.update(move.start - 0.5)
    alone = Character(biped, behaviour=play.inner, name="alone")
    alone.update(move.start - 0.5)
    np.testing.assert_array_equal(person.pose, alone.pose)


def test_the_wave_uses_the_clip_when_the_library_has_one(biped):
    clip = _wave_clip(biped)
    library = MotionLibrary({clip.name: clip, "m_idle_neutral_01": _still(biped, "m_idle_neutral_01")})
    assert classify_clip("m_wave_01") == "wave" and library.names("wave") == ["m_wave_01"]
    assert "m_wave_01" not in [c.name for c in library.choices("idle")]
    behaviour = watch(SPOT, CAMERA, library=library, rng=1, roles=("idle", "listen"))
    assert all(e.clip is None or e.clip.name == "m_idle_neutral_01" for e in behaviour._entries)  # never an idle
    hands = [biped.skeleton.index(f"Bip01 {s} Hand") for s in "LR"]
    for seed in range(4):
        person, _, move = _playing(biped, "wave", seed, library=library, behaviour=behaviour)
        assert move.wave.clip is clip and move.wave.hands in ("right", "left")
        person.update(move.start + 1.6)
        left, right = person.pose[hands, 1, 3]
        raised, down = (right, left) if move.wave.hands == "right" else (left, right)
        assert raised > 1.5 and down < 1.2  # the clip's wave, mirrored for the left hand
    person, _, move = _playing(biped, "wave", 2, library=None)
    assert move.wave.clip is None  # without clips: the procedural wave, the hand beside the head
    person.update(move.start + 0.25 * move.duration + 0.5)
    side = 0 if move.wave.hands == "left" else 1
    assert person.pose[hands[side], 1, 3] > 1.35 and person.pose[hands[1 - side], 1, 3] < 1.2
    phases = []
    for t in np.arange(move.start + 1.0, move.start + 2.0, 1 / 30):  # the forearm swings to and fro
        person.update(float(t))
        phases.append(person.pose[hands[side], 0, 3])
    assert np.ptp(phases) > 0.06 and np.sum(np.diff(np.sign(np.diff(phases))) != 0) >= 3


def test_someone_holding_a_sign_waves_with_the_left_hand(biped, tmp_path):
    hands = [biped.skeleton.index(f"Bip01 {s} Hand") for s in "LR"]
    for library in (None, MotionLibrary({"m_wave_01": _wave_clip(biped)})):
        seed = next(k for k in range(20) if cp.make_move("wave", biped, k, library=library).wave.hands == "right")
        person, _, move = _playing(biped, "wave", seed, library=library)
        accessories.dress(person, "sign", "welcome", t=0.0)
        person.update(move.start + 0.45 * move.duration)
        assert person.pose[hands[0], 1, 3] > 1.35  # the left hand waves; the right one holds the sign
        assert cp.doing(person, move.start + 1.0) == "waves at the camera"


def test_faces_go_to_the_face_rig(biped, monkeypatch):
    calls = []
    monkeypatch.setattr(faces, "face_rig", lambda model: "rig")
    monkeypatch.setattr(faces, "apply_face", lambda local, rig, weights: calls.append((rig, dict(weights))))
    person, _, move = _playing(biped, "faces", 5, gesture="ears")
    assert move.faces and move.gesture == "ears"
    for t in np.arange(move.start - 0.5, move.end + 0.5, 0.1):
        person.update(float(t))
    assert calls and all(rig == "rig" for rig, _ in calls)
    assert all(w and set(w) <= set(move.faces) and all(0.0 < x <= 1.0 for x in w.values()) for _, w in calls)
    assert max(max(w.values()) for _, w in calls) > 0.75
    hands = [biped.skeleton.index(f"Bip01 {s} Hand") for s in "LR"]
    head = biped.skeleton.index("Bip01 Head")
    held = move.start + np.flatnonzero(move.table[:, cp.GESTURE] > 0.999)[0] / cp.RATE + 0.2
    person.update(float(held))
    for k in hands:  # thumbs at the temples: the hands beside the head
        assert np.linalg.norm(person.pose[k, :3, 3] - person.pose[head, :3, 3]) < 0.25


def test_no_play_during_a_yoga_act(biped):
    person, play, move = _playing(biped, "lean_forward", 2)
    performer = Performer(play)
    person.behaviour = performer
    frames = 31
    clip = AnimationClip("yoga_test", 10.0, ["Bip01"], np.tile((0.0, 0.0, 0.0, 1.0), (frames, 1, 1)),
                         np.zeros((frames, 1, 3)), metadata={"pose": "cow", "hold": (0.5, 2.5), "view": 0.0})
    start = move.start + 1.5  # in the middle of the bend
    person.update(start)
    act = performer.perform(clip, start, heading=performer(start).heading)
    middle = act.clip_start + 1.0
    assert play.posed(person, middle) is None and cp.doing(person, middle) == ""
    alone = Character(biped, behaviour=Performer(play.inner), name="alone")
    alone.behaviour.perform(clip, start, heading=performer(start).heading)
    for who in (person, alone):
        who.update(middle)
    np.testing.assert_allclose(person.pose, alone.pose, atol=1e-9)  # the yoga clip alone
    later = act.end + 1.0  # back: the move goes on where it was (the behaviour's time stood still)
    base = performer._base_time(later)
    assert base == pytest.approx(start + 1.0) and base < move.end
    assert play.posed(person, later)[0] is move and base < later
    assert play.posed(person, later)[1] == play.sample(base)[1]


# -- planning --------------------------------------------------------------------------------------------------------


def _audience(model, seed=1):
    """A small audience in front of the camera: three who stay, two visitors walking up between them, a walker."""
    spots = [(CAMERA[0] + 3.9 * math.sin(math.radians(a)), CAMERA[2] - 3.9 * math.cos(math.radians(a))) for a in
             (-36, -22, -7, 7, 22)]
    stays = [watch((x, 0.0, z), CAMERA, library=None, rng=seed + k) for k, (x, z) in enumerate(spots[::2])]
    visits = [approach((x * 0.2, 0.0, -14.0), (x, 0.0, z), CAMERA, library=None, rng=seed + 10 + k, bow=0.0,
                       stop_distance=(3.5, 5.0), watch=CAMERA, idle=(15.0, 30.0), period=360.0)
              for k, (x, z) in enumerate(spots[1::2])]
    walker = walk_by([(-6.0, 0.0, 0.8), (6.0, 0.0, 0.8)], library=None, camera=CAMERA, rng=seed + 20, period=360.0)
    players = [cp.Player(b, cp.figure(model), age=age) for b, age in zip(stays + visits, ("adult", "child", "adult",
                                                                                          "adult", "older"))]
    return players, [walker]


@pytest.fixture(scope="module")
def planned(biped):
    players, others = _audience(biped)
    plans = cp.plan_camera_play(players, CAMERA, others=others, period=360.0, rng=7)
    return players, others, plans


def test_planning_is_deterministic(biped, planned):
    _, _, plans = planned
    again = cp.plan_camera_play(*_audience(biped)[:1], CAMERA, others=_audience(biped)[1], period=360.0, rng=7)
    assert [[(m.kind, m.start) for m in plan] for plan in again] == [[(m.kind, m.start) for m in plan]
                                                                     for plan in plans]
    other = cp.plan_camera_play(*_audience(biped)[:1], CAMERA, others=_audience(biped)[1], period=360.0, rng=8)
    assert [[m.start for m in plan] for plan in other] != [[m.start for m in plan] for plan in plans]


def test_people_play_only_while_they_watch_and_every_move_happens(planned):
    players, _, plans = planned
    kinds = {m.kind for plan in plans for m in plan}
    assert kinds == set(cp.MOVES)
    for player, plan in zip(players, plans):
        assert plan and all(0.0 <= m.start < m.end <= 360.0 for m in plan)
        assert all(a.end <= b.start for a, b in itertools.pairwise(plan))  # one at a time
        for move in plan:  # with time to spare before and after (the head turns to the camera and back meanwhile)
            assert all(player.behaviour.watching(float(t)) for t in np.linspace(move.start - 1.0, move.end + 1.0, 40))
            assert move.heading == pytest.approx(player.behaviour(move.start).heading)
        windows = cp.watch_windows(player.behaviour, 360.0)
        watched = sum(b - a for a, b in windows)
        share = sum(m.duration for m in plan) / watched
        assert 0.03 < share < 0.65  # they mostly just watch
        if len(windows) > 1:  # a visitor: one to three moves per visit
            per_visit = [sum(a <= m.start < b for m in plan) for a, b in windows]
            assert max(per_visit) <= 3 and sum(n >= 1 for n in per_visit) >= 0.6 * len(windows)
    stays = [sum(m.duration for m in plan) for plan in plans[:3]]
    assert 0.15 * 3 * 359 < sum(stays) < 0.55 * 3 * 359  # about a third of the time, all together


def test_moves_keep_everyone_apart(planned):
    players, others, plans = planned
    times = np.arange(0.0, 360.0, 0.1)
    plays = [cp.CameraPlay(p.behaviour, plan, CAMERA, 360.0) for p, plan in zip(players, plans)]
    before = np.stack([p.behaviour.positions(times) for p in players] + [b.positions(times) for b in others])
    after = np.stack([p.positions(times) for p in plays] + [b.positions(times) for b in others])
    assert np.nanmax(np.hypot(*(after - before).transpose(2, 0, 1))) > 0.3  # some do move about
    for i in range(len(players)):
        for j in range(len(after)):
            if i != j:
                d = np.hypot(*(after[i] - after[j]).T)
                d0 = np.hypot(*(before[i] - before[j]).T)
                d, d0 = np.where(np.isnan(d), np.inf, d), np.where(np.isnan(d0), np.inf, d0)
                assert np.all((d >= cp.CLEARANCE) | (d >= d0 - 1e-3)), (i, j, d.min())


def _reaching(model, plays, people, others, rate=5.0):
    """``(metres, metres)`` for every sample (``rate`` a second) of every move of ``plays`` (the plays of ``people``,
    with ``others`` the behaviours of everyone else): how near an elbow, a hand or the head of the player comes to
    someone else's body axis, and how near it would come where the player stands if they did not play (the arms as
    the clips have them)."""
    times = np.arange(0.0, 360.0, 0.1)
    tracks = [p.positions(times) for p in plays] + [b.positions(times) for b in others]
    near = []
    for i, (play, person) in enumerate(zip(plays, people)):
        sk = person.skinned_model.skeleton
        bones = [sk.index(f"Bip01 {n}") for n in ("L Forearm", "L Hand", "R Forearm", "R Hand", "Head")]
        twin = Character(person.skinned_model, behaviour=play.inner, name="twin")
        for move in play.moves:
            for t in np.arange(move.start, move.end, 1.0 / rate):
                person.update(float(t))
                twin.update(float(t))
                k = round(t / 0.1)
                shift = (play.positions(times[k:k + 1]) - play.inner.positions(times[k:k + 1]))[0]
                points, still = _world(person)[bones][:, [0, 2], 3], _world(twin)[bones][:, [0, 2], 3] + shift
                for j, track in enumerate(tracks):
                    if j != i and not np.isnan(track[k]).any():
                        d = np.hypot(*(points - track[k]).T)
                        d0 = np.hypot(*(still - track[k]).T)
                        near += [(a, b) for a, b in zip(d, d0)]
    return np.array(near)


def test_moves_keep_the_arms_off_the_others(biped, planned):
    """Waving, making faces and leaning are kept clear of the others too, not only travelling: the elbows, hands and
    head stay off everybody's body (or no nearer it than they are anyway)."""
    players, others, plans = planned
    plays = [cp.CameraPlay(p.behaviour, plan, CAMERA, 360.0) for p, plan in zip(players, plans)]
    people = []
    for play in plays:
        people.append(Character(biped, behaviour=play, name="p"))
        people[-1].pose_adjusters.append(cp.PlayPose(play, biped))
    near = _reaching(biped, plays, people, others)
    assert len(near) and np.all((near[:, 0] >= 0.3) | (near[:, 0] >= near[:, 1] - 0.02)), near[:, 0].min()


def test_play_along_skips_who_cannot_play(biped):
    from test_behaviour import synthetic_model

    from bifold_demo.scene import Quad

    cutout = Quad(height=1.7)
    other = Character(synthetic_model(), behaviour=watch(SPOT, CAMERA, library=None), name="other")
    person = Character(biped, behaviour=watch((1.0, 0.0, 2.0), CAMERA, library=None, rng=3), name="person")
    inner = person.behaviour
    plays = cp.play_along([cutout, other, person], CAMERA, period=360.0, rng=1)
    assert len(plays) == 1 and person.behaviour is plays[0] and plays[0].inner is inner
    assert len(person.pose_adjusters) == 1 and isinstance(other.behaviour, type(inner)) and not other.pose_adjusters
    assert plays[0].duration == inner.duration and plays[0].watching(3.0)  # the rest is the behaviour's
    found = next(m for m in plays[0].moves)
    assert cp.doing(person, found.start + 0.5) == cp.DOING[found.kind] and cp.doing(other, 1.0) == ""


def test_other_behaviours_and_bad_names(biped):
    from bifold_demo.scenes.fair import Raised

    raised = Raised(watch(SPOT, CAMERA, library=None, rng=2), 1.0)  # on a stage: the moves keep them up there
    move = cp.make_move("closer", biped, 2)
    move.start, move.heading = 10.0, raised(10.0).heading
    play = cp.CameraPlay(raised, [move], CAMERA, 360.0)
    assert all(play(float(t)).position[1] == pytest.approx(1.0) for t in np.linspace(move.start, move.end, 20))
    assert play(move.start + 2.0).position != raised(move.start + 2.0).position
    drifting = watch(SPOT, CAMERA, library=None, rng=2, roles=("idle", "listen"))  # a loop that does not fit
    drifting.watching = lambda t: t % 50.0 < 30.0
    person = Character(biped, behaviour=drifting, name="p")
    assert not cp.play_along([person], CAMERA, period=37.0) and person.behaviour is drifting
    with pytest.raises(ValueError, match="unknown"):
        cp.plan_camera_play(_audience(biped)[0], CAMERA, moves=("wave", "jumping"))


def test_playing_is_cheap(biped):
    budget = {}
    for kind in cp.MOVES:
        person, play, move = _playing(biped, kind, 2, gesture="mouth") if kind == "faces" else \
            _playing(biped, kind, 2)
        alone = Character(biped, behaviour=play.inner, name="alone")
        times = np.arange(move.start, move.end, 1 / 30)
        best = [math.inf, math.inf]
        for _ in range(3):
            for k, who in enumerate((alone, person)):
                t0 = time.perf_counter()
                for t in times:
                    who.update(float(t))
                best[k] = min(best[k], (time.perf_counter() - t0) / len(times))
        budget[kind] = best[1] - best[0]
    assert max(budget.values()) < 0.8e-3, budget  # well under a millisecond per person and frame, while playing
    person, play, move = _playing(biped, "wave", 2)
    t0 = time.perf_counter()
    for t in np.arange(0.0, 5.0, 1 / 30):  # before the move: next to nothing
        play(float(t))
        cp.PlayPose(play, biped).play.posed(person, float(t))
    assert (time.perf_counter() - t0) / 150 < 0.2e-3


# -- helpers ---------------------------------------------------------------------------------------------------------


def _still(model, name):
    """A clip holding ``model``'s rest pose (all its bones), ``name`` d."""
    sk = model.skeleton
    q = np.tile((0.0, 0.0, 0.0, 1.0), (2, len(sk.names), 1))
    t = np.repeat(sk.rest_local[None, :, :3, 3], 2, axis=0)
    return AnimationClip(name, 10.0, list(sk.names), q, t, root="Bip01", metadata={"kind": "idle"})


def _wave_clip(model):
    """A synthetic ``m_wave_01``: the right arm swings up over the head and down again in 3 s."""
    clip = _still(model, "m_wave_01")
    frames = 31
    q = np.tile((0.0, 0.0, 0.0, 1.0), (frames, len(model.skeleton.names), 1))
    up = np.sin(np.pi * np.clip((np.arange(frames) / 10.0 - 0.4) / 2.2, 0.0, 1.0)) ** 2
    angle = np.radians(-125.0 * up) / 2
    q[:, model.skeleton.index("Bip01 R UpperArm")] = np.stack([0 * angle, 0 * angle, np.sin(angle), np.cos(angle)], 1)
    t = np.repeat(clip.translations[:1], frames, axis=0)
    return AnimationClip("m_wave_01", 10.0, clip.bone_names, q, t, root="Bip01", metadata={"kind": "gesture"})


# -- real avatars ----------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real():
    from bifold_demo.characters.library import CharacterLibrary

    if not (CHARACTERS / "avatars").is_dir():
        pytest.skip(f"no prepared characters in {CHARACTERS} (set BIFOLD_CHARACTERS_TEST_DIR)")
    library = CharacterLibrary(CHARACTERS)
    names = sorted(library.available())
    if not names or not library.roles().get("idle"):
        pytest.skip("no prepared avatars with idle clips")
    return library


@pytest.mark.parametrize("kind", BODY)
def test_real_avatars_keep_their_feet_and_face(real, kind):
    names = sorted(real.available())
    motions = real.motions()
    for name in (names[0], names[len(names) // 2]):
        model = real.load(name)
        sex = real.available()[name].get("sex", "male")
        behaviour = watch(SPOT, CAMERA, library=motions, rng=3, style={"sex": sex})
        person, play, move = _playing(model, kind, 3, library=motions, behaviour=behaviour)
        twin = Character(model, behaviour=behaviour, name="twin")
        feet = [model.skeleton.index(f"Bip01 {s} Foot") for s in "LR"]
        planted = [None, None]
        for t in np.arange(move.start, move.end, 1 / 30):
            person.update(float(t))
            twin.update(float(t))
            v = play.sample(float(t))[1]
            for j, (_, _, lift) in enumerate(cp._FEET):
                where = _world(person)[feet[j], :3, 3]
                clip_drift = _world(twin)[feet[j], :3, 3]  # the idle clip's own feet move a little
                if v[lift] > 0.0:
                    planted[j] = None
                elif planted[j] is None:
                    planted[j] = (where - clip_drift).copy()
                else:
                    assert np.linalg.norm(where - clip_drift - planted[j]) < 0.01
            if not (v[cp.HEAD_YAW] or v[cp.HEAD_PITCH]):
                (yaw, pitch), (yaw0, pitch0) = _angles(*_face(person, model)), _angles(*_face(twin, model))
                assert abs(yaw - yaw0) < 3.0 and abs(pitch - pitch0) < 4.0, (name, t, yaw - yaw0, pitch - pitch0)


def test_real_wave_clips_wave(real):
    motions = real.motions()
    waves = motions.choices("wave", "male") + motions.choices("wave", "female")
    if not waves:
        pytest.skip("no wave clips prepared (bifold-characters fetch downloads them)")
    name = min(real.available())
    model = real.load(name)
    fig = cp.figure(model)
    infos = [cp.wave_clip(c, fig) for c in waves]
    assert all(info is not None and 2.0 < info.span <= info.clip.duration for info in infos)
    assert any(not info.both for info in infos)  # the _01 clips wave with the right hand
    hands = [model.skeleton.index(f"Bip01 {s} Hand") for s in "LR"]
    for info in infos:
        for hands_used in ("both",) if info.both else ("right", "left"):
            move = cp.Move("wave", _wave_table(fig, info), wave=cp.Wave(info.clip, hands_used))
            behaviour = watch(SPOT, CAMERA, library=motions, rng=1)
            move.start, move.heading = 10.0, behaviour(10.0).heading
            play = cp.CameraPlay(behaviour, [move], CAMERA, 360.0)
            person = Character(model, behaviour=play, name="p")
            person.pose_adjusters.append(cp.PlayPose(play, model))
            heights = []
            for t in np.arange(move.start, move.end, 0.1):
                person.update(float(t))
                heights.append(person.pose[hands, 1, 3])
            heights = np.array(heights)
            k = 0 if hands_used == "left" else 1
            assert heights[:, k].max() > fig.height * 0.85  # the waving hand goes up above the shoulder ...
            if not info.both:
                assert heights[:, 1 - k].max() < fig.height * 0.6  # ... the other one does not (mirrored on the left)


def test_real_arms_move_smoothly(real):
    names = sorted(real.available())
    motions = real.motions()
    for name in (names[0], names[len(names) // 2]):
        model = real.load(name)
        sex = real.available()[name].get("sex", "male")
        behaviour = watch(SPOT, CAMERA, library=motions, rng=3, style={"sex": sex})
        jumps = [_hand_jumps(model, "faces", 2, behaviour=behaviour, gesture=g) for g in cp.GESTURES]
        jumps += [_hand_jumps(model, "lean_forward", seed, behaviour=behaviour, playful=1.4) for seed in (1, 3)]
        jumps.append(_hand_jumps(model, "wave", 2, behaviour=behaviour))  # the procedural wave
        assert max(jump for _, jump in jumps) < 0.065, (name, [round(jump, 3) for _, jump in jumps])


def _wave_table(fig, info):
    script = cp._Script()
    script.go(cp.WAVE, 0.0, 0.35, 1.0)
    script.run(cp.WAVE_TIME, 0.0, info.span, 1.0)
    script.go(cp.WAVE, info.span - 0.55, info.span, 0.0)
    return script.table(fig)


def test_the_fair_audience_plays_and_keeps_apart(real, monkeypatch):
    from bifold_demo.characters.library import CharacterLibrary
    from bifold_demo.scenes import fair

    monkeypatch.setenv("BIFOLD_CHARACTERS", str(CHARACTERS))
    monkeypatch.setattr(CharacterLibrary, "shared", classmethod(lambda cls, cache_dir=None: real))
    scene = fair.build_scene()
    people = {n: o for n, o in scene.objects.items() if isinstance(o, Character)}
    if not people:
        pytest.skip("the fair could not set up its 3D people")
    players = {n: o for n, o in people.items() if isinstance(o.behaviour, cp.CameraPlay)}
    assert sorted(players) == sorted(n for n in people if n.startswith(("audience", "approach")))
    assert {m.kind for o in players.values() for m in o.behaviour.moves} == set(fair.CAMERA_PLAY_MOVES)
    times = np.arange(0.0, fair.LOOP_PERIOD, 0.1)
    tracks = {n: o.behaviour.positions(times) for n, o in people.items()}
    bare = {n: (o.behaviour.inner if n in players else o.behaviour).positions(times) for n, o in people.items()}
    for n in players:
        for m in people:
            if m != n:
                d = np.hypot(*(tracks[n] - tracks[m]).T)
                d0 = np.hypot(*(bare[n] - bare[m]).T)
                d, d0 = np.where(np.isnan(d), np.inf, d), np.where(np.isnan(d0), np.inf, d0)
                assert np.all((d >= cp.CLEARANCE) | (d >= d0 - 1e-3)), (n, m, d.min())
    for n in players:  # the feet stay in the picture
        spot = bare[n][~np.isnan(bare[n][:, 0])]
        moved = tracks[n][~np.isnan(tracks[n][:, 0])]
        assert all(fair.feet_in_view(p) for p in moved[::5]) or not all(fair.feet_in_view(p) for p in spot[::50])
    names = sorted(players)
    near = _reaching(None, [players[n].behaviour for n in names], [players[n] for n in names],
                     [o.behaviour for n, o in people.items() if n not in players], rate=4.0)
    assert np.all((near[:, 0] >= 0.3) | (near[:, 0] >= near[:, 1] - 0.02)), near[:, 0].min()  # no arm in anyone
    monkeypatch.setattr(fair, "CAMERA_PLAY", False)
    quiet = fair.build_scene()
    assert not any(isinstance(getattr(o, "behaviour", None), cp.CameraPlay) for o in quiet.objects.values())
