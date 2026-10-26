"""Selection mode: keys, people on screen, Tab order, losing the selected person, and yoga poses on request."""

import math
import time
from concurrent.futures import wait

import numpy as np
import pytest
from test_behaviour import (
    BONES,
    CAMERA,
    CHARACTERS,
    _angle_to_camera,
    synthetic_clip,
    synthetic_clips,
    synthetic_model,
)
from test_yoga import synthetic_biped

from bifold_demo.characters import yoga
from bifold_demo.characters.behaviour import MotionLibrary, stand, turn_angle, walk_by
from bifold_demo.characters.character import Character
from bifold_demo.characters.model import AnimationClip, CharacterState, Layer
from bifold_demo.characters.performer import (
    LEAD,
    Performer,
    _rect_distance,
    choose_place,
    floor_height,
    floors,
    footprint,
    obstacles,
    place,
)
from bifold_demo.characters.yoga_clips import YogaClips, cache_path, model_key
from bifold_demo.display.window import BACKSPACE, BACKTAB, DOWN, LEFT, RIGHT, TAB, UP, key_code
from bifold_demo.scene import Box, Camera, Plane, Scene
from bifold_demo.selection import STALE, Selection

SIZE = (640, 360)


def test_key_codes_from_every_backend():
    assert key_code(-1) is None
    assert key_code(ord("A")) == ord("a") and key_code(ord("1")) == ord("1") and key_code(27) == 27
    assert key_code(9) == TAB
    assert key_code(25) == BACKTAB  # macOS: Shift+Tab
    assert key_code(0xFE20 | 1 << 16) == BACKTAB  # GTK ISO_Left_Tab with Shift held
    assert key_code(11) == BACKTAB  # Qt: Shift+Tab
    assert key_code(0x250000) == LEFT and key_code(0x270000) == RIGHT  # Windows arrows
    assert key_code(0xF702) == LEFT and key_code(0xFF53) == RIGHT  # macOS, Qt and GTK (X11 keysyms)
    assert key_code(0xFF51) == LEFT  # GTK: not "q" (which would quit)
    assert key_code(0xFF52) == UP and key_code(0xFF54 | 1 << 16) == DOWN and key_code(0xF700) == UP  # GTK, macOS
    assert key_code(0x260000) == UP and key_code(0x280000) == DOWN  # Windows
    assert key_code(27 | 0x10 << 16) == 27  # GTK Escape with NumLock on
    assert key_code(ord("y") | 0x10 << 16) == ord("y")
    assert key_code(127) == BACKSPACE and key_code(0xFF08) == BACKSPACE
    assert key_code(0x700000) is None and key_code(0xF704) is None and key_code(0xFFBE) is None  # F1 keys


# -- the performer ------------------------------------------------------------------------------------------------


def _clip(duration=8.0, hold=(2.0, 6.0), view=90.0):
    frames = int(duration * 10) + 1
    rotations = np.tile((0.0, 0.0, 0.0, 1.0), (frames, 1, 1))
    translations = np.zeros((frames, 1, 3))
    translations[:, 0, 1] = np.linspace(0.0, 1.0, frames)  # a marker of the clip time
    return AnimationClip("yoga_test", 10.0, ["Bip01"], rotations, translations,
                         metadata={"pose": "cow", "hold": hold, "view": view})


def _walker(t):
    return CharacterState((float(t), 0.0, 0.0), 90.0, [Layer(_clip(), 0.0)], head_yaw=20.0)


def test_a_pose_interrupts_the_behaviour_which_carries_on_where_it_left_off():
    performer = Performer(_walker)
    clip = _clip()
    act = performer.perform(clip, 5.0, heading=0.0)
    assert act is not None and act.turn == pytest.approx(0.6)  # 90 degrees at 150 degrees per second
    stop = 5.0 + LEAD / 2  # walking at 1 m/s, they slow to a stop over LEAD seconds
    track = [performer(t).position[0] for t in np.arange(4.9, 5.0 + LEAD, 1 / 60)]
    assert track[6] == pytest.approx(5.0) and 0.0 <= np.diff(track).min() and np.diff(track).max() <= 1 / 60 + 1e-9
    for t in (5.0 + LEAD, 7.0, 12.0):
        assert performer(t).position == pytest.approx((stop, 0.0, 0.0))  # stays put while doing it
    assert performer(5.6).heading == pytest.approx(0.0) and performer(5.6).head_yaw == pytest.approx(0.0)
    assert performer(7.6).layers[-1].time == pytest.approx(2.0)  # the clip runs after the turn
    assert act.end == pytest.approx(5.0 + 0.6 + 8.0)
    after = act.end + 2.0
    # walking on from where it stopped (the behaviour's time ran on into the stop), starting from a standstill
    # (which loses LEAD / 2 seconds)
    assert performer(after).position == pytest.approx((stop + 2.0 - LEAD / 2, 0.0, 0.0))
    assert performer(after).heading == pytest.approx(90.0)
    times = [4.0, 6.0, 20.0]
    assert np.allclose(performer.positions(times), [[4.0, 0.0], [stop, 0.0], [20.0 - 8.6, 0.0]])


def test_stopping_leaves_the_pose_the_quickest_way():
    clip = _clip()
    holding = Performer(_walker)
    holding.perform(clip, 0.0, heading=90.0)  # no turn needed: 0.6 s anyway
    assert holding.stop(0.6 + 4.0)  # in the hold (2..6): out of it at once
    act = holding.acts[-1]
    assert act.mode == "skip" and act.end == pytest.approx(4.6 + (8.0 - 6.0))
    entering = Performer(_walker)
    entering.perform(clip, 0.0, heading=90.0)
    entering.stop(0.6 + 0.5)  # just started: back the way it came
    act = entering.acts[-1]
    assert act.mode == "back" and act.end == pytest.approx(1.1 + 0.5)
    assert act.clip_time(1.1 + 0.25) == pytest.approx(0.25)
    nearly = Performer(_walker)
    nearly.perform(_clip(hold=(2.0, 7.5)), 0.0, heading=90.0)
    nearly.stop(0.6 + 1.8)  # nearly in a hold that ends near the end: on into it and straight out
    act = nearly.acts[-1]
    assert act.mode == "through" and act.end == pytest.approx(2.4 + 0.2 + 0.5)
    assert act.clip_time(2.4 + 0.3) == pytest.approx(7.6)
    assert not Performer(_walker).stop(3.0)  # nothing to stop


def test_holding_keeps_the_pose_until_stopped_and_a_new_pose_follows_the_current_one():
    clip = _clip()
    performer = Performer(_walker)
    performer.perform(clip, 0.0, heading=90.0)
    assert performer.hold(1.0)
    act = performer.acts[-1]
    assert math.isinf(act.end) and act.phase(100.0) == "holding"
    assert 2.0 <= act.clip_time(100.0) < 6.0
    second = _clip(duration=5.0, hold=(1.0, 3.0), view=0.0)
    assert performer.perform(second, 100.0, heading=45.0) is None  # after the current one
    assert act.cut == 100.0 and performer.queued is not None
    t = act.end + 0.01
    state = performer(t)
    assert performer.acts[-1].clip is second and performer.acts[-1].start == pytest.approx(act.end)
    assert state.position == pytest.approx((LEAD / 2, 0.0, 0.0))  # where the first one was done (after stopping)
    assert performer.positions([t])[0] == pytest.approx((LEAD / 2, 0.0))


def _line(length=1.5, n=7):
    points = np.zeros((2, n, 3))
    points[:, :, 2] = np.linspace(0.0, length, n)  # a pose reaching ``length`` metres ahead of the person
    return points


def test_looking_ahead_changes_nothing_and_a_queued_pose_can_be_held():
    performer = Performer(_walker)
    performer.perform(_clip(), 0.0, heading=90.0)
    performer.perform(_clip(duration=5.0, hold=(1.0, 3.0), view=0.0), 3.0, heading=0.0)  # follows the first
    acts, queued = list(performer.acts), performer.queued
    later = performer.acts[-1].end + 3.0
    ahead, peeked = performer.positions([1.0, later]), performer.peek(later)
    assert performer.acts == acts and performer.queued == queued  # nothing started early
    assert performer.hold(4.0) and performer.queued[3]  # the same key again: hold the one that follows
    real = performer(later)
    assert np.allclose(ahead[1], np.asarray(real.position)[[0, 2]]) and np.allclose(peeked.position, real.position)
    assert performer.acts[-1].looping and math.isinf(performer.acts[-1].end)


def test_going_back_to_the_behaviour_turns_smoothly_and_steps_back_at_a_walking_pace():
    turning = Performer(lambda t: CharacterState((0.0, 0.0, -0.5 * t), 171.0 + 15.0 * t, []))  # walking a curve
    act = turning.perform(_clip(), 0.0, heading=0.0, position=(0.6, 0.0, 0.0))
    times = np.arange(act.end - 0.5, act.end + 4.0, 1 / 60)
    states = [turning(t) for t in times]
    steps = np.abs((np.diff([s.heading for s in states]) + 180) % 360 - 180)
    assert steps.max() < 6.0  # no flip when the behaviour's heading crosses the opposite direction
    speed = np.linalg.norm(np.diff([s.position for s in states], axis=0), axis=1) * 60
    assert speed.max() < 1.6  # the 0.6 m step back is no glide (the behaviour itself moves at 0.5 m/s)
    ahead = turning.positions(times)
    assert np.allclose(ahead, np.array([s.position for s in states])[:, [0, 2]])


def test_a_new_pose_breaks_off_the_walk_back():
    library = MotionLibrary(synthetic_clips())
    performer = Performer(stand((0.0, 0.0), 0.0, library=library))
    act = performer.perform(_clip(), 0.0, heading=90.0, position=(3.0, 0.0, 0.0))
    t = act.clip_end + 0.5 * act.retreat.duration  # on the way back
    here = np.asarray(performer(t).position)
    assert performer.free_at(t) == pytest.approx(t + LEAD / 2)  # once they have slowed to a stop
    planned = performer.peek(performer.free_at(t)).position  # where a place for the pose is looked for
    second = performer.perform(_clip(view=0.0), t, heading=0.0, position=(2.0, 0.0, 1.0))
    assert second is not None and act.end == t and np.allclose(performer(t).position, here)
    stop = np.asarray(second.before.position)  # they walk on a little as they slow to a stop, towards home
    assert 0.05 < here[0] - stop[0] < 0.4 and np.allclose(stop[[1, 2]], 0.0) and np.allclose(stop, planned)
    assert second.home == act.home


def test_the_place_for_a_pose_keeps_the_mat_clear():
    points, nobody, nothing = _line(), np.zeros((2, 0, 2)), np.zeros((0, 6))
    assert choose_place((0, 0, 0), [90.0, -90.0], points, nobody, nothing) == (90.0, (0.0, 0.0, 0.0))
    in_the_way = np.tile((1.4, 0.0), (2, 1, 1))  # someone where the head would go
    heading, spot = choose_place((0, 0, 0), [90.0, -90.0], points, in_the_way, nothing)
    assert heading == -90.0 and spot == (0.0, 0.0, 0.0)  # seen from the other side instead
    lying = np.zeros((2, 3, 2))
    lying[:, :, 0] = (0.1, 0.4, 0.7)  # someone doing a pose, their body right next to ours
    lying[:, :, 1] = 0.75
    heading, spot = choose_place((0, 0, 0), [0.0], points, nobody, nothing, bodies=lying)
    assert (heading, spot) != (0.0, (0.0, 0.0, 0.0))
    counter = np.array([[-3.0, 0.5, 3.0, 0.7, 0.0, 1.0]])  # a counter just in front
    heading, spot = choose_place((0, 0, 0), [0.0], points, nobody, counter)
    assert abs(heading) > 15.0 or spot != (0.0, 0.0, 0.0)


def test_the_place_for_a_pose_is_in_the_picture_and_on_one_floor():
    points, nobody, nothing = _line(), np.zeros((2, 0, 2)), np.zeros((0, 6))
    def seen(world):  # the camera sees only z <= -1
        return (world[..., 2] <= -1.0).mean(axis=(-2, -1))

    heading, spot = choose_place((0, 0, 0), [180.0], points, nobody, nothing, reach=4.0, seen=seen)
    assert seen(place(points, heading, spot)) == 1.0 and spot[2] <= -1.0
    assert math.hypot(spot[0], spot[2]) <= 1.5  # the nearest such spot
    assert choose_place((0, 0, 0), [180.0], points, nobody, nothing, reach=0.6, seen=seen)[1][2] >= -0.6
    platform = np.array([[-5.0, -5.0, 5.0, 5.0, 0.0], [0.5, -5.0, 5.0, 5.0, 0.08]])  # a booth platform at x > 0.5
    heading, spot = choose_place((0, 0, 0), [90.0], points, nobody, nothing, floor_rects=platform)
    assert place(points, heading, spot)[..., 0].max() < 0.5  # all of it off the platform


def test_the_floor_is_the_floor_carpets_and_platforms_not_the_ceiling_or_the_walls():
    scene = Scene()
    scene.add(Plane(size=(20.0, 20.0)), name="floor")
    scene.add(Plane(size=(4.0, 20.0), position=(-6.0, 0.004, 0.0)), name="carpet")
    scene.add(Plane(size=(20.0, 20.0), position=(0.0, 12.0, 0.0), rotation=(180, 0, 0)), name="ceiling")
    scene.add(Plane(size=(20.0, 12.0), position=(0.0, 6.0, -10.0), rotation=(90, 0, 0)), name="wall")
    scene.add(Box(size=(4.0, 0.08, 3.0), position=(3.0, 0.04, 0.0)), name="platform")
    scene.add(Box(size=(1.8, 1.05, 0.6), position=(3.0, 0.605, 1.0)), name="counter")  # on the platform
    scene.update(0.0, 0.0)
    rects = floors(scene)
    heights = floor_height(rects, np.array([(0.0, 0.0), (-6.0, 3.0), (3.0, 0.0), (5.2, 0.0), (0.0, -9.99)]))
    assert heights == pytest.approx([0.0, 0.004, 0.08, 0.0, 0.0])
    assert floor_height(rects, np.array((3.0, 1.0))) == pytest.approx(0.08)  # a counter is no floor
    assert len(obstacles(scene, 0.08)) == 1  # but furniture


def test_the_fair_hall_is_level_but_for_the_booth_platforms():
    from bifold_demo.scenes import fair

    scene = fair.build_scene()
    scene.update(0.0, 0.0)
    rects = floors(scene)
    platform = scene["booth1_floor"]
    lo, hi = (platform.current_position + corner for corner in platform.extents())
    centre = (lo + hi)[[0, 2]] / 2
    beside = np.array((hi[0] + 0.2, centre[1]) if abs(hi[0]) < abs(lo[0]) else (lo[0] - 0.2, centre[1]))
    assert floor_height(rects, np.array([(0.0, -10.0), (12.0, -21.0)])) == pytest.approx([0.0, 0.0], abs=0.01)
    assert floor_height(rects, centre) == pytest.approx(fair.BOOTH_FLOOR)
    assert floor_height(rects, beside) == pytest.approx(0.0, abs=0.01)  # the aisle beside it


def test_no_mat_in_the_fair_is_half_on_a_booth_platform(monkeypatch):
    from test_yoga import CHARACTERS

    from bifold_demo.scenes import fair

    if not (CHARACTERS / "avatars").is_dir():
        pytest.skip("set BIFOLD_CHARACTERS_TEST_DIR to a prepared character library")
    monkeypatch.setenv("BIFOLD_CHARACTERS", str(CHARACTERS))
    scene = fair.build_scene()
    t = 10.0
    scene.update(t, 1 / 30)
    rects = floors(scene)
    platforms = rects[np.isclose(rects[:, 4], fair.BOOTH_FLOOR)]
    beside = [o for o in scene if getattr(o, "skinned_model", None) is not None and yoga.supports(o.skinned_model)
              and 0.0 < _rect_distance(o.current_position[None, [0, 2]], platforms).min() < 1.0]
    assert len(beside) >= 3  # visitors at the booths, a chat group
    selection = Selection(YogaClips(cache_dir=False, process=False))
    selection.update(scene, (1280, 720))
    try:
        for person in beside[:3]:
            for pose in ("warrior_2", "downward_dog"):
                clip = _real_clip(person.skinned_model, pose)
                performer = selection._performer(person)
                selection._place(person, performer, clip, t)
                heading, spot = selection._placing[id(person)].future.result(timeout=120)
                selection._drop(id(person))
                performer.reset()
                _times, points = selection._footprint(person, clip)
                under = floor_height(rects, place(points, heading, spot)[..., [0, 2]])
                level = floor_height(rects, person.current_position[[0, 2]])
                assert np.abs(under - level).max() <= 0.03, (person.name, pose, heading, spot)
    finally:
        selection.close()


def _real_clip(model, pose):
    """``pose`` for a real avatar: from the prepared library's cache if it is there (never written), else solved."""
    from test_yoga import CHARACTERS

    from bifold_demo.characters.rocketbox import read_clip

    path = cache_path(CHARACTERS / "yoga", model, pose)
    if path.is_file():
        clip = read_clip(path)
        if clip.metadata.get("cache_key") == model_key(model):
            return clip
    return yoga.yoga_clip(model, pose)


def test_the_mat_keeps_every_point_of_the_body_clear_of_people(biped):
    person = Character(biped)
    for pose in ("warrior_2", "downward_dog"):
        clip = yoga.yoga_clip(biped, pose)
        times, points = footprint(person, clip)
        ends = points.shape[1] * 3 // 4  # the footprint's segment ends: ... head, ... hands, fingertips, ... toes
        hold = int(np.searchsorted(times, sum(clip.metadata["hold"]) / 2))
        for k in (2, 4, 5, 7, 8, 11, 14):  # the head, a hand, fingertips, a foot, toes, the other fingertips and toes
            tip = points[hold, ends + k, [0, 2]]
            if np.linalg.norm(tip) < 0.3:
                continue  # not out to the side (the head standing, say)
            other = tip * (1.0 + 0.15 / np.linalg.norm(tip))  # someone standing just beyond it
            heading, spot = choose_place((0, 0, 0), [0.0], points, np.tile(other, (len(times), 1, 1)), np.zeros((0, 6)))
            gap = np.linalg.norm(place(points, heading, spot)[..., [0, 2]] - other, axis=-1).min()
            assert gap >= 0.5 - 1e-9, (pose, k, gap)


def test_the_way_back_home_after_a_queued_pose_keeps_clear_of_the_furniture():
    library = MotionLibrary(synthetic_clips())
    performer = Performer(stand((0.0, 0.0), 0.0, library=library))
    first = performer.perform(_clip(), 0.0, heading=90.0, position=(3.0, 0.0, 0.0))
    t = first.clip_start + 3.0  # a second pose asked for during the first: planned from where the first ends
    start = performer.free_at(t)
    here, home = np.asarray(performer.peek(start).position), performer.home(start)
    assert np.allclose(home, (0.0, 0.0, 0.0))
    counter = np.array([[0.5, 1.0, 2.5, 1.4, 0.0, 1.0]])  # between home and the spots in view
    def seen(world):  # the camera sees the pose only beyond z = 1.8
        return (world[..., 2] >= 1.8).mean(axis=(-2, -1))

    nobody = np.zeros((2, 0, 2))
    heading, spot = choose_place(here, [0.0], _line(1.0), nobody, counter, reach=4.0, seen=seen, home=home)
    assert performer.perform(_clip(view=0.0), t, heading, spot) is None  # it follows the first
    second = performer.current(first.end + 0.01)
    way = np.array([performer(u).position for u in np.arange(second.clip_end, second.end, 1 / 30)])[:, [0, 2]]
    assert _rect_distance(way, counter).min() > 0.15  # straight home, past the counter


def test_a_pose_hidden_by_a_counter_is_rated_by_what_can_be_seen(biped):
    scene = Scene(camera=Camera(position=(0.0, 1.6, 6.0), target=(0.0, 1.0, 0.0)))
    scene.add(Box(size=(0.8, 1.1, 0.4), position=(2.45, 0.55, 2.5)), name="counter")
    scene.add(Box(size=(0.6, 1.05, 2.0), position=(-1.3, 0.525, 0.0)), name="side")  # long side towards the camera
    scene.update(0.0, 0.0)
    selection = Selection(_clips())
    selection.update(scene, SIZE)
    seen = selection._rating(np.zeros((3, 0, 2)), obstacles(scene, 0.0), np.arange(3))
    line = np.stack([np.zeros(9), np.full(9, 0.3), np.linspace(-0.8, 0.8, 9)], axis=1)  # a low pose, 3 frames
    low = np.tile(line, (3, 1, 1))
    eye = np.asarray(scene.camera.current_position)
    clip_from_world = scene.camera.projection_matrix(SIZE[0] / SIZE[1]) @ scene.camera.view_matrix()

    def truth(world):  # the share in the picture and not behind a box (ray casts)
        world = world.reshape(-1, 3)
        lo, hi = obstacles(scene, 0.0)[:, [0, 4, 1]], obstacles(scene, 0.0)[:, [2, 5, 3]]
        ray = world[:, None, :] - eye
        with np.errstate(divide="ignore", invalid="ignore"):
            a, b = (lo - eye) / ray, (hi - eye) / ray
        enter, leave = np.minimum(a, b).max(-1), np.maximum(a, b).min(-1)
        behind = ((enter <= leave) & (leave >= 0) & (enter < 1)).any(-1)
        c = np.c_[world, np.ones(len(world))] @ clip_from_world.T
        x = (c[:, 0] / c[:, 3] * 0.5 + 0.5) * SIZE[0]
        return np.mean(~behind & (x >= 0.02 * SIZE[1]) & (x <= SIZE[0] - 0.02 * SIZE[1]))

    for spot in ((4.4, 0.0, 0.0), (5.0, 0.0, 0.0), (5.6, 0.0, 0.0)):  # behind the counter, partly out of the picture
        world = place(low, 90.0, spot)
        assert float(seen(world)) == pytest.approx(truth(world), abs=0.02)
    xs, zs = np.meshgrid(np.linspace(-1.75, -2.35, 7), np.linspace(0.2, 0.8, 5))  # beside "side", level with its front
    beside = np.tile(np.stack([xs.ravel(), np.full(xs.size, 0.3), zs.ravel()], axis=1), (3, 1, 1))
    assert 0.1 < 1.0 - truth(beside) < 0.5 and float(seen(beside)) == pytest.approx(truth(beside), abs=0.02)
    selection.close()


def test_people_with_walk_clips_walk_to_the_place_and_back():
    library = MotionLibrary(synthetic_clips())
    performer = Performer(stand((0.0, 0.0), 0.0, library=library))
    assert performer.can_walk and not Performer(_walker).can_walk
    act = performer.perform(_clip(), 1.0, heading=90.0, position=(2.0, 0.0, 0.0))
    assert act.approach is not None and act.retreat is not None and act.arrived > act.start + 1.0
    times = np.arange(0.0, act.end + 3.0, 1 / 30)
    track = np.array([performer(t).position for t in times])
    states = [performer(t) for t in times]
    headings = np.array([state.heading for state in states])
    assert np.abs(np.diff(track, axis=0)).max() < 0.08  # no jumps: at most a walking step per frame
    # turn clips hand over to their "@after" variant: the heading jumps where the clip's root turned
    handover = np.array([any(layer.clip.name.endswith("@after") for layer in state.layers) for state in states])
    steps = np.abs((np.diff(headings) + 180) % 360 - 180)[~(handover[1:] & ~handover[:-1])]
    assert steps.max() < 12.0
    assert np.allclose(performer(act.arrived + 0.5).position, (2.0, 0.0, 0.0))
    assert np.allclose(performer(act.clip_end - 0.1).position, (2.0, 0.0, 0.0))
    assert np.allclose(performer(act.end + 3.0).position, (0.0, 0.0, 0.0), atol=1e-6)  # back home
    assert np.allclose(performer.positions(times), track[:, [0, 2]], atol=1e-9)
    assert act.phase(act.start + 0.5) == "walking" and act.phase(act.clip_end + 0.5) == "back"


def test_a_pose_cut_short_on_the_way_walks_back_and_a_new_one_starts_where_the_last_ended():
    library = MotionLibrary(synthetic_clips())
    performer = Performer(stand((0.0, 0.0), 0.0, library=library))
    act = performer.perform(_clip(), 0.0, heading=90.0, position=(3.0, 0.0, 0.0))
    t = act.arrived * 0.5
    halfway = np.asarray(performer(t).position)
    assert performer.stop(t) and act.clip_end == t
    stop = np.asarray(act.retreat(0.0).position)  # where they come to a stop, a little further on
    assert 0.05 < stop[0] - halfway[0] < 0.4 and np.allclose(act.final().position, (0, 0, 0))
    assert np.linalg.norm(np.asarray(performer(t + 1 / 30).position) - halfway) < 0.08
    first = Performer(stand((0.0, 0.0), 0.0, library=library))
    act = first.perform(_clip(), 0.0, heading=90.0, position=(2.0, 0.0, 0.0))
    holding = act.clip_start + 3.0
    assert first.perform(_clip(view=0.0), holding, heading=0.0, position=(2.6, 0.0, 0.0)) is None
    assert act.retreat is None  # no walk back in between
    planned = first._planned(act.end)[-1]  # looked ahead to (e.g. by others finding a place) ...
    assert planned is not act and first._planned(act.end + 1.0)[-1] is planned  # ... planned once
    second = first.current(act.end + 0.01)
    assert second is planned and second.approach is not None and second.home == act.home
    assert np.allclose(second.before.position, (2.0, 0.0, 0.0))
    assert np.allclose(first(second.end + 3.0).position, (0.0, 0.0, 0.0), atol=1e-6)


# -- carrying on smoothly: the body's facing in turn clips, no standing starts, the camera on the way ----------------

FRAME = 1 / 60
JOINT_STEP = 0.065  # metres a joint may move in a frame (the feet of the synthetic walk itself move up to 0.056)


def _pose_clip(view=90.0):
    """A standing pose on the synthetic skeleton (so the body faces the heading), held from 2 s to 6 s of 8 s."""
    clip = synthetic_clip("yoga_cow", "idle", 241)
    clip.metadata.update(pose="cow", hold=(2.0, 6.0), view=view)
    return clip


def _turning_round():
    """A walk_by turning round with a 180 degree turn clip at the far end, and a moment 60 % into that clip."""
    library = MotionLibrary(synthetic_clips(full=True))
    base = walk_by([(-3.0, 1.0), (3.0, 1.0)], library=library, offset=0.0)
    for t in np.arange(0.0, base.duration, FRAME):
        layer = base(t).layers[-1]
        if layer.clip.name == "m_turn_right_180" and 0.6 * layer.clip.duration < layer.time:
            return base, float(t)
    raise AssertionError("no turn clip")


def _run(performer, t0, t1, ops=()):
    """Step a synthetic person driven by ``performer`` at 60 Hz from ``t0`` to ``t1``, doing each ``(time,
    op(performer, t))`` then; per frame: the body's facing from the hips (world degrees) and the world joints."""
    person = Character(synthetic_model(1.0), performer)
    left, right = BONES.index("Bip01 L Thigh"), BONES.index("Bip01 R Thigh")
    ops, facings, joints = sorted(ops, key=lambda op: op[0]), [], []
    for t in np.arange(t0, t1, FRAME):
        person.update(float(t))
        model = person.matrices()[0]
        world = person.pose[:, :3, 3] @ model[:3, :3].T + model[:3, 3]
        d = world[left] - world[right]
        facings.append(math.degrees(math.atan2(-d[2], d[0])))
        joints.append(world)
        while ops and ops[0][0] <= t + 1e-9:
            ops.pop(0)[1](performer, float(t))
    turns = np.abs((np.diff(facings) + 180.0) % 360.0 - 180.0)
    return np.array(facings), turns, np.linalg.norm(np.diff(joints, axis=0), axis=-1).max(axis=-1)


def _ahead(state, facing, distance, turn=0.0):
    a = math.radians(facing + turn)
    return (state.position[0] + distance * math.sin(a), 0.0, state.position[2] + distance * math.cos(a))


@pytest.mark.parametrize("turn", [0.0, 90.0, 180.0, None])  # where the spot is from the way the body faces
def test_a_pose_asked_for_while_turning_carries_on_from_the_way_the_body_faces(turn):
    base, mid = _turning_round()
    assert abs(((base(mid).heading - Performer(base).facing(mid)) + 180) % 360 - 180) > 90  # the heading lags
    acts = []

    def ask(p, t):
        facing = p.facing(t)
        spot = None if turn is None else _ahead(p(t), facing, 2.0, turn)
        acts.append(p.perform(_pose_clip(), t, heading=facing + 120.0, position=spot))

    performer = Performer(base)
    facings, turns, moves = _run(performer, mid - 0.5, mid + 8.0, [(mid, ask)])
    act = acts[0]
    stop = act.before.heading + performer.turned(act.before.layers)  # where they face once the turn stops
    if act.approach is not None:  # the walk starts from there, not from the heading
        assert abs((act.approach.behaviour.start_heading - stop + 180) % 360 - 180) < 1e-6
    else:  # as does the turn on the spot (and it takes as long as the body needs)
        assert act.turn == pytest.approx(max(0.6, abs((act.heading - stop + 180) % 360 - 180) / 150.0))
    assert turns.max() < 4.0 and moves.max() < JOINT_STEP  # per frame (240 degrees a second): no swing back, no flip
    assert act.phase(mid + 8.0) in ("into", "holding", "out")  # in the pose by then, facing its way
    assert abs((facings[-1] - act.heading + 180) % 360 - 180) < 1.0


def test_a_new_pose_early_in_the_walk_back_and_a_stop_in_the_first_turn_carry_on_from_the_way_the_body_faces():
    library = MotionLibrary(synthetic_clips(full=True))
    performer = Performer(stand((0.0, 0.0), 0.0, library=library))
    act = performer.perform(_pose_clip(), 0.0, heading=90.0, position=(3.0, 0.0, 0.0))
    first = act.retreat.behaviour._pieces[0]
    assert first.kind == "turn"  # the walk back starts by turning round
    t = act.clip_end + 0.5 * (first.t1 - first.t0)
    again = (t, lambda p, t: p.perform(_pose_clip(view=0.0), t, heading=0.0, position=(2.0, 0.0, 1.5)))
    _, turns, moves = _run(performer, act.clip_end - 0.5, t + 4.0, [again])
    assert turns.max() < 4.0 and moves.max() < JOINT_STEP
    # 0 pressed while turning round to walk to a spot behind: they turn back as fast as the body needs, no faster
    performer = Performer(stand((0.0, 0.0), 0.0, library=library))
    act = performer.perform(_pose_clip(), 0.0, heading=90.0, position=(0.0, 0.0, -2.0))
    first = act.approach.behaviour._pieces[0]
    assert first.kind == "turn" and act.approach.behaviour._entries[0].clip.metadata["kind"] == "turn"
    t = 0.7 * first.t1
    facings, turns, moves = _run(performer, 0.0, t + 3.0, [(t, lambda p, t: p.stop(t))])
    src, start, back = performer._take_over(act, performer.acts)
    body = abs(((src.heading + performer.turned(src.layers)) - start.heading + 180) % 360 - 180)
    assert body > 90.0 and back >= body / 150.0 - 1e-9 and turns.max() < 4.0 and moves.max() < JOINT_STEP
    assert abs((facings[-1] - start.heading + 180) % 360 - 180) < 2.0


def test_a_walking_person_asked_for_a_pose_slows_down_to_a_stop_and_walks_on_from_a_standstill():
    library = MotionLibrary(synthetic_clips(full=True))
    base = walk_by([(-6.0, 0.0), (6.0, 0.0)], library=library, offset=0.0, pause=(0.0, 0.0))
    walking = 4.0  # in mid stride along +x
    speed = np.linalg.norm(np.diff(base.positions([walking, walking + FRAME]), axis=0)) / FRAME
    assert speed > 0.8
    performer, track = Performer(base), []
    for t in np.arange(walking - 0.2, walking + 0.8, FRAME):
        track.append(performer(t))
        if abs(t - walking) < 1e-9:
            act = performer.perform(_pose_clip(), t, heading=0.0, position=(-1.0, 0.0, 2.0))
    speeds = np.linalg.norm(np.diff([s.position for s in track], axis=0), axis=1) / FRAME
    assert np.abs(np.diff(speeds)).max() < 0.1  # slowing down (and turning to go) at most 6 m/s^2, no dead stop
    assert speeds[12:30].min() > 0.2 and speeds[12:30].max() < 1.1 * speed  # on for a moment, then stopping
    stride = [next(layer.time for layer in s.layers if layer.clip.name.startswith("m_walk")) for s in track[12:24]]
    assert np.all(np.diff(stride) > 0.0)  # the walk clip goes on meanwhile: the feet keep stepping
    assert np.isclose(act.before.position[0] - track[12].position[0], speed * LEAD / 2, rtol=0.1)
    # back into the walk afterwards: from a standstill, never faster than the walk itself
    back = performer._take_over(act, performer.acts)[2]
    times = np.arange(act.end - 0.5, act.end + back + 1.0, FRAME)
    states = [performer(t) for t in times]
    speeds = np.linalg.norm(np.diff([s.position for s in states], axis=0), axis=1) / FRAME
    assert speeds.max() < 1.05 * speed and np.abs(np.diff(speeds)).max() < 0.1
    facings = [s.heading + performer.turned(s.layers) for s in states]
    assert np.abs((np.diff(facings) + 180) % 360 - 180).max() < 4.0  # at most 240 degrees per second
    assert np.allclose(performer.positions(times), np.array([s.position for s in states])[:, [0, 2]], atol=1e-9)


def _swaying_walks():
    """The synthetic clips with walks whose hips swing 10 degrees either way, as real walks do (the root of a clip that
    faces the heading turns a little: the turn clips' root yaw is no longer all of the root's turn)."""
    clips = synthetic_clips(full=True)
    for name, clip in list(clips.items()):
        if clip.metadata["kind"] == "walk":
            clips[name] = synthetic_clip(name, "walk", clip.frames, speed=clip.metadata["speed"],
                                         root_yaw=lambda u: 10.0 * np.sin(2 * np.pi * u))
    return MotionLibrary(clips)


@pytest.mark.parametrize("late", [0.8, 0.88, 0.94])
def test_a_pose_asked_for_late_in_a_turn_round_does_not_flip_the_body(late):
    """Late in a 180 degree turn clip its root is about opposite to that of a walk facing the same way: crossfaded
    as they are, the blend of the two roots flipped between the two ways round (by up to 115 degrees in a frame)."""
    library = _swaying_walks()
    base = walk_by([(-3.0, 1.0), (3.0, 1.0)], library=library, offset=0.0)
    mid = next(float(t) for t in np.arange(0.0, base.duration, FRAME)
               if (layer := base(t).layers[-1]).clip.name == "m_turn_right_180" and late * layer.clip.duration < layer.time)

    def ask(p, t):  # a spot ahead of the way the body faces: the walk there starts with a walk
        p.perform(_pose_clip(), t, heading=p.facing(t) + 120.0, position=_ahead(p(t), p.facing(t), 2.0))

    performer = Performer(base)
    _, turns, moves = _run(performer, mid - 0.5, mid + 4.0, [(mid, ask)])
    assert performer.acts[0].approach.behaviour._pieces[0].kind == "walk"
    assert turns.max() < 4.0 and moves.max() < JOINT_STEP
    stop = performer.acts[0].before  # where the turn stops, handed over early to the clip's "@after" variant
    assert stop.layers[-1].clip.name == "m_turn_right_180@after" and abs(performer.turned(stop.layers)) < 90.0


@pytest.mark.skipif(not (CHARACTERS / "clips").is_dir(), reason="no prepared Rocketbox characters")
def test_real_people_asked_for_a_pose_late_in_a_turn_round_do_not_flip():
    from bifold_demo.characters.crowd import avatar_style
    from bifold_demo.characters.library import CharacterLibrary

    library = CharacterLibrary(CHARACTERS)
    motions, available = library.motions(), library.available()
    names = [next((n for n, e in sorted(available.items()) if e.get("sex") == sex and e.get("age_group") == "adult"),
                  None) for sex in ("male", "female")]
    if not library.is_ready() or None in names or not motions.has("turn_right_180"):
        pytest.skip("the prepared library lacks clips or adult avatars")
    idle = next(clip for clip in motions.clips.values() if clip.metadata.get("kind") == "idle")
    pose = AnimationClip("yoga_idle", idle.fps, list(idle.bone_names), idle.rotations, idle.translations,
                         root=idle.root, metadata={"pose": "idle", "hold": (1.0, 2.0), "view": 90.0})
    worst = []
    for name in names:
        model = library.load(name, textures="none")
        base = walk_by([(-4.0, 1.0), (4.0, 1.0)], library=motions, rng=1, style=avatar_style(available[name], model,
                                                                                             motions), offset=0.0)
        times = np.arange(0.0, base.duration, FRAME)
        layers = [base(t).layers for t in times]
        left, right = model.skeleton.find("Bip01 L Thigh"), model.skeleton.find("Bip01 R Thigh")
        for late in (0.8, 0.88):  # the person, a woman most of all, flipped by up to 167 degrees in a frame
            mid = next(float(t) for t, ls in zip(times, layers) if ls[-1].weight > 0.99
                       and abs(turn_angle(ls[-1].clip)) > 150 and "@after" not in ls[-1].clip.name
                       and late * ls[-1].clip.duration < ls[-1].time)
            performer = Performer(base)
            person, facings, asked = Character(model, performer), [], False
            for t in np.arange(mid - 0.2, mid + 3.0, FRAME):
                person.update(float(t))
                world = person.pose[:, :3, 3] @ person.matrices()[0][:3, :3].T
                d = world[left] - world[right]
                facings.append(math.degrees(math.atan2(-d[2], d[0])))
                if not asked and t >= mid - 1e-9:
                    facing = performer.facing(float(t))
                    performer.perform(pose, float(t), facing + 90.0, _ahead(performer(float(t)), facing, 2.0))
                    asked = True
            worst.append(float(np.abs((np.diff(facings) + 180.0) % 360.0 - 180.0).max()))
    assert max(worst) < 6.0, worst


def test_a_walking_person_does_the_pose_where_they_stop_and_walks_on_from_there():
    """A walker slows to a stop LEAD / 2 seconds on: the place for the pose is looked for there (free_at, peek), the
    pose without a place is done there, and their behaviour carries on from there, never going back."""
    library = MotionLibrary(synthetic_clips(full=True))
    base = walk_by([(-6.0, 0.0), (6.0, 0.0)], library=library, offset=0.0, pause=(0.0, 0.0))
    walking = 4.0
    for position in (None, "planned"):
        performer = Performer(base)
        assert performer.free_at(walking) == pytest.approx(walking + LEAD / 2)
        planned = np.asarray(performer.peek(performer.free_at(walking)).position)
        assert planned[0] - base(walking).position[0] > 0.2  # on along the walk
        act = performer.perform(_pose_clip(), walking, heading=0.0,
                                position=None if position is None else tuple(planned))
        assert np.allclose(act.before.position, planned) and np.allclose(act.home, planned)
        assert act.approach is None and act.retreat is None
        times = np.arange(walking - 0.5, act.end + 3.0, FRAME)
        track = performer.positions(times)
        assert np.diff(track[:, 0]).min() > -1e-9  # never backwards
        assert np.allclose(track[times >= act.end + 1.0][:, 1], 0.0)


def test_the_walks_to_a_pose_and_back_keep_the_face_off_the_camera():
    library = MotionLibrary(synthetic_clips(full=True))
    base = walk_by([(-4.0, -2.0), (4.0, -2.0)], camera=CAMERA, library=library, rng=3, offset=0.0)
    performer = Performer(base)
    here = base(1.0)
    towards = performer.walk(here.position, performer.facing(1.0), (0.3, 0.0, 3.5))  # straight at the camera
    assert towards.behaviour.camera.nominal == (CAMERA[0], CAMERA[2]) and towards.behaviour.min_angle == base.min_angle
    assert towards.behaviour.look_feasible
    angles = [_angle_to_camera(towards.behaviour, u) for u in np.arange(0.0, towards.duration, 0.05)]
    assert min(a for a in angles if a is not None) >= base.min_angle - 1e-6


# -- selecting people -------------------------------------------------------------------------------------------


def _standing(x, z, visible=lambda t: True):
    return lambda t: CharacterState((x, 0.0, z), 0.0, [], visible=visible(t))


@pytest.fixture(scope="module")
def biped():
    return synthetic_biped()


def _scene(biped):
    scene = Scene(camera=Camera(position=(0.0, 1.6, 6.0), target=(0.0, 1.0, 0.0)))
    scene.add(Character(biped, _standing(-1.5, 0.0), name="left"))
    scene.add(Character(biped, _standing(0.0, 1.0), name="middle"))  # nearest the camera
    scene.add(Character(biped, _standing(1.5, 0.0), name="right"))
    scene.add(Character(biped, _standing(30.0, 0.0), name="away"))  # out of the picture
    scene.update(0.0, 0.0)
    return scene


def _clips():
    return YogaClips(cache_dir=False, process=False)


def test_tab_goes_through_the_people_in_the_picture_left_to_right(biped):
    scene = _scene(biped)
    selection = Selection(_clips())
    try:
        assert "middle" in selection.enter(scene, SIZE)
        assert [p.character.name for p in selection.people] == ["left", "middle", "right"]
        names = []
        for key in (TAB, TAB, TAB, BACKTAB, LEFT, RIGHT):
            assert selection.handle_key(key, 0.0)[0]
            names.append(selection.selected.name)
        assert names == ["right", "left", "middle", "left", "right", "left"]
        assert selection.handle_key(27, 0.0) == (True, "selection off") and not selection.active
        assert selection.handle_key(TAB, 0.0) == (False, None)  # not in selection mode
    finally:
        selection.close()


def test_a_new_person_is_selected_when_the_selected_one_is_gone(biped):
    scene = _scene(biped)
    walker = Character(biped, lambda t: CharacterState((0.4 + 3.0 * t, 0.0, 2.0), 90.0, []), name="walker")
    scene.add(walker)
    scene.update(0.0, 0.0)
    selection = Selection(_clips())
    try:
        selection.enter(scene, SIZE)
        assert selection.selected is walker  # the nearest
        message = None
        for t in np.arange(0.1, 3.0, 0.1):
            scene.update(t, 0.1)
            message = message or selection.update(scene, SIZE)  # it walks out of the picture on the right
        assert "walker" in message and "left the picture" in message and selection.selected.name == "right"
        scene.remove("right")
        scene.update(3.1, 0.1)
        message = selection.update(scene, SIZE)
        assert "was removed" in message and selection.selected.name == "middle"
        selection.selected.behaviour = _standing(0.0, 1.0, visible=lambda t: False)
        scene.update(3.2, 0.1)
        message = selection.update(scene, SIZE)
        assert "is hidden" in message and selection.selected.name == "left"
        scene.remove("left")
        scene.update(3.3, 0.1)
        assert "nobody else" in selection.update(scene, SIZE) and selection.selected is None
        assert selection.status_lines(3.3) == ["selected: nobody in the picture"]
        scene["middle"].behaviour = _standing(0.0, 1.0)
        scene.update(3.4, 0.1)
        assert selection.update(scene, SIZE) == "selected middle (synthetic_biped)"  # someone is back
    finally:
        selection.close()


def _run_until(scene, selection, t, until, step=1 / 15, searches=False):
    """Run the scene and the selection until ``until(t)``; with ``searches``, scene time waits for spots being found
    (as if they were found at once)."""
    messages = []
    for _ in range(3000):
        if searches:
            wait([entry.future for entry in selection._placing.values() if entry.future is not None], timeout=60)
        t += step
        scene.update(t, step)
        message = selection.update(scene, SIZE)
        if message:
            messages.append(message)
        if until(t):
            return t, messages
        if not messages and (selection._requests or selection._placing):
            time.sleep(0.005)  # let the clip thread work
    raise AssertionError("timed out")


def _posing(person, t) -> bool:
    """Whether ``person`` is doing a pose at ``t`` (not just standing waiting for a spot for it)."""
    act = person.behaviour.current(t) if isinstance(person.behaviour, Performer) else None
    return act is not None and not act.clip.metadata.get("waiting")


def test_number_keys_make_the_selected_person_do_yoga(biped):
    scene = _scene(biped)
    selection = Selection(_clips())
    try:
        selection.enter(scene, SIZE)
        person = selection.selected
        handled, message = selection.handle_key(ord("5"), 0.0)  # Warrior 2: seen from the front
        assert handled and message.startswith("preparing Warrior 2")
        assert "preparing Warrior 2" in selection.describe(person, 0.0)
        t, _ = _run_until(scene, selection, 0.0, lambda t: id(person) in selection._placing)
        waiting = person.behaviour.current(t)  # the clip is ready: they stand still while a spot is found
        assert waiting.clip.metadata["waiting"] and selection.describe(person, t) == "Warrior 2: finding a spot ..."
        t, messages = _run_until(scene, selection, t, lambda t: _posing(person, t))
        assert messages[-1] == "middle (synthetic_biped): Warrior 2"
        act = person.behaviour.current(t)
        camera = scene.camera.current_position
        assert act.heading == pytest.approx(math.degrees(math.atan2(camera[0], camera[2] - 1.0)), abs=1e-6)
        _run_until(scene, selection, t, lambda t: act.phase(t) == "holding")
        assert selection.describe(person, act.start + act.turn + act.hold[0] + 0.1) == "Warrior 2: holding"
        t = act.start + act.turn + act.hold[0] + 0.5
        assert "keeps holding" in selection.handle_key(ord("5"), t)[1] and act.looping
        assert selection.handle_key(ord("0"), t + 1.0) == (True, "middle (synthetic_biped) stops")
        assert act.cut == t + 1.0 and act.mode == "skip"
        scene.update(act.end + 5.0, 0.1)
        assert person.state.position == pytest.approx((0.0, 0.0, 1.0)) and person.state.heading == 0.0
        assert selection.handle_key(BACKSPACE, act.end + 5.0)[1] == "middle (synthetic_biped) is not doing a pose"
        selection.reset()
        assert person.behaviour.acts == []
    finally:
        selection.close()


def test_the_same_number_again_holds_a_pose_that_has_not_started_yet(biped):
    scene = _scene(biped)
    selection = Selection(_clips())
    try:
        selection.enter(scene, SIZE)
        person = selection.selected
        assert selection.handle_key(ord("1"), 0.0)[1].startswith("preparing Yogi Squat")
        assert "will keep holding Yogi Squat" in selection.handle_key(ord("1"), 0.0)[1]
        t, _ = _run_until(scene, selection, 0.0, lambda t: _posing(person, t))
        assert person.behaviour.current(t).looping
    finally:
        selection.close()


def _clip_ready(selection, model, pose):
    for _ in range(3000):
        clip = selection.clips.request(model, pose)
        if clip is not None:
            return clip
        time.sleep(0.01)
    raise AssertionError(f"no {pose} clip")


def _walking_scene(biped):
    scene = Scene(camera=Camera(position=(0.0, 1.6, 6.0), target=(0.0, 1.0, 0.0)))
    walker = Character(biped, lambda t: CharacterState((-1.0 + 1.0 * t, 0.0, 1.0), 90.0, []), name="walker")
    scene.add(walker)
    scene.update(0.0, 0.0)
    return scene, walker


def test_a_walking_person_stops_and_the_pose_starts_where_and_when_planned(biped, monkeypatch):
    """A walker asked for a pose slows to a stop over LEAD (LEAD / 2 seconds on, at 1 m/s 0.25 m past the key),
    stands there while the spot is found, and the search plans from there: the pose starts from there, and the
    behaviour carries on from there afterwards."""
    import bifold_demo.selection as selection_module

    real = selection_module.choose_place

    def slow(*args, **kwargs):  # finding the spot takes a while
        time.sleep(0.6)
        return real(*args, **kwargs)

    monkeypatch.setattr(selection_module, "choose_place", slow)
    scene, walker = _walking_scene(biped)
    selection = Selection(_clips())
    try:
        _clip_ready(selection, biped, "warrior_2")
        t = 0.5
        scene.update(t, t)
        selection.update(scene, SIZE)
        selection.selected = walker
        key = np.asarray(walker.state.position)
        stop = key + (LEAD / 2, 0.0, 0.0)
        assert selection.do("warrior_2", t) is None
        track, planned, begin = [], [], time.perf_counter()
        while not _posing(walker, t):  # scene time is real time, as in the demo
            time.sleep(1 / 60)
            dt, t = 0.5 + time.perf_counter() - begin - t, 0.5 + time.perf_counter() - begin
            scene.update(t, dt)
            entry = selection._placing.get(id(walker))
            if entry is not None and entry.future is not None:
                planned.append((entry.start, np.asarray(entry.here)))
            selection.update(scene, SIZE)
            track.append((t, np.asarray(walker.state.position)))
            assert t < 30.0
        act = walker.behaviour.current(t)
        assert act.start - 0.5 > 0.6  # time enough to walk on a little ...
        times, positions = np.array([u for u, _ in track[:-1]]), np.array([p for _, p in track[:-1]])
        assert np.diff(positions[:, 0]).min() >= -1e-9 and positions[:, 0].max() <= stop[0] + 1e-9  # slowing down
        assert np.allclose(positions[times >= 0.5 + LEAD], stop)  # ... then they stood still where they stopped
        assert np.allclose(planned[-1][1], stop) and np.allclose(act.before.position, stop)  # planned from there
        assert planned[-1][0] <= act.start <= planned[-1][0] + 0.6 + STALE  # not before it was planned for, nor much later
        assert np.allclose(walker.behaviour.home(act.start), stop)  # and carry on from there afterwards
    finally:
        selection.close()


def test_a_spot_found_long_ago_is_looked_for_again(biped):
    scene, walker = _walking_scene(biped)
    selection = Selection(_clips())
    try:
        _clip_ready(selection, biped, "warrior_2")
        t = 0.5
        scene.update(t, t)
        selection.update(scene, SIZE)
        selection.selected = walker
        key = np.asarray(walker.state.position)
        selection.do("warrior_2", t)
        selection._placing[id(walker)].future.result(timeout=60)
        t += 20.0  # found, but the demo showed the webcam meanwhile: no frames, the clock ran on
        scene.update(t, 20.0)
        selection.update(scene, SIZE)
        assert selection._placing[id(walker)].tries == 2 and not _posing(walker, t)  # the others moved on: again
        t, _ = _run_until(scene, selection, t, lambda t: _posing(walker, t))
        act = walker.behaviour.current(t)
        assert np.allclose(act.before.position, key + (LEAD / 2, 0.0, 0.0)) and act.start >= 20.5  # where they stopped
    finally:
        selection.close()


def test_poses_asked_of_neighbours_one_after_another_keep_clear_of_each_other(biped, monkeypatch):
    library = MotionLibrary(synthetic_clips())
    scene = Scene(camera=Camera(position=(0.0, 1.6, 6.0), target=(0.0, 1.0, 0.0)))
    first = Character(biped, stand((-0.35, 1.0), 0.0, library=library), name="first")
    second = Character(biped, stand((0.35, 1.0), 0.0, library=library), name="second")
    scene.add(first)
    scene.add(second)
    scene.update(0.0, 0.0)
    searches = []
    find = Selection._find

    def logged(entry, *args, **kwargs):
        searches.append((entry.character.name, entry.asked))
        return find(entry, *args, **kwargs)

    monkeypatch.setattr(Selection, "_find", staticmethod(logged))
    selection = Selection(_clips())
    try:
        selection.update(scene, SIZE)
        _clip_ready(selection, biped, "warrior_2")
        t = 0.0
        for person in (first, second):  # Warrior 2 for one, Tab, Warrior 2 for the other
            selection.selected = person
            selection.do("warrior_2", t)
            t, _ = _run_until(scene, selection, t, lambda t: True, searches=True)
        t, _ = _run_until(scene, selection, t, lambda t: _posing(first, t) and _posing(second, t), searches=True)
        a, b = first.behaviour.current(t), second.behaviour.current(t)
        assert [name for name, _asked in searches] == ["first", "second"]
        assert searches[1][1] >= a.start  # the second spot was looked for once the first pose had started
        frames, points = selection._footprint(first, a.clip)
        step = frames[1] - frames[0]
        closest = np.inf
        for u in np.arange(max(a.clip_start, b.clip_start), min(a.clip_end, b.clip_end), 0.25):
            body_a = place(points[min(round(a.clip_time(u) / step), len(points) - 1)], a.heading, a.position)[:, [0, 2]]
            body_b = place(points[min(round(b.clip_time(u) / step), len(points) - 1)], b.heading, b.position)[:, [0, 2]]
            closest = min(closest, float(np.linalg.norm(body_a[:, None] - body_b[None], axis=-1).min()))
        assert closest >= 0.25 - 1e-6  # their bodies keep apart (in place, their arms would cross)
    finally:
        selection.close()


def test_a_superseded_or_closed_search_stops_at_once(biped, monkeypatch):
    import threading

    import bifold_demo.selection as selection_module

    nobody, nothing = np.zeros((2, 0, 2)), np.zeros((0, 6))
    assert choose_place((0, 0, 0), [0.0], _line(), nobody, nothing, cancelled=lambda: True) is None

    def endless(*args, cancelled, **kwargs):  # a search that would take long
        deadline = time.monotonic() + 10.0
        while not cancelled() and time.monotonic() < deadline:
            time.sleep(0.01)

    monkeypatch.setattr(selection_module, "choose_place", endless)
    scene = _scene(biped)
    selection = Selection(_clips())
    selection.update(scene, SIZE)
    person = selection.selected = scene["middle"]
    for pose in ("warrior_2", "cow"):
        _clip_ready(selection, biped, pose)
    selection.do("warrior_2", 0.0)
    superseded = selection._placing[id(person)]
    selection.do("cow", 0.0)  # another pose before a spot was found: that search gives up
    assert superseded.cancel.is_set() and wait([superseded.future], timeout=1.0).done
    begin = time.monotonic()
    selection.close()  # quitting while the next search runs
    for thread in threading.enumerate():
        if thread.name.startswith("yoga-place"):
            thread.join(timeout=2.0)
            assert not thread.is_alive()
    assert time.monotonic() - begin < 1.0


def test_the_menus_switch_between_poses_and_items(biped):
    from bifold_demo.characters import accessories
    from bifold_demo.characters.accessories import Wardrobe
    from bifold_demo.selection import menus

    titles = [title for title, _kind, _entries in menus()]
    assert titles[-1] == "Items" and all(len(entries) <= 8 for _t, _k, entries in menus())
    scene = _scene(biped)
    selection = Selection(_clips(), Wardrobe(None))
    try:
        selection.enter(scene, SIZE)
        person = selection.selected
        assert selection.handle_key(UP, 0.0) == (True, "menu: Items")  # round to the last menu
        assert "[Items] 1 hat  2 glasses  3 sign  4 t-shirt" in selection.help_lines()[1]
        hats = [look.name for look in accessories.built_in()["hat"].values()]
        for k, look in enumerate(hats):
            message = selection.handle_key(ord("1"), 0.0)[1]
            assert message.startswith(f"middle (synthetic_biped): {look} ({k + 1} of {len(hats)}")
        assert message.endswith("again: off)")  # the last look
        assert selection.handle_key(ord("1"), 0.0)[1] == "middle (synthetic_biped) takes the hat off"
        for _ in range(list(accessories.built_in()["glasses"]).index("sunglasses") + 1):
            selection.handle_key(ord("2"), 0.0)
        assert selection.wears(person) == "sunglasses"  # not "sunglasses glasses"
        selection.handle_key(ord("2"), 0.0)
        selection.handle_key(ord("2"), 0.0)
        assert "glasses" not in accessories.wearing(person)
        assert "hat" not in accessories.wearing(person)
        selection.handle_key(ord("3"), 0.0)
        selection.handle_key(ord("4"), 0.0)
        assert sorted(accessories.wearing(person)) == ["sign", "tshirt"]
        assert selection.wears(person) == "welcome sign, white t-shirt"
        assert selection.detail(person, 0.0) == "welcome sign, white t-shirt"
        assert "wears: welcome sign, white t-shirt" in selection.status_lines(0.0)
        assert selection.handle_key(ord("9"), 0.0) == (True, None)  # nothing on that key
        assert selection.handle_key(ord("0"), 0.0)[1] == "middle (synthetic_biped) takes everything off"
        assert accessories.wearing(person) == {}
        assert selection.handle_key(ord("i"), 0.0)[1] == f"menu: {titles[0]}"  # back to the poses
        assert selection.handle_key(ord("5"), 0.0)[1].startswith("preparing")
        assert selection.handle_key(ord("i"), 0.0)[1] == "menu: Items"
    finally:
        selection.close()


def test_scenes_without_3d_people_have_nobody_to_select():
    selection = Selection(_clips())
    scene = Scene()
    scene.update(0.0, 0.0)
    assert selection.enter(scene, SIZE) == "this scene has no 3D people that can do yoga" and not selection.active


# -- building clips ------------------------------------------------------------------------------------------------


def test_clips_are_built_in_a_worker_process_and_cached_on_disk(biped, tmp_path):
    clips = YogaClips(cache_dir=tmp_path)
    try:
        assert clips.request(biped, "warrior_2") is None
        deadline = time.time() + 120
        while clips.get(biped, "warrior_2") is None and time.time() < deadline:
            assert clips.error(biped, "warrior_2") is None
            time.sleep(0.05)
        clip = clips.get(biped, "warrior_2")
        assert clips.process  # it did not fall back to building here
    finally:
        clips.close()
    reference = yoga.yoga_clip(biped, "warrior_2")
    assert clip.frames == reference.frames and np.allclose(clip.rotations, reference.rotations)
    path = cache_path(tmp_path / "yoga", biped, "warrior_2")
    assert path.is_file()
    again = YogaClips(cache_dir=tmp_path, process=False)
    try:
        again.request(biped, "warrior_2")
        deadline = time.time() + 30
        while again.get(biped, "warrior_2") is None and time.time() < deadline:
            time.sleep(0.01)
        cached = again.get(biped, "warrior_2")
        assert cached.metadata["cache_key"] == model_key(biped)
        assert np.allclose(cached.rotations, reference.rotations, atol=1e-6)
    finally:
        again.close()
    with pytest.raises(KeyError):
        again.request(biped, "headstand")


def test_selection_keys_in_the_demo_app(gl, biped, tmp_path, monkeypatch):
    from bifold_demo.app import DemoApp, DemoConfig
    from bifold_demo.sources import SourceSwitcher

    monkeypatch.chdir(tmp_path)
    scene = _scene(biped)
    app = DemoApp(DemoConfig(headless=True, frames=1, width=320, height=180, detect=False, uploads=None), scene=scene)
    app.run()
    app.selection.clips = _clips()
    app.config.headless = False  # draw the key help like the windowed demo
    app.switcher = SourceSwitcher({"scene": lambda: app.scene_source}, "scene")
    messages = []
    app.notify = lambda message, *args, **kwargs: messages.append(message)
    window = type("W", (), {"toggle_fullscreen": lambda self: None})()
    try:
        app._handle_key(ord("y"), window)
        assert app.selection.active and app.selection.selected.name == "middle" and "selected middle" in messages[-1]
        app._handle_key(TAB, window)
        assert app.selection.selected.name == "right"
        app._handle_key(ord("1"), window)
        assert messages[-1].startswith("preparing Yogi Squat for right")
        frame = app.scene_source.read()
        app.selection.update(app.scene_source.scene, frame.size)
        plain = frame.image.copy()
        canvas = app._compose(frame, None)
        assert app.selection.box is not None and (canvas != plain).any()  # the tag and the selection help
        assert any(line.startswith("selected: right") for line in app._status_lines(frame, None))
        app._handle_key(27, window)  # Esc ends selection mode, it does not quit
        assert not app.selection.active and app._running
        app._handle_key(27, window)
        assert not app._running
    finally:
        app.selection.close()


def test_the_key_help_of_every_menu_fits_the_picture():
    import cv2

    from bifold_demo.display.overlay import FONT
    from bifold_demo.selection import menus

    selection = Selection.__new__(Selection)
    for menu in range(len(menus())):
        selection.menu = menu
        lines = selection.help_lines()
        widths = [cv2.getTextSize(line, FONT, 0.55, 1)[0][0] for line in lines]
        assert max(widths) + 2 * 10 + 12 < 1280, lines  # the panel at 1280x720, padding and margin included


def test_tab_goes_by_where_people_are_now_even_when_they_pass_each_other(biped):
    scene = Scene(camera=Camera(position=(0.0, 1.6, 6.0), target=(0.0, 1.0, 0.0)))
    scene.add(Character(biped, _standing(-1.5, 0.0), name="left"))
    scene.add(Character(biped, lambda t: CharacterState((-1.0 + 2.0 * min(t, 1.0), 0.0, 0.5), 0.0, []), name="walker"))
    scene.add(Character(biped, _standing(0.5, 0.0), name="right"))
    scene.update(0.0, 0.0)
    selection = Selection(_clips())
    try:
        selection.enter(scene, SIZE)
        selection.selected = scene["left"]
        selection.handle_key(TAB, 0.0)
        assert selection.selected.name == "walker"  # at x = -1: next to the right of "left"
        scene.update(1.0, 1.0)  # the walker passes "right" (now at x = 1)
        selection.update(scene, SIZE)
        selection.handle_key(TAB, 1.0)
        assert selection.selected.name == "left"  # nobody right of the walker any more: round to the leftmost
        selection.handle_key(TAB, 1.0)
        assert selection.selected.name == "right"
        selection.handle_key(BACKTAB, 1.0)
        assert selection.selected.name == "left"
        selection.handle_key(BACKTAB, 1.0)
        assert selection.selected.name == "walker"  # round to the rightmost
    finally:
        selection.close()


def test_a_click_selects_the_person_under_the_cursor_the_nearest_one_when_they_overlap(biped):
    scene = Scene(camera=Camera(position=(0.0, 1.6, 6.0), target=(0.0, 1.0, 0.0)))
    scene.add(Character(biped, _standing(-1.5, 0.0), name="left"))
    scene.add(Character(biped, _standing(0.0, 1.0), name="front"))
    scene.add(Character(biped, _standing(0.5, -2.0), name="behind"))  # partly hidden by "front"
    scene.update(0.0, 0.0)
    selection = Selection(_clips())
    try:
        selection.enter(scene, SIZE)
        boxes = {p.character.name: p.box for p in selection.people}

        def centre(name):
            x0, y0, x1, y1 = boxes[name]
            return (x0 + x1) / 2, (y0 + y1) / 2

        assert selection.select_at(centre("left")).startswith("selected left")
        assert selection.select_at(centre("behind")).startswith("selected front")  # both boxes there: the nearer
        x0, y0, x1, y1 = boxes["behind"]
        beside = (x1 - 1.0, (y0 + y1) / 2)  # right of the front person, inside the box of the one behind
        assert boxes["front"][0] < x0 < boxes["front"][2] < beside[0]
        assert selection.select_at(beside).startswith("selected behind")
        x0, y0, x1, y1 = boxes["left"]
        assert selection.select_at((x0 - 5.0, (y0 + y1) / 2)).startswith("selected left")  # just beside counts
        assert selection.select_at((5.0, 5.0)) == "nobody there to select" and selection.selected.name == "left"
    finally:
        selection.close()


def test_clicks_in_the_demo_app_select_people(gl, biped, tmp_path, monkeypatch):
    from bifold_demo.app import DemoApp, DemoConfig
    from bifold_demo.sources import SourceSwitcher

    monkeypatch.chdir(tmp_path)
    scene = _scene(biped)
    app = DemoApp(DemoConfig(headless=True, frames=1, width=320, height=180, detect=False, uploads=None), scene=scene)
    app.run()
    app.selection.clips = _clips()
    app.switcher = SourceSwitcher({"scene": lambda: app.scene_source}, "scene")
    messages = []
    app.notify = lambda message, *args, **kwargs: messages.append(message)
    try:
        frame = app.scene_source.read()
        app.selection.enter(app.scene_source.scene, frame.size)
        boxes = {p.character.name: p.box for p in app.selection.people}
        app.selection.leave()
        x0, y0, x1, y1 = boxes["right"]
        app._handle_click(((x0 + x1) / 2, (y0 + y1) / 2))  # a click starts selection mode on that person
        assert app.selection.active and app.selection.selected.name == "right"
        assert messages[-1].startswith("selected right")
    finally:
        app.selection.close()


def test_the_window_hands_over_left_clicks():
    import cv2

    from bifold_demo.display.window import Window

    window = Window.__new__(Window)
    window._clicks = []
    window._on_mouse(cv2.EVENT_MOUSEMOVE, 1, 2, 0, None)
    window._on_mouse(cv2.EVENT_LBUTTONDOWN, 10, 20, 0, None)
    window._on_mouse(cv2.EVENT_LBUTTONUP, 10, 20, 0, None)
    assert window.poll_clicks() == [(10, 20)] and window.poll_clicks() == []
