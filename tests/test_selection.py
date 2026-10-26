"""Selection mode: keys, people on screen, Tab order, losing the selected person, and yoga poses on request."""

import math
import time

import numpy as np
import pytest
from test_yoga import synthetic_biped

from bifold_demo.characters import yoga
from bifold_demo.characters.character import Character
from bifold_demo.characters.model import AnimationClip, CharacterState, Layer
from bifold_demo.characters.performer import Performer, choose_place
from bifold_demo.characters.yoga_clips import YogaClips, cache_path, model_key
from bifold_demo.display.window import BACKSPACE, BACKTAB, LEFT, RIGHT, TAB, key_code
from bifold_demo.scene import Camera, Scene
from bifold_demo.selection import Selection

SIZE = (640, 360)


def test_key_codes_from_every_backend():
    assert key_code(-1) is None
    assert key_code(ord("A")) == ord("a") and key_code(ord("1")) == ord("1") and key_code(27) == 27
    assert key_code(9) == TAB
    assert key_code(25) == BACKTAB  # macOS: Shift+Tab
    assert key_code(0xFE20 | 1 << 16) == BACKTAB  # GTK ISO_Left_Tab with Shift held
    assert key_code(0x01000002) == BACKTAB  # Qt Backtab
    assert key_code(0x250000) == LEFT and key_code(0x270000) == RIGHT  # Windows arrows
    assert key_code(0xF702) == LEFT and key_code(0x01000014) == RIGHT  # macOS, Qt
    assert key_code(0xFF51) == LEFT  # GTK: not "q" (which would quit)
    assert key_code(27 | 0x10 << 16) == 27  # GTK Escape with NumLock on
    assert key_code(ord("y") | 0x10 << 16) == ord("y")
    assert key_code(127) == BACKSPACE and key_code(0xFF08) == BACKSPACE
    assert key_code(0x700000) is None and key_code(0xF704) is None and key_code(0x01000030) is None  # F1 keys


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
    for t in (5.0, 5.3, 7.0, 12.0):
        assert performer(t).position == pytest.approx((5.0, 0.0, 0.0))  # stays put while doing it
    assert performer(5.6).heading == pytest.approx(0.0) and performer(5.6).head_yaw == pytest.approx(0.0)
    assert performer(7.6).layers[-1].time == pytest.approx(2.0)  # the clip runs after the turn
    assert act.end == pytest.approx(5.0 + 0.6 + 8.0)
    after = act.end + 2.0
    assert performer(after).position == pytest.approx((5.0 + 2.0, 0.0, 0.0))  # walking on from where it stopped
    assert performer(after).heading == pytest.approx(90.0)
    times = [4.0, 6.0, 20.0]
    assert np.allclose(performer.positions(times), [[4.0, 0.0], [5.0, 0.0], [20.0 - 8.6, 0.0]])


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
    assert state.position == pytest.approx((0.0, 0.0, 0.0))
    assert performer.positions([t])[0] == pytest.approx((0.0, 0.0))


def test_the_place_for_a_pose_keeps_the_mat_clear():
    points = np.zeros((2, 7, 2))
    points[:, :, 1] = np.linspace(0.0, 1.5, 7)  # the pose reaches 1.5 m ahead of the person
    nobody = np.zeros((2, 0, 2))
    assert choose_place((0, 0, 0), [90.0, -90.0], points, nobody, np.zeros((0, 4))) == (90.0, (0.0, 0.0, 0.0))
    in_the_way = np.tile((1.4, 0.0), (2, 1, 1))  # someone where the head would go
    heading, spot = choose_place((0, 0, 0), [90.0, -90.0], points, in_the_way, np.zeros((0, 4)))
    assert heading == -90.0 and spot == (0.0, 0.0, 0.0)  # seen from the other side instead
    wall = np.array([[-3.0, 0.5, 3.0, 0.7]])  # a counter just in front, a person behind
    heading, spot = choose_place((0, 0, 0), [0.0], points, nobody, wall)
    assert abs(heading) > 15.0 or spot != (0.0, 0.0, 0.0)


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


def _run_until(scene, selection, t, until, step=1 / 15):
    messages = []
    for _ in range(3000):
        t += step
        scene.update(t, step)
        message = selection.update(scene, SIZE)
        if message:
            messages.append(message)
        if until(t):
            return t, messages
        if not messages and selection._requests:
            time.sleep(0.005)  # let the clip thread work
    raise AssertionError("timed out")


def test_number_keys_make_the_selected_person_do_yoga(biped):
    scene = _scene(biped)
    selection = Selection(_clips())
    try:
        selection.enter(scene, SIZE)
        person = selection.selected
        handled, message = selection.handle_key(ord("5"), 0.0)  # Warrior 2: seen from the front
        assert handled and message.startswith("preparing Warrior 2")
        assert "preparing Warrior 2" in selection.describe(person, 0.0)
        t, messages = _run_until(scene, selection, 0.0, lambda t: isinstance(person.behaviour, Performer)
                                 and person.behaviour.current(t) is not None)
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
