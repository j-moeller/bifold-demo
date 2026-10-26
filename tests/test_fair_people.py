"""The trade-fair hall with animated 3D people, using synthetic avatars and clips (no downloads needed)."""

import math
import random
from functools import partial

import pytest
from test_behaviour import synthetic_clips, synthetic_model

from bifold_demo.characters.behaviour import MotionLibrary, classify_clip, fits_period, walk_by
from bifold_demo.scenes import fair

CATALOG = {
    "Woman_Business": {"sex": "female", "age_group": "adult", "outfit": "business", "skin_tone": 7, "conference_fit": 3},
    "Man_Business": {"sex": "male", "age_group": "adult", "outfit": "business", "skin_tone": 2, "conference_fit": 3},
    "Woman_Casual": {"sex": "female", "age_group": "adult", "outfit": "casual", "skin_tone": 4, "conference_fit": 3},
    "Man_Casual": {"sex": "male", "age_group": "adult", "outfit": "casual", "skin_tone": 9, "conference_fit": 2},
    "Woman_Older": {"sex": "female", "age_group": "adult", "age": 64, "outfit": "smart_casual", "skin_tone": 3,
                    "conference_fit": 3},
    "Man_Older": {"sex": "male", "age_group": "older", "outfit": "casual", "skin_tone": 5, "conference_fit": 2},
    "Girl": {"sex": "female", "age_group": "child", "outfit": "casual", "skin_tone": 6, "conference_fit": 1},
    "Boy": {"sex": "male", "age_group": "child", "outfit": "casual", "skin_tone": 2, "conference_fit": 1},
}


MODELS = {n: synthetic_model(0.75 if e["age_group"] == "child" else 1.0) for n, e in CATALOG.items()}


class FakeLibrary:
    """Stands in for CharacterLibrary with synthetic avatars and clips."""

    def __init__(self):
        clips = synthetic_clips(full=True)
        self._motions = MotionLibrary(clips)
        self._roles = {}
        for name in clips:
            role = classify_clip(name)
            if role:
                self._roles.setdefault(role, []).append(name)

    def available(self):
        return CATALOG

    def roles(self):
        return self._roles

    def load(self, name):
        return MODELS[name]

    def motions(self):
        return self._motions


@pytest.fixture(scope="module")
def hall():
    original = fair.CharacterLibrary
    fair.CharacterLibrary = FakeLibrary
    try:
        scene = fair.build_scene()
    finally:
        fair.CharacterLibrary = original
    people = [o for o in scene if hasattr(o, "skinned_model")]
    return scene, people


def _kind(name):
    return name.rstrip("0123456789").split("_")[0].rstrip("0123456789")


def test_the_hall_fills_with_animated_people(hall):
    scene, people = hall
    assert len(people) == len(fair.people_slots())
    kinds = {_kind(p.name) for p in people}
    assert {"approach", "walker", "wanderer", "chat", "staff", "guest", "speaker"} <= kinds
    assert "visitor1" not in scene  # no 2D cut-outs
    for t in (0.0, 7.5, 31.0):
        scene.update(t, 0.1)
        for p in people:
            if p.current_visible:
                x, y, z = p.current_position
                assert abs(x) <= fair.HALL_WIDTH / 2 and fair.HALL_FRONT - fair.HALL_LENGTH <= z <= fair.HALL_FRONT
                raised = {"speaker": fair.STAGE_HEIGHT, "staff": fair.BOOTH_FLOOR}.get(_kind(p.name), 0.0)
                assert y == pytest.approx(raised)  # the speaker on the stage, exhibitors on their booth platform
    assert people[0].skin_palette.shape[1:] == (4, 4)


def test_nobody_walks_through_anyone(hall):
    _, people = hall
    behaviours = [p.behaviour for p in people]
    for p in people:
        if _kind(p.name) in fair.MOVING:  # so the whole hall repeats after LOOP_PERIOD
            assert fits_period(p.behaviour.duration, fair.LOOP_PERIOD), (p.name, p.behaviour.duration)
    assert fair.close_encounters(behaviours, 1.5 * fair.LOOP_PERIOD, step=0.05) == []
    kinds = [_kind(p.name) for p in people]
    assert fair.keep_apart(behaviours, kinds, random.Random(1), rounds=0) == 0


def test_keep_apart_moves_people_out_of_each_others_way():
    clips = synthetic_clips(full=True)
    library = MotionLibrary(clips)
    make = partial(walk_by, return_mode="teleport", gap=(1.0, 1.2), period=fair.LOOP_PERIOD, library=library)
    walkers = [make([(-6, 0, 0), (6, 0, 0)], rng=1), make([(0, 0, -6), (0, 0, 6)], rng=2)]
    assert all(fits_period(w.duration, fair.LOOP_PERIOD) for w in walkers)
    assert walkers[0].duration == pytest.approx(walkers[1].duration)
    for w in walkers:
        w.offset = 0.0  # both reach the crossing at the same moment
    assert fair.close_encounters(walkers, fair.LOOP_PERIOD)
    assert fair.keep_apart(walkers, ["walker", "walker"], random.Random(0), rounds=0) == 1  # counts, even unfixed
    assert fair.keep_apart(walkers, ["walker", "walker"], random.Random(0)) == 0
    assert fair.close_encounters(walkers, 2 * fair.LOOP_PERIOD, step=0.05) == []


def test_walkers_vanish_only_out_of_view(hall):
    _, people = hall
    hidden = 0
    for p in people:
        b = p.behaviour
        for start, end in b.teleports():
            for t, side in ((start - 0.05, "vanishes"), (end + 0.05, "appears")):
                state = b(t - b.offset)
                assert state.visible
                assert not fair.in_view((state.position[0], state.position[2])), f"{p.name} {side} in view"
            hidden += 1
    assert hidden  # the near aisle's walkers do vanish at the walls ...
    assert fair.in_view((fair.HALL_WIDTH / 2 - 1, -20.5))  # ... where the far aisles' walkers turn round instead
    assert not fair.in_view((fair.HALL_WIDTH / 2 - 1, -7.0))
    assert fair.in_view((0.0, -12.0)) and not fair.in_view((0.0, 9.0))


def test_every_wander_area_is_in_view():
    for (x0, z0), (x1, z1) in fair.WANDER_AREAS:
        assert fair.in_view(((x0 + x1) / 2, (z0 + z1) / 2), margin=0.0)


def test_nobody_talks_to_their_double(hall):
    _, people = hall
    for group in fair.conversation_groups(fair.people_slots()):
        avatars = [_avatar(people[i]) for i in group]
        assert len(set(avatars)) == len(avatars), [people[i].name for i in group]


def test_children_join_groups_and_business_people_staff_the_booths(hall):
    _, people = hall
    kids = [p for p in people if CATALOG[_avatar(p)]["age_group"] == "child"]
    assert kids and all(_kind(p.name) in ("wanderer", "chat") for p in kids)
    groups = len(fair.CHAT_GROUPS)
    assert sum(_kind(p.name) == "chat" for p in kids) == min(len(kids), groups)  # with grown-ups, not alone
    staff = [p for p in people if _kind(p.name) in ("staff", "speaker")]
    formal = sum(CATALOG[_avatar(p)]["outfit"] in ("business", "smart_casual") for p in staff)
    assert formal >= 2  # as many as the (small) catalog offers before repeating


def _avatar(person):
    return next(n for n, m in MODELS.items() if m is person.skinned_model)


def test_nobody_in_the_hall_looks_at_the_camera(hall):
    _, people = hall
    cx, _, cz = fair.CAMERA_POSITION
    worst = 180.0
    for p in people:
        behaviour = getattr(p.behaviour, "inner", p.behaviour)
        if not behaviour.look_feasible:
            pytest.fail(f"{p.name}: the head plan cannot avoid the camera")
        for k in range(600):
            t = k * 0.1
            state = behaviour(t)
            dx, dz = cx - state.position[0], cz - state.position[2]
            if not state.visible or math.hypot(dx, dz) < 0.75:
                continue
            off = abs((behaviour.face_heading(t) - math.degrees(math.atan2(dx, dz)) + 180) % 360 - 180)
            worst = min(worst, off)
    assert worst >= 35.0 - 1e-6


def test_without_prepared_characters_the_hall_shows_2d_people(capsys, monkeypatch):
    monkeypatch.setattr(fair, "people_for_scene", lambda folder: [])
    scene = fair.build_scene()
    assert not any(hasattr(o, "skinned_model") for o in scene)
    assert "bifold-characters fetch" in capsys.readouterr().out


def test_a_broken_character_cache_falls_back_to_2d_people(capsys, monkeypatch):
    def broken(*args, **kwargs):
        raise OSError("damaged avatar file")

    monkeypatch.setattr(fair, "CharacterLibrary", FakeLibrary)
    monkeypatch.setattr(fair, "populate", broken)
    monkeypatch.setattr(fair, "people_for_scene", lambda folder: [])
    scene = fair.build_scene()
    assert not any(hasattr(o, "skinned_model") for o in scene)
    assert "booth1_wall" in scene
    assert "damaged avatar file" in capsys.readouterr().out


def test_carpets_lie_below_the_blob_shadows():
    from bifold_demo.scene.renderer import SHADOW_LIFT

    scene = fair.build_scene(people=[])
    carpets = [o for o in scene if o.name.startswith("carpet")]
    assert carpets and all(0.0 < o.current_position[1] < SHADOW_LIFT - 0.004 for o in carpets)


def test_assign_people_prefers_formal_clothes_for_staff():
    slots = [("staff", None, False, True), ("wanderer", None, True, False), ("walker", None, False, False)]
    names = ["Man_Casual", "Boy", "Woman_Business"]
    assigned = fair.assign_people(names, slots, CATALOG, random.Random(0))
    assert assigned == ["Woman_Business", "Boy", "Man_Casual"]


def test_assign_people_keeps_doubles_out_of_conversations():
    slots = [("walker", None, False, False), ("chat1_", None, False, False), ("chat1_", None, False, False),
             ("chat1_", None, True, False), ("staff", None, False, True), ("guest", None, False, False),
             ("wanderer", None, True, False)]
    names = ["Man_Casual", "Woman_Casual", "Woman_Casual", "Man_Business", "Man_Business", "Girl", "Man_Older"]
    for seed in range(5):
        assigned = fair.assign_people(names, slots, CATALOG, random.Random(seed))
        assert sorted(assigned) == sorted(names)
        assert assigned[3] == "Girl"  # the child joins the group rather than wandering alone
        assert assigned[4] == "Man_Business"  # the exhibitor keeps the suit
        for group in fair.conversation_groups(slots):
            assert len({assigned[i] for i in group}) == len(group), assigned
