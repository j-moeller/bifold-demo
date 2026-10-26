"""The trade-fair hall with animated 3D people, using synthetic avatars and clips (no downloads needed)."""

import math
import random

import pytest
from test_behaviour import synthetic_clips, synthetic_model

from bifold_demo.characters.behaviour import MotionLibrary, classify_clip
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
                assert y == pytest.approx(1.0 if p.name == "speaker" else 0.0)  # the speaker is on the stage
    assert people[0].skin_palette.shape[1:] == (4, 4)


def test_nobody_walks_through_anyone(hall):
    _, people = hall
    kinds = [_kind(p.name) for p in people]
    assert fair.keep_apart([p.behaviour for p in people], kinds, random.Random(1), seconds=60.0, rounds=0) == 0


def test_children_join_groups_and_business_people_staff_the_booths(hall):
    _, people = hall
    kids = [p for p in people if CATALOG[_avatar(p)]["age_group"] == "child"]
    assert kids and all(_kind(p.name) in ("wanderer", "chat") for p in kids)
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


def test_assign_people_prefers_formal_clothes_for_staff():
    slots = [("staff", None, False, True), ("wanderer", None, True, False), ("walker", None, False, False)]
    names = ["Man_Casual", "Boy", "Woman_Business"]
    assigned = fair.assign_people(names, slots, CATALOG, random.Random(0))
    assert assigned == ["Woman_Business", "Boy", "Man_Casual"]
