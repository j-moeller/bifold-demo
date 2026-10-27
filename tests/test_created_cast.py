"""Created characters in the fair: they take over places of generic people, follow changes and leave again, and every
change is made where nobody sees it.

Synthetic avatars, clips and plain colour textures (no real faces); the tests on the prepared Rocketbox avatars skip
without ``BIFOLD_CHARACTERS_TEST_DIR``.
"""

import json
import math
import os
import random
import shutil
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from test_behaviour import synthetic_clips, synthetic_model
from test_yoga import synthetic_biped

from bifold_demo import pacing
from bifold_demo.characters import accessories, created
from bifold_demo.characters import behaviour as B
from bifold_demo.characters import camera_play as cp
from bifold_demo.characters.behaviour import MotionLibrary, classify_clip, stride_scale
from bifold_demo.characters.created import CharacterBook, CreatedCharacter, save
from bifold_demo.scene import Camera, Scene
from bifold_demo.scenes import created_cast, fair
from bifold_demo.scenes.created_cast import CreatedCast
from bifold_demo.selection import person_name

CHARACTERS = Path(os.environ.get("BIFOLD_CHARACTERS_TEST_DIR", "/nonexistent")).expanduser()
CATALOG = {
    "Woman": {"sex": "female", "age_group": "adult", "outfit": "casual", "skin_tone": 4, "conference_fit": 3},
    "Woman_2": {"sex": "female", "age_group": "adult", "outfit": "casual", "skin_tone": 8, "conference_fit": 3},
    "Man": {"sex": "male", "age_group": "adult", "outfit": "casual", "skin_tone": 2, "conference_fit": 3},
    "Man_2": {"sex": "male", "age_group": "adult", "outfit": "business", "skin_tone": 6, "conference_fit": 3},
    "Tall_Woman": {"sex": "female", "age_group": "adult", "outfit": "casual", "skin_tone": 5, "conference_fit": 3},
    "Girl": {"sex": "female", "age_group": "child", "outfit": "casual", "skin_tone": 6, "conference_fit": 1},
}
MODELS = {name: synthetic_biped() for name in ("Woman", "Woman_2", "Man", "Man_2")}
MODELS["Tall_Woman"] = synthetic_model(1.1)  # longer legs: another stride (and no Biped: no items, no camera play)
MODELS["Girl"] = synthetic_model(0.75)
for _name, _model in MODELS.items():
    _model.name = _name


class FakeLibrary:
    """Stands in for CharacterLibrary with the synthetic avatars and clips."""

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
        if name not in MODELS:
            raise KeyError(f"the avatar {name!r} is not prepared")
        return MODELS[name]

    def motions(self):
        return self._motions


class View:
    """Stands in for the renderer that drew the last frame: how many pixels of each person would show in it
    (``shown(person)``, or a number for everyone) and what is still to upload (``prepare`` says no ``uploads``
    times first)."""

    width, height = 1280, 720

    def __init__(self, shown=0, uploads=0):
        self.shown = shown if callable(shown) else (lambda person: shown)
        self.uploads = uploads
        self.counted, self.prepared = [], []

    def visible_pixels(self, people):
        self.counted.append(list(people))
        return [self.shown(p) for p in people]

    def prepare(self, *things):
        self.prepared.append(things)
        if self.uploads:
            self.uploads -= 1
            return False
        return True


def unseen():
    """A view in which nothing shows: every change is made at the next update."""
    return View(0)


LIBRARY = FakeLibrary()
# two of the audience, two visitors who walk up to the camera, two walkers and a wanderer (a child)
NAMES = ["Man", "Woman", "Woman", "Man", "Woman", "Man", "Girl"]


def small_fair(library=LIBRARY) -> Scene:
    slots = fair.people_slots()
    kinds = [kind for kind, *_ in slots]
    chosen = slots[:4] + [slots[kinds.index("walker")], slots[kinds.index("walker") + 1],
                          slots[kinds.index("wanderer")]]
    scene = Scene()
    scene.camera = Camera(position=fair.CAMERA_POSITION, target=fair.CAMERA_TARGET, fov=fair.CAMERA_FOV)
    fair.add_3d_people(scene, library, random.Random(fair.CROWD_SEED), slots=chosen, names=NAMES)
    return scene


@pytest.fixture(scope="module")
def built():
    """The small fair, once; each test gets a copy of the scene's list of objects (the people themselves are shared,
    and swapping them in and out does not change them, but for being held out of sight: they are shown again)."""
    scene = small_fair()
    return scene, dict(scene.objects)


def _restore(scene, objects):
    scene.objects = dict(objects)
    for person in objects.values():
        person.visible = True


@pytest.fixture
def scene(built):
    scene, objects = built
    _restore(scene, objects)
    scene.update(0.0, 0.0)
    return scene


def _plain(colour, size=16):
    image = np.zeros((size, size, 4), np.uint8)
    image[..., :3], image[..., 3] = colour, 255
    return image


def _character(name, avatar, colour=(200, 40, 40), **options):
    primitive = MODELS[avatar].primitives[0].name
    return CreatedCharacter(name, avatar, textures={primitive: _plain(colour)}, **options)


def _cast(tmp_path, *characters):
    book = CharacterBook(tmp_path / "created")
    for c in characters:
        book.save(c)
        time.sleep(0.002)  # distinct save times: the book lists them oldest first
    return CreatedCast(book, library=LIBRARY, threads=False)


def _people(scene):
    return [o for o in scene if hasattr(o, "skinned_model")]


def _kinds(scene):
    return [p.kind for p in _people(scene)]


def _shown(scene, name):
    """Who is drawn in the place ``name`` (``None``: nobody, the place is empty for a moment)."""
    person = scene.get(name)
    return person if person is not None and person.current_visible else None


def test_the_small_fair_has_places_for_created_characters(scene):
    names = {p.name: p.skinned_model.name for p in _people(scene)}
    assert names == {"audience1": "Man", "approach1": "Woman", "audience2": "Woman", "approach2": "Man",
                     "walker1": "Woman", "walker2": "Man", "wanderer1": "Girl"}
    assert all(p.recipe is not None and p.created is None for p in _people(scene))
    # visitors first, women's places first for a woman; never the child's
    order = [p.name for p in created_cast.places(scene, CATALOG["Woman_2"])]
    assert order == ["approach1", "approach2", "walker1", "walker2"]
    assert [p.name for p in created_cast.places(scene, CATALOG["Man_2"])][:2] == ["approach2", "approach1"]


def test_who_does_a_yoga_pose_gives_up_their_place_last(scene):
    from bifold_demo.characters.performer import Performer

    visitor = scene["approach1"]
    base = visitor.behaviour
    visitor.behaviour = Performer(base)
    try:
        assert created_cast.places(scene, CATALOG["Woman_2"])[0].name == "approach1"  # selected only
        visitor.behaviour.acts.append(object())  # doing (or done) a pose: their timing moved
        assert [p.name for p in created_cast.places(scene, CATALOG["Woman_2"])] == \
            ["approach2", "walker1", "walker2", "approach1"]
    finally:
        visitor.behaviour = base


def test_created_characters_take_over_places_exactly(scene, tmp_path):
    scene.update(40.0, 0.1)
    before = {p.name: p for p in _people(scene)}
    where = {name: p.current_position.copy() for name, p in before.items()}
    ada = _character("Ada", "Woman_2", items={"hat": "#204080"})
    cast = _cast(tmp_path, ada, _character("Ben", "Man_2", (40, 160, 60)))
    messages = cast.update(scene, now=0.0, view=unseen())
    assert messages == ["Ada joined the fair", "Ben joined the fair"]
    first, second = scene["approach1"], scene["approach2"]
    assert (first.created.name, first.skinned_model.name, second.created.name) == ("Ada", "Woman_2", "Ben")
    assert first is not before["approach1"] and all(o is not before["approach1"] for o in scene)
    assert list(scene.objects) == list(before)  # in the same places, in the same order
    assert cast.status() == "created: Ada, Ben in the fair"
    assert person_name(first) == "approach1 (Ada)" and person_name(scene["walker1"]) == "walker1 (Woman)"
    # the place's own behaviour (they are alike): the same position at the same time, the same camera play
    for name in ("approach1", "approach2"):
        person, original = scene[name], before[name]
        assert person.behaviour is original.behaviour and isinstance(person.behaviour, cp.CameraPlay)
        poses = [a for a in person.pose_adjusters if isinstance(a, cp.PlayPose)]
        assert len(poses) == 1 and poses[0].play is person.behaviour and poses[0].fig is not None
        person.update(40.0)
        assert np.allclose(person.current_position, where[name])
    # their looks: the textures and the hat
    head = MODELS["Woman_2"].primitives[0].name
    assert tuple(first.texture_overrides[head].rgba[0, 0, :3]) == (200, 40, 40)
    assert list(accessories.wearing(first)) == ["hat"] and not accessories.wearing(second)
    # the fair's guarantees hold: nobody walks through anybody
    people = _people(scene)
    assert fair.keep_apart([p.behaviour for p in people], _kinds(scene), random.Random(1), rounds=0) == 0
    for t in (41.0, 80.0, 200.0):
        scene.update(t, 0.1)  # they are drawn like everyone (posed, the hat on the head)


def test_a_change_is_made_only_where_nobody_sees_it(scene, tmp_path):
    """The one in the place goes out at a frame where nothing of them shows; the place stays empty until the one
    coming in would show nothing either; then they are there."""
    shown = {"old": 5000, "new": 5000}
    view = View(lambda p: shown["new" if p.created is not None else "old"])
    original = scene["approach1"]
    cast = _cast(tmp_path, _character("Ada", "Woman_2"))
    t = 10.0
    for _ in range(4):  # made (and on the GPU), but both are in the picture
        t += 0.5
        scene.update(t, 0.5)
        assert cast.update(scene, now=t, view=view) == []
        assert _shown(scene, "approach1") is original
    assert cast.status() == "created: 1 on the way" and view.counted[-1][0] is original
    shown["old"] = 0  # the visitor walks behind someone: out, while Ada would still be seen coming in
    scene.update(t + 0.5, 0.5)
    assert cast.update(scene, now=t, view=view) == []
    ada = scene["approach1"]
    assert ada is not original and ada.created.name == "Ada" and _shown(scene, "approach1") is None
    scene.update(t + 1.0, 0.5)  # (the scene does not draw her meanwhile)
    assert _shown(scene, "approach1") is None and cast.update(scene, now=t, view=view) == []
    shown["new"] = 0
    scene.update(t + 1.5, 0.5)
    assert cast.update(scene, now=t, view=view) == ["Ada joined the fair"]
    assert _shown(scene, "approach1") is ada and cast.status() == "created: Ada in the fair"
    scene.update(t + 2.0, 0.5)
    assert _shown(scene, "approach1") is ada and view.prepared and view.prepared[0][0] is ada  # on the GPU first


@pytest.mark.parametrize("pixels, joins", [(300, True), (600, False)])
def test_without_a_moment_nobody_sees_a_small_change_will_do_later(scene, tmp_path, pixels, joins):
    """No moment nobody sees for ``LINE_UP`` seconds: then a change showing at most ``SMALL`` of the picture (460 of
    1280 x 720 pixels: someone far down the hall) will do, never more."""
    assert created_cast.SMALL * 1280 * 720 == pytest.approx(460.8)
    view = View(pixels)
    cast = _cast(tmp_path, _character("Ada", "Woman_2"))
    t, start, messages = 10.0, 10.0, []
    scene.update(t, 0.0)
    while not messages and t < start + 3 * created_cast.LINE_UP:
        messages = cast.update(scene, now=t - start, view=view)
        t += 0.5
        scene.update(t, 0.5)
    if joins:
        assert messages == ["Ada joined the fair"]
        assert created_cast.LINE_UP <= t - start <= created_cast.LINE_UP + 1.0
    else:
        assert messages == [] and scene["approach1"].created is None and cast.status() == "created: 1 on the way"


def test_a_change_waits_for_what_drawing_it_needs_to_be_on_the_gpu(scene, tmp_path):
    view = View(0, uploads=3)  # three textures to upload, one a frame
    cast = _cast(tmp_path, _character("Ada", "Woman_2"))
    found = []
    for k in range(6):
        scene.update(1.0 + k, 1.0)
        found.append(cast.update(scene, now=float(k), view=view))
    assert found == [[], [], [], ["Ada joined the fair"], [], []]
    assert [len(p) for p in view.prepared[:4]] == [1, 1, 1, 1] and view.prepared[0][0].created.name == "Ada"
    assert len(view.counted) == 1  # her pixels are counted only once she could come in


def test_a_change_updates_in_place_and_a_delete_gives_the_place_back(scene, tmp_path):
    ada = _character("Ada", "Woman_2")
    cast = _cast(tmp_path, ada, _character("Ben", "Man_2"))
    originals = {p.name: p for p in _people(scene)}
    view = unseen()
    cast.update(scene, now=0.0, view=view)
    person = scene["approach1"]
    ada.textures = {MODELS["Woman_2"].primitives[0].name: _plain((30, 30, 220))}
    ada.items = {"glasses": "sunglasses"}
    cast.book.save(ada)  # saved again (by a creator in this process)
    assert cast.update(scene, now=0.5, view=view) == ["Ada has new looks"]
    assert scene["approach1"] is person and person.created is ada  # the same 3D person, new looks
    assert tuple(person.texture_overrides[MODELS["Woman_2"].primitives[0].name].rgba[0, 0, :3]) == (30, 30, 220)
    assert list(accessories.wearing(person)) == ["glasses"]
    ada.avatar = "Woman"  # another avatar: a new 3D person in the same place, with its behaviour
    cast.book.save(ada)
    assert cast.update(scene, now=1.0, view=view) == ["Ada is now Woman"]
    assert scene["approach1"] is not person and scene["approach1"].skinned_model.name == "Woman"
    assert scene["approach1"].behaviour is originals["approach1"].behaviour
    other = CharacterBook(tmp_path / "created")  # e.g. the creator, deleting Ben
    other.refresh(force=True)
    assert other.delete("Ben")
    assert cast.update(scene, now=1.2, view=view) == []  # the folder is looked at once a second
    assert cast.update(scene, now=2.5, view=view) == ["Ben left the fair"]
    assert _shown(scene, "approach2") is originals["approach2"] and scene["approach2"].created is None
    save(_character("Cleo", "Man_2"), tmp_path / "created")  # saved while the demo runs: the next free place
    assert cast.update(scene, now=4.0, view=view) == ["Cleo joined the fair"]
    assert scene["approach2"].created.name == "Cleo" and cast.status() == "created: Ada, Cleo in the fair"


def test_new_looks_and_leaving_wait_for_a_moment_nobody_sees_too(scene, tmp_path):
    ada = _character("Ada", "Woman_2")
    cast = _cast(tmp_path, ada)
    shown = {"n": 0}
    view = View(lambda p: shown["n"])
    cast.update(scene, now=0.0, view=view)
    person, head = scene["approach1"], MODELS["Woman_2"].primitives[0].name
    shown["n"] = 3000
    ada.textures = {head: _plain((30, 30, 220))}
    cast.book.save(ada)
    for k in range(3):
        scene.update(1.0 + k, 1.0)
        assert cast.update(scene, now=1.0 + k, view=view) == []
    assert tuple(person.texture_overrides[head].rgba[0, 0, :3]) == (200, 40, 40)  # in the picture: as she was
    assert view.prepared and list(view.prepared[-1]) == list(cast._places["approach1"].looks[2].gpu())
    shown["n"] = 0
    assert cast.update(scene, now=4.0, view=view) == ["Ada has new looks"]
    assert tuple(person.texture_overrides[head].rgba[0, 0, :3]) == (30, 30, 220)
    original = cast.placed["Ada"].original
    cast.book.delete("Ada")
    shown["n"] = 3000
    assert cast.update(scene, now=5.0, view=view) == [] and _shown(scene, "approach1") is person
    assert cast.status() == "created: Ada leaving"
    view.shown = lambda p: 0 if p is person else 3000  # she is hidden, the visitor coming back would be seen
    assert cast.update(scene, now=5.5, view=view) == ["Ada left the fair"]
    assert _shown(scene, "approach1") is None and scene["approach1"] is original and cast.status() is None
    view.shown = lambda p: 0
    assert cast.update(scene, now=6.0, view=view) == [] and _shown(scene, "approach1") is original
    assert "approach1" not in cast._places


def test_a_freed_place_goes_to_one_who_waits_without_its_own_person_in_between(scene, tmp_path):
    characters = [_character(f"P{k}", "Woman_2") for k in range(5)]
    cast = _cast(tmp_path, *characters)
    view = unseen()
    messages = cast.update(scene, now=0.0, view=view)
    assert "P4 waits for a free place in the fair" in messages and cast.waiting == ["P4"]
    place = cast.placed["P1"].person.name
    original = cast.placed["P1"].original
    cast.book.delete("P1")
    drawn = []
    for k in range(3):
        scene.update(1.0 + k, 1.0)
        messages = cast.update(scene, now=1.0 + k, view=view)
        drawn.append((_shown(scene, place), messages))
    assert drawn[0][1] == ["P1 left the fair", "P4 joined the fair"] and drawn[0][0].created.name == "P4"
    assert all(who is not original for who, _ in drawn) and not cast.waiting
    shown = {"n": 0}  # and when P4 cannot come in at once (in the picture): the place stays empty meanwhile
    view.shown = lambda p: shown["n"] if p.created is not None and p.created.name == "P5" else 0
    cast.book.save(_character("P5", "Woman_2"))
    cast.book.delete("P2")
    place, original = cast.placed["P2"].person.name, cast.placed["P2"].original
    shown["n"] = 3000
    drawn = []
    for k in range(4):
        if k == 3:
            shown["n"] = 0
        scene.update(10.0 + k, 1.0)
        messages = cast.update(scene, now=10.0 + k, view=view)
        drawn.append((_shown(scene, place), messages))
    assert [m for _, m in drawn] == [["P2 left the fair"], [], [], ["P5 joined the fair"]]
    assert [who for who, _ in drawn[:3]] == [None, None, None] and drawn[3][0].created.name == "P5"


def test_a_restart_makes_the_changes_that_are_ready_at_once(scene, tmp_path):
    cast = _cast(tmp_path, _character("Ada", "Woman_2"))
    view = View(5000)
    scene.update(30.0, 0.1)
    assert cast.update(scene, now=0.0, view=view) == [] and scene["approach1"].created is None
    scene.reset()  # R: the whole picture changes anyway
    scene.update(0.0, 0.0)
    assert cast.update(scene, now=0.1, view=view) == ["Ada joined the fair"]
    assert _shown(scene, "approach1").created.name == "Ada"


def test_a_sign_is_held_up_from_the_start(scene, tmp_path):
    cast = _cast(tmp_path, _character("Ada", "Woman_2", items={"sign": "welcome"}))
    scene.update(50.0, 0.1)
    cast.update(scene, now=0.0, view=unseen())
    ada = scene["approach1"]
    assert list(accessories.wearing(ada)) == ["sign"]
    assert all(ada.grip.weight(ada, t) == 1.0 for t in (0.0, 0.5, 50.0, 50.5, 51.0, 52.0, 300.0))
    scene.reset()  # and after R
    assert ada.grip.weight(ada, 0.0) == 1.0


def test_a_character_unlike_its_place_gets_a_behaviour_made_for_it(scene, tmp_path):
    scene.update(100.0, 0.1)
    originals = {p.name: p for p in _people(scene)}
    cast = _cast(tmp_path, _character("Ada", "Man_2"), _character("Ben", "Man"), _character("Cleo", "Tall_Woman"))
    messages = cast.update(scene, now=0.0, wait=True)
    assert messages == ["Ada joined the fair", "Ben joined the fair", "Cleo joined the fair"]
    ada, ben, cleo = scene["approach2"], scene["approach1"], scene["walker1"]
    assert (ada.created.name, ben.created.name, cleo.created.name) == ("Ada", "Ben", "Cleo")
    assert ada.behaviour is originals["approach2"].behaviour  # a man in a man's place: as it is
    for person, original in ((ben, originals["approach1"]), (cleo, originals["walker1"])):
        inner = cp.find_play(person.behaviour)
        inner = inner.inner if inner is not None else person.behaviour
        assert inner is not created_cast.slot_behaviour(original)
        assert person.recipe.style["sex"] == CATALOG[person.skinned_model.name]["sex"]
        assert fair.fits_period(inner.duration, fair.LOOP_PERIOD)
        # no sliding feet: its walk clips play at the speed its own legs need
        walks = [e for e in inner._entries if e.walk is not None and e.clip is not None]
        assert walks
        for entry in walks:
            stride = stride_scale(person.skinned_model, entry.clip)
            assert abs(entry.scale * entry.clip.metadata["speed"] * stride - 1.0) <= B.STRIDE_STRETCH + 1e-9
    assert cleo.recipe.style["stride_scale"] == pytest.approx(stride_scale(MODELS["Tall_Woman"],
                                                                           LIBRARY.motions().role("walk", "female")))
    assert ben.behaviour.style.sex == "male" and originals["approach1"].behaviour.style.sex == "female"
    # Ben plays with the camera in the visitor's place, his moves while he watches, the others' stay
    play = ben.behaviour
    assert isinstance(play, cp.CameraPlay)
    assert [a.play for a in ben.pose_adjusters if isinstance(a, cp.PlayPose)] == [play]
    windows = cp.watch_windows(play.inner, fair.LOOP_PERIOD)
    assert all(any(a - 1e-6 <= m.start and m.end <= b + 1e-6 for a, b in windows) for m in play.moves)
    assert cp.find_play(scene["audience1"].behaviour) is cp.find_play(originals["audience1"].behaviour)
    # nobody walks through anybody, with the new timings
    people = _people(scene)
    assert fair.keep_apart([p.behaviour for p in people], _kinds(scene), random.Random(1), rounds=0) == 0
    # and they follow the old timing about: the visitor is back at the same spot for most of the visit
    old = originals["approach1"].behaviour.positions(np.arange(0.0, fair.LOOP_PERIOD, 0.5))
    new = ben.behaviour.positions(np.arange(0.0, fair.LOOP_PERIOD, 0.5))
    assert np.nanmedian(np.hypot(*(new - old).T)) < 0.5


def test_newcomers_made_at_the_same_time_are_planned_around_each_other(scene, tmp_path, monkeypatch):
    seen, real = [], created_cast.stand_in

    def stand_in(original, created, library, people=(), **options):
        seen.append((created.name, {p.name: getattr(p.created, "name", None) for p in people}))
        return real(original, created, library, people, **options)

    monkeypatch.setattr(created_cast, "stand_in", stand_in)
    cast = _cast(tmp_path, _character("Ada", "Tall_Woman"), _character("Ben", "Tall_Woman"))
    cast.update(scene, now=0.0, wait=True)
    (first, before), (second, around) = seen
    assert (first, second) == ("Ada", "Ben") and before["approach1"] is None
    assert around["approach1"] == "Ada"  # Ben's timing was fitted in around Ada's, not the visitor's she replaces
    plan = created_cast._Plan(scene)
    control = pacing.Control()
    control.cancel()
    plan.offer("approach1", "someone", control)  # a job cancelled meanwhile: its person never counts
    assert "someone" not in plan.people()


def test_fit_in_keeps_people_apart_and_follows_the_old_timing():
    motions = LIBRARY.motions()
    cam = fair.CAMERA_POSITION
    a = B.walk_by([(-8.0, 0.0, -7.0), (8.0, 0.0, -7.0)], library=motions, camera=cam, rng=3, period=fair.LOOP_PERIOD,
                  return_mode="turn", offset=0.0)
    b = B.walk_by([(-8.0, 0.0, -7.0), (8.0, 0.0, -7.0)], library=motions, camera=cam, rng=3, period=fair.LOOP_PERIOD,
                  return_mode="turn", offset=17.3)
    assert fair.fit_in(b, [], like=a) == 0
    assert min(abs(b.offset - a.offset), b.duration - abs(b.offset - a.offset)) < 0.15  # back in step with it
    crossing = B.walk_by([(0.0, 0.0, -14.0), (0.0, 0.0, 0.0)], library=motions, camera=cam, rng=4,
                         period=fair.LOOP_PERIOD, offset=0.0)
    assert fair.fit_in(b, [crossing], like=a) == 0  # out of the other's way, as near the old timing as that allows
    assert fair.close_encounters([b, crossing], fair.LOOP_PERIOD, step=0.05, distance=fair.MIN_DISTANCE) == []


def test_never_a_childs_place_and_more_characters_wait(scene, tmp_path):
    characters = [_character(f"P{k}", "Woman_2") for k in range(5)]
    cast = _cast(tmp_path, *characters)
    messages = cast.update(scene, now=0.0, wait=True)
    assert messages == ["P4 waits for a free place in the fair"] + [f"P{k} joined the fair" for k in range(4)]
    assert scene["wanderer1"].created is None and scene["wanderer1"].skinned_model.name == "Girl"
    assert cast.waiting == ["P4"] and cast.status() == "created: P0, P1, P2, P3 in the fair; 1 waiting for a place"
    cast.book.save(characters[0])  # the fair is full and P4 still waits: said once, and not that there is no place
    assert cast.update(scene, now=1.5, wait=True) == ["P0 has new looks"] and cast.waiting == ["P4"]
    cast.book.delete("P1")  # a place comes free: the one who waited takes it
    assert cast.update(scene, now=3.0, wait=True) == ["P1 left the fair", "P4 joined the fair"] and not cast.waiting
    missing = _character("Zed", "Woman_2")
    missing.avatar = "Unknown_Avatar"
    cast.book.save(missing)
    cast.book.delete("P4")
    messages = cast.update(scene, now=4.0, wait=True)
    assert any(m.startswith("cannot put Zed in the fair: the avatar") for m in messages) and len(messages) == 2
    assert "P4 left the fair" in messages
    assert cast.status().endswith("1 cannot be shown") and cast.update(scene, now=5.0) == []  # said once


def test_a_place_freed_by_a_character_that_cannot_be_made_goes_to_one_who_waits(built, scene, tmp_path):
    zed = _character("Zed", "Woman_2")
    zed.avatar = "Unknown_Avatar"  # not prepared: it cannot be made, but takes a place while it is tried
    cast = _cast(tmp_path, *[_character(f"P{k}", "Woman_2") for k in range(1)], zed,
                 *[_character(f"P{k}", "Woman_2") for k in range(1, 4)])
    view = unseen()
    messages = cast.update(scene, now=0.0, view=view)
    assert "P3 waits for a free place in the fair" in messages and any(m.startswith("cannot put Zed") for m in messages)
    assert cast.update(scene, now=0.1, view=view) == ["P3 joined the fair"]  # the next frame: Zed's place is P3's
    assert sorted(cast.placed) == ["P0", "P1", "P2", "P3"] and not cast.waiting
    assert cast.status() == "created: P0, P1, P2, P3 in the fair; 1 cannot be shown"
    _restore(scene, built[1])  # the fair as it was
    again = _cast(tmp_path / "again", _character("P0", "Woman_2"), zed, *[_character(f"P{k}", "Woman_2")
                                                                         for k in range(1, 4)])
    messages = again.update(scene, now=0.0, wait=True)  # all at once (a scene nobody has seen yet)
    assert "P3 joined the fair" in messages and sorted(again.placed) == ["P0", "P1", "P2", "P3"]


def test_one_that_could_not_be_made_is_tried_again(scene, tmp_path, monkeypatch):
    zed = _character("Zed", "Woman_2")
    zed.avatar = "Later_Avatar"  # prepared while the demo runs
    cast = _cast(tmp_path, zed)
    view = unseen()
    tries = []
    real = LIBRARY.load
    monkeypatch.setattr(LIBRARY, "load", lambda name: (tries.append(name), real(name))[1])
    assert cast.update(scene, now=0.0, view=view)[0].startswith("cannot put Zed in the fair")
    assert cast.update(scene, now=created_cast.RETRY - 1.0, view=view) == [] and len(tries) == 1
    assert cast.update(scene, now=created_cast.RETRY + 0.5, view=view) == [] and len(tries) == 2  # (said once)
    MODELS["Later_Avatar"], CATALOG["Later_Avatar"] = MODELS["Woman_2"], CATALOG["Woman_2"]
    try:
        assert cast.update(scene, now=2 * created_cast.RETRY + 1.0, view=view) == ["Zed joined the fair"]
    finally:
        del MODELS["Later_Avatar"], CATALOG["Later_Avatar"]


def test_an_image_written_after_its_character_file_is_read_once_it_is_whole(scene, tmp_path):
    """Copied by hand or by a sync tool: character.json first, the image half written. It cannot be read at first,
    and is read again as soon as it changes (not only when character.json does)."""
    folder = tmp_path / "copy"
    save(_character("Ada", "Woman_2"), folder)
    png = next((folder / "Ada").glob("*.png"))
    whole = png.read_bytes()
    target = tmp_path / "created" / "Ada"
    target.mkdir(parents=True)
    shutil.copy2(folder / "Ada" / created.FILE, target / created.FILE)
    (target / png.name).write_bytes(whole[: len(whole) // 2])
    cast = CreatedCast(CharacterBook(tmp_path / "created"), library=LIBRARY, threads=False)
    view = unseen()
    messages = cast.update(scene, now=0.0, view=view)
    assert len(messages) == 1 and messages[0].startswith("cannot put Ada in the fair: OSError: cannot read")
    (target / png.name).write_bytes(whole)
    os.utime(target / png.name, ns=(time.time_ns(), time.time_ns() + 10**9))
    assert cast.update(scene, now=1.5, view=view) == ["Ada joined the fair"]


def test_a_character_deleted_while_it_is_made_is_no_error(scene, tmp_path, monkeypatch):
    cast = _cast(tmp_path, _character("Ada", "Woman_2"))

    def stand_in(original, created, *args, **kwargs):
        shutil.rmtree(created.folder)  # deleted meanwhile: its images are gone
        raise OSError(f"cannot read {created.folder}/f.png")

    monkeypatch.setattr(created_cast, "stand_in", stand_in)
    assert cast.update(scene, now=0.0, view=unseen()) == []
    assert cast.update(scene, now=1.5, view=unseen()) == [] and cast.status() is None


def test_looks_it_cannot_use_leave_the_character_as_it_was(scene, tmp_path, monkeypatch):
    ada = _character("Ada", "Woman_2", items={"hat": "#204080"})
    cast = _cast(tmp_path, ada)
    view = unseen()
    cast.update(scene, now=0.0, view=view)
    person, head = scene["approach1"], MODELS["Woman_2"].primitives[0].name
    real = created_cast.apply_looks

    def broken(person, created, wardrobe=None, t=None, looks=None):  # e.g. a creator's bug, on the main thread
        raise TypeError("cannot use int as a texture: 5")

    monkeypatch.setattr(created_cast, "apply_looks", broken)
    ada.textures = {head: _plain((30, 30, 220))}
    cast.book.save(ada)
    assert cast.update(scene, now=0.5, view=view) == [
        "cannot give Ada the new looks: TypeError: cannot use int as a texture: 5"]
    assert scene["approach1"] is person and cast.status() == "created: Ada in the fair"  # still in, as before
    assert tuple(person.texture_overrides[head].rgba[0, 0, :3]) == (200, 40, 40)
    assert cast.update(scene, now=2.0, view=view) == []  # not tried again until it changes (or a while later)
    monkeypatch.setattr(created_cast, "apply_looks", real)
    cast.book.save(ada)
    assert cast.update(scene, now=3.5, view=view) == ["Ada has new looks"]
    assert tuple(person.texture_overrides[head].rgba[0, 0, :3]) == (30, 30, 220)
    # her file broken by hand (a look of the wrong kind; a file half written): she stays as she was, and it is said
    file = ada.folder / created.FILE
    good = file.read_text()
    data = json.loads(good)
    data["items"], data["updated"] = {"hat": 5}, data["updated"] + 1.0
    file.write_text(json.dumps(data))
    os.utime(file, ns=(time.time_ns(), time.time_ns() + 10**9))
    messages = cast.update(scene, now=5.0, view=view)
    assert len(messages) == 1 and messages[0].startswith("cannot read the created character Ada: ValueError: the look")
    assert messages[0].endswith("(shown as it was)") and scene["approach1"] is person and "Ada" in cast.placed
    assert cast.update(scene, now=6.5, view=view) == []  # said once
    file.write_text(good)
    os.utime(file, ns=(time.time_ns(), time.time_ns() + 2 * 10**9))
    assert cast.update(scene, now=8.0, view=view) == []  # it reads well again, as it was: nothing changes to say
    assert scene["approach1"] is person and cast.book.problems == {} and cast.status() == "created: Ada in the fair"
    file.write_text("{")
    os.utime(file, ns=(time.time_ns(), time.time_ns() + 3 * 10**9))
    assert cast.update(scene, now=9.5, view=view)[0].startswith("cannot read the created character Ada")  # again


BAD_FILES = {
    "a list": "[]", "null": "null", "a string": '"Ada"', "a number": "5", "empty": "", "cut short": '{"name": "Ada",',
    "not text": b"\x89PNG\r\n\x1a\n\x00\xff", "deeply nested": "[" * 100000 + "]" * 100000,
    "format null": {"format": None}, "format a list": {"format": [1]}, "format a word": {"format": "x"},
    "outfit a number": {"outfit": 5}, "outfit a list": {"outfit": [1, 2]},
    "items a number": {"items": 5}, "items a list": {"items": [1, 2]}, "items a word": {"items": "abc"},
    "an item a mapping": {"items": {"hat": {"colour": "navy"}}},
    "textures a list": {"textures": ["a.png"]}, "textures a word": {"textures": "x"},
    "textures a number": {"textures": 5},
    "a texture a number": {"textures": {"f": 5}}, "a texture null": {"textures": {"f": None}},
    "a texture elsewhere": {"textures": {"f": "../Ben/f.png"}},
    "created a list": {"created": [1]}, "updated a mapping": {"updated": {}}, "updated NaN": {"updated": math.nan},
    "no name": {"name": None}, "name a number": {"name": 5}, "avatar a list": {"avatar": ["Woman_2"]},
    "face a number": {"face": 5},
}


@pytest.mark.parametrize("case", sorted(BAD_FILES))
def test_a_character_file_of_any_shape_never_stops_the_demo(scene, tmp_path, case):
    folder = tmp_path / "created" / "Bad"
    folder.mkdir(parents=True)
    bad = BAD_FILES[case]
    if isinstance(bad, dict):
        bad = json.dumps({"name": "Bad", "avatar": "Woman_2", "format": 1, **bad})
    (folder / created.FILE).write_bytes(bad if isinstance(bad, bytes) else bad.encode())
    with pytest.raises(ValueError):
        created.load(folder)
    cast = CreatedCast(CharacterBook(tmp_path / "created"), library=LIBRARY, threads=False)
    messages = cast.update(scene, now=0.0, view=unseen())
    assert len(messages) == 1 and messages[0].startswith("cannot read the created character Bad: ValueError")
    assert cast.status() == "created: 1 cannot be shown" and cast.update(scene, now=1.5, view=unseen()) == []


def test_a_folder_that_cannot_be_looked_into_leaves_the_characters_as_they_are(scene, tmp_path, monkeypatch):
    cast = _cast(tmp_path, _character("Ada", "Woman_2"))
    view = unseen()
    assert cast.update(scene, now=0.0, view=view) == ["Ada joined the fair"]
    real = Path.iterdir

    def iterdir(path):
        if path == cast.book.folder:
            raise PermissionError(13, "Permission denied", str(path))
        return real(path)

    monkeypatch.setattr(Path, "iterdir", iterdir)
    messages = cast.update(scene, now=1.5, view=view)
    assert len(messages) == 1 and messages[0].startswith(f"cannot look into {cast.book.folder}: [Errno 13]")
    assert "Ada" in cast.placed and cast.update(scene, now=3.0, view=view) == []
    monkeypatch.setattr(Path, "iterdir", real)
    assert cast.update(scene, now=4.5, view=view) == [] and cast.book.trouble is None and "Ada" in cast.placed


def test_notices_put_the_characters_who_join_together():
    notice = created_cast.notice
    assert notice(["Ada joined the fair"]) == "Ada joined the fair"
    assert notice(["Ada joined the fair", "Ben joined the fair"]) == "Ada and Ben joined the fair"
    assert notice(["P9 waits for a free place in the fair"] + [f"P{k} joined the fair" for k in range(6)]) == \
        "P9 waits for a free place in the fair; P0, P1, P2 and 3 more joined the fair"
    assert notice(["Ada left the fair", "Cleo joined the fair", "Ben left the fair", "cannot put Zed in the fair: x"]) \
        == "Ada and Ben left the fair; Cleo joined the fair; cannot put Zed in the fair: x"


def test_a_newcomer_who_cannot_be_kept_apart_is_reported(scene, monkeypatch, capsys):
    monkeypatch.setattr(fair, "fit_in", lambda *args, **kwargs: 2)
    person = created_cast.stand_in(scene["walker1"], _character("Ada", "Man_2"), LIBRARY, _people(scene))
    assert person.created.name == "Ada"
    assert "Ada in walker1 comes nearer than 0.6 m to 2 other(s)" in capsys.readouterr().out


def test_the_camera_play_of_a_newcomer_looks_at_the_others_as_planned(scene, monkeypatch):
    from bifold_demo.characters.performer import Performer

    walker = scene["walker2"]
    base = walker.behaviour
    walker.behaviour = Performer(base)  # doing yoga in selection mode (on the main thread, while the worker plans)
    seen = {}
    monkeypatch.setattr(cp, "join", lambda person, camera, **kwargs: seen.update(kwargs))
    try:
        fair.join_camera_play(scene["approach1"], _people(scene))
    finally:
        walker.behaviour = base
    assert base in seen["others"] and all(not isinstance(b, Performer) for b in seen["others"])
    assert {p.name for p in seen["players"]} == {"audience1", "audience2", "approach2"}


def test_a_scene_without_places_says_so(tmp_path):
    cast = _cast(tmp_path, _character("Ada", "Woman_2"))
    assert cast.update(Scene(), now=0.0) == [
        "created characters need the 3D people of the fair (this scene has no place for them)"]
    assert cast.waiting == ["Ada"]


def test_the_face_cast_and_selection_leave_their_faces_alone(scene, tmp_path, monkeypatch):
    from bifold_demo.characters import face_photo
    from bifold_demo.recognition.cast import FaceCast
    from bifold_demo.selection import Selection

    monkeypatch.setattr(face_photo, "head_layout", lambda model: object())  # every synthetic avatar has a face

    class Gallery:
        def refresh(self):
            pass

        def people(self):
            return {"Zoe": 1}

        def photos(self, name):
            return [tmp_path / "zoe.jpg"]

    cast = _cast(tmp_path, _character("Ada", "Woman_2"))
    cast.update(scene, now=0.0, view=unseen())
    faces = FaceCast(Gallery())
    faces._scene = scene
    monkey = []
    faces._paint = lambda character, *args: monkey.append(character.name)
    candidates = faces._candidates()
    assert scene["approach1"] not in candidates and candidates  # everyone else with a face may get one
    faces._cast()
    assert monkey and "approach1" not in monkey
    assert "a face of their own" in faces.put(scene["approach1"], "Zoe") and monkey.count("approach1") == 0
    selection = Selection()
    selection.cast = faces
    selection.selected = scene["approach1"]
    assert selection.wear("face", 0.0) == "approach1 (Ada) has a face of their own, from the character creator"


def test_restarting_and_reloading_keep_them(scene, tmp_path):
    cast = _cast(tmp_path, _character("Ada", "Woman_2"), _character("Ben", "Man_2"))
    cast.update(scene, now=0.0, view=unseen())
    scene.update(50.0, 0.1)
    scene.reset()  # R: time 0, everyone where they start
    assert cast.update(scene, now=0.2) == [] and scene["approach1"].created.name == "Ada"
    scene.update(0.0, 0.0)
    again = small_fair()  # the scene file reloaded: a new hall
    assert cast.update(again, now=0.4, wait=True) == []  # no news: they were in the fair already
    assert again["approach1"].created.name == "Ada" and again["approach2"].created.name == "Ben"
    assert scene["approach1"].created.name == "Ada"  # the old scene is left as it was


def test_threads_make_the_3d_people_and_swap_when_unseen(scene, tmp_path):
    book = CharacterBook(tmp_path / "created")
    book.save(_character("Ada", "Tall_Woman"))  # another stride: a behaviour is made for her, on the worker
    cast = CreatedCast(book, library=LIBRARY)
    workers = []
    real = created_cast.stand_in

    def stand_in(*args, **kwargs):
        workers.append((threading.current_thread(), getattr(pacing._local, "pacer", None)))
        return real(*args, **kwargs)

    created_cast.stand_in = stand_in
    shown = {"n": 4000}
    view = View(lambda p: shown["n"])
    try:
        messages, t = cast.update(scene, now=0.0, view=view), 20.0
        deadline = time.time() + 60
        while cast._jobs and time.time() < deadline:
            time.sleep(0.02)
            t += 0.25
            scene.update(t, 0.25)
            messages = cast.update(scene, now=0.1, view=view)
        assert not cast._jobs and messages == [] and scene["approach1"].created is None  # ready, but in the picture
        # made on the worker thread, paced (it leaves the main thread the GIL most of the time)
        assert len(workers) == 1 and workers[0][0] is not threading.main_thread() and workers[0][1] is not None
        shown["n"] = 0
        assert cast.update(scene, now=0.2, view=view) == ["Ada joined the fair"]
        assert scene["approach1"].created.name == "Ada"
    finally:
        created_cast.stand_in = real
        cast.close()


def test_waiting_for_the_work_hurries_it_and_closing_stops_it(scene, tmp_path, monkeypatch):
    rested = []

    def slow(original, created, library, people=(), **options):  # a long job, with pause points
        pacer = pacing._local.pacer
        for _ in range(400):
            start = time.perf_counter()
            while time.perf_counter() - start < 0.0005:
                pass
            pacing.pause()
        rested.append(pacer.rested)
        return created_cast.Character(MODELS["Woman_2"], behaviour=created_cast.slot_behaviour(original),
                                      name=original.name)

    monkeypatch.setattr(created_cast, "stand_in", slow)
    book = CharacterBook(tmp_path / "created")
    book.save(_character("Ada", "Woman_2"))
    cast = CreatedCast(book, library=LIBRARY)
    start = time.perf_counter()
    cast.update(scene, now=0.0, wait=True)  # a scene nobody has seen yet: no rests (0.2 s of work)
    assert time.perf_counter() - start < 0.5 and rested[0] < 0.05
    book.save(_character("Ben", "Man_2"))
    cast.update(scene, now=1.5)
    time.sleep(0.05)
    (thread,) = cast._pool._threads
    start = time.perf_counter()
    cast.close()  # quitting: the job stops at its next pause, the program can end at once
    thread.join(timeout=2.0)
    assert not thread.is_alive() and time.perf_counter() - start < 0.2 and len(rested) == 1


def test_problems_of_the_book_are_told_when_they_are_new(scene, tmp_path):
    cast = _cast(tmp_path, _character("Ada", "Woman_2"))
    view = unseen()
    assert cast.update(scene, now=0.0, view=view) == ["Ada joined the fair"]
    bob = tmp_path / "created" / "Bob"
    bob.mkdir()
    (bob / created.FILE).write_text('{"name": "Bob", "avatar": ')  # copied in, broken
    messages = cast.update(scene, now=1.5, view=view)  # told, though nothing else changed
    assert len(messages) == 1 and messages[0].startswith("cannot read the created character Bob: ValueError")
    assert cast.update(scene, now=3.0, view=view) == []
    assert cast.status() == "created: Ada in the fair; 1 cannot be shown"
    shutil.rmtree(bob)
    assert cast.update(scene, now=4.5, view=view) == []
    bob.mkdir()
    (bob / created.FILE).write_text("[]")  # broken again: told again
    assert cast.update(scene, now=6.0, view=view)[0].startswith("cannot read the created character Bob")


def test_a_loop_of_another_length_comes_in_unseen(scene, tmp_path):
    """A child's avatar in a grown-up's walker place: her loop is longer than the place's, so the two drift apart.
    Without a renderer, unseen is being out of the picture or hidden: the walker goes out then, and she comes in
    when she is out of the picture too (not when the walker hides at the end of the aisle while she is in the middle
    of it: she would appear out of thin air)."""
    occupant, people = scene["walker1"], _people(scene)
    slot = created_cast.slot_behaviour(occupant)
    probe = created_cast.stand_in(occupant, _character("Gail", "Girl"), LIBRARY, people)  # what she will do
    assert probe.recipe.style["age_group"] == "child" and slot.teleports()
    assert probe.behaviour.duration > slot.duration + 1.0
    times = np.arange(0.0, fair.LOOP_PERIOD, 0.1)
    old, new = slot.positions(times), probe.behaviour.positions(times)
    trap = next(float(t) for t, a, b in zip(times, old, new)  # the walker hides, she is in view
                if np.isnan(a[0]) and not np.isnan(b[0]) and fair.in_view(tuple(b)))

    def seen(person, t):
        state = person.behaviour(t)
        return bool(state.visible) and fair.in_view((float(state.position[0]), float(state.position[2])))

    t = trap - 0.2
    scene.update(t, 0.0)
    cast = _cast(tmp_path, _character("Ada", "Woman_2"), _character("Ben", "Man_2"), _character("Gail", "Girl"))
    start, gone, came = t, None, None
    while t < start + 3 * created_cast.LINE_UP and came is None:
        before = _shown(scene, "walker1")
        cast.update(scene, now=0.5)
        after = scene["walker1"]
        if gone is None and before is occupant and _shown(scene, "walker1") is None:
            gone = (t, seen(occupant, t))
        if after.created is not None and after.current_visible:
            came = (t, seen(after, t))
        t += 0.1
        scene.update(t, 0.1)
    assert gone is not None and not gone[1], gone  # out where the walker was not seen
    assert came is not None and not came[1] and came[0] > trap + 0.05, came  # in where she is not seen
    assert came[0] - start <= created_cast.LINE_UP  # (such moments come often in a walker's place)


def test_selection_mode_works_on_them(scene, tmp_path):
    from test_selection import _clips

    from bifold_demo.selection import Selection

    cast = _cast(tmp_path, _character("Ada", "Woman_2"))
    cast.update(scene, now=0.0, view=unseen())
    scene.update(12.0, 0.1)
    selection = Selection(clips=_clips())
    try:
        selection.enter(scene, (1280, 720))
        ada = scene["approach1"]
        assert any(p.character is ada for p in selection.people), "Ada can be selected"
        selection._select(next(p for p in selection.people if p.character is ada))
        assert selection.do("yogi_squat", 12.0) == "preparing Yogi Squat for approach1 (Ada) ..."
        assert selection.wear("hat", 12.0).startswith("approach1 (Ada): ")
        assert "hat" in accessories.wearing(ada)
    finally:
        selection.close()


def test_a_character_saved_while_the_demo_runs_joins_it(gl, tmp_path, monkeypatch):
    from bifold_demo.app import DemoApp, DemoConfig
    from bifold_demo.scene.renderer import Renderer
    from bifold_demo.sources import Frame

    monkeypatch.chdir(tmp_path)
    counted = []  # (the small fair has no booths to hide behind: here nobody shows, so she comes in at once)
    monkeypatch.setattr(Renderer, "visible_pixels", lambda self, people: counted.append(people) or [0] * len(people))
    scene = small_fair()
    (tmp_path / "created" / "Bad").mkdir(parents=True)  # a broken file there from the start: the demo goes on
    (tmp_path / "created" / "Bad" / created.FILE).write_text('{"name": "Bad", "avatar": "Woman_2", "items": 5}')
    app = DemoApp(DemoConfig(headless=True, frames=4000, width=160, height=90, detect=False, uploads=None,
                             created=str(tmp_path / "created")), scene=scene)
    app.created._library = LIBRARY
    messages, seen = [], {}
    app.notify = lambda message, *args, **kwargs: messages.append(message)

    @scene.on_update
    def save_and_watch(scene, t, dt):
        status = app.created.status() or ""
        if "saved" not in seen:  # the creator saves a character while the demo runs
            save(_character("Ada", "Woman_2"), tmp_path / "created")
            seen["saved"] = time.monotonic()
        elif "on the way" in status and "picked up" not in seen:
            seen["picked up"] = time.monotonic()
        elif scene["approach1"].created is not None and scene["approach1"].current_visible:
            seen["joined"] = time.monotonic()
            seen["status"] = app._status_lines(Frame(np.zeros((90, 160, 3), np.uint8), 1, 0.0, "scene"), None)
            app._running = False
        if "broken" not in seen and t > 0.5:  # and one broken while it runs
            (tmp_path / "created" / "Bad" / created.FILE).write_text("[1, 2")
            seen["broken"] = t

    app.run()
    assert "joined" in seen and messages[-1] == "Ada joined the fair"
    assert messages[0].startswith("cannot read the created character Bad: ValueError")
    assert seen["picked up"] - seen["saved"] < created.RESCAN + 1.5  # within a look at the folder (once a second)
    assert "created: Ada in the fair; 1 cannot be shown" in seen["status"]
    assert counted and any(p.created is not None for people in counted for p in people)  # by the app's renderer
    assert DemoConfig().created == "created"


def test_characters_who_join_together_get_one_notice(gl, tmp_path, monkeypatch, capsys):
    from bifold_demo.app import DemoApp, DemoConfig

    monkeypatch.chdir(tmp_path)
    for name, avatar in (("Ada", "Woman_2"), ("Ben", "Man_2")):  # saved before the demo starts
        save(_character(name, avatar), tmp_path / "created")
        time.sleep(0.002)
    scene = small_fair()
    app = DemoApp(DemoConfig(headless=True, frames=1, width=160, height=90, detect=False, uploads=None,
                             created=str(tmp_path / "created")), scene=scene)
    app.created._library = LIBRARY
    shown = []
    real = app.notify
    app.notify = lambda message, *args, **kwargs: (shown.append(message), real(message, *args, **kwargs))
    app.run()
    assert "Ada and Ben joined the fair" in shown and not any(m == "Ada joined the fair" for m in shown)
    out = capsys.readouterr().out
    assert "[demo] Ada joined the fair" in out and "[demo] Ben joined the fair" in out  # each in the console
    assert "[demo] Ada and Ben joined the fair" not in out
    # a frame or two apart, while the notice is still on the screen: it takes the new ones in
    app._message = (app._message[0], time.perf_counter() - 1.0, False)  # (that one is gone from the screen)
    updates = iter([["Cleo joined the fair"], ["Dan joined the fair"], [], ["Eve left the fair"]])
    app.created.update = lambda *args, **kwargs: next(updates)
    for _ in range(4):
        app._update_created()
    assert shown[-3:] == ["Cleo joined the fair", "Cleo and Dan joined the fair",
                          "Cleo and Dan joined the fair; Eve left the fair"]
    app._message = (app._message[0], time.perf_counter() - 1.0, False)  # gone from the screen: a notice of its own
    app.created.update = lambda *args, **kwargs: ["Fay joined the fair"]
    app._update_created()
    assert shown[-1] == "Fay joined the fair"

# -- real avatars ----------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_fair():
    from bifold_demo.characters.library import CharacterLibrary

    if not (CHARACTERS / "avatars").is_dir():
        pytest.skip(f"no prepared characters in {CHARACTERS} (set BIFOLD_CHARACTERS_TEST_DIR)")
    library = CharacterLibrary(CHARACTERS)
    if not fair.crowd_ready(library):
        pytest.skip("the prepared characters lack walk or idle clips")
    scene = Scene()
    scene.camera = Camera(position=fair.CAMERA_POSITION, target=fair.CAMERA_TARGET, fov=fair.CAMERA_FOV, near=0.1)
    fair.add_3d_people(scene, library, random.Random(fair.CROWD_SEED))
    return library, scene, dict(scene.objects)


def _adults(library, sex):
    return [n for n, e in sorted(library.available().items()) if e.get("sex") == sex and e.get("age_group") == "adult"]


def _outfit(library, avatar, colour):
    """Plain colour clothes: the avatar's body texture all in ``colour``."""
    body = next(p.name for p in library.load(avatar).primitives if p.name.endswith("_body"))
    return {body: _plain(colour, 64)}


def test_real_created_characters_walk_up_to_the_camera(real_fair, tmp_path):
    from bifold_demo.characters.crowd import same_style

    library, scene, objects = real_fair
    _restore(scene, objects)
    women, men = _adults(library, "female"), _adults(library, "male")
    if not women or len(men) < 2:
        pytest.skip("the prepared characters lack grown-ups")
    book = CharacterBook(tmp_path / "created")
    for name, avatar in (("Ada", women[0]), ("Ben", men[0]), ("Cleo", men[1])):
        book.save(CreatedCharacter(name, avatar, items={"hat": "#e07818"}, textures=_outfit(library, avatar,
                                                                                            (230, 0, 230))))
        time.sleep(0.002)
    scene.update(30.0, 0.1)
    before = {p.name: p for p in _people(scene)}
    men_places = sum(p.recipe.style["sex"] == "male" for p in _people(scene) if p.kind == "approach")
    cast = CreatedCast(book, library=library, threads=False)
    messages = cast.update(scene, now=0.0, wait=True)
    assert messages == ["Ada joined the fair", "Ben joined the fair", "Cleo joined the fair"]
    rebuilt = 0
    for name, placement in cast.placed.items():
        person, original = placement.person, placement.original
        assert person.kind == "approach" and scene[person.name] is person and person.created.name == name
        play = cp.find_play(person.behaviour)
        assert play is not None and [a.play for a in person.pose_adjusters if isinstance(a, cp.PlayPose)] == [play]
        inner = play.inner
        if same_style(person.recipe.style, original.recipe.style):  # the place's behaviour as it is
            assert person.behaviour is created_cast.slot_behaviour(original)
        else:  # made again: its own clips, about the old timing
            rebuilt += 1
            sex = person.recipe.style["sex"][0]
            assert {e.clip.name[:2] for e in inner._entries if e.clip is not None} == {f"{sex}_"}
            old = created_cast.planned_behaviour(original)
            assert inner.duration == pytest.approx(old.duration)
            assert _in_step(inner, old) > 0.35
        for entry in inner._entries:  # the feet do not slide
            if entry.walk is not None and entry.clip is not None:
                stride = stride_scale(person.skinned_model, entry.clip)
                assert abs(entry.scale * entry.clip.metadata["speed"] * stride - 1.0) <= B.STRIDE_STRETCH + 1e-9
        assert list(accessories.wearing(person)) == ["hat"]
    assert rebuilt >= 2 - men_places  # two men: one in a woman's place, when there is just one man's
    people = _people(scene)
    assert fair.keep_apart([p.behaviour for p in people], _kinds(scene), random.Random(1), rounds=0) == 0
    for name, person in before.items():  # nobody else changed
        if all(p.original is not person for p in cast.placed.values()):
            assert scene[name] is person


def _in_step(behaviour, old, step: float = 0.1) -> float:
    """The share of the loop (while both are seen) ``behaviour`` is within half a metre of ``old``."""
    times = np.arange(0.0, fair.LOOP_PERIOD, step)
    d = np.hypot(*(behaviour.positions(times) - old.positions(times)).T)
    return float(np.mean(d[~np.isnan(d)] <= 0.5))


def test_real_visitors_made_again_follow_each_place(real_fair):
    """In every visitor's place, an avatar of the other sex (a behaviour made again): its feet keep up with its
    walk, it watches the camera where the place's person did and plays then, nobody walks through anybody, and it is
    in step with the place's person for a good part of the visit (not all of it: its turns and clips differ)."""
    library, scene, objects = real_fair
    _restore(scene, objects)
    women, men = _adults(library, "female"), _adults(library, "male")
    if not women or not men:
        pytest.skip("the prepared characters lack grown-ups")
    people = _people(scene)
    visitors = [p for p in people if p.kind == "approach"]
    assert len(visitors) >= 3
    for original in visitors:
        avatar = women[0] if original.recipe.style["sex"] == "male" else men[0]
        person = created_cast.stand_in(original, CreatedCharacter("X", avatar), library, people)
        play, old = cp.find_play(person.behaviour), created_cast.planned_behaviour(original)
        assert play is not None and play.inner is not old and play.inner.duration == pytest.approx(old.duration)
        for entry in play.inner._entries:
            if entry.walk is not None and entry.clip is not None:
                stride = stride_scale(person.skinned_model, entry.clip)
                assert abs(entry.scale * entry.clip.metadata["speed"] * stride - 1.0) <= B.STRIDE_STRETCH + 1e-9
        windows = cp.watch_windows(play.inner, fair.LOOP_PERIOD)
        assert all(any(a - 1e-6 <= m.start and m.end <= b + 1e-6 for a, b in windows) for m in play.moves)
        crowd = [person if p is original else p for p in people]
        assert fair.keep_apart([created_cast.planned_behaviour(p) for p in crowd], [p.kind for p in crowd],
                               random.Random(1), rounds=0) == 0, original.name
        assert _in_step(play.inner, old) > 0.35, original.name


def test_real_children_joining_together_keep_apart(real_fair, tmp_path):
    """Seven children's avatars in grown-ups' places, made at the same time (their behaviours made again): each is
    fitted in around the others made before it, so nobody comes nearer anyone than ``MIN_DISTANCE``."""
    library, scene, objects = real_fair
    _restore(scene, objects)
    children = [n for n, e in sorted(library.available().items()) if e.get("age_group") == "child"]
    if not children:
        pytest.skip("the prepared characters lack children")
    book = CharacterBook(tmp_path / "created")
    for k in range(7):
        book.save(CreatedCharacter(f"Kid{k}", children[k % len(children)]))
        time.sleep(0.002)
    cast = CreatedCast(book, library=library, threads=False)
    assert len([m for m in cast.update(scene, now=0.0, wait=True) if m.endswith("joined the fair")]) == 7
    people = _people(scene)
    assert fair.keep_apart([created_cast.planned_behaviour(p) for p in people], [p.kind for p in people],
                           random.Random(1), rounds=0) == 0


def test_real_created_characters_are_drawn_in_their_looks(gl, real_fair, tmp_path):
    from bifold_demo.scene.renderer import Renderer

    library, scene, objects = real_fair
    _restore(scene, objects)
    women = _adults(library, "female")
    if not women:
        pytest.skip("the prepared characters lack grown-up women")
    book = CharacterBook(tmp_path / "created")
    book.save(CreatedCharacter("Ada", women[0], textures=_outfit(library, women[0], (230, 0, 230))))
    cast = CreatedCast(book, library=library, threads=False)
    place = created_cast.places(scene, library.available()[women[0]])[0]
    t = next(float(t) for t in np.arange(0.0, fair.LOOP_PERIOD, 1.0) if place.behaviour.watching(float(t)))
    renderer = Renderer(320, 180, samples=0)
    try:
        scene.update(t, 0.0)
        plain = renderer.render(scene).astype(int)
        cast.update(scene, now=0.0, wait=True)
        scene.update(t, 0.0)
        dressed = renderer.render(scene).astype(int)
    finally:
        renderer.release()

    def magenta(image):  # BGR
        return (image[..., 2] > 90) & (image[..., 0] > 90) & (image[..., 1] < 0.45 * image[..., 2])

    assert (magenta(dressed) & ~magenta(plain)).sum() > 150  # Ada in front of the camera, in her own clothes


def test_real_changes_in_the_picture_are_made_where_nothing_of_them_shows(gl, real_fair, tmp_path):
    """Rendered at 30 frames a second: three characters fill the visitors' places (at once, as at the start), two
    more join live (in walkers' places) and one of them leaves again. At every frame where a place changes, the
    picture changes by at most ``SMALL`` of it against the frame as it would have been without the change."""
    from bifold_demo.scene.renderer import Renderer

    library, scene, objects = real_fair
    _restore(scene, objects)
    women, men = _adults(library, "female"), _adults(library, "male")
    if len(women) < 2 or len(men) < 2:
        pytest.skip("the prepared characters lack grown-ups")
    book = CharacterBook(tmp_path / "created")
    for name, avatar, colour in (("Ada", women[0], (230, 0, 230)), ("Ben", men[0], (0, 200, 0)),
                                 ("Cleo", women[1], (250, 150, 0))):
        book.save(CreatedCharacter(name, avatar, textures=_outfit(library, avatar, colour)))
        time.sleep(0.002)
    cast = CreatedCast(book, library=library, threads=False)
    clock = {"t": 165.0, "frames": 0}  # the walkers of the near aisles are in the picture then
    scene.update(clock["t"], 0.0)
    assert len(cast.update(scene, now=0.0, wait=True)) == 3
    width, height, step = 320, 180, 1.0 / 30.0
    renderer = Renderer(width, height, samples=0)
    names = [p.name for p in _people(scene) if p.kind in created_cast.PLACES]
    changes = []

    def run_until(done, seconds):
        end = clock["t"] + seconds
        while not done():
            assert clock["t"] < end, (cast.status(), changes)
            clock["frames"] += 1
            before = {n: (scene.objects[n], scene.objects[n].visible) for n in names}
            cast.update(scene, now=1.5 + clock["frames"] * step, view=renderer)
            after = {n: (scene.objects[n], scene.objects[n].visible) for n in names}
            changed = [n for n in names if after[n] != before[n]]
            clock["t"] += step
            without = None
            if changed:  # the next frame as it would have been without the change
                for n in changed:
                    if after[n][0] is not before[n][0]:
                        after[n][0].visible = False
                    scene.objects[n] = before[n][0]
                    before[n][0].visible = before[n][1]
                scene.update(clock["t"], step)
                without = renderer.render(scene).astype(int)
                for n in changed:
                    if after[n][0] is not before[n][0]:
                        before[n][0].visible = False
                    scene.objects[n] = after[n][0]
                    after[n][0].visible = after[n][1]
            scene.update(clock["t"], step)
            image = renderer.render(scene)
            if without is not None:
                pixels = int((np.abs(image.astype(int) - without).max(axis=2) > 24).sum())
                changes.append((round(clock["t"], 2), changed, pixels))

    def settled():
        return all(p.settled for p in cast._places.values())

    try:
        scene.update(clock["t"], 0.0)
        renderer.render(scene)
        for name, avatar, colour in (("Dan", men[1], (0, 160, 230)), ("Eve", women[0], (240, 240, 0))):
            book.save(CreatedCharacter(name, avatar, textures=_outfit(library, avatar, colour)))
        run_until(lambda: len(cast.placed) == 5 and settled(), created_cast.LINE_UP + 10)
        dan = cast.placed["Dan"].person.name
        assert cast.placed["Dan"].person.kind == "walker" and cast.placed["Eve"].person.kind == "walker"
        book.delete("Dan")
        run_until(lambda: "Dan" not in cast.placed and dan not in cast._places, created_cast.LINE_UP + 10)
        assert sorted(cast.placed) == ["Ada", "Ben", "Cleo", "Eve"]
    finally:
        cast.close()
        renderer.release()
    print(changes)
    assert len(changes) >= 3 and max(px for *_, px in changes) <= created_cast.SMALL * width * height, changes


def test_the_created_folder_is_an_option():
    from bifold_demo.app import parse_args

    assert parse_args([]).created == "created"
    assert parse_args(["--created", "my_people"]).created == "my_people"
    assert parse_args(["--created", ""]).created is None
