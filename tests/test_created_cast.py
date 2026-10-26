"""Created characters in the fair: they take over places of generic people, follow changes and leave again.

Synthetic avatars, clips and plain colour textures (no real faces); the tests on the prepared Rocketbox avatars skip
without ``BIFOLD_CHARACTERS_TEST_DIR``.
"""

import math
import os
import random
import time
from pathlib import Path

import numpy as np
import pytest
from test_behaviour import synthetic_clips, synthetic_model
from test_yoga import synthetic_biped

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
    and swapping them in and out does not change them)."""
    scene = small_fair()
    return scene, dict(scene.objects)


@pytest.fixture
def scene(built):
    scene, objects = built
    scene.objects = dict(objects)
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


def test_the_small_fair_has_places_for_created_characters(scene):
    names = {p.name: p.skinned_model.name for p in _people(scene)}
    assert names == {"audience1": "Man", "approach1": "Woman", "audience2": "Woman", "approach2": "Man",
                     "walker1": "Woman", "walker2": "Man", "wanderer1": "Girl"}
    assert all(p.recipe is not None and p.created is None for p in _people(scene))
    # visitors first, women's places first for a woman; never the child's
    order = [p.name for p in created_cast.places(scene, CATALOG["Woman_2"])]
    assert order == ["approach1", "approach2", "walker1", "walker2"]
    assert [p.name for p in created_cast.places(scene, CATALOG["Man_2"])][:2] == ["approach2", "approach1"]


def test_created_characters_take_over_places_exactly(scene, tmp_path):
    scene.update(40.0, 0.1)
    before = {p.name: p for p in _people(scene)}
    where = {name: p.current_position.copy() for name, p in before.items()}
    ada = _character("Ada", "Woman_2", items={"hat": "#204080"})
    cast = _cast(tmp_path, ada, _character("Ben", "Man_2", (40, 160, 60)))
    messages = cast.update(scene, now=0.0)
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


def test_a_change_updates_in_place_and_a_delete_gives_the_place_back(scene, tmp_path):
    ada = _character("Ada", "Woman_2")
    cast = _cast(tmp_path, ada, _character("Ben", "Man_2"))
    originals = {p.name: p for p in _people(scene)}
    cast.update(scene, now=0.0)
    person = scene["approach1"]
    ada.textures = {MODELS["Woman_2"].primitives[0].name: _plain((30, 30, 220))}
    ada.items = {"glasses": "sunglasses"}
    cast.book.save(ada)  # saved again (by a creator in this process)
    assert cast.update(scene, now=0.5) == ["Ada has new looks"]
    assert scene["approach1"] is person and person.created is ada  # the same 3D person, new looks
    assert tuple(person.texture_overrides[MODELS["Woman_2"].primitives[0].name].rgba[0, 0, :3]) == (30, 30, 220)
    assert list(accessories.wearing(person)) == ["glasses"]
    ada.avatar = "Woman"  # another avatar: a new 3D person in the same place, with its behaviour
    cast.book.save(ada)
    assert cast.update(scene, now=1.0) == ["Ada is now Woman"]
    assert scene["approach1"] is not person and scene["approach1"].skinned_model.name == "Woman"
    assert scene["approach1"].behaviour is originals["approach1"].behaviour
    other = CharacterBook(tmp_path / "created")  # e.g. the creator, deleting Ben
    other.refresh(force=True)
    assert other.delete("Ben")
    assert cast.update(scene, now=1.2) == []  # the folder is looked at once a second
    assert cast.update(scene, now=2.5) == ["Ben left the fair"]
    assert scene["approach2"] is originals["approach2"] and scene["approach2"].created is None
    save(_character("Cleo", "Man_2"), tmp_path / "created")  # saved while the demo runs: the next free place
    assert cast.update(scene, now=4.0) == ["Cleo joined the fair"] and scene["approach2"].created.name == "Cleo"
    assert cast.status() == "created: Ada, Cleo in the fair"


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
    cast.book.delete("P1")  # a place comes free: the one who waited takes it
    assert cast.update(scene, now=1.0, wait=True) == ["P1 left the fair", "P4 joined the fair"] and not cast.waiting
    missing = _character("Zed", "Woman_2")
    missing.avatar = "Unknown_Avatar"
    cast.book.save(missing)
    cast.book.delete("P4")
    messages = cast.update(scene, now=2.0, wait=True)
    assert messages[0] == "P4 left the fair" and messages[1].startswith("cannot put Zed in the fair: the avatar")
    assert cast.status().endswith("1 cannot be shown") and cast.update(scene, now=3.0) == []  # said once


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
    cast.update(scene, now=0.0)
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
    cast.update(scene, now=0.0)
    scene.update(50.0, 0.1)
    scene.reset()  # R: time 0, everyone where they start
    assert cast.update(scene, now=0.2) == [] and scene["approach1"].created.name == "Ada"
    scene.update(0.0, 0.0)
    again = small_fair()  # the scene file reloaded: a new hall
    assert cast.update(again, now=0.4, wait=True) == []  # no news: they were in the fair already
    assert again["approach1"].created.name == "Ada" and again["approach2"].created.name == "Ben"
    assert scene["approach1"].created.name == "Ada"  # the old scene is left as it was


def test_threads_make_the_3d_people_and_swap_when_ready(scene, tmp_path):
    book = CharacterBook(tmp_path / "created")
    book.save(_character("Ada", "Tall_Woman"))  # another stride: a behaviour is made for her, on the worker
    cast = CreatedCast(book, library=LIBRARY)
    replaced = scene["approach1"]
    try:
        messages, t = cast.update(scene, now=0.0), 20.0
        deadline = time.time() + 60
        while not messages and time.time() < deadline:
            time.sleep(0.02)
            t += 0.25
            scene.update(t, 0.25)
            messages = cast.update(scene, now=0.1)
        assert messages == ["Ada joined the fair"] and scene["approach1"].created.name == "Ada"
        # swapped in where the visitor she replaces was: no jump
        assert t - 20.0 <= created_cast.LINE_UP
        ada = scene["approach1"]
        ada.update(t)
        assert math.dist(ada.current_position, replaced.current_position) <= created_cast.JUMP + 1e-9
    finally:
        cast.close()


def test_selection_mode_works_on_them(scene, tmp_path):
    from test_selection import _clips

    from bifold_demo.selection import Selection

    cast = _cast(tmp_path, _character("Ada", "Woman_2"))
    cast.update(scene, now=0.0)
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
    from bifold_demo.sources import Frame

    monkeypatch.chdir(tmp_path)
    scene = small_fair()
    app = DemoApp(DemoConfig(headless=True, frames=2000, width=160, height=90, detect=False, uploads=None,
                             created=str(tmp_path / "created")), scene=scene)
    app.created._library = LIBRARY
    messages, seen = [], {}
    app.notify = lambda message, *args, **kwargs: messages.append(message)

    @scene.on_update
    def save_and_watch(scene, t, dt):
        if "saved" not in seen:  # the creator saves a character while the demo runs
            save(_character("Ada", "Woman_2"), tmp_path / "created")
            seen["saved"] = time.monotonic()
        elif scene["approach1"].created is not None:
            seen["joined"] = time.monotonic()
            seen["status"] = app._status_lines(Frame(np.zeros((90, 160, 3), np.uint8), 1, 0.0, "scene"), None)
            app._running = False

    app.run()
    assert "joined" in seen and messages[-1] == "Ada joined the fair"
    assert seen["joined"] - seen["saved"] < created.RESCAN + 1.5  # within a look at the folder (once a second)
    assert "created: Ada in the fair" in seen["status"]
    assert DemoConfig().created == "created"


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
    scene.objects = dict(objects)
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
            times = np.arange(0.0, fair.LOOP_PERIOD, 0.5)
            assert inner.duration == pytest.approx(old.duration)
            assert np.nanmedian(np.hypot(*(inner.positions(times) - old.positions(times)).T)) < 0.2
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


def test_real_created_characters_are_drawn_in_their_looks(gl, real_fair, tmp_path):
    from bifold_demo.scene.renderer import Renderer

    library, scene, objects = real_fair
    scene.objects = dict(objects)
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


def test_the_created_folder_is_an_option():
    from bifold_demo.app import parse_args

    assert parse_args([]).created == "created"
    assert parse_args(["--created", "my_people"]).created == "my_people"
    assert parse_args(["--created", ""]).created is None
