"""Created characters in the fair: each takes over the place of one of the generic people who walk about.

A character made in the character creator (:mod:`bifold_demo.characters.created`) appears in the fair in the place of
a generic grown-up and keeps that place's behaviour: where and when they walk, so the spacing and the timing the fair
planned (``fair.keep_apart``) and the camera play stay valid. The best places go first (:data:`PLACES`): the visitors
who walk up the aisle to the camera (``approach``: they are seen best, watch the camera and play with it), then the
people walking by along the cross aisles (``walker``), then those wandering about (``wanderer``). Never a child's place:
its behaviour is made for a child's size. Within a kind, places whose person is like the created character's avatar
(same sex and age group) come first, because there the behaviour fits the avatar as it is
(:func:`~bifold_demo.characters.crowd.same_style`: the same clips, pace and stride; Rocketbox grown-ups of one sex
have the same legs), and the new 3D person simply gets it: the same position at the same time, doing the same.
People doing a yoga pose in selection mode, or who did one, give up their place last.

Elsewhere the behaviour is made again for the new avatar (:meth:`~bifold_demo.characters.crowd.Recipe.remake`: the
same path, pauses and seed, but its own clips, pace and stride, so the feet do not slide), its timing is fitted in
among everyone else's (:func:`~bifold_demo.scenes.fair.fit_in`: never nearer anyone than ``MIN_DISTANCE``, following
the old timing as closely as that allows; the people made for other places before it count too, so newcomers made at
the same time keep apart as well) and, in front of the camera, its camera play is planned around the others' moves,
which stay as they are (:func:`~bifold_demo.scenes.fair.join_camera_play`). It walks as fast as the place's person did
when its clips can keep up (within ``CADENCE``), so the timings differ only by its own turns and clips.

Changes nobody sees. A viewer must never see a person turn into someone else, jump or pop up, so every change of a
place (a character stepping in, another avatar, new looks, a character leaving) is made between two frames where
nothing of it shows: the person who leaves is taken out at a frame where no pixel of them is seen (out of the picture,
hidden while walkers come back to the start of their aisle, or behind a booth, a plant or someone else), the place
stays empty until the person coming in would show no pixel either, and then they are there. Changed looks go on the
same way. :meth:`~bifold_demo.scene.renderer.Renderer.visible_pixels` counts those pixels against the frame just
drawn; without a renderer only being out of the picture or hidden counts (:func:`estimate`). On the fair (its six
minute loop, rendered), the walkers' places have such moments every few seconds (the near ones leave the picture, the
far ones pass behind booths), the visitors' and wanderers' places now and then, when someone passes in front of them;
the visitor walking up the middle of the aisle never. So after ``LINE_UP`` seconds of scene time without such a
moment, a frame where the change shows at most ``SMALL`` of the picture (460 of 1280 x 720 pixels: the person far down
the hall, or nearly hidden) will do; never more. When the whole picture changes anyway (the scene restarted with R, a
new scene), the changes are made at once. Before someone comes in, what drawing them needs goes to the GPU, one
texture a frame (:meth:`~bifold_demo.scene.renderer.Renderer.prepare`), so that the frame they come in has none of
it left to do.

The new 3D person keeps the place's name (``approach1``) and ``kind``; ``person.created`` is the
:class:`~bifold_demo.characters.created.CreatedCharacter`, so other parts can tell (the known faces of
``recognition.cast`` leave it alone, selection mode shows the character's name). It wears the character's textures
and accessories (:func:`~bifold_demo.characters.created.apply_looks`; a sign is held up from the start). The person
whose place it took waits outside the scene and comes back when the character is deleted.

:class:`CreatedCast` keeps a scene in step with a :class:`~bifold_demo.characters.created.CharacterBook` while the
demo runs: a newly saved character takes the next free place, a changed one is updated (new looks on the same 3D
person; another avatar: a new 3D person in the same place), a deleted one gives the place back to its original person
(or to a character waiting for a place). Making a 3D person (loading the avatar, painting its looks, making its
behaviour again) takes from a few milliseconds to under a second of work, so it runs on a worker thread, paced
(:mod:`bifold_demo.pacing`: it rests three quarters of the time, else it would take the GIL from the main thread for
5 ms at a time and the frames would come late); when someone waits for it (a new scene, nobody has seen yet), it
hurries. A new scene (the scene file reloaded) gets the characters again, all at once. What cannot be shown is left
out and the rest goes on: a character whose avatar is not prepared or whose images cannot be read (its place goes to
one waiting for a place; it is tried again every ``RETRY`` seconds, and at once when its files change), new looks it
cannot use (it stays as it was), an item its avatar cannot wear. :func:`notice` puts messages on one line.
"""

from __future__ import annotations

import concurrent.futures
import functools
import math
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

from .. import pacing
from ..characters import camera_play
from ..characters.behaviour import fits_period
from ..characters.character import Character
from ..characters.created import CharacterBook, CreatedCharacter, apply_looks, prepare_looks
from ..characters.crowd import Recipe, age_group, avatar_style, same_style, sex
from ..characters.performer import Performer
from . import fair

PLACES = ("approach", "walker", "wanderer")  # kinds of places created characters take, best first
CADENCE = 1.25  # walk clips of a behaviour made again play at most this much faster (or slower) than their own pace
LINE_UP = 20.0  # seconds of scene time a change waits for a frame where nothing of it shows, before less will do
SMALL = 0.0005  # the most of the picture a change may show after LINE_UP: 460 pixels of 1280 x 720
RETRY = 10.0  # seconds after which a character that could not be put in the fair is tried again
PICTURE = (1280, 720)  # the picture estimate() counts pixels of


def slot_behaviour(person):
    """The behaviour of ``person``'s place: theirs, without what selection mode wraps around it for yoga poses."""
    behaviour = getattr(person, "behaviour", None)
    return behaviour.base if isinstance(behaviour, Performer) else behaviour


def planned_behaviour(person):
    """``person``'s behaviour as ``fair.keep_apart`` planned it (:func:`~bifold_demo.scenes.fair.planned_behaviour`:
    without selection mode's and the camera play's additions)."""
    return fair.planned_behaviour(person)


def _a_place(person) -> bool:
    """Whether ``person`` (one of the fair's own people) has a place a created character may take."""
    recipe = getattr(person, "recipe", None)
    return getattr(person, "kind", None) in PLACES and recipe is not None and getattr(person, "created", None) is None \
        and recipe.style.get("age_group") != "child"


def places(scene, entry: dict | None = None, originals: dict | None = None) -> list:
    """The generic people of ``scene`` whose place a created character may take, best first (see the module docs);
    with ``entry`` (the catalog entry of the character's avatar) the people like it first within each kind. People
    doing a yoga pose in selection mode, or who did one (their timing moved), come last. ``originals``: the fair's
    own people of the places where someone else is now (by place name), who count instead."""
    found = []
    for k, obj in enumerate(scene):
        own = (originals or {}).get(getattr(obj, "name", None), obj)
        if _a_place(own):
            found.append((k, own, obj))

    def rank(item):
        k, person, there = item
        busy = isinstance(there.behaviour, Performer) and bool(getattr(there.behaviour, "acts", ()))
        style = person.recipe.style
        like = entry is not None and (style.get("sex"), style.get("age_group")) == (sex(entry), age_group(entry))
        return busy, PLACES.index(person.kind), not like, k

    return [person for _, person, _ in sorted(found, key=rank)]


def stand_in(original, created: CreatedCharacter, library, people=(), *, wardrobe=None, now: float = 0.0):
    """The 3D person for ``created`` in the place of ``original`` (the generic person of that place), not in the
    scene yet (see the module docs). ``people``: everyone in the hall (the place's own people are left out); their
    timings stay. ``now``: the scene time it is posed for. Can run on a worker thread (it pauses between its steps,
    see :mod:`bifold_demo.pacing`); raises ``KeyError`` or ``ValueError`` when the avatar is not available."""
    pause = pacing.pause
    pause()
    model = library.load(created.avatar)
    pause()
    recipe = original.recipe
    style = avatar_style(library.available().get(created.avatar, {}), model, recipe.library)
    behaviour = slot_behaviour(original)
    play = camera_play.find_play(behaviour)
    others = [p for p in people if getattr(p, "name", None) != original.name and p is not original]
    pause()
    if same_style(style, recipe.style):  # the place's own behaviour (and camera play) fits as it is
        playing = play is not None and camera_play.supports(model)
        person = Character(model, behaviour=behaviour if playing or play is None else play.inner, name=original.name)
        pause()
        if playing:
            person.pose_adjusters.append(camera_play.PlayPose(play, model))
    else:  # made again for this avatar, fitted in among the others
        like = play.inner if play is not None else behaviour
        inner = remade(recipe, style, like)
        close = fair.fit_in(inner, [planned_behaviour(p) for p in others], like=like)
        if close:
            print(f"[created] {created.name} in {original.name} comes nearer than {fair.MIN_DISTANCE} m to {close} "
                  f"other(s) now and then")
        pause()
        person = Character(model, behaviour=inner, name=original.name)
        if play is not None:
            pause()
            fair.join_camera_play(person, others)
    pause()
    person.kind, person.created = original.kind, created
    person.recipe = Recipe(recipe.make, style, recipe.seed, recipe.library)
    looks = prepare_looks(created, model, wardrobe)
    pause()
    for problem in apply_looks(person, created, wardrobe, t=-math.inf, looks=looks):  # (a sign: held up already)
        print(f"[created] {created.name} goes without the {problem}")
    pause()
    person.update(now)
    concurrent.futures.wait(person.texture_jobs, timeout=10.0)  # decoded here, not on the first frame that shows it
    pause()
    return person


def remade(recipe: Recipe, style: dict, like):
    """The behaviour of ``recipe`` made again for ``style`` (see the module docs), as like ``like`` (the place's
    behaviour) as it can be: walking as fast as the place's person (its walk clips then play faster or slower, by at
    most ``CADENCE``, so the feet keep up), else at its own pace, with its loop as long as the place's (the pauses
    stretched to it), so that its timing can follow the place's; else looping in a whole fraction of the fair's
    ``LOOP_PERIOD``, like everyone's."""
    duration = getattr(like, "duration", None)
    speed = like.walk_speed() if hasattr(like, "walk_speed") else None
    if duration is not None and fits_period(float(duration), fair.LOOP_PERIOD):
        for options in ([{**style, "speed": speed}] if speed else []) + [style]:
            try:
                behaviour = recipe.remake(options, period=float(duration))
            except TypeError:  # a behaviour that takes no period
                break
            rates = behaviour.walk_rates() if hasattr(behaviour, "walk_rates") else []
            if abs(behaviour.duration - float(duration)) < 1e-6 and \
                    all(1.0 / CADENCE <= rate <= CADENCE for rate in rates):
                return behaviour
    return recipe.remake(style)


JOINED, LEFT = " joined the fair", " left the fair"


def notice(messages: list[str], names: int = 4) -> str:
    """One line for the screen of ``messages`` (often several characters join at once: at the start, after a scene
    reload, or when the creator saves several): ``"Ada, Ben and Cleo joined the fair"``, at most ``names`` names (then
    ``"and 9 more"``); the other messages follow, after a semicolon."""
    def together(who: list[str], what: str) -> str:
        if len(who) > names:
            who = [*who[:names - 1], f"{len(who) - names + 1} more"]
        return (", ".join(who[:-1]) + " and " + who[-1] if len(who) > 1 else who[0]) + what

    parts, joined, left = [], [], []
    for message in messages:
        for what, who in ((JOINED, joined), (LEFT, left)):
            if message.endswith(what):
                if not who:
                    parts.append(what)  # where the names go
                who.append(message[:-len(what)])
                break
        else:
            parts.append(message)
    return "; ".join(together(joined, p) if p == JOINED else together(left, p) if p == LEFT else p for p in parts)


def put(scene, person) -> None:
    """Put ``person`` into ``scene`` in place of the object of the same name (where it was in the drawing order)."""
    scene.objects[person.name] = person


def _held(obj) -> bool:
    """Whether ``obj`` is held out of sight (by :class:`CreatedCast`: it waits to come in, or it went out)."""
    return getattr(obj, "visible", True) is False


def _hold(obj) -> None:
    obj.visible = False
    obj.current_visible = False


def _version(created) -> tuple:
    """What tells two versions of a created character apart: the book's object (a changed file is read anew) and when
    it was saved (the creator may save the same object again)."""
    return (id(created), created.updated) if created is not None else (None, None)


@dataclass
class Placement:
    """A created character in the fair: its 3D ``person``, in the place of ``original``; ``version`` and ``avatar``:
    of the character as it was made from it."""

    created: CreatedCharacter
    original: object
    person: object
    version: tuple = ()
    avatar: str = ""

    def __post_init__(self):
        self.avatar = self.avatar or self.created.avatar


@dataclass
class _Place:
    """A place a created character has, or had, by the scene name of its person. ``original``: the fair's own person
    of it; ``target``: who is to be seen there (the original, or the 3D person made for ``character`` from
    ``created``, of that ``version`` and ``avatar``); ``looks``: ``(created, version, Looks)`` to put on ``target``
    next; ``since``: the scene time ``target`` or ``looks`` changed; ``settled``: the place shows them;
    ``told``: the character the place was announced with (``None``: its own person)."""

    name: str
    original: object
    target: object
    character: str | None = None
    created: CreatedCharacter | None = None
    version: tuple = ()
    avatar: str = ""
    looks: tuple | None = None
    since: float = 0.0
    settled: bool = True
    told: str | None = None

    def coming(self) -> bool:
        """Whether a character is on the way to this place (its 3D person is being made) while it waits for it."""
        return self.character is not None and self.target is self.original


@dataclass
class _Job:
    """A 3D person being made for (this ``version`` of) a character, for the place ``place``; ``looks``: only the
    looks for the 3D person there."""

    version: tuple
    created: CreatedCharacter
    place: str
    future: Future
    control: pacing.Control
    looks: bool = False


class _Plan:
    """Who is (to be) in each place, by name, for the worker to plan around: the scene's people, and the 3D people
    made for places before they step in (so that people made at the same time keep apart too). Thread-safe."""

    def __init__(self, scene=()):
        self._lock = threading.Lock()
        self._people = {o.name: o for o in scene if getattr(o, "skinned_model", None) is not None}

    def people(self) -> list:
        with self._lock:
            return list(self._people.values())

    def set(self, name: str, person) -> None:
        with self._lock:
            self._people[name] = person

    def offer(self, name: str, person, control: pacing.Control) -> None:
        """``set``, unless the job that made ``person`` was cancelled (its place went back to someone else)."""
        with self._lock:
            if not control.cancelled:
                self._people[name] = person


def _make(original, created, library, plan: _Plan, place: str, control: pacing.Control, wardrobe=None):
    """:func:`stand_in` for the worker, planned around ``plan``'s people; the new person joins the plan."""
    person = stand_in(original, created, library, plan.people(), wardrobe=wardrobe)
    plan.offer(place, person, control)
    return person


class CreatedCast:
    """Keeps the created characters of ``book`` in the scene (see the module docs).

    Call :meth:`update` with the scene after every frame (with the renderer that drew it as ``view``); it returns
    messages for the user (``"Ada joined the fair"``). ``library``: the character library the scene's people come
    from (default the fair's shared one); ``wardrobe``: the :class:`~bifold_demo.characters.accessories.Wardrobe`
    item looks are taken from; ``threads``: make the 3D people on a worker thread (else right away). ``placed`` maps
    the names of the characters seen in the scene to their :class:`Placement`; ``waiting`` names those without a free
    place.
    """

    def __init__(self, book: CharacterBook, library=None, wardrobe=None, threads: bool = True):
        self.book = book
        self.wardrobe = wardrobe
        self.threads = threads
        self._library = library
        self._scene = None
        self._time = None  # the scene time at the last update (it going back: the scene restarted)
        self._now = 0.0  # the clock time of the update going on
        self.placed: dict[str, Placement] = {}
        self.waiting: list[str] = []
        self._places: dict[str, _Place] = {}  # scene name -> a place that is, or was, a created character's
        self._jobs: dict[str, _Job] = {}  # character name -> the 3D person (or looks) being made for it
        self._failed: dict[str, tuple] = {}  # character name -> (version, why, when): tried again after RETRY
        self._announced: set[str] = set()  # characters the user was told joined (kept across scenes)
        self._told: dict = {}  # what was told about (key -> message), so that it is told once
        self._pool: ThreadPoolExecutor | None = None
        self._plan = _Plan()
        self._seen: tuple | None = None  # the book's characters at the last look
        self._again = False  # a place came free: look for the waiting ones again

    @property
    def library(self):
        if self._library is None:
            self._library = fair.shared_library()
        return self._library

    # -- every frame ------------------------------------------------------------------------------------------

    def update(self, scene, now: float | None = None, wait: bool = False, view=None) -> list[str]:
        """Make the changes that are due, at a frame where nobody sees them (see the module docs), and, when the book
        changed (it looks at its folder once a second), start the new ones. ``scene`` is the one shown (another one
        gets the characters anew); ``view``: the renderer that drew its last frame (``visible_pixels`` and
        ``prepare``), or ``None``; ``wait``: finish all of it now and make the changes at once (for a scene nobody
        has seen yet: then its next frame has them). After the scene restarted (its time went back: the whole picture
        changes) the changes that are ready are made at once too."""
        now = time.monotonic() if now is None else now
        self._now = now
        if scene is not self._scene:
            self._forget()
            self._scene, self._plan = scene, _Plan(scene)
        restarted = self._time is not None and scene.time < self._time - 1e-6
        self._time = scene.time
        self.book.refresh(now, force=self._seen is None)
        at_once = wait or restarted
        messages = self._step(view, at_once)
        while wait and (self._jobs or self._again):
            for job in self._jobs.values():
                job.control.hurry()  # someone waits: no more rests
            concurrent.futures.wait([job.future for job in self._jobs.values()])
            messages += self._step(view, at_once)
        return messages

    def status(self) -> str | None:
        """A line for the status panel, or ``None`` without created characters."""
        problems = len([n for n in self._failed if n not in self.placed]) + len(self.book.problems)
        leaving = [n for n in self.placed if not any(p.character == n for p in self._places.values())]
        coming = [p.character for p in self._places.values() if p.character is not None and p.character not in
                  self.placed]
        here = [n for n in self.placed if n not in leaving]
        if not (here or coming or leaving or self.waiting or problems):
            return None
        parts = [", ".join(here) + " in the fair"] if here else []
        if coming:
            parts.append(f"{len(coming)} on the way")
        if leaving:
            parts.append(", ".join(leaving) + " leaving")
        if self.waiting:
            parts.append(f"{len(self.waiting)} waiting for a place")
        if problems:
            parts.append(f"{problems} cannot be shown")
        return "created: " + "; ".join(parts)

    def close(self) -> None:
        """Stop the work going on (a job stops at its next pause, so the program can end at once)."""
        self._forget()
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None

    # -- keeping in step ----------------------------------------------------------------------------------------

    def _step(self, view, at_once: bool) -> list[str]:
        messages = self._tell_problems()
        seen = tuple((c.name, *_version(c)) for c in self.book.characters())
        retry = any(self._now - when >= RETRY for _, _, when in self._failed.values())
        if seen != self._seen or self._again or retry:
            self._seen, self._again = seen, False
            messages += self._reconcile()
        return messages + self._collect() + self._drive(view, at_once)

    def _tell_problems(self) -> list[str]:
        """The book's problems, each when it is new (or comes back after it was fixed)."""
        current = {("book", folder): f"cannot read the created character {folder}: {problem}"
                   for folder, problem in self.book.problems.items()}
        if self.book.trouble:
            current[("folder",)] = self.book.trouble
        messages = [m for key, m in current.items() if self._told.get(key) != m]
        for key in [k for k in self._told if k[0] in ("book", "folder") and k not in current]:
            del self._told[key]
        self._told.update(current)
        return messages

    def _tell(self, key, message: str) -> list[str]:
        if self._told.get(key) == message:
            return []
        self._told[key] = message
        return [message]

    def _reconcile(self) -> list[str]:
        scene, messages = self._scene, []
        wanted = {c.name: c for c in self.book.characters()}
        for place in list(self._places.values()):  # deleted: the place goes back to its own person
            if place.character is not None and place.character not in wanted:
                self._release(place)
        for name in [n for n, job in self._jobs.items() if job.version != _version(wanted.get(n))]:
            self._cancel(name)
        for name, (version, _, when) in list(self._failed.items()):
            if version != _version(wanted.get(name)) or self._now - when >= RETRY:
                del self._failed[name]  # changed, or a while ago: tried again
        for key in [k for k in self._told if k[0] in ("failed", "full") and k[1] not in wanted]:
            del self._told[key]
        self._announced &= set(wanted) | {p.told for p in self._places.values()}
        for place in self._places.values():  # changed: new looks on the same 3D person, or another avatar
            name = place.character
            if name is None or name in self._jobs or name in self._failed:
                continue
            created = wanted[name]
            if (place.looks[1] if place.looks is not None else place.version) != _version(created):
                self._submit(created, place, looks=place.created is not None and created.avatar == place.avatar)
        self.waiting = []
        catalog = None
        for name, created in wanted.items():  # new ones, oldest first: the next free place
            if name in self._failed or any(p.character == name for p in self._places.values()):
                continue
            if catalog is None:
                catalog = self._catalog()
            originals = {n: p.original for n, p in self._places.items()}
            taken = {n for n, p in self._places.items() if p.character is not None}
            free = [p for p in places(scene, catalog.get(created.avatar), originals) if p.name not in taken]
            if free:
                original = free[0]
                place = self._places.get(original.name)
                if place is None:
                    place = self._places[original.name] = _Place(original.name, original, original)
                place.character = name
                self._told.pop(("full", name), None)
                self._submit(created, place)
                continue
            self.waiting.append(name)
            if not places(scene, originals=originals):
                messages += self._tell(("scene", id(scene)), "created characters need the 3D people of the fair "
                                                             "(this scene has no place for them)")
            else:
                messages += self._tell(("full", name), f"{name} waits for a free place in the fair")
        return messages

    def _catalog(self) -> dict:
        try:
            return self.library.available()
        except Exception:  # noqa: BLE001 - no library: the places are ranked without it, the avatar fails later
            return {}

    def _submit(self, created: CreatedCharacter, place: _Place, looks: bool = False) -> None:
        """Make the 3D person for ``created`` in ``place`` (or, ``looks``, the looks for the one there)."""
        control = pacing.Control()
        if looks:
            work = functools.partial(prepare_looks, created, place.target.skinned_model, self.wardrobe)
        else:
            work = functools.partial(_make, place.original, created, self.library, self._plan, place.name, control,
                                     wardrobe=self.wardrobe)
        if self.threads:
            if self._pool is None:
                self._pool = ThreadPoolExecutor(1, thread_name_prefix="created-cast")
            future = self._pool.submit(pacing.run_paced, work, control=control)
        else:
            future = Future()
            try:
                future.set_result(work())
            except Exception as exc:  # noqa: BLE001 - reported by _collect
                future.set_exception(exc)
        self._jobs[created.name] = _Job(_version(created), created, place.name, future, control, looks)

    def _cancel(self, name: str) -> None:
        job = self._jobs.pop(name)
        job.control.cancel()
        job.future.cancel()
        place = self._places.get(job.place)
        if place is not None:
            self._plan.set(place.name, place.target)

    def _release(self, place: _Place) -> None:
        """The character of ``place`` left the book: the place goes back to its own person (or to the next one)."""
        if place.character in self._jobs:
            self._cancel(place.character)
        self._failed.pop(place.character, None)
        place.character, place.created, place.version, place.avatar, place.looks = None, None, (), "", None
        self._retarget(place, place.original)
        self._again = self._again or bool(self.waiting)

    def _retarget(self, place: _Place, target) -> None:
        if target is not place.target:
            if self._scene.get(place.name) is not place.target:
                _hold(place.target)  # it never came in
            place.target = target
        place.since, place.settled = self._scene.time, False
        self._plan.set(place.name, target)

    def _collect(self) -> list[str]:
        """Take the 3D people (and looks) that are ready."""
        messages = []
        for name, job in list(self._jobs.items()):
            if not job.future.done():
                continue
            del self._jobs[name]
            place = self._places.get(job.place)
            if job.future.cancelled() or place is None or place.character != name:
                continue
            try:
                made = job.future.result()  # (the looks, for new looks)
            except pacing.Cancelled:
                continue
            except Exception as exc:  # noqa: BLE001 - e.g. an avatar that is not prepared: the rest go on
                messages += self._fail(name, job, place, exc)
                continue
            self._told.pop(("failed", name), None)
            if job.looks:
                place.looks = (job.created, job.version, made)
                place.since, place.settled = self._scene.time, False
            else:
                _hold(made)
                place.created, place.version, place.avatar, place.looks = job.created, job.version, \
                    job.created.avatar, None
                self._retarget(place, made)
        return messages

    def _fail(self, name: str, job: _Job, place: _Place, exc: Exception) -> list[str]:
        """Remember that ``job`` failed (it is tried again after ``RETRY``, or when the character changes); what to
        tell."""
        self._plan.set(place.name, place.target)
        folder = job.created.folder
        if folder is not None and not (folder / "character.json").is_file():
            return []  # deleted while it was being made: not a problem, the next look at the folder sees it
        problem = str(exc.args[0]) if isinstance(exc, KeyError) and exc.args else f"{type(exc).__name__}: {exc}"
        self._failed[name] = (job.version, problem, self._now)
        if place.created is None:  # never in the fair: its place is free again, for one waiting for a place
            place.character = None
            self._retarget(place, place.original)
            self._again = self._again or bool(self.waiting)
            message = f"cannot put {name} in the fair: {problem}"
        elif job.looks:
            message = f"cannot give {name} the new looks: {problem}"
        else:
            message = f"cannot make {name} {job.created.avatar}: {problem}"
        return self._tell(("failed", name), message)

    # -- the changes, where nobody sees them ---------------------------------------------------------------------

    def _drive(self, view, at_once: bool) -> list[str]:
        """Make the changes of the places that are due (see the module docs)."""
        scene, messages = self._scene, []
        pending = [p for p in self._places.values() if not p.settled]
        if not pending:
            return messages
        ready = pending if at_once else self._prepare(view, pending)
        pixels = {} if at_once else self._pixels(view, [o for p in ready for o in self._to_count(p)])
        for place in ready:
            allowed = math.inf if at_once else self._allowed(place, view)
            current, target = scene.get(place.name), place.target
            if current is not target:  # first the one there goes, at a frame where nothing of them shows
                if current is not None and not _held(current):
                    if pixels.get(id(current), 0) > allowed:
                        continue
                    _hold(current)
                put(scene, target)
                messages += self._left(place)
            if _held(target):  # then the one coming in, where nothing of them would show
                if place.coming() or pixels.get(id(target), 0) > allowed:
                    continue  # (on the way: the place stays empty until they are here)
                target.visible = True
                target.update(scene.time)
            if place.looks is not None:
                if pixels.get(id(target), 0) > allowed:
                    continue
                messages += self._put_on(place)
            messages += self._settle(place)
        return messages

    def _prepare(self, view, pending: list[_Place]) -> list[_Place]:
        """The places of ``pending`` whose newcomer (and new looks) are on the GPU: one more texture (or mesh part)
        goes there each frame, for the first place that needs one."""
        ready, uploading = [], False
        for place in pending:
            things = []
            if place.target is not self._scene.get(place.name) or _held(place.target):
                things.append(place.target)
            if place.looks is not None:
                things += place.looks[2].gpu()
            if not things or place.coming() or (not uploading and _prepared(view, *things)):
                ready.append(place)
            else:
                uploading = True
        return ready

    def _to_count(self, place: _Place) -> list:
        """Who of ``place`` a viewer must not see change now: the one leaving, the one coming in."""
        current, target = self._scene.get(place.name), place.target
        out = [current] if current is not None and current is not target and not _held(current) else []
        return out + ([] if place.coming() else [target])

    def _allowed(self, place: _Place, view) -> float:
        """How many pixels a change of ``place`` may show: none, and after ``LINE_UP`` seconds ``SMALL``."""
        if self._scene.time - place.since < LINE_UP:
            return 0
        width, height = (getattr(view, "width", None), getattr(view, "height", None)) if view is not None else PICTURE
        width, height = (width, height) if width and height else PICTURE
        return SMALL * width * height

    def _pixels(self, view, people) -> dict[int, int]:
        """``{id(person): pixels}``: how much of each of ``people`` would show in the frame just drawn (posed for its
        time; those held out of sight for the count)."""
        scene, unique = self._scene, list({id(p): p for p in people}.values())
        if not unique:
            return {}
        posed = [p for p in unique if _held(p)]
        for person in posed:
            person.visible = True
            person.update(scene.time)
        try:
            counts = None
            if view is not None:
                try:
                    counts = view.visible_pixels(unique)
                except Exception as exc:  # noqa: BLE001 - e.g. before the first frame: as without a renderer
                    if self._tell(("view",), str(exc)):
                        print(f"[created] cannot count pixels, counting less well ({type(exc).__name__}: {exc})")
            if counts is None:
                counts = [estimate(scene, p) for p in unique]
        finally:
            for person in posed:
                _hold(person)
        return {id(p): int(c) for p, c in zip(unique, counts)}

    def _left(self, place: _Place) -> list[str]:
        """The character ``place`` was announced with went out of it for good: what to tell."""
        who, now = place.told, (None if place.target is place.original else place.character)
        if who is None or who == now:
            return []
        place.told = None
        self.placed.pop(who, None)
        if who in self._announced:
            self._announced.discard(who)
            return [who + LEFT]
        return []

    def _put_on(self, place: _Place) -> list[str]:
        """The new looks of ``place`` go on its person: what to tell."""
        (created, version, looks), target = place.looks, place.target
        place.looks = None
        try:
            left_out = apply_looks(target, created, self.wardrobe, t=-math.inf, looks=looks)
        except Exception as exc:  # noqa: BLE001 - a look it cannot use: the demo goes on, as it was
            job = _Job(version, created, place.name, Future(), pacing.Control(), looks=True)
            return self._fail(place.character, job, place, exc)
        for problem in left_out:
            print(f"[created] {place.character} goes without the {problem}")
        target.created, place.created, place.version = created, created, version
        target.update(self._scene.time)
        return [f"{place.character} has new looks"]

    def _settle(self, place: _Place) -> list[str]:
        """``place`` shows its target, with its looks: what to tell."""
        place.settled = True
        name = None if place.target is place.original else place.character
        messages = self._left(place)
        if name is not None:
            old = self.placed.get(name)
            self.placed[name] = Placement(place.created, place.original, place.target, place.version, place.avatar)
            if name not in self._announced:
                self._announced.add(name)
                messages.append(name + JOINED)
            elif old is not None and old.person is not place.target:
                messages.append(f"{name} is now {place.avatar}")
        place.told = name
        if name is None and place.character is None:
            del self._places[place.name]  # its own person again
        return messages

    def _forget(self) -> None:
        """Another scene: the people of the old one are gone with it."""
        for job in self._jobs.values():
            job.control.cancel()
            job.future.cancel()
        self._jobs.clear()
        self._places.clear()
        self.placed.clear()
        self._failed.clear()
        self.waiting = []
        self._told = {k: v for k, v in self._told.items() if k[0] in ("book", "folder")}
        self._seen, self._again, self._time = None, False, None


def _prepared(view, *things) -> bool:
    """Whether what drawing ``things`` (people, textures) needs is on the GPU (one more of it is uploaded now)."""
    prepare = getattr(view, "prepare", None)
    if prepare is None:
        return True
    try:
        return bool(prepare(*things))
    except Exception:  # noqa: BLE001 - then the first frame that draws it uploads it
        return True


def estimate(scene, person) -> int:
    """Without a renderer: about how many pixels of a ``PICTURE`` of ``scene``'s camera ``person`` (posed) covers:
    0 when hidden, or out of the picture (:func:`~bifold_demo.scenes.fair.in_view`); nothing in front counts."""
    if not person.current_visible:
        return 0
    x, _, z = (float(v) for v in person.current_position)
    camera = getattr(scene, "camera", None)
    if not fair.in_view((x, z), camera=camera):
        return 0
    if camera is None:
        return PICTURE[0] * PICTURE[1]
    to_clip = camera.projection_matrix(PICTURE[0] / PICTURE[1]) @ camera.view_matrix()
    feet, head = to_clip @ (x, 0.0, z, 1.0), to_clip @ (x, 1.8, z, 1.0)
    if feet[3] <= 1e-6 or head[3] <= 1e-6:
        return PICTURE[0] * PICTURE[1]
    tall = abs(feet[1] / feet[3] - head[1] / head[3]) / 2.0 * PICTURE[1]
    return math.ceil(0.2 * tall * tall) + 1  # a person covers about a fifth of the square of their height
