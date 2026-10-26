"""Created characters in the fair: each takes over the place of one of the generic people who walk about.

A character made in the character creator (:mod:`bifold_demo.characters.created`) appears in the fair in the place of
a generic grown-up and keeps that place's behaviour: where and when they walk, so the spacing and the timing the fair
planned (``fair.keep_apart``) and the camera play stay valid. The best places go first (:data:`PLACES`): the visitors
who walk up the aisle to the camera (``approach``: they are seen best, watch the camera and play with it), then the
people walking by along the cross aisles (``walker``), then those wandering about (``wanderer``). Never a child's place:
its behaviour is made for a child's size. Within a kind, places whose person is like the created character's avatar
(same sex and age group) come first, because there the behaviour fits the avatar as it is
(:func:`~bifold_demo.characters.crowd.same_style`: the same clips, pace and stride; Rocketbox grown-ups of one sex
have the same legs), and the new 3D person simply gets it: the swap is exact, the same position at the same time.

Elsewhere the behaviour is made again for the new avatar (:meth:`~bifold_demo.characters.crowd.Recipe.remake`: the
same path, pauses and seed, but its own clips, pace and stride, so the feet do not slide), its timing is fitted in
among everyone else's (:func:`~bifold_demo.scenes.fair.fit_in`: never nearer anyone than ``MIN_DISTANCE``, following
the old timing as closely as that allows) and, in front of the camera, its camera play is planned around the others'
moves, which stay as they are (:func:`~bifold_demo.scenes.fair.join_camera_play`). It walks as fast as the place's
person did when its clips can keep up (within ``CADENCE``), so the timings differ only by its own turns and clips:
in some places it is back in step for most of the loop, in others (another loop length, or turns and pauses that
fall differently) metres apart for long stretches. So it is swapped in at a moment nobody sees it happen: when it
stands where that person stands (``JUMP``), or when neither of them is in the camera's picture (see
:class:`CreatedCast`).

The new 3D person keeps the place's name (``approach1``) and ``kind``; ``person.created`` is the
:class:`~bifold_demo.characters.created.CreatedCharacter`, so other parts can tell (the known faces of
``recognition.cast`` leave it alone, selection mode shows the character's name). It wears the character's textures
and accessories (:func:`~bifold_demo.characters.created.apply_looks`). The person whose place it took waits outside
the scene and comes back when the character is deleted.

:class:`CreatedCast` keeps a scene in step with a :class:`~bifold_demo.characters.created.CharacterBook` while the
demo runs: a newly saved character takes the next free place (replacing that place's person in place), a changed one
is updated in place (new looks on the same 3D person; another avatar: a new 3D person in the same place), a deleted
one gives the place back to its original person. Making a 3D person (loading the avatar, painting its looks, making
its behaviour again) takes from a few milliseconds to under a second of work, so it runs on a worker thread, paced
(:mod:`bifold_demo.pacing`: it rests three quarters of the time, else it would take the GIL from the main thread for
5 ms at a time and the frames would come late), and the swap happens on a frame after it is ready. A new scene (the
scene file reloaded) gets the characters again, all at once. What cannot be shown is left out and the rest goes on:
a character whose avatar is not prepared (its place goes to one waiting for a place), new looks it cannot use (it
stays as it was), an item its avatar cannot wear. :func:`notice` puts the messages of one update on one line.
"""

from __future__ import annotations

import concurrent.futures
import math
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

from .. import pacing
from ..characters import camera_play
from ..characters.behaviour import fits_period
from ..characters.character import Character
from ..characters.created import CharacterBook, CreatedCharacter, apply_looks, textures_for
from ..characters.crowd import Recipe, age_group, avatar_style, same_style, sex
from ..characters.performer import Performer
from . import fair

PLACES = ("approach", "walker", "wanderer")  # kinds of places created characters take, best first
LEAD = 1.0  # seconds: the looks are put on for the scene time this much after a 3D person was asked for
CADENCE = 1.25  # walk clips of a behaviour made again play at most this much faster (or slower) than their own pace
JUMP = 0.05  # metres: a 3D person with a behaviour of its own is swapped in when it is this near the one it replaces
LINE_UP = 60.0  # seconds of scene time it waits for that (or for both to be out of sight) before it takes less
POP = 2.0  # metres: a person appearing or vanishing in view counts as a jump this far; allowed after LINE_UP


def slot_behaviour(person):
    """The behaviour of ``person``'s place: theirs, without what selection mode wraps around it for yoga poses."""
    behaviour = getattr(person, "behaviour", None)
    return behaviour.base if isinstance(behaviour, Performer) else behaviour


def planned_behaviour(person):
    """``person``'s behaviour as ``fair.keep_apart`` planned it (:func:`~bifold_demo.scenes.fair.planned_behaviour`:
    without selection mode's and the camera play's additions)."""
    return fair.planned_behaviour(person)


def places(scene, entry: dict | None = None) -> list:
    """The generic people of ``scene`` whose place a created character may take, best first (see the module docs);
    with ``entry`` (the catalog entry of the character's avatar) the people like it first within each kind. People
    doing a yoga pose in selection mode, or who did one (their timing moved), come last."""
    found = []
    for k, obj in enumerate(scene):
        kind, recipe = getattr(obj, "kind", None), getattr(obj, "recipe", None)
        if kind in PLACES and recipe is not None and getattr(obj, "created", None) is None \
                and recipe.style.get("age_group") != "child":
            found.append((k, obj))

    def rank(item):
        k, person = item
        busy = isinstance(person.behaviour, Performer) and bool(getattr(person.behaviour, "acts", ()))
        style = person.recipe.style
        like = entry is not None and (style.get("sex"), style.get("age_group")) == (sex(entry), age_group(entry))
        return busy, PLACES.index(person.kind), not like, k

    return [person for _, person in sorted(found, key=rank)]


def stand_in(original, created: CreatedCharacter, library, people=(), *, wardrobe=None, now: float = 0.0):
    """The 3D person for ``created`` in the place of ``original`` (the generic person of that place), not in the
    scene yet (see the module docs). ``people``: everyone in the hall (the place's own people are left out); their
    timings stay. ``now``: the scene time the swap is for. Can run on a worker thread; raises ``KeyError`` or
    ``ValueError`` when the avatar is not available."""
    model = library.load(created.avatar)
    recipe = original.recipe
    style = avatar_style(library.available().get(created.avatar, {}), model, recipe.library)
    behaviour = slot_behaviour(original)
    play = camera_play.find_play(behaviour)
    others = [p for p in people if getattr(p, "name", None) != original.name and p is not original]
    if same_style(style, recipe.style):  # the place's own behaviour (and camera play) fits as it is
        playing = play is not None and camera_play.supports(model)
        person = Character(model, behaviour=behaviour if playing or play is None else play.inner, name=original.name)
        if playing:
            person.pose_adjusters.append(camera_play.PlayPose(play, model))
    else:  # made again for this avatar, fitted in among the others
        like = play.inner if play is not None else behaviour
        inner = remade(recipe, style, like)
        close = fair.fit_in(inner, [planned_behaviour(p) for p in others], like=like)
        if close:
            print(f"[created] {created.name} in {original.name} comes nearer than {fair.MIN_DISTANCE} m to {close} "
                  f"other(s) now and then")
        person = Character(model, behaviour=inner, name=original.name)
        if play is not None:
            fair.join_camera_play(person, others)
    person.kind, person.created = original.kind, created
    person.recipe = Recipe(recipe.make, style, recipe.seed, recipe.library)
    for problem in apply_looks(person, created, wardrobe, t=now):
        print(f"[created] {created.name} goes without the {problem}")
    person.update(now)
    concurrent.futures.wait(person.texture_jobs, timeout=10.0)  # decoded here, not on the first frame that shows it
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
    """One line for the screen of the ``messages`` of one :meth:`CreatedCast.update` (several characters often join
    at once: at the start, after a scene reload, or when the creator saves several): ``"Ada, Ben and Cleo joined the
    fair"``, at most ``names`` names (then ``"and 9 more"``); the other messages follow, after a semicolon."""
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
class _Job:
    """A 3D person being made for (this ``version`` of) a character, in the place of ``original``."""

    version: tuple
    created: CreatedCharacter
    original: object
    future: Future
    ready: float | None = None  # the scene time it was ready, while it waits to line up with the one it replaces
    looks: object = None  # instead: the 3D person (in the scene) who gets the new looks, once their images are read


class CreatedCast:
    """Keeps the created characters of ``book`` in the scene (see the module docs).

    Call :meth:`update` with the scene after every frame; it returns messages for the user (``"Ada joined the
    fair"``). ``library``: the character library the scene's people come from (default the fair's shared one);
    ``wardrobe``: the :class:`~bifold_demo.characters.accessories.Wardrobe` item looks are taken from; ``threads``:
    make the 3D people on a worker thread (else right away). ``placed`` maps the names of the characters in the
    scene to their :class:`Placement`; ``waiting`` names those without a free place.

    A 3D person with a behaviour of its own (made again for its avatar) follows the place's old timing only roughly,
    so it is swapped in at a moment nobody can see it happen: when it is where the person it replaces is (within
    ``JUMP``; both standing at the end of the aisle, say), or when neither of them is in the camera's picture (out
    of view, or hidden while they come back to the start of their walk). After ``LINE_UP`` seconds of scene time
    without such a moment it takes a jump of up to ``POP`` metres in view, or one of them appearing or vanishing,
    and after twice that whatever there is (:func:`swap_cost`).
    """

    def __init__(self, book: CharacterBook, library=None, wardrobe=None, threads: bool = True):
        self.book = book
        self.wardrobe = wardrobe
        self.threads = threads
        self._library = library
        self._scene = None
        self.placed: dict[str, Placement] = {}
        self.waiting: list[str] = []
        self._jobs: dict[str, _Job] = {}  # character name -> the 3D person being made for it
        self._failed: dict[str, tuple] = {}  # character name -> (version, why): not put in until it changes
        self._announced: set[str] = set()
        self._told: set = set()  # problems already told about
        self._pool: ThreadPoolExecutor | None = None
        self._seen: tuple | None = None  # the book's characters at the last look
        self._again = False  # a place came free (a character could not be made): look for the waiting ones again

    @property
    def library(self):
        if self._library is None:
            self._library = fair.shared_library()
        return self._library

    # -- every frame ------------------------------------------------------------------------------------------

    def update(self, scene, now: float | None = None, wait: bool = False) -> list[str]:
        """Swap in the 3D people made since the last call and, when the book changed (it looks at its folder once a
        second), put new characters in, update changed ones and take deleted ones out. ``scene`` is the one shown
        (another one gets the characters anew); ``wait``: finish all of it now and swap them in at once (for a scene
        nobody has seen yet: then its next frame has them)."""
        now = time.monotonic() if now is None else now
        if scene is not self._scene:
            self._forget()
            self._scene = scene
        self.book.refresh(now, force=self._seen is None)
        messages = self._step(wait)
        while wait and (self._jobs or self._again):
            concurrent.futures.wait([job.future for job in self._jobs.values()])
            messages += self._step(wait)
        return messages

    def status(self) -> str | None:
        """A line for the status panel, or ``None`` without created characters."""
        problems = len([n for n in self._failed if n not in self.placed]) + len(self.book.problems)
        if not (self.placed or self._jobs or self.waiting or problems):
            return None
        parts = [", ".join(self.placed) + " in the fair"] if self.placed else []
        if self._jobs:
            parts.append(f"{len(self._jobs)} on the way")
        if self.waiting:
            parts.append(f"{len(self.waiting)} waiting for a place")
        if problems:
            parts.append(f"{problems} cannot be shown")
        return "created: " + "; ".join(parts)

    def close(self) -> None:
        self._forget()
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None

    # -- keeping in step ----------------------------------------------------------------------------------------

    def _step(self, at_once: bool = False) -> list[str]:
        seen = tuple((c.name, *_version(c)) for c in self.book.characters())
        messages = []
        if seen != self._seen or self._again:
            self._seen, self._again = seen, False
            messages += self._reconcile()
        return messages + self._finish(at_once)

    def _reconcile(self) -> list[str]:
        scene, messages = self._scene, []
        wanted = {c.name: c for c in self.book.characters()}
        for folder, problem in self.book.problems.items():
            messages += self._tell(("book", folder, problem), f"cannot read the created character {folder}: {problem}")
        for name in [n for n in self.placed if n not in wanted]:  # deleted: the original person comes back
            put(scene, self.placed.pop(name).original)
            self._announced.discard(name)
            messages.append(name + LEFT)
        for name in [n for n, job in self._jobs.items() if job.version != _version(wanted.get(n))]:
            self._jobs.pop(name).future.cancel()
        for name in [n for n, failed in self._failed.items() if failed[0] != _version(wanted.get(n))]:
            del self._failed[name]
        for name, placement in list(self.placed.items()):  # changed
            created = wanted[name]
            if placement.version == _version(created) or name in self._jobs or name in self._failed:
                continue
            if created.avatar == placement.avatar:  # new looks on the same 3D person (its images read on the worker)
                self._submit(created, placement.original, looks=placement.person)
            else:  # another avatar: a new 3D person in the same place
                self._submit(created, placement.original)
        self.waiting = []
        catalog = None
        for name, created in wanted.items():  # new ones, oldest first: the next free place
            if name in self.placed or name in self._failed or name in self._jobs:
                continue
            if catalog is None:
                catalog = self._catalog()
            taken = {id(p.original) for p in self.placed.values()} | {id(job.original) for job in self._jobs.values()}
            free = [p for p in places(scene, catalog.get(created.avatar)) if id(p) not in taken]
            if free:
                self._submit(created, free[0])
                continue
            self.waiting.append(name)
            if not places(scene):
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

    def _submit(self, created: CreatedCharacter, original, looks=None) -> None:
        """Make the 3D person for ``created`` in ``original``'s place, or (``looks``: a 3D person) its textures."""
        scene = self._scene
        if looks is not None:
            work, args, options = textures_for, (created, looks.skinned_model), {}
        else:
            people = [o for o in scene if getattr(o, "skinned_model", None) is not None]
            work, args = stand_in, (original, created, self.library, people)
            options = {"wardrobe": self.wardrobe, "now": scene.time + (LEAD if self.threads else 0.0)}
        if self.threads:
            if self._pool is None:
                self._pool = ThreadPoolExecutor(1, thread_name_prefix="created-cast")
            future = self._pool.submit(pacing.run_paced, work, *args, **options)
        else:
            future = Future()
            try:
                future.set_result(work(*args, **options))
            except Exception as exc:  # noqa: BLE001 - reported by _finish
                future.set_exception(exc)
        self._jobs[created.name] = _Job(_version(created), created, original, future, looks=looks)

    def _finish(self, at_once: bool = False) -> list[str]:
        """Swap in the 3D people that are ready (and line up with the people they replace, unless ``at_once``)."""
        messages, scene = [], self._scene
        for name, job in list(self._jobs.items()):
            if not job.future.done():
                continue
            if job.future.cancelled():
                del self._jobs[name]
                continue
            try:
                person = job.future.result()  # (the textures, for new looks)
            except Exception as exc:  # noqa: BLE001 - e.g. an avatar that is not prepared: the rest go on
                del self._jobs[name]
                messages.append(self._fail(name, job, exc))
                continue
            if job.looks is not None:  # the new looks, on the same 3D person
                del self._jobs[name]
                if name in self.placed and self.placed[name].person is job.looks:
                    try:
                        left_out = apply_looks(job.looks, job.created, self.wardrobe, t=scene.time, textures=person)
                    except Exception as exc:  # noqa: BLE001 - a look it cannot use: the demo goes on, as it was
                        messages.append(self._fail(name, job, exc))
                        continue
                    for problem in left_out:
                        print(f"[created] {name} goes without the {problem}")
                    job.looks.created = job.created
                    self.placed[name] = Placement(job.created, job.original, job.looks, job.version)
                    messages.append(f"{name} has new looks")
                continue
            if not at_once:
                if job.ready is None or scene.time < job.ready:  # (or the scene restarted meanwhile)
                    job.ready = scene.time
                waited = scene.time - job.ready
                allowed = JUMP if waited < LINE_UP else POP if waited < 2 * LINE_UP else math.inf
                if swap_cost(scene, scene.get(person.name), person, scene.time) > allowed:
                    continue
            del self._jobs[name]
            before = self.placed.get(name)
            put(scene, person)
            self.placed[name] = Placement(job.created, job.original, person, job.version)
            if before is not None:
                messages.append(f"{name} is now {job.created.avatar}")
            elif name not in self._announced:
                self._announced.add(name)
                messages.append(name + JOINED)
        return messages

    def _fail(self, name: str, job: _Job, exc: Exception) -> str:
        """Remember that ``job`` failed (it is not tried again until the character changes); what to tell."""
        problem = str(exc.args[0]) if isinstance(exc, KeyError) and exc.args else f"{type(exc).__name__}: {exc}"
        self._failed[name] = (job.version, problem)
        if name not in self.placed:  # its place is free again: a character waiting for one may have it
            self._again = self._again or bool(self.waiting)
            return f"cannot put {name} in the fair: {problem}"
        if job.looks is not None:
            return f"cannot give {name} the new looks: {problem}"
        return f"cannot make {name} {job.created.avatar}: {problem}"

    def _tell(self, key, message: str) -> list[str]:
        if key in self._told:
            return []
        self._told.add(key)
        return [message]

    def _forget(self) -> None:
        """Another scene: the people of the old one are gone with it."""
        for job in self._jobs.values():
            job.future.cancel()
        self._jobs.clear()
        self.placed.clear()
        self._failed.clear()
        self.waiting = []
        self._seen, self._again = None, False


def swap_cost(scene, occupant, person, t: float) -> float:
    """How much a viewer sees of ``person`` taking ``occupant``'s place in ``scene`` at scene time ``t``: 0 when
    neither is in the camera's picture (hidden, or out of view), else how far the swap jumps on the floor (metres,
    both in view), or ``POP`` when one of them would appear or vanish in view."""
    if occupant is None or getattr(occupant, "behaviour", None) is person.behaviour:  # the same behaviour: no jump
        return 0.0
    state = person.behaviour(t)
    old = occupant.current_position if getattr(occupant, "current_visible", True) else None
    new = state.position if state.visible else None
    old, new = (p if p is not None and _seen(scene, p) else None for p in (old, new))
    if old is None and new is None:
        return 0.0
    if old is None or new is None:
        return POP
    return math.hypot(float(new[0]) - float(old[0]), float(new[2]) - float(old[2]))


def _seen(scene, position) -> bool:
    """Whether someone standing at ``position`` (``(x, y, z)``) can show up in the picture of ``scene``'s camera."""
    camera = getattr(scene, "camera", None)
    return fair.in_view((float(position[0]), float(position[2])), camera=camera)
