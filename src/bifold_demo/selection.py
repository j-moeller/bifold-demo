"""The demo's selection mode: pick one of the 3D people in the picture, have them do yoga poses, dress them up.

``Y`` (or Tab) starts it with the most prominent person selected; Tab and Shift+Tab (or the left and right arrows) go
to the next person to the right or left; Esc or ``Y`` ends it. The up and down arrows (or ``I``) switch between the
:func:`menus`, whose entries are on the number keys: in a yoga menu a number makes the selected person do that pose,
the same number again keeps them holding it and 0 (or Backspace) stops; in the items menu a number puts that item on
(:mod:`.characters.accessories`), the same number again goes to its next look and, after the last, takes it off, and
0 takes everything off. When the selected person leaves the picture, is hidden or is removed from the scene, the
person nearest to where they were is selected instead.

A person asked for a pose stops at once and stands while a spot for it is found on a thread (:meth:`Selection._place`:
it takes a second or two). The search plans for the pose to start a little later, and the pose waits for that
moment, so where the person stands and where the others will be are what the search assumed; a spot found too late
(after a while on the webcam, say) is looked for again. Spots are found one after another, each once the pose before
has started, so poses asked of several people in a row keep clear of each other.
"""

from __future__ import annotations

import math
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .characters import accessories, yoga
from .characters.accessories import Wardrobe
from .characters.performer import (
    TURN_MIN,
    WALK_FROM,
    Performer,
    choose_place,
    floors,
    footprint,
    obstacles,
    place,
    standing,
)
from .characters.yoga_clips import YogaClips
from .display.window import BACKSPACE, BACKTAB, DOWN, LEFT, RIGHT, TAB, UP

PAGE = 8  # poses to a menu
NUMBER_KEYS = {ord(str(k)): k for k in range(1, 10)}
STOP_KEYS = (ord("0"), BACKSPACE)
MIN_HEIGHT = 0.08  # share of the picture height: people drawn smaller standing (far away) cannot be selected
MIN_INSIDE = 0.6  # share of a person's box that must be in the picture
KEEP = 0.25, 0.8  # the selected person stays selected down to these shares of MIN_INSIDE and MIN_HEIGHT
CLICK_SLACK = 0.03  # share of the picture height: a click this close to a person's box still selects them
WALK_REACH = 4.0  # metres a person walks at most to find room for a pose (people without walk clips: STEP_REACH)
STEP_REACH = 0.6
PERSON = 0.6, 1.8  # width and height of the box that stands for another person hiding the pose
HIDDEN = 0.1  # share of a pose's points that may be hidden behind people or furniture in a frame of its hold
LEAD = 1.0, 4.0  # seconds a spot is planned ahead: at first and at most (else a bit more than the last search took)
STALE = 1.25  # seconds a pose may start later than its spot was planned for (half the plan's steps), else look again
TRIES = 3  # searches at most for one pose (the last result is used whatever)


def menus() -> list[tuple[str, str, list[str]]]:
    """Selection mode's menus, ``(title, kind, entries)`` with the entries on the number keys 1, 2, ...: the yoga
    poses (:data:`PAGE` to a menu, in the order of :data:`yoga.POSES`), then the items (``kind`` "items")."""
    poses = list(yoga.POSES)
    pages = [poses[k:k + PAGE] for k in range(0, len(poses), PAGE)]
    titles = [f"Yoga {k + 1}/{len(pages)}" if len(pages) > 1 else "Yoga" for k in range(len(pages))]
    items = [*accessories.ITEMS, "face"]  # face: a known person's face (recognition.cast)
    return [(title, "yoga", page) for title, page in zip(titles, pages)] + [("Items", "items", items)]


def _distance(a, b) -> float:
    """Distance of two ``(x, y, z)`` points along the floor."""
    return math.hypot(float(a[0]) - float(b[0]), float(a[2]) - float(b[2]))


def short_title(pose: str) -> str:
    """``"Cow Pose"`` for ``"cow"`` (the title without the Sanskrit name)."""
    return yoga.POSES[pose].title.split(" (")[0]


def _pose_of(clip) -> str:
    return str(clip.metadata.get("pose", "")) if not isinstance(clip, str) else clip


def _same(a, b) -> bool:
    """Whether two texture values are the same look (arrays and textures by identity)."""
    if a is b:
        return True
    if isinstance(a, (str, Path)) and isinstance(b, (str, Path)):
        return str(a) == str(b)
    return isinstance(a, tuple) and isinstance(b, tuple) and a == b


_TITLES = {**accessories.TITLES, "face": "face"}


def _item_title(look_name: str, item: str) -> str:
    """``"navy hat"`` for the look ``"navy"`` (or ``"navy hat"``) of the hat, but ``"sunglasses"``, not
    ``"sunglasses glasses"``."""
    title = _TITLES[item]
    return look_name if title in look_name.lower() else f"{look_name} {title}"


def person_name(character) -> str:
    return f"{character.name} ({character.skinned_model.name})"


class OnScreen:
    """A selectable person in the current frame and their box ``(x0, y0, x1, y1)`` in pixels."""

    def __init__(self, character, box, depth: float = 0.0):
        self.character, self.box, self.depth = character, box, depth  # depth: metres from the camera

    @property
    def centre(self) -> tuple[float, float]:
        return (self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2

    @property
    def height(self) -> float:
        return self.box[3] - self.box[1]


def person_box(character, clip_from_world: np.ndarray, size) -> tuple | None:
    """The box around ``character``'s posed skeleton in pixels (padded to the outline of the body), or ``None`` when
    part of them is behind the camera."""
    w, h = size
    joints = character.pose[:, :3, 3]
    model = character.matrices()[0]
    world = joints @ model[:3, :3].T + model[:3, 3]
    clip = np.c_[world, np.ones(len(world))] @ clip_from_world.T
    if (clip[:, 3] <= 1e-6).any():
        return None
    ndc = clip[:, :2] / clip[:, 3:]
    x = (ndc[:, 0] * 0.5 + 0.5) * w
    y = (0.5 - ndc[:, 1] * 0.5) * h
    x0, x1, y0, y1 = x.min(), x.max(), y.min(), y.max()
    pad = 0.06 * max(x1 - x0, y1 - y0)
    return (float(x0 - pad), float(y0 - pad - 0.08 * (y1 - y0)), float(x1 + pad), float(y1 + pad))


class _Placing:
    """A pose asked of someone who stands waiting while a spot for it is found (see :meth:`Selection._place`)."""

    def __init__(self, character, clip, hold: bool):
        self.character, self.clip, self.hold = character, clip, hold
        self.future = None  # the search, once it was started
        self.cancel = threading.Event()
        self.asked = self.start = self.done = 0.0  # scene times: search started, the pose planned to start, found
        self.covers = 0.0  # seconds after ``start`` the clip may start: the walks the search allowed for
        self.seen = False  # the result was looked at
        self.here = None  # where the search had the person stand then
        self.tries = 0

    def drop(self) -> None:
        """Give up the search (a running one stops at its next spot)."""
        self.cancel.set()
        if self.future is not None:
            self.future.cancel()


class Selection:
    """Who is selected (``active`` in selection mode), and the poses asked of people (see the module docs).

    Call :meth:`update` after every rendered frame of the scene and :meth:`handle_key` with each key press.
    """

    def __init__(self, clips: YogaClips | None = None, wardrobe: Wardrobe | None = None):
        self.clips = clips if clips is not None else YogaClips()
        self.wardrobe = wardrobe if wardrobe is not None else Wardrobe()
        self.cast = None  # the known people's faces (recognition.cast.FaceCast), when the demo has some
        self.menu = 0  # which of menus()
        self.active = False
        self.selected = None  # the Character
        self.box: tuple | None = None  # its box in the last frame
        self.people: list[OnScreen] = []  # selectable people in the last frame, left to right
        self._scene = None
        self._size = (1, 1)
        self._last_centre: tuple | None = None
        self._requests: dict[int, tuple] = {}  # id(character) -> (character, pose, hold): waiting for the clip
        self._placing: dict[int, _Placing] = {}  # id(character) -> finding a spot, in the order asked
        self._pool = None  # the thread that looks for spots (one after another; each takes up to a few seconds)
        self._lead = LEAD[0]  # seconds the next spot is planned ahead: about how long finding it takes
        self._performers: list[Performer] = []
        self._can: dict[int, tuple] = {}
        self._footprints: dict[tuple, tuple] = {}
        self._standing: dict[int, tuple] = {}  # id(clip) -> (clip, standing(clip))

    # -- every frame ------------------------------------------------------------------------------------------

    def update(self, scene, size) -> str | None:
        """Start the poses whose clips are ready and keep the selection on someone in the picture (``scene`` was just
        rendered at ``size`` = (width, height)); returns a message for the user, if any."""
        if scene is not self._scene:
            self._forget()
            self._scene = scene
        self._size = tuple(size)
        messages = self._start_ready()
        if self.active:
            self.people = self._on_screen()
            messages.append(self._keep_selection())
        return "; ".join(m for m in messages if m) or None

    def _on_screen(self) -> list[OnScreen]:
        scene, size = self._scene, self._size
        clip_from_world = scene.camera.projection_matrix(size[0] / size[1]) @ scene.camera.view_matrix()
        eye = np.asarray(scene.camera.position, float)
        found = []
        for obj in scene:
            if not self._can_do_yoga(obj) or not obj.current_visible:
                continue
            box = person_box(obj, clip_from_world, size)
            if box is None:
                continue
            area = (box[2] - box[0]) * (box[3] - box[1])
            inside = (max(0.0, min(box[2], size[0]) - max(box[0], 0.0))
                      * max(0.0, min(box[3], size[1]) - max(box[1], 0.0)))
            share, height = KEEP if obj is self.selected else (1.0, 1.0)
            if (area > 0 and inside >= share * MIN_INSIDE * area
                    and self._standing_height(obj, clip_from_world, size) >= height * MIN_HEIGHT * size[1]):
                found.append(OnScreen(obj, box, float(np.linalg.norm(obj.current_position - eye))))
        found.sort(key=lambda p: p.centre[0])
        return found

    def _can_do_yoga(self, obj) -> bool:
        model = getattr(obj, "skinned_model", None)
        return model is not None and self._model_info(model)[0]

    def _model_info(self, model) -> tuple[bool, float]:
        """``(can do yoga, height in metres)``."""
        known = self._can.get(id(model))
        if known is None or known[0] is not model:
            known = self._can[id(model)] = (model, yoga.supports(model), model.height)
        return known[1:]

    def _standing_height(self, character, clip_from_world, size) -> float:
        """How tall ``character`` would be drawn standing where they are, in pixels (whatever pose they are in)."""
        foot = np.append(character.current_position, 1.0)
        head = foot + (0.0, self._model_info(character.skinned_model)[1] * character.current_scale[1], 0.0, 0.0)
        (_fx, fy, _fz, fw), (_hx, hy, _hz, hw) = clip_from_world @ foot, clip_from_world @ head
        if fw <= 1e-6 or hw <= 1e-6:
            return 0.0
        return abs(fy / fw - hy / hw) * 0.5 * size[1]

    def _keep_selection(self) -> str | None:
        here = next((p for p in self.people if p.character is self.selected), None)
        if here is not None:
            self.box, self._last_centre = here.box, here.centre
            return None
        lost = self.selected
        if not self.people:
            self.selected = self.box = None
            if lost is None:
                return None
            return f"{person_name(lost)} {self._gone(lost)}; nobody else in the picture"
        if self._last_centre is None:
            choice = max(self.people, key=lambda p: p.height)
        else:
            cx, cy = self._last_centre
            choice = min(self.people, key=lambda p: math.hypot(p.centre[0] - cx, p.centre[1] - cy))
        self._select(choice)
        if lost is None:
            return f"selected {person_name(choice.character)}"
        return f"{person_name(lost)} {self._gone(lost)}: selected {person_name(choice.character)}"

    def _gone(self, character) -> str:
        if self._scene.get(character.name) is not character:
            return "was removed"
        return "is hidden" if not character.current_visible else "left the picture"

    def _select(self, entry: OnScreen) -> None:
        self.selected, self.box, self._last_centre = entry.character, entry.box, entry.centre
        models, seen = [], set()  # this person's poses first, then those of the others in the picture
        for p in [entry] + sorted(self.people, key=lambda p: -p.height):
            key = self.clips.key(p.character.skinned_model)
            if key not in seen:
                seen.add(key)
                models.append(p.character.skinned_model)
        self.clips.prepare(models)

    # -- keys -------------------------------------------------------------------------------------------------

    def enter(self, scene, size) -> str:
        """Start selection mode on ``scene`` (shown at ``size``); returns a message for the user."""
        if scene is not self._scene:
            self._forget()
            self._scene = scene
        self._size = tuple(size)
        if not any(self._can_do_yoga(obj) for obj in scene):
            return "this scene has no 3D people that can do yoga"
        self.active = True
        self.people = self._on_screen()
        self.selected = None
        self._last_centre = None
        if not self.people:
            return "selection on: nobody is in the picture yet"
        self._select(max(self.people, key=lambda p: p.height))
        return f"selected {person_name(self.selected)}: number keys for yoga poses, Tab for the next person"

    def leave(self) -> None:
        self.active = False
        self.selected = self.box = self._last_centre = None
        self.people = []

    def handle_key(self, key: int, t: float) -> tuple[bool, str | None]:
        """``(handled, message)`` for a key pressed at scene time ``t`` (only keys of selection mode are handled)."""
        if not self.active:
            return False, None
        if key in (TAB, RIGHT, BACKTAB, LEFT):
            return True, self.step(1 if key in (TAB, RIGHT) else -1)
        all_menus = menus()
        if key in (UP, DOWN, ord("i")):
            if key == ord("i"):  # to the items and back to the poses
                items = len(all_menus) - 1
                self.menu = items if self.menu != items else self._yoga_menu
            else:
                self.menu = (self.menu + (1 if key == DOWN else -1)) % len(all_menus)
            if all_menus[self.menu][1] == "yoga":
                self._yoga_menu = self.menu
            return True, f"menu: {all_menus[self.menu][0]}"
        _title, kind, entries = all_menus[min(self.menu, len(all_menus) - 1)]
        if key in NUMBER_KEYS:
            k = NUMBER_KEYS[key] - 1
            if k >= len(entries):
                return True, None
            return True, self.do(entries[k], t) if kind == "yoga" else self.wear(entries[k], t)
        if key in STOP_KEYS:
            return True, self.stop(t) if kind == "yoga" else self.undress(t)
        if key in (27, ord("y")):
            self.leave()
            return True, "selection off"
        return False, None

    def step(self, direction: int) -> str:
        """Select the next person to the right (``1``) or left (``-1``) of the selected one in the picture, wrapping
        around: the leftmost after the rightmost (and the other way round)."""
        if not self.people:
            return "nobody in the picture to select"
        here = next((p for p in self.people if p.character is self.selected), None)
        x = here.centre[0] if here is not None else self._last_centre[0] if self._last_centre is not None else None
        order = self.people if direction > 0 else self.people[::-1]  # people are sorted left to right
        if x is None:
            choice = order[0]
        else:
            ahead = [p for p in order if p is not here and (p.centre[0] - x) * direction > 0]
            choice = ahead[0] if ahead else order[0]
        self._select(choice)
        return f"selected {person_name(self.selected)} ({self.people.index(choice) + 1} of {len(self.people)})"

    def select_at(self, point) -> str:
        """Select the person drawn at ``point`` (``(x, y)`` in pixels of the picture): of the people whose box holds
        it the one nearest the camera, else the box nearest the point if it is at most ``CLICK_SLACK`` of the picture
        height away."""
        if not self.people:
            return "nobody in the picture to select"
        x, y = point

        def gap(p):
            x0, y0, x1, y1 = p.box
            return math.hypot(max(x0 - x, 0.0, x - x1), max(y0 - y, 0.0, y - y1))

        hits = [p for p in self.people if gap(p) == 0.0]
        if hits:
            choice = min(hits, key=lambda p: p.depth)
        else:
            choice = min(self.people, key=gap)
            if gap(choice) > CLICK_SLACK * self._size[1]:
                return "nobody there to select"
        self._select(choice)
        return f"selected {person_name(self.selected)} ({self.people.index(choice) + 1} of {len(self.people)})"

    def do(self, pose: str, t: float) -> str | None:
        """Have the selected person do ``pose`` (from scene time ``t``, as soon as its clip is ready and a spot for it
        is found; :meth:`update` then says so)."""
        character = self.selected
        if character is None:
            return "nobody selected"
        performer = self._performer(character)
        act = performer.current(t)
        title, key = short_title(pose), id(character)
        coming = self._coming(key)  # asked for, not started yet
        if coming == pose:
            self._hold_later(key)
            return f"{person_name(character)} will keep holding {title} (0 stops)"
        if performer.queued is not None and _pose_of(performer.queued[0]) == pose and coming is None:
            performer.hold(t)
            return f"{person_name(character)} will keep holding {title} (0 stops)"
        same = (act is not None and act.name == pose and act.cut is None and performer.queued is None
                and not act.clip.metadata.get("waiting"))
        if same and coming is None and performer.hold(t):
            return f"{person_name(character)} keeps holding {title} (0 stops)"
        clip = self.clips.request(character.skinned_model, pose)
        self._drop(key)
        if clip is None:
            performer.stop(t)  # out of the pose they are in while the new one is prepared (it takes seconds)
            self._requests[key] = (character, pose, False)
            return f"preparing {title} for {person_name(character)} ..."
        self._requests.pop(key, None)
        self._place(character, performer, clip, t)
        return None

    def _coming(self, key: int) -> str | None:
        """The pose asked of someone that has not started yet (its clip is being prepared or a spot found)."""
        if key in self._requests:
            return self._requests[key][1]
        return _pose_of(self._placing[key].clip) if key in self._placing else None

    def _hold_later(self, key: int) -> None:
        if key in self._requests:
            self._requests[key] = (*self._requests[key][:2], True)
        if key in self._placing:
            self._placing[key].hold = True

    def _drop(self, key: int) -> bool:
        """Forget the spot being found for someone (``False`` if none is)."""
        entry = self._placing.pop(key, None)
        if entry is not None:
            entry.drop()
        return entry is not None

    def wear(self, item: str, t: float) -> str:
        """The selected person puts ``item`` on, or changes it to its next look in :attr:`wardrobe`, or (after the
        last look) takes it off."""
        character = self.selected
        if character is None:
            return "nobody selected"
        if item == "face":
            return self._face(character)
        if not accessories.supports(character.skinned_model):
            return f"{person_name(character)} cannot wear items"
        name = accessories.TITLES[item]
        looks = self.wardrobe.looks(item)
        current = accessories.wearing(character).get(item)
        index = 0 if current is None else next((k + 1 for k, (_n, spec) in enumerate(looks) if _same(spec, current.look)),
                                                len(looks))
        if index >= len(looks):
            accessories.dress(character, item, None, t=t)
            return f"{person_name(character)} takes the {name} off"
        try:
            worn = accessories.dress(character, item, looks[index][1], self.wardrobe, t=t)
        except (OSError, ValueError) as exc:
            return f"cannot put the {name} on {person_name(character)}: {exc}"
        again = "off" if index + 1 == len(looks) else "the next"
        return f"{person_name(character)}: {_item_title(worn.name, item)} ({index + 1} of {len(looks)}; again: {again})"

    def _face(self, character) -> str:
        """The selected person gets the next known person's face, or (after the last one) their own back."""
        from .characters import face_photo

        known = self.cast.known() if self.cast is not None else []
        if not known:
            return "nobody known yet: put photos of people in the faces folder (a folder per person)"
        if face_photo.head_layout(character.skinned_model) is None:
            return f"{person_name(character)} has no face to put a photo on"
        current = self.cast.wears(character)
        index = known.index(current) + 1 if current in known else 0
        if index >= len(known):
            return self.cast.put(character, None)
        again = "their own" if index + 1 == len(known) else "the next"
        return f"{self.cast.put(character, known[index])} ({index + 1} of {len(known)}; again: {again})"

    def undress(self, t: float) -> str:
        """The selected person takes everything off (a known person's face too)."""
        character = self.selected
        if character is None:
            return "nobody selected"
        worn = accessories.wearing(character)
        for item in worn:
            accessories.dress(character, item, None, t=t)
        if self.cast is not None and self.cast.wears(character) is not None:
            self.cast.put(character, None)
            worn = {**worn, "face": None}
        return f"{person_name(character)} takes everything off" if worn else f"{person_name(character)} wears nothing"

    def stop(self, t: float) -> str:
        """The selected person leaves their pose (and forgets one being prepared)."""
        character = self.selected
        if character is None:
            return "nobody selected"
        waiting = self._requests.pop(id(character), None) is not None
        waiting = self._drop(id(character)) or waiting
        performer = character.behaviour if isinstance(character.behaviour, Performer) else None
        if (performer is not None and performer.stop(t)) or waiting:
            return f"{person_name(character)} stops"
        return f"{person_name(character)} is not doing a pose"

    def reset(self) -> None:
        """The scene restarted: everyone goes back to just their behaviour."""
        for performer in self._performers:
            performer.reset()
        self._requests.clear()
        self._drop_all()

    def close(self) -> None:
        self.clips.close()
        self._drop_all()  # a search going on stops at its next spot: quitting does not wait for it
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None

    def _drop_all(self) -> None:
        for key in list(self._placing):
            self._drop(key)

    # -- poses ------------------------------------------------------------------------------------------------

    def _performer(self, character) -> Performer:
        if not isinstance(character.behaviour, Performer):
            character.behaviour = Performer(character.behaviour)
            self._performers.append(character.behaviour)
        return character.behaviour

    def _start_ready(self) -> list[str]:
        messages = []
        t = self._scene.time if self._scene is not None else 0.0
        for key, (character, pose, hold) in list(self._requests.items()):
            model = character.skinned_model
            clip = self.clips.get(model, pose)
            if clip is None:
                error = self.clips.error(model, pose)
                if error is None and not self.clips.pending(model, pose):
                    self.clips.request(model, pose)  # dropped meanwhile (the clips were closed): ask again
                if error is not None:
                    del self._requests[key]
                    messages.append(f"cannot do {short_title(pose)} with {model.name}: {error}")
                continue
            del self._requests[key]
            if self._scene.get(character.name) is not character or not character.current_visible:
                continue  # gone while the clip was being prepared
            self._place(character, self._performer(character), clip, t, hold)
        for key, entry in list(self._placing.items()):
            character, performer = entry.character, self._performer(entry.character)
            if self._scene.get(character.name) is not character or not character.current_visible:
                self._drop(key)  # gone meanwhile
                performer.stop(t)
                continue
            if entry.future is None or not entry.future.done():
                continue
            if not entry.seen:  # how long finding it took: plan the next one about that far ahead
                entry.seen = True
                self._lead = min(max(TURN_MIN, 1.25 * (entry.done - entry.asked)), LEAD[1])
            if t < entry.start:
                continue  # the spot is planned for then (the person stands waiting)
            try:
                heading, spot = entry.future.result() or (None, None)
            except Exception as exc:  # noqa: BLE001 - do it where they are then
                print(f"[selection] no spot found for {character.name}: {type(exc).__name__}: {exc}")
                heading, spot = None, None
            now = performer.peek(t)
            if entry.tries < TRIES and (_distance(now.position, entry.here) > WALK_FROM
                                        or t - entry.start + self._walk(performer, now, spot) > entry.covers + STALE):
                entry.future = None  # not from here, or found too late (the others moved on): look again
                continue
            del self._placing[key]
            clip = entry.clip
            act = performer.perform(clip, t, now.heading if heading is None else heading, spot, hold=entry.hold)
            current = performer.current(t)
            started = act is not None or (current is not None and current.clip is clip)  # after the wait, at once
            title = short_title(clip.metadata.get("pose", ""))
            messages.append(f"{person_name(character)}: {title}" + ("" if started else " (next)"))
        self._search(t)
        return messages

    @staticmethod
    def _walk(performer: Performer, state, spot) -> float:
        """Seconds ``performer`` (as ``state`` now) walks to ``spot`` before a pose there."""
        if spot is None or performer.walk is None or _distance(state.position, spot) <= WALK_FROM:
            return 0.0
        return performer.walk(state.position, state.heading, spot).duration

    def _place(self, character, performer: Performer, clip, t: float, hold: bool = False) -> None:
        """Have the person stop and stand waiting (:meth:`.Performer.wait`) from scene time ``t`` while a spot is found
        (on a thread) near them from which the camera sees the whole pose from its best side, with room for it (see
        :func:`choose_place`); :meth:`update` starts ``clip`` there once it is found.

        Spots are found one after another, in the order asked for, each when the pose before has started: so each
        search knows where the poses before it go. A search plans for the pose to start a little later (about as long
        as finding a spot takes, :attr:`_lead`), after a walk there of up to a few seconds. A spot found early waits
        for that moment; one found so late that the pose would start more than ``STALE`` after what the search
        allowed for (e.g. after a while on the webcam) is looked for again, since the others moved on meanwhile."""
        self._drop(id(character))
        performer.wait(self._standing_clip(clip), t)
        self._placing[id(character)] = _Placing(character, clip, hold)
        self._search(t)

    def _search(self, t: float) -> None:
        """Start finding the spot of the first pose waiting for one, unless a spot is being found."""
        if any(entry.future is not None for entry in self._placing.values()):
            return
        entry = next(iter(self._placing.values()), None)
        if entry is None:
            return
        character, clip = entry.character, entry.clip
        performer = self._performer(character)
        start = max(performer.free_at(t), t + self._lead)
        here = np.asarray(performer.peek(start).position, float)
        camera = self._scene.camera.current_position
        bearing = math.degrees(math.atan2(camera[0] - here[0], camera[2] - here[2]))
        view = float(clip.metadata.get("view", 0.0))
        views = [bearing - view] + ([bearing + view] if abs(math.sin(math.radians(view))) > 1e-6 else [])
        times, points = self._footprint(character, clip)
        reach = WALK_REACH if performer.can_walk else STEP_REACH
        # where the others are during each frame of the pose: it starts after the turn, or after a walk of up to
        # about 7 s, so each of those moments counts (as if the people then were all there)
        delays = (0.0, 2.5, 5.0, 7.5) if performer.can_walk else (0.0, 0.6)
        moments = [start + TURN_MIN + delay + times for delay in delays]
        tracks = np.concatenate([self._tracks(character, m) for m in moments], axis=1)
        bodies = np.concatenate([self._bodies(character, m) for m in moments], axis=1)
        span = float(np.abs(points[..., [0, 2]]).max()) + reach + 1.0
        distance = np.linalg.norm(tracks - here[[0, 2]], axis=-1)
        near = tracks[:, np.nan_to_num(distance, nan=np.inf).min(axis=0) < span]
        furniture = obstacles(self._scene, float(here[1]))
        window = clip.metadata.get("hold")
        holding = (times >= window[0]) & (times <= window[1]) if window else np.ones(len(times), bool)
        seen = self._rating(tracks, furniture, np.nonzero(holding)[0] if holding.any() else np.arange(len(times)))
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="yoga-place")
        entry.asked, entry.start, entry.here, entry.seen, entry.tries = t, start, here, False, entry.tries + 1
        entry.covers = delays[-1]
        entry.future = self._pool.submit(self._find, entry, self._scene, here, views, points, near, furniture,
                                         bodies=bodies, floor_rects=floors(self._scene), reach=reach, seen=seen,
                                         home=performer.home(start))

    @staticmethod
    def _find(entry: _Placing, scene, *args, **kwargs):
        """:func:`choose_place` on the thread (``None`` once dropped); notes the scene time it was done (scene time
        stands still while the scene is not shown, e.g. on the webcam)."""
        try:
            return choose_place(*args, cancelled=entry.cancel.is_set, **kwargs)
        finally:
            entry.done = scene.time

    def _standing_clip(self, clip):
        known = self._standing.get(id(clip))
        if known is None or known[0] is not clip:
            known = self._standing[id(clip)] = (clip, standing(clip))
        return known[1]

    def _footprint(self, character, clip):
        key = (id(clip), self.clips.key(character.skinned_model))
        known = self._footprints.get(key)
        if known is None or known[0] is not clip:
            known = self._footprints[key] = (clip, *footprint(character, clip))
        return known[1], known[2]

    def _tracks(self, me, times) -> np.ndarray:
        """``(T, O, 2)`` floor positions of the other people at scene ``times`` (nan while hidden)."""
        columns = []
        for obj in self._scene:
            if obj is me or getattr(obj, "skinned_model", None) is None:
                continue
            behaviour = obj.behaviour
            if behaviour is None:
                track = np.tile(obj.current_position[[0, 2]], (len(times), 1))
            elif hasattr(behaviour, "positions"):
                track = np.asarray(behaviour.positions(times), float)
            else:
                track = np.full((len(times), 2), np.nan)
                for k, t in enumerate(times):
                    state = behaviour(float(t))
                    if state.visible:
                        track[k] = np.asarray(state.position, float)[[0, 2]]
            columns.append(track)
        return np.stack(columns, axis=1) if columns else np.zeros((len(times), 0, 2))

    def _bodies(self, me, times) -> np.ndarray:
        """``(T, Q, 2)`` floor points of the bodies of the other people who are doing a pose at scene ``times`` (nan
        where nobody is)."""
        columns = []
        for obj in self._scene:
            performer = getattr(obj, "behaviour", None)
            if obj is me or not isinstance(performer, Performer) or not (performer.acts or performer.queued):
                continue
            acts = performer._planned(float(times.max()))
            for act in acts:
                if act.end <= times.min() or act.start >= times.max():
                    continue
                frames, points = self._footprint(obj, act.clip)
                track = np.full((len(times), points.shape[1], 2), np.nan)
                for k, t in enumerate(times):
                    if act.clip_start <= t < act.clip_end:
                        f = min(round(act.clip_time(float(t)) / (frames[1] - frames[0])), len(frames) - 1)
                        track[k] = place(points[f], act.heading, act.position)[:, [0, 2]]
                columns.append(track)
        return np.concatenate(columns, axis=1) if columns else np.zeros((len(times), 0, 2))

    def _rating(self, tracks: np.ndarray, furniture: np.ndarray, hold: np.ndarray):
        """``seen(world_points) -> share`` for :func:`choose_place`: the share of a pose's ``(T, P, 3)`` points that are
        in the picture and not hidden. A point is hidden in one of the pose's ``hold`` frames when another person
        (``tracks``, as :data:`PERSON` boxes) or a piece of ``furniture`` is in the line of sight to it; up to ``HIDDEN``
        of the points may be hidden, else the share hidden in the worst of those frames is taken off."""
        w, h = self._size
        camera = self._scene.camera
        clip_from_world = camera.projection_matrix(w / h) @ camera.view_matrix()
        eye = np.asarray(camera.current_position, float)
        margin = 0.02 * h

        def project(points):
            c = (points.reshape(-1, 3) @ clip_from_world[:, :3].T + clip_from_world[:, 3]).reshape(points.shape[:-1] + (4,))
            depth = c[..., 3]
            safe = np.where(depth > 1e-6, depth, np.nan)
            return (c[..., 0] / safe * 0.5 + 0.5) * w, (0.5 - c[..., 1] / safe * 0.5) * h, depth

        unit = np.array([(i & 1, (i >> 1) & 1, (i >> 2) & 1) for i in range(8)], float)  # cube corners

        def screen(lo, hi):  # boxes (..., 3) -> (..., 4) screen rectangles (nan when partly behind the camera), depth
            x, y, depth = project(lo[..., None, :] + (hi - lo)[..., None, :] * unit)
            return np.stack([x.min(-1), y.min(-1), x.max(-1), y.max(-1)], axis=-1), depth.min(-1)

        half = PERSON[0] / 2  # the people as boxes on the floor, in every frame (nan: nowhere)
        people_lo = np.stack([tracks[..., 0] - half, np.zeros(tracks.shape[:2]), tracks[..., 1] - half], axis=-1)
        people_hi = people_lo + (PERSON[0], PERSON[1], PERSON[0])
        people_lo, people_hi = people_lo[hold], people_hi[hold]  # (K, O, 3)
        people_rects, people_near = screen(people_lo, people_hi)
        things_lo, things_hi = furniture[:, [0, 4, 1]].reshape(-1, 3), furniture[:, [2, 5, 3]].reshape(-1, 3)
        things_rects, things_near = screen(things_lo, things_hi)  # (N, 4), (N,)

        def nearby(rects, near, x, y, depth):
            """Which boxes can hide some of the points: on screen around them and nearer than the farthest."""
            with np.errstate(invalid="ignore"):
                overlap = ((rects[..., 0] <= np.nanmax(x)) & (rects[..., 2] >= np.nanmin(x))
                           & (rects[..., 1] <= np.nanmax(y)) & (rects[..., 3] >= np.nanmin(y)) & (near < depth.max()))
            return overlap | (np.isnan(rects).any(axis=-1) & ~np.isnan(near))  # partly behind the camera: maybe

        def blocked(world, lo, hi) -> np.ndarray:
            """Which points ``(..., K, P, 3)`` a box (corners ``(K, O, 3)``, or ``(O, 3)`` in every frame) hides: the
            line of sight to the point goes through it, at least 0.1 m before the point (a slab test)."""
            if lo.shape[-2] == 0:
                return np.zeros(world.shape[:-1], bool)
            if lo.ndim == 2:
                lo, hi = lo[None], hi[None]
            ray = world - eye
            enter, leave = np.full(world.shape[:-1] + lo.shape[-2:-1], -np.inf), np.inf
            with np.errstate(divide="ignore", invalid="ignore"):
                for axis in range(3):  # (..., K, P, 1) against boxes (K, 1, O)
                    inverse = 1.0 / ray[..., axis, None]
                    a, b = (lo[:, None, :, axis] - eye[axis]) * inverse, (hi[:, None, :, axis] - eye[axis]) * inverse
                    enter, leave = np.maximum(enter, np.minimum(a, b)), np.minimum(leave, np.maximum(a, b))
                before = 1.0 - 0.1 / np.linalg.norm(ray, axis=-1)
                return ((enter <= leave) & (leave >= 0.0) & (enter < before[..., None])).any(axis=-1)

        def seen(world: np.ndarray) -> np.ndarray:
            x, y, depth = project(world)  # (..., T, P)
            inside = (x >= margin) & (x <= w - margin) & (y >= margin) & (y <= h - margin)
            share = np.where(np.isfinite(x).all(axis=(-2, -1)), inside.mean(axis=(-2, -1)), 0.0)
            held, xs, ys, ds = world[..., hold, :, :], x[..., hold, :], y[..., hold, :], depth[..., hold, :]
            if not np.isfinite(xs).any():
                return share
            people = nearby(people_rects, people_near, xs, ys, ds).any(axis=0)  # in any of the hold frames
            things = nearby(things_rects, things_near, xs, ys, ds)
            hidden = (blocked(held, people_lo[:, people], people_hi[:, people])
                      | blocked(held, things_lo[things], things_hi[things]))
            worst = (hidden & inside[..., hold, :]).mean(axis=-1).max(axis=-1)  # (...)
            return np.where(worst > HIDDEN, share - worst, share)

        return seen

    # -- what to show -----------------------------------------------------------------------------------------

    def describe(self, character, t: float) -> str:
        """What ``character`` is doing in yoga terms (empty when nothing)."""
        waiting = self._requests.get(id(character))
        if waiting is not None:
            return f"preparing {short_title(waiting[1])} ..."
        placing = self._placing.get(id(character))
        if placing is not None:
            return f"{short_title(_pose_of(placing.clip))}: finding a spot ..."
        performer = character.behaviour if isinstance(character.behaviour, Performer) else None
        act = performer.current(t) if performer is not None else None
        if act is None or act.clip.metadata.get("waiting"):
            return ""
        title = short_title(act.name)
        phase = act.phase(t)
        text = {"walking": f"{title}: walking to a free spot", "turning": f"{title}: getting ready", "into": title,
                "holding": f"{title}: holding" + (" (until 0)" if act.looping else ""),
                "out": f"{title}: back to standing", "back": f"{title}: done, walking back"}[phase]
        if performer.queued is not None:
            text += f", then {short_title(_pose_of(performer.queued[0]))}" + (" (held)" if performer.queued[3] else "")
        return text

    def wears(self, character) -> str:
        """What ``character`` wears, e.g. ``"navy hat, welcome sign, Ada's face"`` (empty when nothing)."""
        items = [_item_title(worn.name, item) for item, worn in accessories.wearing(character).items()]
        face = self.cast.wears(character) if self.cast is not None else None
        return ", ".join(items + ([f"{face}'s face"] if face else []))

    def detail(self, character, t: float) -> str:
        """The second line of the selected person's tag: what they are doing, else what they wear."""
        return self.describe(character, t) or self.wears(character)

    def status_lines(self, t: float) -> list[str]:
        if not self.active:
            return []
        if self.selected is None:
            return ["selected: nobody in the picture"]
        index = next((k for k, p in enumerate(self.people) if p.character is self.selected), 0)
        lines = [f"selected: {person_name(self.selected)}, {index + 1} of {len(self.people)} in the picture"]
        doing, wears = self.describe(self.selected, t), self.wears(self.selected)
        if doing:
            lines.append(f"yoga: {doing}")
        if wears:
            lines.append(f"wears: {wears}")
        return lines

    def help_lines(self) -> list[str]:
        """The key help of the current menu."""
        all_menus = menus()
        title, kind, entries = all_menus[min(self.menu, len(all_menus) - 1)]
        keys = [f"{k + 1} {short_title(e) if kind == 'yoga' else _TITLES[e]}"
                for k, e in enumerate(entries)]
        people = "Click a person, or Tab / Shift+Tab: next to the right / left | Up / Down: menu | Esc: done"
        if kind == "yoga":  # eight pose names do not fit one line: four to a line
            return [people + " | 0 stop",
                    f"[{title}] " + "  ".join(keys[:4]),
                    "  ".join(keys[4:]) + " | same number again: keep holding"]
        return [people,
                f"[{title}] {'  '.join(keys)} | again: next look, after the last: off | 0 all off"]

    _yoga_menu = 0  # the yoga menu ``I`` goes back to

    def _forget(self) -> None:
        """A new scene: the people of the old one are gone."""
        self.selected = self.box = self._last_centre = None
        self.people = []
        self._requests.clear()
        self._drop_all()
        self._performers = []
        self._footprints.clear()
        self._standing.clear()
