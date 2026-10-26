"""The demo's selection mode: pick one of the 3D people in the picture and have them do yoga poses.

``Y`` (or Tab) starts it with the most prominent person selected; Tab and Shift+Tab (or the arrow keys) go to the next
person to the right or left; the number keys make the selected person do a pose (1 = the first of
:data:`yoga.POSES`), the same key again keeps them holding it, 0 (or Backspace) stops; Esc or ``Y`` ends it. When the
selected person leaves the picture, is hidden or is removed from the scene, the person nearest to where they were is
selected instead.
"""

from __future__ import annotations

import math

import numpy as np

from .characters import yoga
from .characters.performer import TURN_MIN, Performer, choose_place, footprint, obstacles
from .characters.yoga_clips import YogaClips
from .display.window import BACKSPACE, BACKTAB, LEFT, RIGHT, TAB

POSE_KEYS = {ord(str(k + 1)): name for k, name in enumerate(yoga.POSES)}  # "1" -> the first pose, ...
STOP_KEYS = (ord("0"), BACKSPACE)
MIN_HEIGHT = 30  # pixels: people who would be drawn smaller than this standing (far away) cannot be selected
MIN_INSIDE = 0.6  # share of a person's box that must be in the picture
KEEP = 0.25, 0.8  # the selected person stays selected down to these shares of MIN_INSIDE and MIN_HEIGHT


def short_title(pose: str) -> str:
    """``"Cow Pose"`` for ``"cow"`` (the title without the Sanskrit name)."""
    return yoga.POSES[pose].title.split(" (")[0]


def person_name(character) -> str:
    return f"{character.name} ({character.skinned_model.name})"


class OnScreen:
    """A selectable person in the current frame and their box ``(x0, y0, x1, y1)`` in pixels."""

    def __init__(self, character, box):
        self.character, self.box = character, box

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


class Selection:
    """Who is selected (``active`` in selection mode), and the poses asked of people (see the module docs).

    Call :meth:`update` after every rendered frame of the scene and :meth:`handle_key` with each key press.
    """

    def __init__(self, clips: YogaClips | None = None):
        self.clips = clips if clips is not None else YogaClips()
        self.active = False
        self.selected = None  # the Character
        self.box: tuple | None = None  # its box in the last frame
        self.people: list[OnScreen] = []  # selectable people in the last frame, left to right
        self._scene = None
        self._size = (1, 1)
        self._last_centre: tuple | None = None
        self._requests: dict[int, tuple] = {}  # id(character) -> (character, pose): waiting for the clip
        self._performers: list[Performer] = []
        self._can: dict[int, tuple] = {}
        self._footprints: dict[tuple, tuple] = {}

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
                    and self._standing_height(obj, clip_from_world, size) >= height * MIN_HEIGHT):
                found.append(OnScreen(obj, box))
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
        if key in POSE_KEYS:
            return True, self.do(POSE_KEYS[key], t)
        if key in STOP_KEYS:
            return True, self.stop(t)
        if key in (27, ord("y")):
            self.leave()
            return True, "selection off"
        return False, None

    def step(self, direction: int) -> str:
        """Select the next person to the right (``1``) or left (``-1``), wrapping around."""
        if not self.people:
            return "nobody in the picture to select"
        index = next((k for k, p in enumerate(self.people) if p.character is self.selected), None)
        if index is None:
            index = 0 if direction > 0 else len(self.people) - 1
        else:
            index = (index + direction) % len(self.people)
        self._select(self.people[index])
        return f"selected {person_name(self.selected)} ({index + 1} of {len(self.people)})"

    def do(self, pose: str, t: float) -> str:
        """Have the selected person do ``pose`` (from scene time ``t``, as soon as its clip is ready)."""
        character = self.selected
        if character is None:
            return "nobody selected"
        performer = self._performer(character)
        act = performer.current(t)
        title = short_title(pose)
        same = act is not None and act.name == pose and act.cut is None and performer.queued is None
        if same and performer.hold(t):
            return f"{person_name(character)} keeps holding {title} (0 stops)"
        clip = self.clips.request(character.skinned_model, pose)
        if clip is None:
            performer.stop(t)  # out of the pose they are in while the new one is prepared
            self._requests[id(character)] = (character, pose)
            return f"preparing {title} for {person_name(character)} ..."
        self._requests.pop(id(character), None)
        return self._begin(character, performer, clip, t)

    def stop(self, t: float) -> str:
        """The selected person leaves their pose (and forgets one being prepared)."""
        character = self.selected
        if character is None:
            return "nobody selected"
        waiting = self._requests.pop(id(character), None) is not None
        performer = character.behaviour if isinstance(character.behaviour, Performer) else None
        if (performer is not None and performer.stop(t)) or waiting:
            return f"{person_name(character)} stops"
        return f"{person_name(character)} is not doing a pose"

    def reset(self) -> None:
        """The scene restarted: everyone goes back to just their behaviour."""
        for performer in self._performers:
            performer.reset()
        self._requests.clear()

    def close(self) -> None:
        self.clips.close()

    # -- poses ------------------------------------------------------------------------------------------------

    def _performer(self, character) -> Performer:
        if not isinstance(character.behaviour, Performer):
            character.behaviour = Performer(character.behaviour)
            self._performers.append(character.behaviour)
        return character.behaviour

    def _start_ready(self) -> list[str]:
        messages = []
        t = self._scene.time if self._scene is not None else 0.0
        for key, (character, pose) in list(self._requests.items()):
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
            messages.append(self._begin(character, self._performer(character), clip, t))
        return messages

    def _begin(self, character, performer: Performer, clip, t: float) -> str:
        """Start ``clip`` (or queue it after the current pose), facing so the camera sees it well and the mat is
        clear of the others."""
        start = performer.free_at(t)
        here = np.asarray(performer(start).position, float)
        camera = self._scene.camera.current_position
        bearing = math.degrees(math.atan2(camera[0] - here[0], camera[2] - here[2]))
        view = float(clip.metadata.get("view", 0.0))
        views = [bearing - view] + ([bearing + view] if abs(math.sin(math.radians(view))) > 1e-6 else [])
        times, points = self._footprint(character, clip)
        others = self._others(character, here, start + TURN_MIN + times, float(np.abs(points).max()) + 1.5)
        furniture = obstacles(self._scene, float(here[1]))
        heading, spot = choose_place(here, views, points, others, furniture)
        act = performer.perform(clip, t, heading, spot)
        title = short_title(clip.metadata.get("pose", ""))
        return f"{person_name(character)}: {title}" + ("" if act is not None else " (next)")

    def _footprint(self, character, clip):
        key = (id(clip), self.clips.key(character.skinned_model))
        known = self._footprints.get(key)
        if known is None or known[0] is not clip:
            known = self._footprints[key] = (clip, *footprint(character, clip))
        return known[1], known[2]

    def _others(self, me, here, times, reach: float) -> np.ndarray:
        """``(T, O, 2)`` floor positions of the other people within ``reach`` of ``here`` at scene ``times``."""
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
            d = np.linalg.norm(track - here[[0, 2]], axis=1)
            if np.isfinite(d).any() and np.nanmin(d) < reach:
                columns.append(track)
        return np.stack(columns, axis=1) if columns else np.zeros((len(times), 0, 2))

    # -- what to show -----------------------------------------------------------------------------------------

    def describe(self, character, t: float) -> str:
        """What ``character`` is doing in yoga terms (empty when nothing)."""
        waiting = self._requests.get(id(character))
        if waiting is not None:
            return f"preparing {short_title(waiting[1])} ..."
        performer = character.behaviour if isinstance(character.behaviour, Performer) else None
        act = performer.current(t) if performer is not None else None
        if act is None:
            return ""
        title = short_title(act.name)
        phase = act.phase(t)
        text = {"turning": f"{title}: getting ready", "into": title,
                "holding": f"{title}: holding" + (" (until 0)" if act.looping else ""),
                "out": f"{title}: back to standing"}[phase]
        if performer.queued is not None:
            text += f", then {short_title(performer.queued[0].metadata.get('pose', ''))}"
        return text

    def status_lines(self, t: float) -> list[str]:
        if not self.active:
            return []
        if self.selected is None:
            return ["selected: nobody in the picture"]
        index = next((k for k, p in enumerate(self.people) if p.character is self.selected), 0)
        lines = [f"selected: {person_name(self.selected)}, {index + 1} of {len(self.people)} in the picture"]
        doing = self.describe(self.selected, t)
        if doing:
            lines.append(f"yoga: {doing}")
        return lines

    @staticmethod
    def help_lines() -> list[str]:
        poses = "  ".join(f"{chr(k)} {short_title(p)}" for k, p in POSE_KEYS.items())
        return ["Tab / Shift+Tab next / previous person | 0 stop | same number again: keep holding | Esc done",
                poses]

    def _forget(self) -> None:
        """A new scene: the people of the old one are gone."""
        self.selected = self.box = self._last_centre = None
        self.people = []
        self._requests.clear()
        self._performers = []
        self._footprints.clear()
