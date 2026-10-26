"""Interrupting what a person does for a yoga pose, and finding room for it.

A :class:`Performer` wraps a character's behaviour. Asked to :meth:`~Performer.perform` a clip, the person stops
where they are, turns to face the right way, does the clip (from standing into the pose and back) and then carries on
with their behaviour exactly where they left it: the behaviour's time stands still while they do yoga. A pose that is
cut short (:meth:`~Performer.stop`, or a new pose) leaves the hold early, or goes back the way it came if that is
quicker; a pose asked to :meth:`~Performer.hold` keeps holding until it is stopped.

:func:`choose_place` picks the heading (and, if need be, a small step aside) that shows the pose from the angle it is
best seen from while keeping the mat clear of the other people and of the furniture.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, replace

import numpy as np

from .model import AnimationClip, CharacterState, Layer

TURN_SPEED = 150.0  # degrees per second when turning on the spot before and after a pose
TURN_MIN = 0.6  # seconds: the shortest turn, also the blend into the clip and back into the behaviour
SKIP_BLEND = 0.6  # seconds of crossfade when a hold is cut short
# chains of bones whose joints (and the points between them) outline the body on the floor
FOOTPRINT_CHAINS = [("Bip01 Pelvis", "Bip01 Spine1", "Bip01 Neck", "Bip01 Head")] + [
    tuple(f"Bip01 {side} {bone}" for bone in chain) for side in "LR"
    for chain in (("UpperArm", "Forearm", "Hand", "Finger2"), ("Thigh", "Calf", "Foot", "Toe0"))]


def _ease(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


def _wrap(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def _turn_time(degrees: float) -> float:
    return max(TURN_MIN, abs(degrees) / TURN_SPEED)


def _mix(before: list, after: list, w: float) -> list:
    """Layers fading from ``before`` (weight 1 - w) to ``after`` (weight w)."""
    if w >= 1.0:
        return list(after)
    if w <= 0.0:
        return list(before)
    return ([replace(layer, weight=layer.weight * (1.0 - w)) for layer in before]
            + [replace(layer, weight=layer.weight * w) for layer in after])


@dataclass
class Act:
    """One clip done by a :class:`Performer`, from scene time ``start``.

    ``before`` is how the person looked when it began (it fades out during the turn), ``turn`` the seconds spent
    turning from ``before.heading`` to ``heading`` (and stepping from ``before.position`` to ``position``) before the
    clip starts. ``hold`` is the clip's main hold ``(t0, t1)`` in clip time (from ``clip.metadata``).
    """

    clip: AnimationClip
    start: float
    position: tuple
    heading: float
    before: CharacterState
    turn: float
    hold: tuple | None
    looping: bool = False
    last: CharacterState | None = None  # how the person looked when cut short while still turning
    cut: float | None = None  # scene time it was cut short
    cut_at: float = 0.0  # clip time then
    mode: str = ""  # "back" the way it came, "through" (on to the hold, then out of it), "skip" (out of the hold)

    @property
    def name(self) -> str:
        return str(self.clip.metadata.get("pose", self.clip.name))

    @property
    def end(self) -> float:
        """Scene time the clip is over (``inf`` while it keeps holding)."""
        if self.cut is None:
            return math.inf if self.looping and self.hold else self.start + self.turn + self.clip.duration
        if self.mode == "back":
            return self.cut + self.cut_at
        if self.mode == "through":
            return self.cut + (self.hold[0] - self.cut_at) + (self.clip.duration - self.hold[1])
        if self.mode == "skip":
            return self.cut + (self.clip.duration - self.hold[1])
        return self.cut + (self.clip.duration - self.cut_at)

    def clip_time(self, t: float) -> float:
        """Where in the clip the person is at scene time ``t``."""
        if self.cut is not None and t >= self.cut:
            u = t - self.cut
            if self.mode == "back":
                return max(0.0, self.cut_at - u)
            if self.mode == "through":
                c = self.cut_at + u
                return c + (self.hold[1] - self.hold[0]) if c >= self.hold[0] else c  # the hold ends as it began
            if self.mode == "skip":
                return self.hold[1] + u
            return self.cut_at + u
        c = max(0.0, t - self.start - self.turn)
        if self.looping and self.hold and c > self.hold[0]:
            c = self.hold[0] + (c - self.hold[0]) % (self.hold[1] - self.hold[0])
        return min(c, self.clip.duration)

    def layers(self, t: float) -> list:
        """The clip layers at scene time ``t`` (two while the cut out of a hold crossfades)."""
        if self.cut is not None and self.mode == "skip" and t - self.cut < SKIP_BLEND:
            u = t - self.cut
            w = _ease(u / SKIP_BLEND)
            return [Layer(self.clip, min(self.cut_at + u, self.clip.duration), 1.0 - w),
                    Layer(self.clip, self.hold[1] + u, w)]
        return [Layer(self.clip, self.clip_time(t))]

    def heading_at(self, t: float) -> float:
        a = self.before.heading
        return a + _wrap(self.heading - a) * _ease((t - self.start) / self.turn)

    def position_at(self, t: float) -> np.ndarray:
        a, b = np.asarray(self.before.position, float), np.asarray(self.position, float)
        return a + (b - a) * _ease((t - self.start) / self.turn)

    def phase(self, t: float) -> str:
        """``"turning"``, ``"into"`` the pose, ``"holding"``, ``"out"`` of it (back to standing)."""
        if self.cut is not None and t >= self.cut:
            return "out"
        if t < self.start + self.turn:
            return "turning"
        c = self.clip_time(t)
        if not self.hold:
            return "into"
        return "into" if c < self.hold[0] else "holding" if c < self.hold[1] or self.looping else "out"


class Performer:
    """A behaviour (``t -> CharacterState``) that runs ``base`` and can break off for clips (see the module docs).

    ``base`` may be ``None`` (standing at the origin). Attributes it does not have itself are looked up on ``base``.
    Scene time must not run backwards past an act; :meth:`reset` forgets every act (call it when the scene restarts).
    """

    def __init__(self, base):
        self.base = base
        self.acts: list[Act] = []  # in time order; acts that are over stay, they shift the base's time
        self.queued: tuple | None = None  # (clip, heading, position) to do when the current act is over

    def __getattr__(self, name):
        if name == "base":
            raise AttributeError(name)
        return getattr(self.base, name)

    def reset(self) -> None:
        self.acts, self.queued = [], None

    # -- asking for poses ---------------------------------------------------------------------------------------

    def current(self, t: float) -> Act | None:
        """The act going on at scene time ``t``, if any."""
        self._advance(t)
        return next((a for a in reversed(self.acts) if a.start <= t < a.end), None)

    def free_at(self, t: float) -> float:
        """When a clip asked for at ``t`` would start: ``t``, or the end of the current act once it is cut short."""
        act = self.current(t)
        if act is None:
            return t
        if act.cut is None:
            probe = replace(act)
            _cut(probe, t)
            return probe.end
        return act.end

    def perform(self, clip: AnimationClip, t: float, heading: float, position=None) -> Act | None:
        """Do ``clip`` from scene time ``t`` facing ``heading`` at ``position`` (default: where the person is).

        If an act is going on, it is cut short and ``clip`` follows it (the act is returned only when it started now).
        """
        self._forget_after(t)
        act = self.current(t)
        if act is not None:
            self._cut(act, t)
            self.queued = (clip, heading, position)
            return None
        return self._start(clip, t, heading, position)

    def hold(self, t: float) -> bool:
        """Keep holding the current pose (until :meth:`stop` or another pose); ``False`` if it is already leaving."""
        act = self.current(t)
        if act is None or act.cut is not None or not act.hold or act.phase(t) == "out":
            return False
        act.looping = True
        return True

    def stop(self, t: float) -> bool:
        """Leave the current pose now (and forget a pose waiting to follow it); ``False`` if there is none."""
        stopped = self._forget_after(t)
        act = self.current(t)
        if act is not None and act.cut is None:
            self._cut(act, t)
            stopped = True
        return stopped

    def _forget_after(self, t: float) -> bool:
        """Drop the queued clip, also if it was already started for a moment after ``t`` (a look ahead does that)."""
        later = [a for a in self.acts if a.start > t]
        self.acts = [a for a in self.acts if a.start <= t]
        dropped = bool(later) or self.queued is not None
        self.queued = None
        return dropped

    def _cut(self, act: Act, t: float) -> None:
        if act.cut is None:
            if t < act.start + act.turn:  # still turning: fade back from how the person looks right now
                act.last = self(t)
            _cut(act, t)

    def _start(self, clip, t, heading, position) -> Act:
        before = self(t)
        position = tuple(before.position) if position is None else tuple(position)
        moved = float(np.hypot(*(np.asarray(position, float) - np.asarray(before.position, float))[[0, 2]]))
        turn = max(_turn_time(_wrap(heading - before.heading)), moved / 0.5)  # step aside at 0.5 m/s
        hold = clip.metadata.get("hold")
        act = Act(clip, t, position, float(heading), before, turn, tuple(hold) if hold else None)
        self.acts.append(act)
        return act

    def _advance(self, t: float) -> None:
        """Start the queued clip when the act before it is over (at that moment, whenever this is evaluated)."""
        while self.queued is not None and self.acts and self.acts[-1].end <= t:
            clip, heading, position = self.queued
            self.queued = None
            self._start(clip, self.acts[-1].end, heading, position)

    # -- evaluating ---------------------------------------------------------------------------------------------

    def _base_time(self, t: float) -> float:
        """The base behaviour's own time: scene time minus the time spent doing yoga."""
        return t - sum(max(0.0, min(t, act.end) - act.start) for act in self.acts if act.start < t)

    def _base(self, t: float) -> CharacterState:
        if self.base is None:
            return CharacterState((0.0, 0.0, 0.0), 0.0, [])
        return self.base(self._base_time(t))

    def __call__(self, t: float) -> CharacterState:
        self._advance(t)
        act = next((a for a in reversed(self.acts) if a.start <= t), None)
        if act is not None and t < act.end:
            w = _ease((t - act.start) / act.turn)
            before = act.before
            return CharacterState(tuple(act.position_at(t)), act.heading_at(t), _mix(before.layers, act.layers(t), w),
                                  head_yaw=before.head_yaw * (1.0 - w), head_pitch=before.head_pitch * (1.0 - w),
                                  visible=before.visible)
        state = self._base(t)
        if act is None:
            return state
        last = act.end
        src = act.last or CharacterState(tuple(act.position_at(last)), act.heading_at(last), act.layers(last))
        back = _turn_time(_wrap(state.heading - src.heading))
        if t - last >= back:
            return state
        w = _ease((t - last) / back)  # back into the behaviour (turning, stepping back to where it is)
        position = np.asarray(src.position, float)
        position = position + (np.asarray(state.position, float) - position) * w
        return CharacterState(tuple(position), src.heading + _wrap(state.heading - src.heading) * w,
                              _mix(src.layers, state.layers, w),
                              head_yaw=src.head_yaw + (state.head_yaw - src.head_yaw) * w,
                              head_pitch=src.head_pitch + (state.head_pitch - src.head_pitch) * w,
                              visible=state.visible)

    def positions(self, times) -> np.ndarray:
        """``(x, z)`` at each of ``times`` (nan while hidden), like ``Behaviour.positions``."""
        times = np.asarray(times, float)
        if len(times):
            self._advance(float(times.max()))
        out = np.full((len(times), 2), np.nan)
        rest = []
        for k, t in enumerate(times):
            act = next((a for a in reversed(self.acts) if a.start <= t), None)
            if act is not None and t < act.end:
                out[k] = act.position_at(t)[[0, 2]]
            else:
                rest.append(k)
        if rest:
            base_times = [self._base_time(float(times[k])) for k in rest]
            if self.base is None:
                out[rest] = 0.0
            elif hasattr(self.base, "positions"):
                out[rest] = self.base.positions(base_times)
            else:
                for k, bt in zip(rest, base_times):
                    s = self.base(bt)
                    if s.visible:
                        out[k] = np.asarray(s.position, float)[[0, 2]]
        return out


def _cut(act: Act, t: float) -> None:
    """Cut ``act`` short at scene time ``t``: the quickest way back to standing."""
    c = act.clip_time(t)
    act.looping, act.cut, act.cut_at = False, t, c
    if t < act.start + act.turn or not act.hold:
        act.mode = "back" if t < act.start + act.turn else "finish"
        return
    h0, h1 = act.hold
    if c < h0:
        act.mode = "back" if c <= (h0 - c) + (act.clip.duration - h1) else "through"
    elif c < h1:
        act.mode = "skip"
    else:
        act.mode = "finish"


# -- where to do it -------------------------------------------------------------------------------------------


def footprint(character, clip: AnimationClip, step: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """``(times, points)``: clip times every ``step`` seconds and the ``(T, P, 2)`` floor positions (character space
    x, z) of points along the trunk, arms and legs of ``character`` doing ``clip`` then (joints and the thirds
    between them, so no thin counter fits between two of them)."""
    skeleton = character.skinned_model.skeleton
    chains = [[b for b in (skeleton.find(n) for n in chain) if b is not None] for chain in FOOTPRINT_CHAINS]
    pairs = [pair for chain in chains for pair in itertools.pairwise(chain)]
    if not pairs:
        pairs = [(b, b) for b in range(len(skeleton))]
    first, second = np.array(pairs).T
    times = np.append(np.arange(0.0, clip.duration, step), clip.duration)
    points = []
    for t in times:
        joints = character.poser.global_pose([Layer(clip, float(t))])[:, [0, 2], 3]
        a, b = joints[first], joints[second]
        points.append(np.concatenate([a, a + (b - a) / 3, a + 2 * (b - a) / 3, b]))
    return times, np.array(points)


def obstacles(scene, floor: float, skip=()) -> np.ndarray:
    """``(N, 4)`` floor rectangles ``(x0, z0, x1, z1)`` of the scene's furniture: objects other than people, floors
    and pictures that reach into knee to chest height above ``floor``; ``skip``: objects to leave out."""
    from ..scene.scene import Plane, Quad

    rects = []
    for obj in scene:
        if obj in skip or hasattr(obj, "skinned_model") or isinstance(obj, (Plane, Quad)) or not obj.current_visible:
            continue
        try:
            low, high = obj.extents()
        except Exception:  # noqa: BLE001 - an object without a usable mesh is no obstacle
            continue
        centre = obj.current_position
        bottom, top = centre[1] + low[1], centre[1] + high[1]
        if top < floor + 0.1 or bottom > floor + 1.2:
            continue
        rects.append((centre[0] + low[0], centre[2] + low[2], centre[0] + high[0], centre[2] + high[2]))
    return np.array(rects, float).reshape(-1, 4)


def _rect_distance(points: np.ndarray, rects: np.ndarray) -> np.ndarray:
    """Distance of each ``(..., 2)`` point to each rectangle: ``(..., N)``."""
    p = points[..., None, :]
    dx = np.maximum(np.maximum(rects[:, 0] - p[..., 0], 0.0), p[..., 0] - rects[:, 2])
    dz = np.maximum(np.maximum(rects[:, 1] - p[..., 1], 0.0), p[..., 1] - rects[:, 3])
    return np.hypot(dx, dz)


def choose_place(position, views, points: np.ndarray, others: np.ndarray, furniture: np.ndarray,
                 people_gap: float = 0.5, furniture_gap: float = 0.15) -> tuple[float, tuple]:
    """``(heading, position)`` for a pose: the first of ``views`` (preferred headings, best first) whose mat is clear,
    turned by up to 45 degrees and stepping up to 0.6 m aside if need be.

    ``points``: the pose's ``(T, P, 2)`` footprint (see :func:`footprint`); ``others``: ``(T, O, 2)`` where the other
    people are at those moments (nan = nowhere); ``furniture``: ``(N, 4)`` rectangles (see :func:`obstacles`). Clear
    means every point at least ``people_gap`` from every person and ``furniture_gap`` from the furniture; when no
    choice is clear, the one with the most room.
    """
    position = np.asarray(position, float)
    steps = [(0.0, 0.0)] + [(r * math.sin(a), r * math.cos(a)) for r in (0.3, 0.6)
                            for a in np.radians(np.arange(0, 360, 45))]
    best, best_room = None, -math.inf
    for dx, dz in steps:
        spot = position + (dx, 0.0, dz)
        for turn in (0.0, 15.0, -15.0, 30.0, -30.0, 45.0, -45.0):
            for view in views:
                heading = float(view) + turn
                c, s = math.cos(math.radians(heading)), math.sin(math.radians(heading))
                x = spot[0] + c * points[..., 0] + s * points[..., 1]
                z = spot[2] - s * points[..., 0] + c * points[..., 1]
                world = np.stack([x, z], axis=-1)  # (T, P, 2)
                room = math.inf
                if others.size:
                    d = np.linalg.norm(world[:, :, None, :] - others[:, None, :, :], axis=-1)
                    d = d[np.isfinite(d)]
                    if d.size:
                        room = float(d.min()) - people_gap
                if len(furniture):
                    room = min(room, float(_rect_distance(world, furniture).min()) - furniture_gap)
                if room >= 0.0:
                    return heading, tuple(spot)
                if room > best_room:
                    best, best_room = (heading, tuple(spot)), room
    return best if best is not None else (float(views[0]), tuple(position))
