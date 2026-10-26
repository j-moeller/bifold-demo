"""Interrupting what a person does for a yoga pose, and finding room for it.

A :class:`Performer` wraps a character's behaviour. Asked to :meth:`~Performer.perform` a clip at a place, the person
stops what they are doing, walks there if it is more than a step away (with their behaviour's own walk clips), turns to
face the right way, does the clip (from standing into the pose and back), walks back and carries on with their
behaviour exactly where they left it: the behaviour's time stands still meanwhile. A pose that is cut short
(:meth:`~Performer.stop`, or a new pose) leaves the hold early, or goes back the way it came if that is quicker; a new
pose follows at once, without walking back first. A pose asked to :meth:`~Performer.hold` keeps holding until it is
stopped. Since the behaviour's time stood still, a person who walks around is later on their way than before; plans
that time several people's loops against each other (``fair.keep_apart``) no longer hold for them.

:func:`choose_place` picks where and facing which way: a spot as near as possible from which the camera sees the
whole pose from the side it is best seen from, with the mat clear of the other people and of the furniture and on one
floor level (not half on a booth platform).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field, replace

import numpy as np

from .model import AnimationClip, CharacterState, Layer

TURN_SPEED = 150.0  # degrees per second when turning on the spot before and after a pose
TURN_MIN = 0.6  # seconds: the shortest turn, also the blend into the clip and back into the behaviour
SKIP_BLEND = 0.6  # seconds of crossfade when a hold is cut short
WALK_BLEND = 0.5  # seconds of crossfade into and out of a walk
WALK_FROM = 0.3  # metres: farther than this the person walks there (when the behaviour has walk clips)
STEP_SPEED = 0.5  # m/s of a shorter step aside (a slide during the turn)
# chains of bones whose joints (and the points between them) outline the body
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


def _blend(a: CharacterState, b: CharacterState, w: float, position=None) -> CharacterState:
    """``a`` fading into ``b``: layers crossfaded, heading turned the short way, position and head in between."""
    if position is None:
        pa = np.asarray(a.position, float)
        position = tuple(pa + (np.asarray(b.position, float) - pa) * w)
    return CharacterState(tuple(position), a.heading + _wrap(b.heading - a.heading) * w, _mix(a.layers, b.layers, w),
                          head_yaw=a.head_yaw + (b.head_yaw - a.head_yaw) * w,
                          head_pitch=a.head_pitch + (b.head_pitch - a.head_pitch) * w, visible=b.visible)


class Leg:
    """A short walk from ``start`` to ``goal`` (``(x, y, z)``; the floor stays at the start's height), played by a
    one-walk :class:`~.behaviour.Behaviour` with ``library``'s clips; ``duration`` seconds until it arrives."""

    def __init__(self, library, style: dict, start, heading: float, goal):
        from .behaviour import Behaviour, Walk

        self.floor = float(start[1])
        self.goal = (float(goal[0]), self.floor, float(goal[2]))
        self.behaviour = Behaviour([Walk([(goal[0], goal[2])])], library, start=(start[0], start[2]),
                                   heading=heading, style=style)
        self.duration = next((p.t1 for p in self.behaviour._pieces if p.kind == "walk"), 0.0)

    def __call__(self, u: float) -> CharacterState:
        state = self.behaviour(min(max(u, 0.0), self.duration))
        x, _y, z = state.position
        if u >= self.duration:
            x, z = self.goal[0], self.goal[2]
        return replace(state, position=(x, self.floor, z))


def walker(base):
    """``(start, heading, goal) -> Leg`` walking like ``base`` (a behaviour with ``library`` and ``style``), or
    ``None`` when it has no walk clips."""
    library, style = getattr(base, "library", None), getattr(base, "style", None)
    role = getattr(style, "walk_role", None)
    if library is None or role is None or not hasattr(library, "choices") or not library.choices(role, style.sex):
        return None
    clip = library.choices(role, style.sex)[0]
    spec = {"sex": style.sex, "age_group": style.age, "walk_role": role, "stride_scale": style.stride,
            "speed": style.walk_speed(clip)}
    return lambda start, heading, goal: Leg(library, spec, start, heading, goal)


@dataclass
class Act:
    """One clip done by a :class:`Performer`, from scene time ``start``.

    The person (as ``before``) first walks to ``position`` (``approach``, if any), then turns for ``turn`` seconds to
    ``heading`` (or, without a walk, steps there meanwhile) and does ``clip``; ``hold`` is its main hold ``(t0, t1)``
    in clip time (from ``clip.metadata``). Afterwards they walk back ``home`` (``retreat``, if any): where their
    behaviour carries on.
    """

    clip: AnimationClip
    start: float
    position: tuple
    heading: float
    before: CharacterState
    turn: float
    hold: tuple | None
    home: tuple = (0.0, 0.0, 0.0)
    approach: Leg | None = None
    retreat: Leg | None = None
    looping: bool = False
    last: CharacterState | None = None  # how the person looked when cut short before the clip began
    stopped: float | None = None  # scene time the walk back was broken off (for a new pose)
    left: CharacterState | None = None  # how the person looked then
    cut: float | None = None  # scene time it was cut short
    cut_at: float = 0.0  # clip time then
    mode: str = ""  # "back" the way it came, "through" (on to the hold, then out of it), "skip" (out of the hold)
    _arrival: CharacterState | None = field(default=None, repr=False)
    _back: tuple | None = field(default=None, repr=False)  # how the behaviour takes over again (see Performer)

    @property
    def name(self) -> str:
        return str(self.clip.metadata.get("pose", self.clip.name))

    @property
    def arrived(self) -> float:
        """Scene time the person is at ``position`` (and starts turning)."""
        return self.start + (self.approach.duration if self.approach is not None else 0.0)

    @property
    def clip_start(self) -> float:
        return self.arrived + self.turn

    @property
    def clip_end(self) -> float:
        """Scene time the clip is over (``inf`` while it keeps holding)."""
        if self.cut is None:
            return math.inf if self.looping and self.hold else self.clip_start + self.clip.duration
        if self.mode == "back":
            return self.cut + self.cut_at
        if self.mode == "through":
            return self.cut + (self.hold[0] - self.cut_at) + (self.clip.duration - self.hold[1])
        if self.mode == "skip":
            return self.cut + (self.clip.duration - self.hold[1])
        return self.cut + (self.clip.duration - self.cut_at)

    @property
    def end(self) -> float:
        """Scene time the person is back where their behaviour carries on (or starts a new pose on the way)."""
        if self.stopped is not None:
            return self.stopped
        return self.clip_end + (self.retreat.duration if self.retreat is not None else 0.0)

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
                return min(self.hold[1] + u, self.clip.duration)
            return min(self.cut_at + u, self.clip.duration)
        c = max(0.0, t - self.clip_start)
        if self.looping and self.hold and c > self.hold[0]:
            c = self.hold[0] + (c - self.hold[0]) % (self.hold[1] - self.hold[0])
        return min(c, self.clip.duration)

    def layers(self, t: float) -> list:
        """The clip layers at scene time ``t`` (two while the cut out of a hold crossfades)."""
        if self.cut is not None and self.mode == "skip" and t - self.cut < SKIP_BLEND:
            u = t - self.cut
            w = _ease(u / SKIP_BLEND)
            return [Layer(self.clip, min(self.cut_at + u, self.clip.duration), 1.0 - w),
                    Layer(self.clip, min(self.hold[1] + u, self.clip.duration), w)]
        return [Layer(self.clip, self.clip_time(t))]

    def arrival(self) -> CharacterState:
        """How the person looks when they start turning into the clip."""
        if self._arrival is None:
            self._arrival = self.approach(self.approach.duration) if self.approach is not None else self.before
        return self._arrival

    def heading_at(self, t: float) -> float:
        a = self.arrival().heading
        return a + _wrap(self.heading - a) * _ease((t - self.arrived) / self.turn)

    def position_at(self, t: float) -> np.ndarray:
        a, b = np.asarray(self.arrival().position, float), np.asarray(self.position, float)
        return a + (b - a) * _ease((t - self.arrived) / self.turn)

    def finished(self) -> CharacterState:
        """How the person looks when the clip part is over (where the walk back starts)."""
        if self.last is not None:
            return self.last
        t = self.clip_end
        return CharacterState(tuple(self.position_at(t)), self.heading_at(t), self.layers(t))

    def final(self) -> CharacterState:
        """How the person looks at :attr:`end`, back home."""
        if self.left is not None:
            return self.left
        return self.retreat(self.retreat.duration) if self.retreat is not None else self.finished()

    def state(self, t: float) -> CharacterState:
        """The person at scene time ``t`` within the act (``start <= t < end``)."""
        if t < self.clip_end:
            if self.approach is not None and t < self.arrived:
                w = _ease((t - self.start) / min(WALK_BLEND, self.approach.duration))
                return _blend(self.before, self.approach(t - self.start), w)
            src = self.arrival()
            w = _ease((t - self.arrived) / self.turn)
            return CharacterState(tuple(self.position_at(t)), self.heading_at(t), _mix(src.layers, self.layers(t), w),
                                  head_yaw=src.head_yaw * (1.0 - w), head_pitch=src.head_pitch * (1.0 - w),
                                  visible=self.before.visible)
        u = t - self.clip_end
        src = self.finished()
        return _blend(src, self.retreat(u), _ease(u / min(WALK_BLEND, self.retreat.duration)))

    def where(self, t: float) -> np.ndarray:
        """Just the position at scene time ``t`` within the act."""
        if t < self.clip_end:
            if self.approach is not None and t < self.arrived:
                return np.asarray(self.approach(t - self.start).position, float)
            return self.position_at(t)
        return np.asarray(self.state(t).position, float)

    def phase(self, t: float) -> str:
        """``"walking"`` there, ``"turning"``, ``"into"`` the pose, ``"holding"``, ``"out"`` of it (back to standing),
        ``"back"`` (walking back)."""
        if t >= self.clip_end:
            return "back"
        if self.cut is not None and t >= self.cut:
            return "out"
        if t < self.arrived:
            return "walking"
        if t < self.clip_start:
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
        self.queued: tuple | None = None  # (clip, heading, position, hold) to do when the current act is over
        self.walk = walker(base)

    def __getattr__(self, name):
        if name in ("base", "walk"):
            raise AttributeError(name)
        return getattr(self.base, name)

    @property
    def can_walk(self) -> bool:
        return self.walk is not None

    def reset(self) -> None:
        self.acts, self.queued = [], None

    # -- asking for poses ---------------------------------------------------------------------------------------

    def current(self, t: float) -> Act | None:
        """The act going on at scene time ``t``, if any."""
        self._advance(t)
        return next((a for a in reversed(self.acts) if a.start <= t < a.end), None)

    def free_at(self, t: float) -> float:
        """When a clip asked for at ``t`` would start: ``t``, or when the current act is over once it is cut short
        (without walking back)."""
        act = self.current(t)
        if act is None:
            return t
        if t > act.clip_end:
            return t  # walking back: a new pose starts from where the person is
        probe = replace(act)
        if probe.cut is None:
            _cut(probe, t)
        probe.retreat = None
        return probe.end

    def perform(self, clip: AnimationClip, t: float, heading: float, position=None, hold: bool = False) -> Act | None:
        """Do ``clip`` from scene time ``t`` facing ``heading`` at ``position`` (default: where the person is); with
        ``hold``, keep holding the pose (see :meth:`hold`).

        If an act is going on, it is cut short and ``clip`` follows it (the act is returned only when it started now).
        """
        self._forget_after(t)
        act = self.current(t)
        if act is not None and t > act.clip_end:  # walking back: start the new pose from here
            act.left, act.stopped = self._evaluate(t, self.acts), t
            act = None
        if act is not None:
            self._cut(act, t)
            if t <= act.clip_end:
                act.retreat = None  # no walk back: the next pose starts from where this one ends
            self.queued = (clip, heading, position, hold)
            return None
        return self._start(clip, t, heading, position, hold)

    def hold(self, t: float) -> bool:
        """Keep holding the current pose (until :meth:`stop` or another pose), or the queued one; ``False`` if the
        current pose is already leaving and none is queued."""
        act = self.current(t)
        if self.queued is not None:
            self.queued = (*self.queued[:3], True)
            return True
        if act is None or act.cut is not None or not act.hold or act.phase(t) in ("out", "back"):
            return False
        act.looping = True
        return True

    def peek(self, t: float) -> CharacterState:
        """The state at scene time ``t`` without starting a queued clip (looking ahead changes nothing)."""
        return self._evaluate(t, self._planned(t))

    def stop(self, t: float) -> bool:
        """Leave the current pose now (and forget a pose waiting to follow it); ``False`` if there is none."""
        stopped = self._forget_after(t)
        act = self.current(t)
        if act is not None and act.cut is None and t < act.clip_end:
            self._cut(act, t)
            stopped = True
        if act is not None and act.retreat is None and self.walk is not None and t <= act.clip_end:
            self._walk_back(act)  # a pose that was to follow was dropped: walk back after all
        return stopped

    def _forget_after(self, t: float) -> bool:
        """Drop the queued clip, also if it was already started after ``t`` (by evaluating a later time)."""
        later = [a for a in self.acts if a.start > t]
        self.acts = [a for a in self.acts if a.start <= t]
        dropped = bool(later) or self.queued is not None
        self.queued = None
        return dropped

    def _cut(self, act: Act, t: float) -> None:
        if act.cut is not None or t >= act.clip_end:
            return
        if t < act.clip_start:  # walking or turning: leave from how the person looks right now
            act.last = self._evaluate(t, self.acts)
            _cut(act, t)
            if act.retreat is not None:
                self._walk_back(act)
        else:
            _cut(act, t)

    def _walk_back(self, act: Act) -> None:
        src = act.finished()
        far = math.hypot(src.position[0] - act.home[0], src.position[2] - act.home[2])
        act.retreat = self.walk(src.position, src.heading, act.home) if far > WALK_FROM else None

    def _start(self, clip, t, heading, position, hold=False) -> Act:
        act = self._new_act(clip, t, heading, position, self(t), self.acts)
        act.looping = bool(hold and act.hold)
        self.acts.append(act)
        return act

    def _new_act(self, clip, t, heading, position, before: CharacterState, acts) -> Act:
        """The act of doing ``clip`` from ``t`` after ``acts`` for someone looking like ``before``."""
        home = tuple(self._base(t, acts).position)
        position = tuple(before.position) if position is None else tuple(position)
        moved = float(np.hypot(*(np.asarray(position, float) - np.asarray(before.position, float))[[0, 2]]))
        hold = clip.metadata.get("hold")
        hold = tuple(hold) if hold else None
        if moved > WALK_FROM and self.walk is not None:
            approach = self.walk(before.position, before.heading, position)
            arrival = approach(approach.duration)
            act = Act(clip, t, position, float(heading), before, _turn_time(_wrap(heading - arrival.heading)), hold,
                      home=home, approach=approach)
        else:
            turn = max(_turn_time(_wrap(heading - before.heading)), moved / STEP_SPEED)
            act = Act(clip, t, position, float(heading), before, turn, hold, home=home)
        far = math.hypot(position[0] - home[0], position[2] - home[2])
        if far > WALK_FROM and self.walk is not None:
            act.retreat = self.walk(position, float(heading), home)
        return act

    def _advance(self, t: float) -> None:
        """Start the queued clip when the act before it is over (at that moment, whenever this is evaluated)."""
        while self.queued is not None and self.acts and self.acts[-1].end <= t:
            clip, heading, position, hold = self.queued
            self.queued = None
            self._start(clip, self.acts[-1].end, heading, position, hold)

    def _planned(self, t: float) -> list[Act]:
        """The acts, with the queued one if it starts by ``t`` (as :meth:`_advance` would start it, but on a copy)."""
        if self.queued is None or not self.acts or self.acts[-1].end > t:
            return self.acts
        clip, heading, position, hold = self.queued
        last = self.acts[-1]
        act = self._new_act(clip, last.end, heading, position, self._evaluate(last.end, self.acts), self.acts)
        act.looping = bool(hold and act.hold)
        return [*self.acts, act]

    # -- evaluating ---------------------------------------------------------------------------------------------

    def _base_time(self, t: float, acts=None) -> float:
        """The base behaviour's own time: scene time minus the time spent doing yoga."""
        acts = self.acts if acts is None else acts
        return t - sum(max(0.0, min(t, act.end) - act.start) for act in acts if act.start < t)

    def _base(self, t: float, acts=None) -> CharacterState:
        if self.base is None:
            return CharacterState((0.0, 0.0, 0.0), 0.0, [])
        return self.base(self._base_time(t, acts))

    def __call__(self, t: float) -> CharacterState:
        self._advance(t)
        return self._evaluate(t, self.acts)

    def _evaluate(self, t: float, acts) -> CharacterState:
        act = next((a for a in reversed(acts) if a.start <= t), None)
        if act is not None and t < act.end:
            return act.state(t)
        state = self._base(t, acts)
        if act is None:
            return state
        src, start, turn, back = self._take_over(act, acts)
        if t - act.end >= back:
            return state
        w = _ease((t - act.end) / back)  # back into the behaviour: the turn is chosen once, then follows it
        heading = src.heading + (turn + _wrap(state.heading - start.heading)) * w
        return replace(_blend(src, state, w), heading=heading)

    def _take_over(self, act: Act, acts) -> tuple:
        """``(final state, the behaviour's state then, degrees to turn, seconds)`` of the blend back after ``act``:
        long enough for the turn and for a step back at ``STEP_SPEED``."""
        if act._back is None or act._back[0] != act.end:
            src, start = act.final(), self._base(act.end, acts)
            turn = _wrap(start.heading - src.heading)
            step = math.hypot(start.position[0] - src.position[0], start.position[2] - src.position[2])
            act._back = (act.end, src, start, turn, max(_turn_time(turn), step / STEP_SPEED))
        return act._back[1:]

    def positions(self, times) -> np.ndarray:
        """``(x, z)`` at each of ``times`` (nan while hidden), like ``Behaviour.positions`` (changes nothing)."""
        times = np.asarray(times, float)
        acts = self._planned(float(times.max())) if len(times) else self.acts
        out = np.full((len(times), 2), np.nan)
        rest = []
        for k, t in enumerate(times):
            act = next((a for a in reversed(acts) if a.start <= t), None)
            if act is not None and t < act.end:
                out[k] = act.where(t)[[0, 2]]
            elif act is not None and t < act.end + self._take_over(act, acts)[3]:
                out[k] = np.asarray(self._evaluate(float(t), acts).position, float)[[0, 2]]  # still blending back
            else:
                rest.append(k)
        if rest:
            base_times = [self._base_time(float(times[k]), acts) for k in rest]
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
    if t < act.clip_start or not act.hold:
        act.mode = "back" if t < act.clip_start else "finish"
        if t < act.clip_start:
            act.cut_at = 0.0
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
    """``(times, points)``: clip times every ``step`` seconds and the ``(T, P, 3)`` positions (character space) of
    points along the trunk, arms and legs of ``character`` doing ``clip`` then (joints and the thirds between them,
    so no thin counter fits between two of them)."""
    skeleton = character.skinned_model.skeleton
    chains = [[b for b in (skeleton.find(n) for n in chain) if b is not None] for chain in FOOTPRINT_CHAINS]
    pairs = [pair for chain in chains for pair in itertools.pairwise(chain)]
    if not pairs:
        pairs = [(b, b) for b in range(len(skeleton))]
    first, second = np.array(pairs).T
    times = np.append(np.arange(0.0, clip.duration, step), clip.duration)
    scale = np.asarray(getattr(character, "current_scale", (1.0, 1.0, 1.0)), float)
    points = []
    for t in times:
        joints = character.poser.global_pose([Layer(clip, float(t))])[:, :3, 3] * scale
        a, b = joints[first], joints[second]
        points.append(np.concatenate([a, a + (b - a) / 3, a + 2 * (b - a) / 3, b]))
    return times, np.array(points)


def _boxes(scene):
    """``(object, low corner, high corner)`` world bounding boxes of the scene's solid things: no people, no
    pictures."""
    from ..scene.scene import Quad

    for obj in scene:
        if hasattr(obj, "skinned_model") or isinstance(obj, Quad) or not obj.current_visible:
            continue
        try:
            low, high = obj.extents()
        except Exception:  # noqa: BLE001 - an object without a usable mesh takes no room
            continue
        centre = obj.current_position
        yield obj, centre + low, centre + high


def obstacles(scene, floor: float) -> np.ndarray:
    """``(N, 6)`` boxes ``(x0, z0, x1, z1, bottom, top)`` of the furniture: things that reach into knee to chest
    height above ``floor`` (counters, walls, plants)."""
    from ..scene.scene import Plane

    rects = [(lo[0], lo[2], hi[0], hi[2], lo[1], hi[1]) for obj, lo, hi in _boxes(scene)
             if not isinstance(obj, Plane) and hi[1] >= floor + 0.1 and lo[1] <= floor + 1.2]
    return np.array(rects, float).reshape(-1, 6)


def floors(scene) -> np.ndarray:
    """``(N, 5)`` rectangles ``(x0, z0, x1, z1, height)`` of what people can stand on: floor planes and flat boxes
    (booth platforms)."""
    from ..scene.scene import Plane

    rects = [(lo[0], lo[2], hi[0], hi[2], hi[1]) for obj, lo, hi in _boxes(scene)
             if isinstance(obj, Plane) or (hi[1] - lo[1] <= 0.35 and hi[1] <= 0.35)]
    return np.array(rects, float).reshape(-1, 5)


def floor_height(rects: np.ndarray, xz: np.ndarray) -> np.ndarray:
    """The floor height under each ``(..., 2)`` point: the highest of ``rects`` (see :func:`floors`) there, else 0."""
    xz = np.asarray(xz, float)
    if not len(rects):
        return np.zeros(xz.shape[:-1])
    x, z = xz[..., None, 0], xz[..., None, 1]
    on = (x >= rects[:, 0]) & (x <= rects[:, 2]) & (z >= rects[:, 1]) & (z <= rects[:, 3])
    return np.where(on, rects[:, 4], -np.inf).max(axis=-1).clip(min=0.0)


def _rect_distance(points: np.ndarray, rects: np.ndarray) -> np.ndarray:
    """Distance of each ``(..., 2)`` point to each rectangle ``(x0, z0, x1, z1, ...)``: ``(..., N)``."""
    p = points[..., None, :]
    dx = np.maximum(np.maximum(rects[:, 0] - p[..., 0], 0.0), p[..., 0] - rects[:, 2])
    dz = np.maximum(np.maximum(rects[:, 1] - p[..., 1], 0.0), p[..., 1] - rects[:, 3])
    return np.hypot(dx, dz)


def place(points: np.ndarray, heading: float, spot) -> np.ndarray:
    """``points`` (character space, ``(..., 3)``) of a person at ``spot`` facing ``heading``, in the scene."""
    c, s = math.cos(math.radians(heading)), math.sin(math.radians(heading))
    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    return np.stack([spot[0] + c * x + s * z, spot[1] + y, spot[2] - s * x + c * z], axis=-1)


def choose_place(position, views, points: np.ndarray, others: np.ndarray, furniture: np.ndarray, *,
                 bodies: np.ndarray | None = None, floor_rects: np.ndarray | None = None, reach: float = 0.6,
                 seen=None, people_gap: float = 0.5, body_gap: float = 0.25,
                 furniture_gap: float = 0.15) -> tuple[float, tuple]:
    """``(heading, position)`` for a pose done by someone standing at ``position``.

    The candidates are ``views`` (preferred headings, best first) turned by up to 45 degrees, at ``position`` or up
    to ``reach`` metres away; the nearest, least turned one is taken whose mat is clear (every point of ``points``,
    the pose's ``(T, P, 3)`` :func:`footprint`, at least ``people_gap`` from ``others``, the ``(T, O, 2)`` positions of
    the other people at those moments (nan = nowhere), ``body_gap`` from ``bodies``, ``(T, Q, 2)`` points of the
    bodies of people doing poses then, and ``furniture_gap`` from the :func:`obstacles` ``furniture``), lies on the
    person's floor level (``floor_rects``, see :func:`floors`) as does the way there,
    and that ``seen`` rates at least 0.97 (``seen(world_points)``: ``(..., T, P, 3)`` scene points of poses to
    ``(...)`` shares in the picture and not hidden). Otherwise the candidate with the best ``seen`` rating among the
    clear ones, else the one with the most room.
    """
    position = np.asarray(position, float)
    points = np.asarray(points, float)
    outline = np.unique(np.round(points.reshape(-1, 3) / 0.05), axis=0) * 0.05  # every frame at once, for the
    span = float(np.hypot(outline[:, 0], outline[:, 2]).max()) + reach + 1.0  # furniture, which does not move
    if len(furniture):
        furniture = furniture[_rect_distance(position[None, [0, 2]], furniture)[0] < span + furniture_gap]
    others = np.asarray(others, float).reshape(len(points), -1, 2)
    if bodies is not None and np.size(bodies):  # body points count like people, with a smaller gap
        bodies = np.asarray(bodies, float).reshape(len(points), -1, 2)
        others = np.concatenate([others, bodies], axis=1)
        gaps = np.r_[np.full(others.shape[1] - bodies.shape[1], people_gap), np.full(bodies.shape[1], body_gap)]
    else:
        gaps = np.full(others.shape[1], people_gap)
    turns = (0.0, 15.0, -15.0, 30.0, -30.0, 45.0, -45.0)
    headings = np.array([float(view) + turn for turn in turns for view in views])
    penalty = np.array([0.6 * abs(turn) / 45.0 + 0.2 * k for turn in turns for k, _view in enumerate(views)])
    turned_outline = _turned(outline, headings)  # (H, U, 3)
    turned_points = _turned(points, headings)  # (H, T, P, 3)
    every = max(1, len(points) // 40)  # the people, who move, are checked at most 40 times, at the joints only
    joints = _turned(points[::every, ::max(1, points.shape[1] // 16)], headings)[..., [0, 2]]
    moments, moment_gaps = others[::every], gaps
    reach_out = float(np.hypot(outline[:, 0], outline[:, 2]).max()) + people_gap
    rings = [r for r in (0.3, 0.6, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0) if r <= reach + 1e-9]
    spots = [(0.0, 0.0)] + [(r * math.sin(a), r * math.cos(a)) for r in rings
                            for a in np.radians(np.arange(0.0, 360.0, 45.0 if r < 1.0 else 30.0))]
    level = float(floor_height(floor_rects, position[[0, 2]])) if floor_rects is not None else None
    found, found_cost = None, math.inf
    best, best_seen, roomiest, most_room = None, -1.0, None, -math.inf
    for dx, dz in spots:
        distance = math.hypot(dx, dz)
        if distance > found_cost:
            break  # nothing farther can be better
        spot = position + (dx, 0.0, dz)
        flat_outline = turned_outline[..., [0, 2]] + spot[[0, 2]]
        room = np.full(len(headings), math.inf)
        if len(furniture):
            room = _rect_distance(flat_outline, furniture).min(axis=(1, 2)) - furniture_gap
        if moments.shape[1]:
            near = np.nan_to_num(np.linalg.norm(moments - spot[[0, 2]], axis=-1), nan=np.inf).min(axis=0) < reach_out
            if near.any():
                flat = joints + spot[[0, 2]]  # (H, T', J, 2)
                d = np.linalg.norm(flat[:, :, :, None, :] - moments[None, :, None, near, :], axis=-1)
                d = np.nan_to_num(d, nan=np.inf) - moment_gaps[near]
                room = np.minimum(room, d.min(axis=(1, 2, 3)))
        k = int(np.argmax(room))
        if room[k] > most_room:
            roomiest, most_room = (float(headings[k]), tuple(spot)), float(room[k])
        ok = room >= 0.0
        if level is not None and ok.any():
            ok &= np.abs(floor_height(floor_rects, flat_outline) - level).max(axis=1) <= 0.03  # one floor level
        if not ok.any() or not _way_clear(position, spot, others, furniture, floor_rects, level):
            continue
        chosen = np.nonzero(ok)[0]
        ratings = np.ones(len(chosen)) if seen is None else np.asarray(seen(turned_points[chosen] + spot), float)
        for h, rating in zip(chosen, ratings):
            cost = distance + penalty[h]
            if rating >= 0.97 and cost < found_cost:
                found, found_cost = (float(headings[h]), tuple(spot)), cost
            if rating > best_seen:
                best, best_seen = (float(headings[h]), tuple(spot)), float(rating)
    if found is not None:
        return found
    if best is not None:
        return best
    return roomiest if roomiest is not None else (float(views[0]), tuple(position))


def _turned(points: np.ndarray, headings: np.ndarray) -> np.ndarray:
    """``points`` (character space, ``(..., 3)``) turned to each of ``headings``: ``(H, ..., 3)``."""
    c = np.cos(np.radians(headings)).reshape((-1,) + (1,) * (points.ndim - 1))
    s = np.sin(np.radians(headings)).reshape(c.shape)
    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    return np.stack([c * x + s * z, np.broadcast_to(y, (len(headings),) + y.shape), -s * x + c * z], axis=-1)


def _way_clear(start, goal, others, furniture, floor_rects, level) -> bool:
    """Whether the straight way from ``start`` to ``goal`` keeps clear of the furniture and of the others (where they
    are now) and stays on one floor level."""
    length = math.hypot(goal[0] - start[0], goal[2] - start[2])
    if length < 1e-6:
        return True
    u = np.linspace(0.0, 1.0, max(2, int(length / 0.2) + 1))[:, None]
    way = start[[0, 2]] + (goal[[0, 2]] - start[[0, 2]]) * u
    if len(furniture) and _rect_distance(way, furniture).min() < 0.2:
        return False
    now = others[0] if others.shape[1] else np.zeros((0, 2))
    now = now[np.isfinite(now).all(axis=1)]
    if len(now) and np.linalg.norm(way[:, None, :] - now[None, :, :], axis=-1).min() < 0.4:
        return False
    return level is None or np.abs(floor_height(floor_rects, way) - level).max() <= 0.03
