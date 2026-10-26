"""Interrupting what a person does for a yoga pose, and finding room for it.

A :class:`Performer` wraps a character's behaviour. Asked to :meth:`~Performer.perform` a clip at a place, the person
stops what they are doing, walks there if it is more than a step away (with their behaviour's own walk clips), turns to
face the right way, does the clip (from standing into the pose and back), walks back and carries on with their
behaviour exactly where they left it: the behaviour's time stands still meanwhile. A pose that is cut short
(:meth:`~Performer.stop`, or a new pose) leaves the hold early, or goes back the way it came if that is quicker; a new
pose follows at once, without walking back first. A pose asked to :meth:`~Performer.hold` keeps holding until it is
stopped. :meth:`~Performer.wait` stops the person (as below: someone moving slows to a stop) and has them stand (in
the clip's first pose) until the clip is asked for, so where they will be stays known while a place for it is found. Since the behaviour's time stood still,
a person who walks around is later on their way than before; plans that time several people's loops against each
other (``fair.keep_apart``) no longer hold for them.

Facing. Inside a turn clip the body turns with the clip's root bone while ``heading`` only takes the turn when the
clip ends (the behaviour's contract), so where someone faces is ``heading`` plus the turn made so far
(:meth:`~.behaviour.MotionLibrary.turned`; :meth:`Performer.facing`). Walks, turns and their timing start from
there, and every crossfade turns the body, not the heading, the short way: a pose asked for while someone turns
carries on from the way they face. A crossfade mixes the clips of two states under one heading, so a state late in a
turn clip is handed over early to the clip's ``"@after"`` variant (:meth:`~.behaviour.MotionLibrary.settled`; the
same pose): otherwise its root would be about opposite to the root of the clip it fades with, and the blend of the
two would flip from one way round to the other.

No standing starts. Someone who is moving when a pose (or a stop) breaks in, walking with their behaviour or to a
spot or back, does not stop dead: what they were doing carries on for a moment, slowing to a stop over ``LEAD``
seconds (its clips slow down with it, so the feet keep stepping), and what follows starts where they stop and fades
in meanwhile. Their behaviour's time ran on by those ``LEAD / 2`` seconds, so afterwards it carries on from where they
stopped (no walking back to where the pose was asked for), and :meth:`~Performer.free_at` and
:meth:`~Performer.peek` tell where that is. Likewise the behaviour starts again from a standstill after an act,
speeding up over ``LEAD`` seconds (so it loses ``LEAD / 2`` seconds). The crossfades into and out of the walks and
back into the behaviour let these move at their own pace and only close the gap from where the person was.

The walks are behaviours of their own, with the base's camera and ``min_angle``: on the way to a spot and back the
face keeps off the camera as everyone's does, once under way (a walk back fades in from the pose's last frame over
``WALK_BLEND``). The rest of an act need not: poses seen best from the front face the camera on purpose, and in the
turns into the pose and back into the behaviour the head turns with the body.

:func:`choose_place` picks where and facing which way: a spot as near as possible from which the camera sees the
whole pose from the side it is best seen from, with the mat clear of the other people (every point of the body, hands
and feet too) and of the furniture and on one floor level (not half on a booth platform: :func:`floors` are the
floor, carpets and platforms, not the ceiling or the walls), and with clear ways there and back home.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable
from dataclasses import dataclass, field, replace

import numpy as np

from .model import AnimationClip, CharacterState, Layer

TURN_SPEED = 150.0  # degrees per second when turning on the spot before and after a pose
TURN_MIN = 0.6  # seconds: the shortest turn, also the blend into the clip and back into the behaviour
SKIP_BLEND = 0.6  # seconds of crossfade when a hold is cut short
WALK_BLEND = 0.5  # seconds of crossfade into and out of a walk
LEAD = 0.5  # seconds over which someone moving slows to a stop when a pose breaks in, and a behaviour starts again
WALK_FROM = 0.3  # metres: farther than this the person walks there (when the behaviour has walk clips)
STEP_SPEED = 0.5  # m/s of a shorter step aside (a slide during the turn)
STEP_UP = 0.35  # metres: flat things up to this high count as floor (booth platforms)
FIELD_CELL = 0.05  # metres between the points of the grids on which the distance to a body is looked up
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


def _slow(u: float) -> float:
    """Seconds something slowing to a stop over ``LEAD`` (``1 - ease``) gets on in ``u`` seconds: ``LEAD / 2`` in
    all. Also what something starting from a standstill over ``LEAD`` (``ease``) lags behind after ``u`` seconds."""
    x = min(max(u / LEAD, 0.0), 1.0)
    return LEAD * (x - x ** 3 + x ** 4 / 2.0)


def _straight(layers) -> float:
    """Without a motion library there are no turn clips: the body faces the heading."""
    return 0.0


def _as_is(state: CharacterState, toward: float = 0.0) -> CharacterState:
    """Without a motion library there are no turn clips to hand over early (see
    :meth:`~.behaviour.MotionLibrary.settled`)."""
    return state


def _facing(state: CharacterState, turned=_straight) -> float:
    """Where the body faces: the heading plus the turn made so far inside a playing turn clip (``turned``, a
    :meth:`~.behaviour.MotionLibrary.turned`)."""
    return state.heading + turned(state.layers)


def _mix(before: list, after: list, w: float) -> list:
    """Layers fading from ``before`` (weight 1 - w) to ``after`` (weight w)."""
    if w >= 1.0:
        return list(after)
    if w <= 0.0:
        return list(before)
    return ([replace(layer, weight=layer.weight * (1.0 - w)) for layer in before]
            + [replace(layer, weight=layer.weight * w) for layer in after])


def _blend(a: CharacterState, b: CharacterState, w: float, turned=_straight, ref=None) -> CharacterState:
    """``a`` fading into ``b``: layers crossfaded, the body (see :func:`_facing`) turned the short way, position and
    head in between.

    With ``ref``, how ``b`` looked when the fade began, ``b`` moves and turns at its own pace and only the gap from
    ``a`` to ``ref`` closes; without, the person heads for wherever ``b`` has got to (which makes them surge when
    ``b`` moves on)."""
    layers = _mix(a.layers, b.layers, w)
    fa, fb = _facing(a, turned), _facing(b, turned)
    pa, pb = np.asarray(a.position, float), np.asarray(b.position, float)
    if ref is None:
        position, facing = pa + (pb - pa) * w, fa + _wrap(fb - fa) * w
    else:
        position = pb + (pa - np.asarray(ref.position, float)) * (1.0 - w)
        facing = fb + _wrap(fa - _facing(ref, turned)) * (1.0 - w)
    return CharacterState(tuple(position), facing - turned(layers), layers,
                          head_yaw=a.head_yaw + (b.head_yaw - a.head_yaw) * w,
                          head_pitch=a.head_pitch + (b.head_pitch - a.head_pitch) * w, visible=b.visible)


def _carry(a: CharacterState, b: CharacterState, w: float, ref: CharacterState, turned=_straight) -> CharacterState:
    """``a`` slowing to a stop at ``ref`` while ``b`` sets off from there: ``b``'s motion since ``ref`` adds to
    ``a``'s (so neither stops dead nor surges), and the layers and the head fade from ``a`` to ``b``."""
    layers = _mix(a.layers, b.layers, w)
    position = np.asarray(a.position, float) + np.asarray(b.position, float) - np.asarray(ref.position, float)
    facing = _facing(a, turned) + _wrap(_facing(b, turned) - _facing(ref, turned))
    return CharacterState(tuple(position), facing - turned(layers), layers,
                          head_yaw=a.head_yaw + (b.head_yaw - a.head_yaw) * w,
                          head_pitch=a.head_pitch + (b.head_pitch - a.head_pitch) * w, visible=b.visible)


class Leg:
    """A short walk from ``start`` to ``goal`` (``(x, y, z)``; the floor stays at the start's height), played by a
    one-walk :class:`~.behaviour.Behaviour` with ``library``'s clips that keeps the face ``min_angle`` off
    ``camera`` (if any); ``duration`` seconds until it arrives, ``first`` its state as it sets off."""

    def __init__(self, library, style: dict, start, heading: float, goal, camera=None, min_angle=None):
        from .behaviour import MIN_ANGLE, Behaviour, Walk

        self.floor = float(start[1])
        self.goal = (float(goal[0]), self.floor, float(goal[2]))
        self.behaviour = Behaviour([Walk([(goal[0], goal[2])])], library, start=(start[0], start[2]),
                                   heading=heading, style=style, camera=camera,
                                   min_angle=MIN_ANGLE if min_angle is None else min_angle)
        self.duration = next((p.t1 for p in self.behaviour._pieces if p.kind == "walk"), 0.0)
        self.first = self(0.0)

    def __call__(self, u: float) -> CharacterState:
        state = self.behaviour(min(max(u, 0.0), self.duration))
        x, _y, z = state.position
        if u >= self.duration:
            x, z = self.goal[0], self.goal[2]
        return replace(state, position=(x, self.floor, z))


def walker(base):
    """``(start, heading, goal) -> Leg`` walking like ``base`` (a behaviour with ``library`` and ``style``) and
    keeping the face off its ``camera`` as it does (by its ``min_angle``), or ``None`` when it has no walk clips. Like
    the base, a walk takes a moving camera at its own time."""
    library, style = getattr(base, "library", None), getattr(base, "style", None)
    role = getattr(style, "walk_role", None)
    if library is None or role is None or not hasattr(library, "choices") or not library.choices(role, style.sex):
        return None
    clip = library.choices(role, style.sex)[0]
    spec = {"sex": style.sex, "age_group": style.age, "walk_role": role, "stride_scale": style.stride,
            "speed": style.walk_speed(clip)}
    camera, min_angle = getattr(base, "camera", None), getattr(base, "min_angle", None)
    return lambda start, heading, goal: Leg(library, spec, start, heading, goal, camera=camera, min_angle=min_angle)


@dataclass
class Act:
    """One clip done by a :class:`Performer`, from scene time ``start``.

    The person (as ``before``: how they look when it starts, once they have come to a stop) first walks to
    ``position`` (``approach``, if any), then turns for ``turn`` seconds to ``heading`` (or, without a walk, steps
    there meanwhile) and does ``clip``; ``hold`` is its main hold ``(t0, t1)`` in clip time (from ``clip.metadata``).
    Afterwards they walk back ``home`` (``retreat``, if any): where their behaviour carries on. ``turned`` tells how
    far turn clips turn the body (see :func:`_facing`).
    """

    clip: AnimationClip
    start: float
    position: tuple
    heading: float
    before: CharacterState
    turn: float
    hold: tuple | None
    home: tuple = (0.0, 0.0, 0.0)
    ran_on: float = 0.0  # seconds (of its own time) the behaviour ran on into the stop before: it stands still after
    approach: Leg | None = None
    retreat: Leg | None = None
    looping: bool = False
    last: CharacterState | None = None  # how the person looked when cut short before the clip began (once stopped)
    stopped: float | None = None  # scene time the walk back was broken off (for a new pose)
    left: CharacterState | None = None  # how the person looked then (once stopped)
    cut: float | None = None  # scene time it was cut short
    cut_at: float = 0.0  # clip time then
    mode: str = ""  # "back" the way it came, "through" (on to the hold, then out of it), "skip" (out of the hold)
    turned: Callable = field(default=_straight, repr=False, compare=False)
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
        """The heading at scene time ``t`` from the arrival on, while the body turns the short way to ``heading``."""
        if t >= self.clip_start:
            return self.heading
        src = self.arrival()
        w = _ease((t - self.arrived) / self.turn)
        a = _facing(src, self.turned)
        return a + _wrap(self.heading - a) * w - self.turned(_mix(src.layers, self.layers(t), w))

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
                u = t - self.start
                w = _ease(u / min(WALK_BLEND, self.approach.duration))
                return _blend(self.before, self.approach(u), w, self.turned, ref=self.approach.first)
            src = self.arrival()
            w = _ease((t - self.arrived) / self.turn)
            return CharacterState(tuple(self.position_at(t)), self.heading_at(t), _mix(src.layers, self.layers(t), w),
                                  head_yaw=src.head_yaw * (1.0 - w), head_pitch=src.head_pitch * (1.0 - w),
                                  visible=self.before.visible)
        u = t - self.clip_end
        w = _ease(u / min(WALK_BLEND, self.retreat.duration))
        return _blend(self.finished(), self.retreat(u), w, self.turned, ref=self.retreat.first)

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
        library = getattr(base, "library", None)
        self.turned = library.turned if hasattr(library, "turned") else _straight
        self.settled = library.settled if hasattr(library, "settled") else _as_is
        # (scene time, acts then, the lead before, where they stop) of the last change someone slowed to a stop for
        self._lead: tuple | None = None
        self._next: tuple | None = None  # (queued, last act, how it ends, the queued act planned after it)

    def __getattr__(self, name):
        if name in ("base", "walk", "turned", "settled", "_lead", "_next"):
            raise AttributeError(name)
        return getattr(self.base, name)

    @property
    def can_walk(self) -> bool:
        return self.walk is not None

    def reset(self) -> None:
        self.acts, self.queued, self._lead, self._next = [], None, None, None

    def facing(self, t: float) -> float:
        """Where the body faces at scene time ``t``: the heading plus the turn made so far inside a turn clip."""
        return _facing(self(t), self.turned)

    # -- asking for poses ---------------------------------------------------------------------------------------

    def current(self, t: float) -> Act | None:
        """The act going on at scene time ``t``, if any."""
        self._advance(t)
        return next((a for a in reversed(self.acts) if a.start <= t < a.end), None)

    def free_at(self, t: float) -> float:
        """When a clip asked for at ``t`` would start: when the current act is over once it is cut short (without
        walking back), else ``t``; for someone moving then, ``LEAD / 2`` seconds later, when they have come to a stop
        (:meth:`_halt`): :meth:`peek` then tells where they stop."""
        act = self.current(t)
        if act is not None and t <= act.clip_end:
            probe = replace(act)
            if probe.cut is None:
                _cut(probe, t)
            probe.retreat = None
            if probe.end > t:
                return probe.end  # out of the pose, standing
        return t + LEAD / 2.0 if self._carry_on(t)[2] is not None else t

    def perform(self, clip: AnimationClip, t: float, heading: float, position=None, hold: bool = False) -> Act | None:
        """Do ``clip`` from scene time ``t`` facing ``heading`` at ``position`` (default: where the person is); with
        ``hold``, keep holding the pose (see :meth:`hold`).

        If an act is going on, it is cut short and ``clip`` follows it (the act is returned only when it started now).
        """
        self._forget_after(t)
        act = self.current(t)
        if act is not None and t > act.clip_end:  # walking back: start the new pose from here
            act.left, ran_on = self._halt(t)
            act.stopped = t
            return self._start(clip, t, heading, position, hold, act.left, ran_on)
        if act is not None:
            self._cut(act, t)
            if t <= act.clip_end:
                act.retreat = None  # no walk back: the next pose starts from where this one ends
            self.queued = (clip, heading, position, hold)
            return None
        return self._start(clip, t, heading, position, hold, *self._halt(t))

    def wait(self, clip: AnimationClip, t: float) -> None:
        """Stop what the person is doing at scene time ``t`` (out of the current pose, without walking back; someone
        moving slows to a stop over ``LEAD``, see :meth:`_halt`) and stand there in ``clip``'s first pose
        (:func:`standing`) until the next :meth:`perform` or :meth:`stop`. The behaviour's time stands still meanwhile
        (from where they stop), so where the person will be when ``clip`` starts stays known while a place is found:
        ``peek(free_at(t))`` and :meth:`home`."""
        self.perform(standing(clip), t, _facing(self.peek(self.free_at(t)), self.turned), hold=True)  # mid turn too

    def home(self, t: float) -> np.ndarray:
        """Where the person walks back to after a clip started at scene time ``t``: where their behaviour carries on."""
        return np.asarray(self._base(t, self._planned(t)).position, float)

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
        return self._show(t, self._planned(t), self._lead)

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
        if t < act.clip_start:  # walking or turning: leave from where the person comes to a stop
            act.last = self._halt(t)[0]
            _cut(act, t)
            if act.retreat is not None:
                self._walk_back(act)
        else:
            _cut(act, t)

    def _carry_on(self, t: float) -> tuple:
        """``(the acts now, the lead now, where the person comes to a stop)`` for a change at ``t``: the last is
        ``None`` for someone standing still (or about to vanish), who stops at once (see :meth:`_halt`)."""
        plan = [replace(act) for act in self.acts]
        lead = self._lead if self._lead is not None and t < self._lead[0] + LEAD else None
        now, later = self._show(t, plan, lead), self._show(t + LEAD / 2.0, plan, lead)
        gone = math.hypot(later.position[0] - now.position[0], later.position[2] - now.position[2])
        turned = abs(_wrap(_facing(later, self.turned) - _facing(now, self.turned)))
        still = gone <= 0.01 and turned <= 2.0
        jumps = gone > 2.5 * LEAD / 2.0  # faster than 2.5 m/s: a teleport
        return plan, lead, None if still or jumps or not (now.visible and later.visible) else later

    def _halt(self, t: float) -> tuple[CharacterState, float]:
        """How the person looks where what follows a change at ``t`` starts (call it before changing the acts), and
        how far (in its own time) the behaviour ran on meanwhile.

        Someone who is moving (or turning) carries on for a moment with what they were doing, slowing to a stop over
        ``LEAD`` seconds (its time runs on by ``LEAD / 2``): that state is returned, and :meth:`_show` fades from the
        slowing old plan into what follows meanwhile. Someone standing still (or about to vanish) stops at once.
        Inside a turn clip the state is :meth:`~.behaviour.MotionLibrary.settled` (what follows is mixed with it).
        """
        plan, lead, later = self._carry_on(t)
        if later is None:
            self._lead = lead
            return self.settled(self._evaluate(t, plan)), 0.0
        self._lead = (t, plan, lead, later)
        return self.settled(later), self._base_time(t + LEAD / 2.0, plan) - self._base_time(t, plan)

    def _walk_back(self, act: Act) -> None:
        src = act.finished()
        far = math.hypot(src.position[0] - act.home[0], src.position[2] - act.home[2])
        act.retreat = self.walk(src.position, _facing(src, self.turned), act.home) if far > WALK_FROM + 1e-6 else None

    def _start(self, clip, t, heading, position, hold, before: CharacterState, ran_on: float = 0.0) -> Act:
        act = self._new_act(clip, t, heading, position, before, self.acts, ran_on)
        act.looping = bool(hold and act.hold)
        self.acts.append(act)
        return act

    def _new_act(self, clip, t, heading, position, before: CharacterState, acts, ran_on: float = 0.0) -> Act:
        """The act of doing ``clip`` from ``t`` after ``acts`` for someone looking like ``before``, who came to a stop
        there after their behaviour ran on by ``ran_on`` seconds of its own time (home is there)."""
        home = tuple(self._base(t, acts, ran_on).position)
        position = tuple(before.position) if position is None else tuple(position)
        moved = float(np.hypot(*(np.asarray(position, float) - np.asarray(before.position, float))[[0, 2]]))
        hold = clip.metadata.get("hold")
        hold = tuple(hold) if hold else None
        facing = _facing(before, self.turned)  # not before.heading: that lags behind inside a turn clip
        if moved > WALK_FROM + 1e-6 and self.walk is not None:  # (a spot 0.3 m away is a step)
            approach = self.walk(before.position, facing, position)
            turn = _turn_time(_wrap(heading - _facing(approach(approach.duration), self.turned)))
            act = Act(clip, t, position, float(heading), before, turn, hold, home=home, approach=approach,
                      ran_on=ran_on, turned=self.turned)
        else:
            turn = max(_turn_time(_wrap(heading - facing)), moved / STEP_SPEED)
            act = Act(clip, t, position, float(heading), before, turn, hold, home=home, ran_on=ran_on,
                      turned=self.turned)
        far = math.hypot(position[0] - home[0], position[2] - home[2])
        if far > WALK_FROM + 1e-6 and self.walk is not None:
            act.retreat = self.walk(position, float(heading), home)
        return act

    def _advance(self, t: float) -> None:
        """Start the queued clip when the act before it is over (at that moment, whenever this is evaluated)."""
        while self.queued is not None and self.acts and self.acts[-1].end <= t:
            act = self._queued_act()
            self.queued, self._next = None, None
            self.acts.append(act)

    def _planned(self, t: float) -> list[Act]:
        """The acts, with the queued one if it starts by ``t`` (as :meth:`_advance` will start it)."""
        if self.queued is None or not self.acts or self.acts[-1].end > t:
            return self.acts
        return [*self.acts, self._queued_act()]

    def _queued_act(self) -> Act:
        """The queued clip's act, starting when the last act is over; planned once (its walks take a while to plan)
        until the queue or the last act changes."""
        last = self.acts[-1]
        after = (len(self.acts), last.end, last.cut, last.stopped)
        if self._next is not None and self._next[0] is self.queued and self._next[1] is last \
                and self._next[2] == after:
            return self._next[3]
        clip, heading, position, hold = self.queued
        act = self._new_act(clip, last.end, heading, position, self._evaluate(last.end, self.acts), self.acts)
        act.looping = bool(hold and act.hold)
        self._next = (self.queued, last, after, act)
        return act

    # -- evaluating ---------------------------------------------------------------------------------------------

    def _base_time(self, t: float, acts=None) -> float:
        """The base behaviour's own time: scene time minus the time spent doing yoga, but for the ``ran_on`` seconds it
        ran on into the stop before each act (as it slowed to a stop over ``LEAD``, :func:`_slow`: the time of the
        clips the person shows meanwhile), and minus what the behaviour lags after each act as it starts again from a
        standstill (:func:`_slow`)."""
        acts = self.acts if acts is None else acts
        lost = 0.0
        for k, act in enumerate(acts):
            if act.start >= t:
                break
            nxt = acts[k + 1].start if k + 1 < len(acts) else t  # the next act stops it again
            lost += min(t, act.end) - act.start
            if act.ran_on:
                lost -= act.ran_on * _slow(min(t, nxt) - act.start) / (LEAD / 2.0)
            if act.end < t:
                lost += _slow(min(t, nxt) - act.end)
        return t - lost

    def _base(self, t: float, acts=None, ahead: float = 0.0) -> CharacterState:
        if self.base is None:
            return CharacterState((0.0, 0.0, 0.0), 0.0, [])
        return self.base(self._base_time(t, acts) + ahead)

    def __call__(self, t: float) -> CharacterState:
        self._advance(t)
        return self._show(t, self.acts, self._lead)

    def _show(self, t: float, acts, lead) -> CharacterState:
        """The state at scene time ``t`` after ``acts``; within ``LEAD`` seconds of the change ``lead`` (see
        :meth:`_halt`) the old plan slowing to a stop carries the person on meanwhile."""
        state = self._evaluate(t, acts)
        if lead is None or not 0.0 <= t - lead[0] < LEAD:
            return state
        start, plan, before, halt = lead
        u = t - start
        old = self.settled(self._show(start + _slow(u), plan, before), self.turned(state.layers))
        return _carry(old, state, _ease(u / LEAD), halt, self.turned)

    def _evaluate(self, t: float, acts) -> CharacterState:
        act = next((a for a in reversed(acts) if a.start <= t), None)
        if act is not None and t < act.end:
            return act.state(t)
        state = self._base(t, acts)
        if act is None:
            return state
        src, start, back = self._take_over(act, acts)
        if t - act.end >= back:
            return state
        # back into the behaviour, which goes on (from a standstill) at its own pace while the gap closes
        state = self.settled(state, self.turned(src.layers))
        return _blend(src, state, _ease((t - act.end) / back), self.turned, ref=start)

    def _take_over(self, act: Act, acts) -> tuple:
        """``(final state, the behaviour's state then, seconds)`` of the blend back after ``act``: long enough for the
        body to turn (the short way) and for a step back at ``STEP_SPEED``."""
        if act._back is None or act._back[0] != act.end:
            src, start = act.final(), self._base(act.end, acts)
            turn = _wrap(_facing(start, self.turned) - _facing(src, self.turned))
            step = math.hypot(start.position[0] - src.position[0], start.position[2] - src.position[2])
            act._back = (act.end, src, start, max(_turn_time(turn), step / STEP_SPEED))
        return act._back[1:]

    def positions(self, times) -> np.ndarray:
        """``(x, z)`` at each of ``times`` (nan while hidden), like ``Behaviour.positions`` (changes nothing)."""
        times = np.asarray(times, float)
        acts = self._planned(float(times.max())) if len(times) else self.acts
        out = np.full((len(times), 2), np.nan)
        rest = []
        lead = self._lead
        for k, t in enumerate(times):
            act = next((a for a in reversed(acts) if a.start <= t), None)
            if lead is not None and 0.0 <= t - lead[0] < LEAD:
                state = self._show(float(t), acts, lead)  # slowing to a stop
                if state.visible:
                    out[k] = np.asarray(state.position, float)[[0, 2]]
            elif act is not None and t < act.end:
                out[k] = act.where(t)[[0, 2]]
            elif act is not None and t < act.end + self._take_over(act, acts)[2]:
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


def standing(clip: AnimationClip) -> AnimationClip:
    """``clip``'s first frame (standing, before the pose) held as a clip of its own: what someone waiting to do
    ``clip`` looks like (``metadata["waiting"]`` is true)."""
    return AnimationClip(f"{clip.name}@standing", clip.fps, list(clip.bone_names), clip.rotations[:1].copy(),
                         clip.translations[:1].copy(), loop=False, root=clip.root,
                         metadata={"pose": clip.metadata.get("pose", clip.name), "hold": (0.0, 1.0 / clip.fps),
                                   "waiting": True})


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
    """``(N, 5)`` rectangles ``(x0, z0, x1, z1, height)`` of what people can stand on: the floor planes, carpets and
    booth platforms, i.e. flat things at most a step (``STEP_UP``) above the ground. The ceiling and the walls are
    planes too, but no floor."""
    rects = [(lo[0], lo[2], hi[0], hi[2], hi[1]) for _obj, lo, hi in _boxes(scene)
             if hi[1] - lo[1] <= STEP_UP and hi[1] <= STEP_UP]
    return np.array(rects, float).reshape(-1, 5)


def floor_height(rects: np.ndarray, xz: np.ndarray) -> np.ndarray:
    """The floor height under each ``(..., 2)`` point: the highest of ``rects`` (see :func:`floors`) there, else 0."""
    xz = np.asarray(xz, float)
    if not len(rects):
        return np.zeros(xz.shape[:-1])
    x, z = xz[..., None, 0], xz[..., None, 1]
    on = (x >= rects[:, 0]) & (x <= rects[:, 2]) & (z >= rects[:, 1]) & (z <= rects[:, 3])
    return np.where(on, rects[:, 4], -np.inf).max(axis=-1).clip(min=0.0)


def _on_level(rects: np.ndarray, xz: np.ndarray, level: float, tolerance: float = 0.03) -> np.ndarray:
    """Whether all the ``(..., M, 2)`` points lie on the floor ``level`` (their :func:`floor_height` within
    ``tolerance`` of it): ``(...)``. None is on something higher, and (above the ground) none off it."""
    x, z = xz[..., 0, None], xz[..., 1, None]

    def on(r):  # (..., M): on any of the rectangles r
        if not len(r):
            return np.zeros(xz.shape[:-1], bool)
        return ((x >= r[:, 0]) & (x <= r[:, 2]) & (z >= r[:, 1]) & (z <= r[:, 3])).any(axis=-1)

    ok = ~on(rects[rects[:, 4] > level + tolerance]).any(axis=-1)
    if level > tolerance:
        ok &= on(rects[rects[:, 4] >= level - tolerance]).all(axis=-1)
    return ok


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
                 seen=None, home=None, people_gap: float = 0.5, body_gap: float = 0.25, furniture_gap: float = 0.15,
                 cancelled=None) -> tuple[float, tuple] | None:
    """``(heading, position)`` for a pose done by someone standing at ``position``.

    The candidates are ``views`` (preferred headings, best first) turned by up to 45 degrees, at ``position`` or up
    to ``reach`` metres away; the nearest, least turned one is taken whose mat is clear (every point of ``points``,
    the pose's ``(T, P, 3)`` :func:`footprint`, at least ``people_gap`` from ``others``, the ``(T, O, 2)`` positions of
    the other people at those moments (nan = nowhere), ``body_gap`` from ``bodies``, ``(T, Q, 2)`` points of the
    bodies of people doing poses then, and ``furniture_gap`` from the :func:`obstacles` ``furniture``), lies on the
    person's floor level (``floor_rects``, see :func:`floors`) as do the way there and the way back to ``home`` (where
    the person carries on afterwards; default ``position``), and that ``seen`` rates at least 0.97
    (``seen(world_points)``: ``(..., T, P, 3)`` scene points of poses to ``(...)`` shares in the picture and not
    hidden). Otherwise the candidate with the best ``seen`` rating among the clear ones, else the one with the most
    room (on the person's floor level, if any is). ``None`` if ``cancelled()`` turns true meanwhile (it is asked before
    each spot).
    """
    position = np.asarray(position, float)
    home = position if home is None else np.asarray(home, float)
    points = np.asarray(points, float)
    outline = np.unique(np.round(points.reshape(-1, 3) / 0.05), axis=0) * 0.05  # every frame at once, for the
    span = float(np.hypot(outline[:, 0], outline[:, 2]).max()) + reach + 1.0  # furniture, which does not move
    if len(furniture):
        furniture = furniture[_rect_distance(position[None, [0, 2]], furniture)[0] < span + furniture_gap]
    if floor_rects is not None and len(floor_rects):
        floor_rects = floor_rects[_rect_distance(position[None, [0, 2]], floor_rects)[0] < span]
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
    flat_points = turned_points[..., [0, 2]].reshape(len(headings), -1, 2)  # every point, for the floor level
    # the people, who move, are checked at most 40 times, against every point of the body (hands and feet too): by
    # their distance to it in each frame, looked up on a grid around the body (turned and moved with the candidates)
    every = max(1, len(points) // 40)
    moments, margin = others[::every], (float(gaps.max()) if len(gaps) else 0.0)
    field, cos, sin = None, np.cos(np.radians(headings))[:, None], np.sin(np.radians(headings))[:, None]
    reach_out = float(np.hypot(outline[:, 0], outline[:, 2]).max()) + margin
    rings = [r for r in (0.3, 0.6, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0) if r <= reach + 1e-9]
    spots = [(0.0, 0.0)] + [(r * math.sin(a), r * math.cos(a)) for r in rings
                            for a in np.radians(np.arange(0.0, 360.0, 45.0 if r < 1.0 else 30.0))]
    level = float(floor_height(floor_rects, position[[0, 2]])) if floor_rects is not None else None
    found, found_cost = None, math.inf
    best, best_seen, roomiest, most_room = None, -1.0, None, (False, -math.inf)
    for dx, dz in spots:
        if cancelled is not None and cancelled():
            return None
        distance = math.hypot(dx, dz)
        if distance > found_cost:
            break  # nothing farther can be better
        spot = position + (dx, 0.0, dz)
        flat_outline = turned_outline[..., [0, 2]] + spot[[0, 2]]
        room = np.full(len(headings), math.inf)
        if len(furniture):
            room = _rect_distance(flat_outline, furniture).min(axis=(1, 2)) - furniture_gap
        with np.errstate(invalid="ignore"):  # nan: nobody there
            when, who = np.nonzero(np.linalg.norm(moments - spot[[0, 2]], axis=-1) < reach_out)  # who is near when
        if len(when):
            if field is None:
                field, corner = _distance_field(points[::every][..., [0, 2]], margin)
            v = moments[when, who] - spot[[0, 2]]  # (K, 2): where they are, in the person's own frame for each heading
            x, z = cos * v[:, 0] - sin * v[:, 1], sin * v[:, 0] + cos * v[:, 1]
            d = _look_up(field, corner, np.broadcast_to(when, x.shape), x, z, margin)  # (H, K)
            room = np.minimum(room, (d - gaps[who]).min(axis=1))
        level_ok = np.ones(len(headings), bool)  # all of it on one floor level
        if level is not None:
            level_ok = _on_level(floor_rects, flat_points + spot[[0, 2]], level)
        k = int(np.argmax(np.where(level_ok, room, -math.inf) if level_ok.any() else room))
        if (bool(level_ok[k]), float(room[k])) > most_room:
            roomiest, most_room = (float(headings[k]), tuple(spot)), (bool(level_ok[k]), float(room[k]))
        ok = (room >= 0.0) & level_ok
        if (not ok.any() or not _way_clear(position, spot, others[0], furniture, floor_rects, level)
                or not _way_clear(spot, home, others[-1], furniture, floor_rects, level)):
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


def _distance_field(body: np.ndarray, margin: float) -> tuple[np.ndarray, np.ndarray]:
    """``(field, corner)``: for each frame of ``body`` (``(T, P, 2)`` floor points) the distance from each point of a
    grid (``FIELD_CELL`` apart from ``corner``, reaching ``margin`` beyond the body) to the nearest body point:
    ``(T, X, Z)``."""
    flat = body.reshape(-1, 2)
    corner = flat.min(axis=0) - margin
    size = np.ceil((flat.max(axis=0) + margin - corner) / FIELD_CELL).astype(int) + 1
    xs, zs = corner[0] + FIELD_CELL * np.arange(size[0]), corner[1] + FIELD_CELL * np.arange(size[1])
    field = np.empty((len(body), *size))
    for k, frame in enumerate(body):
        dx, dz = xs[:, None, None] - frame[:, 0], zs[None, :, None] - frame[:, 1]
        field[k] = np.sqrt((dx * dx + dz * dz).min(axis=-1))
    return field, corner


def _look_up(field, corner, frames, x, z, margin: float) -> np.ndarray:
    """The distance of the points ``(x, z)`` to the body in ``frames`` (see :func:`_distance_field`), rounded down
    (never more than the true distance); at least ``margin`` off the grid."""
    i = np.rint((x - corner[0]) / FIELD_CELL).astype(int)
    j = np.rint((z - corner[1]) / FIELD_CELL).astype(int)
    on = (i >= 0) & (j >= 0) & (i < field.shape[1]) & (j < field.shape[2])
    d = field[frames, np.clip(i, 0, field.shape[1] - 1), np.clip(j, 0, field.shape[2] - 1)] - FIELD_CELL * 0.7072
    return np.where(on, d, margin)


def _turned(points: np.ndarray, headings: np.ndarray) -> np.ndarray:
    """``points`` (character space, ``(..., 3)``) turned to each of ``headings``: ``(H, ..., 3)``."""
    c = np.cos(np.radians(headings)).reshape((-1,) + (1,) * (points.ndim - 1))
    s = np.sin(np.radians(headings)).reshape(c.shape)
    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    return np.stack([c * x + s * z, np.broadcast_to(y, (len(headings),) + y.shape), -s * x + c * z], axis=-1)


def _way_clear(start, goal, others, furniture, floor_rects, level) -> bool:
    """Whether the straight way from ``start`` to ``goal`` keeps clear of the furniture and of the ``(O, 2)``
    ``others`` (where they are then) and stays on one floor level."""
    length = math.hypot(goal[0] - start[0], goal[2] - start[2])
    if length < 1e-6:
        return True
    u = np.linspace(0.0, 1.0, max(2, int(length / 0.2) + 1))[:, None]
    way = start[[0, 2]] + (goal[[0, 2]] - start[[0, 2]]) * u
    if len(furniture) and _rect_distance(way, furniture).min() < 0.2:
        return False
    now = others[np.isfinite(others).all(axis=1)]
    if len(now) and np.linalg.norm(way[:, None, :] - now[None, :, :], axis=-1).min() < 0.4:
        return False
    return level is None or bool(_on_level(floor_rects, way, level))
