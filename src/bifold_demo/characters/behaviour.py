"""Behaviours for animated people: where they walk, which clips they play and where they look.

A behaviour is a callable ``t -> CharacterState`` built from a looping timeline of segments: :class:`Walk`,
:class:`Idle`, :class:`Turn` and :class:`Hide`. Everything is precomputed when it is built, so evaluating a time
is a pure, cheap function of ``t`` (pausing, rewinding and seeking just work). The factories :func:`walk_by`,
:func:`approach`, :func:`wander`, :func:`chat` and :func:`stand` build typical trade-fair behaviours from a
``style`` dict (avatar metadata: ``sex``, ``age_group``, optional ``stride_scale`` / ``speed``) and a seed.

Locomotion. Walks follow a polyline with rounded corners, parametrised by arc length; the heading follows the
tangent and blends in from the previous heading. Speed eases in and out at both ends, and the walk clip's time is
``distance / (clip.metadata["speed"] * stride_scale)``, so the feet never slide whatever the speed (``stride_scale``
is the size ratio the :class:`~bifold_demo.characters.model.Poser` applies to the clip; see :func:`stride_scale`).
Turns of 45 degrees or more play the closest turn clip and add its ``metadata["turn"]`` to the heading when it
ends (the contract of the clip loader), with any remaining angle turned procedurally during the clip; smaller turns,
or turns without clips, rotate the heading procedurally while idling. Position is continuous everywhere except
inside :class:`Hide` (an intentional teleport while invisible), and the body's facing ``heading + clip yaw`` is
continuous (``heading`` itself jumps by the clip's turn exactly when a turn clip ends, as the contract requires).
Consecutive clips crossfade over ``FADE`` seconds with two layers whose weights sum to 1.

Never looking at the camera. The *face heading* is ``heading + clip_yaw + head_yaw``, where ``clip_yaw`` is the
yaw (on the floor plane) of the head bone in the playing clips, blended with the layer weights. It is computed
from each clip's root-to-head bone chain, so it includes the body rotation inside turn clips and the head turns
of look-around or talk clips; without clips it is 0 and the face heading is ``heading + head_yaw``. Given a
camera, a behaviour guarantees for every ``t`` at which the state is visible and the camera is at least
``MIN_CAMERA_DISTANCE`` away on the floor plane:

* ``|head_yaw| <= HEAD_YAW_LIMIT`` (70 degrees), always;
* if ``look_feasible`` is true, the angle on the floor plane between the face heading and the direction to the
  camera position at ``t`` is at least ``min_angle`` (35 degrees by default). This holds wherever the camera is
  in front of, beside or behind the person, for a fixed camera and for a moving one that stays inside the range
  sampled over its first ``CAMERA_HORIZON`` seconds; it is enforced on every evaluation against the actual camera
  position, on top of a head plan made with extra margin.

``look_feasible`` is false only when no continuous head motion within the yaw limit can avoid the camera (e.g. a
path whose heading swings across the camera while it is in front). The factories choose turn directions (away
from the camera), clips (whose own head motion leaves room) and wander paths so that it is true; they retry with
other random choices and only then give up and return a best-effort plan. Head looks are smooth: looking ahead
into curves and turns, glancing at ``look_at`` points (booths) and ``partners`` (people one talks to), small
glances while idling, a lowered head when looking at a phone or walking towards the camera. The angle between
``heading + head_yaw`` and the camera direction (ignoring the clips) is kept above ``min_angle`` too when that
costs little, but that is not guaranteed (clips that turn the head may need it to point elsewhere).
"""

from __future__ import annotations

import bisect
import math
import random
from dataclasses import dataclass

import numpy as np

from .character import NECK_SHARE
from .model import AnimationClip, CharacterModel, CharacterState, Layer, quat_from_axis_angle, quat_multiply, \
    quat_to_matrix

ROLES = ("walk", "walk_slow", "stroll", "idle", "look_around", "talk", "listen", "phone",
         "turn_left_90", "turn_right_90", "turn_left_180", "turn_right_180")
FADE = 0.3  # shortest crossfade between consecutive clips, seconds
HEAD_YAW_LIMIT = 70.0
MIN_ANGLE = 35.0
MIN_CAMERA_DISTANCE = 0.75  # metres on the floor plane; closer than this the direction to the camera is undefined
CAMERA_HORIZON = 600.0  # seconds over which a moving camera's range is sampled

_FALLBACK = {"walk": ("walk_slow", "stroll"), "walk_slow": ("stroll", "walk"), "stroll": ("walk_slow", "walk"),
             "look_around": ("idle",), "talk": ("idle",), "listen": ("idle",), "phone": ("idle",)}
_EXCLUDE = ("sit_", "crouch", "_to_walk", "trolley", "umbrella", "drunk", "injured", "bruised", "run_")
_CHAIN = ("hips", "pelvis", "spine", "spine1", "spine2", "spine3", "chest", "upperchest", "neck", "neck1", "head")
_MARGIN = 5.0  # extra degrees the head plan keeps from the camera (blending and interpolation slack)
_PLAN_DT = 0.1  # head plan time step, seconds
_BIN = 2.5  # head plan yaw resolution, degrees
_HEAD_SPEED = 170.0  # fastest head turn the plan may use, degrees per second
_FADE_SPEED = 200.0  # crossfades last long enough that the clips' head yaw difference moves at most this fast
_MAX_FADE = 1.0
_WALK_TURN = 20.0  # walks starting more than this off the current heading turn in place first
_SHARP = 120.0  # corners sharper than this (degrees of deflection) stop and turn in place
_PACE = {"child": (1.0, 1.25), "adult": (0.92, 1.1), "older": (0.85, 1.0)}  # walking speed / the clip's own speed
_STRIDE = {"child": 0.72, "adult": 1.0, "older": 0.97}  # default stride_scale when the style gives none


def _wrap(a: float) -> float:
    """Degrees into [-180, 180)."""
    return (a + 180.0) % 360.0 - 180.0


def _smooth(u: float) -> float:
    u = 0.0 if u < 0.0 else 1.0 if u > 1.0 else u
    return u * u * (3.0 - 2.0 * u)


def _xz(p) -> tuple[float, float]:
    """A floor point from ``(x, z)`` or ``(x, y, z)``."""
    p = [float(v) for v in np.asarray(p, dtype=np.float64).ravel()]
    if len(p) == 2:
        return p[0], p[1]
    if len(p) == 3:
        return p[0], p[2]
    raise ValueError(f"expected an (x, z) or (x, y, z) point, got {p}")


def _bearing(fx: float, fz: float, tx: float, tz: float) -> float:
    """Heading (degrees) from ``(fx, fz)`` towards ``(tx, tz)``."""
    return math.degrees(math.atan2(tx - fx, tz - fz))


def _as_random(rng) -> random.Random:
    if isinstance(rng, random.Random):
        return rng
    if rng is None or isinstance(rng, (int, np.integer)):
        return random.Random(0 if rng is None else int(rng))
    if hasattr(rng, "integers"):
        return random.Random(int(rng.integers(2 ** 62)))
    raise TypeError(f"rng must be an int seed, random.Random or numpy Generator, not {type(rng).__name__}")


def _choice(rng, items):
    items = list(items)
    if rng is None:
        return items[0]
    if hasattr(rng, "integers"):
        return items[int(rng.integers(len(items)))]
    return items[rng.randrange(len(items))]


def pitch_shift(clip_yaw, elevation, head_yaw, head_pitch, vector: bool = False):
    """Degrees by which ``head_pitch`` moves the face sideways (on the floor plane) as :class:`Character` applies it.

    Character turns the neck by ``NECK_SHARE`` of ``(head_yaw, head_pitch)`` and then the head by the rest, each as
    a pitch about the body's right axis followed by a yaw about +Y, so when the clip already turns the head
    (``clip_yaw``, ``elevation`` = degrees below the horizontal) the pitch also changes the face's yaw a little.
    The face heading is ``heading + clip_yaw + head_yaw + pitch_shift(...)``. ``vector``: numpy arrays in and out.
    """
    if vector:
        sin, cos, atan2 = np.sin, np.cos, np.arctan2
    else:
        if not head_pitch:
            return 0.0
        sin, cos, atan2 = math.sin, math.cos, math.atan2
    r = math.pi / 180.0
    ce = cos(elevation * r)
    x, y, z = sin(clip_yaw * r) * ce, -sin(elevation * r), cos(clip_yaw * r) * ce
    for share in (NECK_SHARE, 1.0 - NECK_SHARE):
        c, s = cos(head_pitch * share * r), sin(head_pitch * share * r)
        y, z = y * c - z * s, y * s + z * c
        c, s = cos(head_yaw * share * r), sin(head_yaw * share * r)
        x, z = x * c + z * s, z * c - x * s
    shift = atan2(x, z) / r - clip_yaw - head_yaw
    return (shift + 180.0) % 360.0 - 180.0


# -- motion library ---------------------------------------------------------------------------------


def clip_sex(name: str) -> str | None:
    """``"female"`` / ``"male"`` from an ``f_`` / ``m_`` clip-name prefix, else ``None``."""
    lowered = name.lower()
    return "female" if lowered.startswith("f_") else "male" if lowered.startswith("m_") else None


def classify_clip(name: str) -> str | None:
    """The role (one of ``ROLES``) a clip name suggests, e.g. ``m_walk_neutral_01`` -> ``"walk"``, else ``None``."""
    base = name.lower().split(".")[0]
    if clip_sex(base):
        base = base[2:]
    if any(token in base for token in _EXCLUDE):
        return None
    for direction in ("left", "right"):
        for angle in ("90", "180"):
            if base == f"turn_{direction}_{angle}":
                return base
    if base.startswith("walk"):
        if "slow" in base:
            return "walk_slow"
        if "stroll" in base:
            return "stroll"
        if base == "walk" or "neutral" in base or "normal" in base:
            return "walk"
        return None
    if "look_around" in base:
        return "look_around"
    if "phone" in base:
        return "phone"
    calm = base in ("talk", "listen", "idle") or "neutral" in base or "relaxed" in base
    if "talk" in base and calm:
        return "talk"
    if "listen" in base and calm:
        return "listen"
    if base.startswith("idle") and (calm or "breathe" in base):
        return "idle"
    return None


def _chain(clip: AnimationClip) -> list[int]:
    """Clip bone indices from the whole-body root down to the head (empty without a head bone)."""
    lowered = [n.lower() for n in clip.bone_names]

    def find(token):
        for k, name in enumerate(lowered):
            if name == token or name.endswith((" " + token, ":" + token, "_" + token)):
                return k
        return None

    if find("head") is None:
        return []
    chain = [k for k in (find(token) for token in _CHAIN) if k is not None]
    root = clip.bone_names.index(clip.root) if clip.root in clip.bone_names else None
    if root is not None and root not in chain:
        chain.insert(0, root)
    return chain


def _chain_rotation(clip: AnimationClip, chain: list[int], frames=slice(None)) -> np.ndarray:
    matrices = quat_to_matrix(np.asarray(clip.rotations, dtype=np.float64)[frames][:, chain])
    out = matrices[:, 0]
    for k in range(1, len(chain)):
        out = out @ matrices[:, k]
    return out


def stride_scale(model: CharacterModel, clip: AnimationClip) -> float:
    """The factor the :class:`Poser` scales ``clip`` by for ``model`` (pass it as ``style["stride_scale"]``)."""
    root = model.skeleton.find(clip.root) if clip.root else None
    if root is None or clip.root not in clip.bone_names:
        return 1.0
    clip_height = float(np.mean(np.asarray(clip.translations)[:, clip.bone_names.index(clip.root), 1]))
    model_height = float(model.skeleton.rest_global[root, 1, 3])
    return model_height / clip_height if clip_height > 1e-3 and model_height > 1e-3 else 1.0


class MotionLibrary:
    """Animation clips by role (see ``ROLES``).

    Args:
        clips: ``{name: AnimationClip}``; roles are guessed from the names (:func:`classify_clip`), more can be
            added with :meth:`register`. Names starting with ``f_`` / ``m_`` are female / male variants.
        head_forward: the head bone's forward axis in its own frame; estimated from the clips' first frames
            (where every clip faces +Z) when omitted.
    """

    def __init__(self, clips: dict[str, AnimationClip], head_forward=None):
        self.clips = dict(clips)
        self._roles: dict[str, list[str]] = {role: [] for role in ROLES}
        for name in sorted(self.clips):
            role = classify_clip(name)
            if role is not None:
                self._roles[role].append(name)
        self.head_forward = self._estimate_head_forward() if head_forward is None else \
            np.asarray(head_forward, dtype=np.float64) / np.linalg.norm(head_forward)
        self._yaw: dict[int, tuple[AnimationClip, tuple[list[float], list[float]]]] = {}
        self._after: dict[int, tuple[AnimationClip, AnimationClip]] = {}

    def register(self, role: str, clip_name: str) -> None:
        """Use the clip called ``clip_name`` for ``role``."""
        if role not in self._roles:
            raise ValueError(f"unknown role {role!r}; choose from {ROLES}")
        if clip_name not in self.clips:
            raise KeyError(f"no clip called {clip_name!r}")
        if clip_name not in self._roles[role]:
            self._roles[role].append(clip_name)

    def names(self, role: str) -> list[str]:
        """Clip names registered for ``role`` (no fallbacks)."""
        return list(self._roles[role])

    def choices(self, role: str, sex: str = "male") -> list[AnimationClip]:
        """Clips :meth:`role` picks from: the role's (else a fallback role's) clips, ``sex`` variants first."""
        for candidate in (role, *_FALLBACK.get(role, ())):
            names = self._roles.get(candidate, [])
            if names:
                sex = "female" if str(sex).lower().startswith("f") else "male"
                preferred = [n for n in names if clip_sex(n) in (sex, None)]
                return [self.clips[n] for n in (preferred or names)]
        return []

    def has(self, role: str) -> bool:
        return bool(self.choices(role))

    def role(self, role: str, sex: str = "male", rng=None) -> AnimationClip:
        """A clip for ``role``, preferring the ``f_`` / ``m_`` variant matching ``sex``; random if ``rng`` is given.

        Falls back to any clip of the role, then to related roles (walk variants, ``idle`` for gestures).
        Raises ``KeyError`` when there is none (turn roles have no fallback: turn procedurally instead).
        """
        clips = self.choices(role, sex)
        if not clips:
            raise KeyError(f"no clip for role {role!r}")
        return _choice(rng, clips)

    def role_of(self, clip: AnimationClip) -> str | None:
        """The first role (in ``ROLES`` order) ``clip`` is registered for, else ``None``."""
        return next((role for role in ROLES if clip.name in self._roles[role]), None)

    def head_yaw(self, clip: AnimationClip) -> list[float]:
        """Per-frame yaw (degrees, unwrapped) of the head the clip produces, in character space (0 = facing +Z)."""
        return self._head(clip)[0]

    def head_elevation(self, clip: AnimationClip) -> list[float]:
        """Per-frame angle (degrees) of the head's forward axis below the horizontal (positive = looking down)."""
        return self._head(clip)[1]

    def _head(self, clip: AnimationClip) -> tuple[list[float], list[float]]:
        cached = self._yaw.get(id(clip))
        if cached is not None and cached[0] is clip:
            return cached[1]
        chain = _chain(clip)
        if chain:
            directions = _chain_rotation(clip, chain) @ self.head_forward
        elif clip.root in clip.bone_names:
            rotations = quat_to_matrix(np.asarray(clip.rotations)[:, clip.bone_names.index(clip.root)])
            directions = rotations @ (rotations[0].T @ np.array([0.0, 0.0, 1.0]))
        else:
            directions = np.tile([0.0, 0.0, 1.0], (clip.frames, 1))
        yaw = np.degrees(np.unwrap(np.arctan2(directions[:, 0], directions[:, 2])))
        yaw -= 360.0 * np.round(yaw[0] / 360.0)
        norm = np.maximum(np.linalg.norm(directions, axis=1), 1e-12)
        elevation = np.degrees(np.arcsin(np.clip(-directions[:, 1] / norm, -1.0, 1.0)))
        self._yaw[id(clip)] = (clip, ([float(v) for v in yaw], [float(v) for v in elevation]))
        return self._yaw[id(clip)][1]

    def after_turn(self, clip: AnimationClip) -> AnimationClip:
        """``clip`` with its root yawed back by its ``turn``: its pose once the heading has taken the turn."""
        cached = self._after.get(id(clip))
        if cached is not None and cached[0] is clip:
            return cached[1]
        turn = turn_angle(clip)
        rotations = np.array(clip.rotations, dtype=np.float64)
        translations = np.array(clip.translations, dtype=np.float64)
        if clip.root in clip.bone_names:
            k = clip.bone_names.index(clip.root)
            rotations[:, k] = quat_multiply(quat_from_axis_angle((0.0, 1.0, 0.0), -turn), rotations[:, k])
            c, s = math.cos(math.radians(-turn)), math.sin(math.radians(-turn))
            x, z = translations[:, k, 0].copy(), translations[:, k, 2].copy()
            translations[:, k, 0], translations[:, k, 2] = c * x + s * z, -s * x + c * z
        after = AnimationClip(f"{clip.name}@after", clip.fps, list(clip.bone_names), rotations, translations,
                              loop=clip.loop, root=clip.root, metadata={**clip.metadata, "turn": 0.0})
        self._after[id(clip)] = (clip, after)
        return after

    def _estimate_head_forward(self) -> np.ndarray:
        total = np.zeros(3)
        for clip in self.clips.values():
            chain = _chain(clip)
            if chain:
                total += _chain_rotation(clip, chain, slice(0, 1))[0].T @ np.array([0.0, 0.0, 1.0])
        norm = np.linalg.norm(total)
        return total / norm if norm > 1e-9 else np.array([0.0, 0.0, 1.0])


def turn_angle(clip: AnimationClip) -> float:
    """A turn clip's yaw change (degrees, positive = left) from its metadata, else from its name."""
    nominal = 0.0
    lowered = clip.name.lower()
    for angle in (180, 120, 90, 60):
        if f"_{angle}" in lowered:
            nominal = float(angle) * (-1.0 if "right" in lowered else 1.0)
            break
    turn = float(clip.metadata.get("turn", 0.0) or 0.0)
    if nominal and (abs(turn) < 30.0 or turn * nominal < 0):
        return nominal
    return turn if abs(turn) >= 1.0 else nominal


# -- segments ---------------------------------------------------------------------------------------


@dataclass
class Walk:
    """Walk from where the person stands through ``path`` (``(x, z)`` or ``(x, y, z)`` points)."""

    path: list
    speed: float | None = None  # m/s; None = from the style
    role: str | None = None  # walk clip role; None = from the style


@dataclass
class Idle:
    """Stand for ``duration`` seconds, cycling through clips of ``roles``."""

    duration: float
    roles: tuple = ("idle", "look_around")


@dataclass
class Turn:
    """Turn in place by ``degrees`` (positive = to the left); with ``either_way`` the behaviour may turn the other
    way round instead (``degrees`` -/+ 360) when that keeps the face off the camera."""

    degrees: float
    either_way: bool = False


@dataclass
class Hide:
    """Be invisible for ``duration`` seconds and reappear at ``to`` (default: where the loop starts)."""

    duration: float
    to: tuple | None = None
    heading: float | None = None


# -- timeline pieces --------------------------------------------------------------------------------


class _Stand:
    """Standing (idle, turning or hidden) at a fixed spot; heading ``h0 + dh * smooth((t - t0) / span)``."""

    __slots__ = ("t0", "t1", "x", "z", "h0", "dh", "span", "visible", "kind", "turn")

    def __init__(self, t0, duration, x, z, h0, dh=0.0, span=None, visible=True, kind="idle", turn=0.0):
        self.t0, self.t1 = t0, t0 + duration
        self.x, self.z, self.h0, self.dh = x, z, h0, dh
        self.span = span or duration
        self.visible, self.kind, self.turn = visible, kind, turn

    def state(self, tl: float) -> tuple[float, float, float]:
        if self.dh:
            return self.x, self.z, self.h0 + self.dh * _smooth((tl - self.t0) / self.span)
        return self.x, self.z, self.h0


class _WalkPiece:
    """A walk along precomputed arc-length samples with eased speed."""

    __slots__ = ("t0", "t1", "s", "px", "pz", "ph", "length", "speed", "ease", "blend", "blend_time", "visible",
                 "kind", "turn")

    def __init__(self, t0, samples, speed, blend):
        self.s, self.px, self.pz, self.ph = samples
        self.length = self.s[-1]
        self.speed = speed
        self.ease = min(0.7, self.length / speed)
        self.t0, self.t1 = t0, t0 + self.length / speed + self.ease
        self.blend, self.blend_time = blend, min(max(0.8, self.ease), self.t1 - t0)
        self.visible, self.kind, self.turn = True, "walk", 0.0

    def distance(self, local: float) -> float:
        v, e = self.speed, self.ease
        duration = self.t1 - self.t0
        if local <= 0.0:
            return 0.0
        if local >= duration:
            return self.length
        if local < e:
            u = local / e
            return v * e * (u ** 3 - u ** 4 / 2)
        if local > duration - e:
            u = (duration - local) / e
            return self.length - v * e * (u ** 3 - u ** 4 / 2)
        return v * e / 2 + v * (local - e)

    def at(self, d: float) -> tuple[float, float, float]:
        s = self.s
        k = bisect.bisect_right(s, d) - 1
        k = 0 if k < 0 else len(s) - 2 if k > len(s) - 2 else k
        a = (d - s[k]) / (s[k + 1] - s[k])
        return (self.px[k] + (self.px[k + 1] - self.px[k]) * a, self.pz[k] + (self.pz[k + 1] - self.pz[k]) * a,
                self.ph[k] + (self.ph[k + 1] - self.ph[k]) * a)

    def state(self, tl: float) -> tuple[float, float, float]:
        local = tl - self.t0
        x, z, h = self.at(self.distance(local))
        if self.blend and local < self.blend_time:
            h += self.blend * (1.0 - _smooth(local / self.blend_time))
        return x, z, h


class _Entry:
    """One clip on the clip track; ``sample(tl)`` gives ``(clip, clip time, head yaw, head elevation)`` of the
    clip (also past ``t1``)."""

    __slots__ = ("t0", "t1", "clip", "after", "head", "after_head", "walk", "phase", "scale", "role", "fade", "fps",
                 "last", "loop")

    def __init__(self, t0, t1, clip, head=None, role=None, walk=None, phase=0.0, scale=1.0, after=None,
                 after_head=None):
        self.t0, self.t1, self.clip, self.head, self.role = t0, t1, clip, head, role
        self.walk, self.phase, self.scale = walk, phase, scale
        self.after, self.after_head = after, after_head
        self.fade = 0.0
        if clip is not None:
            self.fps, self.last, self.loop = clip.fps, clip.frames - 1, clip.loop

    def sample(self, tl: float):
        if self.walk is not None:
            time = (self.walk.distance(tl - self.walk.t0) + self.phase) * self.scale
        else:
            time = self.phase + (tl - self.t0)
        clip, (yaw, elevation) = self.clip, self.head
        if self.after is not None and tl >= self.t1:
            clip, (yaw, elevation) = self.after, self.after_head
        last = self.last
        if last <= 0:
            return clip, time, yaw[0], elevation[0]
        f = time * self.fps
        f = f % last if self.loop else (0.0 if f < 0.0 else last if f > last else f)
        i = int(f)
        j = i + 1 if i < last else last
        a = f - i
        return clip, time, yaw[i] + (yaw[j] - yaw[i]) * a, elevation[i] + (elevation[j] - elevation[i]) * a


def _rounded_path(points: list[tuple[float, float]], radius: float = 1.0, step: float = 0.05):
    """Arc-length samples ``(s, x, z, heading)`` of a polyline whose corners are rounded with circular arcs."""
    pts = np.asarray(points, dtype=np.float64)
    seg = np.diff(pts, axis=0)
    lengths = np.linalg.norm(seg, axis=1)
    headings = np.degrees(np.arctan2(seg[:, 0], seg[:, 1]))
    cut = np.zeros(len(pts))  # distance from each vertex where its arc starts / ends
    turns = np.zeros(len(pts))
    radii = np.zeros(len(pts))
    for i in range(1, len(pts) - 1):
        turns[i] = _wrap(headings[i] - headings[i - 1])
        half = math.radians(abs(turns[i])) / 2
        if abs(turns[i]) < 1e-3:
            continue
        cut[i] = min(radius * math.tan(half), 0.45 * lengths[i - 1], 0.45 * lengths[i])
        radii[i] = cut[i] / math.tan(half)
    s_out, x_out, z_out, h_out = [0.0], [float(pts[0, 0])], [float(pts[0, 1])], []
    h_unwrapped = float(headings[0])
    h_out.append(h_unwrapped)

    def emit(x, z, h):
        s_out.append(s_out[-1] + math.hypot(x - x_out[-1], z - z_out[-1]))
        x_out.append(x)
        z_out.append(z)
        h_out.append(h)

    for i in range(len(seg)):
        a = pts[i] + seg[i] / lengths[i] * cut[i]
        b = pts[i + 1] - seg[i] / lengths[i] * cut[i + 1]
        if i > 0 and cut[i] > 0:  # the arc around vertex i, from the end of segment i-1 to ``a``
            sign = 1.0 if turns[i] > 0 else -1.0
            r = radii[i]
            start = pts[i] - seg[i - 1] / lengths[i - 1] * cut[i]
            h1 = math.radians(h_unwrapped)
            n = max(2, int(math.ceil(r * math.radians(abs(turns[i])) / step)))
            for k in range(1, n + 1):
                h = h1 + sign * math.radians(abs(turns[i])) * k / n
                emit(float(start[0] + sign * r * (math.cos(h1) - math.cos(h))),
                     float(start[1] + sign * r * (math.sin(h) - math.sin(h1))), math.degrees(h))
            h_unwrapped += turns[i]
            x_out[-1], z_out[-1] = float(a[0]), float(a[1])
        straight = float(np.linalg.norm(b - a))
        n = max(1, int(math.ceil(straight / step)))
        for k in range(1, n + 1):
            p = a + (b - a) * k / n
            emit(float(p[0]), float(p[1]), h_unwrapped)
    return s_out, x_out, z_out, h_out


# -- behaviours -------------------------------------------------------------------------------------


class _Camera:
    """A camera position spec (point, function of t, or object with ``position``) reduced to the floor plane."""

    def __init__(self, camera):
        self.fn = None
        self.nominal = None
        self.radius = 0.0
        if camera is None:
            return
        if isinstance(camera, _Camera):
            self.fn, self.nominal, self.radius = camera.fn, camera.nominal, camera.radius
            return
        position = getattr(camera, "position", camera)
        if callable(position):
            samples = np.array([_xz(position(t)) for t in np.arange(0.0, CAMERA_HORIZON, 0.5)])
            lo, hi = samples.min(axis=0), samples.max(axis=0)
            centre = (lo + hi) / 2
            self.nominal = (float(centre[0]), float(centre[1]))
            self.radius = float(np.linalg.norm(samples - centre, axis=1).max())
            self.fn = position
        else:
            self.nominal = _xz(position)

    def at(self, t: float) -> tuple[float, float]:
        if self.fn is None:
            return self.nominal
        p = self.fn(t)
        return float(p[0]), float(p[-1])


class _Style:
    """Avatar metadata reduced to what behaviours use. Without an explicit ``speed`` people walk at their walk
    clip's own speed (times ``stride_scale``) times a random pace, so clips play at about their natural rate."""

    def __init__(self, style, rng: random.Random):
        style = dict(style or {})
        self.sex = "female" if str(style.get("sex", "male")).lower().startswith("f") else "male"
        self.age = str(style.get("age_group", "adult")).lower()
        if self.age not in _PACE:
            self.age = "adult"
        self.walk_role = rng.choice(("walk_slow", "stroll")) if self.age == "older" else "walk"
        self.walk_role = style.get("walk_role", self.walk_role)
        self.pace = rng.uniform(*_PACE[self.age])
        self.speed = float(style["speed"]) if style.get("speed") else None
        self.stride = float(style.get("stride_scale") or _STRIDE[self.age])

    def walk_speed(self, clip: AnimationClip | None) -> float:
        if self.speed is not None:
            return self.speed
        natural = float(clip.metadata.get("speed") or 0.0) if clip is not None else 0.0
        return (natural or 1.2) * self.stride * self.pace


class Behaviour:
    """A looping timeline of segments: call it with a time to get the :class:`CharacterState` (module docs).

    Args:
        segments: :class:`Walk`, :class:`Idle`, :class:`Turn` and :class:`Hide` items, played in a loop. The loop
            is closed automatically (walk back to ``start``, turn to ``heading``) when the segments do not.
        library: the :class:`MotionLibrary` to take clips from (``None``: no clips, the rest pose moves around).
        start, heading: where the loop starts (``(x, z)`` or ``(x, y, z)``; the floor is y = 0) and the heading.
        style: avatar metadata (``sex``, ``age_group``, optional ``stride_scale``, ``speed``, ``walk_role``).
        camera: a point, a function of ``t``, or an object with such a ``position``; ``None`` = no camera to avoid.
        rng: seed (int), ``random.Random`` or numpy ``Generator`` for clip choices and glances.
        look_at: points worth glancing at (booths, screens). partners: people to look at while talking/listening.
        min_angle: degrees the face keeps away from the camera direction.
        offset: seconds added to ``t`` (where in its loop the behaviour is at ``t = 0``).
    """

    def __init__(self, segments, library: MotionLibrary | None = None, *, start=(0.0, 0.0), heading: float = 0.0,
                 style=None, camera=None, rng=0, look_at=(), partners=(), min_angle: float = MIN_ANGLE,
                 offset: float = 0.0):
        self.library = library
        self.rng = _as_random(rng)
        self.style = _Style(style, self.rng)
        self.min_angle = float(min_angle)
        self.offset = float(offset)
        self.camera = _Camera(camera)
        self.look_at = [_xz(p) for p in look_at]
        self.partners = [_xz(p) for p in partners]
        self.start = _xz(start)
        self.start_heading = float(heading)
        segments = list(segments)
        for segment in segments:
            if not isinstance(segment, (Walk, Idle, Turn, Hide)):
                raise TypeError(f"unknown segment {segment!r}")
        state = self.rng.getstate()
        self._build(segments, set())
        flips = self._choose_turns()
        if flips:
            self.rng.setstate(state)
            self._build(segments, flips)
        self._plan_head()

    # -- building ------------------------------------------------------------------------------------

    def _build(self, segments: list, flips: set[int]) -> None:
        """Lay out the pieces and the clip track; the automatic turns numbered in ``flips`` go the other way round."""
        self._pieces: list = []
        self._entries: list[_Entry] = []
        self._walk_clips: dict[str, AnimationClip | None] = {}
        self._side = 0
        self._flips, self._choices = flips, []
        self._t, self._pos, self._heading = 0.0, self.start, self.start_heading
        for k, segment in enumerate(segments):
            following = segments[k + 1] if k + 1 < len(segments) else None
            if isinstance(segment, Walk):
                self._add_walk(segment)
            elif isinstance(segment, Idle):
                self._add_idle(segment, following)
            elif isinstance(segment, Turn):
                self._add_turn(float(segment.degrees), choice=bool(segment.either_way))
            else:
                self._add_hide(segment)
        self._close_loop()
        self.duration = self._t
        self._piece_starts = [p.t0 for p in self._pieces]
        self._entry_starts = [e.t0 for e in self._entries]
        for j, entry in enumerate(self._entries):
            before = self._entries[j - 1]
            if entry.clip is not None and before.clip is not None:
                gap = abs(_wrap(before.sample(entry.t0 if j else entry.t0 + self.duration)[2] - entry.sample(entry.t0)[2]))
                fade = min(max(FADE, gap / _FADE_SPEED), _MAX_FADE)
                entry.fade = min(fade, 0.5 * (before.t1 - before.t0), 0.5 * (entry.t1 - entry.t0))

    def _choose_turns(self) -> set[int] | None:
        """Which turns with a choice of direction (automatic turns, ``Turn(either_way=True)``) to make the other way.

        The face can avoid the camera all along the loop only if ``psi = heading + clip yaw - camera bearing``
        (unwrapped) stays inside one window ``(w - HEAD_YAW_LIMIT, 360 - w + HEAD_YAW_LIMIT) + 360 k``, where ``w`` is
        the angle to keep: the face ``psi + head_yaw`` must stay inside one of the gaps between the camera
        directions. In particular ``psi`` must wind around as often as the camera bearing does over a loop. Flipping
        a turn of ``d`` degrees to ``d -/+ 360`` shifts ``psi`` by -/+ 360 from then on, so this picks the flips
        that fit ``psi`` into a window (with room to spare) and turn the least; ``None`` keeps the turns as they are.
        """
        cam = self.camera.nominal
        if cam is None or not self._choices or any(not p.visible for p in self._pieces):
            return None
        n = max(8, int(math.ceil(self.duration / _PLAN_DT)))
        times = np.arange(n + 1) * (self.duration / n)
        samples = np.array([self._body(tl)[:3] + (self._body(tl)[6],) for tl in times[:n]] +
                           [self._body(0.0)[:3] + (self._body(0.0)[6],)])
        dx, dz = cam[0] - samples[:, 0], cam[1] - samples[:, 1]
        distance = np.hypot(dx, dz)
        if distance.min() < MIN_CAMERA_DISTANCE + self.camera.radius:
            return None
        bearing = np.degrees(np.unwrap(np.radians(np.arctan2(dx, dz) * (180.0 / math.pi))))
        psi = samples[:, 2] + samples[:, 3] - bearing
        psi[-1] = psi[0] + (self._heading - self.start_heading) - (bearing[-1] - bearing[0])  # the loop's closure
        width = self.min_angle + _MARGIN + 3.0 + float(np.degrees(np.arcsin(min(
            self.camera.radius / max(float(distance.min()), 1e-9), 0.5))))
        low, high = width - HEAD_YAW_LIMIT, 360.0 - width + HEAD_YAW_LIMIT
        # intervals of the grid: between turns (shifted by the flips before them) and turns (stretched if flipped)
        bounds = [(int(np.searchsorted(times, t0)), int(np.searchsorted(times, t1)), d) for t0, t1, d in self._choices]
        flippable = [k for k, (_, _, d) in enumerate(bounds) if abs(d) >= 90.0][:10]
        pieces, i = [], 0
        for k, (i0, i1, d) in enumerate(bounds):
            if i0 > i:
                pieces.append((None, float(psi[i:i0].min()), float(psi[i:i0].max())))
            a, b = float(psi[min(i0, n)]), float(psi[min(i1, n)])
            seg = psi[min(i0, n):min(i1, n) + 1]
            pieces.append((k, a, b, float(seg.min()) - min(a, b), float(seg.max()) - max(a, b)))
            i = max(i, i1)
        if i <= n:
            pieces.append((None, float(psi[i:].min()), float(psi[i:].max())))
        best = None
        for mask in range(1 << len(flippable)):
            flips = {flippable[j] for j in range(len(flippable)) if mask >> j & 1}
            offset, lo, hi, turned = 0.0, math.inf, -math.inf, 0.0
            for piece in pieces:
                if piece[0] is None:
                    lo, hi = min(lo, piece[1] + offset), max(hi, piece[2] + offset)
                    continue
                k, a, b, below, above = piece
                d = bounds[k][2]
                shift = -math.copysign(360.0, d) if k in flips else 0.0
                turned += abs(d + shift)
                ends = (a + offset, b + offset + shift)
                lo, hi = min(lo, min(ends) + below), max(hi, max(ends) + above)
                offset += shift
            if abs(offset + psi[-1] - psi[0] - (psi[-1] - psi[0])) > 1e-6 and abs(offset) > 1e-6:
                pass
            if abs(psi[-1] + offset - psi[0] - 360.0 * round((psi[-1] + offset - psi[0]) / 360.0)) > 1e-6 or \
                    round((psi[-1] + offset - psi[0]) / 360.0) != 0:
                continue  # the loop must close without winding around the camera
            k = round(((lo - low) + (hi - high)) / 2.0 / 360.0)
            room = min(lo - low - 360.0 * k, 360.0 * k - (hi - high))
            score = (min(room, 10.0), -turned)
            if best is None or score > best[0]:
                best = (score, flips)
        if best is None or not best[1]:
            return None
        return best[1]

    def _add_auto_turn(self, delta: float) -> None:
        """Turn in place by ``delta`` (wrapped) the way :meth:`_turn_direction` suggests (unless flipped)."""
        self._add_turn(self._turn_direction(delta), choice=True)

    def _walk_clip(self, role: str) -> AnimationClip | None:
        if role not in self._walk_clips:
            lib = self.library
            self._walk_clips[role] = lib.role(role, self.style.sex, self.rng) if lib and lib.has(role) else None
        return self._walk_clips[role]

    def _turn_direction(self, delta: float) -> float:
        """``delta`` or the same turn the other way round, so that the face sweeps behind rather than past the camera."""
        delta = _wrap(delta)
        cam = self.camera.nominal
        if cam is None or abs(delta) < 45.0:
            return delta
        r0 = _wrap(_bearing(*self._pos, *cam) - self._heading)
        r1 = r0 - delta  # relative bearing of the camera after the turn (continuous)
        passes_front = min(r0, r1) < 0.0 < max(r0, r1) and min(abs(r0), abs(r1)) > 20.0
        long_way = delta - math.copysign(360.0, delta)
        return long_way if passes_front and abs(long_way) <= 270.0 else delta

    def _add_walk(self, walk: Walk) -> None:
        points = [self._pos]
        for p in walk.path:
            p = _xz(p)
            if math.hypot(p[0] - points[-1][0], p[1] - points[-1][1]) > 1e-3:
                points.append(p)
        if len(points) < 2:
            return
        legs, leg = [], [points[0]]
        for i in range(1, len(points)):
            leg.append(points[i])
            if i < len(points) - 1:
                h_in = _bearing(*points[i - 1], *points[i])
                h_out = _bearing(*points[i], *points[i + 1])
                if abs(_wrap(h_out - h_in)) > _SHARP:
                    legs.append(leg)
                    leg = [points[i]]
        legs.append(leg)
        role = walk.role or self.style.walk_role
        clip = self._walk_clip(role)
        speed = float(walk.speed or self.style.walk_speed(clip))
        for leg in legs:
            s, px, pz, ph = _rounded_path(leg)
            if s[-1] < 1e-3:
                continue
            delta = _wrap(ph[0] - self._heading)
            if abs(delta) > _WALK_TURN:
                self._add_auto_turn(delta)
                delta = _wrap(ph[0] - self._heading)
            shift = self._heading + delta - ph[0]
            piece = _WalkPiece(self._t, (s, px, pz, [h + shift for h in ph]), speed, -delta)
            self._pieces.append(piece)
            if clip is not None:
                natural = float(clip.metadata.get("speed") or 0.0) or 1.2
                self._entries.append(_Entry(piece.t0, piece.t1, clip, self.library._head(clip), role=role,
                                            walk=piece, phase=self.rng.uniform(0.0, natural * clip.duration),
                                            scale=1.0 / (natural * self.style.stride)))
            else:
                self._entries.append(_Entry(piece.t0, piece.t1, None, role=role))
            self._t, self._pos, self._heading = piece.t1, (px[-1], pz[-1]), piece.ph[-1]

    def _turn_clip(self, degrees: float) -> AnimationClip | None:
        lib = self.library
        if lib is None or abs(degrees) < 45.0:
            return None
        direction = "left" if degrees > 0 else "right"
        options = [lib.role(f"turn_{direction}_{a}", self.style.sex, self.rng) for a in (90, 180)
                   if lib.has(f"turn_{direction}_{a}")]
        if not options:
            return None
        clip = min(options, key=lambda c: abs(abs(turn_angle(c)) - abs(degrees)))
        turn = turn_angle(clip)
        if turn * degrees <= 0 or abs(degrees - turn) > max(45.0, 0.5 * abs(degrees)):
            return None  # e.g. only a 180 degree clip for a 90 degree turn: turn procedurally instead
        return clip

    def _add_turn(self, degrees: float, choice: bool = False) -> None:
        if abs(degrees) < 1e-6:
            return
        if choice:
            if len(self._choices) in self._flips:
                degrees -= math.copysign(360.0, degrees)
            self._choices.append((self._t, None, degrees))
        clip = self._turn_clip(degrees)
        x, z = self._pos
        if clip is not None:
            turn = turn_angle(clip)
            piece = _Stand(self._t, clip.duration, x, z, self._heading, degrees - turn, kind="turn", turn=degrees)
            after = self.library.after_turn(clip)
            self._entries.append(_Entry(piece.t0, piece.t1, clip, self.library._head(clip), role="turn",
                                        after=after, after_head=self.library._head(after)))
        else:
            duration = 0.5 + abs(degrees) / 180.0
            piece = _Stand(self._t, duration, x, z, self._heading, degrees, kind="turn", turn=degrees)
            self._extend_or_add_idle(piece.t0, piece.t1, "idle")
        self._pieces.append(piece)
        if choice:
            self._choices[-1] = (piece.t0, piece.t1, degrees)
        self._t, self._heading = piece.t1, self._heading + degrees

    def _extend_or_add_idle(self, t0: float, t1: float, role: str) -> None:
        last = self._entries[-1] if self._entries else None
        if last is not None and last.walk is None and last.after is None and last.t1 == t0 and \
                (last.clip is not None or self.library is None or not self.library.has(role)):
            last.t1 = t1
            return
        clip = self.library.role(role, self.style.sex, self.rng) if self.library and self.library.has(role) else None
        self._add_idle_entry(t0, t1, clip, role)

    def _add_idle_entry(self, t0: float, t1: float, clip: AnimationClip | None, role: str) -> None:
        """Add a clip from ``t0`` to ``t1``, starting where its head yaw is close to the previous clip's and ending
        with the head near the middle (where walks and turns start), so the crossfades stay gentle."""
        if clip is None:
            self._entries.append(_Entry(t0, t1, None, role=role))
            return
        entry = _Entry(t0, t1, clip, self.library._head(clip), role=role)
        before = self._entries[-1] if self._entries else None
        target = before.sample(t0)[2] if before is not None and before.clip is not None else None
        best = None
        for _ in range(24):
            entry.phase = self.rng.uniform(0.0, clip.duration)
            error = 0.5 * abs(entry.sample(t1)[2])
            if target is not None:
                error += abs(_wrap(entry.sample(t0)[2] - target))
            if best is None or error < best[0]:
                best = (error, entry.phase)
        entry.phase = best[1]
        self._entries.append(entry)

    def _idle_options(self, roles, side: int) -> list[tuple[str, AnimationClip]]:
        """``(role, clip)`` pairs for an idle here whose own head motion leaves room to avoid the camera."""
        lib = self.library
        if lib is None:
            return []
        options = self._role_clips(roles)
        cam = self.camera.nominal
        if cam is None or not options:
            return options
        distance = math.hypot(cam[0] - self._pos[0], cam[1] - self._pos[1])
        if distance < MIN_CAMERA_DISTANCE:
            return options
        r = _wrap(_bearing(*self._pos, *cam) - self._heading)
        width = self.min_angle + _MARGIN + self._camera_spread(distance) + 3.0
        bins = np.arange(-HEAD_YAW_LIMIT, HEAD_YAW_LIMIT + 1e-9, _BIN)

        def fits(clip):  # at every frame, a head yaw on ``side`` of the camera direction or no conflict at all
            centre = _wrap_array(r - np.asarray(lib.head_yaw(clip))[:: max(1, clip.frames // 60)])
            clear = np.abs(centre) >= HEAD_YAW_LIMIT + width
            return bool(np.all(clear | np.any(side * (bins[None, :] - centre[:, None]) >= width, axis=1)))

        good = [(role, clip) for role, clip in options if fits(clip)]
        if good:
            return good
        others = self._role_clips(("idle", "listen", "phone", "look_around", "talk"))
        good = [(role, clip) for role, clip in others if fits(clip)]
        if good:
            return good
        return [min(options, key=lambda rc: float(np.ptp(lib.head_yaw(rc[1]))))]

    def _role_clips(self, roles) -> list[tuple[str, AnimationClip]]:
        """``(role, clip)`` for the clips of ``roles`` (with fallbacks); ``role`` is the one the clip really plays."""
        lib, out = self.library, []
        for role in roles:
            for clip in lib.choices(role, self.style.sex):
                actual = role if clip.name in lib.names(role) else lib.role_of(clip) or role
                if all(c is not clip for _, c in out):
                    out.append((actual, clip))
        return out

    def _idle_side(self, following) -> int:
        if isinstance(following, Turn) and abs(following.degrees) >= 90.0:
            flipped = following.either_way and len(self._choices) in self._flips
            return 1 if (following.degrees > 0) != flipped else -1
        if not self._side:
            cam = self.camera.nominal
            r = _wrap(_bearing(*self._pos, *cam) - self._heading) if cam is not None else 0.0
            self._side = (-1 if r > 0 else 1) if abs(r) > 2.0 else self.rng.choice((-1, 1))
        return self._side

    def _add_idle(self, idle: Idle, following) -> None:
        duration = float(idle.duration)
        if duration <= 0:
            return
        x, z = self._pos
        options = self._idle_options(tuple(idle.roles), self._idle_side(following))
        t, end, previous = self._t, self._t + duration, None
        while end - t > 1e-9:
            length = self.rng.uniform(3.5, 7.0)
            if end - (t + length) < 2.0:
                length = end - t
            if options:
                pool = [o for o in options if o[1] is not previous] or options
                role, clip = _choice(self.rng, pool)
            else:
                role, clip = idle.roles[0] if idle.roles else "idle", None
            t1 = end if end - (t + length) < 1e-9 else t + length
            self._add_idle_entry(t, t1, clip, role)
            previous = clip
            t = t1
        self._pieces.append(_Stand(self._t, duration, x, z, self._heading))
        self._t = end

    def _add_hide(self, hide: Hide) -> None:
        to = _xz(hide.to) if hide.to is not None else self.start
        heading = float(hide.heading) if hide.heading is not None else \
            self.start_heading if hide.to is None else self._heading
        piece = _Stand(self._t, float(hide.duration), *to, heading, visible=False, kind="hide")
        self._pieces.append(piece)
        self._entries.append(_Entry(piece.t0, piece.t1, None, role="hide"))
        self._t, self._pos, self._heading = piece.t1, to, heading

    def _close_loop(self) -> None:
        if math.hypot(self._pos[0] - self.start[0], self._pos[1] - self.start[1]) > 0.01:
            self._add_walk(Walk([self.start]))
        delta = _wrap(self.start_heading - self._heading)
        if abs(delta) > 1e-3:
            self._add_auto_turn(delta)
        if self._t <= 0.0:
            self._add_idle(Idle(5.0), None)

    # -- head plan -----------------------------------------------------------------------------------

    def _camera_spread(self, distance: float) -> float:
        """Degrees the camera direction may move away from its nominal direction (camera motion)."""
        if self.camera.radius <= 0.0 or distance <= 0.0:
            return 0.0
        return math.degrees(math.asin(min(self.camera.radius / distance, 0.5)))

    def _body(self, tl: float):
        """``(x, z, heading, piece, entry, layers, clip_yaw, clip_elevation)`` at loop time ``tl``."""
        piece = self._pieces[bisect.bisect_right(self._piece_starts, tl) - 1]
        x, z, heading = piece.state(tl)
        j = bisect.bisect_right(self._entry_starts, tl) - 1
        entry = self._entries[j]
        if entry.clip is None:
            return x, z, heading, piece, entry, [], 0.0, 0.0
        clip, time, yaw, elevation = entry.sample(tl)
        if entry.fade and tl - entry.t0 < entry.fade:
            w = _smooth((tl - entry.t0) / entry.fade)
            before = self._entries[j - 1]
            pclip, ptime, pyaw, pelevation = before.sample(tl if j else tl + self.duration)
            pyaw += 360.0 * round((yaw - pyaw) / 360.0)
            return x, z, heading, piece, entry, [Layer(pclip, ptime, 1.0 - w), Layer(clip, time, w)], \
                pyaw + (yaw - pyaw) * w, pelevation + (elevation - pelevation) * w
        return x, z, heading, piece, entry, [Layer(clip, time, 1.0)], yaw, elevation

    def _plan_head(self) -> None:
        n = max(8, int(math.ceil(self.duration / _PLAN_DT)))
        self._dt = dt = self.duration / n
        body = [self._body(i * dt) for i in range(n)]
        x = np.array([b[0] for b in body])
        z = np.array([b[1] for b in body])
        heading = np.array([b[2] for b in body])
        clip_yaw = np.array([b[6] for b in body])
        clip_elevation = np.array([b[7] for b in body])
        visible = np.array([b[3].visible for b in body])
        want_yaw, want_pitch = self._desired_looks(body, dt)
        bins = np.arange(-HEAD_YAW_LIMIT, HEAD_YAW_LIMIT + 1e-9, _BIN)
        cam = self.camera.nominal
        constrained = np.zeros(n, dtype=bool)
        if cam is not None:
            dx, dz = cam[0] - x, cam[1] - z
            distance = np.hypot(dx, dz)
            bearing = np.degrees(np.arctan2(dx, dz))
            constrained = visible & (distance >= MIN_CAMERA_DISTANCE)
            face_centre = bearing - heading - clip_yaw  # head yaw that would face the camera
            plain_centre = bearing - heading
            if self.camera.radius > 0.0:
                spread = np.degrees(np.arcsin(np.minimum(self.camera.radius / np.maximum(distance, 1e-9), 0.5)))
            else:
                spread = np.zeros(n)
            facing = np.abs(_wrap_array(face_centre)) < 50.0
            walking = np.array([b[3].kind == "walk" for b in body])
            want_pitch = want_pitch + np.where(facing & walking, 12.0, 0.0)
        pitch = np.clip(_periodic_smooth(want_pitch, 0.4 / dt), -20.0, 40.0)
        deviation = ((bins[None, :] - want_yaw[:, None]) / 20.0) ** 2
        self.look_feasible = True
        path, width = None, np.zeros(n)
        if cam is not None and constrained.any():
            # where the face points (relative to heading + clip yaw) for every head yaw bin, pitch included
            face = bins[None, :] + pitch_shift(clip_yaw[:, None], clip_elevation[:, None], bins[None, :],
                                               pitch[:, None], vector=True)
            off_camera = np.abs(_wrap_array(face - face_centre[:, None]))
            plain = np.abs(_wrap_array(bins[None, :] - plain_centre[:, None])) < self.min_angle + _MARGIN
            soft = deviation + np.where(plain & constrained[:, None], 25.0, 0.0)
            for margin, use_spread in ((_MARGIN, True), (0.0, True), (0.0, False)):
                width = self.min_angle + margin + (spread if use_spread else np.zeros(n))
                blocked = (off_camera < width[:, None]) & constrained[:, None]
                path = _plan(np.where(blocked, np.inf, soft), dt)
                if path is not None:
                    break
            if path is None:
                self.look_feasible = False
                path = _plan(np.where(blocked, 50.0, 0.0) + soft, dt)
        else:
            path = _plan(deviation, dt)
        yaw = _periodic_smooth(bins[path], 0.15 / dt)
        if cam is not None and self.look_feasible:
            d = _wrap_array(face[np.arange(n), path] - face_centre)
            low = np.where(constrained & (d >= 0), bins[path] - (d - width), -HEAD_YAW_LIMIT)
            high = np.where(constrained & (d < 0), bins[path] - d - width, HEAD_YAW_LIMIT)
            yaw = np.clip(yaw, np.maximum(low, -HEAD_YAW_LIMIT), np.minimum(high, HEAD_YAW_LIMIT))
        self._grid_yaw = [float(v) for v in np.clip(yaw, -HEAD_YAW_LIMIT, HEAD_YAW_LIMIT)]
        self._grid_pitch = [float(v) for v in pitch]
        self._grid_yaw.append(self._grid_yaw[0])
        self._grid_pitch.append(self._grid_pitch[0])

    def _desired_looks(self, body, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """Head yaw / pitch the person would like (before avoiding the camera), sampled every ``dt``."""
        n = len(body)
        yaw, pitch = np.zeros(n), np.zeros(n)
        rng = random.Random(self.rng.random())
        glance_end, next_glance, target, piece_seen = -1.0, rng.uniform(0.5, 3.0), None, None
        phone_yaw = {}
        for i, (x, z, heading, piece, entry, _, clip_yaw, _) in enumerate(body):
            t = i * dt
            if not piece.visible:
                continue
            if piece is not piece_seen:
                piece_seen, glance_end = piece, -1.0
                next_glance = t + rng.uniform(0.5, 2.5)
            role = entry.role
            if piece.kind == "walk":
                ahead = piece.at(min(piece.length, piece.distance(t - piece.t0) + 2.0))[2]
                base = float(np.clip(0.7 * (ahead - heading), -45.0, 45.0))
            elif piece.kind == "turn":  # look into the turn until the body has caught up
                done = (heading + clip_yaw - piece.h0) / piece.turn
                base = math.copysign(min(40.0, abs(piece.turn) / 2), piece.turn) * (1.0 - min(max(done, 0.0), 1.0))
            else:
                base = 0.0
            if role == "phone":
                key = id(entry)
                phone_yaw.setdefault(key, rng.uniform(-8.0, 8.0))
                yaw[i], pitch[i] = phone_yaw[key], 28.0
                continue
            if role == "look_around" or piece.kind == "turn":
                yaw[i] = base
                continue
            if t >= next_glance:
                target = self._glance_target(rng, x, z, heading, role, piece.kind)
                glance_end = t + (rng.uniform(1.5, 4.0) if role in ("talk", "listen") else rng.uniform(0.8, 2.2))
                gap = rng.uniform(0.0, 0.6) if role in ("talk", "listen") and self.partners else rng.uniform(1.5, 5.0)
                next_glance = glance_end + gap
            if t < glance_end and target is not None:
                if isinstance(target, tuple):
                    yaw[i] = _wrap(_bearing(x, z, *target) - heading) - 0.6 * clip_yaw
                    pitch[i] = 3.0
                else:
                    yaw[i] = target
            else:
                yaw[i] = base
        yaw = np.clip(yaw, -HEAD_YAW_LIMIT, HEAD_YAW_LIMIT)
        return _periodic_smooth(yaw, 0.3 / dt), pitch

    def _glance_target(self, rng: random.Random, x: float, z: float, heading: float, role, kind: str):
        """A point to look at (partner or booth in view), else a yaw angle."""
        if role in ("talk", "listen") and self.partners:
            return rng.choice(self.partners)
        reach = 15.0 if kind == "walk" else 10.0
        seen = [p for p in self.look_at if math.hypot(p[0] - x, p[1] - z) < reach and
                abs(_wrap(_bearing(x, z, *p) - heading)) < 75.0]
        if seen and rng.random() < 0.7:
            return rng.choice(seen)
        if kind == "walk" and rng.random() < 0.4:
            return None
        return rng.choice((-1.0, 1.0)) * rng.uniform(15.0, 50.0 if kind == "walk" else 40.0)

    # -- evaluation ----------------------------------------------------------------------------------

    def _evaluate(self, t: float):
        tl = (t + self.offset) % self.duration
        x, z, heading, piece, entry, layers, clip_yaw, elevation = self._body(tl)
        u = tl / self._dt
        i = int(u)
        if i >= len(self._grid_yaw) - 1:
            i = len(self._grid_yaw) - 2
        a = u - i
        gy, gp = self._grid_yaw, self._grid_pitch
        yaw = gy[i] + (gy[i + 1] - gy[i]) * a
        pitch = gp[i] + (gp[i + 1] - gp[i]) * a
        visible = piece.visible
        if visible and self.camera.nominal is not None:
            yaw = self._avoid_camera(t, x, z, heading, clip_yaw, elevation, yaw, pitch)
        return CharacterState((x, 0.0, z), heading, layers, yaw, pitch, visible), clip_yaw, elevation

    def _avoid_camera(self, t: float, x: float, z: float, heading: float, clip_yaw: float, elevation: float,
                      yaw: float, pitch: float) -> float:
        """Push ``yaw`` out until the face is ``min_angle`` from the actual camera direction, on the planned side."""
        camera = self.camera
        nx, nz = camera.nominal
        cx, cz = camera.at(t) if camera.fn is not None else (nx, nz)
        if (cx - x) ** 2 + (cz - z) ** 2 < MIN_CAMERA_DISTANCE ** 2:
            return yaw
        nominal = math.degrees(math.atan2(nx - x, nz - z))
        moved = 0.0 if camera.fn is None else _wrap(math.degrees(math.atan2(cx - x, cz - z)) - nominal)
        base = heading + clip_yaw - nominal
        side = 0.0
        for _ in range(4):
            planned = _wrap(base + yaw + pitch_shift(clip_yaw, elevation, yaw, pitch))
            if not side:
                side = 1.0 if planned >= 0 else -1.0
            short = self.min_angle + 1e-6 - side * (planned - moved)
            if short <= 0.0:
                break
            yaw += side * short
            if yaw > HEAD_YAW_LIMIT or yaw < -HEAD_YAW_LIMIT:
                return HEAD_YAW_LIMIT if yaw > 0 else -HEAD_YAW_LIMIT
        return yaw

    def __call__(self, t: float) -> CharacterState:
        return self._evaluate(t)[0]

    def face_heading(self, t: float) -> float:
        """Where the face points on the floor plane at ``t`` (degrees): ``heading + clip head yaw + head_yaw`` plus
        the small sideways effect of ``head_pitch`` (:func:`pitch_shift`)."""
        state, clip_yaw, elevation = self._evaluate(t)
        return state.heading + clip_yaw + state.head_yaw + pitch_shift(clip_yaw, elevation, state.head_yaw,
                                                                      state.head_pitch)

    def body_heading(self, t: float) -> float:
        """``heading + clip head yaw`` at ``t`` (continuous, unlike ``heading`` at the end of a turn clip)."""
        state, clip_yaw, _ = self._evaluate(t)
        return state.heading + clip_yaw

    def teleports(self) -> list[tuple[float, float]]:
        """Loop-time intervals ``(start, end)`` during which the person is hidden (and may jump)."""
        return [(p.t0, p.t1) for p in self._pieces if not p.visible]


def _wrap_array(a):
    return (np.asarray(a, dtype=np.float64) + 180.0) % 360.0 - 180.0


def _periodic_smooth(values: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing of a periodic sequence (``sigma`` in samples)."""
    values = np.asarray(values, dtype=np.float64)
    if sigma < 0.5 or len(values) < 3:
        return values.copy()
    half = int(min(3 * sigma, len(values) // 2))
    kernel = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.concatenate([values[-half:], values, values[:half]]) if half else values
    return np.convolve(padded, kernel, mode="valid")


def _plan(cost: np.ndarray, dt: float) -> np.ndarray | None:
    """Cheapest periodic path through ``cost`` (``(N, bins)``, ``inf`` = forbidden) moving a few bins per step.

    Starts and ends in the same bin at the least constrained step; returns bin indices per step, or ``None``.
    """
    n, bins = cost.shape
    reach = max(1, int(math.ceil(_HEAD_SPEED * dt / _BIN)))
    finite = np.isfinite(cost)
    start = int(np.argmax(finite.sum(axis=1)))
    if not finite[start].any():
        return None
    order = (start + np.arange(n + 1)) % n
    offsets = np.arange(-reach, reach + 1)
    step_cost = 0.02 * offsets.astype(np.float64) ** 2
    padded = np.full(bins + 2 * reach, np.inf)
    previous = padded[reach:reach + bins]
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * reach + 1)  # a view: follows ``padded``
    first = int(np.argmin(cost[start]))
    previous[first] = 0.0
    back = np.empty((n, bins), dtype=np.int8)
    rows = np.arange(bins)
    options = np.empty((bins, 2 * reach + 1))
    for k in range(1, n + 1):
        np.add(windows, step_cost, out=options)
        best = options.argmin(axis=1)
        previous[:] = options[rows, best] + cost[order[k]]
        back[k - 1] = best - reach
    total = previous
    if not np.isfinite(total[first]):
        return None
    path = np.empty(n + 1, dtype=np.int64)
    path[n] = first
    for k in range(n, 0, -1):
        path[k - 1] = path[k] + back[k - 1, path[k]]
    out = np.empty(n, dtype=np.int64)
    out[order[:n]] = path[:n]
    return out


# -- factories --------------------------------------------------------------------------------------


def _offset(rng: random.Random, offset, behaviour: Behaviour) -> Behaviour:
    behaviour.offset = rng.uniform(0.0, behaviour.duration) if offset is None else float(offset)
    return behaviour


def _away_from_camera(position, heading: float, camera: _Camera) -> tuple[float, float]:
    """``(turn, clearance)``: the turn around (+180 or -180) whose face sweeps behind the person rather than past the
    camera, and how clearly (degrees the camera is off the line ahead / behind; 0 = either way would do)."""
    cam = camera.nominal
    if cam is None:
        return 180.0, 0.0
    r = _wrap(_bearing(*position, *cam) - heading)
    return (-180.0 if r > 0 else 180.0), min(abs(r), 180.0 - abs(r))


def _turns_around(far, far_heading, near, near_heading, camera: _Camera, rng: random.Random):
    """Turn-around angles at both ends of an out-and-back path: away from the camera, and always opposite ways.

    Over a loop the face must not wind around the camera (see :meth:`Behaviour._choose_turns`), so two turns of
    the same sense are impossible to watch without the face sweeping across the camera somewhere; the end where
    the camera is more clearly to one side chooses.
    """
    at_far, far_clear = _away_from_camera(far, far_heading, camera)
    at_near, near_clear = _away_from_camera(near, near_heading, camera)
    if max(far_clear, near_clear) < 3.0:
        at_far = rng.choice((180.0, -180.0))
        return at_far, -at_far
    return (at_far, -at_far) if far_clear >= near_clear else (-at_near, at_near)


def walk_by(points, speed: float | None = None, *, library: MotionLibrary | None, return_mode: str = "turn",
            camera=None, style=None, rng=0, pause=(0.0, 2.5), gap=(0.5, 4.0), look_at=(),
            min_angle: float = MIN_ANGLE, offset: float | None = None) -> Behaviour:
    """Walk along ``points`` (start and end, or a polyline), then come back.

    ``return_mode="turn"``: pause, turn around (away from the camera) and walk back, turn again, repeat.
    ``return_mode="teleport"``: disappear at the end for ``gap`` seconds and reappear at the start (put both
    ends outside the camera's view). ``offset`` = where in the loop ``t = 0`` is (random when ``None``).
    """
    rng = _as_random(rng)
    pts = [_xz(p) for p in points]
    if len(pts) < 2:
        raise ValueError("walk_by needs at least two points")
    if return_mode not in ("turn", "teleport"):
        raise ValueError("return_mode must be 'turn' or 'teleport'")
    camera = _Camera(camera)
    heading = _bearing(*pts[0], *pts[1])
    segments: list = [Walk(pts[1:], speed)]
    if return_mode == "turn":
        at_end, at_start = _turns_around(pts[-1], _bearing(*pts[-2], *pts[-1]), pts[0], _bearing(*pts[1], *pts[0]),
                                         camera, rng)
        segments += [Idle(rng.uniform(*pause), ("look_around", "idle")), Turn(at_end), Walk(pts[::-1][1:], speed),
                     Idle(rng.uniform(*pause), ("look_around", "idle")), Turn(at_start)]
        segments = [s for s in segments if not isinstance(s, Idle) or s.duration > 0.3]
    else:
        segments.append(Hide(rng.uniform(*gap)))
    behaviour = Behaviour(segments, library, start=pts[0], heading=heading, style=style, camera=camera, rng=rng,
                          look_at=look_at, min_angle=min_angle)
    return _offset(rng, offset, behaviour)


def approach(start, stop_point=None, camera=None, *, library: MotionLibrary | None, stop_distance=(2.5, 5.0),
             idle=(3.0, 8.0), far_pause=(0.5, 3.0), bow: float | None = None, speed: float | None = None, style=None,
             rng=0, look_at=(), min_angle: float = MIN_ANGLE, offset: float | None = None) -> Behaviour:
    """Walk from ``start`` towards the camera, stop, idle (phone, looking around), turn around, walk back, repeat.

    ``stop_point`` defaults to a point on the way to the camera ``stop_distance`` metres (random in the range)
    from it, and is moved back if it is nearer than ``stop_distance[0]``. ``bow`` bends the path sideways by that
    many metres at its middle (random up to 0.8 m when ``None``; 0 walks straight at the camera).
    """
    rng = _as_random(rng)
    if camera is None:
        raise ValueError("approach needs the camera")
    camera = _Camera(camera)
    cx, cz = camera.nominal
    start = _xz(start)
    near, far = stop_distance
    if stop_point is None:
        away, distance = (start[0] - cx, start[1] - cz), rng.uniform(near, far)
    else:
        stop_point = _xz(stop_point)
        away = (stop_point[0] - cx, stop_point[1] - cz)
        distance = max(near, math.hypot(*away))
        if math.hypot(*away) < 1e-6:
            away = (start[0] - cx, start[1] - cz)
    norm = math.hypot(*away) or 1.0
    if stop_point is None or distance > math.hypot(stop_point[0] - cx, stop_point[1] - cz):
        stop_point = (cx + away[0] / norm * distance, cz + away[1] / norm * distance)
    bow = rng.uniform(-0.8, 0.8) if bow is None else float(bow)
    mid = ((start[0] + stop_point[0]) / 2, (start[1] + stop_point[1]) / 2)
    length = math.hypot(stop_point[0] - start[0], stop_point[1] - start[1]) or 1.0
    side = (-(stop_point[1] - start[1]) / length, (stop_point[0] - start[0]) / length)
    path = [start, (mid[0] + side[0] * bow, mid[1] + side[1] * bow), stop_point] if abs(bow) > 0.05 else \
        [start, stop_point]
    heading = _bearing(*path[0], *path[1])
    at_stop, at_start = _turns_around(stop_point, _bearing(*path[-2], *path[-1]), start, _bearing(*path[1], *path[0]),
                                      camera, rng)
    segments = [Walk(path[1:], speed), Idle(rng.uniform(*idle), ("phone", "look_around", "idle")), Turn(at_stop),
                Walk(path[::-1][1:], speed), Idle(rng.uniform(*far_pause), ("look_around", "idle", "phone")),
                Turn(at_start)]
    behaviour = Behaviour(segments, library, start=start, heading=heading, style=style, camera=camera, rng=rng,
                          look_at=look_at, min_angle=min_angle)
    return _offset(rng, offset, behaviour)


def _region_sampler(region):
    """A function ``rng -> (x, z)`` for an axis-aligned rect ``((x0, z0), (x1, z1))`` or a polygon."""
    pts = [_xz(p) for p in region]
    if len(pts) == 2:
        (x0, z0), (x1, z1) = pts
        lo_x, hi_x, lo_z, hi_z = min(x0, x1), max(x0, x1), min(z0, z1), max(z0, z1)
        return lambda rng: (rng.uniform(lo_x, hi_x), rng.uniform(lo_z, hi_z))
    if len(pts) < 3:
        raise ValueError("region must be ((x0, z0), (x1, z1)) or a polygon of at least three points")
    xs, zs = [p[0] for p in pts], [p[1] for p in pts]

    def inside(x, z):
        hit = False
        for (ax, az), (bx, bz) in zip(pts, pts[1:] + pts[:1]):
            if (az > z) != (bz > z) and x < ax + (z - az) * (bx - ax) / (bz - az):
                hit = not hit
        return hit

    def sample(rng):
        for _ in range(1000):
            x, z = rng.uniform(min(xs), max(xs)), rng.uniform(min(zs), max(zs))
            if inside(x, z):
                return x, z
        return sum(xs) / len(xs), sum(zs) / len(zs)

    return sample


def wander(region, rng=0, *, library: MotionLibrary | None, points: int = 4, pause=(1.0, 6.0),
           pause_probability: float = 0.7, min_leg: float = 2.0, speed: float | None = None, camera=None,
           style=None, look_at=(), min_angle: float = MIN_ANGLE, offset: float | None = None,
           attempts: int = 12) -> Behaviour:
    """Walk between ``points`` random points inside ``region`` (rect ``((x0, z0), (x1, z1))`` or polygon), pausing."""
    rng = _as_random(rng)
    camera = _Camera(camera)
    sample = _region_sampler(region)
    best = None
    for _ in range(max(1, attempts)):
        stops = [sample(rng)]
        while len(stops) < max(2, points):
            p = sample(rng)
            for _ in range(20):
                if math.hypot(p[0] - stops[-1][0], p[1] - stops[-1][1]) >= min_leg:
                    break
                p = sample(rng)
            stops.append(p)
        segments: list = []
        for p in stops[1:] + stops[:1]:
            segments.append(Walk([p], speed))
            if rng.random() < pause_probability:
                segments.append(Idle(rng.uniform(*pause), ("look_around", "idle", "phone")))
        heading = _bearing(*stops[0], *stops[1])
        behaviour = Behaviour(segments, library, start=stops[0], heading=heading, style=style, camera=camera,
                              rng=random.Random(rng.random()), look_at=look_at, min_angle=min_angle)
        best = behaviour
        if behaviour.look_feasible:
            break
    return _offset(rng, offset, best)


def _standing(position, heading: float, roles, rng: random.Random, duration, sway, library, style, camera,
              look_at, partners, min_angle, offset) -> Behaviour:
    roles = tuple(roles)
    segments: list = []
    total, target = 0.0, rng.uniform(*duration)
    k = 0
    while total < target:
        length = rng.uniform(4.0, 9.0)
        segments.append(Idle(length, (roles[k % len(roles)],) if len(roles) > 1 else roles))
        total += length
        k += 1
        if sway and rng.random() < 0.35:
            angle = rng.choice((-1.0, 1.0)) * rng.uniform(*sway)
            segments += [Turn(angle), Idle(rng.uniform(3.0, 7.0), (roles[k % len(roles)],)), Turn(-angle)]
            k += 1
    behaviour = Behaviour(segments, library, start=position, heading=heading, style=style, camera=camera, rng=rng,
                          look_at=look_at, partners=partners, min_angle=min_angle)
    behaviour.offset = rng.uniform(0.0, behaviour.duration) if offset is None else float(offset)
    return behaviour


def chat(position, face_point, partners=(), roles=("talk", "listen"), *, library: MotionLibrary | None,
         duration=(25.0, 45.0), sway=(6.0, 15.0), camera=None, style=None, rng=0, look_at=(),
         min_angle: float = MIN_ANGLE, offset: float | None = None) -> Behaviour:
    """Stand at ``position`` facing ``face_point`` in a group, alternating ``roles`` clips and looking at ``partners``.

    Small weight shifts turn the body by ``sway`` degrees now and then (and back).
    """
    rng = _as_random(rng)
    position = _xz(position)
    heading = _bearing(*position, *_xz(face_point))
    return _standing(position, heading, roles, rng, duration, sway, library, style, _Camera(camera), look_at, partners,
                     min_angle, offset)


def stand(position, heading: float = 0.0, roles=("idle", "look_around", "phone"), *, library: MotionLibrary | None,
          duration=(30.0, 50.0), sway=(10.0, 25.0), camera=None, style=None, rng=0, look_at=(),
          min_angle: float = MIN_ANGLE, offset: float | None = None) -> Behaviour:
    """Stand at ``position`` facing ``heading`` (e.g. booth staff), cycling ``roles`` clips with small turns."""
    rng = _as_random(rng)
    return _standing(_xz(position), float(heading), roles, rng, duration, sway, library, style, _Camera(camera),
                     look_at, (),
                     min_angle, offset)
