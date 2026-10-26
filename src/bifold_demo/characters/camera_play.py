"""People in front of the camera playing with their picture: waving, making faces, stepping and leaning about.

The demo screen shows what the camera sees, so the audience of our exhibit (``scenes.fair``) sees itself like in a
mirror, and now and then someone plays with it, the way people do: they wave at the camera, make faces at it, step to
one side and the other to watch their picture follow, lean from side to side, lean in to peer at it, or walk up closer
and back again. :data:`MOVES` names the six moves; :func:`play_along` gives them to people who watch the camera::

    plays = play_along(audience, camera=(0, 1.6, 6), others=everyone_else, period=360.0, rng=5)

Plan. :func:`plan_camera_play` decides for everyone together, once, who does which move when: a pure function of the
scene time that repeats every ``period`` (the fair's ``LOOP_PERIOD``, so the hall still repeats), from a seeded random
generator. A move only happens while the person watches the camera (``Behaviour.watching``), and only when all of it
fits into that stretch, ``MARGIN`` seconds clear of the turn to the camera and away from it: visitors who stay for
15-30 s do one to three moves. How much someone plays depends on them: children and some adults are playful (more
and bigger moves), some people are shy (few, small ones), and on average people play about a third of the time they
watch (``PLAY_SHARE``; a visitor stops once they have played their share of the visit). Every move is checked
against everybody's positions over the whole period (``Behaviour.positions``, people walking by included) and against
the moves planned before it: the body must keep ``CLEARANCE`` from anyone else, and the elbows, the hands and the head
(:func:`reach_points`: where the arms hang, or where the wave or the gesture takes them, by its weight) ``REACH``
from anyone else's body axis and ``HANDS`` from the others' moving arms, unless they are that near anyway. A move
that would not is turned the other way, made smaller, done with one hand or without the hands at the face, or
replaced by a lean. ``place`` (the fair: where the feet stay in the picture) keeps the travelling moves where they
belong.

Moves. A :class:`Move` is a table of channels (:data:`CHANNELS`) sampled ``RATE`` times a second and interpolated in
between: where the body is (an offset from the person's spot, to their left and forward), how the hips shift, how far
the torso bends forward and tilts sideways, where each foot is (relative to where the idle clip puts it) and how high
it is lifted, deliberate head tilts and turns, how much the arms wave or make a gesture, and facial expressions
(:mod:`.faces`). Every channel starts and ends at zero, so the idle clips that keep playing underneath carry on
seamlessly. A foot moves only while it steps, one foot at a time, in a low arc; the body follows the feet, shifting its
weight onto the standing foot first, the knees a little bent (``SINK``) so the feet lift off and land softly. Timings,
sizes, wave speeds and the pauses to check the picture are random within natural ranges, scaled by the person's size
and playfulness.

Doing them. :class:`CameraPlay` wraps the person's behaviour (like ``fair.Raised``: everything else is the behaviour's)
and moves the person by the body offset, so the rest of the system sees where they are (``positions`` too). It points
the head at the camera from where the eyes now are and makes up for the bent torso, so that the face keeps on the
camera through every move but for the deliberate tilts and wobbles of making faces. :class:`PlayPose`, a pose
adjuster of the :class:`~.character.Character` (``pose_adjusters``), does the rest on the clips' local bone transforms
before forward kinematics: hips and torso; two-bone leg IK that keeps the planted feet where they are, the knees
bending towards the front and the pelvis lowering a little where the legs would otherwise overstretch; arms that keep
hanging while the torso bends, always, so that waves and gestures start and end there; the wave (the Rocketbox
``wave`` clips over the waving arm when the motion library has them, mirrored for the left hand, else a procedural
wave); hand gestures; head tilts; the face. The procedural wave and the gestures reach for their pose like an arm does
(:meth:`PlayPose._reach`): the elbow and the wrist travel on curves from where the clip has them, in front of the
chest and low, never out sideways nor flipping over, about a second each way. A person holding a sign in the right
hand (``accessories``) waves and gestures with the left. The adjuster takes the move sample of the same time as the
state the character got (a :class:`~.performer.Performer` runs the behaviour on its own clock) and fades out while the
Performer does a yoga pose. With nothing to do it costs a few microseconds, while playing a few tenths of a
millisecond per person and frame.
"""

from __future__ import annotations

import bisect
import itertools
import math
import random
from dataclasses import dataclass, field, replace
from typing import NamedTuple

import numpy as np

from . import faces, yoga
from .behaviour import _EYE_HEIGHT, HEAD_YAW_LIMIT, Behaviour, fits_period, pitch_shift
from .character import NECK_SHARE, _FastPoser, _pivot_rotation
from .model import AnimationClip, CharacterModel, Layer, quat_from_matrix, quat_to_matrix

MOVES = ("wave", "faces", "side_steps", "lean_sideways", "lean_forward", "closer")
DOING = {"wave": "waves at the camera", "faces": "makes faces at the camera", "side_steps": "steps from side to side",
         "lean_sideways": "leans from side to side", "lean_forward": "leans in to the camera",
         "closer": "steps up to the camera and back"}
PLAY_SHARE = 0.33  # share of the time they watch that an ordinary person plays (more when playful, less when shy)
CLEARANCE = 0.55  # metres a move keeps people apart (it never makes two people nearer than this, or than they were)
REACH = 0.36  # metres a move keeps the elbows, the hands and the head from anyone else's body axis (or ...)
HANDS = 0.15  # ... from the elbows, hands and head of someone else's move
MARGIN = 2.0  # seconds a move keeps from the start and the end of a stretch of watching
RATE = 30.0  # rows per second of a move's channel table
FADE = 0.5  # seconds the play fades out and back in around a yoga act

CHANNELS = ("u", "v", "hips_u", "hips_v", "pitch", "roll", "head_roll", "head_yaw", "head_pitch", "left_u", "left_v",
            "left_lift", "right_u", "right_v", "right_lift", "wave", "wave_time", "gesture", "wiggle", "face0",
            "face1", "face2", "face3", "head_x", "head_y", "head_z", "sink")
(U, V, HIPS_U, HIPS_V, PITCH, ROLL, HEAD_ROLL, HEAD_YAW, HEAD_PITCH, LEFT_U, LEFT_V, LEFT_LIFT, RIGHT_U, RIGHT_V,
 RIGHT_LIFT, WAVE, WAVE_TIME, GESTURE, WIGGLE, FACE0, FACE1, FACE2, FACE3, HEAD_X, HEAD_Y, HEAD_Z,
 SINK) = range(len(CHANNELS))
# u: metres to the person's left, v: forward (towards the camera); pitch: degrees the torso bends forward, roll: the
# degrees its top tilts to the person's right; feet: metres from where the clip puts them, and how high they are
# lifted; head: deliberate tilt (to the right), turn (to the left) and nod (down) on top of looking at the camera;
# wave, gesture and face*: weights 0..1; wave_time: clip seconds (or the swing's phase), wiggle: the fingers' phase;
# head_*: where the eyes are moved by the hips and the torso (metres, character space; worked out from the others);
# sink: metres the hips go down while stepping (soft knees)
_PHASES = (WAVE_TIME, WIGGLE)  # not weights: they need not end at zero, and a fade leaves them alone
_FEET = ((LEFT_U, LEFT_V, LEFT_LIFT), (RIGHT_U, RIGHT_V, RIGHT_LIFT))
_FACES = (FACE0, FACE1, FACE2, FACE3)
PITCH_SHARES = (0.4, 0.15, 0.2, 0.25)  # how a forward bend is spread over the pelvis and the three spine bones
ROLL_SHARES = (0.15, 0.25, 0.3, 0.3)  # and a tilt to the side
GESTURES = ("ears", "mouth", "cheeks")  # thumbs at the temples wiggling the fingers; pulling the mouth; squashed cheeks
_GESTURE_FOR = {"fish": ("cheeks", "ears"), "puff": ("cheeks", "ears"), "grin": ("mouth", "ears"),
                "tongue": ("ears", "mouth")}  # which gestures go with which faces (others: the ears)
_PULLED = ("smile",)  # too mild for making faces (a smile goes with waving and comes after moves)


def _wrap(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def _ease(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


def _smooth(u: np.ndarray) -> np.ndarray:
    """Minimum-jerk easing 0..1 (zero speed and acceleration at both ends), on arrays."""
    u = np.clip(u, 0.0, 1.0)
    return u * u * u * (u * (6.0 * u - 15.0) + 10.0)


def _bend(pitch: float, roll: float) -> np.ndarray:
    """3x3 rotation that tilts the top to the right by ``roll``, then bends it forward by ``pitch`` (degrees): the
    same as ``yoga.turn(pitch, 0, roll)``."""
    cp, sp = math.cos(math.radians(pitch)), math.sin(math.radians(pitch))
    cr, sr = math.cos(math.radians(roll)), math.sin(math.radians(roll))
    return np.array([[cr, -sr, 0.0], [cp * sr, cp * cr, -sp], [sp * sr, sp * cr, cp]])


def _bends(pitch: np.ndarray, roll: np.ndarray) -> np.ndarray:
    """:func:`_bend` for arrays of angles: ``(n, 3, 3)``."""
    cp, sp, cr, sr = np.cos(np.radians(pitch)), np.sin(np.radians(pitch)), np.cos(np.radians(roll)), \
        np.sin(np.radians(roll))
    zero = np.zeros_like(cp)
    return np.stack([np.stack([cr, -sr, zero], -1), np.stack([cp * sr, cp * cr, -sp], -1),
                     np.stack([sp * sr, sp * cr, cp], -1)], -2)


_STILL = np.array([[0.0, 0.0, 0.0, 1.0]] * 2)  # two unit quaternions that do not turn
_TO_QUAT = np.zeros((9, 4))  # the 3x3 entries -> (m21 - m12, m02 - m20, m10 - m01, trace) = 4w (x, y, z), 4w^2 - 1
_TO_QUAT[[7, 2, 3, 0, 4, 8], [0, 1, 2, 3, 3, 3]] = 1.0
_TO_QUAT[[5, 6, 1], [0, 1, 2]] = -1.0
_TO_MATRIX = np.zeros((16, 9))  # products of quaternion components q_k q_l (k * 4 + l) -> the 3x3 entries less I
for _entry, _terms in enumerate([((1, 1, -2), (2, 2, -2)), ((0, 1, 2), (2, 3, -2)), ((0, 2, 2), (1, 3, 2)),
                                 ((0, 1, 2), (2, 3, 2)), ((0, 0, -2), (2, 2, -2)), ((1, 2, 2), (0, 3, -2)),
                                 ((0, 2, 2), (1, 3, -2)), ((1, 2, 2), (0, 3, 2)), ((0, 0, -2), (1, 1, -2))]):
    for _k, _l, _c in _terms:
        _TO_MATRIX[_k * 4 + _l, _entry] += _c


def _quats(m: np.ndarray) -> np.ndarray:
    """Unit quaternions ``(n, 4)`` (x, y, z, w; w >= 0) of rotation matrices ``(n, 3, 3)``, like
    ``model.quat_from_matrix`` but leaner (it runs every frame on a few dozen bones): from the trace where the
    rotation is less than about 160 degrees, by Shepperd's method (the largest component first) elsewhere."""
    q = m.reshape(len(m), 9) @ _TO_QUAT
    w4 = 1.0 + q[:, 3]
    w = 0.5 * np.sqrt(np.maximum(w4, 0.1))
    q *= (0.25 / w)[:, None]
    q[:, 3] = w
    wide = w4 <= 0.1
    if wide.any():
        q[wide] = _shepperd(m[wide])
    return q


def _shepperd(m: np.ndarray) -> np.ndarray:
    m00, m01, m02 = m[:, 0, 0], m[:, 0, 1], m[:, 0, 2]
    m10, m11, m12 = m[:, 1, 0], m[:, 1, 1], m[:, 1, 2]
    m20, m21, m22 = m[:, 2, 0], m[:, 2, 1], m[:, 2, 2]
    a, b, c = m21 - m12, m02 - m20, m10 - m01  # 4w (x, y, z)
    d, e, f = m01 + m10, m02 + m20, m12 + m21  # 4 (xy, xz, yz)
    w4, x4, y4, z4 = 1 + m00 + m11 + m22, 1 + m00 - m11 - m22, 1 - m00 + m11 - m22, 1 - m00 - m11 + m22  # 4 x^2 ...
    rows = np.stack([np.stack([a, b, c, w4], -1), np.stack([x4, d, e, a], -1), np.stack([d, y4, f, b], -1),
                     np.stack([e, f, z4, c], -1)], -2)
    squares = np.stack([w4, x4, y4, z4], -1)
    k = squares.argmax(-1)
    q = np.take_along_axis(rows, k[:, None, None], -2)[:, 0, :]
    q = q / (2.0 * np.sqrt(np.take_along_axis(squares, k[:, None], -1)))
    return q * np.where(q[:, 3:] < 0.0, -1.0, 1.0)


def _matrices(q: np.ndarray) -> np.ndarray:
    """Rotation matrices ``(n, 3, 3)`` of unit quaternions ``(n, 4)`` (x, y, z, w)."""
    m = (q[:, :, None] * q[:, None, :]).reshape(len(q), 16) @ _TO_MATRIX
    m[:, ::4] += 1.0
    return m.reshape(len(q), 3, 3)


def _nlerp(a: np.ndarray, b: np.ndarray, t) -> np.ndarray:
    """``(K, 4)`` quaternions ``t`` (a number, or one per row) of the way from ``a`` to ``b`` (the shorter way)."""
    t = np.asarray(t, float)
    if t.ndim:
        t = t[:, None]
    q = a * (1.0 - t) + b * np.where(np.einsum("ij,ij->i", a, b)[:, None] < 0.0, -t, t)
    return q / np.sqrt(np.einsum("ij,ij->i", q, q))[:, None]


def _as_random(rng) -> random.Random:
    if isinstance(rng, random.Random):
        return rng
    return random.Random(0 if rng is None else int(rng) if not hasattr(rng, "integers") else int(rng.integers(2**62)))


# -- the body ---------------------------------------------------------------------------------------------------------


def supports(model: CharacterModel) -> bool:
    """Whether ``model`` can play with the camera (the Biped skeleton of the Rocketbox people)."""
    if not yoga.supports(model):
        return False
    sk = model.skeleton
    names = ("Bip01", "Bip01 Pelvis", "Bip01 Spine", "Bip01 Spine1", "Bip01 Spine2", "Bip01 Neck", "Bip01 Head")
    bones = [sk.find(n) for n in names]
    return None not in bones and all(sk.parents[b] == a for a, b in itertools.pairwise(bones))


@dataclass
class _Arm:
    """One arm's bones and rest measurements (see :class:`Figure`)."""

    sign: float  # +1 left, -1 right (the person's left is +X)
    clavicle: int
    upper: int
    fore: int
    hand: int
    bones: np.ndarray  # clavicle, upper arm, forearm, hand and fingers: what a wave clip is blended over
    fingers: np.ndarray  # the fingers but the thumb, joint by joint
    chain: np.ndarray  # upper arm, forearm, hand and those fingers: what the arm IK sets
    curl_axes: np.ndarray  # (len(fingers), 3): each finger joint's curling axis, in its own rest frame
    lengths: tuple  # upper arm, forearm
    rest: tuple  # (upper, fore, hand): rest frames that the IK's frames map onto the bones' rotations
    curls: dict  # gesture -> how far each finger joint bends (degrees)
    wiggle: tuple  # (phase, size) of each finger joint's wiggle
    blend: np.ndarray  # the hand, the fingers and the thumb's joints: what a reach turns (PlayPose._reach)
    blend_rest: np.ndarray  # their rest rotations (local)
    thumb: np.ndarray  # (len(thumb joints), 3, 3): their local rotations when the thumbs go to the temples (the
    # thumb straight, pointing across the palm) ...
    thumb_q: np.ndarray  # ... as quaternions, from the rest pose's
    thumb_tip: np.ndarray  # where the thumb's tip then is (hand bone frame)


class Figure:
    """The bones, sizes and rest pose of one avatar (a Biped skeleton), as the moves need them; see :func:`figure`."""

    def __init__(self, model: CharacterModel):
        if not supports(model):
            raise ValueError(f"{model.name} cannot play with the camera (not a Biped skeleton)")
        body = yoga.Body(model)
        sk = model.skeleton
        self.model, self.skeleton = model, sk
        self.scale = body.height / 1.75
        self.height = body.height
        self.chain = [body.root, body.pelvis, *body.spine, body.neck, body.head]  # each the parent of the next
        rest_local = sk.rest_local
        self.legs = []
        for side in ("left", "right"):
            s = body.sides[side]
            parent = int(sk.parents[s["thigh"]])
            if parent not in self.chain:
                raise ValueError(f"{model.name}: the thighs do not hang from the spine")
            self.legs.append((self.chain.index(parent), s["thigh"], s["calf"], s["foot"],
                              float(np.linalg.norm(rest_local[s["calf"], :3, 3])),
                              float(np.linalg.norm(rest_local[s["foot"], :3, 3]))))
        self.rest_rot = body.rest_rot
        self.rest_local_rot = body.local_rot
        self.arms = {side: self._arm(body, side) for side in ("left", "right")}
        pos = body.rest_pos
        eyes = [sk.find(f"Bip01 {s}Eye") for s in "LR"]
        self.eye = pos[eyes].mean(axis=0) if None not in eyes else pos[body.head] + np.array([0.0, 0.1, 0.09]) * \
            self.scale
        self.joints = pos[self.chain[:-2] + [body.neck]]  # root, pelvis, spine, spine1, spine2, neck
        self.chest_forward = self.rest_rot[body.spine[2]].T @ np.array([0.0, 0.0, 1.0])  # in the chest bone's frame
        self.gestures = self._gestures(pos, body)
        self.reaches = self._reaches(pos, body)
        self._tracks: dict = {}
        self._waves: dict = {}  # wave_clip's

    def _reaches(self, pos, body) -> dict:
        """``{name: (2, 2, 2)}``: where the elbows and the hands get to on the floor plan (left arm, right arm; elbow,
        hand; to the left, forward, from under the hips) in each of :data:`GESTURES`, in the procedural wave
        (``"wave"``, swung out the most) and hanging down (``"rest"``). The planner keeps them clear of others."""
        head, s = pos[body.head], self.scale
        reaches = {name: np.zeros((2, 2, 2)) for name in (*GESTURES, "wave", "rest")}
        for j, side in enumerate(("left", "right")):
            arm = self.arms[side]
            shoulder, (l1, l2), sigma = pos[arm.upper].tolist(), arm.lengths, arm.sign
            for gesture in GESTURES:
                at, way = self.gestures[gesture, side]
                elbow, wrist, _ = _two_bone(shoulder, l1, l2, (head + at[0]).tolist(), way[0].tolist())
                out = sigma * 0.08 * s  # with room for the head's tilts and turns, which take the hands along
                reaches[gesture][j] = (elbow[0] + out, elbow[2]), (wrist[0] + out, wrist[2])
            out = math.radians(45.0)  # the widest swing
            elbow = _plus(shoulder, _unit((sigma * 0.72, -0.3, 0.62)), l1)
            wrist = _plus(elbow, _unit((sigma * math.sin(out), math.cos(out), 0.18)), l2)
            reaches["wave"][j] = (elbow[0], elbow[2]), (wrist[0], wrist[2])
            reaches["rest"][j] = (shoulder[0] + sigma * 0.03 * s, -0.04 * s), (shoulder[0] + sigma * 0.04 * s, 0.0)
        return reaches

    def _arm(self, body, side: str) -> _Arm:
        s = body.sides[side]
        pos, rot, sk = body.rest_pos, body.rest_rot, body.skeleton
        fingers = [b for chain in s["fingers"][1:] for b in chain]
        across = yoga._unit(np.cross(s["hand_back"], s["hand_forward"]))  # curling towards the palm
        axes = np.array([body.local_rot[b].T @ rot[sk.parents[b]].T @ across for b in fingers]) if fingers else \
            np.zeros((0, 3))
        every = [b for chain in s["fingers"] for b in chain]
        tips = [c for chain in s["fingers"] if chain for c in body.finger_tips.get(chain[-1], ())]
        bones = np.array([s["clavicle"], s["upper"], s["fore"], s["hand"], *every, *tips], dtype=int)
        rest = (yoga._frame(pos[s["fore"]] - pos[s["upper"]], s["arm_bend"]).T @ rot[s["upper"]],
                yoga._frame(pos[s["hand"]] - pos[s["fore"]], s["arm_bend"]).T @ rot[s["fore"]],
                yoga._frame(s["hand_forward"], s["hand_back"]).T @ rot[s["hand"]])
        fingers = np.array(fingers, dtype=int)
        chain = np.concatenate([[s["upper"], s["fore"], s["hand"]], fingers]).astype(int)
        thumb = np.array(s["fingers"][0], dtype=int)
        hand = rot[s["hand"]].T  # character space -> the hand bone's frame
        end = thumb[-1] if len(thumb) else s["hand"]  # the tip: past the last joint by about the one before's length
        tip = hand @ (pos[end] + 0.9 * (pos[end] - pos[sk.parents[end]]) - pos[s["hand"]])
        base = hand @ (pos[thumb[0]] - pos[s["hand"]]) if len(thumb) else tip
        radial = yoga._unit(pos[s["fingers"][1][0]] - pos[s["fingers"][4][0]])  # from the little finger to the index
        out = hand @ yoga._unit(radial + 0.35 * s["hand_forward"] - 0.2 * s["hand_back"])
        turn = yoga._align(tip - base, out)  # the thumb as at rest, turned at its base to point across the palm
        poses = body.local_rot[thumb].copy()
        if len(thumb):
            poses[0] = turn @ poses[0]
        k = np.arange(len(fingers))  # finger joints: index, middle, ring and little finger, three each
        blend = np.array([s["hand"], *fingers, *thumb], dtype=int)
        curls = {"ears": np.zeros(len(k)), "mouth": np.where(k < 3, 15.0, 75.0), "cheeks": np.full(len(k), 10.0)}
        return _Arm(s["sign"], s["clavicle"], s["upper"], s["fore"], s["hand"], bones, fingers, chain, axes,
                    (s["upper_length"], s["fore_length"]), rest, curls, (0.9 * (k // 3), 0.6 + 0.2 * (k % 3)),
                    blend, body.local_rot[blend], poses,
                    _quats(np.swapaxes(body.local_rot[thumb], 1, 2) @ poses) if len(thumb) else np.zeros((0, 4)),
                    base + turn @ (tip - base))

    def _gestures(self, pos, body) -> dict:
        """``{(gesture, side): (at, way)}``: where the hands go in each of :data:`GESTURES`. ``at`` rows (the head's
        rest frame, from the head joint): the point the wrist goes to (for "ears": the thumb's tip), the way the
        fingers point and the way the back of the hand faces; ``way`` rows (the chest's rest frame): the way the
        elbow points, and how the arm gets there (``_Reach.rise``, ``arrive``, ``low``)."""
        sk, head, s = self.skeleton, pos[body.head], self.scale

        def bone(name, default):
            b = sk.find(name)
            return pos[b] - head if b is not None else np.asarray(default, float) * s

        gestures = {}
        for side, sign, letter in (("left", 1.0, "L"), ("right", -1.0, "R")):
            eye = bone(f"Bip01 {letter}Eye", (sign * 0.032, 0.1, 0.086))
            temple = eye + np.array([sign * 0.052, 0.012, -0.045]) * s  # the thumb's pad on the temple
            mouth = bone(f"Bip01 {letter}MouthCorner", (sign * 0.028, 0.024, 0.112)) + \
                np.array([sign * 0.095, -0.085, -0.01]) * s  # the wrist, the index finger hooked in the corner
            cheek = bone(f"Bip01 {letter}Cheek", (sign * 0.041, 0.069, 0.108)) + \
                np.array([sign * 0.07, -0.13, -0.02]) * s  # the wrist, the palm on the cheek
            way = [(0.0, 0.15 * s, 0.25 * s), (-sign * 0.04 * s, 0.0, 0.22 * s), (sign * 0.28, -0.94, -0.18)]
            gestures["ears", side] = (np.array([temple, (sign * 0.35, 1.0, 0.12), (0.0, 0.0, -1.0)]),
                                      np.array([(sign * 1.0, -0.2, -0.3), *way]))  # palms forward, elbows out
            gestures["mouth", side] = (np.array([mouth, (-sign * 0.6, 0.75, 0.2), (sign * 0.8, -0.3, 0.45)]),
                                       np.array([(sign * 0.55, -1.0, -0.1), *way]))
            gestures["cheeks", side] = (np.array([cheek, (-sign * 0.15, 1.0, 0.05), (sign * 1.0, 0.0, 0.0)]),
                                        np.array([(sign * 0.5, -1.0, -0.15), *way]))
        return gestures

    def arm_track(self, clip: AnimationClip, side: str, mirror: bool) -> tuple[np.ndarray, np.ndarray]:
        """``(bones, (F, K, 4) local rotations)`` of the ``side`` arm in ``clip``; ``mirror``: the other arm's motion,
        mirrored onto this one (a right-handed wave done with the left hand)."""
        key = (id(clip), side, mirror)
        cached = self._tracks.get(key)
        if cached is not None and cached[0] is clip:
            return cached[1]
        names, index = self.skeleton.names, {n: k for k, n in enumerate(clip.bone_names)}
        known = {n: b for b, n in enumerate(names)}
        letters = ("L", "R") if side == "left" else ("R", "L")

        def source(name):
            return name.replace(f" {letters[0]} ", f" {letters[1]} ") if mirror else name

        bones = [b for b in self.arms[side].bones if source(names[b]) in index]
        columns = [index[source(names[b])] for b in bones]
        q = np.asarray(clip.rotations, float)[:, columns]
        if mirror:  # L' = K_p^T (M L M) K_b, K_b = (M R_m(b) M)^T R_b with R the rest rotations, M the mirror in x
            m = np.diag([-1.0, 1.0, 1.0])
            rest = self.rest_rot

            def k(b):
                return (m @ rest[known.get(source(names[b]), b)] @ m).T @ rest[b]

            local = quat_to_matrix(q)
            kb = np.array([k(b) for b in bones])
            kp = np.array([k(int(self.skeleton.parents[b])) for b in bones])
            q = quat_from_matrix(np.einsum("kji,fkjl->fkil", kp, m @ local @ m) @ kb)
        track = (np.array(bones, dtype=int), q)
        self._tracks[key] = (clip, track)
        return track


_figures: dict[int, tuple[CharacterModel, Figure]] = {}


def figure(model: CharacterModel) -> Figure:
    """The :class:`Figure` of ``model`` (cached per model)."""
    cached = _figures.get(id(model))
    if cached is None or cached[0] is not model:
        cached = _figures[id(model)] = (model, Figure(model))
    return cached[1]


class _Reach(NamedTuple):
    """What one arm reaches for (see :meth:`PlayPose._reach`; points and directions in character space): the wrist
    at ``wrist``, the elbow out towards ``pole``, the fingers along ``fingers`` with the back of the hand towards
    ``back``, each finger joint bent ``curl`` degrees from the rest pose (an array); the way there (``rise``,
    ``arrive``, ``low``); ``thumb``: the thumb as when the thumbs go to the temples (else the clip's)."""

    arm: _Arm
    wrist: tuple
    pole: tuple
    fingers: tuple
    back: tuple
    curl: np.ndarray
    rise: tuple = (0.0, 0.0, 0.0)
    arrive: tuple = (0.0, 0.0, 0.0)
    low: tuple = (0.0, -1.0, 0.0)
    thumb: bool = False


@dataclass(frozen=True)
class WaveClip:
    """A wave clip and what it does: ``both`` arms or the right one, waving until ``span`` seconds into it; ``arms``:
    where it puts the elbows and the hands every ``WAVE_STEP`` seconds (``(n, 2, 2, 2)``, as ``Figure.reaches``)."""

    clip: AnimationClip
    both: bool
    span: float
    arms: np.ndarray = field(default_factory=lambda: np.zeros((1, 2, 2, 2)), compare=False, repr=False)


WAVE_STEP = 0.1  # seconds between the samples of a wave clip (WaveClip.arms)


def wave_clip(clip: AnimationClip, fig: Figure) -> WaveClip | None:
    """Which arms ``clip`` raises above the shoulders and until when (cached per clip and figure); ``None`` if it
    raises none."""
    cached = fig._waves.get(id(clip))
    if cached is not None and cached[0] is clip:
        return cached[1]
    poser = _FastPoser(fig.model)
    times = np.arange(0.0, clip.duration, WAVE_STEP)
    up = np.zeros((len(times), 2), bool)
    arms = np.zeros((len(times), 2, 2, 2))
    root = fig.chain[0]
    for k, t in enumerate(times):
        pose = poser.global_pose([Layer(clip, float(t))])
        for j, side in enumerate(("left", "right")):
            arm = fig.arms[side]
            up[k, j] = pose[arm.hand, 1, 3] > pose[arm.upper, 1, 3] + 0.02 * fig.scale
            arms[k, j] = pose[[arm.fore, arm.hand]][:, [0, 2], 3] - pose[root, [0, 2], 3]  # the elbow, the hand
    info = None
    if up.any():
        raised = up.any(axis=0)
        last = float(times[np.flatnonzero(up.any(axis=1))[-1]])
        info = WaveClip(clip, bool(raised.all()), min(clip.duration, last + 0.8), arms)
        if not info.both and not raised[1]:  # waves with the left hand only: not what the moves expect
            info = None
    fig._waves[id(clip)] = (clip, info)
    return info


# -- moves ------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Wave:
    """How a wave move waves: with ``clip`` (``None``: the procedural wave) and which hands (``"right"``, ``"left"``
    or ``"both"``)."""

    clip: AnimationClip | None
    hands: str
    amplitude: float = 28.0  # degrees the forearm swings each way (the procedural wave)


class Move:
    """One move of one person: its ``kind`` (one of :data:`MOVES`), when it starts (scene time, within the period),
    the ``heading`` the person has then, and its channel ``table`` (``RATE`` rows a second, see :data:`CHANNELS`).

    ``faces`` names the expression of each face channel in use, ``gesture`` is the hand gesture made with them (one of
    :data:`GESTURES`) and ``wave`` how the arms wave, if they do.
    """

    __slots__ = ("faces", "gesture", "heading", "kind", "points", "start", "table", "travels", "wave")

    def __init__(self, kind: str, table: np.ndarray, faces=(), gesture: str | None = None, wave: Wave | None = None):
        self.kind, self.table = kind, table
        self.faces, self.gesture, self.wave = tuple(faces), gesture, wave
        self.start, self.heading, self.points = 0.0, 0.0, None  # points: reach_points' (cached)
        self.travels = bool(np.abs(table[:, [U, V]]).max() > 1e-9)  # the person leaves their spot

    @property
    def duration(self) -> float:
        return (len(self.table) - 1) / RATE

    @property
    def end(self) -> float:
        return self.start + self.duration

    def values(self, u: float) -> list[float]:
        """The channels ``u`` seconds into the move."""
        f = u * RATE
        i = int(f)
        last = len(self.table) - 1
        if i >= last or i < 0:
            return self.table[last if i >= last else 0].tolist()
        row = self.table[i]
        return (row + (self.table[i + 1] - row) * (f - i)).tolist()

    def offsets(self, u: np.ndarray) -> np.ndarray:
        """``(n, 2)`` body offsets (to the left, forward) ``u`` seconds into the move."""
        grid = np.arange(len(self.table)) / RATE
        return np.stack([np.interp(u, grid, self.table[:, U]), np.interp(u, grid, self.table[:, V])], axis=1)

    def __repr__(self) -> str:
        return f"Move({self.kind!r}, {self.start:.1f}-{self.end:.1f} s)"


class _Script:
    """A move's channels while it is being made up: eased changes, foot lifts and linearly running phases."""

    def __init__(self):
        self.changes: list[tuple[int, float, float, float]] = []  # (channel, t0, t1, change)
        self.bumps: list[tuple[int, float, float, float]] = []  # (channel, t0, t1, height): up and down again
        self.runs: list[tuple[int, float, float, float]] = []  # (channel, t0, t1, rate): rising from t0 to t1
        self.value = [0.0] * len(CHANNELS)  # where each channel ends up
        self.faces: list[str] = []
        self.end = 0.0

    def go(self, channel: int, t0: float, t1: float, value: float) -> float:
        """Ease ``channel`` from where it is to ``value`` between ``t0`` and ``t1``; returns ``t1``."""
        t1 = max(t1, t0 + 1e-3)
        if value != self.value[channel]:
            self.changes.append((channel, t0, t1, value - self.value[channel]))
            self.value[channel] = value
        self.end = max(self.end, t1)
        return t1

    def shift(self, channel: int, t0: float, t1: float, change: float) -> float:
        return self.go(channel, t0, t1, self.value[channel] + change)

    def bump(self, channel: int, t0: float, t1: float, height: float) -> None:
        self.bumps.append((channel, t0, t1, height))
        self.end = max(self.end, t1)

    def run(self, channel: int, t0: float, t1: float, rate: float) -> None:
        """Let phase ``channel`` rise at ``rate`` per second from ``t0`` until ``t1`` (past the move's end is fine)."""
        self.runs.append((channel, t0, t1, rate))

    def face(self, name: str) -> int:
        """The face channel of expression ``name`` (one of four)."""
        if name not in self.faces:
            if len(self.faces) == len(_FACES):
                raise ValueError("a move makes at most four different faces")
            self.faces.append(name)
        return _FACES[self.faces.index(name)]

    def table(self, fig: Figure, rest: float = 0.25) -> np.ndarray:
        """The channel table, ``rest`` seconds past the last change (the head channels worked out for ``fig``)."""
        if any(abs(v) > 1e-9 for k, v in enumerate(self.value) if k not in _PHASES):
            raise ValueError("a move must end where it started")
        n = math.ceil((self.end + rest) * RATE) + 1
        grid = np.arange(n) / RATE
        table = np.zeros((n, len(CHANNELS)))
        for channel, t0, t1, change in self.changes:
            if t0 < 0.0:
                raise ValueError("a move must start where the person stands")
            table[:, channel] += change * _smooth((grid - t0) / (t1 - t0))
        for channel, t0, t1, height in self.bumps:
            table[:, channel] += height * np.sin(np.pi * np.clip((grid - t0) / (t1 - t0), 0.0, 1.0)) ** 2
        for channel, t0, t1, rate in self.runs:
            table[:, channel] += rate * np.clip(grid - t0, 0.0, t1 - t0)
        table[:, [HEAD_X, HEAD_Y, HEAD_Z]] = _eye_shift(fig, table)
        table[np.abs(table) < 1e-12] = 0.0
        return table


def _eye_shift(fig: Figure, table: np.ndarray) -> np.ndarray:
    """``(n, 3)`` how far the hips and the bent torso move the eyes (character space), per row of ``table``: each bend
    (pelvis, spine, spine1, spine2, see ``PITCH_SHARES``) swings everything above its joint."""
    pivots = fig.joints[1:5]
    ends = np.vstack([pivots[1:], fig.eye[None]])
    turned = np.repeat(np.eye(3)[None], len(table), axis=0)
    eye = np.repeat(pivots[0][None], len(table), axis=0)
    for ps, rs, pivot, end in zip(PITCH_SHARES, ROLL_SHARES, pivots, ends):
        turned = turned @ _bends(table[:, PITCH] * ps, table[:, ROLL] * rs)
        eye = eye + turned @ (end - pivot)
    shift = eye - fig.eye
    shift[:, 0] += table[:, HIPS_U]
    shift[:, 1] -= table[:, SINK]
    shift[:, 2] += table[:, HIPS_V]
    return shift


# -- making up moves --------------------------------------------------------------------------------------------------


@dataclass
class _Person:
    """What the moves of one person are made from: their body, how playful they are, their wave clips."""

    fig: Figure
    playful: float  # 0.3 (shy) .. 1.5 (a playful child)
    amp: float  # how big their moves are
    pace: float  # how quick
    waves: list  # WaveClip of their sex
    expressions: list  # faces they make


def _step(script: _Script, foot: int, t0: float, swing: float, du: float, dv: float, lift: float) -> float:
    """Foot ``foot`` (0 left, 1 right) steps by ``(du, dv)``, lifted ``lift`` metres at the middle; returns when it
    lands."""
    cu, cv, cl = _FEET[foot]
    if du:
        script.shift(cu, t0, t0 + swing, du)
    if dv:
        script.shift(cv, t0, t0 + swing, dv)
    script.bump(cl, t0, t0 + swing, lift)
    return t0 + swing


def _soft_knees(script: _Script, person: _Person, t0: float, t1: float) -> None:
    """Bend the knees a little (the hips going down) from just before ``t0`` until just after ``t1``, as people do who
    step about: so a foot lifts off and lands without the knee snapping out of a straight leg."""
    script.go(SINK, max(0.0, t0 - 0.4), t0, 0.018 * person.fig.scale)
    script.go(SINK, t1 - 0.1, t1 + 0.25, 0.0)


def _traverse(script: _Script, rng: random.Random, person: _Person, t: float, delta: float) -> float:
    """Side steps moving the person ``delta`` metres to their left (right if negative): the leading foot steps out,
    the other one closes up, the body shifting its weight onto the standing foot first. Returns when it is done."""
    s, pace, start = person.fig.scale, person.pace, t
    pairs = max(1, math.ceil(abs(delta) / (0.34 * s)))
    part, sign = delta / pairs, 1.0 if delta > 0 else -1.0
    lead, trail = (0, 1) if delta > 0 else (1, 0)
    for _ in range(pairs):
        swing, lift = rng.uniform(0.42, 0.55) / pace, rng.uniform(0.035, 0.055) * s
        unload = 0.018 * s
        script.shift(U, t - 0.18, t + 0.08, -sign * unload)  # onto the trailing foot
        script.shift(ROLL, t - 0.18, t + 0.1, sign * 2.0)
        _step(script, lead, t, swing, part, 0.0, lift)
        script.shift(U, t + 0.08, t + swing + 0.1, 0.5 * part + sign * unload)
        script.shift(ROLL, t + 0.15, t + swing, -sign * 2.0)
        t2 = t + swing + rng.uniform(0.04, 0.14) / pace
        swing2 = swing * rng.uniform(0.9, 1.05)
        _step(script, trail, t2, swing2, part, 0.0, lift * 0.8)
        script.shift(U, t2 - 0.05, t2 + swing2 + 0.15, 0.5 * part)
        t = t2 + swing2 + rng.uniform(0.08, 0.2) / pace
    _soft_knees(script, person, start - 0.18, t)
    return t + 0.1


def _walk(script: _Script, rng: random.Random, person: _Person, t: float, distance: float, steps: int, lead: int,
          swing: tuple) -> float:
    """``steps`` steps straight forward by ``distance`` metres (back if negative), feet together at the end."""
    s, pace, start = person.fig.scale, person.pace, t
    places = [distance * k / (steps - 1) for k in range(1, steps)] + [distance]
    feet, body = [0.0, 0.0], 0.0
    for k, place in enumerate(places):
        foot = lead if k % 2 == 0 else 1 - lead
        stride = abs(place - feet[foot])  # longer steps take a little longer
        time, lift = (rng.uniform(*swing) + 0.35 * stride / s) / pace, rng.uniform(0.03, 0.05) * s
        _step(script, foot, t, time, 0.0, place - feet[foot], lift)
        feet[foot] = place
        script.shift(V, t - 0.1, t + time + 0.12, 0.5 * (feet[0] + feet[1]) - body)
        body = 0.5 * (feet[0] + feet[1])
        t += time + rng.uniform(0.02, 0.1) / pace
    _soft_knees(script, person, start - 0.1, t)
    return t + 0.1


def _smile(script: _Script, person: _Person, t0: float, t1: float, strength: float) -> None:
    if "smile" in person.expressions and t1 > t0 + 0.6:
        channel = script.face("smile")
        script.go(channel, t0, t0 + 0.4, strength)
        script.go(channel, t1 - 0.4, t1, 0.0)


def _make_faces(script: _Script, rng: random.Random, person: _Person, t: float, count: int,
                gesture: float, hands: str | None = None) -> tuple[float, str | None]:
    """Make ``count`` faces in a row from ``t`` with a hand gesture (``hands``, or one that goes with the first face
    with probability ``gesture``), the head tilting and wobbling; returns when the face is back to normal, and the
    gesture."""
    names = [n for n in person.expressions if n not in _PULLED] or list(person.expressions)
    if not names:
        return t, None
    chosen = rng.sample(names, min(count, len(names), len(_FACES)))
    if hands is None and rng.random() < gesture:
        hands = rng.choice(_GESTURE_FOR.get(chosen[0], ("ears",)))
    if hands is not None:  # the hands come up to the face (about 0.8 m in about a second) as the face starts
        script.go(GESTURE, t, t + rng.uniform(0.8, 1.0) / person.pace, 1.0)
        script.run(WIGGLE, t, t + 30.0, 2 * math.pi * rng.uniform(2.5, 3.5))
        t += 0.45
    amp = person.amp
    for name in chosen:
        channel, ramp = script.face(name), rng.uniform(0.22, 0.38)
        script.go(channel, t, t + ramp, rng.uniform(0.8, 1.0))
        script.go(HEAD_ROLL, t, t + 0.4, rng.choice((-1.0, 1.0)) * rng.uniform(4.0, 13.0) * amp)
        script.go(HEAD_PITCH, t, t + 0.4, rng.uniform(-6.0, 4.0))
        hold = rng.uniform(0.8, 1.7) / person.pace
        if name in ("tongue", "puff", "fish") or rng.random() < 0.35:  # waggle the head
            swing, k, a = rng.uniform(0.22, 0.32), t + ramp, rng.uniform(4.0, 8.0) * amp
            while k + 2 * swing < t + ramp + hold:
                script.go(HEAD_YAW, k, k + swing, a)
                a = -a
                k += swing
            script.go(HEAD_YAW, k, k + swing, 0.0)
        end = t + ramp + hold
        script.go(channel, end, end + ramp, 0.0)
        t = end + 0.5 * ramp
    script.go(HEAD_ROLL, t, t + 0.45, 0.0)
    script.go(HEAD_PITCH, t, t + 0.45, 0.0)
    script.go(HEAD_YAW, t, t + 0.45, 0.0)
    if hands is not None:
        script.go(GESTURE, t - 0.1, t - 0.1 + rng.uniform(0.75, 0.95) / person.pace, 0.0)
    return t + 0.5, hands


def _wave(rng: random.Random, person: _Person, hands: str | None = None, **_) -> Move:
    script = _Script()
    t = rng.uniform(0.1, 0.25)
    one = [w for w in person.waves if not w.both]
    both = [w for w in person.waves if w.both] if hands is None else []  # hands: "right" or "left" (one hand)
    if one or both:
        info = rng.choice(both) if both and (not one or rng.random() < 0.15 + 0.25 * person.playful) else \
            rng.choice(one or both)
        wave = Wave(info.clip, "both" if info.both else hands or ("right" if rng.random() < 0.75 else "left"))
        speed = rng.uniform(0.88, 1.08)  # no two waves alike
        end = info.span / speed
        script.go(WAVE, 0.0, 0.35, 1.0)
        script.run(WAVE_TIME, 0.0, end, speed)
        script.go(WAVE, end - 0.55, end, 0.0)
    else:
        lift = rng.uniform(0.65, 0.85) / person.pace
        swings, rate = rng.randint(3, 7), rng.uniform(2.0, 2.8)  # swings to and fro, per second
        wave = Wave(None, hands or ("right" if rng.random() < 0.75 else "left"),
                    rng.uniform(22.0, 32.0) * min(person.amp, 1.1))
        script.go(WAVE, t, t + lift, 1.0)
        script.run(WAVE_TIME, t + 0.6 * lift, t + 30.0, 2 * math.pi * rate)
        end = t + lift + swings / rate
        script.go(WAVE, end, end + rng.uniform(0.6, 0.8) / person.pace, 0.0)
        end = script.end
    _smile(script, person, rng.uniform(0.1, 0.4), end, rng.uniform(0.5, 0.8))
    if rng.random() < 0.5:
        tilt = rng.choice((-1.0, 1.0)) * rng.uniform(3.0, 7.0)
        script.go(HEAD_ROLL, 0.3, 0.9, tilt)
        script.go(HEAD_ROLL, max(1.0, end - 0.8), max(1.6, end - 0.1), 0.0)
    return Move("wave", script.table(person.fig), script.faces, wave=wave)


def _faces(rng: random.Random, person: _Person, gesture: str | None = None, arms: bool = True, **_) -> Move:
    script = _Script()
    t = rng.uniform(0.15, 0.3)
    lean = rng.uniform(3.0, 9.0) * person.amp if rng.random() < 0.5 else 0.0
    if lean:
        script.go(PITCH, t, t + 0.7, lean)
        script.go(HIPS_V, t, t + 0.7, -0.15 * person.fig.scale * math.sin(math.radians(lean)))
    count = 1 + (rng.random() < 0.35 + 0.25 * person.playful) + (rng.random() < 0.1 + 0.15 * person.playful)
    end, gesture = _make_faces(script, rng, person, t + 0.3, count, (0.3 + 0.25 * person.playful) * arms,
                               gesture if arms else None)
    if lean:
        script.go(PITCH, end - 0.2, end + 0.6, 0.0)
        script.go(HIPS_V, end - 0.2, end + 0.6, 0.0)
    return Move("faces", script.table(person.fig), script.faces, gesture=gesture)


def _side_steps(rng: random.Random, person: _Person, first: float | None = None, size: float = 1.0) -> Move:
    script, s, amp = _Script(), person.fig.scale, min(person.amp, 1.15) * size
    first = first or rng.choice((-1.0, 1.0))
    out, back = rng.uniform(0.25, 0.4) * s * amp, rng.uniform(0.22, 0.36) * s * amp
    t = _traverse(script, rng, person, rng.uniform(0.6, 0.75), first * out)
    look = rng.uniform(0.6, 1.4) / person.pace
    if rng.random() < 0.4:  # a tilt of the head at the picture
        tilt = rng.choice((-1.0, 1.0)) * rng.uniform(4.0, 9.0)
        script.go(HEAD_ROLL, t, t + 0.4, tilt)
        script.go(HEAD_ROLL, t + look - 0.2, t + look + 0.3, 0.0)
    t = _traverse(script, rng, person, t + look, -first * (out + back))
    t = _traverse(script, rng, person, t + rng.uniform(0.5, 1.3) / person.pace, first * back)
    if rng.random() < 0.5:
        _smile(script, person, t - 0.6, t + 1.2, rng.uniform(0.4, 0.7))
    return Move("side_steps", script.table(person.fig), script.faces)


def _lean_sideways(rng: random.Random, person: _Person, **_) -> Move:
    script, s, pace = _Script(), person.fig.scale, person.pace
    side = rng.choice((-1.0, 1.0))  # +1: the top to the right first
    t = rng.uniform(0.15, 0.3)
    swings = 2 + (rng.random() < 0.25 + 0.2 * person.playful)
    for k in range(swings + 1):
        last = k == swings
        angle = 0.0 if last else side * rng.uniform(12.0, 20.0) * min(person.amp, 1.2) * (0.8 if k >= 2 else 1.0)
        time = rng.uniform(0.7, 1.0) / pace if k == 0 or last else rng.uniform(1.0, 1.4) / pace
        script.go(ROLL, t, t + time, angle)
        script.go(HIPS_U, t, t + time, 0.004 * s * angle)  # the hips the other way (to the left for a right tilt)
        script.go(HEAD_ROLL, t, t + time, -0.25 * angle)  # the head a little more upright than the torso
        t += time + (0.0 if last else rng.uniform(0.35, 1.0) / pace)
        side = -side
    if rng.random() < 0.4:
        _smile(script, person, t - 0.8, t + 0.8, rng.uniform(0.4, 0.7))
    return Move("lean_sideways", script.table(person.fig), script.faces)


def _lean_forward(rng: random.Random, person: _Person, arms: bool = True, **_) -> Move:
    script, s, pace = _Script(), person.fig.scale, person.pace
    angle = min(32.0, max(14.0, rng.uniform(18.0, 30.0) * person.amp))
    back = 0.2 * s * math.sin(math.radians(angle))
    t = rng.uniform(0.15, 0.3)
    down = rng.uniform(0.8, 1.2) / pace
    script.go(PITCH, t, t + down, angle)
    script.go(HIPS_V, t, t + down, -back)
    t += down
    end = t + rng.uniform(0.9, 2.2) / pace
    gesture = None
    if rng.random() < 0.25 + 0.3 * person.playful:  # a face at the camera from close up
        done, gesture = _make_faces(script, rng, person, t + 0.15, 1 + (rng.random() < 0.3),
                                    0.25 * person.playful * arms)
        end = max(end, done + 0.1)
    elif rng.random() < 0.5:  # a closer look: a little further in and back
        script.shift(PITCH, t + 0.3, t + 0.8, 5.0)
        script.shift(PITCH, end - 0.6, end - 0.1, -5.0)
        tilt = rng.choice((-1.0, 1.0)) * rng.uniform(4.0, 10.0)
        script.go(HEAD_ROLL, t + 0.2, t + 0.7, tilt)
        script.go(HEAD_ROLL, end - 0.5, end, 0.0)
    up = rng.uniform(0.8, 1.1) / pace
    script.go(PITCH, end, end + up, 0.0)
    script.go(HIPS_V, end, end + up, 0.0)
    if rng.random() < 0.4:
        _smile(script, person, end + 0.3 * up, end + up + 1.0, rng.uniform(0.4, 0.7))
    return Move("lean_forward", script.table(person.fig), script.faces, gesture=gesture)


def _closer(rng: random.Random, person: _Person, size: float = 1.0, arms: bool = True, **_) -> Move:
    script, s, pace = _Script(), person.fig.scale, person.pace
    distance = min(0.9 * s, max(0.4 * s * min(size, 1.0), rng.uniform(0.5, 0.85) * s * person.amp * size))
    steps = 3 if distance > 0.55 * s else 2
    t = _walk(script, rng, person, rng.uniform(0.5, 0.65), distance, steps, rng.randrange(2), (0.5, 0.62))
    end = t + rng.uniform(1.2, 2.4) / pace  # a pause, peering at the picture
    lean = rng.uniform(6.0, 13.0) * min(person.amp, 1.2) if rng.random() < 0.7 else 0.0
    if lean:
        script.go(PITCH, t - 0.2, t + 0.5, lean)
        script.go(HIPS_V, t - 0.2, t + 0.5, -0.15 * s * math.sin(math.radians(lean)))
    gesture = None
    if rng.random() < 0.3 + 0.3 * person.playful:  # at the near end: a face
        done, gesture = _make_faces(script, rng, person, t + 0.4, 1 + (rng.random() < 0.35),
                                    (0.2 + 0.2 * person.playful) * arms)
        end = max(end, done + 0.1)
    if lean:
        script.go(PITCH, end - 0.3, end + 0.3, 0.0)
        script.go(HIPS_V, end - 0.3, end + 0.3, 0.0)
    back = steps + (rng.random() < 0.6)  # smaller steps backwards
    t = _walk(script, rng, person, end + 0.2, -distance, back, rng.randrange(2), (0.55, 0.7))
    return Move("closer", script.table(person.fig), script.faces, gesture=gesture)


_MAKERS = {"wave": _wave, "faces": _faces, "side_steps": _side_steps, "lean_sideways": _lean_sideways,
           "lean_forward": _lean_forward, "closer": _closer}


def make_move(kind: str, model_or_figure, rng=0, *, playful: float = 1.0, library=None, sex: str = "male",
              **options) -> Move:
    """One move of ``kind`` for the avatar ``model_or_figure`` (a model or its :class:`Figure`), starting at 0.

    ``playful``: 0.3 (shy, small) .. 1.5 (playful, big); ``library``: a MotionLibrary with ``wave`` clips (else the
    procedural wave). ``options``: ``first`` (+1 / -1, the side the side steps go to first), ``size`` (a factor on
    the distance of the side steps and of going closer), ``gesture`` (the hand gesture of making faces, one of
    :data:`GESTURES`), ``arms=False`` (faces without a hand gesture), ``hands`` (``"right"`` or ``"left"``: a wave
    with that hand only).
    """
    fig = model_or_figure if isinstance(model_or_figure, Figure) else figure(model_or_figure)
    rng = _as_random(rng)
    return _MAKERS[kind](rng, _person_for(fig, rng, playful, library, sex), **options)


def _person_for(fig: Figure, rng: random.Random, playful: float, library, sex: str) -> _Person:
    waves = []
    if library is not None and hasattr(library, "choices"):
        waves = [w for w in (wave_clip(c, fig) for c in library.choices("wave", sex)) if w is not None]
    return _Person(fig, playful, min(1.25, 0.55 + 0.4 * playful), rng.uniform(0.9, 1.12) * (1.08 if playful > 1.2
                   else 1.0), waves, list(faces.FAR_FACES))  # the faces that read from the camera's distance


# -- planning ---------------------------------------------------------------------------------------------------------


def watch_windows(behaviour, period: float, step: float = 0.1, margin: float = MARGIN) -> list[tuple[float, float]]:
    """Scene-time stretches within ``[0, period)`` during which ``behaviour`` watches (``watching``), ``margin``
    seconds in from both ends (a stretch that runs on round the period is cut there, half a second in)."""
    times = np.arange(0.0, period, step)
    watching = np.array([bool(behaviour.watching(float(t))) for t in times])
    padded = np.concatenate([[False], watching, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    windows = []
    for a, b in zip(edges[::2], edges[1::2]):
        start = float(times[a]) + (margin if a > 0 else 0.5)
        end = float(times[b - 1]) + step - (margin if b < len(times) else 0.5)
        if end - start > 2.0:
            windows.append((start, end))
    return windows


def _track(behaviour, times) -> np.ndarray:
    if hasattr(behaviour, "positions"):
        return np.asarray(behaviour.positions(times), float)
    track = np.full((len(times), 2), np.nan)
    for k, t in enumerate(times):
        state = behaviour(float(t))
        if state.visible:
            track[k] = state.position[0], state.position[2]
    return track


def _turned(move: Move, offsets: np.ndarray) -> np.ndarray:
    """Floor offsets ``(..., 2)`` to the person's left and forward as world ``(x, z)`` offsets (``move.heading``)."""
    h = math.radians(move.heading)
    c, s = math.cos(h), math.sin(h)
    return np.stack([offsets[..., 0] * c + offsets[..., 1] * s, -offsets[..., 0] * s + offsets[..., 1] * c], axis=-1)


def _world(move: Move, u: np.ndarray) -> np.ndarray:
    """``(n, 2)`` world-floor offsets ``(x, z)`` of ``move`` ``u`` seconds into it."""
    return _turned(move, move.offsets(u))


def reach_points(move: Move, fig: Figure) -> np.ndarray:
    """``(n, 5, 2)``: where ``move`` puts the elbows and the hands (the left arm's, then the right one's) and the head
    on the floor plan at each row of its table, to the person's left and forward from their spot. The planner keeps
    them clear of other people (``REACH``). The arms hang (``Figure.reaches``) and go with the body and the bent
    torso; a wave or a gesture takes them out towards its pose (the wave clip's, sampled) by its weight."""
    if move.points is not None:
        return move.points
    table, rest = move.table, fig.reaches["rest"]
    arms = np.repeat(rest[None], len(table), axis=0)
    if move.gesture is not None:
        arms += np.clip(table[:, GESTURE], 0.0, 1.0)[:, None, None, None] * (fig.reaches[move.gesture] - rest)
    wave = move.wave
    if wave is not None:
        info = wave_clip(wave.clip, fig) if wave.clip is not None else None
        if info is not None:
            k = np.clip(np.rint(table[:, WAVE_TIME] / WAVE_STEP).astype(int), 0, len(info.arms) - 1)
            target = info.arms[k]
            if wave.hands == "left":  # the right arm's wave, mirrored
                target = target[:, ::-1] * (-1.0, 1.0)
        else:
            target = np.repeat(fig.reaches["wave"][None], len(table), axis=0)
        sides = {"left": [0], "right": [1]}.get(wave.hands, [0, 1])
        weight = np.clip(table[:, WAVE], 0.0, 1.0)[:, None, None]
        arms[:, sides] += weight[..., None] * (target[:, sides] - rest[sides])
    shift = table[:, [HEAD_X, HEAD_Z]]
    points = np.concatenate([arms.reshape(len(table), 4, 2) + 0.8 * shift[:, None], shift[:, None]], axis=1)
    move.points = points + table[:, None, [U, V]]
    return move.points


def _rest_points(fig: Figure) -> np.ndarray:
    """``(5, 2)``: :func:`reach_points` of standing still."""
    return np.concatenate([fig.reaches["rest"].reshape(4, 2), np.zeros((1, 2))])


@dataclass
class Player:
    """Someone who plays with the camera, as the planner sees them: their behaviour, body and wave clips."""

    behaviour: object
    figure: Figure
    library: object = None  # MotionLibrary (for the wave clips)
    sex: str = "male"
    age: str = "adult"

    @classmethod
    def of(cls, character) -> Player:
        """The player a :class:`~.character.Character` is (its behaviour's style and motion library)."""
        behaviour = character.behaviour
        style = getattr(behaviour, "style", None)
        return cls(behaviour, figure(character.skinned_model), getattr(behaviour, "library", None),
                   getattr(style, "sex", "male"), getattr(style, "age", "adult"))


def _playfulness(age: str, rng: random.Random) -> float:
    if age == "child":
        return rng.uniform(1.15, 1.5)
    if age == "older":
        return rng.uniform(0.3, 0.75)
    mood = rng.random()
    return rng.uniform(0.3, 0.55) if mood < 0.25 else rng.uniform(1.0, 1.3) if mood > 0.75 else rng.uniform(0.6, 1.0)


@dataclass
class _Room:
    """Where everybody is while the planner plans: ``tracks`` ``(people, len(times), 2)`` (players first, then the
    others; nan while hidden), ``reach`` ``(players, len(times), 5, 2)``: the players' planned reach points (nan
    when they make no move), both world floor ``(x, z)``; ``place``: where the moves may take people."""

    times: np.ndarray
    step: float
    tracks: np.ndarray
    reach: np.ndarray
    place: object = None
    camera: tuple = (0.0, 0.0)  # (x, z)


def plan_camera_play(players, camera, *, others=(), period: float = 360.0, rng=0, moves=MOVES,
                     share: float = PLAY_SHARE, step: float = 0.1, place=None) -> list[list[Move]]:
    """The moves of each of ``players`` (:class:`Player`) over one ``period`` of scene time (see the module docs).

    ``others``: behaviours of everyone else, whose positions the moves keep clear of (``CLEARANCE``, ``REACH``);
    ``moves``: the kinds to choose from (names of :data:`MOVES`); ``share``: the share of the watching time an
    ordinary person plays; ``place``: where the moves may take people, ``(x, z) -> bool`` (where their feet stay in
    the picture, say: a move never takes someone out of it, nor nearer the camera when they are). Deterministic for a
    seed.
    """
    unknown = [m for m in moves if m not in _MAKERS]
    if unknown:
        raise ValueError(f"unknown camera play moves {unknown} (the moves are {', '.join(MOVES)})")
    rng, moves = _as_random(rng), list(moves)
    times = np.arange(0.0, period, step)
    tracks = np.stack([_track(p.behaviour, times) for p in players] + [_track(b, times) for b in others]) \
        if players else np.zeros((0, len(times), 2))
    cam = np.asarray(camera, float).ravel()
    room = _Room(times, step, tracks, np.full((len(players), len(times), 5, 2), np.nan), place, (cam[0], cam[2]))
    plans = []
    for i, player in enumerate(players):
        prng = random.Random(rng.random())
        playful = _playfulness(player.age, prng)
        person = _person_for(player.figure, prng, playful, player.library, player.sex)
        mine = min(0.6, max(0.08, share * playful))
        plan, done, last = [], set(), None
        for start, end in (watch_windows(player.behaviour, period, step) if moves else []):
            visit = end - start < 60.0
            budget = (1 + (playful > 0.7 and prng.random() < 0.7) + (playful > 1.05 and prng.random() < 0.6)) \
                if visit else 10**6
            t = start + (prng.uniform(1.0, 5.0) if visit else prng.uniform(0.0, 12.0))
            played = 0.0  # a visitor stops when they have played their share of the visit
            while budget > 0 and t < end - 2.0 and not (visit and played > mine * (end - start)):
                kind = _pick(moves, done, last, playful, prng)
                move = _fitting(kind, prng, person, player.behaviour, t, end, i, room, moves)
                if move is None:
                    t += prng.uniform(1.0, 3.0)
                    if visit:
                        budget -= 1
                    continue
                plan.append(move)
                done.add(move.kind)
                last, budget, played = move.kind, budget - 1, played + move.duration
                gap = 5.5 * (1.0 - mine) / mine
                t = move.end + (prng.uniform(4.0, 9.0) if visit else max(2.5, gap * prng.uniform(0.4, 1.6)))
        for move in plan:  # now the others keep clear of these moves
            k0, k1 = _span(move, times, step)
            u = times[k0:k1] - move.start
            room.reach[i, k0:k1] = tracks[i, k0:k1, None] + _turned(move, _sampled(reach_points(move, person.fig), u))
            if move.travels:
                tracks[i, k0:k1] += _world(move, u)
        plans.append(plan)
    return plans


def _sampled(rows: np.ndarray, u: np.ndarray) -> np.ndarray:
    """The rows (``RATE`` a second) of a move's table, or of its :func:`reach_points`, nearest ``u`` seconds in."""
    return rows[np.clip(np.rint(u * RATE).astype(int), 0, len(rows) - 1)]


def _pick(moves, done: set, last, playful: float, rng: random.Random) -> str:
    """The next kind of move: any but the last one, the ones not done yet first; playful people travel more."""
    weights = {"wave": 1.2, "faces": 0.6 + 0.5 * playful, "side_steps": 0.4 + 0.6 * playful, "lean_sideways": 1.0,
               "lean_forward": 1.0, "closer": 0.15 + 0.5 * playful}
    options = [m for m in moves if m != last] or list(moves)
    w = [weights.get(m, 1.0) * (3.0 if m not in done else 1.0) for m in options]
    return rng.choices(options, w)[0]


def _span(move: Move, times: np.ndarray, step: float) -> tuple[int, int]:
    return math.ceil(move.start / step - 1e-9), min(len(times), math.floor(move.end / step) + 1)


def _fitting(kind, rng, person, behaviour, t, end, i, room: _Room, moves) -> Move | None:
    """A move of ``kind`` (or a smaller one, one with the arms closer in, or another kind) starting at ``t`` that
    ends by ``end`` and keeps everyone clear."""
    heading = float(behaviour(t).heading)
    tries = [(kind, {})]
    if kind == "side_steps":
        first = rng.choice((-1.0, 1.0))
        tries = [(kind, {"first": first}), (kind, {"first": -first}), (kind, {"first": first, "size": 0.6}),
                 (kind, {"first": -first, "size": 0.6})]
    elif kind == "closer":
        tries = [(kind, {}), (kind, {"size": 0.6}), (kind, {"size": 0.6, "arms": False})]
    elif kind == "wave":  # one hand, the one away from the neighbour (whichever that is)
        tries += [(kind, {"hands": "right"}), (kind, {"hands": "left"})]
    elif kind in ("faces", "lean_forward"):  # without the hands at the face
        tries.append((kind, {"arms": False}))
    still = [m for m in ("lean_sideways", "lean_forward") if m in moves and m != kind]  # moves that need little room
    if still:
        tries.append((rng.choice(still), {}))
    for which, options in tries:
        move = _MAKERS[which](rng, person, **options)
        move.start, move.heading = t, heading
        if move.end <= end and _clear(move, i, room, person.fig):
            return move
    return None


def _placed(room: _Room, moved, base) -> bool:
    """Whether a move that takes someone from ``base`` to ``moved`` keeps them in ``room.place`` (or, out of it
    already, no nearer the camera)."""
    if room.place((float(moved[0]), float(moved[1]))):
        return True
    cx, cz = room.camera
    return not room.place((float(base[0]), float(base[1]))) and \
        math.hypot(moved[0] - cx, moved[1] - cz) >= math.hypot(base[0] - cx, base[1] - cz) - 1e-3


def _clear(move: Move, i: int, room: _Room, fig: Figure) -> bool:
    """Whether ``move`` keeps person ``i`` ``CLEARANCE`` from everyone, their elbows, hands and head ``REACH`` from
    everyone else's body and ``HANDS`` from the others' moves' (or no nearer than they would be anyway), and in the
    ``place`` the moves may take them."""
    times, tracks = room.times, room.tracks
    k0, k1 = _span(move, times, room.step)
    if k1 <= k0:
        return True
    u = times[k0:k1] - move.start
    base = tracks[i, k0:k1]
    moved = base + _world(move, u) if move.travels else base
    if move.travels and room.place is not None and not all(_placed(room, p, q) for p, q in zip(moved[::2], base[::2])):
        return False
    others = np.delete(tracks[:, k0:k1], i, axis=0)[:, :, None]  # (people, n, 1, 2)
    theirs = np.delete(room.reach[:, k0:k1], i, axis=0)  # (players, n, 5, 2)
    mine = base[:, None] + _turned(move, _sampled(reach_points(move, fig), u))  # (n, 5, 2)
    still = base[:, None] + _turned(move, _rest_points(fig))

    def apart(a, b, now, then, least):  # at least ``least`` apart, or no nearer than then (nan: nobody there)
        d = np.hypot(*np.moveaxis(a - b, -1, 0))
        d0 = np.hypot(*np.moveaxis(now - then, -1, 0)) if now is not None else d
        return bool(np.all(np.isnan(d) | (d >= least) | (d >= d0 - 1e-3)))

    return (apart(moved[:, None], others, base[:, None], others, CLEARANCE)
            and apart(mine, others, still, others, REACH)
            and apart(moved[:, None], theirs, base[:, None], theirs, REACH)
            and apart(mine[:, :, None], theirs[:, :, None, :], still[:, :, None], theirs[:, :, None, :], HANDS))


# -- doing the moves --------------------------------------------------------------------------------------------------


class CameraPlay:
    """A behaviour that plays with the camera: ``inner`` with ``moves`` (from :func:`plan_camera_play`) on top.

    The moves repeat every ``period`` seconds of the time it is called with. Everything else (``duration``,
    ``offset``, ``watching``, ``style`` ...) is the ``inner`` behaviour's; see the module docs.
    """

    def __init__(self, inner, moves, camera, period: float = 360.0):
        cam = [float(v) for v in np.asarray(camera, float).ravel()]
        moves = sorted(moves, key=lambda m: m.start)
        style = getattr(inner, "style", None)
        self.__dict__.update(inner=inner, moves=moves, camera=(cam[0], cam[1], cam[2]), period=float(period),
                             _starts=[m.start for m in moves], _travelling=[m for m in moves if m.travels],
                             _eye=_EYE_HEIGHT * float(getattr(style, "stride", 1.0) or 1.0), _last=(None, None))

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def __setattr__(self, name, value):
        setattr(self.inner, name, value)

    def sample(self, t: float):
        """``(move, channel values)`` at time ``t``, or ``None`` when no move is going on."""
        last = self.__dict__["_last"]
        if last[0] == t:
            return last[1]
        tm = t % self.period
        k = bisect.bisect_right(self._starts, tm) - 1
        found = None
        if k >= 0:
            move = self.moves[k]
            u = tm - move.start
            if u < move.duration:
                found = (move, move.values(u))
        self.__dict__["_last"] = (t, found)
        return found

    def move_at(self, t: float) -> Move | None:
        """The move going on at ``t``, if any."""
        found = self.sample(t)
        return found[0] if found is not None else None

    def __call__(self, t: float):
        found = self.sample(t)
        inner = self.inner
        if found is None:
            return inner(t)
        if isinstance(inner, Behaviour):  # with the clip's own head turn (others: as their state has it)
            state, clip_yaw, elevation = inner._evaluate(t)
        else:
            state, clip_yaw, elevation = inner(t), 0.0, 0.0
        return self._moved(state, found[1], clip_yaw, elevation)

    def _moved(self, state, v: list, clip_yaw: float, elevation: float):
        """``state`` moved by the body offset, the head looking at the camera from where the eyes now are."""
        h = math.radians(state.heading)
        sh, ch = math.sin(h), math.cos(h)
        x, y, z = state.position
        px, pz = x + v[U] * ch + v[V] * sh, z - v[U] * sh + v[V] * ch
        if not state.visible:
            return replace(state, position=(px, y, pz))
        ex, ez = px + v[HEAD_X] * ch + v[HEAD_Z] * sh, pz - v[HEAD_X] * sh + v[HEAD_Z] * ch
        cx, cy, cz = self.camera
        turn = _wrap(math.degrees(math.atan2(cx - ex, cz - ez) - math.atan2(cx - x, cz - z)))
        tilt = math.degrees(math.atan2(cy - self._eye, max(math.hypot(cx - x, cz - z), 0.3))
                            - math.atan2(cy - self._eye - v[HEAD_Y], max(math.hypot(cx - ex, cz - ez), 0.3)))
        c0, e0 = math.radians(clip_yaw), math.radians(elevation)
        d = _bend(v[PITCH], v[ROLL]) @ (math.sin(c0) * math.cos(e0), -math.sin(e0), math.cos(c0) * math.cos(e0))
        bent_yaw = math.degrees(math.atan2(d[0], d[2]))
        bent_elevation = math.degrees(math.asin(max(-1.0, min(1.0, -d[1]))))
        yaw0, pitch0 = state.head_yaw, state.head_pitch
        pitch = pitch0 + tilt - (bent_elevation - elevation) + v[HEAD_PITCH]
        face = clip_yaw + yaw0 + pitch_shift(clip_yaw, elevation, yaw0, pitch0) + turn + v[HEAD_YAW]
        yaw = face - bent_yaw
        for _ in range(2):  # the pitch turns the face a little sideways when the head is turned
            yaw = face - bent_yaw - pitch_shift(bent_yaw, bent_elevation, yaw, pitch)
        yaw = min(max(_wrap(yaw), -HEAD_YAW_LIMIT), HEAD_YAW_LIMIT)
        return replace(state, position=(px, y, pz), head_yaw=yaw, head_pitch=pitch)

    def positions(self, times) -> np.ndarray:
        """``(x, z)`` at each of ``times`` (nan while hidden), the moves included."""
        times = np.asarray(times, float)
        track = np.array(self.inner.positions(times), float)
        tm = times % self.period
        for move in self._travelling:
            inside = (tm >= move.start) & (tm < move.end)
            if inside.any():
                track[inside] += _world(move, tm[inside] - move.start)
        return track

    def posed(self, character, t: float):
        """``(move, channel values)`` for posing ``character`` at scene time ``t``: the sample of the same time as
        the state it got (a Performer's own clock), faded while the Performer does a yoga act; ``None``: nothing."""
        behaviour, weight = character.behaviour, 1.0
        if behaviour is not self and getattr(behaviour, "base", None) is self and hasattr(behaviour, "_base_time"):
            weight = 1.0 - _acting(behaviour, t)
            t = behaviour._base_time(t)
        if weight <= 0.0:
            return None
        found = self.sample(t)
        if found is None or weight >= 1.0:
            return found
        move, v = found
        return move, [x if k in _PHASES else x * weight for k, x in enumerate(v)]


def _acting(performer, t: float) -> float:
    """1 while ``performer`` does a yoga act (from when it starts walking there until it is back), easing in and out."""
    for act in reversed(getattr(performer, "acts", ())):
        if act.start > t:
            continue
        if t < act.end:
            return _ease((t - act.start) / FADE)
        return 1.0 - _ease((t - act.end) / FADE)
    return 0.0


def find_play(behaviour) -> CameraPlay | None:
    """The :class:`CameraPlay` a character's behaviour is (or wraps, as a Performer's base), if any."""
    for _ in range(4):
        if isinstance(behaviour, CameraPlay):
            return behaviour
        behaviour = behaviour.__dict__.get("base") if hasattr(behaviour, "__dict__") else None
        if behaviour is None:
            return None
    return None


def doing(character, t: float) -> str:
    """What ``character`` does with the camera at scene time ``t`` (e.g. ``"waves at the camera"``), else ``""``."""
    play = find_play(character.behaviour)
    found = play.posed(character, t) if play is not None else None
    return DOING.get(found[0].kind, "") if found is not None else ""


class PlayPose:
    """The pose adjuster (``Character.pose_adjusters``) that does a :class:`CameraPlay`'s moves on the bones."""

    def __init__(self, play: CameraPlay, model: CharacterModel):
        self.play, self.fig = play, figure(model)
        self.rig = faces.face_rig(model)

    def __call__(self, local: np.ndarray, character, t: float) -> None:
        found = self.play.posed(character, t)
        if found is None:
            return
        state = character.state
        look = (state.head_yaw, state.head_pitch) if state is not None else (0.0, 0.0)
        self.pose(local, *found, sign=_holds_sign(character), look=look)

    def pose(self, local: np.ndarray, move: Move, v: list, *, sign: bool = False, look=(0.0, 0.0),
             legs: bool = True, face: bool = True) -> list:
        """Do ``move`` with the channel values ``v`` on the ``local`` bone transforms (in place). ``sign``: the right
        hand holds a sign; ``look``: the head turn the character gives the head after this (``head_yaw``,
        ``head_pitch``); ``legs`` / ``face``: whether to pose those too. Returns the spine chain's transforms
        (character space, root to head)."""
        fig, ch = self.fig, self.fig.chain
        g = [local[ch[0]]]
        for b in ch[1:]:
            g.append(g[-1] @ local[b])
        torso = bool(v[PITCH] or v[ROLL] or v[HIPS_U] or v[HIPS_V])
        legs = legs and (torso or any(v[k] for k in (U, V, LEFT_U, LEFT_V, LEFT_LIFT, RIGHT_U, RIGHT_V, RIGHT_LIFT,
                                                     SINK)))
        feet = [_clip_leg(local, g, leg) for leg in fig.legs] if legs else None
        if torso:  # each bend turns its bone about its joint in character space: G' = R G (local: P^T R P L)
            for k, (ps, rs) in enumerate(zip(PITCH_SHARES, ROLL_SHARES), start=1):
                parent = g[k - 1][:3, :3]
                local[ch[k], :3, :3] = parent.T @ _bend(v[PITCH] * ps, v[ROLL] * rs) @ parent @ local[ch[k], :3, :3]
            local[ch[0], 0, 3] += v[HIPS_U]
            local[ch[0], 2, 3] += v[HIPS_V]
        gm = [local[ch[0]].copy()]
        for b in ch[1:-1]:
            gm.append(gm[-1] @ local[b])
        if v[HEAD_ROLL]:  # a tilt of the head about its joint (in character space)
            neck = gm[5][:3, :3]
            local[ch[6], :3, :3] = neck.T @ _bend(0.0, v[HEAD_ROLL]) @ neck @ local[ch[6], :3, :3]
        gm.append(gm[5] @ local[ch[6]])
        if feet is not None:
            self._legs(local, gm, feet, v)
        if torso:  # the arms keep hanging down while the torso bends; waves and gestures start from there
            hang = _bend(0.85 * v[PITCH], 0.85 * v[ROLL]).T
            for arm in fig.arms.values():
                clavicle = (gm[5] @ local[arm.clavicle])[:3, :3]
                local[arm.upper, :3, :3] = clavicle.T @ hang @ clavicle @ local[arm.upper, :3, :3]
        if move.wave is not None and v[WAVE] > 0.0:
            self._wave(local, gm, move.wave, v, sign)
        if move.gesture is not None and v[GESTURE] > 0.0:
            self._gesture(local, gm, move.gesture, v, sign, look)
        if face and self.rig is not None and move.faces:
            weights = {name: v[_FACES[k]] for k, name in enumerate(move.faces) if v[_FACES[k]] > 0.0}
            if weights:
                faces.apply_face(local, self.rig, weights)
        return gm

    def _legs(self, local, gm, feet, v) -> None:
        """Two-bone IK of both legs to the feet's places; the pelvis goes down by ``SINK`` and further where the legs
        would be straighter than ``0.985`` of their length (or than the clip has them)."""
        fig, legs, sink = self.fig, [], v[SINK]
        drop = sink
        for (parent, thigh, _, _, first, second), (_, _, ankle0, _, reach0), (cu, cv, cl) in zip(fig.legs, feet, _FEET):
            gt = gm[parent] @ local[thigh]
            hip = gt[:3, 3].tolist()
            target = (ankle0[0] + v[cu] - v[U], ankle0[1] + v[cl], ankle0[2] + v[cv] - v[V])
            limit = max(0.985 * (first + second), reach0)  # never straighter than the clip has it
            flat = (target[0] - hip[0]) ** 2 + (target[2] - hip[2]) ** 2
            if flat < limit * limit:
                drop = max(drop, hip[1] - target[1] - math.sqrt(limit * limit - flat))
            legs.append((gt, hip, target))
        if drop > 0.0:
            local[fig.chain[0], 1, 3] -= drop
            for m in gm:
                m[1, 3] -= drop
            for gt, hip, _ in legs:
                gt[1, 3] -= drop
                hip[1] -= drop
        for (parent, thigh, calf, foot, first, second), (hip0, knee0, ankle0, rot0, _), (gt, hip, target) in \
                zip(fig.legs, feet, legs):
            pole = _off_axis(_sub(knee0, hip0), _unit(_sub(ankle0, hip0))) or (0.0, 0.0, 1.0)  # as the clip bends
            knee, ankle, _ = _two_bone(hip, first, second, target, pole)
            rot = gt[:3, :3]
            rt = _align((rot @ local[calf, :3, 3]).tolist(), _sub(knee, hip)) @ rot
            rc = rt @ local[calf, :3, :3]
            rc = _align((rc @ local[foot, :3, 3]).tolist(), _sub(ankle, knee)) @ rc
            local[thigh, :3, :3] = gm[parent][:3, :3].T @ rt
            local[calf, :3, :3] = rt.T @ rc
            local[foot, :3, :3] = rc.T @ rot0  # the foot keeps the clip's lie (flat on the floor)

    def _wave(self, local, gm, wave: Wave, v, sign: bool) -> None:
        """The waving arm(s): the clip's (mirrored for the left hand) blended over the clips', or the procedural
        wave."""
        fig, weight = self.fig, min(1.0, v[WAVE])
        hands = {"right": [("right", False)], "left": [("left", True)], "both": [("right", False), ("left", False)]}
        arms = hands[wave.hands]
        if sign:  # the right hand holds a sign: the left one waves
            arms = [("left", False)] if wave.hands == "both" else [("left", wave.clip is not None)]
        if wave.clip is None:
            self._procedural_wave(local, gm, [fig.arms[side] for side, _ in arms], v, wave.amplitude, weight)
            return
        blends = []
        for side, mirror in arms:
            bones, track = fig.arm_track(wave.clip, side, mirror)
            f = min(max(v[WAVE_TIME], 0.0), wave.clip.duration) * wave.clip.fps
            i = min(int(f), len(track) - 1)
            j = min(i + 1, len(track) - 1)
            blends.append((bones, _nlerp(track[i], track[j], f - i), weight))
        _apply(local, blends)

    def _procedural_wave(self, local, gm, arms: list, v, amplitude: float, weight: float) -> None:
        """The hand raised beside the head, palm to the camera, the forearm swinging to and fro, the wrist after it."""
        s, reaches = self.fig.scale, []
        swing = amplitude * math.sin(v[WAVE_TIME])
        out = math.radians(10.0 + swing)
        lag = math.radians(10.0 + swing + 0.6 * amplitude * math.sin(v[WAVE_TIME] - 0.9))
        for arm in arms:
            sigma = arm.sign
            shoulder = (gm[5] @ local[arm.clavicle] @ local[arm.upper])[:3, 3].tolist()
            upper, fore = arm.lengths
            elbow = _plus(shoulder, _unit((sigma * 0.72, -0.3, 0.62)), upper)
            wrist = _plus(elbow, _unit((sigma * math.sin(out), math.cos(out), 0.18)), fore)
            pole = _off_axis(_sub(elbow, shoulder), _unit(_sub(wrist, shoulder))) or (sigma * 0.7, -0.7, 0.0)
            reaches.append(_Reach(arm, wrist, pole, (sigma * math.sin(lag), math.cos(lag), 0.1),
                                  (sigma * 0.15, 0.0, -1.0), arm.curls["ears"], rise=(sigma * 0.05 * s, 0.15 * s, 0.32 * s),
                                  arrive=(sigma * 0.05 * s, -0.15 * s, 0.15 * s),
                                  low=_unit((sigma * 0.45, -0.85, 0.25))))
        self._reach(local, gm, reaches, weight)

    def _gesture(self, local, gm, gesture: str, v, sign: bool, look) -> None:
        """Hands to the face (thumbs at the temples, fingers at the mouth corners, palms on the cheeks); the hands
        come up in front of the chest and the face, the elbows low."""
        fig = self.fig
        yaw, pitch = look
        turned = _pivot_rotation(yaw * NECK_SHARE, pitch * NECK_SHARE, gm[5][:3, 3]) @ gm[6]  # as Character turns it
        head = _pivot_rotation(yaw * (1.0 - NECK_SHARE), pitch * (1.0 - NECK_SHARE), turned[:3, 3]) @ turned
        rot, centre = head[:3, :3] @ fig.rest_rot[fig.chain[6]].T, head[:3, 3]
        chest = gm[4][:3, :3] @ fig.rest_rot[fig.chain[4]].T
        reaches = []
        for side in ["left"] if sign else ["left", "right"]:
            arm = fig.arms[side]
            at, way = fig.gestures[gesture, side]
            at, way = at @ rot.T, way @ chest.T
            wrist, fingers, back, curl = centre + at[0], at[1], at[2], arm.curls[gesture]
            if gesture == "ears":  # the thumb's tip at the temple, the fingers wiggling
                wrist = wrist - (_frame(fingers, back) @ arm.rest[2]) @ arm.thumb_tip
                phase, size = arm.wiggle
                curl = curl + 30.0 * (0.5 + 0.5 * np.sin(v[WIGGLE] + phase)) * size
            reaches.append(_Reach(arm, wrist.tolist(), way[0].tolist(), fingers.tolist(), back.tolist(), curl,
                                  *way[1:].tolist(), gesture == "ears"))
        self._reach(local, gm, reaches, min(1.0, v[GESTURE]))

    def _reach(self, local, gm, reaches: list, s: float) -> None:
        """Take each arm of ``reaches`` (:class:`_Reach`) the ``s`` (0..1) part of the way from where the clips have
        it to the pose it reaches for.

        The elbow travels from where it is to its place in the pose along a curve drawn towards the upper arm's
        direction ``low`` (low elbows, for instance), and the wrist on a curve that leaves its place towards
        ``rise`` and comes in from ``arrive`` (in front of the chest and of the face, say): the forearm points from
        the elbow at that curve. Meanwhile the twist of the arm's bones about the arm's plane and the bend of the
        hand and fingers go from the clip's to the pose's. So the arm reaches like an arm (it does not swing out on
        the way, nor flip over: the elbow never has to jump round a nearly straight arm the way IK would), every
        frame of the way is a pose by itself, and ``s = 0`` is the clip's arm."""
        k = 1.0 - s
        kk, ks, ss = k * k, 2.0 * k * s, s * s
        c0, c1, c2, c3 = k * kk, 3.0 * kk * s, 3.0 * k * ss, s * ss  # a cubic Bezier curve
        arms, clip, axes, angles = [], [], [], []
        for r in reaches:
            arm, n = r.arm, len(r.arm.fingers)
            clavicle = gm[5] @ local[arm.clavicle]
            upper = clavicle @ local[arm.upper]
            shoulder, (l1, l2) = upper[:3, 3].tolist(), arm.lengths
            e1, w1, b1 = _two_bone(shoulder, l1, l2, r.wrist, r.pole)  # the arm in the pose
            hand = (_frame(_sub(w1, e1), b1) @ arm.rest[1]).T @ (_frame(r.fingers, r.back) @ arm.rest[2])
            bones = arm.blend if r.thumb else arm.blend[:1 + n]  # the hand, the fingers (, the thumb)
            axes.append(arm.curl_axes)
            angles.append(r.curl)
            if s >= 1.0:  # there
                arms.append((arm, n, bones, clavicle, shoulder, e1, w1, b1, hand))
                continue
            fore = upper @ local[arm.fore]
            w0, e0 = (fore @ local[arm.hand])[:3, 3].tolist(), fore[:3, 3].tolist()
            aim = (upper[:3, :3] @ arm.rest[0].T)[:, 1].tolist()  # the side the clip bends the elbow to
            aim_at = [c0 * a + c1 * (a + p) + c2 * (b + q) + c3 * b for a, b, p, q in zip(w0, w1, r.rise, r.arrive)]
            elbow = _plus(shoulder, _unit([kk * (a - o) + ks * l1 * d + ss * (b - o) for a, b, d, o in
                                           zip(e0, e1, r.low, shoulder)]), l1)
            reached = _plus(elbow, _unit(_sub(aim_at, elbow)), l2)
            bend = _off_axis([(a - o) / l1 + k * q for a, o, q in zip(elbow, shoulder, aim)],
                             _unit(_sub(reached, shoulder))) or aim
            b0 = _off_axis([(a - o) / l1 + q for a, o, q in zip(e0, shoulder, aim)], _unit(_sub(w0, shoulder))) or aim
            rest = arm.blend_rest[:len(bones)]
            clip += [(_frame(_sub(e0, shoulder), b0).T @ upper[:3, :3] @ arm.rest[0].T)[None],  # the clip's twists
                     (_frame(_sub(w0, e0), b0).T @ fore[:3, :3] @ arm.rest[1].T)[None],  # about the arm's frames,
                     np.swapaxes(rest, 1, 2) @ local[bones, :3, :3],  # its hand, fingers and thumb from the rest's
                     (rest[0].T @ hand)[None]]  # and the pose's hand
            arms.append((arm, n, bones, clavicle, shoulder, elbow, reached, bend, rest))
        half = np.radians(np.concatenate(angles)) * 0.5  # the fingers' bends from the rest pose, as quaternions
        bent = np.concatenate([np.sin(half)[:, None] * np.concatenate(axes), np.cos(half)[:, None]], axis=1)
        if s >= 1.0:
            bent, start = _matrices(bent), 0
            for arm, n, bones, clavicle, shoulder, elbow, reached, bend, hand in arms:
                pose = np.concatenate([hand[None], arm.blend_rest[1:1 + n] @ bent[start:start + n],
                                       arm.thumb[:len(bones) - 1 - n]])
                self._arm(local, arm, bones, clavicle, shoulder, elbow, reached, bend, arm.rest[0], arm.rest[1], pose)
                start += n
            return
        q, start, at = _quats(np.concatenate(clip)), 0, 0  # blend from the rest pose's: small turns, no flips
        clip, pose = [], []
        for arm, n, bones, *_ in arms:
            clip.append(q[at:at + 2 + len(bones)])
            pose += [_STILL, q[at + 2 + len(bones):at + 3 + len(bones)], bent[start:start + n],
                     arm.thumb_q[:len(bones) - 1 - n]]
            at, start = at + 3 + len(bones), start + n
        m, at = _matrices(_nlerp(np.concatenate(clip), np.concatenate(pose), s)), 0
        for arm, n, bones, clavicle, shoulder, elbow, reached, bend, rest in arms:
            self._arm(local, arm, bones, clavicle, shoulder, elbow, reached, bend, m[at] @ arm.rest[0],
                      m[at + 1] @ arm.rest[1], rest @ m[at + 2:at + 2 + len(bones)])
            at += 2 + len(bones)

    @staticmethod
    def _arm(local, arm: _Arm, bones, clavicle, shoulder, elbow, reached, bend, upper, fore, pose) -> None:
        """Pose ``arm``: the upper arm and forearm along the elbow and the wrist, turned about them as ``upper`` and
        ``fore`` say (the bones' rotations from the arm's frames), and ``bones`` to ``pose``."""
        ru = _frame(_sub(elbow, shoulder), bend) @ upper
        rf = _frame(_sub(reached, elbow), bend) @ fore
        local[arm.upper, :3, :3] = clavicle[:3, :3].T @ ru
        local[arm.fore, :3, :3] = ru.T @ rf
        local[bones, :3, :3] = pose


def _clip_leg(local, g, leg) -> tuple:
    """Where the clip puts one leg: hip, knee and ankle (character space), the foot's rotation, the leg's reach."""
    parent, thigh, calf, foot, _, _ = leg
    gt = g[parent] @ local[thigh]
    gc = gt @ local[calf]
    gf = gc @ local[foot]
    hip, knee, ankle = gt[:3, 3].tolist(), gc[:3, 3].tolist(), gf[:3, 3].tolist()
    return hip, knee, ankle, gf[:3, :3], math.dist(hip, ankle)


def _apply(local, blends: list) -> None:
    """Set the local rotations of each blend's bones to its targets (quaternions or 3x3 matrices), weighted: the
    partial ones together, so that the quaternion conversions run once a frame."""
    partial = []
    for bones, target, weight in blends:
        if weight >= 1.0:
            local[bones, :3, :3] = target if target.shape[-1] == 3 else _matrices(target)
        else:
            partial.append((bones, target, weight))
    if not partial:
        return
    partial.sort(key=lambda blend: blend[1].shape[-1])  # matrices first: converted in one go
    matrices = [q for _, q, _ in partial if q.shape[-1] == 3]
    target = [_quats(np.concatenate(matrices))] if matrices else []
    target = np.concatenate(target + [q for _, q, _ in partial if q.shape[-1] == 4])
    bones = np.concatenate([b for b, _, _ in partial])
    weights = np.concatenate([np.full(len(b), w) for b, _, w in partial])
    local[bones, :3, :3] = _matrices(_nlerp(_quats(local[bones, :3, :3]), target, weights))


# small vectors as plain floats: much quicker than numpy for one 3-vector at a time


def _sub(a, b) -> tuple:
    return a[0] - b[0], a[1] - b[1], a[2] - b[2]


def _plus(a, b, k: float = 1.0) -> tuple:
    return a[0] + k * b[0], a[1] + k * b[1], a[2] + k * b[2]


def _unit(a) -> tuple:
    n = math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]) or 1e-12
    return a[0] / n, a[1] / n, a[2] / n


def _off_axis(v, axis) -> tuple | None:
    """The part of ``v`` across the unit ``axis``, as a unit vector (``None`` when ``v`` lies along it)."""
    d = v[0] * axis[0] + v[1] * axis[1] + v[2] * axis[2]
    w = (v[0] - d * axis[0], v[1] - d * axis[1], v[2] - d * axis[2])
    n = math.sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2])
    return (w[0] / n, w[1] / n, w[2] / n) if n > 1e-6 else None


def _two_bone(start, first: float, second: float, target, pole) -> tuple:
    """``yoga._two_bone`` on plain floats: middle and end joint of a two-bone limb from ``start`` reaching for
    ``target``, bending towards ``pole``, and the direction it bends in."""
    reach = _sub(target, start)
    d = math.sqrt(reach[0] ** 2 + reach[1] ** 2 + reach[2] ** 2)
    u = (reach[0] / d, reach[1] / d, reach[2] / d) if d > 1e-9 else (0.0, -1.0, 0.0)
    d = min(max(d, abs(first - second) + 1e-4), (first + second) * 0.9999)
    v = _off_axis(pole, u) or _off_axis((0.0, 0.0, 1.0), u) or _off_axis((0.0, 1.0, 0.0), u)
    cos_a = (first * first + d * d - second * second) / (2.0 * first * d)
    sin_a = math.sqrt(max(0.0, 1.0 - cos_a * cos_a))
    middle = (start[0] + first * (cos_a * u[0] + sin_a * v[0]), start[1] + first * (cos_a * u[1] + sin_a * v[1]),
              start[2] + first * (cos_a * u[2] + sin_a * v[2]))
    return middle, (start[0] + d * u[0], start[1] + d * u[1], start[2] + d * u[2]), v


def _align(a, b) -> np.ndarray:
    """The smallest rotation (3x3) turning direction ``a`` into direction ``b``."""
    a, b = _unit(a), _unit(b)
    x, y, z = a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]
    c = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    if c < -0.999999:
        return yoga._align(np.asarray(a), np.asarray(b))
    k = 1.0 / (1.0 + c)
    return np.array([[c + k * x * x, k * x * y - z, k * x * z + y],
                     [k * x * y + z, c + k * y * y, k * y * z - x],
                     [k * x * z - y, k * y * z + x, c + k * z * z]])


def _frame(main, second) -> np.ndarray:
    """``yoga._frame`` on plain floats: columns ``main``, ``second`` made perpendicular to it, and their cross."""
    u = _unit(main)
    v = _off_axis(second, u) or _off_axis((0.0, 0.0, 1.0) if abs(u[2]) < 0.9 else (1.0, 0.0, 0.0), u)
    w = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])
    return np.array([[u[0], v[0], w[0]], [u[1], v[1], w[1]], [u[2], v[2], w[2]]])


def _holds_sign(character) -> bool:
    items = getattr(character, "items", None)
    return bool(items) and "sign" in items


# -- putting it together ----------------------------------------------------------------------------------------------


def play_along(people, camera, *, others=(), period: float = 360.0, rng=0, moves=MOVES,
               share: float = PLAY_SHARE, place=None) -> list[CameraPlay]:
    """Let ``people`` (:class:`~.character.Character` objects whose behaviours watch ``camera`` now and then) play
    with the camera: plan their moves together (clear of ``others``, characters or behaviours), wrap each behaviour in
    a :class:`CameraPlay` and give the character a :class:`PlayPose`. People who cannot are left as they are: 2D
    cut-outs, other skeletons, and behaviours that would not watch the same stretches every ``period`` (a loop that
    does not fit it, unless they watch all the time). Returns the plays, in the order of the people who got one."""
    from .character import Character

    chosen = [p for p in people if isinstance(p, Character) and p.behaviour is not None and
              hasattr(p.behaviour, "watching") and supports(p.skinned_model) and _repeats(p.behaviour, period)]
    rest = [getattr(o, "behaviour", o) for o in others] + \
        [p.behaviour for p in people if p not in chosen and getattr(p, "behaviour", None) is not None]
    rest = [b for b in rest if b is not None]
    plans = plan_camera_play([Player.of(p) for p in chosen], camera, others=rest, period=period, rng=rng,
                             moves=moves, share=share, place=place)
    plays = []
    for person, plan in zip(chosen, plans):
        play = CameraPlay(person.behaviour, plan, camera, period)
        person.behaviour = play
        person.pose_adjusters.append(PlayPose(play, person.skinned_model))
        plays.append(play)
    return plays


def _repeats(behaviour, period: float, step: float = 0.5) -> bool:
    """Whether ``behaviour`` watches the same stretches every ``period`` (its loop fits it, or it always watches)."""
    duration = getattr(behaviour, "duration", None)
    if duration is not None and fits_period(float(duration), period):
        return True
    return all(behaviour.watching(float(t)) for t in np.arange(0.0, period, step))
