"""Yoga poses as animation clips for the Rocketbox avatars (3ds Max Biped skeletons).

A pose is described where it happens, in character space (the person faces +Z, Y is up, their left is +X, metres,
the floor is y = 0): where the hips are (the ``Bip01`` root) and how the pelvis, the spine and the head are turned
(:class:`BodyPose`), and for each arm and leg where the wrist or ankle goes, which way the elbow or knee points and
how the hand or foot lies (:class:`Limb`). :func:`solve` turns that into bone rotations for one avatar's own
skeleton, with two-bone IK for the limbs, so the same description fits a child and a tall man. Limbs marked
``contact`` (and the knees, with ``knees_down``) are then moved until the skinned mesh just touches the floor
(:func:`settle`).

Each pose in :data:`POSES` is a short sequence: from a starting pose (standing, or on hands and knees) into the
pose, holding it while breathing, and back, so that the clip loops. :func:`yoga_clip` builds such a clip for an
avatar::

    from bifold_demo.characters.yoga import yoga_clip
    clip = yoga_clip(library.load("Female_Adult_01"), "yogi_squat")   # an AnimationClip, ~13 s, looping
    person = Character(model, behaviour=lambda t: CharacterState((0, 0, 0), 0.0, [Layer(clip, t)]))

Angles are degrees. Rotations of the pelvis, spine and head are ``(pitch, yaw, roll)``: pitch bends forward
(positive) or back, yaw turns to the person's left, roll tilts the top towards their right; each is relative to
the body part below it (the pelvis to the character).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from functools import partial

import numpy as np

from .character import skin_vertices
from .model import AnimationClip, CharacterModel, quat_from_matrix, quat_to_matrix

BREATH = 4.5  # seconds per breath while holding a pose
HAND_REACH = 0.95  # arms on the floor reach this share of their length (straight, the elbows soft, room to breathe)
FPS = 15.0  # frames per second of the clips (they are interpolated when played)


# -- describing a pose ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class Limb:
    """Where an arm or a leg goes (character space).

    ``target``: the wrist or ankle joint. ``pole``: the direction the elbow or knee points (only its part across
    the limb counts). ``forward``: where the fingers or toes point; ``up``: the back of the hand or the top of the
    foot. ``contact``: the hand or foot rests on the floor (:func:`settle` moves the target until it just touches).
    ``shrug``: arms only, the shoulder raised by that many degrees (arms overhead). ``toes``: legs only, the toes bent
    up towards the top of the foot by that many degrees (tucked under when kneeling).
    """

    target: tuple
    pole: tuple
    forward: tuple
    up: tuple
    contact: bool = False
    shrug: float = 0.0
    toes: float = 0.0


@dataclass(frozen=True)
class BodyPose:
    """A whole-body pose; see the module docs for the conventions."""

    root: tuple  # the Bip01 joint (between the hips)
    left_arm: Limb
    right_arm: Limb
    left_leg: Limb
    right_leg: Limb
    pelvis: tuple = (0.0, 0.0, 0.0)
    spine: tuple = (0.0, 0.0, 0.0)  # spread over the three spine bones
    head: tuple = (0.0, 0.0, 0.0)  # relative to the chest
    knees_down: bool = False  # both knees rest on the floor (settle moves the hips until they do)
    flat_hands: bool = False  # fingers straight, as when the palms are on the floor
    spine_shares: tuple = (0.1, 0.45, 0.45)  # how ``spine`` is spread over Spine, Spine1 and Spine2


def _lerp(a, b, w):
    return tuple(float(x) + (float(y) - float(x)) * w for x, y in zip(a, b))


def _direction_lerp(a, b, w):
    v = np.asarray(a, float) * (1 - w) + np.asarray(b, float) * w
    n = np.linalg.norm(v)
    return tuple(v / n) if n > 1e-6 else tuple(np.asarray(b, float))


def _slerp(a: np.ndarray, b: np.ndarray, w: float) -> np.ndarray:
    """Rotation ``w`` of the way from 3x3 ``a`` to ``b`` (the shorter way)."""
    qa, qb = quat_from_matrix(a), quat_from_matrix(b)
    dot = float(np.dot(qa, qb))
    if dot < 0:
        qb, dot = -qb, -dot
    if dot > 0.9995:
        q = qa + (qb - qa) * w
    else:
        angle = math.acos(min(dot, 1.0))
        q = (math.sin((1 - w) * angle) * qa + math.sin(w * angle) * qb) / math.sin(angle)
    return quat_to_matrix(q / np.linalg.norm(q))


def _limb_blend(a: Limb, b: Limb, w: float, lift: float = 0.0) -> Limb:
    target = _lerp(a.target, b.target, w)
    if lift:  # a step: the foot or hand rises on the way
        target = (target[0], target[1] + lift * math.sin(math.pi * w), target[2])
    turn = _slerp(_frame(a.forward, a.up), _frame(b.forward, b.up), w)  # the hand or foot turns smoothly
    return Limb(target, _direction_lerp(a.pole, b.pole, w), tuple(turn[:, 0]), tuple(turn[:, 1]),
                a.contact if w < 0.5 else b.contact, a.shrug + (b.shrug - a.shrug) * w, a.toes + (b.toes - a.toes) * w)


def blend(a: BodyPose, b: BodyPose, w: float, lift: dict | None = None) -> BodyPose:
    """The pose ``w`` of the way from ``a`` to ``b`` (0..1); ``lift``: limb name -> metres it rises mid-way."""
    lift = lift or {}
    limbs = {name: _limb_blend(getattr(a, name), getattr(b, name), w, lift.get(name, 0.0)) for name in LIMBS}
    return BodyPose(_lerp(a.root, b.root, w), pelvis=_lerp(a.pelvis, b.pelvis, w), spine=_lerp(a.spine, b.spine, w),
                    head=_lerp(a.head, b.head, w), knees_down=a.knees_down if w < 0.5 else b.knees_down,
                    flat_hands=a.flat_hands if w < 0.5 else b.flat_hands,
                    spine_shares=_lerp(a.spine_shares, b.spine_shares, w), **limbs)


LIMBS = ("left_arm", "right_arm", "left_leg", "right_leg")


# -- rotations -----------------------------------------------------------------------------------------


def axis_rotation(axis, degrees: float) -> np.ndarray:
    """3x3 rotation by ``degrees`` about ``axis`` (right-handed)."""
    x, y, z = np.asarray(axis, float) / np.linalg.norm(axis)
    a = math.radians(degrees)
    c, s, t = math.cos(a), math.sin(a), 1 - math.cos(a)
    return np.array([[t * x * x + c, t * x * y - s * z, t * x * z + s * y],
                     [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
                     [t * x * z - s * y, t * y * z + s * x, t * z * z + c]])


def turn(pitch: float = 0.0, yaw: float = 0.0, roll: float = 0.0) -> np.ndarray:
    """Pitch forward about the body's left-right axis, yaw to the left, roll the top to the right (degrees)."""
    return axis_rotation((0, 1, 0), yaw) @ axis_rotation((1, 0, 0), pitch) @ axis_rotation((0, 0, 1), roll)


def _cross(a, b) -> np.ndarray:  # np.cross is slow for single 3-vectors
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


def _unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    return v / max(math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]), 1e-12)


def _across(v, axis) -> np.ndarray | None:
    """The part of ``v`` across ``axis`` (unit), or None when ``v`` lies along it."""
    v = np.asarray(v, float)
    v = v - (v[0] * axis[0] + v[1] * axis[1] + v[2] * axis[2]) * axis
    n = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    return v / n if n > 1e-6 else None


def _frame(main, second) -> np.ndarray:
    """Orthonormal frame (columns): ``main``, ``second`` made perpendicular to it, and their cross product."""
    u = _unit(main)
    v = _across(second, u)
    if v is None:
        v = _across((0, 0, 1) if abs(u[2]) < 0.9 else (1, 0, 0), u)
    frame = np.empty((3, 3))
    frame[:, 0], frame[:, 1], frame[:, 2] = u, v, _cross(u, v)
    return frame


def _align(a, b) -> np.ndarray:
    """The smallest rotation that turns direction ``a`` into direction ``b``."""
    a, b = _unit(a), _unit(b)
    axis = _cross(a, b)
    s, c = math.sqrt(float(axis @ axis)), float(a @ b)
    if s < 1e-9:
        return np.eye(3) if c > 0 else axis_rotation(_across((1, 0, 0), a) if abs(a[0]) < 0.9 else (0, 1, 0), 180.0)
    return axis_rotation(axis / s, math.degrees(math.atan2(s, c)))


# -- the avatar --------------------------------------------------------------------------------------


_BIPED = ("Bip01", "Bip01 Spine2", "Bip01 L Thigh", "Bip01 R Hand")


def supports(model: CharacterModel) -> bool:
    """Whether ``model`` has the (3ds Max Biped) skeleton the poses are made for."""
    return all(model.skeleton.find(n) is not None for n in _BIPED)


class Body:
    """Bones and rest measurements of an avatar (``Bip01 ...`` Biped names), for posing it."""

    def __init__(self, model: CharacterModel):
        sk = model.skeleton
        self.model, self.skeleton = model, sk
        missing = [n for n in _BIPED if sk.find(n) is None]
        if missing:
            raise ValueError(f"{model.name} is not a Biped skeleton (no {', '.join(missing)})")

        def bone(name):
            return sk.index(f"Bip01 {name}")

        self.root, self.pelvis = sk.index("Bip01"), bone("Pelvis")
        self.spine = [bone("Spine"), bone("Spine1"), bone("Spine2")]
        self.neck, self.head = bone("Neck"), bone("Head")
        self.sides = {}
        for side, sign in (("left", 1.0), ("right", -1.0)):
            s = "L" if side == "left" else "R"
            fingers = [[sk.find(f"Bip01 {s} Finger{k}{j}") for j in ("", "1", "2")] for k in range(5)]
            self.sides[side] = {
                "sign": sign, "clavicle": bone(f"{s} Clavicle"), "upper": bone(f"{s} UpperArm"),
                "fore": bone(f"{s} Forearm"), "hand": bone(f"{s} Hand"), "thigh": bone(f"{s} Thigh"),
                "calf": bone(f"{s} Calf"), "foot": bone(f"{s} Foot"), "toe": sk.find(f"Bip01 {s} Toe0"),
                "fingers": [[b for b in chain if b is not None] for chain in fingers],
            }
        rest = sk.rest_global
        self.rest_pos = rest[:, :3, 3].copy()
        self.rest_rot = rest[:, :3, :3] / np.linalg.norm(rest[:, :3, :3], axis=1, keepdims=True)
        local = sk.rest_local
        self.local_rot = local[:, :3, :3] / np.linalg.norm(local[:, :3, :3], axis=1, keepdims=True)
        self.local_t = local[:, :3, 3].copy()
        # the mesh, for floor contacts: every vertex with the bone that moves it most
        self.vertices = np.concatenate([p.positions for p in model.primitives]).astype(np.float64)
        joints = np.concatenate([p.joints for p in model.primitives]).astype(np.int64)
        weights = np.concatenate([p.weights for p in model.primitives])
        self.owner = joints[np.arange(len(joints)), weights.argmax(axis=1)]
        self.skin_joints, self.skin_weights = joints, weights.astype(np.float64)
        self.floor = float(self.vertices[:, 1].min())  # where the soles are in the rest pose
        pos = self.rest_pos
        for side in self.sides.values():
            side["thigh_length"] = float(np.linalg.norm(pos[side["calf"]] - pos[side["thigh"]]))
            side["shin_length"] = float(np.linalg.norm(pos[side["foot"]] - pos[side["calf"]]))
            side["upper_length"] = float(np.linalg.norm(pos[side["fore"]] - pos[side["upper"]]))
            side["fore_length"] = float(np.linalg.norm(pos[side["hand"]] - pos[side["fore"]]))
            foot = self.group(side["foot"], side["toe"])
            side["ankle_height"] = float(pos[side["foot"], 1] - self.vertices[foot, 1].min())  # standing flat
            side["foot_forward"] = _unit((pos[side["toe"]] - pos[side["foot"]]) * (1, 0, 1)) if side["toe"] is not None \
                else np.array([0.0, 0.0, 1.0])
            f0 = _unit(pos[side["fingers"][2][0]] - pos[side["hand"]])  # towards the middle finger
            across = _unit(pos[side["fingers"][1][0]] - pos[side["fingers"][4][0]])  # pinky to index
            back = _unit(np.cross(f0, across)) * side["sign"]
            tip = pos[side["fingers"][2][-1]] - pos[side["fingers"][2][0]]
            if np.dot(back, tip) > 0:  # the fingers curl towards the palm, so the back is the other way
                back = -back
            side["hand_forward"], side["hand_back"] = f0, back
            shoulder, elbow, wrist = pos[side["upper"]], pos[side["fore"]], pos[side["hand"]]
            bend = _across(elbow - shoulder, _unit(wrist - shoulder))
            side["arm_bend"] = bend if bend is not None else np.array([0.0, 0.0, -1.0])
            side["leg_bend"] = np.array([0.0, 0.0, 1.0])  # the rest legs are straight: knees point forward
        self.finger_tips = {chain[-1]: [c for c in sk.descendants(chain[-1]) if c != chain[-1]]
                            for side in self.sides.values() for chain in side["fingers"] if chain}
        left, right = self.sides["left"], self.sides["right"]
        self.leg = 0.5 * (left["thigh_length"] + left["shin_length"] + right["thigh_length"] + right["shin_length"])
        self.arm = 0.5 * (left["upper_length"] + left["fore_length"] + right["upper_length"] + right["fore_length"])
        self.hip_width = float(pos[left["thigh"], 0] - pos[right["thigh"], 0])
        self.shoulder_width = float(pos[left["upper"], 0] - pos[right["upper"], 0])
        self.root_height = float(pos[self.root, 1] - self.floor)  # standing straight
        self.hip_drop = float(pos[self.root, 1] - 0.5 * (pos[left["thigh"], 1] + pos[right["thigh"], 1]))
        self.torso = float(np.linalg.norm(0.5 * (pos[left["upper"]] + pos[right["upper"]]) - pos[self.root]))
        self.height = float(self.vertices[:, 1].max() - self.floor)

    @property
    def names(self) -> list[str]:
        return self.skeleton.names

    def group(self, *bones) -> np.ndarray:
        """Indices of the vertices moved most by any of ``bones``."""
        return np.nonzero(np.isin(self.owner, [b for b in bones if b is not None]))[0]

    def hand_group(self, side: str) -> np.ndarray:
        s = self.sides[side]
        return self.group(s["hand"], *[b for chain in s["fingers"] for b in chain])

    def foot_group(self, side: str) -> np.ndarray:
        s = self.sides[side]
        return self.group(s["foot"], s["toe"])

    def knee_group(self, side: str) -> np.ndarray:
        """The knee (what rests on the floor when kneeling on the toes)."""
        s = self.sides[side]
        leg = self.group(s["calf"], s["thigh"])
        distance = np.linalg.norm(self.vertices[leg] - self.rest_pos[s["calf"]], axis=1)
        near = leg[distance < 0.09]
        return near if len(near) >= 20 else leg[np.argsort(distance)[:40]]  # coarse meshes: the closest ones

    def shin_group(self, side: str, within: float = 0.075) -> np.ndarray:
        """The knee and the shin (what rests on the floor when kneeling on the tops of the feet): the lower leg's
        vertices ``within`` metres of the bone, so that a skirt or dress hem does not count. Wide trouser legs
        reach about 0.11 m out at the cuffs."""
        s = self.sides[side]
        knee, ankle = self.rest_pos[s["calf"]], self.rest_pos[s["foot"]]
        calf = self.group(s["calf"])
        axis = _unit(ankle - knee)
        along = np.clip((self.vertices[calf] - knee) @ axis, 0.0, float(np.linalg.norm(ankle - knee)))
        off = np.linalg.norm(self.vertices[calf] - (knee + along[:, None] * axis), axis=1)
        return np.union1d(calf[off < within], self.knee_group(side))


# -- solving -------------------------------------------------------------------------------------------


def _two_bone(start, first: float, second: float, target, pole):
    """Middle and end joint of a two-bone limb from ``start`` reaching for ``target``, bending towards ``pole``."""
    start, target = np.asarray(start, float), np.asarray(target, float)
    reach = target - start
    d = float(np.linalg.norm(reach))
    u = reach / max(d, 1e-9)
    d = min(max(d, abs(first - second) + 1e-4), (first + second) * 0.9999)
    v = _across(pole, u)
    if v is None:
        v = _across((0, 0, 1), u) if abs(u[2]) < 0.9 else _across((0, 1, 0), u)
    cos_a = (first * first + d * d - second * second) / (2 * first * d)
    sin_a = math.sqrt(max(0.0, 1 - cos_a * cos_a))
    middle = start + first * (cos_a * u + sin_a * v)
    return middle, start + d * u, v


def solve(body: Body, pose: BodyPose) -> tuple[np.ndarray, np.ndarray]:
    """``(local, global)`` ``(B, 4, 4)`` bone transforms of ``body`` in ``pose`` (local translations stay the
    skeleton's own, but the root's, so the result works as a clip frame)."""
    sk = body.skeleton
    n = len(sk.names)
    G = np.empty((n, 3, 3))
    P = np.empty((n, 3))
    pelvis = turn(*pose.pelvis)
    chest = pelvis.copy()
    special: dict[int, np.ndarray] = {body.root: pelvis @ body.rest_rot[body.root]}
    for bone, share in zip(body.spine, pose.spine_shares):  # (the thighs hang from Spine: keep its share small)
        chest = chest @ turn(*(share * np.asarray(pose.spine, float)))
        special[bone] = chest @ body.rest_rot[bone]
    special[body.neck] = chest @ body.rest_rot[body.neck]  # the clavicles hang from the neck: it follows the chest
    special[body.head] = chest @ turn(*pose.head) @ body.rest_rot[body.head]
    limbs = {"left": (pose.left_arm, pose.left_leg), "right": (pose.right_arm, pose.right_leg)}
    for side, s in body.sides.items():
        shrug = limbs[side][0].shrug
        special[s["clavicle"]] = chest @ axis_rotation((0, 0, 1), s["sign"] * shrug) @ body.rest_rot[s["clavicle"]]
    ik: dict[int, tuple] = {}
    toes: dict[int, np.ndarray] = {}
    for side, s in body.sides.items():
        arm, leg = limbs[side]
        ik[s["upper"]] = ("arm", side, arm)
        ik[s["thigh"]] = ("leg", side, leg)
        if leg.toes and s["toe"] is not None:  # bent about the foot's left-right axis, towards its top
            axis = _cross(_unit(leg.forward), _across(leg.up, _unit(leg.forward)))
            toes[s["toe"]] = axis_rotation(axis, leg.toes)
    done: set[int] = set()
    for b in range(n):
        parent = sk.parents[b]
        if b in done:
            continue
        P[b] = pose.root if parent < 0 else P[parent] + G[parent] @ body.local_t[b]
        if b in special:
            G[b] = special[b]
        elif b in ik:
            kind, side, limb = ik[b]
            s = body.sides[side]
            if kind == "arm":
                a, m, e, lengths, bend = s["upper"], s["fore"], s["hand"], (s["upper_length"], s["fore_length"]), \
                    s["arm_bend"]
            else:
                a, m, e, lengths, bend = s["thigh"], s["calf"], s["foot"], (s["thigh_length"], s["shin_length"]), \
                    s["leg_bend"]
            middle, end, out = _two_bone(P[b], *lengths, limb.target, limb.pole)
            rest = body.rest_pos
            for bone, child, new_dir in ((a, m, middle - P[b]), (m, e, end - middle)):
                old = _frame(rest[child] - rest[bone], bend)
                new = _frame(new_dir, out)
                G[bone] = new @ old.T @ body.rest_rot[bone]
            P[m] = P[a] + G[a] @ body.local_t[m]
            if kind == "arm":
                f0, u0 = s["hand_forward"], s["hand_back"]
            else:
                f0, u0 = s["foot_forward"], np.array([0.0, 1.0, 0.0])
            G[e] = _frame(limb.forward, limb.up) @ _frame(f0, u0).T @ body.rest_rot[e]
            P[e] = P[m] + G[m] @ body.local_t[e]
            done.update((m, e))
            continue
        else:
            G[b] = G[parent] @ body.local_rot[b]
            if b in toes:
                G[b] = toes[b] @ G[b]
    if pose.flat_hands:  # every finger but the thumb straight along the hand
        for side, s in body.sides.items():
            hand_dir = G[s["hand"]] @ body.rest_rot[s["hand"]].T @ s["hand_forward"]
            for chain in s["fingers"][1:]:
                for k, b in enumerate(chain):  # each joint follows the one before, then is turned straight
                    G[b] = G[sk.parents[b]] @ body.local_rot[b]
                    if k + 1 < len(chain):
                        now = G[b] @ body.rest_rot[b].T @ (body.rest_pos[chain[k + 1]] - body.rest_pos[b])
                        G[b] = _align(now, hand_dir) @ G[b]
                    P[b] = P[sk.parents[b]] + G[sk.parents[b]] @ body.local_t[b]
                for c in body.finger_tips.get(chain[-1], ()):  # finger nubs, if any
                    G[c] = G[sk.parents[c]] @ body.local_rot[c]
                    P[c] = P[sk.parents[c]] + G[sk.parents[c]] @ body.local_t[c]
    glob = np.zeros((n, 4, 4))
    glob[:, :3, :3], glob[:, :3, 3], glob[:, 3, 3] = G, P, 1.0
    local = np.zeros((n, 4, 4))
    for b in range(n):
        parent = sk.parents[b]
        local[b, :3, :3] = G[b] if parent < 0 else G[parent].T @ G[b]
        local[b, :3, 3] = pose.root if parent < 0 else body.local_t[b]
        local[b, 3, 3] = 1.0
    return local, glob


def skin(body: Body, glob: np.ndarray, vertices: np.ndarray | None = None) -> np.ndarray:
    """The mesh vertices of ``body`` (all, or those numbered in ``vertices``) posed by the global transforms ``glob``."""
    if vertices is None:
        return np.concatenate(skin_vertices(body.model, body.model.skinning_matrices(glob)))
    palette = body.model.skinning_matrices(glob)
    m = np.einsum("nk,nkij->nij", body.skin_weights[vertices], palette[body.skin_joints[vertices]])
    return np.einsum("nij,nj->ni", m[:, :3, :3], body.vertices[vertices]) + m[:, :3, 3]


def _dorsum_down(limb: Limb) -> bool:
    """Whether the top of the foot faces the floor (kneeling with the shins down)."""
    return limb.up[1] < -0.5


def settle(body: Body, pose: BodyPose, rounds: int = 4) -> BodyPose:
    """``pose`` with its contacts on the floor: each ``contact`` hand or foot is moved up or down until its lowest
    vertex is at y = 0. With ``knees_down`` the hips move until the knees touch; a leg kneeling on the top of the
    foot moves with them as a whole (hips and ankle together) until the lowest point of its knee, shin or foot
    touches, and its foot then points its toes down until the top of the foot rests on the floor."""
    groups = {f"{side}_{part}": getter(side) for side in body.sides
              for part, getter in (("hand", body.hand_group), ("foot", body.foot_group), ("knee", body.knee_group),
                                   ("shin", body.shin_group), ("cuffs", partial(body.shin_group, within=0.12)))}
    used = np.unique(np.concatenate(list(groups.values())))
    where = {name: np.searchsorted(used, group) for name, group in groups.items()}  # positions within ``used``
    for _ in range(rounds):
        verts = skin(body, solve(body, pose)[1], used)
        changes, largest = {}, 0.0

        def moved(limb, dy):
            return replace(limb, target=(limb.target[0], limb.target[1] + dy, limb.target[2]))

        shins = {f"{side}_leg" for side in body.sides if pose.knees_down and _dorsum_down(getattr(pose, f"{side}_leg"))}
        for name in LIMBS:
            limb = getattr(pose, name)
            if limb.contact and name not in shins:
                side = name.split("_")[0]
                part = where[f"{side}_{'hand' if name.endswith('arm') else 'foot'}"]
                if name.endswith("leg") and _dorsum_down(limb):  # on the top of the foot: the shin may be lowest,
                    part = np.concatenate([part, where[f"{side}_cuffs"]])  # trouser cuffs included
                dy = max(-float(verts[part, 1].min()), -limb.target[1])  # a wrist or ankle never below the floor
                changes[name] = moved(limb, dy)
                largest = max(largest, abs(dy))
        root = pose.root
        if pose.knees_down:
            touching = [np.concatenate([where[f"{side}_shin"], where[f"{side}_foot"]]) if f"{side}_leg" in shins
                        else where[f"{side}_knee"] for side in body.sides]
            dy = -float(verts[np.concatenate(touching), 1].min())
            root = (root[0], root[1] + dy, root[2])
            changes.update({name: moved(getattr(pose, name), dy) for name in shins})
            largest = max(largest, abs(dy))
        pose = replace(pose, root=root, **changes)
        resting = shins | {name for name in ("left_leg", "right_leg")
                           if getattr(pose, name).contact and _dorsum_down(getattr(pose, name))}
        if resting:
            pose = _rest_feet(body, pose, resting)
        if largest < 2e-4:
            break
    return pose


def _rest_feet(body: Body, pose: BodyPose, legs, most: float = 58.0) -> BodyPose:
    """Kneeling ``legs`` point their toes further down (at most ``most`` degrees in all) until the foot touches."""
    for name in sorted(legs):
        side = name.split("_")[0]
        limb = getattr(pose, name)
        down = np.array([0.0, -1.0, 0.0])
        axis = _across(_cross(limb.forward, down), _unit(limb.forward))
        if axis is None:
            continue
        tilted = math.degrees(math.acos(min(1.0, max(-1.0, -float(_unit(limb.up)[1])))))  # already, from flat
        foot = body.foot_group(side)
        for angle in np.arange(0.0, max(0.0, most - tilted) + 1e-9, 2.5):
            r = axis_rotation(axis, angle)
            tried = replace(limb, forward=tuple(r @ limb.forward), up=tuple(r @ limb.up))
            if skin(body, solve(body, replace(pose, **{name: tried}))[1], foot)[:, 1].min() <= 0.003:
                break
        pose = replace(pose, **{name: tried})
    return pose


def lowest(body: Body, pose: BodyPose) -> float:
    """The lowest point of the posed mesh (should not be much below the floor)."""
    return float(skin(body, solve(body, pose)[1])[:, 1].min())


# -- sequences and clips -------------------------------------------------------------------------------


@dataclass(frozen=True)
class Step:
    """Move into ``pose`` over ``seconds`` (eased), then hold it ``hold`` seconds (breathing when ``breathe``).
    ``lift``: limb name -> metres a hand or foot rises on the way (stepping)."""

    pose: BodyPose
    seconds: float = 0.0
    hold: float = 0.0
    breathe: bool = True
    lift: dict = field(default_factory=dict)


def breathe(pose: BodyPose, phase: float, depth: float = 1.0) -> BodyPose:
    """``pose`` at ``phase`` (0..1) of a breath: the chest lifts and opens a little, the head follows."""
    w = 0.5 - 0.5 * math.cos(2 * math.pi * phase)  # 0 at the start of the breath in, 1 at the top
    pitch, yaw, roll = pose.spine
    head = pose.head
    return replace(pose, spine=(pitch - 2.0 * depth * w, yaw, roll), head=(head[0] + 1.0 * depth * w, head[1], head[2]))


def timeline(steps: list[Step]) -> tuple[list[tuple], float]:
    """``[(t0, t1, from_pose, to_pose, kind, step)]`` pieces (kind ``"move"`` or ``"hold"``) and the loop length;
    the loop ends where it began (the first step's pose)."""
    pieces, t = [], 0.0
    current = steps[0].pose
    for k, step in enumerate(steps + [replace(steps[0], hold=0.0)] if steps[-1].pose != steps[0].pose else steps):
        if k and step.seconds > 0:
            pieces.append((t, t + step.seconds, current, step.pose, "move", step))
            t += step.seconds
        current = step.pose
        if step.hold > 0:
            pieces.append((t, t + step.hold, current, current, "hold", step))
            t += step.hold
    return pieces, t


def _smooth(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


def pose_at(pieces, t: float) -> BodyPose:
    """The pose at time ``t`` of a :func:`timeline`."""
    for t0, t1, a, b, kind, step in pieces:
        if t <= t1 or (t0, t1) == pieces[-1][:2]:
            if kind == "hold":  # whole breaths, so the hold ends as it began
                breaths = max(1, round((t1 - t0) / BREATH))
                return breathe(a, (t - t0) / (t1 - t0) * breaths) if step.breathe else a
            return blend(a, b, _smooth((t - t0) / (t1 - t0)), step.lift)
    return pieces[-1][3]


def yoga_clip(model: CharacterModel, name: str, fps: float = FPS) -> AnimationClip:
    """The looping clip of pose ``name`` (see :data:`POSES`) for ``model``."""
    if name not in POSES:
        raise KeyError(f"unknown yoga pose {name!r}; known: {', '.join(POSES)}")
    body = Body(model)
    spec = POSES[name]
    steps = [replace(step, pose=settle(body, step.pose)) for step in spec.steps(body)]
    pieces, duration = timeline(steps)
    frames = max(2, round(duration * fps) + 1)
    times = np.linspace(0.0, duration, frames)
    rotations = np.empty((frames, len(body.names), 4))
    translations = np.empty((frames, len(body.names), 3))
    for f, t in enumerate(times):
        local, _ = solve(body, pose_at(pieces, t))
        rotations[f] = quat_from_matrix(local[:, :3, :3])
        translations[f] = local[:, :3, 3]
    holds = [(t0, t1) for t0, t1, *_rest, kind, _step in pieces if kind == "hold"]
    main = max((p for p in pieces if p[4] == "hold" and p[2] == steps[spec.main].pose), key=lambda p: p[1] - p[0],
               default=None)
    metadata = {"kind": "yoga", "pose": name, "title": spec.title, "speed": 0.0, "turn": 0.0,
                "root_height": float(np.mean(translations[:, body.root, 1])), "holds": holds,
                "hold": (main[0], main[1]) if main else None, "view": spec.view}
    return AnimationClip(f"yoga_{name}", (frames - 1) / duration, list(body.names), rotations, translations,
                         loop=True, root="Bip01", metadata=metadata)


# -- poses ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PoseSpec:
    """A yoga pose: ``steps(body)`` gives the :class:`Step` sequence; ``main`` is the index of the step that is
    the pose itself; ``view``: the side the pose is best seen from (degrees the camera is round from the
    person's front, positive = from their left)."""

    title: str
    steps: callable
    main: int
    view: float = 0.0


def _foot(body: Body, side: str, x: float, z: float, toe_out: float = 0.0, contact: bool = True, pole=None) -> Limb:
    """A foot flat on the floor with the ankle above ``(x, z)``, toes turned out by ``toe_out`` degrees."""
    s = body.sides[side]
    forward = axis_rotation((0, 1, 0), s["sign"] * toe_out) @ s["foot_forward"]
    return Limb((x, s["ankle_height"], z), tuple(pole if pole is not None else forward), tuple(forward), (0, 1, 0),
                contact)


def mountain(body: Body) -> BodyPose:
    """Tadasana: standing tall, feet under the hips, arms along the sides, palms facing the thighs."""
    arms = {}
    for side, s in body.sides.items():
        shoulder = body.rest_pos[s["upper"]]
        wrist = shoulder + body.arm * 0.96 * _unit((s["sign"] * 0.18, -1.0, 0.02))
        arms[f"{side}_arm"] = Limb(tuple(wrist), (s["sign"] * 0.3, 0.0, -1.0), (0.0, -1.0, 0.05),
                                   (s["sign"] * 1.0, 0.0, -0.1))
    legs = {f"{side}_leg": _foot(body, side, body.rest_pos[s["foot"], 0], body.rest_pos[s["foot"], 2])
            for side, s in body.sides.items()}
    root = (0.0, body.root_height - 0.004, float(body.rest_pos[body.root, 2]))
    return BodyPose(root, **arms, **legs)


def _yogi_squat_steps(body: Body) -> list[Step]:
    """Malasana: a deep squat, feet wider than the hips and turned out, heels down, knees wide, the torso upright
    between the knees, the palms together at the chest and the elbows pressing the knees apart."""
    stand = mountain(body)
    leg, width = body.leg, body.hip_width
    feet = {}
    for side, s in body.sides.items():
        x = s["sign"] * (0.5 * width + 0.15 * leg)
        feet[f"{side}_leg"] = _foot(body, side, x, 0.05 * leg, toe_out=35.0,
                                    pole=axis_rotation((0, 1, 0), s["sign"] * 45.0) @ (0, 0, 1))
    hip_height = feet["left_leg"].target[1] + 0.21 * leg  # deep: the hips not far above the heels
    root = (0.0, hip_height + body.hip_drop, -0.20 * leg)
    probe = BodyPose(root, stand.left_arm, stand.right_arm, feet["left_leg"], feet["right_leg"], pelvis=(30.0, 0, 0),
                     spine=(-14.0, 0, 0))  # the pelvis tips forward, the back stays long and upright
    _, glob = solve(body, probe)
    shoulders = 0.5 * (glob[body.sides["left"]["upper"], :3, 3] + glob[body.sides["right"]["upper"], :3, 3])
    heart = shoulders + np.array([0.0, -0.30 * body.torso, 0.24 * body.torso])  # palms together at the sternum
    arms = {}
    for side, s in body.sides.items():
        wrist = heart + np.array([s["sign"] * 0.03, 0.0, 0.0])
        knee = glob[s["calf"], :3, 3]
        inside = knee - np.array([s["sign"] * 0.08, -0.02, 0.0])  # the elbow presses the inner knee
        pole = inside - 0.5 * (glob[s["upper"], :3, 3] + wrist)
        arms[f"{side}_arm"] = Limb(tuple(wrist), tuple(pole), (-s["sign"] * 0.1, 1.0, 0.2), (s["sign"] * 1.0, 0.0, 0.0))
    squat = replace(probe, head=(-8.0, 0, 0), **arms)
    return [Step(stand, hold=1.5, breathe=False), Step(squat, seconds=3.0, hold=8.0), Step(stand, seconds=3.0)]


def _shoulders(body: Body, pose: BodyPose) -> np.ndarray:
    """The point between the shoulder joints in ``pose``."""
    _, glob = solve(body, pose)
    return 0.5 * (glob[body.sides["left"]["upper"], :3, 3] + glob[body.sides["right"]["upper"], :3, 3])


def _pitch_for_shoulders(body: Body, pose: BodyPose, height: float, low: float = 40.0, high: float = 130.0) -> float:
    """The pelvis pitch (degrees, forward) that puts the shoulders of ``pose`` at ``height`` above the floor."""
    def at(pitch):
        return _shoulders(body, replace(pose, pelvis=(pitch, *pose.pelvis[1:])))[1] - height

    if at(low) * at(high) > 0:  # out of range: whichever end comes closer
        return low if abs(at(low)) < abs(at(high)) else high
    for _ in range(30):
        middle = 0.5 * (low + high)
        if at(low) * at(middle) <= 0:
            high = middle
        else:
            low = middle
    return 0.5 * (low + high)


def _hands_down(body: Body, shoulders: np.ndarray, ahead: float = 0.0, width: float = 1.0) -> dict:
    """Both hands flat on the floor below ``shoulders`` (plus ``ahead`` metres), fingers forward, elbows back."""
    arms = {}
    for side, s in body.sides.items():
        x = shoulders[0] + s["sign"] * 0.5 * body.shoulder_width * width
        arms[f"{side}_arm"] = Limb((x, 0.05, shoulders[2] + ahead), (s["sign"] * 0.25, 0.0, -1.0), (0.0, 0.0, 1.0),
                                   (0.0, 1.0, 0.0), contact=True)
    return arms


def _shin_down(body: Body, side: str, knee_z: float) -> Limb:
    """A kneeling leg: the knee at ``knee_z`` below the hip, the shin and the top of the foot on the floor."""
    s = body.sides[side]
    return Limb((s["sign"] * 0.5 * body.hip_width, 0.06, knee_z - s["shin_length"]), (0.0, -1.0, 0.25),
                (0.0, 0.0, -1.0), (0.0, -1.0, 0.0), contact=True)


def _toes_tucked(body: Body, side: str, ankle_z: float) -> Limb:
    """A kneeling leg on the ball of the foot: the foot upright behind the knee, the toes bent forward on the floor."""
    s = body.sides[side]
    return Limb((s["sign"] * 0.5 * body.hip_width, 0.16, ankle_z), (0.0, -1.0, 0.25), (0.0, -0.96, -0.28),
                (0.0, -0.28, 0.96), contact=True, toes=70.0)


def table_knees(body: Body) -> float:
    """Where the knees go on hands and knees: the feet (tops down) end up where they stand."""
    return float(body.rest_pos[body.sides["left"]["foot"], 2]) + body.sides["left"]["shin_length"]


def _reach_floor(body: Body, pose: BodyPose, ahead: float = 0.0, low: float = 40.0, high: float = 150.0) -> BodyPose:
    """``pose`` leaning (pelvis pitch) until straight arms reach the floor, the hands flat below the shoulders."""
    pose = replace(pose, pelvis=(_pitch_for_shoulders(body, pose, HAND_REACH * body.arm + 0.05, low, high), 0.0, 0.0),
                   flat_hands=True)
    return replace(pose, **_hands_down(body, _shoulders(body, pose), ahead=ahead))


def tabletop(body: Body, spine=(0.0, 0.0, 0.0), head=(-12.0, 0.0, 0.0), shares=(0.1, 0.45, 0.45),
             tucked: bool = False, hips_back: float = 0.0) -> BodyPose:
    """On hands and knees: knees under the hips, hands under the shoulders, the back flat (or bent by ``spine``);
    ``tucked``: on the balls of the feet instead of the tops; ``hips_back``: the hips that far behind the knees
    (sitting back towards the heels)."""
    stand = mountain(body)
    knee_z = table_knees(body)
    thigh = body.sides["left"]["thigh_length"]
    if tucked:
        legs = {f"{side}_leg": _toes_tucked(body, side, knee_z - 0.85 * body.sides[side]["shin_length"])
                for side in body.sides}
    else:
        legs = {f"{side}_leg": _shin_down(body, side, knee_z) for side in body.sides}
    hip_height = 0.06 + math.sqrt(max((0.97 * thigh) ** 2 - hips_back ** 2, 0.01))  # the knees on the floor
    probe = BodyPose((0.0, hip_height + body.hip_drop, knee_z - hips_back), stand.left_arm, stand.right_arm,
                     pelvis=(90.0, 0, 0), spine=spine, head=head, knees_down=True, flat_hands=True,
                     spine_shares=shares, **legs)
    probe = settle(body, probe)  # the knees, shins and feet on the floor first: that sets the hip height
    return _reach_floor(body, probe)


def crouch(body: Body) -> BodyPose:
    """Half way down from standing: knees bent, the torso folded forward, the hands on the floor below the shoulders."""
    stand = mountain(body)
    leg = body.leg
    root = (0.0, stand.left_leg.target[1] + 0.40 * leg, -0.12 * leg)
    probe = replace(stand, root=root, spine=(25.0, 0, 0), head=(-15.0, 0, 0))
    for side, s in body.sides.items():  # the knees bend forward over the toes
        probe = replace(probe, **{f"{side}_leg": replace(getattr(probe, f"{side}_leg"), pole=(s["sign"] * 0.2, 0.0, 1.0))})
    return _reach_floor(body, probe, ahead=0.05)


def kneel_tucked(body: Body) -> BodyPose:
    """Down on the knees from the crouch: knees on the floor in front of the feet, toes tucked, sitting back."""
    return tabletop(body, spine=(15.0, 0.0, 0.0), tucked=True, hips_back=0.22 * body.leg)


def _down_to_table(body: Body) -> list[Step]:
    """From standing to hands and knees (tops of the feet down): crouch, knees down, walk the hands forward, untuck."""
    hands = {"left_arm": 0.06, "right_arm": 0.06}
    feet = {"left_leg": 0.05, "right_leg": 0.05}
    heels = {"left_leg": 0.075, "right_leg": 0.075, **hands}  # the heels rise while the toes tuck under
    return [Step(mountain(body), hold=1.0, breathe=False), Step(crouch(body), seconds=2.0),
            Step(kneel_tucked(body), seconds=1.5, lift=heels), Step(tabletop(body, tucked=True), seconds=1.2, lift=hands),
            Step(tabletop(body), seconds=1.2, hold=1.0, breathe=False, lift=feet)]


def _up_from(steps: list[Step]) -> list[Step]:
    """The way back up: ``steps`` in reverse, each move as long as on the way down."""
    back = []
    for k in range(len(steps) - 2, -1, -1):
        back.append(replace(steps[k], seconds=steps[k + 1].seconds, hold=steps[k].hold if k else 0.0,
                            breathe=False, lift=steps[k + 1].lift))
    return back


def _cow_steps(body: Body) -> list[Step]:
    """Bitilasana: on hands and knees, the belly sinks, the tailbone and the chest lift, the gaze goes forward."""
    down = _down_to_table(body)
    table = down[-1].pose
    cow = tabletop(body, spine=(-55.0, 0.0, 0.0), head=(-48.0, 0.0, 0.0), shares=(0.3, 0.4, 0.3))
    cow = replace(cow, root=table.root, **{name: getattr(table, name) for name in LIMBS})  # hands and knees stay
    hand = table.left_arm.target
    forward = _shoulders(body, cow)[2] - hand[2]  # the chest moves back: the arms slant a little
    height = hand[1] + math.sqrt(max((HAND_REACH * body.arm) ** 2 - forward ** 2, 0.0))  # as high as the arms reach
    cow = replace(cow, pelvis=(_pitch_for_shoulders(body, cow, height, low=60.0, high=150.0), 0, 0))
    return down + [Step(cow, seconds=2.5, hold=9.0), Step(table, seconds=2.5, hold=1.0, breathe=False)] + _up_from(down)


ARM_STRAIGHT = 0.995  # an arm held out straight reaches this share of its length (the elbow barely bent)


def _arms_out(body: Body, pose: BodyPose, lift: float = 0.0) -> dict:
    """Both arms straight out to the sides at shoulder height (raised by ``lift`` degrees), palms down."""
    _, glob = solve(body, pose)
    chest = glob[body.spine[2], :3, :3] @ body.rest_rot[body.spine[2]].T
    up, forward = chest @ (0.0, 1.0, 0.0), chest @ (0.0, 0.0, 1.0)
    arms = {}
    for side, s in body.sides.items():
        out = chest @ (s["sign"] * 1.0, 0.0, 0.0)
        direction = _unit(out * math.cos(math.radians(lift)) + up * math.sin(math.radians(lift)))
        wrist = glob[s["upper"], :3, 3] + ARM_STRAIGHT * (s["upper_length"] + s["fore_length"]) * direction
        arms[f"{side}_arm"] = Limb(tuple(wrist), tuple(-forward - 0.3 * up), tuple(direction), tuple(up))
    return arms


def _arms_up(body: Body, pose: BodyPose, apart: float = 1.0, shrug: float = 14.0) -> dict:
    """Both arms straight overhead (as wide as the shoulders times ``apart``), palms facing each other."""
    pose = replace(pose, left_arm=replace(pose.left_arm, shrug=shrug), right_arm=replace(pose.right_arm, shrug=shrug))
    _, glob = solve(body, pose)
    chest = glob[body.spine[2], :3, :3] @ body.rest_rot[body.spine[2]].T  # the chest's turn from standing
    up, forward = chest @ (0.0, 1.0, 0.0), chest @ (0.0, 0.0, 1.0)
    arms = {}
    for side, s in body.sides.items():
        shoulder = glob[s["upper"], :3, 3]
        out = chest @ (s["sign"] * 1.0, 0.0, 0.0)
        direction = _unit(up + 0.08 * (apart - 1.0) * out + 0.06 * forward)
        wrist = shoulder + ARM_STRAIGHT * (s["upper_length"] + s["fore_length"]) * direction
        arms[f"{side}_arm"] = Limb(tuple(wrist), tuple(out - 0.5 * forward), tuple(up), tuple(out), shrug=shrug)
    return arms


def _warrior1_steps(body: Body) -> list[Step]:
    """Virabhadrasana I: a long lunge, the front (left) knee bent over its ankle, the back leg straight with the
    foot turned out and the heel down, the hips square to the front, the arms raised overhead, the chest lifted."""
    stand = mountain(body)
    left, right = body.sides["left"], body.sides["right"]
    front = stand.left_leg
    knee_height = left["ankle_height"] + 0.97 * left["shin_length"]  # the front shin upright
    hip_z = front.target[2] - 0.9 * left["thigh_length"]  # the front thigh about level
    hip_height = knee_height + 0.12 * left["thigh_length"]
    drop = hip_height - right["ankle_height"]
    back_z = hip_z - math.sqrt(max((0.985 * (right["thigh_length"] + right["shin_length"])) ** 2 - drop ** 2, 0.0))
    back = _foot(body, "right", right["sign"] * (0.5 * body.hip_width + 0.03), back_z, toe_out=45.0)
    back = replace(back, pole=(right["sign"] * 0.4, 0.0, 1.0))
    front = replace(front, pole=(0.15, 0.0, 1.0))
    lunge = replace(stand, root=(0.0, hip_height + body.hip_drop, hip_z), left_leg=front, right_leg=back,
                    pelvis=(4.0, 0.0, 0.0), spine=(-14.0, 0.0, 0.0), head=(-12.0, 0.0, 0.0))
    warrior = replace(lunge, **_arms_up(body, lunge))
    sweep = replace(lunge, **_arms_out(body, lunge, lift=10.0))  # the arms rise out to the sides, then overhead
    step_back = {"right_leg": 0.08}
    return [Step(stand, hold=1.5, breathe=False), Step(sweep, seconds=2.2, lift=step_back), Step(warrior, seconds=1.3, hold=9.0),
            Step(sweep, seconds=1.3), Step(stand, seconds=2.2, lift=step_back)]


def downward_dog(body: Body, hands: BodyPose, trunk: float = 48.0) -> BodyPose:
    """Adho Mukha Svanasana with the hands of ``hands``: hips high, arms and back in one line ``trunk`` degrees
    up from the floor, the legs straight and the heels down, the head between the arms."""
    left = body.sides["left"]
    hand = np.asarray(hands.left_arm.target, float)
    reach = ARM_STRAIGHT * (left["upper_length"] + left["fore_length"])
    legs = ARM_STRAIGHT * (left["thigh_length"] + left["shin_length"])  # straight legs too
    down_line = np.array([0.0, -math.sin(math.radians(trunk)), math.cos(math.radians(trunk))])  # hips to hands
    hips = hand - (reach + body.torso) * down_line
    arms = {}
    for name in ("left_arm", "right_arm"):
        sign = 1.0 if name == "left_arm" else -1.0  # the elbow points up and out, its crease forward
        arms[name] = replace(getattr(hands, name), pole=(sign * 0.5, 0.7, -0.6))
    dog = None
    for _ in range(6):  # slide the hips along the line until straight arms just reach the hands
        rise = hips[1] - left["ankle_height"]
        ankle_z = hips[2] - math.sqrt(max(legs ** 2 - rise ** 2, 0.0))
        feet = {f"{side}_leg": replace(_foot(body, side, s["sign"] * 0.5 * body.hip_width, ankle_z),
                                       pole=(s["sign"] * 0.1, 0.3, 1.0)) for side, s in body.sides.items()}
        dog = BodyPose((0.0, hips[1] + body.hip_drop, hips[2]), pelvis=(90.0 + trunk, 0, 0), spine=(-8.0, 0.0, 0.0),
                       head=(10.0, 0.0, 0.0), flat_hands=True, **arms, **feet)
        _, glob = solve(body, dog)
        miss = float(np.linalg.norm(glob[left["upper"], :3, 3] - hand)) - reach
        if abs(miss) < 1e-3:
            break
        hips = hips + miss * down_line
    return dog


def _downward_dog_steps(body: Body) -> list[Step]:
    """From standing down to hands and knees, the toes tucked, the hips lifting up and back into the dog."""
    down = _down_to_table(body)
    table = settle(body, down[-1].pose)  # the hands where they will be, so the arms end up straight
    tucked = tabletop(body, tucked=True)
    tucked = replace(tucked, left_arm=table.left_arm, right_arm=table.right_arm)
    dog = downward_dog(body, table)
    feet = {"left_leg": 0.05, "right_leg": 0.05}
    step = {"left_leg": 0.08, "right_leg": 0.08}  # the feet step back while the heels come down
    into = [Step(tucked, seconds=1.2, lift=feet), Step(dog, seconds=2.5, hold=9.0, lift=step)]
    return down + into + [Step(tucked, seconds=2.5, lift=step), Step(table, seconds=1.2, hold=1.0, breathe=False,
                                                                     lift=feet)] + _up_from(down)


def _warrior2_steps(body: Body) -> list[Step]:
    """Virabhadrasana II: the feet wide apart, the front (left) foot turned out and its knee bent over the ankle,
    the back leg straight with the foot turned slightly in, hips and chest open to the front, the arms straight
    out at shoulder height, palms down, the gaze over the front hand."""
    stand = mountain(body)
    left, right = body.sides["left"], body.sides["right"]
    z = stand.left_leg.target[2]
    knee_height = left["ankle_height"] + 0.97 * left["shin_length"]  # the front shin upright
    hip_height = knee_height + 0.1 * left["thigh_length"]  # the front thigh about level
    front_x = 0.5 * body.hip_width + 0.95 * left["thigh_length"]
    drop = hip_height - right["ankle_height"]
    back_x = -0.5 * body.hip_width - math.sqrt(max((ARM_STRAIGHT * (right["thigh_length"] + right["shin_length"])) ** 2
                                                    - drop ** 2, 0.0))
    front = Limb((front_x, left["ankle_height"], z), (1.0, 0.0, 0.15), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), contact=True)
    back_toes = axis_rotation((0, 1, 0), 12.0) @ (0.0, 0.0, 1.0)  # turned in a little, towards the front foot
    back = Limb((back_x, right["ankle_height"], z), (0.0, 0.0, 1.0), tuple(back_toes), (0.0, 1.0, 0.0), contact=True)
    wide = replace(stand, root=(0.0, hip_height + body.hip_drop, float(body.rest_pos[body.root, 2])), left_leg=front,
                   right_leg=back, spine=(-4.0, 0.0, 0.0), head=(-4.0, 72.0, 0.0))
    warrior = replace(wide, **_arms_out(body, wide))
    steps = {"left_leg": 0.03, "right_leg": 0.08}  # the back foot steps out, the front one turns
    return [Step(stand, hold=1.5, breathe=False), Step(warrior, seconds=3.0, hold=9.0, lift=steps),
            Step(stand, seconds=3.0, lift=steps)]


def plank(body: Body, hands: BodyPose) -> BodyPose:
    """High plank with the hands of ``hands``: straight arms upright under the shoulders, the body one straight line
    down to the heels, on the balls of the feet."""
    left = body.sides["left"]
    hand = np.asarray(hands.left_arm.target, float)
    shoulder = hand + np.array([0.0, ARM_STRAIGHT * (left["upper_length"] + left["fore_length"]), 0.0])
    legs = ARM_STRAIGHT * (left["thigh_length"] + left["shin_length"])
    ankle_height = 0.14 * body.leg  # on the balls of the feet, the heels up
    arms = {name: replace(getattr(hands, name), pole=(0.0, 0.0, -1.0)) for name in ("left_arm", "right_arm")}
    base = BodyPose((0.0, 0.0, 0.0), pelvis=(80.0, 0.0, 0.0), head=(-8.0, 0.0, 0.0), flat_hands=True, **arms,
                    left_leg=hands.left_leg, right_leg=hands.right_leg)

    def place(pitch):  # the shoulders right above the hands, the straight legs down to the floor
        _, glob = solve(body, replace(base, pelvis=(pitch, 0.0, 0.0)))
        root = shoulder - (glob[left["upper"], :3, 3] - glob[body.root, :3, 3])
        rise = root[1] - body.hip_drop - ankle_height
        ankle_z = root[2] - math.sqrt(max(legs ** 2 - rise ** 2, 0.0))
        feet = {f"{side}_leg": replace(_toes_tucked(body, side, ankle_z),
                                       target=(s["sign"] * 0.5 * body.hip_width, ankle_height, ankle_z),
                                       pole=(0.0, -1.0, 0.0), forward=tuple(_unit((0.0, -0.75, -0.66))),
                                       up=tuple(_unit((0.0, -0.66, 0.75))), toes=55.0)
                for side, s in body.sides.items()}
        pose = replace(base, root=tuple(root), pelvis=(pitch, 0.0, 0.0), **feet)
        trunk = math.atan2(shoulder[1] - root[1], shoulder[2] - root[2])  # the slope hips -> shoulders
        legs_slope = math.asin(min(1.0, max(-1.0, rise / legs)))  # the slope heels -> hips
        return pose, trunk - legs_slope

    low, high = 55.0, 110.0
    for _ in range(30):  # the pitch that puts the hips on the line from the heels to the shoulders
        middle = 0.5 * (low + high)
        if place(middle)[1] * place(low)[1] <= 0:
            high = middle
        else:
            low = middle
    return place(0.5 * (low + high))[0]


def upward_dog(body: Body, hands: BodyPose) -> BodyPose:
    """Urdhva Mukha Svanasana with the hands of ``hands``: the legs straight back with the tops of the feet down
    and the knees off the floor, the hips low, the chest lifted on straight arms through a backbend, gaze forward."""
    left = body.sides["left"]
    hand = np.asarray(hands.left_arm.target, float)
    shoulder_height = hand[1] + ARM_STRAIGHT * (left["upper_length"] + left["fore_length"])  # arms upright
    hip_height = 0.32 * body.leg  # low, but the thighs off the floor
    legs = ARM_STRAIGHT * (left["thigh_length"] + left["shin_length"])
    arms = {name: replace(getattr(hands, name), pole=(0.0, 0.0, -1.0)) for name in ("left_arm", "right_arm")}
    feet = {f"{side}_leg": Limb((s["sign"] * 0.5 * body.hip_width, 0.06, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0),
                                (0.0, -1.0, 0.0), contact=True) for side, s in body.sides.items()}
    base = BodyPose((0.0, hip_height + body.hip_drop, 0.0), pelvis=(70.0, 0, 0), spine=(-50.0, 0.0, 0.0),
                    head=(-22.0, 0.0, 0.0), flat_hands=True, spine_shares=(0.35, 0.35, 0.3), **arms, **feet)

    def place(pitch):  # the hips where the shoulders come out right above the hands
        _, glob = solve(body, replace(base, pelvis=(pitch, 0.0, 0.0)))
        offset = glob[left["upper"], :3, 3] - glob[body.root, :3, 3]
        root = (0.0, base.root[1], float(hand[2] - offset[2]))
        rise = base.root[1] - body.hip_drop - feet["left_leg"].target[1]
        ankle_z = root[2] - math.sqrt(max(legs ** 2 - rise ** 2, 0.0))
        legs_back = {name: replace(limb, target=(limb.target[0], limb.target[1], ankle_z)) for name, limb in feet.items()}
        return replace(base, root=root, pelvis=(pitch, 0.0, 0.0), **legs_back), base.root[1] + offset[1]

    low, high = 30.0, 110.0
    for _ in range(30):  # the pitch that lifts the shoulders to straight-arm height
        middle = 0.5 * (low + high)
        if (place(middle)[1] - shoulder_height) * (place(low)[1] - shoulder_height) <= 0:
            high = middle
        else:
            low = middle
    return place(0.5 * (low + high))[0]


def _upward_dog_steps(body: Body) -> list[Step]:
    """From standing down to hands and knees, a step back to plank, then the hips lower, the feet roll onto their
    tops and the chest lifts."""
    down = _down_to_table(body)
    table = settle(body, down[-1].pose)
    board = plank(body, table)
    dog = upward_dog(body, table)
    back = {"left_leg": 0.12, "right_leg": 0.12}  # stepping back from the knees
    roll = {"left_leg": 0.05, "right_leg": 0.05}  # the feet roll over the toes
    return down + [Step(board, seconds=2.0, hold=0.5, breathe=False, lift=back), Step(dog, seconds=2.5, hold=9.0, lift=roll),
                   Step(board, seconds=2.5, hold=0.5, breathe=False, lift=roll),
                   Step(table, seconds=2.0, hold=1.0, breathe=False, lift=back)] + _up_from(down)


def _lean(body: Body, pose: BodyPose) -> float:
    """How far (degrees, forward) the line from the hips to the shoulders leans from upright in ``pose``."""
    _, glob = solve(body, pose)
    up = _shoulders(body, pose) - glob[body.root, :3, 3]
    return math.degrees(math.atan2(up[2], up[1]))


def _pitch_for_lean(body: Body, pose: BodyPose, lean: float, low: float = -20.0, high: float = 100.0) -> float:
    """The pelvis pitch (degrees, forward) that leans the hips-to-shoulders line of ``pose`` by ``lean`` degrees."""
    def at(pitch):
        return _lean(body, replace(pose, pelvis=(pitch, *pose.pelvis[1:]))) - lean

    for _ in range(30):
        middle = 0.5 * (low + high)
        if at(low) * at(middle) <= 0:
            high = middle
        else:
            low = middle
    return 0.5 * (low + high)


def _hips_at(body: Body, pose: BodyPose, hips) -> BodyPose:
    """``pose`` moved so that the point between its hip joints is at ``hips``."""
    _, glob = solve(body, pose)
    middle = 0.5 * (glob[body.sides["left"]["thigh"], :3, 3] + glob[body.sides["right"]["thigh"], :3, 3])
    return replace(pose, root=tuple(np.asarray(hips, float) + glob[body.root, :3, 3] - middle))


def chair(body: Body, lean: float = 38.0, shin_tilt: float = 25.0, knee_bend: float = 100.0) -> BodyPose:
    """Utkatasana: the feet where they stand, the knees bent ``knee_bend`` degrees with the shins ``shin_tilt``
    degrees forward and the heels down, the hips back as if sitting on a chair, the long back leaning ``lean``
    degrees forward, the arms straight up alongside the ears in line with the back, the gaze forward."""
    stand = mountain(body)
    thigh = 0.5 * (body.sides["left"]["thigh_length"] + body.sides["right"]["thigh_length"])
    shin = 0.5 * (body.sides["left"]["shin_length"] + body.sides["right"]["shin_length"])
    thigh_down = math.radians(shin_tilt + math.degrees(math.asin(math.cos(math.radians(knee_bend)))))
    tilt = math.radians(shin_tilt)
    ankle = np.array([0.0, stand.left_leg.target[1], 0.5 * (stand.left_leg.target[2] + stand.right_leg.target[2])])
    knee = ankle + shin * np.array([0.0, math.cos(tilt), math.sin(tilt)])
    hips = knee + thigh * np.array([0.0, math.sin(thigh_down), -math.cos(thigh_down)])
    legs = {name: replace(getattr(stand, name), pole=(0.0, 0.0, 1.0)) for name in ("left_leg", "right_leg")}
    pose = replace(stand, spine=(-6.0, 0.0, 0.0), head=(-30.0, 0.0, 0.0), **legs)  # the chest lifts, the gaze ahead;
    pose = replace(pose, pelvis=(_pitch_for_lean(body, pose, lean), 0.0, 0.0))  # the knees point straight ahead
    pose = _hips_at(body, pose, hips)
    return replace(pose, **_arms_up(body, pose))


def _chair_steps(body: Body) -> list[Step]:
    """From standing the arms sweep out and up overhead, then the knees bend and the hips sit back into the chair;
    the way back the same in reverse."""
    stand = mountain(body)
    sweep = replace(stand, **_arms_out(body, stand, lift=10.0))
    reach = replace(stand, **_arms_up(body, stand))  # Urdhva Hastasana: standing, the arms overhead
    seat = chair(body)
    return [Step(stand, hold=1.5, breathe=False), Step(sweep, seconds=1.5), Step(reach, seconds=1.2, hold=0.5, breathe=False),
            Step(seat, seconds=2.5, hold=9.0), Step(reach, seconds=2.5, hold=0.5, breathe=False), Step(sweep, seconds=1.2),
            Step(stand, seconds=1.5)]


def _cactus_arms(body: Body, pose: BodyPose) -> dict:
    """Both arms in 'cactus': the upper arms straight out to the sides at shoulder height, the elbows bent at a
    right angle, the forearms upright, the palms facing forward."""
    _, glob = solve(body, pose)
    chest = glob[body.spine[2], :3, :3] @ body.rest_rot[body.spine[2]].T
    up, forward = chest @ (0.0, 1.0, 0.0), chest @ (0.0, 0.0, 1.0)
    arms = {}
    for side, s in body.sides.items():
        out = chest @ (s["sign"] * 1.0, 0.0, 0.0)
        elbow = glob[s["upper"], :3, 3] + s["upper_length"] * out
        wrist = elbow + s["fore_length"] * up
        arms[f"{side}_arm"] = Limb(tuple(wrist), tuple(out - up), tuple(up), tuple(-forward))
    return arms


def _wide_feet(body: Body, width: float, toe_out: float, sides=("left", "right")) -> dict:
    """The feet of ``sides`` flat ``width`` metres apart (ankle to ankle), turned out by ``toe_out`` degrees; the
    other feet where they stand."""
    stand = mountain(body)
    feet = {}
    for side, s in body.sides.items():
        if side in sides:
            toes = axis_rotation((0, 1, 0), s["sign"] * toe_out) @ (0.0, 0.0, 1.0)
            feet[f"{side}_leg"] = replace(_foot(body, side, s["sign"] * 0.5 * width, stand.left_leg.target[2]),
                                          forward=tuple(toes), pole=tuple(toes))
        else:
            feet[f"{side}_leg"] = getattr(stand, f"{side}_leg")
    return feet


def _straddle(body: Body, feet: dict, pose: BodyPose) -> BodyPose:
    """``pose`` standing on ``feet`` with the legs long (the knees soft), the hips centred between the ankles."""
    left, right = feet["left_leg"].target, feet["right_leg"].target
    centre = 0.5 * (left[0] + right[0])
    across = 0.5 * (left[0] - right[0]) - 0.5 * body.hip_width
    reach = 0.985 * body.leg
    height = left[1] + math.sqrt(max(reach ** 2 - across ** 2, 0.0))
    moved = _hips_at(body, replace(pose, **feet), (centre, height, 0.5 * (left[2] + right[2])))
    shift = np.asarray(moved.root) - np.asarray(pose.root)  # the hands go along with the hips
    arms = {name: replace(getattr(pose, name), target=tuple(np.asarray(getattr(pose, name).target) + shift))
            for name in ("left_arm", "right_arm")}
    return replace(moved, **arms)


def goddess(body: Body, width: float = 1.05, toe_out: float = 50.0, thigh_down: float = 10.0) -> BodyPose:
    """Utkata Konasana: the feet ``width`` leg lengths apart and turned out ``toe_out`` degrees, the knees bent out
    over the toes with the shins upright, the thighs ``thigh_down`` degrees from level, the torso upright, the arms
    in cactus, the gaze forward."""
    feet = _wide_feet(body, width * body.leg, toe_out)
    hips, legs = [], {}
    for side, s in body.sides.items():
        foot = feet[f"{side}_leg"]
        knee = np.asarray(foot.target, float) + np.array([0.0, s["shin_length"], 0.0])  # right above the ankle
        hip_x = s["sign"] * 0.5 * body.hip_width
        level = s["thigh_length"] * math.cos(math.radians(thigh_down))  # the thigh's reach along the floor
        back = math.sqrt(max(level ** 2 - (knee[0] - hip_x) ** 2, 0.0))  # the hips sit back between the heels
        hip = np.array([hip_x, knee[1] + s["thigh_length"] * math.sin(math.radians(thigh_down)), knee[2] - back])
        hips.append(hip)
        legs[f"{side}_leg"] = replace(foot, pole=tuple(knee - hip))  # the knee points the way the thigh goes
    pose = replace(mountain(body), spine=(-4.0, 0.0, 0.0), head=(-2.0, 0.0, 0.0), **legs)
    pose = replace(pose, pelvis=(_pitch_for_lean(body, pose, 2.0), 0.0, 0.0))  # upright over the hips
    pose = _hips_at(body, pose, 0.5 * (hips[0] + hips[1]))
    return replace(pose, **_cactus_arms(body, pose))


def _goddess_steps(body: Body) -> list[Step]:
    """From standing the left foot steps out wide and turns out, then the right one while the arms rise out to the
    sides, then the knees bend out over the toes and the elbows bend into cactus; back the same way."""
    stand = mountain(body)
    width, toe_out = 1.05 * body.leg, 50.0
    half = _straddle(body, _wide_feet(body, width, toe_out, sides=("left",)), stand)
    wide = _straddle(body, _wide_feet(body, width, toe_out), stand)
    wide = replace(wide, **_arms_out(body, wide))
    pose = goddess(body, width=width / body.leg, toe_out=toe_out)
    return [Step(stand, hold=1.5, breathe=False), Step(half, seconds=1.3, lift={"left_leg": 0.08}),
            Step(wide, seconds=1.6, lift={"right_leg": 0.08}), Step(pose, seconds=2.5, hold=9.0),
            Step(wide, seconds=2.5), Step(half, seconds=1.6, lift={"right_leg": 0.08}),
            Step(stand, seconds=1.3, lift={"left_leg": 0.08})]


# -- standing on one leg -------------------------------------------------------------------------------

FAR = 3.0  # a limb held straight reaches for a point this many limb lengths away (out of reach, the solver keeps it
# straight), so that it stays straight while it swings from one such point to another: the straight line between
# them never comes within its reach

# shares of the body's mass (after Dempster): trunk, head; per side upper arm, forearm and hand, thigh, shank and foot
_MASS = {"trunk": 0.497, "head": 0.081, "upper": 0.028, "fore": 0.022, "thigh": 0.100, "shank": 0.061}


def _centre_of_mass(body: Body, glob: np.ndarray) -> np.ndarray:
    """About where the centre of mass of ``body`` posed by the global transforms ``glob`` is (body segments)."""
    p = glob[:, :3, 3]
    head_up = glob[body.head, :3, :3] @ body.rest_rot[body.head].T @ (0.0, 1.0, 0.0)
    total = _MASS["trunk"] * 0.5 * (p[body.root] + p[body.neck]) + _MASS["head"] * (p[body.head] + 0.07 * head_up)
    for s in body.sides.values():
        total = total + _MASS["upper"] * 0.5 * (p[s["upper"]] + p[s["fore"]]) \
            + _MASS["fore"] * (0.4 * p[s["fore"]] + 0.6 * p[s["hand"]]) \
            + _MASS["thigh"] * (0.57 * p[s["thigh"]] + 0.43 * p[s["calf"]]) \
            + _MASS["shank"] * (0.4 * p[s["calf"]] + 0.6 * p[s["foot"]])
    return total / (_MASS["trunk"] + _MASS["head"] + 2 * sum(_MASS[k] for k in ("upper", "fore", "thigh", "shank")))


def _sole_point(body: Body, side: str, along: float = 0.5) -> np.ndarray:
    """A point ``(x, z)`` of the ``side`` foot standing where it stands at rest (and in :func:`mountain`): across the
    middle, ``along`` of the way from the heel to the toes."""
    foot = body.vertices[body.foot_group(side)][:, [0, 2]]
    low, high = foot.min(axis=0), foot.max(axis=0)
    return np.array([0.5 * (low[0] + high[0]), low[1] + along * (high[1] - low[1])])


def _stand_on(body: Body, pose: BodyPose, reach: dict) -> BodyPose:
    """``pose`` with the hips as high as the legs in ``reach`` (side -> ``(ankle, share)``) reach those ankle
    points with that share of their length (the hips at the height the shortest of them allows)."""
    _, glob = solve(body, pose)
    rises = []
    for side, (ankle, share) in reach.items():
        s = body.sides[side]
        hip = glob[s["thigh"], :3, 3]
        ankle = np.asarray(ankle, float)
        flat = (hip - ankle)[[0, 2]]
        length = share * (s["thigh_length"] + s["shin_length"])
        rises.append(ankle[1] + math.sqrt(max(length ** 2 - float(flat @ flat), 0.0)) - hip[1])
    return replace(pose, root=(pose.root[0], pose.root[1] + min(rises), pose.root[2]))


def _reach_far(start, direction, length: float) -> tuple:
    """The target of a limb from ``start`` held straight along ``direction`` (:data:`FAR`)."""
    return tuple(np.asarray(start, float) + FAR * length * _unit(direction))


def _balanced(body: Body, build, side: str, along: float = 0.5, rounds: int = 3) -> BodyPose:
    """The pose ``build(x, z)`` (the hips above ``(x, z)``) with ``x, z`` found so that the centre of mass is above
    the ``side`` foot (``along`` of the way from the heel to the toes): standing on that foot."""
    centre = _sole_point(body, side, along)
    x, z = float(centre[0]), float(centre[1])
    for _ in range(rounds):
        com = _centre_of_mass(body, solve(body, build(x, z))[1])
        x, z = x + float(centre[0] - com[0]), z + float(centre[1] - com[2])
    return build(x, z)


def _chest_frame(body: Body, pose: BodyPose) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(global transforms, the chest's turn from standing, the point between the shoulders)`` of ``pose``."""
    _, glob = solve(body, pose)
    chest = glob[body.spine[2], :3, :3] @ body.rest_rot[body.spine[2]].T
    return glob, chest, 0.5 * (glob[body.sides["left"]["upper"], :3, 3] + glob[body.sides["right"]["upper"], :3, 3])


def _palms_together(body: Body, pose: BodyPose, at, fingers=(0.0, 1.0, 0.2), elbows=(1.0, -0.6, -0.3),
                    apart: float = 0.03) -> dict:
    """Both hands palm to palm (the wrists ``apart`` metres either side) at ``at``, the fingers pointing towards
    ``fingers`` and the left elbow out towards ``elbows`` (the right one mirrored); all in the chest's frame (right,
    up, forward), ``at`` from the point between the shoulders."""
    _, chest, shoulders = _chest_frame(body, pose)
    arms = {}
    for side, s in body.sides.items():
        sign = s["sign"]
        wrist = shoulders + chest @ (at[0] + sign * apart, at[1], at[2])
        arms[f"{side}_arm"] = Limb(tuple(wrist), tuple(chest @ (sign * elbows[0], elbows[1], elbows[2])),
                                   tuple(chest @ fingers), tuple(chest @ (sign * 1.0, 0.0, 0.0)))
    return arms


def _prayer(body: Body, pose: BodyPose) -> dict:
    """Anjali mudra: the palms together in front of the breastbone, fingers up, elbows out to the sides."""
    return _palms_together(body, pose, (0.0, -0.30 * body.torso, 0.24 * body.torso))


def _palms_overhead(body: Body, pose: BodyPose, shrug: float = 14.0, reach: float = 0.99) -> dict:
    """The arms overhead (``reach`` of their length), the palms together above the head, fingers up."""
    pose = replace(pose, left_arm=replace(pose.left_arm, shrug=shrug), right_arm=replace(pose.right_arm, shrug=shrug))
    glob, chest, shoulders = _chest_frame(body, pose)
    left = body.sides["left"]
    offset = chest.T @ (glob[left["upper"], :3, 3] - shoulders)  # the left shoulder, chest frame
    apart, ahead = 0.03, 0.04
    length = reach * (left["upper_length"] + left["fore_length"])
    up = offset[1] + math.sqrt(max(length ** 2 - (offset[0] - apart) ** 2 - (offset[2] - ahead) ** 2, 0.0))
    arms = _palms_together(body, pose, (0.0, up, ahead), fingers=(0.0, 1.0, 0.05), elbows=(1.0, 0.0, -0.5), apart=apart)
    return {name: replace(limb, shrug=shrug) for name, limb in arms.items()}


def _foot_on_leg(body: Body, pose: BodyPose, side: str, part: str, at: float, pole, gap: float = 0.004,
                 toes_above: float | None = None, heel_below: float = 0.0, ahead: float = 0.15) -> Limb:
    """The ``side`` leg bent, the sole of its foot against the inside of the other leg's ``part`` ("thigh" or
    "calf"), the toes pointing down (and ``ahead`` forward): the ankle ``at`` of the way up that part (from its
    lower end), the knee towards ``pole``; the foot as close as leaves ``gap`` metres between the sole and the other
    leg's skin. ``toes_above``: instead of ``at``, the ankle as high as puts the tips of the toes that far above the
    part's lower joint, but the top of the foot (the heel) at least ``heel_below`` metres below its upper joint."""
    s = body.sides[side]
    other = body.sides["right" if side == "left" else "left"]
    lower, upper = (other["calf"], other["thigh"]) if part == "thigh" else (other["foot"], other["calf"])
    _, glob = solve(body, pose)
    a, b = glob[lower, :3, 3], glob[upper, :3, 3]
    axis = _unit(b - a)
    inward = _across((s["sign"], 0.0, 0.0), axis)  # from the other leg towards this one's side
    third = _cross(axis, inward)  # forward for the right foot, back for the left
    forward = _unit(-axis - ahead * s["sign"] * third)  # the toes down along the leg, a little ahead
    foot_ids, leg_ids = body.foot_group(side), body.group(upper)  # (the thigh or calf bone moves that part)
    distance = 0.2
    limb = None
    for _ in range(3 if toes_above is not None else 1):
        centre = a + at * (b - a)
        for _ in range(4):  # the foot translates with its ankle: a few steps close the gap
            limb = Limb(tuple(centre + distance * inward), tuple(pole), tuple(forward), tuple(inward))
            g = solve(body, replace(pose, **{f"{side}_leg": limb}))[1]
            foot, leg = skin(body, g, foot_ids) - centre, skin(body, g, leg_ids) - centre
            fa, ft, la, lt = foot @ axis, foot @ third, leg @ axis, leg @ third
            behind = (la > fa.min()) & (la < fa.max()) & (lt > ft.min()) & (lt < ft.max())  # the leg under the sole
            if not behind.any():  # a coarse mesh: the whole part
                behind = np.ones(len(leg), bool)
            distance -= float((foot @ inward).min() - (leg[behind] @ inward).max()) - gap
        if toes_above is not None:  # the foot moves (about) with its ankle along the leg
            low, high = centre[1] + foot[:, 1].min(), centre[1] + foot[:, 1].max()
            at += min(float(a[1] + toes_above - low), float(b[1] - heel_below - high)) / float(b[1] - a[1])
    return limb


def _tree_steps(body: Body) -> list[Step]:
    """Vrksasana: standing on the straight left leg, the sole of the right foot against the inner left thigh above
    the knee (toes down), the right knee open to the side, the hips level; the palms together at the chest, then
    overhead; the gaze forward. From standing: the weight shifts onto the left foot, the right foot comes to the
    inner calf, slides up to the thigh, then the hands rise from the chest through in front of the face."""
    stand = mountain(body)
    standing = replace(stand.left_leg, pole=(0.0, 0.0, 1.0))

    def on_left(arms, foot=None):
        def build(x, z):
            pose = replace(stand, root=(x, stand.root[1], z), left_leg=standing)
            reach = {"left": (standing.target, ARM_STRAIGHT)}
            if foot is None:  # both feet on the floor, the weight over the left one
                reach["right"] = (stand.right_leg.target, ARM_STRAIGHT)
            pose = _stand_on(body, pose, reach)
            if foot is not None:
                pose = replace(pose, right_leg=foot(pose))
            return replace(pose, **arms(pose))
        return _balanced(body, build, "left")

    in_front = partial(_palms_together, body, at=(0.0, -0.55 * body.torso, 0.65 * body.torso), fingers=(0.0, 0.5, 1.0),
                       elbows=(1.0, -0.8, -0.4), apart=0.05)  # on the way up from the sides, in front of the belly
    face = partial(_palms_together, body, at=(0.0, 0.30 * body.torso, 0.62 * body.torso), fingers=(0.0, 1.0, 0.1),
                   elbows=(1.0, -0.5, -0.2))  # on the way overhead, clear of the face
    calf = partial(_foot_on_leg, body, side="right", part="calf", at=0.6, pole=(-1.0, -0.5, 0.8), gap=0.015)
    thigh = partial(_foot_on_leg, body, side="right", part="thigh", at=0.5, pole=(-1.0, -0.4, 0.4), ahead=0.4,
                    toes_above=0.06 * body.leg, heel_below=0.12 * body.leg)  # the toes above the knee
    prayer, overhead = partial(_prayer, body), partial(_palms_overhead, body)
    shift = on_left(in_front)
    on_calf, on_thigh = on_left(prayer, calf), on_left(prayer, thigh)
    rising, tree = on_left(face, thigh), on_left(overhead, thigh)
    into = [Step(stand, hold=1.5, breathe=False), Step(shift, seconds=1.5),
            Step(on_calf, seconds=1.5, lift={"right_leg": 0.04}), Step(on_thigh, seconds=1.2, hold=1.0, breathe=False),
            Step(rising, seconds=0.8)]
    return into + [Step(tree, seconds=1.0, hold=9.0), Step(rising, seconds=1.0)] + _up_from(into)


def _warrior3_steps(body: Body) -> list[Step]:
    """Virabhadrasana III: balancing on the straight left leg, the torso hinged forward to level, the right leg
    lifted straight back in line with it (hips square, toes down), the arms reaching forward alongside the ears:
    one horizontal line from the hands to the heel, the gaze down and a little ahead. From standing: the weight
    shifts onto the left foot, then the body tips forward like a seesaw as the right leg rises."""
    stand = mountain(body)
    standing = replace(stand.left_leg, pole=(0.0, 0.0, 1.0))
    right = body.sides["right"]
    leg_length = right["thigh_length"] + right["shin_length"]
    arm_length = right["upper_length"] + right["fore_length"]

    def build(hinge, x, z):
        base = replace(stand, root=(x, stand.root[1], z), left_leg=standing, pelvis=(hinge, 0.0, 0.0),
                       spine=(-3.0 * hinge / 90.0, 0.0, 0.0), head=(-22.0 * hinge / 90.0, 0.0, 0.0))
        pelvis = turn(*base.pelvis)
        reach = {"left": (standing.target, ARM_STRAIGHT)}
        if hinge == 0.0:  # the right foot still on the floor where it stands
            reach["right"] = (stand.right_leg.target, ARM_STRAIGHT)
        pose = _stand_on(body, base, reach)
        glob, chest, _ = _chest_frame(body, pose)
        hip = glob[right["thigh"], :3, 3]
        # the right leg straight back in line with the pelvis (from the foot on the floor, the knee bends a little
        # while the heel peels off and straightens as the target moves out of reach)
        back = stand.right_leg if hinge == 0.0 else \
            Limb(_reach_far(hip, pelvis @ (0.0, -1.0, 0.0), leg_length), tuple(pelvis @ (0.0, 0.0, 1.0)),
                 tuple(pelvis @ right["foot_forward"]), tuple(pelvis @ (0.0, 1.0, 0.0)))
        shrug = 10.0 * hinge / 90.0
        arms = {}
        for side, s in body.sides.items():
            sign = s["sign"]
            if hinge == 0.0:  # hanging straight down by the sides, the palms facing the thighs
                direction, pole, back_of_hand = (sign * 0.18, -1.0, 0.02), (sign * 0.3, 0.0, -1.0), (sign, 0.0, -0.1)
            else:  # reaching forward in line with the chest, the palms facing each other
                direction, pole, back_of_hand = chest @ (0, 1, 0), chest @ (sign, 0.0, -0.5), chest @ (sign, 0, 0)
            arms[f"{side}_arm"] = Limb(_reach_far(glob[s["upper"], :3, 3], direction, arm_length), tuple(pole),
                                       tuple(_unit(direction)), tuple(back_of_hand), shrug=shrug)
        return replace(pose, right_leg=back, **arms)

    shift = _balanced(body, partial(build, 0.0), "left")
    warrior = _balanced(body, partial(build, 90.0), "left", along=0.72)  # the weight on the ball of the foot
    return [Step(stand, hold=1.5, breathe=False), Step(shift, seconds=1.5), Step(warrior, seconds=3.0, hold=9.0),
            Step(shift, seconds=3.0), Step(stand, seconds=1.5)]
OVERREACH = 1.08  # a free arm aimed this far (times its length) stays straight while it swings: the IK stops short


def _straight_legs_root(body: Body, pose: BodyPose) -> tuple:
    """Where the root of ``pose`` goes (its z kept) for both legs to reach their ankle targets straight
    (``ARM_STRAIGHT`` of their length): the higher crossing of the two circles the root may lie on."""
    _, glob = solve(body, replace(pose, root=(0.0, 0.0, pose.root[2])))
    circles = []
    for side, s in body.sides.items():
        offset = glob[s["thigh"], :3, 3] - glob[body.root, :3, 3]  # the hip joint from the root
        centre = np.asarray(getattr(pose, f"{side}_leg").target, float) - offset
        reach = ARM_STRAIGHT * (s["thigh_length"] + s["shin_length"])
        dz = centre[2] - pose.root[2]
        circles.append((centre[:2], math.sqrt(max(reach * reach - dz * dz, 1e-6))))
    (c0, r0), (c1, r1) = circles
    gap = c1 - c0
    d = max(float(np.linalg.norm(gap)), 1e-6)
    along = (r0 * r0 - r1 * r1 + d * d) / (2 * d)
    height = math.sqrt(max(r0 * r0 - along * along, 0.0))
    middle = c0 + along * gap / d
    across = np.array([-gap[1], gap[0]]) / d
    root = max(middle + height * across, middle - height * across, key=lambda p: p[1])
    return (float(root[0]), float(root[1]), float(pose.root[2]))


def _arm_along(body: Body, glob: np.ndarray, side: str, direction, back, pole, reach: float = OVERREACH) -> Limb:
    """A straight arm from its shoulder in ``glob`` along ``direction``, the back of the hand towards ``back``."""
    s = body.sides[side]
    direction = _unit(direction)
    wrist = glob[s["upper"], :3, 3] + reach * (s["upper_length"] + s["fore_length"]) * direction
    return Limb(tuple(wrist), tuple(pole), tuple(direction), tuple(back))


def _farther(body: Body, pose: BodyPose, names=("left_arm", "right_arm")) -> BodyPose:
    """``pose`` with the targets of the arms ``names`` pushed out along the arm to ``OVERREACH`` of its length."""
    _, glob = solve(body, pose)
    arms = {}
    for name in names:
        s = body.sides[name.split("_")[0]]
        shoulder = glob[s["upper"], :3, 3]
        limb = getattr(pose, name)
        aim = _unit(np.asarray(limb.target, float) - shoulder)
        arms[name] = replace(limb, target=tuple(shoulder + OVERREACH * (s["upper_length"] + s["fore_length"]) * aim))
    return replace(pose, **arms)


def _gap_along(body: Body, pose: BodyPose, top: np.ndarray, below: np.ndarray, axis: int = 1,
               radius: float = 0.02) -> float | None:
    """How far the vertices ``top`` of the posed mesh are beyond the vertices ``below`` along character ``axis``
    (the least difference over the ``top`` vertices with ``below`` vertices within ``radius`` across the axis;
    negative when they sink in), or None when none are in line."""
    verts = skin(body, solve(body, pose)[1], np.concatenate([top, below]))
    a, b = verts[:len(top)], verts[len(top):]
    others = [k for k in range(3) if k != axis]
    across = np.linalg.norm(a[:, None, others] - b[None, :, others], axis=2)
    near = across < radius
    if not near.any():
        return None
    under = np.where(near, b[None, :, axis], -np.inf).max(axis=1)
    has = near.any(axis=1)
    return float((a[has, axis] - under[has]).min())


def _wide_stance(body: Body, width: float, turned: bool) -> BodyPose:
    """Standing with the feet ``width`` apart (ankle to ankle) across the body, both legs straight, the arms out
    to the sides; ``turned``: the front (left) foot turned out to point along the stance, the back foot a little in."""
    stand = mountain(body)
    left, right = body.sides["left"], body.sides["right"]
    z = stand.left_leg.target[2]
    if turned:
        front = Limb((0.5 * width, left["ankle_height"], z), (1.0, 0.0, 0.15), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
                     contact=True)
        back_toes = axis_rotation((0, 1, 0), 12.0) @ (0.0, 0.0, 1.0)
        back = Limb((-0.5 * width, right["ankle_height"], z), (0.0, 0.0, 1.0), tuple(back_toes), (0.0, 1.0, 0.0),
                    contact=True)
    else:
        front = _foot(body, "left", 0.5 * width, z)
        back = _foot(body, "right", -0.5 * width, z)
    wide = replace(stand, left_leg=front, right_leg=back, flat_hands=True)
    wide = replace(wide, root=_straight_legs_root(body, wide))
    return _farther(body, replace(wide, **_arms_out(body, wide)))


def triangle(body: Body, stance: BodyPose, tilt: float | None = None, spine_roll: float = -14.0,
             opening: float = -10.0, gaze: float = -55.0, pole=(0.6, 0.0, -0.8), shrug: float = 10.0) -> BodyPose:
    """Utthita Trikonasana on the feet of ``stance`` (turned out, :func:`_wide_stance`): both legs straight, the
    hips back, the torso tilted ``tilt`` degrees (pelvis roll, plus ``spine_roll`` in the waist) over the front
    leg with the chest turned open (``opening``), the lower arm straight down to the front shin, the palm on its
    front, the top arm straight up, the gaze turned up (``gaze``, head yaw). Without ``tilt`` the torso tilts
    until the lower hand comes down onto the shin."""
    left = body.sides["left"]
    hand, shin = body.hand_group("left"), body.group(left["calf"], left["foot"])

    def place(angle, ahead=0.0):
        pose = replace(stance, pelvis=(0.0, 0.0, -angle), spine=(0.0, opening, spine_roll), head=(0.0, gaze, 0.0),
                       spine_shares=(0.2, 0.4, 0.4), right_arm=replace(stance.right_arm, shrug=shrug))
        pose = replace(pose, root=_straight_legs_root(body, pose))
        _, glob = solve(body, pose)
        down = _unit(glob[left["foot"], :3, 3] - glob[left["calf"], :3, 3])  # the fingers point down the shin
        lower = _arm_along(body, glob, "left", (0.0, -1.0, ahead), _across((0.0, 0.0, 1.0), down), pole,
                           reach=ARM_STRAIGHT)
        upper = _arm_along(body, glob, "right", (0.0, 1.0, 0.0), (0.0, 0.0, -1.0), (0.0, 0.0, -1.0))
        return replace(pose, left_arm=replace(lower, forward=tuple(down)), right_arm=replace(upper, shrug=shrug))

    if tilt is not None:
        return place(tilt)

    def off_shin(pose):  # how far the wrist is above the shin's line, seen from the front
        _, glob = solve(body, pose)
        knee, ankle = glob[left["calf"], :3, 3], glob[left["foot"], :3, 3]
        normal = _across((0.0, 1.0, 0.0), _unit((ankle - knee) * (1, 1, 0)))
        return float(((glob[left["hand"], :3, 3] - knee) * (1, 1, 0)) @ normal)

    low, high = 30.0, 89.0
    for _ in range(16):  # the tilt where the straight arm comes down onto the shin
        middle = 0.5 * (low + high)
        low, high = (middle, high) if off_shin(place(middle)) > 0 else (low, middle)
    angle = 0.5 * (low + high)
    low, high = 0.0, 0.4
    for _ in range(10):  # the hand just in front of the shin (the arm leaning forward that much)
        middle = 0.5 * (low + high)
        gap = _gap_along(body, place(angle, middle), hand, shin, axis=2)
        low, high = (middle, high) if gap is not None and gap < 0.003 else (low, middle)
    return place(angle, high)


def _triangle_steps(body: Body) -> list[Step]:
    """Utthita Trikonasana: from standing the feet step wide apart and the arms open to the sides, the front (left)
    foot turns out, the hips shift back while the torso reaches out over the front leg and tilts down, the lower
    hand to the shin, the top arm straight up, the gaze to the top hand; up the same way and back to standing."""
    stand = mountain(body)
    width = 1.2 * body.leg
    wide = _wide_stance(body, width, turned=False)
    turned = _wide_stance(body, width, turned=True)
    pose = triangle(body, turned)
    tilt = -pose.pelvis[2]
    reach = triangle(body, turned, tilt=0.45 * tilt, gaze=0.0, shrug=0.0)
    reach = _farther(body, replace(reach, **_arms_out(body, reach)))
    step = {"left_leg": 0.05, "right_leg": 0.05}
    pivot = {"left_leg": 0.02}
    return [Step(stand, hold=1.5, breathe=False), Step(wide, seconds=2.5, lift=step), Step(turned, seconds=1.0, lift=pivot),
            Step(reach, seconds=1.5), Step(pose, seconds=2.0, hold=9.0), Step(reach, seconds=2.0),
            Step(turned, seconds=1.5), Step(wide, seconds=1.0, lift=pivot), Step(stand, seconds=2.5, lift=step)]


POSES: dict[str, PoseSpec] = {
    "yogi_squat": PoseSpec("Yogi Squat (Malasana)", _yogi_squat_steps, main=1, view=0.0),
    "cow": PoseSpec("Cow Pose (Bitilasana)", _cow_steps, main=5, view=90.0),
    "warrior_1": PoseSpec("Warrior 1 (Virabhadrasana I)", _warrior1_steps, main=2, view=90.0),
    "downward_dog": PoseSpec("Downward Facing Dog (Adho Mukha Svanasana)", _downward_dog_steps, main=6, view=90.0),
    "warrior_2": PoseSpec("Warrior 2 (Virabhadrasana II)", _warrior2_steps, main=1, view=0.0),
    "upward_dog": PoseSpec("Upward Facing Dog (Urdhva Mukha Svanasana)", _upward_dog_steps, main=6, view=90.0),
    "chair": PoseSpec("Chair Pose (Utkatasana)", _chair_steps, main=3, view=90.0),
    "goddess": PoseSpec("Goddess Pose (Utkata Konasana)", _goddess_steps, main=3, view=0.0),
    "tree": PoseSpec("Tree Pose (Vrksasana)", _tree_steps, main=5, view=0.0),
    "warrior_3": PoseSpec("Warrior 3 (Virabhadrasana III)", _warrior3_steps, main=2, view=90.0),
    "triangle": PoseSpec("Triangle Pose (Utthita Trikonasana)", _triangle_steps, main=4, view=0.0),
}
