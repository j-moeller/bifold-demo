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

import numpy as np

from .character import skin_vertices
from .model import AnimationClip, CharacterModel, quat_from_matrix

SPINE_SHARES = (0.1, 0.45, 0.45)  # how a spine bend is spread over Spine, Spine1 and Spine2 (the thighs hang from Spine)
BREATH = 4.5  # seconds per breath while holding a pose
FPS = 15.0  # frames per second of the clips (they are interpolated when played)


# -- describing a pose ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class Limb:
    """Where an arm or a leg goes (character space).

    ``target``: the wrist or ankle joint. ``pole``: the direction the elbow or knee points (only its part across
    the limb counts). ``forward``: where the fingers or toes point; ``up``: the back of the hand or the top of the
    foot. ``contact``: the hand or foot rests on the floor (:func:`settle` moves the target until it just touches).
    ``shrug``: arms only, the shoulder raised by that many degrees (arms overhead).
    """

    target: tuple
    pole: tuple
    forward: tuple
    up: tuple
    contact: bool = False
    shrug: float = 0.0


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


def _lerp(a, b, w):
    return tuple(float(x) + (float(y) - float(x)) * w for x, y in zip(a, b))


def _direction_lerp(a, b, w):
    v = np.asarray(a, float) * (1 - w) + np.asarray(b, float) * w
    n = np.linalg.norm(v)
    return tuple(v / n) if n > 1e-6 else tuple(np.asarray(b, float))


def _limb_blend(a: Limb, b: Limb, w: float, lift: float = 0.0) -> Limb:
    target = _lerp(a.target, b.target, w)
    if lift:  # a step: the foot or hand rises on the way
        target = (target[0], target[1] + lift * math.sin(math.pi * w), target[2])
    return Limb(target, _direction_lerp(a.pole, b.pole, w), _direction_lerp(a.forward, b.forward, w),
                _direction_lerp(a.up, b.up, w), a.contact if w < 0.5 else b.contact, a.shrug + (b.shrug - a.shrug) * w)


def blend(a: BodyPose, b: BodyPose, w: float, lift: dict | None = None) -> BodyPose:
    """The pose ``w`` of the way from ``a`` to ``b`` (0..1); ``lift``: limb name -> metres it rises mid-way."""
    lift = lift or {}
    limbs = {name: _limb_blend(getattr(a, name), getattr(b, name), w, lift.get(name, 0.0)) for name in LIMBS}
    return BodyPose(_lerp(a.root, b.root, w), pelvis=_lerp(a.pelvis, b.pelvis, w), spine=_lerp(a.spine, b.spine, w),
                    head=_lerp(a.head, b.head, w), knees_down=a.knees_down if w < 0.5 else b.knees_down,
                    flat_hands=a.flat_hands if w < 0.5 else b.flat_hands, **limbs)


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


def _unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    return v / max(np.linalg.norm(v), 1e-12)


def _across(v, axis) -> np.ndarray | None:
    """The part of ``v`` across ``axis`` (unit), or None when ``v`` lies along it."""
    v = np.asarray(v, float) - np.dot(v, axis) * axis
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else None


def _frame(main, second) -> np.ndarray:
    """Orthonormal frame (columns): ``main``, ``second`` made perpendicular to it, and their cross product."""
    u = _unit(main)
    v = _across(second, u)
    if v is None:
        v = _across((0, 0, 1) if abs(u[2]) < 0.9 else (1, 0, 0), u)
    return np.column_stack([u, v, np.cross(u, v)])


def _align(a, b) -> np.ndarray:
    """The smallest rotation that turns direction ``a`` into direction ``b``."""
    a, b = _unit(a), _unit(b)
    axis = np.cross(a, b)
    s, c = np.linalg.norm(axis), float(np.dot(a, b))
    if s < 1e-9:
        return np.eye(3) if c > 0 else axis_rotation(_across((1, 0, 0), a) if abs(a[0]) < 0.9 else (0, 1, 0), 180.0)
    return axis_rotation(axis / s, math.degrees(math.atan2(s, c)))


# -- the avatar --------------------------------------------------------------------------------------


class Body:
    """Bones and rest measurements of an avatar (``Bip01 ...`` Biped names), for posing it."""

    def __init__(self, model: CharacterModel):
        sk = model.skeleton
        self.model, self.skeleton = model, sk
        missing = [n for n in ("Bip01", "Bip01 Spine2", "Bip01 L Thigh", "Bip01 R Hand") if sk.find(n) is None]
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
        s = self.sides[side]
        near = np.linalg.norm(self.vertices - self.rest_pos[s["calf"]], axis=1) < 0.09
        return np.intersect1d(self.group(s["calf"], s["thigh"]), np.nonzero(near)[0])


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
    for bone, share in zip(body.spine, SPINE_SHARES):
        chest = chest @ turn(*(share * np.asarray(pose.spine, float)))
        special[bone] = chest @ body.rest_rot[bone]
    special[body.neck] = chest @ body.rest_rot[body.neck]  # the clavicles hang from the neck: it follows the chest
    special[body.head] = chest @ turn(*pose.head) @ body.rest_rot[body.head]
    limbs = {"left": (pose.left_arm, pose.left_leg), "right": (pose.right_arm, pose.right_leg)}
    for side, s in body.sides.items():
        shrug = limbs[side][0].shrug
        special[s["clavicle"]] = chest @ axis_rotation((0, 0, 1), s["sign"] * shrug) @ body.rest_rot[s["clavicle"]]
    ik: dict[int, tuple] = {}
    for side, s in body.sides.items():
        arm, leg = limbs[side]
        ik[s["upper"]] = ("arm", side, arm)
        ik[s["thigh"]] = ("leg", side, leg)
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
    if pose.flat_hands:  # every finger but the thumb straight along the hand
        for side, s in body.sides.items():
            hand_dir = G[s["hand"]] @ body.rest_rot[s["hand"]].T @ s["hand_forward"]
            for chain in s["fingers"][1:]:
                for b in chain:
                    child = [c for c in sk.children(b)]
                    if child:
                        now = G[b] @ body.rest_rot[b].T @ (body.rest_pos[child[0]] - body.rest_pos[b])
                        G[b] = _align(now, hand_dir) @ G[b]
                    else:
                        G[b] = G[sk.parents[b]] @ body.local_rot[b]
                    for c in sk.descendants(b):
                        if c != b:
                            G[c] = G[sk.parents[c]] @ body.local_rot[c]
        for b in range(n):  # positions again below the hands
            if sk.parents[b] >= 0:
                P[b] = P[sk.parents[b]] + G[sk.parents[b]] @ body.local_t[b]
    glob = np.zeros((n, 4, 4))
    glob[:, :3, :3], glob[:, :3, 3], glob[:, 3, 3] = G, P, 1.0
    local = np.zeros((n, 4, 4))
    for b in range(n):
        parent = sk.parents[b]
        local[b, :3, :3] = G[b] if parent < 0 else G[parent].T @ G[b]
        local[b, :3, 3] = pose.root if parent < 0 else body.local_t[b]
        local[b, 3, 3] = 1.0
    return local, glob


def skin(body: Body, glob: np.ndarray) -> np.ndarray:
    """All mesh vertices of ``body`` posed by the global transforms ``glob``."""
    return np.concatenate(skin_vertices(body.model, body.model.skinning_matrices(glob)))


def settle(body: Body, pose: BodyPose, rounds: int = 4) -> BodyPose:
    """``pose`` with its contacts on the floor: each ``contact`` hand or foot moved up or down until its lowest
    vertex is at y = 0, and with ``knees_down`` the hips until the lower knee touches."""
    for _ in range(rounds):
        verts = skin(body, solve(body, pose)[1])
        changes, largest = {}, 0.0
        for name in LIMBS:
            limb = getattr(pose, name)
            if limb.contact:
                side = name.split("_")[0]
                group = body.hand_group(side) if name.endswith("arm") else body.foot_group(side)
                dy = -float(verts[group, 1].min())
                changes[name] = replace(limb, target=(limb.target[0], limb.target[1] + dy, limb.target[2]))
                largest = max(largest, abs(dy))
        root = pose.root
        if pose.knees_down:
            dy = -min(float(verts[body.knee_group(side), 1].min()) for side in body.sides)
            root = (root[0], root[1] + dy, root[2])
            largest = max(largest, abs(dy))
        pose = replace(pose, root=root, **changes)
        if largest < 2e-4:
            break
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


POSES: dict[str, PoseSpec] = {
    "yogi_squat": PoseSpec("Yogi Squat (Malasana)", _yogi_squat_steps, main=1, view=0.0),
}
