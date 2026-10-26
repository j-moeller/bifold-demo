"""Core data for skinned 3D characters: skeletons, skinned meshes, animation clips and posing.

Conventions (every loader produces data in this form):

* character space: metres, Y up, the character faces +Z, feet on y = 0 in the rest pose;
* matrices are 4x4 ``float64`` for column vectors (``p' = M @ p``);
* quaternions are ``(x, y, z, w)`` and unit length;
* a skeleton's bones are sorted so that every parent comes before its children.

Skinning: ``vertex(t) = sum_k weight_k * (global(t)[joint_k] @ inverse_bind[joint_k]) @ vertex_bind``,
where ``vertex_bind`` is the primitive's position in character space in the rest pose.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# -- quaternions (vectorised over leading axes) ---------------------------------------------------


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product ``a * b`` (apply ``b`` first, then ``a``)."""
    ax, ay, az, aw = np.moveaxis(np.asarray(a, dtype=np.float64), -1, 0)
    bx, by, bz, bw = np.moveaxis(np.asarray(b, dtype=np.float64), -1, 0)
    return np.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], axis=-1)


def quat_from_axis_angle(axis, degrees) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis, axis=-1, keepdims=True)
    half = np.radians(np.asarray(degrees, dtype=np.float64))[..., None] / 2
    return np.concatenate([axis * np.sin(half), np.cos(half)], axis=-1)


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """``(..., 4)`` quaternions to ``(..., 3, 3)`` rotation matrices."""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    x, y, z, w = np.moveaxis(q, -1, 0)
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
    ], axis=-2)


def quat_from_matrix(m: np.ndarray) -> np.ndarray:
    """``(..., 3, 3)`` rotation matrices (scale is removed first) to ``(..., 4)`` quaternions with w >= 0."""
    m = np.asarray(m, dtype=np.float64)[..., :3, :3]
    m = m / np.linalg.norm(m, axis=-2, keepdims=True)  # normalise columns (drops scale)
    shape = m.shape[:-2]
    m = m.reshape(-1, 3, 3)
    q = np.empty((len(m), 4))
    trace = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    for i, r in enumerate(m):
        if trace[i] > 0:
            s = 2 * np.sqrt(trace[i] + 1)
            q[i] = [(r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s, s / 4]
        elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
            s = 2 * np.sqrt(1 + r[0, 0] - r[1, 1] - r[2, 2])
            q[i] = [s / 4, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s, (r[2, 1] - r[1, 2]) / s]
        elif r[1, 1] > r[2, 2]:
            s = 2 * np.sqrt(1 + r[1, 1] - r[0, 0] - r[2, 2])
            q[i] = [(r[0, 1] + r[1, 0]) / s, s / 4, (r[1, 2] + r[2, 1]) / s, (r[0, 2] - r[2, 0]) / s]
        else:
            s = 2 * np.sqrt(1 + r[2, 2] - r[0, 0] - r[1, 1])
            q[i] = [(r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, s / 4, (r[1, 0] - r[0, 1]) / s]
    q *= np.where(q[:, 3:] < 0, -1.0, 1.0)
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    return q.reshape(*shape, 4)


def quat_nlerp(a: np.ndarray, b: np.ndarray, t) -> np.ndarray:
    """Normalised linear interpolation along the shorter arc (fast, fine for nearby rotations)."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    if t.ndim:
        t = t[..., None]
    b = np.where(np.sum(a * b, axis=-1, keepdims=True) < 0, -b, b)
    q = a * (1 - t) + b * t
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def compose(rotations: np.ndarray, translations: np.ndarray) -> np.ndarray:
    """Quaternions ``(B, 4)`` and translations ``(B, 3)`` to rigid ``(B, 4, 4)`` transforms."""
    m = np.zeros((len(rotations), 4, 4))
    m[:, :3, :3] = quat_to_matrix(rotations)
    m[:, :3, 3] = translations
    m[:, 3, 3] = 1.0
    return m


def decompose(matrices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rigid ``(B, 4, 4)`` transforms to quaternions ``(B, 4)`` and translations ``(B, 3)``."""
    return quat_from_matrix(matrices[:, :3, :3]), matrices[:, :3, 3].copy()


# -- skeleton --------------------------------------------------------------------------------------


class Skeleton:
    """Bone hierarchy with rest-pose local transforms (relative to the parent bone)."""

    def __init__(self, names, parents, rest_local):
        self.names = list(names)
        self.parents = np.asarray(parents, dtype=np.int64)
        self.rest_local = np.asarray(rest_local, dtype=np.float64).reshape(-1, 4, 4)
        n = len(self.names)
        if self.parents.shape != (n,) or len(self.rest_local) != n:
            raise ValueError("names, parents and rest_local must have one entry per bone")
        for i, p in enumerate(self.parents):
            if p >= i:
                raise ValueError(f"bone {self.names[i]!r} comes before its parent; sort bones parents-first")
        self._index = {name: i for i, name in enumerate(self.names)}
        depth = np.zeros(n, dtype=np.int64)
        for i, p in enumerate(self.parents):
            depth[i] = 0 if p < 0 else depth[p] + 1
        self._levels = [np.nonzero(depth == d)[0] for d in range(int(depth.max(initial=-1)) + 1)]
        self.rest_global = self.global_transforms(self.rest_local)

    def __len__(self) -> int:
        return len(self.names)

    def index(self, name: str) -> int:
        return self._index[name]

    def find(self, *candidates: str) -> int | None:
        """Index of the first bone whose name equals, or else ends with, one of ``candidates`` (case-insensitive)."""
        lowered = [n.lower() for n in self.names]
        for candidate in candidates:
            c = candidate.lower()
            if c in lowered:
                return lowered.index(c)
        for candidate in candidates:
            c = candidate.lower()
            for i, name in enumerate(lowered):
                if name.endswith(c):
                    return i
        return None

    def children(self, bone: int) -> np.ndarray:
        return np.nonzero(self.parents == bone)[0]

    def descendants(self, bone: int) -> np.ndarray:
        """``bone`` and every bone below it."""
        keep = np.zeros(len(self), dtype=bool)
        keep[bone] = True
        for i in range(bone + 1, len(self)):
            if self.parents[i] >= 0 and keep[self.parents[i]]:
                keep[i] = True
        return np.nonzero(keep)[0]

    def global_transforms(self, local: np.ndarray, root: np.ndarray | None = None) -> np.ndarray:
        """Forward kinematics: local ``(B, 4, 4)`` -> character-space ``(B, 4, 4)`` (vectorised per depth)."""
        out = np.empty_like(local)
        for level, bones in enumerate(self._levels):
            if level == 0:
                out[bones] = local[bones] if root is None else root @ local[bones]
            else:
                out[bones] = out[self.parents[bones]] @ local[bones]
        return out


# -- meshes and models -----------------------------------------------------------------------------


@dataclass
class SkinnedPrimitive:
    """Part of a character's mesh with one material.

    Attributes:
        positions, normals: ``(N, 3)`` float32, rest pose, character space.
        uvs: ``(N, 2)`` float32 texture coordinates (v = 0 at the bottom of the image).
        indices: ``(M, 3)`` uint32 triangles.
        joints: ``(N, 4)`` int bone indices, weights: ``(N, 4)`` float32 summing to 1.
        texture: anything :func:`bifold_demo.scene.as_texture` accepts (RGBA with alpha = cut-out).
    """

    name: str
    positions: np.ndarray
    normals: np.ndarray
    uvs: np.ndarray
    indices: np.ndarray
    joints: np.ndarray
    weights: np.ndarray
    texture: object = None

    def interleaved(self) -> np.ndarray:
        """Rows of ``x y z nx ny nz u v j0 j1 j2 j3 w0 w1 w2 w3`` as float32."""
        return np.hstack([self.positions, self.normals, self.uvs, self.joints.astype(np.float32),
                          self.weights]).astype(np.float32)


@dataclass
class CharacterModel:
    """A rigged, skinned character in its rest pose.

    ``inverse_bind[b]`` maps character space to bone ``b``'s space in the rest pose
    (normally ``inv(skeleton.rest_global[b])``).
    """

    name: str
    skeleton: Skeleton
    primitives: list[SkinnedPrimitive]
    inverse_bind: np.ndarray
    metadata: dict = field(default_factory=dict)

    @property
    def height(self) -> float:
        """Top of the mesh in the rest pose, in metres."""
        return float(max(p.positions[:, 1].max() for p in self.primitives))

    def skinning_matrices(self, global_transforms: np.ndarray) -> np.ndarray:
        """Per-bone ``(B, 4, 4)`` matrices for the vertex shader."""
        return global_transforms @ self.inverse_bind


# -- animation -------------------------------------------------------------------------------------


@dataclass
class AnimationClip:
    """Local bone transforms sampled at a fixed frame rate.

    Attributes:
        bone_names: the ``K`` animated bones (matched to a skeleton by name).
        rotations: ``(F, K, 4)`` local rotations; translations: ``(F, K, 3)`` local translations (metres).
            On construction each quaternion's sign is chosen so that consecutive frames lie on the same
            hemisphere (non-negative dot product): neighbouring frames blend without sign checks.
        root: name of the bone that carries the whole-body motion (pelvis / centre of mass), if any.
        metadata: e.g. ``speed`` (m/s of forward motion that was removed), ``turn`` (degrees of yaw
            over the clip, positive = to the left / counter-clockwise seen from above), ``kind``.
    """

    name: str
    fps: float
    bone_names: list[str]
    rotations: np.ndarray
    translations: np.ndarray
    loop: bool = True
    root: str | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        q = np.asarray(self.rotations)
        if q.ndim == 3 and len(q) > 1:
            flips = np.einsum("fkc,fkc->fk", q[1:], q[:-1]) < 0
            if flips.any():
                sign = np.ones(q.shape[:2], dtype=q.dtype)
                sign[1:] -= 2 * (np.cumsum(flips, axis=0) % 2)
                q = q * sign[..., None]
        self.rotations = q
        self._continuous_rotations = q  # what the sign alignment was done for (``rotations`` may be reassigned)

    @property
    def frames(self) -> int:
        return len(self.rotations)

    @property
    def duration(self) -> float:
        """Length in seconds (a looping clip's last frame joins its first)."""
        return max(self.frames - 1, 1) / self.fps

    def sample(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Interpolated ``(K, 4)`` rotations and ``(K, 3)`` translations at clip time ``t`` (seconds)."""
        if self.frames == 1:
            return self.rotations[0], self.translations[0]
        f = t * self.fps
        last = self.frames - 1
        f = f % last if self.loop else min(max(f, 0.0), float(last))
        i = int(f)
        j = min(i + 1, last)
        a = f - i
        return (quat_nlerp(self.rotations[i], self.rotations[j], a),
                self.translations[i] * (1 - a) + self.translations[j] * a)

    def bone_map(self, skeleton: Skeleton) -> tuple[np.ndarray, np.ndarray]:
        """``(clip_indices, skeleton_indices)`` of the bones both share."""
        pairs = [(k, skeleton._index[n]) for k, n in enumerate(self.bone_names) if n in skeleton._index]
        if not pairs:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
        clip_idx, skel_idx = zip(*pairs)
        return np.asarray(clip_idx), np.asarray(skel_idx)


_LEG_CHAINS = (  # hip, knee, ankle, toe joints of each leg (3ds Max Biped names, then Mixamo-style names)
    (("L Thigh", "LeftUpLeg"), ("L Calf", "LeftLeg"), ("L Foot", "LeftFoot"), ("L Toe0", "LeftToeBase")),
    (("R Thigh", "RightUpLeg"), ("R Calf", "RightLeg"), ("R Foot", "RightFoot"), ("R Toe0", "RightToeBase")),
)


def _leg_chain(skeleton: Skeleton, joints, present) -> list[int]:
    """Bones from the hip down a leg, up to the first joint that is missing (or not in ``present``) or not a child."""
    chain: list[int] = []
    for candidates in joints:
        bone = skeleton.find(*candidates)
        if bone is None or skeleton.names[bone] not in present or (chain and skeleton.parents[bone] != chain[-1]):
            break
        chain.append(bone)
    return chain


def _leg_height(skeleton: Skeleton, chain: list[int], offsets: np.ndarray) -> float:
    """Height of the hip ``chain[0]`` above the lowest joint below it in the rest pose, with bone ``offsets``."""
    m = skeleton.rest_global[chain[0]].copy()
    lowest = hip = m[1, 3]
    for bone, offset in zip(chain[1:], offsets):
        local = skeleton.rest_local[bone].copy()
        local[:3, 3] = offset
        m = m @ local
        lowest = min(lowest, m[1, 3])
    return float(hip - lowest)


def root_scale(skeleton: Skeleton, clip: AnimationClip) -> float:
    """Size of ``skeleton`` relative to the skeleton ``clip`` was made for (scales the clip's root motion).

    It is the ratio of the leg heights, hip joint to toe joint, in the rest pose: the model's from its own bone
    offsets, the clip's from the clip's (constant) translations of the same bones, both with the model's rest
    rotations. That keeps the soles on the floor when the clip's knees bend and makes the same bone lengths give
    exactly 1. Without a common leg, it falls back to the ratio of the model's rest root height to the clip's
    highest root height (its legs are straightest there), and to 1.
    """
    names = {name: k for k, name in enumerate(clip.bone_names)}
    translations = np.asarray(clip.translations, dtype=np.float64)
    for joints in _LEG_CHAINS:
        chain = _leg_chain(skeleton, joints, names)
        if len(chain) < 3:
            continue
        clip_offsets = translations[:, [names[skeleton.names[b]] for b in chain[1:]]].mean(axis=0)
        model_leg = _leg_height(skeleton, chain, skeleton.rest_local[chain[1:], :3, 3])
        clip_leg = _leg_height(skeleton, chain, clip_offsets)
        if model_leg > 1e-3 and clip_leg > 1e-3:
            return model_leg / clip_leg
    root = skeleton.find(clip.root) if clip.root else None
    if root is None or clip.root not in names:
        return 1.0
    clip_height = float(translations[:, names[clip.root], 1].max())
    model_height = float(skeleton.rest_global[root, 1, 3])
    return model_height / clip_height if clip_height > 1e-3 and model_height > 1e-3 else 1.0


@dataclass
class Layer:
    """One clip contributing to a pose: ``clip`` at ``time`` seconds with blend ``weight``."""

    clip: AnimationClip
    time: float
    weight: float = 1.0


class Poser:
    """Turns animation layers into bone transforms for one character model.

    Clips drive bone *rotations*. Bone translations stay those of the model's own
    skeleton (so one clip fits children and adults alike), except for the clip's
    ``root`` bone, whose translation is taken from the clip and scaled by the size
    of the model relative to the clip's skeleton (:func:`root_scale`).
    """

    def __init__(self, model: CharacterModel):
        self.model = model
        self.skeleton = model.skeleton
        rest_q, rest_t = decompose(self.skeleton.rest_local)
        self.rest_rotations, self.rest_translations = rest_q, rest_t
        self._maps: dict[int, tuple] = {}

    def _map(self, clip: AnimationClip):
        key = id(clip)
        if key not in self._maps or self._maps[key][0] is not clip:
            clip_idx, skel_idx = clip.bone_map(self.skeleton)
            root = self.skeleton.find(clip.root) if clip.root else None
            root_k = clip.bone_names.index(clip.root) if clip.root in clip.bone_names else None
            scale = root_scale(self.skeleton, clip) if root is not None and root_k is not None else 1.0
            self._maps[key] = (clip, clip_idx, skel_idx, root, root_k, scale)
        return self._maps[key][1:]

    def local_pose(self, layers: list[Layer]) -> np.ndarray:
        """Blend the layers (weights are normalised) into local ``(B, 4, 4)`` transforms."""
        rotations = self.rest_rotations.copy()
        translations = self.rest_translations.copy()
        total = 0.0
        for layer in layers:
            if layer.weight <= 0:
                continue
            clip_idx, skel_idx, root, root_k, scale = self._map(layer.clip)
            q, tr = layer.clip.sample(layer.time)
            share = layer.weight / (total + layer.weight)  # running normalised blend
            rotations[skel_idx] = quat_nlerp(rotations[skel_idx], q[clip_idx], share) if total else q[clip_idx]
            if root is not None and root_k is not None:
                target = tr[root_k] * scale
                translations[root] = translations[root] * (1 - share) + target * share if total else target
            total += layer.weight
        return compose(rotations, translations)

    def global_pose(self, layers: list[Layer], adjust=None) -> np.ndarray:
        """Character-space bone transforms; ``adjust(local, poser)`` may edit the local pose first."""
        local = self.local_pose(layers)
        if adjust is not None:
            adjust(local, self)
        return self.skeleton.global_transforms(local)

    def rotate_in_character_space(self, local: np.ndarray, bone: int, rotation: np.ndarray) -> None:
        """Pre-rotate ``bone`` by the 3x3 ``rotation`` expressed in character space (e.g. a yaw about +Y).

        Edits ``local`` in place. Use it for procedural offsets such as turning the head.
        """
        parent = self.skeleton.parents[bone]
        parents_global = self.skeleton.global_transforms(local) if parent >= 0 else None
        p = parents_global[parent][:3, :3] if parent >= 0 else np.eye(3)
        p = p / np.linalg.norm(p, axis=0, keepdims=True)
        local[bone, :3, :3] = p.T @ rotation @ p @ local[bone, :3, :3]


@dataclass
class CharacterState:
    """What a behaviour decides for one character at one moment.

    Attributes:
        position: ``(x, y, z)`` of the character's root on the floor, in scene coordinates.
        heading: degrees about +Y; 0 faces +Z, 90 faces +X; increasing = turning to the character's left.
        layers: animation clips to blend (see :class:`Poser`).
        head_yaw: extra head turn in degrees (positive = to the character's left), split over neck and head.
        head_pitch: extra head tilt in degrees (positive = looking down).
        visible: draw the character at all.
    """

    position: tuple
    heading: float
    layers: list
    head_yaw: float = 0.0
    head_pitch: float = 0.0
    visible: bool = True
