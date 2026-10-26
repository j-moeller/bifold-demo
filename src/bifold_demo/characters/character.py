"""Animated people as scene objects: posed on the CPU every frame, skinned on the GPU by the renderer."""

from __future__ import annotations

import math

import numpy as np

from ..scene.animation import resolve
from ..scene.geometry import Mesh
from ..scene.scene import SceneObject
from ..scene.textures import parse_color
from ..scene.transforms import vec3
from .model import CharacterModel, CharacterState, Layer, Poser, quat_nlerp

NECK_SHARE = 0.4  # part of a head turn done by the neck; the head does the rest


class Character(SceneObject):
    """A skinned, animated character driven by a behaviour.

    Args:
        model: the rigged mesh (shared freely between characters).
        behaviour: callable ``t -> CharacterState``; ``None`` stands still in the rest pose at the origin facing +Z.
        name: scene object name.
        scale: a number or ``(sx, sy, sz)`` (or a function of time) applied in character space.
        shadow: draw a soft blob shadow on the floor under the character.
        head_bones: ``(neck, head)`` bone names used for ``head_yaw`` / ``head_pitch``; looked up as
            ``"Neck"`` and ``"Head"`` when omitted. Without a head bone the look offsets are ignored.

    Every :meth:`update` evaluates the behaviour and stores ``state``, ``current_position``,
    ``current_rotation = (0, heading, 0)``, ``current_visible``, ``pose`` (character-space bone transforms),
    ``skin_palette`` (``(B, 4, 4)`` float32 skinning matrices) and ``shadow_ellipse``. The renderer draws
    ``skinned_model`` with ``skin_palette``; the ``mesh`` property is only the static rest pose (for bounds).
    """

    def __init__(self, model: CharacterModel, behaviour=None, *, name: str | None = None, scale=1.0,
                 shadow: bool = True, head_bones: tuple[str, str] | None = None):
        self.skinned_model = model
        self.poser = _FastPoser(model)
        self.behaviour = behaviour
        self.shadow = shadow
        self.state: CharacterState | None = None
        skeleton = model.skeleton
        neck_name, head_name = head_bones if head_bones is not None else ("Neck", "Head")
        neck = skeleton.find(neck_name) if neck_name else None
        head = skeleton.find(head_name) if head_name else None
        if head is not None and neck is not None and head not in skeleton.descendants(neck):
            neck = None
        self._look_bones = [] if head is None else [(head, 1.0)] if neck is None else \
            [(neck, NECK_SHARE), (head, 1.0 - NECK_SHARE)]
        self._look_subtrees = [skeleton.descendants(bone) for bone, _share in self._look_bones]
        self._rest_pose = skeleton.rest_global
        self._rest_palette = model.skinning_matrices(skeleton.rest_global).astype(np.float32)
        centre_bone = skeleton.find("Pelvis", "Hips")
        self._centre_bone = 0 if centre_bone is None else centre_bone
        cx, cz, rx, rz = _footprint(model)
        rest_centre = skeleton.rest_global[self._centre_bone, [0, 2], 3]
        self._shadow_offset = np.array([cx, cz]) - rest_centre
        self._shadow_radii = (rx, rz)
        super().__init__(None, name=name, scale=scale)
        self.pose = self._rest_pose
        self.skin_palette = self._rest_palette
        self.shadow_ellipse = (cx, cz, rx, rz)
        self.update(0.0)

    def _build_mesh(self) -> Mesh:
        prims = self.skinned_model.primitives
        offsets = np.cumsum([0] + [len(p.positions) for p in prims[:-1]])
        return Mesh(np.concatenate([p.positions for p in prims]),
                    np.concatenate([p.indices.astype(np.int64) + o for p, o in zip(prims, offsets)]),
                    uvs=np.concatenate([p.uvs for p in prims]), normals=np.concatenate([p.normals for p in prims]))

    def update(self, t: float) -> None:
        """Evaluate the behaviour at time ``t`` and pose the skeleton (skipped while invisible)."""
        self.current_color = parse_color(resolve(self.color, t))
        self.current_scale = vec3(resolve(self.scale, t), "scale")
        state = self.behaviour(t) if self.behaviour is not None else None
        self.state = state
        if state is None:
            self.current_position = np.zeros(3)
            self.current_rotation = np.zeros(3)
            visible = True
        else:
            self.current_position = vec3(state.position, "position")
            self.current_rotation = np.array([0.0, float(state.heading), 0.0])
            visible = bool(state.visible)
        self.current_visible = visible and bool(resolve(self.visible, t))
        if self.current_visible:
            self._pose(state)

    def _pose(self, state: CharacterState | None) -> None:
        look = state is not None and self._look_bones and (state.head_yaw or state.head_pitch)
        if state is None or (not state.layers and not look):
            pose, palette = self._rest_pose, self._rest_palette
        else:
            pose = self.poser.global_pose(state.layers)
            if look:
                self._look(pose, float(state.head_yaw), float(state.head_pitch))
            palette = self.skinned_model.skinning_matrices(pose).astype(np.float32)
        self.pose, self.skin_palette = pose, palette
        cx, cz = pose[self._centre_bone, [0, 2], 3] + self._shadow_offset
        self.shadow_ellipse = (cx, cz, *self._shadow_radii)

    def _look(self, pose: np.ndarray, yaw: float, pitch: float) -> None:
        """Turn the neck and head subtrees about their joints, in character space (edits ``pose`` in place).

        Same result as :meth:`Poser.rotate_in_character_space` on the neck and then the head followed by forward
        kinematics, without re-running forward kinematics for every bone.
        """
        for (bone, share), subtree in zip(self._look_bones, self._look_subtrees):
            m = np.eye(4)
            m[:3, :3] = _yaw_pitch(yaw * share, pitch * share)
            pivot = pose[bone, :3, 3]
            m[:3, 3] = pivot - m[:3, :3] @ pivot
            pose[subtree] = m @ pose[subtree]

    def matrices(self, camera_position=None) -> tuple[np.ndarray, np.ndarray]:
        """Model and normal matrix: ``translation(position) @ rotation_y(heading) @ scaling(scale)``."""
        rx, heading, rz = self.current_rotation
        if rx or rz or self.face_camera:
            return super().matrices(camera_position)
        key = (self.current_position.tobytes(), heading, self.current_scale.tobytes())
        cached = self.__dict__.get("_matrix_cache")
        if cached is not None and cached[0] == key:
            return cached[1], cached[2]
        c, s = math.cos(math.radians(heading)), math.sin(math.radians(heading))
        sx, sy, sz = self.current_scale
        model = np.array([[c * sx, 0.0, s * sz, 0.0], [0.0, sy, 0.0, 0.0], [-s * sx, 0.0, c * sz, 0.0],
                          [0.0, 0.0, 0.0, 1.0]])
        model[:3, 3] = self.current_position
        if abs(sx * sy * sz) > 1e-12:
            normal = np.array([[c / sx, 0.0, s / sz], [0.0, 1.0 / sy, 0.0], [-s / sx, 0.0, c / sz]])
        else:
            normal = np.eye(3)
        self.__dict__["_matrix_cache"] = (key, model, normal)
        return model, normal


def skin_vertices(model: CharacterModel, palette: np.ndarray) -> list[np.ndarray]:
    """CPU linear-blend skinning (a reference for tests and debugging): ``(N, 3)`` points per primitive."""
    palette = np.asarray(palette, dtype=np.float64)
    out = []
    for prim in model.primitives:
        m = np.einsum("nk,nkij->nij", prim.weights.astype(np.float64), palette[prim.joints.astype(np.int64)])
        out.append(np.einsum("nij,nj->ni", m[:, :3, :3], prim.positions.astype(np.float64)) + m[:, :3, 3])
    return out


def _yaw_pitch(yaw: float, pitch: float) -> np.ndarray:
    """Pitch about the character's right axis (positive = looking down), then yaw about +Y (positive = left)."""
    cy, sy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    cp, sp = math.cos(math.radians(pitch)), math.sin(math.radians(pitch))
    return np.array([[cy, sy * sp, sy * cp], [0.0, cp, -sp], [-sy, cy * sp, cy * cp]])


def _footprint(model: CharacterModel) -> tuple[float, float, float, float]:
    """``(cx, cz, rx, rz)``: a blob-shadow ellipse around the legs and feet in the rest pose (character space)."""
    points = np.concatenate([p.positions for p in model.primitives]).astype(np.float64)
    height = max(float(points[:, 1].max()), 1e-3)
    low = points[points[:, 1] < 0.3 * height]
    if len(low) < 3:
        low = points
    lo, hi = low[:, [0, 2]].min(axis=0), low[:, [0, 2]].max(axis=0)
    cx, cz = (lo + hi) / 2
    rx, rz = np.maximum((hi - lo) / 2, 0.06 * height) * 1.5 + 0.08 * height
    return float(cx), float(cz), float(rx), float(rz)


# -- faster posing ---------------------------------------------------------------------------------

_SKEW = np.zeros((3, 9))  # (x, y, z) @ _SKEW = the cross-product matrix [v]x, flattened row by row
_SKEW[[2, 1, 2, 0, 1, 0], [1, 2, 3, 5, 6, 7]] = [-1, 1, 1, -1, -1, 1]
_DIAGONAL = np.arange(3)


def _compose(rotations: np.ndarray, translations: np.ndarray) -> np.ndarray:
    """Same as :func:`model.compose` with fewer numpy calls: ``R = (1 - 2|v|^2) I + 2 v v^T + 2 w [v]x``."""
    n = len(rotations)
    q = rotations / np.sqrt(np.einsum("ij,ij->i", rotations, rotations))[:, None]
    v, w = q[:, :3], q[:, 3:]
    m = np.zeros((n, 4, 4))
    r = m[:, :3, :3]
    np.multiply(v[:, :, None], 2.0 * v[:, None, :], out=r)
    r += ((2.0 * w * v) @ _SKEW).reshape(n, 3, 3)
    r[:, _DIAGONAL, _DIAGONAL] += (1.0 - 2.0 * np.einsum("ij,ij->i", v, v))[:, None]
    m[:, :3, 3] = translations
    m[:, 3, 3] = 1.0
    return m


class _FastPoser(Poser):
    """A :class:`Poser` giving the same poses with fewer numpy calls (dozens of characters are posed per frame).

    Forward kinematics runs on bones sorted by depth, so each level is one gather and one in-place matmul.
    """

    def __init__(self, model: CharacterModel):
        super().__init__(model)
        skeleton = self.skeleton
        order = np.concatenate(skeleton._levels) if len(skeleton) else np.zeros(0, dtype=np.int64)
        rank = np.empty_like(order)
        rank[order] = np.arange(len(order))
        self._order, self._rank = order, rank
        self._spans = []
        start = 0
        for level, bones in enumerate(skeleton._levels):
            end = start + len(bones)
            self._spans.append((start, end, rank[skeleton.parents[bones]] if level else None))
            start = end

    def local_pose(self, layers: list[Layer]) -> np.ndarray:
        rotations = self.rest_rotations.copy()
        translations = self.rest_translations.copy()
        total = 0.0
        for layer in layers:
            if layer.weight <= 0:
                continue
            clip_idx, skel_idx, root, root_k, scale = self._map(layer.clip)
            q, tr = layer.clip.sample(layer.time)
            share = layer.weight / (total + layer.weight)
            rotations[skel_idx] = quat_nlerp(rotations[skel_idx], q[clip_idx], share) if total else q[clip_idx]
            if root is not None and root_k is not None:
                target = tr[root_k] * scale
                translations[root] = translations[root] * (1 - share) + target * share if total else target
            total += layer.weight
        return _compose(rotations, translations)

    def global_pose(self, layers: list[Layer], adjust=None) -> np.ndarray:
        local = self.local_pose(layers)
        if adjust is not None:
            adjust(local, self)
        return self.forward_kinematics(local)

    def forward_kinematics(self, local: np.ndarray) -> np.ndarray:
        """Same as ``skeleton.global_transforms(local)``."""
        ordered = local[self._order]
        out = np.empty_like(ordered)
        for start, end, parents in self._spans:
            if parents is None:
                out[start:end] = ordered[start:end]
            else:
                np.matmul(out[parents], ordered[start:end], out=out[start:end])
        return out[self._rank]
