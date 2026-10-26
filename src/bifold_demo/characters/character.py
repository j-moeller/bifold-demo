"""Animated people as scene objects: posed on the CPU every frame, skinned on the GPU by the renderer."""

from __future__ import annotations

import math
import os
import queue
import threading
from concurrent.futures import Future
from pathlib import Path

import numpy as np

from ..scene.animation import resolve
from ..scene.geometry import Mesh
from ..scene.scene import SceneObject
from ..scene.textures import IMAGE_EXTENSIONS, as_texture, parse_color
from ..scene.transforms import vec3
from .model import AnimationClip, CharacterModel, CharacterState, Layer, Poser

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
            Other children of the neck (a Biped parents the clavicles to it) keep their pose, so looking
            around never swings the arms.

    Every :meth:`update` evaluates the behaviour and stores ``state``, ``current_position``,
    ``current_rotation = (0, heading, 0)``, ``current_visible``, ``pose`` (character-space bone transforms),
    ``skin_palette`` (``(B, 4, 4)`` float32 skinning matrices) and ``shadow_ellipse``. The renderer draws
    ``skinned_model`` with ``skin_palette``; the ``mesh`` property is only the static rest pose (for bounds).
    Creating a character starts decoding its texture files, visible or not (``texture_jobs``: the futures
    from :func:`preload_textures`).
    """

    def __init__(self, model: CharacterModel, behaviour=None, *, name: str | None = None, scale=1.0,
                 shadow: bool = True, head_bones: tuple[str, str] | None = None):
        self.skinned_model = model
        self.texture_jobs = preload_textures(model)
        self.poser = _FastPoser(model)
        self.behaviour = behaviour
        self.shadow = shadow
        self.state: CharacterState | None = None
        skeleton = model.skeleton
        self._look = _look_bones(skeleton, *(head_bones if head_bones is not None else ("Neck", "Head")))
        self._rest_pose = skeleton.rest_global.copy()
        self._rest_palette = model.skinning_matrices(skeleton.rest_global).astype(np.float32)
        self._rest_pose.flags.writeable = self._rest_palette.flags.writeable = False  # shared by every rest frame
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
        color, scale = self.color, self.scale
        if callable(color) or color is not self.__dict__.get("_color_spec"):  # constants are parsed once
            self.current_color = parse_color(resolve(color, t))
            self._color_spec = None if callable(color) else color
        if callable(scale) or scale is not self.__dict__.get("_scale_spec"):
            self.current_scale = vec3(resolve(scale, t), "scale")
            self._scale_spec = None if callable(scale) else scale
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
        look = state is not None and self._look and (state.head_yaw or state.head_pitch)
        if state is None or (not state.layers and not look):
            pose, palette = self._rest_pose, self._rest_palette
        else:
            pose = self.poser.global_pose(state.layers)
            if look:
                self._turn_head(pose, float(state.head_yaw), float(state.head_pitch))
            palette = self.skinned_model.skinning_matrices(pose).astype(np.float32)
        self.pose, self.skin_palette = pose, palette
        centre = pose[self._centre_bone]
        dx, dz = self._shadow_offset
        self.shadow_ellipse = (float(centre[0, 3]) + dx, float(centre[2, 3]) + dz, *self._shadow_radii)

    def _turn_head(self, pose: np.ndarray, yaw: float, pitch: float) -> None:
        """Turn the head by ``yaw`` / ``pitch`` degrees in character space, editing the global ``pose`` in place.

        The neck turns by 40 % about its joint, then the head by the remaining 60 % about its own joint: the same
        result as :meth:`Poser.rotate_in_character_space` on the neck and then the head before forward kinematics,
        except that the neck's children outside the head chain stay where they are.
        """
        for bone, share, bones in self._look:
            pose[bones] = _pivot_rotation(yaw * share, pitch * share, pose[bone, :3, 3]) @ pose[bones]

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


_texture_jobs: dict[str, Future] = {}  # image file -> its decode, queued by the first character that uses it
_texture_queue: queue.SimpleQueue | None = None
_texture_lock = threading.Lock()


def preload_textures(model: CharacterModel) -> list[Future]:
    """Start decoding the image files among ``model``'s textures on a background thread (each file once).

    The decoded textures land in the shared file cache of :func:`bifold_demo.scene.as_texture`, where the renderer
    finds them, so the first frame that shows a character does not decode its PNGs on the render thread. Textures
    given as arrays or :class:`~bifold_demo.scene.Texture` need no decoding. Returns one future per file.
    """
    global _texture_queue
    files = sorted({os.fspath(p.texture) for p in model.primitives if isinstance(p.texture, (str, os.PathLike))
                    and Path(p.texture).suffix.lower() in IMAGE_EXTENSIONS})
    jobs = []
    with _texture_lock:
        for file in files:
            job = _texture_jobs.get(file)
            if job is None:
                if _texture_queue is None:
                    _texture_queue = queue.SimpleQueue()
                    threading.Thread(target=_decode_textures, args=(_texture_queue,), name="character-textures",
                                     daemon=True).start()
                job = _texture_jobs[file] = Future()
                _texture_queue.put((file, job))
            jobs.append(job)
    return jobs


def _decode_textures(jobs: queue.SimpleQueue) -> None:
    """Background worker: one at a time, since the file cache decodes under its lock anyway."""
    while True:
        file, job = jobs.get()
        if job.set_running_or_notify_cancel():
            try:
                job.set_result(as_texture(file))
            except Exception as exc:  # noqa: BLE001 - the renderer raises it again when it draws the character
                job.set_exception(exc)


def skin_vertices(model: CharacterModel, palette: np.ndarray) -> list[np.ndarray]:
    """CPU linear-blend skinning (a reference for tests and debugging): ``(N, 3)`` points per primitive."""
    palette = np.asarray(palette, dtype=np.float64)
    out = []
    for prim in model.primitives:
        m = np.einsum("nk,nkij->nij", prim.weights.astype(np.float64), palette[prim.joints.astype(np.int64)])
        out.append(np.einsum("nij,nj->ni", m[:, :3, :3], prim.positions.astype(np.float64)) + m[:, :3, 3])
    return out


def _look_bones(skeleton, neck_name: str | None, head_name: str | None) -> list[tuple[int, float, np.ndarray]]:
    """``(pivot bone, share of the turn, bones it moves)`` for the neck (if any) and the head."""
    head = skeleton.find(head_name) if head_name else None
    if head is None:
        return []
    head_bones = skeleton.descendants(head)
    neck = skeleton.find(neck_name) if neck_name else None
    if neck is None or neck == head or head not in skeleton.descendants(neck):
        return [(head, 1.0, head_bones)]
    chain = [neck]
    bone = int(skeleton.parents[head])
    while bone != neck:
        chain.insert(1, bone)
        bone = int(skeleton.parents[bone])
    return [(neck, NECK_SHARE, np.concatenate([chain, head_bones])), (head, 1.0 - NECK_SHARE, head_bones)]


def _pivot_rotation(yaw: float, pitch: float, pivot: np.ndarray) -> np.ndarray:
    """4x4 rotation about ``pivot``: pitch about the right axis (positive = down), then yaw about +Y (positive = left).

    The character faces +Z, so its right is -X and a positive pitch is a positive rotation about +X.
    """
    cy, sy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    cp, sp = math.cos(math.radians(pitch)), math.sin(math.radians(pitch))
    r0, r1, r2 = (cy, sy * sp, sy * cp), (0.0, cp, -sp), (-sy, cy * sp, cy * cp)
    px, py, pz = pivot.tolist()
    return np.array([
        [*r0, px - r0[0] * px - r0[1] * py - r0[2] * pz],
        [*r1, py - r1[0] * px - r1[1] * py - r1[2] * pz],
        [*r2, pz - r2[0] * px - r2[1] * py - r2[2] * pz],
        [0.0, 0.0, 0.0, 1.0],
    ])


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
# The Poser in model.py is the reference; these produce the same poses (to ~1e-15) with far fewer numpy calls,
# because dozens of characters with ~80 bones are posed every frame.


def _quat_matrix_map() -> np.ndarray:
    """``(16, 16)`` map from a quaternion's flattened outer product ``q q^T`` to ``|q|^2 *`` its 4x4 rotation."""
    x, y, z, w = range(4)
    terms = {
        (0, 0): [(1, w, w), (1, x, x), (-1, y, y), (-1, z, z)], (1, 1): [(1, w, w), (-1, x, x), (1, y, y), (-1, z, z)],
        (2, 2): [(1, w, w), (-1, x, x), (-1, y, y), (1, z, z)], (3, 3): [(1, w, w), (1, x, x), (1, y, y), (1, z, z)],
        (0, 1): [(2, x, y), (-2, z, w)], (1, 0): [(2, x, y), (2, z, w)],
        (0, 2): [(2, x, z), (2, y, w)], (2, 0): [(2, x, z), (-2, y, w)],
        (1, 2): [(2, y, z), (-2, x, w)], (2, 1): [(2, y, z), (2, x, w)],
    }
    m = np.zeros((16, 16))
    for (row, col), products in terms.items():
        for coefficient, a, b in products:
            m[a * 4 + b, row * 4 + col] += coefficient
    return m


_QUAT_MATRIX = _quat_matrix_map()


def _compose(rotations: np.ndarray, translations: np.ndarray) -> np.ndarray:
    """Same as :func:`model.compose` (quaternions need not be unit length)."""
    n = len(rotations)
    m = (rotations[:, :, None] * rotations[:, None, :]).reshape(n, 16) @ _QUAT_MATRIX
    m /= m[:, 15:]
    m = m.reshape(n, 4, 4)
    m[:, :3, 3] = translations
    return m


def _nlerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """:func:`model.quat_nlerp` for ``(K, 4)`` arrays and a scalar ``t``."""
    q = a * (1 - t) + b * np.where(np.einsum("ij,ij->i", a, b) < 0, -t, t)[:, None]
    return q / np.sqrt(np.einsum("ij,ij->i", q, q))[:, None]


def _sample(clip: AnimationClip, t: float, root_k: int | None) -> tuple[np.ndarray, np.ndarray | None]:
    """:meth:`AnimationClip.sample`, but returns only the translation of bone ``root_k``."""
    if clip.frames == 1:
        return clip.rotations[0], None if root_k is None else clip.translations[0, root_k]
    f = t * clip.fps
    last = clip.frames - 1
    f = f % last if clip.loop else min(max(f, 0.0), float(last))
    i = int(f)
    j = min(i + 1, last)
    a = f - i
    rotations = clip.rotations
    if getattr(clip, "_continuous_rotations", None) is rotations:  # signs aligned when the clip was made
        q = rotations[i] * (1 - a) + rotations[j] * a
        q /= np.sqrt(np.einsum("ij,ij->i", q, q))[:, None]
    else:
        q = _nlerp(rotations[i], rotations[j], a)
    if root_k is None:
        return q, None
    return q, clip.translations[i, root_k] * (1 - a) + clip.translations[j, root_k] * a


class _FastPoser(Poser):
    """A :class:`Poser` giving the same poses with fewer numpy calls.

    Forward kinematics uses pointer jumping: after step ``k`` every bone holds the product of the local transforms
    of itself and its ``2**k - 1`` nearest ancestors, so a skeleton of depth ``d`` needs ``ceil(log2(d))`` steps.
    """

    def __init__(self, model: CharacterModel):
        super().__init__(model)
        n = len(self.skeleton)
        ancestors = np.append(np.where(self.skeleton.parents < 0, n, self.skeleton.parents), n)  # n = identity
        self._jumps = []
        while np.any(ancestors[:n] < n):
            self._jumps.append(ancestors[:n].copy())
            ancestors = ancestors[ancestors]

    def local_pose(self, layers: list[Layer]) -> np.ndarray:
        rotations = self.rest_rotations.copy()
        translations = self.rest_translations.copy()
        total = 0.0
        for layer in layers:
            if layer.weight <= 0:
                continue
            clip_idx, skel_idx, root, root_k, scale = self._map(layer.clip)
            q, root_translation = _sample(layer.clip, layer.time, root_k if root is not None else None)
            share = layer.weight / (total + layer.weight)
            rotations[skel_idx] = _nlerp(rotations[skel_idx], q[clip_idx], share) if total else q[clip_idx]
            if root_translation is not None:
                target = root_translation * scale
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
        n = len(local)
        a, b = np.empty((n + 1, 4, 4)), np.empty((n + 1, 4, 4))
        a[n] = b[n] = np.eye(4)
        a[:n] = local
        for ancestors in self._jumps:
            np.matmul(a[ancestors], a[:n], out=b[:n])
            a, b = b, a
        return a[:n]
