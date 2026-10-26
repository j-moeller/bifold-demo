"""Convert binary FBX scenes (Microsoft Rocketbox avatars and animations) into :mod:`.model` data.

* :func:`load_character` builds a :class:`CharacterModel` (skeleton, skinned primitives per material,
  textures) in character space: metres, Y up, facing +Z, feet on y = 0.
* :func:`load_clip` samples an animation into an in-place :class:`AnimationClip`.

Both apply the same conversion from the file's axis system and units: one rigid transform plus a
uniform scale above the root bone. Bone-local rotations are kept; local translations are scaled.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

from .fbx import FBXNode, read_fbx
from .model import AnimationClip, CharacterModel, Skeleton, SkinnedPrimitive

KTIME_PER_SECOND = 46186158000
_TIME_MODES = {1: 120.0, 2: 100.0, 3: 60.0, 4: 50.0, 5: 48.0, 6: 30.0, 7: 30.0, 8: 29.97, 9: 29.97, 10: 25.0,
               11: 24.0, 12: 1000.0, 13: 23.976, 15: 96.0, 16: 72.0, 17: 59.94, 18: 119.88}
_EULER_ORDERS = ("xyz", "xzy", "yzx", "yxz", "zxy", "zyx", "xyz")
_BONE_TYPES = {"Root", "LimbNode", "Limb"}
_LOD_PATTERN = re.compile(r"(ultra_?low|low|mid|medium|hi|high)_?poly|lod_?(\d+)", re.IGNORECASE)
_LOD_RANK = {"hipoly": 0, "highpoly": 0, "midpoly": 1, "mediumpoly": 1, "lowpoly": 2, "ultralowpoly": 3}
_AUTO_TRIANGLES = (4000, 12000)
_BIPED_PREFIX = re.compile(r"^Bip\d+(?=\s|$)")


# -- public API ------------------------------------------------------------------------------------


def load_character(fbx_path, texture_dir=None, lod="auto", max_texture_size=1024) -> CharacterModel:
    """Load a skinned avatar from a binary FBX file.

    Args:
        fbx_path: the ``.fbx`` file (Rocketbox: ``<Avatar>/Export/<Avatar>.fbx``).
        texture_dir: folder with the texture images (default: ``../Textures`` next to the file's folder,
            then the file's folder); names are matched case-insensitively by basename.
        lod: ``"auto"`` (the most detailed level with roughly 4k-12k triangles, else the one closest to
            that range) or a level name such as ``"hipoly"`` / ``"lowpoly"`` / ``"lod1"`` or a mesh name.
        max_texture_size: textures are downscaled so that their longer side is at most this.
    """
    fbx_path = Path(fbx_path)
    scene = _Scene(read_fbx(fbx_path))
    bones = scene.skeleton_nodes()
    if not bones:
        raise ValueError(f"{fbx_path}: no skeleton found")
    levels = scene.lod_levels()
    if not levels:
        raise ValueError(f"{fbx_path}: no skinned mesh found")
    level = _choose_lod(levels, lod, fbx_path)
    meshes = levels[level]["meshes"] + levels.get("", {"meshes": []})["meshes"] if level else levels[level]["meshes"]

    index = {node_id: i for i, node_id in enumerate(bones)}
    parents = [index.get(scene.parent(node_id), -1) for node_id in bones]
    conversion = scene.conversion()
    clusters = [scene.clusters(mesh_id) for mesh_id in meshes]

    # Bind pose (file space): cluster TransformLink, else the BindPose entry, else the scene pose below the parent.
    bind = {}
    for mesh_clusters in clusters:
        for bone_id, _indexes, _weights, _transform, link in mesh_clusters:
            bind.setdefault(bone_id, link)
    pose = scene.bind_pose()
    rest_file = np.empty((len(bones), 4, 4))
    for i, node_id in enumerate(bones):
        if node_id in bind:
            rest_file[i] = bind[node_id]
        elif node_id in pose:
            rest_file[i] = pose[node_id]
        elif parents[i] >= 0:
            rest_file[i] = rest_file[parents[i]] @ scene.local_matrix(node_id)
        else:
            rest_file[i] = scene.global_matrix(node_id)
    conversion = _facing_correction(scene, bones, rest_file, conversion)
    rest_global = _to_character(rest_file, conversion)
    rest_global[:, :3, :3] = _orthonormalize(rest_global[:, :3, :3])
    rest_local = rest_global.copy()
    has_parent = np.asarray(parents) >= 0
    rest_local[has_parent] = np.linalg.inv(rest_global[np.asarray(parents)[has_parent]]) @ rest_global[has_parent]
    skeleton = Skeleton([scene.bone_name(n) for n in bones], parents, rest_local)

    textures: dict = {}
    folders = _texture_folders(fbx_path, texture_dir)
    primitives = []
    for mesh_id, mesh_clusters in zip(meshes, clusters):
        primitives += _mesh_primitives(scene, mesh_id, mesh_clusters, index, skeleton, conversion, folders,
                                       max_texture_size, textures, prefix=len(meshes) > 1)
    triangles = int(sum(len(p.indices) for p in primitives))
    metadata = {
        "source": str(fbx_path),
        "lod": level or scene.name(meshes[0]),
        "triangles": triangles,
        "lods": {name or "default": info["triangles"] for name, info in levels.items()},
        "meshes": [scene.name(m) for m in meshes],
    }
    return CharacterModel(fbx_path.name.split(".")[0], skeleton, primitives, np.linalg.inv(skeleton.rest_global),
                          metadata)


def load_clip(fbx_path, name=None) -> AnimationClip:
    """Sample the first animation stack of an FBX file into an in-place clip (see the module docs).

    The root bone's horizontal travel is measured (``metadata["speed"]`` for walk clips) and removed,
    the clip is rotated about +Y so that the travel direction (walk) or the initial facing (other clips)
    is +Z, and turn clips report their total yaw in ``metadata["turn"]`` (degrees, positive = left).
    ``metadata["kind"]`` comes from the file name: walk, turn, idle or gesture.
    """
    fbx_path = Path(fbx_path)
    name = name or fbx_path.name.split(".")[0]
    scene = _Scene(read_fbx(fbx_path))
    bones = scene.skeleton_nodes()
    if not bones:
        raise ValueError(f"{fbx_path}: no skeleton found")
    fps, times = scene.frame_times()
    index = {node_id: i for i, node_id in enumerate(bones)}
    parents = np.asarray([index.get(scene.parent(node_id), -1) for node_id in bones])
    local = np.empty((len(times), len(bones), 4, 4))
    animated = np.zeros(len(bones), dtype=bool)
    for i, node_id in enumerate(bones):
        local[:, i], animated[i] = scene.animated_local(node_id, times)
    axes, scale = _split_conversion(scene.conversion())
    local[..., :3, 3] *= scale
    for i in np.nonzero(parents < 0)[0]:
        above = scene.global_matrix(scene.parent(bones[i])).copy()
        above[:3, 3] *= scale
        local[:, i] = axes @ above @ local[:, i]
    local[..., :3, :3] /= np.linalg.norm(local[..., :3, :3], axis=-2, keepdims=True)  # rigid: drop bone scale

    root = _root_bone(scene, bones, parents)
    kind = _clip_kind(name)
    names = [scene.bone_name(n) for n in bones]
    global_ = _forward_kinematics(local, parents)
    heading = _facing_headings(global_, names, root)
    travel = global_[:, root, :3, 3]
    duration = max(len(times) - 1, 1) / fps
    delta = travel[-1] - travel[0]
    speed = float(math.hypot(delta[0], delta[2]) / duration) if kind == "walk" else 0.0
    if kind == "walk" and math.hypot(delta[0], delta[2]) > 1e-6:
        align = math.degrees(math.atan2(delta[0], delta[2]))
    else:
        align = float(heading[0])
    turn = float(np.degrees(np.unwrap(np.radians(heading)))[-1] - heading[0]) if kind == "turn" else 0.0

    yaw = _rotation_y(-align)
    local[:, root] = yaw @ local[:, root]
    xz = local[:, root, :3, 3][:, [0, 2]]
    ramp = np.linspace(0.0, 1.0, len(times))[:, None]
    xz = xz - xz[0] - (xz[-1] - xz[0]) * ramp
    loop = kind != "turn"
    if loop:
        xz -= xz.mean(axis=0)
    local[:, root, 0, 3], local[:, root, 2, 3] = xz[:, 0], xz[:, 1]

    keep = animated.copy()
    keep[root] = True
    keep_idx = np.nonzero(keep)[0]
    kept = local[:, keep_idx]
    rotations = _quaternions(kept[..., :3, :3])
    translations = kept[..., :3, 3].copy()
    metadata = {"kind": kind, "speed": speed, "turn": turn, "source": str(fbx_path)}
    return AnimationClip(name, fps, [names[i] for i in keep_idx], rotations, translations, loop=loop,
                         root=names[root], metadata=metadata)


# -- FBX scene graph -------------------------------------------------------------------------------


def _properties(node: FBXNode | None) -> dict[str, list]:
    out: dict[str, list] = {}
    p70 = node.find("Properties70") if node is not None else None
    if p70 is not None:
        for p in p70.children:
            if p.props:
                out[p.props[0]] = p.props[4:]
    return out


class _Scene:
    """Objects, connections, property templates and global settings of a parsed FBX file."""

    def __init__(self, root: FBXNode):
        objects = root.find("Objects")
        self.objects = {o.props[0]: o for o in objects.children if o.props} if objects is not None else {}
        self.templates: dict[str, dict] = {}
        definitions = root.find("Definitions")
        for object_type in definitions.find_all("ObjectType") if definitions is not None else []:
            template = object_type.find("PropertyTemplate")
            if template is not None:
                self.templates[object_type.props[0]] = _properties(template)
        self.sources_of: dict[int, list[tuple[int, str | None]]] = {}
        self.targets_of: dict[int, list[tuple[int, str | None]]] = {}
        connections = root.find("Connections")
        for c in connections.children if connections is not None else []:
            if len(c.props) < 3:
                continue
            prop = c.props[3] if c.props[0] == "OP" and len(c.props) > 3 else None
            self.sources_of.setdefault(c.props[2], []).append((c.props[1], prop))
            self.targets_of.setdefault(c.props[1], []).append((c.props[2], prop))
        self.settings = _properties(root.find("GlobalSettings"))
        self._props: dict[int, dict] = {}

    # objects and connections

    def name(self, node_id: int) -> str:
        return str(self.objects[node_id].props[1]).split("\x00")[0]

    def bone_name(self, node_id: int) -> str:
        """Model name with the Biped prefix unified (``Bip02 Head`` -> ``Bip01 Head``) so clips fit every rig."""
        return _BIPED_PREFIX.sub("Bip01", self.name(node_id))

    def subtype(self, node_id: int) -> str:
        props = self.objects[node_id].props
        return str(props[2]) if len(props) > 2 else ""

    def props(self, node_id: int) -> dict:
        """Properties70 of an object merged over its type's template defaults."""
        if node_id not in self._props:
            node = self.objects[node_id]
            merged = dict(self.templates.get(node.name, {}))
            merged.update(_properties(node))
            self._props[node_id] = merged
        return self._props[node_id]

    def prop(self, node_id: int, key: str, default):
        value = self.props(node_id).get(key)
        if not value:
            return default
        return value if len(value) > 1 else value[0]

    def sources(self, node_id: int, kind: str | None = None, prop: str | None = None) -> list[int]:
        """Objects connected into ``node_id`` (its children, materials, deformers, ...)."""
        return [s for s, p in self.sources_of.get(node_id, ())
                if s in self.objects and (kind is None or self.objects[s].name == kind) and (prop is None or p == prop)]

    def source_props(self, node_id: int, kind: str) -> list[tuple[int, str | None]]:
        return [(s, p) for s, p in self.sources_of.get(node_id, ()) if s in self.objects and self.objects[s].name == kind]

    def parent(self, node_id: int) -> int:
        """Parent model of a model (0 = the scene root)."""
        for target, prop in self.targets_of.get(node_id, ()):
            if prop is None and (target == 0 or (target in self.objects and self.objects[target].name == "Model")):
                return target
        return 0

    def models(self, subtype: str | None = None) -> list[int]:
        return [i for i, o in self.objects.items() if o.name == "Model" and (subtype is None or self.subtype(i) == subtype)]

    def skeleton_nodes(self) -> list[int]:
        """Skeleton models sorted parents-first: bones, their ancestors and non-leaf helpers below them."""
        bones = {i for i in self.models() if self.subtype(i) in _BONE_TYPES}
        keep = set(bones)
        for i in bones:
            p = self.parent(i)
            while p and p in self.objects and self.subtype(p) != "Mesh":
                keep.add(p)
                p = self.parent(p)
        for i in self.models():
            if i not in keep and self.subtype(i) != "Mesh" and self.parent(i) in keep and self._has_bone_below(i, bones):
                keep.add(i)
        order: list[int] = []

        def visit(node_id: int) -> None:
            order.append(node_id)
            for child in self.sources(node_id, "Model"):
                if child in keep and child not in seen:
                    seen.add(child)
                    visit(child)

        seen: set[int] = set()
        for i in self.models():
            if i in keep and self.parent(i) not in keep and i not in seen:
                seen.add(i)
                visit(i)
        return order

    def _has_bone_below(self, node_id: int, bones: set[int]) -> bool:
        return any(c in bones or self._has_bone_below(c, bones) for c in self.sources(node_id, "Model"))

    # transforms

    def local_matrix(self, node_id: int) -> np.ndarray:
        return self.animated_local(node_id, None)[0][0]

    def global_matrix(self, node_id: int) -> np.ndarray:
        if not node_id or node_id not in self.objects:
            return np.eye(4)
        return self.global_matrix(self.parent(node_id)) @ self.local_matrix(node_id)

    def geometric_matrix(self, node_id: int) -> np.ndarray:
        t = np.asarray(self.prop(node_id, "GeometricTranslation", (0.0, 0.0, 0.0)), dtype=np.float64)
        r = np.asarray(self.prop(node_id, "GeometricRotation", (0.0, 0.0, 0.0)), dtype=np.float64)
        s = np.asarray(self.prop(node_id, "GeometricScaling", (1.0, 1.0, 1.0)), dtype=np.float64)
        m = np.eye(4)
        m[:3, :3] = _euler_matrix(r[None], "xyz")[0] * s
        m[:3, 3] = t
        return m

    def animated_local(self, node_id: int, times: np.ndarray | None) -> tuple[np.ndarray, bool]:
        """Local transforms ``(F, 4, 4)`` of a model at ``times`` (KTime ticks; ``None`` = static pose)."""
        frames = 1 if times is None else len(times)
        channels = {}
        animated = False
        for key, default in (("Lcl Translation", (0.0, 0.0, 0.0)), ("Lcl Rotation", (0.0, 0.0, 0.0)),
                             ("Lcl Scaling", (1.0, 1.0, 1.0))):
            values = np.tile(np.asarray(self.prop(node_id, key, default), dtype=np.float64), (frames, 1))
            if times is not None:
                for curve_node in self.sources(node_id, "AnimationCurveNode", key):
                    node_props = self.props(curve_node)
                    for axis, channel in enumerate(("d|X", "d|Y", "d|Z")):
                        if channel in node_props and node_props[channel]:
                            values[:, axis] = float(node_props[channel][0])
                        for curve in self.sources(curve_node, "AnimationCurve", channel):
                            values[:, axis] = _sample_curve(self.objects[curve], times, values[0, axis])
                            animated = True
            channels[key] = values
        order = _EULER_ORDERS[int(self.prop(node_id, "RotationOrder", 0)) % len(_EULER_ORDERS)]
        vec = {k: np.asarray(self.prop(node_id, k, (0.0, 0.0, 0.0)), dtype=np.float64)
               for k in ("RotationOffset", "RotationPivot", "ScalingOffset", "ScalingPivot", "PreRotation",
                         "PostRotation")}
        pre = _euler_matrix(vec["PreRotation"][None], "xyz")[0]
        post_inv = _euler_matrix(vec["PostRotation"][None], "xyz")[0].T
        rotation = pre @ _euler_matrix(channels["Lcl Rotation"], order) @ post_inv
        out = np.zeros((frames, 4, 4))
        out[:, 3, 3] = 1.0
        # T * Roff * Rp * Rpre * R * Rpost^-1 * Rp^-1 * Soff * Sp * S * Sp^-1
        scaling = channels["Lcl Scaling"]
        out[:, :3, :3] = rotation * scaling[:, None, :]
        rp, sp = vec["RotationPivot"], vec["ScalingPivot"]
        inner = vec["ScalingOffset"] + sp - scaling * sp  # Soff * Sp * S * Sp^-1 applied to the origin
        out[:, :3, 3] = (channels["Lcl Translation"] + vec["RotationOffset"] + rp
                         + np.einsum("fij,fj->fi", rotation, inner - rp))
        return out, animated

    def bind_pose(self) -> dict[int, np.ndarray]:
        out: dict[int, np.ndarray] = {}
        for i, o in self.objects.items():
            if o.name == "Pose" and self.subtype(i) == "BindPose":
                for pose_node in o.find_all("PoseNode"):
                    node, matrix = pose_node.find("Node"), pose_node.find("Matrix")
                    if node is not None and matrix is not None:
                        out.setdefault(node.props[0], _matrix(matrix.props[0]))
        return out

    def conversion(self) -> np.ndarray:
        """File space -> character space: axis change times unit scale (to metres) as a 4x4 matrix."""
        s = self.settings
        get = lambda key, default: int(s[key][0]) if s.get(key) else default  # noqa: E731
        m = np.zeros((3, 3))
        m[0, get("CoordAxis", 0)] = get("CoordAxisSign", 1)
        m[1, get("UpAxis", 1)] = get("UpAxisSign", 1)
        m[2, get("FrontAxis", 2)] = get("FrontAxisSign", 1)
        if abs(np.linalg.det(m)) < 0.5:
            m = np.eye(3)
        if np.linalg.det(m) < 0:  # left-handed file: mirror X
            m[0] = -m[0]
        unit = float(s["UnitScaleFactor"][0]) if s.get("UnitScaleFactor") else 1.0
        out = np.eye(4)
        out[:3, :3] = m * (unit / 100.0)
        return out

    # animation

    def frame_times(self) -> tuple[float, np.ndarray]:
        """Frame rate and sample times (KTime) spanning the first animation stack."""
        mode = int(self.settings["TimeMode"][0]) if self.settings.get("TimeMode") else 0
        fps = _TIME_MODES.get(mode, 0.0)
        if not fps and self.settings.get("CustomFrameRate"):
            fps = float(self.settings["CustomFrameRate"][0])
        stacks = [i for i, o in self.objects.items() if o.name == "AnimationStack"]
        start = stop = 0
        if stacks:
            start = int(self.prop(stacks[0], "LocalStart", 0))
            stop = int(self.prop(stacks[0], "LocalStop", 0))
        keys = [self.objects[i].find("KeyTime") for i, o in self.objects.items() if o.name == "AnimationCurve"]
        keys = [np.asarray(k.props[0], dtype=np.int64) for k in keys if k is not None and len(k.props[0])]
        if stop <= start and keys:
            start = min(int(k[0]) for k in keys)
            stop = max(int(k[-1]) for k in keys)
        if fps <= 0:
            steps = [np.diff(k).min() for k in keys if len(k) > 1]
            fps = KTIME_PER_SECOND / float(min(steps)) if steps else 30.0
        count = int(round((stop - start) * fps / KTIME_PER_SECOND)) + 1
        return fps, start + np.arange(max(count, 1)) * (KTIME_PER_SECOND / fps)

    # meshes

    def lod_levels(self) -> dict[str, dict]:
        """Skinned (or bone-parented) meshes grouped by LOD name (``""`` = meshes without a LOD name)."""
        levels: dict[str, dict] = {}
        for mesh_id in self.models("Mesh"):
            geometry = self.sources(mesh_id, "Geometry")
            if not geometry:
                continue
            pvi = self.objects[geometry[0]].find("PolygonVertexIndex")
            if pvi is None:
                continue
            ends = np.asarray(pvi.props[0]) < 0
            triangles = int(len(ends) - 2 * ends.sum())
            match = _LOD_PATTERN.search(self.name(mesh_id))
            key = ""
            if match:
                key = f"lod{match.group(2)}" if match.group(2) else match.group(0).lower().replace("_", "")
                key = key.replace("highpoly", "hipoly").replace("mediumpoly", "midpoly")
            level = levels.setdefault(key, {"meshes": [], "triangles": 0})
            level["meshes"].append(mesh_id)
            level["triangles"] += triangles
        return levels

    def clusters(self, mesh_id: int) -> list[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """``(bone, indexes, weights, transform, transform_link)`` for each skin cluster of a mesh model."""
        out = []
        for geometry in self.sources(mesh_id, "Geometry"):
            for skin in self.sources(geometry, "Deformer"):
                if self.subtype(skin) != "Skin":
                    continue
                for cluster in self.sources(skin, "Deformer"):
                    node = self.objects[cluster]
                    links = self.sources(cluster, "Model")
                    if not links:
                        continue
                    indexes, weights = node.find("Indexes"), node.find("Weights")
                    transform, link = node.find("Transform"), node.find("TransformLink")
                    out.append((
                        links[0],
                        np.asarray(indexes.props[0], dtype=np.int64) if indexes is not None else np.zeros(0, np.int64),
                        np.asarray(weights.props[0], dtype=np.float64) if weights is not None else np.zeros(0),
                        _matrix(transform.props[0]) if transform is not None else np.eye(4),
                        _matrix(link.props[0]) if link is not None else self.global_matrix(links[0]),
                    ))
        return out


# -- helpers: math ---------------------------------------------------------------------------------


def _matrix(values) -> np.ndarray:
    """FBX stores 4x4 matrices column by column (translation in elements 12..14)."""
    return np.asarray(values, dtype=np.float64).reshape(4, 4).T.copy()


def _euler_matrix(degrees: np.ndarray, order: str) -> np.ndarray:
    """``(F, 3)`` Euler angles (degrees about X, Y, Z) to ``(F, 3, 3)``; ``order`` lists the axes applied first to last."""
    a = np.radians(np.asarray(degrees, dtype=np.float64))
    c, s = np.cos(a), np.sin(a)
    n = len(a)
    one, zero = np.ones(n), np.zeros(n)
    rx = np.stack([one, zero, zero, zero, c[:, 0], -s[:, 0], zero, s[:, 0], c[:, 0]], -1).reshape(n, 3, 3)
    ry = np.stack([c[:, 1], zero, s[:, 1], zero, one, zero, -s[:, 1], zero, c[:, 1]], -1).reshape(n, 3, 3)
    rz = np.stack([c[:, 2], -s[:, 2], zero, s[:, 2], c[:, 2], zero, zero, zero, one], -1).reshape(n, 3, 3)
    axes = {"x": rx, "y": ry, "z": rz}
    first, second, third = (axes[k] for k in order)
    return third @ second @ first


def _rotation_y(degrees: float) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = _euler_matrix(np.array([[0.0, degrees, 0.0]]), "xyz")[0]
    return m


def _orthonormalize(r: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(r)
    out = u @ vt
    flip = np.linalg.det(out) < 0
    if np.any(flip):
        u[flip, :, -1] *= -1
        out = u @ vt
    return out


def _quaternions(m: np.ndarray) -> np.ndarray:
    """``(..., 3, 3)`` rotation matrices to ``(..., 4)`` quaternions ``(x, y, z, w)`` with w >= 0 (vectorised)."""
    shape = m.shape[:-2]
    m = m.reshape(-1, 3, 3)
    m00, m01, m02 = m[:, 0, 0], m[:, 0, 1], m[:, 0, 2]
    m10, m11, m12 = m[:, 1, 0], m[:, 1, 1], m[:, 1, 2]
    m20, m21, m22 = m[:, 2, 0], m[:, 2, 1], m[:, 2, 2]
    case = np.argmax(np.stack([m00 + m11 + m22, m00, m11, m22], axis=1), axis=1)
    s = 2 * np.sqrt(np.maximum(np.stack([1 + m00 + m11 + m22, 1 + m00 - m11 - m22, 1 - m00 + m11 - m22,
                                         1 - m00 - m11 + m22], axis=1), 1e-12))
    candidates = np.stack([
        np.stack([(m21 - m12), (m02 - m20), (m10 - m01), s[:, 0] ** 2 / 4], axis=1) / s[:, :1],
        np.stack([s[:, 1] ** 2 / 4, (m01 + m10), (m02 + m20), (m21 - m12)], axis=1) / s[:, 1:2],
        np.stack([(m01 + m10), s[:, 2] ** 2 / 4, (m12 + m21), (m02 - m20)], axis=1) / s[:, 2:3],
        np.stack([(m02 + m20), (m12 + m21), s[:, 3] ** 2 / 4, (m10 - m01)], axis=1) / s[:, 3:4],
    ], axis=1)
    q = candidates[np.arange(len(m)), case]
    q *= np.where(q[:, 3:] < 0, -1.0, 1.0)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return q.reshape(*shape, 4)


def _sample_curve(curve: FBXNode, times: np.ndarray, default: float) -> np.ndarray:
    key_time, key_value = curve.find("KeyTime"), curve.find("KeyValueFloat")
    if key_time is None or key_value is None or not len(key_time.props[0]):
        value = curve.find("Default")
        return np.full(len(times), float(value.props[0]) if value is not None else default)
    return np.interp(times, np.asarray(key_time.props[0], dtype=np.float64),
                     np.asarray(key_value.props[0], dtype=np.float64))


def _split_conversion(conversion: np.ndarray) -> tuple[np.ndarray, float]:
    """``conversion`` = ``axes`` (rigid 4x4) times a uniform ``scale``."""
    scale = float(np.cbrt(abs(np.linalg.det(conversion[:3, :3]))))
    axes = conversion.copy()
    axes[:3, :3] /= scale
    return axes, scale


def _to_character(matrices: np.ndarray, conversion: np.ndarray) -> np.ndarray:
    """File-space global transforms to rigid character-space transforms (bone axes keep their rotation)."""
    axes, scale = _split_conversion(conversion)
    scaled = matrices.copy()
    scaled[..., :3, 3] *= scale
    return axes @ scaled


def _forward_kinematics(local: np.ndarray, parents: np.ndarray) -> np.ndarray:
    out = local.copy()
    for i, p in enumerate(parents):
        if p >= 0:
            out[:, i] = out[:, p] @ local[:, i]
    return out


# -- helpers: facing and clips ---------------------------------------------------------------------


def _find(names: list[str], *suffixes: str) -> int | None:
    lowered = [n.lower() for n in names]
    for suffix in suffixes:
        for i, n in enumerate(lowered):
            if n.endswith(suffix.lower()):
                return i
    return None


def _facing_correction(scene: _Scene, bones: list[int], rest_file: np.ndarray, conversion: np.ndarray) -> np.ndarray:
    """Add a yaw (multiple of 90 degrees) to ``conversion`` so that the toes point to +Z in the rest pose."""
    names = [scene.bone_name(b) for b in bones]
    rest = _to_character(rest_file, conversion)
    forward = np.zeros(3)
    for side in ("l", "r"):
        foot = _find(names, f"{side} foot", f"{side}_foot", f"{side}foot")
        toe = _find(names, f"{side} toe0", f"{side} toe", f"{side}_toe", f"{side}toe0", f"{side}toebase")
        if foot is not None and toe is not None:
            forward += rest[toe, :3, 3] - rest[foot, :3, 3]
    if math.hypot(forward[0], forward[2]) < 1e-6:
        return conversion
    yaw = math.degrees(math.atan2(forward[0], forward[2]))
    snapped = 90.0 * round(yaw / 90.0)
    if snapped % 360 == 0:
        return conversion
    return _rotation_y(-snapped) @ conversion


def _facing_headings(global_: np.ndarray, names: list[str], root: int) -> np.ndarray:
    """Heading (degrees, 0 = +Z, positive = left) of the body for every frame.

    The root's local axis that points forward is found from the hips (left thigh minus right thigh)
    at the first frame; the heading follows that axis.
    """
    left = _find(names, "l thigh", "l_thigh", "leftupleg")
    right = _find(names, "r thigh", "r_thigh", "rightupleg")
    rotation = global_[:, root, :3, :3]
    if left is not None and right is not None:
        side = global_[0, left, :3, 3] - global_[0, right, :3, 3]
        forward0 = np.cross(side, [0.0, 1.0, 0.0])
    else:
        forward0 = np.array([0.0, 0.0, 1.0])
    forward0[1] = 0.0
    forward0 /= max(np.linalg.norm(forward0), 1e-9)
    axes = rotation[0].T  # rows: the root's local axes in character space
    scores = np.concatenate([axes @ forward0, -(axes @ forward0)])
    k = int(np.argmax(scores))
    local_axis = np.zeros(3)
    local_axis[k % 3] = 1.0 if k < 3 else -1.0
    forward = rotation @ local_axis
    return np.degrees(np.arctan2(forward[:, 0], forward[:, 2]))


def _root_bone(scene: _Scene, bones: list[int], parents: np.ndarray) -> int:
    names = [scene.bone_name(b).lower() for b in bones]
    if "bip01" in names:
        return names.index("bip01")
    for i, b in enumerate(bones):
        if scene.subtype(b) in _BONE_TYPES:
            return i
    return 0


def _clip_kind(name: str) -> str:
    lowered = name.lower()
    if "walk" in lowered or "run" in lowered:
        return "walk"
    if "turn" in lowered:
        return "turn"
    if any(k in lowered for k in ("talk", "listen", "phone", "look_around", "lookaround", "gestic")):
        return "gesture"
    return "idle" if "idle" in lowered else "gesture"


# -- helpers: meshes, skins and textures -----------------------------------------------------------


def _choose_lod(levels: dict[str, dict], lod: str, path: Path) -> str:
    named = {k: v for k, v in levels.items() if k}
    if not named:
        return ""
    if lod != "auto":
        wanted = str(lod).lower().replace("_", "")
        for key in named:
            if key == wanted:
                return key
        for key, info in named.items():
            if wanted in key or any(wanted in str(m) for m in info["meshes"]):
                return key
        raise ValueError(f"{path}: no LOD {lod!r}; available: {sorted(named)}")
    low, high = _AUTO_TRIANGLES
    ranked = sorted(named, key=lambda k: _LOD_RANK.get(k, 10 + int(k[3:]) if k[3:].isdigit() else 5))
    for key in ranked:
        if low <= named[key]["triangles"] <= high:
            return key
    return min(ranked, key=lambda k: abs(math.log(max(named[k]["triangles"], 1) / math.sqrt(low * high))))


def _mesh_bind_matrix(scene: _Scene, mesh_id: int, clusters: list) -> np.ndarray:
    """Mesh global transform at bind time (file space)."""
    reference = scene.bind_pose().get(mesh_id)
    if reference is None:
        reference = scene.global_matrix(mesh_id)
    if not clusters:
        return reference
    _bone, _idx, _w, transform, link = clusters[0]
    candidates = [link @ transform, transform]  # FBX writers store inv(link) @ mesh; some store the mesh matrix
    errors = [np.abs(c - reference).max() for c in candidates]
    return candidates[int(np.argmin(errors))]


def _layer_values(layer: FBXNode | None, data_key: str, index_key: str, width: int, control: np.ndarray,
                  polygon: np.ndarray) -> np.ndarray | None:
    """Per polygon-vertex values of a LayerElement (normals, UVs, ...)."""
    if layer is None or layer.find(data_key) is None:
        return None
    data = np.asarray(layer.find(data_key).props[0], dtype=np.float64).reshape(-1, width)
    mapping_node, reference_node = layer.find("MappingInformationType"), layer.find("ReferenceInformationType")
    mapping = str(mapping_node.props[0]) if mapping_node is not None else "ByPolygonVertex"
    reference = str(reference_node.props[0]) if reference_node is not None else "Direct"
    if mapping == "ByPolygonVertex":
        idx = np.arange(len(control))
    elif mapping in ("ByVertice", "ByVertex", "ByControlPoint"):
        idx = control
    elif mapping == "ByPolygon":
        idx = polygon
    else:  # AllSame
        idx = np.zeros(len(control), dtype=np.int64)
    if reference in ("IndexToDirect", "Index") and layer.find(index_key) is not None:
        lookup = np.asarray(layer.find(index_key).props[0], dtype=np.int64)
        idx = lookup[np.minimum(idx, len(lookup) - 1)]
    return data[np.clip(idx, 0, len(data) - 1)]


def _mesh_primitives(scene: _Scene, mesh_id: int, clusters: list, bone_index: dict, skeleton: Skeleton,
                     conversion: np.ndarray, folders: list[Path], max_texture_size: int, cache: dict,
                     prefix: bool) -> list[SkinnedPrimitive]:
    geometry = scene.objects[scene.sources(mesh_id, "Geometry")[0]]
    vertices = np.asarray(geometry.find("Vertices").props[0], dtype=np.float64).reshape(-1, 3)
    pvi = np.asarray(geometry.find("PolygonVertexIndex").props[0], dtype=np.int64)
    ends = pvi < 0
    ends[-1] = True
    control = np.where(pvi < 0, ~pvi, pvi)
    polygon = np.concatenate([[0], np.cumsum(ends)[:-1]])
    starts = np.concatenate([[0], np.nonzero(ends)[0][:-1] + 1])
    first = starts[polygon]
    corner = np.nonzero(np.arange(len(pvi)) - first >= 2)[0]
    triangles = np.stack([first[corner], corner - 1, corner], axis=1)
    triangle_polygon = polygon[corner]

    normals = _layer_values(geometry.find("LayerElementNormal"), "Normals", "NormalsIndex", 3, control, polygon)
    uvs = _layer_values(geometry.find("LayerElementUV"), "UV", "UVIndex", 2, control, polygon)
    material_layer = geometry.find("LayerElementMaterial")
    polygon_material = np.zeros(int(polygon[-1]) + 1, dtype=np.int64)
    if material_layer is not None and material_layer.find("Materials") is not None:
        values = np.asarray(material_layer.find("Materials").props[0], dtype=np.int64)
        mapping = material_layer.find("MappingInformationType")
        if mapping is not None and str(mapping.props[0]) == "AllSame" or len(values) == 1:
            polygon_material[:] = values[0] if len(values) else 0
        else:
            polygon_material[:min(len(values), len(polygon_material))] = values[:len(polygon_material)]

    # bind transform into character space
    bind = _mesh_bind_matrix(scene, mesh_id, clusters) @ scene.geometric_matrix(mesh_id)
    to_char = conversion @ bind
    positions = vertices @ to_char[:3, :3].T + to_char[:3, 3]
    if normals is None:
        normals = np.zeros((len(pvi), 3))
    normal_matrix = np.linalg.inv(to_char[:3, :3]).T
    normals = normals @ normal_matrix.T
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    if uvs is None:
        uvs = np.zeros((len(pvi), 2))
    if np.linalg.det(to_char[:3, :3]) < 0:
        triangles = triangles[:, ::-1]

    joints, weights = _skin_weights(clusters, bone_index, len(vertices), positions, skeleton,
                                    bone_index.get(scene.parent(mesh_id)))

    # de-duplicate polygon vertices by (control point, normal, uv)
    rows = np.empty((len(pvi), 6), dtype=np.float32)
    rows[:, 0] = control.astype(np.int32).view(np.float32)
    rows[:, 1:4] = normals
    rows[:, 4:6] = uvs
    keys = np.ascontiguousarray(rows).view(np.dtype((np.void, rows.itemsize * 6))).ravel()
    _, unique_pv, vertex_of_pv = np.unique(keys, return_index=True, return_inverse=True)
    vertex_of_pv = vertex_of_pv.ravel()

    materials = scene.sources(mesh_id, "Material")
    primitives = []
    for m in np.unique(polygon_material[triangle_polygon]):
        tri = vertex_of_pv[triangles[polygon_material[triangle_polygon] == m]]
        used, local = np.unique(tri, return_inverse=True)
        pv = unique_pv[used]
        cp = control[pv]
        material = materials[m] if 0 <= m < len(materials) else None
        name = scene.name(material) if material is not None else f"material{m}"
        primitives.append(SkinnedPrimitive(
            name=f"{scene.name(mesh_id)}/{name}" if prefix else name,
            positions=positions[cp].astype(np.float32),
            normals=normals[pv].astype(np.float32),
            uvs=uvs[pv].astype(np.float32),
            indices=local.reshape(-1, 3).astype(np.uint32),
            joints=joints[cp],
            weights=weights[cp],
            texture=_material_texture(scene, material, folders, max_texture_size, cache) if material else None,
        ))
    return primitives


def _skin_weights(clusters: list, bone_index: dict, count: int, positions: np.ndarray, skeleton: Skeleton,
                  parent_bone: int | None) -> tuple[np.ndarray, np.ndarray]:
    """Four strongest influences per control point (renormalised); unweighted points get one bone."""
    dense = np.zeros((count, len(skeleton)), dtype=np.float32)
    for bone, indexes, weights, _transform, _link in clusters:
        b = bone_index.get(bone)
        if b is None or not len(indexes):
            continue
        valid = (indexes >= 0) & (indexes < count)
        np.add.at(dense[:, b], indexes[valid], weights[valid].astype(np.float32))
    k = min(4, dense.shape[1])
    joints = np.argsort(-dense, axis=1, kind="stable")[:, :k]
    weights = np.take_along_axis(dense, joints, axis=1)
    if k < 4:
        joints = np.pad(joints, ((0, 0), (0, 4 - k)))
        weights = np.pad(weights, ((0, 0), (0, 4 - k)))
    total = weights.sum(axis=1)
    missing = total <= 1e-8
    if np.any(missing):
        if parent_bone is not None:
            fallback = np.full(int(missing.sum()), parent_bone)
        else:
            heads = skeleton.rest_global[:, :3, 3]
            d = np.linalg.norm(positions[missing][:, None, :] - heads[None], axis=2)
            fallback = np.argmin(d, axis=1)
        joints[missing] = 0
        joints[missing, 0] = fallback
        weights[missing] = 0
        weights[missing, 0] = 1
        total[missing] = 1
    weights = weights / total[:, None]
    weights[weights <= 0] = 0
    joints[weights <= 0] = 0
    return joints.astype(np.int32), weights.astype(np.float32)


def _texture_folders(fbx_path: Path, texture_dir) -> list[Path]:
    if texture_dir is not None:
        return [Path(texture_dir)]
    return [fbx_path.parent.parent / "Textures", fbx_path.parent / "Textures", fbx_path.parent]


def _resolve(file_name: str, folders: list[Path]) -> Path | None:
    base = re.split(r"[\\/]", file_name)[-1].lower()
    if not base:
        return None
    for folder in folders:
        if folder.is_dir():
            for candidate in folder.iterdir():
                if candidate.name.lower() == base:
                    return candidate
    return None


def _texture_path(scene: _Scene, material: int, slots: tuple[str, ...], folders: list[Path]) -> Path | None:
    for slot in slots:
        for texture in scene.sources(material, "Texture", slot):
            node = scene.objects[texture]
            for key in ("FileName", "RelativeFilename", "Filename"):
                child = node.find(key)
                if child is not None and child.props and child.props[0]:
                    path = _resolve(str(child.props[0]), folders)
                    if path is not None:
                        return path
            for video in scene.sources(texture, "Video"):
                for key in ("Filename", "RelativeFilename"):
                    child = scene.objects[video].find(key)
                    if child is not None and child.props and child.props[0]:
                        path = _resolve(str(child.props[0]), folders)
                        if path is not None:
                            return path
    return None


def _material_texture(scene: _Scene, material: int, folders: list[Path], max_size: int, cache: dict):
    """RGBA ``uint8`` texture (row 0 = top) of a material; alpha from its opacity map, else 255."""
    color = _texture_path(scene, material, ("DiffuseColor", "Diffuse", "Maya|baseColor", "base_color"), folders)
    if color is None:
        return None
    opacity = _texture_path(scene, material, ("TransparentColor", "TransparencyFactor", "Opacity",
                                              "Maya|opacity", "opacity"), folders)
    key = (str(color), str(opacity), max_size)
    if key not in cache:
        rgba, _ = _read_image(color, max_size, cache)
        rgba = rgba.copy()
        rgba[..., 3] = 255
        if opacity is not None:
            image, has_alpha = _read_image(opacity, max_size, cache)
            if has_alpha:  # e.g. Rocketbox *_opacity_color.tga: colour plus coverage in alpha
                mask = image[..., 3]
            else:
                mask = (image[..., :3].astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32))
                mask = mask.round().clip(0, 255).astype(np.uint8)
            if mask.shape != rgba.shape[:2]:
                from PIL import Image

                mask = np.asarray(Image.fromarray(mask).resize(rgba.shape[1::-1], Image.BILINEAR))
            rgba[..., 3] = mask
        cache[key] = rgba
    return cache[key]


def _read_image(path: Path, max_size: int, cache: dict) -> tuple[np.ndarray, bool]:
    """RGBA ``uint8`` pixels (row 0 = top) downscaled to ``max_size``, and whether the file has alpha."""
    key = ("image", str(path), max_size)
    if key not in cache:
        from PIL import Image

        with Image.open(path) as image:
            has_alpha = image.mode in ("RGBA", "LA", "PA") or "transparency" in image.info
            image.load()
            longest = max(image.size)
            if longest > max_size:
                factor = longest / max_size
                if factor == int(factor) and image.width % int(factor) == 0 and image.height % int(factor) == 0:
                    image = image.reduce(int(factor))
                else:
                    size = (max(1, round(image.width / factor)), max(1, round(image.height / factor)))
                    image = image.resize(size, Image.BOX)
            cache[key] = (np.asarray(image.convert("RGBA"), dtype=np.uint8), has_alpha)
    return cache[key]
