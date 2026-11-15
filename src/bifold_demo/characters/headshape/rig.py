"""S5 and S7: a fitted shape on the avatar's rig, and stored as ``shape.npz``.

**The shape** (:class:`Shape`): offsets of the head primitive's vertices in the bind pose (metres, character space),
of the other parts the head moves (lashes and hair cards, glasses, a keffiyeh: by primitive name), and of the face
bones' rest positions (by bone name), with what the fit found (the coefficients and the identity space they belong
to, where the scale came from, the iris size, the focal lengths, the landmark errors before and after) but no photos
and no photo landmarks. It is made from a fit's skin offsets by :func:`make_shape` (:meth:`~.head.HeadSpace.transfer`:
the eyeballs moved rigidly into their sockets, teeth, gums and tongue with the lips and the jaw, the hair with the
scalp, the bones by their regressors) and kept in a ``shape.npz`` of about 30 KB (float32, no pickles):

* ``format``, ``avatar``, ``base`` (the head's geometry fingerprint: a shape made for another version of the avatar is
  refused), ``head`` (its primitive's name);
* ``offsets`` ``(N, 3)``; ``other_names`` and ``other_<k>`` ``(M, 3)``; ``bone_names`` and ``bones`` ``(B, 3)``;
* ``beta``, ``space``, ``method``, ``scale_from``, ``iris``, ``focal``, ``error`` (before, after), ``photos``;
* ``shrink``: where the fitted offsets were kept smaller to pass the checks (JSON: ``{"left eye": 0.75}``; ``"all"``
  for the whole head). Format 1 (before the shrink was kept) is still read.

A fitted face shape is biometric data like the photos it comes from: it is kept only in git-ignored folders (a
created character's, the known faces' cache) and deleted with the person or character.

**On the rig** (:func:`apply_shape`): a copy of the model that shares everything (textures, UVs, skin weights,
triangles, the body) but the head primitive, the other parts that move, and the skeleton. The parts' positions take
the offsets; their normals turn as the surface around them turns (the stored normals rotated by the change of the
geometric normals, so the artist's normals stay as they were where nothing moves); tangents are made again
(:func:`~bifold_demo.characters.tangents.vertex_tangents`); the face bones' rest positions move by their offsets, the
skeleton's local rest transforms are made again from the global ones (rotations kept) and the inverse binds of the
moved bones follow, so the rig, the face rig of :mod:`~bifold_demo.characters.faces` (whose slides scale with the
new eye distance) and the animations apply as before. The renderer gives the copy's parts their own buffers by
itself (it keys them by object).

**Checks** (:func:`rig_checks`): the fitted head and the avatar's own are posed in each of the 12 expressions of
:mod:`~bifold_demo.characters.faces` (the tongue among them) and seen from the front and the side (z-buffers); the
fitted head must not show more of the eyeballs through the lids or of the teeth and tongue through the lips than the
avatar's own does, the neck's edge must not move, and nothing may jump (no offset beyond :data:`MAX_OFFSET`, no
flipped or degenerate triangle). In detail: no lid over an eyeball's cap sinks into it more than 1 mm further than
the avatar's (:data:`LID_TOLERANCE`); a face that shuts an eye shows no more of that eye; the front teeth come no
nearer the lips' front than the avatar's do (:data:`TEETH_TOLERANCE`); the lips' front (where the avatar's is
:data:`LIP_FRONT` or more in front of the tooth behind it, seen from the front, in each face) does not come behind
that tooth by more than :data:`LIP_TOLERANCE` (a lip pulled behind the teeth; the lips' inside, against the teeth
already, is hidden whatever it does); and where the avatar's mouth is shut seen from the front (fewer than :data:`CLOSED_MOUTH` pixels of its teeth), the
fitted head shows no more of them from either side. Each problem is put down to a part of the head
(:attr:`RigCheck.where`: an eye, the mouth, a fold, the neck, the size), so that a shape can be kept smaller there
only (:func:`~.fitted.fit_shape`).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..model import CharacterModel, Skeleton, SkinnedPrimitive
from . import mesh

FORMAT = 2  # 2: with the shrink (format 1 is still read)
FORMATS = (1, 2)
FILE = "shape.npz"
MAX_OFFSET = 0.05  # metres: no vertex of a fitted head moves further (a stylised avatar to a person: up to 35 mm)
CHECK_SIZE = 256  # pixels: the z-buffers of the checks
SHOW_TOLERANCE = 0.25  # the fitted head may show this much more of the eyes or teeth than the avatar's (relative) ...
SHOW_PIXELS = 12  # ... plus this many pixels


@dataclass
class Shape:
    """A head shape fitted to a person (see the module docs)."""

    avatar: str
    base: str
    head: str
    offsets: np.ndarray
    others: dict = field(default_factory=dict)  # primitive name -> (M, 3)
    bones: dict = field(default_factory=dict)  # bone name -> (3,)
    beta: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    space: str = ""
    method: str = ""
    scale_from: str = ""
    iris: float = float("nan")
    focal: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    error: tuple = (float("nan"), float("nan"))
    photos: int = 0
    created: float = 0.0
    shrink: dict = field(default_factory=dict)  # part of the head (or "all") -> the share of the offsets kept there

    def save(self, path) -> Path:
        """Write the shape (``path`` a file, or a folder that gets :data:`FILE`)."""
        path = Path(path)
        if path.suffix != ".npz":
            path = path / FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        names = list(self.others)
        values = {"format": np.array(FORMAT), "avatar": np.array(self.avatar), "base": np.array(self.base),
                  "head": np.array(self.head), "offsets": np.asarray(self.offsets, np.float32),
                  "other_names": np.array(names, dtype=str),
                  "bone_names": np.array(list(self.bones), dtype=str),
                  "bones": np.asarray(list(self.bones.values()), np.float32).reshape(-1, 3),
                  "beta": np.asarray(self.beta, np.float32), "space": np.array(self.space),
                  "method": np.array(self.method), "scale_from": np.array(self.scale_from),
                  "iris": np.array(self.iris, np.float32), "focal": np.asarray(self.focal, np.float32),
                  "error": np.asarray(self.error, np.float32), "photos": np.array(self.photos),
                  "created": np.array(self.created or time.time()),
                  "shrink": np.array(json.dumps({str(k): float(v) for k, v in self.shrink.items()}))}
        for k, name in enumerate(names):
            values[f"other_{k}"] = np.asarray(self.others[name], np.float32)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
        np.savez_compressed(tmp, **values)
        os.replace(tmp, path)
        return path

    @classmethod
    def load(cls, path) -> Shape:
        """The shape in ``path`` (a file or a folder with :data:`FILE`). Raises ``OSError`` when it cannot be read and
        ``ValueError`` when it is not a shape of this format (no pickles are ever loaded)."""
        path = Path(path)
        if path.is_dir():
            path = path / FILE
        try:
            with np.load(path, allow_pickle=False) as data:
                if int(data["format"]) not in FORMATS:
                    raise ValueError(f"{path}: a shape of format {int(data['format'])}, not one of {FORMATS}")
                names = [str(n) for n in data["other_names"]]
                others = {name: np.asarray(data[f"other_{k}"], np.float32) for k, name in enumerate(names)}
                bone_names = [str(n) for n in data["bone_names"]]
                bones = {n: np.asarray(v, np.float32) for n, v in zip(bone_names, data["bones"])}
                shape = cls(str(data["avatar"]), str(data["base"]), str(data["head"]),
                            np.asarray(data["offsets"], np.float32), others, bones,
                            np.asarray(data["beta"], np.float32), str(data["space"]), str(data["method"]),
                            str(data["scale_from"]), float(data["iris"]), np.asarray(data["focal"], np.float32),
                            tuple(float(v) for v in data["error"]), int(data["photos"]), float(data["created"]))
                if "shrink" in data.files:
                    shape.shrink = {str(k): float(v) for k, v in json.loads(str(data["shrink"])).items()}
        except (KeyError, TypeError, json.JSONDecodeError, AttributeError) as exc:
            raise ValueError(f"{path}: not a head shape ({exc})") from None
        if shape.offsets.ndim != 2 or shape.offsets.shape[1] != 3 or not np.isfinite(shape.offsets).all():
            raise ValueError(f"{path}: the head's offsets are not N x 3 numbers")
        for name, values in shape.others.items():
            if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
                raise ValueError(f"{path}: the offsets of {name} are not M x 3 numbers")
        for name, value in shape.bones.items():
            if value.shape != (3,) or not np.isfinite(value).all():
                raise ValueError(f"{path}: the move of bone {name} is not 3 numbers")
        return shape

    def problems(self, model) -> list[str]:
        """Why this shape cannot go on ``model`` (empty when it can)."""
        from ..face_photo import head_layout

        out = []
        if model.name != self.avatar:
            out.append(f"made for {self.avatar}, not {model.name}")
        layout = head_layout(model)
        if layout is None:
            return out + [f"{model.name} has no head"]
        head = model.primitives[layout.primitive]
        if head.name != self.head or len(head.positions) != len(self.offsets):
            out.append(f"made for a head {self.head} of {len(self.offsets)} vertices, not {head.name} of "
                       f"{len(head.positions)}")
        elif self.base and self.base != base_key(model):
            out.append(f"made for another version of {model.name}'s head (prepared again since)")
        names = {p.name: p for p in model.primitives}
        for name, values in self.others.items():
            if name not in names or len(names[name].positions) != len(values):
                out.append(f"no part {name} of {len(values)} vertices")
        for name in self.bones:
            if model.skeleton.find(name) is None:
                out.append(f"no bone {name}")
        if np.abs(self.offsets).max(initial=0.0) > MAX_OFFSET:
            out.append(f"an offset of {np.abs(self.offsets).max() * 1000:.0f} mm (at most {MAX_OFFSET * 1000:.0f})")
        return out

    def describe(self) -> str:
        moved = np.linalg.norm(self.offsets, axis=1) * 1000.0
        kept = "".join(f"; kept at {100 * v:.0f} % near {'the whole head' if k == 'all' else 'the ' + k}"
                       for k, v in self.shrink.items())
        return (f"{self.method or 'shape'} from {self.photos} photos, scale from {self.scale_from or '?'}; head moved "
                f"up to {moved.max(initial=0.0):.1f} mm (mean {moved.mean() if len(moved) else 0.0:.1f}){kept}")


def base_key(model) -> str:
    """The fingerprint of ``model``'s head a shape is made for (stage 0's geometry key)."""
    from ..headtex.landmarks import geometry_key

    return geometry_key(model)


def untangle(points, triangles, skin, offsets, tie=None, rounds: int = 40) -> np.ndarray:
    """``offsets`` (``(N, 3)``) smoothed where they would turn a skin triangle inside out or squash or stretch it
    beyond :data:`AREA_LIMITS`: the offsets of such triangles' vertices and their neighbours are averaged with their
    neighbours', round by round, until none is left (at most ``rounds`` rounds); a fold that stays (its surroundings
    hold it) widens the smoothed region by a ring every :data:`UNTANGLE_GROW` rounds. Elsewhere the offsets stay as
    they are. ``tie``: groups of vertices that move as one (:attr:`~.head.HeadSpace.tie`)."""
    points = np.asarray(points, np.float64)
    tri = np.asarray(triangles, np.int64).reshape(-1, 3)
    tri = tri[np.asarray(skin)[tri].all(axis=1)]
    out = np.array(offsets, np.float64)
    if not len(tri):
        return out
    node, count = mesh.weld(points)
    if tie is not None and (np.asarray(tie) >= 0).any():  # tied vertices are one node
        tie = np.asarray(tie)
        merged = np.where(tie >= 0, count + tie, node)
        _, node = np.unique(merged, return_inverse=True)
        node = node.ravel()
        count = int(node.max()) + 1
    e = mesh.edges(node[tri])
    neighbours = [[] for _ in range(count)]
    for a, b in e:
        neighbours[a].append(b)
        neighbours[b].append(a)
    before = np.cross(points[tri[:, 1]] - points[tri[:, 0]], points[tri[:, 2]] - points[tri[:, 0]])
    area = np.linalg.norm(before, axis=1)
    usable = area > 1e-12
    for k in range(rounds):
        moved = points + out
        after = np.cross(moved[tri[:, 1]] - moved[tri[:, 0]], moved[tri[:, 2]] - moved[tri[:, 0]])
        ratio = np.linalg.norm(after, axis=1) / np.maximum(area, 1e-30)
        bad = usable & ((np.einsum("tk,tk->t", before, after) <= 0) | (ratio < AREA_LIMITS[0])
                        | (ratio > AREA_LIMITS[1]))
        if not bad.any():
            break
        nodes = np.unique(node[tri[bad]])
        for _ring in range(1 + k // UNTANGLE_GROW):
            nodes = np.unique(np.concatenate([nodes] + [np.asarray(neighbours[n], np.int64) for n in nodes]))
        node_offsets = np.zeros((count, 3))
        np.add.at(node_offsets, node, out)
        node_offsets /= np.maximum(np.bincount(node, minlength=count), 1)[:, None]
        smoothed = np.array([node_offsets[neighbours[n]].mean(axis=0) if neighbours[n] else node_offsets[n]
                             for n in nodes])
        node_offsets[nodes] = 0.5 * node_offsets[nodes] + 0.5 * smoothed
        changed = np.isin(node, nodes) & np.asarray(skin)
        out[changed] = node_offsets[node[changed]]
    return out


AREA_LIMITS = (0.15, 8.0)  # a skin triangle's area may shrink or grow this much
UNTANGLE_GROW = 10  # rounds: a fold still there after so many smooths a ring more round it


def make_shape(model, hs, offsets, fit=None) -> Shape:
    """The :class:`Shape` of skin ``offsets`` (``(N, 3)`` on the head primitive, e.g. a
    :class:`~.fit.ShapeFit`'s) on ``model`` with its :class:`~.head.HeadSpace` ``hs``: untangled where they would
    fold the skin (:func:`untangle`), moved onto the whole head (:meth:`~.head.HeadSpace.transfer`)."""
    from .head import SKIN

    prim = model.primitives[hs.primitive]
    skin = hs.parts == SKIN
    offsets = untangle(prim.positions, prim.indices, skin, offsets, hs.tie)
    moved = hs.transfer(model, offsets)
    again = untangle(prim.positions, prim.indices, skin, moved.head, hs.tie)  # what the ties and the skin with no
    if not np.allclose(again[skin], moved.head[skin], atol=1e-9):  # counterpart folded again, smoothed; the rest
        moved = hs.transfer(model, again, keep_skin=True)  # (eyes, mouth, bones, parts) from that skin
    skeleton = model.skeleton
    head = model.primitives[hs.primitive]
    others = {model.primitives[k].name: v.astype(np.float32) for k, v in moved.others.items()}
    bones = {skeleton.names[b]: np.asarray(v, np.float32) for b, v in moved.bones.items()}
    shape = Shape(model.name, base_key(model), head.name, moved.head.astype(np.float32), others, bones,
                  created=time.time())
    if fit is not None:
        shape.beta = np.asarray(fit.beta if fit.beta is not None else [], np.float32)
        shape.space = hs.space if fit.beta is not None else ""
        shape.method, shape.scale_from = fit.method, fit.scale_from
        shape.iris = float(fit.iris_radius)
        shape.focal = np.asarray(fit.focal, np.float32)
        shape.error = (float(fit.error_before), float(fit.error_after))
        shape.photos = int(fit.photos)
    return shape


def rotate_like(vectors, before, after) -> np.ndarray:
    """``vectors`` (``(N, 3)``) each turned by the smallest rotation that takes the unit vector ``before`` to
    ``after`` (Rodrigues)."""
    v = np.asarray(vectors, np.float64)
    a = np.asarray(before, np.float64)
    b = np.asarray(after, np.float64)
    axis = np.cross(a, b)
    s = np.linalg.norm(axis, axis=1)
    c = np.einsum("nk,nk->n", a, b)
    k = axis / np.maximum(s, 1e-12)[:, None]
    turned = (v * c[:, None] + np.cross(k, v) * s[:, None]
              + k * np.einsum("nk,nk->n", k, v)[:, None] * (1.0 - c)[:, None])
    return np.where((s > 1e-9)[:, None], turned, v)


def moved_primitive(prim: SkinnedPrimitive, offsets) -> SkinnedPrimitive:
    """A copy of ``prim`` with its positions moved by ``offsets``, its normals turned with the surface and its
    tangents made again; everything else shared."""
    from ..tangents import vertex_tangents

    old = np.asarray(prim.positions, np.float64)
    new = old + np.asarray(offsets, np.float64)
    tri = np.asarray(prim.indices, np.int64).reshape(-1, 3)
    node = mesh.weld(old)[0]
    before = mesh.vertex_normals(old, tri, node)
    after = mesh.vertex_normals(new, tri, node)
    normals = rotate_like(prim.normals, before, after)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    tangents = None
    if prim.tangents is not None:
        tangents = vertex_tangents(new, normals, prim.uvs, prim.indices)
    return SkinnedPrimitive(prim.name, new.astype(np.float32), normals.astype(np.float32), prim.uvs, prim.indices,
                            prim.joints, prim.weights, prim.texture, tangents, prim.maps, prim.detail)


def apply_shape(model: CharacterModel, shape: Shape) -> CharacterModel:
    """``model`` wearing ``shape`` (see the module docs): a copy sharing everything but the moved parts and the
    skeleton. Raises ``ValueError`` (with :meth:`Shape.problems`) when the shape was made for another head."""
    problems = shape.problems(model)
    if problems:
        raise ValueError(f"the head shape does not fit {model.name}: " + "; ".join(problems))
    from ..face_photo import head_layout

    head_index = head_layout(model).primitive
    primitives = list(model.primitives)
    primitives[head_index] = moved_primitive(model.primitives[head_index], shape.offsets)
    by_name = {p.name: k for k, p in enumerate(model.primitives)}
    for name, offsets in shape.others.items():
        k = by_name[name]
        primitives[k] = moved_primitive(model.primitives[k], offsets)
    skeleton = model.skeleton
    rest_global = skeleton.rest_global.copy()
    moved = np.zeros(len(skeleton), bool)
    for name, offset in shape.bones.items():
        b = skeleton.find(name)
        rest_global[b, :3, 3] += np.asarray(offset, np.float64)
        moved[b] = bool(np.any(np.asarray(offset) != 0))
    rest_local = skeleton.rest_local.copy()
    for b, p in enumerate(skeleton.parents):
        if p >= 0 and (moved[b] or moved[p]):
            rest_local[b] = np.linalg.inv(rest_global[p]) @ rest_global[b]
        elif p < 0 and moved[b]:
            rest_local[b] = rest_global[b]
    new_skeleton = Skeleton(skeleton.names, skeleton.parents, rest_local)
    inverse_bind = np.array(model.inverse_bind, copy=True)
    changed = np.abs(new_skeleton.rest_global - skeleton.rest_global).max(axis=(1, 2)) > 0
    inverse_bind[changed] = np.linalg.inv(new_skeleton.rest_global[changed])
    metadata = dict(model.metadata)
    metadata["head_shape"] = shape.describe()
    return CharacterModel(model.name, new_skeleton, primitives, inverse_bind, metadata)


# -- checks --------------------------------------------------------------------------------------------------------


@dataclass
class RigCheck:
    """What :func:`rig_checks` found: ``ok``, the ``problems`` in words, and per expression and view the pixels of
    eyeball and of the mouth's inside each head shows (``((avatar's eyes, inside), (fitted's eyes, inside))``)."""

    ok: bool
    problems: list = field(default_factory=list)
    shown: dict = field(default_factory=dict)
    where: dict = field(default_factory=dict)  # part of the head -> its problems: "right eye", "left eye", "mouth",
    # "fold", "neck", "size"
    folds: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))  # the vertices of folded triangles


def posed_head(model, primitive: int, face: str | None) -> np.ndarray:
    """The head primitive's vertices in the rest pose with ``face`` (an expression of :mod:`..faces`) at full
    weight (CPU skinning)."""
    from ..faces import apply_face, face_rig
    from ..model import Poser

    local = Poser(model).local_pose([])
    if face is not None:
        rig = face_rig(model)
        if rig is not None:
            apply_face(local, rig, {face: 1.0})
    palette = model.skinning_matrices(model.skeleton.global_transforms(local))
    prim = model.primitives[primitive]
    m = np.einsum("nk,nkij->nij", np.asarray(prim.weights, np.float64), palette[np.asarray(prim.joints, np.int64)])
    return np.einsum("nij,nj->ni", m[:, :3, :3], np.asarray(prim.positions, np.float64)) + m[:, :3, 3]


def rig_checks(model, fitted, hs, size: int = CHECK_SIZE) -> RigCheck:
    """The checks of a fitted head (``fitted``, from :func:`apply_shape` on ``model``, whose
    :class:`~.head.HeadSpace` is ``hs``): see the module docs."""
    from ..faces import EXPRESSIONS
    from ..headtex import raster
    from ..headtex.camera import PhotoCamera
    from ..headtex.landmarks import face_frame
    from .head import SKIN

    where = {}

    def problem(part, text):
        where.setdefault(part, []).append(text)

    primitive = hs.primitive
    old = np.asarray(model.primitives[primitive].positions, np.float64)
    new = np.asarray(fitted.primitives[primitive].positions, np.float64)
    tri = np.asarray(model.primitives[primitive].indices, np.int64).reshape(-1, 3)
    if not np.isfinite(new).all():
        return RigCheck(False, ["the fitted head has points that are not numbers"],
                        where={"size": ["the fitted head has points that are not numbers"]})
    moved = np.linalg.norm(new - old, axis=1)
    if moved.max() > MAX_OFFSET:
        problem("size", f"a vertex moves {moved.max() * 1000:.0f} mm (at most {MAX_OFFSET * 1000:.0f})")
    edge = hs.fade <= 0.0
    if edge.any() and moved[edge].max() > 1e-5:
        problem("neck", f"the neck's edge moves {moved[edge].max() * 1000:.2f} mm: a gap to the body")
    skin_tri = tri[(hs.parts[tri] == SKIN).all(axis=1)]
    before = np.cross(old[skin_tri[:, 1]] - old[skin_tri[:, 0]], old[skin_tri[:, 2]] - old[skin_tri[:, 0]])
    after = np.cross(new[skin_tri[:, 1]] - new[skin_tri[:, 0]], new[skin_tri[:, 2]] - new[skin_tri[:, 0]])
    area_b, area_a = np.linalg.norm(before, axis=1), np.linalg.norm(after, axis=1)
    usable = area_b > 1e-12
    flips = (np.einsum("tk,tk->t", before, after) <= 0) & usable
    ratio = np.where(usable, area_a / np.maximum(area_b, 1e-30), 1.0)
    squashed = usable & ((ratio < 0.1) | (ratio > 10.0))
    if flips.any():
        problem("fold", f"{int(flips.sum())} triangles of the skin are turned inside out")
    if squashed.any():
        problem("fold", f"a triangle of the skin grows or shrinks too much ({ratio[usable].min():.2f} to "
                f"{ratio[usable].max():.1f})")
    folds = np.unique(skin_tri[flips | squashed])
    frame = face_frame(model)
    cameras = {"front": PhotoCamera.around(frame.centre, 0.45, 0.0, 0.0, fov=24.0, size=(size, size)),
               "side": PhotoCamera.around(frame.centre, 0.45, 75.0, 0.0, fov=24.0, size=(size, size))}
    eye_tri = [(hs.eye == e)[tri].all(axis=1) for e in (1, 2)]
    inner_tri = (hs.inner > 0)[tri].all(axis=1)
    lids = [_lids(hs, old, e) for e in range(2)]
    teeth, lips = _teeth_and_lips(model, hs, old)
    shown = {}
    for face in (None, *EXPRESSIONS):
        posed = {"avatar": posed_head(model, primitive, face), "fitted": posed_head(fitted, primitive, face)}
        key = face or "neutral"
        for e, (lid, eyeball) in enumerate(lids):
            if not len(lid):
                continue
            own, theirs = (_clearance(posed[which], lid, eyeball) for which in ("avatar", "fitted"))
            if theirs < min(own, 0.0) - LID_TOLERANCE:
                side = ("right", "left")[e]
                problem(f"{side} eye", f"{key}: the {side} eyeball comes through the lid ({theirs * 1000:.1f} mm, the "
                        f"avatar's {own * 1000:.1f} mm)")
        if len(teeth) and len(lips):  # the teeth stay behind the lips (as far as the avatar's do)
            own, theirs = (float(posed[w][teeth, 2].max() - posed[w][lips, 2].max()) for w in ("avatar", "fitted"))
            if theirs > max(own, -TEETH_TOLERANCE) + TEETH_TOLERANCE:
                problem("mouth", f"{key}: the teeth come through the lips ({theirs * 1000:.1f} mm from the lips' "
                        f"front, the avatar's {own * 1000:.1f} mm)")
        lip, tooth = _lips_before_teeth(posed["avatar"], teeth, lips)  # (paired in this face: the jaw moves)
        if len(lip):  # the lips' front stays in front of the teeth behind it (where the avatar's is)
            own, theirs = (posed[w][lip, 2] - posed[w][tooth, 2] for w in ("avatar", "fitted"))
            front = own >= LIP_FRONT  # (not the lips' inside, against the teeth already: hidden whatever it does)
            worst = np.argmin(np.where(front, theirs, np.inf)) if front.any() else 0
            if front.any() and theirs[worst] < -LIP_TOLERANCE:
                problem("mouth", f"{key}: a lip comes behind the teeth ({theirs[worst] * 1000:.1f} mm in front of "
                        f"them, the avatar's {own[worst] * 1000:.1f} mm)")
        counts = {}
        for view, camera in cameras.items():
            counts[view] = {}
            for which, points in posed.items():
                r = raster.rasterise(camera, points[tri], cull=True)
                seen = r.triangle[r.triangle >= 0]
                counts[view][which] = (int(eye_tri[0][seen].sum()), int(eye_tri[1][seen].sum()),
                                       int(inner_tri[seen].sum()))
            shown[f"{key}, {view}"] = (counts[view]["avatar"], counts[view]["fitted"])
        shut = counts["front"]["avatar"][2] < CLOSED_MOUTH  # (seen from the front: from the side an open mouth hides)
        for view in cameras:
            parts = [("mouth", "the teeth and tongue", 2)] if shut else []
            for e in CLOSED_EYES.get(face, ()):
                side = ("right", "left")[e - 1]
                parts.append((f"{side} eye", f"the {side} eyeball", e - 1))
            for part, words, k in parts:
                own, theirs = (counts[view][w][k] for w in ("avatar", "fitted"))
                if theirs > own * (1.0 + SHOW_TOLERANCE) + SHOW_PIXELS:
                    problem(part, f"{key}, {view}: the fitted head shows more of {words} ({theirs} pixels against "
                            f"{own})")
    problems = [text for texts in where.values() for text in texts]
    return RigCheck(not problems, problems, shown, where, folds)


LID_TOLERANCE = 0.001  # metres: a lid may sink this much further into its eyeball than the avatar's does
CLOSED_EYES = {"scrunch": (1, 2), "wink": (2,)}  # the eyes (1 right, 2 left) these faces shut: no more of them may show
CLOSED_MOUTH = 2 * SHOW_PIXELS  # pixels: where the avatar shows fewer of its teeth its mouth is shut: no more may show
TEETH_TOLERANCE = 0.0015  # metres: the teeth may come this much nearer the lips' front than the avatar's do
LIP_TOLERANCE = 0.0005  # metres: the lips' front may come this much behind the front of the teeth behind it ...
LIP_FRONT = 0.001  # ... where the avatar's is at least this much in front of them (the lips' inside touches them)
LIP_REACH = 0.002  # metres across (seen from the front): the frontmost tooth this near a lip vertex is behind it
LID_REACH = 0.006  # metres outside an eyeball: the skin there is its lids
LID_MARGIN = 5.0  # degrees: the lids within this of the eyeball cap's edge are not over it (the eye's corners)


def _lids(hs, points, e) -> tuple[np.ndarray, np.ndarray]:
    """The skin vertices round eyeball ``e`` (its lids) and the eyeball's vertices, in the rest pose."""
    from .head import SKIN

    eyeball = np.nonzero(hs.eye == e + 1)[0]
    if not len(eyeball):
        return np.zeros(0, np.int64), eyeball
    centre = hs.eye_centre[e]
    d = np.linalg.norm(points - centre, axis=1) - hs.eye_radius[e]
    directions = points - centre
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)
    axis = directions[eyeball].mean(axis=0)
    axis /= np.linalg.norm(axis)
    cap = np.degrees(np.arccos(np.clip(directions[eyeball] @ axis, -1.0, 1.0))).max()
    over = np.degrees(np.arccos(np.clip(directions @ axis, -1.0, 1.0))) < cap - LID_MARGIN  # over the eyeball's cap
    return np.nonzero((hs.parts == SKIN) & (d < LID_REACH) & over)[0], eyeball


def _teeth_and_lips(model, hs, points) -> tuple[np.ndarray, np.ndarray]:
    """The teeth and gums (the mouth's inside but the tongue) and the lips (the skin within 2 cm of the mouth's
    corners, in front of them)."""
    from .head import SKIN

    teeth = np.nonzero((hs.inner == 1) | (hs.inner == 2))[0]
    skeleton = model.skeleton
    corners = [skeleton.find(n) for n in ("Bip01 RMouthCorner", "Bip01 LMouthCorner")]
    if any(c is None for c in corners) or not len(teeth):
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    mouth = np.mean([skeleton.rest_global[c, :3, 3] for c in corners], axis=0)
    near = (hs.parts == SKIN) & (np.linalg.norm(points - mouth, axis=1) < 0.02) & (points[:, 2] > mouth[2])
    return teeth, np.nonzero(near)[0]


def _lips_before_teeth(points, teeth, lips) -> tuple[np.ndarray, np.ndarray]:
    """The lip vertices with a tooth behind them in the pose ``points`` (seen from the front, within
    :data:`LIP_REACH`), and for each the frontmost such tooth vertex."""
    if not len(teeth) or not len(lips):
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    across = np.linalg.norm(points[lips, None, :2] - points[None, teeth, :2], axis=2)  # (L, T)
    depth = np.where((across < LIP_REACH) & (points[None, teeth, 2] < points[lips, None, 2]),
                     points[None, teeth, 2], -np.inf)
    front = depth.argmax(axis=1)
    keep = np.isfinite(depth[np.arange(len(lips)), front])
    return lips[keep], teeth[front[keep]]


def _clearance(points, lid, eyeball) -> float:
    """How far the nearest lid vertex is outside the eyeball's sphere (negative: inside), in the pose ``points``."""
    centre, radius = mesh.sphere_fit(points[eyeball])
    return float((np.linalg.norm(points[lid] - centre, axis=1) - radius).min())
