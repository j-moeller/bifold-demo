"""S0: the identity space on one avatar's head, worked out once per head model and cached.

**Parts of the head.** Every vertex of the head primitive is one of:

* ``SKIN``: the face, the scalp, the ears and the neck, the surface the space deforms;
* ``EYE``: an eyeball (vertices led by an eye bone), moved rigidly into its new socket;
* ``INNER``: the mouth's inside (teeth, gums, tongue; what no camera outside the head sees: z-buffers from 33
  directions around it, the body closing the open neck, as :mod:`~bifold_demo.characters.headtex.fuse` finds the
  inner texels), moved by similarity transforms that follow the lips and the jaw.

The skin's offsets fade out towards the neck's open edge (:data:`NECK_FADE`: from the edge, and from what the spine,
the clavicles and the arms move), so the head stays joined to the body. The avatar's other parts that the head moves
(the lashes and hair cards of the ``*_opacity`` primitive, glasses, a keffiyeh) follow the nearest skin.

**Alignment.** ICT's neutral head is put on the avatar's head by the similarity transform (Umeyama) that best lines
up MediaPipe's landmarks on both (the avatar's from stage 0 of the head textures, ICT's shipped with the code; the
reliable ones inside the face). The avatar's coefficients are fitted to its landmarks first (ridge least squares).
Then the avatar's skin is laid onto ICT's head by a non-rigid registration (Amberg et al.'s optimal-step idea on the
avatar's side, which is the small mesh: a thin-plate spline on the landmarks to start, then closest points with a
graph-Laplacian stiffness relaxed step by step, the landmarks held, matches farther than :data:`MATCH_DISTANCE` or
turned more than 60 degrees left out). Each skin vertex then has its point on ICT's surface (a triangle and
barycentric coordinates).

**The space on the head.** The 100 identity modes and the 53 expressions, read at those points, turned and scaled
by the similarity and faded at the neck, are offsets of the avatar's vertices (:meth:`HeadSpace.modes`,
:meth:`HeadSpace.expressions`; an eyeball moves with its socket's rim). The avatar's own coefficients
(``alpha``: least squares over the whole skin with a unit-Gaussian prior, alternated with the similarity) explain
the avatar as far as the space can; the rest, ``residual``, is the artist's detail, which a fit keeps: a person's
head is ``x_avatar + modes @ (beta - alpha)``.

**Regressors** (fixed weights of skin vertices): each eyeball moves with the rim of skin round it (a translation and
a uniform scale, at most :data:`EYE_SCALE` from 1; its bone with it); the upper teeth and gums with the skin round the
upper lip, the lower ones and the tongue with the lower lip and the chin (a translation and a uniform scale each); each
face bone by the skin near its rest position (Gaussian weights, :data:`BONE_SIGMA`); the other parts by their nearest
skin vertices. :meth:`HeadSpace.transfer` applies them to any skin offsets.

The result is cached in the head-texture cache (``$BIFOLD_HEADTEX_CACHE``, else ``headtex/`` in the characters
folder) as ``headspace-<key>.npz``, keyed by the head's geometry and skinning, its stage-0 landmarks, the other parts'
geometry, the identity space and this code's version.
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np

from ..headtex import canonical as cf
from . import mesh

VERSION = 4
SKIN, EYE, INNER = 1, 2, 3
NECK_FADE = 0.05  # metres from the neck's edge over which the skin's offsets fade in
MATCH_DISTANCE = (0.02, 0.008)  # metres: the registration's closest points further than this (first, last) are left out
STIFFNESS = (40.0, 15.0, 6.0, 2.5, 1.0, 0.5)  # the registration's Laplacian weights, step by step
ICP_ROUNDS = 3  # closest-point rounds per stiffness step
LANDMARK_WEIGHT = (8.0, 1.0)  # the landmarks' weight in the registration, first and last step
ALPHA_NOISE = 0.01  # metres: how far the space misses an artist's head, per vertex, for the avatar's coefficients
# (the misses are smooth over a few centimetres, so the vertices are far from independent: 3.5 mm measured, counted
# as 10)
EYE_RIM = 0.005  # metres outside an eyeball: the skin within this of it is the eye's rim
EYE_SCALE = 0.05  # an eyeball's scale stays within this of 1
EYE_CAP_MARGIN = 10.0  # degrees: the rim reaches this far beyond the edge of the eyeball's cap (its corners)
INNER_REACH = 0.025  # metres: the skin within this of the mouth's inside moves it
INNER_SIGMA = 0.012
INNER_SCALE = 0.1  # the mouth's inside's scale stays within this of 1
BONE_SIGMA = 0.01  # metres: the Gaussian of the skin near a face bone that moves it
NEAREST = 6  # skin vertices each vertex of another part follows
TIE = 0.0012  # metres: skin vertices this close but not joined by an edge (the closed lips) move together
MOUTH_REACH = 0.04  # metres from the mouth: the mouth's inside; inner parts further away (the neck's, the
# nostrils') follow the skin
JAW_BONES = ("MJaw", "MBottomLip", "LMouthBottom", "RMouthBottom", "MTongue")
LID_BONES = ("LEyeBlinkTop", "REyeBlinkTop")
JAW_SCALE, LID_SCALE = 0.015, 0.004  # metres ICT's jawOpen and eyeBlink move a vertex for it to count fully
SIDE_TOLERANCE = 0.6  # a match must lie on the same side of the lips and the lids: features this close
FACING = -0.2  # ... and face the same way (the cosine between the normals at least this)
LEFT_OUT = tuple(sorted(set(cf.FACE_OVAL) | set(range(468, 478))))  # landmarks the alignment does not use: the
# outline (where the face's edge is seen from, not a point) and the irises (on the eyeballs)
BROWS = tuple(sorted(set(cf.RIGHT_BROW) | set(cf.LEFT_BROW)))  # ICT's head has no painted brows: half weight


@dataclass
class HeadSpace:
    """The identity space on one head model (see the module docs). Per-vertex arrays have one row per vertex of the
    head primitive."""

    model: str
    primitive: int
    key: str
    parts: np.ndarray  # (N,) int8: SKIN, EYE, INNER
    fade: np.ndarray  # (N,) float32: 0 at the neck's edge, 1 a NECK_FADE above it
    triangle: np.ndarray  # (N,) int32: ICT's triangle under each vertex (-1: none: eyes, the inside)
    barycentric: np.ndarray  # (N, 3) float32
    scale: float
    rotation: np.ndarray  # (3, 3): ICT -> the avatar's character space, with scale and translation
    translation: np.ndarray  # (3,)
    alpha: np.ndarray  # (K,) float32: the avatar's own coefficients
    residual: np.ndarray  # (N, 3) float32: the avatar minus the space's version of it (0 off the skin)
    eye: np.ndarray  # (N,) int8: 0, or 1 (right) / 2 (left) for the eyeballs
    eye_centre: np.ndarray  # (2, 3): the eyeballs' sphere centres (right, left)
    eye_radius: np.ndarray  # (2,)
    rim: np.ndarray  # (2, N) float32: each eye's rim weights (sum 1)
    inner: np.ndarray  # (N,) int8: 0, or 1 (upper teeth and gums), 2 (lower), 3 (tongue)
    inner_weights: np.ndarray  # (3, N) float32: the skin that moves each inner group (sum 1, or all 0)
    bones: np.ndarray  # (F,) int: skeleton indices of the face bones moved by the skin
    bone_weights: np.ndarray  # (F, N) float32 (sum 1)
    others: np.ndarray  # (P,) int: the other primitives that follow the skin
    other_nearest: np.ndarray  # (P, M, NEAREST) int: per vertex of each (padded with -1), its skin vertices
    other_weights: np.ndarray  # (P, M, NEAREST) float32
    tie: np.ndarray = None  # (N,) int: the group of skin vertices that move together (the closed lips), or -1
    hidden_nearest: np.ndarray = None  # (N, NEAREST) int: for inner vertices away from the mouth, their skin
    hidden_weights: np.ndarray = None  # (N, NEAREST) float32 (0 for the others)
    landmarks: str = ""  # the stage-0 landmarks' key it was made with
    space: str = ""  # the identity space's commit
    timings: dict = field(default_factory=dict)

    @property
    def vertices(self) -> int:
        return len(self.parts)

    @property
    def skin(self) -> np.ndarray:
        return self.parts == SKIN

    def _at(self, space, arrays: np.ndarray, tied: bool) -> np.ndarray:
        """``(K, N, 3)``: ICT's per-vertex ``arrays`` (``(K, V, 3)``) at the head's points, turned and scaled into the
        avatar's space and faded at the neck (``tied``: the tied skin vertices moved alike); the eyeballs moved with
        their rims, the inner parts away from the mouth with their skin, the mouth's inside left still."""
        has = self.triangle >= 0
        out = np.zeros((len(arrays), self.vertices, 3), np.float32)
        tri = space.triangles[self.triangle[has]]  # (n, 3) ICT vertices
        sampled = np.einsum("nc,kncd->knd", self.barycentric[has], arrays[:, tri])
        sampled = self.scale * np.einsum("ij,knj->kni", self.rotation, sampled)
        out[:, has] = sampled * self.fade[has, None]
        if tied:
            out = _tied(out, self.tie)
        if self.hidden_weights is not None and self.hidden_weights.any():
            hidden = self.hidden_weights.sum(axis=1) > 0
            out[:, hidden] = np.einsum("nk,mnkd->mnd", self.hidden_weights[hidden],
                                       out[:, np.maximum(self.hidden_nearest[hidden], 0)])
        for e in range(2):
            mine = self.eye == e + 1
            out[:, mine] = np.einsum("n,knd->kd", self.rim[e], out)[:, None, :]
        return out

    def modes(self, space) -> np.ndarray:
        """``(K, N, 3)`` float32: the identity modes as offsets of the head's vertices (metres per unit; the closed
        lips stay closed)."""
        return self._at(space, space.identity, True)

    def expressions(self, space) -> np.ndarray:
        """``(E, N, 3)`` float32: ICT's expressions as offsets of the head's vertices (the lips may part)."""
        return self._at(space, space.expressions, False)

    def neutral(self, space, beta=None) -> np.ndarray:
        """``(N, 3)``: the space's own head (``beta``, default ``alpha``) at the head's points, in the avatar's space
        (the avatar minus ``residual`` for ``alpha``)."""
        beta = self.alpha if beta is None else np.asarray(beta, np.float64)
        has = self.triangle >= 0
        shape = space.shape(beta)
        out = np.zeros((self.vertices, 3))
        points = np.einsum("nc,ncd->nd", self.barycentric[has], shape[space.triangles[self.triangle[has]]])
        out[has] = self.scale * points @ self.rotation.T + self.translation
        return out

    # -- moving the rest of the head with the skin ----------------------------------------------------------------

    def transfer(self, model, offsets) -> Transfer:
        """What skin ``offsets`` (``(N, 3)``, bind pose; their values on the skin are used as given) do to the whole
        head: the eyeballs and the mouth's inside moved rigidly with their surroundings, the face bones' rest
        positions, the other parts' offsets (see the module docs)."""
        head = np.array(offsets, np.float64)
        prim = model.primitives[self.primitive]
        points = np.asarray(prim.positions, np.float64)
        head[~self.skin] = 0.0
        head = _tied(head[None], self.tie)[0]
        if self.hidden_weights is not None and self.hidden_weights.any():
            hidden = self.hidden_weights.sum(axis=1) > 0
            head[hidden] = np.einsum("nk,nkd->nd", self.hidden_weights[hidden],
                                     head[np.maximum(self.hidden_nearest[hidden], 0)])
        eyes = []
        for e in range(2):
            mine = self.eye == e + 1
            t, s = _translation_scale(points, head, self.rim[e], self.eye_centre[e], EYE_SCALE)
            head[mine] = t + (s - 1.0) * (points[mine] - self.eye_centre[e])
            eyes.append((t, s))
        inner_moves = []
        for g in range(3):
            mine = self.inner == g + 1
            w = self.inner_weights[g]
            if not mine.any() or w.sum() <= 0:
                inner_moves.append((np.zeros(3), 1.0))
                continue
            centre = points[mine].mean(axis=0)
            t, s = _translation_scale(points, head, w, centre, INNER_SCALE)
            head[mine] = t + (s - 1.0) * (points[mine] - centre)
            inner_moves.append((t, s))
        skeleton = model.skeleton
        bones = {}
        for b, w in zip(self.bones, self.bone_weights):
            bones[int(b)] = w.astype(np.float64) @ head
        for e, name in enumerate(("Bip01 REye", "Bip01 LEye")):
            b = skeleton.find(name)
            if b is not None:
                rest = skeleton.rest_global[b, :3, 3]
                t, s = eyes[e]
                bones[int(b)] = t + (s - 1.0) * (rest - self.eye_centre[e])
        tongue = skeleton.find("Bip01 MTongue")
        if tongue is not None and (self.inner == 3).any():
            rest = skeleton.rest_global[tongue, :3, 3]
            t, s = inner_moves[2]
            bones[int(tongue)] = t + (s - 1.0) * (rest - points[self.inner == 3].mean(axis=0))
        others = {}
        for k, prim_index in enumerate(self.others):
            count = len(model.primitives[prim_index].positions)
            near, w = self.other_nearest[k, :count], self.other_weights[k, :count]
            others[int(prim_index)] = np.einsum("mk,mkd->md", w, head[np.maximum(near, 0)])
        return Transfer(head, bones, others, eyes, inner_moves)

    # -- storing ---------------------------------------------------------------------------------------------------

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp.npz")
        values = {f.name: np.asarray(getattr(self, f.name)) for f in fields(self) if f.name != "timings"}
        np.savez_compressed(tmp, version=np.array(VERSION), **values)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path) -> HeadSpace:
        with np.load(path, allow_pickle=False) as data:
            if int(data["version"]) != VERSION:
                raise ValueError("another version")
            values = {f.name: data[f.name] for f in fields(cls) if f.name != "timings" and f.name in data}
        for name in ("model", "key", "landmarks", "space"):
            values[name] = str(values[name])
        values["primitive"] = int(values["primitive"])
        values["scale"] = float(values["scale"])
        return cls(**values)


@dataclass
class Transfer:
    """What skin offsets do to the whole head (:meth:`HeadSpace.transfer`): ``head`` ``(N, 3)`` offsets of the
    head primitive's vertices, ``bones`` the face bones' rest-position offsets by skeleton index, ``others`` the
    offsets of the other parts by primitive index, ``eyes`` and ``inner``: ``(translation, scale)`` of each eyeball
    and inner group."""

    head: np.ndarray
    bones: dict
    others: dict
    eyes: list
    inner: list


def _tied(values: np.ndarray, tie) -> np.ndarray:
    """``values`` (``(K, N, 3)``) with each group of tied vertices (``tie`` >= 0) given its mean."""
    if tie is None or not (np.asarray(tie) >= 0).any():
        return values
    tie = np.asarray(tie)
    mine = np.nonzero(tie >= 0)[0]
    groups = tie[mine]
    count = np.bincount(groups).astype(np.float64)
    out = np.array(values, copy=True)
    sums = np.zeros((values.shape[0], len(count), 3))
    np.add.at(sums, (slice(None), groups), values[:, mine])
    out[:, mine] = (sums / np.maximum(count, 1.0)[None, :, None])[:, groups]
    return out


def _translation_scale(points, offsets, weights, centre, limit) -> tuple[np.ndarray, float]:
    """``(t, s)``: the translation and uniform scale about ``centre`` that best explain ``offsets`` of the
    ``weights``-weighted ``points`` (``s`` within ``limit`` of 1)."""
    w = np.asarray(weights, np.float64)
    total = w.sum()
    if total <= 0:
        return np.zeros(3), 1.0
    w = w / total
    r = points - centre
    mr = w @ r
    md = w @ offsets
    rc = r - mr
    var = float((w * (rc * rc).sum(axis=1)).sum())
    sigma = float((w * ((offsets - md) * rc).sum(axis=1)).sum() / var) if var > 1e-12 else 0.0
    sigma = float(np.clip(sigma, -limit, limit))
    return md - sigma * mr, 1.0 + sigma


# -- making it ---------------------------------------------------------------------------------------------------


def space_key(model, head, primitive: int, space_commit: str) -> str:
    """The cache key: the head's geometry and skinning, the other parts, the stage-0 landmarks, the face bones, the
    space and this code's version."""
    h = hashlib.blake2b(f"headspace-{VERSION}-{space_commit}".encode(), digest_size=12)
    prim = model.primitives[primitive]
    arrays = [prim.positions, prim.indices, prim.joints, prim.weights, head.points, head.reliability]
    arrays += [model.skeleton.rest_global[:, :3, 3]]
    for other in model.primitives:
        if other is not prim:
            arrays += [other.positions, other.joints]
    for array in map(np.asarray, arrays):
        h.update(f"{array.dtype}{array.shape}".encode())
        h.update(np.ascontiguousarray(array).tobytes())
    h.update("\n".join(model.skeleton.names).encode())
    return h.hexdigest()


_memory: dict[str, HeadSpace] = {}


def head_space(model, head=None, space="load", *, cache_dir=None, landmarker=None, refresh: bool = False,
               progress=None) -> HeadSpace:
    """The :class:`HeadSpace` of ``model``'s head (see the module docs), cached. ``head``: its stage-0
    :class:`~bifold_demo.characters.headtex.landmarks.HeadLandmarks` (found and cached when not given, with
    ``landmarker``); ``space``: the :class:`~.space.IdentitySpace` (``"load"``: :func:`.space.load`, which raises
    ``FileNotFoundError`` without it; ``None``: without the space, for the free-form fallback: the parts and the
    regressors only); ``cache_dir``: as for the head textures (``False``: memory only)."""
    from ..headtex.landmarks import cache_folder, head_landmarks
    from . import space as identity

    if isinstance(space, str) and space == "load":
        space = identity.load()
    if head is None:
        head = head_landmarks(model, landmarker, cache_dir=cache_dir)
    key = space_key(model, head, head.primitive, space.commit if space is not None else "")
    if not refresh and key in _memory:
        return _memory[key]
    path = None if cache_dir is False else cache_folder(cache_dir) / f"headspace-{key}.npz"
    if not refresh and path is not None and path.is_file():
        try:
            found = HeadSpace.load(path)
        except Exception:  # noqa: BLE001 - a damaged or old file: made again
            found = None
        if found is not None and found.key == key:
            _memory[key] = found
            return found
    result = make_head_space(model, head, space, progress=progress)
    result.key = key
    _memory[key] = result
    if path is not None:
        try:
            result.save(path)
        except OSError:
            pass  # a read-only folder: kept in memory only
    return result


def head_parts(model, primitive: int) -> np.ndarray:
    """``(N,)`` int8 per vertex of the head primitive: :data:`SKIN`, :data:`EYE` (led by an eye bone) or
    :data:`INNER` (on no triangle any camera around the head sees)."""
    from ..headtex import fuse, raster
    from ..headtex.camera import PhotoCamera
    from ..headtex.landmarks import face_frame

    prim = model.primitives[primitive]
    skeleton = model.skeleton
    joints, weights = np.asarray(prim.joints, np.int64), np.asarray(prim.weights, np.float64)
    eye_bones = [skeleton.find(n) for n in ("Bip01 REye", "Bip01 LEye")]
    eye_share = sum(np.where(joints == b, weights, 0.0).sum(axis=1) for b in eye_bones if b is not None)
    parts = np.full(len(prim.positions), SKIN, np.int8)
    parts[np.asarray(eye_share) > 0.5] = EYE
    tri = np.asarray(prim.indices, np.int64).reshape(-1, 3)
    corners = np.asarray(prim.positions, np.float64)[tri]
    occluders = fuse._occluders(model, primitive)
    everything = np.concatenate([corners, occluders]) if len(occluders) else corners
    pivot = face_frame(model).pivot
    seen = np.zeros(len(tri), bool)
    rasters = []
    for yaw, pitch in fuse.INNER_VIEWS:
        camera = PhotoCamera.around(pivot, 0.8, yaw, pitch, fov=32.0, size=(fuse.INNER_SIZE, fuse.INNER_SIZE))
        r = raster.rasterise(camera, everything, cull=True)
        shown = np.unique(r.triangle[r.triangle >= 0])
        seen[shown[shown < len(tri)]] = True
        rasters.append((camera, r.nearest(1)))
    outer = np.zeros(len(prim.positions), bool)
    outer[tri[seen].ravel()] = True
    rest = np.nonzero(~outer & (parts == SKIN))[0]
    if len(rest):  # small triangles may own no pixel: a depth test for their vertices
        points = np.asarray(prim.positions, np.float64)[rest]
        normals = np.asarray(prim.normals, np.float64)[rest]
        found = np.zeros(len(rest), bool)
        for camera, near in rasters:
            to_camera = camera.position - points
            facing = np.einsum("nk,nk->n", normals, to_camera) / np.linalg.norm(to_camera, axis=1)
            found |= (facing > 0) & fuse._depth_test(camera, near, points, facing, fuse.INNER_TOLERANCE, 0.5)
        outer[rest[found]] = True
    parts[(parts == SKIN) & ~outer] = INNER
    return parts


def neck_fade(model, primitive: int, parts: np.ndarray, distance: float = NECK_FADE) -> np.ndarray:
    """``(N,)`` 0-1 per vertex: 0 at the neck's open edge and on what bones outside the head and neck move (the
    spine, the clavicles, the arms), rising smoothly to 1 ``distance`` away."""
    prim = model.primitives[primitive]
    points = np.asarray(prim.positions, np.float64)
    skeleton = model.skeleton
    head_bone = skeleton.find("Head")
    neck_bone = skeleton.find("Neck")
    ours = np.zeros(len(skeleton), bool)
    if head_bone is not None:
        ours[skeleton.descendants(head_bone)] = True
    if neck_bone is not None:
        ours[neck_bone] = True
    joints, weights = np.asarray(prim.joints, np.int64), np.asarray(prim.weights, np.float64)
    share = (weights * ours[np.clip(joints, 0, len(skeleton) - 1)]).sum(axis=1)
    anchored = share < 0.5
    tri = np.asarray(prim.indices, np.int64).reshape(-1, 3)
    skin_tri = tri[(parts[tri] == SKIN).all(axis=1)]
    node, count = mesh.weld(points)
    open_nodes = np.unique(mesh.open_edges(node[skin_tri]))
    if len(open_nodes):  # the open edges low on the head: the neck (not the eyes' and the mouth's rims)
        frame_y = _mouth_height(model)
        low = np.zeros(count, bool)
        low[open_nodes] = True
        node_y = np.full(count, np.inf)
        np.minimum.at(node_y, node, points[:, 1])
        low &= node_y < frame_y - 0.03
        anchored |= low[node]
    if not anchored.any():
        return np.ones(len(points), np.float32)
    base = points[anchored]
    d = np.full(len(points), np.inf)
    for s in range(0, len(points), 512):
        d[s:s + 512] = np.min(np.linalg.norm(points[s:s + 512, None] - base[None], axis=2), axis=1)
    return mesh.smoothstep(d / distance).astype(np.float32)


def _mouth_height(model) -> float:
    skeleton = model.skeleton
    ys = [skeleton.rest_global[skeleton.index(n), 1, 3] for n in ("Bip01 RMouthCorner", "Bip01 LMouthCorner")]
    return float(np.mean(ys))


def make_head_space(model, head, space, progress=None) -> HeadSpace:
    """:class:`HeadSpace` of ``model`` with its stage-0 landmarks ``head`` and the identity ``space`` (not cached;
    see :func:`head_space`)."""
    from . import space as identity

    timings = {}
    start = time.perf_counter()
    primitive = head.primitive
    prim = model.primitives[primitive]
    points = np.asarray(prim.positions, np.float64)
    tri = np.asarray(prim.indices, np.int64).reshape(-1, 3)
    parts = head_parts(model, primitive)
    fade = neck_fade(model, primitive, parts)
    timings["parts"] = time.perf_counter() - start
    if space is None:  # the free-form fallback: the parts and the regressors only
        n = len(points)
        result = _empty(model.name, primitive, parts, fade, n, head.key, timings)
        start = time.perf_counter()
        _regressors(result, model, points, tri, matched=parts == SKIN)
        timings["regressors"] = time.perf_counter() - start
        return result

    # -- the similarity from the landmarks, and the avatar's coefficients from them --------------------------------
    start = time.perf_counter()
    ict_marks = identity.space_landmarks()
    use = np.ones(cf.COUNT, bool)
    use[list(LEFT_OUT)] = False
    weight = np.minimum(head.reliability, ict_marks.reliability) * use
    weight[list(BROWS)] *= 0.5
    on_skin = (parts[tri[head.triangle]] == SKIN).all(axis=1)
    weight = weight * on_skin
    target = head.points
    neutral_marks = ict_marks.points(space.neutral)
    s, r, t = mesh.umeyama(neutral_marks, target, weight)
    mode_marks = ict_marks.offsets(space.identity)  # (K, 478, 3) in ICT's space
    alpha0 = _ridge(np.einsum("ij,klj->kli", s * r, mode_marks), target - (s * neutral_marks @ r.T + t), weight,
                    ALPHA_NOISE)
    instance = space.shape(alpha0)
    ict_points = s * instance @ r.T + t
    timings["alignment"] = time.perf_counter() - start

    # -- laying the avatar's skin onto ICT's head ------------------------------------------------------------------
    start = time.perf_counter()
    node, count = mesh.weld(points)
    skin_tri = tri[(parts[tri] != EYE).all(axis=1)]
    node_tri = node[skin_tri]
    node_points = np.zeros((count, 3))
    node_points[node] = points
    data = np.zeros(count, bool)  # a node is skin where any of its vertices is (the mouth's corners: skin and inside)
    np.logical_or.at(data, node, parts == SKIN)
    ict_index = mesh.SurfaceIndex(ict_points, space.triangles)
    boundary = np.zeros(space.vertices, bool)
    boundary[mesh.open_edges(space.triangles).ravel()] = True
    allowed = ~boundary[space.triangles].any(axis=1)  # matches on ICT's open edges (the neck's end) are no matches
    mark_nodes = node[tri[head.triangle]]  # (478, 3)
    target_marks = ict_marks.points(ict_points)
    vertex_sides = _avatar_sides(model, primitive)
    node_sides = np.zeros((count, 2))
    np.maximum.at(node_sides, node, vertex_sides)
    triangle_sides = _space_sides(space)[space.triangles].mean(axis=1)
    deformed = _register(node_points, node_tri, count, data, mark_nodes, head.barycentric, target_marks, weight,
                         ict_index, allowed, node_sides, triangle_sides, progress)
    deformed_normals = mesh.vertex_normals(deformed, node_tri)
    distance, ict_tri, ict_bary = ict_index.closest(
        deformed, allowed, _compatible(np.arange(count), node_sides, deformed_normals, ict_index, triangle_sides))
    plain = ict_tri < 0  # (nothing on the same side near: the nearest point)
    if plain.any():
        distance[plain], ict_tri[plain], ict_bary[plain] = ict_index.closest(deformed[plain], allowed)
    timings["registration"] = time.perf_counter() - start

    # -- the avatar's coefficients over the whole skin, alternated with the similarity -----------------------------
    start = time.perf_counter()
    has = data & np.isfinite(distance) & (ict_tri >= 0)
    good = has & (distance < MATCH_DISTANCE[1] * 2)
    corners = space.triangles[ict_tri[has]]
    bary = ict_bary[has]
    neutral_at = np.einsum("nc,ncd->nd", bary, space.neutral[corners].astype(np.float64))
    modes_at = np.einsum("nc,kncd->knd", bary, space.identity[:, corners].astype(np.float64))
    x = node_points[has]
    w = good[has].astype(np.float64)
    alpha = alpha0.copy()
    for _ in range(4):
        model_points = neutral_at + np.einsum("k,knd->nd", alpha, modes_at)
        s, r, t = mesh.umeyama(model_points, x, w)
        alpha = _ridge(np.einsum("ij,knj->kni", s * r, modes_at), x - (s * neutral_at @ r.T + t), w, ALPHA_NOISE)
    timings["coefficients"] = time.perf_counter() - start

    # -- per vertex ------------------------------------------------------------------------------------------------
    n = len(points)
    triangle = np.full(n, -1, np.int32)
    barycentric = np.zeros((n, 3), np.float32)
    vertex_has = has[node] & (parts == SKIN)
    triangle[vertex_has] = ict_tri[node][vertex_has]
    barycentric[vertex_has] = ict_bary[node][vertex_has]
    result = _empty(model.name, primitive, parts, fade, n, head.key, timings)
    result.triangle, result.barycentric = triangle, barycentric
    result.scale, result.rotation, result.translation = float(s), r, t
    result.alpha, result.space = alpha.astype(np.float32), space.commit
    own = result.neutral(space)
    result.residual[vertex_has] = (points - own)[vertex_has]
    start = time.perf_counter()
    _regressors(result, model, points, tri)
    timings["regressors"] = time.perf_counter() - start
    return result


def _empty(name, primitive, parts, fade, n, landmarks, timings) -> HeadSpace:
    """A :class:`HeadSpace` without the identity space (no correspondence, no coefficients), to be filled in."""
    return HeadSpace(name, primitive, "", parts, fade, np.full(n, -1, np.int32), np.zeros((n, 3), np.float32), 1.0,
                     np.eye(3), np.zeros(3), np.zeros(0, np.float32), np.zeros((n, 3), np.float32),
                     np.zeros(n, np.int8), np.zeros((2, 3)), np.zeros(2), np.zeros((2, n), np.float32),
                     np.zeros(n, np.int8), np.zeros((3, n), np.float32), np.zeros(0, np.int64),
                     np.zeros((0, n), np.float32), np.zeros(0, np.int64), np.zeros((0, 0, NEAREST), np.int64),
                     np.zeros((0, 0, NEAREST), np.float32), landmarks=landmarks, timings=timings)


def _ridge(modes, residual, weights, noise) -> np.ndarray:
    """The coefficients ``a`` minimising ``sum w |residual - a @ modes|^2 / noise^2 + |a|^2`` (``modes``
    ``(K, P, 3)``)."""
    k = len(modes)
    m = modes.reshape(k, -1)  # (K, 3P)
    w = np.repeat(np.asarray(weights, np.float64), 3)
    a = (m * w) @ m.T / noise ** 2 + np.eye(k)
    b = (m * w) @ residual.reshape(-1) / noise ** 2
    return np.linalg.solve(a, b)


def _register(points, triangles, count, data, mark_nodes, mark_bary, mark_targets, mark_weight, index, allowed,
              node_sides, triangle_sides, progress=None) -> np.ndarray:
    """The nodes ``points`` (``(count, 3)``) laid onto the surface of ``index`` (see the module docs): ``data`` the
    nodes matched to their closest points (on the same side of the lips and lids: ``node_sides`` and
    ``triangle_sides``, see :func:`_compatible`), the landmarks (their nodes and barycentric weights) held at
    ``mark_targets`` with ``mark_weight``. Returns the deformed nodes."""
    lap = mesh.laplacian(triangles, count)
    smooth = lap.T @ lap
    lonely = np.ones(count, bool)  # nodes in no triangle (the eyeballs): held still
    lonely[np.asarray(triangles).ravel()] = False
    hold = np.diag(np.where(lonely, 1.0, 1e-6))
    marks = mark_weight > 0
    rows = np.zeros((int(marks.sum()), count))
    for i, (nodes, bary) in enumerate(zip(mark_nodes[marks], mark_bary[marks])):
        np.add.at(rows[i], nodes, bary)
    targets = mark_targets[marks]
    mw = mark_weight[marks]
    start_spline = mesh.ThinPlate3D(rows @ points, targets, smoothing=1e-4)
    displacement = start_spline(points) - points
    steps = len(STIFFNESS)
    for step, stiffness in enumerate(STIFFNESS):
        f = step / max(steps - 1, 1)
        reach = MATCH_DISTANCE[0] * (1 - f) + MATCH_DISTANCE[1] * f
        landmark = LANDMARK_WEIGHT[0] * (1 - f) + LANDMARK_WEIGHT[1] * f
        for _ in range(ICP_ROUNDS):
            current = points + displacement
            all_normals = mesh.vertex_normals(current, triangles)
            idx = np.nonzero(data)[0]
            dist, tri, bary = index.closest(current[data], allowed,
                                            _compatible(idx, node_sides, all_normals, index, triangle_sides))
            closest = index.point(np.maximum(tri, 0), bary)
            normals = all_normals[data]
            corners = index.vertices[index.triangles[np.maximum(tri, 0)]]
            tn = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
            tn /= np.maximum(np.linalg.norm(tn, axis=1, keepdims=True), 1e-18)
            ok = (tri >= 0) & (dist < reach) & (np.einsum("nk,nk->n", normals, tn) > 0.5)
            w = np.zeros(count)
            target = points.copy()
            w[idx[ok]] = 1.0
            target[idx[ok]] = closest[ok]
            a = np.diag(w) + hold + stiffness * smooth + landmark * (rows.T * mw) @ rows
            b = w[:, None] * (target - points) + landmark * (rows.T * mw) @ (targets - rows @ points)
            displacement = np.linalg.solve(a, b)
        if progress:
            progress(f"[headshape] registration step {step + 1}/{steps}: {int(ok.sum())} of {int(data.sum())} "
                     f"matched")
    return points + displacement


def _ties(points, triangles, skin) -> np.ndarray:
    """``(N,)``: groups of skin vertices closer than :data:`TIE` that no edge joins (the two lips along a closed
    mouth, a lid's corners), -1 for the rest: they move together, so the lips stay closed."""
    node, count = mesh.weld(points)
    joined = set(map(tuple, mesh.edges(node[triangles]).tolist()))
    nodes = np.unique(node[np.nonzero(skin)[0]])
    where = np.zeros((count, 3))
    where[node] = points
    parent = np.arange(count)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    linked = np.zeros(count, bool)
    for start in range(0, len(nodes), 256):
        chunk = nodes[start:start + 256]
        d = np.linalg.norm(where[chunk][:, None] - where[nodes][None], axis=2)
        for a, b in zip(*np.nonzero(d < TIE)):
            u, v = int(chunk[a]), int(nodes[b])
            if u < v and (u, v) not in joined:
                ru, rv = find(u), find(v)
                if ru != rv:
                    parent[max(ru, rv)] = min(ru, rv)
                linked[u] = linked[v] = True
    group = np.array([find(i) for i in range(count)])
    out = np.full(len(points), -1, np.int64)
    mine = linked[node] & skin
    if mine.any():
        out[mine] = np.unique(group[node[mine]], return_inverse=True)[1].ravel()
    return out


def _avatar_sides(model, primitive: int) -> np.ndarray:
    """``(N, 2)`` per vertex of the head: how much the jaw's bones move it (the lower lip and the chin) and how much
    the upper lids' bones do."""
    prim = model.primitives[primitive]
    skeleton = model.skeleton
    joints, weights = np.asarray(prim.joints, np.int64), np.asarray(prim.weights, np.float64)
    out = np.zeros((len(joints), 2))
    for column, names in enumerate((JAW_BONES, LID_BONES)):
        for name in names:
            b = skeleton.find("Bip01 " + name)
            if b is not None:
                out[:, column] += np.where(joints == b, weights, 0.0).sum(axis=1)
    return np.clip(out, 0.0, 1.0)


def _space_sides(space) -> np.ndarray:
    """``(V, 2)`` per vertex of ICT's head: how far ``jawOpen`` moves it (the jaw's side) and how far ``eyeBlink``
    does (the upper lids), each relative to :data:`JAW_SCALE` and :data:`LID_SCALE` and at most 1."""
    names = list(space.expression_names)

    def moved(*wanted):
        out = np.zeros(space.vertices)
        for name in wanted:
            if name in names:  # (a space without the expression: no side there)
                out += np.linalg.norm(space.expressions[names.index(name)], axis=1)
        return out

    return np.clip(np.c_[moved("jawOpen") / JAW_SCALE, moved("eyeBlink_L", "eyeBlink_R") / LID_SCALE], 0.0, 1.0)


def _compatible(nodes, node_sides, node_normals, index, triangle_sides):
    """The test :meth:`~.mesh.SurfaceIndex.closest` applies to the matches of ``nodes`` (the queries, in order): a
    triangle of ICT's on the same side of the lips and the lids (:func:`_avatar_sides`, :func:`_space_sides`) that
    does not face away from the node's normal."""
    corners = index.vertices[index.triangles]
    tn = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    tn /= np.maximum(np.linalg.norm(tn, axis=1, keepdims=True), 1e-18)

    def check(queries, triangles):
        q = nodes[queries]
        same_side = (np.abs(node_sides[q] - triangle_sides[triangles]) < SIDE_TOLERANCE).all(axis=1)
        return same_side & (np.einsum("nk,nk->n", node_normals[q], tn[triangles]) > FACING)

    return check


def _regressors(result: HeadSpace, model, points, tri, matched=None) -> None:
    """Fill in the eyes, the inside's groups, the face bones and the other parts (see the module docs)."""
    from ..faces import FACE_BONES
    from ..outfit.regions import body_primitive

    skeleton = model.skeleton
    prim = model.primitives[result.primitive]
    joints, weights = np.asarray(prim.joints, np.int64), np.asarray(prim.weights, np.float64)
    skin = result.parts == SKIN
    skin_idx = np.nonzero(skin)[0]
    n = len(points)
    # eyes: sphere fits, rims
    for e, name in enumerate(("Bip01 REye", "Bip01 LEye")):
        b = skeleton.find(name)
        mine = np.zeros(n, bool)
        if b is not None:
            mine = (result.parts == EYE) & (np.where(joints == b, weights, 0.0).sum(axis=1) > 0.5)
        if mine.sum() < 4:
            continue
        result.eye[mine] = e + 1
        centre, radius = mesh.sphere_fit(points[mine])
        result.eye_centre[e], result.eye_radius[e] = centre, radius
        d = np.linalg.norm(points[skin_idx] - centre, axis=1) - radius
        directions = points - centre
        directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)
        axis = directions[mine].mean(axis=0)
        axis /= np.linalg.norm(axis)
        cap = np.degrees(np.arccos(np.clip(directions[mine] @ axis, -1.0, 1.0))).max()
        angle = np.degrees(np.arccos(np.clip(directions[skin_idx] @ axis, -1.0, 1.0)))
        near = (d < EYE_RIM) & (angle < cap + EYE_CAP_MARGIN)  # the lids over the eyeball's cap, not the brow
        if near.sum() < 8:
            near = np.zeros(len(skin_idx), bool)
            near[np.argsort(d)[:12]] = True
        w = np.exp(-((np.maximum(d[near], 0.0) / EYE_RIM) ** 2))
        result.rim[e, skin_idx[near]] = w / w.sum()
    # the mouth's inside: upper, lower, tongue
    jaw_side = np.zeros(len(skeleton), bool)
    for name in JAW_BONES:
        b = skeleton.find("Bip01 " + name)
        if b is not None:
            jaw_side[b] = True
    on_jaw = (weights * jaw_side[np.clip(joints, 0, len(skeleton) - 1)]).sum(axis=1) > 0.5
    tongue_bone = skeleton.find("Bip01 MTongue")
    on_tongue = np.where(joints == tongue_bone, weights, 0.0).sum(axis=1) > 0.5 if tongue_bone is not None \
        else np.zeros(n, bool)
    mouth = np.mean([skeleton.rest_global[skeleton.index(n), :3, 3] for n in ("Bip01 RMouthCorner",
                                                                               "Bip01 LMouthCorner")], axis=0)
    inner = (result.parts == INNER) & (np.linalg.norm(points - mouth, axis=1) < MOUTH_REACH)
    matched = skin & (result.triangle >= 0) if matched is None else np.asarray(matched)
    hidden = ((result.parts == INNER) & ~inner) | (skin & ~matched)  # (inside the neck, a nostril; unmatched skin)
    result.hidden_nearest = np.full((n, NEAREST), -1, np.int64)
    result.hidden_weights = np.zeros((n, NEAREST), np.float32)
    source = np.nonzero(matched)[0]
    if hidden.any() and len(source):
        d = np.linalg.norm(points[hidden][:, None] - points[source][None], axis=2)
        order = np.argsort(d, axis=1)[:, :NEAREST]
        dd = np.take_along_axis(d, order, axis=1)
        w = 1.0 / (dd + 0.002) ** 2
        result.hidden_nearest[hidden] = source[order]
        result.hidden_weights[hidden] = (w / w.sum(axis=1, keepdims=True)) * result.fade[hidden, None]
    result.tie = _ties(points, tri, skin)
    result.inner[inner & ~on_jaw] = 1
    result.inner[inner & on_jaw & ~on_tongue] = 2
    result.inner[inner & on_tongue] = 3
    for g in range(3):
        mine = result.inner == g + 1
        if not mine.any():
            continue
        candidates = skin & (on_jaw if g > 0 else ~on_jaw)
        cand_idx = np.nonzero(candidates)[0]
        d = np.min(np.linalg.norm(points[cand_idx, None] - points[mine][None], axis=2), axis=1)
        near = d < INNER_REACH
        if near.sum() < 4:
            near = np.zeros(len(cand_idx), bool)
            near[np.argsort(d)[:8]] = True
        w = np.exp(-((d[near] / INNER_SIGMA) ** 2))
        result.inner_weights[g, cand_idx[near]] = w / w.sum()
    # face bones (the eyes and the tongue go with their parts)
    bones, rows = [], []
    for name in FACE_BONES:
        if name in ("REye", "LEye", "MTongue"):
            continue
        b = skeleton.find("Bip01 " + name)
        if b is None:
            continue
        rest = skeleton.rest_global[b, :3, 3]
        d = np.linalg.norm(points[skin_idx] - rest, axis=1)
        near = d < 3.0 * BONE_SIGMA
        if near.sum() < 6:
            near = np.zeros(len(skin_idx), bool)
            near[np.argsort(d)[:8]] = True
        w = np.exp(-((d[near] / BONE_SIGMA) ** 2)) + 1e-9
        row = np.zeros(n, np.float32)
        row[skin_idx[near]] = w / w.sum()
        bones.append(b)
        rows.append(row)
    result.bones = np.asarray(bones, np.int64)
    result.bone_weights = np.asarray(rows, np.float32).reshape(len(rows), n)
    # the other parts the head moves
    head_bone = skeleton.find("Head")
    ours = np.zeros(len(skeleton), bool)
    if head_bone is not None:
        ours[skeleton.descendants(head_bone)] = True
    body = body_primitive(model)
    others = []
    for k, other in enumerate(model.primitives):
        if k in (result.primitive, body):
            continue
        j, w = np.asarray(other.joints, np.int64), np.asarray(other.weights, np.float64)
        share = (w * ours[np.clip(j, 0, len(skeleton) - 1)]).sum(axis=1)
        if share.mean() >= 0.5:
            others.append(k)
    longest = max((len(model.primitives[k].positions) for k in others), default=0)
    nearest = np.full((len(others), longest, NEAREST), -1, np.int64)
    near_w = np.zeros((len(others), longest, NEAREST), np.float32)
    skin_points = points[skin_idx]
    for i, k in enumerate(others):
        q = np.asarray(model.primitives[k].positions, np.float64)
        for s in range(0, len(q), 256):
            d = np.linalg.norm(q[s:s + 256, None] - skin_points[None], axis=2)
            order = np.argsort(d, axis=1)[:, :NEAREST]
            dd = np.take_along_axis(d, order, axis=1)
            w = 1.0 / (dd + 0.002) ** 2
            nearest[i, s:s + len(order)] = skin_idx[order]
            near_w[i, s:s + len(order)] = w / w.sum(axis=1, keepdims=True)
    result.others = np.asarray(others, np.int64)
    result.other_nearest, result.other_weights = nearest, near_w
