"""Subdivided heads for close-ups: a finer head mesh where a window shows a face large (step S4 of
``docs/research/face_recognition_focus.md``; the approach: ``docs/research/high_res_faces.md``, sections 2.3 and 3.1,
A4; the measurements: ``docs/research/closeups_results.md``).

The Rocketbox heads have about 1,700 vertices and 3,100 triangles; a face edge is 10 mm long on the median and 23 mm
at the 90th percentile. In a window with 100 pixels or more between the eyes the jaw line, the nose, the chin and the
ears show their facets. :func:`subdivide` makes a copy of a model whose head primitive is split into 4 triangles per
triangle (level 1, "x4") or 16 (level 2, "x16"), on a smooth surface through the original vertices: once per head
geometry (the avatar's own head, or a head fitted to a person after :func:`~.headshape.rig.apply_shape`), cached.

**The surface: curved triangles.** Every triangle becomes a cubic Bezier triangle (Vlachos et al.'s PN triangles)
through its three corners, tangent there to the surface's tangent planes, and the new vertices are points of these
patches (level 1: the middle of each edge; level 2: four points per edge side and three inside). Each edge's curve
depends only on its two ends, so neighbouring triangles share it and no crack opens. The tangent planes are the mesh's
own, welded (the angle-weighted normals of the triangles round a position, across UV seams). This is an interpolating
scheme like the modified Butterfly the plan named, but local: a new point depends on one triangle's corners and
normals, not on two rings of neighbours with fixed weights, so the coarse heads' uneven edges (2 to 50 mm side by side)
do not make it ripple (measured against a Butterfly scheme in the results note). Along an edge of a circle sampled at
angle steps of ``2 t`` a new point misses the circle by ``R t^4 / 4``, as the Butterfly's four-point rule does on even
steps and on uneven ones too.

**What keeps its shape** (the edge rules, :data:`RULES`):

* *Creases* (``CREASE``): the lip line and other skin that touches without an edge (:func:`.headshape.head._ties`,
  the closed lips and the lids' corners), the lid margins (where the skin turns from facing out to facing the eyeball,
  within :data:`LID_REACH` of it), where the visible skin meets the mouth's inside, and folds sharper than
  :data:`CREASE_FOLD`. A crease's new points lie on a curve along the crease (tangent at a vertex the mean direction
  of its two crease edges; straight at a corner, where a crease turns more than :data:`CREASE_TURN` or meets others),
  and the surface on either side keeps its own tangent plane there (the normals of the triangles on that side): the
  folds stay folds. The closed lips' two creases get the same new points (their shared offset from the edges), so
  closed lips stay closed.
* *The neck's open edge* and the *mouth's inside* (teeth, gums, tongue: the head's ``INNER`` part) are split along
  their straight edges (``LINEAR``): the neck's edge stays on its original edges, so the seam to the body's collar
  does not open, and the inside of the mouth keeps its exact shape.
* *Eyeballs* (``SPHERE``): new points on their spheres (the radius interpolated, so a cornea keeps its bulge).

**The rest of a vertex** comes from its triangle by the same barycentric weights (its "stencil"): UVs per triangle
(seams stay exact, a new vertex on a seam is split like its ends), the skin weights (the four strongest kept,
renormalised), the artist's normals interpolated (what the coarse head's shading shows at that point of its surface:
the shading inside the face stays as it was, the outline is what changes; a fitted head's normals were turned with
its shape already, :func:`~.headshape.rig.apply_shape`), tangents made again (:func:`~.tangents.vertex_tangents`).
The original vertices come first and keep everything as they were, bit for bit.

**Labels** (:class:`HeadLabels`): the parts of the head (skin, eyeballs, the mouth's inside: found by visibility, about
1.2 s once per avatar head, cached in the head-texture cache), the eyes, the inner groups, the neck's fade and the ties
that :func:`~.headshape.rig.rig_checks` needs; :func:`head_labels` makes them, :func:`labels_from` takes them from a
:class:`~.headshape.head.HeadSpace`. They are carried to the subdivided head (a new vertex between two parts gets
none, so its triangles count as they did), so the S3 checks run on it: :func:`check` compares the head subdivided on
its curved surface with the same head split along its flat triangles (the coarse head's surface, vertex for vertex) in
the 12 faces of the face rig. The subdivided model shares the skeleton, the inverse binds and the face rig
(:func:`~.faces.share_rig`) of the model it was made from, so it poses exactly as that model does.

**Cost** (measured in the results note): about 0.1-0.4 s per head and level, 0.2 MB (x4) and 2-3 MB (x16) of arrays
and GPU buffers per head, cached in memory by geometry (fitted heads, biometric, only there).
"""

from __future__ import annotations

import hashlib
import threading
import time
import weakref
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np

from .model import CharacterModel, SkinnedPrimitive

VERSION = 1
LEVELS = (1, 2)  # 4 and 16 triangles per triangle
JOINTS = 4  # skin weights kept per vertex
CREASE_FOLD = 100.0  # degrees between two skin triangles' normals beyond which their edge is a crease
CREASE_TURN = 60.0  # degrees a crease may turn at a vertex and stay smooth there (else a corner: straight ends)
LID_REACH = 0.006  # metres outside an eyeball: skin there that faces it or turns to face it is its lids
LID_CONE = 15.0  # degrees beyond the edge of the eyeball's cap the lids reach
CACHED = 64  # subdivided head primitives kept in memory
LINEAR, CURVED, CREASE, SPHERE = 0, 1, 2, 3
RULES = {LINEAR: "straight", CURVED: "curved", CREASE: "crease", SPHERE: "sphere"}


@dataclass
class HeadLabels:
    """Per-vertex labels of a head primitive, as :func:`~.headshape.rig.rig_checks` reads them from a
    :class:`~.headshape.head.HeadSpace`: ``parts`` (``SKIN``, ``EYE``, ``INNER`` of :mod:`.headshape.head`; 0 for a new
    vertex between two parts of a subdivided head), ``eye`` (0, 1 right, 2 left), ``inner`` (0, 1 the upper teeth and
    gums, 2 the lower ones, 3 the tongue), ``fade`` (0 on the neck's edge), ``tie`` (groups of skin that touches without
    an edge, the closed lips; -1 for the rest), and the eyeballs' spheres."""

    primitive: int
    parts: np.ndarray
    eye: np.ndarray
    inner: np.ndarray
    fade: np.ndarray
    tie: np.ndarray
    eye_centre: np.ndarray = field(default_factory=lambda: np.zeros((2, 3)))
    eye_radius: np.ndarray = field(default_factory=lambda: np.zeros(2))

    @property
    def vertices(self) -> int:
        return len(self.parts)


@dataclass
class Subdivided:
    """A head primitive subdivided (:func:`subdivide_primitive`): the new ``primitive``, its carried ``labels``, the
    coarse triangle each new triangle lies in (``parent``), the coarse mesh's welded ``edges`` (node pairs) with the
    ``rules`` they were split by (:data:`RULES`), and how long it took."""

    primitive: SkinnedPrimitive
    labels: HeadLabels
    level: int
    smooth: bool
    parent: np.ndarray
    edges: np.ndarray
    rules: np.ndarray
    seconds: float = 0.0

    def counts(self) -> dict:
        """How many coarse edges each rule split (by name)."""
        return {name: int((self.rules == rule).sum()) for rule, name in RULES.items()}


# -- labels ------------------------------------------------------------------------------------------------------------


def labels_from(hs) -> HeadLabels:
    """The labels of a :class:`~.headshape.head.HeadSpace` (its own fade, which the jaw line ends)."""
    n = hs.vertices
    tie = np.full(n, -1, np.int64) if hs.tie is None else np.asarray(hs.tie, np.int64)
    return HeadLabels(int(hs.primitive), np.asarray(hs.parts, np.int8), np.asarray(hs.eye, np.int8),
                      np.asarray(hs.inner, np.int8), np.asarray(hs.fade, np.float32), tie,
                      np.asarray(hs.eye_centre, np.float64), np.asarray(hs.eye_radius, np.float64))


def head_labels(model, primitive: int | None = None, cache_dir=None) -> HeadLabels:
    """The :class:`HeadLabels` of ``model``'s head (``primitive``: its index, default the head of
    :func:`~.face_photo.head_layout`), made the way :func:`~.headshape.head.head_space` makes them, without the
    identity space or the landmarks (MediaPipe): the parts by visibility (cached on disk for an avatar's own head, in
    memory for a fitted one), the eyes by their bones, the mouth's inside by the jaw's and the tongue's bones, the fade
    from the neck's edge (not ended by the jaw line, which needs the landmarks), the ties. Raises ``ValueError`` for a
    model without a head with face bones."""
    from .face_photo import head_layout
    from .headshape import head as hd

    if primitive is None:
        layout = head_layout(model)
        if layout is None:
            raise ValueError(f"{model.name} has no head with face bones")
        primitive = layout.primitive
    prim = model.primitives[primitive]
    points = np.asarray(prim.positions, np.float64)
    tri = np.asarray(prim.indices, np.int64).reshape(-1, 3)
    parts = _parts(model, primitive, cache_dir)
    eye, centres, radii = _eyes(model, prim, parts, points)
    inner = _inner(model, prim, parts, points)
    fade = hd.neck_fade(model, primitive, parts).astype(np.float32)
    tie = hd._ties(points, tri, parts == hd.SKIN)
    return HeadLabels(int(primitive), parts, eye, inner, fade, np.asarray(tie, np.int64), centres, radii)


_parts_memory: OrderedDict[str, np.ndarray] = OrderedDict()


def _parts(model, primitive: int, cache_dir) -> np.ndarray:
    """:func:`.headshape.head.head_parts`, cached: on disk (the head-texture cache) for an avatar's own head, in
    memory for one fitted to a person (``metadata["head_shape"]``)."""
    from .headshape import head as hd
    from .headtex.landmarks import cache_folder
    from .outfit.regions import body_primitive

    prim = model.primitives[primitive]
    h = hashlib.blake2b(f"subdivide-parts-{VERSION}-{hd.VERSION}".encode(), digest_size=12)
    arrays = [prim.positions, prim.indices, prim.joints, prim.weights, model.skeleton.rest_global[:, :3, 3]]
    body = body_primitive(model)
    if body is not None and body != primitive:
        arrays += [model.primitives[body].positions, model.primitives[body].indices]
    for array in map(np.asarray, arrays):
        h.update(f"{array.dtype}{array.shape}".encode())
        h.update(np.ascontiguousarray(array).tobytes())
    h.update("\n".join(model.skeleton.names).encode())
    key = h.hexdigest()
    if key in _parts_memory:
        _parts_memory.move_to_end(key)
        return _parts_memory[key]
    personal = "head_shape" in getattr(model, "metadata", {})  # a head fitted to a person: biometric, memory only
    path = None if personal or cache_dir is False else cache_folder(cache_dir) / f"subdivide-parts-{key}.npz"
    parts = None
    if path is not None and path.is_file():
        try:
            with np.load(path, allow_pickle=False) as data:
                found = np.asarray(data["parts"], np.int8)
            if len(found) == len(prim.positions):
                parts = found
        except Exception:  # noqa: BLE001 - a damaged file: made again
            parts = None
    if parts is None:
        parts = hd.head_parts(model, primitive)
        if path is not None:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_name(f".{path.name}.{threading.get_ident()}.tmp.npz")
                np.savez_compressed(tmp, parts=parts)
                tmp.replace(path)
            except OSError:
                pass  # a read-only folder: kept in memory only
    _parts_memory[key] = parts
    while len(_parts_memory) > CACHED:
        _parts_memory.popitem(last=False)
    return parts


def _eyes(model, prim, parts, points) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The eyeballs (vertices of the ``EYE`` part led by an eye bone; 1 right, 2 left) and their spheres, as
    :mod:`.headshape.head` finds them."""
    from .headshape import head as hd
    from .headshape import mesh

    skeleton = model.skeleton
    joints, weights = np.asarray(prim.joints, np.int64), np.asarray(prim.weights, np.float64)
    eye = np.zeros(len(points), np.int8)
    centres, radii = np.zeros((2, 3)), np.zeros(2)
    for e, name in enumerate(("Bip01 REye", "Bip01 LEye")):
        b = skeleton.find(name)
        if b is None:
            continue
        mine = (parts == hd.EYE) & (np.where(joints == b, weights, 0.0).sum(axis=1) > 0.5)
        if mine.sum() < 4:
            continue
        eye[mine] = e + 1
        centres[e], radii[e] = mesh.sphere_fit(points[mine])
    return eye, centres, radii


def _inner(model, prim, parts, points) -> np.ndarray:
    """The mouth's inside in groups (1 the upper teeth and gums, 2 the lower ones, 3 the tongue), as
    :mod:`.headshape.head` finds them: the ``INNER`` part near the mouth, by the bones that move it."""
    from .headshape import head as hd

    skeleton = model.skeleton
    joints, weights = np.asarray(prim.joints, np.int64), np.asarray(prim.weights, np.float64)
    out = np.zeros(len(points), np.int8)
    corners = [skeleton.find(n) for n in ("Bip01 RMouthCorner", "Bip01 LMouthCorner")]
    if any(c is None for c in corners):
        return out
    jaw_side = np.zeros(len(skeleton), bool)
    for name in hd.JAW_BONES:
        b = skeleton.find("Bip01 " + name)
        if b is not None:
            jaw_side[b] = True
    on_jaw = (weights * jaw_side[np.clip(joints, 0, len(skeleton) - 1)]).sum(axis=1) > 0.5
    tongue = skeleton.find("Bip01 MTongue")
    on_tongue = np.where(joints == tongue, weights, 0.0).sum(axis=1) > 0.5 if tongue is not None \
        else np.zeros(len(points), bool)
    mouth = np.mean([skeleton.rest_global[c, :3, 3] for c in corners], axis=0)
    inner = (parts == hd.INNER) & (np.linalg.norm(points - mouth, axis=1) < hd.MOUTH_REACH)
    out[inner & ~on_jaw] = 1
    out[inner & on_jaw & ~on_tongue] = 2
    out[inner & on_tongue] = 3
    return out


# -- the subdivision ---------------------------------------------------------------------------------------------------


def _unit(v: np.ndarray) -> np.ndarray:
    length = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(length, 1e-30)


class _Union:
    """Union-find over ``n`` items."""

    def __init__(self, n: int):
        self.parent = np.arange(n)

    def find(self, i: int) -> int:
        parent = self.parent
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def join(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)

    def roots(self) -> np.ndarray:
        return np.array([self.find(i) for i in range(len(self.parent))], np.int64)


def _grid(n: int):
    """The barycentric grid of a triangle split into ``n`` segments per edge: per point ``(i, j, k)`` (the weights of
    corners 0, 1, 2 times ``n``), its kind (0 corner, 1 edge, 2 inside), the corner or local edge (0: corner 0 to 1,
    1: 1 to 2, 2: 2 to 0) and the step along that edge from its first corner; and the sub-triangles (index triples,
    turned as the triangle is)."""
    points, where = [], {}
    for j in range(n + 1):
        for k in range(n + 1 - j):
            where[(j, k)] = len(points)
            points.append((n - j - k, j, k))
    kind, local, step = [], [], []
    for i, j, k in points:
        zeros = (i == 0) + (j == 0) + (k == 0)
        if zeros == 2:
            kind.append(0)
            local.append(0 if i else (1 if j else 2))
            step.append(0)
        elif zeros == 1:
            kind.append(1)
            if k == 0:
                local.append(0)
                step.append(j)
            elif i == 0:
                local.append(1)
                step.append(k)
            else:
                local.append(2)
                step.append(i)
        else:
            kind.append(2)
            local.append(-1)
            step.append(0)
    triangles = []
    for j in range(n):
        for k in range(n - j):
            triangles.append((where[(j, k)], where[(j + 1, k)], where[(j, k + 1)]))
            if j + k <= n - 2:
                triangles.append((where[(j + 1, k)], where[(j + 1, k + 1)], where[(j, k + 1)]))
    return (np.array(points, np.float64) / n, np.array(kind), np.array(local), np.array(step),
            np.array(triangles, np.int64))


def subdivide_primitive(prim: SkinnedPrimitive, labels: HeadLabels, level: int = 1,
                        smooth: bool = True) -> Subdivided:
    """``prim`` (a head primitive with its ``labels``) split into ``4 ** level`` triangles per triangle (see the
    module docs); ``smooth`` False splits it along its flat triangles instead (the same vertices, triangles and
    labels on the coarse head's own surface: what :func:`check` compares with)."""
    from .headshape import head as hd
    from .headshape import mesh
    from .tangents import vertex_tangents

    if level < 1:
        raise ValueError(f"level {level}: at least 1")
    start = time.perf_counter()
    n = 2 ** int(level)
    p = np.asarray(prim.positions, np.float64)
    tri = np.asarray(prim.indices, np.int64).reshape(-1, 3)
    count_v, count_t = len(p), len(tri)
    parts = np.asarray(labels.parts)
    if len(parts) != count_v:
        raise ValueError(f"labels for {len(parts)} vertices, the primitive has {count_v}")

    # -- the welded mesh: nodes, edges, the triangles round each edge ----------------------------------------------
    node, nodes = mesh.weld(p)
    P = np.zeros((nodes, 3))
    P[node] = p
    nt = node[tri]
    a_of, b_of = nt, np.roll(nt, -1, axis=1)  # local edge k runs from corner k to corner k + 1
    lo, hi = np.minimum(a_of, b_of), np.maximum(a_of, b_of)
    edges, edge_of, tri_count = np.unique(np.stack([lo.ravel(), hi.ravel()], axis=1), axis=0, return_inverse=True,
                                          return_counts=True)
    edge_of = edge_of.reshape(-1, 3)
    forward = a_of < b_of  # the triangle's edge runs from the edge's first node to its second
    count_e = len(edges)
    order = np.argsort(edge_of.ravel(), kind="stable")
    first = np.full(count_e, -1, np.int64)
    second = np.full(count_e, -1, np.int64)
    flat_index = np.arange(3 * count_t)[order]
    sorted_edges = edge_of.ravel()[order]
    starts = np.searchsorted(sorted_edges, np.arange(count_e))
    first[:] = flat_index[starts]
    has_two = tri_count >= 2
    second[has_two] = flat_index[starts[has_two] + 1]
    t1, k1 = first // 3, first % 3
    t2, k2 = np.where(has_two, second // 3, -1), np.where(has_two, second % 3, -1)

    # -- face normals, corner angles, node labels ------------------------------------------------------------------
    corners = P[nt]
    face = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    face_n = _unit(face)
    angle = np.zeros((count_t, 3))
    for c in range(3):
        u = corners[:, (c + 1) % 3] - corners[:, c]
        v = corners[:, (c + 2) % 3] - corners[:, c]
        cos = np.einsum("ij,ij->i", u, v) / np.maximum(np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1), 1e-30)
        angle[:, c] = np.arccos(np.clip(cos, -1.0, 1.0))
    node_part = np.zeros(nodes, np.int8)
    for part in (hd.INNER, hd.EYE, hd.SKIN):  # (a node is skin where any of its vertices is: the mouth's corners)
        has = np.zeros(nodes, bool)
        np.logical_or.at(has, node, parts == part)
        node_part[has] = part
    node_eye = np.zeros(nodes, np.int8)
    np.maximum.at(node_eye, node, np.asarray(labels.eye, np.int8))
    tie = np.asarray(labels.tie, np.int64) if labels.tie is not None else np.full(count_v, -1, np.int64)
    node_tie = np.full(nodes, -1, np.int64)
    np.maximum.at(node_tie, node, tie)
    node_fade = np.full(nodes, np.inf)
    np.minimum.at(node_fade, node, np.asarray(labels.fade, np.float64))
    tri_parts = parts[tri]
    all_skin = (tri_parts == hd.SKIN).all(axis=1)

    # -- the edge rules ----------------------------------------------------------------------------------------------
    ea, eb = edges[:, 0], edges[:, 1]
    skin_edge = (node_part[ea] == hd.SKIN) & (node_part[eb] == hd.SKIN)
    eye_edge = (node_part[ea] == hd.EYE) & (node_part[eb] == hd.EYE) & (node_eye[ea] == node_eye[eb]) \
        & (node_eye[ea] > 0)
    interior = tri_count == 2
    safe_t2 = np.maximum(t2, 0)
    fold = np.where(interior, np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", face_n[t1], face_n[safe_t2]),
                                                           -1.0, 1.0))), 0.0)
    crease = skin_edge & ((node_tie[ea] >= 0) & (node_tie[eb] >= 0))  # the lip line, the lids' corners
    crease |= skin_edge & interior & (all_skin[t1] != all_skin[safe_t2])  # the visible skin against the inside
    crease |= skin_edge & interior & (fold > CREASE_FOLD)
    crease |= skin_edge & (tri_count > 2)
    crease |= skin_edge & interior & _lid_margins(P, nt, face_n, all_skin, node_eye, t1, safe_t2, labels, p)
    rules = np.full(count_e, LINEAR, np.int8)
    rules[skin_edge & interior] = CURVED
    rules[crease & (tri_count >= 2)] = CREASE
    rules[eye_edge] = SPHERE
    rules[(node_fade[ea] <= 0.0) & (node_fade[eb] <= 0.0)] = LINEAR  # the neck's edge and the skin the body moves
    rules[edges[:, 0] == edges[:, 1]] = LINEAR

    # -- the tangent planes: per corner, the normal of its side of the creases ---------------------------------------
    union = _Union(3 * count_t)
    joined = np.nonzero(interior & (rules != CREASE))[0]
    at_a1 = np.where(forward[t1, k1], k1, (k1 + 1) % 3)
    at_b1 = np.where(forward[t1, k1], (k1 + 1) % 3, k1)
    at_a2 = np.where(forward[safe_t2, np.maximum(k2, 0)], k2, (k2 + 1) % 3)
    at_b2 = np.where(forward[safe_t2, np.maximum(k2, 0)], (k2 + 1) % 3, k2)
    for e in joined.tolist():
        union.join(3 * int(t1[e]) + int(at_a1[e]), 3 * int(t2[e]) + int(at_a2[e]))
        union.join(3 * int(t1[e]) + int(at_b1[e]), 3 * int(t2[e]) + int(at_b2[e]))
    sector = union.roots()
    sums = np.zeros((3 * count_t, 3))
    np.add.at(sums, sector, (angle[:, :, None] * face_n[:, None, :]).reshape(-1, 3))
    corner_n = _unit(sums[sector]).reshape(count_t, 3, 3)  # (T, 3 corners, xyz)
    n_a = corner_n[t1, at_a1]
    n_b = corner_n[t1, at_b1]

    # -- the creases' curves: tangents along them ---------------------------------------------------------------------
    d = P[eb] - P[ea]
    length = np.linalg.norm(d, axis=1)
    creased = np.nonzero(rules == CREASE)[0]
    tangent = np.zeros((nodes, 3))
    smooth_node = np.zeros(nodes, bool)
    if len(creased):
        ends = edges[creased].ravel()
        degree = np.bincount(ends, minlength=nodes)
        twos = np.nonzero(degree == 2)[0]
        if len(twos):
            other = np.full((nodes, 2), -1, np.int64)
            filled = np.zeros(nodes, np.int64)
            for a, b in edges[creased].tolist():
                for x, y in ((a, b), (b, a)):
                    if degree[x] == 2:
                        other[x, filled[x]] = y
                        filled[x] += 1
            u1 = _unit(P[twos] - P[other[twos, 0]])
            u2 = _unit(P[other[twos, 1]] - P[twos])
            turn = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", u1, u2), -1.0, 1.0)))
            ok = turn < CREASE_TURN
            tangent[twos[ok]] = _unit(u1[ok] + u2[ok])
            smooth_node[twos[ok]] = True
    along = np.where(np.einsum("ij,ij->i", tangent[ea], d) >= 0, 1.0, -1.0)[:, None]
    t_a = np.where(smooth_node[ea, None], tangent[ea] * along, _unit(d))
    along = np.where(np.einsum("ij,ij->i", tangent[eb], d) >= 0, 1.0, -1.0)[:, None]
    t_b = np.where(smooth_node[eb, None], tangent[eb] * along, _unit(d))

    # -- each edge's cubic: its two inner control points ---------------------------------------------------------------
    c1 = P[ea] + d / 3.0
    c2 = P[ea] + 2.0 * d / 3.0
    if smooth:
        curved = rules == CURVED
        da, db = np.einsum("ij,ij->i", d, n_a), np.einsum("ij,ij->i", d, n_b)
        c1[curved] = ((2.0 * P[ea] + P[eb] - da[:, None] * n_a) / 3.0)[curved]
        c2[curved] = ((2.0 * P[eb] + P[ea] + db[:, None] * n_b) / 3.0)[curved]
        cr = rules == CREASE
        c1[cr] = (P[ea] + t_a * length[:, None] / 3.0)[cr]
        c2[cr] = (P[eb] - t_b * length[:, None] / 3.0)[cr]
    sphere = rules == SPHERE if smooth else np.zeros(count_e, bool)
    centres, radii = _spheres(p, labels)
    centre_of_node = np.zeros((nodes, 3))
    for e in (1, 2):
        mine = node_eye == e
        centre_of_node[mine] = centres[e - 1]

    # -- the edges' new points (canonical: from the edge's first node) -------------------------------------------------
    s = np.arange(1, n) / n  # (n - 1,)
    edge_points = np.zeros((count_e, n - 1, 3))
    for j, t in enumerate(s):
        edge_points[:, j] = ((1 - t) ** 3 * P[ea] + 3 * (1 - t) ** 2 * t * c1 + 3 * (1 - t) * t * t * c2
                             + t ** 3 * P[eb])
    if sphere.any():
        sel = np.nonzero(sphere)[0]
        c = centre_of_node[ea[sel]]
        ra, rb = np.linalg.norm(P[ea[sel]] - c, axis=1), np.linalg.norm(P[eb[sel]] - c, axis=1)
        ua, ub = _unit(P[ea[sel]] - c), _unit(P[eb[sel]] - c)
        for j, t in enumerate(s):
            edge_points[sel, j] = c + ((1 - t) * ra + t * rb)[:, None] * _unit((1 - t) * ua + t * ub)
    tied_new = _tie_lips(edge_points, rules, edges, node_tie, P, s) if smooth else None
    if not smooth or tied_new is None:
        tied_new = _tie_groups(rules, edges, node_tie, n)

    # -- the template grid over every triangle -------------------------------------------------------------------------
    bary, kind, local, step, sub = _grid(n)
    g = len(bary)
    tt = np.repeat(np.arange(count_t), g)  # (T * g,)
    gg = np.tile(np.arange(g), count_t)
    kk, ll, st = kind[gg], local[gg], step[gg]
    out = np.full(count_t * g, -1, np.int64)
    corner_rec = kk == 0
    out[corner_rec] = tri[tt[corner_rec], ll[corner_rec]]
    edge_rec = np.nonzero(kk == 1)[0]
    e_id = edge_of[tt[edge_rec], ll[edge_rec]]
    fwd = forward[tt[edge_rec], ll[edge_rec]]
    jc = np.where(fwd, st[edge_rec], n - st[edge_rec])
    first_corner = np.where(fwd, ll[edge_rec], (ll[edge_rec] + 1) % 3)
    second_corner = np.where(fwd, (ll[edge_rec] + 1) % 3, ll[edge_rec])
    va, vb = tri[tt[edge_rec], first_corner], tri[tt[edge_rec], second_corner]
    keys = np.stack([e_id, jc, va, vb], axis=1)
    unique_keys, edge_new = np.unique(keys, axis=0, return_inverse=True)
    edge_new = edge_new.ravel()
    out[edge_rec] = count_v + edge_new
    inside_rec = np.nonzero(kk == 2)[0]
    inside_local = np.cumsum(kind == 2) - 1  # the index of each inside grid point among them
    per_tri = int((kind == 2).sum())
    out[inside_rec] = count_v + len(unique_keys) + tt[inside_rec] * per_tri + inside_local[gg[inside_rec]]
    total = count_v + len(unique_keys) + count_t * per_tri

    # -- one record per new vertex: its triangle and barycentric weights -----------------------------------------------
    rep = np.full(total, -1, np.int64)
    rep[out[::-1]] = np.arange(len(out))[::-1]  # the first record of each vertex
    new_ids = np.arange(count_v, total)
    rec = rep[new_ids]
    rt, rb_ = tt[rec], bary[gg[rec]]  # (M, 3)
    rv = tri[rt]  # (M, 3) corner vertices

    positions = np.zeros((total, 3))
    positions[:count_v] = p
    is_edge = kk[rec] == 1
    edge_part = new_ids[is_edge] - count_v
    positions[new_ids[is_edge]] = edge_points[unique_keys[edge_part, 0], unique_keys[edge_part, 1] - 1]
    is_inside = ~is_edge
    if is_inside.any():
        positions[new_ids[is_inside]] = _patch_points(rt[is_inside], rb_[is_inside], P, nt, edge_of, forward, c1, c2,
                                                      sphere, rules, centre_of_node, smooth)

    def mix(values):
        values = np.asarray(values, np.float64)
        return np.einsum("mc,mc...->m...", rb_, values[rv])

    new_tie = np.full(total - count_v, -1, np.int64)
    if tied_new is not None and is_edge.any():
        new_tie[np.nonzero(is_edge)[0]] = tied_new[unique_keys[edge_part, 0], unique_keys[edge_part, 1] - 1]
    indices = out.reshape(count_t, g)[:, sub].reshape(-1, 3)
    parent = np.repeat(np.arange(count_t), len(sub))
    if smooth:  # (a new point split by a seam is several vertices: they move as one)
        same = np.arange(total - count_v) + count_e * n
        same[np.nonzero(is_edge)[0]] = unique_keys[edge_part, 0] * n + unique_keys[edge_part, 1]
        positions[count_v:] = _untangle(positions, mix(p), indices, parent, count_v, new_tie, same)

    uvs = np.zeros((total, 2))
    uvs[:count_v] = prim.uvs
    uvs[count_v:] = mix(prim.uvs)
    normals = np.zeros((total, 3))
    normals[:count_v] = prim.normals
    normals[count_v:] = _unit(mix(prim.normals))
    joints = np.zeros((total, JOINTS), np.asarray(prim.joints).dtype)
    weights = np.zeros((total, JOINTS), np.float32)
    joints[:count_v], weights[:count_v] = prim.joints, prim.weights
    joints[count_v:], weights[count_v:] = _blend_weights(prim.joints, prim.weights, rv, rb_)

    tangents = None
    if prim.tangents is not None:
        tangents = vertex_tangents(positions, normals, uvs, indices)

    # -- the labels, carried -------------------------------------------------------------------------------------------
    def common(values):
        values = np.asarray(values)
        picked = values[rv]
        used = rb_ > 0
        first_value = np.take_along_axis(picked, np.argmax(used, axis=1)[:, None], axis=1)[:, 0]
        same = np.where(used, picked == first_value[:, None], True).all(axis=1)
        return np.concatenate([values, np.where(same, first_value, 0).astype(values.dtype)])

    base = int(tie.max(initial=-1)) + 1
    carried = HeadLabels(labels.primitive, common(parts).astype(np.int8), common(labels.eye).astype(np.int8),
                         common(labels.inner).astype(np.int8),
                         np.concatenate([np.asarray(labels.fade, np.float32),
                                         np.einsum("mc,mc->m", rb_, np.asarray(labels.fade, np.float64)[rv])
                                         .astype(np.float32)]),
                         np.concatenate([tie, np.where(new_tie >= 0, new_tie + base, -1)]),
                         centres.copy(), radii.copy())
    made = SkinnedPrimitive(prim.name, positions.astype(np.float32), normals.astype(np.float32),
                            uvs.astype(np.float32), indices.astype(np.uint32), joints, weights, prim.texture,
                            tangents, prim.maps, prim.detail)
    return Subdivided(made, carried, int(level), bool(smooth), parent, edges, rules,
                      time.perf_counter() - start)


def _untangle(positions, flat, indices, parent, count_v, tie, same=None) -> np.ndarray:
    """The new vertices' positions (``positions[count_v:]``) with the curved patches that would fold flattened: where
    a new triangle turns over against its flat version or its area changes beyond :data:`UNTANGLE_AREA`, the offsets
    from the flat surface (``flat``: the new vertices' places on the coarse triangles) of all new vertices of its
    coarse triangle (``parent``) are halved, round by round (at most :data:`UNTANGLE_ROUNDS`), and alike those of the
    vertices at the same point (``same``: an id per point, for the copies a UV seam makes) and of tied points (``tie``).
    Measured on the avatars: a few slivers in the eye sockets (1 by 20 mm), where the two long edges' curves twist."""
    offsets = positions[count_v:] - flat
    every = np.concatenate([positions[:count_v], flat])
    a = np.cross(every[indices[:, 1]] - every[indices[:, 0]], every[indices[:, 2]] - every[indices[:, 0]])
    area = np.linalg.norm(a, axis=1)
    usable = area > 1e-14
    low, high = UNTANGLE_AREA
    for _ in range(UNTANGLE_ROUNDS):
        moved = np.concatenate([positions[:count_v], flat + offsets])
        b = np.cross(moved[indices[:, 1]] - moved[indices[:, 0]], moved[indices[:, 2]] - moved[indices[:, 0]])
        ratio = np.linalg.norm(b, axis=1) / np.maximum(area, 1e-30)
        bad = usable & ((np.einsum("ij,ij->i", a, b) <= 0) | (ratio < low) | (ratio > high))
        if not bad.any():
            break
        corners = indices[np.isin(parent, np.unique(parent[bad]))].ravel()
        mine = np.zeros(len(offsets), bool)
        mine[corners[corners >= count_v] - count_v] = True
        groups = np.unique(tie[mine & (tie >= 0)])
        if len(groups):
            mine |= np.isin(tie, groups)
        if same is not None:
            mine |= np.isin(same, np.unique(same[mine]))
        offsets[mine] *= 0.5
    return flat + offsets


UNTANGLE_AREA = (0.2, 5.0)  # a new triangle's area may change this much against its flat version
UNTANGLE_ROUNDS = 12


def _spheres(points, labels) -> tuple[np.ndarray, np.ndarray]:
    """The eyeballs' spheres fitted to their vertices as they are now (a fitted head moves them)."""
    from .headshape import mesh

    centres, radii = np.array(labels.eye_centre, np.float64).reshape(2, 3), np.array(labels.eye_radius,
                                                                                    np.float64).reshape(2)
    for e in (1, 2):
        mine = np.asarray(labels.eye) == e
        if mine.sum() >= 4:
            centres[e - 1], radii[e - 1] = mesh.sphere_fit(points[mine])
    return centres, radii


def _lid_margins(P, nt, face_n, all_skin, node_eye, t1, t2, labels, points) -> np.ndarray:
    """Per edge: a lid's margin, where skin facing the eyeball meets skin facing out, both within :data:`LID_REACH`
    of the eyeball and over its cap (:data:`LID_CONE` beyond its edge)."""
    centres, radii = _spheres(points, labels)
    centroid = P[nt].mean(axis=1)
    margin = np.zeros(len(t1), bool)
    for e in (1, 2):
        mine = np.nonzero(node_eye == e)[0]
        if len(mine) < 4:
            continue
        c, r = centres[e - 1], radii[e - 1]
        directions = _unit(P[mine] - c)
        axis = _unit(directions.mean(axis=0))
        cap = np.degrees(np.arccos(np.clip(directions @ axis, -1.0, 1.0))).max()
        to = centroid - c
        distance = np.linalg.norm(to, axis=1) - r
        cone = np.degrees(np.arccos(np.clip(_unit(to) @ axis, -1.0, 1.0)))
        near = all_skin & (distance < LID_REACH) & (cone < cap + LID_CONE)
        inward = np.einsum("ij,ij->i", face_n, -to) > 0
        margin |= near[t1] & near[t2] & (inward[t1] != inward[t2])
    return margin


def _tie_groups(rules, edges, node_tie, n) -> np.ndarray | None:
    """``(E, n - 1)``: per new point of a creased edge between two tied nodes of different groups, a group shared with
    the point of the other lip's edge between the same groups (-1 for the others); None when there is none."""
    ga, gb = node_tie[edges[:, 0]], node_tie[edges[:, 1]]
    tied = (rules == CREASE) & (ga >= 0) & (gb >= 0) & (ga != gb)
    if not tied.any():
        return None
    out = np.full((len(edges), n - 1), -1, np.int64)
    pairs = np.stack([np.minimum(ga, gb), np.maximum(ga, gb)], axis=1)[tied]
    _, which = np.unique(pairs, axis=0, return_inverse=True)
    which = which.ravel()
    flip = (ga > gb)[tied]
    for j in range(n - 1):
        aligned = np.where(flip, n - 2 - j, j)
        out[np.nonzero(tied)[0], j] = which * (n - 1) + aligned
    return out


def _tie_lips(edge_points, rules, edges, node_tie, P, s) -> np.ndarray | None:
    """Give the closed lips' creases the same offsets from their edges (in place on ``edge_points``): each creased
    edge between two tied nodes of different groups takes the mean offset of the edges between the same groups, so
    the two lips' new points are as far apart as their straight midpoints are (no further). Returns the groups of
    :func:`_tie_groups`."""
    n = len(s) + 1
    groups = _tie_groups(rules, edges, node_tie, n)
    if groups is None:
        return None
    tied = np.nonzero(groups[:, 0] >= 0)[0]
    straight = P[edges[tied, 0], None] * (1 - s)[None, :, None] + P[edges[tied, 1], None] * s[None, :, None]
    offsets = edge_points[tied] - straight  # (K, n - 1, 3), canonical order
    ga, gb = node_tie[edges[tied, 0]], node_tie[edges[tied, 1]]
    flip = ga > gb
    aligned = np.where(flip[:, None, None], offsets[:, ::-1], offsets)
    keys = groups[tied, 0] // (n - 1)
    total = np.zeros((int(keys.max()) + 1, n - 1, 3))
    np.add.at(total, keys, aligned)
    mean = total[keys] / np.bincount(keys)[keys][:, None, None]
    back = np.where(flip[:, None, None], mean[:, ::-1], mean)
    edge_points[tied] = straight + back
    return groups


def _patch_points(rt, rb, P, nt, edge_of, forward, c1, c2, sphere, rules, centre_of_node, smooth) -> np.ndarray:
    """Points inside triangles ``rt`` at barycentric ``rb``: on each triangle's cubic Bezier patch (the PN centre
    control point from its edges'), on the eyeball's sphere for an eyeball's triangle, flat when not ``smooth``."""
    corners = P[nt[rt]]  # (M, 3, 3)
    if not smooth:
        return np.einsum("mc,mcd->md", rb, corners)
    near = np.zeros((len(rt), 3, 2, 3))  # per local edge: the control points near its first and second corner
    for k in range(3):
        e = edge_of[rt, k]
        f = forward[rt, k][:, None]
        near[:, k, 0] = np.where(f, c1[e], c2[e])
        near[:, k, 1] = np.where(f, c2[e], c1[e])
    b0, b1, b2 = corners[:, 0], corners[:, 1], corners[:, 2]
    b210, b120 = near[:, 0, 0], near[:, 0, 1]
    b021, b012 = near[:, 1, 0], near[:, 1, 1]
    b102, b201 = near[:, 2, 0], near[:, 2, 1]
    e_mean = (b210 + b120 + b021 + b012 + b102 + b201) / 6.0
    v_mean = (b0 + b1 + b2) / 3.0
    b111 = e_mean + (e_mean - v_mean) / 2.0
    u, v, w = rb[:, 0:1], rb[:, 1:2], rb[:, 2:3]
    out = (b0 * u ** 3 + b1 * v ** 3 + b2 * w ** 3 + 3 * b210 * u * u * v + 3 * b120 * u * v * v
           + 3 * b021 * v * v * w + 3 * b012 * v * w * w + 3 * b102 * u * w * w + 3 * b201 * u * u * w
           + 6 * b111 * u * v * w)
    eyes = sphere[edge_of[rt]].all(axis=1)
    if eyes.any():
        c = centre_of_node[nt[rt[eyes], 0]]
        rel = corners[eyes] - c[:, None]
        r = np.linalg.norm(rel, axis=2)
        direction = np.einsum("mc,mcd->md", rb[eyes], rel / np.maximum(r, 1e-30)[:, :, None])
        out[eyes] = c + np.einsum("mc,mc->m", rb[eyes], r)[:, None] * _unit(direction)
    return out


def _blend_weights(joints, weights, corners, bary) -> tuple[np.ndarray, np.ndarray]:
    """The skin weights of new vertices: their corners' by the barycentric weights ``bary``, the :data:`JOINTS`
    strongest kept and renormalised to 1."""
    joints = np.asarray(joints, np.int64)
    weights = np.asarray(weights, np.float64)
    bones = int(joints.max(initial=0)) + 1
    m = len(corners)
    dense = np.zeros((m, bones))
    rows = np.repeat(np.arange(m), 3 * joints.shape[1])
    cols = joints[corners].reshape(m, -1)
    vals = (bary[:, :, None] * weights[corners]).reshape(m, -1)
    np.add.at(dense, (rows, cols.ravel()), vals.ravel())
    keep = np.argsort(-dense, axis=1, kind="stable")[:, :JOINTS]
    kept = np.take_along_axis(dense, keep, axis=1)
    kept = np.maximum(kept, 0.0)
    total = kept.sum(axis=1, keepdims=True)
    kept = np.where(total > 0, kept / np.maximum(total, 1e-30), np.eye(JOINTS)[0])
    keep = np.where(kept > 0, keep, 0)
    return keep.astype(np.asarray(joints).dtype), kept.astype(np.float32)


# -- models, cached ----------------------------------------------------------------------------------------------------

_primitives: OrderedDict[str, Subdivided] = OrderedDict()  # geometry key -> the subdivision (bounded)
_models: dict[tuple, tuple] = {}  # (id(model), level, smooth) -> (weakref to the model, the subdivided model)
_lock = threading.Lock()


def _key(prim, labels, level: int, smooth: bool) -> str:
    h = hashlib.blake2b(f"subdivide-{VERSION}-{level}-{smooth}".encode(), digest_size=16)
    for array in (prim.positions, prim.normals, prim.uvs, prim.indices, prim.joints, prim.weights, labels.parts,
                  labels.eye, labels.inner, labels.tie, labels.fade):
        array = np.asarray(array)
        h.update(f"{array.dtype}{array.shape}".encode())
        h.update(np.ascontiguousarray(array).tobytes())
    h.update(f"{prim.tangents is not None}".encode())
    return h.hexdigest()


def subdivision(model: CharacterModel, level: int = 1, labels: HeadLabels | None = None,
                smooth: bool = True) -> Subdivided:
    """The :class:`Subdivided` head of ``model`` (``labels``: its :class:`HeadLabels`, :func:`head_labels` by
    default), cached in memory by the head's geometry, the labels and the level (at most :data:`CACHED`)."""
    if labels is None:
        labels = head_labels(model)
    prim = model.primitives[labels.primitive]
    key = _key(prim, labels, level, smooth)
    with _lock:
        found = _primitives.get(key)
        if found is not None:
            _primitives.move_to_end(key)
            return found
    made = subdivide_primitive(prim, labels, level, smooth)
    with _lock:
        _primitives[key] = made
        while len(_primitives) > CACHED:
            _primitives.popitem(last=False)
    return made


def subdivide(model: CharacterModel, level: int = 1, labels: HeadLabels | None = None,
              smooth: bool = True) -> CharacterModel:
    """A copy of ``model`` whose head is subdivided (:func:`subdivision`): it shares the skeleton, the inverse binds,
    the other primitives and the face rig (:func:`~.faces.share_rig`) of ``model``, so it poses as ``model`` does;
    the same object for the same model and level while the model lives."""
    from .faces import share_rig

    key = (id(model), int(level), bool(smooth))
    with _lock:
        known = _models.get(key)
        if known is not None and known[0]() is model:
            return known[1]
    done = subdivision(model, level, labels, smooth)
    primitives = list(model.primitives)
    primitives[done.labels.primitive] = done.primitive
    metadata = dict(model.metadata)
    metadata["subdivided"] = {"level": int(level), "smooth": bool(smooth), "of": model.name}
    made = CharacterModel(model.name, model.skeleton, primitives, model.inverse_bind, metadata)
    share_rig(made, model)
    with _lock:
        _models[key] = (weakref.ref(model), made)
        weakref.finalize(model, _models.pop, key, None)
    return made


def check(model: CharacterModel, level: int = 1, labels: HeadLabels | None = None, size: int | None = None):
    """The rig's checks (:func:`~.headshape.rig.rig_checks`) of ``model``'s subdivided head in the 12 faces of the
    face rig: the head subdivided on its curved surface against the same head split along its flat triangles (the
    coarse head's own surface), so that no eyeball shows through a lid, no tooth through the lips, no mouth opens that
    was shut, the neck's edge stays where it is and no triangle folds, as far as the coarse head does."""
    from .headshape import rig

    if labels is None:
        labels = head_labels(model)
    flat = subdivide(model, level, labels, smooth=False)
    curved = subdivide(model, level, labels)
    carried = subdivision(model, level, labels).labels
    return rig.rig_checks(flat, curved, carried, size or rig.CHECK_SIZE)
