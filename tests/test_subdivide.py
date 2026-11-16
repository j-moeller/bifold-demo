"""Subdivided heads (step S4), on small meshes and the synthetic avatar of the head-shape tests (no avatars, no
OpenGL): the original vertices kept, new points on the curved surface, the open edge on its edges, creases, closed
lips, eyeballs on their spheres, the mouth's inside left as it is, the skin weights, UVs, tangents, the labels carried,
the caches and the checks of the rig."""

import math

import numpy as np
import pytest

from bifold_demo.characters import faces
from bifold_demo.characters import subdivide as sd
from bifold_demo.characters.headshape import head as s0
from bifold_demo.characters.headshape import mesh
from bifold_demo.characters.model import SkinnedPrimitive
from bifold_demo.characters.tangents import vertex_tangents

RADIUS = 0.1


def _primitive(points, uvs, tri, normals=None, joints=None, weights=None) -> SkinnedPrimitive:
    points = np.asarray(points, np.float64)
    tri = np.asarray(tri, np.int64)
    if normals is None:
        normals = mesh.vertex_normals(points, tri, mesh.weld(points)[0])
    if joints is None:
        joints = np.zeros((len(points), 4), np.int64)
        weights = np.tile([1.0, 0.0, 0.0, 0.0], (len(points), 1))
    return SkinnedPrimitive("t_head", points.astype(np.float32), np.asarray(normals, np.float32),
                            np.asarray(uvs, np.float32), tri.astype(np.uint32), np.asarray(joints),
                            np.asarray(weights, np.float32), np.full((2, 2, 4), 128, np.uint8),
                            tangents=vertex_tangents(points, normals, uvs, tri))


def _labels(count, parts=None, fade=None, tie=None, eye=None) -> sd.HeadLabels:
    parts = np.full(count, s0.SKIN, np.int8) if parts is None else np.asarray(parts, np.int8)
    return sd.HeadLabels(0, parts, np.zeros(count, np.int8) if eye is None else np.asarray(eye, np.int8),
                         np.zeros(count, np.int8), np.ones(count, np.float32) if fade is None else
                         np.asarray(fade, np.float32), np.full(count, -1, np.int64) if tie is None else
                         np.asarray(tie, np.int64))


def _band(rows=6, cols=12, theta=(0.3, 1.4)):
    """A band of a sphere (two open edges) with a UV seam where the azimuth wraps round; the bottom ring is the
    'neck' (fade 0)."""
    v, u = np.meshgrid(np.linspace(0.0, 1.0, rows + 1), np.linspace(0.0, 1.0, cols + 1), indexing="ij")
    t, f = theta[0] + v * (theta[1] - theta[0]), u * 2 * math.pi
    d = np.stack([np.sin(t) * np.sin(f), np.cos(t), np.sin(t) * np.cos(f)], axis=-1).reshape(-1, 3)
    tri = []
    for r in range(rows):
        for c in range(cols):
            a, b = r * (cols + 1) + c, r * (cols + 1) + c + 1
            tri += [(a, a + cols + 1, b), (b, a + cols + 1, b + cols + 1)]
    fade = np.where(v.ravel() >= 1.0, 0.0, 1.0)
    return d * RADIUS, d, np.c_[u.ravel(), 1.0 - v.ravel()], np.array(tri), fade


def _open_edges(prim) -> int:
    node = mesh.weld(prim.positions)[0]
    return len(mesh.open_edges(node[np.asarray(prim.indices, np.int64)]))


def _segment_distance(p, a, b) -> np.ndarray:
    ab = b - a
    t = np.clip(np.einsum("ij,ij->i", p - a, ab) / np.einsum("ij,ij->i", ab, ab), 0.0, 1.0)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=1)


@pytest.fixture(scope="module")
def band():
    points, normals, uvs, tri, fade = _band()
    prim = _primitive(points, uvs, tri, normals)
    return prim, _labels(len(points), fade=fade)


@pytest.mark.parametrize("level", [1, 2])
def test_the_original_vertices_stay_and_the_new_points_lie_on_the_curved_surface(band, level):
    prim, labels = band
    made = sd.subdivide_primitive(prim, labels, level)
    flat = sd.subdivide_primitive(prim, labels, level, smooth=False)
    q, n = made.primitive, len(prim.positions)
    for name in ("positions", "normals", "uvs", "joints", "weights"):
        assert np.array_equal(getattr(q, name)[:n], getattr(prim, name)), name  # bit for bit
    assert len(q.indices) == 4 ** level * len(prim.indices)
    assert np.array_equal(made.primitive.indices, flat.primitive.indices)
    curved = np.abs(np.linalg.norm(q.positions[n:].astype(np.float64), axis=1) - RADIUS)
    straight = np.abs(np.linalg.norm(flat.primitive.positions[n:].astype(np.float64), axis=1) - RADIUS)
    assert np.median(curved) < 0.1 * np.median(straight)  # (measured: 0.03-0.05)
    assert _open_edges(q) == 2 ** level * _open_edges(prim)  # watertight: only the old open edges, split


def test_the_open_edges_stay_on_their_edges(band):
    """The neck's edge (and any open edge) is split on its straight edges: the seam to the body cannot open."""
    prim, labels = band
    made = sd.subdivide_primitive(prim, labels, 2)
    q = made.primitive
    node = mesh.weld(q.positions)[0]
    edges = mesh.open_edges(node[np.asarray(q.indices, np.int64)])
    coarse_node = mesh.weld(prim.positions)[0]
    coarse = mesh.open_edges(coarse_node[np.asarray(prim.indices, np.int64)])
    where = np.zeros((coarse_node.max() + 1, 3))
    where[coarse_node] = prim.positions
    a, b = where[coarse[:, 0]], where[coarse[:, 1]]
    points = np.zeros((node.max() + 1, 3))
    points[node] = q.positions
    on_edge = points[np.unique(edges)].astype(np.float64)
    nearest = np.min([_segment_distance(on_edge, np.repeat(a[k:k + 1], len(on_edge), 0),
                                        np.repeat(b[k:k + 1], len(on_edge), 0)) for k in range(len(a))], axis=0)
    assert nearest.max() < 1e-7
    neck = made.labels.fade <= 0
    assert neck.sum() == 13 + 12 * 3  # the ring's 13 vertices (one split by the seam) and 3 new ones per edge


def _roof(alpha=30.0, cells=6):
    """Two planes meeting at a straight ridge (along x) at an angle of ``2 alpha``: the fold between their normals is
    ``180 - 2 alpha`` degrees. Each plane has its own vertices; the ridge's are at the same places."""
    a = math.radians(alpha)
    points, uvs, tri = [], [], []
    for side in (1.0, -1.0):
        down = np.array([0.0, -math.cos(a), side * math.sin(a)])
        start = len(points)
        for i in range(cells + 1):
            for j in range(cells + 1):
                x = (i / cells - 0.5) * 0.06
                points.append(np.array([x, 0.0, 0.0]) + down * (j / cells) * 0.03)
                uvs.append((i / cells, j / cells * 0.5 + (0.5 if side < 0 else 0.0)))
        for i in range(cells):
            for j in range(cells):
                p0 = start + i * (cells + 1) + j
                quad = (p0, p0 + cells + 1, p0 + 1, p0 + cells + 2)
                if side > 0:
                    tri += [(quad[0], quad[2], quad[1]), (quad[1], quad[2], quad[3])]
                else:
                    tri += [(quad[0], quad[1], quad[2]), (quad[1], quad[3], quad[2])]
    return np.array(points), np.array(uvs), np.array(tri)


@pytest.mark.parametrize("level", [1, 2])
def test_a_crease_keeps_its_fold(level):
    """A fold sharper than CREASE_FOLD is a crease: the planes on either side stay planes (the surface is not
    rounded over the ridge) and the ridge's new points lie on the ridge."""
    points, uvs, tri = _roof(30.0)  # a fold of 120 degrees
    prim = _primitive(points, uvs, tri)
    made = sd.subdivide_primitive(prim, _labels(len(points)), level)
    assert made.counts()["crease"] == 6
    q = made.primitive.positions.astype(np.float64)
    n = len(points)
    a = math.radians(30.0)
    normal = np.array([0.0, math.sin(a), math.cos(a)])  # of the first plane (z >= 0)
    other = normal * [1.0, 1.0, -1.0]
    off = np.minimum(np.abs(q[n:] @ normal), np.abs(q[n:] @ other))
    assert off.max() < 1e-7


def _lips(gap=0.0, cells=8, lower_bend=12.0):
    """Two closed lips: an upper and a lower sheet, each folding back into the mouth, touching along a curved lip line
    (their own vertices, tied), ``gap`` apart (the lower lip's line bent by ``lower_bend``, the upper's by 12)."""
    points, uvs, tri, tie = [], [], [], []
    xs = np.linspace(-0.025, 0.025, cells + 1)
    for side in (1.0, -1.0):
        start = len(points)
        rows = [np.array([0.0, side * 0.012, 0.004]), np.array([0.0, side * 0.006, 0.003]), np.zeros(3),
                np.array([0.0, side * 0.004, -0.008]), np.array([0.0, side * 0.006, -0.016])]
        for i, x in enumerate(xs):
            for j, r in enumerate(rows):
                bend = 12.0 if side > 0 else lower_bend
                points.append(np.array([x, -gap * (side < 0), -bend * x * x]) + r)
                uvs.append((i / cells, (j / 4) * 0.5 + (0.5 if side < 0 else 0.0)))
                tie.append(i if j == 2 else -1)
        for i in range(cells):
            for j in range(len(rows) - 1):
                p0 = start + i * len(rows) + j
                quad = (p0, p0 + len(rows), p0 + 1, p0 + len(rows) + 1)
                if side > 0:
                    tri += [(quad[0], quad[1], quad[2]), (quad[2], quad[1], quad[3])]
                else:
                    tri += [(quad[0], quad[2], quad[1]), (quad[2], quad[3], quad[1])]
    return np.array(points), np.array(uvs), np.array(tri), np.array(tie)


def _lip_gaps(subdivided, count) -> np.ndarray:
    """The distances between the two lips' new points on their lip lines (rows of v = 0.25 and 0.75), paired by
    their place along the line."""
    q = subdivided.primitive.positions[count:].astype(np.float64)
    v = subdivided.primitive.uvs[count:, 1]
    upper, lower = q[np.abs(v - 0.25) < 1e-6], q[np.abs(v - 0.75) < 1e-6]
    assert len(upper) == len(lower) > 0
    upper, lower = upper[np.argsort(upper[:, 0])], lower[np.argsort(lower[:, 0])]
    return np.linalg.norm(upper - lower, axis=1)


@pytest.mark.parametrize("gap", [0.0001, 0.0005])
def test_closed_lips_stay_closed(gap):
    """The lip line is a crease on each lip, and the two lips' new points are as far apart as their straight
    midpoints (no further): closed lips stay closed, and new points along it are tied too. Without the crease (the
    ties left out) each lip would curve its own way and they would part."""
    points, uvs, tri, tie = _lips(gap)
    prim = _primitive(points, uvs, tri)
    labels = _labels(len(points), tie=tie)
    for level in (1, 2):
        made = sd.subdivide_primitive(prim, labels, level)
        assert (made.labels.tie[len(points):] >= 0).sum() == 2 * 8 * (2 ** level - 1)
        assert made.counts()["crease"] == 16
        assert np.abs(_lip_gaps(made, len(points)) - gap).max() < 1e-7
        tie_new = made.labels.tie
        q = made.primitive.positions.astype(np.float64)
        spread = max(float(np.ptp(q[tie_new == g], axis=0).max()) for g in np.unique(tie_new[tie_new >= 0]))
        assert spread <= gap + 1e-7
    untied = sd.subdivide_primitive(prim, _labels(len(points)), 1)
    assert np.abs(_lip_gaps(untied, len(points)) - gap).max() > 2e-5  # (what the crease prevents)


def test_lips_of_two_shapes_part_no_more_than_their_edges_do():
    """Where the two lips' lines differ (bent unlike, 0.2 mm apart at their ends), each tied pair of new points takes
    the same offset from its straight edges, so the lips are exactly as far apart as the straight midpoints are."""
    points, uvs, tri, tie = _lips(0.0002, lower_bend=10.0)
    prim = _primitive(points, uvs, tri)
    labels = _labels(len(points), tie=tie)
    for level in (1, 2):
        made = sd.subdivide_primitive(prim, labels, level)
        flat = sd.subdivide_primitive(prim, labels, level, smooth=False)
        assert np.abs(_lip_gaps(made, len(points)) - _lip_gaps(flat, len(points))).max() < 1e-7


def test_the_lid_margins_are_where_the_skin_turns_to_face_the_eyeball():
    """Skin near an eyeball (a 12 mm sphere looking along +z): a band facing out round its front, folding over at the
    eye opening into a band facing the eyeball; the edges between the two bands (along the fold) are the margin, and
    only they."""
    cols = 16
    angles = np.linspace(0.0, 2 * math.pi, cols, endpoint=False)
    rings = []
    for cone, radius in ((50.0, 0.0150), (40.0, 0.0145), (45.0, 0.0125)):  # outside, the margin, folded in
        c = math.radians(cone)
        rings.append(np.stack([radius * math.sin(c) * np.cos(angles), radius * math.sin(c) * np.sin(angles),
                               np.full(cols, radius * math.cos(c))], axis=1))
    eyeball = _band(4, 8, (0.1, 1.2))[0][:, [0, 2, 1]] * 0.12  # (its cap round +z)
    points = np.concatenate([*rings, eyeball])
    tri = []
    for r in range(2):
        for k in range(cols):
            a, b = r * cols + k, r * cols + (k + 1) % cols
            tri += [(a, b, a + cols), (b, b + cols, a + cols)]
    tri = np.array(tri)
    face_n = sd._unit(np.cross(points[tri[:, 1]] - points[tri[:, 0]], points[tri[:, 2]] - points[tri[:, 0]]))
    out = np.einsum("ij,ij->i", face_n, points[tri].mean(axis=1)) > 0
    assert out[:cols * 2].all() and not out[cols * 2:].any()  # (the construction: the first band faces out)
    edges, edge_of = np.unique(np.sort(np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), axis=1),
                               axis=0, return_inverse=True)
    edge_of = edge_of.reshape(3, -1).T
    t1, t2 = np.full(len(edges), -1), np.full(len(edges), -1)
    for t in range(len(tri)):
        for e in edge_of[t]:
            if t1[e] < 0:
                t1[e] = t
            else:
                t2[e] = t
    eye = np.r_[np.zeros(3 * cols, np.int8), np.ones(len(eyeball), np.int8)]
    margin = sd._lid_margins(points, tri, face_n, np.ones(len(tri), bool), eye, t1, np.maximum(t2, 0),
                             _labels(len(points), eye=eye), points) & (t2 >= 0)
    ring = edges // cols  # (0 outside, 1 the margin, 2 folded in)
    assert np.array_equal(margin, (ring[:, 0] == 1) & (ring[:, 1] == 1)) and margin.sum() == cols


def test_eyeballs_stay_on_their_spheres_and_the_mouths_inside_is_split_flat():
    eye_p, _eye_n, eye_uv, eye_tri, _fade = _band(5, 10, (0.2, 2.6))
    eye_p = eye_p * 0.12 + [0.03, 0.0, 0.0]  # an eyeball of 12 mm, open at the back
    in_p, in_uv, in_tri = _roof(70.0, 3)  # the mouth's inside, folded
    in_p = in_p + [0.0, -0.05, 0.0]
    points = np.concatenate([eye_p, in_p])
    tri = np.concatenate([eye_tri, in_tri + len(eye_p)])
    uvs = np.concatenate([eye_uv, in_uv])
    parts = np.r_[np.full(len(eye_p), s0.EYE), np.full(len(in_p), s0.INNER)]
    eye = np.r_[np.ones(len(eye_p)), np.zeros(len(in_p))]
    labels = _labels(len(points), parts=parts, eye=eye)
    prim = _primitive(points, uvs, tri)
    made = sd.subdivide_primitive(prim, labels, 2)
    counts = made.counts()
    assert counts["sphere"] > 0 and counts["straight"] > 0 and counts["curved"] == counts["crease"] == 0
    q = made.primitive.positions.astype(np.float64)
    new_eye = (made.labels.eye == 1) & (np.arange(len(q)) >= len(points))
    centre = np.array([0.03, 0.0, 0.0])
    assert np.abs(np.linalg.norm(q[new_eye] - centre, axis=1) - 0.012).max() < 1e-6
    flat = sd.subdivide_primitive(prim, labels, 2, smooth=False).primitive.positions
    inside = made.labels.parts == s0.INNER
    assert np.abs(q[inside] - flat[inside]).max() < 1e-7


def test_the_skin_weights_are_the_four_strongest_renormalised(band):
    prim, labels = band
    n = len(prim.positions)
    joints = np.zeros((n, 4), np.int64)
    weights = np.zeros((n, 4), np.float32)
    for v in range(n):  # every vertex on four bones of its own, equally
        joints[v] = 4 * (v % 7) + np.arange(4)
        weights[v] = 0.25
    joints[0], weights[0] = (30, 0, 0, 0), (1.0, 0, 0, 0)
    joints[1], weights[1] = (31, 0, 0, 0), (1.0, 0, 0, 0)
    weighted = _primitive(prim.positions, prim.uvs, prim.indices, prim.normals, joints, weights)
    made = sd.subdivide_primitive(weighted, labels, 2).primitive
    total = made.weights.sum(axis=1)
    assert np.abs(total - 1.0).max() < 1e-6  # (a new vertex inside a triangle draws on 12 bones: 4 kept)
    assert (made.weights >= 0).all() and made.joints.shape[1] == 4
    # the middle of the edge between the vertices on bones 30 and 31: half each
    between = np.nonzero(np.all(np.abs(made.positions - (prim.positions[0] + prim.positions[1]) / 2) < 1e-3,
                                axis=1))[0]
    assert len(between)
    k = between[np.argmin(np.linalg.norm(made.uvs[between] - (prim.uvs[0] + prim.uvs[1]) / 2, axis=1))]
    pairs = {int(j): float(w) for j, w in zip(made.joints[k], made.weights[k]) if w > 0}
    assert pairs.keys() == {30, 31} and abs(pairs[30] - 0.5) < 1e-6


def test_every_new_vertex_follows_its_stencil(band):
    """A new vertex's UV, normal and skin weights are its coarse triangle's corners' by its barycentric weights (the
    weights' four strongest, renormalised); on an open edge its place is too."""
    prim, labels = band
    n = len(prim.positions)
    joints = np.stack([np.arange(n) % 5, (np.arange(n) + 1) % 5, np.zeros(n), np.zeros(n)], axis=1).astype(np.int64)
    weights = np.tile([0.75, 0.25, 0.0, 0.0], (n, 1)).astype(np.float32)
    weighted = _primitive(prim.positions, prim.uvs, prim.indices, prim.normals, joints, weights)
    made = sd.subdivide_primitive(weighted, labels, 2)
    q = made.primitive
    corners, bary = made.corners, made.bary
    assert corners.shape == bary.shape == (len(q.positions) - n, 3) and np.allclose(bary.sum(axis=1), 1.0)
    assert np.allclose(q.uvs[n:], np.einsum("mc,mcd->md", bary, prim.uvs[corners].astype(np.float64)), atol=1e-6)
    mixed = np.einsum("mc,mcd->md", bary, prim.normals[corners].astype(np.float64))
    assert np.allclose(q.normals[n:], mixed / np.linalg.norm(mixed, axis=1, keepdims=True), atol=1e-5)
    dense = np.zeros((len(corners), 5))
    for c in range(3):
        for k in range(4):
            share = bary[:, c] * weights[corners[:, c], k]
            np.add.at(dense, (np.arange(len(corners)), joints[corners[:, c], k]), share)
    got = np.zeros_like(dense)
    np.add.at(got, (np.repeat(np.arange(len(corners)), 4), q.joints[n:].ravel()), q.weights[n:].ravel())
    top = np.sort(dense, axis=1)[:, ::-1][:, :4]
    assert np.allclose(np.sort(got, axis=1)[:, ::-1][:, :4], top / top.sum(axis=1, keepdims=True), atol=1e-5)
    on_open = made.edge >= 0
    open_edge = np.isin(made.edge, np.flatnonzero(made.rules == sd.LINEAR))
    assert on_open.sum() > 0 and open_edge.any()
    flat = np.einsum("mc,mcd->md", bary, prim.positions[corners].astype(np.float64))
    assert np.abs(q.positions[n:][open_edge] - flat[open_edge]).max() < 1e-6


def test_uvs_are_per_triangle_and_a_seam_splits_new_vertices(band):
    prim, labels = band
    made = sd.subdivide_primitive(prim, labels, 1)
    q, n = made.primitive, len(prim.positions)
    seam = (np.abs(q.uvs[n:, 0] - 0.0) < 1e-9) | (np.abs(q.uvs[n:, 0] - 1.0) < 1e-9)
    seam_points = q.positions[n:][seam]
    _, counts = np.unique(np.round(seam_points / 1e-6).astype(np.int64), axis=0, return_counts=True)
    assert len(counts) and (counts == 2).all()  # each new point on the seam twice, once per side
    tri = np.asarray(q.indices, np.int64)
    parent = np.asarray(prim.indices, np.int64)[made.parent]
    for t in range(0, len(tri), 37):  # a new triangle's UVs lie in its coarse triangle's (barycentric, same weights)
        system = np.r_[prim.uvs[parent[t]].astype(np.float64).T, np.ones((1, 3))]
        for v in tri[t]:
            assert np.linalg.solve(system, np.r_[q.uvs[v], 1.0]).min() > -1e-6


def test_the_tangents_are_made_again(band):
    prim, labels = band
    q = sd.subdivide_primitive(prim, labels, 2).primitive
    again = vertex_tangents(q.positions, q.normals, q.uvs, q.indices)
    assert q.tangents.shape == (len(q.positions), 4) and np.allclose(q.tangents, again, atol=1e-6)
    assert np.abs(np.einsum("ij,ij->i", q.tangents[:, :3], q.normals)).max() < 1e-4


def test_the_labels_are_carried():
    """A new vertex between two parts belongs to neither, so the new triangles of a coarse triangle that mixes parts
    count as it did (neither skin nor inside); the fade is interpolated."""
    points, normals, uvs, tri, _fade = _band()
    parts = np.where(points[:, 0] > 0.02, s0.INNER, s0.SKIN).astype(np.int8)
    fade = np.clip((points[:, 1] + 0.05) / 0.15, 0.0, 1.0)  # (linear where it is not clipped)
    labels = _labels(len(points), parts=parts, fade=fade)
    labels.inner = np.where(parts == s0.INNER, 2, 0).astype(np.int8)
    prim = _primitive(points, uvs, tri, normals)
    made = sd.subdivide_primitive(prim, labels, 1)
    new = made.labels
    coarse_tri = np.asarray(prim.indices, np.int64)
    for kind in (s0.SKIN, s0.INNER):
        coarse_all = (labels.parts[coarse_tri] == kind).all(axis=1)
        fine_all = (new.parts[np.asarray(made.primitive.indices, np.int64)] == kind).all(axis=1)
        assert np.array_equal(fine_all, coarse_all[made.parent])
    assert set(np.unique(new.parts)) <= {0, s0.SKIN, s0.INNER} and (new.parts == 0).any()
    assert set(np.unique(new.inner[new.parts == s0.INNER])) == {2} and (new.inner[new.parts == 0] == 0).all()
    flat = sd.subdivide_primitive(prim, labels, 1, smooth=False).primitive.positions.astype(np.float64)
    inside = (flat[:, 1] + 0.05) / 0.15
    linear = (inside > 0.0) & (inside < 1.0)  # (where the fade is the linear function of the place, so is it after)
    assert np.allclose(new.fade[linear], inside[linear], atol=1e-5)


def test_a_patch_that_would_fold_is_flattened():
    """Where curved edges would turn a new triangle over against the flat one, its coarse triangle's new points
    move back towards the flat surface (halved, round by round) until none is."""
    flat = np.array([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.5, 0.5, 0.0]])
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-0.1, 0.4, 0.0], [0.0, 0.5, 0.0],
                          [0.5, 0.5, 0.0]])  # new point 3 pulled past point 4: triangle (0, 3, 4) turned over
    indices = np.array([[0, 3, 4], [3, 1, 5], [4, 5, 2], [3, 5, 4]])
    out = sd._untangle(positions.copy(), flat, indices, np.zeros(4, np.int64), 3, np.full(3, -1))
    moved = np.concatenate([positions[:3], out])
    a = np.cross(np.r_[positions[:3], flat][indices[:, 1]] - np.r_[positions[:3], flat][indices[:, 0]],
                 np.r_[positions[:3], flat][indices[:, 2]] - np.r_[positions[:3], flat][indices[:, 0]])
    b = np.cross(moved[indices[:, 1]] - moved[indices[:, 0]], moved[indices[:, 2]] - moved[indices[:, 0]])
    assert (np.einsum("ij,ij->i", a, b) > 0).all()
    assert np.linalg.norm(out - flat) < np.linalg.norm(positions[3:] - flat)


# -- on a model --------------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def avatar():
    from test_headshape import synthetic_avatar

    return synthetic_avatar()


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("BIFOLD_HEADTEX_CACHE", str(tmp_path))
    monkeypatch.setattr(sd, "_parts_memory", type(sd._parts_memory)())
    return tmp_path


def test_head_labels_of_a_model(avatar, cache):
    labels = sd.head_labels(avatar)
    head = avatar.primitives[labels.primitive]
    skin = avatar.metadata["skin vertices"]
    assert labels.primitive == 1 and len(labels.parts) == len(head.positions)
    assert (labels.eye[skin:] > 0).all() and (labels.eye[:skin] == 0).all()
    assert (labels.parts[labels.eye > 0] == s0.EYE).all()
    assert np.allclose(labels.eye_radius, 0.012, atol=1e-3)
    assert labels.fade.min() == 0.0 and labels.fade.max() == 1.0


def test_the_parts_are_cached_for_an_avatars_head_only(avatar, cache, monkeypatch):
    first = sd.head_labels(avatar)
    files = list(cache.glob("subdivide-parts-*.npz"))
    assert len(files) == 1
    monkeypatch.setattr(sd, "_parts_memory", type(sd._parts_memory)())
    monkeypatch.setattr(s0, "head_parts", lambda *a: (_ for _ in ()).throw(AssertionError("made again")))
    assert np.array_equal(sd.head_labels(avatar).parts, first.parts)  # read from the file
    from bifold_demo.characters.model import CharacterModel

    fitted = CharacterModel(avatar.name, avatar.skeleton, avatar.primitives, avatar.inverse_bind,
                            {**avatar.metadata, "head_shape": "a person's"})
    head = avatar.primitives[1]
    moved = SkinnedPrimitive(head.name, head.positions + np.float32(0.001), head.normals, head.uvs, head.indices,
                             head.joints, head.weights, head.texture, head.tangents)
    fitted.primitives = [avatar.primitives[0], moved, avatar.primitives[2]]
    monkeypatch.setattr(s0, "head_parts", lambda model, primitive: np.full(len(moved.positions), s0.SKIN, np.int8))
    sd.head_labels(fitted)
    assert len(list(cache.glob("subdivide-parts-*.npz"))) == 1  # a fitted head (biometric): in memory only


def test_the_subdivided_model_shares_the_rest_and_is_cached(avatar, cache):
    made = sd.subdivide(avatar, 1)
    assert made is sd.subdivide(avatar, 1) and made is not sd.subdivide(avatar, 2)
    assert made.skeleton is avatar.skeleton and made.inverse_bind is avatar.inverse_bind
    assert made.primitives[0] is avatar.primitives[0] and made.primitives[2] is avatar.primitives[2]
    head, sub = avatar.primitives[1], made.primitives[1]
    assert len(sub.indices) == 4 * len(head.indices) and sub.name == head.name and sub.texture is head.texture
    assert made.metadata["subdivided"]["level"] == 1
    assert faces.face_rig(made) is faces.face_rig(avatar)
    from bifold_demo.characters.model import CharacterModel

    copy = CharacterModel(avatar.name, avatar.skeleton, list(avatar.primitives), avatar.inverse_bind,
                          dict(avatar.metadata))
    assert sd.subdivision(copy, 1) is sd.subdivision(avatar, 1)  # cached by the head's geometry


def test_two_models_of_one_head_wear_their_own_textures(avatar, cache):
    """The subdivided geometry is cached by itself: another model with the same head but other textures (a person's
    own) shares it and keeps its own textures and maps."""
    import dataclasses

    from bifold_demo.characters.model import CharacterModel

    head = avatar.primitives[1]
    other = dataclasses.replace(head, texture=np.full((4, 4, 4), 90, np.uint8), maps=np.zeros((4, 4, 4), np.uint8))
    copy = CharacterModel(avatar.name, avatar.skeleton, [avatar.primitives[0], other, avatar.primitives[2]],
                          avatar.inverse_bind, dict(avatar.metadata))
    a, b = sd.subdivide(avatar, 1).primitives[1], sd.subdivide(copy, 1).primitives[1]
    assert a.texture is head.texture and a.maps is head.maps
    assert b.texture is other.texture and b.maps is other.maps
    assert b.positions is a.positions and b.indices is a.indices  # (one geometry)


def test_a_head_that_cannot_be_subdivided_is_drawn_as_it_is_and_not_tried_again(avatar, monkeypatch, capsys):
    """Whatever goes wrong making the finer head, the detailed model is None (the picture draws the coarse head), it
    is said once, and the next frames do not try again; a caller that waits gets the background's result."""
    import threading

    from bifold_demo.characters.model import CharacterModel

    copy = CharacterModel(avatar.name, avatar.skeleton, list(avatar.primitives), avatar.inverse_bind, {})
    calls = []

    def broken(model, level):
        calls.append(level)
        raise IndexError("a part the subdivision did not expect")

    monkeypatch.setattr(sd, "subdivide", broken)
    assert sd.detail_model(copy, 1) is None and sd.detail_model(copy, 1) is None
    assert sd.detail_model(copy, 1, wait=False) is None and calls == [1]
    assert "cannot subdivide the head of" in capsys.readouterr().out
    other = CharacterModel(avatar.name, avatar.skeleton, list(avatar.primitives), avatar.inverse_bind, {})
    gate = threading.Event()
    monkeypatch.setattr(sd, "subdivide", lambda model, level: (gate.wait(10), ("finer", level))[1])
    assert sd.detail_model(other, 2, wait=False) is None  # on its way
    threading.Timer(0.2, gate.set).start()
    assert sd.detail_model(other, 2) == ("finer", 2)  # (waited for it)


def test_the_face_rig_is_shared(monkeypatch, avatar):
    marker = object()
    monkeypatch.setattr(faces, "face_rig", lambda model: marker)
    from bifold_demo.characters.model import CharacterModel

    copy = CharacterModel(avatar.name, avatar.skeleton, list(avatar.primitives), avatar.inverse_bind, {})
    faces.share_rig(copy, avatar)
    monkeypatch.undo()
    assert faces.face_rig(copy) is marker


@pytest.mark.parametrize("level", [1, 2])
def test_the_subdivided_head_passes_the_rigs_checks(avatar, cache, level):
    check = sd.check(avatar, level, size=128)
    assert check.ok, check.problems
    curved = sd.subdivide(avatar, level).primitives[1].positions
    flat = sd.subdivide(avatar, level, smooth=False).primitives[1].positions
    neck = sd.subdivision(avatar, level).labels.fade <= 0
    assert neck.any() and np.abs(curved[neck] - flat[neck]).max() == 0.0  # the neck's edge does not move
