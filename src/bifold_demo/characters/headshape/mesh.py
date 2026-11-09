"""Mesh tools for the shape fit, in numpy: welding, open edges, graph Laplacians, nearest points on a large mesh,
similarity transforms, thin-plate splines in 3D and vertex normals.

The Rocketbox heads have about 1,800 vertices, ICT-FaceKit's face and head 11,248, so dense linear algebra on the
heads and a candidate search on ICT's mesh are enough; nothing here needs scipy.
"""

from __future__ import annotations

import numpy as np

WELD = 1e-5  # metres: vertices this close are one point of the surface (split only by UV seams or hard edges)


def weld(points, tolerance: float = WELD) -> tuple[np.ndarray, int]:
    """``(node per vertex, nodes)``: vertices at the same position (to ``tolerance``) share one node."""
    keys = np.round(np.asarray(points, np.float64) / tolerance).astype(np.int64)
    _, node = np.unique(keys, axis=0, return_inverse=True)
    node = node.ravel()
    return node, int(node.max()) + 1 if len(node) else 0


def edges(triangles) -> np.ndarray:
    """``(E, 2)`` sorted unique undirected edges of ``triangles`` (``(T, 3)``)."""
    t = np.asarray(triangles, np.int64)
    e = np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
    return np.unique(np.sort(e, axis=1), axis=0)


def open_edges(triangles) -> np.ndarray:
    """``(E, 2)``: the edges only one triangle uses (the mesh's open boundary)."""
    t = np.asarray(triangles, np.int64)
    e = np.sort(np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), axis=1)
    unique, count = np.unique(e, axis=0, return_counts=True)
    return unique[count == 1]


def components(triangles, nodes: int) -> np.ndarray:
    """The connected component of each node (``(nodes,)`` int; nodes in no triangle are their own)."""
    parent = np.arange(nodes)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in edges(triangles):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return np.array([find(i) for i in range(nodes)])


def laplacian(triangles, nodes: int, weights=None) -> np.ndarray:
    """The dense ``(nodes, nodes)`` graph Laplacian ``I - D^-1 A`` of the mesh (uniform weights; ``weights`` per
    edge of :func:`edges` to weight them): row ``i`` takes the mean of a node's neighbours from it. Nodes without
    neighbours get a zero row."""
    e = edges(triangles)
    w = np.ones(len(e)) if weights is None else np.asarray(weights, np.float64)
    a = np.zeros((nodes, nodes))
    np.add.at(a, (e[:, 0], e[:, 1]), w)
    np.add.at(a, (e[:, 1], e[:, 0]), w)
    degree = a.sum(axis=1)
    lap = -a / np.maximum(degree, 1e-12)[:, None]
    lap[np.arange(nodes), np.arange(nodes)] = np.where(degree > 0, 1.0, 0.0)
    return lap


def vertex_normals(points, triangles, node=None) -> np.ndarray:
    """``(N, 3)`` unit area-weighted vertex normals; with ``node`` (from :func:`weld`) vertices of one node share
    one (smooth across UV seams)."""
    p = np.asarray(points, np.float64)
    t = np.asarray(triangles, np.int64)
    face = np.cross(p[t[:, 1]] - p[t[:, 0]], p[t[:, 2]] - p[t[:, 0]])
    key = np.arange(len(p)) if node is None else np.asarray(node)
    sums = np.zeros((int(key.max()) + 1 if len(key) else 0, 3))
    for k in range(3):
        np.add.at(sums, key[t[:, k]], face)
    n = sums[key]
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-18)


def umeyama(source, target, weights=None, scale: bool = True) -> tuple[float, np.ndarray, np.ndarray]:
    """``(s, R, t)`` with ``target ~ s R source + t`` (weighted least squares; Umeyama 1991)."""
    src = np.asarray(source, np.float64)
    dst = np.asarray(target, np.float64)
    w = np.ones(len(src)) if weights is None else np.asarray(weights, np.float64)
    w = w / w.sum()
    ms, md = w @ src, w @ dst
    a, b = src - ms, dst - md
    cov = (b * w[:, None]).T @ a
    u, d, vt = np.linalg.svd(cov)
    sign = np.ones(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[2] = -1.0
    r = u @ np.diag(sign) @ vt
    var = float((w * (a * a).sum(axis=1)).sum())
    s = float((d * sign).sum() / var) if scale and var > 0 else 1.0
    return s, r, md - s * r @ ms


class ThinPlate3D:
    """A thin-plate spline in 3D (the kernel ``|r|``, biharmonic in 3D) taking ``source`` points exactly (or, with
    ``smoothing``, nearly) onto ``target`` points, affine far away."""

    def __init__(self, source, target, smoothing: float = 0.0):
        self.source = np.asarray(source, np.float64)
        target = np.asarray(target, np.float64)
        n = len(self.source)
        k = np.linalg.norm(self.source[:, None] - self.source[None], axis=2) + smoothing * np.eye(n)
        p = np.c_[np.ones(n), self.source]
        a = np.zeros((n + 4, n + 4))
        a[:n, :n], a[:n, n:], a[n:, :n] = k, p, p.T
        rhs = np.zeros((n + 4, 3))
        rhs[:n] = target
        solution = np.linalg.lstsq(a, rhs, rcond=None)[0]
        self.weights, self.affine = solution[:n], solution[n:]

    def __call__(self, points) -> np.ndarray:
        points = np.asarray(points, np.float64)
        out = np.c_[np.ones(len(points)), points] @ self.affine
        for start in range(0, len(points), 4096):
            chunk = points[start:start + 4096]
            out[start:start + 4096] += np.linalg.norm(chunk[:, None] - self.source[None], axis=2) @ self.weights
        return out


def paired_barycentric(p, a, b, c) -> np.ndarray:
    """``(P, 3)``: barycentric coordinates of the nearest point of triangle ``(a[i], b[i], c[i])`` to ``p[i]``
    (Ericson, *Real-Time Collision Detection*, 5.1.5), for pairs."""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = np.einsum("nk,nk->n", ab, ap), np.einsum("nk,nk->n", ac, ap)
    bp = p - b
    d3, d4 = np.einsum("nk,nk->n", ab, bp), np.einsum("nk,nk->n", ac, bp)
    cp = p - c
    d5, d6 = np.einsum("nk,nk->n", ab, cp), np.einsum("nk,nk->n", ac, cp)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    out = np.zeros((len(p), 3))
    done = np.zeros(len(p), bool)

    def put(mask, u, v, w):
        nonlocal done
        mask = mask & ~done
        if mask.any():
            out[mask] = np.stack([np.broadcast_to(u, mask.shape)[mask], np.broadcast_to(v, mask.shape)[mask],
                                  np.broadcast_to(w, mask.shape)[mask]], axis=-1)
        done |= mask

    with np.errstate(divide="ignore", invalid="ignore"):
        put((d1 <= 0) & (d2 <= 0), 1.0, 0.0, 0.0)
        put((d3 >= 0) & (d4 <= d3), 0.0, 1.0, 0.0)
        put((d6 >= 0) & (d5 <= d6), 0.0, 0.0, 1.0)
        t = d1 / (d1 - d3)
        put((vc <= 0) & (d1 >= 0) & (d3 <= 0), 1.0 - t, t, 0.0)
        t = d2 / (d2 - d6)
        put((vb <= 0) & (d2 >= 0) & (d6 <= 0), 1.0 - t, 0.0, t)
        t = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        put((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0), 0.0, 1.0 - t, t)
        denom = va + vb + vc
        denom = np.where(np.abs(denom) < 1e-30, 1e-30, denom)
        v, w = vb / denom, vc / denom
        put(np.ones(len(p), bool), 1.0 - v - w, v, w)
    return np.nan_to_num(out)


class SurfaceIndex:
    """Nearest points on a large triangle mesh (``vertices``, ``triangles``): each query looks at the triangles
    around its ``k`` nearest vertices (exact unless the mesh's triangles are much larger than their neighbours'
    spacing, as on a regular scan mesh they are not)."""

    def __init__(self, vertices, triangles, k: int = 8):
        self.vertices = np.asarray(vertices, np.float64)
        self.triangles = np.asarray(triangles, np.int64)
        self.k = k
        order = np.argsort(self.triangles.ravel(), kind="stable")
        vertex_of = self.triangles.ravel()[order]
        self._tri_of = order // 3
        self._start = np.searchsorted(vertex_of, np.arange(len(self.vertices) + 1))
        counts = np.diff(self._start)
        self._fan = int(counts.max()) if len(counts) else 0
        table = np.full((len(self.vertices), max(self._fan, 1)), -1, np.int64)
        for j in range(self._fan):
            has = counts > j
            table[has, j] = self._tri_of[self._start[:-1][has] + j]
        self._table = table
        self._sq = (self.vertices ** 2).sum(axis=1)

    def nearest_vertices(self, points, k=None) -> np.ndarray:
        """``(P, k)`` indices of the nearest vertices."""
        points = np.asarray(points, np.float64)
        k = min(k or self.k, len(self.vertices))
        out = np.empty((len(points), k), np.int64)
        for s in range(0, len(points), 512):
            q = points[s:s + 512]
            d = (q ** 2).sum(axis=1)[:, None] - 2.0 * q @ self.vertices.T + self._sq[None]
            near = np.argpartition(d, k - 1, axis=1)[:, :k]
            out[s:s + 512] = near
        return out

    def closest(self, points, allowed=None, compatible=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(distance, triangle, barycentric)`` of the nearest point on the mesh to each of ``points`` (triangle -1
        and distance inf where none is allowed). ``allowed`` (``(T,)`` bool) leaves the other triangles out (their
        vertices still lead to their neighbours); ``compatible(queries, triangles) -> bool`` (index arrays of equal
        length) leaves out the pairs it rejects."""
        points = np.asarray(points, np.float64)
        near = self.nearest_vertices(points)
        cand = self._table[near].reshape(len(points), -1)  # (P, k * fan)
        if allowed is not None:
            cand = np.where((cand >= 0) & np.asarray(allowed)[np.maximum(cand, 0)], cand, -1)
        if compatible is not None:
            rows = np.repeat(np.arange(len(points)), cand.shape[1])
            flat = cand.ravel()
            ok = flat >= 0
            keep = np.zeros(len(flat), bool)
            keep[ok] = compatible(rows[ok], flat[ok])
            cand = np.where(keep.reshape(cand.shape), cand, -1)
        best_d = np.full(len(points), np.inf)
        best_t = np.full(len(points), -1, np.int64)
        best_b = np.zeros((len(points), 3))
        corners = self.vertices[self.triangles]
        for j in range(cand.shape[1]):
            t = cand[:, j]
            ok = t >= 0
            if not ok.any():
                continue
            idx = np.nonzero(ok)[0]
            tri = corners[t[idx]]
            bary = paired_barycentric(points[idx], tri[:, 0], tri[:, 1], tri[:, 2])
            q = np.einsum("nc,ncd->nd", bary, tri)
            d = np.linalg.norm(q - points[idx], axis=1)
            better = d < best_d[idx]
            sel = idx[better]
            best_d[sel], best_t[sel], best_b[sel] = d[better], t[idx][better], bary[better]
        return best_d, best_t, best_b

    def point(self, triangle, barycentric, vertices=None) -> np.ndarray:
        """The points at ``barycentric`` coordinates in ``triangle`` (of ``vertices``, default the index's)."""
        v = self.vertices if vertices is None else np.asarray(vertices, np.float64)
        return np.einsum("nc,ncd->nd", np.asarray(barycentric, np.float64), v[self.triangles[np.asarray(triangle)]])


def smoothstep(x) -> np.ndarray:
    x = np.clip(np.asarray(x, np.float64), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def sphere_fit(points) -> tuple[np.ndarray, float]:
    """``(centre, radius)`` of the least-squares sphere through ``points``."""
    p = np.asarray(points, np.float64)
    a = np.c_[2.0 * p, np.ones(len(p))]
    b = (p * p).sum(axis=1)
    x = np.linalg.lstsq(a, b, rcond=None)[0]
    centre = x[:3]
    return centre, float(np.sqrt(max(x[3] + centre @ centre, 0.0)))
