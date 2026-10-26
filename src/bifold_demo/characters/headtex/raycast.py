"""Rays and nearest points on a triangle mesh, in numpy (a head has about 3000 triangles, so brute force is fast).

Triangles are given by their corners, ``(T, 3, 3)``. Results name the triangle and the barycentric coordinates of the
point in it (weights of the three corners), which is how a point on a skinned mesh follows the skinning.
"""

from __future__ import annotations

import numpy as np

CHUNK = 64  # rays or points per batch (keeps the (chunk, T, 3) arrays small)


def corners_of(positions, indices) -> np.ndarray:
    """``(T, 3, 3)`` corners of the triangles ``indices`` (``(T, 3)`` or flat) of ``positions``."""
    return np.asarray(positions, np.float64)[np.asarray(indices, np.int64).reshape(-1, 3)]


def triangle_normals(corners: np.ndarray) -> np.ndarray:
    n = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-18)


def point_on(corners: np.ndarray, triangle, barycentric) -> np.ndarray:
    """The points at ``barycentric`` coordinates in the triangles ``triangle``."""
    return np.einsum("nk,nkd->nd", np.asarray(barycentric, np.float64), corners[np.asarray(triangle)])


def intersect(origins, directions, corners: np.ndarray, two_sided: bool = True):
    """The first triangle each ray hits: ``(distance, triangle, barycentric)``, with triangle -1 (and distance inf)
    where a ray hits nothing. ``origins`` is one point or one per ray; ``directions`` ``(R, 3)``. Without
    ``two_sided`` only triangles facing the ray (counter-clockwise seen from its origin) count."""
    directions = np.asarray(directions, np.float64).reshape(-1, 3)
    origins = np.broadcast_to(np.asarray(origins, np.float64), directions.shape)
    v0 = corners[:, 0]
    e1, e2 = corners[:, 1] - v0, corners[:, 2] - v0
    n = len(directions)
    best_t = np.full(n, np.inf)
    best_tri = np.full(n, -1, np.int64)
    best_uv = np.zeros((n, 2))
    for s in range(0, n, CHUNK):
        d, o = directions[s:s + CHUNK, None, :], origins[s:s + CHUNK, None, :]
        p = np.cross(d, e2)  # (r, T, 3)
        det = np.einsum("tk,rtk->rt", e1, p)
        ok = np.abs(det) > 1e-14 if two_sided else det > 1e-14
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        tvec = o - v0  # (r, T, 3)
        u = np.einsum("rtk,rtk->rt", tvec, p) * inv
        q = np.cross(tvec, e1)
        v = np.einsum("rtk,rtk->rt", d, q) * inv
        t = np.einsum("tk,rtk->rt", e2, q) * inv
        eps = -1e-9
        hit = ok & (u >= eps) & (v >= eps) & (u + v <= 1.0 - eps) & (t > 1e-9)
        t = np.where(hit, t, np.inf)
        k = t.argmin(axis=1)
        rows = np.arange(len(k))
        tk = t[rows, k]
        found = np.isfinite(tk)
        idx = np.arange(s, s + len(k))[found]
        best_t[idx] = tk[found]
        best_tri[idx] = k[found]
        best_uv[idx] = np.c_[u[rows, k], v[rows, k]][found]
    bary = np.c_[1.0 - best_uv.sum(axis=1), best_uv]
    bary[best_tri < 0] = 0.0
    return best_t, best_tri, np.clip(bary, 0.0, 1.0)


def closest(points, corners: np.ndarray, allowed=None):
    """The nearest point on the mesh to each of ``points`` (``(P, 3)``): ``(distance, triangle, barycentric)``.
    ``allowed`` (``(T,)`` bool) restricts it to some triangles."""
    points = np.asarray(points, np.float64).reshape(-1, 3)
    if allowed is not None:
        index = np.nonzero(allowed)[0]
        corners = corners[index]
    a, b, c = corners[:, 0], corners[:, 1], corners[:, 2]
    ab, ac = b - a, c - a
    n = len(points)
    best_d = np.full(n, np.inf)
    best_tri = np.zeros(n, np.int64)
    best_bary = np.zeros((n, 3))
    for s in range(0, n, CHUNK):
        bary = _closest_barycentric(points[s:s + CHUNK], a, b, c, ab, ac)  # (p, T, 3)
        q = bary[..., 0:1] * a + bary[..., 1:2] * b + bary[..., 2:3] * c
        d = np.linalg.norm(q - points[s:s + CHUNK, None, :], axis=2)
        k = d.argmin(axis=1)
        rows = np.arange(len(k))
        best_d[s:s + len(k)] = d[rows, k]
        best_tri[s:s + len(k)] = k
        best_bary[s:s + len(k)] = bary[rows, k]
    if allowed is not None:
        best_tri = index[best_tri]
    return best_d, best_tri, best_bary


def _closest_barycentric(p, a, b, c, ab, ac):
    """Barycentric coordinates of the nearest point of each triangle to each point (Ericson, *Real-Time Collision
    Detection*, 5.1.5), vectorised: ``(P, T, 3)``."""
    ap = p[:, None, :] - a
    d1, d2 = np.einsum("tk,ptk->pt", ab, ap), np.einsum("tk,ptk->pt", ac, ap)
    bp = p[:, None, :] - b
    d3, d4 = np.einsum("tk,ptk->pt", ab, bp), np.einsum("tk,ptk->pt", ac, bp)
    cp = p[:, None, :] - c
    d5, d6 = np.einsum("tk,ptk->pt", ab, cp), np.einsum("tk,ptk->pt", ac, cp)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    shape = d1.shape
    out = np.zeros(shape + (3,))
    done = np.zeros(shape, bool)

    def put(mask, u, v, w):
        nonlocal done
        mask = mask & ~done
        out[mask] = np.stack([np.broadcast_to(u, shape)[mask], np.broadcast_to(v, shape)[mask],
                              np.broadcast_to(w, shape)[mask]], axis=-1)
        done |= mask

    with np.errstate(divide="ignore", invalid="ignore"):
        put((d1 <= 0) & (d2 <= 0), 1.0, 0.0, 0.0)  # corner a
        put((d3 >= 0) & (d4 <= d3), 0.0, 1.0, 0.0)  # corner b
        put((d6 >= 0) & (d5 <= d6), 0.0, 0.0, 1.0)  # corner c
        t = d1 / (d1 - d3)
        put((vc <= 0) & (d1 >= 0) & (d3 <= 0), 1.0 - t, t, 0.0)  # edge ab
        t = d2 / (d2 - d6)
        put((vb <= 0) & (d2 >= 0) & (d6 <= 0), 1.0 - t, 0.0, t)  # edge ac
        t = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        put((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0), 0.0, 1.0 - t, t)  # edge bc
        denom = va + vb + vc
        denom = np.where(np.abs(denom) < 1e-30, 1e-30, denom)
        v, w = vb / denom, vc / denom
        put(np.ones(shape, bool), 1.0 - v - w, v, w)  # inside the face
    return np.nan_to_num(out)
