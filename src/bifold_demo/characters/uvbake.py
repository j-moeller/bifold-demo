"""What every pixel of an avatar's texture is: its point on the body, its normal, its triangle, bone and UV island.

Painting into a texture atlas (a person's face from photos, a new fabric on the clothes) needs to know, for each
texel, where on the 3D body it lands. :func:`bake` rasterises one primitive's triangles in UV space and keeps, per
texel: the bind-pose position and normal (character space: the person faces +Z, +X is their left, Y up), the
triangle, the bone that moves it most, and the UV island it belongs to (triangles joined by shared UV corners).
Texels outside every triangle are not ``covered``. Where triangles overlap in UV (mirrored or stacked parts) the
last one drawn wins and ``overlap`` marks the texel.

A bake takes a second or two for a Rocketbox primitive at 1024x1024 (on a CPU) and is kept per primitive (see
:func:`cached_bake`).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class UVBake:
    """The baked maps of one primitive at ``size`` x ``size`` texels (row 0 = the top of the texture image, i.e.
    ``v`` near 1). Arrays are ``(size, size, ...)``."""

    size: int
    covered: np.ndarray  # bool
    triangle: np.ndarray  # int32, -1 outside
    position: np.ndarray  # float32 (.., 3), NaN outside
    normal: np.ndarray  # float32 (.., 3), unit, NaN outside
    bone: np.ndarray  # int32, -1 outside: the bone with the most weight at the texel
    island: np.ndarray  # int32, -1 outside
    overlap: np.ndarray  # bool: more than one triangle covers the texel

    def texel(self, uv) -> tuple[int, int]:
        """``(row, column)`` of the texel at texture coordinates ``uv``."""
        u, v = float(uv[0]), float(uv[1])
        return (min(self.size - 1, max(0, int((1.0 - v) * self.size))), min(self.size - 1, max(0, int(u * self.size))))


def uv_pixels(uvs: np.ndarray, size: int) -> np.ndarray:
    """Texture coordinates ``(N, 2)`` as ``(x, y)`` pixel positions in a ``size`` x ``size`` image (y down)."""
    uvs = np.asarray(uvs, np.float64)
    return np.c_[uvs[:, 0] * size - 0.5, (1.0 - uvs[:, 1]) * size - 0.5]


def islands(primitive) -> np.ndarray:
    """``(M,)`` the UV island of each triangle: triangles sharing a UV corner (the same texture coordinates at a
    shared vertex) are in one island."""
    tri = np.asarray(primitive.indices, np.int64).reshape(-1, 3)
    uv = np.round(np.asarray(primitive.uvs, np.float64) * 65536.0).astype(np.int64)
    _, corner = np.unique(uv[tri.ravel()], axis=0, return_inverse=True)  # corners with equal texture coordinates
    corner = corner.reshape(-1, 3)
    parent = np.arange(corner.max() + 1 if corner.size else 0)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b, c in corner:
        ra, rb, rc = find(a), find(b), find(c)
        parent[rb] = ra
        parent[find(rc)] = ra
    roots = np.array([find(a) for a in corner[:, 0]]) if len(corner) else np.zeros(0, np.int64)
    _, island = np.unique(roots, return_inverse=True)
    return island.astype(np.int32)


def bake(model, primitive: int, size: int = 1024) -> UVBake:
    """The maps of ``model.primitives[primitive]`` (see the module docs)."""
    prim = model.primitives[primitive]
    tri = np.asarray(prim.indices, np.int64).reshape(-1, 3)
    pixels = uv_pixels(prim.uvs, size)
    ids = np.full((size, size), -1, np.int32)
    counts = np.zeros((size, size), np.uint8)
    shift = 4  # sub-pixel precision of cv2.fillConvexPoly
    corners = np.round(pixels * (1 << shift)).astype(np.int64)
    lo = np.clip(np.floor(pixels[tri].min(axis=1)).astype(int) - 1, 0, size - 1)
    hi = np.clip(np.ceil(pixels[tri].max(axis=1)).astype(int) + 2, 1, size)
    for k, t in enumerate(tri):
        (x0, y0), (x1, y1) = lo[k], hi[k]
        if x1 <= x0 or y1 <= y0:
            continue
        offset = np.array([x0, y0], np.float64)
        poly = (corners[t] - (np.array([x0, y0]) << shift)).astype(np.int32)
        layer = np.zeros((y1 - y0, x1 - x0), np.uint8)  # this triangle alone, in its bounding box
        cv2.fillConvexPoly(layer, poly, 1, lineType=cv2.LINE_8, shift=shift)
        ids[y0:y1, x0:x1][layer.astype(bool)] = k
        # for the overlaps: only texel centres strictly inside, so neighbours sharing an edge do not both count
        counts[y0:y1, x0:x1] += _strictly_inside(pixels[t] - offset, layer.shape)
    covered = ids >= 0
    ys, xs = np.nonzero(covered)
    k = ids[ys, xs]
    a, b, c = pixels[tri[k, 0]], pixels[tri[k, 1]], pixels[tri[k, 2]]
    v0, v1 = b - a, c - a
    p = np.c_[xs, ys].astype(np.float64) - a
    det = v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0]
    det = np.where(np.abs(det) < 1e-12, 1e-12, det)
    l1 = (p[:, 0] * v1[:, 1] - p[:, 1] * v1[:, 0]) / det
    l2 = (v0[:, 0] * p[:, 1] - v0[:, 1] * p[:, 0]) / det
    lam = np.clip(np.c_[1.0 - l1 - l2, l1, l2], 0.0, 1.0)
    lam /= lam.sum(axis=1, keepdims=True)
    positions = np.asarray(prim.positions, np.float64)
    normals = np.asarray(prim.normals, np.float64)
    position = np.full((size, size, 3), np.nan, np.float32)
    normal = np.full((size, size, 3), np.nan, np.float32)
    position[ys, xs] = np.einsum("nk,nkd->nd", lam, positions[tri[k]])
    n = np.einsum("nk,nkd->nd", lam, normals[tri[k]])
    normal[ys, xs] = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    joints, weights = np.asarray(prim.joints), np.asarray(prim.weights, np.float64)
    corner = lam.argmax(axis=1)
    vertex = tri[k, corner]  # the nearest corner's bone
    bone = np.full((size, size), -1, np.int32)
    bone[ys, xs] = joints[vertex, weights[vertex].argmax(axis=1)]
    island = np.full((size, size), -1, np.int32)
    island[ys, xs] = islands(prim)[k]
    return UVBake(size, covered, ids, position, normal, bone, island, counts > 1)


def _strictly_inside(corners: np.ndarray, shape) -> np.ndarray:
    """``uint8`` mask of the texel centres strictly inside the triangle ``corners`` (pixels of a ``shape`` box)."""
    gy, gx = np.mgrid[0:shape[0], 0:shape[1]]
    (ax, ay), (bx, by), (cx, cy) = corners
    det = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
    if abs(det) < 1e-12:
        return np.zeros(shape, np.uint8)
    l0 = ((by - cy) * (gx - cx) + (cx - bx) * (gy - cy)) / det
    l1 = ((cy - ay) * (gx - cx) + (ax - cx) * (gy - cy)) / det
    eps = 1e-6
    return ((l0 > eps) & (l1 > eps) & (1.0 - l0 - l1 > eps)).astype(np.uint8)


_cache: dict[tuple, tuple] = {}


def cached_bake(model, primitive: int, size: int = 1024) -> UVBake:
    """:func:`bake`, kept per (model, primitive, size) while the model lives."""
    key = (id(model), primitive, size)
    hit = _cache.get(key)
    if hit is not None and hit[0] is model:
        return hit[1]
    result = bake(model, primitive, size)
    _cache[key] = (model, result)
    return result
