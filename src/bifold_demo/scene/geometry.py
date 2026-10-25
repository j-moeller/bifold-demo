"""Triangle meshes for the built-in shapes.

Every mesh carries positions, normals and texture coordinates (UVs, with
``v = 0`` at the bottom of the image). Faces wind counter-clockwise when seen
from the side their normal points to.
"""

from __future__ import annotations

import math

import numpy as np


class Mesh:
    """An indexed triangle mesh.

    Build your own geometry by passing arrays directly::

        Mesh(positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
             uvs=[[0, 0], [1, 0], [0, 1]],
             indices=[[0, 1, 2]])

    Normals are computed from the faces when omitted.
    """

    def __init__(self, positions, indices, uvs=None, normals=None):
        self.positions = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
        self.indices = np.asarray(indices, dtype=np.uint32).reshape(-1, 3)
        n = len(self.positions)
        if self.indices.size and self.indices.max() >= n:
            raise ValueError("mesh indices reference missing vertices")
        self.uvs = np.zeros((n, 2), np.float32) if uvs is None else np.asarray(uvs, np.float32).reshape(n, 2)
        self.normals = (
            _face_normals(self.positions, self.indices)
            if normals is None
            else np.asarray(normals, np.float32).reshape(n, 3)
        )

    def interleaved(self) -> np.ndarray:
        """Vertex data as rows of ``x y z nx ny nz u v`` (float32)."""
        return np.hstack([self.positions, self.normals, self.uvs]).astype(np.float32)

    def __repr__(self) -> str:
        return f"Mesh(vertices={len(self.positions)}, triangles={len(self.indices)})"


def _face_normals(positions: np.ndarray, indices: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(positions)
    a, b, c = (positions[indices[:, k]] for k in range(3))
    face = np.cross(b - a, c - a)
    for k in range(3):
        np.add.at(normals, indices[:, k], face)
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    return (normals / np.maximum(length, 1e-12)).astype(np.float32)


def _quad_indices(count: int) -> np.ndarray:
    """Two triangles per quad for ``count`` quads stored as 4 consecutive vertices."""
    base = np.arange(count, dtype=np.uint32)[:, None] * 4
    return (base + np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)).reshape(-1, 3)


def quad_mesh(width: float, height: float) -> Mesh:
    """Vertical rectangle in the XY plane, centred on the origin, facing +Z."""
    w, h = width / 2, height / 2
    return Mesh(
        positions=[[-w, -h, 0], [w, -h, 0], [w, h, 0], [-w, h, 0]],
        normals=[[0, 0, 1]] * 4,
        uvs=[[0, 0], [1, 0], [1, 1], [0, 1]],
        indices=_quad_indices(1),
    )


def plane_mesh(width: float, depth: float, uv_repeat=(1.0, 1.0)) -> Mesh:
    """Horizontal rectangle in the XZ plane, centred on the origin, facing +Y.

    The top edge of the texture lies towards -Z (away from the default camera).
    """
    w, d = width / 2, depth / 2
    ru, rv = (uv_repeat, uv_repeat) if np.isscalar(uv_repeat) else uv_repeat
    return Mesh(
        positions=[[-w, 0, d], [w, 0, d], [w, 0, -d], [-w, 0, -d]],
        normals=[[0, 1, 0]] * 4,
        uvs=[[0, 0], [ru, 0], [ru, rv], [0, rv]],
        indices=_quad_indices(1),
    )


# (normal, u axis, v axis) per face; u x v == normal so faces wind CCW from outside.
_BOX_FACES = [
    ((0, 0, 1), (1, 0, 0), (0, 1, 0)),  # front  (+Z)
    ((0, 0, -1), (-1, 0, 0), (0, 1, 0)),  # back   (-Z)
    ((1, 0, 0), (0, 0, -1), (0, 1, 0)),  # right  (+X)
    ((-1, 0, 0), (0, 0, 1), (0, 1, 0)),  # left   (-X)
    ((0, 1, 0), (1, 0, 0), (0, 0, -1)),  # top    (+Y)
    ((0, -1, 0), (1, 0, 0), (0, 0, 1)),  # bottom (-Y)
]


def box_mesh(width: float, height: float, depth: float) -> Mesh:
    """Axis-aligned box centred on the origin; every face shows the whole texture."""
    half = np.array([width, height, depth], dtype=np.float64) / 2
    positions, normals, uvs = [], [], []
    for n, u, v in _BOX_FACES:
        n, u, v = (np.array(x, dtype=np.float64) for x in (n, u, v))
        for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            positions.append((n + su * u + sv * v) * half)
            normals.append(n)
            uvs.append(((su + 1) / 2, (sv + 1) / 2))
    return Mesh(positions, _quad_indices(6), uvs=uvs, normals=normals)


def _grid_indices(rows: int, cols: int) -> np.ndarray:
    """Triangles for a (rows+1) x (cols+1) vertex grid; row 0 is the top edge."""
    r, c = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    a = (r * (cols + 1) + c).ravel()
    b = a + cols + 1
    tris = np.stack([np.stack([a, b, b + 1], 1), np.stack([a, b + 1, a + 1], 1)], 1)
    return tris.reshape(-1, 3).astype(np.uint32)


def sphere_mesh(radius: float, segments: int = 48, rings: int = 24) -> Mesh:
    """UV sphere with an equirectangular texture mapping (u = longitude, v = latitude)."""
    phi = np.linspace(0, math.pi, rings + 1)[:, None]  # 0 at the top pole
    lon = np.linspace(0, 2 * math.pi, segments + 1)[None, :]
    normals = np.stack(
        np.broadcast_arrays(np.sin(phi) * np.sin(lon), np.cos(phi), np.sin(phi) * np.cos(lon)), -1
    ).reshape(-1, 3)
    u, v = np.meshgrid(np.linspace(0, 1, segments + 1), np.linspace(1, 0, rings + 1))
    uvs = np.stack([u, v], -1).reshape(-1, 2)
    return Mesh(normals * radius, _grid_indices(rings, segments), uvs=uvs, normals=normals)


def cylinder_mesh(radius: float, height: float, segments: int = 48, caps: bool = True) -> Mesh:
    """Upright cylinder centred on the origin; the texture wraps once around the side."""
    lon = np.linspace(0, 2 * math.pi, segments + 1)
    x, z = np.sin(lon), np.cos(lon)
    h = height / 2
    # side: top row then bottom row, so row 0 is the top edge like the sphere grid
    side_pos = np.concatenate([np.stack([x * radius, np.full_like(x, y), z * radius], 1) for y in (h, -h)])
    side_nrm = np.concatenate([np.stack([x, np.zeros_like(x), z], 1)] * 2)
    u = np.linspace(0, 1, segments + 1)
    side_uv = np.concatenate([np.stack([u, np.full_like(u, v)], 1) for v in (1.0, 0.0)])
    parts = [(side_pos, side_nrm, side_uv, _grid_indices(1, segments))]
    if caps:
        for sign in (1.0, -1.0):
            ring = np.stack([x * radius, np.full_like(x, sign * h), z * radius], 1)
            pos = np.vstack([[0.0, sign * h, 0.0], ring])
            nrm = np.tile([0.0, sign, 0.0], (len(pos), 1))
            uv = np.vstack([[0.5, 0.5], np.stack([0.5 + x / 2, 0.5 - sign * z / 2], 1)])
            j = np.arange(1, segments + 1)
            tris = np.stack([np.zeros_like(j), j, j + 1], 1) if sign > 0 else np.stack([np.zeros_like(j), j + 1, j], 1)
            parts.append((pos, nrm, uv, tris))
    return _merge(parts)


def _merge(parts) -> Mesh:
    positions, normals, uvs, indices, offset = [], [], [], [], 0
    for pos, nrm, uv, tris in parts:
        positions.append(pos)
        normals.append(nrm)
        uvs.append(uv)
        indices.append(np.asarray(tris) + offset)
        offset += len(pos)
    return Mesh(np.vstack(positions), np.vstack(indices), uvs=np.vstack(uvs), normals=np.vstack(normals))
