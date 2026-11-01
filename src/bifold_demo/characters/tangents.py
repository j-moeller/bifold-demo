"""Tangent frames for normal mapping, and the maps they serve: what a normal map's colours mean on a mesh.

A tangent-space normal map stores, per texel, a normal relative to the surface: red along the texture's ``+u``
direction (the *tangent*), green along ``+v`` (the *bitangent*; see :data:`GREEN`), blue along the surface normal. To
use it, every vertex needs the directions in which ``u`` and ``v`` grow on the mesh.

:func:`vertex_tangents` computes them the way MikkTSpace (the common convention for baking and rendering) does in its
essentials:

* per triangle, the derivatives of the position by ``u`` and by ``v`` from its edges and their UV differences;
* per corner, that tangent laid into the plane of the corner's vertex normal, normalised and weighted by the corner's
  angle (so neither the triangles' sizes nor the texture's scale matter);
* summed over the *welded* mesh: vertices at the same position with the same UV (split only because their normals
  differ, at hard edges) share one sum, so the tangent field runs on smoothly across such splits; vertices split
  by UV seams keep their own, since the texture's directions differ on either side of a seam;
* the handedness (``w`` = +1 or -1) says whether ``v`` grows along ``normal x tangent`` (+1) or against it (-1, a
  mirrored UV layout); the bitangent is ``w * cross(normal, tangent)``. Where mirrored and unmirrored triangles meet at
  one vertex (a mirror seam with shared UVs), the side with the larger summed angle decides it.

:func:`normal_green` finds out which way a normal map's green channel points (towards ``+v`` or ``-v``): a normal map
baked from a height field holds its gradient, and a gradient field has no curl, except when one of its components is
read backwards. Rocketbox's heads are ``+v`` (checked on the 20 prepared heads; the README section "High-resolution
heads"), like Unity's and OpenGL's convention.

:func:`pack_maps` packs a head's normal and specular maps into the one RGBA image the renderer reads: the normal's x
and y in red and green (green towards ``+v``; z = sqrt(1 - x^2 - y^2) is rebuilt when drawing), the specular level in
blue and, in alpha, room for ambient occlusion (255: none yet).
"""

from __future__ import annotations

import numpy as np

GREEN = "+v"  # the convention of the packed maps (pack_maps): green grows towards +v, as in OpenGL and Unity
WELD_DIGITS = 5  # positions are compared to 10 micrometres and UVs to 1e-6 when welding
FOLD_SCALE = 4.0  # texels at 2048: the blur under which normal_green judges a map (folds, not pores)


def vertex_tangents(positions, normals, uvs, indices) -> np.ndarray:
    """``(N, 4)`` float32 per vertex: the unit tangent (the direction ``u`` grows in, at right angles to the
    vertex normal) and the handedness ``w`` (+1, or -1 where the UV layout is mirrored); see the module docs.

    Vertices that no triangle with a proper UV area touches get any tangent at right angles to their normal, with
    ``w = +1``."""
    p = np.asarray(positions, np.float64).reshape(-1, 3)
    n = np.asarray(normals, np.float64).reshape(-1, 3)
    n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    uv = np.asarray(uvs, np.float64).reshape(-1, 2)
    tri = np.asarray(indices, np.int64).reshape(-1, 3)
    count = len(p)
    out = np.zeros((count, 4), np.float32)
    if count == 0:
        return out

    e1, e2 = p[tri[:, 1]] - p[tri[:, 0]], p[tri[:, 2]] - p[tri[:, 0]]
    d1, d2 = uv[tri[:, 1]] - uv[tri[:, 0]], uv[tri[:, 2]] - uv[tri[:, 0]]
    det = d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1]
    scale = np.maximum(np.abs(d1).max(axis=1), np.abs(d2).max(axis=1))
    usable = np.abs(det) > 1e-12 * np.maximum(scale * scale, 1e-30)
    safe = np.where(usable, det, 1.0)[:, None]
    du = (e1 * d2[:, 1:2] - e2 * d1[:, 1:2]) / safe  # dP/du
    dv = (e2 * d1[:, 0:1] - e1 * d2[:, 0:1]) / safe  # dP/dv

    group = _weld(p, uv)
    groups = int(group.max()) + 1
    sums = np.zeros((2, groups, 3))  # per handedness (+1, -1): angle-weighted tangents
    weight = np.zeros((2, groups))
    for k in range(3):
        v = tri[:, k]
        a, b = p[tri[:, (k + 1) % 3]] - p[v], p[tri[:, (k + 2) % 3]] - p[v]
        cos = np.einsum("ij,ij->i", a, b) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-30)
        angle = np.arccos(np.clip(cos, -1.0, 1.0))
        nv = n[v]
        t = du - nv * np.einsum("ij,ij->i", nv, du)[:, None]
        length = np.linalg.norm(t, axis=1)
        ok = usable & (length > 1e-12) & (angle > 0)
        t = t / np.maximum(length, 1e-30)[:, None]
        hand = np.einsum("ij,ij->i", np.cross(nv, t), dv) >= 0  # True: +1
        side = np.where(hand, 0, 1)
        g = group[v]
        for s in (0, 1):
            pick = ok & (side == s)
            np.add.at(sums[s], g[pick], t[pick] * angle[pick, None])
            np.add.at(weight[s], g[pick], angle[pick])

    mirrored = weight[1] > weight[0]  # per group: the side with the larger summed angle decides
    chosen = np.where(mirrored[:, None], sums[1], sums[0])[group]
    t = chosen - n * np.einsum("ij,ij->i", n, chosen)[:, None]
    length = np.linalg.norm(t, axis=1)
    missing = length < 1e-9
    if missing.any():
        t[missing] = _any_perpendicular(n[missing])
        length[missing] = 1.0
    out[:, :3] = t / length[:, None]
    out[:, 3] = np.where(mirrored[group] & ~missing, -1.0, 1.0)
    return out


def _weld(positions: np.ndarray, uvs: np.ndarray) -> np.ndarray:
    """A group per vertex: vertices at the same position with the same UV share one."""
    keys = np.concatenate([np.round(positions * 10 ** WELD_DIGITS), np.round(uvs * 1e6)], axis=1).astype(np.int64)
    _, group = np.unique(keys, axis=0, return_inverse=True)
    return group.ravel()


def _any_perpendicular(n: np.ndarray) -> np.ndarray:
    """Unit vectors at right angles to the unit vectors ``n``."""
    helper = np.where((np.abs(n[:, 1]) < 0.9)[:, None], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0])
    t = np.cross(helper, n)
    return t / np.linalg.norm(t, axis=1, keepdims=True)


def normal_green(normal_rgb: np.ndarray, margin: float = 0.15) -> tuple[str, float, float]:
    """Which way the green channel of the tangent-space normal map ``normal_rgb`` (HxWx3 ``uint8``, row 0 at the
    top of the image, as image files are) points: ``("+v" or "-v", residual if +v, residual if -v)``.

    The residual is the mean curl of the map's gradient field relative to the mean size of the two terms it is made
    of (0 for a perfect height field; around 1 for a green channel read backwards), over the texels that are not
    flat, after a blur of :data:`FOLD_SCALE` texels at 2048 (in proportion at other sizes): it measures the folds,
    which matter for the shading, not the pores. (One Rocketbox head, ``Male_Adult_04``, has a fine skin texture that
    reads the other way below about 2 texels, while its folds read like every other head's.) When the two residuals
    differ by less than ``margin`` the map cannot tell, and the answer is :data:`GREEN`."""
    import cv2

    rgb = np.asarray(normal_rgb, np.float32)[..., :3] / 127.5 - 1.0
    nx, ny = rgb[..., 0], rgb[..., 1]
    nz = np.maximum(rgb[..., 2], 0.2)
    detail = (np.hypot(nx, ny) > 0.02).astype(np.uint8)
    detail = cv2.erode(detail, np.ones((5, 5), np.uint8)).astype(bool)
    if detail.sum() < 64:
        return GREEN, float("nan"), float("nan")
    sigma = max(1.0, FOLD_SCALE * max(rgb.shape[:2]) / 2048)
    p = cv2.GaussianBlur(-nx / nz, (0, 0), sigma)
    dp_dv = -np.gradient(p, axis=0)  # rows run down the image, v up
    residual = []
    for sign in (1.0, -1.0):  # green towards +v, then towards -v
        q = cv2.GaussianBlur(-sign * ny / nz, (0, 0), sigma)
        dq_du = np.gradient(q, axis=1)
        curl = np.abs(dp_dv - dq_du)[detail].mean()
        residual.append(float(curl / max((np.abs(dp_dv) + np.abs(dq_du))[detail].mean(), 1e-12)))
    up, down = residual
    if abs(up - down) < margin:
        return GREEN, up, down
    return ("+v" if up < down else "-v"), up, down


def pack_maps(normal_rgb: np.ndarray, specular=None, green: str = GREEN, occlusion=None) -> np.ndarray:
    """The packed map (HxWx4 ``uint8``) of a normal map (HxWx3 ``uint8``, green towards ``green``) and a specular map
    (HxW or HxWx3 ``uint8``, its mean over the channels, which are nearly equal; ``None``: no specular, 0) and an
    ambient-occlusion map (HxW ``uint8``; ``None``: none, 255): red and green the normal's x and y with green
    towards ``+v`` (:data:`GREEN`), blue the specular level, alpha the ambient occlusion. The maps must have the same
    size."""
    normal = np.asarray(normal_rgb, np.uint8)
    height, width = normal.shape[:2]
    out = np.empty((height, width, 4), np.uint8)
    out[..., 0] = normal[..., 0]
    out[..., 1] = normal[..., 1] if green == GREEN else 255 - normal[..., 1]
    if specular is None:
        out[..., 2] = 0
    else:
        spec = np.asarray(specular)
        if spec.shape[:2] != (height, width):
            raise ValueError(f"the specular map is {spec.shape[1]}x{spec.shape[0]}, the normal map {width}x{height}")
        out[..., 2] = spec if spec.ndim == 2 else np.round(spec[..., :3].astype(np.float32).mean(axis=2))
    if occlusion is None:
        out[..., 3] = 255
    else:
        out[..., 3] = occlusion
    return out
