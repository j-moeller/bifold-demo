"""S3: the head's outline against each photo's face-skin mask, alternated with the landmarks in :mod:`.fit`.

Landmarks tell the depth of a face poorly (a profile's nose, chin and forehead are where they constrain least); the
outline of the head in a photo tells exactly that. Per photo with the segmenter's masks: the outline that can be
trusted is where the face skin meets the background (not hair, which differs from the avatar's, nor the neck's body
skin, nor clothes); a signed distance to it is computed once (negative inside the face skin), on a crop round the
face at :data:`WORK_SIZE` pixels. Every round of the fit, the head's outline in that photo is found: the skin vertices
of the face (where ICT's face area is, not the ears, scalp or neck) on the contour as the current camera sees it
(their normal turns from facing it to facing away among their neighbours) and on the outer edge of the head's
silhouette (a z-buffer of the head). Each such vertex's residual is the signed distance at its projection, in
:data:`OUTLINE_NOISE` of the face's size, with a Cauchy weight; vertices further than :data:`REACH` from any trusted
outline do not count (another part of the outline: hair, an ear, the neck).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..headtex import raster
from ..headtex.detect import mask_at
from . import mesh
from .head import SKIN

WORK_SIZE = 240  # pixels: the face's crop is scaled to about this height
TURNED = 40.0  # degrees: the photos with a face turned at least this far count for their outline (from the front the
# jaw line's landmarks tell the outline already)
OUTLINE_NOISE = 0.02  # of the face's size: the masks' outline error
REACH = 0.07  # of the face's size: outline points further from a trusted edge do not count
ROBUST = 3.0
BACKGROUND = 0.5
FACE_AREA = 9409  # ICT's face area: vertices below this


@dataclass
class OutlineData:
    """One photo's trusted outline: the signed distance field (``field``, crop pixels; negative inside the face
    skin) with its gradient, the crop (``offset`` and ``scale`` from photo pixels), the face's size in crop pixels,
    and the vertices assigned to the outline this round."""

    field: np.ndarray
    grad_x: np.ndarray
    grad_y: np.ndarray
    offset: np.ndarray
    scale: float
    face: float
    vertices: np.ndarray | None = None


def trusted_outline(masks: dict, size, box) -> OutlineData | None:
    """The :class:`OutlineData` of a photo of ``size`` with segmenter ``masks``, round the face's ``box``
    (``x0, y0, x1, y1`` pixels); ``None`` without the face and background masks or a trusted edge."""
    if not masks or "face" not in masks or "background" not in masks:
        return None
    x0, y0, x1, y1 = box
    w, h = size
    face_size = max(x1 - x0, y1 - y0)
    pad = 0.6 * face_size
    cx0, cy0 = int(max(0, x0 - pad)), int(max(0, y0 - pad))
    cx1, cy1 = int(min(w, x1 + pad)), int(min(h, y1 + pad))
    scale = WORK_SIZE / max(face_size, 1.0)
    face = mask_at(masks["face"], size)[cy0:cy1, cx0:cx1]
    background = mask_at(masks["background"], size)[cy0:cy1, cx0:cx1]
    out_size = (max(8, round((cx1 - cx0) * scale)), max(8, round((cy1 - cy0) * scale)))
    face = cv2.resize(face, out_size, interpolation=cv2.INTER_LINEAR) > 0.5
    background = cv2.resize(background, out_size, interpolation=cv2.INTER_LINEAR) > BACKGROUND
    inside = face.astype(np.uint8)
    edge = (inside - cv2.erode(inside, np.ones((3, 3), np.uint8))) > 0
    near_background = cv2.dilate(background.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    trusted = edge & near_background
    if trusted.sum() < 10:
        return None
    distance = cv2.distanceTransform((~trusted).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
    signed = np.where(face, -distance, distance).astype(np.float32)
    grad_x = cv2.Sobel(signed, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    grad_y = cv2.Sobel(signed, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    return OutlineData(signed, grad_x, grad_y, np.array([cx0, cy0], np.float64), scale, face_size * scale)


def _sample(image, points):
    """Bilinear samples of ``image`` at ``points`` (``(P, 2)`` x, y), and whether each is inside."""
    h, w = image.shape
    x, y = points[:, 0], points[:, 1]
    inside = (x >= 0) & (y >= 0) & (x <= w - 1) & (y <= h - 1)
    x, y = np.clip(x, 0, w - 1.001), np.clip(y, 0, h - 1.001)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    v = (image[y0, x0] * (1 - fx) * (1 - fy) + image[y0, x0 + 1] * fx * (1 - fy) + image[y0 + 1, x0] * (1 - fx) * fy
         + image[y0 + 1, x0 + 1] * fx * fy)
    return v, inside


def prepare_outlines(state) -> None:
    """The trusted outline of the photos of the fit's ``state`` that have masks and show the face turned
    :data:`TURNED` or more (or no face: a profile placed by its outline)."""
    for i, photo in enumerate(state.photos):
        if photo.points is not None:
            if abs(photo.camera.angles()[0]) < TURNED:
                continue
            lo, hi = photo.points.min(axis=0), photo.points.max(axis=0)
        else:
            head = _head_box(photo)
            if head is None:
                continue
            lo, hi = head
        state.outline[i] = trusted_outline(photo.masks, photo.size, (lo[0], lo[1], hi[0], hi[1]))


def _head_box(photo):
    """The face skin's box in a photo without a face's landmarks, from its mask."""
    if not photo.masks or "face" not in photo.masks:
        return None
    face = mask_at(photo.masks["face"], photo.size) > 0.5
    ys, xs = np.nonzero(face)
    if len(xs) < 50:
        return None
    return np.array([xs.min(), ys.min()], np.float64), np.array([xs.max(), ys.max()], np.float64)


def _face_vertices(m) -> np.ndarray:
    """The skin vertices of the face (on ICT's face area: not the ears, the scalp or the neck), away from the neck."""
    hs = m.hs
    if m.space_triangles is None:
        return np.zeros(0, np.int64)
    has = hs.triangle >= 0
    face = np.zeros(hs.vertices, bool)
    face[has] = (m.space_triangles[hs.triangle[has]] < FACE_AREA).all(axis=1)
    return np.nonzero(face & (hs.parts == SKIN) & (hs.fade > 0.9))[0]


def assign_outlines(state) -> None:
    """The face's vertices on the head's outline in each photo, for the current state (see the module docs)."""
    m = state.m
    if getattr(m, "_outline_candidates", None) is None:
        m._outline_candidates = _face_vertices(m)
        node, _count = mesh.weld(m.points)
        m._outline_node = node
        m._outline_edges = mesh.edges(node[m.triangles])
    candidates = m._outline_candidates
    if len(candidates) == 0:
        return
    delta = state.beta - m.alpha
    for i, data in enumerate(state.outline):
        if data is None:
            continue
        camera = state.camera(i)
        x = m.points + m.vertex_offsets(delta, state.expressions[i])
        normals = mesh.vertex_normals(x, m.triangles, m._outline_node)
        to_camera = camera.position - x
        facing = np.einsum("nk,nk->n", normals, to_camera)
        node = m._outline_node
        node_facing = np.zeros(int(node.max()) + 1)
        node_facing[node] = facing
        a, b = m._outline_edges[:, 0], m._outline_edges[:, 1]
        flips = np.sign(node_facing[a]) != np.sign(node_facing[b])
        contour_nodes = np.zeros(len(node_facing), bool)
        contour_nodes[a[flips]] = True
        contour_nodes[b[flips]] = True
        on_contour = candidates[contour_nodes[node[candidates]] & (facing[candidates] >= 0)]
        if len(on_contour) == 0:
            data.vertices = None
            continue
        crop = _crop_camera(camera, data)
        r = raster.rasterise(crop, x[m.triangles], cull=False)
        silhouette = (r.triangle >= 0).astype(np.uint8)
        rim = (cv2.dilate(silhouette, np.ones((3, 3), np.uint8)) - cv2.erode(silhouette, np.ones((5, 5), np.uint8)))
        u = crop.project(x[on_contour])
        px = np.clip(np.round(u).astype(int), 0, np.array(rim.shape[::-1]) - 1)
        data.vertices = on_contour[rim[px[:, 1], px[:, 0]] > 0]


def _crop_camera(camera, data: OutlineData):
    from ..headtex.camera import PhotoCamera

    k = camera.K.copy()
    k[:2] *= data.scale
    k[:2, 2] = (camera.K[:2, 2] - data.offset + 0.5) * data.scale - 0.5
    h, w = data.field.shape
    return PhotoCamera(k, camera.R, camera.t, (w, h))


def outline_rows(data: OutlineData, m, state, i, delta, e, r, t, f, centre, sigma, cols, n, free_modes,
                 with_jacobian):
    """The outline's weighted residuals of photo ``i`` (and their Jacobian), or ``None``."""
    if data is None or data.vertices is None or len(data.vertices) == 0:
        return None
    from .fit import _project, _skew

    verts = data.vertices
    xv = m.points[verts] + m.vertex_offsets(delta, e)[verts]
    xc = xv @ r.T + t
    u, du = _project(xc, f, centre)
    crop = (u - data.offset + 0.5) * data.scale - 0.5
    value, inside = _sample(data.field, crop)
    gx, _ = _sample(data.grad_x, crop)
    gy, _ = _sample(data.grad_y, crop)
    noise = OUTLINE_NOISE * data.face
    near = inside & (np.abs(value) < REACH * data.face)
    res = value / noise
    weight = near / (1.0 + (res / ROBUST) ** 2)
    sw = np.sqrt(weight)
    rows = res * sw
    if not with_jacobian:
        return rows, None
    grad = np.c_[gx, gy] * data.scale / noise * sw[:, None]  # d residual / d photo pixel
    d = np.einsum("ld,ldk->lk", grad, du)  # by the camera-space point
    dr = d @ r
    jac = np.zeros((len(verts), n))
    if free_modes:
        a, b = cols["beta"]
        jac[:, a:b] = np.einsum("ld,kld->lk", dr, m.modes[:free_modes, verts])
    a, b = cols[("pose", i)]
    jac[:, a:a + 3] = -np.einsum("lj,ljk->lk", d, _skew(xv @ r.T))
    jac[:, a + 3:b] = d
    if m.e:
        a, b = cols[("expr", i)]
        jac[:, a:b] = np.einsum("ld,eld->le", dr, m.expression_modes[:, verts])
    if state.focal_free[state.photos[i].group]:
        a, _ = cols["focal"]
        jac[:, a + state.photos[i].group] = np.einsum("ld,ld->l", grad, u - centre)
    return rows, jac
