"""How far one head is from another: the geometry measures of the shape fit's evaluation.

Two heads (a fitted head and the person it was fitted to: in the evaluation, renders of avatar A fitted onto avatar
B's head and compared with A) are compared as the NoW benchmark does (Sanyal et al. 2019): aligned by their
landmarks, once rigidly and once with a scale (a similarity), and measured

* at MediaPipe's landmarks, which both heads have from stage 0 of the head textures (the 468 of the face mesh; the
  irises lie on eyeballs whose size is the artist's), and
* by the distance between the surfaces over the face (the skin inside each head's face oval seen from the front),
  both ways,

in millimetres, as the median and the mean, and per region as REALY does (Chai et al. 2022): the eyes, the brows, the
nose, the mouth, the cheeks, the forehead, the outline (the face oval) and the profile (the midline). The alignment
uses the landmarks inside the face (not the outline, which is a silhouette, nor the irises).
"""

from __future__ import annotations

import numpy as np

from ..headtex import canonical as cf
from . import mesh

REGIONS = ("eyes", "brows", "nose", "mouth", "cheeks", "forehead", "outline", "profile")
MIDLINE = 0.004  # metres from the nose's midline: the profile's landmarks


def regions(points) -> dict[str, np.ndarray]:
    """The landmark indices of each region (see the module docs), from a head's 478 landmark ``points``."""
    p = np.asarray(points, np.float64)
    eyes = np.array(sorted(set(cf.RIGHT_EYE) | set(cf.LEFT_EYE)))
    brows = np.array(sorted(set(cf.RIGHT_BROW) | set(cf.LEFT_BROW)))
    nose = np.array(sorted(cf.NOSE))
    mouth = np.array(sorted(set(cf.LIPS_OUTER) | set(cf.LIPS_INNER)))
    outline = np.array(sorted(cf.FACE_OVAL))
    mesh_points = np.arange(cf.MESH_COUNT)
    taken = set(eyes) | set(brows) | set(nose) | set(mouth) | set(outline)
    rest = np.array([k for k in mesh_points if k not in taken])
    middle = p[cf.NOSE_TIP, 0]
    profile = mesh_points[np.abs(p[:cf.MESH_COUNT, 0] - middle) < MIDLINE]
    brow_top = p[brows, 1].max()
    eye_y = p[eyes, 1].mean()
    mouth_y = p[mouth, 1].mean()
    forehead = rest[p[rest, 1] > brow_top]
    nose_half = 0.5 * (p[nose, 0].max() - p[nose, 0].min())
    cheeks = rest[(p[rest, 1] < eye_y) & (p[rest, 1] > mouth_y) & (np.abs(p[rest, 0] - middle) > nose_half)]
    return {"eyes": eyes, "brows": brows, "nose": nose, "mouth": mouth, "cheeks": cheeks, "forehead": forehead,
            "outline": outline, "profile": profile}


def _aligning() -> np.ndarray:
    keep = np.ones(cf.COUNT, bool)
    keep[list(cf.FACE_OVAL)] = False
    keep[cf.MESH_COUNT:] = False
    return keep


def align(source_marks, target_marks, scale: bool, weights=None) -> tuple[float, np.ndarray, np.ndarray]:
    """``(s, R, t)`` taking ``source_marks`` onto ``target_marks`` (the landmarks inside the face; ``scale``: with a
    scale, else rigid)."""
    keep = _aligning()
    w = None if weights is None else np.asarray(weights, np.float64)[keep]
    return mesh.umeyama(np.asarray(source_marks)[keep], np.asarray(target_marks)[keep], w, scale=scale)


def face_vertices(points, triangles, marks) -> np.ndarray:
    """Indices of the vertices of the face: inside the face oval of ``marks`` seen from the front, on the front of
    the head (their normals facing forward)."""
    import cv2

    p = np.asarray(points, np.float64)
    oval = np.asarray(marks, np.float64)[list(cf.FACE_OVAL)][:, :2]
    normals = mesh.vertex_normals(p, triangles, mesh.weld(p)[0])
    scale = 2000.0  # pixels per metre for the polygon test
    lo = oval.min(axis=0) - 0.01
    contour = np.round((oval - lo) * scale).astype(np.int32)
    inside = np.array([cv2.pointPolygonTest(contour, (float(x), float(y)), False) >= 0
                       for x, y in np.round((p[:, :2] - lo) * scale)])
    front = normals[:, 2] > 0.2
    behind = p[:, 2] > np.asarray(marks)[list(cf.FACE_OVAL), 2].min() - 0.01
    return np.nonzero(inside & front & behind)[0]


def compare(points, triangles, marks, target_points, target_triangles, target_marks, *, skin=None,
            target_skin=None, weights=None) -> dict:
    """The measures (see the module docs) of a head (``points``, ``triangles``, its 478 landmarks ``marks``, its
    skin vertices ``skin``) against a target head: ``{"rigid": {...}, "similarity": {...}, "scale": s}``, each
    ``{"landmarks": {"median", "mean", region: median}, "surface": {"median", "mean", "there", "back",
    region: median}}``, in millimetres."""
    points = np.asarray(points, np.float64)
    target_points = np.asarray(target_points, np.float64)
    marks, target_marks = np.asarray(marks, np.float64), np.asarray(target_marks, np.float64)
    groups = regions(target_marks)
    tri = np.asarray(triangles, np.int64).reshape(-1, 3)
    ttri = np.asarray(target_triangles, np.int64).reshape(-1, 3)
    if skin is not None:
        tri = tri[np.asarray(skin)[tri].all(axis=1)]
    if target_skin is not None:
        ttri = ttri[np.asarray(target_skin)[ttri].all(axis=1)]
    face = face_vertices(points, tri, marks)
    target_face = face_vertices(target_points, ttri, target_marks)
    target_index = _index(target_points, ttri)
    out = {}
    for kind, scaled in (("rigid", False), ("similarity", True)):
        s, r, t = align(marks, target_marks, scaled, weights)
        moved = s * points @ r.T + t
        moved_marks = s * marks @ r.T + t
        d = np.linalg.norm(moved_marks - target_marks, axis=1)[:cf.MESH_COUNT] * 1000.0
        lm = {"median": float(np.median(d)), "mean": float(np.mean(d))}
        for name, ids in groups.items():
            lm[name] = float(np.median(d[ids])) if len(ids) else float("nan")
        there, _tri, _bary = target_index.closest(moved[face])
        index = _index(moved, tri)
        back, _tri, _bary = index.closest(target_points[target_face])
        both = np.r_[there, back] * 1000.0
        nearest = _nearest_region(target_points[target_face], target_marks, groups)
        sf = {"median": float(np.median(both)), "mean": float(np.mean(both)),
              "there": float(np.median(there) * 1000.0), "back": float(np.median(back) * 1000.0)}
        for name in REGIONS:
            mine = nearest == name
            sf[name] = float(np.median(back[mine]) * 1000.0) if mine.any() else float("nan")
        out[kind] = {"landmarks": lm, "surface": sf}
        if scaled:
            out["scale"] = float(s)
    return out


def _index(points, triangles) -> mesh.SurfaceIndex:
    """A :class:`~.mesh.SurfaceIndex` of the triangles' own vertices only (others would lead nowhere)."""
    used = np.unique(triangles)
    remap = np.full(len(points), -1, np.int64)
    remap[used] = np.arange(len(used))
    return mesh.SurfaceIndex(np.asarray(points)[used], remap[triangles])


def _nearest_region(points, marks, groups) -> np.ndarray:
    """The region of the landmark nearest each point (for the surface's regions)."""
    label = np.full(cf.MESH_COUNT, "", dtype=object)
    for name in ("cheeks", "forehead", "profile", "outline", "nose", "mouth", "brows", "eyes"):  # last wins
        label[groups[name]] = name
    known = np.nonzero(label != "")[0]
    d = np.linalg.norm(np.asarray(points)[:, None] - np.asarray(marks)[known][None], axis=2)
    return label[known][np.argmin(d, axis=1)]
