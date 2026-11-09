"""S1-S4: a person's head shape fitted to all their photos at once.

**What is fitted** (a bundle adjustment, Levenberg-Marquardt on dense normal equations in numpy):

* the shape: the person's identity coefficients ``beta`` on the avatar's head (:mod:`.head`): the head is
  ``x_avatar + modes @ (beta - alpha)``, the avatar's own detail kept; the first :attr:`FitOptions.first_modes`
  first, then all of them with a firmer prior;
* per photo, the camera (a turn and a shift: the head's pose) and the expression: ICT's expression shapes (ARKit's
  names, as MediaPipe's blendshape scores have them: ``eyeBlinkLeft`` is the person's left, like ICT's
  ``eyeBlink_L``), started from MediaPipe's scores and pulled towards them and towards a neutral face, kept between
  0 and 1; the eyes' gaze shapes are left out (the irises do not take part);
* per camera session (photos of one picture shape without EXIF data) the focal length, free with a prior around
  the registration's (:data:`FOCAL_SPREAD`); with EXIF data it is held;
* the size, apart from the shape: the landmarks cannot tell a head from a larger one farther away, so the face's
  overall size (the spread of its landmarks) is held (:data:`SIZE_HOLD`) at a target, with a uniform scale of the
  head as an extra unknown, and the coefficients shape it. The target is set every round: the irises are a ruler
  (a horizontal diameter of 11.7 mm across people, 4 % spread): in the frontal photos where they are large enough,
  the iris radius against the distance between the iris centres (both in pixels, so the camera's distance and lens
  cancel) measures the person's eye distance, which with the shape's proportions gives a size; weighed against the
  avatar's own size (people's spread :data:`SIZE_SPREAD`) that is the target; without irises the avatar's size
  (:attr:`ShapeFit.scale_from` says which). Left free instead, with the irises' pixels against their depth as a few
  more residuals, the turned photos made the head 5-15 % too big (``docs/research/head_shape_results.md``).

**Residuals**: MediaPipe's landmarks against the avatar's stage-0 landmarks as the shape and the camera put them,
weighted by their reliability on the mesh, by how squarely the surface faces the camera (MediaPipe only guesses the
far side of a turned face) and by a robust (Cauchy) weight; the jaw line and the cheeks' outline (MediaPipe's face
oval below the ears) are not points on the mesh but where the head's outline is in that view, so each such landmark
is re-assigned every round to the vertex of the current head that is outermost there (along its marching line, after
Zhu et al.) and only its distance across the outline counts; the head's outline against the photo's face-skin mask
where that mask meets the background (:mod:`.outline`), alternated with the landmarks; and the priors: a unit
Gaussian on ``beta``, the expressions' pulls, the focal length's and the size held.

**A small free-form correction** (:func:`free_form`) then removes what the photos agree on and the space cannot
express: per landmark, the 3D correction that its residuals in at least two photos triangulate to (with the photos'
agreement as its weight), spread smoothly over the skin by a bi-Laplacian, capped at :data:`FREE_CAP`, held at the
neck. Without the identity space this alone is the fit, with stronger smoothing (the fallback: proportions, but no
plausible depth where no photo looks).

**When it is not used.** :func:`fit_person` returns ``usable = False`` with the reason, and the avatar's own head
is kept, when fewer than :attr:`FitOptions.min_photos` photos show a usable face, or the fitted head explains the
photos no better than the avatar's own head (its robust landmark error not :data:`MIN_GAIN` lower), or its
coefficients are implausible (:data:`MAX_MAHALANOBIS`).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from ..headtex import canonical as cf
from ..headtex.camera import PhotoCamera
from . import mesh
from .head import SKIN, HeadSpace

LOWER_OVAL = (454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234)
UPPER_OVAL = tuple(sorted(set(cf.FACE_OVAL) - set(LOWER_OVAL)))  # on the forehead under the hair: left out
IRISES = tuple(range(468, 478))
POINTS = tuple(k for k in range(cf.MESH_COUNT) if k not in set(cf.FACE_OVAL))  # landmarks fitted as points
LANDMARK_NOISE = 0.03  # of the eye distance in pixels: MediaPipe's error per landmark (about 2.5-3.2 %)
ROBUST = 3.0  # Cauchy scale, in landmark noises
IRIS_RADIUS, IRIS_SPREAD = 0.00585, 0.00025  # metres: half of 11.7 +/- 0.5 mm
IRIS_NOISE = 0.06  # of the radius: one photo's measurement
IRIS_YAW = 15.0  # degrees: photos turned further do not measure the irises
IRIS_MIN = 4.0  # pixels: smaller iris radii do not count
SIZE_SPREAD = 0.05  # people's head sizes spread so (relative): the avatar's size is the prior on the person's
SIZE_HOLD = 0.0002  # metres: the face's size (its landmarks' RMS spread) is held at the target so firmly
SCALE_STEP = 0.01  # the uniform scale's coefficient: this share of the head's size per unit
FOCAL_SPREAD = 0.5  # log focal length
EXPRESSION_PULL, EXPRESSION_NEUTRAL = 0.35, 0.6  # spreads: around MediaPipe's score, around 0
EXPRESSION_HELD = 0.01  # ... of the expressions MediaPipe scores below the gate: held at 0
FREE_CAP = 0.003  # metres: the free-form correction's largest offset
FREE_SMOOTH = 30.0  # its bi-Laplacian weight (the fallback's is FALLBACK_SMOOTH)
FALLBACK_SMOOTH = 120.0
MIN_GAIN = 0.03  # the fitted head must lower the robust landmark error by this share ...
MIN_GAIN_ABSOLUTE = 0.02  # ... and by this much (landmark noises): photos the avatar explains need no new head
MAX_MAHALANOBIS = 6.0  # ... and its coefficients' RMS (per mode, in standard deviations) stay below this
FACING_LOW, FACING_HIGH = 0.05, 0.3  # cosines: landmarks count from 0 to fully between these
MARCH_BAND, MARCH_REACH = 0.006, 0.045  # metres: a contour landmark's marching line: across, along


@dataclass
class FitOptions:
    """How :func:`fit_person` fits (the defaults were chosen on the measurements: ``docs/research/
    head_shape_results.md``)."""

    first_modes: int = 40
    prior: float = 3.0  # the Gaussian prior's weight on beta (in unit variances)
    prior_high: float = 6.0  # ... on the modes after first_modes
    centre: str = "avatar"  # what the prior pulls beta to: "avatar" (alpha: keep the avatar where the photos say
    # nothing) or "population" (0: ICT's mean head)
    landmark_noise: float = LANDMARK_NOISE
    expressions: bool = True
    expression_gate: float = 0.3  # an expression is fitted in a photo where MediaPipe scores it at least this
    trust: bool = True  # weight the landmarks by how faithfully MediaPipe's follow a head's shape (landmark_trust)
    contour: bool = True
    outline: bool = True
    iris: bool = True
    free: bool = False  # the free-form correction after the space's fit (always in the fallback without the space)
    free_cap: float = FREE_CAP
    min_photos: int = 2
    rounds: int = 4
    iterations: int = 10
    space: bool = True  # use the identity space (False: the free-form fallback only)


@dataclass
class Photo:
    """One photo's measurements for the fit (:func:`photos_from_views`)."""

    name: str
    size: tuple
    points: np.ndarray | None  # (478, 2) MediaPipe's landmarks, pixels; None: a photo placed by its outline
    camera: PhotoCamera  # the registration's camera (the avatar's own head)
    focal_known: bool
    blendshapes: dict = field(default_factory=dict)
    masks: dict | None = None
    group: int = 0  # the camera session (shared focal length)

    @property
    def eye_distance(self) -> float:
        if self.points is None:
            return 0.0
        return float(np.linalg.norm(self.points[cf.LEFT_IRIS[0]] - self.points[cf.RIGHT_IRIS[0]]))


OUTLINE_MIN_IOU = 0.6  # a photo without a face (a profile) counts for its outline when the registration placed it so


def fitted_view(view) -> bool:
    """Whether the fit uses a registered view: a usable one placed by its face's landmarks, or one placed by the
    head's outline with the segmenter's masks (a profile, for its outline only; the registration's overlap
    :data:`OUTLINE_MIN_IOU` at least: another person's hair rarely lets it reach the 0.8 the textures need)."""
    if view.camera is None:
        return False
    if view.method == "landmarks":
        return bool(view.usable and view.marks is not None)
    return bool(view.method == "silhouette" and view.masks is not None and view.iou >= OUTLINE_MIN_IOU)


def photos_from_views(views) -> list[Photo]:
    """The usable photos with a face among registered ``views`` (:func:`~bifold_demo.characters.headtex.register.
    register_photos`; :func:`fitted_view`), each with its focal-length session: photos of one picture shape without
    a known focal length share one."""
    out = []
    groups: dict = {}
    for k, view in enumerate(views):
        if not fitted_view(view):
            continue
        w, h = view.size
        if view.focal_known:
            group = ("own", k)
        else:
            group = ("shape", round(max(w, h) / min(w, h), 2))
        g = groups.setdefault(group, len(groups))
        face = view.method == "landmarks"
        out.append(Photo(view.source or view.name, (w, h),
                         np.asarray(view.marks.points, np.float64) if face else None, view.camera,
                         bool(view.focal_known), dict(view.marks.blendshapes or {}) if face else {}, view.masks, g))
    return out


@dataclass
class ShapeFit:
    """What :func:`fit_person` found. ``offsets``: ``(N, 3)`` of the head primitive's skin (bind pose), the model's
    and the free-form correction's together (``model_offsets`` without the latter)."""

    usable: bool
    reason: str = ""
    beta: np.ndarray | None = None
    offsets: np.ndarray | None = None
    model_offsets: np.ndarray | None = None
    free: np.ndarray | None = None
    cameras: list = field(default_factory=list)
    expressions: list = field(default_factory=list)  # per photo {name: weight} above 0.05
    focal: list = field(default_factory=list)
    scale_from: str = ""  # "the irises (n photos; ...)" or "the avatar's own size"
    scale: float = 1.0  # the head's size against the avatar's
    iris_radius: float = math.nan  # metres, the fitted person's iris
    error_before: float = math.nan  # robust landmark error, in landmark noises, of the avatar's own head
    error_after: float = math.nan
    photos: int = 0
    mahalanobis: float = math.nan  # RMS of beta per mode
    method: str = ""  # "space", "space + free-form", "free-form"
    warnings: list = field(default_factory=list)
    timings: dict = field(default_factory=dict)

    def summary(self) -> str:
        if not self.usable:
            return f"the head keeps the avatar's shape: {self.reason}"
        lines = [f"head shape fitted to {self.photos} photos ({self.method}); scale from {self.scale_from}",
                 (f"  landmark error {self.error_before:.2f} -> {self.error_after:.2f} (landmark noises, robust); "
                  f"coefficients {self.mahalanobis:.2f} sd per mode")]
        if self.free is not None:
            free = np.linalg.norm(self.free, axis=1) * 1000.0
            lines.append(f"  free-form correction: up to {free.max():.1f} mm, mean {free.mean():.2f} mm")
        if not math.isnan(self.iris_radius):
            lines.append(f"  iris diameter {2000.0 * self.iris_radius:.1f} mm")
        lines += [f"  warning: {w}" for w in self.warnings]
        return "\n".join(lines)


# -- the model of the landmarks ------------------------------------------------------------------------------------


class _Model:
    """The head as the fit sees it: per vertex and per landmark, the base position and the offsets per coefficient
    and per expression."""

    def __init__(self, model, head, hs: HeadSpace, space, options: FitOptions):
        prim = model.primitives[hs.primitive]
        self.points = np.asarray(prim.positions, np.float64)
        self.triangles = np.asarray(prim.indices, np.int64).reshape(-1, 3)
        self.normals = np.asarray(prim.normals, np.float64)
        self.hs = hs
        self.head = head
        self.weight = np.asarray(head.reliability, np.float64) * (landmark_trust() if options.trust else 1.0)
        self.space_triangles = space.triangles if space is not None and options.space else None
        self.extra = 0  # coefficients before the space's: the uniform scale (with the space)
        if space is not None and options.space:
            self.modes = np.concatenate([_scale_mode(model, hs)[None], hs.modes(space).astype(np.float64)])
            self.extra = 1
            names = list(space.expression_names)
            keep = [k for k, n in enumerate(names) if not n.startswith("eyeLook")] if options.expressions else []
            self.expression_names = [names[k] for k in keep]
            self.expression_modes = hs.expressions(space)[keep].astype(np.float64) if keep else \
                np.zeros((0, len(self.points), 3))
            self.alpha = np.r_[0.0, hs.alpha.astype(np.float64)]
        else:
            self.modes = np.zeros((0, len(self.points), 3))
            self.expression_names = []
            self.expression_modes = np.zeros((0, len(self.points), 3))
            self.alpha = np.zeros(0)
        tri = self.triangles[head.triangle]  # (478, 3)
        self.mark_vertices, self.mark_bary = tri, np.asarray(head.barycentric, np.float64)
        self.mark_base = np.einsum("lc,lcd->ld", self.mark_bary, self.points[tri])
        self.mark_modes = np.einsum("lc,klcd->kld", self.mark_bary, self.modes[:, tri])
        self.mark_expr = np.einsum("lc,klcd->kld", self.mark_bary, self.expression_modes[:, tri])
        self.mark_normals = np.einsum("lc,lcd->ld", self.mark_bary, self.normals[tri])
        self.mark_normals /= np.maximum(np.linalg.norm(self.mark_normals, axis=1, keepdims=True), 1e-12)
        self.marches = _marching_lines(self, hs)
        self.sized = np.array([k for k in POINTS if k not in IRISES])  # the landmarks whose spread is the size
        self.size = _spread(self.mark_base[self.sized])  # the avatar's

    @property
    def k(self) -> int:
        return len(self.modes)

    @property
    def e(self) -> int:
        return len(self.expression_modes)

    def vertex_offsets(self, delta, expression=None) -> np.ndarray:
        out = np.einsum("k,knd->nd", delta, self.modes) if len(delta) else np.zeros_like(self.points)
        if expression is not None and len(expression):
            out = out + np.einsum("e,end->nd", expression, self.expression_modes)
        return out


def _scale_mode(model, hs: HeadSpace) -> np.ndarray:
    """``(N, 3)``: the head scaled up by :data:`SCALE_STEP` about the middle between the eyes (faded at the neck,
    the eyeballs and the mouth's inside with it)."""
    points = np.asarray(model.primitives[hs.primitive].positions, np.float64)
    middle = hs.eye_centre.mean(axis=0)
    return hs.transfer(model, SCALE_STEP * hs.fade[:, None] * (points - middle)).head


def _spread(points) -> float:
    """The RMS distance of ``points`` from their middle."""
    return float(np.sqrt(np.mean(np.sum((points - points.mean(axis=0)) ** 2, axis=1))))


def landmark_trust() -> np.ndarray:
    """``(478,)`` 0-1: how faithfully MediaPipe's landmarks follow a change of a head's shape (``landmark_trust.json``:
    the share of their movement between renders of a head and of the same head changed that the change explains,
    over 1 for 0.6 and more, at least 0.1; measured on six avatars)."""
    import json
    from pathlib import Path

    data = json.loads((Path(__file__).resolve().parent / "landmark_trust.json").read_text(encoding="utf-8"))
    return np.asarray(data["trust"], np.float64)


def _marching_lines(m: _Model, hs: HeadSpace) -> dict:
    """For each landmark of the lower face oval: the skin vertices of its marching line (across the oval's tangent
    through the landmark's point, :data:`MARCH_BAND` wide, :data:`MARCH_REACH` long)."""
    skin = (hs.parts == SKIN) & (hs.fade > 0.5)
    idx = np.nonzero(skin)[0]
    oval = list(cf.FACE_OVAL)
    points = m.mark_base
    lines = {}
    for k in LOWER_OVAL:
        i = oval.index(k)
        tangent = points[oval[(i + 1) % len(oval)]] - points[oval[i - 1]]
        tangent /= max(np.linalg.norm(tangent), 1e-9)
        rel = m.points[idx] - points[k]
        across = np.abs(rel @ tangent)
        near = (across < MARCH_BAND) & (np.linalg.norm(rel, axis=1) < MARCH_REACH)
        lines[k] = idx[near] if near.any() else np.array([m.mark_vertices[k][0]])
    return lines


# -- the fit -------------------------------------------------------------------------------------------------------


def _skew(v):
    """``(..., 3, 3)`` cross-product matrices of ``v`` (``(..., 3)``)."""
    out = np.zeros(v.shape[:-1] + (3, 3))
    out[..., 0, 1], out[..., 0, 2] = -v[..., 2], v[..., 1]
    out[..., 1, 0], out[..., 1, 2] = v[..., 2], -v[..., 0]
    out[..., 2, 0], out[..., 2, 1] = -v[..., 1], v[..., 0]
    return out


def _rotation(omega) -> np.ndarray:
    angle = float(np.linalg.norm(omega))
    if angle < 1e-12:
        return np.eye(3)
    k = _skew(np.asarray(omega) / angle)
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * k @ k


def _project(x_cam, f, centre):
    """Pixels and ``(P, 2, 3)`` derivatives by the camera-space point of points ``x_cam`` ``(P, 3)``."""
    z = np.maximum(x_cam[:, 2], 1e-6)
    u = f * x_cam[:, :2] / z[:, None] + centre
    d = np.zeros((len(x_cam), 2, 3))
    d[:, 0, 0] = f / z
    d[:, 1, 1] = f / z
    d[:, 0, 2] = -f * x_cam[:, 0] / z ** 2
    d[:, 1, 2] = -f * x_cam[:, 1] / z ** 2
    return u, d


class _State:
    """The unknowns: beta, per photo a rotation, a translation and expressions, per session a log focal length;
    and the size the head is held at (:func:`_size`)."""

    def __init__(self, m: _Model, photos, options: FitOptions):
        self.m = m
        self.photos = photos
        self.options = options
        self.beta = m.alpha.copy()
        self.centre = np.zeros(m.k) if options.centre == "population" else m.alpha.copy()
        self.rotations = [p.camera.R.copy() for p in photos]
        self.translations = [p.camera.t.copy() for p in photos]
        groups = max(p.group for p in photos) + 1
        self.log_focal = np.zeros(groups)
        self.focal_prior = np.zeros(groups)
        self.focal_free = np.zeros(groups, bool)
        for p in photos:
            self.log_focal[p.group] = math.log(p.camera.focal)
            self.focal_prior[p.group] = math.log(p.camera.focal)
            self.focal_free[p.group] |= not p.focal_known
        self.measured = [_mediapipe_expressions(p.blendshapes, m.expression_names) for p in photos]
        self.active = [e >= options.expression_gate for e in self.measured]  # the expressions fitted per photo
        self.expressions = [np.where(a, np.clip(e, 0.0, 1.0), 0.0) for e, a in zip(self.measured, self.active)]
        self.size = None  # the face's size held (metres: its landmarks' RMS spread), set each round by _size
        self.contours = [{} for _ in photos]  # landmark -> vertex, per photo
        self.outline = [None for _ in photos]  # per photo: (vertices, gradients, values) of the outline term

    def camera(self, i) -> PhotoCamera:
        p = self.photos[i]
        f = math.exp(self.log_focal[p.group])
        k = PhotoCamera.intrinsics(f, p.size)
        return PhotoCamera(k, self.rotations[i], self.translations[i], p.size)


def _mediapipe_expressions(shapes: dict, names) -> np.ndarray:
    """MediaPipe's blendshape scores for ICT's expression ``names`` (``browInnerUp`` and ``cheekPuff`` for both
    sides; ``_L`` is ``Left``)."""
    out = np.zeros(len(names))
    for k, name in enumerate(names):
        base, side = (name[:-2], name[-1]) if name[-2:] in ("_L", "_R") else (name, "")
        key = base + ("Left" if side == "L" else "Right" if side == "R" else "")
        out[k] = float(shapes.get(key, shapes.get(base, 0.0)))
    return out


def _columns(state: _State, free_modes: int):
    """The unknowns' column ranges: ``beta[:free_modes]``, then per photo 6 + E, then the focal lengths."""
    m = state.m
    cols = {"beta": (0, free_modes)}
    at = free_modes
    for i in range(len(state.photos)):
        cols[("pose", i)] = (at, at + 6)
        at += 6
        cols[("expr", i)] = (at, at + m.e)
        at += m.e
    cols["focal"] = (at, at + len(state.log_focal))
    at += len(state.log_focal)
    return cols, at


def _residuals(state: _State, free_modes: int, weights=None, with_jacobian: bool = True):
    """The weighted residuals (and their Jacobian) of the current state: rows of landmarks, contours, outlines,
    the size and priors. ``weights``: the robust weights per photo and landmark (``None``: from the residuals)."""
    m, options = state.m, state.options
    cols, n = _columns(state, free_modes)
    rows_r, rows_j = [], []
    delta = state.beta - m.alpha
    marks = m.mark_base + (np.einsum("k,kld->ld", delta, m.mark_modes) if m.k else 0.0)
    robust = []
    for i, photo in enumerate(state.photos):
        e = state.expressions[i]
        x = marks + (np.einsum("e,eld->ld", e, m.mark_expr) if m.e else 0.0)
        r, t = state.rotations[i], state.translations[i]
        f = math.exp(state.log_focal[photo.group])
        centre = np.array([(photo.size[0] - 1) / 2.0, (photo.size[1] - 1) / 2.0])
        if photo.points is None:  # a profile placed by its outline: the outline alone
            robust.append(None)
            got = _outline_rows(state, m, i, delta, e, r, t, f, centre, cols, n, free_modes, with_jacobian)
            if got is not None:
                rows_r.append(got[0])
                if with_jacobian:
                    rows_j.append(got[1])
            continue
        xc = x @ r.T + t
        u, du = _project(xc, f, centre)
        sigma = max(options.landmark_noise * max(photo.eye_distance, 5.0), 0.5)
        to_camera = -r.T @ t - x
        facing = np.einsum("lk,lk->l", m.mark_normals, to_camera) / np.maximum(np.linalg.norm(to_camera, axis=1),
                                                                               1e-9)
        visible = mesh.smoothstep((facing - FACING_LOW) / (FACING_HIGH - FACING_LOW))
        w = np.zeros(cf.COUNT)
        w[list(POINTS)] = (m.weight[list(POINTS)] * visible[list(POINTS)])
        res = (u - photo.points) / sigma
        if weights is None:
            cauchy = 1.0 / (1.0 + (np.linalg.norm(res, axis=1) / ROBUST) ** 2)
        else:
            cauchy = weights[i]
        robust.append(cauchy)
        sw = np.sqrt(w * cauchy)
        use = sw > 1e-4
        rows_r.append((res[use] * sw[use, None]).ravel())
        if with_jacobian:
            jac = np.zeros((int(use.sum()), 2, n))
            d = du[use] / sigma * sw[use, None, None]  # (L, 2, 3)
            dr = np.einsum("lij,jk->lik", d, r)  # by the point in character space
            if free_modes:
                a, b = cols["beta"]
                jac[:, :, a:b] = np.einsum("lid,kld->lik", dr, m.mark_modes[:free_modes, use])
            a, b = cols[("pose", i)]
            jac[:, :, a:a + 3] = -np.einsum("lij,ljk->lik", d, _skew(x[use] @ r.T))
            jac[:, :, a + 3:b] = d
            if m.e:
                a, b = cols[("expr", i)]
                jac[:, :, a:b] = np.einsum("lid,eld->lie", dr, m.mark_expr[:, use])
            if state.focal_free[photo.group]:
                a, _ = cols["focal"]
                jac[:, :, a + photo.group] = (u[use] - centre) / sigma * sw[use, None]
            rows_j.append(jac.reshape(-1, n))
        # the jaw line and cheeks: across the head's outline, at the vertex outermost along each marching line
        if options.contour and state.contours[i]:
            ks = list(state.contours[i])
            verts = np.array([state.contours[i][k] for k in ks])
            xv = m.points[verts] + m.vertex_offsets(delta, e)[verts]
            uv, duv = _project(xv @ r.T + t, f, centre)
            normal2d = _outward(photo.points, ks)
            cw = np.ones(len(ks))
            res_c = np.einsum("ld,ld->l", uv - photo.points[ks], normal2d) / sigma
            cauchy_c = 1.0 / (1.0 + (res_c / ROBUST) ** 2)
            swc = np.sqrt(cw * cauchy_c)
            rows_r.append(res_c * swc)
            if with_jacobian:
                jac = np.zeros((len(ks), n))
                d = np.einsum("ld,ldk->lk", normal2d, duv) / sigma * swc[:, None]  # (C, 3)
                dr = d @ r
                if free_modes:
                    a, b = cols["beta"]
                    jac[:, a:b] = np.einsum("ld,kld->lk", dr, m.modes[:free_modes, verts])
                a, b = cols[("pose", i)]
                jac[:, a:a + 3] = -np.einsum("lj,ljk->lk", d, _skew(xv @ r.T))
                jac[:, a + 3:b] = d
                if m.e:
                    a, b = cols[("expr", i)]
                    jac[:, a:b] = np.einsum("ld,eld->le", dr, m.expression_modes[:, verts])
                if state.focal_free[photo.group]:
                    a, _ = cols["focal"]
                    jac[:, a + photo.group] = np.einsum("ld,ld->l", uv - centre, normal2d) / sigma * swc
                rows_j.append(jac)
        # the outline against the photo's face-skin mask
        got = _outline_rows(state, m, i, delta, e, r, t, f, centre, cols, n, free_modes, with_jacobian)
        if got is not None:
            rows_r.append(got[0])
            if with_jacobian:
                rows_j.append(got[1])
    # priors
    if free_modes:
        a, b = cols["beta"]
        weight = np.full(free_modes, math.sqrt(options.prior))
        weight[m.extra + options.first_modes:] = math.sqrt(options.prior_high)
        weight[:m.extra] = 1e-3  # (the uniform scale: none to speak of; the size is held)
        rows_r.append(weight * (state.beta[:free_modes] - state.centre[:free_modes]))
        if with_jacobian:
            jac = np.zeros((free_modes, n))
            jac[np.arange(free_modes), a + np.arange(free_modes)] = weight
            rows_j.append(jac)
    for i in range(len(state.photos)):
        if not m.e:
            continue
        a, b = cols[("expr", i)]
        e = state.expressions[i]
        active = state.active[i]
        pull = np.where(active, 1.0 / EXPRESSION_PULL, 0.0)
        neutral = np.where(active, 1.0 / EXPRESSION_NEUTRAL, 1.0 / EXPRESSION_HELD)
        rows_r.append(np.r_[(e - state.measured[i]) * pull, e * neutral])
        if with_jacobian:
            jac = np.zeros((2 * m.e, n))
            jac[np.arange(m.e), a + np.arange(m.e)] = pull
            jac[m.e + np.arange(m.e), a + np.arange(m.e)] = neutral
            rows_j.append(jac)
    a, _ = cols["focal"]
    for g in range(len(state.log_focal)):
        if state.focal_free[g]:
            rows_r.append(np.array([(state.log_focal[g] - state.focal_prior[g]) / FOCAL_SPREAD]))
            if with_jacobian:
                jac = np.zeros((1, n))
                jac[0, a + g] = 1.0 / FOCAL_SPREAD
                rows_j.append(jac)
    if free_modes and state.size is not None:  # the size held at the target
        rows = _size_rows(state, free_modes, cols, n, with_jacobian)
        rows_r.append(rows[0])
        if with_jacobian:
            rows_j.append(rows[1])
    r = np.concatenate(rows_r)
    j = np.concatenate(rows_j) if with_jacobian else None
    return r, j, robust


def _outline_rows(state, m, i, delta, e, r, t, f, centre, cols, n, free_modes, with_jacobian):
    outline = state.outline[i]
    if not state.options.outline or outline is None:
        return None
    from .outline import outline_rows

    return outline_rows(outline, m, state, i, delta, e, r, t, f, centre, 1.0, cols, n, free_modes, with_jacobian)


def _size_rows(state, free_modes, cols, n, with_jacobian):
    m = state.m
    delta = state.beta - m.alpha
    marks = m.mark_base[m.sized] + np.einsum("k,kld->ld", delta[:free_modes], m.mark_modes[:free_modes, m.sized])
    centred = marks - marks.mean(axis=0)
    size = float(np.sqrt(np.mean(np.sum(centred ** 2, axis=1))))
    r = np.array([(size - state.size) / SIZE_HOLD])
    if not with_jacobian:
        return r, None
    jac = np.zeros((1, n))
    a, b = cols["beta"]
    jac[0, a:b] = np.einsum("ld,kld->k", centred, m.mark_modes[:free_modes, m.sized]) / (len(marks) * size) / SIZE_HOLD
    return r, jac


def _size(state: _State) -> tuple[float, str, float]:
    """The face's size to hold the head at (its landmarks' RMS spread, metres), what it came from in words, and the
    person's iris radius at that size (``nan`` without irises): see the module docs. The shape's own proportions
    (from the current coefficients) turn the eye distance the irises measure into a size."""
    m = state.m
    delta = state.beta - m.alpha
    marks = m.mark_base + (np.einsum("k,kld->ld", delta, m.mark_modes) if m.k else 0.0)
    size = _spread(marks[m.sized])
    pupils = float(np.linalg.norm(marks[cf.LEFT_IRIS[0]] - marks[cf.RIGHT_IRIS[0]])) * m.size / size  # at the avatar's
    ratios = [_iris_radius(p) / p.eye_distance for i, p in enumerate(state.photos)
              if state.options.iris and _iris_usable(p, state.camera(i)) and p.eye_distance > 0]
    if not ratios or pupils <= 0:
        return m.size, "the avatar's own size", math.nan
    ratio = float(np.mean(ratios))
    measured = IRIS_RADIUS / ratio / pupils  # the irises alone: the head's size against the avatar's
    spread = math.hypot(IRIS_SPREAD / IRIS_RADIUS, IRIS_NOISE / math.sqrt(len(ratios)))
    w_own, w_measured = SIZE_SPREAD ** -2, spread ** -2
    scale = math.exp(w_measured * math.log(measured) / (w_own + w_measured))
    words = (f"the irises ({len(ratios)} photos; the head {100 * scale:.0f} % of the avatar's size, the irises alone "
             f"{100 * measured:.0f} %)")
    return m.size * scale, words, ratio * pupils * scale


def _outward(points, ks) -> np.ndarray:
    """``(C, 2)`` unit 2D normals of the face oval at landmarks ``ks`` in a photo, pointing out of the face."""
    oval = list(cf.FACE_OVAL)
    centre = points[list(cf.FACE_OVAL)].mean(axis=0)
    out = []
    for k in ks:
        i = oval.index(k)
        tangent = points[oval[(i + 1) % len(oval)]] - points[oval[i - 1]]
        normal = np.array([tangent[1], -tangent[0]])
        normal /= max(np.linalg.norm(normal), 1e-9)
        if normal @ (points[k] - centre) < 0:
            normal = -normal
        out.append(normal)
    return np.array(out)


def _iris_radius(photo: Photo) -> float:
    """Half the mean horizontal diameter of the irises in pixels (rim points 469/471 and 474/476)."""
    p = photo.points
    return 0.25 * (np.linalg.norm(p[469] - p[471]) + np.linalg.norm(p[474] - p[476]))


def _iris_usable(photo: Photo, camera: PhotoCamera) -> bool:
    if photo.points is None:
        return False
    yaw = camera.angles()[0]
    shut = max(photo.blendshapes.get("eyeBlinkLeft", 0.0), photo.blendshapes.get("eyeBlinkRight", 0.0))
    return abs(yaw) < IRIS_YAW and shut < 0.5 and _iris_radius(photo) >= IRIS_MIN


def _assign_contours(state: _State) -> None:
    """Each lower-oval landmark of each photo to the vertex of the current head outermost along its marching line in
    that photo (the head's outline there, seen from the camera)."""
    m = state.m
    delta = state.beta - m.alpha
    for i, photo in enumerate(state.photos):
        if photo.points is None:
            continue
        camera = state.camera(i)
        x = m.points + m.vertex_offsets(delta, state.expressions[i])
        normal2d = _outward(photo.points, LOWER_OVAL)
        found = {}
        for k, n2d in zip(LOWER_OVAL, normal2d):
            verts = m.marches[k]
            if len(verts) == 0:
                continue
            u = camera.project(x[verts])
            found[k] = int(verts[int(np.argmax(u @ n2d))])
        state.contours[i] = found


def _solve(state: _State, free_modes: int, iterations: int) -> float:
    """Levenberg-Marquardt on the state; returns the final cost."""
    cols, _n = _columns(state, free_modes)
    r, j, robust = _residuals(state, free_modes)
    cost = float(r @ r)
    lam = 1e-3
    for _ in range(iterations):
        jtj = j.T @ j
        g = j.T @ r
        diag = np.maximum(np.diag(jtj), 1e-9)
        improved = False
        for _try in range(6):
            try:
                step = -np.linalg.solve(jtj + lam * np.diag(diag), g)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue
            saved = _snapshot(state)
            _apply(state, step, cols, free_modes)
            r2, _j2, _rb = _residuals(state, free_modes, weights=robust, with_jacobian=False)
            cost2 = float(r2 @ r2)
            if cost2 < cost:
                lam = max(lam / 3.0, 1e-7)
                improved = True
                break
            _restore(state, saved)
            lam *= 10.0
        if not improved:
            break
        r, j, robust = _residuals(state, free_modes)
        new_cost = float(r @ r)
        if abs(cost - new_cost) < 1e-5 * max(cost, 1e-9):
            cost = new_cost
            break
        cost = new_cost
    return cost


def _snapshot(state: _State):
    return (state.beta.copy(), [r.copy() for r in state.rotations], [t.copy() for t in state.translations],
            [e.copy() for e in state.expressions], state.log_focal.copy())


def _restore(state: _State, saved) -> None:
    (state.beta, state.rotations, state.translations, state.expressions, state.log_focal) = saved


def _apply(state: _State, step, cols, free_modes: int) -> None:
    a, b = cols["beta"]
    if free_modes:
        state.beta = state.beta.copy()
        state.beta[:free_modes] += step[a:b]
    for i in range(len(state.photos)):
        a, b = cols[("pose", i)]
        omega, shift = step[a:a + 3], step[a + 3:b]
        state.rotations[i] = _rotation(omega) @ state.rotations[i]  # (the turn is about the camera: x_cam = R x + t)
        state.translations[i] = state.translations[i] + shift
        if state.m.e:
            a, b = cols[("expr", i)]
            state.expressions[i] = np.clip(state.expressions[i] + step[a:b], 0.0, 1.0)
    a, _ = cols["focal"]
    state.log_focal = state.log_focal + np.where(state.focal_free, step[a:a + len(state.log_focal)], 0.0)


def _robust_error(state: _State, offsets=None) -> float:
    """The mean robust landmark error (in landmark noises) over the photos: the Cauchy loss of the points' residuals,
    weighted like the fit's; with ``offsets`` (``(N, 3)``) the head displaced by them instead of the coefficients."""
    m, options = state.m, state.options
    if offsets is not None:
        marks = np.einsum("lc,lcd->ld", m.mark_bary, (m.points + offsets)[m.mark_vertices])
    else:
        delta = state.beta - m.alpha
        marks = m.mark_base + (np.einsum("k,kld->ld", delta, m.mark_modes) if m.k else 0.0)
    losses, weights = [], []
    for i, photo in enumerate(state.photos):
        if photo.points is None:
            continue
        x = marks + (np.einsum("e,eld->ld", state.expressions[i], m.mark_expr) if m.e else 0.0)
        camera = state.camera(i)
        u = camera.project(x)
        sigma = max(options.landmark_noise * max(photo.eye_distance, 5.0), 0.5)
        res = np.linalg.norm(u - photo.points, axis=1) / sigma
        to_camera = camera.position - x
        facing = np.einsum("lk,lk->l", m.mark_normals, to_camera) / np.linalg.norm(to_camera, axis=1)
        w = (m.weight * mesh.smoothstep((facing - FACING_LOW) / (FACING_HIGH - FACING_LOW)))[list(POINTS)]
        loss = np.log1p((res[list(POINTS)] / ROBUST) ** 2) * ROBUST ** 2
        losses.append(loss * w)
        weights.append(w)
    return float(np.sqrt(np.sum(np.concatenate(losses)) / max(np.sum(np.concatenate(weights)), 1e-9)))


def fit_person(photos, model, head, hs: HeadSpace, space=None, options: FitOptions | None = None,
               progress=None) -> ShapeFit:
    """Fit the head's shape to ``photos`` (:class:`Photo`, e.g. :func:`photos_from_views`) of one person: see the
    module docs. ``head``: the avatar's stage-0 landmarks; ``hs``: its :class:`~.head.HeadSpace`; ``space``: the
    :class:`~.space.IdentitySpace` (``None``, or ``options.space = False``: the free-form fallback)."""
    options = options or FitOptions()
    timings = {}
    start = time.perf_counter()
    photos = list(photos)
    faces = sum(p.points is not None for p in photos)
    if faces == 0 or len(photos) < options.min_photos:
        what = "no photo shows" if not faces else f"only {len(photos)} photo shows"
        return ShapeFit(False, f"{what} a usable face; fitting the shape needs {options.min_photos} or more "
                        "(a frontal one, and turned ones, best with profiles)", photos=len(photos))
    use_space = space is not None and options.space
    m = _Model(model, head, hs, space if use_space else None, options)
    state = _State(m, photos, options)
    timings["setup"] = time.perf_counter() - start
    start = time.perf_counter()
    # the cameras and expressions on the avatar's own head first
    if options.contour:
        _assign_contours(state)
    _solve(state, 0, options.iterations)
    before = _robust_error(state)
    scale_from, iris_radius = "the avatar's own size", math.nan
    timings["cameras"] = time.perf_counter() - start
    start = time.perf_counter()
    warnings = []
    if use_space:
        if options.outline:
            from .outline import prepare_outlines

            prepare_outlines(state)
        stages = [min(m.extra + options.first_modes, m.k), m.k]
        for free_modes in stages:
            for _round in range(options.rounds):
                state.size, scale_from, iris_radius = _size(state)
                if options.contour:
                    _assign_contours(state)
                if options.outline:
                    from .outline import assign_outlines

                    assign_outlines(state)
                _solve(state, free_modes, options.iterations)
    timings["shape"] = time.perf_counter() - start
    after = _robust_error(state)
    delta = state.beta - m.alpha
    model_offsets = m.vertex_offsets(delta) if use_space else np.zeros_like(m.points)
    beta = state.beta[m.extra:]  # the identity coefficients (the uniform scale apart)
    mahalanobis = float(np.sqrt(np.mean(beta ** 2))) if use_space else math.nan
    scale = 1.0  # the face's size against the avatar's (the uniform scale's coefficient also undoes what the
    if use_space:  # identity modes do to the size: often 20 % and more)
        scale = _spread((m.mark_base + np.einsum("k,kld->ld", delta, m.mark_modes))[m.sized]) / m.size
    free = None
    start = time.perf_counter()
    if options.free or not use_space:
        free = free_form(state, model_offsets, FALLBACK_SMOOTH if not use_space else FREE_SMOOTH, options.free_cap)
        after = _robust_error(state, model_offsets + free)
    timings["free-form"] = time.perf_counter() - start
    offsets = model_offsets + (free if free is not None else 0.0)
    fit = ShapeFit(True, beta=beta.copy(), offsets=offsets, model_offsets=model_offsets, free=free,
                   cameras=[state.camera(i) for i in range(len(photos))],
                   expressions=[{n: round(float(v), 3) for n, v in zip(m.expression_names, e) if v > 0.05}
                                for e in state.expressions],
                   focal=[float(math.exp(state.log_focal[p.group])) for p in photos],
                   scale_from=scale_from, scale=scale, iris_radius=iris_radius, error_before=before,
                   error_after=after,
                   photos=len(photos), mahalanobis=mahalanobis,
                   method=("space + free-form" if free is not None else "space") if use_space else "free-form",
                   warnings=warnings, timings=timings)
    if not np.isfinite(offsets).all():
        return ShapeFit(False, "the fit did not converge (not a number)", photos=len(photos), timings=timings)
    if after > before * (1.0 - MIN_GAIN) or before - after < MIN_GAIN_ABSOLUTE:
        fit.usable = False
        fit.reason = (f"the fitted head explains the photos no better than the avatar's own ({before:.2f} -> "
                      f"{after:.2f})")
    elif use_space and mahalanobis > MAX_MAHALANOBIS:
        fit.usable = False
        fit.reason = f"the fitted coefficients are implausible ({mahalanobis:.1f} sd per mode)"
    return fit


# -- the free-form correction --------------------------------------------------------------------------------------


def free_form(state: _State, offsets, smooth: float, cap: float) -> np.ndarray:
    """The smooth, capped correction of the skin (``(N, 3)``) that removes what the photos agree on beyond the
    head displaced by ``offsets`` (see the module docs)."""
    m = state.m
    hs = m.hs
    x = m.points + offsets
    marks = np.einsum("lc,lcd->ld", m.mark_bary, x[m.mark_vertices])
    # per landmark: its 3D correction triangulated from the photos where it is seen squarely
    normal_eq = np.zeros((cf.COUNT, 3, 3))
    rhs = np.zeros((cf.COUNT, 3))
    seen = np.zeros(cf.COUNT, int)
    residual2 = np.zeros(cf.COUNT)
    weight_sum = np.zeros(cf.COUNT)
    per_photo = []
    for i, photo in enumerate(state.photos):
        if photo.points is None:
            continue
        camera = state.camera(i)
        e = state.expressions[i]
        xi = marks + (np.einsum("e,eld->ld", e, m.mark_expr) if m.e else 0.0)
        xc = camera.to_camera(xi)
        u, du = _project(xc, camera.focal, camera.K[:2, 2])
        sigma = max(state.options.landmark_noise * max(photo.eye_distance, 5.0), 0.5)
        to_camera = camera.position - xi
        facing = np.einsum("lk,lk->l", m.mark_normals, to_camera) / np.linalg.norm(to_camera, axis=1)
        w = m.weight * mesh.smoothstep((facing - FACING_HIGH) / 0.3)
        res = (photo.points - u) / sigma
        w = w / (1.0 + (np.linalg.norm(res, axis=1) / ROBUST) ** 2)
        a = np.einsum("lij,jk->lik", du / sigma, camera.R)  # (L, 2, 3)
        normal_eq += w[:, None, None] * np.einsum("lij,lik->ljk", a, a)
        rhs += w[:, None] * np.einsum("lij,li->lj", a, res)
        seen += (w > 0.1).astype(int)
        per_photo.append((w, a, res))
    ridge = 0.05 * np.eye(3) * np.trace(normal_eq, axis1=1, axis2=2)[:, None, None] / 3.0 + 1e-9 * np.eye(3)
    corrections = np.linalg.solve(normal_eq + ridge, rhs[..., None])[..., 0]  # metres
    for w, a, res in per_photo:
        left = res - np.einsum("lij,lj->li", a, corrections)
        residual2 += w * (left ** 2).sum(axis=1)
        weight_sum += w
    agree = np.exp(-residual2 / np.maximum(weight_sum, 1e-9) / 2.0)  # 1: the photos agree on what is left
    trust = (seen >= 2) & np.isin(np.arange(cf.COUNT), POINTS) & (hs.parts[m.mark_vertices] == SKIN).all(axis=1)
    lw = np.where(trust, np.minimum(weight_sum, 4.0) / 4.0 * agree, 0.0)
    if not (lw > 0).any():
        return np.zeros_like(m.points)
    # spread over the skin: a smooth field through the corrections
    node, count = mesh.weld(m.points)
    skin_tri = m.triangles[(hs.parts[m.triangles] == SKIN).all(axis=1)]
    lap = mesh.laplacian(node[skin_tri], count)
    bilap = lap.T @ lap
    rows = np.zeros((cf.COUNT, count))
    for k in np.nonzero(lw > 0)[0]:
        np.add.at(rows[k], node[m.mark_vertices[k]], m.mark_bary[k])
    fade = np.zeros(count)
    np.maximum.at(fade, node, np.where(hs.parts == SKIN, hs.fade, 0.0))
    hold = np.diag(1e-4 + 10.0 * (1.0 - fade))
    a = (rows.T * lw) @ rows + smooth * bilap + hold
    b = (rows.T * lw) @ corrections
    field_nodes = np.linalg.solve(a, b)
    out = field_nodes[node] * np.where(hs.parts == SKIN, hs.fade, 0.0)[:, None]
    length = np.linalg.norm(out, axis=1, keepdims=True)
    scale = np.where(length > 0, cap * np.tanh(length / cap) / np.maximum(length, 1e-12), 1.0)
    return out * scale
