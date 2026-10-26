"""Stage 1: each photo registered to the avatar's head model, a :class:`View`.

**Photos with a face.** The Face Landmarker finds the person's 478 landmarks in the photo; stage 0 has the same 478
on the avatar's head (:class:`~.landmarks.HeadLandmarks`). The photo's camera is the one that makes the avatar's
landmarks fall on the person's: ``cv2.solvePnPRansac`` (EPnP) finds which landmarks agree within 6 % of the face's
size (the others are where the person's face differs from the avatar's, or MediaPipe guessed), then SQPnP and a
Levenberg-Marquardt refinement fit the camera to those, and the fit is repeated without the landmarks on the far side
of the face (MediaPipe only guesses where those are). The camera is in the avatar's bind-pose character space;
:meth:`~.camera.PhotoCamera.angles` gives the head's yaw, pitch and roll.

The focal length comes from the photo's EXIF data when it has a 35 mm equivalent and was not cropped. Otherwise it is
searched for (the reprojection error for views from 8 to 90 degrees across the photo's longer side, refined by a
golden-section search) with a firm preference for a 50 degree view: MediaPipe's landmarks tell the perspective
poorly, and a face off the middle of the picture or turned away pulls the search to too short a focal length, which
the pose then makes up for. On renders with known cameras ("fake photos") the preference cut the worst angle errors
from 17 to 7 degrees (90th percentile: from 11 to 4). Photos of one session share one camera: :func:`register_photos`
gives those without EXIF data the focal length of the most frontal one.

The person's face is not the avatar's, so the projected avatar landmarks miss the person's by a few pixels (more at
the jaw line and the mouth). A thin-plate spline (:class:`~.warp.ThinPlateWarp`) takes the avatar's projected
landmarks onto the person's, smoothly, and fades out to no movement on a ring around the face, so the hair and the
background stay where they are. It is a backward map: ``view.photo_pixels(points)`` projects 3D points on the head
with the camera and moves them through the warp, so the colour a later stage reads for the avatar's eye corner is the
person's eye corner, and the person's eyes, brows, nose and mouth move with the avatar's face bones. Only landmarks on
the side of the face turned to the camera, that the camera fit trusts, steer the warp.

**Photos without a face** (the side or the back of the head, where MediaPipe finds nothing, or a face that cannot be
what the caller named). The caller names the view (``"left"``: the photo shows the left side of the head,
``"right"``, ``"back"``) and the camera is placed there, at the distance of a photo with a face from the middle of
the head (a person turning in front of a camera, or a photographer walking round them, keeps that distance) and with
its focal length, then refined by fitting the outline of the avatar's head and hair to the photo's head mask (hair,
face skin and "others" from the segmenter, which takes blond hair seen from behind for an accessory; or a mask the
caller gives) above the chin: a pattern search over the turn and the tilts (within 30 degrees of the named view, with
a pull towards it, stronger for the angles out of the picture's plane, which the outline hardly shows), the position
and, without a photo with a face, the distance, for the best overlap (IoU). Below an IoU of 0.8 (0.9 without the
distance) the photo is reported unusable, and angles the overlap hardly changes with are named in a warning.

Checked on renders of four avatars (their own hair, so the outlines agree: the best case): side views came within
6 degrees of turn and 3 of nod, back views within 8 degrees of turn but up to 13 of nod; the segmenter labelled one
child's ponytail as clothes from behind, and those photos were rejected. With a real person's hair, which is never
the avatar's, it is only as good as the hair outlines agree.

Every :class:`View` carries what the later stages need: the image, the camera, the warp, the segmentation masks, the
landmarks, a quality score from 0 to 1 and warnings (open mouth, closed eyes, low resolution, blur, a large
reprojection error, an unknown focal length, a pose MediaPipe may get wrong), or ``usable = False`` with the reason.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import canonical as cf
from . import raycast
from .camera import PhotoCamera, head_rotation
from .detect import FaceMarks
from .warp import ThinPlateWarp

VIEW_YAWS = {"front": 0.0, "right": 90.0, "left": -90.0, "back": 180.0}  # the side of the head the photo shows
MIN_RELIABILITY = 0.4  # landmarks less reliable on the mesh stay out of the camera fit
RANSAC_THRESHOLD = 0.06  # of the face's size: landmarks further from the avatar's projected ones are outliers
MIN_INLIERS = 12
HIDDEN = 0.1  # landmarks where the surface faces the camera less than this (cosine) are hidden in the photo
FOV_RANGE = (8.0, 90.0)  # degrees across the photo's longer side: the focal lengths searched
FOV_PRIOR = 50.0  # ... and the one preferred
PRIOR_SPREAD = 0.5  # how firmly (log focal length): this far off costs as much as doubling the squared error
MIN_NOISE = 0.3  # pixels: the reprojection error never counts as smaller than this
WARP_LIMIT = 0.2  # of the face's size: landmarks the warp would move further stay out of it
WARP_SPACING = 0.025  # of the face's size: control points closer than this to one already taken are skipped
WARP_RING = 1.7  # the warp is pinned on a ring this many face radii around the face's centre
WARP_FACING = 0.15  # landmarks steer the warp where the surface faces the camera at least this much (cosine)
SILHOUETTE_SIZE = 160  # pixels: the height of the head in the silhouette fit
SILHOUETTE_MIN_IOU = 0.8  # the outline fit's overlap a photo without a face needs to be usable
SILHOUETTE_MIN_IOU_FREE = 0.9  # ... when no photo with a face gives the distance (the scale is free then)
OUTLINE_LIMIT = 30.0  # degrees: the outline fit keeps the turn and the tilts this close to the named view
OUTLINE_PRIOR_IN, OUTLINE_PRIOR_OUT = 0.01, 0.05  # overlap given up for being 30 degrees off the named view in
# the tilt in the picture's plane, and in an angle out of it
OUTLINE_FLAT = 0.01  # an angle 12 degrees off that costs less overlap than this is not pinned down by the outline
MOUTH_OPEN, EYES_CLOSED = 0.35, 0.6  # MediaPipe blendshape scores that earn a warning
GOOD_FACE_WIDTH = 300  # pixels across the face oval for full marks on resolution
SHARP = 40.0  # the variance of the Laplacian over the face for full marks on sharpness


@dataclass
class CameraFit:
    """The camera found for a photo's landmarks."""

    camera: PhotoCamera
    inliers: np.ndarray  # (478,) bool: landmarks the fit agrees with
    errors: np.ndarray  # (478,) pixels between each photo landmark and the avatar's projected one
    error: float  # RMS of the inliers' errors, pixels
    focal_known: bool  # the focal length was given (not searched for)


@dataclass
class View:
    """One photo registered to a head model (see the module docs)."""

    image: np.ndarray  # BGR, as given
    name: str  # "front", "left", "right", "back", "three-quarter left", ...
    usable: bool
    reason: str = ""  # why it is not usable
    camera: PhotoCamera | None = None  # character space (bind pose) -> photo pixels
    method: str = ""  # "landmarks" or "silhouette"
    marks: FaceMarks | None = None  # the person's landmarks (None without a face)
    projected: np.ndarray | None = None  # (478, 2) the avatar's landmarks seen by the camera
    inliers: np.ndarray | None = None  # (478,) landmarks the camera fit agrees with
    error: float = math.nan  # RMS reprojection error of the inliers, pixels
    warp: ThinPlateWarp = field(default_factory=ThinPlateWarp.identity)  # avatar-projected -> photo pixels
    masks: dict | None = None  # segmentation confidences (see detect.SEGMENT_CLASSES), each (H, W) float32
    quality: float = 0.0  # 0-1
    warnings: list = field(default_factory=list)
    focal_known: bool = False
    iou: float = math.nan  # the silhouette fit's overlap (photos without a face)
    source: str = ""  # the photo's file, if it came from one

    @property
    def size(self) -> tuple[int, int]:
        return self.image.shape[1], self.image.shape[0]

    @property
    def angles(self) -> tuple[float, float, float]:
        """``(yaw, pitch, roll)`` of the head in the photo (degrees; see :mod:`.camera`)."""
        return self.camera.angles() if self.camera is not None else (math.nan,) * 3

    def photo_pixels(self, points) -> np.ndarray:
        """Where in the photo to read the colour of 3D ``points`` on the avatar's head (bind pose, character space):
        their projection, moved by the warp."""
        return self.warp(self.camera.project(points))

    def warped(self) -> np.ndarray:
        """The photo with the person's features moved onto the avatar's projected ones."""
        return self.warp.apply(self.image)

    def summary(self) -> str:
        if not self.usable:
            return f"{self.name}: not usable: {self.reason}"
        yaw, pitch, roll = self.angles
        fit = (f"reprojection error {self.error:.1f} px" if self.method == "landmarks"
               else f"outline overlap {self.iou:.2f}")
        return (f"{self.name}: yaw {yaw:+.0f}, pitch {pitch:+.0f}, roll {roll:+.0f} degrees, {fit}, focal "
                f"{self.camera.focal:.0f} px ({'known' if self.focal_known else 'searched'}), "
                f"quality {self.quality:.2f}" + "".join(f"; {w}" for w in self.warnings))


# -- the camera ----------------------------------------------------------------------------------------------------


def face_size(points: np.ndarray) -> float:
    """A face's size in pixels: the square root of its landmarks' bounding-box area."""
    lo, hi = points.min(axis=0), points.max(axis=0)
    return float(math.sqrt(max((hi[0] - lo[0]) * (hi[1] - lo[1]), 1.0)))


def focal_for_fov(fov: float, size) -> float:
    """Focal length in pixels of a ``fov`` degree view across the longer side of an image of ``size``."""
    return 0.5 * max(size) / math.tan(math.radians(fov) / 2.0)


def exif_focal(path, size) -> float | None:
    """The focal length in pixels from a photo's EXIF data (its 35 mm equivalent), or ``None`` when it has none or
    the photo was cropped (its shape no longer matches the camera's)."""
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        with Image.open(path) as image:
            exif = image.getexif()
            sub = exif.get_ifd(0x8769)  # the Exif IFD
    except (OSError, ValueError, SyntaxError):
        return None
    f35 = sub.get(0xA405)  # FocalLengthIn35mmFilm
    if not f35:
        return None
    w, h = sub.get(0xA002), sub.get(0xA003)  # the image size the camera wrote
    if w and h and abs(max(w, h) / min(w, h) - max(size) / min(size)) > 0.01:
        return None  # cropped: the 35 mm equivalent no longer applies
    return float(f35) / 43.27 * math.hypot(*size)  # the 35 mm frame's diagonal is 43.27 mm


def _pnp(obj, img, focal, size, guess=None):
    k = PhotoCamera.intrinsics(focal, size)
    if guess is None:
        ok, rvec, tvec = cv2.solvePnP(obj, img, k, None, flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            return None
    else:
        rvec, tvec = (np.array(v, np.float64).reshape(3, 1) for v in guess)
    rvec, tvec = cv2.solvePnPRefineLM(obj, img, k, None, rvec, tvec)
    projected, _ = cv2.projectPoints(obj, rvec, tvec, k, None)
    rms = float(np.sqrt(((projected.reshape(-1, 2) - img) ** 2).sum(axis=1).mean()))
    return rms, rvec, tvec


def _camera(rvec, tvec, focal, size) -> PhotoCamera:
    return PhotoCamera(PhotoCamera.intrinsics(focal, size), cv2.Rodrigues(rvec)[0], tvec.ravel(), size)


def solve_camera(image_points, object_points, size, *, focal: float | None = None, weights=None) -> CameraFit | None:
    """The camera that projects ``object_points`` (``(N, 3)``, character space) onto ``image_points`` (``(N, 2)``)
    in an image of ``size``, robust to outliers; the focal length (pixels) is searched when not given. ``weights``
    (``(N,)`` reliabilities) leaves points under :data:`MIN_RELIABILITY` out. ``None`` when no camera fits."""
    known = focal is not None
    img_all = np.asarray(image_points, np.float64).reshape(-1, 2)
    obj_all = np.asarray(object_points, np.float64).reshape(-1, 3)
    use = np.ones(len(img_all), bool) if weights is None else np.asarray(weights) >= MIN_RELIABILITY
    if use.sum() < MIN_INLIERS:
        use[:] = True
    obj, img = obj_all[use], img_all[use]
    threshold = RANSAC_THRESHOLD * face_size(img)
    prior = focal_for_fov(FOV_PRIOR, size)
    start = focal or prior
    ok, _rvec, _tvec, inl = cv2.solvePnPRansac(obj, img, PhotoCamera.intrinsics(start, size), None,
                                             flags=cv2.SOLVEPNP_EPNP, reprojectionError=threshold,
                                             iterationsCount=300, confidence=0.999)
    if not ok or inl is None or len(inl) < MIN_INLIERS:
        return None
    inside = np.zeros(len(obj), bool)
    inside[inl.ravel()] = True
    o, i = obj[inside], img[inside]
    if focal is None:
        focal = _search_focal(o, i, size, prior)
        if focal is None:
            return None
    fitted = _pnp(o, i, focal, size)
    if fitted is None:
        return None
    # with the final camera: every landmark that agrees counts, and the camera is refitted to them
    for _ in range(2):
        camera = _camera(fitted[1], fitted[2], focal, size)
        errors = np.linalg.norm(camera.project(obj) - img, axis=1)
        inside = errors <= threshold
        if inside.sum() < MIN_INLIERS:
            return None
        fitted = _pnp(obj[inside], img[inside], focal, size, guess=fitted[1:])
    camera = _camera(fitted[1], fitted[2], focal, size)
    if (camera.depth(obj) <= 0).any():
        return None
    errors_all = np.linalg.norm(camera.project(obj_all) - img_all, axis=1)
    inliers = np.zeros(len(img_all), bool)
    inliers[np.nonzero(use)[0][inside]] = True
    return CameraFit(camera, inliers, errors_all, float(np.sqrt((errors_all[inliers] ** 2).mean())), known)


def _search_focal(obj, img, size, prior) -> float | None:
    """The focal length that best explains the landmarks, with a preference for ``prior``: the squared reprojection
    error in units of its smallest value (the landmarks' own noise) plus the squared distance from ``prior`` in log
    focal length over :data:`PRIOR_SPREAD`. Near-exact landmarks outweigh the preference; MediaPipe's, whose error
    curve is shallow and biased, defer to it."""

    def rms2(log_f):
        fitted = _pnp(obj, img, math.exp(log_f), size)
        return math.inf if fitted is None else fitted[0] ** 2

    grid = np.linspace(math.log(focal_for_fov(FOV_RANGE[1], size)), math.log(focal_for_fov(FOV_RANGE[0], size)), 24)
    errors = np.array([rms2(g) for g in grid])
    if not np.isfinite(errors).any():
        return None
    noise = max(float(errors.min()), MIN_NOISE ** 2)

    def cost(log_f, error=None):
        error = rms2(log_f) if error is None else error
        return error / noise + ((log_f - math.log(prior)) / PRIOR_SPREAD) ** 2

    costs = [cost(g, e) for g, e in zip(grid, errors)]
    k = int(np.argmin(costs))
    lo, hi = grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]
    golden = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = hi - golden * (hi - lo), lo + golden * (hi - lo)
    ca, cb = cost(a), cost(b)
    for _ in range(14):
        if ca < cb:
            hi, b, cb = b, a, ca
            a = hi - golden * (hi - lo)
            ca = cost(a)
        else:
            lo, a, ca = a, b, cb
            b = lo + golden * (hi - lo)
            cb = cost(b)
    return math.exp(a if ca < cb else b)


# -- the warp ------------------------------------------------------------------------------------------------------


def facing(head, model, camera: PhotoCamera) -> np.ndarray:
    """How squarely the surface at each of ``head``'s landmarks faces ``camera`` (cosine; negative: turned away, so
    hidden in the photo)."""
    prim = model.primitives[head.primitive]
    corners = raycast.corners_of(prim.positions, prim.indices)[head.triangle]
    normals = raycast.triangle_normals(corners)
    to_camera = camera.position - head.points
    return np.einsum("nk,nk->n", normals, to_camera) / np.linalg.norm(to_camera, axis=1)


def face_warp(projected, found, usable, size_px) -> ThinPlateWarp:
    """The thin-plate warp from the avatar's projected landmarks to the person's (``(478, 2)`` each), steered by the
    ``usable`` ones (``(478,)`` bool), fading out to no movement on a ring around the face."""
    moved = np.linalg.norm(found - projected, axis=1)
    usable = usable & (moved <= WARP_LIMIT * size_px)
    features = set(cf.FEATURES)
    order = [k for k in cf.FEATURES if usable[k]] + [k for k in range(cf.COUNT) if usable[k] and k not in features]
    taken: list[int] = []
    spacing = WARP_SPACING * size_px
    for k in order:  # the features first, then the rest, each far enough from those taken
        if not taken or np.linalg.norm(projected[taken] - projected[k], axis=1).min() >= spacing:
            taken.append(k)
    if len(taken) < 3:
        return ThinPlateWarp.identity()
    source, target = projected[taken], found[taken]
    centre = source.mean(axis=0)
    face = float(np.linalg.norm(projected - centre, axis=1).max())
    angles = np.linspace(0.0, 2.0 * math.pi, 24, endpoint=False)
    ring = centre + WARP_RING * face * np.c_[np.cos(angles), np.sin(angles)]
    inner = max(float(np.linalg.norm(source - centre, axis=1).max()), face) * 1.05
    return ThinPlateWarp(np.r_[source, ring], np.r_[target, ring], falloff=(centre, inner, WARP_RING * face))


# -- photos without a face: the head's outline ------------------------------------------------------------------


@dataclass
class Outline:
    """The triangles that make the outline of an avatar's head: the head as far as the head bone moves it (the neck
    below is body skin to the segmenter) and the hair, without the see-through parts of the hair cards."""

    triangles: np.ndarray  # (T, 3, 3) character space
    normals: np.ndarray  # (T, 3) outward; zero for the hair cards (seen from both sides)

    @classmethod
    def of(cls, model, primitive: int) -> Outline:
        from ...scene.textures import as_texture

        skeleton = model.skeleton
        bone = skeleton.find("Head")
        moved_by_head = np.zeros(len(skeleton), bool)
        if bone is not None:
            moved_by_head[skeleton.descendants(bone)] = True
        triangles, normals = [], []
        for k, prim in enumerate(model.primitives):
            name = prim.name.lower()
            if k != primitive and not any(w in name for w in ("hair", "opacity")):
                continue
            tri = np.asarray(prim.indices, np.int64).reshape(-1, 3)
            if bone is not None:
                joints, weights = np.asarray(prim.joints, np.int64), np.asarray(prim.weights, np.float64)
                head_share = (weights * moved_by_head[np.clip(joints, 0, len(skeleton) - 1)]).sum(axis=1)
                tri = tri[(head_share[tri] >= 0.8).all(axis=1)]
            if k != primitive:
                try:
                    alpha = as_texture(prim.texture).rgba[..., 3] if prim.texture is not None else None
                except (OSError, ValueError):
                    alpha = None
                if alpha is not None:
                    uv = np.asarray(prim.uvs, np.float64)[tri].mean(axis=1)
                    h, w = alpha.shape
                    rows = np.clip(((1.0 - uv[:, 1]) * h).astype(int), 0, h - 1)
                    cols = np.clip((uv[:, 0] * w).astype(int), 0, w - 1)
                    tri = tri[alpha[rows, cols] >= 128]
            triangles.append(np.asarray(prim.positions, np.float64)[tri])
            if k == primitive:  # a closed surface: only the side facing the camera makes its outline
                n = np.asarray(prim.normals, np.float64)[tri].sum(axis=1)
                normals.append(n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12))
            else:
                normals.append(np.zeros((len(tri), 3)))
        return cls(np.concatenate(triangles), np.concatenate(normals))

    def mask(self, camera: PhotoCamera, shape) -> np.ndarray:
        """The outline as ``camera`` sees it: ``(H, W)`` uint8, 1 inside. The triangles are filled one by one
        (``cv2.fillPoly`` would fill them as one even-odd shape, with holes where they overlap)."""
        points = self.triangles.reshape(-1, 3)
        pixels = camera.project(points).reshape(-1, 3, 2)
        depth = camera.depth(points).reshape(-1, 3)
        to_camera = camera.position - self.triangles.mean(axis=1)
        h, w = shape
        lo, hi = pixels.min(axis=1), pixels.max(axis=1)
        keep = ((hi[:, 0] >= 0) & (hi[:, 1] >= 0) & (lo[:, 0] < w) & (lo[:, 1] < h) & (depth > 0).all(axis=1)
                & (np.einsum("tk,tk->t", self.normals, to_camera) >= 0))
        mask = np.zeros(shape, np.uint8)
        for tri in np.round(pixels[keep] * 4).astype(np.int32):
            cv2.fillConvexPoly(mask, tri, 1, lineType=cv2.LINE_8, shift=2)
        return mask


def _above(camera: PhotoCamera, point, shape) -> np.ndarray:
    """The part of the image above the horizontal plane through ``point`` (character space), as the camera sees it:
    the pixels on the same side of the plane's image as the head's top."""
    right = camera.R[0].copy()
    right[1] = 0.0
    right = right / max(np.linalg.norm(right), 1e-9) * 0.1
    a, b = camera.project(np.array([point - right, point + right]))
    up = camera.project(np.array([point + np.array([0.0, 0.1, 0.0])]))[0]
    h, w = shape
    gy, gx = np.mgrid[0:h, 0:w]
    side = (b[0] - a[0]) * (gy - a[1]) - (b[1] - a[1]) * (gx - a[0])
    up_side = (b[0] - a[0]) * (up[1] - a[1]) - (b[1] - a[1]) * (up[0] - a[0])
    return (side * np.sign(up_side)) > 0


@dataclass
class OutlineFit:
    """The camera found from the head's outline."""

    camera: PhotoCamera | None
    iou: float
    uncertain: list  # the angles ("turn", "tilt", "lean") the outline hardly changes with: left near the named view


def fit_silhouette(model, primitive: int, chin, centre, mask: np.ndarray, yaw: float, focal: float,
                   distance: float | None = None) -> OutlineFit:
    """The camera of a photo without a face from the head's outline (see the module docs): ``mask`` is the photo's
    head (hair and face, 0-1), ``yaw`` the named view's, ``centre`` and ``chin`` points on the avatar's head
    (character space), ``distance`` the camera's from ``centre`` if known."""
    h, w = mask.shape
    count, labels, stats, _centres = cv2.connectedComponentsWithStats((mask > 0.5).astype(np.uint8), connectivity=8)
    if count < 2:
        return OutlineFit(None, 0.0, [])
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))  # the head; stray blobs elsewhere are mistakes
    mask = (labels == biggest).astype(np.float32)
    ys, xs = np.nonzero(mask > 0.5)
    if len(ys) < 50:
        return OutlineFit(None, 0.0, [])
    # work on a crop around the head, scaled so the head is about SILHOUETTE_SIZE pixels tall
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    scale = SILHOUETTE_SIZE / max(y1 - y0, x1 - x0, 1)
    margin = int(0.4 * max(y1 - y0, x1 - x0))
    cx0, cy0 = max(0, x0 - margin), max(0, y0 - margin)
    cx1, cy1 = min(w, x1 + margin + 1), min(h, y1 + margin + 1)
    crop = cv2.resize(mask[cy0:cy1, cx0:cx1].astype(np.float32), None, fx=scale, fy=scale,
                      interpolation=cv2.INTER_AREA) > 0.5
    shape = crop.shape
    small_size = (shape[1], shape[0])
    k = np.array([[focal * scale, 0.0, ((w - 1) / 2.0 - cx0 + 0.5) * scale - 0.5],
                  [0.0, focal * scale, ((h - 1) / 2.0 - cy0 + 0.5) * scale - 0.5], [0.0, 0.0, 1.0]])
    outline = Outline.of(model, primitive)
    chin, centre = np.asarray(chin, np.float64), np.asarray(centre, np.float64)

    def camera_of(p):
        dyaw, dpitch, roll, u, v, depth = p
        r = np.diag([1.0, -1.0, -1.0]) @ head_rotation(yaw + dyaw, dpitch, roll)
        position = np.array([(u - k[0, 2]) * depth / k[0, 0], (v - k[1, 2]) * depth / k[1, 1], depth])
        return PhotoCamera(k, r, position - r @ centre, small_size)

    def overlap(p):
        if p[5] <= 0.05:
            return 0.0
        camera = camera_of(p)
        region = _above(camera, chin, shape)
        avatar = outline.mask(camera, shape).astype(bool) & region
        photo = crop & region
        union = (avatar | photo).sum()
        return float((avatar & photo).sum() / union) if union else 0.0

    # the overlap, with a pull towards the named view for what the outline leaves open: faint for the tilt in the
    # picture's plane (the nod seen from the side, the lean seen from behind), stronger for the turns out of it
    prior = np.full(3, OUTLINE_PRIOR_OUT)
    prior[1 if abs(math.sin(math.radians(yaw))) > 0.7 else 2] = OUTLINE_PRIOR_IN

    def score(p):
        return overlap(p) - float((prior * (p[:3] / 30.0) ** 2).sum())

    # start: the head's top and middle where the mask's are, at the given distance or the one matching its width
    top = np.nonzero(crop.any(axis=1))[0][0]
    cols = np.nonzero(crop.any(axis=0))[0]
    width = cols[-1] - cols[0] + 1
    seen = outline.mask(camera_of((0.0, 0.0, 0.0, shape[1] / 2, shape[0] / 2, 1.0)), shape)
    seen_cols = np.nonzero(seen.any(axis=0))[0]
    seen_width = (seen_cols[-1] - seen_cols[0] + 1) if len(seen_cols) else shape[1] / 2
    depth0 = distance if distance else max(0.1, seen_width / max(width, 1))
    steps = np.array([8.0, 6.0, 6.0, 0.05 * shape[1], 0.05 * shape[0], 0.06 * depth0])
    if distance:
        steps[5] = 0.0
    starts = []
    for dyaw in (-16.0, 0.0, 16.0):
        p = np.array([dyaw, 0.0, 0.0, (cols[0] + cols[-1]) / 2.0, shape[0] / 2.0, depth0])
        rows = np.nonzero(outline.mask(camera_of(p), shape).any(axis=1))[0]
        if len(rows):
            p[4] += top - rows[0]  # the tops of the heads meet
        starts.append(_pattern_search(score, p, steps, rounds=2))
    best, _value = max(starts, key=lambda s: s[1])
    best, _value = _pattern_search(score, best, steps / 4.0, rounds=3)
    iou = overlap(best)
    uncertain = []
    for i, name in enumerate(("turn", "tilt", "lean")):  # angles the overlap hardly changes with
        drop = iou - max(overlap(best + np.eye(6)[i] * sign * 12.0) for sign in (1.0, -1.0))
        if drop < OUTLINE_FLAT:
            uncertain.append(name)
    small = camera_of(best)
    full_k = np.array([[focal, 0.0, (w - 1) / 2.0], [0.0, focal, (h - 1) / 2.0], [0.0, 0.0, 1.0]])
    return OutlineFit(PhotoCamera(full_k, small.R, small.t, (w, h)), iou, uncertain)


def _pattern_search(f, p, steps, rounds: int = 4):
    """Maximise ``f`` from ``p`` by trying each parameter a step up and down, halving the steps when nothing helps;
    the angles (the first three) stay within :data:`OUTLINE_LIMIT` degrees of the named view."""
    p = p.astype(np.float64).copy()
    value = f(p)
    steps = steps.astype(np.float64).copy()
    for _ in range(rounds):
        improved = True
        while improved:
            improved = False
            for i in range(len(p)):
                if steps[i] == 0.0:
                    continue
                for sign in (1.0, -1.0):
                    q = p.copy()
                    q[i] += sign * steps[i]
                    if i < 3 and abs(q[i]) > OUTLINE_LIMIT:
                        continue
                    v = f(q)
                    if v > value + 1e-6:
                        p, value, improved = q, v, True
                        break
        steps *= 0.5
    return p, value


# -- registering photos --------------------------------------------------------------------------------------------


def _load(photo):
    if isinstance(photo, np.ndarray):
        return photo, ""
    image = cv2.imread(str(photo), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"cannot read {photo}")
    return image, str(photo)


def _name_for(yaw: float) -> str:
    side = "left" if yaw < 0 else "right"  # the side of the head it shows
    if abs(yaw) < 25:
        return "front"
    if abs(yaw) < 65:
        return f"three-quarter {side}"
    return side if abs(yaw) < 135 else "back"


def _sharpness(image, marks) -> float:
    """The variance of the Laplacian over the face, at most 256 pixels wide (a small face is not enlarged, which
    would blur it)."""
    x, y, w, h = (round(v) for v in marks.box)
    crop = image[max(0, y):y + h, max(0, x):x + w]
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    if gray.shape[1] > 256:
        gray = cv2.resize(gray, (256, max(1, round(256 * gray.shape[0] / gray.shape[1]))), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def register_photo(photo, head, model, *, view: str | None = None, landmarker=None, segmenter=None, masks=None,
             focal: float | None = None, reference=None, faces=None) -> View:
    """Register ``photo`` (a path or a BGR image) to the head model ``model`` with its stage-0 landmarks ``head``.

    ``view``: what the photo shows (``"front"``, ``"left"``, ``"right"``, ``"back"``), needed for photos without a
    face. ``landmarker``: a Face Landmarker (default: MediaPipe's); ``faces``: the faces it found already (then it is
    not run). ``segmenter``: a :class:`~.detect.Segmenter` for the masks, or ``masks`` given directly (at least
    ``"hair"`` and ``"face"``, each (H, W) 0-1). ``focal``: the focal length in pixels if known. ``reference``: a
    registered :class:`View` of the same session whose focal length and distance a photo without a face takes."""
    image, source = _load(photo)
    size = (image.shape[1], image.shape[0])
    if view is not None and view not in VIEW_YAWS:
        raise ValueError(f"unknown view {view!r} (one of {', '.join(VIEW_YAWS)})")
    focal_known = focal is not None
    if focal is None and source:
        focal = exif_focal(source, size)
        focal_known = focal is not None
    if masks is None and segmenter is not None:
        masks = segmenter.segment(image)
    if faces is None:
        if landmarker is None:
            from .detect import Landmarker

            landmarker = Landmarker()
        faces = landmarker.detect(image)
    warnings: list[str] = []
    if len(faces) > 1:
        warnings.append(f"{len(faces)} faces: the biggest is used")
    if faces:
        result = _register_face(image, faces[0], head, model, size, focal, focal_known, view, list(warnings))
        if view in ("left", "right", "back") and masks is not None and _contradicts(result, view):
            yaw, _pitch, roll = result.angles
            warnings.append(f"the face MediaPipe found (yaw {yaw:+.0f}, roll {roll:+.0f}) does not fit a photo of the "
                            f"{view}: ignored")
            result = _register_outline(image, head, model, size, focal, focal_known, view, masks, reference,
                                       warnings)
    else:
        result = _register_outline(image, head, model, size, focal, focal_known, view, masks, reference, warnings)
    result.masks, result.source = masks, source
    return result


def _contradicts(view: View, name: str) -> bool:
    """Whether a registration by landmarks cannot be a photo of ``name``: MediaPipe sometimes "finds" a face in a
    profile or the back of a head, turned or upside down."""
    if not view.usable:
        return True
    yaw, _pitch, roll = view.angles
    off = abs((yaw - VIEW_YAWS[name] + 180.0) % 360.0 - 180.0)
    return off > 70.0 or abs(roll) > 45.0


def _register_face(image, marks, head, model, size, focal, focal_known, view, warnings) -> View:
    fit = solve_camera(marks.points, head.points, size, focal=focal, weights=head.reliability)
    if fit is not None:  # again without the landmarks on the far side of the face: MediaPipe only guesses those
        seen = facing(head, model, fit.camera) >= HIDDEN
        if (~seen & (head.reliability >= MIN_RELIABILITY)).sum() > 10:
            refit = solve_camera(marks.points, head.points, size, focal=focal, weights=head.reliability * seen)
            fit = refit or fit
    if fit is None:
        return View(image, view or "front", False, "the avatar's landmarks cannot be lined up with the photo's",
                    marks=marks, warnings=warnings)
    camera = fit.camera
    yaw = camera.angles()[0]
    name = view or _name_for(yaw)
    if view is not None and view != "front" and VIEW_YAWS[view] * yaw < 0 and abs(yaw) > 20:
        warnings.append(f"named {view!r} but the head is turned the other way (yaw {yaw:+.0f})")
    if view == "front" and abs(yaw) > 45:
        warnings.append(f"named 'front' but the head is turned {yaw:+.0f} degrees")
    projected = camera.project(head.points)
    size_px = face_size(marks.points)
    seen = facing(head, model, camera) >= WARP_FACING
    steer = (fit.inliers | (head.reliability >= MIN_RELIABILITY)) & seen
    warp = face_warp(projected, marks.points, steer, size_px)
    # the quality: resolution, how well the avatar's landmarks fit, how many agree, sharpness, expression, pose
    oval = projected[list(cf.FACE_OVAL)]
    width = float(oval[:, 0].max() - oval[:, 0].min())
    q_resolution = min(1.0, width / GOOD_FACE_WIDTH)
    relative = fit.error / size_px
    q_fit = math.exp(-((relative / 0.03) ** 2))
    used = head.reliability >= MIN_RELIABILITY
    q_agree = min(1.0, fit.inliers[used].sum() / max(used.sum(), 1) / 0.8)
    sharp = _sharpness(image, marks)
    q_sharp = min(1.0, max(0.3, sharp / SHARP))
    q_other = 1.0
    shapes = marks.blendshapes or {}
    if shapes.get("jawOpen", 0.0) > MOUTH_OPEN:
        warnings.append("the mouth is open (its inside would be painted on the lips)")
        q_other *= 0.7
    if max(shapes.get("eyeBlinkLeft", 0.0), shapes.get("eyeBlinkRight", 0.0)) > EYES_CLOSED:
        warnings.append("the eyes are closed")
        q_other *= 0.7
    if width < 0.5 * GOOD_FACE_WIDTH:
        warnings.append(f"low resolution: the face is {width:.0f} pixels wide")
    if q_sharp < 0.6:
        warnings.append("the face looks blurred")
    if relative > 0.04:
        warnings.append(f"large reprojection error ({fit.error:.1f} px, {100 * relative:.1f} % of the face)")
    if not focal_known:
        warnings.append("no EXIF focal length: searched for it")
    if abs(yaw) > 60:
        warnings.append("seen from the side: MediaPipe guesses the hidden half of the face and its pose may be "
                        "off by 10-15 degrees")
        q_other *= 0.6
    quality = q_resolution * q_fit * q_agree * q_sharp * q_other
    return View(image, name, True, camera=camera, method="landmarks", marks=marks, projected=projected,
                inliers=fit.inliers, error=fit.error, warp=warp, quality=float(quality), warnings=warnings,
                focal_known=focal_known)


def _register_outline(image, head, model, size, focal, focal_known, view, masks, reference, warnings) -> View:
    name = view or "unknown"
    if view is None or view == "front":
        reason = "no face found" + ("" if view else " (name the view, 'left', 'right' or 'back', for a photo of the "
                                    "side or the back of the head)")
        return View(image, name, False, reason, warnings=warnings)
    if masks is None or "hair" not in masks or "face" not in masks:
        return View(image, name, False, "no face found and no head mask to fit the head's outline to (use the "
                    "segmenter)", warnings=warnings)
    from .landmarks import face_frame

    frame = face_frame(model)
    distance = None
    if reference is not None and reference.usable and reference.camera is not None:
        distance = float(np.linalg.norm(reference.camera.position - frame.pivot))  # the same distance from the head
        if focal is None:
            focal = reference.camera.focal * max(size) / max(reference.size)
            focal_known = reference.focal_known
        warnings.append(f"the distance{'' if focal_known else ' and focal length'} of the {reference.name} photo")
    elif focal is None:
        focal = focal_for_fov(FOV_PRIOR, size)
        warnings.append("no EXIF focal length and no photo with a face: a focal length was assumed")
    # the head: hair and face skin, and "others" (the segmenter takes blond hair seen from behind for an accessory)
    head_mask = np.clip(masks["hair"] + masks["face"] + masks.get("others", 0.0), 0.0, 1.0)
    fit = fit_silhouette(model, head.primitive, frame.chin, frame.pivot, head_mask, VIEW_YAWS[view], focal,
                         distance)
    camera, iou = fit.camera, fit.iou
    if camera is None or iou < (SILHOUETTE_MIN_IOU if distance else SILHOUETTE_MIN_IOU_FREE):
        return View(image, name, False, f"the head's outline does not fit the photo's head (overlap {iou:.2f})",
                    camera=camera, method="silhouette", iou=iou, warnings=warnings, focal_known=focal_known)
    quality = (iou - SILHOUETTE_MIN_IOU) / (1.0 - SILHOUETTE_MIN_IOU) * 0.6 * 0.75 ** len(fit.uncertain)
    if distance is None:
        quality *= 0.7
    warnings.append("placed by the head's outline (no landmarks)")
    if fit.uncertain:
        warnings.append(f"the outline hardly fixes the {' and the '.join(fit.uncertain)}: kept near the named view")
    return View(image, name, True, camera=camera, method="silhouette", projected=camera.project(head.points),
                quality=float(quality), warnings=warnings, focal_known=focal_known, iou=iou)


def register_photos(photos, head, model, *, views=None, focals=None, landmarker=None, segmenter=None) -> list[View]:
    """Register several photos of one person (one session, one camera). The most frontal photo with a face is the
    reference: photos of the same size without a focal length (given, or from EXIF data) take its focal length (a
    turned face tells the focal length poorly), and photos without a face its distance (and focal length).
    ``views``: a view name (or ``None``) per photo; ``focals``: a focal length in pixels (or ``None``) per photo."""
    views = list(views) if views is not None else [None] * len(photos)
    focals = list(focals) if focals is not None else [None] * len(photos)
    if landmarker is None:
        from .detect import Landmarker

        landmarker = Landmarker()
    found = [landmarker.detect(_load(photo)[0]) for photo in photos]
    results: list[View | None] = [None] * len(photos)
    for k, (photo, faces) in enumerate(zip(photos, found)):
        if faces or views[k] in (None, "front"):
            results[k] = register_photo(photo, head, model, view=views[k], focal=focals[k], segmenter=segmenter,
                                        faces=faces)
    with_faces = [v for v in results if v is not None and v.usable and v.method == "landmarks"]
    reference = min(with_faces, key=lambda v: abs(v.angles[0])) if with_faces else None
    for k, view in enumerate(results):
        if view is None:  # no face: placed by the outline, at the reference's distance
            results[k] = register_photo(photos[k], head, model, view=views[k], focal=focals[k], segmenter=segmenter,
                                        faces=[], reference=reference)
        elif (reference is not None and view is not reference and view.method == "landmarks" and view.usable
              and not view.focal_known and not reference.focal_known and view.size == reference.size):
            again = register_photo(photos[k], head, model, view=views[k], masks=view.masks, faces=found[k],
                                   focal=reference.camera.focal)
            again.focal_known = False
            again.warnings = [w for w in again.warnings if "focal" not in w]
            again.warnings.append(f"the focal length of the {reference.name} photo")
            results[k] = again
    return results


def draw_registration(view: View, model, head, wireframe: bool = True, arrows: bool = False) -> np.ndarray:
    """The photo with what the registration found drawn over it: the avatar's head mesh as the camera sees it (its
    triangles facing the camera, through the warp), the avatar's projected landmarks on the side of the head the
    camera sees (green: agreeing with the camera fit, red: not) and the person's (small white dots). ``arrows``: the
    warp instead of the dots, as arrows from the avatar's projected features to the person's, three times as long."""
    out = view.image.copy()
    if view.camera is None:
        return out
    shown = facing(head, model, view.camera) > 0  # the landmarks on the side of the head the camera sees
    size_px = face_size(view.projected[shown]) if view.projected is not None and shown.any() else 300.0
    dot = int(np.clip(round(size_px / 110), 1, 4))
    if wireframe:
        prim = model.primitives[head.primitive]
        corners = raycast.corners_of(prim.positions, prim.indices)
        normals = raycast.triangle_normals(corners)
        to_camera = view.camera.position - corners.mean(axis=1)
        front = np.einsum("nk,nk->n", normals, to_camera) > 0
        pixels = view.photo_pixels(corners[front].reshape(-1, 3)).reshape(-1, 3, 2)
        layer = out.copy()
        cv2.polylines(layer, np.round(pixels * 4).astype(np.int32), True, (255, 200, 60), 1, cv2.LINE_AA, shift=2)
        out = cv2.addWeighted(layer, 0.45, out, 0.55, 0.0)
    if arrows and view.marks is not None and view.projected is not None:
        for k in cf.FEATURES + cf.FACE_OVAL:
            if shown[k]:
                a = view.projected[k]
                b = a + 3.0 * (view.warp(a[None])[0] - a)
                cv2.arrowedLine(out, (round(a[0]), round(a[1])), (round(b[0]), round(b[1])), (60, 230, 255), 1,
                                cv2.LINE_AA, tipLength=0.3)
        return out
    if view.marks is not None:
        for p in view.marks.points:
            cv2.circle(out, (round(p[0]), round(p[1])), max(1, dot - 1), (255, 255, 255), -1, cv2.LINE_AA)
    if view.projected is not None:
        inliers = view.inliers if view.inliers is not None else np.ones(len(view.projected), bool)
        for p, ok in zip(view.projected[shown], inliers[shown]):
            cv2.circle(out, (round(p[0]), round(p[1])), dot, (60, 220, 60) if ok else (40, 40, 230), -1, cv2.LINE_AA)
    return out
