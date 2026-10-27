"""Stage 1: each photo registered to the avatar's head model, a :class:`View`.

**Photos with a face.** The Face Landmarker finds the person's 478 landmarks in the photo; stage 0 has the same 478
on the avatar's head (:class:`~.landmarks.HeadLandmarks`). The photo's camera is the one that makes the avatar's
landmarks fall on the person's: ``cv2.solvePnPRansac`` (EPnP) finds which landmarks agree within 6 % of the face's
size (the others are where the person's face differs from the avatar's, or MediaPipe guessed), then SQPnP and a
Levenberg-Marquardt refinement fit the camera to those, and the fit is repeated without the landmarks on the far side
of the face (MediaPipe only guesses where those are). The camera is in the avatar's bind-pose character space;
:meth:`~.camera.PhotoCamera.angles` gives the head's yaw, pitch and roll.

The focal length comes from the caller or from the photo's EXIF data (a 35 mm equivalent, the photo not cropped).
Otherwise it is searched for (the reprojection error for views from 8 to 90 degrees across the photo's longer side,
refined by a golden-section search) with a firm preference for a 50 degree view: without it, a face off the middle of
the picture or turned away pulled the search to too short a focal length, which the pose then made up for (on renders
with known cameras the worst angle errors were 17 degrees instead of 7). MediaPipe's landmarks tell the perspective so
poorly that the search ends near 50 degrees whatever the lens (on portrait renders a 70 degree lens came out at about
53, a 25 degree one at about 47), so in effect a 50 degree view is assumed, and the view says so: the camera of a
wide-angle selfie comes out too far away, though its pose hardly suffers. Photos of one session share one camera:
:func:`register_photos` gives a focal length known for one photo to the others of the same picture shape, then the
most frontal photo's to those still without one (a turned face tells the focal length even worse).

The person's face is not the avatar's, so the projected avatar landmarks miss the person's by a few pixels (more at
the jaw line and the mouth). A thin-plate spline (:class:`~.warp.ThinPlateWarp`) takes the avatar's projected
landmarks onto the person's, smoothly, and fades out to no movement on a ring around the face, so the hair and the
background stay where they are. It is a backward map: ``view.photo_pixels(points)`` projects 3D points on the head
with the camera and moves them through the warp, so the colour a later stage reads for the avatar's eye corner is the
person's eye corner, and the person's eyes, brows, nose and mouth move with the avatar's face bones. Only landmarks the
camera fit trusts and the camera sees steer the warp (:func:`visible`: the surface faces the camera, by the vertex
normals interpolated over the landmark's triangle, as one triangle's own normal is too coarse, pointing sideways at
the tip of the nose; and no other part of the head, such as the nose, is in front).

**Faces that are not there.** MediaPipe makes up faces in profiles and on the backs of heads: at its default confidence,
which :class:`~.detect.Landmarker` keeps, in 16 of 220 renders of the 20 test avatars turned 75 to 180 degrees (35 at
0.3). The camera fitted to a face is not used when it leans more than 45 degrees or looks up or down more than 35, when
it is more than 25 degrees from MediaPipe's own estimate of the pose (its facial transformation matrix), or, with the
segmenter's masks, when the segmenter finds face skin in less than 30 % of the face's oval or the avatar's head is put
where the photo has none. Without masks, a face turned less than 35 degrees needs a second opinion: the face
recognition's YuNet detector must be at least 0.8 sure of a face there (on the renders, 95 % of the real faces turned up
to 45 degrees got 0.92 or more, the real test photos 0.89-0.95, the false faces on the backs of heads 0.17-0.74; YuNet
misses many faces turned 60 degrees, so it is not asked about those). With these checks no made-up face on the back of a
head was used, with masks or without. MediaPipe also finds the faces of profiles turned 50 to 70 degrees whether they
are turned 60 or 90, and an unnamed photo of a profile is registered that way, with a warning and a lower quality. In a
photo the caller names as a side or the back, the face is used only when the head's outline (below) turns the camera
within 15 degrees of it, or, without masks, when it is within 20 degrees of the named view. In a side photo whose face
is turned less than its outline (by up to 50 degrees), the truth mostly lies between the two: MediaPipe turns profiles
too little, and the outline of another person's head is pulled towards the named view. The camera is then turned half
way and fitted to the outline again; else the outline's camera is used. A named photo whose face is not used is placed
by its outline; an unnamed one is reported unusable, with the reason.

**Photos without a face** (the side or the back of the head, or a face that is not used). The caller names the view
(``"left"``: the photo shows the left side of the head, ``"right"``, ``"back"``) and the camera is placed there, at the
distance of a photo with a face from the middle of the head (a person turning in front of a camera, or a photographer
walking round them, keeps that distance) and with its focal length, then refined by fitting the outline of the
avatar's head and hair to the photo's head mask (hair, face skin and "others" from the segmenter, which takes blond
hair seen from behind for an accessory; or masks the caller gives) above the chin: a pattern search over the turn and
the tilts, the position and, without a photo with a face, the distance, for the best overlap (IoU), with a pull
towards the named view. Another person's hair never has the avatar's outline, and the outline alone hardly tells the
turn, so the fit also lines up the face skin: :func:`head_regions` finds once per head model which parts of the
avatar's head the segmenter calls face skin and hair (renders from 12 sides, segmented; about 6 s, cached like the
landmarks), and the overlap of the avatar's face skin with the photo's (blurred, so that a small patch compares
smoothly) counts as much as the head's. The turn may then go up to 45 degrees from the named view; the tilt out of the
picture's plane (the nod seen from behind, the lean from a side) stays within 20 degrees, as the skin would tip a back
view over to hide the avatar's ears where the person's hair covers theirs, the one in it within 30. Below a head IoU
of 0.8 (0.9 without the distance) the photo is reported unusable, and angles the overlap hardly changes with are named
in a warning.

Measured on renders of six avatars turned 60 to 180 degrees (14 turns), each named after the nearest side or the back
and registered to the head of another avatar (as a person's photos are): on Male_Adult_05's head (short hair) 69 of 83
came out usable, turned within 4 degrees of the truth (the median; 90 % within 15, all within 25); on
Business_Female_02's (long hair) 39 of 83, within 10 degrees (90 % within 22, the worst 41 off). The outline alone was
13 and 20 degrees off (the medians), up to 38 and 58. Nothing in a fit tells a bad one from a good one, so a view
placed by its outline gets a quality of at most 0.6, falling to 0 at an IoU of 0.8, and a warning.

Every :class:`View` carries what the later stages need: the image, the camera, the warp, the segmentation masks (small
copies: :meth:`View.mask` gives one at the photo's size), the landmarks, a quality score from 0 to 1 and warnings (open
mouth, closed eyes, low resolution, blur, a large reprojection error, an assumed focal length, a pose MediaPipe may get
wrong), or ``usable = False`` with the reason.
"""
from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import canonical as cf
from . import raycast
from .camera import PhotoCamera, head_rotation
from .detect import FaceMarks, compact_mask, mask_at
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
OCCLUSION_TOLERANCE = 0.002  # metres: a landmark is hidden when the head is hit this much nearer the camera
SILHOUETTE_SIZE = 160  # pixels: the height of the head in the silhouette fit
SILHOUETTE_MIN_IOU = 0.8  # the outline fit's overlap a photo without a face needs to be usable
SILHOUETTE_MIN_IOU_FREE = 0.9  # ... when no photo with a face gives the distance (the scale is free then)
OUTLINE_LIMIT = 30.0  # degrees: the outline fit keeps the turn and the tilts this close to the named view
OUTLINE_TURN_LIMIT = 45.0  # ... the turn this close when it also lines up the face skin (see head_regions)
OUTLINE_TILT_LIMIT = 20.0  # ... and the tilt out of the picture's plane this close then (the skin would tip it over)
OUTLINE_PRIOR_IN, OUTLINE_PRIOR_OUT = 0.01, 0.05  # overlap given up for being 30 degrees off the named view in
# the tilt in the picture's plane, and in an angle out of it
OUTLINE_PRIOR_SKIN = 0.02  # ... in an angle out of it when the face skin is lined up too (it tells the turn)
SKIN_WEIGHT = 1.0  # the face skin's overlap counts this much beside the head's in the outline fit
SKIN_BLUR, SKIN_FLOOR = 0.03, 0.02  # the skin's overlap: of blurred masks (sigma: this share of the crop's height),
# with this share of the head's area added to both parts, so a small or no patch of skin compares smoothly
OUTLINE_FLAT = 0.01  # an angle 12 degrees off that costs less overlap than this is not pinned down by the outline
SKIN, HAIR, OTHER, UNSEEN = 1, 2, 3, -1  # what the segmenter calls a triangle of the head's outline (head_regions)
REGIONS_VERSION = 1
REGION_VIEWS = tuple((float(yaw), 0.0) for yaw in range(0, 360, 45)) + ((0.0, 35.0), (180.0, 35.0), (90.0, 30.0),
                                                                         (-90.0, 30.0))  # (yaw, pitch) of renders
REGION_SIZE, REGION_DISTANCE = 512, 0.6  # pixels, metres
MAX_ROLL, MAX_PITCH = 45.0, 35.0  # degrees: a face found leaning or looking up or down more cannot be right
POSE_AGREEMENT = 25.0  # degrees: ... nor one whose camera fit is further than this from MediaPipe's own pose
NAMED_VIEW_LIMIT = 20.0  # degrees: ... nor one turned further than this from the side or back the caller named
FACE_OUTLINE_AGREEMENT = 15.0  # degrees: ... (unless the head's outline turns its camera within this of the face's)
HALF_WAY_LIMIT = 50.0  # degrees: a side's face and outline further apart than this are not met half way
SIDES = ("left", "right", "back")  # the views a face in a photo is checked by the head's outline for
MIN_FACE_SKIN = 0.3  # ... nor one whose face oval the segmenter finds less face skin in than this share
MIN_FACE_OVERLAP = 0.2  # ... nor one that puts the avatar's head where the photo's head mask is not (IoU)
CONFIRM_YAW, MIN_DETECTOR_SCORE = 35.0, 0.8  # ... nor, without masks, one turned less than this (degrees) that
# YuNet is less sure than this is a face
SIDE_YAW = 50.0  # degrees: faces turned further are seen from the side (MediaPipe's turn is unsure there)
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
    masks: dict | None = None  # segmentation confidences by class (detect.SEGMENT_CLASSES): detect.compact_mask's
    quality: float = 0.0  # 0-1
    warnings: list = field(default_factory=list)
    focal_known: bool = False  # the focal length was given or read from EXIF data (this photo's or the reference's)
    focal_from: str = ""  # where it came from: "given", "EXIF", "assumed", "the front photo's, EXIF", ...
    iou: float = math.nan  # the outline fit's overlap of the head (photos without a face)
    skin: float = math.nan  # ... and of the face skin, when it was lined up too (see head_regions)
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
        their projection, moved by the warp (:meth:`~.warp.ThinPlateWarp.on_grid`: fast for a texture's texels)."""
        return self.warp.on_grid(self.camera.project(points))

    def warped(self) -> np.ndarray:
        """The photo with the person's features moved onto the avatar's projected ones."""
        return self.warp.apply(self.image)

    def mask(self, name: str) -> np.ndarray | None:
        """The segmenter's confidence that each pixel of the photo shows ``name`` (one of
        :data:`~.detect.SEGMENT_CLASSES`: ``"hair"``, ``"face"`` skin, ``"body"`` skin, ``"clothes"``, ...):
        ``(H, W)`` float32 0-1, or ``None`` without masks."""
        if not self.masks or name not in self.masks:
            return None
        return mask_at(self.masks[name], self.size)

    def summary(self) -> str:
        if not self.usable:
            return f"{self.name}: not usable: {self.reason}"
        yaw, pitch, roll = self.angles
        fit = (f"reprojection error {self.error:.1f} px" if self.method == "landmarks"
               else f"outline overlap {self.iou:.2f}" + ("" if math.isnan(self.skin) else f", skin {self.skin:.2f}"))
        return (f"{self.name}: yaw {yaw:+.0f}, pitch {pitch:+.0f}, roll {roll:+.0f} degrees, {fit}, focal "
                f"{self.camera.focal:.0f} px ({self.focal_from or ('known' if self.focal_known else 'assumed')}), "
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
    """``(rms, rvec, tvec)`` of the camera with ``focal`` fitted to the points, or ``None``. SQPnP refuses points
    that look nearly alike from the camera (a small face at a long focal length): EPnP starts the fit then."""
    k = PhotoCamera.intrinsics(focal, size)
    try:
        if guess is None:
            try:
                ok, rvec, tvec = cv2.solvePnP(obj, img, k, None, flags=cv2.SOLVEPNP_SQPNP)
            except cv2.error:
                ok, rvec, tvec = cv2.solvePnP(obj, img, k, None, flags=cv2.SOLVEPNP_EPNP)
            if not ok:
                return None
        else:
            rvec, tvec = (np.array(v, np.float64).reshape(3, 1) for v in guess)
        rvec, tvec = cv2.solvePnPRefineLM(obj, img, k, None, rvec, tvec)
        projected, _ = cv2.projectPoints(obj, rvec, tvec, k, None)
    except cv2.error:
        return None
    rms = float(np.sqrt(((projected.reshape(-1, 2) - img) ** 2).sum(axis=1).mean()))
    return (rms, rvec, tvec) if math.isfinite(rms) else None


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
    try:
        ok, _rvec, _tvec, inl = cv2.solvePnPRansac(obj, img, PhotoCamera.intrinsics(start, size), None,
                                                 flags=cv2.SOLVEPNP_EPNP, reprojectionError=threshold,
                                                 iterationsCount=300, confidence=0.999)
    except cv2.error:  # degenerate points
        return None
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


def surface_normals(head, model) -> np.ndarray:
    """``(478, 3)`` unit normals of the head's surface at ``head``'s landmarks: the vertex normals interpolated over
    each landmark's triangle (a triangle's own normal is too coarse: at the tip of the nose it points sideways)."""
    prim = model.primitives[head.primitive]
    tri = np.asarray(prim.indices, np.int64).reshape(-1, 3)[head.triangle]
    normals = np.einsum("lc,lcd->ld", head.barycentric, np.asarray(prim.normals, np.float64)[tri])
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    flat = raycast.triangle_normals(raycast.corners_of(prim.positions, prim.indices)[head.triangle])
    return np.where(length > 1e-6, normals / np.maximum(length, 1e-12), flat)  # (the flat one where they cancel)


def facing(head, model, camera: PhotoCamera) -> np.ndarray:
    """How squarely the surface at each of ``head``'s landmarks faces ``camera``: the cosine between its normal
    (:func:`surface_normals`) and the direction to the camera (negative: turned away, so hidden in the photo)."""
    to_camera = camera.position - head.points
    return np.einsum("nk,nk->n", surface_normals(head, model), to_camera) / np.linalg.norm(to_camera, axis=1)


def visible(head, model, camera: PhotoCamera, at_least: float = 0.0) -> np.ndarray:
    """``(478,)`` bool: the landmarks ``camera`` sees: the surface faces it (:func:`facing` above ``at_least``) and
    no other part of the head is in front (the nose hides some of the far cheek of a turned face)."""
    seen = facing(head, model, camera) > at_least
    if seen.any():
        prim = model.primitives[head.primitive]
        corners = raycast.corners_of(prim.positions, prim.indices)
        to_camera = camera.position - corners.mean(axis=1)
        corners = corners[np.einsum("tk,tk->t", raycast.triangle_normals(corners), to_camera) > 0]  # the near side
        rays = head.points[seen] - camera.position
        distance = np.linalg.norm(rays, axis=1)
        hit, _triangle, _bary = raycast.intersect(camera.position, rays / distance[:, None], corners)
        seen[np.nonzero(seen)[0][hit < distance - OCCLUSION_TOLERANCE]] = False
    return seen


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
    labels: np.ndarray | None = None  # (T,) SKIN, HAIR, OTHER or UNSEEN, once known (see head_regions)

    @classmethod
    def of(cls, model, primitive: int) -> Outline:
        """The outline of ``model``'s head primitive ``primitive`` and hair (kept for the last few models)."""
        key = (id(model), primitive)
        hit = _outlines.get(key)
        if hit is not None and hit[0] is model:
            return hit[1]
        outline = cls._make(model, primitive)
        _outlines[key] = (model, outline)
        while len(_outlines) > 4:
            _outlines.pop(next(iter(_outlines)))
        return outline

    @classmethod
    def _make(cls, model, primitive: int) -> Outline:
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

    def _seen(self, camera: PhotoCamera, shape):
        """The triangles' pixels (``(T, 3, 2)``), depths (``(T, 3)``) and which of them ``camera`` may see."""
        points = self.triangles.reshape(-1, 3)
        pixels = camera.project(points).reshape(-1, 3, 2)
        depth = camera.depth(points).reshape(-1, 3)
        to_camera = camera.position - self.triangles.mean(axis=1)
        h, w = shape
        lo, hi = pixels.min(axis=1), pixels.max(axis=1)
        keep = ((hi[:, 0] >= 0) & (hi[:, 1] >= 0) & (lo[:, 0] < w) & (lo[:, 1] < h) & (depth > 0).all(axis=1)
                & (np.einsum("tk,tk->t", self.normals, to_camera) >= 0))
        return pixels, depth, keep

    def mask(self, camera: PhotoCamera, shape) -> np.ndarray:
        """The outline as ``camera`` sees it: ``(H, W)`` uint8, 1 inside. The triangles are filled one by one
        (``cv2.fillPoly`` would fill them as one even-odd shape, with holes where they overlap)."""
        pixels, _depth, keep = self._seen(camera, shape)
        mask = np.zeros(shape, np.uint8)
        for tri in np.round(pixels[keep] * 4).astype(np.int32):
            cv2.fillConvexPoly(mask, tri, 1, lineType=cv2.LINE_8, shift=2)
        return mask

    def labelled(self, camera: PhotoCamera, shape) -> np.ndarray:
        """What the segmenter would make of the outline ``camera`` sees: ``(H, W)`` uint8, :data:`SKIN`,
        :data:`HAIR` or :data:`OTHER` from the nearest triangle (the far ones are drawn first), 0 outside. Needs
        :attr:`labels`."""
        pixels, depth, keep = self._seen(camera, shape)
        order = np.nonzero(keep & (self.labels != UNSEEN))[0]
        order = order[np.argsort(-depth[order].mean(axis=1), kind="stable")]
        out = np.zeros(shape, np.uint8)
        for tri, label in zip(np.round(pixels[order] * 4).astype(np.int32), self.labels[order].tolist()):
            cv2.fillConvexPoly(out, tri, label, lineType=cv2.LINE_8, shift=2)
        return out


_outlines: dict[tuple, tuple] = {}


def region_views(model) -> list[PhotoCamera]:
    """The cameras of :func:`head_regions`' renders: round the head every 45 degrees, and from above."""
    from .landmarks import face_frame

    pivot = face_frame(model).pivot
    return [PhotoCamera.around(pivot, REGION_DISTANCE, yaw, pitch, fov=35.0, size=(REGION_SIZE, REGION_SIZE))
            for yaw, pitch in REGION_VIEWS]


def head_regions(model, primitive: int, segmenter=None, *, render=None, cache_dir=None,
                 refresh: bool = False) -> np.ndarray | None:
    """What MediaPipe's segmenter calls each triangle of the avatar's head :class:`Outline`: ``(T,)`` int8,
    :data:`SKIN` (face skin), :data:`HAIR` (hair, and "others", which takes in blond hair seen from behind),
    :data:`OTHER` (body skin, clothes) or :data:`UNSEEN`; kept on the outline for the outline fit, which then lines up
    the face skin too. Once per head model, like stage 0: the avatar is rendered with its own texture from
    :data:`REGION_VIEWS` (``render(model, cameras) -> images``, default the offscreen renderer), each render is
    segmented, and every triangle a render shows gets the segmenter's confidences at its middle, weighted by how
    squarely it faces the camera; the class with the most wins. Cached like the landmarks (``cache_dir``, see
    :func:`~.landmarks.cache_folder`; ``False``: memory only). ``None`` when not cached and no ``segmenter`` is
    given."""
    from .detect import SEGMENT_CLASSES
    from .landmarks import cache_folder, render_views

    outline = Outline.of(model, primitive)
    if outline.labels is not None and not refresh:
        return outline.labels
    key = hashlib.blake2b(f"headtex-regions-{REGIONS_VERSION}".encode(), digest_size=12)
    key.update(np.ascontiguousarray(outline.triangles, np.float64).tobytes())
    path = None if cache_dir is False else cache_folder(cache_dir) / f"regions-{key.hexdigest()}.npz"
    if not refresh and path is not None and path.is_file():
        try:
            with np.load(path, allow_pickle=False) as data:
                labels = np.asarray(data["labels"], np.int8)
        except Exception:  # noqa: BLE001 - a damaged or foreign file: made again
            labels = None
        if labels is not None and labels.shape == (len(outline.triangles),):
            outline.labels = labels
            return labels
    if segmenter is None:
        return None
    cameras = region_views(model)
    images = (render or render_views)(model, cameras)
    tri = outline.triangles
    centres = tri.mean(axis=1)
    cards = ~outline.normals.any(axis=1)  # hair cards, seen from both sides
    votes = np.zeros((len(tri), len(SEGMENT_CLASSES)))
    for camera, image in zip(cameras, images):
        masks = segmenter.segment(image)
        confidence = np.stack([mask_at(masks[name], camera.size) for name in SEGMENT_CLASSES], axis=-1)
        to_camera = camera.position - centres
        distance = np.linalg.norm(to_camera, axis=1)
        facing = np.einsum("tk,tk->t", outline.normals, to_camera) / distance
        candidates = np.nonzero(cards | (facing > 0.05))[0]
        hit, _triangle, _bary = raycast.intersect(camera.position, -to_camera[candidates] / distance[candidates, None],
                                                  tri)
        shown = candidates[hit >= distance[candidates] - 0.003]  # nothing nearer in the way
        pixels = np.round(camera.project(centres[shown])).astype(int)
        inside = ((pixels >= 0) & (pixels < np.array(camera.size) - 1)).all(axis=1)
        shown, pixels = shown[inside], pixels[inside]
        weight = np.where(cards[shown], 0.5, np.clip(facing[shown], 0.05, 1.0))
        votes[shown] += confidence[pixels[:, 1], pixels[:, 0]] * weight[:, None]
    best = votes.argmax(axis=1)
    hair = [SEGMENT_CLASSES.index("hair"), SEGMENT_CLASSES.index("others")]
    labels = np.where(best == SEGMENT_CLASSES.index("face"), SKIN, np.where(np.isin(best, hair), HAIR, OTHER))
    labels = np.where(votes.sum(axis=1) > 0, labels, UNSEEN).astype(np.int8)
    outline.labels = labels
    if path is not None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp.npz")
            np.savez_compressed(tmp, labels=labels)
            os.replace(tmp, path)
        except OSError:
            pass  # a read-only folder: kept in memory only
    return labels


def outline_overlap(model, primitive: int, camera: PhotoCamera, mask: np.ndarray, longest: int = 640) -> float:
    """How well the outline of the avatar's head and hair seen by ``camera`` covers the photo's head ``mask`` (0-1,
    the photo's size) above the chin: their IoU, on the photo scaled to at most ``longest`` pixels. Only the parts of
    the mask that touch the avatar's outline count (other people's heads are not this one)."""
    from .landmarks import face_frame

    h, w = mask.shape[:2]
    scale = min(1.0, longest / max(h, w))
    small = camera.scaled(scale) if scale < 1.0 else camera
    shape = (small.size[1], small.size[0])
    photo = (cv2.resize(np.asarray(mask, np.float32), small.size, interpolation=cv2.INTER_AREA) if scale < 1.0
             else np.asarray(mask)) > 0.5
    region = _above(small, face_frame(model).chin, shape)
    avatar = Outline.of(model, primitive).mask(small, shape).astype(bool) & region
    _count, labels = cv2.connectedComponents((photo & region).astype(np.uint8), connectivity=8)
    touching = np.unique(labels[avatar])
    photo = np.isin(labels, touching[touching > 0])
    union = (avatar | photo).sum()
    return float((avatar & photo).sum() / union) if union else 0.0


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
    iou: float  # the head's overlap
    uncertain: list  # the angles ("turn", "tilt", "lean") the outline hardly changes with: left near the named view
    skin: float = math.nan  # the face skin's overlap, when it was lined up too
    overlap: float = math.nan  # what the fit made best: the head's overlap, or its mean with the face skin's


def _iou(a: np.ndarray, b: np.ndarray, empty: float = 0.0) -> float:
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else empty


def _soft_iou(a: np.ndarray, b: np.ndarray, blur: float, floor: float) -> float:
    """The IoU of two masks blurred by a Gaussian (sigma ``blur`` pixels), with ``floor`` pixels added to both their
    intersection and their union: it changes smoothly as a small patch moves, shrinks or disappears."""
    a = cv2.GaussianBlur(a.astype(np.float32), (0, 0), blur)
    b = cv2.GaussianBlur(b.astype(np.float32), (0, 0), blur)
    return float((np.minimum(a, b).sum() + floor) / (np.maximum(a, b).sum() + floor))


def fit_silhouette(model, primitive: int, chin, centre, mask: np.ndarray, yaw: float, focal: float,
                   distance: float | None = None, skin: np.ndarray | None = None,
                   turn: float | None = None) -> OutlineFit:
    """The camera of a photo without a face from the head's outline (see the module docs): ``mask`` is the photo's
    head (hair and face, 0-1), ``yaw`` the named view's, ``centre`` and ``chin`` points on the avatar's head
    (character space), ``distance`` the camera's from ``centre`` if known. ``skin``: the photo's face skin (0-1);
    with the outline's labels (:func:`head_regions`) the fit lines up the avatar's face skin with it too, which tells
    the turn far better than the outline does, and lets the turn go further from the named view. ``turn``: the head's
    yaw held there (the rest fitted), to see how well a camera turned so fits."""
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
    lined_up = skin is not None and outline.labels is not None  # the face skin too
    if lined_up:
        skin_crop = cv2.resize(np.asarray(skin, np.float32)[cy0:cy1, cx0:cx1], small_size,
                               interpolation=cv2.INTER_AREA) > 0.5
        skin_crop &= crop
        blur, floor = SKIN_BLUR * shape[0], SKIN_FLOOR * float(crop.sum())
    chin, centre = np.asarray(chin, np.float64), np.asarray(centre, np.float64)

    def camera_of(p):
        dyaw, dpitch, roll, u, v, depth = p
        r = np.diag([1.0, -1.0, -1.0]) @ head_rotation(yaw + dyaw, dpitch, roll)
        position = np.array([(u - k[0, 2]) * depth / k[0, 0], (v - k[1, 2]) * depth / k[1, 1], depth])
        return PhotoCamera(k, r, position - r @ centre, small_size)

    def overlaps(p) -> tuple[float, float]:
        """The head's and the face skin's IoU above the chin."""
        if p[5] <= 0.05:
            return 0.0, 0.0
        camera = camera_of(p)
        region = _above(camera, chin, shape)
        photo = crop & region
        if not lined_up:
            return _iou(outline.mask(camera, shape).astype(bool) & region, photo), math.nan
        drawn = outline.labelled(camera, shape)
        head_iou = _iou(((drawn == SKIN) | (drawn == HAIR)) & region, photo)
        return head_iou, _soft_iou((drawn == SKIN) & region, skin_crop & region, blur, floor)

    def overlap(p) -> float:
        head_iou, skin_iou = overlaps(p)
        return (head_iou + SKIN_WEIGHT * skin_iou) / (1.0 + SKIN_WEIGHT) if lined_up else head_iou

    # the overlap, with a pull towards the named view for what the outline leaves open: faint for the tilt in the
    # picture's plane (the nod seen from the side, the lean seen from behind), stronger for the turns out of it
    # (fainter for the turn when the face skin is lined up too: it tells the turn)
    in_plane = 1 if abs(math.sin(math.radians(yaw))) > 0.7 else 2  # the nod seen from a side, the lean from behind
    prior = np.array([OUTLINE_PRIOR_SKIN if lined_up else OUTLINE_PRIOR_OUT, OUTLINE_PRIOR_OUT, OUTLINE_PRIOR_OUT])
    prior[in_plane] = OUTLINE_PRIOR_IN
    limits = np.array([OUTLINE_TURN_LIMIT, OUTLINE_TILT_LIMIT, OUTLINE_TILT_LIMIT]) if lined_up else \
        np.full(3, OUTLINE_LIMIT)
    limits[in_plane] = OUTLINE_LIMIT

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
    turns = (-16.0, 0.0, 16.0)
    if turn is not None:
        turns = ((turn - yaw + 180.0) % 360.0 - 180.0,)
        steps[0], limits[0] = 0.0, 180.0
    starts = []
    for dyaw in turns:
        p = np.array([dyaw, 0.0, 0.0, (cols[0] + cols[-1]) / 2.0, shape[0] / 2.0, depth0])
        rows = np.nonzero(outline.mask(camera_of(p), shape).any(axis=1))[0]
        if len(rows):
            p[4] += top - rows[0]  # the tops of the heads meet
        starts.append(_pattern_search(score, p, steps, limits, rounds=2))
    best, _value = max(starts, key=lambda s: s[1])
    best, _value = _pattern_search(score, best, steps / 4.0, limits, rounds=3)
    value = overlap(best)
    uncertain = []
    for i, name in enumerate(("turn", "tilt", "lean")):  # angles the overlap hardly changes with
        if steps[i] == 0.0:
            continue  # held
        drop = value - max(overlap(best + np.eye(6)[i] * sign * 12.0) for sign in (1.0, -1.0))
        if drop < OUTLINE_FLAT:
            uncertain.append(name)
    head_iou, skin_iou = overlaps(best)
    small = camera_of(best)
    full_k = np.array([[focal, 0.0, (w - 1) / 2.0], [0.0, focal, (h - 1) / 2.0], [0.0, 0.0, 1.0]])
    return OutlineFit(PhotoCamera(full_k, small.R, small.t, (w, h)), head_iou, uncertain, skin_iou, value)


def _pattern_search(f, p, steps, limits, rounds: int = 4):
    """Maximise ``f`` from ``p`` by trying each parameter a step up and down, halving the steps when nothing helps;
    the angles (the first three) stay within ``limits`` (degrees) of the named view."""
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
                    if i < 3 and abs(q[i]) > limits[i]:
                        continue
                    v = f(q)
                    if v > value + 1e-6:
                        p, value, improved = q, v, True
                        break
        steps *= 0.5
    return p, value


# -- registering photos --------------------------------------------------------------------------------------------


@dataclass
class _Photo:
    """A photo on its way to a :class:`View`."""

    image: np.ndarray
    source: str  # its file, if it came from one
    view: str | None  # what the caller says it shows
    masks: dict | None
    focal: float | None = None  # pixels, when known
    focal_known: bool = False  # given, or from EXIF data (this photo's or another one's of the session)
    focal_from: str = ""  # see View.focal_from
    shared: str = ""  # the photo of the session whose focal length it took: "the front photo", "IMG_2.jpg", ...

    @property
    def size(self) -> tuple[int, int]:
        return self.image.shape[1], self.image.shape[0]


def _load(photo):
    if isinstance(photo, np.ndarray):
        return photo, ""
    image = cv2.imread(str(photo), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"cannot read {photo}")
    return image, str(photo)


def _prepare(photo, view, focal, masks, segmenter) -> _Photo:
    if view is not None and view not in VIEW_YAWS:
        raise ValueError(f"unknown view {view!r} (one of {', '.join(VIEW_YAWS)})")
    image, source = _load(photo)
    if masks is not None:
        masks = {name: compact_mask(m) for name, m in masks.items()}
    elif segmenter is not None:
        masks = segmenter.segment(image)
    p = _Photo(image, source, view, masks)
    if focal is not None:
        p.focal, p.focal_known, p.focal_from = float(focal), True, "given"
    elif source:
        found = exif_focal(source, p.size)
        if found is not None:
            p.focal, p.focal_known, p.focal_from = found, True, "EXIF"
    return p


def _share_focal(p: _Photo, focal: float, size, known: bool, origin: str, name: str) -> None:
    """A photo without a focal length of its own takes ``focal`` (pixels) of the session's photo ``name`` (of
    ``size``) when their pictures have the same shape: the same camera, perhaps turned or scaled (a turned face tells
    the focal length poorly). ``known``: that one's was given or read from EXIF data (``origin``)."""
    if p.focal is not None:
        return
    (w, h), (rw, rh) = p.size, size
    if abs(max(w, h) / min(w, h) - max(rw, rh) / min(rw, rh)) > 0.01:
        return  # another shape of picture: another camera, or cropped
    p.focal = focal * max(w, h) / max(rw, rh)
    p.focal_known, p.shared = known, name
    p.focal_from = f"{name}'s" + (f", {origin}" if known else "")


def _share_reference_focal(p: _Photo, reference: View | None) -> None:
    if reference is not None and reference.usable and reference.camera is not None:
        _share_focal(p, reference.camera.focal, reference.size, reference.focal_known, reference.focal_from,
                     f"the {reference.name} photo")


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
    ``"hair"`` and ``"face"``, each 0-1 floats or 0-255 bytes covering the photo, at its size or smaller; kept as
    :func:`~.detect.compact_mask` copies). ``focal``: the focal length in pixels if known. ``reference``: a
    registered :class:`View` of the same session (see :func:`register_photos`): a photo without a focal length of
    its own takes the reference's, and a photo placed by its outline also its distance from the head."""
    p = _prepare(photo, view, focal, masks, segmenter)
    if faces is None:
        if landmarker is None:
            from .detect import Landmarker

            landmarker = Landmarker()
        faces = landmarker.detect(p.image)
    _share_reference_focal(p, reference)
    return _finish(p, _by_landmarks(p, faces, head, model), head, model, reference, segmenter)


def register_photos(photos, head, model, *, views=None, focals=None, landmarker=None, segmenter=None) -> list[View]:
    """Register several photos of one person (one session, one camera). A focal length given for one photo (or read
    from its EXIF data) holds for the others of the same picture shape (turned or scaled: the same camera; cropped:
    not). The most frontal photo registered by its face is the reference: the photos still without a focal length
    take its estimated one (a turned face tells the focal length poorly), and the photos without a usable face are
    placed by their outline at its distance from the head. ``views``: a view name (or ``None``) per photo;
    ``focals``: a focal length in pixels (or ``None``) per photo."""
    views = list(views) if views is not None else [None] * len(photos)
    focals = list(focals) if focals is not None else [None] * len(photos)
    if landmarker is None:
        from .detect import Landmarker

        landmarker = Landmarker()
    prepared = [_prepare(photo, v, f, None, segmenter) for photo, v, f in zip(photos, views, focals)]
    faces = [landmarker.detect(p.image) for p in prepared]
    # by the landmarks alone first (the reference comes from these, and the outline fits need the reference); the
    # photos with a focal length of their own before the others, which take it
    own = [p.focal_known for p in prepared]
    first = [_by_landmarks(p, f, head, model) if known else None for p, f, known in zip(prepared, faces, own)]
    for k, (known, found) in enumerate(zip(prepared, first)):
        if own[k]:
            name = (f"the {found.view.name} photo" if found.view is not None else f"the {known.view} photo"
                    if known.view else Path(known.source).name if known.source else f"photo {k + 1}")
            for p in prepared:
                _share_focal(p, known.focal, known.size, True, known.focal_from, name)
    first = [found or _by_landmarks(p, f, head, model) for p, f, found in zip(prepared, faces, first)]
    with_faces = [found.view for found in first if found.view is not None]
    reference = min(with_faces, key=lambda v: abs(v.angles[0])) if with_faces else None
    results = []
    for p, f, found in zip(prepared, faces, first):
        if found.view is not reference and p.focal is None:
            _share_reference_focal(p, reference)
            if p.focal is not None and f:
                found = _by_landmarks(p, f, head, model)  # again, with the reference's focal length
        results.append(_finish(p, found, head, model, reference, segmenter))
    return results


@dataclass
class _Found:
    """What the landmarks make of a photo: ``view``, its registration by the biggest face, when nothing is wrong with
    that; else ``doubt`` says what is (empty: no face). ``face``: the registration by the face also when only the
    view the caller named speaks against it. ``notes``: warnings about the photo however it is registered."""

    view: View | None
    notes: list
    doubt: str = ""
    face: View | None = None


def _by_landmarks(p: _Photo, faces, head, model) -> _Found:
    if not faces:
        return _Found(None, [])
    notes = [f"{len(faces)} faces: the biggest is used"] if len(faces) > 1 else []
    view = _register_face(p, faces[0], head, model, list(notes))
    doubt = _doubt(view, faces[0], p, head, model)
    if doubt:
        return _Found(None, notes, doubt)
    yaw, pitch, roll = view.angles
    if p.view in SIDES and _off(yaw, VIEW_YAWS[p.view]) > NAMED_VIEW_LIMIT:
        return _Found(None, notes, f"(yaw {yaw:+.0f}, pitch {pitch:+.0f}, roll {roll:+.0f}) does not show the "
                      f"{p.view} of the head", face=view)
    return _Found(view, notes, face=view)


def _finish(p: _Photo, found: _Found, head, model, reference, segmenter=None) -> View:
    if found.face is not None and p.view in SIDES and _head_masks(p):
        view = _checked_by_outline(p, found, head, model, reference, segmenter)
    else:
        view = found.view or _register_outline(p, head, model, reference, found.notes, found.doubt, segmenter)
    view.masks, view.source = p.masks, p.source
    return view


def _checked_by_outline(p: _Photo, found: _Found, head, model, reference, segmenter) -> View:
    """A face in a photo of a side or the back of the head. MediaPipe finds the faces of profiles turned 50 to 70
    degrees whether they are turned 60 or 90, and makes up faces on the backs of heads: the face is used when the
    head's outline turns the camera within :data:`FACE_OUTLINE_AGREEMENT` of the face's turn. Further off, in a
    photo of a side whose face is turned less than the outline (up to :data:`HALF_WAY_LIMIT` less), the truth mostly
    lies between the two (MediaPipe turns profiles too little; the outline of another person's head is pulled towards
    the named view): the camera is turned half way and fitted to the outline again. Else the outline's camera is
    used. When the outline does not fit the photo, a face is used if it shows about the named view."""
    face = found.face
    outline = _register_outline(p, head, model, reference, found.notes, found.doubt, segmenter)
    if not outline.usable:
        return found.view or outline
    face_yaw, outline_yaw = face.angles[0], outline.angles[0]
    gap = _off(outline_yaw, face_yaw)
    if gap <= FACE_OUTLINE_AGREEMENT:
        turned = "" if found.view is not None else f"turned {face_yaw:+.0f} degrees, not quite the {p.view}, but "
        face.warnings.append(f"{turned}the head's outline agrees (yaw {outline_yaw:+.0f})")
        return face
    named = VIEW_YAWS[p.view]
    if p.view != "back" and gap <= HALF_WAY_LIMIT and _off(face_yaw, named) > _off(outline_yaw, named):
        half = face_yaw + 0.5 * ((outline_yaw - face_yaw + 180.0) % 360.0 - 180.0)
        between = _register_outline(p, head, model, reference, found.notes, "", segmenter, turn=half)
        if between.usable:
            between.warnings.insert(len(found.notes), f"the face MediaPipe found (yaw {face_yaw:+.0f}) and the head's "
                                    f"outline (yaw {outline_yaw:+.0f}) disagree: turned half way")
            return between
    if found.view is not None:  # it had passed: say why it is not used
        outline.warnings.insert(len(found.notes), f"the face MediaPipe found (yaw {face_yaw:+.0f}) is "
                                f"{gap:.0f} degrees from where the head's outline turns it: ignored")
    return outline


def _head_masks(p: _Photo) -> bool:
    return p.masks is not None and "hair" in p.masks and "face" in p.masks


def _off(yaw: float, target: float) -> float:
    """Degrees between two turns of the head."""
    return abs((yaw - target + 180.0) % 360.0 - 180.0)


def _doubt(view: View, marks: FaceMarks, p: _Photo, head, model) -> str:
    """Why the face found cannot be what the photo shows (empty when nothing is wrong with it). MediaPipe "finds"
    faces in profiles and the backs of heads; the camera fitted to such landmarks has an absurd pose, or one that does
    not agree with MediaPipe's own estimate of the pose, or the segmenter finds hardly any face skin in the face, or
    the avatar's head is put where the photo has none, or (without masks) YuNet sees no face where a nearly frontal
    one was found. (The view the caller named is checked in _by_landmarks.)"""
    if not view.usable:
        return "cannot be lined up with the avatar's landmarks"
    yaw, pitch, roll = view.angles
    pose = f"(yaw {yaw:+.0f}, pitch {pitch:+.0f}, roll {roll:+.0f})"
    if abs(roll) > MAX_ROLL:
        return f"{pose} lies on its side or upside down"
    if abs(pitch) > MAX_PITCH:
        return f"{pose} looks further up or down than a photo of a face does"
    gap = _mediapipe_gap(view.camera, marks)
    if gap > POSE_AGREEMENT:
        return f"{pose} is {gap:.0f} degrees from MediaPipe's own estimate of its pose"
    if _head_masks(p):
        skin = _skin_share(p.masks["face"], marks, p.size)
        if skin < MIN_FACE_SKIN:
            return f"{pose} is hardly skin to the segmenter ({100 * skin:.0f} % of its oval)"
        overlap = outline_overlap(model, head.primitive, view.camera, _head_mask(p))
        if overlap < MIN_FACE_OVERLAP:
            return f"{pose} puts the avatar's head where the photo's is not (outline overlap {overlap:.2f})"
    elif marks.detector_score is not None and abs(yaw) < CONFIRM_YAW and marks.detector_score < MIN_DETECTOR_SCORE:
        return (f"{pose} is no face to YuNet, the face recognition's detector (its confidence "
                f"{marks.detector_score:.2f})")
    return ""


def _skin_share(mask, marks: FaceMarks, size) -> float:
    """The share of the face oval of ``marks`` that the segmenter's ``mask`` (face skin, any resolution) calls face
    skin: nearly all of it in a real face (58 % with a headscarf), none for a "face" on the back of a head."""
    m = np.asarray(mask)
    scale = np.array([m.shape[1] / size[0], m.shape[0] / size[1]])
    oval = np.zeros(m.shape[:2], np.uint8)
    cv2.fillPoly(oval, [np.round((marks.points[list(cf.FACE_OVAL)] + 0.5) * scale - 0.5).astype(np.int32)], 1)
    inside = mask_at(m, (m.shape[1], m.shape[0]))[oval > 0]
    return float((inside > 0.5).mean()) if inside.size else 0.0


def _mediapipe_gap(camera: PhotoCamera, marks: FaceMarks) -> float:
    """Degrees between the head's rotation found by the camera fit and MediaPipe's own (its facial transformation
    matrix, which turns its canonical face, oriented like an avatar's character space, into its camera); 0 without
    one."""
    if marks.matrix is None:
        return 0.0
    u, _s, vt = np.linalg.svd(np.asarray(marks.matrix, np.float64)[:3, :3])  # the nearest rotation
    own = u @ np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))]) @ vt
    cosine = (np.trace(camera.rotation.T @ own) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _head_mask(p: _Photo) -> np.ndarray:
    """The photo's head: hair and face skin, and "others" (the segmenter takes blond hair seen from behind for an
    accessory), 0-1 at the photo's size."""
    masks = p.masks
    return np.clip(sum(mask_at(masks[n], p.size) for n in ("hair", "face", "others") if n in masks), 0.0, 1.0)


def _register_face(p: _Photo, marks, head, model, warnings) -> View:
    fit = solve_camera(marks.points, head.points, p.size, focal=p.focal, weights=head.reliability)
    if fit is not None:  # again without the landmarks on the far side of the face: MediaPipe only guesses those
        seen = facing(head, model, fit.camera) >= HIDDEN  # (those behind the nose count: they fix the turn)
        if (~seen & (head.reliability >= MIN_RELIABILITY)).sum() > 10:
            refit = solve_camera(marks.points, head.points, p.size, focal=p.focal, weights=head.reliability * seen)
            fit = refit or fit
    if fit is None:
        return View(p.image, p.view or "front", False, "the avatar's landmarks cannot be lined up with the photo's",
                    marks=marks, warnings=warnings)
    camera = fit.camera
    yaw = camera.angles()[0]
    name = p.view or _name_for(yaw)
    if p.view is not None and p.view != "front" and VIEW_YAWS[p.view] * yaw < 0 and abs(yaw) > 20:
        warnings.append(f"named {p.view!r} but the head is turned the other way (yaw {yaw:+.0f})")
    if p.view == "front" and abs(yaw) > 45:
        warnings.append(f"named 'front' but the head is turned {yaw:+.0f} degrees")
    projected = camera.project(head.points)
    size_px = face_size(marks.points)
    seen = visible(head, model, camera, WARP_FACING)
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
    sharp = _sharpness(p.image, marks)
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
    focal_from = p.focal_from
    if p.shared:
        warnings.append(f"the focal length of {p.shared}")
    elif p.focal is None:
        focal_from = "assumed"
        warnings.append(f"no EXIF focal length: about a {FOV_PRIOR:.0f} degree view assumed (the landmarks hardly tell "
                        "the focal length: a wide-angle selfie then comes out too far away, a zoomed photo too near)")
    if abs(yaw) > SIDE_YAW:
        warnings.append(f"seen from the side: MediaPipe finds such faces turned {SIDE_YAW:.0f}-60 degrees whether "
                        "they are turned 60 or 90, and guesses their hidden half (name side photos and segment them, "
                        "so their outline places them)")
        q_other *= 0.6
    quality = q_resolution * q_fit * q_agree * q_sharp * q_other
    return View(p.image, name, True, camera=camera, method="landmarks", marks=marks, projected=projected,
                inliers=fit.inliers, error=fit.error, warp=warp, quality=float(quality), warnings=warnings,
                focal_known=p.focal_known, focal_from=focal_from)


def _register_outline(p: _Photo, head, model, reference, notes, doubt: str, segmenter=None,
                      turn: float | None = None) -> View:
    """The photo placed by the head's outline: ``notes`` are warnings about it, ``doubt`` what is wrong with the face
    MediaPipe found (empty: no face); ``turn``: the head's yaw, held (see fit_silhouette)."""
    name = p.view or "unknown"
    what = f"the face MediaPipe found {doubt}" if doubt else "no face found"
    warnings = list(notes) + ([f"{what}: ignored"] if doubt else [])
    if p.view is None or p.view == "front":
        hint = "" if p.view else ("; if the photo shows the side or the back of the head, name the view ('left', "
                                  "'right' or 'back')")
        return View(p.image, name, False, what + hint, warnings=warnings)
    if not _head_masks(p):
        return View(p.image, name, False, f"{what}, and no head mask to fit the head's outline to (use the "
                    "segmenter)", warnings=warnings)
    from .landmarks import face_frame

    frame = face_frame(model)
    distance = None
    focal, focal_from = p.focal, p.focal_from
    if reference is not None and reference.usable and reference.camera is not None:
        distance = float(np.linalg.norm(reference.camera.position - frame.pivot))  # the same distance from the head
        own = f"the {reference.name} photo"
        warnings.append(f"the distance and focal length of {own}" if p.shared == own else f"the distance of {own}")
        if p.shared and p.shared != own:
            warnings.append(f"the focal length of {p.shared}")
    elif p.shared:
        warnings.append(f"the focal length of {p.shared}")
    if focal is None:
        focal, focal_from = focal_for_fov(FOV_PRIOR, p.size), "assumed"
        other = "" if reference is None else " and the reference photo has another shape"
        warnings.append(f"no EXIF focal length{other}: a {FOV_PRIOR:.0f} degree view was assumed")
    try:  # where the segmenter finds face skin on the avatar's head: rendered once per head model, then cached
        regions = head_regions(model, head.primitive, segmenter)
    except Exception as exc:  # noqa: BLE001 - no OpenGL for the renders, say: the outline alone then
        regions = None
        warnings.append(f"the outline alone (the avatar's face skin could not be found: {exc})")
    skin = mask_at(p.masks["face"], p.size) if regions is not None else None
    fit = fit_silhouette(model, head.primitive, frame.chin, frame.pivot, _head_mask(p), VIEW_YAWS[p.view], focal,
                         distance, skin, turn)
    camera, iou = fit.camera, fit.iou
    if camera is None or iou < (SILHOUETTE_MIN_IOU if distance else SILHOUETTE_MIN_IOU_FREE):
        return View(p.image, name, False, f"{what}, and the head's outline does not fit the photo's head (overlap "
                    f"{iou:.2f})", camera=camera, method="silhouette", iou=iou, skin=fit.skin, warnings=warnings,
                    focal_known=p.focal_known, focal_from=focal_from)
    quality = (iou - SILHOUETTE_MIN_IOU) / (1.0 - SILHOUETTE_MIN_IOU) * 0.6 * 0.75 ** len(fit.uncertain)
    if distance is None:
        quality *= 0.7
    warnings.append("placed by the head's outline alone, not by landmarks: its turn may be 10-20 degrees off, at times "
                    "40" if math.isnan(fit.skin) else "placed by the head's outline and face skin, not by landmarks: "
                    "its turn may be 5-10 degrees off, at times 20-40")
    if fit.uncertain:
        warnings.append(f"the outline hardly fixes the {' and the '.join(fit.uncertain)}: kept near the named view")
    return View(p.image, name, True, camera=camera, method="silhouette", projected=camera.project(head.points),
                quality=float(quality), warnings=warnings, focal_known=p.focal_known, focal_from=focal_from,
                iou=iou, skin=fit.skin)


def draw_registration(view: View, model, head, wireframe: bool = True, arrows: bool = False) -> np.ndarray:
    """The photo with what the registration found drawn over it: the avatar's head mesh as the camera sees it (its
    triangles facing the camera, through the warp), the avatar's projected landmarks on the side of the head the
    camera sees (green: agreeing with the camera fit, red: not) and the person's (small white dots). ``arrows``: the
    warp instead of the dots, as arrows from the avatar's projected features to the person's, three times as long."""
    out = view.image.copy()
    if view.camera is None:
        return out
    shown = visible(head, model, view.camera)  # the landmarks the camera sees
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
