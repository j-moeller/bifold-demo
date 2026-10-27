"""Stage 0: MediaPipe's 478 face landmarks on an avatar's head mesh, found once per head model.

The avatar's head is rendered with its own texture in the bind pose (flat, even light, no shading) from five views:
straight in front, turned 25 degrees to each side and seen from 12 degrees above and below. The Face Landmarker runs
on each render, and each landmark it finds is lifted onto the head mesh: the ray from the render camera through the
landmark's pixel is cast against the head's triangles, and the first one it hits is where the landmark sits. A
landmark seen at a grazing angle (the ray nearly along the surface, as on the far side of a turned face) is not
trusted in that view.

The views are combined per landmark: the median of the points, then the mean of the points within 4 mm of it (or of
three times their median distance), snapped back onto the mesh. Each landmark keeps its triangle and barycentric
coordinates there, so it follows the skinning (the jaw, the face bones), and a *reliability* from 0 to 1: how many
views agree (three or more for full marks) and how closely (falling off over 3 mm).

When MediaPipe finds no face on any render (a stylised or covered face), the fallback fits MediaPipe's canonical face
model (:mod:`.canonical`) to the avatar: a width and a height scale and a shift (least squares) that put its eyes,
nose tip, mouth corners, chin and cheek sides where the avatar's eye and mouth-corner bones are and where its nose
tip, chin and ears are found on the mesh (the most forward point of the midline between the eyes and the mouth; the
point of the midline below the mouth furthest forward and down at 27 degrees; the outermost points at the ears'
height, of which MediaPipe's cheek sides lie at 70 %, the median on the 20 avatars of the test set); the depth is
scaled by the mean of the two and shifted onto the nose, mouth and chin. Each fitted point goes to the nearest point
of the head mesh, and the irises are cast from the front at the eye bones. Its landmarks get a reliability of 0.3;
forced on the 20 avatars, they lie 7 mm from MediaPipe's on average (5 at the eyes, mouth and nose, 17 on the face's
outline).

Results are cached in ``<characters folder>/headtex/landmarks-<key>.npz`` (or ``$BIFOLD_HEADTEX_CACHE``), the key a
fingerprint of the head's geometry, the face bones and this code's version (in memory only when the folder cannot be
written).
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

from .. import face_photo
from . import canonical as cf
from . import raycast
from .camera import PhotoCamera

VERSION = 1
VIEWS = ((0.0, 0.0), (25.0, 0.0), (-25.0, 0.0), (0.0, 12.0), (0.0, -12.0))  # (yaw, pitch) of the renders
RENDER_SIZE = 512
FOV = 30.0  # degrees: the render camera's field of view
DISTANCE = 0.55  # metres from the face centre: the head about 0.25 m tall in a 0.3 m wide view
GRAZING = 0.25  # a landmark counts in a view when the surface faces the camera at least this much (cosine)
AGREE = 0.004  # metres: views within this of the median agree
SPREAD = 0.003  # metres: reliability falls off with the spread of the agreeing views over this
FALLBACK_RELIABILITY = 0.3
EYEBALL = 0.011  # metres: an eye bone sits this far behind the eye's surface (only used to place the irises)
IRIS_RADIUS = 0.0058  # metres: the iris' radius on the canonical face (scaled with the face)
IRIS_RIM = ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0))  # 469..472 / 474..477: +X (face's left), up, ...
EAR_TO_CHEEK = 0.7  # MediaPipe's cheek sides (234, 454) lie at this share of the ears' distance from the middle

CACHE_ENV = "BIFOLD_HEADTEX_CACHE"
_memory: dict[str, HeadLandmarks] = {}


@dataclass
class HeadLandmarks:
    """The 478 face landmarks on one head model (bind pose, character space)."""

    points: np.ndarray  # (478, 3) metres
    triangle: np.ndarray  # (478,) triangle of the head primitive each landmark lies on
    barycentric: np.ndarray  # (478, 3) its position in that triangle
    reliability: np.ndarray  # (478,) 0-1
    primitive: int  # the head primitive (see :func:`bifold_demo.characters.face_photo.head_layout`)
    method: str  # "mediapipe" or "canonical" (the fallback fit)
    views: int  # renders in which MediaPipe found the face
    key: str = ""

    def posed(self, model, palette) -> np.ndarray:
        """The landmarks moved by the skinning matrices ``palette`` (``(B, 4, 4)``, e.g. a character's
        ``skin_palette``)."""
        head = model.primitives[self.primitive]
        tri = np.asarray(head.indices, np.int64).reshape(-1, 3)[self.triangle]  # (478, 3) vertex indices
        positions = np.asarray(head.positions, np.float64)[tri]
        joints, weights = np.asarray(head.joints)[tri], np.asarray(head.weights, np.float64)[tri]
        matrices = np.einsum("lcj,lcjab->lcab", weights, np.asarray(palette, np.float64)[joints])
        moved = np.einsum("lcab,lcb->lca", matrices[..., :3, :3], positions) + matrices[..., :3, 3]
        return np.einsum("lc,lcd->ld", self.barycentric, moved)

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp.npz")
        np.savez_compressed(tmp, **{f.name: np.asarray(getattr(self, f.name)) for f in fields(self)})
        os.replace(tmp, path)

    @classmethod
    def load(cls, path) -> HeadLandmarks:
        with np.load(path, allow_pickle=False) as data:
            values = {f.name: data[f.name] for f in fields(cls)}
        for name in ("primitive", "views"):
            values[name] = int(values[name])
        for name in ("method", "key"):
            values[name] = str(values[name])
        return cls(**values)


@dataclass
class FaceFrame:
    """Where an avatar's face is, from its face bones and head mesh (bind pose)."""

    primitive: int
    eyes: np.ndarray  # (2, 3) right, left eye bone
    mouth: np.ndarray  # (2, 3) right, left mouth-corner bone
    nose: np.ndarray  # (3,) the tip of the nose: the most forward point of the midline
    chin: np.ndarray  # (3,) the corner of the chin's profile
    ears: np.ndarray  # (2, 3) right, left: the outermost points of the head at the ears' height

    @property
    def centre(self) -> np.ndarray:
        """Half way between the eyes and the mouth."""
        return 0.5 * (self.eyes.mean(axis=0) + self.mouth.mean(axis=0))

    @property
    def pivot(self) -> np.ndarray:
        """The middle of the head, between the ears: what a camera circling the head stays at a distance from."""
        return self.ears.mean(axis=0)

    @property
    def eye_distance(self) -> float:
        return float(np.linalg.norm(self.eyes[1] - self.eyes[0]))


def face_frame(model) -> FaceFrame:
    """The :class:`FaceFrame` of ``model``, or :class:`ValueError` when it has no head or face bones."""
    layout = face_photo.head_layout(model)
    if layout is None:
        raise ValueError(f"{model.name} has no head with face bones")
    skeleton = model.skeleton
    rest = skeleton.rest_global

    def bone(name):
        return rest[skeleton.index(name), :3, 3].astype(np.float64)

    eyes = np.array([bone("Bip01 REye"), bone("Bip01 LEye")])
    mouth = np.array([bone("Bip01 RMouthCorner"), bone("Bip01 LMouthCorner")])
    points = np.asarray(model.primitives[layout.primitive].positions, np.float64)
    width = layout.eye_distance
    middle = points[np.abs(points[:, 0]) < 0.12 * width]
    between = middle[(middle[:, 1] < eyes[:, 1].mean()) & (middle[:, 1] > mouth[:, 1].mean())]
    nose = between[between[:, 2].argmax()] if len(between) else middle[middle[:, 2].argmax()]
    drop = 0.75 * (eyes[:, 1].mean() - mouth[:, 1].mean())  # the chin is less than this below the mouth
    below = middle[(middle[:, 1] < mouth[:, 1].mean()) & (middle[:, 1] > mouth[:, 1].mean() - drop)]
    chin = below[(below[:, 2] - 0.5 * below[:, 1]).argmax()]  # furthest forward and down (27 degrees down)
    level = points[(points[:, 1] < eyes[:, 1].mean()) & (points[:, 1] > nose[1] - 0.3 * width)]
    ears = np.array([level[level[:, 0].argmin()], level[level[:, 0].argmax()]])
    ears[:, 0] = np.array([-1.0, 1.0]) * np.abs(ears[:, 0]).min()  # hair on one side can stick out further
    return FaceFrame(layout.primitive, eyes, mouth, nose, chin, ears)


def views(frame: FaceFrame, size: int = RENDER_SIZE) -> list[PhotoCamera]:
    """The render cameras of stage 0, looking at the face centre (slightly above it: the forehead is taller than the
    chin)."""
    target = frame.centre + np.array([0.0, 0.3 * frame.eye_distance, 0.0])
    return [PhotoCamera.around(target, DISTANCE, yaw, pitch, fov=FOV, size=(size, size)) for yaw, pitch in VIEWS]


def render_views(model, cameras, renderer=None) -> list[np.ndarray]:
    """``model`` in its bind pose with its own textures, unlit, seen by ``cameras`` (BGR images; needs OpenGL)."""
    from ...scene import Scene
    from ...scene.renderer import Renderer
    from ..character import Character

    size = cameras[0].size
    own = renderer is None
    renderer = renderer or Renderer(*size)
    try:
        scene = Scene(background=(128, 128, 128), ambient=1.0)
        scene.add(Character(model, name="avatar", shadow=False))
        scene.update(0.0, 0.0)
        images = []
        for camera in cameras:
            scene.camera = camera.scene_camera()
            images.append(renderer.render(scene))
        return images
    finally:
        if own:
            renderer.release()


def geometry_key(model) -> str:
    """A fingerprint of what the landmarks depend on: the head's mesh and the face bones, and this code."""
    frame_bones = ("Bip01 REye", "Bip01 LEye", "Bip01 RMouthCorner", "Bip01 LMouthCorner")
    layout = face_photo.head_layout(model)
    head = model.primitives[layout.primitive]
    h = hashlib.blake2b(f"headtex-landmarks-{VERSION}".encode(), digest_size=12)
    skeleton = model.skeleton
    arrays = [head.positions, head.indices, *(skeleton.rest_global[skeleton.index(b)] for b in frame_bones)]
    for array in map(np.asarray, arrays):
        h.update(f"{array.dtype}{array.shape}".encode())
        h.update(np.ascontiguousarray(array).tobytes())
    return h.hexdigest()


def cache_folder(cache_dir=None) -> Path:
    """``$BIFOLD_HEADTEX_CACHE`` if set, else ``headtex/`` in ``cache_dir`` (default: the prepared characters'
    folder)."""
    from ..library import default_folder

    if os.environ.get(CACHE_ENV):
        return Path(os.environ[CACHE_ENV]).expanduser()
    return (Path(cache_dir).expanduser() if cache_dir is not None else default_folder()) / "headtex"


def head_landmarks(model, landmarker=None, *, cache_dir=None, render=None, canonical=None, method: str = "auto",
                   refresh: bool = False) -> HeadLandmarks:
    """The face landmarks on ``model``'s head (see the module docs), cached.

    ``landmarker``: anything with ``detect(image) -> list[FaceMarks]`` (default: a MediaPipe
    :class:`~.detect.Landmarker`). ``render(model, cameras) -> images`` makes the views (default: the offscreen
    renderer). ``canonical``: the :class:`~.canonical.CanonicalFace` for the fallback (default: read from the Face
    Landmarker's bundle). ``method``: ``"auto"`` (MediaPipe, else the fallback), ``"mediapipe"`` (raise when no face
    is found) or ``"canonical"`` (the fallback only). ``cache_dir``: the characters folder whose ``headtex/``
    subfolder keeps the results (default: the prepared characters' folder); ``False``: memory only.
    """
    if method not in ("auto", "mediapipe", "canonical"):
        raise ValueError(f"unknown method {method!r}")
    frame = face_frame(model)
    key = geometry_key(model) + ("" if method == "auto" else f"-{method}")
    path = None if cache_dir is False else cache_folder(cache_dir) / f"landmarks-{key}.npz"
    if not refresh:
        if key in _memory:
            return _memory[key]
        if path is not None and path.is_file():
            try:
                found = HeadLandmarks.load(path)
            except Exception:  # noqa: BLE001 - a damaged or foreign file (zipfile and zlib raise several kinds)
                found = None  # made again
            if found is not None and found.key == key:
                _memory[key] = found
                return found
    result = None
    if method != "canonical":
        if landmarker is None:
            from .detect import Landmarker

            landmarker = Landmarker()
        result = find_landmarks(model, frame, landmarker, render or render_views)
        if result is None and method == "mediapipe":
            raise ValueError(f"MediaPipe finds no face on the renders of {model.name}")
    if result is None:
        if canonical is None:
            canonical = _canonical_from(landmarker)
        result = fit_canonical(model, frame, canonical)
    result.key = key
    _memory[key] = result
    if path is not None:
        try:
            result.save(path)
        except OSError:
            pass  # a read-only folder: kept in memory only
    return result


def _canonical_from(landmarker):
    from .detect import FACE_LANDMARKER, model_path

    task = getattr(landmarker, "task_path", None) or model_path(FACE_LANDMARKER)
    return cf.read_canonical_face(task)


def _plausible(marks, camera: PhotoCamera, frame: FaceFrame) -> bool:
    """Whether a face found on a render is the avatar's: its nose near the avatar's nose, its eyes about as far
    apart."""
    nose = camera.project(frame.nose)[0]
    eyes = camera.project(frame.eyes)
    distance = float(np.linalg.norm(eyes[1] - eyes[0]))
    found = marks.points
    found_distance = float(np.linalg.norm(found[cf.LEFT_IRIS[0]] - found[cf.RIGHT_IRIS[0]]))
    return (np.linalg.norm(found[cf.NOSE_TIP] - nose) < 0.6 * distance
            and 0.5 * distance < found_distance < 2.0 * distance)


def lift(points, camera: PhotoCamera, corners: np.ndarray, normals: np.ndarray):
    """Landmarks seen at ``points`` (``(N, 2)`` pixels) by ``camera``, on the mesh ``corners``:
    ``(xyz, triangle, barycentric, facing)``, where ``facing`` is how squarely the surface faces the camera there
    (cosine; 0 where the ray misses: then the landmark goes to the mesh point nearest the ray)."""
    origin, directions = camera.rays(points)
    distance, triangle, bary = raycast.intersect(origin, directions, corners)
    facing = np.zeros(len(points))
    hit = triangle >= 0
    facing[hit] = np.abs(np.einsum("nk,nk->n", normals[triangle[hit]], directions[hit]))
    xyz = np.zeros((len(points), 3))
    xyz[hit] = origin + directions[hit] * distance[hit, None]
    if (~hit).any():  # past the silhouette: the vertex nearest the ray
        vertices = corners.reshape(-1, 3)
        rel = vertices - origin
        for k in np.nonzero(~hit)[0]:
            along = rel @ directions[k]
            off = np.linalg.norm(rel - along[:, None] * directions[k], axis=1)
            best = int(off.argmin())
            triangle[k], corner = divmod(best, 3)
            bary[k] = np.eye(3)[corner]
            xyz[k] = vertices[best]
    return xyz, triangle, bary, facing


def combine(candidates: np.ndarray, facing: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Robust per-landmark combination of ``(V, N, 3)`` points seen in ``V`` views with their ``(V, N)`` facing:
    ``(points (N, 3), reliability (N,))``."""
    n = candidates.shape[1]
    out = np.zeros((n, 3))
    reliability = np.zeros(n)
    for i in range(n):
        good = facing[:, i] >= GRAZING
        if not good.any():  # seen only at grazing angles: the most squarely seen view, barely trusted
            v = int(facing[:, i].argmax())
            out[i] = candidates[v, i]
            reliability[i] = 0.1 if facing[v, i] > 0 else 0.05
            continue
        c, w = candidates[good, i], facing[good, i]
        median = np.median(c, axis=0)
        d = np.linalg.norm(c - median, axis=1)
        keep = d <= max(AGREE, 3.0 * float(np.median(d)))
        c, w = c[keep], w[keep]
        out[i] = (c * w[:, None]).sum(axis=0) / w.sum()
        spread = float(np.sqrt((w * ((c - out[i]) ** 2).sum(axis=1)).sum() / w.sum()))
        reliability[i] = min(1.0, len(c) / 3.0) * float(np.exp(-((spread / SPREAD) ** 2)))
    return out, reliability


def _head_mesh(model, frame: FaceFrame):
    head = model.primitives[frame.primitive]
    corners = raycast.corners_of(head.positions, head.indices)
    return corners, raycast.triangle_normals(corners)


def find_landmarks(model, frame: FaceFrame, landmarker, render) -> HeadLandmarks | None:
    """The MediaPipe path of stage 0 (``None`` when no render shows a face)."""
    cameras = views(frame)
    images = render(model, cameras)
    corners, normals = _head_mesh(model, frame)
    candidates, facing = [], []
    for image, camera in zip(images, cameras):
        faces = landmarker.detect(image)
        if not faces or not _plausible(faces[0], camera, frame):
            continue
        xyz, _tri, _bary, face = lift(faces[0].points, camera, corners, normals)
        candidates.append(xyz)
        facing.append(face)
    if not candidates:
        return None
    points, reliability = combine(np.array(candidates), np.array(facing))
    # near a crease (the lips, the eyelids) the mean can leave the surface: back onto the mesh
    _d, triangle, bary = raycast.closest(points, corners)
    points = raycast.point_on(corners, triangle, bary)
    return HeadLandmarks(points, triangle, bary, reliability, frame.primitive, "mediapipe", len(candidates))


def _axis_fit(source: np.ndarray, target: np.ndarray, weight: np.ndarray) -> tuple[float, float]:
    """Scale and shift with ``target ~ scale * source + shift`` (weighted least squares)."""
    w = weight / weight.sum()
    ms, mt = (w * source).sum(), (w * target).sum()
    var = (w * (source - ms) ** 2).sum()
    scale = (w * (source - ms) * (target - mt)).sum() / max(var, 1e-12)
    return float(scale), float(mt - scale * ms)


def fit_canonical(model, frame: FaceFrame, canonical: cf.CanonicalFace) -> HeadLandmarks:
    """The fallback of stage 0: MediaPipe's canonical face fitted to the avatar (see the module docs)."""
    c = np.asarray(canonical.points, np.float64)
    eye_r, eye_l = c[list(cf.RIGHT_EYE)].mean(axis=0), c[list(cf.LEFT_EYE)].mean(axis=0)
    cheek = EAR_TO_CHEEK * frame.ears
    pairs = [  # canonical point, avatar point, weights per axis (x, y, z)
        (eye_r, frame.eyes[0], (1, 1, 0)), (eye_l, frame.eyes[1], (1, 1, 0)),
        (c[cf.NOSE_TIP], frame.nose, (1, 1, 1)),
        (c[cf.MOUTH_RIGHT], frame.mouth[0], (1, 1, 1)), (c[cf.MOUTH_LEFT], frame.mouth[1], (1, 1, 1)),
        (c[cf.CHIN], frame.chin, (0, 1, 1)),
        (c[cf.RIGHT_CHEEK_SIDE], cheek[0], (0.5, 0, 0)), (c[cf.LEFT_CHEEK_SIDE], cheek[1], (0.5, 0, 0)),
    ]
    source = np.array([p[0] for p in pairs])
    target = np.array([p[1] for p in pairs])
    weight = np.array([p[2] for p in pairs], np.float64)
    (sx, tx), (sy, ty) = (_axis_fit(source[:, a], target[:, a], weight[:, a]) for a in range(2))
    sz = 0.5 * (sx + sy)  # the depth: as the face's width and height, shifted onto the nose, mouth and chin
    w = weight[:, 2]
    tz = float((w * (target[:, 2] - sz * source[:, 2])).sum() / w.sum())
    scale, shift = np.array([sx, sy, sz]), np.array([tx, ty, tz])
    placed = c * scale + shift
    corners, _normals = _head_mesh(model, frame)
    _d, triangle, bary = raycast.closest(placed, corners)
    # the irises: cast from straight in front at the eye bones, on the eyes' surface
    size = float(np.mean(np.abs(scale[:2])))
    rays, order = [], []
    for eye, ids in ((frame.eyes[0], cf.RIGHT_IRIS), (frame.eyes[1], cf.LEFT_IRIS)):
        offsets = [(0.0, 0.0)] + [(dx * IRIS_RADIUS * size * 100, dy * IRIS_RADIUS * size * 100)
                                  for dx, dy in IRIS_RIM]
        for k, (dx, dy) in zip(ids, offsets):
            rays.append(eye + np.array([dx, dy, 0.0]))
            order.append(k)
    rays = np.array(rays)
    _dist, tri, b = raycast.intersect(rays + np.array([0.0, 0.0, 0.5]), np.tile([0.0, 0.0, -1.0], (len(rays), 1)),
                                      corners)
    triangle = np.r_[triangle, np.zeros(cf.COUNT - len(triangle), np.int64)]
    bary = np.r_[bary, np.zeros((cf.COUNT - len(bary), 3))]
    for k, t_, bb, r in zip(order, tri, b, rays):
        if t_ < 0:  # no surface in front of the eye: the surface nearest the bone pushed forward
            _d, t1, b1 = raycast.closest((r + np.array([0.0, 0.0, EYEBALL]))[None], corners)
            t_, bb = int(t1[0]), b1[0]
        triangle[k], bary[k] = t_, bb
    points = raycast.point_on(corners, triangle, bary)
    reliability = np.full(cf.COUNT, FALLBACK_RELIABILITY)
    return HeadLandmarks(points, triangle, bary, reliability, frame.primitive, "canonical", 0)
