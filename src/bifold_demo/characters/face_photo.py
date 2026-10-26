"""A real person's face on a 3D person: a photo painted into the avatar's head texture.

The photo's face is lined up with the avatar's by the five landmarks the face detector finds (the eyes, the nose tip,
the mouth corners) and the same five points on the avatar (its eye bones, the tip of its nose, its mouth-corner
bones), seen from the front: a similarity transform from the avatar's front view to the photo. Every part of the
head mesh that can be seen from the front (a depth buffer of the front view leaves out the inside of the mouth, the
teeth and the backs of the eyes) is then painted with the photo pixels in front of it, in the head texture's own
layout (each texture pixel looks up its 3D point, its place in the front view and so its photo pixel). The edge is
feathered into the avatar's own skin and the photo's skin tone is moved part of the way towards the avatar's, so no
seam shows. The face is a texture on the same mesh, so it moves with the head, the jaw and the funny faces.

The painted texture belongs to one person (``character.texture_overrides``); the avatar's other users keep theirs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from ..scene.textures import Texture, as_texture

LANDMARK_BONES = ("Bip01 REye", "Bip01 LEye", None, "Bip01 RMouthCorner", "Bip01 LMouthCorner")  # None: nose tip
GRID = 0.002  # metres: the front view's depth-buffer cell
DEPTH_SLACK = 0.004  # metres: surfaces this far behind the front one still count as seen
OVAL = (1.45, 1.75)  # half width and half height of the painted oval, in eye distances, around the face centre
FEATHER = 0.4  # share of the oval's size over which the photo fades into the avatar's skin
TONE = 0.5  # how far the photo's skin tone moves to the avatar's (0: keep the photo's, 1: the avatar's)
TURNED = (0.0, 0.3)  # the photo fades out where the surface faces sideways (the normal's forward part), to 0 where
# the triangles stop being painted, so no edge shows


@dataclass
class HeadLayout:
    """What the painting needs to know about one avatar's head (made once per model by :func:`head_layout`)."""

    primitive: int  # index of the head primitive in the model
    front: np.ndarray  # (N, 2) the head vertices in the front view: x to the viewer's right, y down (metres)
    seen: np.ndarray  # (T,) bool: triangles seen from the front
    landmarks: np.ndarray  # (5, 2) the avatar's five landmarks in the front view
    eye_distance: float


_layouts: dict[int, tuple] = {}


def _head_primitive(model) -> int | None:
    best = None
    for k, primitive in enumerate(model.primitives):
        name = primitive.name.lower()
        if "head" in name and not any(word in name for word in ("hair", "hat", "helmet", "opacity")):
            best = k
    return best


def head_layout(model) -> HeadLayout | None:
    """The front view of ``model``'s head, or ``None`` when it has no head primitive or face bones."""
    cached = _layouts.get(id(model))
    if cached is not None and cached[0] is model:
        return cached[1]
    layout = _make_layout(model)
    _layouts[id(model)] = (model, layout)
    return layout


def _make_layout(model) -> HeadLayout | None:
    index = _head_primitive(model)
    skeleton = model.skeleton
    if index is None or any(b is not None and skeleton.find(b) is None for b in LANDMARK_BONES):
        return None
    head = model.primitives[index]
    points = np.asarray(head.positions, float)  # the bind pose, character space: facing +Z, +X to the left
    front = np.c_[points[:, 0], -points[:, 1]]  # seen from in front: the person's left is the viewer's right
    rest = skeleton.rest_global
    marks = []
    for bone in LANDMARK_BONES:
        if bone is None:  # the nose tip: the most forward point between the eyes and the mouth
            middle = np.abs(points[:, 0]) < 0.01
            marks.append(None if not middle.any() else points[middle][points[middle][:, 2].argmax()])
        else:
            marks.append(rest[skeleton.index(bone), :3, 3])
    if any(m is None for m in marks):
        return None
    marks = np.array(marks)
    landmarks = np.c_[marks[:, 0], -marks[:, 1]]
    eye_distance = float(np.linalg.norm(landmarks[1] - landmarks[0]))
    triangles = np.asarray(head.indices, np.int64).reshape(-1, 3)
    return HeadLayout(index, front, _seen(points, front, triangles), landmarks, eye_distance)


def _seen(points: np.ndarray, front: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Which triangles are the front surface: facing forward and not behind another surface (a depth buffer of the
    front view, filled at the triangles' corners and centres)."""
    a, b, c = points[triangles[:, 0]], points[triangles[:, 1]], points[triangles[:, 2]]
    normal = np.cross(b - a, c - a)
    facing = normal[:, 2] > TURNED[0] * np.linalg.norm(normal, axis=1)
    samples = np.concatenate([a, b, c, (a + b + c) / 3, (a + b) / 2, (b + c) / 2, (a + c) / 2])
    lo = front.min(axis=0)
    cells = np.floor((np.c_[samples[:, 0], -samples[:, 1]] - lo) / GRID).astype(int)
    shape = cells.max(axis=0) + 1
    depth = np.full(shape, -np.inf)
    np.maximum.at(depth, (cells[:, 0], cells[:, 1]), samples[:, 2])
    centre = (a + b + c) / 3
    cell = np.floor((np.c_[centre[:, 0], -centre[:, 1]] - lo) / GRID).astype(int)
    # a steep surface spans more depth within one cell: allow for its slope
    slope = np.hypot(normal[:, 0], normal[:, 1]) / np.maximum(normal[:, 2], 0.05 * np.linalg.norm(normal, axis=1))
    return facing & (centre[:, 2] >= depth[cell[:, 0], cell[:, 1]] - DEPTH_SLACK - 1.5 * GRID * slope)


def _similarity(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """The 2x3 rotation + uniform scale + shift taking the points ``src`` closest to ``dst`` (least squares)."""
    matrix, _ = cv2.estimateAffinePartial2D(src.astype(np.float32), dst.astype(np.float32), method=cv2.LMEDS)
    if matrix is None:
        raise ValueError("cannot line the face up with the photo")
    return matrix.astype(np.float64)


def paint_face(model, photo: np.ndarray, landmarks, box=None, base_texture=None) -> np.ndarray:
    """``model``'s head texture (RGBA, as an image: first row at the top) with the face of ``photo`` (BGR) painted
    on, lined up by the photo's five ``landmarks`` (as :class:`~bifold_demo.recognition.engine.Face` gives them).
    ``box`` (``(x, y, w, h)``, the face detector's box): only the photo inside the oval it frames is used, so the
    hair and the background beside a narrow face stay off a wider head."""
    layout = head_layout(model)
    if layout is None:
        raise ValueError(f"{model.name} has no head to paint a face on")
    head = model.primitives[layout.primitive]
    texture = as_texture(base_texture if base_texture is not None else head.texture)
    atlas = texture.rgba.astype(np.float32)  # RGBA, first row at the top
    th, tw = atlas.shape[:2]
    to_photo = _similarity(layout.landmarks, np.asarray(landmarks, float))
    photo_points = layout.front @ to_photo[:, :2].T + to_photo[:, 2]  # every head vertex's photo pixel

    # where each painted texture pixel looks in the photo, and how much of the photo it takes
    eyes, mouth = layout.landmarks[:2].mean(axis=0), layout.landmarks[3:].mean(axis=0)
    centre = 0.5 * (eyes + mouth)
    radii = np.array(OVAL) * layout.eye_distance
    map_x = np.full((th, tw), -1.0, np.float32)
    map_y = np.full((th, tw), -1.0, np.float32)
    weight = np.zeros((th, tw), np.float32)
    uv = np.asarray(head.uvs, float)
    pixels = np.c_[uv[:, 0] * tw - 0.5, (1.0 - uv[:, 1]) * th - 0.5]  # texture pixel of each vertex
    oval = np.hypot(*((layout.front - centre) / radii).T)  # 1 at the oval's edge
    normals = np.asarray(head.normals, float)
    forward = normals[:, 2] / np.maximum(np.linalg.norm(normals, axis=1), 1e-9)
    fade = _smooth((1.0 - oval) / FEATHER) * _smooth((forward - TURNED[0]) / (TURNED[1] - TURNED[0]))
    triangles = np.asarray(head.indices, np.int64).reshape(-1, 3)
    for tri in triangles[layout.seen]:
        if fade[tri].max() <= 0.0:
            continue
        _fill(pixels[tri], photo_points[tri], fade[tri], map_x, map_y, weight)
    if not weight.any():
        raise ValueError("the face does not reach the head texture")
    sampled = cv2.remap(photo, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if box is not None:
        weight *= cv2.remap(_face_mask(photo.shape[:2], box), map_x, map_y, cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    sampled = cv2.cvtColor(sampled, cv2.COLOR_BGR2RGB).astype(np.float32)
    inside = weight > 0.5
    if inside.sum() > 50:  # move the photo's skin tone towards the avatar's
        own, theirs = atlas[..., :3][inside].mean(axis=0), sampled[inside].mean(axis=0)
        sampled = np.clip(sampled + TONE * (own - theirs), 0.0, 255.0)
    w = weight[..., None]
    atlas[..., :3] = atlas[..., :3] * (1.0 - w) + sampled * w
    return atlas.clip(0, 255).astype(np.uint8)


def _face_mask(shape, box) -> np.ndarray:
    """1 inside the oval the face box frames in the photo, fading to 0 just outside it."""
    x, y, w, h = box
    scale = 0.25 * min(w, h)  # the mask is made small and blurred, then scaled up
    small = (max(4, round(shape[1] / scale * 8)), max(4, round(shape[0] / scale * 8)))
    mask = np.zeros((small[1], small[0]), np.float32)
    fx, fy = small[0] / shape[1], small[1] / shape[0]
    axes = (max(1, round(w * 0.48 * fx)), max(1, round(h * 0.55 * fy)))  # narrower than the box: its sides hold hair
    cv2.ellipse(mask, (round((x + w / 2) * fx), round((y + h * 0.55) * fy)), axes, 0.0, 0.0, 360.0, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), 0.08 * w * fx + 0.5)
    return cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)


def _smooth(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def _fill(corners: np.ndarray, photo: np.ndarray, fade: np.ndarray, map_x, map_y, weight) -> None:
    """Rasterise one triangle in the texture: each pixel gets its photo position and fade (barycentric)."""
    th, tw = weight.shape
    x0, y0 = np.floor(corners.min(axis=0)).astype(int) - 1
    x1, y1 = np.ceil(corners.max(axis=0)).astype(int) + 1
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, tw - 1), min(y1, th - 1)
    if x1 < x0 or y1 < y0:
        return
    (ax, ay), (bx, by), (cx, cy) = corners
    det = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
    if abs(det) < 1e-9:
        return
    gx, gy = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
    l0 = ((by - cy) * (gx - cx) + (cx - bx) * (gy - cy)) / det
    l1 = ((cy - ay) * (gx - cx) + (ax - cx) * (gy - cy)) / det
    l2 = 1.0 - l0 - l1
    pad = -0.6 / max(math.sqrt(abs(det)), 1.0)  # a little over the edges, so neighbours meet without gaps
    inside = (l0 >= pad) & (l1 >= pad) & (l2 >= pad)
    if not inside.any():
        return
    lam = np.stack([l0[inside], l1[inside], l2[inside]], axis=1)
    ys, xs = gy[inside], gx[inside]
    w = lam @ fade
    better = w > weight[ys, xs]
    ys, xs, lam, w = ys[better], xs[better], lam[better], w[better]
    map_x[ys, xs] = lam @ photo[:, 0]
    map_y[ys, xs] = lam @ photo[:, 1]
    weight[ys, xs] = w


def put_face(character, photo, landmarks=None, engine=None, box=None) -> None:
    """Paint the face in ``photo`` (a path or a BGR image; its biggest face, found with ``engine``, a
    :class:`~bifold_demo.recognition.engine.FaceEngine`, unless ``landmarks`` (and ``box``) are given) on
    ``character``."""
    image = cv2.imread(str(photo), cv2.IMREAD_COLOR) if not isinstance(photo, np.ndarray) else photo
    if image is None:
        raise FileNotFoundError(f"cannot read {photo}")
    if landmarks is None:
        if engine is None:
            from ..recognition.engine import FaceEngine

            engine = FaceEngine()
        faces = engine.detect(image, min_size=0)
        if not faces:
            raise ValueError(f"no face found in {photo if not isinstance(photo, np.ndarray) else 'the photo'}")
        landmarks, box = faces[0].landmarks, faces[0].box
    model = character.skinned_model
    painted = paint_face(model, image, landmarks, box)
    layout = head_layout(model)
    character.texture_overrides[model.primitives[layout.primitive].name] = Texture.from_array(
        painted, order="rgb", name=f"{character.name} face")


def take_face_off(character) -> None:
    layout = head_layout(character.skinned_model)
    if layout is not None:
        character.texture_overrides.pop(character.skinned_model.primitives[layout.primitive].name, None)
