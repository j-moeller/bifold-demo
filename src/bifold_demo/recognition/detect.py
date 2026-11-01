"""Finding faces: a box, five landmarks and a score per face, the same way from every detector.

**The landmark order** is the same everywhere, the order of the ArcFace alignment template (:mod:`.align`): the eye
on the image's left, the eye on the image's right, the nose tip, the mouth corner on the image's left, the one on
the image's right. (For a person facing the camera that is their right eye first; SCRFD and YuNet both give them so.)

**SCRFD-10G** (InsightFace's ``det_10g.onnx``, run by ONNX Runtime) sees the image letterboxed into a *canvas*: scaled
to fit with its aspect kept, placed at the top left, the rest black. For each of its three strides (8, 16 and 32
pixels) it scores two anchors per cell and gives, per anchor, the distances from the cell's centre to the box's four
sides and the offsets of the five landmarks, in units of the stride. :func:`decode_scrfd` turns those into boxes and
landmarks above a score, and :func:`nms` keeps the best of overlapping boxes; both are written from InsightFace's
reference code (``python-package/insightface/model_zoo/scrfd.py``: anchor centres in (x, y) order, two per cell;
``distance2bbox``, ``distance2kps``; NMS with the "+1" pixel areas at an overlap of 0.4; score 0.5). The input is
red-green-blue, ``(pixel - 127.5) / 128``.

**YuNet** (OpenCV Zoo, run by OpenCV's ``FaceDetectorYN``) sees the image itself (scaled down above
:data:`MAX_SIDE`), as the demo always did.

The *canvas* decides what size of face can be found. ``canvas=640`` (InsightFace's default) suits photos with a
big face; a whole camera frame with small faces needs its own resolution (``canvas="image"``: the image's size,
rounded up to a multiple of 32, at most :data:`MAX_SIDE` on the longer side); a head cut out of a frame is best
scaled up to a canvas of 160-192 pixels (an integer ``canvas`` scales the image up or down to fit). Faces come back
in the image's pixels, biggest first.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import cv2
import numpy as np

MAX_SIDE = 1600  # longer images are scaled down for detection
STRIDES = (8, 16, 32)  # SCRFD-10G's feature strides ...
ANCHORS = 2  # ... and anchors per cell
SCRFD_SCORE, SCRFD_NMS = 0.5, 0.4  # InsightFace's defaults
YUNET_SCORE, YUNET_NMS = 0.7, 0.3  # the demo's YuNet settings


@dataclass(frozen=True)
class Face:
    """A face in an image: ``box`` ``(x, y, w, h)`` in pixels, five ``landmarks`` ``(x, y)`` (the eyes as seen from
    the camera left to right, the nose tip, the mouth corners left to right) and the detector's ``score``."""

    box: tuple[float, float, float, float]
    landmarks: tuple[tuple[float, float], ...]
    score: float

    @property
    def size(self) -> float:
        return min(self.box[2], self.box[3])

    @property
    def points(self) -> np.ndarray:
        """The landmarks as a (5, 2) array."""
        return np.asarray(self.landmarks, np.float64).reshape(5, 2)

    @property
    def eye_distance(self) -> float:
        """Pixels between the centres of the eyes (the inter-eye distance, IED)."""
        eyes = self.points[:2]
        return float(np.linalg.norm(eyes[1] - eyes[0]))

    def mapped(self, scale: float, offset=(0.0, 0.0)) -> Face:
        """This face with every coordinate ``c`` turned into ``c * scale + offset`` (from a crop back to its frame)."""
        ox, oy = offset
        x, y, w, h = self.box
        box = (x * scale + ox, y * scale + oy, w * scale, h * scale)
        marks = tuple((px * scale + ox, py * scale + oy) for px, py in self.landmarks)
        return Face(box, marks, self.score)

    def row(self) -> np.ndarray:
        """The detector's own row (what ``FaceRecognizerSF.alignCrop`` takes)."""
        return np.array([*self.box, *np.ravel(self.landmarks), self.score], np.float32)


def _biggest_first(faces: list[Face], min_size: float) -> list[Face]:
    faces = [f for f in faces if f.size >= min_size]
    faces.sort(key=lambda f: -f.box[2] * f.box[3])
    return faces


def canvas_size(shape, canvas) -> tuple[int, int]:
    """The canvas ``(width, height)`` for an image of ``shape`` (see the module docs): an int gives a square, a pair
    itself, ``"image"`` the image's size (at most :data:`MAX_SIDE`) rounded up to a multiple of 32."""
    if isinstance(canvas, str):
        if canvas != "image":
            raise ValueError(f"canvas must be a size, a (width, height) pair or 'image', not {canvas!r}")
        h, w = shape[:2]
        scale = min(1.0, MAX_SIDE / max(h, w))
        return (max(32, -(-round(w * scale) // 32) * 32), max(32, -(-round(h * scale) // 32) * 32))
    if np.ndim(canvas) == 0:
        return int(canvas), int(canvas)
    return int(canvas[0]), int(canvas[1])


def letterbox(image: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, float]:
    """``image`` scaled to fit in ``size`` ``(width, height)`` with its aspect kept, at the top left of a black
    canvas; and the scale (canvas pixels per image pixel). As InsightFace's SCRFD does."""
    width, height = size
    h, w = image.shape[:2]
    if h / w > height / width:
        new_h, new_w = height, int(height / (h / w))
    else:
        new_w, new_h = width, int(width * (h / w))
    new_w, new_h = max(1, new_w), max(1, new_h)
    scale = new_h / h
    canvas = np.zeros((height, width, 3), np.uint8)
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    canvas[:new_h, :new_w] = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    return canvas, scale


def anchor_centres(height: int, width: int, stride: int, anchors: int = ANCHORS) -> np.ndarray:
    """The anchor centres of a ``height`` x ``width`` feature map, ``(height * width * anchors, 2)`` as ``(x, y)``
    pixels of the canvas, row by row, each cell's anchors one after the other (as the reference code orders them)."""
    centres = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
    centres = (centres * stride).reshape(-1, 2)
    return np.repeat(centres, anchors, axis=0) if anchors > 1 else centres


def decode_scrfd(outputs, canvas: tuple[int, int], threshold: float = SCRFD_SCORE, strides=STRIDES,
                 anchors: int = ANCHORS):
    """SCRFD's raw ``outputs`` (scores, box distances and landmark offsets, three strides each, in that order) for a
    canvas of ``(width, height)``: ``(boxes (N, 4) as x1, y1, x2, y2; scores (N,); landmarks (N, 5, 2))`` in canvas
    pixels, for the anchors scoring at least ``threshold``, best first. Outputs with a batch axis of 1 are taken too."""
    width, height = canvas
    count = len(strides)
    if len(outputs) != 3 * count:
        raise ValueError(f"SCRFD with landmarks gives {3 * count} outputs, not {len(outputs)}")
    boxes, scores, marks = [], [], []
    for k, stride in enumerate(strides):
        score = np.asarray(outputs[k]).reshape(-1)
        distance = np.asarray(outputs[k + count]).reshape(-1, 4) * stride
        offsets = np.asarray(outputs[k + 2 * count]).reshape(-1, 10) * stride
        centres = anchor_centres(height // stride, width // stride, stride, anchors)
        if len(centres) != len(score):
            raise ValueError(f"stride {stride}: {len(score)} scores for {len(centres)} anchors (wrong canvas?)")
        keep = np.nonzero(score >= threshold)[0]
        c, d = centres[keep], distance[keep]
        boxes.append(np.c_[c[:, 0] - d[:, 0], c[:, 1] - d[:, 1], c[:, 0] + d[:, 2], c[:, 1] + d[:, 3]])
        marks.append(offsets[keep].reshape(-1, 5, 2) + c[:, None, :])
        scores.append(score[keep])
    boxes, scores, marks = np.concatenate(boxes), np.concatenate(scores), np.concatenate(marks)
    order = np.argsort(-scores, kind="stable")
    return boxes[order], scores[order], marks[order]


def nms(boxes: np.ndarray, scores: np.ndarray, overlap: float = SCRFD_NMS) -> list[int]:
    """The indices of the boxes (x1, y1, x2, y2) to keep, best score first: each box drops the lower-scoring ones
    that overlap it by more than ``overlap`` (intersection over union, with the reference code's "+1" pixel areas)."""
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = np.argsort(-scores, kind="stable")
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        rest = order[1:]
        w = np.maximum(0.0, np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]) + 1)
        h = np.maximum(0.0, np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]) + 1)
        inter = w * h
        order = rest[inter / (areas[i] + areas[rest] - inter) <= overlap]
    return keep


class ScrfdDetector:
    """SCRFD-10G with five landmarks (``spec``: :data:`.registry.SCRFD`) at ``path``, run by ONNX Runtime (see the
    module docs). ``canvas``: the default canvas; ``score``: the score a face must reach."""

    def __init__(self, path, spec=None, canvas=640, score: float = SCRFD_SCORE, overlap: float = SCRFD_NMS,
                 provider: str | None = None):
        from . import registry
        from .runtime import OnnxModel

        self.spec = spec or registry.SCRFD
        self.canvas = canvas
        self.score = float(score)
        self.overlap = float(overlap)
        width, height = canvas_size((640, 640), canvas if canvas != "image" else 640)
        self.model = OnnxModel(path, (1, 3, height, width), provider=provider)

    def detect(self, image: np.ndarray, min_size: float = 0.0, canvas=None) -> list[Face]:
        """Faces in ``image`` (BGR) at least ``min_size`` pixels across, biggest first."""
        size = canvas_size(image.shape, self.canvas if canvas is None else canvas)
        board, scale = letterbox(image, size)
        blob = (board[..., ::-1].astype(np.float32) - self.spec.mean) / self.spec.std  # BGR -> RGB
        outputs = self.model.run(blob.transpose(2, 0, 1)[None])
        boxes, scores, marks = decode_scrfd(outputs, size, self.score)
        faces = []
        for i in nms(boxes, scores, self.overlap):
            x1, y1, x2, y2 = (boxes[i] / scale).tolist()
            points = tuple(map(tuple, (marks[i] / scale).tolist()))
            faces.append(Face((x1, y1, x2 - x1, y2 - y1), points, float(scores[i])))
        return _biggest_first(faces, min_size)


class YuNetDetector:
    """YuNet (``spec``: :data:`.registry.YUNET`) at ``path``, run by OpenCV's ``FaceDetectorYN`` (see the module
    docs); thread safe (a lock: OpenCV's detector keeps state between calls)."""

    def __init__(self, path, spec=None, score: float = YUNET_SCORE, overlap: float = YUNET_NMS):
        from . import registry

        self.spec = spec or registry.YUNET
        self.canvas = None
        self._detector = cv2.FaceDetectorYN.create(str(path), "", (320, 320), float(score), float(overlap), 5000)
        self._lock = threading.Lock()

    def detect(self, image: np.ndarray, min_size: float = 0.0, canvas=None) -> list[Face]:
        """Faces in ``image`` (BGR) at least ``min_size`` pixels across, biggest first. ``canvas``: an int scales
        the image to fit in that square (up or down); default: the image itself, scaled down above
        :data:`MAX_SIDE`."""
        h, w = image.shape[:2]
        if canvas is None or canvas == "image":
            scale = min(1.0, MAX_SIDE / max(h, w))
        else:
            width, height = canvas_size(image.shape, canvas)
            scale = min(width / w, height / h)
        if scale != 1.0:
            interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
            image = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))),
                               interpolation=interpolation)
        with self._lock:
            self._detector.setInputSize((image.shape[1], image.shape[0]))
            _, rows = self._detector.detect(np.ascontiguousarray(image))
        faces = []
        for row in rows if rows is not None else []:
            row = row.astype(np.float64)
            sx, sy = w / image.shape[1], h / image.shape[0]
            box = (row[0] * sx, row[1] * sy, row[2] * sx, row[3] * sy)
            marks = tuple((float(x * sx), float(y * sy)) for x, y in row[4:14].reshape(5, 2))
            faces.append(Face(tuple(map(float, box)), marks, float(row[14])))
        return _biggest_first(faces, min_size)
