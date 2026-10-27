"""MediaPipe for the head texture: the Face Landmarker and the multiclass selfie segmenter, behind small wrappers.

* :class:`Landmarker` finds faces and puts MediaPipe's 478 landmarks on each (see :mod:`.canonical`), with the
  head's pose as MediaPipe estimates it (its *facial transformation matrix*) and 52 expression scores
  (blendshapes: ``jawOpen``, ``eyeBlinkLeft``, ...). About 20 ms per image on a CPU. With the face recognition's
  YuNet detector (``bifold-faces fetch``) it also finds small faces in big photos, and gives each face YuNet's
  confidence as a second opinion.
* :class:`Segmenter` labels every pixel of a photo as background, hair, body skin, face skin, clothes or other
  (accessories), as confidences from 0 to 1. About 120 ms per image. The masks are kept small (:func:`compact_mask`:
  0-255 in one byte, at most :data:`MASK_SIZE` pixels on the longer side, as the segmenter sees the photo at 256 x
  256 pixels anyway) and are copies of MediaPipe's: it reuses its buffers on the next call. :func:`mask_at` gives one
  at any size.

Both models come from Google's MediaPipe model bucket (Apache 2.0) and are downloaded into the face models folder
(:func:`bifold_demo.recognition.models.models_dir`) on first use, checked by size and SHA-256.

MediaPipe itself is an optional dependency (``pip install --no-deps mediapipe``, see the README: its package asks for
``opencv-contrib-python``, which must not be installed beside the project's ``opencv-python``). Everything else in the
demo works without it; :func:`mediapipe` raises :class:`MediaPipeMissing` with the install command when a feature
needs it. Anything with a ``detect(image) -> list[FaceMarks]`` method can stand in for :class:`Landmarker` (the tests
use fakes).
"""

from __future__ import annotations

import atexit
import threading
import weakref
from dataclasses import dataclass, field

import cv2
import numpy as np

from ...downloads import ModelFile, ensure
from .canonical import NOSE_TIP

MODELS_URL = "https://storage.googleapis.com/mediapipe-models/"
FACE_LANDMARKER = ModelFile("face_landmarker.task", MODELS_URL + "face_landmarker/face_landmarker/float16/1/"
                            "face_landmarker.task",
                            "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff", 3_758_596)
SELFIE_MULTICLASS = ModelFile("selfie_multiclass_256x256.tflite", MODELS_URL + "image_segmenter/"
                              "selfie_multiclass_256x256/float32/1/selfie_multiclass_256x256.tflite",
                              "c6748b1253a99067ef71f7e26ca71096cd449baefa8f101900ea23016507e0e0", 16_371_837)
FILES = (FACE_LANDMARKER, SELFIE_MULTICLASS)
SEGMENT_CLASSES = ("background", "hair", "body", "face", "clothes", "others")  # the segmenter's labels, in order
MASK_SIZE = 1024  # pixels: masks are kept at most this big on the longer side
SMALL_FACE = 0.15  # faces narrower than this share of a photo's longer side may need a closer look (see detect)
SMALL_CROP = 3.0  # ... in a crop this many times their size, round a face YuNet finds
YUNET_SCORE, YUNET_SIDE = 0.6, 1600  # YuNet's confidence for a face to look at; the longest copy it looks at
INSTALL = 'pip install -e ".[headtex]" && pip install --no-deps "mediapipe>=1.0"'


class MediaPipeMissing(ImportError):
    """MediaPipe is not installed (or cannot load); the message says how to install it."""


def mediapipe():
    """The ``mediapipe`` module, or :class:`MediaPipeMissing` explaining how to get it."""
    try:
        import mediapipe as mp
        from mediapipe.tasks.python import vision  # noqa: F401 - the part we use, imported to check it loads
    except ImportError as exc:
        raise MediaPipeMissing(f"head textures from photos need MediaPipe ({exc}); install it with: {INSTALL}") \
            from exc
    except OSError as exc:  # its native library: on Linux it needs libEGL and libGLESv2
        raise MediaPipeMissing(f"MediaPipe cannot load its native library ({exc}); on Linux install libEGL and "
                               "libGLESv2 (Debian/Ubuntu: sudo apt install libegl1 libgles2)") from exc
    return mp


def model_path(model: ModelFile, folder=None, download: bool = True, progress=print):
    from ...recognition.models import models_dir

    return ensure(model, models_dir(folder), download, progress, fetch_hint="bifold-faces fetch --headtex",
                  tag="headtex")


@dataclass
class FaceMarks:
    """One face's 478 landmarks in a photo."""

    points: np.ndarray  # (478, 2) pixels (pixel centres at whole numbers)
    depth: np.ndarray  # (478,) MediaPipe's relative depth, in pixels like x (smaller: nearer the camera)
    matrix: np.ndarray | None = None  # 4x4: the canonical face (cm) to MediaPipe's camera (looking along -z)
    blendshapes: dict = field(default_factory=dict)  # expression scores 0-1, e.g. "jawOpen"
    detector_score: float | None = None  # a second opinion: YuNet's confidence in a face round the nose tip (0: none
    # above YUNET_SCORE there; None: not asked)

    @property
    def box(self) -> tuple[float, float, float, float]:
        lo, hi = self.points.min(axis=0), self.points.max(axis=0)
        return float(lo[0]), float(lo[1]), float(hi[0] - lo[0]), float(hi[1] - lo[1])

    @property
    def eye_distance(self) -> float:
        """Pixels between the outer eye corners."""
        from .canonical import LEFT_EYE_OUTER, RIGHT_EYE_OUTER

        return float(np.linalg.norm(self.points[LEFT_EYE_OUTER] - self.points[RIGHT_EYE_OUTER]))


def _rgb_image(mp, image: np.ndarray):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))


_open_tasks: weakref.WeakSet = weakref.WeakSet()


def _drop_at_exit() -> None:
    """Let go of the MediaPipe tasks still open while the interpreter is intact. This module registers it before
    MediaPipe is imported, so it runs after MediaPipe's own exit hook: their clean-up then knows the program ends
    (left to the interpreter's final tear-down, it fails with a traceback)."""
    for task in list(_open_tasks):
        task._task = None


atexit.register(_drop_at_exit)


class _Task:
    """A MediaPipe task object (closed by :meth:`close`, at the end of a ``with`` block or when it goes away)."""

    def __init__(self, task):
        self._task = task
        self._lock = threading.Lock()  # MediaPipe tasks take one image at a time
        _open_tasks.add(self)

    def close(self) -> None:
        task, self._task = self._task, None
        if task is not None:
            task.close()

    def _open(self):
        if self._task is None:
            raise RuntimeError(f"the {type(self).__name__} is closed")
        return self._task

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class Landmarker(_Task):
    """MediaPipe's Face Landmarker on BGR images. ``min_confidence`` is MediaPipe's own default: below it, MediaPipe
    finds more faces in profiles and the backs of heads, all of them wrong (renders of 20 avatars turned 75 to 180
    degrees: "faces" in 35 of 220 at 0.3, in 16 at 0.5), and only a few more real ones (217 against 215 of 220
    turned up to 60 degrees)."""

    def __init__(self, models=None, download: bool = True, max_faces: int = 1, min_confidence: float = 0.5):
        mp = mediapipe()
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions

        path = model_path(FACE_LANDMARKER, models, download)
        self.task_path = path
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path)), num_faces=max_faces,
            output_facial_transformation_matrixes=True, output_face_blendshapes=True,
            min_face_detection_confidence=min_confidence, min_face_presence_confidence=min_confidence)
        self._mp = mp
        self._models, self._download, self._max_faces = models, download, max_faces
        self._boxes = None  # YuNet, made on first need (False: not available)
        super().__init__(vision.FaceLandmarker.create_from_options(options))

    def detect(self, image: np.ndarray) -> list[FaceMarks]:
        """The faces in ``image`` (BGR), biggest first. MediaPipe looks for faces in a 128-pixel copy of the image,
        so small faces in a big photo are missed: when it finds none, YuNet (the face recognition's detector, which
        finds faces down to about 10 pixels in a copy at most 1600 pixels long) says where to look, and MediaPipe
        looks again in a crop three times the size of each face YuNet finds (the biggest three). Without YuNet's
        model it looks in overlapping crops instead (a 2x2 and a 3x3 grid), which finds faces down to about 7 % of the
        photo's longer side. Each face gets YuNet's opinion of it (:attr:`FaceMarks.detector_score`): MediaPipe also
        "finds" faces on the backs of heads, which YuNet mostly does not (it does not see many faces turned 60
        degrees either)."""
        faces = self._detect(image)
        h, w = image.shape[:2]
        rows = self._yunet(image)
        if not faces and max(h, w) >= 320:
            faces = self._in_grid(image) if rows is None else self._around(image, rows)
        if rows is not None:
            for face in faces:
                x, y = face.points[NOSE_TIP]
                holding = (rows[:, 0] <= x) & (x <= rows[:, 0] + rows[:, 2]) & (rows[:, 1] <= y) & \
                    (y <= rows[:, 1] + rows[:, 3])
                face.detector_score = float(rows[holding, 4].max()) if holding.any() else 0.0
        return faces

    def _around(self, image: np.ndarray, rows: np.ndarray) -> list[FaceMarks]:
        """MediaPipe's faces in crops round the biggest faces YuNet found (``rows`` from :meth:`_yunet`)."""
        faces: list[FaceMarks] = []
        h, w = image.shape[:2]
        rows = rows[np.argsort(-rows[:, 2] * rows[:, 3], kind="stable")][:max(3, self._max_faces)]
        for x, y, bw, bh in rows[:, :4].tolist():
            side = min(max(h, w), round(SMALL_CROP * max(bw, bh)))
            x0 = int(np.clip(round(x + bw / 2 - side / 2), 0, max(0, w - side)))
            y0 = int(np.clip(round(y + bh / 2 - side / 2), 0, max(0, h - side)))
            for face in self._detect(image[y0:y0 + side, x0:x0 + side]):
                face.points += (x0, y0)
                nose = face.points[NOSE_TIP]
                if not (x - 0.25 * bw <= nose[0] <= x + 1.25 * bw and y - 0.25 * bh <= nose[1] <= y + 1.25 * bh):
                    continue  # not the face YuNet found
                if not any(np.linalg.norm(f.points[NOSE_TIP] - nose) < 0.3 * f.box[2] for f in faces):
                    faces.append(face)
        faces.sort(key=lambda f: -f.box[2] * f.box[3])
        return faces

    def _yunet(self, image: np.ndarray) -> np.ndarray | None:
        """YuNet's faces in ``image`` with a confidence of at least :data:`YUNET_SCORE`: ``(N, 5)`` rows ``x, y, w,
        h, score`` (pixels of ``image``), or ``None`` without its model (not downloaded, or no downloads allowed)."""
        if self._boxes is None:
            from ...recognition import models as face_models

            try:
                path = face_models.ensure(face_models.DETECTOR, self._models, self._download)
                self._boxes = cv2.FaceDetectorYN.create(str(path), "", (320, 320), YUNET_SCORE, 0.3, 5000)
            except (OSError, cv2.error):
                self._boxes = False
        if self._boxes is False:
            return None
        h, w = image.shape[:2]
        scale = min(1.0, YUNET_SIDE / max(h, w))
        small = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA) \
            if scale < 1.0 else image
        try:
            with self._lock:
                self._boxes.setInputSize((small.shape[1], small.shape[0]))
                _ok, rows = self._boxes.detect(np.ascontiguousarray(small))
        except cv2.error:  # an image too small for it, say
            rows = None
        if rows is None or len(rows) == 0:
            return np.zeros((0, 5))
        rows = np.asarray(rows, np.float64)
        return np.c_[rows[:, :4] / scale, rows[:, -1]]

    def _in_grid(self, image: np.ndarray) -> list[FaceMarks]:
        """Faces narrower than :data:`SMALL_FACE` of the photo, in overlapping crops (a 2x2, then a 3x3 grid)."""
        faces: list[FaceMarks] = []
        h, w = image.shape[:2]
        for cells in (2, 3):
            if faces or max(h, w) < 160 * cells:
                break
            ch, cw = round(1.5 * h / cells), round(1.5 * w / cells)
            for y in np.linspace(0, h - ch, cells).round().astype(int):
                for x in np.linspace(0, w - cw, cells).round().astype(int):
                    for face in self._detect(image[y:y + ch, x:x + cw]):
                        face.points += (x, y)
                        if face.box[2] > SMALL_FACE * max(h, w):  # big enough to be found in the whole photo:
                            continue  # in a crop MediaPipe also "finds" faces in profiles, with a wrong pose
                        if not any(np.linalg.norm(f.points[NOSE_TIP] - face.points[NOSE_TIP]) < 0.3 * f.box[2]
                                   for f in faces):
                            faces.append(face)  # (a face in two crops is kept once)
            faces.sort(key=lambda f: -f.box[2] * f.box[3])
        return faces

    def _detect(self, image: np.ndarray) -> list[FaceMarks]:
        h, w = image.shape[:2]
        with self._lock:
            result = self._open().detect(_rgb_image(self._mp, image))
        faces = []
        for k, marks in enumerate(result.face_landmarks):
            xyz = np.array([(m.x, m.y, m.z) for m in marks], np.float64)
            matrix = (np.asarray(result.facial_transformation_matrixes[k], np.float64)
                      if k < len(result.facial_transformation_matrixes) else None)
            shapes = ({c.category_name: float(c.score) for c in result.face_blendshapes[k]}
                      if k < len(result.face_blendshapes) else {})
            faces.append(FaceMarks(np.c_[xyz[:, 0] * w - 0.5, xyz[:, 1] * h - 0.5], xyz[:, 2] * w, matrix, shapes))
        faces.sort(key=lambda f: -f.box[2] * f.box[3])
        return faces


class Segmenter(_Task):
    """MediaPipe's multiclass selfie segmenter on BGR images."""

    def __init__(self, models=None, download: bool = True):
        mp = mediapipe()
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions

        path = model_path(SELFIE_MULTICLASS, models, download)
        options = vision.ImageSegmenterOptions(base_options=BaseOptions(model_asset_path=str(path)),
                                               output_confidence_masks=True, output_category_mask=False)
        self._mp = mp
        super().__init__(vision.ImageSegmenter.create_from_options(options))

    def segment(self, image: np.ndarray) -> dict[str, np.ndarray]:
        """``{class name: confidence}`` for :data:`SEGMENT_CLASSES`, each a :func:`compact_mask` of the photo
        (:func:`mask_at` gives it at the photo's size)."""
        with self._lock:
            result = self._open().segment(_rgb_image(self._mp, image))
            # MediaPipe's masks are views of its own buffers, which the next call frees or reuses: copied here
            return {name: compact_mask(np.asarray(mask.numpy_view()).reshape(mask.height, mask.width))
                    for name, mask in zip(SEGMENT_CLASSES, result.confidence_masks)}


def compact_mask(mask) -> np.ndarray:
    """A confidence mask (0-1 floats, or 0-255 bytes) the way a :class:`~.register.View` keeps it: a new ``uint8``
    array (0-255) at most :data:`MASK_SIZE` pixels on the longer side."""
    m = np.asarray(mask)
    if m.dtype != np.uint8:
        m = np.asarray(m, np.float32)
    h, w = m.shape[:2]
    scale = MASK_SIZE / max(h, w, 1)
    if scale < 1.0:
        m = cv2.resize(m, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    if m.dtype != np.uint8:
        return np.clip(m * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
    return np.array(m)  # always a copy


def mask_at(mask, size) -> np.ndarray:
    """A confidence mask (0-1 floats or 0-255 bytes, any resolution) as ``(h, w)`` float32 confidences 0-1 for an
    image of ``size`` (w, h)."""
    m = np.asarray(mask)
    m = m.astype(np.float32) / 255.0 if m.dtype == np.uint8 else np.asarray(m, np.float32)
    w, h = int(size[0]), int(size[1])
    if m.shape[:2] != (h, w):
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
    return m
