"""MediaPipe for the head texture: the Face Landmarker and the multiclass selfie segmenter, behind small wrappers.

* :class:`Landmarker` finds faces and puts MediaPipe's 478 landmarks on each (see :mod:`.canonical`), with the
  head's pose as MediaPipe estimates it (its *facial transformation matrix*) and 52 expression scores
  (blendshapes: ``jawOpen``, ``eyeBlinkLeft``, ...). About 20 ms per image on a CPU.
* :class:`Segmenter` labels every pixel of a photo as background, hair, body skin, face skin, clothes or other
  (accessories), as confidences from 0 to 1. About 120 ms per image.

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

MODELS_URL = "https://storage.googleapis.com/mediapipe-models/"
FACE_LANDMARKER = ModelFile("face_landmarker.task", MODELS_URL + "face_landmarker/face_landmarker/float16/1/"
                            "face_landmarker.task",
                            "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff", 3_758_596)
SELFIE_MULTICLASS = ModelFile("selfie_multiclass_256x256.tflite", MODELS_URL + "image_segmenter/"
                              "selfie_multiclass_256x256/float32/1/selfie_multiclass_256x256.tflite",
                              "c6748b1253a99067ef71f7e26ca71096cd449baefa8f101900ea23016507e0e0", 16_371_837)
FILES = (FACE_LANDMARKER, SELFIE_MULTICLASS)
SEGMENT_CLASSES = ("background", "hair", "body", "face", "clothes", "others")  # the segmenter's labels, in order
SMALL_FACE = 0.15  # faces narrower than this share of a photo's longer side may need a closer look (see detect)
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
    """MediaPipe's Face Landmarker on BGR images."""

    def __init__(self, models=None, download: bool = True, max_faces: int = 1, min_confidence: float = 0.3):
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
        super().__init__(vision.FaceLandmarker.create_from_options(options))

    def detect(self, image: np.ndarray) -> list[FaceMarks]:
        """The faces in ``image`` (BGR), biggest first. MediaPipe looks for faces in a 128-pixel copy of the image,
        so small faces in a big photo are missed: when it finds none, it looks again in overlapping crops (a 2x2 and
        a 3x3 grid) for faces narrower than :data:`SMALL_FACE` of the photo."""
        faces = self._detect(image)
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
                        if not any(np.linalg.norm(f.points[1] - face.points[1]) < 0.3 * f.box[2] for f in faces):
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
        """``{class name: (H, W) float32 confidence}`` for :data:`SEGMENT_CLASSES`."""
        h, w = image.shape[:2]
        with self._lock:
            result = self._open().segment(_rgb_image(self._mp, image))
        masks = {}
        for name, mask in zip(SEGMENT_CLASSES, result.confidence_masks):
            m = np.asarray(mask.numpy_view(), np.float32).reshape(mask.height, mask.width)
            masks[name] = m if m.shape == (h, w) else cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
        return masks
