"""Finding faces in an image and turning them into embeddings, with the models of a profile (:mod:`.registry`).

A :class:`FaceEngine` puts the stages together: the profile's detector (:mod:`.detect`), the ArcFace alignment
(:mod:`.align`), its embedder (:mod:`.embed`; the first one of the profile that is there, so ``best`` falls back to
``buffalo_l``'s ResNet-50 when AdaFace cannot be had) and its quality model (:mod:`.quality`). The embedder and the
quality model are loaded when first needed, so code that only finds faces (painting a photo onto an avatar) does not
pay for a 260 MB recogniser.

* :meth:`FaceEngine.detect`, :meth:`~FaceEngine.embed` and :meth:`~FaceEngine.faces` are the demo's old interface
  (biggest face first; a unit embedding per face).
* :meth:`FaceEngine.analyse` gives a :class:`FaceRecord` per face (the face, its aligned crop, its embedding, the
  embedding's raw length and a :class:`~.quality.Quality`), embedding all faces of an image in one batch; this is
  what the person tracks of the next step (F2) build on. With ``region`` it looks only inside a part of the frame (a
  person's head, scaled to :data:`HEAD_CANVAS` pixels for the detector) and gives the faces in the frame's pixels,
  aligned from the frame itself.

Embeddings of different models cannot be compared: :attr:`FaceEngine.model_id` names the model, and everything made
from the embeddings (the gallery's caches, thresholds, calibration) is kept per model. :attr:`FaceEngine.threshold`
is the model's provisional cosine cut (``ModelSpec.threshold``) until F2 calibrates the decisions. Thread safe.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

from . import registry
from .align import Aligned, align, crops
from .detect import Face, ScrfdDetector, YuNetDetector
from .embed import Embeddings, embedder
from .quality import FiqaModel, Quality, assess

MIN_FACE = 24  # pixels: smaller faces are found but too small to recognise reliably
DETECT_SCORE = None  # the detector's own default (SCRFD 0.5, YuNet 0.7)
HEAD_CANVAS = 192  # pixels: the canvas a person's head region is scaled to (the review suggests 160-192)

__all__ = ["DETECT_SCORE", "MIN_FACE", "Face", "FaceEngine", "FaceRecord"]


@dataclass(frozen=True)
class FaceRecord:
    """One face in an image, analysed: the ``face``, its ``aligned`` crop, its unit ``vector`` (the embedding), the
    embedding's raw ``norm`` and its ``quality``."""

    face: Face
    aligned: Aligned
    vector: np.ndarray
    norm: float
    quality: Quality


class FaceEngine:
    """The face recognition's models for ``profile`` (default: ``$BIFOLD_FACE_PROFILE``, else ``best``), from
    ``models_folder`` (see :mod:`.models`), downloaded when missing if ``download``. ``score``: the detection score
    a face must reach (default: the detector's own); ``provider``: ``cpu`` or ``coreml`` (default: automatic);
    ``quality``: also use the profile's quality model."""

    def __init__(self, models_folder=None, download: bool = True, score: float | None = DETECT_SCORE,
                 profile: str | None = None, provider: str | None = None, quality: bool = True, progress=print):
        self.profile = registry.profile(profile)
        self.models_folder = models_folder
        self.provider = provider
        self.notes: list[str] = []  # what was chosen instead of what, for the user
        spec = registry.model(self.profile.detector)
        path = registry.path(spec, models_folder, download, progress)
        kwargs = {} if score is None else {"score": score}
        if spec.form == "opencv":
            self.detector = YuNetDetector(path, spec, **kwargs)
        else:
            self.detector = ScrfdDetector(path, spec, canvas="image", provider=provider, **kwargs)
        self.embedder_spec, self._embedder_path = self._choose(self.profile.embedders, download, progress)
        self.quality_spec = None
        if quality and self.profile.quality:
            spec = registry.model(self.profile.quality)
            try:
                self._fiqa_path = registry.path(spec, models_folder, download, progress)
                self.quality_spec = spec
            except OSError as exc:  # the geometric measures still work
                self.notes.append(f"no {spec.title}: {exc}")
        self._embedder = None
        self._fiqa = None
        self._lock = threading.Lock()

    def _choose(self, ids, download, progress):
        """The first embedder of ``ids`` whose file is (or can be made) present."""
        errors = []
        for model_id in ids:
            spec = registry.model(model_id)
            try:
                path = registry.path(spec, self.models_folder, download, progress)
            except OSError as exc:  # missing without downloads, or the download failed
                errors.append(f"{spec.title}: {exc}")
                continue
            if errors:
                self.notes.append(f"using {spec.title} ({'; '.join(errors)})")
            return spec, path
        raise FileNotFoundError("no face recogniser of the profile "
                                f"{self.profile.name} is available: {'; '.join(errors)}")

    # -- the models -------------------------------------------------------------------------------------------------

    @property
    def model_id(self) -> str:
        """The embedder's id: the gallery's caches and every threshold are kept per model."""
        return self.embedder_spec.id

    @property
    def dim(self) -> int:
        return self.embedder_spec.dim

    @property
    def threshold(self) -> float:
        """The embedder's provisional cosine cut (see the module docs)."""
        return float(self.embedder_spec.threshold)

    @property
    def provider_name(self) -> str:
        """What computes the embeddings: ``opencv``, or ONNX Runtime's ``cpu`` or ``coreml`` (embeddings made by
        different providers differ slightly, so the gallery's caches remember it)."""
        if self.embedder_spec.form == "opencv":
            return "opencv"
        from .runtime import provider_choice

        return provider_choice(self.provider)

    @property
    def embedder(self):
        with self._lock:
            if self._embedder is None:
                self._embedder = embedder(self.embedder_spec, self._embedder_path, self.provider)
            return self._embedder

    @property
    def fiqa(self) -> FiqaModel | None:
        """The quality model, or ``None`` (the profile has none, or its file cannot be had: then a note says so)."""
        if self.quality_spec is None:
            return None
        with self._lock:
            if self._fiqa is None:
                try:
                    self._fiqa = FiqaModel(self._fiqa_path, self.quality_spec, self.provider)
                except Exception as exc:  # noqa: BLE001 - (ONNX Runtime's errors too) geometry still works
                    self.notes.append(f"no {self.quality_spec.title}: {exc}")
                    self.quality_spec = None
                    return None
            return self._fiqa

    def describe(self) -> str:
        """``best: SCRFD-10G ... + AdaFace ...`` for the user."""
        parts = [registry.model(self.profile.detector).title, self.embedder_spec.title]
        if self.quality_spec is not None:
            parts.append(self.quality_spec.title)
        return f"{self.profile.name}: {' + '.join(parts)}"

    # -- faces -----------------------------------------------------------------------------------------------------

    def detect(self, image: np.ndarray, min_size: float = MIN_FACE, canvas=None, region=None) -> list[Face]:
        """Faces in ``image`` (BGR) at least ``min_size`` pixels across, biggest first. ``canvas``: what the
        detector looks at (see :mod:`.detect`; default: the whole image at its own resolution, and when that finds
        nothing in a big image, the image shrunk to 640 pixels, where a face filling a close-up photo is the size
        SCRFD knows best). ``region`` ``(x, y, w, h)``: look only there (a person's head), scaled to ``canvas``
        (default :data:`HEAD_CANVAS`); the faces come back in the image's pixels."""
        if region is not None:
            h, w = image.shape[:2]
            x0, y0 = max(0, round(region[0])), max(0, round(region[1]))
            x1, y1 = min(w, round(region[0] + region[2])), min(h, round(region[1] + region[3]))
            if x1 - x0 < 16 or y1 - y0 < 16:  # (no usable face in fewer pixels)
                return []
            found = self.detector.detect(image[y0:y1, x0:x1], min_size, HEAD_CANVAS if canvas is None else canvas)
            return [f.mapped(1.0, (x0, y0)) for f in found]
        faces = self.detector.detect(image, min_size, canvas)
        if not faces and canvas is None and isinstance(self.detector, ScrfdDetector) and max(image.shape[:2]) > 800:
            faces = self.detector.detect(image, min_size, 640)
        return faces

    def align(self, image: np.ndarray, face: Face) -> Aligned:
        return align(image, face.landmarks)

    def embed_faces(self, image: np.ndarray, faces) -> Embeddings:
        """The embeddings of ``faces`` in ``image``, in one batch."""
        aligned = [self.align(image, f) for f in faces]
        return self.embedder.embed(crops(aligned))

    def embed(self, image: np.ndarray, face: Face) -> np.ndarray:
        """The unit-length embedding of ``face`` in ``image`` (float64)."""
        return self.embed_faces(image, [face]).vectors[0].astype(np.float64)

    def faces(self, image: np.ndarray, min_size: float = MIN_FACE) -> list[tuple[Face, np.ndarray]]:
        """``(face, embedding)`` for every face in ``image``."""
        found = self.detect(image, min_size)
        if not found:
            return []
        vectors = self.embed_faces(image, found).vectors.astype(np.float64)
        return list(zip(found, vectors))

    def analyse(self, image: np.ndarray, faces=None, min_size: float = 0.0, canvas=None, region=None,
                quality: bool = True) -> list[FaceRecord]:
        """A :class:`FaceRecord` per face of ``image`` (``faces``, else every face found, in ``region`` if given:
        see :meth:`detect`), aligned from the image itself and embedded in one batch; ``quality``: also eDifFIQA's
        score where the profile has it (the geometric measures are always there)."""
        found = list(faces) if faces is not None else self.detect(image, min_size, canvas, region)
        if not found:
            return []
        aligned = [self.align(image, f) for f in found]
        stack = crops(aligned)
        embedded = self.embedder.embed(stack)
        fiqa = self.fiqa if quality else None
        scores = fiqa.score(stack) if fiqa is not None else [None] * len(found)
        use_norm = self.embedder_spec.norm_quality
        return [FaceRecord(f, a, embedded.vectors[i], float(embedded.norms[i]),
                           assess(f, a, scores[i], float(embedded.norms[i]) if use_norm else None))
                for i, (f, a) in enumerate(zip(found, aligned))]
