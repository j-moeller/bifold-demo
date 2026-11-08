"""Finding faces in an image and turning them into embeddings, with the models of a profile (:mod:`.registry`).

A :class:`FaceEngine` puts the stages together: the profile's detector (:mod:`.detect`), the ArcFace alignment
(:mod:`.align`), its embedder (:mod:`.embed`; the first one of the profile that can be had, has its registered
SHA-256 and loads, so ``best`` falls back to ``buffalo_l``'s ResNet-50 when AdaFace is missing or damaged) and its
quality model (:mod:`.quality`). The embedder and the quality model are found, downloaded and loaded when first
needed, so code that only finds faces (painting a photo onto an avatar) needs neither, offline too.

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
FALLBACK_CANVASES = (640, 320)  # tried in turn when SCRFD finds no face in a whole image at its own size

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
    ``quality``: also use the profile's quality model.

    Only the detector is loaded here. The recogniser and the quality model are found when first needed: the file
    (downloaded if allowed), its registered SHA-256 (checked once per process, so a damaged file of the right size is
    caught) and a model that loads; when the profile's first recogniser fails any of these, the next one is used and
    :attr:`notes` says why. Without a usable model, the error names the commercial profile, which needs only YuNet
    and SFace."""

    def __init__(self, models_folder=None, download: bool = True, score: float | None = DETECT_SCORE,
                 profile: str | None = None, provider: str | None = None, quality: bool = True, progress=print):
        self.profile = registry.profile(profile)
        self.models_folder = models_folder
        self.download = bool(download)
        self.provider = provider
        self.progress = progress
        self.notes: list[str] = []  # what was chosen instead of what, for the user
        self._lock = threading.RLock()
        self.detector_spec = spec = registry.model(self.profile.detector)
        kwargs = {} if score is None else {"score": score}
        try:
            path = self._file(spec)
            if spec.form == "opencv":
                self.detector = YuNetDetector(path, spec, **kwargs)
            else:
                self.detector = ScrfdDetector(path, spec, canvas="image", provider=provider, **kwargs)
        except Exception as exc:  # missing, damaged or not loading: say what to do instead
            raise FileNotFoundError(f"the face detector of the profile {self.profile.name} ({spec.title}) cannot "
                                    f"be had: {exc}{self._hint()}") from exc
        self._embedder = self._embedder_spec = self._embedder_path = None
        self._quality = bool(quality) and bool(self.profile.quality)
        self._fiqa = self._fiqa_spec = None
        self._fiqa_checked = False

    def _file(self, spec):
        """The file of ``spec`` (downloaded if allowed) with its registered SHA-256 (hashed once per process)."""
        from .. import downloads

        path = registry.path(spec, self.models_folder, self.download, self.progress)
        if not downloads.verified(path, spec.file.sha256):
            raise OSError(f"{path} is damaged: its SHA-256 is not the registered one (fetch it again: bifold-faces "
                          f"fetch --model {spec.id})")
        return path

    def _hint(self) -> str:
        """What to do when a model of the profile cannot be had (offline, say): the commercial profile."""
        if self.profile.name == "commercial":
            return ""
        from .. import downloads
        from .models import models_dir

        commercial = registry.PROFILES["commercial"]
        folder = models_dir(self.models_folder)
        needed = [registry.model(commercial.detector), registry.model(commercial.embedders[-1])]
        here = all(downloads.present(spec.file, folder) is not None for spec in needed)
        where = f"both are in {folder}" if here else "0.2 and 39 MB to download"
        return (f". The commercial profile needs only YuNet and SFace ({where}): --face-profile commercial or "
                f"BIFOLD_FACE_PROFILE=commercial")

    def _resolve_embedder(self) -> None:
        with self._lock:
            if self._embedder is not None:
                return
            errors = []
            for model_id in self.profile.embedders:
                spec = registry.model(model_id)
                try:
                    path = self._file(spec)
                    made = embedder(spec, path, self.provider)
                except Exception as exc:  # noqa: BLE001 - missing, damaged or not loading: the next one
                    errors.append(f"{spec.title}: {exc}")
                    continue
                if errors:
                    self.notes.append(f"using {spec.title} ({'; '.join(errors)})")
                self._embedder, self._embedder_spec, self._embedder_path = made, spec, path
                return
            raise FileNotFoundError(f"no face recogniser of the profile {self.profile.name} can be had: "
                                    f"{'; '.join(errors)}{self._hint()}")

    def _resolve_fiqa(self) -> None:
        with self._lock:
            if self._fiqa_checked:
                return
            self._fiqa_checked = True
            if not self._quality:
                return
            spec = registry.model(self.profile.quality)
            try:
                self._fiqa = FiqaModel(self._file(spec), spec, self.provider)
                self._fiqa_spec = spec
            except Exception as exc:  # noqa: BLE001 - (ONNX Runtime's errors too) the geometric measures still work
                self.notes.append(f"no {spec.title}: {exc}")

    # -- the models -------------------------------------------------------------------------------------------------

    @property
    def embedder_spec(self):
        """The recogniser in use (found when first asked for, see the class docs)."""
        self._resolve_embedder()
        return self._embedder_spec

    @property
    def embedder_path(self):
        self._resolve_embedder()
        return self._embedder_path

    @property
    def embedder(self):
        self._resolve_embedder()
        return self._embedder

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
    def quality_spec(self):
        """The quality model in use, or ``None`` (the profile has none, ``quality=False``, or it cannot be had or
        loaded: then a note says why)."""
        self._resolve_fiqa()
        return self._fiqa_spec

    @property
    def fiqa(self) -> FiqaModel | None:
        """The quality model, or ``None`` (see :attr:`quality_spec`)."""
        self._resolve_fiqa()
        return self._fiqa

    def describe(self) -> str:
        """``best: SCRFD-10G ... + AdaFace ...`` for the user (finds the recogniser if it is not found yet)."""
        parts = [self.detector_spec.title, self.embedder_spec.title]
        if self.quality_spec is not None:
            parts.append(self.quality_spec.title)
        return f"{self.profile.name}: {' + '.join(parts)}"

    # -- faces -----------------------------------------------------------------------------------------------------

    def detect(self, image: np.ndarray, min_size: float = MIN_FACE, canvas=None, region=None) -> list[Face]:
        """Faces in ``image`` (BGR) at least ``min_size`` pixels across, biggest first. ``canvas``: what the
        detector looks at (see :mod:`.detect`; default: the whole image at its own resolution, and when SCRFD finds
        nothing there, the image shrunk to :data:`FALLBACK_CANVASES`, 640 and then 320 pixels: on the test photos
        SCRFD misses most faces filling more than three quarters of a 640 canvas's height, and finds them all at
        320). ``region`` ``(x, y, w, h)``: look only there (a person's head), scaled to ``canvas`` (default
        :data:`HEAD_CANVAS`); the faces come back in the image's pixels."""
        if region is not None:
            h, w = image.shape[:2]
            x0, y0 = max(0, round(region[0])), max(0, round(region[1]))
            x1, y1 = min(w, round(region[0] + region[2])), min(h, round(region[1] + region[3]))
            if x1 - x0 < 16 or y1 - y0 < 16:  # (no usable face in fewer pixels)
                return []
            found = self.detector.detect(image[y0:y1, x0:x1], min_size, HEAD_CANVAS if canvas is None else canvas)
            return [f.mapped(1.0, (x0, y0)) for f in found]
        faces = self.detector.detect(image, min_size, canvas)
        if canvas is None and isinstance(self.detector, ScrfdDetector):
            for smaller in FALLBACK_CANVASES:
                if faces:
                    break
                if max(image.shape[:2]) > 1.25 * smaller:  # (else much the same canvas as the image's own)
                    faces = self.detector.detect(image, min_size, smaller)
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
