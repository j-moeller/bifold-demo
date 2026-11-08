"""Turning aligned faces into embeddings, the same way for every model.

An embedder takes a batch of aligned 112 x 112 crops (:mod:`.align`; in the image's blue-green-red order) and gives
:class:`Embeddings`: a unit vector per crop, so that the dot product of two is their cosine similarity, and the raw
length of each embedding before it was made unit length. AdaFace learns that length as a measure of how recognisable
a face is (its loss widens the margin for faces with long embeddings), so for AdaFace it is a quality signal
(``ModelSpec.norm_quality``); for the other models it is kept but means little.

* :class:`OnnxEmbedder` runs an ONNX file with ONNX Runtime: the crops in the model's colour order, ``(pixel - mean)
  / std``, all in one run on the CPU, in batches of 8 and then single crops with CoreML (see :mod:`.runtime`). The
  AdaFace export divides its embedding by its length inside the graph; that length is read out of the graph as an
  extra output.
* :class:`SFaceEmbedder` runs SFace with OpenCV's ``FaceRecognizerSF`` (its graph normalises its input itself), one
  crop at a time.

``flip=True`` sums the raw embeddings of each crop and its mirror image, as the verification benchmarks (LFW and
others) are usually run; the live path does not need it. Embeddings of different models, even of the same length,
cannot be compared: anything made from them is kept per model (:attr:`OnnxEmbedder.id`).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

from .runtime import OnnxModel, batches


@dataclass(frozen=True)
class Embeddings:
    """``vectors`` ``(N, dim)``: unit length (float32); ``norms`` ``(N,)``: the raw embeddings' lengths."""

    vectors: np.ndarray
    norms: np.ndarray

    def __len__(self) -> int:
        return len(self.vectors)


def _unit(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    norms = np.linalg.norm(raw, axis=1)
    return (raw / np.maximum(norms, 1e-12)[:, None]).astype(np.float32), norms.astype(np.float32)


def _check(crops: np.ndarray, size) -> np.ndarray:
    crops = np.asarray(crops)
    if crops.ndim == 3:
        crops = crops[None]
    if crops.ndim != 4 or crops.shape[1:3] != (size[1], size[0]) or crops.shape[3] != 3:
        raise ValueError(f"expected aligned crops of shape (N, {size[1]}, {size[0]}, 3), not {crops.shape}")
    return crops


class OnnxEmbedder:
    """The embedder ``spec`` (a :class:`~.registry.ModelSpec`) at ``path``, run by ONNX Runtime. ``bgr`` overrides
    the colour order the registry gives (for the colour-order check)."""

    def __init__(self, path, spec, provider: str | None = None, bgr: bool | None = None):
        self.spec = spec
        self.bgr = spec.bgr if bgr is None else bool(bgr)
        width, height = spec.size
        extra = (spec.norm_output,) if spec.norm_output else ()
        self.model = OnnxModel(path, (1, 3, height, width), extra_outputs=extra, provider=provider)

    @property
    def id(self) -> str:
        return self.spec.id

    @property
    def dim(self) -> int:
        return self.spec.dim

    def blob(self, crops: np.ndarray) -> np.ndarray:
        """The network's input for ``crops`` (BGR): ``(N, 3, height, width)`` float32."""
        x = crops.astype(np.float32)
        if not self.bgr:
            x = x[..., ::-1]
        x = (x - self.spec.mean) / self.spec.std
        return np.ascontiguousarray(x.transpose(0, 3, 1, 2))

    def raw(self, crops: np.ndarray) -> np.ndarray:
        """The raw (not normalised) embeddings of ``crops``: ``(N, dim)``."""
        crops = _check(crops, self.spec.size)
        out = np.zeros((len(crops), self.dim), np.float32)
        for part in batches(len(crops), self.model.fixed):
            outputs = self.model.run(self.blob(crops[part]))
            vectors = np.asarray(outputs[0], np.float32).reshape(-1, self.dim)
            if self.spec.norm_output:  # the graph's output is unit length; its length is the extra output
                vectors = vectors * np.asarray(outputs[1], np.float32).reshape(-1, 1)
            out[part] = vectors
        return out

    def embed(self, crops: np.ndarray, flip: bool = False) -> Embeddings:
        """Unit embeddings and raw lengths of ``crops`` (``(N, 112, 112, 3)`` BGR, aligned; see the module docs)."""
        crops = _check(crops, self.spec.size)
        raw = self.raw(crops)
        vectors, norms = _unit(raw + self.raw(crops[:, :, ::-1]) if flip else raw)
        return Embeddings(vectors, np.linalg.norm(raw, axis=1).astype(np.float32) if flip else norms)


class SFaceEmbedder:
    """SFace (``spec``) at ``path``, run by OpenCV's ``FaceRecognizerSF``; thread safe (a lock: OpenCV's objects keep
    state between calls)."""

    def __init__(self, path, spec):
        import cv2

        self.spec = spec
        self.bgr = spec.bgr
        self._recognizer = cv2.FaceRecognizerSF.create(str(path), "")
        self._lock = threading.Lock()

    @property
    def id(self) -> str:
        return self.spec.id

    @property
    def dim(self) -> int:
        return self.spec.dim

    def raw(self, crops: np.ndarray) -> np.ndarray:
        crops = _check(crops, self.spec.size)
        out = np.zeros((len(crops), self.dim), np.float32)
        with self._lock:
            for i, crop in enumerate(crops):
                out[i] = self._recognizer.feature(np.ascontiguousarray(crop)).ravel()
        return out

    def embed(self, crops: np.ndarray, flip: bool = False) -> Embeddings:
        crops = _check(crops, self.spec.size)
        raw = self.raw(crops)
        vectors, norms = _unit(raw + self.raw(crops[:, :, ::-1]) if flip else raw)
        return Embeddings(vectors, np.linalg.norm(raw, axis=1).astype(np.float32) if flip else norms)


def embedder(spec, path, provider: str | None = None):
    """The embedder for ``spec`` with its file at ``path``."""
    if spec.task != "embedder":
        raise ValueError(f"{spec.id} is a {spec.task}, not an embedder")
    if spec.form == "opencv":
        return SFaceEmbedder(path, spec)
    return OnnxEmbedder(path, spec, provider)
