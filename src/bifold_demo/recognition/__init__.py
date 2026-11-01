"""Face recognition: who the people in the picture are.

Faces are found by YuNet and turned into a 128-number embedding (a face's "fingerprint") by SFace, both small
OpenCV Zoo models that OpenCV runs itself (MIT / Apache 2.0, see :mod:`.models`). The embeddings of a face seen in
different photos lie close together: their cosine similarity is high (about 0.6-1.0 for the same person, below
about 0.3 for others). A :class:`~.gallery.Gallery` keeps the embeddings of the photos of known people (a folder per
person) and names a face by its nearest match above :data:`~.gallery.THRESHOLD`.

The new recognition core is being built beside this (see docs/research/face_recognition_focus.md, step F1):
:mod:`.registry` lists every face model with its checked download, licence and profile (``best``, ``licensed``,
``commercial``), and :mod:`.runtime` runs ONNX files with ONNX Runtime. Importing this package switches ONNX Runtime's
telemetry off (``ORT_DISABLE_TELEMETRY=1``, unless set already) before anything can import ONNX Runtime.
"""

import os

os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")  # before ONNX Runtime is first imported

from .engine import Face, FaceEngine  # (after the telemetry switch)
from .gallery import THRESHOLD, Gallery, Match

__all__ = ["THRESHOLD", "Face", "FaceEngine", "Gallery", "Match"]
