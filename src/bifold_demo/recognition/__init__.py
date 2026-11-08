"""Face recognition: who the people in the picture are.

The core (docs/research/face_recognition_focus.md, step F1), stage by stage:

* :mod:`.registry`: every face model with its checked download, input, output, licence and provenance, and the
  three profiles behind one switch (``--face-profile`` or ``BIFOLD_FACE_PROFILE``): ``best`` (SCRFD-10G and AdaFace
  IR-101; research-only weights), ``licensed`` (SCRFD-10G and InsightFace's ``buffalo_l`` ResNet-50) and
  ``commercial`` (YuNet with AuraFace-v1 where Hugging Face can be reached, else SFace).
* :mod:`.detect`: faces with a box, five landmarks and a score, the same from every detector.
* :mod:`.align`: the ArcFace five-point similarity transform to a 112 x 112 crop.
* :mod:`.embed`: a batch of crops to unit embeddings (a face's "fingerprint") and their raw lengths. Two photos of one
  person give embeddings with a high cosine similarity (their dot product), other people's a low one.
* :mod:`.quality`: a :class:`~.quality.Quality` record per face (eye distance, pose, sharpness, light, eDifFIQA-T's
  score, AdaFace's norm).
* :mod:`.engine`: :class:`FaceEngine`, the stages of one profile together.
* :mod:`.gallery`: the known people (a folder of photos per person), checked at enrolment, cached per model, and
  matched by photo and by pose; the decision whom a face shows is a replaceable step.
* :mod:`.checks`: checking every model before trusting it (``bifold-faces check``).
* :mod:`.runtime`: ONNX Runtime sessions. Importing this package switches ONNX Runtime's telemetry off
  (``ORT_DISABLE_TELEMETRY=1``, unless set already) before anything can import ONNX Runtime.

:mod:`.recognizer` follows faces over the frames of the demo and :mod:`.cast` puts the known people's faces on the 3D
people.
"""

import os

os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")  # before ONNX Runtime is first imported

from .engine import Face, FaceEngine, FaceRecord  # (after the telemetry switch)
from .gallery import THRESHOLD, Gallery, Match, PersonScore  # (THRESHOLD: SFace's; see gallery.THRESHOLD)

__all__ = ["THRESHOLD", "Face", "FaceEngine", "FaceRecord", "Gallery", "Match", "PersonScore"]
