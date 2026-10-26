"""The face models: downloaded once into the models folder (next to the YOLO weights) and checked.

Both come from the `OpenCV Zoo <https://github.com/opencv/opencv_zoo>`_:

* ``face_detection_yunet_2023mar.onnx`` (YuNet, MIT licence): finds faces and five landmarks (eyes, nose tip, mouth
  corners), about 0.8 AP on WIDER Face "medium", fast on a CPU;
* ``face_recognition_sface_2021dec.onnx`` (SFace, Apache 2.0): a 128-number embedding of an aligned face, about
  99.6 % on LFW.

Stronger recognisers exist (ArcFace / AdaFace ResNet-100s trained on Glint360K or WebFace), but their usual
pretrained weights (InsightFace's ``buffalo_l``, ``antelopev2``) are for non-commercial research only; SFace is free
to use and good enough to tell a handful of known people apart.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

BASE_URL = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"


@dataclass(frozen=True)
class ModelFile:
    name: str
    path: str  # below BASE_URL
    sha256: str
    size: int


DETECTOR = ModelFile("face_detection_yunet_2023mar.onnx", "face_detection_yunet/face_detection_yunet_2023mar.onnx",
                     "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4", 232_589)
RECOGNIZER = ModelFile("face_recognition_sface_2021dec.onnx",
                       "face_recognition_sface/face_recognition_sface_2021dec.onnx",
                       "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79", 38_696_353)
FILES = (DETECTOR, RECOGNIZER)


def models_dir(folder=None) -> Path:
    """``folder``, else ``$BIFOLD_FACE_MODELS``, else ``models/`` in the working directory (as for YOLO)."""
    return Path(folder or os.environ.get("BIFOLD_FACE_MODELS") or "models").expanduser()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure(model: ModelFile, folder=None, download: bool = True, progress=print) -> Path:
    """The path of ``model`` in the models folder, downloading it first if it is missing (or damaged)."""
    path = models_dir(folder) / model.name
    if path.is_file() and path.stat().st_size == model.size:
        return path
    if not download:
        raise FileNotFoundError(f"{path} is missing (run: bifold-faces fetch)")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    url = BASE_URL + model.path
    if progress:
        progress(f"[faces] downloading {model.name} ({model.size / 1e6:.1f} MB) from {url}")
    try:
        with urllib.request.urlopen(url, timeout=60) as response, open(tmp, "wb") as out:
            while block := response.read(1 << 20):
                out.write(block)
        if _sha256(tmp) != model.sha256:
            raise OSError(f"{model.name}: the download does not match its checksum (a proxy page or a cut-off file?)")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def ensure_all(folder=None, download: bool = True, progress=print) -> dict[str, Path]:
    return {m.name: ensure(m, folder, download, progress) for m in FILES}
