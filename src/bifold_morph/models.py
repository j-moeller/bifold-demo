"""The model files: where they are kept and how they are fetched (all from GitHub releases, each checked by its
SHA-256).

They live in ``$BIFOLD_MORPH_MODELS``, else ``~/.cache/bifold-morph``.

=================================== ============================================================ ======= =========
file                                what                                                          size    needed for
=================================== ============================================================ ======= =========
``face_detection_yunet_2023mar``    YuNet face detector (OpenCV Zoo, MIT)                          0.2 MB  the app
``fairface.onnx``                   FairFace gender/age/race classifier (CC BY 4.0)               85 MB   read-out, training
``genderage.onnx``                  InsightFace's gender/age model (from ``buffalo_l``)           1.3 MB  read-out
``arcface_w600k_r50.onnx``          ArcFace identity embedding (InsightFace)                     174 MB   validation
``GFPGANv1.pth``                    GFPGAN v1: StyleGAN2 FFHQ-512 + its encoder (Apache 2.0 code) 615 MB   training
=================================== ============================================================ ======= =========

The student network itself (``weights/student.pt``, 12 MB) comes with the package.
"""

from __future__ import annotations

import hashlib
import io
import os
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

MODELS_ENV = "BIFOLD_MORPH_MODELS"
FACEFUSION = "https://github.com/facefusion/facefusion-assets/releases/download/"
STUDENT = Path(__file__).resolve().parent / "weights" / "student.pt"


@dataclass(frozen=True)
class ModelFile:
    name: str
    what: str
    url: str
    sha256: str
    size: int
    member: str | None = None  # the file inside a zip at ``url``


YUNET = ModelFile("face_detection_yunet_2023mar.onnx", "YuNet face detector",
                  "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/"
                  "face_detection_yunet_2023mar.onnx",
                  "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4", 232_589)
FAIRFACE = ModelFile("fairface.onnx", "FairFace classifier", FACEFUSION + "models-3.0.0/fairface.onnx",
                     "b176b1e4b0739496e7943af7ba4ff306271fecf837bca5ec5d9a48201959db37", 85_170_772)
GENDERAGE = ModelFile("genderage.onnx", "InsightFace gender/age model",
                      "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
                      "4fde69b1c810857b88c64a335084f1c3fe8f01246c9a191b48c7bb756d6652fb", 1_322_532,
                      member="genderage.onnx")
ARCFACE = ModelFile("arcface_w600k_r50.onnx", "ArcFace identity embedding",
                    FACEFUSION + "models-3.0.0/arcface_w600k_r50.onnx",
                    "f1f79dc3b0b79a69f94799af1fffebff09fbd78fd96a275fd8f0cbbea23270d1", 174_388_474)
GFPGAN = ModelFile("GFPGANv1.pth", "GFPGAN v1 (StyleGAN2 FFHQ-512 and its encoder)",
                   "https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/GFPGANv1.pth",
                   "6db3a33dd00dd427b8a70a7e3c6244a5bcccb818736f4861ce1d609024a991de", 615_378_983)
APP = (YUNET, FAIRFACE, GENDERAGE)
ALL = APP + (ARCFACE, GFPGAN)


def models_dir(folder=None) -> Path:
    """``folder``, else ``$BIFOLD_MORPH_MODELS``, else ``~/.cache/bifold-morph``."""
    return Path(folder or os.environ.get(MODELS_ENV) or Path.home() / ".cache" / "bifold-morph").expanduser()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _download(url: str, progress=None) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as response:
        total = int(response.headers.get("Content-Length") or 0)
        chunks, done, shown = [], 0, 0
        while block := response.read(1 << 20):
            chunks.append(block)
            done += len(block)
            if progress and total and done - shown > total / 10:
                progress(f"[models]   {done / total:.0%} of {total / 1e6:.0f} MB")
                shown = done
    return b"".join(chunks)


def ensure(model: ModelFile, folder=None, download: bool = True, progress=print) -> Path:
    """The path of ``model`` in the models folder, downloading it first if it is missing (or damaged)."""
    path = models_dir(folder) / model.name
    if path.is_file() and path.stat().st_size == model.size:
        return path
    if not download:
        raise FileNotFoundError(f"{path} is missing (run: bifold-morph fetch)")
    path.parent.mkdir(parents=True, exist_ok=True)
    if progress:
        progress(f"[models] downloading {model.what} from {model.url}")
    data = _download(model.url, progress)
    if model.member:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            name = next(n for n in z.namelist() if n.endswith("/" + model.member) or n == model.member)
            data = z.read(name)
    if _sha256(data) != model.sha256:
        raise RuntimeError(f"{model.name}: the download does not match its checksum (a proxy page or a cut-off "
                           f"file?). Get it from {model.url} and save it as {path}")
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return path


def ensure_all(models=APP, folder=None, download: bool = True, progress=print) -> dict[str, Path]:
    return {m.name: ensure(m, folder, download, progress) for m in models}
