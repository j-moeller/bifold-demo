"""The model files: where they are kept and how they are fetched.

They live in ``$BIFOLD_MORPH_MODELS``, else ``~/.cache/bifold-morph``:

==================================== ============================================================ =========
file                                 what                                                         size
==================================== ============================================================ =========
``e4e_ffhq_encode.pt``               e4e encoder + StyleGAN2 FFHQ generator (Google Drive)         ~1.2 GB
``mivolo_face_imdb.pth.tar``         MiVOLO, face only, age + gender, IMDB-clean (Google Drive)
``face_detection_yunet_2023mar.onnx`` YuNet face detector (OpenCV Zoo, GitHub)                     0.2 MB
``directions.npz``                   the age and gender directions, made by ``bifold-morph directions``
==================================== ============================================================ =========
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

MODELS_ENV = "BIFOLD_MORPH_MODELS"
DIRECTIONS = "directions.npz"
PREVIEW = "directions_preview.jpg"


@dataclass(frozen=True)
class ModelFile:
    name: str
    what: str
    gdrive_id: str | None = None  # a file on Google Drive (fetched with gdown) ...
    url: str | None = None  # ... or a plain download
    sha256: str | None = None
    size: int | None = None
    min_size: int = 0  # bytes: anything smaller is an error page, not the model

    @property
    def page(self) -> str:
        return f"https://drive.google.com/file/d/{self.gdrive_id}/view" if self.gdrive_id else self.url


E4E = ModelFile("e4e_ffhq_encode.pt", "e4e encoder + StyleGAN2 FFHQ generator",
                gdrive_id="1cUv_reLE6k3604or78EranS7XzuVMWeO", min_size=500_000_000)
MIVOLO = ModelFile("mivolo_face_imdb.pth.tar", "MiVOLO age + gender (face only, IMDB-clean)",
                   gdrive_id="1NlsNEVijX2tjMe8LBb1rI56WB_ADVHeP", min_size=50_000_000)
YUNET = ModelFile("face_detection_yunet_2023mar.onnx", "YuNet face detector",
                  url="https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
                      "face_detection_yunet/face_detection_yunet_2023mar.onnx",
                  sha256="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4", size=232_589)
FILES = (E4E, MIVOLO, YUNET)


def models_dir(folder=None) -> Path:
    """``folder``, else ``$BIFOLD_MORPH_MODELS``, else ``~/.cache/bifold-morph``."""
    return Path(folder or os.environ.get(MODELS_ENV) or Path.home() / ".cache" / "bifold-morph").expanduser()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _looks_complete(model: ModelFile, path: Path) -> bool:
    if not path.is_file():
        return False
    size = path.stat().st_size
    if model.size is not None:
        return size == model.size
    if size < model.min_size:
        return False
    with open(path, "rb") as f:
        return not f.read(16).lstrip().startswith(b"<")  # an HTML page from Google Drive (quota, sign-in)


def _download_url(url: str, out: Path) -> None:
    with urllib.request.urlopen(url, timeout=60) as response, open(out, "wb") as f:
        while block := response.read(1 << 20):
            f.write(block)


def _download_gdrive(file_id: str, out: Path) -> None:
    import gdown  # imported here: only the Google Drive files need it

    if gdown.download(id=file_id, output=str(out), quiet=False) is None:
        raise OSError("gdown could not fetch the file")


def ensure(model: ModelFile, folder=None, download: bool = True, progress=print) -> Path:
    """The path of ``model`` in the models folder, downloading it first if it is missing (or damaged)."""
    path = models_dir(folder) / model.name
    if _looks_complete(model, path):
        return path
    if not download:
        raise FileNotFoundError(f"{path} is missing (run: bifold-morph fetch)")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    if progress:
        progress(f"[models] downloading {model.what} to {path}")
    try:
        if model.gdrive_id:
            _download_gdrive(model.gdrive_id, tmp)
        else:
            _download_url(model.url, tmp)
        if model.sha256 and _sha256(tmp) != model.sha256:
            raise OSError("the download does not match its checksum (a proxy page or a cut-off file?)")
        if not _looks_complete(model, tmp):
            raise OSError("the download is not the model (an error page from the server?)")
        os.replace(tmp, path)
    except Exception as error:
        raise RuntimeError(f"could not download {model.name}: {error}. Download it in a browser from "
                           f"{model.page} and save it as {path}") from error
    finally:
        tmp.unlink(missing_ok=True)
    return path


def directions_path(folder=None) -> Path:
    return models_dir(folder) / DIRECTIONS
