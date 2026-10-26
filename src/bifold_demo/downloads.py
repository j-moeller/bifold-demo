"""Model files downloaded once and checked: the face models of :mod:`.recognition` and of
:mod:`.characters.headtex`.

A :class:`ModelFile` names a file, the URL it comes from, its size and its SHA-256. :func:`ensure` gives the file's
path in a folder, downloading it first when it is missing or has the wrong size. A download goes to a temporary file
next to the target and only replaces it once its checksum matches, so a cut-off download or a proxy's error page never
takes the place of a model.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelFile:
    name: str  # the file name in the models folder
    url: str
    sha256: str
    size: int  # bytes


def sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure(model: ModelFile, folder, download: bool = True, progress=print, fetch_hint: str = "",
           tag: str = "models") -> Path:
    """The path of ``model`` in ``folder``, downloading it first if it is missing (or damaged). Without
    ``download`` a missing file raises :class:`FileNotFoundError`, naming ``fetch_hint`` (the command that fetches
    it)."""
    path = Path(folder).expanduser() / model.name
    if path.is_file() and path.stat().st_size == model.size:
        return path
    if not download:
        raise FileNotFoundError(f"{path} is missing" + (f" (run: {fetch_hint})" if fetch_hint else ""))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    if progress:
        progress(f"[{tag}] downloading {model.name} ({model.size / 1e6:.1f} MB) from {model.url}")
    try:
        with urllib.request.urlopen(model.url, timeout=60) as response, open(tmp, "wb") as out:
            while block := response.read(1 << 20):
                out.write(block)
        if sha256(tmp) != model.sha256:
            raise OSError(f"{model.name}: the download does not match its checksum (a proxy page or a cut-off file?)")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path
