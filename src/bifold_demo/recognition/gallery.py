"""The people the recogniser knows: a folder of photos per person, their embeddings, and matching a face to them.

Layout of the faces folder (``faces/`` by default)::

    faces/
      Ada Lovelace/        a folder per person, named as they should be called;
        portrait.jpg       as many photos as you like (different days, light, angles: the more, the better)
        holiday.png
      Grace Hopper.jpg     or a single photo named after the person

Each photo should show the person's face (the biggest face in it is used). Their embeddings are cached in
``faces/.embeddings.npz``, so only new or changed photos are looked at again; the folder is checked again at most
once a second, so photos dropped in while the demo runs are known a moment later. A face is the person whose closest
photo is the most similar, if that similarity reaches :data:`THRESHOLD`.
"""

from __future__ import annotations

import math
import os
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..scene.textures import IMAGE_EXTENSIONS

THRESHOLD = 0.363  # cosine similarity: OpenCV's recommended cut for SFace (99.6 % on LFW)
CACHE = ".embeddings.npz"
RESCAN = 1.0  # seconds between looks at the folder


@dataclass(frozen=True)
class Match:
    """Who a face is: ``name`` (``None``: nobody known), the ``score`` of the closest photo of them and the score of
    the closest other person (``runner_up``)."""

    name: str | None
    score: float
    runner_up: float = -1.0


@dataclass
class Photo:
    path: Path
    name: str
    stamp: tuple[int, int]  # (size, mtime_ns): looked at again when it changes
    vector: np.ndarray | None  # None: no face in it
    face_size: float = 0.0


def _person(path: Path, folder: Path) -> str | None:
    relative = path.relative_to(folder)
    if len(relative.parts) == 1:
        return path.stem.replace("_", " ").strip() or None
    if len(relative.parts) == 2:
        return relative.parts[0].strip() or None
    return None


def _photos(folder: Path) -> list[Path]:
    found = []
    for root, dirs, files in os.walk(folder):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for name in sorted(files):
            path = Path(root) / name
            if not name.startswith(".") and path.suffix.lower() in IMAGE_EXTENSIONS:
                found.append(path)
    return found


class Gallery:
    """Known people from ``folder`` (see the module docs), embedded with ``engine`` (a :class:`~.engine.FaceEngine`;
    made when first needed). Thread safe."""

    def __init__(self, folder="faces", engine=None, threshold: float = THRESHOLD):
        self.folder = Path(folder).expanduser()
        self.threshold = float(threshold)
        self._engine = engine
        self._photos: dict[Path, Photo] = {}
        self._checked = -math.inf
        self._lock = threading.RLock()
        self._names: list[str] = []
        self._matrix = np.zeros((0, 128))
        self._owners = np.zeros(0, int)
        self._loaded = False

    @property
    def engine(self):
        if self._engine is None:
            from .engine import FaceEngine

            self._engine = FaceEngine()
        return self._engine

    # -- the folder -------------------------------------------------------------------------------------------

    def refresh(self, now: float | None = None, force: bool = False) -> bool:
        """Look at the folder again (at most every ``RESCAN`` seconds unless ``force``); True if anything changed."""
        now = time.monotonic() if now is None else now
        with self._lock:
            if not force and now - self._checked < RESCAN:
                return False
            self._checked = now
            first = not self._loaded
            if first:
                self._load_cache()
                self._loaded = True
            paths = _photos(self.folder) if self.folder.is_dir() else []
            changed = first
            keep: dict[Path, Photo] = {}
            for path in paths:
                name = _person(path, self.folder)
                if name is None:
                    continue
                try:
                    st = path.stat()
                except OSError:
                    continue
                stamp = (st.st_size, st.st_mtime_ns)
                old = self._photos.get(path)
                if old is not None and old.stamp == stamp:
                    old.name = name  # a renamed folder keeps the embeddings
                    keep[path] = old
                    continue
                keep[path] = self._embed(path, name, stamp)
                changed = True
            if set(keep) != set(self._photos):
                changed = True
            self._photos = keep
            if changed:
                self._rebuild()
                self._save_cache()
            return changed

    def _embed(self, path: Path, name: str, stamp) -> Photo:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            return Photo(path, name, stamp, None)
        faces = self.engine.detect(image, min_size=0)
        if not faces:
            return Photo(path, name, stamp, None)
        face = faces[0]  # the biggest
        return Photo(path, name, stamp, self.engine.embed(image, face), face.size)

    def _rebuild(self) -> None:
        good = [p for p in self._photos.values() if p.vector is not None]
        self._names = sorted({p.name for p in good})
        index = {n: k for k, n in enumerate(self._names)}
        self._matrix = np.array([p.vector for p in good]) if good else np.zeros((0, 128))
        self._owners = np.array([index[p.name] for p in good], int)

    def _cache_path(self) -> Path:
        return self.folder / CACHE

    def _load_cache(self) -> None:
        try:
            data = np.load(self._cache_path(), allow_pickle=False)
            paths, stamps, vectors, sizes = data["paths"], data["stamps"], data["vectors"], data["sizes"]
        except (OSError, KeyError, ValueError):
            return
        for path, stamp, vector, size in zip(paths, stamps, vectors, sizes):
            path = self.folder / str(path)
            name = _person(path, self.folder) if path.is_relative_to(self.folder) else None
            if name is not None:
                found = np.isfinite(vector).all()
                self._photos[path] = Photo(path, name, (int(stamp[0]), int(stamp[1])), vector if found else None,
                                           float(size))

    def _save_cache(self) -> None:
        if not self.folder.is_dir():
            return
        photos = list(self._photos.values())
        tmp = self.folder / f"{CACHE}.{os.getpid()}.tmp.npz"
        try:
            np.savez(tmp, paths=np.array([str(p.path.relative_to(self.folder)) for p in photos], str),
                     stamps=np.array([p.stamp for p in photos], np.int64).reshape(-1, 2),
                     vectors=np.array([p.vector if p.vector is not None else np.full(128, np.nan) for p in photos],
                                      np.float64).reshape(-1, 128),
                     sizes=np.array([p.face_size for p in photos], np.float64))
            os.replace(tmp, self._cache_path())
        except OSError:
            tmp.unlink(missing_ok=True)

    # -- who is it --------------------------------------------------------------------------------------------

    def people(self) -> dict[str, int]:
        """``{name: number of photos with a face}``."""
        with self._lock:
            counts: dict[str, int] = {}
            for p in self._photos.values():
                if p.vector is not None:
                    counts[p.name] = counts.get(p.name, 0) + 1
            return dict(sorted(counts.items()))

    def problems(self) -> list[Path]:
        """Photos in which no face was found."""
        with self._lock:
            return sorted(p.path for p in self._photos.values() if p.vector is None)

    def photos(self, name: str) -> list[Path]:
        """The photos of ``name`` with a face, the one with the biggest face first."""
        with self._lock:
            mine = [p for p in self._photos.values() if p.name == name and p.vector is not None]
            return [p.path for p in sorted(mine, key=lambda p: -p.face_size)]

    def match(self, vector: np.ndarray) -> Match:
        """Who the face with embedding ``vector`` is."""
        with self._lock:
            matrix, owners, names = self._matrix, self._owners, self._names
        if not len(matrix):
            return Match(None, -1.0)
        scores = matrix @ np.asarray(vector, np.float64)
        best = np.full(len(names), -1.0)
        np.maximum.at(best, owners, scores)
        order = np.argsort(-best)
        top = float(best[order[0]])
        second = float(best[order[1]]) if len(order) > 1 else -1.0
        return Match(names[order[0]] if top >= self.threshold else None, top, second)

    def score(self, vector: np.ndarray, name: str) -> float:
        """How similar the face with embedding ``vector`` is to the closest photo of ``name`` (-1: not known)."""
        with self._lock:
            matrix, owners, names = self._matrix, self._owners, self._names
        if name not in names:
            return -1.0
        mine = matrix[owners == names.index(name)]
        return float((mine @ np.asarray(vector, np.float64)).max()) if len(mine) else -1.0

    # -- adding people ----------------------------------------------------------------------------------------

    def add(self, name: str, photos) -> list[str]:
        """Copy ``photos`` of ``name`` into their folder (``faces/<name>/``); returns a problem per photo without a
        face (those are not copied)."""
        name = name.strip()
        if not name or any(c in name for c in "/\\") or name.startswith("."):
            raise ValueError(f"not a usable name for a person: {name!r}")
        target = self.folder / name
        problems = []
        for photo in photos:
            photo = Path(photo).expanduser()
            image = cv2.imread(str(photo), cv2.IMREAD_COLOR)
            if image is None:
                problems.append(f"{photo}: not an image")
                continue
            if not self.engine.detect(image, min_size=0):
                problems.append(f"{photo}: no face found")
                continue
            target.mkdir(parents=True, exist_ok=True)
            dest = target / photo.name
            k = 2
            while dest.exists():
                dest = target / f"{photo.stem}_{k}{photo.suffix}"
                k += 1
            shutil.copy2(photo, dest)
        self.refresh(force=True)
        return problems
