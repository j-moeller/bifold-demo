"""Textures: anything you can paint onto an object.

``as_texture`` turns any of the following into a :class:`Texture`:

* a path to an image (``"cat.jpg"``; PNG transparency is respected)
* a path to a video (``"clip.mp4"``; plays in simulation time)
* a NumPy array (OpenCV-style BGR or BGRA, ``uint8``)
* a PIL image
* a colour: ``(r, g, b)`` with 0-255 components, or a hex string ``"#ff8800"``
* a :class:`Texture` (returned unchanged)

Image files are re-read automatically when they change on disk, so replacing a
file is enough to update every object that uses it.
"""

from __future__ import annotations

import importlib.util
import itertools
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff", ".gif"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}

_ids = itertools.count(1)


class Texture:
    """RGBA image data (row 0 = top of the image).

    ``version`` increases whenever the pixels change so the renderer knows when
    to re-upload them to the GPU.
    """

    def __init__(self, rgba: np.ndarray, name: str = ""):
        self.id = next(_ids)
        self.name = name
        self.version = 0
        self.rgba = _check_rgba(rgba)

    @classmethod
    def from_file(cls, path, max_size: int = 2048) -> "Texture":
        path = Path(path).expanduser()
        return cls(_read_image(path, max_size), name=path.name)

    @classmethod
    def from_array(cls, image: np.ndarray, order: str = "bgr", name: str = "array") -> "Texture":
        """Create a texture from a ``uint8`` array in ``"bgr"`` (OpenCV) or ``"rgb"`` channel order."""
        return cls(_array_to_rgba(np.asarray(image), order), name=name)

    @classmethod
    def solid(cls, color, name: str = "") -> "Texture":
        rgb = parse_color(color)
        return cls(np.array([[[*rgb, 255]]], dtype=np.uint8), name=name or f"color{tuple(rgb)}")

    @property
    def width(self) -> int:
        return self.rgba.shape[1]

    @property
    def height(self) -> int:
        return self.rgba.shape[0]

    @property
    def aspect(self) -> float:
        """Width divided by height."""
        return self.width / self.height

    def update(self, image: np.ndarray, order: str = "bgr") -> None:
        """Replace the pixels (the renderer picks the change up on the next frame).

        ``image`` is a ``uint8`` array like for :meth:`from_array`: 1, 3 or 4 channels in
        ``"bgr"`` (OpenCV) or ``"rgb"`` order. The size may differ from the previous image.
        """
        self._set_rgba(_array_to_rgba(np.asarray(image), order))

    def _set_rgba(self, rgba: np.ndarray) -> None:
        self.rgba = _check_rgba(rgba)
        self.version += 1

    def at_time(self, t: float) -> "Texture":
        """Advance animated textures to simulation time ``t``; static textures return themselves."""
        return self

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r}, {self.width}x{self.height})"


class VideoTexture(Texture):
    """A texture that plays a video file, synchronised to simulation time."""

    def __init__(self, path, loop: bool = True, speed: float = 1.0, max_size: int = 1024):
        self.path = Path(path).expanduser()
        self.loop, self.speed, self.max_size = loop, speed, max_size
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise FileNotFoundError(f"cannot open video {self.path}")
        self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, frame = self._cap.read()
        if not ok:
            raise ValueError(f"video {self.path} has no frames")
        self._next_index = 1  # index of the frame the next read() returns
        self._shown = 0  # index of the frame currently in `rgba`
        super().__init__(_fit(_array_to_rgba(frame, "bgr"), max_size), name=self.path.name)

    def at_time(self, t: float) -> "Texture":
        index = max(0, int(t * self.fps * self.speed))
        if self.frame_count > 0:
            index = index % self.frame_count if self.loop else min(index, self.frame_count - 1)
        if index == self._shown:
            return self
        if index < self._next_index or index > self._next_index + 30:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            self._next_index = index
        frame = None
        while self._next_index <= index:
            ok, frame = self._cap.read()
            if not ok:  # unknown length or decoder hiccup: restart from the beginning
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self._next_index = 0
                ok, frame = self._cap.read()
                if not ok:
                    return self
                index = 0
            self._next_index += 1
        if frame is not None:
            self._set_rgba(_fit(_array_to_rgba(frame, "bgr"), self.max_size))
            self._shown = index
        return self


# ---------------------------------------------------------------------------
# Conversion


def as_texture(spec) -> Texture | None:
    """Convert a texture specification (see module docs) into a :class:`Texture`."""
    if spec is None or isinstance(spec, Texture):
        return spec
    if isinstance(spec, str) and spec.startswith("#"):
        return _cached_color(spec)
    if isinstance(spec, (str, os.PathLike)):
        return _file_cache.get(Path(spec))
    if isinstance(spec, np.ndarray):
        return Texture.from_array(spec)
    if hasattr(spec, "convert") and hasattr(spec, "size"):  # PIL.Image
        return Texture(np.asarray(spec.convert("RGBA")), name=getattr(spec, "filename", "") or "PIL image")
    if isinstance(spec, (tuple, list)):
        return _cached_color(spec)
    raise TypeError(f"cannot use {type(spec).__name__} as a texture: {spec!r}")


def parse_color(color) -> tuple[int, int, int]:
    """Parse ``(r, g, b)`` (0-255) or ``"#rrggbb"`` into an RGB tuple."""
    if isinstance(color, str):
        s = color.lstrip("#")
        if len(s) == 3:
            s = "".join(ch * 2 for ch in s)
        if len(s) != 6:
            raise ValueError(f"invalid hex colour {color!r}")
        return tuple(int(s[i : i + 2], 16) for i in (0, 2, 4))
    values = tuple(color)
    if len(values) not in (3, 4):
        raise ValueError(f"colour must have 3 components (r, g, b), got {color!r}")
    return tuple(int(np.clip(round(float(v)), 0, 255)) for v in values[:3])


_color_cache: dict[tuple[int, int, int], Texture] = {}


def _cached_color(color) -> Texture:
    rgb = parse_color(color)
    if rgb not in _color_cache:
        _color_cache[rgb] = Texture.solid(rgb)
    return _color_cache[rgb]


class _FileCache:
    """Shares one Texture per file and reloads it when the file changes."""

    check_interval = 0.5  # seconds between modification-time checks per file

    def __init__(self):
        self._entries: dict[Path, list] = {}  # path -> [texture, mtime, last_check]
        self._lock = threading.Lock()

    def get(self, path: Path) -> Texture:
        path = path.expanduser().resolve()
        now = time.monotonic()
        with self._lock:
            entry = self._entries.get(path)
            if entry is not None and now - entry[2] < self.check_interval:
                return entry[0]
            mtime = path.stat().st_mtime if path.exists() else None
            if entry is not None:
                entry[2] = now
                if mtime is None or mtime == entry[1] or isinstance(entry[0], VideoTexture):
                    return entry[0]
                try:  # file was replaced: refresh the pixels of the shared texture
                    entry[0]._set_rgba(_read_image(path, 2048))
                    entry[0].name = path.name
                    entry[1] = mtime
                except (OSError, ValueError):
                    pass  # half-written file; retry on the next check
                return entry[0]
            if mtime is None:
                raise FileNotFoundError(f"texture file not found: {path}")
            if path.suffix.lower() in VIDEO_EXTENSIONS:
                texture = VideoTexture(path)
            else:
                texture = Texture.from_file(path)
            self._entries[path] = [texture, mtime, now]
            return texture


_file_cache = _FileCache()


# ---------------------------------------------------------------------------
# Procedural textures and bundled sample images


def checkerboard(squares: int = 8, size: int = 512, color_a=(225, 225, 225), color_b=(70, 70, 75)) -> Texture:
    """A checkerboard pattern, handy for floors."""
    cell = max(1, size // squares)
    yy, xx = np.mgrid[0 : cell * squares, 0 : cell * squares] // cell
    mask = ((xx + yy) % 2 == 0)[..., None]
    rgb = np.where(mask, parse_color(color_a), parse_color(color_b)).astype(np.uint8)
    return Texture(np.dstack([rgb, np.full(rgb.shape[:2], 255, np.uint8)]), name="checkerboard")


def gradient(top=(70, 90, 120), bottom=(20, 22, 28), size: int = 256) -> Texture:
    """A vertical colour gradient (16:9, so it fills a widescreen background), handy as a background."""
    width = size * 16 // 9
    t = np.linspace(0, 1, size)[:, None]
    rgb = (np.array(parse_color(top)) * (1 - t) + np.array(parse_color(bottom)) * t).astype(np.uint8)
    rgba = np.dstack([np.repeat(rgb[:, None, :], width, axis=1), np.full((size, width), 255, np.uint8)])
    return Texture(rgba, name="gradient")


def sample_image(name: str) -> Path | None:
    """Path to an example photo bundled with Ultralytics (``"bus.jpg"`` or ``"zidane.jpg"``)."""
    spec = importlib.util.find_spec("ultralytics")
    if spec is None or not spec.submodule_search_locations:
        return None
    path = Path(next(iter(spec.submodule_search_locations))) / "assets" / name
    return path if path.exists() else None


# ---------------------------------------------------------------------------
# Pixel helpers


def _check_rgba(rgba: np.ndarray) -> np.ndarray:
    rgba = np.asarray(rgba)
    if rgba.ndim != 3 or rgba.shape[2] != 4 or rgba.dtype != np.uint8:
        raise ValueError(f"expected an HxWx4 uint8 array, got {rgba.dtype} {rgba.shape}")
    return np.ascontiguousarray(rgba)


def _array_to_rgba(image: np.ndarray, order: str) -> np.ndarray:
    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating) and image.max(initial=0) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        image = image[..., None]
    channels = image.shape[2]
    if order not in ("bgr", "rgb"):
        raise ValueError("order must be 'bgr' or 'rgb'")
    if channels == 1:
        rgb = np.repeat(image, 3, axis=2)
        alpha = np.full(image.shape[:2], 255, np.uint8)
    elif channels in (3, 4):
        rgb = image[..., 2::-1] if order == "bgr" else image[..., :3]
        alpha = image[..., 3] if channels == 4 else np.full(image.shape[:2], 255, np.uint8)
    else:
        raise ValueError(f"unsupported number of channels: {channels}")
    return np.ascontiguousarray(np.dstack([rgb, alpha]))


def _fit(rgba: np.ndarray, max_size: int) -> np.ndarray:
    h, w = rgba.shape[:2]
    scale = max_size / max(h, w)
    if scale >= 1:
        return rgba
    return cv2.resize(rgba, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)


def _read_image(path: Path, max_size: int) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED) if data.size else None
    if image is not None:
        if image.dtype == np.uint16:
            image = (image >> 8).astype(np.uint8)
        return _fit(_array_to_rgba(image, "bgr"), max_size)
    try:  # formats OpenCV cannot decode (e.g. animated GIF)
        from PIL import Image

        with Image.open(path) as img:
            return _fit(np.asarray(img.convert("RGBA")), max_size)
    except Exception as exc:  # noqa: BLE001 - report any decoder failure the same way
        raise ValueError(f"cannot read image {path}: {exc}") from exc
