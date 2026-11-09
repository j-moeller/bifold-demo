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

**Colour spaces.** Image files hold sRGB-encoded colours, and the renderer works in linear light: it uploads a
:class:`Texture` as an sRGB texture, so the GPU decodes its colours when it samples them, and builds its mipmaps by
averaging light (:func:`mip_chain`: a fine black-and-white pattern becomes the grey of half the light, level 188,
not the darker 128 of averaged sRGB values; transparent texels lend no colour to the edges of cut-outs). Data such
as normal maps are not colours (``Texture.srgb`` is False, :func:`as_data_texture`): they are averaged as they are.
:func:`prepare_upload` lays a texture out for the GPU, mipmaps included, on any thread, so that the thread that
draws does not have to.
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
BASE_SIZE = 1024  # pixels: the largest side of the textures every picture draws (see scene.detail)


class Texture:
    """RGBA image data (row 0 = top of the image).

    ``version`` increases whenever the pixels change so the renderer knows when
    to re-upload them to the GPU. ``srgb``: the pixels are sRGB-encoded colours
    (the default); False for data such as normal maps (see the module docs).
    """

    srgb = True

    def __init__(self, rgba: np.ndarray, name: str = "", srgb: bool = True):
        self.id = next(_ids)
        self.name = name
        self.version = 0
        self.rgba = _check_rgba(rgba)
        if not srgb:
            self.srgb = False

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
        super().__init__(_fit(_array_to_rgba(frame, "bgr"), max_size, linear=False), name=self.path.name)

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
            self._set_rgba(_fit(_array_to_rgba(frame, "bgr"), self.max_size, linear=False))
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


def as_data_texture(spec) -> Texture | None:
    """:func:`as_texture` for data that are not colours (a normal map): marks the texture so that its pixels are
    neither decoded from sRGB when drawn nor averaged in linear light for its mipmaps. An image file used this way
    must not be used as a colour texture too (the file cache shares one texture per file)."""
    if isinstance(spec, np.ndarray) and spec.ndim == 3 and spec.shape[2] in (3, 4):
        return Texture(_array_to_rgba(spec, "rgb"), name="data", srgb=False)  # arrays of characters are RGB(A)
    texture = as_texture(spec)
    if texture is not None and texture.srgb:
        texture.srgb = False
    return texture


def prepare_upload(texture: Texture) -> Texture:
    """Lay out the pixels of ``texture`` the way the renderer uploads them (rows bottom first, for OpenGL), with its
    mipmap levels (:func:`mip_chain`), now, on the calling thread, so that the thread that draws does not have to:
    for a texture made or read on a worker thread. The renderer uses (and drops) them on the upload, unless the
    pixels changed meanwhile; when a renderer uploaded these pixels already, nothing is kept. A texture larger than
    :data:`BASE_SIZE` that a person wears (a face from photos at 2048) also gets the level the renderer draws where a
    picture does not need more (:func:`~.detail.base_copy`), laid out too. Returns ``texture``."""
    rgba, version = texture.rgba, texture.version
    chain = mip_chain(rgba, texture.srgb)
    levels = [np.ascontiguousarray(level[::-1]).tobytes() for level in chain]
    if texture.__dict__.get("_on_gpu") != version:  # (else they would never be used)
        texture._upload = (version, np.ascontiguousarray(rgba[::-1]).tobytes(), levels)
    if max(texture.width, texture.height) > BASE_SIZE and not isinstance(texture, VideoTexture) \
            and not texture.__dict__.get("drop_after_upload"):  # (a head's own detail texture never needs one)
        k = next(i for i, level in enumerate(chain) if max(level.shape[:2]) <= BASE_SIZE)
        copy = Texture(np.ascontiguousarray(chain[k]), name=f"{texture.name} (base)", srgb=texture.srgb)
        copy._upload = (copy.version, levels[k], levels[k + 1:])  # (its mipmaps are the texture's own below it)
        texture._base_copy = (version, copy)
    return texture


def drop_pixels(texture: Texture) -> None:
    """Give up the memory of ``texture``'s pixels once a renderer holds them on the GPU for good (a head's detail
    textures, which are read again when they are needed again): ``rgba`` keeps its shape but takes no memory, and the
    texture cannot be uploaded again (``pixels_dropped``)."""
    texture.__dict__.pop("_upload", None)
    texture.__dict__.pop("_base_copy", None)
    texture.rgba = np.broadcast_to(np.zeros(4, np.uint8), texture.rgba.shape)
    texture.pixels_dropped = True


def upload_levels(texture: Texture) -> list[bytes]:
    """The mipmap levels below ``texture``'s own, laid out for the renderer's upload (rows bottom first): those
    :func:`prepare_upload` made for its current pixels when there are any (dropped then), else made now."""
    prepared = texture.__dict__.get("_upload")
    if prepared is not None and prepared[0] == texture.version and prepared[2] is not None:
        texture.__dict__.pop("_upload", None)
        return prepared[2]
    return [np.ascontiguousarray(level[::-1]).tobytes() for level in mip_chain(texture.rgba, texture.srgb)]


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
    return tuple(min(255, max(0, int(round(float(v))))) for v in values[:3])


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
# Colour spaces: image files hold sRGB-encoded colours; light adds up, and pixels average, in linear light


def _srgb_decode_table() -> np.ndarray:
    c = np.arange(256, dtype=np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)


def _srgb_encode_table(steps: int = 65536) -> np.ndarray:
    lin = np.arange(steps, dtype=np.float64) / (steps - 1)
    c = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055)
    return np.clip(np.round(c * 255.0), 0, 255).astype(np.uint8)


_DECODE = _srgb_decode_table()
_ENCODE = _srgb_encode_table()
_LINEAR = (np.arange(256, dtype=np.float32) / 255.0).astype(np.float32)
_RGBA_TABLES = {True: np.stack([_DECODE, _DECODE, _DECODE, _LINEAR], axis=1).reshape(1, 256, 4),  # colour, alpha
                False: np.stack([_LINEAR] * 4, axis=1).reshape(1, 256, 4)}  # data


def to_linear(values) -> np.ndarray:
    """sRGB-encoded ``uint8`` values (any shape) as linear light, ``float32`` 0-1."""
    return _DECODE[np.asarray(values, dtype=np.uint8)]


def to_srgb(linear) -> np.ndarray:
    """Linear light (``float`` 0-1, any shape; clipped) as sRGB-encoded ``uint8`` values (exact to the nearest
    level)."""
    steps = len(_ENCODE) - 1
    index = np.clip(np.asarray(linear, dtype=np.float32) * steps + 0.5, 0, steps).astype(np.uint16)
    return _ENCODE[index]


def color_to_linear(color) -> tuple[float, float, float]:
    """A colour (anything :func:`parse_color` takes) as linear-light ``(r, g, b)``, 0-1 (for a shader)."""
    return tuple(float(v) for v in _DECODE[list(parse_color(color))])


def _premultiplied(rgba: np.ndarray, srgb: bool) -> tuple[np.ndarray, bool]:
    """``(HxWx4 float32, opaque)``: colour in linear light (``srgb``) or as it is, times alpha; alpha 0-1. Data (not
    ``srgb``) are not weighted: their alpha is data too (a packed map's occlusion), not how much of a texel is there."""
    out = cv2.LUT(np.ascontiguousarray(rgba), _RGBA_TABLES[bool(srgb)]).reshape(rgba.shape)
    opaque = not srgb or bool(rgba[..., 3].min(initial=255) == 255)
    if not opaque:
        out[..., :3] *= out[..., 3:]
    return out, opaque


def _unpremultiplied(level: np.ndarray, plain: np.ndarray | None, srgb: bool) -> np.ndarray:
    """``uint8`` RGBA of a premultiplied level (:func:`_premultiplied`); ``plain``: the same level averaged without
    the alpha weights, for the texels no coverage reached (``None``: the image is opaque)."""
    out = np.empty(level.shape, np.uint8)
    rgb = level[..., :3]
    if plain is not None:
        alpha = level[..., 3:]
        rgb = np.where(alpha > 1e-6, rgb / np.maximum(alpha, 1e-6), plain[..., :3])
    out[..., :3] = to_srgb(rgb) if srgb else np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)
    out[..., 3] = np.clip(level[..., 3] * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return out


def _shrink(values: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """``float32`` HxWxC averaged over the area of each new pixel (an exact 2x2 mean when the size halves)."""
    width, height = size
    if values.shape[2] <= 4:
        return cv2.resize(values, (width, height), interpolation=cv2.INTER_AREA).reshape(height, width, -1)
    return np.concatenate([_shrink(values[..., k:k + 4], size) for k in range(0, values.shape[2], 4)], axis=2)


def downsample(rgba: np.ndarray, size: tuple[int, int], srgb: bool = True) -> np.ndarray:
    """``rgba`` (HxWx4 ``uint8``) made smaller, to ``size`` = ``(width, height)``, by averaging over each new
    pixel's area: in linear light for colours (``srgb``; data such as normal maps are averaged as they are, channel by
    channel), the colour weighted by alpha, so that transparent texels lend no colour to the edges of cut-outs
    (hair)."""
    rgba = np.asarray(rgba, np.uint8)
    if (rgba.shape[1], rgba.shape[0]) == tuple(size):
        return rgba.copy()
    values, opaque = _premultiplied(rgba, srgb)
    plain = None if opaque else _shrink(_premultiplied(np.dstack([rgba[..., :3], np.full(rgba.shape[:2], 255,
                                                                                         np.uint8)]), srgb)[0], size)
    return _unpremultiplied(_shrink(values, size), plain, srgb)


def mip_chain(rgba: np.ndarray, srgb: bool = True) -> list[np.ndarray]:
    """The mipmap levels of ``rgba`` (HxWx4 ``uint8``) below it, each half the size of the one before (rounded
    down, at least 1 pixel), down to 1x1: ``uint8`` RGBA, averaged as :func:`downsample` does (in linear light for
    colours), each level from the unrounded values of the level above."""
    rgba = np.asarray(rgba, np.uint8)
    values, opaque = _premultiplied(rgba, srgb)
    plain = None
    if not opaque:
        plain, _ = _premultiplied(np.dstack([rgba[..., :3], np.full(rgba.shape[:2], 255, np.uint8)]), srgb)
    levels = []
    width, height = rgba.shape[1], rgba.shape[0]
    while width > 1 or height > 1:
        width, height = max(1, width // 2), max(1, height // 2)
        values = _shrink(values, (width, height))
        if plain is not None:
            plain = _shrink(plain, (width, height))
        levels.append(_unpremultiplied(values, plain, srgb))
    return levels


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


def _fit(rgba: np.ndarray, max_size: int, linear: bool = True) -> np.ndarray:
    """``rgba`` at most ``max_size`` pixels wide and high: made smaller in linear light (:func:`downsample`, as a
    colour image), or (``linear`` False: video frames, every frame) by averaging the stored values, which is faster."""
    h, w = rgba.shape[:2]
    scale = max_size / max(h, w)
    if scale >= 1:
        return rgba
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    if linear:
        return downsample(rgba, size)
    return cv2.resize(rgba, size, interpolation=cv2.INTER_AREA)


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
