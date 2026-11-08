"""Detail levels: the 2048-pixel textures of heads only where a picture needs them.

A prepared head has its colour texture and its packed normal and specular map at 1024 pixels (``texture``,
``maps``) and at 2048 (``detail``, see :mod:`bifold_demo.characters.model`). At 1024 a head texture has about 112
texels between the eyes, so the 2048 one adds nothing to a face smaller than that, while it needs four times the
memory: 21 MiB per texture with its mipmaps, 43 MiB per head with its maps. So the renderer draws a person's head with
its detail textures only in pictures where that person's eyes are at least ``Renderer.detail_from`` pixels apart (64:
the 1024 textures begin to blur there on surfaces seen at an angle, where anisotropic filtering wants finer levels),
whatever the picture is (a window of the virtual sensor, or a close camera), and with the 1024 ones everywhere else.

:class:`DetailCache` holds the detail textures of the last few heads drawn that way (``Renderer.detail_heads``; the
least recently used are dropped, and the renderer frees their GPU memory at once). It reads a head's detail files when
a picture first needs them: at once by default (``Renderer.detail_wait``: the picture is as it should be, as tests and
measurements want), or on a background thread, drawing the 1024 textures until they are ready (the demo, so that a
frame never waits for a 2048 PNG: about 0.2 s to decode here, 0.1 s more for its mipmaps).

A texture larger than 1024 pixels that a person wears instead of the model's (``texture_overrides``: a face made from
photos at 2048) gets the same treatment: :func:`base_copy` is its 1024 mipmap level, drawn where the picture does not
need more.
"""

from __future__ import annotations

import threading
import weakref
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .textures import Texture, VideoTexture, mip_chain, prepare_upload

BASE_SIZE = 1024  # pixels: the size of the textures every picture can draw (larger ones are detail)
EYE_DISTANCE = 0.063  # metres between the eyes of an adult, for the size of a face in a picture
_pool: ThreadPoolExecutor | None = None
_pool_lock = threading.Lock()


def eye_pixels(head, view_proj: np.ndarray, size, right) -> float:
    """How many pixels apart the eyes of a face at ``head`` (a point in the scene) are in a picture of ``size`` =
    (width, height) drawn with ``view_proj``: a frontal face's eye distance, whatever way it is turned. ``right``:
    the camera's right in the scene (unit). 0 behind the camera."""
    p = np.asarray(head, np.float64)
    a = view_proj @ np.append(p, 1.0)
    b = view_proj @ np.append(p + EYE_DISTANCE * np.asarray(right, np.float64), 1.0)
    if a[3] <= 1e-6 or b[3] <= 1e-6:
        return 0.0
    d = (b[:2] / b[3] - a[:2] / a[3]) * (0.5 * size[0], 0.5 * size[1])
    return float(np.hypot(d[0], d[1]))


def is_large(texture) -> bool:
    """Whether ``texture`` is larger than :data:`BASE_SIZE` and does not change (a video does)."""
    return (isinstance(texture, Texture) and not isinstance(texture, VideoTexture)
            and max(texture.width, texture.height) > BASE_SIZE)


def base_copy(texture: Texture) -> Texture:
    """The mipmap level of a large ``texture`` that is at most :data:`BASE_SIZE` pixels (averaged in linear light,
    :func:`~.textures.mip_chain`), made once per version of its pixels and kept with it."""
    cached = texture.__dict__.get("_base_copy")
    if cached is not None and cached[0] == texture.version:
        return cached[1]
    levels = mip_chain(texture.rgba, texture.srgb)
    level = next(lv for lv in levels if max(lv.shape[:2]) <= BASE_SIZE)
    copy = Texture(level, name=f"{texture.name} (base)", srgb=texture.srgb)
    texture._base_copy = (texture.version, copy)
    return copy


def load(detail) -> tuple[Texture, Texture | None]:
    """The colour texture and the packed map of a :class:`~bifold_demo.characters.model.Detail` (image paths or RGBA
    arrays in RGB order), laid out for the upload with their mipmaps (on the calling thread)."""
    return _texture(detail.texture, True), (_texture(detail.maps, False) if detail.maps is not None else None)


def _texture(spec, srgb: bool) -> Texture:
    from .textures import _array_to_rgba, _read_image

    if isinstance(spec, Texture):
        return prepare_upload(spec)
    if isinstance(spec, np.ndarray):
        rgba, name = _array_to_rgba(spec, "rgb"), "detail"
    else:
        rgba, name = _read_image(Path(spec), 1 << 14), Path(spec).name
    return prepare_upload(Texture(rgba, name=name, srgb=srgb))


def _background() -> ThreadPoolExecutor:
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="detail-textures")
        return _pool


class DetailCache:
    """The detail textures of the heads drawn with them lately (see the module docs), by mesh part."""

    def __init__(self):
        self._entries: OrderedDict[int, list] = OrderedDict()  # id(primitive) -> [weakref, textures or None, job]
        self.loads = 0  # detail textures read (a pair counts once)

    def get(self, primitive, wait: bool) -> tuple[Texture, Texture | None] | None:
        """The detail ``(colour, maps)`` of ``primitive`` (one with a ``detail``), read now when ``wait`` and not
        read yet, else asked for on a background thread; ``None`` until they are there (or when they cannot be
        read: it is then not tried again)."""
        key = id(primitive)
        entry = self._entries.get(key)
        if entry is None or entry[0]() is not primitive:
            entry = [weakref.ref(primitive), None, None]
            self._entries[key] = entry
            if wait:
                entry[1] = self._read(primitive.detail)
            else:
                entry[2] = _background().submit(self._read, primitive.detail)
        if entry[2] is not None and (wait or entry[2].done()):
            job, entry[2] = entry[2], None
            entry[1] = job.result()
        self._entries.move_to_end(key)
        return entry[1] or None

    def _read(self, detail):
        try:
            textures = load(detail)
        except (OSError, ValueError) as exc:  # a damaged file: the 1024 textures are drawn
            print(f"[renderer] cannot read a head's detail textures, drawing it at 1024: {exc}")
            return False
        self.loads += 1
        return textures

    def trim(self, keep: int) -> list[Texture]:
        """Drop all but the ``keep`` heads used last (and those whose mesh part is gone); returns the textures dropped
        (the renderer frees their GPU memory)."""
        dropped = []
        for key in [k for k, entry in self._entries.items() if entry[0]() is None]:
            dropped += self._drop(key)
        while len(self._entries) > max(0, keep):
            dropped += self._drop(next(iter(self._entries)))
        return dropped

    def _drop(self, key: int) -> list[Texture]:
        _ref, textures, job = self._entries.pop(key)
        if isinstance(job, Future):
            job.cancel()
        return [t for t in (textures or ()) if t is not None]

    def __len__(self) -> int:
        return len(self._entries)

    def textures(self) -> list[Texture]:
        """The detail textures held now."""
        return [t for _ref, textures, _job in self._entries.values() for t in (textures or ()) if t is not None]
