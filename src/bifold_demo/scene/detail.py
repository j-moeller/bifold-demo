"""Detail levels: the 2048-pixel textures of heads only where a picture needs them.

A prepared head has its colour texture and its packed normal and specular map at 1024 pixels (``texture``,
``maps``) and at 2048 (``detail``, see :mod:`bifold_demo.characters.model`). At 1024 a head texture has about 112
texels between the eyes where the eyes are, fewer on the sparser parts of the face (a tenth of the face has 66-70), so
the 2048 one adds nothing to a face much smaller than that, while it needs four times the memory: 21 MiB per texture
with its mipmaps, 43 MiB per head with its maps. So the renderer draws a person's head with its detail textures only in
pictures where that person's eyes are at least ``Renderer.detail_from`` pixels apart (64: where the sparsest parts of
the face begin to be magnified), whatever the picture is (a window of the virtual sensor, or a close camera), and with
the 1024 ones everywhere else; a head outside the picture does not count.

:class:`DetailCache` holds the detail textures of at most ``Renderer.detail_heads`` heads, those that need them most:
a head's claim is the largest eye distance a picture drew it with, fading by :data:`FADE` each frame no picture draws
it. A head that is not held is read only when there is room, or when its eyes are more than :data:`HYSTERESIS` times
as far apart as the weakest claim of a held head that no picture of the frame drew and that is not being read; that
one is dropped (the renderer frees its GPU memory at once). So with more heads near the camera than are kept, the
largest faces keep theirs and the others are drawn at 1024, instead of each picture reading its head again. Detail
textures are read when a picture first needs them: at once by default (``Renderer.detail_wait``: the picture is as it
should be, as tests and measurements want), or on a background thread, drawing the 1024 textures until they are ready
(the demo, so that a frame never waits for a 2048 PNG: about 0.45 s for a head's two here). Once on the GPU their
pixels are not kept in memory.

A texture larger than 1024 pixels that a person wears instead of the model's (``texture_overrides``: a face made from
photos at 2048) gets the same treatment: :func:`base_copy` is its 1024 mipmap level, drawn where the picture does not
need more.
"""

from __future__ import annotations

import threading
import weakref
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .textures import BASE_SIZE, Texture, VideoTexture, mip_chain, prepare_upload

EYE_DISTANCE = 0.063  # metres between the eyes of an adult, for the size of a face in a picture
HEAD_REACH = 0.3  # metres from the head bone that a head reaches (its top, the face in front): in the picture or not
FADE = 0.9  # per frame: how a held head's claim to its detail textures fades while no picture draws it
HYSTERESIS = 1.25  # a head takes the place of a held one only when its eyes are this much further apart
_pool: ThreadPoolExecutor | None = None
_pool_lock = threading.Lock()


def eye_pixels(head, view_proj: np.ndarray, size, right) -> float:
    """How many pixels apart the eyes of a face at ``head`` (a point in the scene) are in a picture of ``size`` =
    (width, height) drawn with ``view_proj``: a frontal face's eye distance, whatever way it is turned. ``right``:
    the camera's right in the scene (unit). 0 behind the camera, and when the head (:data:`HEAD_REACH` round the
    point) is outside the picture."""
    p = np.asarray(head, np.float64)
    a = view_proj @ np.append(p, 1.0)
    b = view_proj @ np.append(p + EYE_DISTANCE * np.asarray(right, np.float64), 1.0)
    if a[3] <= 1e-6 or b[3] <= 1e-6:
        return 0.0
    half = np.array([0.5 * size[0], 0.5 * size[1]])
    centre = (a[:2] / a[3] + 1.0) * half  # pixels, from the picture's corner
    d = (b[:2] / b[3] - a[:2] / a[3]) * half
    eye = float(np.hypot(d[0], d[1]))
    reach = eye * HEAD_REACH / EYE_DISTANCE
    if np.any(centre < -reach) or np.any(centre > 2.0 * half + reach):
        return 0.0
    return eye


def is_large(texture) -> bool:
    """Whether ``texture`` is larger than :data:`BASE_SIZE` and does not change (a video does)."""
    return (isinstance(texture, Texture) and not isinstance(texture, VideoTexture)
            and max(texture.width, texture.height) > BASE_SIZE)


def base_copy(texture: Texture) -> Texture:
    """The mipmap level of a large ``texture`` that is at most :data:`BASE_SIZE` pixels (averaged in linear light,
    :func:`~.textures.mip_chain`), made once per version of its pixels and kept with it: by
    :func:`~.textures.prepare_upload` where the texture was made (a worker thread), else now."""
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
    arrays in RGB order), laid out for the upload with their mipmaps (on the calling thread). Those read or made here
    give their pixels up once the renderer has them on the GPU (``Texture.drop_after_upload``)."""
    return _texture(detail.texture, True), (_texture(detail.maps, False) if detail.maps is not None else None)


def _texture(spec, srgb: bool) -> Texture:
    from .textures import _array_to_rgba, _read_image

    if isinstance(spec, Texture):  # (someone else's: its pixels stay)
        return prepare_upload(spec)
    if isinstance(spec, np.ndarray):
        rgba, name = _array_to_rgba(spec, "rgb"), "detail"
    else:
        rgba, name = _read_image(Path(spec), 1 << 14), Path(spec).name
    texture = Texture(rgba, name=name, srgb=srgb)
    texture.drop_after_upload = True
    return prepare_upload(texture)


def _background() -> ThreadPoolExecutor:
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="detail-textures")
        return _pool


class DetailCache:
    """The detail textures of the heads that need them most (see the module docs), by mesh part."""

    def __init__(self):
        self._entries: dict[int, list] = {}  # id(primitive) -> [weakref, textures or None, job, claim, frame]
        self._dropped: list[Texture] = []  # textures of heads dropped for others, for the renderer to free
        self.loads = 0  # detail textures read (a pair counts once)
        self.refused = 0  # pictures that asked for a head's detail textures and were refused them
        self.frame = 0  # the frame the pictures belong to (Renderer.render starts one)

    def next_frame(self) -> None:
        """A new frame: the claims of the heads its pictures do not draw fade (:data:`FADE`)."""
        self.frame += 1

    def get(self, primitive, eye: float, wait: bool, keep: int) -> tuple[Texture, Texture | None] | None:
        """The detail ``(colour, maps)`` of ``primitive`` (one with a ``detail``), drawn in a picture with its eyes
        ``eye`` pixels apart, at most ``keep`` heads holding theirs: read now when ``wait`` and not read yet (else
        asked for on a background thread) if it is held or may be (see the module docs); ``None`` until they are
        there, when it may not hold them, or when they cannot be read (it is then not tried again)."""
        key = id(primitive)
        entry = self._entries.get(key)
        if entry is not None and entry[0]() is not primitive:  # the id of a mesh part that is gone
            self._dropped += self._drop(key)
            entry = None
        if entry is None:
            if not self._room(eye, keep):
                self.refused += 1
                return None
            entry = self._entries[key] = [weakref.ref(primitive), None, None, 0.0, self.frame]
            if wait:
                entry[1] = self._read(primitive.detail)
            else:
                entry[2] = _background().submit(self._read, primitive.detail)
        entry[3], entry[4] = max(self._claim(entry), float(eye)), self.frame
        if entry[2] is not None and (wait or entry[2].done()):
            job, entry[2] = entry[2], None
            entry[1] = job.result()
        return entry[1] or None

    def _claim(self, entry) -> float:
        return entry[3] * FADE ** (self.frame - entry[4])

    def _room(self, eye: float, keep: int) -> bool:
        """Whether a head drawn with its eyes ``eye`` pixels apart may hold detail textures now: there is room, or it
        outranks the weakest held head that no picture of this frame drew and that is not being read, which is
        dropped for it."""
        if keep <= 0:
            return False
        if len(self._entries) < keep:
            return True
        for entry in self._entries.values():  # reads finished for heads no picture asked for since
            if entry[2] is not None and entry[2].done():
                job, entry[2] = entry[2], None
                entry[1] = job.result()
        free = [(self._claim(entry), key) for key, entry in self._entries.items()
                if entry[4] != self.frame and entry[2] is None]
        if not free:
            return False
        claim, key = min(free)
        if eye <= HYSTERESIS * claim:
            return False
        self._dropped += self._drop(key)
        return len(self._entries) < keep

    def _read(self, detail):
        try:
            textures = load(detail)
        except (OSError, ValueError) as exc:  # a damaged file: the 1024 textures are drawn
            print(f"[renderer] cannot read a head's detail textures, drawing it at 1024: {exc}")
            return False
        self.loads += 1
        return textures

    def trim(self, keep: int) -> list[Texture]:
        """Drop the heads whose mesh part is gone, and the weakest beyond ``keep``; returns the textures dropped since
        the last call (the renderer frees their GPU memory)."""
        dropped, self._dropped = self._dropped, []
        for key in [k for k, entry in self._entries.items() if entry[0]() is None]:
            dropped += self._drop(key)
        while len(self._entries) > max(0, keep):
            dropped += self._drop(min(self._entries, key=lambda k: self._claim(self._entries[k])))
        return dropped

    def clear(self) -> list[Texture]:
        """Drop every head; returns the textures dropped (the renderer frees their GPU memory)."""
        dropped, self._dropped = self._dropped, []
        for key in list(self._entries):
            dropped += self._drop(key)
        return dropped

    def _drop(self, key: int) -> list[Texture]:
        _ref, textures, job, _claim, _frame = self._entries.pop(key)
        if isinstance(job, Future):
            job.cancel()
        return [t for t in (textures or ()) if t is not None]

    def __len__(self) -> int:
        return len(self._entries)

    def textures(self) -> list[Texture]:
        """The detail textures held now."""
        return [t for _ref, textures, _job, _c, _f in self._entries.values() for t in (textures or ()) if t is not None]

    def cpu_bytes(self) -> int:
        """What the held detail textures still take in memory (their pixels, until the renderer has them on the GPU,
        and what was laid out for the upload)."""
        total = 0
        for texture in self.textures():
            if not texture.__dict__.get("pixels_dropped"):
                total += texture.rgba.nbytes
            prepared = texture.__dict__.get("_upload")
            if prepared is not None:
                total += len(prepared[1]) + sum(len(level) for level in (prepared[2] or ()))
        return total
