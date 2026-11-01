"""The camera model: what a real camera does to a window's pixels, applied at the window's own pixel scale (V4).

A render is noiseless, perfectly focused and frozen in time; a camera's picture is not. :meth:`CameraModel.apply`
runs a window through a camera's faults in the order light meets them (``docs/research/high_res_faces.md``, 2.1):

1. **Linear light.** The render's values are taken as sRGB and decoded to linear light, where blur and noise happen.
2. **Lens blur**: a Gaussian of ``lens_blur`` sensor pixels (``lens_blur / k`` window pixels at level ``k``).
3. **Motion blur**: the person moves ``speed x exposure`` window pixels while the shutter is open: a line kernel along
   their projected motion. With the person's coverage (``mask``) only the person is smeared, over the still
   background (``blur(F M) + (1 - blur(M)) B``, with the window itself standing in for the background ``B`` behind
   the person); without it the whole window is.
4. **Bayer mosaic** (optional): one colour per pixel (RGGB), before the noise, as on a sensor.
5. **Noise**: Poisson-Gaussian (Foi et al. 2008) in electrons: a pixel that sees full white collects ``electrons``
   (at this exposure), shot noise with the variance of the signal, plus ``read_noise`` electrons. A binned pixel
   (level ``k``) collects ``k x k`` times the light. The Gaussian approximation of the Poisson part is used (fine
   above a few electrons); values are clipped to the sensor's range.
6. **Demosaicing** with OpenCV (bilinear, edge-aware or VNG) when there is a mosaic.
7. **Tone**: the sRGB curve back, then an S-curve of strength ``tone`` (contrast).
8. **Sharpening**: an unsharp mask of strength ``sharpen``, as cameras do.
9. **Compression**: JPEG at quality ``jpeg``, standing in for the video codec.

Everything random is drawn from a generator seeded by the model's ``seed`` and the window (frame, tag, place), so runs
reproduce. The ``ideal`` preset changes nothing. A preset may bring its own sensor (``lines``: a 4K camera has 2160),
which then replaces the declared sensor for its windows: "budget 1080p" shows what such a camera gives the
recogniser. The presets' numbers are plausible values for such cameras, not measurements of one; calibrate them
against footage of the intended camera when there is some.
"""

from __future__ import annotations

import functools
import zlib
from dataclasses import dataclass, replace

import cv2
import numpy as np

_DEMOSAIC = {  # an RGGB mosaic (red at the top-left pixel) to BGR; OpenCV names it by the second row
    "bilinear": (cv2.COLOR_BayerBG2BGR, np.uint16),
    "ea": (cv2.COLOR_BayerBG2BGR_EA, np.uint16),  # edge-aware
    "vng": (cv2.COLOR_BayerBG2BGR_VNG, np.uint8),  # variable number of gradients (8 bits only)
}


@dataclass(frozen=True)
class CameraModel:
    """A camera's faults (see the module docs). The defaults change nothing (the ideal camera).

    Attributes:
        name: what results and logs call it.
        lines: the camera's sensor height in pixels (its width follows the view's shape); ``None``: the declared
            sensor of the window service.
        lens_blur: Gaussian sigma of the optics, sensor pixels.
        exposure: seconds the shutter is open (motion blur).
        electrons: photo-electrons a pixel collects from full white at this exposure (0: no noise at all).
        read_noise: electrons of read noise per pixel.
        bayer: a colour filter mosaic, demosaiced with ``demosaic`` ("bilinear", "ea" or "vng").
        tone: strength of the contrast S-curve (0-1).
        sharpen: strength of the unsharp mask.
        jpeg: JPEG quality (1-100); ``None``: no compression.
        seed: the base of every random draw.
    """

    name: str = "ideal"
    lines: int | None = None
    lens_blur: float = 0.0
    exposure: float = 0.0
    electrons: float = 0.0
    read_noise: float = 0.0
    bayer: bool = False
    demosaic: str = "ea"
    tone: float = 0.0
    sharpen: float = 0.0
    jpeg: int | None = None
    seed: int = 0

    def __post_init__(self):
        if self.demosaic not in _DEMOSAIC:
            raise ValueError(f"unknown demosaicing {self.demosaic!r}; choose from {', '.join(_DEMOSAIC)}")
        if self.jpeg is not None and not 1 <= self.jpeg <= 100:
            raise ValueError(f"JPEG quality must be 1-100, got {self.jpeg}")

    @property
    def ideal(self) -> bool:
        """Changes nothing."""
        return not (self.lens_blur > 0 or self.exposure > 0 or self.electrons > 0 or self.bayer or self.tone > 0
                    or self.sharpen > 0 or self.jpeg is not None)

    @property
    def needs_motion(self) -> bool:
        """Uses the person's speed (and coverage)."""
        return self.exposure > 0

    def with_exposure(self, seconds: float) -> CameraModel:
        """The same camera with another exposure: as much less (or more) light, so more (or less) noise."""
        scale = seconds / self.exposure if self.exposure > 0 else 1.0
        return replace(self, exposure=float(seconds), electrons=self.electrons * scale)

    def window_seed(self, frame: int, tag, rect) -> int:
        """The seed of a window's random draws: this model's seed, the frame, the tag and the window's place."""
        return zlib.crc32(repr((self.seed, int(frame), repr(tag), tuple(int(v) for v in rect))).encode())

    def apply(self, image: np.ndarray, level: int = 1, velocity=(0.0, 0.0), mask: np.ndarray | None = None,
              seed: int = 0) -> np.ndarray:
        """``image`` (HxWx3 BGR ``uint8``, a window at binning ``level``) as this camera delivers it. ``velocity``:
        the person's motion in window pixels a second; ``mask``: their coverage (HxW, 0-1), so that only they are
        smeared; ``seed``: for the noise (see :meth:`window_seed`)."""
        if self.ideal:
            return image
        level = max(1, int(level))
        light = cv2.LUT(np.ascontiguousarray(image), _DECODE_TABLE)  # sRGB to linear light, by table
        if self.lens_blur > 0:
            sigma = self.lens_blur / level
            if sigma >= 0.05:
                light = cv2.GaussianBlur(light, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)
        if self.exposure > 0:
            kernel = motion_kernel(np.asarray(velocity, dtype=np.float64) * self.exposure)
            if kernel is not None:
                light = _smear(light, kernel, mask)
        rng = np.random.default_rng(seed)
        if self.bayer:
            raw = mosaic(light)
            raw = self._noise(raw, level, rng)
            light = demosaiced(raw, self.demosaic)
        else:
            light = self._noise(light, level, rng)
        index = np.clip(light, 0.0, 1.0)  # the sRGB curve and the S-curve, by a table of the linear values
        index *= len(_LEVELS) - 1
        index += 0.5
        out = np.take(_encoder(round(float(self.tone), 6)), index.astype(np.uint16))
        if self.sharpen > 0:
            out = out + self.sharpen * (out - cv2.GaussianBlur(out, (0, 0), 1.0, borderType=cv2.BORDER_REFLECT))
        result = np.clip(np.rint(out * 255.0), 0, 255).astype(np.uint8)
        if self.jpeg is not None:
            ok, data = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, int(self.jpeg)])
            if ok:
                result = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return result

    def _noise(self, light: np.ndarray, level: int, rng: np.random.Generator) -> np.ndarray:
        """Poisson-Gaussian noise for pixels of ``level x level`` sensor pixels (values in units of full white)."""
        if self.electrons <= 0:
            return light
        full = self.electrons * level * level  # electrons a binned pixel collects from full white
        signal = np.clip(light, 0.0, None) * full
        reads = level * level * self.read_noise ** 2  # each of the binned pixels is read
        noisy = signal + np.sqrt(signal + reads) * rng.standard_normal(light.shape, dtype=np.float32)
        return np.clip(noisy / full, 0.0, 1.0).astype(np.float32)


PRESETS: dict[str, CameraModel] = {
    preset.name: preset
    for preset in (
        CameraModel("ideal"),
        # a good 8K camera: the declared default sensor's size, small pixels (fewer electrons), a fast shutter
        CameraModel("good 8K", lines=4320, lens_blur=0.7, exposure=1 / 250, electrons=2500, read_noise=1.5,
                    bayer=True, demosaic="ea", tone=0.25, sharpen=0.3, jpeg=90),
        CameraModel("good 4K", lines=2160, lens_blur=0.6, exposure=1 / 250, electrons=6000, read_noise=2.0,
                    bayer=True, demosaic="ea", tone=0.25, sharpen=0.3, jpeg=90),
        # a cheap 1080p camera: a soft lens, a slow shutter indoors, a small sensor, strong sharpening and compression
        CameraModel("budget 1080p", lines=1080, lens_blur=1.0, exposure=1 / 60, electrons=1500, read_noise=4.0,
                    bayer=True, demosaic="bilinear", tone=0.4, sharpen=0.6, jpeg=70),
    )
}


def camera_preset(preset) -> CameraModel:
    """A :class:`CameraModel`, or the preset of that name (case and ``_`` / ``-`` / space do not matter)."""
    if isinstance(preset, CameraModel):
        return preset
    if preset is None:
        return PRESETS["ideal"]
    key = _key(str(preset))
    for name, model in PRESETS.items():
        if _key(name) == key:
            return model
    raise ValueError(f"unknown camera preset {preset!r}; choose from {', '.join(repr(n) for n in PRESETS)}")


def _key(name: str) -> str:
    return name.lower().replace("_", " ").replace("-", " ").strip()


# -- the steps --------------------------------------------------------------------------------------------------------


def srgb_to_linear(values: np.ndarray) -> np.ndarray:
    """The sRGB decoding (values 0-1)."""
    values = np.asarray(values, dtype=np.float64)
    return np.where(values <= 0.04045, values / 12.92, ((values + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(values: np.ndarray) -> np.ndarray:
    """The sRGB encoding (values 0-1)."""
    values = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    return np.where(values <= 0.0031308, values * 12.92, 1.055 * values ** (1 / 2.4) - 0.055).astype(np.float32)


_DECODE_TABLE = srgb_to_linear(np.arange(256) / 255.0).reshape(1, 256)  # 8-bit sRGB to linear light (cv2.LUT)
_LEVELS = np.linspace(0.0, 1.0, 65536)  # linear light, finely enough for 8-bit output (16-bit steps)


@functools.lru_cache(maxsize=8)
def _encoder(tone: float) -> np.ndarray:
    """Linear light (65536 steps) to sRGB, then an S-curve of strength ``tone``: a smoothstep blend."""
    out = linear_to_srgb(_LEVELS).astype(np.float64)
    if tone > 0:
        out = (1.0 - tone) * out + tone * out * out * (3.0 - 2.0 * out)
    return out.astype(np.float32)


def motion_kernel(shift) -> np.ndarray | None:
    """The normalised kernel of a move of ``shift`` = ``(dx, dy)`` pixels during the exposure: how long the moving
    point spends over each pixel, centred on the middle of the move (the frame's moment), as small as the path
    (``None`` below half a pixel)."""
    dx, dy = (float(v) for v in shift)
    length = float(np.hypot(dx, dy))
    if length < 0.5:
        return None
    steps = max(64, int(np.ceil(length * 16)))
    t = (np.arange(steps) + 0.5) / steps - 0.5  # evenly along the path, symmetric about its middle
    xs, ys = np.rint(t * dx).astype(np.int64), np.rint(t * dy).astype(np.int64)
    rx, ry = int(np.abs(xs).max()), int(np.abs(ys).max())
    kernel = np.zeros((2 * ry + 1, 2 * rx + 1), np.float32)
    np.add.at(kernel, (ys + ry, xs + rx), 1.0)
    return kernel / steps


def _smear(light: np.ndarray, kernel: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """``light`` seen through a line kernel: all of it, or only the masked layer over the rest."""
    if mask is None:
        return cv2.filter2D(light, -1, kernel, borderType=cv2.BORDER_REFLECT)
    m = np.clip(mask.astype(np.float32), 0.0, 1.0)[..., None]
    if not m.any():
        return light
    layer = cv2.filter2D(light * m, -1, kernel, borderType=cv2.BORDER_REFLECT)
    cover = cv2.filter2D(m[..., 0], -1, kernel, borderType=cv2.BORDER_REFLECT)[..., None]
    return layer + (1.0 - cover) * light


def mosaic(light: np.ndarray) -> np.ndarray:
    """One colour per pixel, RGGB (red at the top-left pixel), from a BGR image."""
    raw = np.empty(light.shape[:2], np.float32)
    raw[0::2, 0::2] = light[0::2, 0::2, 2]  # red
    raw[0::2, 1::2] = light[0::2, 1::2, 1]  # green
    raw[1::2, 0::2] = light[1::2, 0::2, 1]  # green
    raw[1::2, 1::2] = light[1::2, 1::2, 0]  # blue
    return raw


def demosaiced(raw: np.ndarray, method: str = "ea") -> np.ndarray:
    """A BGR image (linear, 0-1) from an RGGB mosaic (0-1), by OpenCV's ``method``."""
    code, dtype = _DEMOSAIC[method]
    top = np.iinfo(dtype).max
    quantised = np.clip(np.rint(raw * top), 0, top).astype(dtype)
    return cv2.cvtColor(quantised, code).astype(np.float32) / top
