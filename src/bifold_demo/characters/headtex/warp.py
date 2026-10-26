"""A smooth warp of the image plane: a thin-plate spline through control points.

``ThinPlateWarp(source, target)`` is the smoothest map of the plane (least bending energy) that takes every
``source[i]`` exactly to ``target[i]``: an affine part plus a sum of ``r^2 log r`` bumps centred on the control
points. With ``smoothing > 0`` it only comes close to the targets and bends less. A spline keeps bending beyond its
control points, so ``falloff=(centre, inner, outer)`` fades the movement out between the two radii around ``centre``:
the warp is the spline within ``inner`` and leaves everything beyond ``outer`` where it is. Stage 1 uses it as a
backward map: from where the avatar's landmarks are in a photo to where the person's are, so sampling the photo
through it puts the person's eyes, brows, nose and mouth on the avatar's and leaves the hair and the background alone.
"""

from __future__ import annotations

import cv2
import numpy as np


def _kernel(r2: np.ndarray) -> np.ndarray:
    """``r^2 log r`` from squared distances (0 at r = 0)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(r2 > 1e-20, 0.5 * r2 * np.log(np.maximum(r2, 1e-20)), 0.0)


class ThinPlateWarp:
    """The thin-plate spline taking ``source`` points ``(K, 2)`` to ``target`` points ``(K, 2)``."""

    def __init__(self, source=(), target=(), smoothing: float = 0.0, falloff=None):
        self.falloff = None if falloff is None else (np.asarray(falloff[0], np.float64).reshape(2),
                                                     float(falloff[1]), float(falloff[2]))
        source = np.asarray(source, np.float64).reshape(-1, 2)
        target = np.asarray(target, np.float64).reshape(-1, 2)
        if source.shape != target.shape:
            raise ValueError("source and target need the same number of points")
        self.source, self.target = source, target
        if len(source) < 3:  # too few points for a spline: the identity, or a shift
            self._shift = (target - source).mean(axis=0) if len(source) else np.zeros(2)
            self._weights = None
            return
        # in normalised coordinates, so the system is well conditioned whatever the image size
        self._centre = source.mean(axis=0)
        self._scale = float(np.sqrt(((source - self._centre) ** 2).sum(axis=1).mean())) or 1.0
        s = (source - self._centre) / self._scale
        t = (target - self._centre) / self._scale
        k = len(s)
        a = np.zeros((k + 3, k + 3))
        a[:k, :k] = _kernel(((s[:, None] - s[None]) ** 2).sum(axis=2)) + smoothing * np.eye(k)
        a[:k, k:] = np.c_[np.ones(k), s]
        a[k:, :k] = a[:k, k:].T
        b = np.zeros((k + 3, 2))
        b[:k] = t
        try:
            solution = np.linalg.solve(a, b)
        except np.linalg.LinAlgError:  # repeated control points: the least-squares solution
            solution = np.linalg.lstsq(a, b, rcond=None)[0]
        self._weights, self._affine = solution[:k], solution[k:]
        self._points = s

    @classmethod
    def identity(cls) -> ThinPlateWarp:
        return cls()

    def __call__(self, points) -> np.ndarray:
        """Where the warp takes ``points`` (``(N, 2)``)."""
        points = np.asarray(points, np.float64).reshape(-1, 2)
        moved = self._spline(points)
        if self.falloff is None:
            return moved
        centre, inner, outer = self.falloff
        u = np.clip((np.linalg.norm(points - centre, axis=1) - inner) / max(outer - inner, 1e-9), 0.0, 1.0)
        keep = 1.0 - u * u * (3.0 - 2.0 * u)  # 1 within inner, 0 beyond outer, smooth in between
        return points + (moved - points) * keep[:, None]

    def _spline(self, points: np.ndarray) -> np.ndarray:
        if self._weights is None:
            return points + self._shift
        p = (points - self._centre) / self._scale
        out = np.empty_like(p)
        for start in range(0, len(p), 4096):
            chunk = p[start:start + 4096]
            u = _kernel(((chunk[:, None] - self._points[None]) ** 2).sum(axis=2))
            out[start:start + 4096] = u @ self._weights + self._affine[0] + chunk @ self._affine[1:]
        return out * self._scale + self._centre

    def bending_energy(self) -> float:
        """The spline's bending energy (0 for an affine map), in normalised coordinates."""
        if self._weights is None:
            return 0.0
        k = _kernel(((self._points[:, None] - self._points[None]) ** 2).sum(axis=2))
        return float(np.trace(self._weights.T @ k @ self._weights))

    def jacobian_determinant(self, points, step: float = 0.5) -> np.ndarray:
        """``(N,)`` area scale of the warp at ``points`` (negative where it folds the plane over)."""
        points = np.asarray(points, np.float64).reshape(-1, 2)
        dx, dy = np.array([step, 0.0]), np.array([0.0, step])
        jx = (self(points + dx) - self(points - dx)) / (2 * step)
        jy = (self(points + dy) - self(points - dy)) / (2 * step)
        return jx[:, 0] * jy[:, 1] - jx[:, 1] * jy[:, 0]

    def maps(self, size, step: int = 4) -> tuple[np.ndarray, np.ndarray]:
        """``cv2.remap`` maps of an image of ``size`` (w, h): pixel ``(x, y)`` of the result shows the source image at
        the warp of ``(x, y)``. The spline is evaluated every ``step`` pixels and interpolated in between."""
        w, h = int(size[0]), int(size[1])
        if self._weights is None:
            gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
            return gx + np.float32(self._shift[0]), gy + np.float32(self._shift[1])
        cols = np.r_[np.arange(0, w - 1, step), w - 1].astype(np.float64)
        rows = np.r_[np.arange(0, h - 1, step), h - 1].astype(np.float64)
        gx, gy = np.meshgrid(cols, rows)
        mapped = self(np.c_[gx.ravel(), gy.ravel()]).reshape(len(rows), len(cols), 2)
        # bilinear in between (the grid's last row and column sit on the image's edge, so it is exact there)
        fx = np.interp(np.arange(w), cols, np.arange(len(cols)))
        fy = np.interp(np.arange(h), rows, np.arange(len(rows)))
        coarse_x = mapped[..., 0].astype(np.float32)
        coarse_y = mapped[..., 1].astype(np.float32)
        mx, my = np.meshgrid(fx.astype(np.float32), fy.astype(np.float32))
        map_x = cv2.remap(coarse_x, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        map_y = cv2.remap(coarse_y, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return map_x, map_y

    def apply(self, image: np.ndarray, size=None, step: int = 4) -> np.ndarray:
        """``image`` resampled through the warp (the result has ``size``, default the image's)."""
        size = size or (image.shape[1], image.shape[0])
        map_x, map_y = self.maps(size, step)
        return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
