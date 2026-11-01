"""A small z-buffer: which triangle of a mesh each pixel of a camera's picture shows, and how far away it is.

Stages 3 and 4 need it in three places: the depth test of every texel of the head texture against the head mesh as a
photo's camera sees it (the head hides parts of itself: the nose a strip of the far cheek, an ear the skin behind it,
the chin the neck), finding the parts of the head that no camera outside ever sees (the inside of the mouth), and
pictures of synthetic heads for the tests. It is plain numpy, so it runs without OpenGL: the pixels of every
triangle's bounding box are enumerated at once, each pixel centre is tested against the triangle's edges, its depth is
interpolated perspective-correctly, and the nearest triangle wins each pixel. A head of 2000 triangles, 400 pixels
tall, takes about 20 ms.

Pixel centres sit at whole numbers, as in :mod:`.camera`; depths are distances along the camera's viewing direction.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .camera import PhotoCamera

NEAR = 1e-3  # metres: triangles with a corner nearer the camera than this are not drawn
CHUNK = 1 << 21  # candidate pixels per batch


@dataclass
class Raster:
    """What a camera sees of a mesh: per pixel the nearest surface's depth (``inf`` where there is none), its
    triangle (``-1`` there) and, when asked for, the perspective-correct barycentric coordinates of the point."""

    depth: np.ndarray  # (H, W) float32
    triangle: np.ndarray  # (H, W) int32
    barycentric: np.ndarray | None = None  # (H, W, 3) float32

    @property
    def shape(self) -> tuple[int, int]:
        return self.depth.shape

    def nearest(self, radius: int = 1) -> np.ndarray:
        """The depth map with each pixel the nearest depth within ``radius`` pixels: a texel next to the edge of a
        nearer surface (the side of the nose) then counts as hidden, so samples never straddle the edge."""
        if radius <= 0:
            return self.depth
        finite = np.where(np.isfinite(self.depth), self.depth, np.float32(3e38)).astype(np.float32)
        kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
        return cv2.erode(finite, kernel, borderType=cv2.BORDER_REPLICATE)


def rasterise(camera: PhotoCamera, corners: np.ndarray, *, cull: bool = False, barycentric: bool = False) -> Raster:
    """The z-buffer of the triangles ``corners`` (``(T, 3, 3)``, character space) seen by ``camera``, at its image
    size. ``cull``: leave out the triangles facing away from the camera (by their corner order, counter-clockwise
    seen from outside). ``barycentric``: also keep each pixel's barycentric coordinates."""
    corners = np.asarray(corners, np.float64).reshape(-1, 3, 3)
    w, h = camera.size
    depth = np.full(h * w, np.inf, np.float32)
    triangle = np.full(h * w, -1, np.int32)
    bary = np.zeros((h * w, 3), np.float32) if barycentric else None
    if len(corners) == 0:
        return _raster(depth, triangle, bary, h, w)
    flat = corners.reshape(-1, 3)
    pixels = camera.project(flat).reshape(-1, 3, 2)
    z = camera.depth(flat).reshape(-1, 3)
    keep = (z > NEAR).all(axis=1)
    if cull:
        normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        keep &= np.einsum("tk,tk->t", normals, camera.position - corners.mean(axis=1)) > 0
    lo = np.ceil(pixels.min(axis=1)).astype(np.int64)
    hi = np.floor(pixels.max(axis=1)).astype(np.int64)
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi, np.array([w - 1, h - 1]))
    keep &= (hi >= lo).all(axis=1)
    a, b, c = pixels[:, 0], pixels[:, 1], pixels[:, 2]
    area = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    keep &= np.abs(area) > 1e-9
    ids = np.nonzero(keep)[0]
    if len(ids) == 0:
        return _raster(depth, triangle, bary, h, w)
    widths = hi[ids, 0] - lo[ids, 0] + 1
    counts = widths * (hi[ids, 1] - lo[ids, 1] + 1)
    ends = np.cumsum(counts)
    start = 0
    while start < len(ids):
        stop = int(np.searchsorted(ends, (ends[start - 1] if start else 0) + CHUNK, side="right"))
        stop = max(stop, start + 1)
        _draw(ids[start:stop], counts[start:stop], widths[start:stop], lo, pixels, z, area, w, depth, triangle, bary)
        start = stop
    return _raster(depth, triangle, bary, h, w)


def _raster(depth, triangle, bary, h, w) -> Raster:
    return Raster(depth.reshape(h, w), triangle.reshape(h, w), None if bary is None else bary.reshape(h, w, 3))


def _draw(ids, counts, widths, lo, pixels, z, area, w, depth, triangle, bary) -> None:
    """One batch of triangles: every pixel of their boxes tested, the nearest inside each pixel kept."""
    owner = np.repeat(np.arange(len(ids)), counts)  # the batch's triangle of each candidate pixel
    first = np.cumsum(counts) - counts
    local = np.arange(int(counts.sum())) - first[owner]
    t = ids[owner]
    x = lo[t, 0] + local % widths[owner]
    y = lo[t, 1] + local // widths[owner]
    px, py = x.astype(np.float64), y.astype(np.float64)
    a, b, c = pixels[t, 0], pixels[t, 1], pixels[t, 2]
    # edge functions: the weight of each corner is the area of the triangle the pixel makes with the other two
    w0 = ((b[:, 0] - px) * (c[:, 1] - py) - (b[:, 1] - py) * (c[:, 0] - px)) / area[t]
    w1 = ((c[:, 0] - px) * (a[:, 1] - py) - (c[:, 1] - py) * (a[:, 0] - px)) / area[t]
    w2 = 1.0 - w0 - w1
    eps = -1e-7
    inside = (w0 >= eps) & (w1 >= eps) & (w2 >= eps)
    if not inside.any():
        return
    t, x, y = t[inside], x[inside], y[inside]
    q = np.c_[w0[inside], w1[inside], w2[inside]] / z[t]  # perspective-correct: 1/z is linear on the screen
    inverse = q.sum(axis=1)
    d = (1.0 / inverse).astype(np.float32)
    index = y * w + x
    np.minimum.at(depth, index, d)  # the nearest depth of each pixel so far
    win = d <= depth[index]  # (of candidates at the same depth, any one wins)
    index = index[win]
    triangle[index] = t[win]
    if bary is not None:
        bary[index] = (q[win] / inverse[win, None]).astype(np.float32)


@dataclass
class Crop:
    """A part ``box`` (``x0, y0, x1, y1``, pixels) of a camera's picture, scaled by ``factor``, and the ``camera``
    that sees it."""

    camera: PhotoCamera
    box: tuple[int, int, int, int]
    factor: float

    def pixels(self, pixels) -> np.ndarray:
        """Pixels of the original picture (``(N, 2)``) as pixels of the crop."""
        return (np.asarray(pixels, np.float64) - self.box[:2] + 0.5) * self.factor - 0.5

    def image(self, image: np.ndarray) -> np.ndarray:
        """The crop of ``image`` (the original picture, or anything at its size; area-averaged when scaled down)."""
        x0, y0, x1, y1 = self.box
        part = image[y0:y1, x0:x1]
        if self.factor == 1.0:
            return part
        # (the same rounding as PhotoCamera.scaled: the result has the camera's size)
        return cv2.resize(part, None, fx=self.factor, fy=self.factor, interpolation=cv2.INTER_AREA)


def crop(camera: PhotoCamera, points, margin: float = 0.1, longest: int | None = None) -> Crop:
    """The part of ``camera``'s picture that holds ``points`` (``(N, 3)``, with ``margin`` of their extent around
    them, inside the picture), scaled down so its longer side is at most ``longest`` pixels: the z-buffer of a head
    in a 12-megapixel photo needs only the head, at a few hundred pixels."""
    projected = camera.project(np.asarray(points, np.float64).reshape(-1, 3))
    lo, hi = projected.min(axis=0), projected.max(axis=0)
    pad = margin * float((hi - lo).max())
    w, h = camera.size
    x0, y0 = max(0, int(np.floor(lo[0] - pad))), max(0, int(np.floor(lo[1] - pad)))
    x1, y1 = min(w, int(np.ceil(hi[0] + pad)) + 1), min(h, int(np.ceil(hi[1] + pad)) + 1)
    x0, y0 = min(x0, w - 1), min(y0, h - 1)
    x1, y1 = max(x1, x0 + 1), max(y1, y0 + 1)
    k = camera.K.copy()
    k[0, 2] -= x0
    k[1, 2] -= y0
    part = PhotoCamera(k, camera.R, camera.t, (x1 - x0, y1 - y0))
    factor = 1.0
    if longest is not None and max(part.size) > longest:
        factor = longest / max(part.size)
        part = part.scaled(factor)
    return Crop(part, (x0, y0, x1, y1), factor)
