"""The camera of a photo of a head: a pinhole camera in the avatar's bind-pose character space.

OpenCV's conventions, so the numbers go straight into ``cv2.solvePnP``: the camera looks along its +z, +x is to
the right in the image and +y down, and pixel centres sit at whole numbers (the top-left pixel is ``(0, 0)``).
``R`` and ``t`` take a point from character space (metres, Y up, the person facing +Z, +X their left) into the
camera: ``p_camera = R @ p + t``.

The head's pose in a photo is given as three angles, the way a person would describe it:

* ``yaw``: the head turned to its own left (positive) or right, as :class:`~bifold_demo.characters.model.CharacterState`
  counts ``head_yaw``; 90 shows the right side of the head, -90 the left side, 180 the back;
* ``pitch``: looking down (positive) or up;
* ``roll``: leaning towards the left shoulder (positive) or the right one.

They are the rotation of the head relative to a camera straight in front of it, upright:
``Ry(yaw) @ Rx(pitch) @ Rz(-roll)``, so the yaw is applied last.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

FRONT = np.diag([1.0, -1.0, -1.0])  # character space -> a camera in front of the face, upright (x right, y down)


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def head_rotation(yaw: float, pitch: float = 0.0, roll: float = 0.0) -> np.ndarray:
    """The 3x3 rotation of a head turned by ``yaw``, ``pitch`` and ``roll`` (degrees, see the module docs)."""
    return _ry(math.radians(yaw)) @ _rx(math.radians(pitch)) @ _rz(-math.radians(roll))


def head_angles(rotation: np.ndarray) -> tuple[float, float, float]:
    """``(yaw, pitch, roll)`` in degrees of a head rotation (the inverse of :func:`head_rotation`)."""
    h = np.asarray(rotation, float)
    pitch = math.asin(max(-1.0, min(1.0, -h[1, 2])))
    yaw = math.atan2(h[0, 2], h[2, 2])
    roll = -math.atan2(h[1, 0], h[1, 1])
    return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)


def focal_from_fov(fov_y: float, height: int) -> float:
    """The focal length in pixels of a vertical field of view ``fov_y`` (degrees) on an image ``height`` tall."""
    return 0.5 * height / math.tan(math.radians(fov_y) / 2.0)


@dataclass
class PhotoCamera:
    """A pinhole camera: intrinsics ``K`` (3x3), extrinsics ``R`` (3x3) and ``t`` (3,), image ``size`` (w, h)."""

    K: np.ndarray
    R: np.ndarray
    t: np.ndarray
    size: tuple[int, int]

    def __post_init__(self):
        self.K = np.asarray(self.K, np.float64).reshape(3, 3)
        self.R = np.asarray(self.R, np.float64).reshape(3, 3)
        self.t = np.asarray(self.t, np.float64).reshape(3)
        self.size = (int(self.size[0]), int(self.size[1]))

    @property
    def focal(self) -> float:
        return float(self.K[0, 0])

    @property
    def position(self) -> np.ndarray:
        """Where the camera is, in character space."""
        return -self.R.T @ self.t

    @property
    def forward(self) -> np.ndarray:
        """The direction the camera looks, in character space."""
        return self.R[2].copy()

    @property
    def fov(self) -> float:
        """The vertical field of view in degrees."""
        return math.degrees(2.0 * math.atan(0.5 * self.size[1] / self.K[1, 1]))

    @property
    def rotation(self) -> np.ndarray:
        """The head's rotation seen from this camera (see the module docs)."""
        return FRONT.T @ self.R

    def angles(self) -> tuple[float, float, float]:
        """``(yaw, pitch, roll)`` of the head in this photo, in degrees."""
        return head_angles(self.rotation)

    def to_camera(self, points) -> np.ndarray:
        return np.asarray(points, np.float64).reshape(-1, 3) @ self.R.T + self.t

    def project(self, points) -> np.ndarray:
        """``(N, 2)`` pixel positions of character-space ``points`` (in front of the camera)."""
        p = self.to_camera(points)
        z = np.where(np.abs(p[:, 2]) < 1e-12, 1e-12, p[:, 2])
        k = self.K
        return np.c_[k[0, 0] * p[:, 0] / z + k[0, 1] * p[:, 1] / z + k[0, 2], k[1, 1] * p[:, 1] / z + k[1, 2]]

    def depth(self, points) -> np.ndarray:
        """Distance of ``points`` along the viewing direction."""
        return self.to_camera(points)[:, 2]

    def rays(self, pixels) -> tuple[np.ndarray, np.ndarray]:
        """The camera position and ``(N, 3)`` unit directions, in character space, of the rays through ``pixels``."""
        pixels = np.asarray(pixels, np.float64).reshape(-1, 2)
        local = np.c_[pixels, np.ones(len(pixels))] @ np.linalg.inv(self.K).T
        directions = local @ self.R  # R.T applied to each row
        return self.position, directions / np.linalg.norm(directions, axis=1, keepdims=True)

    @classmethod
    def intrinsics(cls, focal: float, size) -> np.ndarray:
        """``K`` of a camera with square pixels and the principal point in the middle of an image of ``size``."""
        w, h = size
        return np.array([[focal, 0.0, (w - 1) / 2.0], [0.0, focal, (h - 1) / 2.0], [0.0, 0.0, 1.0]])

    @classmethod
    def looking_at(cls, eye, target, fov: float, size, up=(0.0, 1.0, 0.0)) -> PhotoCamera:
        """The camera at ``eye`` looking at ``target`` with a vertical field of view ``fov`` (degrees), as
        :class:`bifold_demo.scene.Camera` would show it on a renderer of ``size``."""
        eye, target, up = (np.asarray(v, np.float64) for v in (eye, target, up))
        forward = target - eye
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        r = np.stack([right, down, forward])
        return cls(cls.intrinsics(focal_from_fov(fov, size[1]), size), r, -r @ eye, size)

    @classmethod
    def around(cls, target, distance: float, yaw: float = 0.0, pitch: float = 0.0, roll: float = 0.0, *,
               fov: float = 30.0, size=(512, 512), focal: float | None = None) -> PhotoCamera:
        """A camera ``distance`` metres from ``target`` (character space) showing the head turned by ``yaw``,
        ``pitch`` and ``roll`` (degrees), with ``target`` in the middle of the picture."""
        r = FRONT @ head_rotation(yaw, pitch, roll)
        target = np.asarray(target, np.float64)
        f = focal if focal is not None else focal_from_fov(fov, size[1])
        return cls(cls.intrinsics(f, size), r, np.array([0.0, 0.0, distance]) - r @ target, size)

    def scene_camera(self, near: float = 0.05, far: float = 50.0):
        """A :class:`bifold_demo.scene.Camera` that renders what this camera sees (on a renderer of ``size``), for
        a camera with its principal point in the middle of the image (the renderer has no other)."""
        from ...scene import Camera

        position = self.position
        return Camera(position=tuple(position), target=tuple(position + self.R[2]), fov=self.fov,
                      up=tuple(-self.R[1]), near=near, far=far)

    def scaled(self, factor: float) -> PhotoCamera:
        """The same camera for the image resized by ``factor``."""
        k = self.K.copy()
        k[:2] *= factor
        k[:2, 2] = (self.K[:2, 2] + 0.5) * factor - 0.5  # pixel centres at whole numbers
        return PhotoCamera(k, self.R, self.t, (round(self.size[0] * factor), round(self.size[1] * factor)))
