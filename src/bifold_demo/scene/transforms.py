"""4x4 matrix helpers (column-vector convention, angles in degrees)."""

from __future__ import annotations

import math

import numpy as np


def vec3(value, name: str = "value") -> np.ndarray:
    """Coerce a scalar or 3-sequence into a float64 array of shape (3,)."""
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        return np.full(3, float(arr))
    if arr.shape != (3,):
        raise ValueError(f"{name} must be a number or a sequence of 3 numbers, got {value!r}")
    return arr.copy()


def translation(offset) -> np.ndarray:
    m = np.eye(4)
    m[:3, 3] = vec3(offset, "offset")
    return m


def scaling(factor) -> np.ndarray:
    return np.diag([*vec3(factor, "scale"), 1.0])


def _axis_rotation(axis: int, degrees: float) -> np.ndarray:
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    i, j = [(1, 2), (2, 0), (0, 1)][axis]
    m = np.eye(4)
    m[i, i], m[i, j] = c, -s
    m[j, i], m[j, j] = s, c
    return m


def rotation_x(degrees: float) -> np.ndarray:
    return _axis_rotation(0, degrees)


def rotation_y(degrees: float) -> np.ndarray:
    return _axis_rotation(1, degrees)


def rotation_z(degrees: float) -> np.ndarray:
    return _axis_rotation(2, degrees)


def euler_rotation(degrees) -> np.ndarray:
    """Rotation from Euler angles ``(x, y, z)`` in degrees, applied in X, Y, Z order."""
    rx, ry, rz = vec3(degrees, "rotation")
    return rotation_z(rz) @ rotation_y(ry) @ rotation_x(rx)


def perspective(fov_y: float, aspect: float, near: float, far: float) -> np.ndarray:
    """OpenGL-style perspective projection; ``fov_y`` is the vertical field of view in degrees."""
    f = 1.0 / math.tan(math.radians(fov_y) / 2.0)
    m = np.zeros((4, 4))
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2.0 * far * near / (near - far)
    m[3, 2] = -1.0
    return m


def look_at(eye, target, up=(0.0, 1.0, 0.0)) -> np.ndarray:
    """View matrix for a camera at ``eye`` looking at ``target``."""
    eye, target, up = vec3(eye), vec3(target), vec3(up)
    forward = target - eye
    norm = np.linalg.norm(forward)
    forward = forward / norm if norm > 1e-9 else np.array([0.0, 0.0, -1.0])
    side = np.cross(forward, up)
    if np.linalg.norm(side) < 1e-6:  # looking straight along `up`
        side = np.cross(forward, [0.0, 0.0, -1.0] if abs(forward[1]) > 0.5 else [0.0, 1.0, 0.0])
    side /= np.linalg.norm(side)
    true_up = np.cross(side, forward)
    m = np.eye(4)
    m[0, :3], m[1, :3], m[2, :3] = side, true_up, -forward
    m[:3, 3] = -m[:3, :3] @ eye
    return m


def gl_bytes(matrix: np.ndarray) -> bytes:
    """Serialize a matrix in the column-major float32 layout OpenGL expects."""
    return np.ascontiguousarray(matrix.T, dtype="f4").tobytes()
