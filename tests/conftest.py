import functools

import numpy as np
import pytest


@functools.lru_cache(maxsize=1)
def _gl_error() -> str | None:
    try:
        import moderngl

        moderngl.create_standalone_context(require=330).release()
    except Exception as exc:  # noqa: BLE001
        return f"no OpenGL 3.3 context available: {exc}"
    return None


@pytest.fixture
def gl():
    """Skip the test when no OpenGL context can be created (e.g. a Linux box without a display)."""
    error = _gl_error()
    if error:
        pytest.skip(error)


@pytest.fixture
def two_tone_png(tmp_path):
    """A 40x20 image: red on top, blue at the bottom (written as BGR by OpenCV)."""
    import cv2

    image = np.zeros((20, 40, 3), np.uint8)
    image[:10] = (0, 0, 255)  # red (BGR)
    image[10:] = (255, 0, 0)  # blue
    path = tmp_path / "two_tone.png"
    cv2.imwrite(str(path), image)
    return path
