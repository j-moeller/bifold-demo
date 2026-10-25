import os
import time

import cv2
import numpy as np
import pytest

from bifold_demo.scene import Texture, VideoTexture, as_texture, checkerboard, gradient
from bifold_demo.scene.textures import parse_color


def test_colours():
    assert parse_color("#ff8800") == (255, 136, 0)
    assert parse_color("#f80") == (255, 136, 0)
    assert parse_color((10, 20, 30)) == (10, 20, 30)
    tex = as_texture((10, 20, 30))
    assert tex.rgba[0, 0].tolist() == [10, 20, 30, 255]
    assert as_texture("#0a141e") is tex  # solid colours are shared


def test_numpy_arrays_are_bgr_by_default():
    bgr = np.zeros((2, 2, 3), np.uint8)
    bgr[..., 2] = 255  # red in OpenCV order
    assert as_texture(bgr).rgba[0, 0].tolist() == [255, 0, 0, 255]
    assert Texture.from_array(bgr, order="rgb").rgba[0, 0].tolist() == [0, 0, 255, 255]


def test_grayscale_float_and_alpha_arrays():
    assert as_texture(np.full((2, 2), 0.5)).rgba[0, 0].tolist() == [127, 127, 127, 255]
    bgra = np.zeros((2, 2, 4), np.uint8)
    bgra[..., 3] = 7
    assert as_texture(bgra).rgba[0, 0, 3] == 7


def test_pil_images():
    from PIL import Image

    tex = as_texture(Image.new("RGB", (4, 2), (1, 2, 3)))
    assert (tex.width, tex.height) == (4, 2)
    assert tex.rgba[0, 0].tolist() == [1, 2, 3, 255]


def test_file_texture_is_cached_and_reloads_when_file_changes(two_tone_png, monkeypatch):
    from bifold_demo.scene import textures

    monkeypatch.setattr(textures._FileCache, "check_interval", 0.0)
    tex = as_texture(str(two_tone_png))
    assert tex is as_texture(two_tone_png)  # same object for str and Path
    assert tex.rgba[0, 0].tolist() == [255, 0, 0, 255]  # top row red
    assert tex.aspect == 2.0
    version = tex.version

    cv2.imwrite(str(two_tone_png), np.full((10, 10, 3), (0, 255, 0), np.uint8))
    stamp = time.time() + 5
    os.utime(two_tone_png, (stamp, stamp))
    assert as_texture(two_tone_png) is tex
    assert tex.version > version
    assert tex.rgba[0, 0].tolist() == [0, 255, 0, 255] and tex.aspect == 1.0


def test_missing_file_and_bad_values():
    with pytest.raises(FileNotFoundError):
        as_texture("does/not/exist.png")
    with pytest.raises(TypeError):
        as_texture(42)


def test_update_accepts_opencv_images():
    tex = Texture.from_array(np.zeros((4, 4, 3), np.uint8))
    tex.update(np.full((2, 8, 3), (255, 0, 0), np.uint8))
    assert tex.version == 1 and (tex.width, tex.height) == (8, 2)
    assert tex.rgba[0, 0].tolist() == [0, 0, 255, 255]


def test_procedural_textures():
    board = checkerboard(squares=2, size=4)
    assert board.rgba[0, 0].tolist() != board.rgba[0, 2].tolist()
    grad = gradient((255, 255, 255), (0, 0, 0))
    assert grad.rgba[0, 0, 0] == 255 and grad.rgba[-1, 0, 0] == 0
    assert grad.aspect == pytest.approx(16 / 9, rel=0.01)


def test_video_texture_follows_simulation_time(tmp_path):
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 16))
    if not writer.isOpened():
        pytest.skip("no video encoder available")
    for i in range(10):
        writer.write(np.full((16, 32, 3), i * 25, np.uint8))
    writer.release()

    video = as_texture(str(path))
    assert isinstance(video, VideoTexture)
    assert video.fps == pytest.approx(10)
    brightness = [int(video.at_time(t).rgba[8, 16, 0]) for t in (0.0, 0.5, 0.9, 1.25)]
    assert brightness[0] < brightness[1] < brightness[2]  # plays forward
    assert abs(brightness[3] - 50) < 15  # loops: t=1.25 s -> frame 2
