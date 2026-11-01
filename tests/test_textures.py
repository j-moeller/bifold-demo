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


def test_srgb_and_linear_light():
    from bifold_demo.scene.textures import color_to_linear, to_linear, to_srgb

    levels = np.arange(256, dtype=np.uint8)
    assert np.array_equal(to_srgb(to_linear(levels)), levels)  # every level comes back exactly
    assert to_linear(levels).dtype == np.float32 and to_linear(255) == 1.0 and to_linear(0) == 0.0
    assert to_linear(128) == pytest.approx(0.2158605, abs=1e-6)  # the sRGB curve, not a plain 2.2 power
    assert to_srgb(0.5) == 188 and to_srgb([-1.0, 2.0]).tolist() == [0, 255]
    assert color_to_linear("#ffffff") == (1.0, 1.0, 1.0) and color_to_linear((128, 0, 0))[0] == pytest.approx(0.2159, abs=1e-4)


def test_downsampling_averages_light_and_keeps_cut_out_colours():
    from bifold_demo.scene.textures import downsample

    checker = np.zeros((4, 4, 4), np.uint8)
    checker[..., 3] = 255
    checker[::2, ::2, :3] = checker[1::2, 1::2, :3] = 255
    assert downsample(checker, (2, 2))[0, 0].tolist() == [188, 188, 188, 255]  # half the light, not level 128
    assert downsample(checker, (2, 2), srgb=False)[0, 0].tolist() == [128, 128, 128, 255]  # data: as it is
    hair = np.zeros((2, 2, 4), np.uint8)
    hair[0, 0] = (200, 120, 40, 255)  # one opaque texel; the transparent ones' colour (black) must not darken it
    assert downsample(hair, (1, 1))[0, 0].tolist() == [200, 120, 40, 64]
    assert np.array_equal(downsample(checker, (4, 4)), checker)


def test_mip_chains_go_down_to_one_pixel_in_linear_light():
    from bifold_demo.scene.textures import mip_chain

    rgba = np.zeros((8, 32, 4), np.uint8)
    rgba[..., 3] = 255
    rgba[:, ::2, :3] = 255  # white and black stripes
    levels = mip_chain(rgba)
    assert [lv.shape[:2] for lv in levels] == [(4, 16), (2, 8), (1, 4), (1, 2), (1, 1)]
    assert all(lv.dtype == np.uint8 for lv in levels)
    assert all((lv[..., :3] == 188).all() for lv in levels)  # the stripes' mean light at every level
    assert len(mip_chain(np.zeros((1, 1, 4), np.uint8))) == 0
    odd = mip_chain(np.full((5, 3, 4), 255, np.uint8))
    assert [lv.shape[:2] for lv in odd] == [(2, 1), (1, 1)] and (odd[-1] == 255).all()
