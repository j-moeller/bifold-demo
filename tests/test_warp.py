import cv2
import numpy as np
import pytest

from bifold_morph import warp

LANDMARKS = np.array([[270.0, 200.0], [330.0, 200.0], [300.0, 235.0], [275.0, 270.0], [325.0, 270.0]])


def test_similarity_recovers_a_known_transform():
    rng = np.random.default_rng(0)
    src = rng.uniform(0, 100, (5, 2))
    theta, s, t = np.radians(25), 1.7, np.array([12.0, -4.0])
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    dst = s * src @ rot.T + t
    m = warp.similarity(src, dst)
    assert np.allclose(m[:, :2], s * rot) and np.allclose(m[:, 2], t)


def test_matrix_puts_the_landmarks_on_the_template():
    m = warp.matrix(LANDMARKS, "ffhq_512", 256)
    pts = np.c_[LANDMARKS, np.ones(5)] @ m.T
    # a similarity cannot fit five points exactly, but these are close to the template's shape
    assert np.abs(pts - warp.TEMPLATES["ffhq_512"] * 256).max() < 12


def test_warp_and_paste_round_trip():
    rng = np.random.default_rng(1)
    frame = cv2.GaussianBlur(rng.integers(0, 256, (480, 640, 3)).astype(np.uint8), (0, 0), 3)
    m = warp.matrix(LANDMARKS, "ffhq_512", 256)
    crop = warp.warp(frame, m, 256)
    out = warp.paste(frame.copy(), crop.astype(np.float32), np.ones((256, 256), np.float32), m)
    cx, cy = LANDMARKS.mean(0).astype(int)
    region = (slice(cy - 40, cy + 40), slice(cx - 40, cx + 40))
    assert np.abs(out[region].astype(int) - frame[region]).mean() < 3  # back where it came from
    assert np.array_equal(out[:20, :20], frame[:20, :20])  # far outside the crop: untouched


def test_paste_outside_the_frame_does_nothing():
    frame = np.zeros((100, 100, 3), np.uint8)
    m = warp.matrix(LANDMARKS + 5000, "ffhq_512", 64)
    assert warp.paste(frame, np.full((64, 64, 3), 255, np.float32), np.ones((64, 64), np.float32), m) is frame
    assert frame.max() == 0


def test_box_mask():
    mask = warp.box_mask(100, 0.3)
    assert mask[50, 50] == pytest.approx(1.0, abs=1e-3) and mask[0, 50] < 0.05 and mask[50, 99] < 0.05
