import numpy as np
import pytest

from bifold_morph import align

# an upright face: eyes 60 px apart at y=200, the mouth 70 px below them
LANDMARKS = np.array([[270.0, 200.0], [330.0, 200.0], [300.0, 235.0], [275.0, 270.0], [325.0, 270.0]])


def test_quad_of_an_upright_face():
    quad = align.ffhq_quad(LANDMARKS)
    # side: max(2 x eye distance, 1.8 x eye-to-mouth) x 2 = max(120, 126) x 2
    assert align.quad_side(quad) == pytest.approx(252.0)
    centre = quad.mean(0)
    assert centre == pytest.approx([300.0, 207.0])  # eye midpoint + 0.1 x eye-to-mouth
    tl, bl, br, tr = quad
    assert tl[0] < tr[0] and tl[1] < bl[1]  # top-left, bottom-left, bottom-right, top-right
    assert align.rotation_deg(quad) == pytest.approx(0.0)


def test_quad_turns_with_the_face():
    theta = np.radians(30)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    turned = (LANDMARKS - 300) @ rot.T + 300
    assert align.rotation_deg(align.ffhq_quad(turned)) == pytest.approx(30.0)
    assert align.quad_side(align.ffhq_quad(turned)) == pytest.approx(252.0)


def test_crop_puts_the_landmarks_where_ffhq_has_them():
    size = 256
    quad = align.ffhq_quad(LANDMARKS)
    m = align.quad_matrix(quad, size)
    pts = np.c_[LANDMARKS, np.ones(5)] @ m.T
    # the eyes level and centred left-right, the eyes above the middle, the mouth well below it
    assert pts[0, 1] == pytest.approx(pts[1, 1])
    assert (pts[0, 0] + pts[1, 0]) / 2 == pytest.approx((size - 1) / 2)
    assert pts[0, 1] < size / 2 < pts[3, 1]


@pytest.mark.parametrize("scale", [1.0, 4.0])
def test_crop_samples_the_right_pixels(scale):
    """A dot drawn at the left eye shows up at the left eye's place in the crop, also for a big (shrunk) face."""
    h, w = round(480 * scale), round(640 * scale)
    image = np.zeros((h, w, 3), np.uint8)
    lm = LANDMARKS * scale
    x, y = np.round(lm[0]).astype(int)
    r = round(3 * scale)
    image[y - r:y + r + 1, x - r:x + r + 1] = 255
    quad = align.ffhq_quad(lm)
    out = align.crop(image, quad, 256)
    ys, xs = np.nonzero(out[..., 0] > 60)
    expect = np.r_[lm[0], 1] @ align.quad_matrix(quad, 256).T
    assert out.shape == (256, 256, 3)
    assert xs.mean() == pytest.approx(expect[0], abs=1.0) and ys.mean() == pytest.approx(expect[1], abs=1.0)


def test_paste_a_zero_change_keeps_the_frame():
    frame = np.random.default_rng(0).integers(0, 256, (480, 640, 3), dtype=np.uint8)
    out = align.paste(frame.copy(), np.zeros((64, 64, 3), np.float32), align.border_mask(64),
                      align.ffhq_quad(LANDMARKS), replace=False)
    assert np.array_equal(out, frame)


def test_paste_adds_inside_the_square_only():
    frame = np.full((480, 640, 3), 100, np.uint8)
    quad = align.ffhq_quad(LANDMARKS)
    out = align.paste(frame.copy(), np.full((64, 64, 3), 50, np.float32), np.ones((64, 64), np.float32), quad,
                      replace=False)
    cx, cy = np.round(quad.mean(0)).astype(int)
    assert out[cy, cx].tolist() == [150, 150, 150]
    assert out[10, 10].tolist() == [100, 100, 100]  # far outside the square
    assert out.max() <= 150


def test_paste_replace_blends_with_the_mask():
    frame = np.full((480, 640, 3), 200, np.uint8)
    quad = align.ffhq_quad(LANDMARKS)
    layer = np.zeros((64, 64, 3), np.float32)
    out = align.paste(frame.copy(), layer, np.full((64, 64), 0.25, np.float32), quad, replace=True)
    cx, cy = np.round(quad.mean(0)).astype(int)
    assert out[cy, cx].tolist() == [150, 150, 150]


def test_paste_near_the_edge():
    frame = np.full((300, 300, 3), 10, np.uint8)
    quad = align.ffhq_quad(LANDMARKS - [200, 100])  # partly outside the frame
    out = align.paste(frame, np.full((32, 32, 3), 5, np.float32), np.ones((32, 32), np.float32), quad, False)
    assert out.shape == (300, 300, 3) and out.max() == 15
    assert align.paste(frame, np.ones((32, 32, 3), np.float32), np.ones((32, 32), np.float32),
                       quad + 5000, False) is frame  # entirely outside: nothing to do


def test_masks():
    b = align.border_mask(100)
    assert b[50, 50] == 1.0 and b[0, 50] == 0.0 and b[50, 99] == 0.0
    f = align.face_mask(100)
    assert f[54, 50] == 1.0 and f[0, 0] == 0.0 and f[99, 50] < 0.5
