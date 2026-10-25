import numpy as np
import pytest

from bifold_demo.scene import Body, Box, Keyframes, Scene, Sphere, linear, orbit, oscillate, resolve, spin


def test_keyframes_interpolate_and_hold():
    k = Keyframes({0: (0, 0, 0), 2: (2, 4, 0)})
    np.testing.assert_allclose(k(1), (1, 2, 0))
    np.testing.assert_allclose(k(-1), (0, 0, 0))
    np.testing.assert_allclose(k(10), (2, 4, 0))


def test_keyframes_loop_and_pingpong():
    loop = Keyframes({0: 0.0, 2: 2.0}, loop=True)
    assert loop(2.5) == pytest.approx(0.5)
    ping = Keyframes({0: 0.0, 2: 2.0}, pingpong=True)
    assert ping(2.5) == pytest.approx(1.5)
    assert ping(4.5) == pytest.approx(0.5)


def test_keyframes_switch_non_numeric_values():
    k = Keyframes({0: "a.jpg", 1: "b.jpg", 2: "a.jpg"}, loop=True)
    assert [k(t) for t in (0.0, 0.5, 1.0, 1.9, 2.1)] == ["a.jpg", "a.jpg", "b.jpg", "b.jpg", "a.jpg"]


def test_keyframes_smooth_easing_keeps_endpoints():
    k = Keyframes({0: 0.0, 1: 10.0}, easing="smooth")
    assert k(0) == 0 and k(1) == 10 and k(0.5) == pytest.approx(5)
    assert k(0.1) < 1.0  # starts slower than linear


def test_motion_helpers():
    np.testing.assert_allclose(oscillate((1, 0, 0), (0, 2, 0), period=4)(1), (1, 2, 0))
    np.testing.assert_allclose(spin(90)(2), (0, 180, 0))
    np.testing.assert_allclose(orbit((0, 1, 0), radius=2, period=4)(1), (2, 1, 0), atol=1e-12)
    np.testing.assert_allclose(linear((0, 0, 0), (1, 2, 3))(2), (2, 4, 6))
    assert resolve(5, 1.0) == 5 and resolve(lambda t: t * 2, 1.5) == 3


def _run(scene, seconds, fps=60):
    for i in range(int(seconds * fps) + 1):
        scene.update(i / fps, 0 if i == 0 else 1 / fps)


def test_ball_bounces_and_settles_on_floor():
    scene = Scene()
    ball = scene.add(Sphere(radius=0.5, position=(0, 3, 0), body=Body(gravity=-9.81, bounce=0.6, floor=0.0)))
    heights = []
    for i in range(60 * 8):
        scene.update(i / 60, 1 / 60)
        heights.append(ball.current_position[1])
    assert min(heights) >= 0.5 - 1e-6  # never sinks into the floor
    assert any(h1 > h0 for h0, h1 in zip(heights[60:], heights[61:]))  # it bounced back up at least once
    assert heights[-1] == pytest.approx(0.5, abs=1e-3)  # and came to rest
    assert abs(ball.body.velocity[1]) < 1e-6


def test_bounds_keep_object_inside():
    scene = Scene()
    box = scene.add(Box(size=1, body=Body(velocity=(5, 3, -4), angular_velocity=(30, 60, 0), bounce=1.0,
                                          bounds=((-2, -2, -2), (2, 2, 2)))))
    for i in range(1, 600):
        scene.update(i / 60, 1 / 60)
        low, high = box.extents()
        assert np.all(box.current_position + low >= -2 - 1e-6)
        assert np.all(box.current_position + high <= 2 + 1e-6)


def test_reset_restores_initial_state():
    scene = Scene()
    box = scene.add(Box(position=(0, 1, 0), body=Body(velocity=(1, 0, 0), angular_velocity=(0, 90, 0))))
    _run(scene, 1.0)
    assert box.current_position[0] == pytest.approx(1.0, abs=0.05)
    scene.reset()
    scene.update(0.0, 0.0)
    np.testing.assert_allclose(box.current_position, (0, 1, 0))
    np.testing.assert_allclose(box.body.velocity, (1, 0, 0))


def test_large_time_steps_are_subdivided():
    scene = Scene()
    ball = scene.add(Sphere(radius=0.5, position=(0, 1, 0), body=Body(velocity=(0, -50, 0), floor=0.0)))
    scene.update(0.0, 0.0)
    scene.update(0.25, 0.25)  # would tunnel through the floor in one step
    assert ball.current_position[1] >= 0.5 - 1e-9
