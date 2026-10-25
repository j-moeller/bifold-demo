import numpy as np
import pytest

from bifold_demo.scene import Box, Camera, Quad, Scene, Texture

pytestmark = pytest.mark.gl


def _render(scene, size=(160, 90)):
    from bifold_demo.scene.renderer import Renderer

    renderer = Renderer(*size, samples=0)
    try:
        scene.update(0.0, 0.0)
        return renderer.render(scene)
    finally:
        renderer.release()


def _front_scene(**quad_args):
    scene = Scene(background=(0, 255, 0))
    scene.camera = Camera(position=(0, 0, 2), target=(0, 0, 0), fov=60)
    scene.add(Quad(position=(0, 0, 0), lit=False, **quad_args), name="q")
    return scene


def test_background_colour_and_bgr_output(gl):
    image = _render(Scene(background=(255, 0, 0)))
    assert image.shape == (90, 160, 3) and image.dtype == np.uint8
    assert image[45, 80].tolist() == [0, 0, 255]  # red, in OpenCV's BGR order


def test_texture_is_upright_and_not_mirrored(gl, two_tone_png):
    image = _render(_front_scene(texture=two_tone_png, height=0.8))
    assert image[38, 80].tolist() == [0, 0, 255]  # top half red
    assert image[52, 80].tolist() == [255, 0, 0]  # bottom half blue
    assert image[2, 2].tolist() == [0, 255, 0]  # background around it

    rgb = np.zeros((8, 8, 3), np.uint8)
    rgb[:, :4] = (255, 255, 255)  # white on the left, black on the right
    image = _render(_front_scene(texture=Texture.from_array(rgb, order="rgb"), height=0.8))
    assert image[45, 70].tolist() == [255, 255, 255]
    assert image[45, 90].tolist() == [0, 0, 0]


def test_transparent_pixels_show_whats_behind(gl):
    rgba = np.zeros((4, 4, 4), np.uint8)
    rgba[..., 0] = 255  # red but fully transparent
    image = _render(_front_scene(texture=Texture(rgba), height=0.8))
    assert image[45, 80].tolist() == [0, 255, 0]


def test_depth_and_lighting(gl):
    scene = Scene(background=(0, 0, 0), ambient=0.2, light_direction=(0, 0, -1))
    scene.camera = Camera(position=(0, 0, 3), target=(0, 0, 0))
    scene.add(Box(size=1, texture="#ffffff", position=(0, 0, -1)), name="far")
    scene.add(Box(size=0.5, texture="#ff0000", position=(0, 0, 0.5)), name="near")
    image = _render(scene)
    b, g, r = image[45, 80].tolist()
    assert r > 200 and g < 30  # the nearer red box hides the white one, lit head-on

    scene.light_direction = (0, 0, 1)  # light from behind: only ambient on the visible face
    image = _render(scene)
    assert 30 < image[45, 80, 2] < 80


def test_image_background_and_texture_updates(gl, two_tone_png):
    from bifold_demo.scene.renderer import Renderer

    scene = Scene(background=two_tone_png)
    screen = Texture.from_array(np.zeros((2, 2, 3), np.uint8))
    scene.camera = Camera(position=(0, 0, 2), target=(0, 0, 0))
    scene.add(Quad(texture=screen, size=0.3, lit=False), name="screen")
    renderer = Renderer(160, 90, samples=4)
    try:
        scene.update(0, 0)
        image = renderer.render(scene)
        assert image[5, 5].tolist() == [0, 0, 255] and image[85, 5].tolist() == [255, 0, 0]
        assert image[45, 80].tolist() == [0, 0, 0]
        screen.update(np.full((2, 2, 3), 255, np.uint8))
        image = renderer.render(scene)
        assert image[45, 80].tolist() == [255, 255, 255]
    finally:
        renderer.release()
