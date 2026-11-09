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


def test_fog_fades_distant_surfaces(gl):
    from bifold_demo.scene import Fog

    scene = Scene(background=(0, 0, 0), fog=Fog(start=1.0, end=5.0, color=(0, 0, 255)))
    scene.camera = Camera(position=(0, 0, 0), target=(0, 0, -1))
    scene.add(Quad(size=1.0, texture="#ff0000", position=(0, 0, -3), lit=False), name="mid")
    b, g, r = _render(scene)[45, 80].tolist()
    assert 180 < r < 195 and 180 < b < 195  # halfway: half the light of the red and of the blue fog (level 188)
    scene.fog = None
    assert _render(scene)[45, 80].tolist() == [0, 0, 255]


def test_every_level_comes_back_through_srgb_textures_and_the_srgb_picture(gl):
    """Colours are decoded from sRGB when sampled and encoded again when drawn: unlit, nothing changes."""
    scene = Scene(background=(17, 34, 51))
    scene.camera = Camera(position=(0, 0, 2.0), target=(0, 0, 0), fov=60)
    cell = 0.07
    for level in range(256):
        row, col = divmod(level, 16)
        scene.add(Quad(position=((col - 7.5) * cell, (7.5 - row) * cell, 0), size=cell * 0.9, lit=False,
                       texture=(level, level, 255 - level)), name=f"q{level}")
    image = _render(scene, size=(320, 320))
    assert image[2, 2].tolist() == [51, 34, 17]  # the background colour too (BGR)
    from bifold_demo.scene.renderer import Renderer

    renderer = Renderer(320, 320, samples=0)
    try:
        scene.update(0.0, 0.0)
        renderer.render(scene)
        view_proj = scene.camera.projection_matrix(1.0) @ scene.camera.view_matrix()
    finally:
        renderer.release()
    for level in range(256):
        row, col = divmod(level, 16)
        clip = view_proj @ np.array([(col - 7.5) * cell, (7.5 - row) * cell, 0.0, 1.0])
        x, y = ((clip[:2] / clip[3]) * [0.5, -0.5] + 0.5) * 320
        assert image[int(y), int(x)].tolist() == [255 - level, level, level], level


def test_mipmaps_average_light():
    """A fine black-and-white checker seen small is the grey of half the light (188), not the level of half the
    value (128), and so are the mipmaps prepared off the render thread."""
    from bifold_demo.scene.textures import prepare_upload

    rgb = np.zeros((256, 256, 3), np.uint8)
    rgb[(np.add.outer(np.arange(256), np.arange(256)) % 2) == 0] = 255
    prepared = prepare_upload(Texture.from_array(rgb, order="rgb"))
    assert len(prepared._upload[2]) == 8  # levels 128 ... 1
    assert set(np.frombuffer(prepared._upload[2][-1], np.uint8).tolist()) == {188, 255}


@pytest.mark.parametrize("prepared", [False, True])
def test_small_textures_show_the_light_of_their_texels(gl, prepared):
    from bifold_demo.scene.textures import prepare_upload

    rgb = np.zeros((256, 256, 3), np.uint8)
    rgb[(np.add.outer(np.arange(256), np.arange(256)) % 2) == 0] = 255
    texture = Texture.from_array(rgb, order="rgb")
    if prepared:
        prepare_upload(texture)
    scene = Scene(background=(0, 0, 0))
    scene.camera = Camera(position=(0, 0, 2), target=(0, 0, 0), fov=60)
    scene.add(Quad(texture=texture, size=0.2, lit=False), name="far")  # 256 texels on about 14 pixels
    value = _render(scene)[45, 80]
    assert (np.abs(value.astype(int) - 188) <= 3).all(), value


def test_cut_outs_seen_small_keep_their_colour(gl):
    """A cut-out (three opaque white texels and a transparent black one in every 2x2 block, like the gaps between
    hair strands) seen small is drawn white where it shows: its mipmaps weight the colour by coverage, so the
    transparent texels lend it no black (averaged without the weights it would come out at about 225)."""
    rgba = np.zeros((256, 256, 4), np.uint8)
    rgba[...] = (255, 255, 255, 255)
    rgba[1::2, 1::2] = (0, 0, 0, 0)
    scene = Scene(background=(0, 0, 0))
    scene.camera = Camera(position=(0, 0, 2), target=(0, 0, 0), fov=60)
    scene.add(Quad(texture=Texture(rgba), size=0.2, lit=False), name="far")  # 256 texels on about 14 pixels
    value = _render(scene)[45, 80]
    assert (value.astype(int) >= 252).all(), value


def test_fog_and_light_colours_come_out_as_given(gl):
    """Colours the sRGB curve changes (not the primaries' 0 and 255): a surface deep in the fog shows the fog's colour,
    a white surface lit squarely by a coloured light without ambient light shows the light's colour, each as given
    (each converted to linear light once, and encoded back once)."""
    from bifold_demo.scene import Fog

    fog = (128, 64, 200)
    scene = Scene(background=(0, 0, 0), fog=Fog(start=0.5, end=1.0, color=fog))
    scene.camera = Camera(position=(0, 0, 0), target=(0, 0, -1))
    scene.add(Quad(size=4.0, texture="#ffffff", position=(0, 0, -3), lit=False), name="deep in the fog")
    assert np.abs(_render(scene)[45, 80].astype(int) - fog[::-1]).max() <= 1
    light = (255, 200, 120)
    scene = Scene(background=(0, 0, 0), ambient=0.0, light_direction=(0, 0, -1), light_color=light)
    scene.camera = Camera(position=(0, 0, 2), target=(0, 0, 0))
    scene.add(Quad(texture="#ffffff", size=1.0), name="lit squarely")
    assert np.abs(_render(scene)[45, 80].astype(int) - light[::-1]).max() <= 1


def test_light_falls_off_in_linear_light_and_shadow_keeps_its_level(gl):
    """A surface turned 60 degrees from the light gets half the light (level 188, not 128); a surface in shadow is
    as dark as the ambient level says (0.2 of white: 51)."""
    scene = Scene(background=(0, 0, 0), ambient=0.0, light_direction=(0, -np.sin(np.radians(60)),
                                                                       -np.cos(np.radians(60))))
    scene.camera = Camera(position=(0, 0, 2), target=(0, 0, 0))
    scene.add(Quad(texture="#ffffff", size=1.0), name="turned")  # facing the camera, 60 degrees from the light
    level = int(_render(scene)[45, 80, 1])
    assert abs(level - 188) <= 3, level
    scene.ambient = 0.2
    scene.light_direction = (0, 0, 1)
    assert abs(int(_render(scene)[45, 80, 1]) - 51) <= 2


def test_the_stored_light_model_draws_as_before(gl):
    """``Renderer.linear = False``: lighting, fog and filtering on the stored sRGB values (the renderer before linear
    light); switching back uploads the textures again as sRGB textures."""
    from bifold_demo.scene import Fog
    from bifold_demo.scene.renderer import Renderer

    rgb = np.zeros((256, 256, 3), np.uint8)
    rgb[(np.add.outer(np.arange(256), np.arange(256)) % 2) == 0] = 255
    small = Texture.from_array(rgb, order="rgb")
    scene = Scene(background=(0, 0, 0), ambient=0.0, light_direction=(0, -np.sin(np.radians(60)),
                                                                       -np.cos(np.radians(60))))
    scene.camera = Camera(position=(0, 0, 2), target=(0, 0, 0), fov=60)
    scene.add(Quad(texture=small, size=0.2, lit=False), name="small")
    scene.add(Quad(texture="#ffffff", size=0.3, position=(0.6, 0, 0)), name="turned")
    foggy = Scene(background=(0, 0, 0), fog=Fog(start=1.0, end=5.0, color=(0, 0, 255)))
    foggy.camera = Camera(position=(0, 0, 0), target=(0, 0, -1))
    foggy.add(Quad(size=1.0, texture="#ff0000", position=(0, 0, -3), lit=False), name="mid")
    renderer = Renderer(160, 90, samples=0)
    try:
        for linear, half in ((False, 128), (True, 188), (False, 128)):
            renderer.linear = linear
            scene.update(0.0, 0.0)
            image = renderer.render(scene)
            assert abs(int(image[45, 80, 1]) - half) <= 4, (linear, image[45, 80])  # the small checker
            x = round(80 + 0.6 / (2 * np.tan(np.radians(30)) * 160 / 90) * 80)  # the turned quad's centre
            assert abs(int(image[45, x, 1]) - half) <= 4, (linear, image[45, x])  # half the light, 60 degrees off
            foggy.update(0.0, 0.0)
            b, _g, r = renderer.render(foggy)[45, 80].tolist()
            assert abs(r - half) <= 6 and abs(b - half) <= 6
    finally:
        renderer.release()
