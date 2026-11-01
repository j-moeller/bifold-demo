"""The virtual sensor's camera model (bifold_demo.sensor.camera_model): each fault, the presets, their seeds, and
how windows get them (the presets' own sensors, the person's coverage for motion blur)."""

import math
from dataclasses import replace

import cv2
import numpy as np
import pytest
from test_sensor import VIEW, sample_scene

from bifold_demo.characters.character import Character
from bifold_demo.characters.model import CharacterModel, CharacterState, Skeleton, SkinnedPrimitive
from bifold_demo.scene import Box, Camera, Scene
from bifold_demo.scene.geometry import box_mesh
from bifold_demo.sensor import PRESETS, CameraModel, Sensor, WindowRequest, camera_preset, render_windows
from bifold_demo.sensor.camera_model import linear_to_srgb, motion_kernel, srgb_to_linear
from bifold_demo.sensor.service import Motion


def _flat(value, size=(64, 64)):
    return np.full((*size, 3), value, np.uint8)


def _bars(size=64):
    image = np.full((size, size, 3), 40, np.uint8)
    image[:, ::4] = 220  # thin bright columns
    return image


def test_the_ideal_camera_changes_nothing_and_presets_are_found_by_name():
    image = _bars()
    assert PRESETS["ideal"].ideal and PRESETS["ideal"].apply(image, 3, (500, 0), None, 7) is image
    assert [name for name in PRESETS] == ["ideal", "good 8K", "good 4K", "budget 1080p"]
    assert camera_preset("good_4k") is PRESETS["good 4K"] and camera_preset("Budget-1080p").lines == 1080
    assert camera_preset(None) is PRESETS["ideal"]
    custom = CameraModel("mine", jpeg=50)
    assert camera_preset(custom) is custom and not custom.ideal
    with pytest.raises(ValueError, match="unknown camera preset"):
        camera_preset("hd")
    with pytest.raises(ValueError, match="demosaicing"):
        CameraModel(demosaic="magic")
    faster = PRESETS["good 4K"].with_exposure(1 / 1000)
    assert faster.exposure == 1 / 1000 and faster.electrons == pytest.approx(PRESETS["good 4K"].electrons / 4)


def test_presets_reproduce_with_their_seed_and_differ_without_it():
    image = cv2.resize(_bars(), (96, 96))
    for preset in PRESETS.values():
        a = preset.apply(image, 1, (300, 40), None, seed=11)
        assert a.shape == image.shape and a.dtype == np.uint8
        assert np.array_equal(a, preset.apply(image, 1, (300, 40), None, seed=11))
        if preset.electrons:
            assert not np.array_equal(a, preset.apply(image, 1, (300, 40), None, seed=12))
    model = PRESETS["good 4K"]
    seed = model.window_seed(5, "track 3", (0, 0, 64, 64))
    assert seed == model.window_seed(5, "track 3", (0, 0, 64, 64))
    assert len({seed, model.window_seed(6, "track 3", (0, 0, 64, 64)), model.window_seed(5, "track 4", (0, 0, 64, 64)),
                model.window_seed(5, "track 3", (8, 0, 72, 64))}) == 4


def test_srgb_curves_invert_each_other():
    values = np.linspace(0, 1, 101)
    assert np.allclose(linear_to_srgb(srgb_to_linear(values)), values, atol=1e-5)
    assert srgb_to_linear(np.array([0.5]))[0] == pytest.approx(0.214, abs=1e-3)
    only_exposure = CameraModel("exposure only", exposure=1 / 100)  # no motion given: the round trip is exact
    image = np.random.default_rng(1).integers(0, 256, (32, 48, 3), dtype=np.uint8)
    assert np.abs(only_exposure.apply(image).astype(int) - image).max() <= 1


def test_lens_blur_softens_less_on_binned_pixels():
    image = _bars()
    lens = CameraModel("lens", lens_blur=1.2)

    def contrast(out):
        return float(out[:, 8:56, 1].astype(float).std())

    sharp, blurred, binned = contrast(image), contrast(lens.apply(image)), contrast(lens.apply(image, level=3))
    assert blurred < 0.5 * sharp and blurred < binned < sharp  # sigma 1.2 sensor pixels is 0.4 binned pixels


def test_noise_follows_the_poisson_gaussian_model_and_binning():
    grey = _flat(128, (256, 256))
    level = srgb_to_linear(np.array([128 / 255]))[0]
    slope = (linear_to_srgb(np.array([level + 1e-4]))[0] - linear_to_srgb(np.array([level - 1e-4]))[0]) / 2e-4

    def predicted(k, electrons, read_noise):  # the sRGB curve's slope turns the linear spread into 8-bit levels
        full = electrons * k * k  # a binned pixel's light, and each of its k x k pixels read
        return math.sqrt(level * full + k * k * read_noise ** 2) / full * slope * 255

    # mostly shot noise, then mostly read noise (which binning adds up too: half the spread at level 2, not a quarter)
    for electrons, read_noise in ((2000, 3.0), (4000, 60.0)):
        camera = CameraModel("noisy", electrons=electrons, read_noise=read_noise)
        for k in (1, 2, 3):
            out = camera.apply(grey, level=k, seed=3).astype(float)
            assert out.std() == pytest.approx(predicted(k, electrons, read_noise), rel=0.1), (read_noise, k)
            assert out.mean() == pytest.approx(128, abs=0.5)
    camera = CameraModel("noisy", electrons=2000, read_noise=3.0)
    dark = camera.apply(_flat(10, (256, 256)), seed=3).astype(float).std()
    assert dark > camera.apply(grey, seed=3).astype(float).std()  # the sRGB curve lifts the dark noise


def test_sharpening_is_an_unsharp_mask_of_its_strength():
    """out = x + s (x - G(x)), G a Gaussian of one pixel, on the sRGB values (nothing else changes them here)."""
    image = _bars()
    x = image.astype(np.float32) / 255
    detail = x - cv2.GaussianBlur(x, (0, 0), 1.0, borderType=cv2.BORDER_REFLECT)
    for strength in (0.3, 0.8):
        expected = np.clip(np.rint((x + strength * detail) * 255), 0, 255)
        out = CameraModel("sharp", sharpen=strength).apply(image)
        assert np.abs(out.astype(int) - expected).max() <= 1, strength


def test_motion_blur_smears_along_the_motion_and_only_the_masked_person():
    kernel = motion_kernel((6.0, 0.0))
    assert kernel.sum() == pytest.approx(1.0) and kernel.shape[0] < kernel.shape[1]  # cut to a horizontal line
    _rows, cols = np.indices(kernel.shape)
    assert (kernel * cols).sum() == pytest.approx((kernel.shape[1] - 1) / 2) and motion_kernel((0.3, 0.2)) is None
    diagonal = motion_kernel((4.0, -3.0))
    rows, cols = np.indices(diagonal.shape)  # centred on the middle of the move
    assert (diagonal * cols).sum() == pytest.approx(2.0) and (diagonal * rows).sum() == pytest.approx(1.0)
    assert (kernel > 0).sum(axis=1).max() >= 6
    moving = CameraModel("shutter", exposure=1 / 50)
    dot = np.full((40, 60, 3), 30, np.uint8)
    dot[18:22, 28:32] = 230
    out = moving.apply(dot, velocity=(400.0, 0.0))  # 8 pixels across during the exposure
    lit = out[..., 1] > 60
    assert 9 <= lit[20].sum() <= 14 and not lit[:17].any() and not lit[23:].any()  # wider, not taller
    image = np.full((40, 60, 3), 30, np.uint8)
    image[:, 28:32] = 230  # a bright upright bar
    mask = np.zeros(image.shape[:2], np.float32)
    mask[:20] = 1.0  # only the top half is the moving person
    half = moving.apply(image, velocity=(400.0, 0.0), mask=mask)
    assert (half[5, :, 1] > 60).sum() >= 9 and (half[35, :, 1] > 60).sum() == 4  # the bottom stays sharp


def test_bayer_mosaic_keeps_colours_and_tone_adds_contrast():
    colour = np.zeros((32, 32, 3), np.uint8)
    colour[:] = (40, 90, 200)  # BGR: a red-orange
    for method in ("bilinear", "ea", "vng"):
        out = CameraModel("bayer", bayer=True, demosaic=method).apply(colour)
        assert np.abs(out[8:24, 8:24].astype(int) - colour[8:24, 8:24]).max() <= 2, method
    ramp = np.repeat(np.linspace(0, 255, 64).astype(np.uint8)[None, :, None], 3, axis=2)
    toned = CameraModel("tone", tone=0.5).apply(ramp).astype(int)[0, :, 0]
    assert toned[8] < ramp[0, 8, 0] and toned[56] > ramp[0, 56, 0] and abs(toned[32] - int(ramp[0, 32, 0])) <= 3
    sharpened = CameraModel("sharp", sharpen=0.8).apply(_bars())
    assert sharpened[:, 8:56, 1].astype(float).std() > _bars()[:, 8:56, 1].astype(float).std()
    jpeg = CameraModel("jpeg", jpeg=90).apply(cv2.resize(_bars(), (96, 96)))
    assert not np.array_equal(jpeg, cv2.resize(_bars(), (96, 96)))


# -- through the windows ----------------------------------------------------------------------------------------------


def _walker_scene():
    """A person with eye bones walking across in front of a wall: x = 0.6 t."""
    rest = np.repeat(np.eye(4)[None], 3, axis=0)
    rest[1, :3, 3], rest[2, :3, 3] = (-0.032, 1.6, 0.1), (0.032, 1.6, 0.1)
    skeleton = Skeleton(["Root", "Bip01 REye", "Bip01 LEye"], [-1, 0, 0], rest)
    body = box_mesh(0.3, 1.75, 0.2)
    joints, weights = np.zeros((len(body.positions), 4), np.int64), np.zeros((len(body.positions), 4), np.float32)
    weights[:, 0] = 1.0
    stripes = np.zeros((8, 8, 4), np.uint8)
    stripes[...] = (230, 60, 40, 255)
    stripes[:, ::2] = (30, 30, 200, 255)
    part = SkinnedPrimitive("body", body.positions + np.float32([0, 0.875, 0]), body.normals, body.uvs, body.indices,
                            joints, weights, stripes)
    model = CharacterModel("walker", skeleton, [part], np.linalg.inv(skeleton.rest_global))
    scene = Scene(background=(90, 90, 90), ambient=1.0)
    scene.camera = Camera(position=(0, 1.6, 6), target=(0, 1.6, -6), fov=55)
    scene.add(Box(size=(6, 3, 0.1), texture="#c8d2a0", position=(0, 1.5, -1.0), lit=False), name="wall")
    walker = scene.add(Character(model, lambda t: CharacterState((0.6 * t, 0.0, 1.5), 0.0, [])), name="walker")
    return scene, walker


def test_the_window_mask_is_the_person_in_front(gl):
    from bifold_demo.scene.renderer import Renderer

    scene = sample_scene()
    renderer = Renderer(*VIEW)
    try:
        renderer.render(scene)
        sensor = Sensor.scaled(4, VIEW)
        window = sensor.window((100, 40, 160, 140), 1)  # the near person's upper body
        image = renderer.render_window(scene, window.ndc, window.size)
        mask = renderer.window_mask(scene["near"])
        assert mask.shape == image.shape[:2] and mask.dtype == np.float32
        red = (image[..., 2] > 120) & (image[..., 1] < 80)  # the body's red
        assert red.sum() > 200 and mask[red].mean() > 0.95
        background = np.all(image == (90, 60, 40), axis=2)  # the scene's background colour (BGR)
        assert background.sum() > 1000 and mask[background].max() == 0.0
        steps = np.unique(mask) * 4  # 4 MSAA samples a pixel (resolved to 8 bits)
        assert np.abs(steps - np.round(steps)).max() < 0.02 and 0 < len(steps) <= 5
        assert np.array_equal(renderer.render_window(scene, window.ndc, window.size), image)  # drawn as before
        hidden = renderer.window_mask(scene["far"])  # the far person is not in this window
        assert hidden.max() == 0.0
        scene.add(Box(size=(0.4, 0.4, 0.05), texture="#ffffff", position=(-0.55, 1.0, 3.2), lit=False), name="cover")
        scene.update(0.0, 0.0)
        renderer.render(scene)
        renderer.render_window(scene, window.ndc, window.size)
        covered = renderer.window_mask(scene["near"])
        assert covered.sum() < mask.sum() - 100  # what is in front is not the person
    finally:
        renderer.release()


def test_windows_blur_the_moving_person_over_a_still_background(gl):
    from bifold_demo.scene.renderer import Renderer

    scene, walker = _walker_scene()
    motion = Motion()
    renderer = Renderer(*VIEW)
    try:
        for t in (0.9, 1.0):
            scene.update(t, 0.1)
            motion.observe(scene)
        renderer.render(scene)
        sensor = Sensor.scaled(4, VIEW)
        shutter = CameraModel("shutter", exposure=1 / 30)
        request = WindowRequest("w", head=(0.6, 1.2, 1.6), head_size=1.6, face_px=None)
        (still,) = render_windows(scene, renderer, [request], sensor, motion=motion)
        (moving,) = render_windows(scene, renderer, [request], sensor, motion=motion, preset=shutter)
        assert moving.preset == "shutter" and moving.truth.name == "walker"
        speed = moving.truth.velocity[0]
        assert speed > 50 and abs(moving.truth.velocity[1]) < 0.05 * speed  # walking to the camera's right
        changed = np.abs(moving.image.astype(int) - still.image.astype(int)).max(axis=2) > 8
        mask = renderer.window_mask(walker)
        near = cv2.dilate((mask > 0).astype(np.uint8), np.ones((3, int(speed / 30) + 3), np.uint8)) > 0
        assert changed.sum() > 50 and not (changed & ~near).any()  # only on and around the person
    finally:
        renderer.release()


def test_the_native_level_keeps_to_the_pixel_budget():
    camera = PRESETS["good 4K"]
    assert camera.native and camera.native_pixels == 1_000_000
    assert camera.native_level((100, 80), 4) == 1  # 400 x 320 of the camera's own pixels
    assert camera.native_level((300, 300), 4) == 2  # 1200 x 1200 is over the budget: 600 x 600
    assert camera.native_level((400, 400), 3) == 3  # no level between
    assert camera.native_level((100, 80), 4, max_side=256) == 2  # what the renderer can draw
    assert camera.native_level((100, 80), 1) == 1
    assert PRESETS["ideal"].native_level((100, 80), 4) == 4  # nothing to do: no pixels drawn for it
    assert replace(camera, native=False).native_level((100, 80), 4) == 4  # a binned read-out


def _psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 10 * math.log10(255.0 ** 2 / max(mse, 1e-12))


def test_binned_windows_meet_the_camera_at_its_own_pixels(gl):
    """A window binned by k is drawn at the camera's own pixels, goes through the faults there and is area-averaged
    down (a camera's full picture, its faces scaled down where they are used); native=False: a binned read-out."""
    from bifold_demo.scene.renderer import Renderer

    scene = sample_scene()
    renderer = Renderer(*VIEW)
    try:
        renderer.render(scene)
        sensor = Sensor.scaled(4, VIEW)
        camera = CameraModel("mine", lens_blur=0.7, electrons=2000, read_noise=2.0, bayer=True, sharpen=0.5, jpeg=85)
        request = WindowRequest("p", region=(100, 40, 160, 100), eye_distance=20, face_px=20)
        (native,) = render_windows(scene, renderer, [request], sensor, frame=2, preset=camera)
        window = native.window
        assert window.level == 4 and camera.native_level(window.size, 4) == 1
        fine = window.at_level(1)
        assert fine.sensor_rect == window.sensor_rect and fine.size == (window.width * 4, window.height * 4)
        assert window.at_level(4) is window and window.at_level(2).size == (window.width * 2, window.height * 2)
        with pytest.raises(ValueError, match="does not divide"):
            window.at_level(3)
        seed = camera.window_seed(2, "p", window.sensor_rect)
        drawn = renderer.render_window(scene, fine.ndc, fine.size)
        own = cv2.resize(camera.apply(drawn, 1, seed=seed), window.size, interpolation=cv2.INTER_AREA)
        assert np.array_equal(native.image, own)
        read_out = replace(camera, native=False)
        (binned,) = render_windows(scene, renderer, [request], sensor, frame=2, preset=read_out)
        clean = renderer.render_window(scene, window.ndc, window.size)
        assert np.array_equal(binned.image, read_out.apply(clean, 4, seed=seed))
        # the mosaic's, the sharpening's and the blocks' marks at the camera's pitch average away in the scaling down
        assert _psnr(native.image, clean) > _psnr(binned.image, clean) + 3.0  # 38 dB against 30
    finally:
        renderer.release()


def test_a_presets_own_sensor_places_its_windows(gl):
    from bifold_demo.scene.renderer import Renderer

    scene = sample_scene()
    renderer = Renderer(*VIEW)
    try:
        renderer.render(scene)
        region = (100, 40, 160, 100)
        own = Sensor.scaled(2, VIEW)
        requests = [WindowRequest("a", region=region, face_px=None),
                    WindowRequest("b", region=region, face_px=None, preset="budget 1080p"),
                    WindowRequest("c", region=region, face_px=None, preset="budget 1080p", sensor=own)]
        a, b, c = render_windows(scene, renderer, requests, Sensor.scaled(6, VIEW), frame=4, preset="good 4K")
        assert (a.preset, b.preset, c.preset) == ("good 4K", "budget 1080p", "budget 1080p")
        assert a.window.sensor.size == (3840, 2160)  # the presets' own sensors, whatever the view's size
        assert b.window.sensor.size == (1920, 1080) and c.window.sensor is own
        again = render_windows(scene, renderer, requests[:1], Sensor.scaled(6, VIEW), frame=4, preset="good 4K")
        assert np.array_equal(again[0].image, a.image)  # the same frame, the same noise
        later = render_windows(scene, renderer, requests[:1], Sensor.scaled(6, VIEW), frame=5, preset="good 4K")
        assert not np.array_equal(later[0].image, a.image)
        assert a.camera_ms > 0
    finally:
        renderer.release()
