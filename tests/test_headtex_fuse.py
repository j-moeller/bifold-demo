"""Head textures from photos, stages 3 and 4, on synthetic data (no MediaPipe, no OpenGL, no avatars needed).

The synthetic head of ``test_headtex`` (a sphere with a nose and the Rocketbox face bones) gets a known texture, and
"photos" of it are drawn by the numpy z-buffer of :mod:`~bifold_demo.characters.headtex.raster` from known cameras
under known light (diffuse, in linear light). The fake Face Landmarker of ``test_headtex`` projects the head's true
landmarks, and a fake segmenter calls the head face skin (or hair where a test says so).
"""

import threading
import time

import cv2
import numpy as np
import pytest
from test_headtex import CENTRE, FakeLandmarker, fake_render, synthetic_head_model, true_landmarks

from bifold_demo.characters.headtex import fuse, light, raster, raycast
from bifold_demo.characters.headtex import landmarks as stage0
from bifold_demo.characters.headtex import register as stage1
from bifold_demo.characters.headtex import texture as stages
from bifold_demo.characters.headtex.camera import PhotoCamera
from bifold_demo.characters.headtex.detect import SEGMENT_CLASSES

TEX = 256  # the synthetic head texture's size
SIZE = (480, 400)  # the photos'
SKIN = np.array([205, 150, 128], np.float64)  # sRGB of the skin


def checker_texture() -> np.ndarray:
    """The synthetic head's texture (RGB, row 0 at the top): skin with a soft checker (cells of 16 texels, 12 %
    lighter or darker) and a gentle gradient from the forehead down, so misplaced or blurred texels show."""
    yy, xx = np.mgrid[0:TEX, 0:TEX]
    checker = np.where(((xx // 16) + (yy // 16)) % 2 == 0, 1.12, 0.88)
    gradient = 1.0 + 0.1 * np.cos(np.pi * yy / TEX)
    rgb = SKIN[None, None] * (checker * gradient)[..., None]
    return np.clip(rgb, 0, 255).astype(np.uint8)


def textured_head(texture=None):
    model = synthetic_head_model()
    model.primitives[0].texture = checker_texture() if texture is None else texture
    return model


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    """No downloads, renders or the user's cache (as in test_headtex)."""
    from bifold_demo.characters.headtex import detect

    def refuse(model, *args, **kwargs):
        raise FileNotFoundError(f"{model.name}: no downloads in the unit tests")

    def no_renders(*args, **kwargs):
        raise RuntimeError("no renders in the unit tests")

    monkeypatch.setattr(detect, "model_path", refuse)
    monkeypatch.setattr(stage0, "render_views", no_renders)
    monkeypatch.setenv(stage0.CACHE_ENV, str(tmp_path / "headtex-cache"))


def draw(model, camera: PhotoCamera, texture=None, ambient=0.6, direction=(0.0, 0.0, -1.0), exposure=1.0,
         background=(120, 140, 160), index=0):
    """A photo (BGR) of ``model``'s head by ``camera``: its texture (or ``texture``) under an even light plus a
    directional one travelling along ``direction``, in linear light, times ``exposure``. Pixel (0, 0) holds
    ``index``, which the fake landmarker reads."""
    prim = model.primitives[0]
    corners = raycast.corners_of(prim.positions, prim.indices)
    r = raster.rasterise(camera, corners, barycentric=True)
    hit = r.triangle >= 0
    tri = np.asarray(prim.indices, np.int64).reshape(-1, 3)[r.triangle[hit]]
    bary = r.barycentric[hit].astype(np.float64)
    uv = np.einsum("nk,nkd->nd", bary, np.asarray(prim.uvs, np.float64)[tri])
    image = texture if texture is not None else prim.texture
    h, w = image.shape[:2]
    texels = np.c_[uv[:, 0] * w - 0.5, (1.0 - uv[:, 1]) * h - 0.5]
    colour = light.to_linear(fuse.sample(image, texels).astype(np.uint8))
    normal = np.einsum("nk,nkd->nd", bary, np.asarray(prim.normals, np.float64)[tri])
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    lit = np.clip(normal @ -np.asarray(direction, np.float64) / np.linalg.norm(direction), 0.0, 1.0)
    shading = ambient + (1.0 - ambient) * lit
    photo = np.empty((camera.size[1], camera.size[0], 3), np.uint8)
    photo[:] = background[::-1]
    photo[hit] = light.to_srgb(colour * (shading * exposure)[:, None])[:, ::-1]
    photo[0, 0] = index
    return photo


class HeadSegmenter:
    """The segmenter's masks of the synthetic head seen by the camera of the view whose number the photo holds: the
    head is face skin, and hair where ``hair(pixels mask)`` says (in the photo); background elsewhere."""

    def __init__(self, model, cameras, hair=None):
        prim = model.primitives[0]
        self.corners = raycast.corners_of(prim.positions, prim.indices)
        self.cameras, self.hair = cameras, hair

    def segment(self, image):
        camera = self.cameras[int(image[0, 0, 0])]
        head = np.isfinite(raster.rasterise(camera, self.corners).depth).astype(np.float32)
        masks = {name: np.zeros(head.shape, np.float32) for name in SEGMENT_CLASSES}
        hair = self.hair(head) if self.hair is not None else np.zeros_like(head)
        masks["hair"] = head * hair
        masks["face"] = head * (1.0 - hair)
        masks["background"] = 1.0 - head
        return masks


@pytest.fixture(scope="module")
def truth():
    """``test_headtex``'s landmarks of the synthetic head, with the brows, the cheeks (50, 280) and the forehead
    (151) where a face has them (there they are spread at random over the face)."""
    from bifold_demo.characters.headtex import canonical as cf

    model = synthetic_head_model()
    points = true_landmarks(model)
    prim = model.primitives[0]
    corners = raycast.corners_of(prim.positions, prim.indices)
    eyes = stage0.face_frame(model).eyes
    eye_y = eyes[:, 1].mean()

    def cast(xy):
        xy = np.asarray(xy, np.float64).reshape(-1, 2)
        origins = np.c_[xy, np.ones(len(xy))]
        _d, tri, bary = raycast.intersect(origins, np.tile([0.0, 0.0, -1.0], (len(xy), 1)), corners)
        return raycast.point_on(corners, tri, bary)

    for ids, eye in ((cf.RIGHT_BROW, eyes[0]), (cf.LEFT_BROW, eyes[1])):
        offsets = np.r_[np.c_[np.linspace(-0.016, 0.016, 5), np.full(5, 0.019)],
                        np.c_[np.linspace(-0.016, 0.016, 5), np.full(5, 0.013)]]
        points[list(ids)] = cast(eye[:2] + offsets)
    points[[50, 280, 151]] = cast([(-0.036, eye_y - 0.03), (0.036, eye_y - 0.03), (0.0, eye_y + 0.042)])
    # the face's outline, in order round the face (from the top, as MediaPipe's runs)
    t = np.linspace(0.0, 2.0 * np.pi, len(cf.FACE_OVAL), endpoint=False)
    oval = np.c_[0.058 * np.sin(t), CENTRE[1] - 0.008 + 0.072 * np.cos(t)]
    points[list(cf.FACE_OVAL)] = cast(oval)
    return points


def _head(model, truth):
    frame = stage0.face_frame(model)
    return stage0.head_landmarks(model, FakeLandmarker(truth, stage0.views(frame)), render=fake_render,
                                 cache_dir=False, refresh=True)


@pytest.fixture(scope="module")
def avatar(truth):
    """The textured synthetic head, its landmarks and its atlas."""
    model = textured_head()
    head = _head(model, truth)
    return model, head, fuse.head_atlas(model, head, labels=None, cache_dir=False)


def cameras_for(yaws, pitch=0.0, distance=0.45):
    return [PhotoCamera.around(CENTRE, distance, yaw, pitch, fov=40.0, size=SIZE) for yaw in yaws]


def session(model, head, truth, cameras, *, texture=None, lights=None, exposures=None, segmenter=None, **options):
    """Photos of the head by ``cameras``, registered (focal length known) and fused: the :class:`HeadTexture`."""
    lights = lights or [{}] * len(cameras)
    exposures = exposures or [1.0] * len(cameras)
    photos = [draw(model, c, texture, exposure=e, index=k, **lit)
              for k, (c, lit, e) in enumerate(zip(cameras, lights, exposures))]
    segmenter = segmenter or HeadSegmenter(model, cameras)
    views = stage1.register_photos(photos, head, model, focals=[c.focal for c in cameras],
                                   landmarker=FakeLandmarker(truth, cameras), segmenter=segmenter)
    assert all(v.usable for v in views), [v.summary() for v in views]
    return stages.fuse_views(views, model, head, labels=None, cache_dir=False, **options)


def lab_of(rgb_uint8):
    return light.lab(light.to_linear(rgb_uint8))


def up_to_exposure(got_rgb, want_rgb):
    """``(gain, delta E)``: the one gain per channel (linear light) that best maps ``got_rgb`` onto ``want_rgb``
    (``(N, 3)`` sRGB bytes) and the colour differences after it. A photo's light and its exposure are known only up
    to such a gain: how bright the skin really is no photo tells."""
    got, want = light.to_linear(got_rgb), light.to_linear(want_rgb)
    gain = np.median(want / np.maximum(got, 1e-4), axis=0)
    return gain, light.delta_e(light.lab(got * gain), light.lab(want))


# -- the z-buffer -------------------------------------------------------------------------------------------------


def test_the_zbuffer_agrees_with_ray_casting():
    model = synthetic_head_model()
    prim = model.primitives[0]
    corners = raycast.corners_of(prim.positions, prim.indices)
    camera = PhotoCamera.around(CENTRE, 0.5, 30.0, 10.0, fov=40.0, size=(160, 120))
    r = raster.rasterise(camera, corners, barycentric=True)
    ys, xs = np.mgrid[0:120:3, 0:160:3]
    pixels = np.c_[xs.ravel(), ys.ravel()].astype(float)
    origin, directions = camera.rays(pixels)
    distance, triangle, _bary = raycast.intersect(origin, directions, corners)
    hit = triangle >= 0
    depth = r.depth[ys.ravel(), xs.ravel()]
    assert np.array_equal(hit, np.isfinite(depth))
    assert np.abs(distance[hit] * (directions[hit] @ camera.forward) - depth[hit]).max() < 1e-5
    assert (r.triangle[ys.ravel(), xs.ravel()][hit] == triangle[hit]).mean() > 0.99
    points = raycast.point_on(corners, r.triangle[ys.ravel(), xs.ravel()][hit], r.barycentric[ys.ravel(), xs.ravel()]
                              [hit])
    assert np.abs(points - (origin + directions[hit] * distance[hit, None])).max() < 1e-5
    # culled: from inside the head nothing faces the camera
    inside = PhotoCamera.around(CENTRE, 0.01, 0.0, fov=60.0, size=(40, 30))
    assert (raster.rasterise(inside, corners, cull=True).triangle < 0).all()
    # a crop sees the same head at its own pixels
    big = PhotoCamera.around(CENTRE, 0.5, 20.0, fov=40.0, size=(4000, 3000))
    part = raster.crop(big, corners.reshape(-1, 3), 0.05, 200)
    assert max(part.camera.size) <= 200 and part.factor < 0.2
    probe = CENTRE + np.array([[0.0, 0.0, 0.09], [0.03, 0.02, 0.08]])
    assert np.allclose(part.pixels(big.project(probe)), part.camera.project(probe), atol=1e-9)


# -- the light --------------------------------------------------------------------------------------------------


def test_the_light_fit_recovers_a_known_light_despite_outliers():
    rng = np.random.default_rng(0)
    normals = rng.normal(size=(6000, 3))
    normals[:, 2] = np.abs(normals[:, 2]) + 0.3  # a photo sees about half of the sphere of normals
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    direction = np.array([0.6, 0.5, 0.62])
    direction /= np.linalg.norm(direction)
    true = 0.35 + 0.65 * np.clip(normals @ direction, 0.0, 1.0)  # an even light and a directional one
    albedo = 0.4 * (1.0 + 0.05 * rng.normal(size=len(normals)))  # skin varies a little
    values = albedo * true
    values[:600] *= 0.3  # a tenth in a cast shadow, hair or misregistered
    fit = light.fit_shading(values, normals)
    assert fit is not None and fit.inliers < 0.95
    relative = fit(normals[600:]) / true[600:]
    relative /= np.median(relative)
    lit, dark = true[600:] > 0.8, true[600:] < 0.4
    # the shape of the light: within 2 % where it is lit; where only the even light reaches (a third as bright),
    # truncating a clamped cosine at order 2 leaves about 10 % (where this stops working for harsh light)
    assert np.abs(relative[lit] - 1.0).mean() < 0.02 and np.abs(relative[dark] - 1.0).mean() < 0.15
    # scaled to 1 at the brightest light the skin shows
    assert np.isclose(light.weighted_percentile(fit(normals[600:]), np.ones(5400), light.NORMALISE_AT), 1.0,
                      atol=0.02)
    assert fit.direction @ direction > 0.9 and fit.strength > 0.3
    # never darker than the skin showed (the light's third, give or take): no division by an extrapolation
    assert 0.2 < fit.range[0] < 0.4 and fit.clamped(-normals).min() >= fit.range[0]
    flat = light.fit_shading(np.full(500, 0.3), normals[:500])
    assert flat.strength < 0.01 and np.allclose(flat(normals[:500]), 1.0, atol=1e-6)
    assert light.fit_shading(values[:50], normals[:50]) is None  # too few to fit


def test_colours_and_skin_tones():
    rgb = np.arange(256, dtype=np.uint8)
    assert np.array_equal(light.to_srgb(light.to_linear(rgb)), rgb)
    lab = light.lab(light.to_linear(np.array([[205, 150, 128]], np.uint8)))
    assert np.allclose(light.to_srgb(light.from_lab(lab)), [[205, 150, 128]], atol=1)
    rng = np.random.default_rng(1)
    skin = lab + rng.normal(0, 1.0, (400, 3))
    shadows = lab + np.array([-30.0, 0.0, 0.0]) + rng.normal(0, 1.0, (100, 3))  # a fifth in shadow
    stray = np.tile([[20.0, 2.0, 5.0]], (40, 1))  # hair
    tone = light.tone({"a": np.concatenate([skin[:200], shadows]), "b": np.concatenate([skin[200:], stray])})
    assert light.delta_e(tone.lab, lab[0]) < 2.0 and set(tone.per_view) == {"a", "b"}
    assert tone.rgb == pytest.approx((205, 150, 128), abs=4) and str(tone).startswith("#")
    beard = light.to_linear(np.array([[150, 150, 150]] * 30 + [[205, 150, 128]] * 100, np.uint8))
    keep = light.like_skin(beard)
    assert not keep[:30].any() and keep[30:].all()


# -- the pieces of stage 3 --------------------------------------------------------------------------------------


def test_push_pull_fills_from_the_mask_only():
    image = np.zeros((64, 64, 3), np.float32)
    mask = np.zeros((64, 64), bool)
    image[10:20, 10:20] = (1.0, 0.0, 0.0)
    mask[10:20, 10:20] = True
    image[40:50, 40:50] = (0.0, 0.0, 1.0)
    mask[40:50, 40:50] = True
    image[~mask] = 7.0  # whatever is outside must not count
    filled = fuse.push_pull(image, mask)
    assert np.array_equal(filled[mask], image[mask])
    assert filled.max() <= 1.0 + 1e-6 and filled.min() >= 0.0  # only the patches' colours, mixed
    assert filled[15, 22, 0] > 0.6 > filled[15, 22, 2] and filled[45, 52, 2] > 0.6 > filled[45, 52, 0]
    gradient = np.abs(np.diff(filled[..., 0], axis=1))[~(mask[:, 1:] | mask[:, :-1])]
    assert gradient.max() < 0.15  # smooth: no edges of its own
    single = fuse.push_pull(image[..., 0], mask)
    assert single.shape == (64, 64) and np.allclose(single, filled[..., 0])
    assert np.array_equal(fuse.push_pull(image, np.ones((64, 64), bool)), image)


def test_the_blend_stays_inside_its_island_and_hides_seams():
    size = 128
    islands = np.full((size, size), -1, np.int32)
    islands[:, :64] = 0
    islands[:, 64:] = 1  # two islands touching in the layout, not on the body
    red = np.zeros((size, size, 3), np.float32)
    red[..., 0] = 0.8
    blue = np.zeros((size, size, 3), np.float32)
    blue[..., 2] = 0.8
    w_red = np.where(islands == 0, 1.0, 0.0).astype(np.float32)
    w_blue = np.where(islands == 1, 1.0, 0.0).astype(np.float32)
    out, share = fuse.blend([red, blue], [w_red, w_blue], islands, 5)
    assert np.abs(out[:, :64] - red[:, :64]).max() < 1e-4 and np.abs(out[:, 64:] - blue[:, 64:]).max() < 1e-4
    assert np.allclose(share[0][:, :64], 1.0) and np.allclose(share[1][:, :64], 0.0)
    # within one island two photos that differ in tone: no step where one takes over, the detail of the better one
    islands[:] = 0
    yy, xx = np.mgrid[0:size, 0:size]
    detail = np.where(((xx // 4) + (yy // 4)) % 2 == 0, 0.05, -0.05).astype(np.float32)
    left = np.repeat((0.5 + detail)[..., None], 3, axis=2)
    right = left + 0.1  # a brighter photo
    w_left = np.clip((80 - xx) / 32.0, 0.0, 1.0).astype(np.float32)
    out, _share = fuse.blend([left, right], [w_left, 1.0 - w_left], islands, 5)
    row = out[64, :, 0] - detail[64]  # the tone without the detail
    assert np.abs(np.diff(row)).max() < 0.01  # it changes smoothly, over many texels
    assert abs(row[5] - 0.5) < 0.01 and abs(row[-5] - 0.6) < 0.01
    assert np.abs(out[64, :40, 0] - left[64, :40, 0]).max() < 0.02  # the detail kept


def test_gains_even_out_exposure_and_white_balance():
    size = 64
    rng = np.random.default_rng(2)
    colour = (0.3 + 0.1 * rng.random((size, size, 3))).astype(np.float32)
    mask = np.ones((size, size), bool)

    def projection(gain, weight):
        return fuse.Projection("p", colour * np.asarray(gain, np.float32), weight.astype(np.float32),
                               np.full((size, size), 255, np.uint8), np.zeros((size, size), np.uint8),
                               np.ones((size, size), np.float16), 1.0)

    xx = np.mgrid[0:size, 0:size][1]
    a = projection((1.0, 1.0, 1.0), np.clip((48 - xx) / 16, 0, 1))
    b = projection((1.3, 1.2, 0.9), np.clip((xx - 16) / 16, 0, 1))  # brighter, warmer
    g = fuse.gains([a, b], mask)
    assert np.allclose(g[0] * 1.0 / (g[1] * np.array([1.3, 1.2, 0.9])), 1.0, atol=0.02)
    assert np.allclose(fuse.gains([a], mask), 1.0)  # one photo: nothing to compare with


def test_padding_fills_the_gutters_with_the_islands_own_colours():
    rgba = np.zeros((64, 64, 4), np.uint8)
    covered = np.zeros((64, 64), bool)
    covered[8:24, 8:24] = True
    rgba[8:24, 8:24] = (200, 40, 40, 255)
    covered[30:60, 34:60] = True
    rgba[30:60, 34:60] = (40, 40, 200, 255)
    out = fuse.pad(rgba, covered)
    assert np.array_equal(out[covered], rgba[covered])
    assert (out[..., 3] == 255).all()
    # the texels next to an island: its colour (the islands grow by PAD_DILATE texels at 1024, one at 64)
    assert tuple(out[16, 24, :3]) == (200, 40, 40) and tuple(out[45, 33, :3]) == (40, 40, 200)
    assert out[16, 27, 0] > out[16, 27, 2] and out[45, 30, 2] > out[45, 30, 0]  # further out: mostly its colour
    # mipmaps of the padded texture keep the islands' colours at their edges
    small = cv2.resize(out, (16, 16), interpolation=cv2.INTER_AREA)
    assert small[2:6, 2:6, 0].min() > 150


# -- the atlas --------------------------------------------------------------------------------------------------


def test_the_atlas_knows_the_face_its_features_and_the_cheeks(avatar):
    model, head, atlas = avatar
    assert atlas.size == TEX and atlas.scalp_from == "colour"
    counts = {fuse.PARTS[k]: int((atlas.parts == k).sum()) for k in range(len(fuse.PARTS))}
    assert counts["face"] > 0.8 * TEX * TEX and counts["scalp"] == 0  # all skin: no hair by colour
    assert counts["eye"] > 0  # the eye openings (the synthetic head has no eyeballs)
    for bit, ids in ((fuse.BROWS, cf_ids("RIGHT_BROW")), (fuse.LIPS, cf_ids("LIPS_OUTER"))):
        texel = [atlas.bake.texel(_uv_of(model, head, k)) for k in ids]
        assert np.mean([(atlas.features[t] & bit) > 0 for t in texel]) > 0.8
    cheeks = atlas.bake.position[atlas.patch == fuse.CHEEKS]
    forehead = atlas.bake.position[atlas.patch == fuse.FOREHEAD]
    frame = stage0.face_frame(model)
    assert len(cheeks) > 50 and len(forehead) > 50
    assert (cheeks[:, 1] < frame.eyes[:, 1].mean()).all() and (forehead[:, 1] > frame.eyes[:, 1].mean()).all()
    # the texture's gradient (lighter towards the top) is what the avatar's painted soft light would be
    up = np.nan_to_num(atlas.bake.normal[..., 1])
    assert atlas.shading[atlas.covered & (up > 0.5)].mean() > atlas.shading[atlas.covered & (up < -0.5)].mean() + 0.1
    assert atlas.own_fit.direction[1] > 0.7
    assert np.allclose((atlas.own_albedo * atlas.shading[..., None])[atlas.covered], atlas.own[atlas.covered])
    assert fuse.head_atlas(model, head, labels=None, cache_dir=False) is atlas  # kept in memory


def cf_ids(name):
    from bifold_demo.characters.headtex import canonical as cf

    return getattr(cf, name)


def _uv_of(model, head, k):
    prim = model.primitives[0]
    tri = np.asarray(prim.indices, np.int64).reshape(-1, 3)[head.triangle[k]]
    return head.barycentric[k] @ np.asarray(prim.uvs, np.float64)[tri]


def test_the_atlas_is_cached_on_disk(truth, tmp_path):
    model = textured_head()
    head = _head(model, truth)
    first = fuse.head_atlas(model, head, labels=None, cache_dir=tmp_path)
    files = list(stage0.cache_folder(tmp_path).glob("atlas-*.npz"))
    assert len(files) == 1
    fuse._atlases.clear()
    again = fuse.head_atlas(model, head, labels=None, cache_dir=tmp_path)
    assert again is not first and np.array_equal(again.parts, first.parts)
    assert np.array_equal(again.bake.triangle, first.bake.triangle)
    files[0].write_bytes(b"not an atlas")
    fuse._atlases.clear()
    made = fuse.head_atlas(model, head, labels=None, cache_dir=tmp_path)
    assert np.array_equal(made.parts, first.parts)


# -- projecting and fusing --------------------------------------------------------------------------------------


def test_a_photo_lands_on_the_right_texels(avatar, truth):
    model, head, atlas = avatar
    camera = cameras_for([25.0])[0]
    photo = draw(model, camera, ambient=1.0)
    view = stage1.register_photo(photo, head, model, landmarker=FakeLandmarker(truth, [camera]), focal=camera.focal,
                                 masks=HeadSegmenter(model, [camera]).segment(photo))
    p = fuse.project(view, atlas, debug=True)
    assert p.weight.max() == pytest.approx(view.quality, rel=0.1)  # (a small face: its quality caps the weights)
    seen = p.weight > 0.3 * p.weight.max()
    assert seen.sum() > 3000 and set(p.factors) == {"angle", "class", "edge", "resolution"}
    expected = light.to_linear(checker_texture())
    error = light.delta_e(light.lab(p.colour[seen]), light.lab(expected[seen]))
    assert np.median(error) < 2.5
    # the far side of the head is not seen, nor the side of the nose turned away
    x = np.nan_to_num(atlas.bake.position[..., 0] - CENTRE[0])
    z = np.nan_to_num(atlas.bake.position[..., 2] - CENTRE[2])
    assert not (p.weight[(z < -0.02)] > 0).any()
    assert (p.weight[(x < -0.04) & (z > 0.0)] > 0).mean() > (p.weight[(x > 0.04) & (z > 0.0)] > 0).mean()


def test_fused_photos_give_back_the_texture(avatar, truth):
    """Three photos (front and turned 40 degrees each way) with different exposures and light: the face comes back
    within a few delta E, the light and the exposures evened out; what no photo shows is the avatar's own."""
    model, head, atlas = avatar
    lights = [{"direction": (0.3, -0.2, -1.0)}, {"direction": (-0.5, -0.3, -0.8)}, {"direction": (0.6, -0.4, -0.7)}]
    result = session(model, head, truth, cameras_for([0.0, 40.0, -40.0]), lights=lights,
                     exposures=[1.0, 0.75, 1.2])
    face = (atlas.parts == fuse.FACE) & (result.trimap == fuse.KEEP)
    assert face.sum() > 4000  # (of 64k texels round the whole head)
    gain, error = up_to_exposure(result.rgba[..., :3][face], checker_texture()[face])
    assert np.median(error) < 3.0 and np.percentile(error, 90) < 6.0
    assert np.all((gain > 0.8) & (gain < 1.35)) and np.ptp(gain) < 0.05  # the level about right, no colour cast
    assert result.skin is not None and light.delta_e(result.skin.lab, result.own_skin.lab) < 6.0
    gains = np.array([p.gain for p in result.projections])
    assert gains[1].mean() / gains[2].mean() == pytest.approx(1.2 / 0.75, rel=0.08)
    assert set(result.shares) == {"front", "three-quarter right", "three-quarter left", "avatar"}
    assert sum(result.shares.values()) == pytest.approx(1.0, abs=0.01)
    back = np.nan_to_num(atlas.bake.position[..., 2]) < CENTRE[2] - 0.05
    assert (result.trimap[back & atlas.covered] == fuse.GENERATE).mean() > 0.95
    # what no photo shows: the avatar's own texture times the change from its skin tone to the person's
    tone = result.skin.linear / result.own_skin.linear
    assert np.all((tone > 0.8) & (tone < 1.25))  # (the person is the avatar here)
    expected = light.lab(light.to_linear(checker_texture()[back]) * tone)
    assert np.median(light.delta_e(lab_of(result.rgba[..., :3][back]), expected)) < 1.5
    assert (result.trimap[atlas.parts == fuse.EYE] == fuse.KEEP).all()
    assert any("back of the head" in w for w in result.warnings)
    assert result.rgba.shape == (TEX, TEX, 4) and result.rgba.dtype == np.uint8
    assert set(result.timings) >= {"projection", "light and gains", "blend", "finish"}


def test_a_larger_texture_marks_the_photos_as_the_ordinary_size_does(avatar, truth):
    """A head texture made at twice its ordinary size (as at 2048 for a Rocketbox head) from small photos: the photos'
    resolution is marked per texel of the ordinary texture, so the result is as sure and its trimap the same, and the
    photos do not lose weight against the avatar's own texture (counted per texel of the larger texture, they had
    under one pixel each and lost a third of their weight, and stage 5 would have filled in much of the face)."""
    model, head, _atlas = avatar
    assert fuse.marks_size(model.primitives[0]) == TEX
    cameras = cameras_for([0.0, 35.0], distance=0.9)  # small in the photos
    results = {size: session(model, head, truth, cameras, size=size) for size in (TEX, 2 * TEX)}
    small, large = results[TEX], results[2 * TEX]
    assert large.size == 2 * TEX and large.atlas.marks_size == TEX
    face = (small.atlas.parts == fuse.FACE) & (small.confidence > 0.05)
    assert face.sum() > 1500
    seen = [p.footprint[p.weight > 0].astype(np.float64) for p in large.projections]
    assert np.median(np.concatenate(seen)) < 0.8  # under one photo pixel per texel of the larger texture
    shrunk = cv2.resize(large.confidence, (TEX, TEX), interpolation=cv2.INTER_AREA)
    assert np.median(np.abs(shrunk[face] - small.confidence[face])) < 0.02
    small_share = (small.trimap[face] != fuse.GENERATE).mean()
    large_face = cv2.resize(face.astype(np.uint8), (2 * TEX, 2 * TEX), interpolation=cv2.INTER_NEAREST) > 0
    assert abs((large.trimap[large_face] != fuse.GENERATE).mean() - small_share) < 0.03
    assert abs(large.shares["avatar"] - small.shares["avatar"]) < 0.03


def test_a_strong_shadow_over_half_the_face_is_not_baked_in(avatar, truth):
    """A light from the person's left with little light elsewhere: the right half of the face is several times
    darker in the photo. De-lit, the halves of the texture agree; without it, the shadow stays."""
    model, head, atlas = avatar
    harsh = {"direction": (-1.0, -0.2, -0.3), "ambient": 0.25}  # travelling to the person's right: lit from the left
    x = np.nan_to_num(atlas.bake.position[..., 0] - CENTRE[0])

    def halves(result):
        sure = (atlas.parts == fuse.FACE) & (result.confidence > 0.3) & (atlas.features == 0)
        lab = lab_of(result.rgba[..., :3])
        return lab[sure & (x > 0.02), 0].mean(), lab[sure & (x < -0.02), 0].mean()

    cameras = cameras_for([0.0, 30.0, -30.0])
    raw = session(model, head, truth, cameras, lights=[harsh] * 3, delight=False)
    left, right = halves(raw)
    assert left - right > 15.0  # the shadow, baked in
    fused = session(model, head, truth, cameras, lights=[harsh] * 3)
    left, right = halves(fused)
    assert abs(left - right) < 4.0
    # the front photo sees the light and the shadow; the one of the shadowed side sees hardly any contrast, and its
    # gain (it took the shadow for the light) is the larger
    front, shadowed, lit = fused.projections
    assert front.shading.contrast > 3.0 and shadowed.shading.contrast < front.shading.contrast
    assert shadowed.gain.mean() > 1.5 * lit.gain.mean()


def test_hair_only_goes_to_the_scalp_and_the_hairline(truth):
    """A photo with a fringe over the forehead: the fringe's hair never lands on the avatar's forehead skin (it keeps
    its own there), while the photo's hair does land on the avatar's scalp."""
    model = textured_head()
    head = _head(model, truth)
    frame = stage0.face_frame(model)
    outline = stage1.Outline.of(model, 0)
    middle = outline.triangles.mean(axis=1)
    scalp = middle[:, 1] > frame.eyes[:, 1].mean() + 0.045  # the avatar's scalp: the top of the head
    labels = np.where(scalp, stage1.HAIR, stage1.SKIN).astype(np.int8)
    atlas = fuse.head_atlas(model, head, labels=labels, cache_dir=False)
    assert atlas.scalp_from == "segmenter" and (atlas.parts == fuse.SCALP).sum() > 1000
    camera = cameras_for([0.0], pitch=15.0)[0]
    brow_y = camera.project(head.points[list(cf_ids("RIGHT_BROW"))])[:, 1].min()

    def fringe(head_mask):  # hair over the upper half of the forehead, and on the head above it
        out = np.zeros_like(head_mask)
        out[: int(brow_y) - 12] = 1.0
        return out

    photo = draw(model, camera, ambient=1.0)
    photo[: int(brow_y) - 12][(np.isfinite(raster.rasterise(camera, outline.triangles).depth))[: int(brow_y) - 12]] = (
        30, 30, 40)  # dark hair over the forehead and the top of the head (BGR)
    masks = HeadSegmenter(model, [camera], fringe).segment(photo)
    view = stage1.register_photo(photo, head, model, landmarker=FakeLandmarker(truth, [camera]), focal=camera.focal,
                                 masks=masks)
    result = stages.fuse_views([view], model, head, labels=labels, cache_dir=False)
    lab = lab_of(result.rgba[..., :3])
    skin_lab = lab_of(SKIN.astype(np.uint8)[None])[0]
    p = result.projections[0]
    hit = camera.project(np.nan_to_num(atlas.bake.position.reshape(-1, 3))).reshape(TEX, TEX, 2)
    under_fringe = (hit[..., 1] < brow_y - 14) & (atlas.parts == fuse.FACE) & ~atlas.hairline
    assert under_fringe.sum() > 100 and (p.weight[under_fringe] < 0.01).all()  # not from the photo
    assert np.median(light.delta_e(lab[under_fringe], skin_lab)) < 15.0  # skin, not hair
    on_scalp = (atlas.parts == fuse.SCALP) & (p.weight > 0.2)
    assert on_scalp.sum() > 100 and np.median(lab[on_scalp][:, 0]) < 30.0  # the photo's dark hair
    assert result.hair is not None and result.hair.lab[0] < 30.0


def test_tone_matching_and_a_photo_without_masks(avatar, truth):
    """A person darker than the avatar: the texels no photo shows take the person's tone; without the segmenter
    the face's landmark oval stands in for the face skin."""
    model, head, atlas = avatar
    person = light.to_srgb(light.to_linear(checker_texture()) * np.array([0.45, 0.4, 0.36], np.float32))
    cameras = cameras_for([0.0, 35.0])
    photos = [draw(model, c, person, index=k) for k, c in enumerate(cameras)]
    views = stage1.register_photos(photos, head, model, focals=[c.focal for c in cameras],
                                   landmarker=FakeLandmarker(truth, cameras))
    assert all(v.usable and v.masks is None for v in views)
    result = stages.fuse_views(views, model, head, labels=None, cache_dir=False)
    assert result.skin.lab[0] < result.own_skin.lab[0] - 15.0
    back = (np.nan_to_num(atlas.bake.position[..., 2]) < CENTRE[2] - 0.05) & atlas.covered
    expected = lab_of(person[back])
    assert np.median(light.delta_e(lab_of(result.rgba[..., :3][back]), expected)) < 4.0
    face = (atlas.parts == fuse.FACE) & (result.trimap == fuse.KEEP)
    assert face.sum() > 2000  # the oval gave the face
    assert np.median(light.delta_e(lab_of(result.rgba[..., :3][face]), lab_of(person[face]))) < 4.0


def test_other_people_in_the_photo_do_not_count():
    masks = {name: np.zeros((100, 200), np.float32) for name in ("face", "hair", "body", "others")}
    masks["face"][20:60, 20:60] = 1.0  # the person
    masks["face"][20:60, 120:160] = 1.0  # someone else
    view = stage1.View(np.zeros((100, 200, 3), np.uint8), "front", True,
                       projected=np.array([[40.0, 40.0], [30.0, 30.0], [50.0, 50.0]]))
    person = fuse._person(view, masks, (200, 100))
    assert person[40, 40] == 1.0 and person[40, 140] == 0.0


def test_the_whole_way_and_wearing_it(truth, tmp_path, monkeypatch):
    """head_texture() from photos with fakes for MediaPipe, put on a character, taken off; in the background too."""
    from bifold_demo.characters.character import Character

    model = textured_head()
    head = _head(model, truth)
    cameras = cameras_for([0.0, -30.0])
    photos = [draw(model, c, index=k) for k, c in enumerate(cameras)]
    segmenter = HeadSegmenter(model, cameras)
    monkeypatch.setattr(stage1, "head_regions", lambda *args, **kwargs: None)  # (renders of the avatar otherwise)
    result = stages.head_texture(photos, model, focals=[c.focal for c in cameras], head=head,
                                 landmarker=FakeLandmarker(truth, cameras), segmenter=segmenter, cache_dir=False)
    assert isinstance(result, stages.HeadTexture) and result.primitive == "s_head"
    assert "registration" in result.timings and "shares:" in result.summary()
    path = result.save(tmp_path / "head.png")
    assert cv2.imread(str(path), cv2.IMREAD_UNCHANGED).shape == (TEX, TEX, 4)
    person = Character(model, name="ada", shadow=False)
    texture = stages.wear(person, result)
    assert person.texture_overrides["s_head"] is texture and np.array_equal(texture.rgba, result.rgba)
    stages.take_off(person, result)
    assert "s_head" not in person.texture_overrides
    # off the main thread: the main thread keeps running while it works
    main = threading.current_thread()
    seen = []
    future = stages.head_texture_async(photos, model, focals=[c.focal for c in cameras], head=head,
                                       landmarker=FakeLandmarker(truth, cameras), segmenter=segmenter,
                                       cache_dir=False)
    ticks = 0
    while not future.done():
        ticks += 1
        time.sleep(0.01)
    seen.append(future.result())
    assert ticks > 0 and threading.current_thread() is main
    assert np.array_equal(seen[0].rgba, result.rgba)
    with pytest.raises(ValueError):  # no usable photo: said, not a texture of nothing
        stages.fuse_views([stage1.View(photos[0], "back", False, "no face")], model, head, labels=None,
                          cache_dir=False)
