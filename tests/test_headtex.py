"""Head textures from photos, stages 0 and 1, on synthetic data (no MediaPipe, no OpenGL, no avatars needed).

A synthetic head (a sphere with a nose and the Rocketbox face bones) stands in for an avatar, a fake renderer and a
fake Face Landmarker for MediaPipe: the "landmarker" knows the 478 true landmark points on the head and projects them
with the camera of the view it is shown.
"""

import math
import struct
import sys
import zipfile

import cv2
import numpy as np
import pytest

from bifold_demo import downloads
from bifold_demo.characters.headtex import canonical as cf
from bifold_demo.characters.headtex import detect, raycast
from bifold_demo.characters.headtex import landmarks as stage0
from bifold_demo.characters.headtex import register as stage1
from bifold_demo.characters.headtex.camera import PhotoCamera, head_angles, head_rotation
from bifold_demo.characters.headtex.detect import FaceMarks
from bifold_demo.characters.headtex.warp import ThinPlateWarp
from bifold_demo.characters.model import CharacterModel, Skeleton, SkinnedPrimitive

CENTRE = np.array([0.0, 1.6, 0.0])  # the synthetic head's centre
RADIUS = 0.09


# -- a synthetic head ------------------------------------------------------------------------------------------


def _sphere(rows=40, cols=64):
    v, u = np.meshgrid(np.linspace(0.0, 1.0, rows + 1), np.linspace(0.0, 1.0, cols + 1), indexing="ij")
    theta, phi = v * math.pi, u * 2 * math.pi  # theta from the top
    d = np.stack([np.sin(theta) * np.sin(phi), np.cos(theta), np.sin(theta) * np.cos(phi)], axis=-1).reshape(-1, 3)
    p = d * RADIUS
    nose = np.exp(-((p[:, 0] / 0.012) ** 2 + ((p[:, 1] + 0.012) / 0.016) ** 2)) * (p[:, 2] > 0)
    p = p + d * (0.022 * nose)[:, None]
    tri = []
    for r in range(rows):
        for c in range(cols):
            a, b = r * (cols + 1) + c, r * (cols + 1) + c + 1
            e, f = a + cols + 1, b + cols + 1
            tri += [(a, e, b), (b, e, f)]  # counter-clockwise seen from outside
    tri = np.array(tri)
    corners = p[tri]
    keep = np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1) > 1e-12
    tri = tri[keep]
    normals = np.zeros_like(p)
    face_n = np.cross(p[tri[:, 1]] - p[tri[:, 0]], p[tri[:, 2]] - p[tri[:, 0]])
    for k in range(3):
        np.add.at(normals, tri[:, k], face_n)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    uvs = np.c_[u.ravel(), 1.0 - v.ravel()]
    return p + CENTRE, normals, uvs, tri


def synthetic_head_model() -> CharacterModel:
    """A sphere head with a nose, a "head" primitive and the five face bones of a Rocketbox avatar."""
    bones = [("Bip01", None, (0.0, 0.0, 0.0)), ("Bip01 Head", "Bip01", tuple(CENTRE)),
             ("Bip01 REye", "Bip01 Head", (-0.032, 1.62, 0.07)), ("Bip01 LEye", "Bip01 Head", (0.032, 1.62, 0.07)),
             ("Bip01 RMouthCorner", "Bip01 Head", (-0.024, 1.556, 0.07)),
             ("Bip01 LMouthCorner", "Bip01 Head", (0.024, 1.556, 0.07))]
    names = [b[0] for b in bones]
    parents = [names.index(p) if p else -1 for _, p, _ in bones]
    position = np.array([b[2] for b in bones], float)
    rest = np.repeat(np.eye(4)[None], len(bones), axis=0)
    for k, p in enumerate(parents):
        rest[k, :3, 3] = position[k] - (position[p] if p >= 0 else 0)
    skeleton = Skeleton(names, parents, rest)
    p, n, uv, tri = _sphere()
    joints = np.zeros((len(p), 4), np.int64)
    joints[:, 0] = 1
    weights = np.zeros((len(p), 4), np.float32)
    weights[:, 0] = 1.0
    head = SkinnedPrimitive("s_head", p.astype(np.float32), n.astype(np.float32), uv.astype(np.float32),
                            tri.astype(np.uint32), joints, weights, np.full((4, 4, 4), 200, np.uint8))
    return CharacterModel("Synthetic", skeleton, [head], np.linalg.inv(skeleton.rest_global))


def true_landmarks(model) -> np.ndarray:
    """478 points on the front of the synthetic head, cast from straight in front: the eyes, irises, mouth, nose
    tip and chin where a face has them (around the face bones), the others spread over the face."""
    head = model.primitives[0]
    corners = raycast.corners_of(head.positions, head.indices)
    frame = stage0.face_frame(model)
    rng = np.random.default_rng(7)
    r = 0.055 * np.sqrt(rng.random(cf.COUNT))
    a = rng.random(cf.COUNT) * 2 * math.pi
    xy = np.c_[r * np.cos(a), CENTRE[1] - 0.005 + 1.1 * r * np.sin(a)]

    def ring(ids, centre, rx, ry):
        t = np.linspace(0.0, 2 * math.pi, len(ids), endpoint=False)
        xy[list(ids)] = centre[:2] + np.c_[rx * np.cos(t), ry * np.sin(t)]

    for eye, contour, iris in ((frame.eyes[0], cf.RIGHT_EYE, cf.RIGHT_IRIS), (frame.eyes[1], cf.LEFT_EYE,
                                                                              cf.LEFT_IRIS)):
        ring(contour, eye, 0.014, 0.006)
        xy[iris[0]] = eye[:2]
        xy[list(iris[1:])] = eye[:2] + 0.0055 * np.array(stage0.IRIS_RIM)
    mouth = frame.mouth.mean(axis=0)
    ring(cf.LIPS_OUTER, mouth, 0.024, 0.008)
    ring(cf.LIPS_INNER, mouth, 0.02, 0.002)
    xy[cf.MOUTH_RIGHT], xy[cf.MOUTH_LEFT] = frame.mouth[0, :2], frame.mouth[1, :2]
    xy[cf.NOSE_TIP] = frame.nose[:2]
    xy[cf.CHIN] = (0.0, mouth[1] - 0.03)
    origins = np.c_[xy, np.full(cf.COUNT, 1.0)]
    _dist, tri, bary = raycast.intersect(origins, np.tile([0.0, 0.0, -1.0], (cf.COUNT, 1)), corners)
    assert (tri >= 0).all()
    return raycast.point_on(corners, tri, bary)


@pytest.fixture(autouse=True)
def no_downloads(monkeypatch):
    """Unit tests never download MediaPipe's models."""

    def refuse(model, *args, **kwargs):
        raise FileNotFoundError(f"{model.name}: no downloads in the unit tests")

    monkeypatch.setattr(detect, "model_path", refuse)


class FakeLandmarker:
    """Projects known 3D landmarks with the camera of the view whose number the (fake) render holds."""

    def __init__(self, points, cameras=None, person=None):
        self.points, self.cameras, self.person, self.calls = points, cameras, person, 0

    def detect(self, image):
        self.calls += 1
        if self.cameras is None:
            return []
        camera = self.cameras[int(image[0, 0, 0])]
        pixels = camera.project(self.points)
        if self.person is not None:
            pixels = self.person(pixels)
        return [FaceMarks(pixels, np.zeros(len(pixels)))]


def fake_render(model, cameras):
    return [np.full((c.size[1], c.size[0], 3), k, np.uint8) for k, c in enumerate(cameras)]


@pytest.fixture(scope="module")
def head_model():
    return synthetic_head_model()


@pytest.fixture(scope="module")
def truth(head_model):
    return true_landmarks(head_model)


@pytest.fixture(scope="module")
def head(head_model, truth, tmp_path_factory):
    frame = stage0.face_frame(head_model)
    fake = FakeLandmarker(truth, stage0.views(frame))
    return stage0.head_landmarks(head_model, fake, render=fake_render, cache_dir=tmp_path_factory.mktemp("head"),
                                 refresh=True)


# -- the camera ------------------------------------------------------------------------------------------------


def test_head_angles_round_trip_and_mean_what_they_say():
    rng = np.random.default_rng(0)
    for yaw, pitch, roll in rng.uniform([-170, -60, -60], [170, 60, 60], (20, 3)):
        assert np.allclose(head_angles(head_rotation(yaw, pitch, roll)), (yaw, pitch, roll), atol=1e-9)
    # turned to its left: the camera sees the head's right side (-X); looking down: the camera is above
    camera = PhotoCamera.around(CENTRE, 1.0, yaw=30.0)
    assert camera.position[0] < -0.4 and np.allclose(camera.angles(), (30, 0, 0), atol=1e-9)
    assert PhotoCamera.around(CENTRE, 1.0, pitch=20.0).position[1] > CENTRE[1] + 0.3
    # leaning towards its left shoulder: the top of the head moves to the person's left, the right of the photo
    lean = PhotoCamera.around(CENTRE, 1.0, roll=20.0, size=(400, 400))
    assert lean.project(CENTRE + [0, 0.1, 0])[0, 0] > 199.5 + 25


def test_the_camera_projects_like_the_renderer_and_back():
    camera = PhotoCamera.around(CENTRE, 0.7, 25.0, -10.0, 5.0, fov=35.0, size=(640, 480))
    assert np.allclose(camera.project(CENTRE), [[319.5, 239.5]])
    assert camera.fov == pytest.approx(35.0)
    # the same camera as the scene's look-at camera, in OpenCV's axes
    from bifold_demo.scene.transforms import look_at

    scene = camera.scene_camera()
    view = look_at(scene.position, scene.target, scene.up)
    assert np.allclose(np.diag([1.0, -1.0, -1.0]) @ view[:3, :3], camera.R, atol=1e-9)
    assert np.allclose(PhotoCamera.looking_at(scene.position, scene.target, scene.fov, (640, 480), scene.up).R,
                       camera.R, atol=1e-9)
    points = CENTRE + np.random.default_rng(1).normal(0, 0.05, (30, 3))
    origin, directions = camera.rays(camera.project(points))
    along = np.einsum("nk,nk->n", points - origin, directions)
    assert np.allclose(origin + directions * along[:, None], points, atol=1e-9)
    half = camera.scaled(0.5)
    assert np.allclose(half.project(points), (camera.project(points) + 0.5) * 0.5 - 0.5)


# -- stage 1: the camera of a photo ------------------------------------------------------------------------------


def _face_points(n=150, seed=3):
    """A face-like point cloud: a half ellipsoid with a nose (metres, character space)."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(-1.2, 1.2, n)
    b = rng.uniform(-1.0, 1.0, n)
    x, y = 0.07 * np.sin(a), 0.1 * b
    z = 0.08 * np.cos(a) * np.sqrt(np.clip(1 - b * b, 0.05, 1)) + 0.02 * np.exp(-(x / 0.015) ** 2 - (y / 0.02) ** 2)
    return np.c_[x, y, z] + CENTRE


@pytest.mark.parametrize("yaw,pitch,roll,distance,fov", [(0, 0, 0, 0.5, 50), (35, -10, 5, 0.6, 40),
                                                          (-55, 12, -8, 0.4, 60)])
def test_pnp_recovers_a_known_camera(yaw, pitch, roll, distance, fov):
    points = _face_points()
    truth = PhotoCamera.around(CENTRE, distance, yaw, pitch, roll, fov=fov, size=(800, 600))
    pixels = truth.project(points) + np.random.default_rng(4).normal(0, 0.3, (len(points), 2))
    pixels[::7] += 60.0  # outliers: a seventh of the points far off
    fit = stage1.solve_camera(pixels, points, (800, 600), focal=truth.focal)
    assert np.allclose(fit.camera.angles(), (yaw, pitch, roll), atol=0.5)
    assert fit.error < 0.6 and not fit.inliers[::7].any() and fit.inliers.sum() >= len(points) - len(points[::7])
    assert np.linalg.norm(fit.camera.position - truth.position) < 0.01
    # the focal length searched for: close up, the perspective tells it
    searched = stage1.solve_camera(pixels, points, (800, 600))
    assert searched.camera.focal == pytest.approx(truth.focal, rel=0.15)
    assert np.allclose(searched.camera.angles(), (yaw, pitch, roll), atol=1.5)
    assert np.median(np.linalg.norm(searched.camera.project(points) - truth.project(points), axis=1)) < 1.0


def test_pnp_gives_up_on_nonsense():
    points = _face_points()
    rng = np.random.default_rng(5)
    assert stage1.solve_camera(rng.uniform(0, 800, (len(points), 2)), points, (800, 600), focal=700.0) is None


def test_the_focal_length_from_exif(tmp_path):
    from PIL import Image

    def photo(size, exif_size, f35):
        path = tmp_path / f"p{size[0]}x{size[1]}-{f35}.jpg"
        exif = Image.Exif()
        exif[0x8769] = {0xA405: f35, 0xA002: exif_size[0], 0xA003: exif_size[1]}
        Image.new("RGB", size).save(path, exif=exif)
        return path

    assert stage1.exif_focal(photo((400, 300), (4000, 3000), 26), (400, 300)) == pytest.approx(26 / 43.27 * 500)
    assert stage1.exif_focal(photo((400, 400), (4000, 3000), 26), (400, 400)) is None  # cropped
    path = tmp_path / "plain.jpg"
    Image.new("RGB", (40, 30)).save(path)
    assert stage1.exif_focal(path, (40, 30)) is None


# -- stage 1: the warp -------------------------------------------------------------------------------------------


def test_the_warp_maps_its_control_points_exactly_and_bends_smoothly():
    rng = np.random.default_rng(1)
    source = rng.uniform(100, 400, (120, 2))
    target = source + 6.0 * np.c_[np.sin(source[:, 1] / 50), np.cos(source[:, 0] / 60)]
    warp = ThinPlateWarp(source, target)
    assert np.abs(warp(source) - target).max() < 1e-8
    grid = np.mgrid[100:400:6, 100:400:6].reshape(2, -1).T.astype(float)
    assert warp.jacobian_determinant(grid).min() > 0.8  # no folds, little squeezing
    moved = np.linalg.norm(warp(grid) - grid, axis=1)
    assert moved.max() < 9.0  # nothing moves much further than the control points do (6 px at most)
    map_x, map_y = warp.maps((500, 450))
    for x, y in ((150, 222), (321, 287)):
        assert np.allclose((map_x[y, x], map_y[y, x]), warp([[x, y]])[0], atol=0.05)
    affine = np.array([[1.05, 0.1], [-0.08, 0.95]])
    assert np.allclose(ThinPlateWarp(source, source @ affine + 3.0)(grid), grid @ affine + 3.0, atol=1e-6)
    assert np.allclose(ThinPlateWarp.identity()(grid), grid) and ThinPlateWarp.identity().bending_energy() == 0.0


def test_the_face_warp_moves_the_face_and_leaves_the_rest_alone():
    rng = np.random.default_rng(2)
    projected = 300 + rng.normal(0, 40, (cf.COUNT, 2))
    found = projected + rng.normal(0, 2.0, (cf.COUNT, 2))
    warp = stage1.face_warp(projected, found, np.ones(cf.COUNT, bool), stage1.face_size(projected))
    taken = np.linalg.norm(warp(projected) - found, axis=1) < 1e-6
    assert taken.sum() > 50 and taken[list(cf.FEATURES)].mean() > taken.mean() - 0.2  # the features come first
    far = 300 + 400 * np.c_[np.cos(np.linspace(0, 6, 20)), np.sin(np.linspace(0, 6, 20))]
    assert np.abs(warp(far) - far).max() < 1e-9  # the photo beyond the ring around the face stays put
    image = np.zeros((600, 600, 3), np.uint8)
    cv2.circle(image, (300, 300), 50, (0, 0, 255), -1)
    assert ThinPlateWarp.identity().apply(image).shape == image.shape


# -- ray casting -------------------------------------------------------------------------------------------------


def test_rays_and_nearest_points_on_a_mesh():
    rng = np.random.default_rng(0)
    corners = rng.normal(size=(60, 3, 3))
    points = rng.normal(size=(20, 3))
    distance, tri, bary = raycast.closest(points, corners)
    assert np.allclose(np.linalg.norm(raycast.point_on(corners, tri, bary) - points, axis=1), distance)
    u = rng.random((4000, 2))
    u[u.sum(axis=1) > 1] = 1 - u[u.sum(axis=1) > 1]
    samples = np.einsum("sk,tkd->tsd", np.c_[1 - u.sum(axis=1), u], corners).reshape(-1, 3)
    brute = np.linalg.norm(samples[None] - points[:, None], axis=2).min(axis=1)
    assert (distance <= brute + 1e-9).all() and (brute - distance).max() < 0.05
    square = np.array([[[-1, -1, 0], [1, -1, 0], [1, 1, 0]], [[-1, -1, 0], [1, 1, 0], [-1, 1, 0]]], float)
    d = np.c_[rng.uniform(-0.5, 0.5, (30, 2)), -np.ones(30)]
    t, tri, bary = raycast.intersect([0.0, 0.0, 1.0], d / np.linalg.norm(d, axis=1, keepdims=True), square)
    assert (tri >= 0).all() and np.abs(raycast.point_on(square, tri, bary)[:, 2]).max() < 1e-12
    t, tri, _ = raycast.intersect([0.0, 0.0, 1.0], [[0.0, 0.0, 1.0]], square)
    assert tri[0] == -1 and math.isinf(t[0])


# -- stage 0 -----------------------------------------------------------------------------------------------------


def test_landmarks_are_lifted_onto_the_mesh(head_model, truth, head):
    assert head.method == "mediapipe" and head.views == len(stage0.VIEWS)
    error = np.linalg.norm(head.points - truth, axis=1)
    assert np.median(error) < 0.0005 and np.percentile(error, 95) < 0.002
    corners = raycast.corners_of(head_model.primitives[0].positions, head_model.primitives[0].indices)
    assert np.allclose(raycast.point_on(corners, head.triangle, head.barycentric), head.points, atol=1e-9)
    assert head.reliability.min() >= 0.0 and np.median(head.reliability) > 0.9
    # the landmarks follow the skinning: the head bone moved 5 cm up moves them 5 cm up
    palette = np.repeat(np.eye(4)[None], len(head_model.skeleton), axis=0)
    assert np.allclose(head.posed(head_model, palette), head.points, atol=1e-6)
    palette[1, 1, 3] = 0.05
    assert np.allclose(head.posed(head_model, palette), head.points + [0, 0.05, 0], atol=1e-6)


def test_landmarks_are_cached_by_the_head_geometry(head_model, truth, tmp_path, monkeypatch):
    frame = stage0.face_frame(head_model)
    fake = FakeLandmarker(truth, stage0.views(frame))
    first = stage0.head_landmarks(head_model, fake, render=fake_render, cache_dir=tmp_path, refresh=True)
    files = list((tmp_path / "headtex").glob("landmarks-*.npz"))
    assert len(files) == 1 and first.key in files[0].name and fake.calls == len(stage0.VIEWS)
    stage0._memory.clear()
    again = stage0.head_landmarks(head_model, fake, render=fake_render, cache_dir=tmp_path)
    assert fake.calls == len(stage0.VIEWS) and np.array_equal(again.points, first.points)  # from the file
    monkeypatch.setenv(stage0.CACHE_ENV, str(tmp_path / "elsewhere"))
    assert stage0.cache_folder(tmp_path) == tmp_path / "elsewhere"


def test_landmarks_without_a_face_on_the_renders_fall_back_to_the_canonical_face(head_model, truth, tmp_path):
    nobody = FakeLandmarker(truth, cameras=None)
    with pytest.raises(ValueError):
        stage0.head_landmarks(head_model, nobody, render=fake_render, cache_dir=False, method="mediapipe",
                              refresh=True)
    frame = stage0.face_frame(head_model)
    canonical = cf.CanonicalFace((truth[:cf.MESH_COUNT] - frame.nose) * 100.0, np.zeros((cf.MESH_COUNT, 2)),
                                 np.zeros((0, 3), np.int64))  # the true face, in centimetres
    fallback = stage0.head_landmarks(head_model, nobody, render=fake_render, canonical=canonical, cache_dir=False,
                                     refresh=True)
    assert fallback.method == "canonical" and fallback.views == 0
    assert np.allclose(fallback.reliability, stage0.FALLBACK_RELIABILITY)
    corners = raycast.corners_of(head_model.primitives[0].positions, head_model.primitives[0].indices)
    assert raycast.closest(fallback.points, corners)[0].max() < 1e-9  # on the mesh
    assert np.median(np.linalg.norm(fallback.points[:cf.MESH_COUNT] - truth[:cf.MESH_COUNT], axis=1)) < 0.012
    for iris, eye in ((cf.RIGHT_IRIS, frame.eyes[0]), (cf.LEFT_IRIS, frame.eyes[1])):
        centre = fallback.points[iris[0]]
        assert np.linalg.norm(centre[:2] - eye[:2]) < 0.002 and centre[2] > eye[2]  # in front of the eye bone


def test_a_head_without_face_bones_has_no_landmarks():
    from test_yoga import synthetic_biped

    with pytest.raises(ValueError):
        stage0.face_frame(synthetic_biped())


# -- stage 1: views --------------------------------------------------------------------------------------------


def test_a_photo_with_a_face_is_registered(head_model, truth, head):
    camera = PhotoCamera.around(CENTRE, 0.6, 20.0, 8.0, -4.0, fov=40.0, size=(640, 480))
    centre = camera.project(CENTRE)[0]

    def wider(pixels):  # the person's face is a little wider than the avatar's
        return pixels + np.c_[0.06 * (pixels[:, 0] - centre[0]), np.zeros(len(pixels))]

    photo = np.full((480, 640, 3), 0, np.uint8)  # the fake landmarker reads view 0 from the first pixel
    fake = FakeLandmarker(truth, [camera], person=wider)
    view = stage1.register_photo(photo, head, head_model, landmarker=fake, focal=camera.focal)
    assert view.usable and view.method == "landmarks" and view.name == "front"
    assert np.allclose(view.angles, (20.0, 8.0, -4.0), atol=3.5)  # the wider face costs a little yaw
    assert 0.0 < view.quality <= 1.0 and view.error < 6.0 and "front: yaw" in view.summary()
    person = wider(camera.project(truth))
    taken = np.linalg.norm(view.warp(view.projected) - person, axis=1) < 1e-3  # the warp's control points
    assert taken.sum() > 30
    seen = stage1.facing(head, head_model, view.camera) > 0.3
    assert np.median(np.linalg.norm(view.photo_pixels(head.points)[seen] - person[seen], axis=1)) < 1.0
    assert view.warped().shape == photo.shape
    drawn = stage1.draw_registration(view, head_model, head)
    assert drawn.shape == photo.shape and (drawn != photo).any()


def test_photos_without_a_face_need_a_view_name_and_a_head_mask(head_model, truth, head):
    photo = np.zeros((300, 400, 3), np.uint8)
    nobody = FakeLandmarker(truth, cameras=None)
    unnamed = stage1.register_photo(photo, head, head_model, landmarker=nobody)
    assert not unnamed.usable and "name the view" in unnamed.reason and unnamed.camera is None
    no_mask = stage1.register_photo(photo, head, head_model, landmarker=nobody, view="back")
    assert not no_mask.usable and "head mask" in no_mask.reason
    with pytest.raises(ValueError):
        stage1.register_photo(photo, head, head_model, landmarker=nobody, view="sideways")


def test_a_side_photo_is_placed_by_the_head_outline(head_model, truth, head):
    size = (480, 400)
    front_camera = PhotoCamera.around(CENTRE, 0.8, 0.0, fov=35.0, size=size)
    front = stage1.register_photo(np.zeros((400, 480, 3), np.uint8), head, head_model,
                            landmarker=FakeLandmarker(truth, [front_camera]), focal=front_camera.focal)
    side_camera = PhotoCamera.around(CENTRE, 0.8, 90.0, fov=35.0, size=size)
    mask = stage1.Outline.of(head_model, 0).mask(side_camera, (400, 480)).astype(np.float32)
    masks = {"hair": mask, "face": np.zeros_like(mask)}
    view = stage1.register_photo(np.zeros((400, 480, 3), np.uint8), head, head_model, view="right", masks=masks,
                           landmarker=FakeLandmarker(truth, cameras=None), reference=front)
    assert view.usable and view.method == "silhouette" and view.iou > 0.95
    yaw, pitch, _roll = view.angles
    assert abs(yaw - 90.0) < 10.0 and abs(pitch) < 6.0
    assert np.linalg.norm(view.camera.position - side_camera.position) < 0.1
    bar = np.zeros_like(mask)
    bar[20:380, 200:230] = 1.0  # a tall thin bar: no head looks like that
    wrong = {"hair": bar, "face": np.zeros_like(mask)}
    assert not stage1.register_photo(np.zeros((400, 480, 3), np.uint8), head, head_model, view="right", masks=wrong,
                               landmarker=FakeLandmarker(truth, cameras=None), reference=front).usable


def test_photos_of_one_session_share_the_focal_length(head_model, truth, head):
    size = (640, 480)
    cameras = [PhotoCamera.around(CENTRE, 0.6, yaw, 0.0, fov=40.0, size=size) for yaw in (0.0, 40.0)]
    photos = [np.full((480, 640, 3), k, np.uint8) for k in range(2)]
    views = stage1.register_photos(photos, head, head_model, landmarker=FakeLandmarker(truth, cameras))
    assert all(v.usable for v in views) and views[0].name == "front"
    assert views[1].camera.focal == pytest.approx(views[0].camera.focal)
    assert any("focal length of the front photo" in w for w in views[1].warnings)


# -- the canonical face, the downloads, MediaPipe missing ---------------------------------------------------------


def _varint(value):
    out = b""
    while True:
        byte, value = value & 0x7F, value >> 7
        out += bytes([byte | (0x80 if value else 0)])
        if not value:
            return out


def _field(number, wire, payload):
    key = _varint(number << 3 | wire)
    return key + (_varint(len(payload)) + payload if wire == 2 else payload)


def test_the_canonical_face_is_read_from_the_task_bundle(tmp_path):
    rng = np.random.default_rng(0)
    vertices = rng.normal(size=(cf.MESH_COUNT, 5)).astype("<f4")
    indices = rng.integers(0, cf.MESH_COUNT, 30)
    mesh = _field(1, 0, _varint(0)) + _field(2, 0, _varint(0))
    mesh += b"".join(_field(3, 5, struct.pack("<f", v)) for v in vertices.ravel())  # unpacked floats
    mesh += b"".join(_field(4, 0, _varint(int(i))) for i in indices)  # unpacked varints
    basis = _field(1, 0, _varint(4)) + _field(2, 5, struct.pack("<f", 0.5))
    message = _field(1, 2, mesh) + _field(2, 2, basis) + _field(3, 0, _varint(1))
    face = cf.parse_canonical_face(message)
    assert np.allclose(face.points, vertices[:, :3]) and np.allclose(face.uvs, vertices[:, 3:])
    assert np.array_equal(face.triangles.ravel(), indices)
    packed = _field(3, 2, vertices.tobytes()) + _field(4, 2, b"".join(_varint(int(i)) for i in indices))
    assert np.allclose(cf.parse_canonical_face(_field(2, 2, packed)).points, vertices[:, :3])
    bundle = tmp_path / "face_landmarker.task"
    with zipfile.ZipFile(bundle, "w") as z:
        z.writestr(cf.METADATA, message)
    assert np.allclose(cf.read_canonical_face(bundle).points, face.points)
    with pytest.raises(ValueError):
        cf.parse_canonical_face(_field(2, 2, basis))


def test_landmark_groups_name_478_points():
    groups = (cf.FACE_OVAL, cf.RIGHT_EYE, cf.LEFT_EYE, cf.RIGHT_BROW, cf.LEFT_BROW, cf.LIPS_OUTER, cf.LIPS_INNER,
              cf.NOSE, cf.RIGHT_IRIS, cf.LEFT_IRIS)
    for group in groups:
        assert len(set(group)) == len(group) and all(0 <= k < cf.COUNT for k in group)
    assert len(cf.FACE_OVAL) == 36 and cf.RIGHT_IRIS[0] == 468 and cf.LEFT_IRIS[-1] == 477


def test_model_files_are_checked_after_download(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"model weights" * 100)
    good = downloads.ModelFile("model.bin", source.as_uri(), downloads.sha256(source), source.stat().st_size)
    folder = tmp_path / "models"
    with pytest.raises(FileNotFoundError, match="bifold-faces fetch"):
        downloads.ensure(good, folder, download=False, fetch_hint="bifold-faces fetch")
    path = downloads.ensure(good, folder, progress=None)
    assert path.read_bytes() == source.read_bytes()
    bad = downloads.ModelFile("other.bin", source.as_uri(), "0" * 64, source.stat().st_size)
    with pytest.raises(OSError, match="checksum"):
        downloads.ensure(bad, folder, progress=None)
    assert sorted(p.name for p in folder.iterdir()) == ["model.bin"]  # no half-written files left


def test_without_mediapipe_the_error_says_how_to_install_it(monkeypatch):
    monkeypatch.setitem(sys.modules, "mediapipe", None)
    with pytest.raises(detect.MediaPipeMissing, match="pip install"):
        detect.mediapipe()
    with pytest.raises(ImportError):
        detect.Landmarker()
    from bifold_demo.recognition.cli import main

    assert main(["landmarks", "--characters", "/nonexistent-folder"]) == 2  # no avatars: a message, not a crash


def test_the_command_line_splits_off_view_names():
    from bifold_demo.characters.headtex.cli import _photo_and_view

    assert _photo_and_view("a/b.jpg:left") == ("a/b.jpg", "left")
    assert _photo_and_view("C:/photos/back.jpg") == ("C:/photos/back.jpg", None)
    assert _photo_and_view("x.jpg:back") == ("x.jpg", "back")

