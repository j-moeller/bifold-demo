"""Head textures from photos, stages 0 and 1, on the real avatars with MediaPipe.

Skipped without a prepared character library (``BIFOLD_CHARACTERS_TEST_DIR``), MediaPipe or OpenGL. MediaPipe's models
are downloaded into the face models folder on first use (``BIFOLD_NO_DOWNLOADS`` stops that). The tests with real
photos of people also need ``BIFOLD_FACE_PHOTOS_TEST_DIR``, a folder of portrait photos (never committed).
"""

import os
from pathlib import Path

import numpy as np
import pytest

from bifold_demo.characters.headtex import canonical as cf
from bifold_demo.characters.headtex import landmarks as stage0
from bifold_demo.characters.headtex import register as stage1
from bifold_demo.characters.headtex.camera import PhotoCamera

CHARACTERS = Path(os.environ.get("BIFOLD_CHARACTERS_TEST_DIR", "/nonexistent")).expanduser()
PHOTOS = Path(os.environ.get("BIFOLD_FACE_PHOTOS_TEST_DIR", "/nonexistent")).expanduser()
AVATARS = ("Male_Adult_05", "Business_Female_02", "Female_Child_01", "Male_Adult_21")  # glasses, a child, a keffiyeh


@pytest.fixture(scope="module")
def library():
    if not CHARACTERS.is_dir():
        pytest.skip("set BIFOLD_CHARACTERS_TEST_DIR to a prepared character library")
    from bifold_demo.characters.library import CharacterLibrary

    library = CharacterLibrary(CHARACTERS)
    missing = [name for name in AVATARS if name not in library.available()]
    if missing:
        pytest.skip(f"the character library lacks {', '.join(missing)}")
    return library


def _download() -> bool:
    return os.environ.get("BIFOLD_NO_DOWNLOADS") is None


@pytest.fixture(scope="module")
def landmarker():
    from bifold_demo.characters.headtex.detect import Landmarker, MediaPipeMissing

    try:
        found = Landmarker(download=_download(), max_faces=2)
    except MediaPipeMissing as exc:
        pytest.skip(str(exc))
    except OSError as exc:
        pytest.skip(f"MediaPipe's face landmarker is unavailable: {exc}")
    yield found
    found.close()


@pytest.fixture(scope="module")
def segmenter(landmarker):
    from bifold_demo.characters.headtex.detect import Segmenter

    try:
        found = Segmenter(download=_download())
    except OSError as exc:
        pytest.skip(f"MediaPipe's segmenter is unavailable: {exc}")
    yield found
    found.close()


@pytest.fixture(scope="module")
def renders():
    """Renderers for stage 0 and for the fake photos (one OpenGL context each for the module)."""
    from conftest import _gl_error

    if _gl_error():
        pytest.skip(_gl_error())
    from bifold_demo.scene.renderer import Renderer

    made = {"stage0": Renderer(stage0.RENDER_SIZE, stage0.RENDER_SIZE), "photo": Renderer(640, 480)}
    yield made
    for renderer in made.values():
        renderer.release()


def _landmarks(library, landmarker, renders, name, **options):
    model = library.load(name)
    marks = stage0.head_landmarks(model, landmarker, cache_dir=False,
                                  render=lambda m, cameras: stage0.render_views(m, cameras, renders["stage0"]),
                                  **options)
    return model, marks


@pytest.mark.parametrize("name", AVATARS)
def test_mediapipe_finds_the_landmarks_on_the_avatars_faces(library, landmarker, renders, name):
    model, marks = _landmarks(library, landmarker, renders, name)
    frame = stage0.face_frame(model)
    assert marks.method == "mediapipe" and marks.views >= 4 and marks.reliability.mean() > 0.65
    mm = 1000.0
    for iris, contour, eye in ((cf.RIGHT_IRIS, cf.RIGHT_EYE, frame.eyes[0]), (cf.LEFT_IRIS, cf.LEFT_EYE,
                                                                              frame.eyes[1])):
        centre = marks.points[iris[0]]
        assert np.linalg.norm(centre[:2] - eye[:2]) * mm < 3.0 and centre[2] > eye[2]  # on the eye, in front
        corners = marks.points[[contour[0], contour[8]]].mean(axis=0)  # half way between the eye's corners
        assert np.linalg.norm(corners[:2] - eye[:2]) * mm < 6.0
    assert marks.points[cf.LEFT_EYE_OUTER][0] > 0 > marks.points[cf.RIGHT_EYE_OUTER][0]  # the person's left is +X
    for k, bone in ((cf.MOUTH_RIGHT, frame.mouth[0]), (cf.MOUTH_LEFT, frame.mouth[1])):
        assert np.linalg.norm(marks.points[k] - bone) * mm < 8.0
    assert np.linalg.norm(marks.points[cf.NOSE_TIP] - frame.nose) * mm < 10.0
    assert marks.points[cf.CHIN][1] < frame.mouth[:, 1].mean() - 0.02
    assert marks.points[cf.FOREHEAD][1] > frame.eyes[:, 1].mean() + 0.03


@pytest.mark.parametrize("name", AVATARS[:2])
def test_the_canonical_face_fallback_comes_close(library, landmarker, renders, name):
    _, found = _landmarks(library, landmarker, renders, name)
    canonical = cf.read_canonical_face(landmarker.task_path)
    _model, fallback = _landmarks(library, landmarker, renders, name, canonical=canonical, method="canonical")
    assert fallback.method == "canonical" and fallback.views == 0
    error = np.linalg.norm(fallback.points - found.points, axis=1) * 1000.0
    assert error.mean() < 12.0 and error[list(cf.FEATURES)].mean() < 10.0
    assert error[list(cf.RIGHT_IRIS + cf.LEFT_IRIS)].max() < 5.0


def _photo_scene(model):
    from bifold_demo.characters.character import Character
    from bifold_demo.scene import Scene

    scene = Scene(background=(170, 180, 190), ambient=0.55, light_direction=(-0.4, -0.6, -0.7))
    scene.add(Character(model, name="person", shadow=False))
    scene.update(0.0, 0.0)
    return scene


@pytest.mark.parametrize("yaw,pitch,roll", [(0, 0, 0), (30, 0, 0), (-30, 0, 5), (60, 0, 0), (-60, 0, 0), (0, 15, 0),
                                            (0, -15, 0), (25, 10, -8)])
def test_fake_photos_register_back_to_their_cameras(library, landmarker, renders, yaw, pitch, roll):
    """Renders of a textured avatar from known cameras ("fake photos"), registered to its own head model."""
    model, marks = _landmarks(library, landmarker, renders, "Male_Adult_05")
    frame = stage0.face_frame(model)
    scene = _photo_scene(model)
    truth = PhotoCamera.around(frame.pivot, 0.75, yaw, pitch, roll, fov=35.0, size=(640, 480))
    scene.camera = truth.scene_camera()
    photo = renders["photo"].render(scene)
    side = abs(yaw) >= 50
    for focal in (truth.focal, None):
        view = stage1.register_photo(photo, marks, model, landmarker=landmarker, focal=focal)
        assert view.usable and view.method == "landmarks"
        searched = 1.0 if focal is None else 0.0  # a degree more when the focal length is searched for
        assert abs(view.angles[0] - yaw) < (7.0 if side else 3.0) + searched
        assert abs(view.angles[1] - pitch) < 3.5 + searched and abs(view.angles[2] - roll) < 3.5 + searched
        pixels = np.linalg.norm(view.camera.project(marks.points) - truth.project(marks.points), axis=1)
        assert np.median(pixels) < (6.0 if side else 3.0)
        seen = stage1.facing(marks, model, view.camera) > 0.3  # through the warp: MediaPipe's points
        warped = np.linalg.norm(view.photo_pixels(marks.points)[seen] - view.marks.points[seen], axis=1)
        assert np.median(warped) < (1.5 if side else 1.0)


def test_side_and_back_photos_are_placed_by_the_head_outline(library, landmarker, segmenter, renders):
    model, marks = _landmarks(library, landmarker, renders, "Male_Adult_05")
    frame = stage0.face_frame(model)
    scene = _photo_scene(model)
    photos, cameras = [], []
    for yaw in (0.0, 90.0, -90.0, 180.0):
        cameras.append(PhotoCamera.around(frame.pivot, 0.8, yaw, fov=35.0, size=(640, 480)))
        scene.camera = cameras[-1].scene_camera()
        photos.append(renders["photo"].render(scene))
    views = stage1.register_photos(photos, marks, model, views=["front", "right", "left", "back"],
                                   landmarker=landmarker, segmenter=segmenter)
    assert [v.method for v in views] == ["landmarks", "silhouette", "silhouette", "silhouette"]
    for view, truth, back in zip(views[1:], cameras[1:], (False, False, True)):
        assert view.usable and view.iou > 0.9, view.summary()
        yaw_off = abs((view.angles[0] - truth.angles()[0] + 180.0) % 360.0 - 180.0)
        # seen from behind, the outline hardly shows the nod (measured up to 13 degrees off)
        assert yaw_off < 8.0 and abs(view.angles[1]) < (15.0 if back else 6.0), view.summary()
        assert np.linalg.norm(view.camera.position - truth.position) < (0.2 if back else 0.08)


def test_the_ultralytics_sample_photo_registers(library, landmarker, renders):
    import cv2

    from bifold_demo.scene.textures import sample_image

    path = sample_image("zidane.jpg")
    if not path:
        pytest.skip("the Ultralytics sample photo is not installed")
    model, marks = _landmarks(library, landmarker, renders, "Male_Adult_05")
    view = stage1.register_photo(path, marks, model, landmarker=landmarker)
    assert view.usable and abs(view.angles[0]) < 25 and view.error < 0.04 * stage1.face_size(view.marks.points)
    assert cv2.imread(str(path)).shape == view.image.shape


def test_real_photos_register_with_plausible_poses(library, landmarker, renders):
    if not PHOTOS.is_dir():
        pytest.skip("set BIFOLD_FACE_PHOTOS_TEST_DIR to a folder of portrait photos")
    photos = sorted(p for p in PHOTOS.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not photos:
        pytest.skip(f"no photos in {PHOTOS}")
    model, marks = _landmarks(library, landmarker, renders, "Male_Adult_05")
    registered = 0
    for path in photos:
        view = stage1.register_photo(path, marks, model, landmarker=landmarker)
        if view.marks is None:
            continue  # no face in it
        registered += 1
        yaw, pitch, roll = view.angles
        assert view.usable and abs(yaw) < 40 and abs(pitch) < 25 and abs(roll) < 20, f"{path}: {view.summary()}"
        assert view.error < 0.04 * stage1.face_size(view.marks.points), f"{path}: {view.summary()}"
        assert 0.0 < view.quality <= 1.0
    assert registered >= max(1, len(photos) // 2)
