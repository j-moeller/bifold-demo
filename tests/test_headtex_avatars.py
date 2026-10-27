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
    """The landmarks where the face bones and the mesh say the features are (the largest distances measured on
    these four avatars: irises 2.4 mm from the eye bones, eye corners 2.2, mouth corners 8.0 from their bones, which
    sit inside the mouth, the nose tip 7.5 from the mesh's most forward point)."""
    model, marks = _landmarks(library, landmarker, renders, name)
    frame = stage0.face_frame(model)
    assert marks.method == "mediapipe" and marks.views >= 4 and marks.reliability.mean() > 0.65
    mm = 1000.0
    for iris, contour, eye in ((cf.RIGHT_IRIS, cf.RIGHT_EYE, frame.eyes[0]), (cf.LEFT_IRIS, cf.LEFT_EYE,
                                                                              frame.eyes[1])):
        centre = marks.points[iris[0]]
        assert np.linalg.norm(centre[:2] - eye[:2]) * mm < 4.0 and centre[2] > eye[2]  # on the eye, in front
        corners = marks.points[[contour[0], contour[8]]].mean(axis=0)  # half way between the eye's corners
        assert np.linalg.norm(corners[:2] - eye[:2]) * mm < 6.0
    assert marks.points[cf.LEFT_EYE_OUTER][0] > 0 > marks.points[cf.RIGHT_EYE_OUTER][0]  # the person's left is +X
    for k, bone in ((cf.MOUTH_RIGHT, frame.mouth[0]), (cf.MOUTH_LEFT, frame.mouth[1])):
        assert np.linalg.norm(marks.points[k] - bone) * mm < 11.0
    assert np.linalg.norm(marks.points[cf.NOSE_TIP] - frame.nose) * mm < 12.0
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
    """Renders of a textured avatar from known cameras ("fake photos"), registered to its own head model (measured:
    within 2.4 degrees and 1.4 pixels up to 30 degrees of yaw, 6.9 degrees and 5.3 pixels at 60)."""
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
        searched = 1.0 if focal is None else 0.0  # a degree more when the focal length is not known
        assert abs(view.angles[0] - yaw) < (9.0 if side else 3.5) + searched
        assert abs(view.angles[1] - pitch) < 5.0 + searched and abs(view.angles[2] - roll) < 5.0 + searched
        pixels = np.linalg.norm(view.camera.project(marks.points) - truth.project(marks.points), axis=1)
        assert np.median(pixels) < (8.0 if side else 3.0)
        seen = stage1.facing(marks, model, view.camera) > 0.3  # through the warp: MediaPipe's points
        warped = np.linalg.norm(view.photo_pixels(marks.points)[seen] - view.marks.points[seen], axis=1)
        assert np.median(warped) < (1.5 if side else 1.0)


@pytest.mark.parametrize("source", ["Female_Adult_04", "Male_Child_02"])
def test_fake_photos_of_other_avatars_register_to_a_head_model(library, landmarker, renders, source):
    """The research doc's sandbox test: renders of another avatar (a woman, a child) from known cameras, registered
    to Male_Adult_05's head model as a person's photos are. The faces differ, so the poses are a few degrees off (a
    child's face on an adult's head looks 5-7 degrees further down), but the warp still reads each of the avatar's
    landmarks where the person's is (measured on these two: turns within 3.4 degrees, nods within 6.9; the person's
    landmarks within 1.8 % of the face's size, the median, and 3.8 % for 90 % of them)."""
    if source not in library.available():
        pytest.skip(f"the character library lacks {source}")
    model, marks = _landmarks(library, landmarker, renders, "Male_Adult_05")
    other, person = _landmarks(library, landmarker, renders, source)
    frame = stage0.face_frame(other)
    scene = _photo_scene(other)
    for yaw, pitch, roll in ((0, 0, 0), (30, 0, 0), (-45, 0, 0), (0, 15, 0), (25, 10, -8)):
        truth = PhotoCamera.around(frame.pivot, 0.75, yaw, pitch, roll, fov=35.0, size=(640, 480))
        scene.camera = truth.scene_camera()
        photo = renders["photo"].render(scene)
        for focal in (truth.focal, None):
            view = stage1.register_photo(photo, marks, model, landmarker=landmarker, focal=focal)
            assert view.usable and view.method == "landmarks", view.summary()
            off = np.abs(np.array(view.angles) - (yaw, pitch, roll))
            assert off[0] < 6.0 and off[1] < 10.0 and off[2] < 4.0, (yaw, pitch, roll, view.summary())
            seen = stage1.visible(marks, model, view.camera, 0.3)
            read = np.linalg.norm(view.photo_pixels(marks.points[seen]) - truth.project(person.points[seen]), axis=1)
            size = stage1.face_size(view.marks.points)
            assert np.median(read) < 0.03 * size and np.percentile(read, 90) < 0.06 * size, (yaw, pitch, roll)


def test_faces_made_up_on_the_backs_of_heads_are_not_used(library, landmarker, renders):
    """MediaPipe "finds" nearly frontal faces on the backs of these two heads (turned 150 degrees). Without masks to
    check them, YuNet's second opinion turns them down: it sees no face there."""
    if landmarker._yunet(np.zeros((240, 320, 3), np.uint8)) is None:
        pytest.skip("YuNet's model is not available (bifold-faces fetch)")
    model, marks = _landmarks(library, landmarker, renders, "Male_Adult_05")
    for name in ("Male_Adult_12", "Male_Child_02"):
        if name not in library.available():
            pytest.skip(f"the character library lacks {name}")
        other = library.load(name)
        scene = _photo_scene(other)
        truth = PhotoCamera.around(stage0.face_frame(other).pivot, 0.8, 150.0, fov=35.0, size=(640, 480))
        scene.camera = truth.scene_camera()
        photo = renders["photo"].render(scene)
        faces = landmarker.detect(photo)
        view = stage1.register_photo(photo, marks, model, landmarker=landmarker, focal=truth.focal, faces=faces)
        assert not view.usable and view.camera is None, view.summary()
        if faces:  # (as with this MediaPipe build)
            assert "no face to YuNet" in view.reason and faces[0].detector_score < stage1.MIN_DETECTOR_SCORE


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
    for view, photo in zip(views, photos):  # each view keeps its own photo's masks
        assert np.array_equal(view.masks["hair"], segmenter.segment(photo)["hair"])
    for view, truth, back in zip(views[1:], cameras[1:], (False, False, True)):
        assert view.usable and view.iou > 0.9, view.summary()
        yaw_off = abs((view.angles[0] - truth.angles()[0] + 180.0) % 360.0 - 180.0)
        # seen from behind, the outline hardly shows the nod (measured up to 13 degrees off)
        assert yaw_off < 8.0 and abs(view.angles[1]) < (15.0 if back else 6.0), view.summary()
        assert np.linalg.norm(view.camera.position - truth.position) < (0.2 if back else 0.08)


def test_the_segmenters_masks_outlive_the_next_photo(segmenter):
    """MediaPipe reuses its buffers for the next photo: the masks it gave for one photo must be copies."""
    import cv2

    from bifold_demo.characters.headtex import detect
    from bifold_demo.scene.textures import sample_image

    path = sample_image("zidane.jpg")
    if not path:
        pytest.skip("the Ultralytics sample photo is not installed")
    photo = cv2.imread(str(path))
    first = segmenter.segment(photo)
    kept = {name: mask.copy() for name, mask in first.items()}
    segmenter.segment(np.ascontiguousarray(photo[:, ::-1]))  # mirrored: other masks
    segmenter.segment(np.ascontiguousarray(photo[100:, 300:]))
    for name, mask in first.items():
        assert mask.dtype == np.uint8 and mask.flags.owndata and max(mask.shape) <= detect.MASK_SIZE
        assert np.array_equal(mask, kept[name]), name
    assert kept["hair"].max() > 200 and kept["face"].max() > 200  # two people: hair and faces found


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


def test_a_small_face_in_a_big_photo_is_found(landmarker):
    """MediaPipe looks for faces in a 128-pixel copy of a photo; YuNet says where to look closer: a face 3 % as wide
    as a 12-megapixel photo is found, its landmarks where they are in the original."""
    import cv2

    from bifold_demo.scene.textures import sample_image

    path = sample_image("zidane.jpg")
    if not path:
        pytest.skip("the Ultralytics sample photo is not installed")
    photo = cv2.imread(str(path))
    face = landmarker.detect(photo)[0]
    scale = 0.03 * 4032 / face.box[2]
    small = cv2.resize(photo, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    canvas = np.full((3024, 4032, 3), 120, np.uint8)
    x, y = 600, 3024 - small.shape[0] - 200
    canvas[y:y + small.shape[0], x:x + small.shape[1]] = small
    if landmarker._yunet(canvas) is None:
        pytest.skip("YuNet's model is not available (bifold-faces fetch)")
    found = landmarker.detect(canvas)
    assert found and found[0].box[2] == pytest.approx(0.03 * 4032, rel=0.15)
    back = (found[0].points - (x, y)) / scale  # in the original photo
    assert np.median(np.linalg.norm(back - face.points, axis=1)) * scale < 2.0


def test_real_photos_register_with_plausible_poses(library, landmarker, renders):
    if not PHOTOS.is_dir():
        pytest.skip("set BIFOLD_FACE_PHOTOS_TEST_DIR to a folder of portrait photos")
    photos = sorted(p for p in PHOTOS.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not photos:
        pytest.skip(f"no photos in {PHOTOS}")
    import cv2

    model, marks = _landmarks(library, landmarker, renders, "Male_Adult_05")
    registered = 0
    for path in photos:
        faces = landmarker.detect(cv2.imread(str(path)))
        if not faces:
            continue  # no face in it
        assert faces[0].detector_score is None or faces[0].detector_score > 0.85, path  # YuNet sees it too
        view = stage1.register_photo(path, marks, model, landmarker=landmarker, faces=faces)
        registered += 1
        yaw, pitch, roll = view.angles
        assert view.usable and abs(yaw) < 40 and abs(pitch) < 25 and abs(roll) < 20, f"{path}: {view.summary()}"
        assert view.error < 0.04 * stage1.face_size(view.marks.points), f"{path}: {view.summary()}"
        assert 0.0 < view.quality <= 1.0
    assert registered >= max(1, len(photos) // 2)


def test_the_command_line_says_what_went_wrong(library, capsys):
    from bifold_demo.recognition.cli import main

    assert main(["landmarks", "Male_Adult_5", "--characters", str(CHARACTERS)]) == 2
    assert main(["register", "photo.jpg", "--avatar", "Male_Adult_5", "--characters", str(CHARACTERS)]) == 2
    err = capsys.readouterr().err
    assert err.count("no avatar or clip 'Male_Adult_5'") == 2 and "Traceback" not in err


def test_the_command_line_names_a_forced_fallback(library, landmarker, tmp_path, monkeypatch, capsys):
    from bifold_demo.recognition.cli import main

    monkeypatch.setenv(stage0.CACHE_ENV, str(tmp_path))  # not into the character library
    assert main(["landmarks", "Male_Adult_05", "--fallback", "--characters", str(CHARACTERS)]) == 0
    assert "fitted canonical face (--fallback)" in capsys.readouterr().out
