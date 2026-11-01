"""The recognition core with the real models: every profile whose models are here, on the sample photo, on the local
test photos of people (``BIFOLD_FACE_PHOTOS_TEST_DIR``) and on avatar renders (``BIFOLD_CHARACTERS_TEST_DIR`` and
OpenGL; renders of different avatars are different people).

Skipped without what they need. Nothing is downloaded (``BIFOLD_FACE_MODELS`` must hold the models: ``bifold-faces
fetch --profile best``, ``licensed``, ``commercial``) and nothing is written: photos of people and anything made from
them stay private."""

import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from bifold_demo.recognition import checks, registry
from bifold_demo.recognition.engine import FaceEngine

PHOTOS = Path(os.environ.get("BIFOLD_FACE_PHOTOS_TEST_DIR", "/nonexistent")).expanduser()
CHARACTERS = Path(os.environ.get("BIFOLD_CHARACTERS_TEST_DIR", "/nonexistent")).expanduser()
AVATARS = ("Male_Adult_05", "Female_Adult_04", "Business_Male_02", "Female_Child_01")
YAWS = (0.0, -30.0, 30.0)


def _engine(profile):
    try:
        return FaceEngine(download=False, profile=profile, progress=None)
    except (OSError, ImportError) as exc:
        pytest.skip(f"the {profile} profile's models are not here: {exc}")


@pytest.fixture(scope="module", params=list(registry.PROFILES))
def engine(request):
    return _engine(request.param)


@pytest.fixture(scope="module")
def zidane():
    from bifold_demo.scene.textures import sample_image

    path = sample_image("zidane.jpg")
    if not path:
        pytest.skip("the Ultralytics sample photo is not installed")
    return cv2.imread(str(path))


def test_each_profile_tells_two_men_apart_and_finds_them_again(engine, zidane):
    found = engine.faces(zidane)
    assert len(found) == 2
    (_, a), (_, b) = found
    assert abs(np.linalg.norm(a) - 1.0) < 1e-6 and a @ b < engine.threshold  # two different men
    brighter = cv2.convertScaleAbs(zidane, alpha=1.2, beta=25)
    smaller = cv2.resize(zidane, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    for other in (brighter, smaller, zidane[:, ::-1].copy()):
        scores = np.array([[x @ y for _, y in engine.faces(other)] for x in (a, b)])
        assert (scores.max(axis=1) > engine.threshold + 0.2).all(), (engine.model_id, scores)  # each man again ...
        assert scores[0].argmax() != scores[1].argmax()  # ... in a face of his own


def test_records_are_deterministic_and_the_same_alone_or_in_a_batch(engine, zidane):
    first, again = engine.analyse(zidane), engine.analyse(zidane)
    assert len(first) == 2 and all(np.array_equal(x.vector, y.vector) for x, y in zip(first, again))
    alone = engine.analyse(zidane, faces=[first[1].face])[0]
    assert np.allclose(alone.vector, first[1].vector, atol=1e-5)
    for record in first:
        q = record.quality
        assert q.ied == pytest.approx(record.face.eye_distance) and 0 < q.sharpness and 0 <= q.outside < 0.5
        assert (q.fiqa is not None) == (engine.quality_spec is not None)
        assert (q.norm is not None) == engine.embedder_spec.norm_quality


def _people():
    """Aligned faces of the test photos of people and who each shows (a folder per person)."""
    if not PHOTOS.is_dir():
        pytest.skip("set BIFOLD_FACE_PHOTOS_TEST_DIR to a folder of photos of people (a folder per person)")
    folder = PHOTOS / "train" if (PHOTOS / "train").is_dir() else PHOTOS
    return folder


def test_photos_of_one_person_score_above_other_peoples(engine):
    crops, labels = checks.labelled_faces(_people(), engine.detector)
    if len(set(labels.tolist())) < 2:
        pytest.skip("the photos show fewer than two people")
    vectors = engine.embedder.embed(crops).vectors
    scores = vectors @ vectors.T
    same = labels[:, None] == labels[None, :]
    upper = np.triu(np.ones_like(same), 1).astype(bool)
    mated, nonmated = scores[same & upper], scores[~same & upper]
    assert mated.min() > nonmated.max(), (engine.model_id, mated, nonmated.max())
    assert mated.min() > engine.threshold, (engine.model_id, mated.min())
    assert checks.separation(mated, nonmated)["d_prime"] > 3.0


def test_the_registered_colour_order_separates_people_better():
    folder = _people()
    for profile in ("best", "licensed"):
        engine = _engine(profile)
        spec = engine.embedder_spec
        crops, labels = checks.labelled_faces(folder, engine.detector)
        report = checks.colour_order(spec, engine._embedder_path, crops, labels)
        assert report.ok, str(report)
        mine, other = ("BGR", "RGB") if spec.bgr else ("RGB", "BGR")
        assert report.values[mine]["d_prime"] > report.values[other]["d_prime"] + 0.5, str(report)


# -- avatar renders -------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def renders():
    """``{(avatar, yaw): (image, projected five points)}``: each avatar's head from the front and turned 30 degrees
    both ways, at about 75 pixels between the eyes."""
    from conftest import _gl_error

    if not CHARACTERS.is_dir():
        pytest.skip("set BIFOLD_CHARACTERS_TEST_DIR to a prepared character library")
    if _gl_error():
        pytest.skip(_gl_error())
    from bifold_demo.characters.character import Character
    from bifold_demo.characters.headtex import landmarks as stage0
    from bifold_demo.characters.headtex.camera import PhotoCamera
    from bifold_demo.characters.library import CharacterLibrary
    from bifold_demo.scene import Scene
    from bifold_demo.scene.renderer import Renderer

    library = CharacterLibrary(CHARACTERS)
    names = [n for n in AVATARS if n in library.available()]
    if len(names) < 3:
        pytest.skip("the character library lacks the avatars")
    renderer = Renderer(640, 480)
    out = {}
    try:
        for name in names:
            model = library.load(name)
            frame = stage0.face_frame(model)
            points = np.vstack([frame.eyes, frame.nose[None], frame.mouth])  # the right eye (the image's left) first
            scene = Scene(background=(170, 180, 190), ambient=0.5, light_direction=(-0.6, -0.5, -0.6))
            scene.add(Character(model, name="person", shadow=False))
            scene.update(0.0, 0.0)
            for yaw in YAWS:
                camera = PhotoCamera.around(frame.pivot, 0.7, yaw, 0.0, fov=35.0, size=(640, 480))
                scene.camera = camera.scene_camera()
                out[(name, yaw)] = (renderer.render(scene), camera.project(points))
    finally:
        renderer.release()
    return out


def test_faces_and_landmarks_are_found_on_renders(engine, renders):
    for (name, yaw), (image, points) in renders.items():
        faces = engine.detect(image)
        assert len(faces) == 1, (name, yaw, faces)
        face = faces[0]
        ied = np.linalg.norm(points[1] - points[0])
        assert face.points[0, 0] < face.points[1, 0] and face.points[3, 0] < face.points[4, 0]  # one order
        if yaw == 0:  # (turned, the eye bones, the eyeballs' centres, lie behind the pupils the detectors see)
            error = np.linalg.norm(face.points - points, axis=1) / ied
            limit = 0.2 if engine.profile.detector == "scrfd_10g_kps" else 0.3
            assert error.max() < limit, (name, error)
        yaw_found = engine.analyse(image, faces=[face])[0].quality.yaw
        assert abs(yaw_found - yaw) < (8 if engine.profile.detector == "scrfd_10g_kps" else 15), (name, yaw, yaw_found)


def test_renders_of_one_avatar_score_above_other_avatars(engine, renders):
    keys = sorted(renders)
    records = [engine.analyse(renders[k][0])[0] for k in keys]
    vectors = np.array([r.vector for r in records])
    labels = np.array([name for name, _ in keys])
    scores = vectors @ vectors.T
    same = labels[:, None] == labels[None, :]
    upper = np.triu(np.ones_like(same), 1).astype(bool)
    mated, nonmated = scores[same & upper], scores[~same & upper]
    assert mated.mean() > nonmated.mean() + 0.3, (engine.model_id, mated.mean(), nonmated.mean())
    front = [i for i, (_, yaw) in enumerate(keys) if yaw == 0]  # a frontal gallery: every turned render named
    for i, (name, yaw) in enumerate(keys):
        if yaw != 0:
            best = max(front, key=lambda j: scores[i, j])
            assert keys[best][0] == name, (engine.model_id, name, yaw)
