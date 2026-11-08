"""Head textures from photos, stages 5 to 7, on the real avatars: another person's fake photos, all the way.

Renders of Male_Adult_05 stand in for a person's photos (as in ``test_headtex_fuse_avatars``) and are finished onto
Female_Adult_04, whose open neckline shows the neck seam: stage 6 must bring the neck and the hands closer to the face
than stage 4 left them. Skipped without a prepared character library (``BIFOLD_CHARACTERS_TEST_DIR``), MediaPipe or
OpenGL; the identity check also needs the face recognition's models (``BIFOLD_FACE_MODELS``). Nothing is written into
the character library.
"""

import cv2
import numpy as np
import test_headtex_fuse_avatars as fuse_tests
from test_headtex_fuse_avatars import CHARACTERS, OTHER, PERSON, PHOTO_SIZE, fused, register

from bifold_demo.characters.headtex import fuse
from bifold_demo.characters.headtex import landmarks as stage0
from bifold_demo.characters.headtex import person as stages
from bifold_demo.characters.headtex import texture as stage4
from bifold_demo.characters.headtex.camera import PhotoCamera

# the fixtures of the stages 3 and 4 tests, used here too
library, mediapipe, renderers = fuse_tests.library, fuse_tests.mediapipe, fuse_tests.renderers
private_cache, avatars, person_photos = fuse_tests.private_cache, fuse_tests.avatars, fuse_tests.person_photos


def _engine():
    """The face recognition, when its models are there (never downloaded here)."""
    try:
        from bifold_demo.recognition.engine import FaceEngine

        return FaceEngine(download=False)
    except Exception:  # noqa: BLE001 - no models: no identity check
        return None


def test_another_persons_photos_finished_onto_an_avatar(avatars, mediapipe, person_photos, renderers, tmp_path):
    """His four photos on her head, all stages: the neck seam, the neck against her body's skin and the face against
    her hands all closer after stage 6 than before (measured: 3.5 -> 1.4, 9.5 -> 2.3, 7.0 -> 0.2 delta E); the
    identity check says the renders look more like him than her own head does; the textures make a created
    character (measured: SFace 0.67 against 0.23 for her own head, from the front)."""
    avatar = avatars[OTHER]
    photos = person_photos[0]
    result = fused(avatar, register(avatar, mediapipe, *person_photos))
    engine = _engine()
    person = stages.finish(result, avatar.model, check=engine is not None, engine=engine, photos=photos,
                           renderer=renderers["preview"])
    checks = person.checks
    assert checks["seam pairs"] > 0 and checks["neck seam after"] < checks["neck seam before"], checks
    assert checks["neck step after"] < checks["neck step before"], checks
    assert checks["face to hands after"] < 2.0 < checks["face to hands before"], checks
    body = person.match.body_primitive
    assert set(person.textures) >= {result.primitive, body} and person.match.hair  # her hair cards recoloured
    if engine is not None:
        mine, own = checks["identity"]["front"], checks["identity of the avatar's own head"]["front"]
        assert mine > own + 0.1, (mine, own)
    created = stages.PersonTextures.load(person.save(tmp_path / "saved")).created("Him at the fair")
    assert created.avatar == OTHER and set(created.textures) == set(person.textures)
    views = stage4.preview(avatar.model, person.textures, renderer=renderers["preview"])
    assert np.abs(views.astype(int) - stage4.preview(avatar.model, None, renderer=renderers["preview"])).mean() > 1.0


def test_the_command_line_finishes_and_saves_a_created_character(library, tmp_path, capsys, renderers):
    from bifold_demo.characters.character import Character
    from bifold_demo.characters.created import load
    from bifold_demo.recognition.cli import main
    from bifold_demo.scene import Scene

    model = library.load(PERSON)
    frame = stage0.face_frame(model)
    paths = []
    for k, yaw in enumerate((0.0, 35.0)):
        scene = Scene(background=(170, 180, 190), ambient=0.55, light_direction=(-0.4, -0.6, -0.7))
        scene.add(Character(model, name="person", shadow=False))
        scene.update(0.0, 0.0)
        scene.camera = PhotoCamera.around(frame.pivot, 0.7, yaw, fov=35.0, size=PHOTO_SIZE).scene_camera()
        paths.append(tmp_path / f"photo{k}.jpg")
        cv2.imwrite(str(paths[-1]), renderers["photo"].render(scene))
    out = tmp_path / "head.png"
    code = main(["headtex", *map(str, paths), "--avatar", OTHER, "-o", str(out), "--full", "--views",
                 str(tmp_path / "views.jpg"), "--character", "Test person", "--created", str(tmp_path / "created"),
                 "--characters", str(CHARACTERS)])
    printed = capsys.readouterr().out
    assert code == 0 and "stage 6: the body's skin times" in printed and "the face's skin against the hands'" in printed
    other = library.load(OTHER)
    side = fuse.texture_size(other.primitives[stage0.face_photo.head_layout(other).primitive])  # 2048 or 1024
    assert cv2.imread(str(out), cv2.IMREAD_UNCHANGED).shape == (side, side, 4)
    assert cv2.imread(str(tmp_path / "head_body.png"), cv2.IMREAD_UNCHANGED).shape[2] == 4
    assert cv2.imread(str(tmp_path / "views.jpg")).shape[0] == 2 * 480
    created = load(tmp_path / "created" / "Test person")
    assert created.avatar == OTHER and len(created.textures) >= 2
