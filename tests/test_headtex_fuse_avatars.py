"""Head textures from photos, stages 3 and 4, on the real avatars: render, register, fuse, compare.

Renders of an avatar with its own texture, from several angles under varied light, stand in for a person's photos
("fake photos"): fused back onto the same avatar's head they should give its texture back, and fused onto another
avatar's head they show what a person's photos would. Skipped without a prepared character library
(``BIFOLD_CHARACTERS_TEST_DIR``), MediaPipe or OpenGL. The test with real photos of people also needs
``BIFOLD_FACE_PHOTOS_TEST_DIR``; it writes no image (photos of people and textures made from them stay private).
Nothing is written into the character library: every cache goes to memory or a temporary folder.
"""

import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from bifold_demo.characters.headtex import fuse, light
from bifold_demo.characters.headtex import landmarks as stage0
from bifold_demo.characters.headtex import register as stage1
from bifold_demo.characters.headtex import texture as stages
from bifold_demo.characters.headtex.camera import PhotoCamera

CHARACTERS = Path(os.environ.get("BIFOLD_CHARACTERS_TEST_DIR", "/nonexistent")).expanduser()
PHOTOS = Path(os.environ.get("BIFOLD_FACE_PHOTOS_TEST_DIR", "/nonexistent")).expanduser()
PERSON, OTHER = "Male_Adult_05", "Female_Adult_04"  # a man with a beard; a woman with an open neckline
POSES = ((0.0, 5.0), (40.0, 0.0), (-40.0, 0.0), (0.0, -12.0))  # (yaw, pitch): front, both three-quarters, below
VARIED = ((0.5, (-0.6, -0.5, -0.6), (255, 255, 255)), (0.4, (0.7, -0.4, -0.6), (255, 244, 230)),
          (0.6, (0.0, -0.8, -0.6), (235, 240, 255)), (0.45, (-0.3, -0.3, -0.9), (255, 250, 240)))
PHOTO_SIZE = (1280, 960)


@pytest.fixture(scope="module")
def library():
    if not CHARACTERS.is_dir():
        pytest.skip("set BIFOLD_CHARACTERS_TEST_DIR to a prepared character library")
    from bifold_demo.characters.library import CharacterLibrary

    library = CharacterLibrary(CHARACTERS)
    missing = [name for name in (PERSON, OTHER) if name not in library.available()]
    if missing:
        pytest.skip(f"the character library lacks {', '.join(missing)}")
    return library


@pytest.fixture(scope="module")
def mediapipe():
    from bifold_demo.characters.headtex.detect import Landmarker, MediaPipeMissing, Segmenter

    download = os.environ.get("BIFOLD_NO_DOWNLOADS") is None
    try:
        landmarker = Landmarker(download=download, max_faces=2)
        segmenter = Segmenter(download=download)
    except MediaPipeMissing as exc:
        pytest.skip(str(exc))
    except OSError as exc:
        pytest.skip(f"MediaPipe's models are unavailable: {exc}")
    yield landmarker, segmenter
    landmarker.close()
    segmenter.close()


@pytest.fixture(scope="module")
def renderers():
    from conftest import _gl_error

    if _gl_error():
        pytest.skip(_gl_error())
    from bifold_demo.scene.renderer import Renderer

    made = {"stage0": Renderer(stage0.RENDER_SIZE, stage0.RENDER_SIZE), "photo": Renderer(*PHOTO_SIZE),
            "preview": Renderer(320, 320)}
    yield made
    for renderer in made.values():
        renderer.release()


@pytest.fixture(autouse=True)
def private_cache(tmp_path, monkeypatch):
    monkeypatch.setenv(stage0.CACHE_ENV, str(tmp_path / "headtex"))  # never into the character library
    monkeypatch.setenv("BIFOLD_OUTFIT_CACHE", str(tmp_path / "outfit"))


class Avatar:
    """An avatar's model, landmarks, the segmenter's labels and atlas (made without disk caches)."""

    def __init__(self, library, mediapipe, renderers, name):
        landmarker, segmenter = mediapipe
        self.model = library.load(name)
        render = lambda m, cameras: stage0.render_views(m, cameras, renderers["stage0"])
        self.head = stage0.head_landmarks(self.model, landmarker, cache_dir=False, render=render)
        self.labels = stage1.head_regions(self.model, self.head.primitive, segmenter, cache_dir=False, render=render)
        self.atlas = fuse.head_atlas(self.model, self.head, labels=self.labels, cache_dir=False)


@pytest.fixture(scope="module")
def avatars(library, mediapipe, renderers):
    return {name: Avatar(library, mediapipe, renderers, name) for name in (PERSON, OTHER)}


def fake_photos(model, renderers, lights, poses=POSES):
    """Renders of ``model`` (its own texture) by cameras round its head, each under its own light."""
    from bifold_demo.characters.character import Character
    from bifold_demo.scene import Scene

    frame = stage0.face_frame(model)
    photos, cameras = [], []
    for (yaw, pitch), (ambient, direction, colour) in zip(poses, lights):
        scene = Scene(background=(170, 180, 190), ambient=ambient, light_direction=direction, light_color=colour)
        scene.add(Character(model, name="person", shadow=False))
        scene.update(0.0, 0.0)
        camera = PhotoCamera.around(frame.pivot, 0.7, yaw, pitch, fov=35.0, size=PHOTO_SIZE)
        scene.camera = camera.scene_camera()
        photos.append(renderers["photo"].render(scene))
        cameras.append(camera)
    return photos, cameras


def register(avatar, mediapipe, photos, cameras):
    landmarker, segmenter = mediapipe
    views = stage1.register_photos(photos, avatar.head, avatar.model, landmarker=landmarker, segmenter=segmenter,
                                   focals=[cameras[0].focal] + [None] * (len(photos) - 1))  # (EXIF of the first)
    assert all(v.usable for v in views), [v.summary() for v in views]
    return views


def fused(avatar, views, **options):
    return stages.fuse_views(views, avatar.model, avatar.head, labels=avatar.labels, cache_dir=False, **options)


@pytest.fixture(scope="module")
def person_photos(avatars, renderers):
    return fake_photos(avatars[PERSON].model, renderers, VARIED)


def test_fake_photos_fused_onto_their_own_head_give_its_texture_back(avatars, mediapipe, person_photos):
    """Four renders (front, turned 40 degrees each way, from below) under four lights of different direction and
    colour: on the face texels the photos show surely the texture comes back within a few delta E (measured: the
    median 4.5, 90 % within 9.3, PSNR 24.4 dB; the lightness 1.6-1.9 lower), its skin tone within 3.3."""
    avatar = avatars[PERSON]
    result = fused(avatar, register(avatar, mediapipe, *person_photos))
    atlas = avatar.atlas
    sure = (atlas.parts == fuse.FACE) & (result.trimap == fuse.KEEP)
    assert sure.sum() > 0.5 * (atlas.parts == fuse.FACE).sum()
    got, want = light.lab(light.to_linear(result.rgba[..., :3][sure])), light.lab(atlas.own[sure])
    error = light.delta_e(got, want)
    assert np.median(error) < 6.5 and np.percentile(error, 90) < 13.0
    mse = ((result.rgba[..., :3][sure].astype(float) - light.to_srgb(atlas.own[sure]).astype(float)) ** 2).mean()
    assert 10 * np.log10(255 ** 2 / mse) > 21.0
    assert abs(float(np.mean(got[:, 0] - want[:, 0]))) < 5.0
    assert light.delta_e(result.skin.lab, result.own_skin.lab) < 6.0
    # the eyes stay the avatar's own, the inside of the mouth too (tone-matched where it looks like skin), the
    # islands' gutters are padded
    eyes, inner = atlas.parts == fuse.EYE, atlas.parts == fuse.INNER
    assert np.array_equal(result.rgba[eyes], atlas.own_rgba[eyes])
    assert np.median(light.delta_e(light.lab(light.to_linear(result.rgba[..., :3][inner])),
                                   light.lab(atlas.own[inner]))) < 3.0
    assert (result.trimap[eyes | inner] == fuse.KEEP).all() and (result.rgba[..., 3] == 255).all()
    gutter = ~atlas.covered
    assert result.rgba[..., :3][gutter].mean() > 40  # not the black of the avatar's own texture there
    assert set(result.shares) == {"front", "three-quarter right", "three-quarter left", "front 2", "avatar"}
    assert result.timings["projection"] + result.timings["blend"] < 30.0  # (a few seconds on a free machine)


def test_a_strong_shadow_over_half_the_face_is_mostly_taken_out(avatars, mediapipe, renderers):
    """A light from the person's left with a third of it elsewhere: the right half of the face is far darker in
    the photos (in this renderer's light about eight times, in linear light). De-lit, most of the difference goes
    (measured: 30 of 39 delta L, the halves 8.8 apart; with ambient light 0.55 it is 21 to 2.8)."""
    avatar = avatars[PERSON]
    harsh = [(0.35, (-0.9, -0.3, -0.3), (255, 255, 255))] * len(POSES)
    views = register(avatar, mediapipe, *fake_photos(avatar.model, renderers, harsh))
    atlas = avatar.atlas
    x = np.nan_to_num(atlas.bake.position[..., 0])
    own = light.lab(atlas.own)[..., 0]

    def gap(result):
        sure = (atlas.parts == fuse.FACE) & (result.trimap == fuse.KEEP) & (atlas.features == 0)
        d = light.lab(light.to_linear(result.rgba[..., :3]))[..., 0] - own
        return float(d[sure & (x > 0.01)].mean() - d[sure & (x < -0.01)].mean())

    raw, delit = gap(fused(avatar, views, delight=False)), gap(fused(avatar, views))
    assert raw > 20.0 and delit < 13.0 and delit < 0.45 * raw, (raw, delit)


def test_another_persons_photos_on_an_avatars_head(avatars, mediapipe, person_photos, renderers):
    """The man's photos on the woman's head: his face (the beard, the brows) where hers was, blended into her head;
    the neck tone-matched. Her body keeps her skin until stage 6: the gaps at the neckline and to the hands are
    measured (measured: the seam itself 1.9 -> 3.5 delta E over 7 points, the neck against the body's skin near it
    14.6 -> 8.8, the face against her hands 6.8 -> 7.0)."""
    avatar = avatars[OTHER]
    result = fused(avatar, register(avatar, mediapipe, *person_photos))
    atlas = avatar.atlas
    assert len(result.projections) == 4 and result.coverage()["face"][0] > 0.4
    sure = (atlas.parts == fuse.FACE) & (result.trimap == fuse.KEEP)
    changed = light.delta_e(light.lab(light.to_linear(result.rgba[..., :3][sure])), light.lab(atlas.own[sure]))
    assert np.median(changed) > 6.0  # his face, not hers
    chin = sure & (np.nan_to_num(atlas.bake.position[..., 1]) < stage0.face_frame(avatar.model).mouth[:, 1].mean())
    beard = light.lab(light.to_linear(result.rgba[..., :3][chin]))
    assert np.median(beard[:, 1]) < np.median(light.lab(atlas.own[chin])[:, 1]) - 3.0  # greyer: his grey beard
    seam = stages.neck_seam(avatar.model, result.rgba, atlas)
    assert seam["pairs"] > 0 and seam["seam"] < 10.0 and seam["step"] < 15.0, seam
    gap = stages.skin_gap(result, avatar.model)
    assert 0.0 < gap < 25.0  # (stage 6 closes it)
    views = stages.preview(avatar.model, result, renderer=renderers["preview"])
    own = stages.preview(avatar.model, None, renderer=renderers["preview"])
    assert views.shape == (320, 960, 3) and np.abs(views.astype(int) - own).mean() > 1.0


def test_real_photos_make_a_head_texture(avatars, mediapipe):
    """Real portraits (checks only: nothing is written): each usable one gives a texture with the face on it."""
    if not PHOTOS.is_dir():
        pytest.skip("set BIFOLD_FACE_PHOTOS_TEST_DIR to a folder of portrait photos")
    photos = sorted(p for p in PHOTOS.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not photos:
        pytest.skip(f"no photos in {PHOTOS}")
    landmarker, segmenter = mediapipe
    avatar = avatars[PERSON]
    made = 0
    for path in photos[:4]:
        view = stage1.register_photo(path, avatar.head, avatar.model, landmarker=landmarker, segmenter=segmenter)
        if not view.usable:
            continue
        result = fused(avatar, [view])
        face = avatar.atlas.parts == fuse.FACE
        assert (result.trimap[face] >= fuse.REFINE).mean() > 0.1, path
        assert result.skin is not None and 20.0 < result.skin.lab[0] < 90.0, path
        made += 1
    assert made >= 1


def test_the_command_line_makes_a_texture_and_renders(library, tmp_path, monkeypatch, capsys, renderers):
    from bifold_demo.characters.character import Character
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
    out, views = tmp_path / "head.png", tmp_path / "views.jpg"
    code = main(["headtex", *map(str, paths), "--avatar", PERSON, "-o", str(out), "--views", str(views),
                 "--debug", str(tmp_path / "debug"), "--characters", str(CHARACTERS)])
    printed = capsys.readouterr().out
    assert code == 0 and "skin #" in printed and "timings:" in printed
    assert cv2.imread(str(out), cv2.IMREAD_UNCHANGED).shape == (1024, 1024, 4)
    assert cv2.imread(str(views)).shape[0] == 2 * 480  # with the texture, and the avatar's own
    assert (tmp_path / "debug" / "trimap.png").is_file() and (tmp_path / "debug" / "parts.png").is_file()
    assert main(["headtex", str(paths[0]), "--avatar", "Male_Adult_5", "-o", str(out), "--characters",
                 str(CHARACTERS)]) == 2
    assert main(["headtex", str(tmp_path / "missing.jpg"), "--avatar", PERSON, "-o", str(out), "--characters",
                 str(CHARACTERS)]) == 2
    err = capsys.readouterr().err
    assert "no avatar or clip 'Male_Adult_5'" in err and "no photo" in err and "Traceback" not in err
