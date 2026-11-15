"""The head's shape fitted to a person, on the real avatars (D1: renders of one avatar fitted onto another's head).

Skipped without a prepared character library (``BIFOLD_CHARACTERS_TEST_DIR``), the identity space
(``BIFOLD_IDENTITY_SPACE``, ``bifold-characters identity-space``), MediaPipe or OpenGL. MediaPipe's models are
downloaded on first use (``BIFOLD_NO_DOWNLOADS`` stops that).
"""

import os
from pathlib import Path

import numpy as np
import pytest

from bifold_demo.characters.headshape import evaluate as ev
from bifold_demo.characters.headshape import fit as shapefit
from bifold_demo.characters.headshape import head as s0
from bifold_demo.characters.headshape import rig
from bifold_demo.characters.headshape import space as identity

CHARACTERS = Path(os.environ.get("BIFOLD_CHARACTERS_TEST_DIR", "/nonexistent")).expanduser()
PERSON, HEAD = "Male_Adult_04", "Male_Adult_10"  # (a pair of the tuning half)


@pytest.fixture(scope="module")
def library():
    if not CHARACTERS.is_dir():
        pytest.skip("set BIFOLD_CHARACTERS_TEST_DIR to a prepared character library")
    from bifold_demo.characters.library import CharacterLibrary

    library = CharacterLibrary(CHARACTERS)
    if any(name not in library.available() for name in (PERSON, HEAD)):
        pytest.skip(f"the character library lacks {PERSON} or {HEAD}")
    return library


@pytest.fixture(scope="module")
def space():
    if not identity.available():
        pytest.skip("set BIFOLD_IDENTITY_SPACE to a folder with the identity space (bifold-characters identity-space)")
    return identity.load()


def _download() -> bool:
    return os.environ.get("BIFOLD_NO_DOWNLOADS") is None


@pytest.fixture(scope="module")
def mediapipe():
    from bifold_demo.characters.headtex.detect import Landmarker, MediaPipeMissing, Segmenter

    try:
        landmarker = Landmarker(download=_download())
        segmenter = Segmenter(download=_download())
    except MediaPipeMissing as exc:
        pytest.skip(str(exc))
    except OSError as exc:
        pytest.skip(f"MediaPipe's models are unavailable: {exc}")
    yield landmarker, segmenter
    landmarker.close()
    segmenter.close()


@pytest.fixture(scope="module")
def renderer():
    from conftest import _gl_error

    if _gl_error():
        pytest.skip(_gl_error())
    from bifold_demo.scene.renderer import Renderer

    made = Renderer(1024, 768)
    yield made
    made.release()


@pytest.fixture(scope="module")
def heads(library, space, mediapipe, renderer):
    from bifold_demo.characters.headtex import landmarks as stage0_module
    from bifold_demo.characters.headtex.landmarks import head_landmarks, render_views
    from bifold_demo.scene.renderer import Renderer

    landmarker = mediapipe[0]
    stage0 = Renderer(stage0_module.RENDER_SIZE, stage0_module.RENDER_SIZE)

    def render(model, cameras):
        return render_views(model, cameras, stage0)

    out = {}
    for name in (PERSON, HEAD):
        model = library.load(name)
        head = head_landmarks(model, landmarker, render=render)
        out[name] = (model, head, s0.head_space(model, head, space))
    stage0.release()
    return out


def test_the_space_on_an_avatars_head(heads, space):
    """The parts, the alignment and the avatar's coefficients of two real heads (measured: the residual's median
    3.5-7 mm, the coefficients' RMS 1-1.5 standard deviations, 73 vertices per eyeball)."""
    for name in (PERSON, HEAD):
        _model, _head, hs = heads[name]
        assert (hs.eye == 1).sum() == 73 and (hs.eye == 2).sum() == 73
        assert (hs.inner > 0).sum() > 50 and (hs.parts == s0.SKIN).sum() > 1000
        skin = (hs.parts == s0.SKIN) & (hs.triangle >= 0)
        assert skin.sum() > 0.9 * (hs.parts == s0.SKIN).sum()
        assert 0.8 < hs.scale < 1.4 and np.isfinite(hs.alpha).all() and np.sqrt(np.mean(hs.alpha ** 2)) < 2.5
        assert np.median(np.linalg.norm(hs.residual[skin], axis=1)) < 0.008
        modes = hs.modes(space)
        assert np.abs(modes[:, hs.fade == 0]).max() == 0.0  # the neck's edge never moves
        assert (hs.tie >= 0).sum() > 20  # the closed lips are tied
        assert len(hs.bones) >= 20 and 0 < len(hs.others) <= 2


def test_renders_of_one_avatar_fitted_onto_anothers_head(heads, space, mediapipe, renderer):
    """D1: six renders of Male_Adult_04 (front, 30 and 60 degrees each way, nodded) fitted onto Male_Adult_10's
    head come closer to Male_Adult_04 than Male_Adult_10's own head (measured: the landmark median 3.75 -> 2.5 mm,
    the surface median 1.8 -> 1.3 mm), as fitted and as it goes on the rig, and the fitted head passes the rig's
    checks."""
    landmarker, segmenter = mediapipe
    model_a, head_a, _hs_a = heads[PERSON]
    model_b, head_b, hs_b = heads[HEAD]
    person = ev.Person.of(model_a, head_a).photograph(ev.views("three-quarter"), renderer, landmarker, segmenter)
    result = ev.run(person, model_b, head_b, hs_b, space, applied=True)
    fit = result["fit"]
    assert fit.usable, fit.reason
    before = result["before"]["similarity"]
    for after in (result["after"]["similarity"], result["applied"]["similarity"]):
        assert after["landmarks"]["median"] < before["landmarks"]["median"] - 0.5
        assert after["surface"]["median"] < before["surface"]["median"]
    assert "irises" in fit.scale_from or "avatar" in fit.scale_from
    shape = rig.make_shape(model_b, hs_b, fit.offsets, fit)
    fitted = rig.apply_shape(model_b, shape)
    check = rig.rig_checks(model_b, fitted, hs_b)
    assert check.ok, check.problems
    assert len(check.shown) == 2 * 13  # the neutral face and the 12 expressions, from the front and the side


def test_one_photo_keeps_the_avatars_head(heads, space, mediapipe, renderer):
    landmarker, segmenter = mediapipe
    model_a, head_a, _hs_a = heads[PERSON]
    model_b, head_b, hs_b = heads[HEAD]
    person = ev.Person.of(model_a, head_a).photograph(ev.views("front"), renderer, landmarker, segmenter)
    result = ev.run(person, model_b, head_b, hs_b, space)
    assert not result["fit"].usable and "needs 2" in result["fit"].reason and result["after"] is None
    _ = shapefit


def _broken(model, hs, offsets):
    """``model`` with its head moved by ``offsets`` as they are (not through make_shape, which mends some)."""
    from dataclasses import replace

    from bifold_demo.characters.model import CharacterModel

    prim = model.primitives[hs.primitive]
    prims = list(model.primitives)
    prims[hs.primitive] = replace(prim, positions=(np.asarray(prim.positions, np.float64) + offsets)
                                  .astype(np.float32))
    return CharacterModel(model.name, model.skeleton, prims, model.inverse_bind, dict(model.metadata))


def test_the_rig_checks_catch_broken_heads(heads):
    """Each check finds its defect on a real head and puts it down to the right part: an eyeball through its lid,
    lids sunk into it, teeth through the lips, a lip pulled behind the teeth, a lip pulled down so that the teeth
    show where the avatar's mouth is shut, a gap at the neck, a fold. The head as it is passes."""
    model, _head, hs = heads[HEAD]
    p = np.asarray(model.primitives[hs.primitive].positions, np.float64)
    assert rig.rig_checks(model, model, hs).ok
    teeth = np.nonzero((hs.inner == 1) | (hs.inner == 2))[0]
    mouth = np.mean([model.skeleton.rest_global[model.skeleton.index(n), :3, 3]
                     for n in ("Bip01 RMouthCorner", "Bip01 LMouthCorner")], axis=0)
    jaw = s0._avatar_sides(model, hs.primitive)[:, 0] > 0.5
    lower = (hs.parts == s0.SKIN) & jaw & (np.linalg.norm(p - mouth, axis=1) < 0.02) & (p[:, 2] > mouth[2] - 0.005)
    lid, _eyeball = rig._lids(hs, p, 0)
    tri = np.asarray(model.primitives[hs.primitive].indices, np.int64).reshape(-1, 3)
    victim = tri[(hs.parts[tri] == s0.SKIN).all(axis=1)][500]

    def moved(where, by):
        off = np.zeros_like(p)
        off[where] = by
        return off

    sink = np.zeros_like(p)
    sink[lid] = -(p[lid] - hs.eye_centre[0]) / np.linalg.norm(p[lid] - hs.eye_centre[0], axis=1)[:, None] * 0.003
    fold = np.zeros_like(p)
    fold[victim[0]] = (p[victim[1]] - p[victim[0]]) * 2.2
    defects = [("the right eyeball 3 mm forward", moved(hs.eye == 1, [0, 0, 0.003]), "right eye", "through the lid"),
               ("the right lids sunk 3 mm", sink, "right eye", "through the lid"),
               ("the teeth 8 mm forward", moved(teeth, [0, 0, 0.008]), "mouth", "teeth come through the lips"),
               ("the lower lip 3 mm back", moved(lower, [0, 0, -0.003]), "mouth", "lip comes behind the teeth"),
               ("the lower lip 4 mm down", moved(lower, [0, -0.004, 0]), "mouth", "shows more of the teeth"),
               ("the neck's edge 0.5 mm up", moved(hs.fade <= 0.0, [0, 0.0005, 0]), "neck", "gap to the body"),
               ("a fold", fold, "fold", "inside out")]
    for label, offsets, part, words in defects:
        check = rig.rig_checks(model, _broken(model, hs, offsets), hs)
        assert any(words in problem for problem in check.where.get(part, [])), (label, check.where)
    teeth_all, lips = rig._teeth_and_lips(model, hs, p)  # the lips' inside, against the teeth already: hidden
    lip, tooth = rig._lips_before_teeth(p, teeth_all, lips)
    inside = lip[p[lip, 2] - p[tooth, 2] < rig.LIP_FRONT]
    assert len(inside) >= 3
    check = rig.rig_checks(model, _broken(model, hs, moved(inside, [0, 0, -0.0015])), hs)
    assert not any("lip comes behind" in problem for problem in check.problems), check.problems


def test_a_failing_part_is_kept_smaller_there_only(heads, space, tmp_path):
    """A shape that sinks the right lids into the eyeball is kept smaller round the right eye until it passes; the
    rest of the head keeps the full shape, and the shape says what was kept smaller."""
    from bifold_demo.characters.headshape.fitted import put_on

    model, _head, hs = heads[HEAD]
    p = np.asarray(model.primitives[hs.primitive].positions, np.float64)
    rng = np.random.default_rng(1)
    offsets = np.einsum("k,knd->nd", 0.3 * rng.normal(size=len(hs.alpha)), hs.modes(space).astype(np.float64))
    lid, _eyeball = rig._lids(hs, p, 0)
    offsets[lid] -= (p[lid] - hs.eye_centre[0]) / np.linalg.norm(p[lid] - hs.eye_centre[0], axis=1)[:, None] * 0.004
    assert "right eye" in rig.rig_checks(model, rig.apply_shape(model, rig.make_shape(model, hs, offsets)), hs).where
    shape, _fitted, checks, shrink = put_on(model, hs, offsets)
    assert shape is not None and checks.ok and list(shrink) == ["right eye"] and shrink["right eye"] < 1.0
    assert rig.Shape.load(shape.save(tmp_path / "shrunk.npz")).shrink == shrink
    full = rig.make_shape(model, hs, offsets)
    far = (np.linalg.norm(p - hs.eye_centre[0], axis=1) > 0.07) & (hs.parts == s0.SKIN)
    assert np.allclose(shape.offsets[far], full.offsets[far], atol=2e-5)


def test_no_fold_is_left_by_the_transfer(heads, space):
    """Two identities of the space on Male_Adult_10 that left a triangle folded at a mouth corner (one sd per mode):
    no skin triangle is turned inside out on the shape."""
    model, _head, hs = heads[HEAD]
    p = np.asarray(model.primitives[hs.primitive].positions, np.float64)
    tri = np.asarray(model.primitives[hs.primitive].indices, np.int64).reshape(-1, 3)
    skin_tri = tri[(hs.parts[tri] == s0.SKIN).all(axis=1)]
    before = np.cross(p[skin_tri[:, 1]] - p[skin_tri[:, 0]], p[skin_tri[:, 2]] - p[skin_tri[:, 0]])
    modes = hs.modes(space).astype(np.float64)
    for seed in (0, 6):
        beta = np.random.default_rng(seed).normal(0.0, 1.0, len(hs.alpha))
        new = p + rig.make_shape(model, hs, np.einsum("k,knd->nd", beta, modes)).offsets
        after = np.cross(new[skin_tri[:, 1]] - new[skin_tri[:, 0]], new[skin_tri[:, 2]] - new[skin_tri[:, 0]])
        usable = np.linalg.norm(before, axis=1) > 1e-12
        assert not ((np.einsum("tk,tk->t", before, after) <= 0) & usable).any(), seed


def test_the_jaw_line_is_free_of_the_neck(heads):
    """The fade towards the neck's edge goes along the skin and is done by the jaw line: on Male_Adult_10 (a short
    neck: straight from the edge its jaw line was within 5 cm, faded to 0.96) the jaw line moves freely."""
    model, head, hs = heads[HEAD]
    points = np.asarray(model.primitives[hs.primitive].positions, np.float64)
    jaw = np.asarray(head.points, np.float64)[list(s0.JAW_LINE)]
    near = np.argmin(np.linalg.norm(points[None] - jaw[:, None], axis=2), axis=1)
    assert hs.fade[near].min() > 0.999
    assert hs.fade[hs.fade <= 0.0].size > 0  # (the edge itself stays)


def test_the_camera_plays_face_rig_follows_a_new_head(heads, space):
    """A character that wears a fitted head since its camera play began makes its faces with the fitted head's face
    rig (its slides scaled to the new eye distance), not the avatar's."""
    from bifold_demo.characters import camera_play, faces

    model, _head, hs = heads[HEAD]
    offsets = np.einsum("k,knd->nd", 0.5 * np.ones(len(hs.alpha)), hs.modes(space).astype(np.float64))
    fitted = rig.apply_shape(model, rig.make_shape(model, hs, offsets))
    play = type("Play", (), {"posed": lambda self, character, t: None})()
    pose = camera_play.PlayPose(play, model)
    assert pose.rig is faces.face_rig(model)
    character = type("Person", (), {"skinned_model": fitted, "state": None})()
    pose(np.repeat(np.eye(4)[None], len(model.skeleton), axis=0), character, 0.0)
    assert pose.rig is faces.face_rig(fitted) and pose.rig is not faces.face_rig(model)
