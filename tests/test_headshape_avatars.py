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
    head come closer to Male_Adult_04 than Male_Adult_10's own head (measured: the landmark median 3.75 -> 2.6 mm,
    the surface median 1.8 -> 1.3 mm), and the fitted head passes the rig's checks."""
    landmarker, segmenter = mediapipe
    model_a, head_a, _hs_a = heads[PERSON]
    model_b, head_b, hs_b = heads[HEAD]
    person = ev.Person.of(model_a, head_a).photograph(ev.views("three-quarter"), renderer, landmarker, segmenter)
    result = ev.run(person, model_b, head_b, hs_b, space)
    fit = result["fit"]
    assert fit.usable, fit.reason
    before, after = result["before"]["similarity"], result["after"]["similarity"]
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
