"""The evaluation end to end with real avatars and models (skipped without them): a tiny P2 rendered, its faces found
and embedded by the commercial profile (YuNet and SFace: quick), scored, deterministic; every condition drawn as
asked (the scene measures each still's yaw, pitch and eye distance); a face made from renders of another avatar (with
MediaPipe). Needs ``BIFOLD_CHARACTERS_TEST_DIR`` (the prepared avatars), OpenGL and ``BIFOLD_FACE_MODELS``."""

import os
from pathlib import Path

import numpy as np
import pytest

CHARACTERS = Path(os.environ.get("BIFOLD_CHARACTERS_TEST_DIR", "/nonexistent")).expanduser()
PEOPLE = ("Male_Adult_05", "Male_Adult_12", "Female_Adult_01")


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    from conftest import _gl_error

    if not all((CHARACTERS / "avatars" / f"{n}.json").is_file() for n in PEOPLE):
        pytest.skip(f"no prepared characters in {CHARACTERS} (set BIFOLD_CHARACTERS_TEST_DIR)")
    if _gl_error():
        pytest.skip(_gl_error())
    from bifold_demo.recognition.engine import FaceEngine

    try:
        engine = FaceEngine(download=False, profile="commercial", progress=None)
        engine.embedder  # noqa: B018 - (found now: skip without it)
    except (OSError, ImportError) as exc:
        pytest.skip(f"the commercial profile's models are not here: {exc}")
    from bifold_demo.recognition.evalrender import StillRenderer

    renderer = StillRenderer(CHARACTERS, cache=tmp_path_factory.mktemp("faces"), progress=None)
    yield renderer, engine, tmp_path_factory.mktemp("eval")
    renderer.close()


def _manifest(items, folder="."):
    from bifold_demo.recognition import evaluate as E

    return E.parse({"format": 1, "name": "tiny", "settings": {"splits": 3},
                    "items": [{"id": f"i{k}", "kind": "still", "source": "render", **item}
                              for k, item in enumerate(items)]}, folder)


def test_conditions_are_drawn_as_asked(world):
    renderer, _engine, _out = world
    cases = [{"ied": 24.0}, {"ied": 64.0, "yaw": 45.0}, {"ied": 38.0, "yaw": -30.0}, {"ied": 38.0, "pitch": 25.0},
             {"ied": 38.0, "pitch": -25.0}, {"view": "frame", "ied": 14.0}, {"ied": 32.0, "distance": 7.0},
             {"ied": 38.0, "occluder": "low", "light": "side", "preset": "good 4K"}]
    manifest = _manifest([{"role": "probe", "identity": "Male_Adult_05", "conditions": c} for c in cases] +
                         [{"role": "gallery", "identity": "Male_Adult_05", "conditions": {"ied": 100.0}}])
    for item, asked in zip(manifest.stills("probe"), cases):
        got = renderer.still(item)
        truth = got["truth"]
        ied = asked["ied"]
        assert truth["ied"] == pytest.approx(ied, rel=0.002 if got["where"] == "window" else 0.03), asked
        assert truth["yaw"] == pytest.approx(asked.get("yaw", 0.0), abs=3.0), asked
        assert truth["pitch"] == pytest.approx(asked.get("pitch", 0.0), abs=4.0), asked
        if asked.get("distance"):
            assert truth["distance"] == pytest.approx(asked["distance"], rel=0.05)
        if got["where"] == "window":
            side = got["image"].shape[0]
            assert 2.5 * ied < side < 12 * ied  # (the head's window: a square about half a metre wide)
        else:
            assert got["image"].shape[:2] == (720, 1280) and got["region"] is not None


def test_occluders_cover_what_they_say(world):
    """The board covers the lower face and leaves the eyes; the pillar covers one side of the face."""
    renderer, *_ = world
    cases = [{"ied": 40.0}, {"ied": 40.0, "occluder": "low"}, {"ied": 40.0, "occluder": "side"}]
    manifest = _manifest([{"role": "probe", "identity": "Male_Adult_12", "conditions": c} for c in cases] +
                         [{"role": "gallery", "identity": "Male_Adult_12", "conditions": {"ied": 100.0}}])
    plain, low, side = (renderer.still(item) for item in manifest.stills("probe"))
    image = plain["image"].astype(float)
    h, w = image.shape[:2]
    cy, cx = h // 2, w // 2
    eyes = (slice(cy - 8, cy + 4), slice(cx - 20, cx + 20))
    chin = (slice(cy + 28, cy + 48), slice(cx - 12, cx + 12))
    assert np.abs(low["image"].astype(float)[eyes] - image[eyes]).mean() < 4
    assert np.abs(low["image"].astype(float)[chin] - image[chin]).mean() > 20
    halves = [(slice(cy - 30, cy + 40), slice(cx - 35, cx - 5)), (slice(cy - 30, cy + 40), slice(cx + 5, cx + 35))]
    changed = [np.abs(side["image"].astype(float)[r] - image[r]).mean() for r in halves]
    assert max(changed) > 20 and min(changed) < 6


def test_the_lights_light_the_face(world):
    """The side light leaves one side of the face darker than the fair's does; the warm light makes it redder."""
    renderer, *_ = world
    cases = [{"ied": 40.0}, {"ied": 40.0, "light": "side"}, {"ied": 40.0, "light": "warm"}]
    manifest = _manifest([{"role": "probe", "identity": "Male_Adult_05", "conditions": c} for c in cases] +
                         [{"role": "gallery", "identity": "Male_Adult_05", "conditions": {"ied": 100.0}}])
    fair, side, warm = (renderer.still(item)["image"].astype(float) for item in manifest.stills("probe"))
    h, w = fair.shape[:2]
    cy, cx = h // 2, w // 2
    left, right = (slice(cy - 10, cy + 30), slice(cx - 30, cx - 8)), (slice(cy - 10, cy + 30), slice(cx + 8, cx + 30))

    def contrast(image):
        a, b = image[left].mean(), image[right].mean()
        return max(a, b) / max(1.0, min(a, b))

    assert contrast(side) > contrast(fair) + 0.25
    face = (slice(cy - 10, cy + 30), slice(cx - 25, cx + 25))
    assert warm[face][..., 2].mean() / warm[face][..., 0].mean() > 1.1 * fair[face][..., 2].mean() / \
        fair[face][..., 0].mean()


def test_a_still_is_drawn_the_same_twice(world):
    renderer, *_ = world
    item = _manifest([{"role": "probe", "identity": "Male_Adult_12",
                       "conditions": {"ied": 30.0, "preset": "budget 1080p", "light": "warm", "seed": 4}},
                      {"role": "gallery", "identity": "Male_Adult_12", "conditions": {"ied": 100.0}}]).items[0]
    a, b = renderer.still(item)["image"], renderer.still(item)["image"]
    assert np.array_equal(a, b)


def test_a_tiny_p2_end_to_end(world):
    """Three avatars, their gallery (set B) and probes of their own faces, scored by the commercial profile; run again
    from the caches with the same results."""
    from bifold_demo.recognition import evaluate as E
    from bifold_demo.recognition import protocols

    renderer, engine, out = world
    raw = protocols.p2(PEOPLE, "tiny", variants=("own",))
    manifest = E.parse(raw, ".")
    assert manifest.sides["evaluation"] == sorted(PEOPLE)

    def measured():
        run = E.Run({"commercial": engine}, out, library=str(CHARACTERS), renderer=lambda: renderer, progress=None,
                    draws=100)
        report = run.stills(manifest)["commercial"]
        return {k: v for k, v in report.items() if k != "seconds"}

    first = measured()
    assert first["counts"]["probes"] == 12 and first["counts"]["people"] == 3 and first["counts"]["found"] == 12
    assert first["open_set"]["rank1"] == 1.0  # (frontal and modestly turned faces of three different avatars)
    at = first["system"]["targets"]["0.01"]
    assert at["known"]["named_wrong"]["rate"] == 0.0 and at["strangers"]["named"]["rate"] == 0.0
    turned = first["conditions"]["turn"]["variant=own, turn=45.0"]
    assert 40.0 < abs(turned["drawn"]["yaw"]) < 50.0  # (to the left for some, the right for others)
    assert measured() == first  # (deterministic: the same numbers from the caches)


def test_a_tiny_p3_end_to_end(world):
    """A short parade (one known person, enrolled from set B, and one stranger, both with their own faces; people
    found from the truth) through the whole pipeline, twice: the same tracks, decisions and scores."""
    from bifold_demo.recognition import evalscene
    from bifold_demo.recognition import evaluate as E

    renderer, engine, out = world
    raw = {"format": 1, "name": "p3-tiny", "items": [{
        "id": "run", "kind": "scene", "scene": "parade", "seconds": 4.0, "fps": 5.0, "seed": 3,
        "known": ["Male_Adult_05"], "detector": "truth",
        "people": [{"identity": "Male_Adult_05", "role": "lane"}, {"identity": "Male_Adult_12", "role": "lane"}]}]}
    manifest = E.parse(raw, ".")
    run = E.Run({"commercial": engine}, out, library=str(CHARACTERS), renderer=lambda: renderer, progress=None,
                draws=50)
    first = evalscene.run_scenes(run, manifest)["commercial"]
    again = evalscene.run_scenes(run, manifest)["commercial"]
    a, b = first["runs"][0], again["runs"][0]
    assert a["frames"] == 20 and a["known"] == ["Male_Adult_05"]
    assert a["segments"] == b["segments"] and a["timeline"] == b["timeline"]
    people = {g["person"]: g for g in a["segments"]}
    assert people and all(g["avatar"] in ("Male_Adult_05", "Male_Adult_12") for g in a["segments"])
    assert not any(g["wrong"] for g in a["segments"])  # (no wrong name)
    assert all(g["named_at"] is None for g in a["segments"] if not g["known"])  # (no stranger named)
    score = first["score"]
    assert score["runs"] == 1 and score["acceptance"]["verdict"] in ("pass", "fail")
    assert score["replay_agrees"]["agree"] == score["replay_agrees"]["segments"]


def test_a_face_made_from_renders_of_another_avatar(world):
    """Male_Adult_05's face (textures made from a render of him) on Male_Adult_12: it is still Male_Adult_05 to the
    recogniser, more than it is Male_Adult_12."""
    pytest.importorskip("mediapipe")
    renderer, engine, _out = world
    item = _manifest([{"role": "probe", "identity": "Male_Adult_05",
                       "conditions": {"ied": 64.0, "face": "renders", "body": "Male_Adult_12", "shots": "fronts"}},
                      {"role": "gallery", "identity": "Male_Adult_05", "conditions": {"ied": 100.0}}]).items[0]
    try:
        got = renderer.still(item)
    except (ImportError, OSError) as exc:  # MediaPipe's models missing
        pytest.skip(f"no face textures from photos here: {exc}")
    own = [_vector(engine, renderer.still(_manifest([{"role": "probe", "identity": n, "conditions": {"ied": 64.0}},
                                                     {"role": "gallery", "identity": n, "conditions": {"ied": 100.0}}]
                                                    ).items[0])["image"]) for n in ("Male_Adult_05", "Male_Adult_12")]
    worn = _vector(engine, got["image"])
    assert worn @ own[0] > worn @ own[1] + 0.05


def _vector(engine, image):
    from bifold_demo.recognition.evaluate import acquire

    found = acquire(image, engine, "window")
    assert found["found"]
    return np.asarray(engine.embedder.embed(found["crop"][None]).vectors[0], np.float64)
