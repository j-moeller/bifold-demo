"""Person, face, identity with the real models and avatars, end to end in the fair (skipped without them): the
prepared avatars (``BIFOLD_CHARACTERS_TEST_DIR``), OpenGL and the face models (``BIFOLD_FACE_MODELS``; nothing is
downloaded). The known people are avatars enrolled from renders of themselves; the person detector is simulated from
the scene's truth (boxes jittered, weak when half hidden), so that YOLO's weights are not needed."""

import os
from pathlib import Path

import numpy as np
import pytest

CHARACTERS = Path(os.environ.get("BIFOLD_CHARACTERS_TEST_DIR", "/nonexistent")).expanduser()
VIEW = (640, 360)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """The fair (its library's avatars), the avatars of its audience (who always face the camera) enrolled from
    renders of themselves, and the commercial profile's engine (YuNet and SFace: quick)."""
    from conftest import _gl_error

    if not (CHARACTERS / "avatars").is_dir():
        pytest.skip(f"no prepared characters in {CHARACTERS} (set BIFOLD_CHARACTERS_TEST_DIR)")
    if _gl_error():
        pytest.skip(_gl_error())
    from bifold_demo.characters.library import CharacterLibrary
    from bifold_demo.recognition.calibrate import gallery_photos
    from bifold_demo.recognition.engine import FaceEngine

    try:
        engine = FaceEngine(download=False, profile="commercial", progress=None)
        engine.embedder  # noqa: B018 - (found now: skip without it)
    except (OSError, ImportError) as exc:
        pytest.skip(f"the commercial profile's models are not here: {exc}")
    library = CharacterLibrary(CHARACTERS)
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("BIFOLD_CHARACTERS", str(CHARACTERS))
        if hasattr(CharacterLibrary, "shared"):
            patch.setattr(CharacterLibrary, "shared", lambda: library)
        from bifold_demo.scenes import fair

        scene = fair.build_scene()
    audience = {obj.name: obj.skinned_model.name for obj in scene
                if obj.name.startswith("audience") and getattr(obj, "skinned_model", None) is not None}
    if len(audience) < 3:
        pytest.skip("the fair has no 3D audience")
    folder = tmp_path_factory.mktemp("faces")
    gallery_photos(library, sorted(set(audience.values())), folder)
    return scene, engine, folder, audience


class _TruthDetector:
    """A person detector simulated from the scene's truth (see the module docs)."""

    def __init__(self, scene, seed=3):
        self.scene, self.rng = scene, np.random.default_rng(seed)

    def detect(self, image, conf=None):
        from bifold_demo.detection.results import Detection
        from bifold_demo.recognition.truth import frame_truth, simulated_detections

        truth = [p for p in frame_truth(self.scene, VIEW) if p.box[3] - p.box[1] >= 20]
        boxes, scores = simulated_detections(truth, self.rng, jitter=0.01, size=VIEW)
        return [Detection(tuple(b), float(s), 0, "person") for b, s in zip(boxes, scores)]


def _run(world, seconds=4.0, fps=10.0, windows=True):
    from bifold_demo.recognition.gallery import Gallery
    from bifold_demo.recognition.people import PeopleRecognizer
    from bifold_demo.recognition.truth import Outcomes, frame_truth, match_tracks
    from bifold_demo.sensor import WindowService
    from bifold_demo.sources import FixedStepClock, SceneSource

    scene, engine, folder, audience = world
    gallery = Gallery(folder, engine)
    gallery.refresh(force=True)
    service = WindowService(max_windows=4) if windows else None
    source = SceneSource(scene, *VIEW, fps=fps, clock=FixedStepClock(fps), windows=service)
    source.reset()
    recognizer = PeopleRecognizer(gallery, _TruthDetector(scene), windows=service)
    outcomes = Outcomes(set(audience.values()), min_eyes=2.0)
    results = []
    try:
        for _ in range(int(seconds * fps)):
            frame = source.read()
            result = recognizer.process(frame)
            truth = frame_truth(scene, VIEW)
            outcomes.update(result.time, truth, match_tracks(truth, result.people),
                            {p.id: p.decision for p in result.people})
            results.append(result)
    finally:
        source.stop()
    return results, outcomes, recognizer


def test_the_fair_end_to_end_names_the_known_people_and_nobody_else(world):
    """Four seconds of the fair: the audience (known) are named by their own names from the windows' faces; nobody
    is named wrongly; the run is deterministic."""
    results, outcomes, recognizer = _run(world)
    _scene, _engine, _folder, audience = world
    named = {g.person: g.correct_at for g in outcomes.segments.values() if g.correct_at is not None}
    assert len(set(named) & set(audience)) >= 2, (named, outcomes.summary(shown=0.0))
    assert not any(g.wrong for g in outcomes.segments.values()), [g for g in outcomes.segments.values() if g.wrong]
    sources = {f.source for info in recognizer.tracks.values() for f in info.evidence.frames}
    assert "window" in sources  # the tracks asked for windows, and their faces came from them
    assert all(r.rate <= recognizer.budget.rate + 1.0 for r in results)
    # the same frames give the same decisions
    again, _, _ = _run(world)
    first = [[(p.id, p.state, p.name, round(p.decision.raw or 0.0, 6)) for p in r.people] for r in results]
    second = [[(p.id, p.state, p.name, round(p.decision.raw or 0.0, 6)) for p in r.people] for r in again]
    assert first == second
