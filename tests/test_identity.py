"""Person, face, identity (step F2) with stand-ins, runnable anywhere: faces in head regions and whose they are, the
evidence per track, calibration, open-set decisions and the pipeline. (The real models and avatars:
``test_identity_real.py``.)"""

import json
import math
import threading
from pathlib import Path

import cv2
import numpy as np
import pytest

from bifold_demo.recognition.calibration import (
    Calibration,
    Cohort,
    Curve,
    calibration_path,
    find_calibration,
    per_person,
)
from bifold_demo.recognition.decision import (
    NAMED,
    UNDECIDED,
    UNKNOWN,
    DecisionLog,
    DecisionSettings,
    IdentityPolicy,
)
from bifold_demo.recognition.detect import Face
from bifold_demo.recognition.evidence import (
    Aggregate,
    Candidate,
    EmbeddingBudget,
    EmbeddingScheduler,
    FrameEvidence,
    QualityGate,
    TrackEvidence,
    rate_for,
)
from bifold_demo.recognition.gallery import RawScores
from bifold_demo.recognition.heads import BOX_PER_EYES, FaceSighting, HeadModel, assign_faces, face_size, find_faces
from bifold_demo.recognition.quality import Quality
from bifold_demo.recognition.truth import Outcomes, PersonTruth


def _face(middle, eyes, score=0.9):
    """A frontal face with its eyes ``eyes`` pixels apart around ``middle``."""
    x, y = middle
    height = BOX_PER_EYES * eyes
    box = (x - 0.45 * height, y - 0.41 * height, 0.9 * height, height)
    marks = ((x - eyes / 2, y), (x + eyes / 2, y), (x, y + 0.5 * eyes), (x - 0.4 * eyes, y + eyes),
             (x + 0.4 * eyes, y + eyes))
    return Face(box, marks, score)


HEAD = HeadModel()


def _head_face(box, head=HEAD, dx=0.0, dy=0.0, scale=1.0):
    """The face of a person in ``box``, where the head model expects it (moved by ``dx``, ``dy`` eye distances)."""
    (x, y), eyes = head.expected(box)
    return _face((x + dx * eyes, y + dy * eyes), eyes * scale)


def _sighting(face, source="frame"):
    return FaceSighting(face, face, np.zeros((4, 4, 3), np.uint8), source)


# -- head regions and whose face a face is ----------------------------------------------------------------------------


def test_a_face_fits_the_head_it_belongs_to():
    head = HeadModel()
    box = (100.0, 100.0, 180.0, 400.0)
    face = _head_face(box)
    assert head.cost(face, box) == pytest.approx(0.0, abs=1e-6)
    assert face_size(face) == pytest.approx(head.expected(box)[1])
    assert 0 < head.cost(_head_face(box, dx=1.0), box) < head.cost(_head_face(box, dx=2.0), box)
    assert head.cost(_head_face(box, dy=1.0), box) > head.cost(_head_face(box, dx=1.0), box)  # rows are tighter
    assert head.cost(_head_face(box, scale=0.5), box) > 2.0
    assert head.cost(_head_face(box, dy=6.0), box) > head.limit  # (on the neck: in the region, far too low)
    assert head.cost(_head_face(box, dy=8.0), box) == math.inf  # (on the chest: outside the head region)
    assert head.cost(_face((500, 500), 11.0), box) == math.inf


def test_overlapping_head_regions_give_each_face_to_its_own_person():
    """B stands behind A, a little to the side: B's face lies in A's head region too."""
    head = HeadModel()
    a = (100.0, 120.0, 200.0, 420.0)  # nearer, bigger
    b = (150.0, 100.0, 220.0, 310.0)
    face_a, face_b = _head_face(a), _head_face(b)
    x0, y0, x1, y1 = head.square(a)
    middle = face_b.points[:2].mean(axis=0)
    assert x0 <= middle[0] <= x1 and y0 <= middle[1] <= y1  # in A's region as well as B's
    given = assign_faces([_sighting(face_b), _sighting(face_a)], {7: a, 9: b}, head)
    assert [s.track for s in given] == [9, 7]
    # with A's own face turned away (not found), B's face still does not become A's
    given = assign_faces([_sighting(face_b)], {7: a, 9: b}, head)
    assert [s.track for s in given] == [9]
    # and without B's track (B not detected), B's face fits A too badly to be given to them
    given = assign_faces([_sighting(face_b)], {7: a}, head)
    assert [s.track for s in given] == [None]


def test_each_track_gets_at_most_one_face_and_each_face_one_track():
    head = HeadModel()
    box = (100.0, 100.0, 180.0, 400.0)
    near, off = _head_face(box), _head_face(box, dx=1.5)
    given = assign_faces([_sighting(off), _sighting(near)], {3: box}, head)
    assert [s.track for s in given] == [None, 3]  # the better fit wins
    assert given[1].cost == pytest.approx(head.cost(near, box))
    assert assign_faces([], {3: box}) == [] and [s.track for s in assign_faces([_sighting(near)], {})] == [None]


class _Window:
    """A stand-in :class:`~bifold_demo.sensor.WindowResult`: frame pixels ``p`` are ``(p - offset) * scale`` here."""

    def __init__(self, image, offset, scale, tag=1):
        self.image, self.offset, self.tag = image, np.asarray(offset, float), scale
        self.window = type("W", (), {"scale": scale})()
        self.tag = tag

    def to_view(self, points):
        return np.asarray(points, float) / self.window.scale + self.offset


class _Engine:
    """Finds the faces it was given (frame pixels) in the frame, a region of it, or a window's image."""

    def __init__(self, faces, frame):
        self.faces, self.frame, self.windows, self.calls = faces, frame, {}, []

    def detect(self, image, min_size=0.0, canvas=None, region=None):
        self.calls.append((id(image), region, canvas))
        if image is self.frame:
            offset, scale = np.zeros(2), 1.0
        else:
            offset, scale = self.windows[id(image)]
        found = []
        for face in self.faces:
            if region is not None:
                x, y, w, h = region
                middle = face.points[:2].mean(axis=0)
                if not (x <= middle[0] <= x + w and y <= middle[1] <= y + h):
                    continue
            box = ((face.box[0] - offset[0]) * scale, (face.box[1] - offset[1]) * scale, face.box[2] * scale,
                   face.box[3] * scale)
            marks = tuple(tuple((np.asarray(p) - offset) * scale) for p in face.landmarks)
            local = Face(box, marks, face.score)
            inside = image is self.frame or (0 <= box[0] and box[0] + box[2] <= image.shape[1] and 0 <= box[1]
                                             and box[1] + box[3] <= image.shape[0])
            if inside and local.size >= min_size:
                found.append(local)
        return found


def test_faces_are_found_in_windows_and_crops_and_mapped_to_the_frame():
    frame = np.zeros((720, 1280, 3), np.uint8)
    head = HeadModel()
    a, b = (100.0, 100.0, 180.0, 400.0), (600.0, 200.0, 650.0, 380.0)
    face_a, face_b = _head_face(a), _head_face(b)
    engine = _Engine([face_a, face_b], frame)
    x0, y0, x1, y1 = head.square(b)
    picture = np.zeros((round((y1 - y0) * 6), round((x1 - x0) * 6), 3), np.uint8)
    window = _Window(picture, (x0, y0), 6.0)
    engine.windows[id(picture)] = (window.offset, 6.0)
    found = find_faces(engine, frame, {1: head.square(a), 2: head.square(b)}, [window])
    assert [s.source for s in found] == ["window", "frame"]  # b's face found twice: the window's is kept
    window_face, frame_face = found
    assert np.allclose(window_face.face.points, face_b.points) and np.allclose(frame_face.face.points, face_a.points)
    assert window_face.image is picture and window_face.scale == 6.0
    assert np.allclose(window_face.local.points, (face_b.points - window.offset) * 6.0)  # aligned from the window
    assert window_face.size == pytest.approx(6.0 * face_size(face_b))
    given = assign_faces(found, {1: a, 2: b}, head)
    assert [s.track for s in given] == [2, 1]
    # a region reaching out of the frame is cut to it; one with no pixels left is not looked at
    engine.calls.clear()
    find_faces(engine, frame, {1: (-50.0, -50.0, 40.0, 40.0), 2: (1300.0, 0.0, 1400.0, 50.0)})
    assert [call[1] for call in engine.calls] == [(0.0, 0.0, 40.0, 40.0)]


# -- the evidence -----------------------------------------------------------------------------------------------------

def _quality(ied=30.0, yaw=0.0, sharpness=1.0, outside=0.0, detector=0.9, fiqa=None):
    return Quality(ied, yaw, 0.0, 0.0, sharpness, 120.0, 40.0, 0.0, outside, detector, fiqa)


def test_the_gate_lets_through_only_faces_worth_embedding():
    gate = QualityGate()
    assert gate.check(_quality(), 30.0) is None
    assert gate.check(_quality(fiqa=0.6), 30.0) is None
    assert gate.check(_quality(), 6.0) == "small"
    assert gate.check(_quality(yaw=-70.0), 30.0) == "turned"
    assert gate.check(_quality(sharpness=0.02), 30.0) == "blurred"
    assert gate.check(_quality(outside=0.5), 30.0) == "cut off"
    assert gate.check(_quality(detector=0.3), 30.0) == "unsure"
    assert gate.check(_quality(fiqa=0.2), 30.0) == "hard"
    assert rate_for("adaface_ir101_wf12m") < rate_for("buffalo_l_r50") < rate_for("sface_2021dec")


def test_the_budget_allows_its_rate_by_the_frames_clock():
    budget = EmbeddingBudget(1.0)
    made = 0
    for k in range(300):  # ten seconds at 30 frames a second
        if budget.available(k / 30) >= 1:
            budget.take()
            made += 1
    assert made == 10  # one at once, then one a second
    assert budget.available(9.99) == 0 and budget.spent == 10
    fast = EmbeddingBudget(20.0)
    assert fast.available(0.0) == 1 and fast.available(10.0) == 20  # never more than a second's worth at once
    assert EmbeddingBudget(0.5).available(1.0) == 1 and EmbeddingBudget(0.5).burst == 1.0
    with pytest.raises(ValueError):
        EmbeddingBudget(0.0)


def test_undecided_tracks_and_better_faces_come_first():
    s = EmbeddingScheduler(recheck=5.0, per_track=2.0, median_after=4)
    faces = [Candidate(1, 0.9, "a", decided=True, last=0.0), Candidate(2, 0.3, "b"), Candidate(3, 0.6, "c"),
             Candidate(3, 0.8, "c2"), Candidate(4, 0.95, "d", decided=True, last=9.0)]
    chosen = s.choose(faces, 10, now=10.0)
    assert [c.key for c in chosen] == ["c2", "b", "a", "d"]  # undecided first (the better of track 3's two), then
    # the decided track due for a check, then the one checked a second ago (with what the budget leaves)
    assert [c.key for c in s.choose(faces, 1, now=10.0)] == ["c2"]
    # faces about as good take turns: the one waiting longest first
    turns = [Candidate(6, 0.4501, last=9.0), Candidate(7, 0.45, last=8.0)]
    assert [c.track for c in s.choose(turns, 1, now=10.0)] == [7]
    assert s.choose([Candidate(5, 0.9, last=9.8)], 1, now=10.0) == []  # at most twice a second
    assert s.choose([Candidate(5, 0.4, median=0.5)], 1, now=10.0) == []  # worse than its median: not worth it
    assert s.choose([Candidate(5, 0.6, median=0.5)], 1, now=10.0)
    assert s.choose([Candidate(5, 0.5, median=0.5)], 1, now=10.0)  # (as good: a still face keeps its turn)


def _frame(vector, weight=1.0, t=0.0, size=30.0, yaw=0.0, model="m"):
    v = np.asarray(vector, float)
    return FrameEvidence(v / np.linalg.norm(v), weight, t, size, yaw, model)


def test_a_tracks_evidence_is_the_quality_weighted_mean_of_its_frames():
    evidence = TrackEvidence("m", keep=3, half_life=10.0)
    assert evidence.aggregate() is None
    evidence.add(_frame([1, 0, 0], 1.0, 0.0, size=40.0, yaw=10.0))
    evidence.add(_frame([0, 1, 0], 0.25, 0.0, size=10.0, yaw=-30.0))
    agg = evidence.aggregate(0.0)
    expected = np.array([1.0, 0.25, 0.0])
    assert np.allclose(agg.vector, expected / np.linalg.norm(expected))
    assert agg.weight == pytest.approx(1.25) and agg.frames == 2 and agg.best == 1.0
    assert agg.size == pytest.approx((40 + 0.25 * 10) / 1.25) and agg.yaw == pytest.approx((10 + 0.25 * 30) / 1.25)
    # time decay: ten seconds on, the first frame counts half
    assert evidence.aggregate(10.0).weight == pytest.approx(0.625)
    # the same frame many times is no more evidence of who it is: the mean's direction does not change
    same = TrackEvidence("m")
    for t in range(5):
        same.add(_frame([1, 1, 0], 0.5, float(t)))
    assert np.allclose(same.aggregate().vector, [2 ** -0.5, 2 ** -0.5, 0])
    with pytest.raises(ValueError):
        evidence.add(_frame([1, 0, 0], model="other"))


def test_a_track_keeps_its_best_frames_and_can_start_over():
    evidence = TrackEvidence("m", keep=3, half_life=100.0)
    for t, w in enumerate([0.2, 0.9, 0.1, 0.8, 0.7]):
        evidence.add(_frame([1, t, 0], w, float(t)))
    assert sorted(f.weight for f in evidence.frames) == [0.7, 0.8, 0.9]
    assert [f.time for f in evidence.frames] == sorted(f.time for f in evidence.frames)
    assert evidence.median_weight(after=3) == pytest.approx(0.8) and evidence.median_weight(after=4) is None
    left = evidence.frames[0]
    assert np.allclose(evidence.aggregate(without=left).vector, evidence.aggregate(4.0, without=left).vector)
    evidence.reset(10.0, keep_after=4.0)
    assert [f.time for f in evidence.frames] == [4.0] and evidence.started == 4.0
    evidence.reset(12.0)
    assert evidence.frames == [] and evidence.started == 12.0


# -- calibration ------------------------------------------------------------------------------------------------------


def test_a_curve_gives_the_false_match_rate_asked_for():
    rng = np.random.default_rng(0)
    nonmated = rng.normal(0.0, 0.1, 50000)
    curve = Curve.fit(nonmated, rng.normal(0.6, 0.1, 500))
    fresh = rng.normal(0.0, 0.1, 400000)
    for fmr in (0.1, 0.01, 1e-3):
        t = curve.threshold(fmr)
        assert (fresh >= t).mean() == pytest.approx(fmr, rel=0.15)
        assert curve.rarity(t) == pytest.approx(-math.log10(fmr), abs=1e-6)
    assert curve.rarity(-1.0) == 0.0 and curve.rarity(0.9) > curve.rarity(0.5) > curve.rarity(0.3)
    # beyond the scores measured (here below 1e-4), the exponential tail keeps going, on the safe side for a Gaussian
    assert curve.threshold(1e-6) > curve.threshold(1e-5) > curve.threshold(1e-4)
    assert (fresh >= curve.threshold(1e-5)).mean() <= 1e-5
    assert curve.mated_quantile(0.05) == pytest.approx(0.6 - 1.645 * 0.1, abs=0.02)
    assert curve.fnmr_at(0.6) == pytest.approx(0.5, abs=0.05) and curve.fnmr_at(-1.0) == 0.0


def test_the_target_false_alarm_rate_holds_per_decision_whatever_the_gallery_size():
    rng = np.random.default_rng(1)
    curve = Curve.fit(rng.normal(0.0, 0.1, 400000))  # (the 1e-3 quantile of 50,000 scores is off by 0.004)
    for people in (1, 10, 50):
        t = curve.threshold(per_person(0.01, people))
        strangers = rng.normal(0.0, 0.1, (40000, people)).max(axis=1)  # a stranger's best score among n people
        assert (strangers >= t).mean() == pytest.approx(0.01, rel=0.2), people
    assert per_person(0.01, 1) == pytest.approx(0.01) and per_person(0.01, 10) == pytest.approx(0.001005, rel=1e-3)


def _synthetic_calibration(model="m", rng=None, n=60000, normalisation="none"):
    """Strangers score N(0, 0.1) at every size; the same person N(m, 0.1) with m from 0.15 (tiny faces) to 0.6 (32
    pixels between the eyes and more): small faces are hard to tell apart, large ones easy."""
    rng = rng or np.random.default_rng(2)
    sizes = rng.uniform(4.0, 100.0, n)
    mated = rng.random(n) < 0.3
    mean = 0.15 + 0.45 * np.clip((sizes - 4.0) / 28.0, 0.0, 1.0)
    scores = np.where(mated, rng.normal(mean, 0.1), rng.normal(0.0, 0.1, n))
    return Calibration.fit(model, scores, mated, sizes, np.zeros(n), rng.integers(0, 30, n),
                           normalisation=normalisation, made={"by": "a test"})


def test_a_calibration_has_curves_per_face_size_and_turns_one_target_into_thresholds(tmp_path):
    calibration = _synthetic_calibration()
    assert calibration.bin(12.0, 0.0) == (1, 0) and calibration.bin(9.99, -30.0) == (0, 1)
    assert calibration.bin(10.0, 24.9) == (1, 0) and calibration.bin(500.0, 170.0) == (7, 2)
    # strangers score alike at every size: the accept threshold hardly moves; the reject threshold rises with size
    small, large = calibration.curve(6.0, 0.0), calibration.curve(60.0, 0.0)
    assert small.threshold(1e-3) == pytest.approx(large.threshold(1e-3), abs=0.03)
    assert large.mated_quantile(0.02) > small.mated_quantile(0.02) + 0.3
    # each bin has its own mated curve: a face of 12 pixels is judged by the 10-14 bin, not a neighbour
    for size, lo, hi in ((12.0, 10, 14), (17.0, 14, 20), (24.0, 20, 28)):
        expected = 0.15 + 0.45 * np.clip((np.mean([lo, hi]) - 4.0) / 28.0, 0.0, 1.0)
        assert calibration.mated_quantile(0.5, size, 0.0) == pytest.approx(expected, abs=0.03), size
    # turned faces (no data here) borrow the frontal bin of the same size
    assert calibration.curves[3][1].borrowed == (3, 0)
    assert calibration.threshold(0.01, 10, 30.0, 0.0) == pytest.approx(calibration.curve(30.0, 0.0).threshold(
        per_person(0.01, 10)))
    path = calibration.save(tmp_path / "calibration" / "m.json")
    again = Calibration.load(path)
    assert again.model == "m" and again.made == {"by": "a test"}
    for size in (6.0, 12.0, 60.0):
        for score in (0.1, 0.3, 0.5):
            assert again.rarity(score, size, 0.0) == pytest.approx(calibration.rarity(score, size, 0.0), abs=1e-4)
    with pytest.raises(ValueError):
        Calibration.fit("m", [0.1, 0.2], [True, False], [20, 20], [0, 0])


def _scores_for_fit(n=20000, seed=3):
    rng = np.random.default_rng(seed)
    mated = rng.random(n) < 0.3
    return rng.normal(np.where(mated, 0.5, 0.0), 0.1), mated, np.full(n, 30.0), np.zeros(n)


def test_calibrations_are_found_per_model_and_a_model_without_one_gets_a_provisional_one(tmp_path, monkeypatch):
    from bifold_demo.recognition import calibration as module

    monkeypatch.setattr(module, "PACKAGED", tmp_path / "packaged")
    models = tmp_path / "models"
    provisional = find_calibration("adaface_ir101_wf12m", models)
    assert provisional.is_provisional and provisional.curve(30.0, 0.0).mated_quantile(0.05) is None
    assert provisional.rarity(0.22, 30.0, 0.0) == pytest.approx(3.0)  # the registered threshold: 1e-3 on LFW
    shipped = _synthetic_calibration("adaface_ir101_wf12m")
    shipped.save(tmp_path / "packaged" / "adaface_ir101_wf12m.json")
    assert find_calibration("adaface_ir101_wf12m", models).made == {"by": "a test"}
    assert not find_calibration("adaface_ir101_wf12m", models).is_provisional
    own = Calibration.fit("adaface_ir101_wf12m", *_scores_for_fit(), made={"by": "me"})
    own.save(calibration_path("adaface_ir101_wf12m", models))
    assert find_calibration("adaface_ir101_wf12m", models).made == {"by": "me"}  # the models folder's first
    (tmp_path / "packaged" / "other.json").write_text((tmp_path / "packaged" / "adaface_ir101_wf12m.json").read_text())
    with pytest.raises(ValueError):
        find_calibration("other", models)


def test_cohort_normalisation_takes_out_a_faces_pull_towards_everyone():
    rng = np.random.default_rng(4)
    dim = 64
    common = rng.normal(size=dim)  # what all faces share (embeddings of faces are not spread over the whole sphere)
    common /= np.linalg.norm(common)
    cohort_vectors = rng.normal(size=(300, dim)) / math.sqrt(dim) + 0.4 * common
    cohort_vectors /= np.linalg.norm(cohort_vectors, axis=1, keepdims=True)
    cohort = Cohort("m", cohort_vectors, tuple(f"c{k // 3}" for k in range(300)), top=20)
    gallery = rng.normal(size=(5, dim)) / math.sqrt(dim) + 0.4 * common
    gallery /= np.linalg.norm(gallery, axis=1, keepdims=True)
    plain = rng.normal(size=dim)
    plain /= np.linalg.norm(plain)
    lamb = plain + 2.0 * common  # a poor face drifting towards what all faces share: high against everyone
    lamb /= np.linalg.norm(lamb)
    for probe in (plain, lamb):
        raw = gallery @ probe
        t = cohort.normalise(raw, probe, gallery, "tnorm")
        mean, spread = cohort.stats(probe[None], top=0)
        assert np.allclose(t, (raw - mean[0]) / spread[0])
        a = cohort.normalise(raw, probe, gallery, "asnorm")
        p_mean, p_spread = cohort.stats(probe[None])
        g_mean, g_spread = cohort.stats(gallery)
        assert np.allclose(a, 0.5 * ((raw - p_mean[0]) / p_spread[0] + (raw - g_mean) / g_spread))
    assert cohort.stats(lamb[None], top=0)[0][0] > cohort.stats(plain[None], top=0)[0][0] + 0.2
    # after T-norm the poor face no longer outscores the plain one against the gallery as a whole
    assert (gallery @ lamb).mean() > (gallery @ plain).mean() + 0.2
    assert abs(cohort.normalise(gallery @ lamb, lamb, gallery, "tnorm").mean()) < 1.0
    assert np.allclose(cohort.normalise(gallery @ plain, plain, gallery, "none"), gallery @ plain)
    assert len(cohort._rows({"c0", "c1"})) == 294  # a person's own faces can be left out
    with pytest.raises(ValueError):
        cohort.normalise(gallery @ plain, plain, gallery, "znorm")


# -- open-set decisions -----------------------------------------------------------------------------------------------

DIM = 16


class _Gallery:
    """A stand-in gallery: one photo per person, each person a unit vector."""

    def __init__(self, people: dict):
        self._lock = threading.RLock()
        self._names = sorted(people)
        self._matrix = np.array([people[n] for n in self._names], float).reshape(len(people), DIM)
        self._templates = np.zeros((0, DIM))

    def people(self):
        return {n: 1 for n in self._names}

    def raw_scores(self, vector, model=None):
        n = len(self._names)
        return RawScores("m", tuple(self._names), self._matrix @ np.asarray(vector), np.arange(n), ("frontal",) * n,
                         tuple(Path(f"{name}.jpg") for name in self._names), np.zeros(0), np.zeros(0, int), ())


def _axis(k):
    v = np.zeros(DIM)
    v[k] = 1.0
    return v


def _people(twin=None):
    people = {"Ada": _axis(0), "Ben": _axis(1), "Cleo": _axis(2)}
    if twin is not None:  # someone the model finds alike to Ada
        people["Adele"] = twin * _axis(0) + math.sqrt(1 - twin ** 2) * _axis(3)
    return people


def _toward(scores: dict, people: dict, rest_axis: int = DIM - 1):
    """A unit vector scoring ``scores[name]`` against those people (by least squares) and about 0 against the rest
    (what is left over lies along ``rest_axis``: faces with different ones differ there too)."""
    names = sorted(people)
    matrix = np.array([people[n] for n in names])
    target = np.array([scores.get(n, 0.0) for n in names])
    v, *_ = np.linalg.lstsq(matrix, target, rcond=None)
    rest = math.sqrt(max(1e-9, 1.0 - float(v @ v)))
    return v + rest * _axis(rest_axis)


def _agg(scores, people, size=40.0, weight=2.0, frames=3, yaw=0.0):
    return Aggregate(_toward(scores, people), weight, size, yaw, frames, "m", 0.9)


@pytest.fixture(scope="module")
def calibration():
    return _synthetic_calibration()


def _policy(calibration, people=None, **settings):
    people = people or _people()
    return IdentityPolicy(_Gallery(people), calibration, settings=DecisionSettings(**settings),
                          log=DecisionLog()), people


def test_a_track_is_named_only_with_a_rare_score_a_margin_and_enough_evidence(calibration):
    policy, people = _policy(calibration)
    named = policy.decide({1: _agg({"Ada": 0.5}, people)}, 0.0, frame=7)[1]
    assert (named.state, named.name, named.reason) == (NAMED, "Ada", "accepted")
    assert named.raw == pytest.approx(0.5) and named.calibrated == pytest.approx(calibration.rarity(0.5, 40, 0))
    assert named.runner_up in ("Ben", "Cleo") and named.runner_up_raw == pytest.approx(0.0, abs=1e-6)
    assert named.accept_rarity == pytest.approx(-math.log10(per_person(0.01, 3)))
    assert named.accept == pytest.approx(calibration.threshold(0.01, 3, 40.0, 0.0))
    assert named.model == "m" and named.frame == 7 and named.evidence["frames"] == 3 and named.label == "Ada"
    policy, people = _policy(calibration)
    little = policy.decide({1: _agg({"Ada": 0.5}, people, weight=0.2)}, 0.0)[1]  # (0.5 needed)
    assert (little.state, little.reason, little.progress) == (UNDECIDED, "collecting", 0.4) and little.label == "? 40%"
    one = policy.decide({1: _agg({"Ada": 0.5}, people, frames=1)}, 0.1)[1]
    assert (one.state, one.progress) == (UNDECIDED, 0.5)  # a single frame is never enough
    waiting = policy.decide({1: None}, 0.2, waiting={1: "turned"})[1]
    assert (waiting.state, waiting.reason, waiting.raw) == (UNDECIDED, "turned", None)


def test_a_runner_up_too_close_leaves_the_track_undecided(calibration):
    policy, people = _policy(calibration, _people(twin=0.97))
    close = policy.decide({1: _agg({"Ada": 0.31, "Adele": 0.30}, people)}, 0.0)[1]
    assert close.raw > close.accept and close.runner_up_raw > close.accept  # both pass the threshold ...
    assert (close.state, close.reason) == (UNDECIDED, "too close to another")  # ... but neither beats the other
    clear = policy.decide({2: _agg({"Ada": 0.5, "Adele": 0.30}, people)}, 0.0)[2]
    assert (clear.state, clear.name) == (NAMED, "Ada")


def test_unknown_needs_good_evidence_held_below_the_reject_threshold(calibration):
    policy, people = _policy(calibration, sustain_unknown=1.0)
    stranger = _agg({}, people, size=40.0)
    first = policy.decide({1: stranger}, 0.0)[1]
    assert first.state == UNDECIDED and first.reason.startswith("waiting: unknown in 1.0")
    assert policy.decide({1: stranger}, 0.5)[1].state == UNDECIDED
    later = policy.decide({1: stranger}, 1.0)[1]
    assert (later.state, later.label, later.changed) == (UNKNOWN, "unknown", True)
    assert later.reject == pytest.approx(calibration.mated_quantile(0.02, 40.0, 0.0))
    # a tiny face: a known person's would score as low, so it is never called unknown
    policy, people = _policy(calibration)
    for t in (0.0, 1.0, 2.0, 5.0):
        tiny = policy.decide({1: _agg({}, people, size=8.0)}, t)[1]
        assert (tiny.state, tiny.reason) == (UNDECIDED, "faces of this quality cannot be told apart")
    # nor a turned face whose evidence is thin
    policy, people = _policy(calibration)
    assert policy.decide({1: _agg({}, people, weight=0.3)}, 0.0)[1].reason == "collecting"
    # nor a face below the reject threshold whose best score strangers seldom reach (a known person's face from
    # photos on an avatar scores lower against their photos than the renders the calibration comes from)
    policy, people = _policy(calibration, sustain_unknown=0.0)
    odd = policy.decide({1: _agg({"Ada": 0.2}, people, size=40.0)}, 0.0)[1]
    assert odd.raw < odd.reject and odd.raw < odd.accept
    assert (odd.state, odd.reason) == (UNDECIDED, "unlike strangers too")
    usual = policy.decide({2: _agg({"Ada": 0.05}, people, size=40.0)}, 0.0)[2]
    assert usual.state == UNKNOWN


def test_a_named_track_keeps_its_name_down_to_the_keep_threshold(calibration):
    """In the 14-20 pixel bin of the made-up calibration, with 3 known people: accept at 0.274, keep at 0.182, reject
    at 0.139."""
    policy, people = _policy(calibration, sustain=1.0, sustain_unknown=1.0)
    size = 16.0
    d = policy.decide({1: _agg({"Ada": 0.5}, people, size=size)}, 0.0)[1]
    assert d.state == NAMED and d.reject < d.keep < d.accept
    assert d.keep == pytest.approx(calibration.curve(size, 0).threshold(10 ** -(d.accept_rarity - 1.0)))
    between = policy.decide({1: _agg({"Ada": 0.25}, people, size=size)}, 1.0)[1]
    assert (between.state, between.name, between.reason) == (NAMED, "Ada", "kept")  # below accept, above keep
    below = policy.decide({1: _agg({"Ada": 0.17}, people, size=size)}, 2.0)[1]  # below keep: the name goes
    assert (below.state, below.reason, below.changed) == (UNDECIDED, "between thresholds", True)
    low = policy.decide({1: _agg({"Ada": 0.10}, people, size=size)}, 3.0)[1]  # below reject: unknown, when it holds
    assert low.state == UNDECIDED and low.reason.startswith("waiting: unknown")
    assert policy.decide({1: _agg({"Ada": 0.10}, people, size=size)}, 4.0)[1].state == UNKNOWN
    # unknown stays while nobody comes within keep of a name
    assert policy.decide({1: _agg({"Ada": 0.16}, people, size=size)}, 5.0)[1].state == UNKNOWN
    assert policy.decide({1: _agg({"Ada": 0.24}, people, size=size)}, 6.0)[1].state == UNDECIDED
    # a fresh track is not named at the keep level
    assert policy.decide({2: _agg({"Ada": 0.25}, people, size=size)}, 7.0)[2].state == UNDECIDED


def test_switching_to_another_name_needs_accept_and_margin_sustained(calibration):
    policy, people = _policy(calibration, sustain=1.0)
    size = 24.0
    assert policy.decide({1: _agg({"Ada": 0.5}, people, size=size)}, 0.0)[1].name == "Ada"
    held = policy.decide({1: _agg({"Ada": 0.2, "Ben": 0.5}, people, size=size)}, 1.0)[1]
    assert (held.state, held.name) == (NAMED, "Ada") and "Ben in 1.0 s" in held.reason  # Ada still above keep
    assert policy.decide({1: _agg({"Ada": 0.2, "Ben": 0.5}, people, size=size)}, 1.5)[1].name == "Ada"
    switched = policy.decide({1: _agg({"Ada": 0.2, "Ben": 0.5}, people, size=size)}, 2.0)[1]
    assert (switched.state, switched.name, switched.changed) == (NAMED, "Ben", True)
    # Ada gone below keep: the name goes at once, Ben still has to wait
    policy, people = _policy(calibration, sustain=1.0)
    policy.decide({1: _agg({"Ada": 0.5}, people, size=size)}, 0.0)
    dropped = policy.decide({1: _agg({"Ada": 0.05, "Ben": 0.5}, people, size=size)}, 1.0)[1]
    assert dropped.state == UNDECIDED and dropped.reason == "waiting: Ben in 1.0 s"
    # a brief look of someone else does not count: the wait starts again when the evidence changes its mind
    assert policy.decide({1: _agg({"Ada": 0.05, "Cleo": 0.5}, people, size=size)}, 1.6)[1].state == UNDECIDED
    assert policy.decide({1: _agg({"Ada": 0.05, "Ben": 0.5}, people, size=size)}, 2.1)[1].state == UNDECIDED
    assert policy.decide({1: _agg({"Ada": 0.05, "Ben": 0.5}, people, size=size)}, 3.1)[1].name == "Ben"


def test_one_identity_per_track_and_one_track_per_identity(calibration):
    policy, people = _policy(calibration)
    both = policy.decide({1: _agg({"Ada": 0.45}, people), 2: _agg({"Ada": 0.6}, people)}, 0.0)
    assert (both[2].state, both[2].name) == (NAMED, "Ada")
    assert (both[1].state, both[1].reason) == (UNDECIDED, "contested")
    # a track holding a name keeps it against a slightly stronger newcomer
    policy, people = _policy(calibration)
    policy.decide({1: _agg({"Ada": 0.45}, people)}, 0.0)
    again = policy.decide({1: _agg({"Ada": 0.45}, people), 3: _agg({"Ada": 0.46}, people)}, 1.0)
    assert (again[1].name, again[3].reason) == ("Ada", "contested")
    # different names: no contest
    two = policy.decide({1: _agg({"Ada": 0.45}, people), 3: _agg({"Ben": 0.5}, people)}, 2.0)
    assert (two[1].name, two[3].name) == ("Ada", "Ben")
    policy.decide({3: _agg({"Ben": 0.5}, people)}, 3.0)  # track 1 lost for a moment: remembered
    back = policy.decide({1: _agg({"Ada": 0.45}, people), 3: _agg({"Ben": 0.5}, people)}, 3.5)
    assert (back[1].name, back[1].changed) == ("Ada", False)
    policy.forget(1)  # removed by the tracker
    assert set(policy.tracks) == {3}


def test_a_track_found_to_show_someone_else_loses_its_name_at_once(calibration):
    policy, people = _policy(calibration, sustain=1.0)
    policy.decide({1: _agg({"Ada": 0.5}, people)}, 0.0)
    ada = _toward({"Ada": 0.5}, people)
    assert policy.consistent(policy.self_score(_toward({"Ada": 0.55}, people), ada), 40.0, 0.0)
    ben = _toward({"Ben": 0.6}, people, rest_axis=DIM - 2)
    assert not policy.consistent(policy.self_score(ben, ada), 40.0, 0.0)  # Ben's face on Ada's track
    assert policy.consistent(policy.self_score(ben, ada), 8.0, 0.0)  # (tiny faces cannot tell)
    policy.reset(1, 1.0)
    fresh = policy.decide({1: _agg({"Ben": 0.5}, people)}, 1.1)[1]
    assert fresh.state == UNDECIDED and fresh.reason.startswith("waiting: Ben")  # not straight to another name
    assert policy.decide({1: _agg({"Ben": 0.5}, people)}, 2.1)[1].name == "Ben"
    # after a reset the old name is gone, even where the new evidence would have kept it (above keep, below accept)
    policy, people = _policy(calibration)
    assert policy.decide({1: _agg({"Ada": 0.5}, people, size=16.0)}, 0.0)[1].name == "Ada"
    assert policy.decide({1: _agg({"Ada": 0.25}, people, size=16.0)}, 1.0)[1].name == "Ada"  # kept, normally
    policy.reset(1, 1.5)
    after = policy.decide({1: _agg({"Ada": 0.25}, people, size=16.0)}, 2.0)[1]
    assert (after.state, after.changed) == (UNDECIDED, False)


def test_decisions_are_logged_for_the_evaluation(calibration, tmp_path):
    log = DecisionLog(tmp_path / "decisions.jsonl")
    people = _people()
    policy = IdentityPolicy(_Gallery(people), calibration, log=log)
    policy.decide({1: _agg({"Ada": 0.5}, people), 2: None}, 0.0, frame=1, evidence={1: {"sources": ["window"]}})
    policy.decide({1: _agg({"Ada": 0.5}, people), 2: None}, 0.1, frame=2)
    lines = [json.loads(line) for line in (tmp_path / "decisions.jsonl").read_text().splitlines()]
    assert len(lines) == 4 and len(log.entries) == 4 and [d.track for d in log.changes()] == [1]
    first = lines[0]
    assert first["state"] == "named" and first["name"] == "Ada" and first["model"] == "m" and first["changed"]
    for key in ("raw", "score", "calibrated", "runner_up", "runner_up_raw", "runner_up_calibrated", "accept", "keep",
                "reject", "bin", "normalisation"):
        assert key in first
    assert first["evidence"]["sources"] == ["window"] and first["evidence"]["frames"] == 3
    assert lines[1]["state"] == "undecided" and not lines[2]["changed"]


# -- the pipeline ------------------------------------------------------------------------------------------------------

COLOURS = {"Ada": (200, 60, 40), "Ben": (40, 200, 60), "Cleo": (60, 40, 200), "Stranger": (180, 180, 40)}


def _identity(colour) -> np.ndarray:
    """The made-up embedding of a face of ``colour`` (BGR): a direction of its own for each person of
    :data:`COLOURS` (the nearest colour), the same in every frame: a perfect recogniser."""
    names = list(COLOURS)
    nearest = min(range(len(names)), key=lambda k: np.linalg.norm(np.subtract(COLOURS[names[k]], colour)))
    return _axis(4 + nearest)


class _PipelineEngine:
    """Finds the faces it was told about (frame pixels) in head regions and windows, aligns them with the real
    alignment, and embeds a face by its colour."""

    model_id = "m"
    threshold = 0.3
    fiqa = None
    profile = type("P", (), {"name": "test"})()

    def __init__(self):
        self.faces = []  # this frame's faces (frame pixels)
        self.windows = {}  # id(window image) -> (offset, scale)
        self.embedded = 0
        self.embedder = self

    def detect(self, image, min_size=0.0, canvas=None, region=None):
        offset, scale = self.windows.get(id(image), (np.zeros(2), 1.0))
        found = []
        for face in self.faces:
            middle = face.points[:2].mean(axis=0)
            if region is not None:
                x, y, w, h = region
                if not (x <= middle[0] <= x + w and y <= middle[1] <= y + h):
                    continue
            local = Face(((face.box[0] - offset[0]) * scale, (face.box[1] - offset[1]) * scale, face.box[2] * scale,
                          face.box[3] * scale), tuple(tuple((np.asarray(p) - offset) * scale) for p in face.landmarks),
                         face.score)
            if 0 <= local.points[:, 0].min() and local.points[:, 0].max() < image.shape[1]:
                found.append(local)
        return found

    def align(self, image, face):
        from bifold_demo.recognition.align import align

        return align(image, face.landmarks)

    def embed(self, crops):
        from bifold_demo.recognition.embed import Embeddings

        self.embedded += len(crops)
        vectors = [_identity(np.median(c[40:90, 35:77].reshape(-1, 3), axis=0)) for c in crops]
        return Embeddings(np.array(vectors, np.float32), np.ones(len(crops), np.float32))


def _paint(image, face, colour, seed):
    """A textured face of ``colour`` (so that it is sharp) around ``face``'s box."""
    x, y, w, h = (round(v) for v in face.box)
    pad = int(0.4 * w)
    rng = np.random.default_rng(seed)
    patch = np.clip(np.asarray(colour, float) + rng.normal(0, 25, (h + 2 * pad, w + 2 * pad, 3)), 0, 255)
    y0, x0 = max(0, y - pad), max(0, x - pad)
    image[y0:y + h + pad, x0:x + w + pad] = patch[:image[y0:y + h + pad, x0:x + w + pad].shape[0],
                                                  :image[y0:y + h + pad, x0:x + w + pad].shape[1]]


class _Detector:
    def __init__(self):
        self.boxes = []

    def detect(self, image, conf=None):
        from bifold_demo.detection.results import Detection

        return [Detection(tuple(box), 0.9, 0, "person") for box in self.boxes]


class _Service:
    """A stand-in window service: keeps the requests; its scheduler takes the priority hook."""

    def __init__(self):
        self.scheduler = type("S", (), {"priority": None})()
        self.asked = []

    def request(self, requests):
        self.asked = list(requests)


def _scene(t, who=("Ada", "Stranger", "Ben"), turned=("Ben",)):
    """People walking slowly: their boxes (frame pixels), their faces (where the head model expects them), and the
    frame with the faces painted on."""
    image = np.full((720, 1280, 3), 90, np.uint8)
    boxes, faces = [], []
    for k, name in enumerate(who):
        x = 100 + 400 * k + 20 * t
        box = (x, 60.0, x + 220, 660.0)  # (near the camera: 22 pixels between the eyes)
        boxes.append(box)
        if name in turned:
            continue  # the back of the head: no face
        face = _head_face(box)
        faces.append(face)
        _paint(image, face, COLOURS[name], k)
    return image, boxes, faces


def _video_frame(image, index, t, windows=()):
    from bifold_demo.sources.base import Frame

    return Frame(image, index, float(index), "scene", t, list(windows))


def _recognizer(calibration, service=None, rate=2.0, **settings):
    from bifold_demo.recognition.people import PeopleRecognizer

    people = {name: _identity(COLOURS[name]) for name in ("Ada", "Ben", "Cleo")}
    engine, detector = _PipelineEngine(), _Detector()
    recognizer = PeopleRecognizer(_Gallery(people), detector, engine, calibration=calibration, windows=service,
                                  rate=rate, settings=DecisionSettings(**settings), log=DecisionLog())
    return recognizer, engine, detector


def _run(recognizer, engine, detector, seconds=4.0, fps=10.0, **scene):
    results = []
    for k in range(int(seconds * fps)):
        t = k / fps
        image, boxes, faces = _scene(t, **scene)
        detector.boxes, engine.faces = boxes, faces
        results.append(recognizer.process(_video_frame(image, k, t)))
    return results


def test_the_pipeline_names_known_people_says_unknown_for_strangers_and_waits_for_turned_heads(calibration):
    recognizer, engine, detector = _recognizer(calibration, rate=2.0)
    results = _run(recognizer, engine, detector)
    last = {p.id: p for p in results[-1].people}
    ada, stranger, ben = (last[i] for i in sorted(last))
    assert (ada.state, ada.name) == (NAMED, "Ada")
    assert stranger.state == UNKNOWN
    assert (ben.state, ben.decision.reason) == (UNDECIDED, "no face found")  # turned away all along: no guess
    assert ada.face is not None and ada.face.source == "frame" and ben.face is None
    # the budget: about 2 embeddings a second for everyone (one at once), decided tracks re-checked only now and then
    assert engine.embedded <= 2 * 4 + 1
    first_named = next(r.time for r in results if any(p.name == "Ada" for p in r.people))
    assert first_named <= 1.5
    assert results[-1].counts() == {NAMED: 1, UNKNOWN: 1, UNDECIDED: 1}
    assert all(set(r.timings) == {"people", "track", "faces", "quality", "embed", "decide"} for r in results)
    # never a wrong name, and the same run gives the same decisions
    assert not any(p.name not in (None, "Ada") for r in results for p in r.people)
    again, engine2, detector2 = _recognizer(calibration, rate=2.0)
    replay = _run(again, engine2, detector2)
    assert [[(p.id, p.state, p.name, p.decision.reason) for p in r.people] for r in results] == \
        [[(p.id, p.state, p.name, p.decision.reason) for p in r.people] for r in replay]


def test_a_track_that_turns_out_to_show_someone_else_loses_its_name(calibration):
    """The tracker keeps one number while the person under it changes (two people crossed): the new person's first
    good face disagrees with the track's evidence, so the old evidence and the name go at once."""
    recognizer, engine, detector = _recognizer(calibration, rate=5.0, sustain=1.0)
    _run(recognizer, engine, detector, seconds=2.0, who=("Ada",), turned=())
    assert recognizer.log.entries[-1].name == "Ada"
    results = []
    for k in range(20, 50):
        t = k / 10
        image, boxes, faces = _scene(t, who=("Cleo",), turned=())
        detector.boxes, engine.faces = boxes, faces
        results.append(recognizer.process(_video_frame(image, k, t)))
    states = [(r.people[0].state, r.people[0].name) for r in results]
    assert results[0].people[0].id == 1 and recognizer.tracks[1].resets == 1
    assert ("named", "Ada") not in states[1:]  # gone with the first good face of Cleo
    assert states[-1] == (NAMED, "Cleo")  # once Cleo's evidence held for the switch time
    switched = next(k for k, s in enumerate(states) if s == (NAMED, "Cleo"))
    assert switched >= 10  # (1 s at 10 frames a second)


def test_the_tracks_ask_for_windows_and_their_faces_come_from_the_windows(calibration):
    service = _Service()
    recognizer, engine, detector = _recognizer(calibration, service=service, rate=5.0)
    assert service.scheduler.priority == recognizer.priority
    image, boxes, faces = _scene(0.0, who=("Ada", "Ben"), turned=("Ben",))
    detector.boxes, engine.faces = boxes, faces
    recognizer.process(_video_frame(image, 0, 0.0))
    assert sorted(r.tag for r in service.asked) == [1, 2]  # undecided, faces small in the frame
    assert all(r.face_px == recognizer.window_face_px and r.box is not None for r in service.asked)
    # the next frame comes with a window on Ada's head: her face is found and aligned there
    region = HEAD.square(boxes[0])
    x0, y0, x1, y1 = (round(v) for v in region)
    picture = cv2.resize(image[y0:y1, x0:x1], None, fx=4.0, fy=4.0, interpolation=cv2.INTER_CUBIC)
    window = _Window(picture, (x0, y0), 4.0, tag=1)
    engine.windows[id(picture)] = (np.array([x0, y0], float), 4.0)
    result = recognizer.process(_video_frame(image, 1, 0.6, [window]))  # (a track is embedded twice a second)
    ada = next(p for p in result.people if p.id == 1)
    assert ada.face.source == "window" and ada.face.image is picture and ada.embedded
    assert recognizer.tracks[1].evidence.frames[-1].source == "window"
    assert recognizer.tracks[1].evidence.frames[-1].size == pytest.approx(4.0 * face_size(faces[0]), rel=0.02)
    # the window order: undecided before decided, waiting adds up so that tracks take turns
    from bifold_demo.sensor import WindowRequest

    high = WindowRequest(1, box=(0, 0, 10, 30), priority=3.0)
    low = WindowRequest(2, box=(0, 0, 10, 30), priority=1.0)
    assert recognizer.priority(high, 0) > recognizer.priority(low, 0)
    assert recognizer.priority(low, 41) > recognizer.priority(high, 0)


def test_a_webcams_windows_are_cut_from_its_full_picture_and_serve_the_same_way(calibration):
    """A real camera: the frame is a 1280x720 copy of a 2560x1440 picture, and the window service cuts the tracks'
    windows from the full picture (``WindowService.for_image``, real window geometry): the faces found there map back
    to the frame and are aligned from the full picture's pixels."""
    from bifold_demo.sensor import WindowService
    from bifold_demo.sources.base import Frame

    service = WindowService(max_windows=4)
    service.bind((1280, 720))
    recognizer, engine, detector = _recognizer(calibration, service=service, rate=5.0)
    image, boxes, faces = _scene(0.0, who=("Ada",), turned=())
    full = cv2.resize(image, (2560, 1440), interpolation=cv2.INTER_CUBIC)
    detector.boxes, engine.faces = boxes, faces
    recognizer.process(_video_frame(image, 0, 0.0))  # the track asks for a window
    assert [r.tag for r in service.requests] == [1]
    frame = Frame(image, 1, 0.6, "webcam")  # (a camera: no scene time; the clock is its own)
    frame.windows = service.for_image(full, frame)
    [window] = frame.windows
    assert window.window.scale == pytest.approx(2.0) and window.time is None
    x0, y0 = window.view_rect[:2]
    engine.windows[id(window.image)] = (np.array([x0, y0]), window.window.scale)
    result = recognizer.process(frame)
    ada = result.people[0]
    assert ada.face.source == "window" and ada.embedded
    assert np.allclose(ada.face.face.points, faces[0].points, atol=0.5)  # back in the frame's pixels
    assert ada.face.size == pytest.approx(2.0 * face_size(faces[0]), rel=0.02)  # aligned from the full picture


# -- scoring decisions against the truth ------------------------------------------------------------------------------


def _truth(name, avatar, yaw=0.0, eyes=10.0, covered=False):
    return PersonTruth(name, avatar, (0, 0, 10, 30), 4.0, ((0, 0), (1, 0)), eyes, abs(yaw), yaw, 0.0,
                       eyes_covered=covered)


class _D:
    def __init__(self, state, name=None):
        self.state, self.name = state, name


def test_outcomes_count_names_per_segment_from_when_the_face_was_shown():
    outcomes = Outcomes(known={"Ada", "Ben"})
    timeline = [  # (time, Ada's yaw, Ada's decision, the stranger's decision)
        (0.0, 80.0, _D("undecided"), _D("undecided")),  # Ada turned away: not shown yet
        (0.5, 10.0, _D("undecided"), _D("undecided")),  # shown from here
        (1.0, 10.0, _D("undecided"), _D("unknown")),
        (2.0, 10.0, _D("named", "Ada"), _D("unknown")),  # named 1.5 s after the face was shown
        (3.0, 10.0, _D("named", "Ada"), _D("named", "Ben")),  # the stranger wrongly named
    ]
    for t, yaw, ada, stranger in timeline:
        truth = [_truth("p1", "Ada", yaw), _truth("p2", "Zoe")]
        outcomes.update(t, truth, {"p1": 1, "p2": 2}, {1: ada, 2: stranger})
    ada, zoe = outcomes.segments[("p1", 1)], outcomes.segments[("p2", 2)]
    assert ada.first_shown == 0.5 and ada.shown == pytest.approx(2.5) and ada.correct_at == 2.0
    assert ada.named_within(1.5) and not ada.named_within(1.0) and ada.known and not zoe.known  # (from 0.5 s)
    assert zoe.wrong == ("Ben",) and zoe.unknown_at == 1.0 and zoe.named_at == 3.0
    summary = outcomes.summary(shown=1.0)
    assert summary["named_within"] == {"2.0": 1.0, "5.0": 1.0} and summary["time_to_name"]["median"] == 1.5
    assert summary["fpir"] == 1.0 and summary["called_unknown"] == 1.0 and summary["confused"] == 0.0
    # another track for Ada (a new segment); a face hidden behind someone does not count as shown
    outcomes.update(4.0, [_truth("p1", "Ada", covered=True)], {"p1": 7}, {7: _D("undecided")})
    outcomes.update(5.0, [_truth("p1", "Ada", covered=True)], {"p1": 7}, {7: _D("unknown")})
    later = outcomes.segments[("p1", 7)]
    assert later.first_shown is None and later.unknown_at == 5.0
    assert outcomes.summary(shown=0.0)["unknown"] == 0.0  # (never shown: not eligible even at 0 s)
    assert outcomes.summary(shown=10.0)["known_eligible"] == 0 and outcomes.summary(shown=10.0)["named"] is None


# -- the calibration builder ------------------------------------------------------------------------------------------


def _avatar_world(avatars=12, per=40, dim=32, seed=5, spread=0.55):
    """Made-up faces of ``avatars`` identities (``per`` faces each at swept sizes), and a gallery of each: an
    identity direction plus noise that grows as faces get smaller, and a part all faces share."""
    from bifold_demo.recognition.calibrate import Faces, GalleryRows

    rng = np.random.default_rng(seed)
    common = rng.normal(size=dim)
    common /= np.linalg.norm(common)
    ids = rng.normal(size=(avatars, dim))
    ids /= np.linalg.norm(ids, axis=1, keepdims=True)
    names = [f"A{k:02d}" for k in range(avatars)]
    vectors, people, sizes = [], [], []
    for k, name in enumerate(names):
        for j in range(per):
            size = [9.0, 12.0, 16.0, 24.0, 34.0, 50.0, 70.0][j % 7]
            noise = spread * 12.0 / size
            v = ids[k] + 0.5 * common + noise * rng.normal(size=dim) / math.sqrt(dim)
            vectors.append(v / np.linalg.norm(v))
            people.append(name)
            sizes.append(size)
    rows = ids + 0.5 * common
    rows /= np.linalg.norm(rows, axis=1, keepdims=True)
    second = rows + 0.4 * rng.normal(size=rows.shape) / math.sqrt(dim)  # a second photo of each, a little off
    second /= np.linalg.norm(second, axis=1, keepdims=True)
    faces = Faces(np.array(vectors), np.array(people), np.array(sizes), np.zeros(len(sizes)),
                  np.ones(len(sizes), bool), np.array(sizes))
    return faces, GalleryRows(np.vstack([rows, second]), np.array(names + names))


def test_the_calibration_builder_scores_faces_against_galleries_with_their_own_left_out_of_the_cohort():
    from bifold_demo.recognition.calibrate import fit, scores

    faces, gallery = _avatar_world()
    raw, people = scores(faces, gallery)
    assert people == sorted(set(gallery.owners.tolist())) and raw.shape == (len(faces.vectors), len(people))
    both = faces.vectors @ gallery.rows.T  # a person's score is the best of their photos (two each here)
    assert np.allclose(raw[:, 4], both[:, gallery.owners == people[4]].max(axis=1))
    own = np.array([people.index(p) for p in faces.people])
    assert (raw.argmax(axis=1) == own).mean() > 0.9  # the made-up faces are recognisable
    t, _ = scores(faces, gallery, "tnorm", faces)
    # T-norm with the face's own avatar and the gallery person left out of the cohort: by hand for one face and person
    k, j = 3, 5
    keep = (faces.people != faces.people[k]) & (faces.people != people[j])
    cohort_scores = faces.vectors[keep] @ faces.vectors[k]
    expected = (raw[k, j] - cohort_scores.mean()) / cohort_scores.std()
    assert t[k, j] == pytest.approx(expected, rel=1e-6)
    a, _ = scores(faces, gallery, "asnorm", faces)
    assert np.isfinite(a).all() and (a.argmax(axis=1) == own).mean() > 0.9
    calibration = fit("m", faces, gallery, made={"by": "a test"})
    assert calibration.model == "m" and calibration.normalisation == "none"
    big, small = calibration.curve(60.0, 0.0), calibration.curve(9.0, 0.0)
    assert big.mated_quantile(0.5) > small.mated_quantile(0.5)  # bigger faces score higher against their own


def test_the_held_out_check_measures_the_false_alarm_rate_asked_for(monkeypatch):
    from bifold_demo.recognition import calibration as calibration_module
    from bifold_demo.recognition.calibrate import check, choose

    monkeypatch.setattr(calibration_module, "MIN_NONMATED", 50)  # (a small made-up world)
    monkeypatch.setattr(calibration_module, "MIN_MATED", 10)
    faces, gallery = _avatar_world(avatars=16, per=42)
    results = check("m", faces, gallery, kinds=("none", "tnorm"), splits=4, targets=(0.1,))
    for kind in ("none", "tnorm"):
        entry = results[kind]["0.1"]
        assert entry["strangers"] > 0 and entry["known"] > 0
        assert entry["strangers"] + entry["known"] == 4 * 8 * 42  # only the held-out half's faces are searched
        assert entry["fpir"] <= 0.3, (kind, entry)  # near the target (few identities: a coarse check)
        assert entry["tpir"] > 0.3 and entry["wrong"] <= entry["named"]
    assert choose(results, 0.1, kinds=("none", "tnorm")) in ("none", "tnorm")
    worse = {"none": {"0.01": {"fpir": 0.01, "tpir": 0.8}}, "tnorm": {"0.01": {"fpir": 0.01, "tpir": 0.805}},
             "asnorm": {"0.01": {"fpir": 0.05, "tpir": 0.95}}}
    assert choose(worse) == "none"  # half a point more is no reason; a normalisation missing the target is out
    better = {"none": {"0.01": {"fpir": 0.01, "tpir": 0.8}}, "tnorm": {"0.01": {"fpir": 0.012, "tpir": 0.85}},
              "asnorm": {"0.01": {"fpir": 0.01, "tpir": 0.83}}}
    assert choose(better) == "tnorm"
