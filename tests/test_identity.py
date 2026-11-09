"""Person, face, identity (step F2) with stand-ins, runnable anywhere: faces in head regions and whose they are, the
evidence per track, calibration, open-set decisions and the pipeline. (The real models and avatars:
``test_identity_real.py``.)"""

import json
import math
import threading
from pathlib import Path

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
    assert [c.key for c in chosen] == ["c2", "b", "a"]  # undecided first (the better of track 3's two), then the
    # decided track due for a check; track 4 was checked a second ago
    assert [c.key for c in s.choose(faces, 1, now=10.0)] == ["c2"]
    assert s.choose([Candidate(5, 0.9, last=9.8)], 1, now=10.0) == []  # at most twice a second
    assert s.choose([Candidate(5, 0.4, median=0.5)], 1, now=10.0) == []  # no better than its median: not worth it
    assert s.choose([Candidate(5, 0.6, median=0.5)], 1, now=10.0)


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
    little = policy.decide({1: _agg({"Ada": 0.5}, people, weight=0.4)}, 0.0)[1]
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
    assert set(policy.tracks) == {1, 3}
    policy.decide({3: _agg({"Ben": 0.5}, people)}, 3.0)
    assert set(policy.tracks) == {3}  # a track no longer given is forgotten


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
