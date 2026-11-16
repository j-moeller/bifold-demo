"""The evaluation's metrics (recognition/metrics.py) against exact hand-computed cases, scikit-learn's roc_curve
(skipped without scikit-learn), and the mutations they must catch: a false-accept rate off by one pair, FNIR counted
per frame, a bootstrap over frames instead of people."""

import math

import numpy as np
import pytest

from bifold_demo.recognition import metrics
from bifold_demo.recognition.truth import Segment

GENUINE = [0.6, 0.8, 0.9]
IMPOSTOR = [0.1, 0.2, 0.7]


def test_roc_by_hand():
    far, tar, thresholds = metrics.roc(GENUINE, IMPOSTOR)
    assert thresholds.tolist() == [math.inf, 0.9, 0.8, 0.7, 0.6, 0.2, 0.1]
    assert np.allclose(far, [0, 0, 0, 1 / 3, 1 / 3, 2 / 3, 1])
    assert np.allclose(tar, [0, 1 / 3, 2 / 3, 2 / 3, 1, 1, 1])
    far_d, frr, _ = metrics.det(GENUINE, IMPOSTOR)
    assert np.allclose(frr, 1 - tar) and np.allclose(far_d, far)


def test_tar_at_far_by_hand():
    strict = metrics.tar_at_far(GENUINE, IMPOSTOR, 0.0)
    assert strict["tar"] == pytest.approx(2 / 3) and strict["threshold"] == 0.8 and strict["far"] == 0.0
    assert not strict["resolvable"]  # (no rate of zero is shown by three pairs)
    third = metrics.tar_at_far(GENUINE, IMPOSTOR, 1 / 3)
    assert third["tar"] == 1.0 and third["threshold"] == 0.6 and third["far"] == pytest.approx(1 / 3)
    assert third["resolvable"] and third["impostors_allowed"] == 1
    assert metrics.tar_at_far(GENUINE, IMPOSTOR, 0.5)["far"] == pytest.approx(1 / 3)  # (never more than asked)


def test_tar_at_far_lets_exactly_the_asked_impostors_through():
    """1,000 impostors at 1e-3: exactly one may pass. A threshold off by one pair (none, or two) gives another TAR."""
    impostor = np.r_[np.linspace(0.0, 0.5, 997), 0.70, 0.80, 0.90]  # the three highest: 0.9, 0.8, 0.7
    genuine = np.array([0.75, 0.85, 0.95, 0.99])
    got = metrics.tar_at_far(genuine, impostor, 1e-3)
    assert got["far"] == pytest.approx(1e-3) and got["threshold"] == pytest.approx(0.85)
    assert got["tar"] == pytest.approx(3 / 4)  # 0.85, 0.95, 0.99 pass; 0.9 is the one impostor through
    assert metrics.tar_at_far(genuine, impostor, 2e-3)["tar"] == pytest.approx(1.0)
    assert metrics.tar_at_far(genuine, impostor, 0.99e-3)["tar"] == pytest.approx(2 / 4)


def test_far_levels_are_those_the_counts_resolve():
    assert metrics.far_levels(1000) == [0.1, 0.01, 0.001]
    assert metrics.far_levels(138_510) == [0.1, 0.01, 0.001, 1e-4, 1e-5]
    assert metrics.far_levels(5) == []


def test_eer_and_d_prime_by_hand():
    assert metrics.eer(GENUINE, IMPOSTOR) == {"eer": pytest.approx(1 / 3), "threshold": 0.7}
    crossing = metrics.eer([0.4, 0.8], [0.3, 0.6, 0.7])  # FRR - FAR: -1/6 at 0.6, +1/6 at 0.7
    assert crossing["eer"] == pytest.approx(0.5) and crossing["threshold"] == pytest.approx(0.65)
    assert metrics.d_prime([1.0, 3.0], [-1.0, 1.0]) == pytest.approx(2.0)
    assert metrics.d_prime([1.0, 3.0], [-1.0, 1.0], [1, 1], [1, 3]) == pytest.approx(
        metrics.d_prime([1.0, 3.0], [-1.0, 1.0, 1.0, 1.0]))


def test_probit():
    assert metrics.probit(0.5) == pytest.approx(0.0, abs=1e-9)
    assert metrics.probit(0.975) == pytest.approx(1.959963985, abs=1e-8)
    assert metrics.probit(0.001) == pytest.approx(-3.090232306, abs=1e-8)
    assert metrics.probit(1e-6) == pytest.approx(-4.753424309, abs=1e-7)
    assert np.isneginf(metrics.probit(0.0)) and np.isposinf(metrics.probit(1.0))


OPEN = np.array([[0.9, 0.1, 0.2],  # mate 0, ranked first at 0.9
                 [0.3, 0.7, 0.75],  # mate 1, ranked second: never identified
                 [0.2, 0.1, 0.6],  # mate 2, first at 0.6
                 [0.5, 0.4, 0.3],  # a stranger, best 0.5
                 [0.1, 0.8, 0.2]])  # a stranger, best 0.8
MATE = [0, 1, 2, -1, -1]


def test_open_set_by_hand():
    fpir, fnir, thresholds = metrics.open_set(OPEN, MATE)
    assert thresholds.tolist() == [math.inf, 0.9, 0.8, 0.6, 0.5]
    assert np.allclose(fpir, [0, 0, 0.5, 0.5, 1.0])
    assert np.allclose(fnir, [1, 2 / 3, 2 / 3, 1 / 3, 1 / 3])
    strict = metrics.fnir_at_fpir(OPEN, MATE, 0.0)
    assert strict["fnir"] == pytest.approx(2 / 3) and strict["threshold"] == 0.9
    half = metrics.fnir_at_fpir(OPEN, MATE, 0.5)
    assert half["fnir"] == pytest.approx(1 / 3) and half["dir"] == pytest.approx(2 / 3) and half["threshold"] == 0.6
    assert metrics.ranks(OPEN, MATE).tolist() == [1, 2, 1, 0, 0]
    assert np.allclose(metrics.cmc(OPEN, MATE, 3), [2 / 3, 1, 1])


def test_a_tie_with_another_person_is_not_rank_one():
    scores = np.array([[0.7, 0.7, 0.1]])
    assert metrics.ranks(scores, [0]).tolist() == [2]
    _, fnir, _ = metrics.open_set(scores, [0])
    assert fnir.min() == 1.0


def test_open_set_refuses_bad_searches():
    with pytest.raises(ValueError):
        metrics.open_set(OPEN, [0, 1, 2, -1])
    with pytest.raises(ValueError):
        metrics.open_set(OPEN, [0, 1, 3, -1, -1])
    with pytest.raises(ValueError):
        metrics.roc([0.5, float("nan")], [0.1])


def test_binomial_interval():
    assert metrics.binomial_interval(0, 100) == [0.0, pytest.approx(1 - 0.025 ** (1 / 100), abs=1e-9)]
    lo, hi = metrics.binomial_interval(5, 100)
    assert lo == pytest.approx(0.01643187, abs=1e-6) and hi == pytest.approx(0.11283491, abs=1e-6)
    assert metrics.binomial_interval(10, 10)[1] == 1.0


def test_bootstrap_counts_people_not_frames():
    """Ten people, a hundred frames each, all frames of a person alike: over people the interval is wide; a bootstrap
    over frames would make it about ten times narrower."""
    flags = np.repeat([True] * 5 + [False] * 5, 100)
    people = np.repeat([f"p{k}" for k in range(10)], 100)
    over_people = metrics.bootstrap_rate(flags, people, seed=1)
    over_frames = metrics.bootstrap_rate(flags, None, seed=1)
    assert over_people["rate"] == over_frames["rate"] == 0.5
    assert over_people["clusters"] == 10 and over_people["kind"] == "bootstrap"
    width = over_people["interval"][1] - over_people["interval"][0]
    assert width > 0.4 > 4 * (over_frames["interval"][1] - over_frames["interval"][0])


def test_a_rate_with_no_event_gets_the_exact_bound_over_clusters():
    flags = np.zeros(300, bool)
    people = np.repeat([f"p{k}" for k in range(30)], 10)
    r = metrics.bootstrap_rate(flags, people)
    assert r["rate"] == 0.0 and r["kind"] == "exact, clusters"
    assert r["interval"] == [0.0, pytest.approx(1 - 0.025 ** (1 / 30))]  # (30 people, not 300 tracks: 11.6 %)


def test_bootstrap_is_deterministic():
    flags = np.arange(40) % 3 == 0
    people = np.arange(40) // 4
    assert metrics.bootstrap_rate(flags, people, seed=4) == metrics.bootstrap_rate(flags, people, seed=4)


def test_pair_weights_by_hand():
    """A draw with person a twice, b never, c once: a's genuine pairs count 2, c's 1; the impostor pair (a, b) 0,
    (a, c) 2, (b, c) 0; a pair that is its own cluster (-1) counts as its own draw."""
    genuine_w, impostor_w = metrics.pair_weights([2, 0, 1, 3], [0, 2, 2], [[0, 1], [0, 2], [1, 2], [3, -1]])
    assert genuine_w.tolist() == [2, 1, 1] and impostor_w.tolist() == [0, 2, 0, 3]


def test_pairs_report_counts_people():
    rng = np.random.default_rng(0)
    people = [f"p{k}" for k in range(12)]
    genuine, g_people, impostor, i_people = [], [], [], []
    for a in people:
        genuine += list(rng.normal(0.6, 0.1, 3))
        g_people += [a] * 3
        for b in people:
            if a < b:
                impostor += list(rng.normal(0.1, 0.1, 2))
                i_people += [(a, b)] * 2
    report = metrics.pairs_report(genuine, impostor, g_people, i_people, draws=200)
    assert report["people"] == 12 and report["genuine"] == 36 and report["impostor"] == 132
    assert list(report["tar_at_far"]) == ["0.1", "0.01"]
    for key in ("d_prime", "eer"):
        lo, hi = report[f"{key}_interval"]
        assert lo <= report[key] <= hi
    lo, hi = report["tar_at_far"]["0.01"]["interval"]
    assert lo <= report["tar_at_far"]["0.01"]["tar"] <= hi
    again = metrics.pairs_report(genuine, impostor, g_people, i_people, draws=200)
    assert again == report


def test_open_set_report():
    report = metrics.open_set_report(OPEN, MATE, people=["a", "b", "c", "d", "e"], draws=100)
    assert report["searches"] == 5 and report["mated"] == 3 and report["nonmated"] == 2 and report["people"] == 5
    assert report["rank1"] == pytest.approx(2 / 3)
    assert list(report["fnir_at_fpir"]) == []  # two strangers resolve no rate of 10 % or less
    assert metrics.open_set_report(OPEN, MATE, levels=(0.5,), draws=50)["fnir_at_fpir"]["0.5"]["fnir"] == \
        pytest.approx(1 / 3)


def test_a_rate_at_an_edge_is_not_a_point():
    """Twenty people, three searches each, never missed, and every genuine pair accepted: the bootstrap's intervals
    would be points; they are the exact bounds over the twenty people (not over the sixty searches)."""
    rng = np.random.default_rng(3)
    people = [f"p{k}" for k in range(20)]
    mate = np.repeat(np.arange(10), 3).tolist() + [-1] * 30  # (ten enrolled, ten strangers)
    scores = rng.uniform(0.0, 0.3, (60, 10))
    scores[np.arange(30), mate[:30]] = rng.uniform(0.8, 1.0, 30)
    report = metrics.open_set_report(scores, mate, people=np.repeat(people, 3), levels=(0.1,), draws=100)
    edge = 1 - 0.025 ** (1 / 10)  # (the ten enrolled people searched)
    assert report["fnir_at_fpir"]["0.1"]["fnir"] == 0.0
    assert report["fnir_at_fpir"]["0.1"]["interval"] == [0.0, pytest.approx(edge)]
    assert report["rank1"] == 1.0 and report["rank1_interval"] == [pytest.approx(0.025 ** (1 / 10)), 1.0]
    genuine = scores[np.arange(30), mate[:30]]
    impostor = scores[30:].ravel()
    i_people = [(people[10 + k // 10 // 3], people[k % 10]) for k in range(len(impostor))]  # (stranger, enrolled)
    pairs = metrics.pairs_report(genuine, impostor, np.repeat(people[:10], 3), i_people, draws=100)
    assert pairs["tar_at_far"]["0.01"]["tar"] == 1.0
    assert pairs["tar_at_far"]["0.01"]["interval"] == [pytest.approx(0.025 ** (1 / 10)), 1.0]
    assert pairs["eer"] == 0.0 and pairs["eer_interval"] == [0.0, pytest.approx(edge)]


def _segment(person, known, shown=2.0, first=0.0, correct=None, named=None, wrong=(), unknown=None, end=10.0,
             frames=50, track=1):
    return Segment(person, person, track, known, 0.0, end, shown, first, correct, tuple(wrong), unknown,
                   named if named is not None else correct, frames)


def test_track_summary_counts_tracks_not_frames():
    """One known track named at once over many frames, one never named over few: half the tracks are named (a count
    of frames would say nearly all)."""
    segs = [_segment("Ada", True, correct=0.5, frames=500), _segment("Bob", True, frames=2)]
    summary = metrics.track_summary(segs)
    assert summary["known"]["named_within"]["2"]["rate"] == 0.5
    assert summary["known"]["named"]["rate"] == 0.5 and summary["known"]["eligible"] == 2


def test_track_summary_strangers_wrong_names_and_hours():
    segs = [_segment("Ada", True, correct=0.4), _segment("Bob", True, wrong=("Ada",), named=1.0),
            _segment("Cy", False, named=3.0, wrong=()), _segment("Di", False), _segment("Ed", False, shown=0.2)]
    s = metrics.track_summary(segs)
    assert s["strangers"]["eligible"] == 2 and s["strangers"]["all"] == 3
    assert s["strangers"]["fpir"]["rate"] == 0.5
    assert s["strangers"]["fpir_all"]["rate"] == pytest.approx(1 / 3)
    assert s["known"]["wrong_all"]["rate"] == 0.5
    assert s["wrong_names"] == 2 and s["wrong_per_track"] == pytest.approx(2 / 5)
    assert s["hours_followed"] == pytest.approx(50 / 3600, abs=1e-4)
    assert s["wrong_per_hour"] == pytest.approx(2 / (50 / 3600), rel=1e-3)


class _D:
    def __init__(self, state, name=None):
        self.state, self.name = state, name


def test_name_events_switches_and_one_name_on_two_tracks():
    events = metrics.NameEvents()
    events.update(0.0, {1: _D("named", "Ada"), 2: _D("undecided")})
    events.update(0.2, {1: _D("undecided"), 2: _D("named", "Bob")})
    events.update(0.4, {1: _D("named", "Bob"), 2: _D("named", "Bob")})
    events.update(0.6, {1: _D("named", "Bob"), 2: _D("unknown")})
    summary = events.summary()
    assert summary["switches"] == 1 and summary["tracks_switching"] == 1
    assert summary["shared_frames"] == 1 and summary["shared_seconds"] == pytest.approx(0.2)
    assert summary["shared"] == {"Bob": [1, 2]}


def test_achieved_against_asked():
    named = np.r_[np.zeros(990, bool), np.ones(10, bool)]
    groups = np.r_[np.full(500, "small"), np.full(500, "big")]
    people = np.arange(1000) // 10
    out = metrics.achieved(named, 0.01, groups, people)
    assert out["big"]["rate"] == pytest.approx(0.02) and out["big"]["ratio"] == pytest.approx(2.0)
    assert out["big"]["within"] == "yes"
    assert out["small"]["rate"] == 0.0 and out["small"]["within"] == "consistent"  # 50 people: up to 7 %
    many = metrics.achieved(np.zeros(5000, bool), 0.01, clusters=np.arange(5000))
    assert many["all"]["within"] == "no"  # 5,000 independent strangers, none named: far stricter than asked


def _summary(known_named, known_total, strangers_named, strangers_total, people=None):
    segs = [_segment(f"k{k}", True, correct=0.5 if k < known_named else None) for k in range(known_total)]
    segs += [_segment(f"s{k}" if people is None else f"s{k % people}", False,
                      named=1.0 if k < strangers_named else None) for k in range(strangers_total)]
    return metrics.track_summary(segs, within=(2.0,))


def test_acceptance_on_made_up_outcomes():
    ok = metrics.acceptance(_summary(20, 20, 0, 50), {"shared_frames": 0})
    assert ok["verdict"] == "pass" and not ok["confident"]  # 50 strangers cannot show 1 %
    assert ok["criteria"]["fpir_within_factor"]["status"] == "consistent"
    assert metrics.acceptance(_summary(18, 20, 0, 50))["verdict"] == "fail"  # 90 % named
    assert metrics.acceptance(_summary(20, 20, 0, 50), {"shared_frames": 3})["verdict"] == "fail"
    assert metrics.acceptance(_summary(20, 20, 10, 100))["verdict"] == "fail"  # 10 % of strangers named
    strict = metrics.acceptance(_summary(20, 20, 0, 2000))  # 2,000 strangers, none named: far stricter than asked
    assert strict["criteria"]["fpir_within_factor"]["status"] == "fail" and strict["verdict"] == "fail"
    one = metrics.acceptance(_summary(100, 100, 10, 1000), {"shared_frames": 0})
    assert one["verdict"] == "pass" and one["criteria"]["fpir_within_factor"]["status"] == "pass"


def test_cost_summary():
    cost = metrics.cost_summary({"embed": [10.0, 20.0, 30.0], "faces": []}, embeddings=6, seconds=3.0, frames=3)
    assert cost["stages"]["embed"]["median"] == 20.0 and "faces" not in cost["stages"]
    assert cost["embeddings_per_s"] == 2.0
    assert metrics.peak_memory_mb() > 1.0


def test_roc_matches_scikit_learn():
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(3)
    genuine = np.round(rng.normal(0.5, 0.2, 400), 3)  # (rounded: ties within and across the two sets)
    impostor = np.round(rng.normal(0.1, 0.2, 3000), 3)
    labels = np.r_[np.ones(len(genuine)), np.zeros(len(impostor))]
    fpr, tpr, thresholds = sklearn_metrics.roc_curve(labels, np.r_[genuine, impostor], drop_intermediate=False)
    far, tar, ours = metrics.roc(genuine, impostor)
    index = {round(float(t), 9): k for k, t in enumerate(ours)}
    rows = [index[round(float(t), 9)] for t in thresholds[1:]]  # (its first threshold accepts nothing, as our +inf)
    assert np.allclose(far[rows], fpr[1:]) and np.allclose(tar[rows], tpr[1:])
    assert len(rows) == len(ours) - 1  # every distinct score, as scikit-learn's curve without dropping points
    assert fpr[0] == tpr[0] == far[0] == tar[0] == 0.0
