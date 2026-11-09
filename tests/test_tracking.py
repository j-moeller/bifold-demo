"""Person tracks: the assignment solver, the ByteTrack-style tracker on hand-made boxes, and the tracker in the fair
with its people's truth (boxes from their posed skeletons, a detector simulated by how much nearer people cover
them). The fair test needs the prepared avatars (``BIFOLD_CHARACTERS_TEST_DIR``), not OpenGL."""

import itertools
import os
from pathlib import Path

import numpy as np
import pytest

from bifold_demo.detection.assignment import linear_sum_assignment, match
from bifold_demo.detection.tracker import LOST, PersonTracker, TrackerSettings, iou
from bifold_demo.recognition.truth import IdSwitches, PersonTruth, match_tracks

CHARACTERS = Path(os.environ.get("BIFOLD_CHARACTERS_TEST_DIR", "/nonexistent")).expanduser()


# -- the assignment ---------------------------------------------------------------------------------------------------


def _brute(cost):
    n, m = cost.shape
    if n <= m:
        return min(sum(cost[i, p[i]] for i in range(n)) for p in itertools.permutations(range(m), n))
    return _brute(cost.T)


def test_the_assignment_is_the_cheapest_on_random_matrices():
    rng = np.random.default_rng(0)
    for _ in range(300):
        n, m = rng.integers(1, 6, size=2)
        cost = rng.integers(0, 6, size=(n, m)).astype(float) if rng.random() < 0.5 else rng.normal(size=(n, m))
        rows, cols = linear_sum_assignment(cost)
        assert len(rows) == min(n, m) and len(set(rows)) == len(rows) and len(set(cols)) == len(cols)
        assert list(rows) == sorted(rows)
        assert cost[rows, cols].sum() == pytest.approx(_brute(cost))


def test_the_assignment_is_deterministic_and_takes_empty_matrices():
    cost = np.ones((4, 4))  # every assignment costs the same: the first indices win
    rows, cols = linear_sum_assignment(cost)
    assert sorted(cols) == [0, 1, 2, 3]
    assert [a.tolist() for a in linear_sum_assignment(cost)] == [rows.tolist(), cols.tolist()]
    assert [len(a) for a in linear_sum_assignment(np.zeros((0, 3)))] == [0, 0]
    with pytest.raises(ValueError):
        linear_sum_assignment([[1.0, np.inf]])


def test_matching_leaves_pairs_over_the_limit_unpaired():
    cost = np.array([[0.1, 0.9], [0.3, 0.95]])
    pairs, rows, cols = match(cost, 0.5)
    assert pairs == [(0, 0)] and rows == [1] and cols == [1]
    # an expensive pair does not push a cheap one aside: solving first and dropping afterwards would pair (0, 1) and
    # (1, 0) (0.85 in all, below 0.1 + 0.9), both under the limit, and lose row 0's good match; with the unpaired
    # rows and columns at limit / 2 each, (0, 0) alone costs 0.1 + 0.5
    cost = np.array([[0.1, 0.45], [0.4, 0.9]])
    rows, cols = linear_sum_assignment(cost)
    assert list(zip(rows.tolist(), cols.tolist())) == [(0, 1), (1, 0)]
    assert match(cost, 0.5)[0] == [(0, 0)]
    assert match(np.array([[np.inf, 0.2]]), 1.0)[0] == [(0, 1)]
    assert match(np.zeros((0, 2)), 1.0) == ([], [], [0, 1])


# -- the tracker on hand-made boxes -------------------------------------------------------------------------------------


def _walk(start, velocity, t, size=(60, 160)):
    x, y = start[0] + velocity[0] * t, start[1] + velocity[1] * t
    return np.array([x, y, x + size[0], y + size[1]])


def _run(tracker, frames, fps=30.0):
    """``frames``: a list of ``(boxes, scores)``; returns the ids found per frame (in the order of the boxes that
    match them best)."""
    out = []
    for k, (boxes, scores) in enumerate(frames):
        tracks = tracker.update(boxes, scores, k / fps, k)
        out.append({t.id: t.box for t in tracks})
    return out


def _owner(found, box):
    """The id whose box overlaps ``box`` most (None: none overlaps by half)."""
    if not found:
        return None
    ids = list(found)
    overlap = iou([box], [found[i] for i in ids])[0]
    return ids[int(overlap.argmax())] if overlap.max() >= 0.5 else None


def test_one_person_walking_keeps_one_number():
    tracker = PersonTracker()
    frames = [([_walk((100, 200), (90, 0), k / 30)], [0.9]) for k in range(60)]
    found = _run(tracker, frames)
    assert all(list(f) == [1] for f in found)
    track = tracker.active()[0]
    assert track.velocity[0] == pytest.approx(3.0, abs=0.3)  # pixels a frame, from 90 a second


def test_a_single_false_box_never_becomes_a_track():
    tracker = PersonTracker()
    person = [_walk((100, 200), (0, 0), 0)]
    frames = [(person, [0.9])] * 5 + [(person + [np.array([600, 100, 650, 240])], [0.9, 0.8])] + [(person, [0.9])] * 5
    found = _run(tracker, frames)
    assert all(list(f) == [1] for f in found)
    assert tracker._next_id == 3  # the false box had a tentative number, never shown and never given again


def test_two_people_crossing_keep_their_numbers():
    """Two people cross; while one walks behind the other, the hidden one's box is weak (half covered) and then
    missing for a few frames: the second pass keeps them on the weak box, the lost track picks them up after."""
    tracker = PersonTracker()
    frames, truth = [], []
    for k in range(90):
        t = k / 30
        front = _walk((100, 220), (150, 0), t, (70, 190))  # nearer: bigger, lower
        back = _walk((520, 200), (-150, 0), t, (55, 150))
        overlap = iou([back], [front])[0, 0]
        boxes, scores = [front], [0.9]
        if overlap < 0.15:
            boxes.append(back)
            scores.append(0.85)
        elif overlap < 0.45:
            boxes.append(back)
            scores.append(0.3)  # weak: partly hidden
        frames.append((boxes, scores))
        truth.append((front, back))
    found = _run(tracker, frames)
    a = {_owner(f, front) for f, (front, _) in zip(found, truth)} - {None}
    b = {_owner(f, back) for f, (_, back) in zip(found, truth)} - {None}
    assert a == {1} and b == {2}, (a, b)


def test_a_person_hidden_for_half_a_second_is_found_again_but_not_after_two():
    for gap, same in ((15, True), (60, False)):
        tracker = PersonTracker()
        frames = []
        for k in range(40 + gap):
            box = _walk((100, 200), (60, 0), k / 30)
            frames.append(([box], [0.9]) if not 20 <= k < 20 + gap else ([], []))
        found = _run(tracker, frames)
        before, after = found[19], found[-1]
        assert list(before) == [1] and len(after) == 1
        assert (list(after) == [1]) == same, (gap, after)
        if same:
            assert tracker.active()[0].found_again == 1
    assert tracker.lost == [] and tracker.removed == []


def test_uneven_frame_times_follow_the_motion():
    """A detector on a worker thread skips frames: the steps between updates vary, the model counts them in time."""
    tracker = PersonTracker()
    times = np.cumsum([0.0] + [0.033, 0.2, 0.033, 0.1, 0.3, 0.033, 0.25] * 4)
    for k, t in enumerate(times):
        tracks = tracker.update([_walk((50, 200), (120, 0), t)], [0.9], float(t), k)
        assert [tr.id for tr in tracks] == [1]
    predicted = tracker.kalman.predict(tracker.active()[0].mean, tracker.active()[0].covariance, 0.3 * 30)[0]
    assert predicted[0] == pytest.approx(_walk((50, 200), (120, 0), times[-1] + 0.3)[[0, 2]].mean(), abs=6)


def test_the_tracker_is_deterministic():
    rng = np.random.default_rng(3)
    frames = []
    for k in range(50):
        boxes = [_walk((50 + 120 * j, 200 + 10 * j), (40 * (-1) ** j, 0), k / 30) + rng.normal(0, 2, 4)
                 for j in range(5)]
        scores = rng.uniform(0.2, 0.95, 5)
        frames.append((boxes, scores))
    first = _run(PersonTracker(), frames)
    again = _run(PersonTracker(), frames)
    assert [{i: tuple(np.round(b, 9)) for i, b in f.items()} for f in first] == \
        [{i: tuple(np.round(b, 9)) for i, b in f.items()} for f in again]


def test_lost_tracks_are_kept_about_a_second():
    tracker = PersonTracker(TrackerSettings(lost_for=1.0))
    tracker.update([[0, 0, 50, 150]], [0.9], 0.0)
    tracker.update([[2, 0, 52, 150]], [0.9], 0.033)
    tracker.update([], [], 0.066)
    assert [t.state for t in tracker.tracks] == [LOST]
    tracker.update([], [], 1.0)
    assert len(tracker.tracks) == 1
    tracker.update([], [], 1.1)
    assert tracker.tracks == [] and [t.id for t in tracker.removed] == [1]


# -- scoring against the truth ----------------------------------------------------------------------------------------


def _person(name, box):
    return PersonTruth(name, name, box, 3.0, ((0, 0), (1, 0)), 10.0, 0.0, 0.0, 0.0)


def test_identity_switches_are_counted_against_the_truth():
    class T:
        def __init__(self, i, box):
            self.id, self.box = i, box

    switches = IdSwitches(gap=1.0)
    a, b = _person("a", (0, 0, 10, 30)), _person("b", (50, 0, 60, 30))
    assert switches.update([a, b], match_tracks([a, b], [T(1, a.box), T(2, b.box)]), 0.0) == 0
    assert switches.update([a, b], match_tracks([a, b], [T(1, a.box)]), 0.1) == 0  # b not followed: no switch (yet)
    assert switches.update([a, b], match_tracks([a, b], [T(1, a.box), T(3, b.box)]), 2.0) == 1  # b back as 3
    assert switches.update([a, b], match_tracks([a, b], [T(2, a.box), T(1, b.box)]), 2.1) == 2  # a and b swapped
    assert (switches.switches, switches.swaps, switches.returns) == (3, 2, 1)
    assert switches.tracks == {"a": [1, 2], "b": [2, 3, 1]}
    assert switches.coverage == pytest.approx(7 / 8)


# -- the fair ---------------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fair_scene():
    if not (CHARACTERS / "avatars").is_dir():
        pytest.skip(f"no prepared characters in {CHARACTERS} (set BIFOLD_CHARACTERS_TEST_DIR)")
    from bifold_demo.characters.library import CharacterLibrary

    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("BIFOLD_CHARACTERS", str(CHARACTERS))
        if hasattr(CharacterLibrary, "shared"):
            patch.setattr(CharacterLibrary, "shared", lambda: CharacterLibrary(CHARACTERS))
        from bifold_demo.scenes import fair

        scene = fair.build_scene()
    if sum(getattr(obj, "skinned_model", None) is not None for obj in scene) < 20:
        pytest.skip("the fair has no 3D people (the library lacks walk or idle clips)")
    return scene


def test_people_crossing_and_hiding_each_other_in_the_fair_keep_their_tracks(fair_scene):
    """30 s of the fair at 10 frames a second: 30-odd people, walkers crossing in front of others, the audience in
    front of the camera, visitors coming and going. The detector is simulated from the truth: boxes jittered, weak
    when half hidden by nearer people, missing when mostly hidden. Identity switches are counted against who is
    where (the truth is never shown to the tracker): swaps (a person taken over by another track) are rare; people
    out of sight for longer than the lost tracks are kept come back with a new number, by design."""
    from bifold_demo.recognition.truth import frame_truth, simulated_detections

    size = (1280, 720)
    rng = np.random.default_rng(11)
    tracker = PersonTracker()
    switches = IdSwitches()
    crossings = 0
    fps = 10.0
    for k in range(300):
        t = k / fps
        fair_scene.update(t, 1.0 / fps if k else 0.0)
        truth = [p for p in frame_truth(fair_scene, size) if p.box[3] - p.box[1] >= 40]  # people a detector finds
        crossings += sum(p.covered > 0.3 for p in truth)
        boxes, scores = simulated_detections(truth, rng, size=size)
        tracks = tracker.update(boxes, scores, t, k)
        switches.update([p for p in truth if p.covered <= 0.75], match_tracks(truth, tracks), t)
    assert crossings > 100  # people do hide each other here
    assert switches.coverage > 0.88, switches.coverage
    # measured: 5 swaps and 29 returns (30 s, seed 11); a tracker without its second pass or its motion model loses
    # far more (see the results note of F2)
    assert switches.swaps <= 8, (switches.swaps, {n: ids for n, ids in switches.tracks.items() if len(ids) > 1})
    assert switches.returns <= 40, switches.returns
