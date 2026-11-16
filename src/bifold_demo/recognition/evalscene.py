"""Scene runs for the evaluation (P3, step E1): scripted fair scenes with known people and strangers, through F2's
whole pipeline (:class:`~.people.PeopleRecognizer`), scored per track against the scene's truth. Needs OpenGL.

**The scene** (``parade``, :func:`parade`): the fair's hall without its crowd, and a scripted one. ``lanes`` people
walk up the hall towards the camera, stop 3 to 7 m from it, watch it a few seconds, turn round, walk back and come
again (the fair's ``approach`` with ``watch``): their faces are shown while they come and while they watch, small
and then bigger. A few ``crossers`` walk across in front of them and hide them for a moment. Every timing is seeded.
A lane person is **known** (one of the run's enrolled identities: its face made from set A and fitted to it, worn by
its carrier avatar, as in P2's ``fitted`` stills) or a **stranger** (an avatar with its own face, not enrolled). The
gallery is the known identities' set B (P2's). A carrier is never enrolled in the same run (:mod:`.protocols`'
``p3`` picks the known halves so), since a carrier's head and hair show through a face it wears.

**The run** (:func:`run_scene`): deterministic (a fixed-step clock: the 2048 heads are read before each frame), the
profile's person detector (YOLO26n, or YOLOX-s for the commercial profile; ``truth``: one simulated from the scene's
truth), up to four windows a frame through the run's camera preset, the profile's embedding budget (or the run's
``rate``), the target false-alarm rate per stranger track ``fpir``. Per frame: the truth (each person's identity is
the face's, not the carrier's), :func:`~.truth.match_tracks`, :class:`~.truth.Outcomes`, :class:`~.truth.IdSwitches`,
:class:`~.metrics.NameEvents`, :class:`~.truth.FaceAttribution` and the stages' timings.

**What is scored** (:func:`score_runs`): per segment (one person followed by one track) as :func:`~.metrics.
track_summary` counts it, with intervals over people: known people named correctly within 2 and 5 s of their face
being shown (turned less than 45 degrees, eyes not hidden), the time to the first correct name, wrong names per track
and per hour, strangers named (the false-alarm rate per track) and called unknown; name switches and one name on two
tracks at once; the acceptance test (:func:`~.metrics.acceptance`).

**Other margins, replayed** (:func:`replay`). Every segment keeps its decisions' timeline: the candidate, its rarity,
the runner-up's, the accept rarity (the single decision's for the target plus the calibration's offset per stranger
track), the evidence's progress, and whether the face was shown. :func:`replay` decides the timeline again with the
accept rarity moved by ``delta`` decades (another margin of the per-track offset: :mod:`.margins`), the live rules
otherwise (the progress complete, the best rarity at the accept one, the runner-up a margin behind). Up to the moment
the looser rule names someone, the evidence is what the live run saw (an undecided track is scheduled the same
whatever its margin), so a replay with a looser margin than the run's is faithful to that moment, a stricter one is
not (a track named early gets fewer faces afterwards); runs are made with the strictest margin (the shipped one).
Hysteresis and one-name-per-person are not replayed: the replay with ``delta`` 0 is checked against the live outcome.
"""

from __future__ import annotations

import dataclasses
import math
import random
import time
from functools import partial

import numpy as np

from . import metrics

SCENES = ("parade",)
VIEW = (1280, 720)
STOP = (3.0, 7.0)  # metres from the camera where the lane people stop and watch it
SPREAD = 17.0  # degrees either side of the camera's axis the lanes take
START = (18.0, 26.0)  # metres from the camera the lanes start
CROSSING_Z = (2.5, 0.5, -1.5, 3.5)  # where the crossers walk across (the camera is at z 6, looking to -z)
SHOWN_EYES = 3.0  # (as Outcomes: a face shown has at least this many view pixels between its eyes)

__all__ = ["SCENES", "parade", "replay", "run_scene", "run_scenes", "score_runs"]


def _lanes(count: int, rng: random.Random, camera) -> list[tuple]:
    """``(start, stop)`` floor points of ``count`` lanes fanning out from the camera."""
    cx, _, cz = camera
    out = []
    for k in range(count):
        angle = math.radians(-SPREAD + 2 * SPREAD * (k + 0.5) / count + rng.uniform(-2.0, 2.0))
        stop = rng.uniform(*STOP)
        start = rng.uniform(*START)
        out.append(((cx + start * math.sin(angle), 0.0, cz - start * math.cos(angle)),
                    (cx + stop * math.sin(angle), 0.0, cz - stop * math.cos(angle))))
    return out


def parade(people: list[dict], seed: int, faces, library=None) -> tuple:
    """The ``parade`` scene (see the module docs) with ``people`` (``{identity, body, face, head, role}``; ``role``
    ``"lane"`` or ``"crosser"``), its timings seeded by ``seed``; ``faces``: a :class:`~.evalrender.FaceMaker` (the
    faces made from set A). Returns ``(scene, {scene name: identity})``."""
    from ..characters.behaviour import approach, walk_by
    from ..characters.crowd import populate
    from ..characters.headtex.person import wear_textures
    from ..scenes import fair

    library = library or faces.library
    rng = random.Random(f"parade {seed}")
    scene = fair.build_scene(people=[])
    camera = fair.CAMERA_POSITION
    lanes = [p for p in people if p.get("role", "lane") == "lane"]
    crossers = [p for p in people if p.get("role") == "crosser"]
    makers, names = [], []
    for p, (start, stop) in zip(lanes, _lanes(len(lanes), rng, camera)):
        makers.append(partial(approach, start, stop, camera=camera, stop_distance=(2.5, 8.0), bow=0.0, watch=camera,
                              idle=(3.0, 7.0), far_pause=(0.5, 3.0)))
        names.append(f"lane {p['identity']}")
    for k, p in enumerate(crossers):
        z = CROSSING_Z[k % len(CROSSING_Z)]
        makers.append(partial(walk_by, [(-7.0, 0.0, z), (7.0, 0.0, z)], camera=camera, return_mode="turn",
                              pause=(1.0, 4.0)))
        names.append(f"crosser {p['identity']}")
    ordered = lanes + crossers
    bodies = [p.get("body") or p["identity"] for p in ordered]
    characters = populate(library, bodies, makers, rng=rng.randrange(2 ** 31), shadow=True)
    who = {}
    for p, character, name in zip(ordered, characters, names):
        if p.get("face", "own") == "renders":
            model, textures = faces.person(p["identity"], p["body"], "renders", p.get("head", "fitted"),
                                           p.get("shots", "three-quarter"))
            if p.get("head", "fitted") == "fitted":
                character.use_model(model)  # (the carrier's head fitted to the identity's set A)
            wear_textures(character, textures)
        scene.add(character, name=name)
        who[name] = p["identity"]
    return scene, who


class _TruthDetector:
    """A person detector simulated from the scene's truth (boxes jittered by 2 % of their height, weak when half
    hidden, missing when three quarters hidden), seeded: for runs without the profile's detector."""

    def __init__(self, scene, seed: int = 0):
        self.scene, self.rng = scene, np.random.default_rng(seed)

    def detect(self, image, conf=None):
        from ..detection.results import Detection
        from .truth import frame_truth, simulated_detections

        truth = [p for p in frame_truth(self.scene, VIEW) if p.box[3] - p.box[1] >= 20]
        boxes, scores = simulated_detections(truth, self.rng, size=VIEW)
        return [Detection(tuple(b), float(s), 0, "person") for b, s in zip(boxes, scores)]


def person_detector(profile: str, scene, kind: str = "profile", yolo=None, models_folder=None, seed: int = 0):
    """The run's person detector: the profile's own (``kind`` ``"profile"``: YOLOX-s for the commercial profile, else
    YOLO26n from ``yolo``, its weights) or one simulated from the truth (``"truth"``)."""
    from . import registry

    if kind == "truth":
        return _TruthDetector(scene, seed)
    own = registry.profile(profile).people
    if own:
        from ..detection.yolox import YoloxDetector

        return YoloxDetector(own, models_folder=models_folder, download=False)
    from ..detection import YoloDetector

    return YoloDetector(str(yolo or "yolo26n.pt"), device="cpu")


def run_scene(run, item, profile: str, engine, rate: float | None = None, yolo=None, progress=None) -> dict:
    """One scene item of a manifest (``item.extra``: ``scene``, ``seconds``, ``fps``, ``seed``, ``people``, ``known``,
    ``windows``, ``preset``, ``detector``, ``fpir``, ``rate``; ``unverified_after``: the decisions' seconds a named
    track may go without a confirming face, default 0: no such limit) through the pipeline with ``engine`` (a
    profile's): its outcomes, name events, switches, faces' attribution, costs and each segment's decision timeline
    (see the module docs). ``run``: the :class:`~.evaluate.Run` (its gallery folders and caches); ``rate``:
    embeddings a second (default: the item's, else the profile's)."""
    from ..sensor import WindowService
    from ..sources import FixedStepClock, SceneSource
    from . import evaluate, protocols
    from .decision import DecisionLog, DecisionSettings
    from .evalrender import FaceMaker
    from .gallery import Gallery
    from .people import PeopleRecognizer
    from .truth import FaceAttribution, IdSwitches, Outcomes, frame_truth, match_tracks

    progress = progress or (lambda *_: None)
    p = dict(item.extra)
    if p.get("scene", "parade") not in SCENES:
        raise ValueError(f"unknown scene {p.get('scene')!r} (one of {', '.join(SCENES)})")
    known = sorted(p.get("known") or [])
    fps, seconds = float(p["fps"]), float(p["seconds"])
    gallery_items = evaluate.parse({"format": 1, "name": f"{item.id}-gallery",
                                    "items": protocols.gallery_items(known)}, run.out)
    gallery = Gallery(run.gallery_folder(gallery_items), engine)
    gallery.refresh(force=True)
    faces = FaceMaker(run.library, progress=progress)
    try:
        scene, who = parade(p["people"], int(p.get("seed", 0)), faces)
    finally:
        faces.close()
    service = WindowService(max_windows=4, preset=p.get("preset", "ideal")) if p.get("windows", True) else None
    source = SceneSource(scene, *VIEW, fps=fps, clock=FixedStepClock(fps), windows=service)
    source.start()
    detector = person_detector(profile, scene, p.get("detector", "profile"), yolo, run.models_folder,
                               int(p.get("seed", 0)))
    log = DecisionLog(keep=10)
    rate = rate if rate is not None else p.get("rate")
    recognizer = PeopleRecognizer(gallery, detector, windows=service, rate=rate, log=log,
                                  settings=DecisionSettings(fpir=float(p.get("fpir", 0.01)),
                                                            unverified_after=float(p.get("unverified_after", 0.0))),
                                  models_folder=run.models_folder)
    outcomes, switches, names, given = Outcomes(known), IdSwitches(), metrics.NameEvents(), FaceAttribution()
    timeline: dict = {}
    timings: dict = {}
    embedded = 0
    started = time.perf_counter()
    frames = round(seconds * fps)
    try:
        for k in range(frames):
            frame = source.read()
            result = recognizer.process(frame)
            truth = [dataclasses.replace(t, avatar=who.get(t.name, t.avatar)) for t in frame_truth(scene, VIEW)]
            matched = match_tracks(truth, result.people)
            decisions = {person.id: person.decision for person in result.people}
            outcomes.update(result.time, truth, matched, decisions)
            switches.update([t for t in truth if t.covered <= 0.75 and t.box[3] - t.box[1] >= 40], matched,
                            result.time)
            names.update(result.time, decisions)
            given.update(truth, matched, result.people)
            embedded += sum(person.embedded for person in result.people)
            for key, value in result.timings.items():
                timings.setdefault(key, []).append(value)
            for t in truth:
                track = matched.get(t.name)
                d = decisions.get(track) if track is not None else None
                if d is None:
                    continue
                shown = t.frontal and not t.eyes_covered and t.eye_distance >= SHOWN_EYES
                timeline.setdefault(f"{t.name}|{track}", []).append(
                    [round(result.time, 4), bool(shown), d.candidate, _r(d.calibrated), _r(d.runner_up_calibrated),
                     _r(d.accept_rarity), round(float(d.progress), 3), d.state, d.name, round(t.eye_distance, 2),
                     bool(getattr(d, "confirmed", True))])
            if k % 50 == 0:
                progress(f"[eval] {item.id} {profile}: t={result.time:5.1f} s, {len(result.people)} tracks, "
                         f"{embedded} embedded ({time.perf_counter() - started:.0f} s)")
    finally:
        source.stop()
    identity_of = {name: identity for name, identity in who.items()}
    segments = [dataclasses.asdict(g) | {"wrong": list(g.wrong)} for g in outcomes.segments.values()]
    return {"id": item.id, "profile": profile, "model": recognizer.model, "rate": recognizer.budget.rate,
            "calibration": {"digest": recognizer.calibration.digest, "path": recognizer.calibration.path,
                            "offset": recognizer.calibration.track_offset(float(p.get("fpir", 0.01)))},
            "fpir": float(p.get("fpir", 0.01)), "known": known, "people": identity_of, "seconds": seconds,
            "fps": fps, "frames": frames, "segments": segments, "names": names.summary(),
            "switches": {"swaps": switches.swaps, "returns": switches.returns, "coverage": switches.coverage},
            "attribution": given.summary(), "cost": metrics.cost_summary(timings, embedded, seconds, frames)
            | {"peak_memory_mb": metrics.peak_memory_mb(), "wall_s": round(time.perf_counter() - started, 1)},
            "timeline": timeline}


def _r(value):
    return None if value is None or not np.isfinite(value) else round(float(value), 4)


def _segments(results) -> list:
    """The :class:`~.truth.Segment` items of runs (their JSON form back)."""
    from .truth import Segment

    out = []
    for r in results:
        for g in r["segments"]:
            g = dict(g)
            g["wrong"] = tuple(g.get("wrong") or ())
            out.append(Segment(**g))
    return out


def score_runs(results: list, acceptance_rule=None, draws: int = metrics.DRAWS, seed: int = 0) -> dict:
    """The track-level measures of one profile's runs pooled (see the module docs): ``summary``
    (:func:`~.metrics.track_summary`, intervals over people), ``names`` summed, ``acceptance`` and ``replayed``: the
    replay with the run's own margin against the live outcome (how far the replay can be trusted)."""
    segments = _segments(results)
    summary = metrics.track_summary(segments, within=(2.0, 5.0), draws=draws, seed=seed)
    names = {"shared_frames": sum(r["names"]["shared_frames"] for r in results),
             "shared_seconds": sum(r["names"]["shared_seconds"] for r in results),
             "switches": sum(r["names"]["switches"] for r in results),
             "tracks_switching": sum(r["names"]["tracks_switching"] for r in results)}
    out = {"runs": len(results), "summary": summary, "names": names,
           "switches": {k: sum(r["switches"][k] for r in results) for k in ("swaps", "returns")},
           "attribution": {k: sum((r["attribution"].get(k) or 0) for r in results)
                           for k in ("given", "right", "wrong", "embedded_wrong")}}
    out["acceptance"] = metrics.acceptance(summary, names, acceptance_rule)
    live = {(g.person, g.track): g for g in segments}
    agree = total = 0
    for r in results:
        for key, rows in r["timeline"].items():
            person, track = key.rsplit("|", 1)
            g = live.get((person, int(track)))
            if g is None:
                continue
            again = replay(rows, 0.0)
            total += 1
            agree += (again["named_at"] is None) == (g.named_at is None) and \
                (again["named_at"] is None or abs(again["named_at"] - g.named_at) <= 1.0 + 1e-9)
    out["replay_agrees"] = {"segments": total, "agree": agree}
    return out


def replay(rows: list, delta: float, margin: float = 1.0) -> dict:
    """A segment's decisions taken again with the accept rarity moved by ``delta`` decades (see the module docs):
    ``{first_shown, named_at, name, correct_at, shown}`` (``correct_at`` needs the caller's identity: the name given
    first is ``name``; :func:`replayed` scores a run)."""
    first_shown, shown, last, named_at, name = None, 0.0, None, None, None
    names = []
    for t, is_shown, candidate, best, runner, accept, progress, *_rest in rows:
        if is_shown:
            if first_shown is None:
                first_shown = t
            elif last is not None:
                shown += t - last
        last = t
        if best is None or accept is None or progress < 1.0:
            continue
        if best >= accept + delta - 1e-9 and best - (runner if runner is not None else 0.0) >= margin - 1e-9:
            if named_at is None:
                named_at, name = t, candidate
            if candidate not in names:
                names.append(candidate)
    return {"first_shown": first_shown, "shown": shown, "named_at": named_at, "name": name, "names": names}


def replayed(results: list, delta: float, margin: float = 1.0, shown: float = 1.0, within=(2.0, 5.0),
             draws: int = metrics.DRAWS, seed: int = 0) -> dict:
    """Runs' segments decided again with the accept rarity moved by ``delta`` decades (:func:`replay`), scored as
    :func:`~.metrics.track_summary` scores them (the replayed names; a known person's first replayed name right or
    wrong)."""
    from .truth import Segment

    segments = []
    for r in results:
        known = set(r["known"])
        identity = r["people"]
        for key, rows in r["timeline"].items():
            person, track = key.rsplit("|", 1)
            got = replay(rows, delta, margin)
            who = identity.get(person, person)
            right = [n for n in got["names"] if n == who]
            wrong = tuple(n for n in got["names"] if n != who)
            correct_at = None
            if right:
                correct_at = next(row[0] for row in rows if row[2] == who and row[3] is not None and row[5] is not None
                                  and row[6] >= 1.0 and row[3] >= row[5] + delta - 1e-9
                                  and row[3] - (row[4] if row[4] is not None else 0.0) >= margin - 1e-9)
            segments.append(Segment(person, who, int(track), who in known, rows[0][0], rows[-1][0], got["shown"],
                                    got["first_shown"], correct_at, wrong, None, got["named_at"], len(rows)))
    return metrics.track_summary(segments, shown=shown, within=within, draws=draws, seed=seed)


def run_scenes(run, manifest, rate: float | None = None, yolo=None, acceptance: bool = False, rule=None,
               done=None) -> dict:
    """Every profile of ``run`` through every scene item of ``manifest``: ``{profile: {"runs": [...], "score":
    score_runs(...)}}`` (the profile keyed with ``@<rate>/s`` when ``rate`` is given); ``done``: called with the
    results so far after each run."""
    out = {}
    for profile, engine in run.engines.items():
        key = profile if rate is None else f"{profile}@{rate:g}/s"
        results = []
        for item in manifest.of_kind("scene"):
            run.progress(f"[eval] {manifest.name}: {item.id} with the {profile} profile")
            results.append(run_scene(run, item, profile, engine, rate, yolo, run.progress))
            out[key] = {"runs": results}
            if done is not None:
                done(out)
        out[key]["score"] = score_runs(results, rule, run.draws, run.seed)
        if done is not None:
            done(out)
    return out
