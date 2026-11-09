"""Open-set decisions per person track: a name, "unknown", or "not decided yet", never a guess.

:class:`IdentityPolicy` decides, for every track with evidence (:class:`~.evidence.TrackEvidence`), whom it shows. It
sits above the gallery (:class:`~.gallery.Gallery`): a track's evidence is one aggregate embedding, scored once against
every known person (:meth:`~.gallery.Gallery.raw_scores`, each person's best photo or pose template), normalised
against a cohort when the calibration says so (:class:`~.calibration.Cohort`), and turned into **rarities** by the
model's calibration for the aggregate's face size and yaw (:class:`~.calibration.Calibration`): ``-log10`` of how
often a stranger scores that high, the same currency for every model and face size. With ``n`` known people and the
user's target false-alarm rate per decision ``fpir``, a name needs the rarity ``accept = -log10(per-person rate)``
(about ``log10(n / fpir)``: 3 for 10 people at 1 %).

**Three states** (:data:`NAMED`, :data:`UNKNOWN`, :data:`UNDECIDED`):

* **a name** when the best person's score reaches ``accept``, beats the runner-up by ``margin`` (in rarity: the
  runner-up's score is that many decades more common among strangers) and enough good evidence has been seen (summed
  quality weight ``min_weight``, ``min_frames`` frames);
* **unknown** when enough good evidence has been seen and the best score stays below the **reject** threshold (the
  score a known person's face of that quality falls below only ``reject_miss`` of the time: the calibration's mated
  curve) for ``sustain_unknown`` seconds, in a bin where that threshold is *decisive*: above all but a tenth of
  strangers' scores (a rarity of ``decisive``). Where known and unknown faces cannot be told apart that well (turned
  or tiny faces), nothing is called unknown. The best score must also be one that strangers' best scores reach at
  least ``stranger_like`` of the time (with the gallery's size): a face unlike everyone known *and* unlike strangers
  (a known person's face from photos painted on an avatar, whose own scores are lower than the calibration's renders
  give) is "not decided yet", not "unknown";
* **not decided yet** otherwise, with how far along it is (the evidence gathered against what is needed) and why
  (no face yet, too little evidence, between the thresholds, contested by another track).

**Hysteresis.** A named track keeps its name while its score for that name stays at the rarity ``accept - keep``;
switching to another name (or to "unknown") needs that name's ``accept`` and ``margin`` sustained for ``sustain``
seconds, and a track that lost a name needs the same before it takes another one (the wait starts again whenever the
evidence changes its mind). "Unknown" stays while no name comes
within ``keep`` of ``accept``. **One identity per track, one track per identity**: the names proposed (and those
held within their keep threshold) are given by an assignment between tracks and known people at the highest summed
rarity (a track holding a name counts ``incumbent`` more for it), and a track whose claim loses goes back to "not
decided yet" ("contested").

**Someone else on the track.** A track can jump to another person (two people crossed, a lost track was picked up by
someone else). So every new face is first compared with the track's own evidence (:meth:`IdentityPolicy.
consistent`): a score below what a known person's face of that quality reaches all but ``switch_miss`` (0.5 %) of the
time, where that floor is decisive, means someone else. The track's old evidence is then dropped and its name
withdrawn (:meth:`IdentityPolicy.reset`) before the new face counts.

Every decision is a :class:`Decision` with the raw, normalised and calibrated scores of the best person and the
runner-up, the thresholds, the model and the evidence; :class:`DecisionLog` keeps them (and writes them as JSON lines
for the evaluation, E1).
"""

from __future__ import annotations

import json
import math
import threading
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..detection.assignment import match
from .calibration import Calibration, Cohort, per_person

NAMED, UNKNOWN, UNDECIDED = "named", "unknown", "undecided"

__all__ = ["NAMED", "UNDECIDED", "UNKNOWN", "Decision", "DecisionLog", "DecisionSettings", "IdentityPolicy",
           "Scores"]


@dataclass(frozen=True)
class DecisionSettings:
    """The policy's settings (see the module docs). ``fpir``: the target false-alarm rate per decision; ``margin``,
    ``keep``: decades of rarity; ``reject_miss``: how often a known person's face may score below the reject
    threshold; ``switch_miss``: how often a new face of the same person may score below the floor that says the track
    shows someone else; ``decisive``: the rarity the reject threshold (and that floor) needs among strangers' scores;
    ``stranger_like``: how often strangers' best scores must reach a best score called unknown;
    ``min_weight``, ``min_frames``: enough good evidence (summed quality weights, :attr:`~.quality.Quality.weight`:
    two frontal faces of 24 pixels between the eyes, or one of 40, and at least two frames; the calibration's bins
    already count how good the faces are); ``sustain``: seconds a switch to another name or to
    unknown must hold; ``sustain_unknown``: seconds the score must stay below the reject threshold (from not decided);
    ``incumbent``: decades a track holding a name counts more for it in the assignment."""

    fpir: float = 0.01
    margin: float = 1.0
    keep: float = 1.0
    reject_miss: float = 0.02
    switch_miss: float = 0.005
    decisive: float = 1.0
    stranger_like: float = 0.25
    min_weight: float = 0.5
    min_frames: int = 2
    sustain: float = 1.0
    sustain_unknown: float = 1.0
    incumbent: float = 1.0

    def accept(self, people: int) -> float:
        """The rarity a name needs with ``people`` known people."""
        return -math.log10(per_person(self.fpir, people))


@dataclass(frozen=True)
class Scores:
    """A track's aggregate scored against the gallery: per known person (best first) the ``raw`` cosine, the
    ``normalised`` score and its ``rarity``; and the thresholds for its bin (in normalised score units): ``accept``,
    ``keep``, ``reject`` (``None``: the calibration has no mated curve there) and whether the reject threshold is
    ``decisive`` (see the module docs)."""

    names: tuple
    raw: tuple
    normalised: tuple
    rarity: tuple
    accept: float
    keep: float
    reject: float | None
    bin: str
    decisive: bool = False

    def of(self, name: str) -> tuple[float, float, float] | None:
        """``(raw, normalised, rarity)`` of ``name`` (``None``: not known)."""
        if name not in self.names:
            return None
        k = self.names.index(name)
        return self.raw[k], self.normalised[k], self.rarity[k]


@dataclass(frozen=True)
class Decision:
    """One decision about a track (see the module docs): its ``state`` (:data:`NAMED`, :data:`UNKNOWN`,
    :data:`UNDECIDED`), ``name`` (when named), ``reason`` and ``progress`` (0-1, how far along a track not decided yet
    is), the best ``candidate`` and the ``runner_up`` with their raw, normalised and calibrated (rarity) scores, the
    thresholds of the face's bin (``accept``, ``keep``, ``reject``, in normalised score units, and ``accept_rarity``),
    the ``model``, the ``evidence`` (frames, summed weight, face size, yaw, and per frame its time, weight, size, yaw
    and source), the ``time`` and ``frame`` it was made at, and whether the state ``changed``."""

    track: int
    time: float
    state: str
    name: str | None = None
    reason: str = ""
    progress: float = 0.0
    frame: int | None = None
    model: str = ""
    candidate: str | None = None
    raw: float | None = None
    score: float | None = None
    calibrated: float | None = None
    runner_up: str | None = None
    runner_up_raw: float | None = None
    runner_up_score: float | None = None
    runner_up_calibrated: float | None = None
    accept: float | None = None
    keep: float | None = None
    reject: float | None = None
    accept_rarity: float | None = None
    bin: str = ""
    normalisation: str = "none"
    evidence: dict = field(default_factory=dict)
    changed: bool = False

    @property
    def label(self) -> str:
        """What the overlay shows: the name, ``unknown`` or ``?`` with how far along (``? 40%``)."""
        if self.state == NAMED:
            return str(self.name)
        if self.state == UNKNOWN:
            return "unknown"
        return f"? {round(100 * self.progress)}%" if self.progress < 1.0 else f"? {self.reason}"

    def to_json(self) -> dict:
        return asdict(self)


class DecisionLog:
    """The decisions made, newest last: the last ``keep`` in memory (``entries``), and every one (or only the changes,
    ``changes_only``) as a line of JSON in ``path`` when given. Thread safe."""

    def __init__(self, path=None, keep: int = 10000, changes_only: bool = False):
        self.path = None if path is None else Path(path)
        self.entries: deque = deque(maxlen=keep)
        self.changes_only = changes_only
        self._lock = threading.Lock()
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, decision: Decision) -> None:
        with self._lock:
            self.entries.append(decision)
            if self.path is not None and (decision.changed or not self.changes_only):
                with self.path.open("a") as out:
                    out.write(json.dumps(decision.to_json(), default=_plain) + "\n")

    def changes(self) -> list[Decision]:
        with self._lock:
            return [d for d in self.entries if d.changed]


def _plain(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


@dataclass
class _Track:
    """What the policy remembers of a track."""

    state: str = UNDECIDED
    name: str | None = None
    since: float = 0.0  # when the state began
    pending: tuple | None = None  # (state, name, since): a switch waiting to be sustained
    dropped: tuple | None = None  # (name, time) of the last name it lost
    reset_at: float | None = None  # when it was last found to show someone else
    last: Decision | None = None


class IdentityPolicy:
    """The open-set decisions (see the module docs) for a ``gallery`` of one model, with its ``calibration`` (a
    :class:`~.calibration.Calibration`), a ``cohort`` when the calibration normalises, ``settings`` and a ``log``.
    Not thread safe: the recognition's worker thread uses it."""

    def __init__(self, gallery, calibration: Calibration, cohort: Cohort | None = None,
                 settings: DecisionSettings | None = None, log: DecisionLog | None = None):
        self.gallery = gallery
        self.calibration = calibration
        self.cohort = cohort
        self.settings = settings or DecisionSettings()
        self.log = log
        if calibration.normalisation != "none" and cohort is None:
            raise ValueError(f"the calibration of {calibration.model} normalises scores ({calibration.normalisation})"
                             " but no cohort was given")
        made_with = (calibration.made.get("cohort") or {}).get("digest")
        if cohort is not None and made_with and cohort.digest != made_with:
            raise ValueError(f"the cohort given ({cohort.digest}) is not the one the calibration of "
                             f"{calibration.model} was made with ({made_with})")
        self.tracks: dict[int, _Track] = {}
        self._gallery_stats = None  # (gallery rows id, means, spreads) for AS-norm

    @property
    def normalisation(self) -> str:
        return self.calibration.normalisation

    # -- scores ----------------------------------------------------------------------------------------------------

    def score(self, vector, size: float, yaw: float, model: str | None = None) -> Scores:
        """The aggregate ``vector`` (a face ``size`` pixels between the eyes, turned ``yaw`` degrees) scored against
        every known person (see the module docs)."""
        vector = np.asarray(vector, dtype=np.float64)
        lock = getattr(self.gallery, "_lock", None)
        if lock is not None:
            with lock:  # (one state of the gallery for the scores and the normalisation's rows)
                raw = self.gallery.raw_scores(vector, model)
                rows_of_gallery = self._gallery_rows() if self.normalisation != "none" else None
        else:
            raw = self.gallery.raw_scores(vector, model)
            rows_of_gallery = None
        names = raw.names
        rows = np.concatenate([np.ravel(raw.photos), np.ravel(raw.templates)])
        owners = np.concatenate([raw.photo_owners, raw.template_owners]).astype(int)
        normalised = rows
        if self.normalisation != "none" and len(rows):
            normalised = self.cohort.normalise(rows, vector, rows_of_gallery, self.normalisation,
                                               gallery_stats=self._stats(rows_of_gallery))
        best_raw = np.full(len(names), -math.inf)
        best_norm = np.full(len(names), -math.inf)
        for k in range(len(names)):
            mine = owners == k
            if mine.any():
                j = int(np.argmax(normalised[mine]))
                best_norm[k] = float(normalised[mine][j])
                best_raw[k] = float(rows[mine][j])
        curve = self.calibration.curve(size, yaw)
        rarity = np.array([curve.rarity(v) if np.isfinite(v) else 0.0 for v in best_norm])
        order = np.lexsort((np.arange(len(names)), -best_norm))
        s = self.settings
        accept_rarity = s.accept(len(names))
        accept = curve.threshold(10.0 ** -accept_rarity)
        keep = curve.threshold(10.0 ** -(accept_rarity - s.keep))
        reject = curve.mated_quantile(s.reject_miss)
        decisive = reject is not None and curve.rarity(reject) >= s.decisive
        return Scores(tuple(names[k] for k in order), tuple(float(best_raw[k]) for k in order),
                      tuple(float(best_norm[k]) for k in order), tuple(float(rarity[k]) for k in order),
                      float(accept), float(keep), None if reject is None else float(reject),
                      self.calibration.bin_label(size, yaw), bool(decisive))

    def self_score(self, vector, reference) -> float:
        """A new face's ``vector`` scored against its track's own aggregate ``reference``, normalised as the gallery's
        scores are (for :meth:`consistent`)."""
        vector, reference = np.asarray(vector, dtype=np.float64), np.asarray(reference, dtype=np.float64)
        raw = float(vector @ reference)
        if self.normalisation == "none":
            return raw
        return float(self.cohort.normalise(np.array([raw]), vector, reference[None], self.normalisation)[0])

    def _gallery_rows(self) -> np.ndarray:
        """The gallery's photos and templates, in the order :meth:`~.gallery.Gallery.raw_scores` scores them."""
        g = self.gallery
        parts = [m for m in (g._matrix, g._templates) if len(m)]
        return np.vstack(parts) if parts else np.zeros((0, 1))

    def _stats(self, rows):
        """The cohort statistics of the gallery's ``rows`` (AS-norm), made again when the gallery changes."""
        if self.normalisation != "asnorm" or rows is None:
            return None
        key = (rows.shape, hash(rows.tobytes()) if rows.size < 1_000_000 else id(rows))
        if self._gallery_stats is None or self._gallery_stats[0] != key:
            self._gallery_stats = (key, self.cohort.stats(rows) if len(rows) else (np.zeros(0), np.zeros(0)))
        return self._gallery_stats[1]

    def consistent(self, score: float, size: float, yaw: float) -> bool:
        """Whether a new face whose (normalised) score against its track's own evidence is ``score`` can be the same
        person: not when that score is below what a known person's face of this quality reaches all but
        ``switch_miss`` of the time (see the module docs). Where the calibration cannot tell (no mated curve, or that
        floor is not decisive among strangers' scores), every face is."""
        curve = self.calibration.curve(size, yaw)
        floor = curve.mated_quantile(self.settings.switch_miss)
        if floor is None or curve.rarity(floor) < self.settings.decisive:
            return True
        return score >= floor

    # -- decisions -------------------------------------------------------------------------------------------------

    def forget(self, track: int) -> None:
        self.tracks.pop(track, None)

    def reset(self, track: int, now: float) -> None:
        """The track turned out to show someone else (:meth:`consistent`): its name (or "unknown") is withdrawn at
        once; the next decision starts from "not decided yet", and another name needs to be sustained."""
        memory = self.tracks.get(track)
        if memory is None:
            return
        if memory.state == NAMED:
            memory.dropped = (memory.name, now)
        memory.state, memory.name, memory.since, memory.pending = UNDECIDED, None, now, None
        memory.reset_at = now

    def decide(self, aggregates: dict, now: float, frame: int | None = None, evidence: dict | None = None,
               waiting: dict | None = None) -> dict[int, Decision]:
        """Decide every track: ``aggregates`` maps a track's id to its :class:`~.evidence.Aggregate` (``None``: no
        evidence yet); ``evidence``: per track a dict describing it for the log; ``waiting``: per track why it has no
        evidence yet (``"no face"``, ``"turned"``, ...). Returns ``{track: Decision}``. Tracks not given keep what the
        policy remembers of them (a track lost for a moment: its name holds when it is found again); :meth:`forget`
        drops a track."""
        known = len(self.gallery.people()) if hasattr(self.gallery, "people") else 0
        proposals: dict[int, tuple] = {}
        scored: dict[int, Scores] = {}
        for track, agg in aggregates.items():
            self.tracks.setdefault(track, _Track(since=now))
            if agg is None:
                proposals[track] = (UNDECIDED, None, (waiting or {}).get(track, "no face yet"), 0.0)
            elif not known:
                proposals[track] = (UNDECIDED, None, "nobody known", self._progress(agg))
            else:
                scored[track] = self.score(agg.vector, agg.size, agg.yaw, agg.model)
                proposals[track] = self._propose(scored[track], agg)
        self._exclusive(proposals, scored)
        out = {}
        for track, proposal in proposals.items():
            decision = self._settle(track, self.tracks[track], proposal, now, frame, scored.get(track),
                                    aggregates.get(track), (evidence or {}).get(track, {}))
            out[track] = decision
            if self.log is not None:
                self.log.add(decision)
        return out

    def _progress(self, agg) -> float:
        s = self.settings
        return float(min(1.0, agg.weight / s.min_weight, agg.frames / max(1, s.min_frames)))

    def _propose(self, scores: Scores, agg) -> tuple:
        """What this evidence says on its own: ``(state, name, reason, progress)``."""
        s = self.settings
        progress = self._progress(agg)
        if progress < 1.0:
            return UNDECIDED, None, "collecting", progress
        best_rarity = scores.rarity[0]
        runner = scores.rarity[1] if len(scores.names) > 1 else 0.0
        if best_rarity >= s.accept(len(scores.names)):
            if best_rarity - runner >= s.margin:
                return NAMED, scores.names[0], "accepted", 1.0
            return UNDECIDED, None, "too close to another", 1.0
        if scores.reject is None or not scores.decisive:
            return UNDECIDED, None, "faces of this quality cannot be told apart", 1.0
        if scores.normalised[0] < scores.reject:
            if scores.rarity[0] <= -math.log10(per_person(s.stranger_like, len(scores.names))):
                return UNKNOWN, None, "below reject", 1.0
            return UNDECIDED, None, "unlike strangers too", 1.0
        return UNDECIDED, None, "between thresholds", 1.0

    def _keeps(self, scores: Scores | None, name: str | None) -> bool:
        """Whether ``name``'s score is still at its keep threshold."""
        got = scores.of(name) if scores is not None and name else None
        return got is not None and got[1] >= scores.keep

    def _exclusive(self, proposals: dict, scored: dict) -> None:
        """One identity per track, one track per identity: the claims (names proposed, and names held within their
        keep threshold) are assigned at the highest summed rarity; a held name that wins is kept, every losing claim
        becomes "contested" (in ``proposals``, in place)."""
        claims = {}
        for t, (state, name, _reason, _progress) in proposals.items():
            memory = self.tracks[t]
            if state == NAMED:
                claims[t] = (name, False)
            elif memory.state == NAMED and self._keeps(scored.get(t), memory.name):
                claims[t] = (memory.name, True)
        if not claims:
            return
        tracks = sorted(claims)
        names = sorted({name for name, _held in claims.values()})
        cost = np.full((len(tracks), len(names)), math.inf)
        for i, t in enumerate(tracks):
            name, _held = claims[t]
            rarity = scored[t].of(name)[2]
            memory = self.tracks[t]
            bonus = self.settings.incumbent if memory.state == NAMED and memory.name == name else 0.0
            cost[i, names.index(name)] = -(max(rarity, 0.0) + bonus)
        pairs, _, _ = match(cost, 1.0)  # every claim costs less than 1 (rarities are not negative)
        winners = {tracks[i]: names[j] for i, j in pairs}
        for t, (name, held) in claims.items():
            if winners.get(t) != name:
                proposals[t] = (UNDECIDED, None, "contested", 1.0)
            elif held:
                proposals[t] = (NAMED, name, "kept", 1.0)

    def _settle(self, track, memory: _Track, proposal, now, frame, scores, agg, described) -> Decision:
        """Apply hysteresis and sustained switches to ``proposal`` (see the module docs), remember the outcome and
        make the :class:`Decision`."""
        s = self.settings
        state, name, reason, progress = proposal

        def sustained(target_state, target_name, hold) -> bool:
            if memory.pending is None or memory.pending[:2] != (target_state, target_name):
                memory.pending = (target_state, target_name, now)
            return now - memory.pending[2] >= hold - 1e-9

        def waiting(target_state, target_name, hold) -> str:
            what = "unknown" if target_state == UNKNOWN else target_name
            return f"{what} in {max(0.0, hold - (now - memory.pending[2])):.1f} s"

        if memory.state == NAMED:
            held = memory.name
            if state == NAMED and name == held:
                result = (NAMED, held, reason, 1.0)
            elif state in (NAMED, UNKNOWN):  # another name, or unknown: only when it holds
                if sustained(state, name, s.sustain):
                    result = (state, name, reason, 1.0)
                elif self._keeps(scores, held):
                    result = (NAMED, held, f"kept ({waiting(state, name, s.sustain)}?)", 1.0)
                else:
                    result = (UNDECIDED, None, f"waiting: {waiting(state, name, s.sustain)}", 1.0)
            elif reason != "contested" and self._keeps(scores, held):
                result = (NAMED, held, "kept", 1.0)
            else:
                result = (UNDECIDED, None, reason, progress)
        elif memory.state == UNKNOWN:
            if state == UNKNOWN:
                result = (UNKNOWN, None, reason, 1.0)
            elif state == NAMED:
                result = (NAMED, name, reason, 1.0) if sustained(NAMED, name, s.sustain) else \
                    (UNDECIDED, None, f"waiting: {waiting(NAMED, name, s.sustain)}", 1.0)
            elif scores is not None and progress >= 1.0 and scores.normalised[0] < scores.keep:
                result = (UNKNOWN, None, "kept", 1.0)  # unknown stays while nobody comes near a name
            else:
                result = (UNDECIDED, None, reason, progress)
        elif state == NAMED:
            other = memory.dropped is not None and memory.dropped[0] != name  # it had another name before
            if not other or sustained(NAMED, name, s.sustain):
                result = (NAMED, name, reason, 1.0)
            else:
                result = (UNDECIDED, None, f"waiting: {waiting(NAMED, name, s.sustain)}", 1.0)
        elif state == UNKNOWN:
            result = (UNKNOWN, None, reason, 1.0) if sustained(UNKNOWN, None, s.sustain_unknown) else \
                (UNDECIDED, None, f"waiting: {waiting(UNKNOWN, None, s.sustain_unknown)}", 1.0)
        else:
            result = (UNDECIDED, None, reason, progress)
        if state == UNDECIDED or result[:2] == (state, name):
            memory.pending = None  # nothing to wait for any more
        if memory.state == NAMED and result[:2] != (NAMED, memory.name):
            memory.dropped = (memory.name, now)
        changed = result[:2] != (memory.state, memory.name)
        if changed:
            memory.state, memory.name, memory.since = result[0], result[1], now
        decision = self._decision(track, now, frame, *result, scores, agg, described, changed)
        memory.last = decision
        return decision

    def _decision(self, track, now, frame, state, name, reason, progress, scores: Scores | None, agg, described,
                  changed) -> Decision:
        evidence = dict(described)
        if agg is not None:
            evidence.update(frames=agg.frames, weight=round(agg.weight, 4), size=round(agg.size, 2),
                            yaw=round(agg.yaw, 2), best=round(agg.best, 4))
        common = {"track": track, "time": float(now), "state": state, "name": name, "reason": reason,
                  "progress": round(float(progress), 3), "frame": frame, "model": self.calibration.model,
                  "normalisation": self.normalisation, "evidence": evidence, "changed": changed}
        if scores is None or not scores.names:
            return Decision(**common)
        second = len(scores.names) > 1
        return Decision(**common, candidate=scores.names[0], raw=scores.raw[0], score=scores.normalised[0],
                        calibrated=scores.rarity[0], runner_up=scores.names[1] if second else None,
                        runner_up_raw=scores.raw[1] if second else None,
                        runner_up_score=scores.normalised[1] if second else None,
                        runner_up_calibrated=scores.rarity[1] if second else None, accept=scores.accept,
                        keep=scores.keep, reject=scores.reject,
                        accept_rarity=self.settings.accept(len(scores.names)), bin=scores.bin)
