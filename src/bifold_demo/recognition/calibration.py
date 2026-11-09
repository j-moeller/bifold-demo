"""Calibration: one false-alarm rate the user sets, turned into each model's own score thresholds per face size.

**The target.** The user sets one thing, the **false positive identification rate per decision** (FPIR, for example
1 %): how often a decision about a stranger names somebody. A stranger is compared with every known person, so with
``n`` known people the false-match rate allowed per person is ``1 - (1 - FPIR) ** (1 / n)`` (about ``FPIR / n``;
:func:`per_person`). Each model's calibration turns that rate into a score threshold for the face's quality, so that
the same target gives the same behaviour whichever profile runs.

**What a calibration holds** (:class:`Calibration`, one per model): for each **bin** of face quality (the face's eye
distance turned to the camera, :func:`~.heads.face_size`, and how far the head is turned; :data:`SIZE_EDGES`,
:data:`YAW_EDGES`) two curves of person-level scores (a face against all of one known person's photos and pose
templates, the best: :meth:`~.gallery.Gallery.scores`):

* the **non-mated** curve, how often a stranger's score reaches a value (the false-match rate, FMR), from scores of
  faces against other people; beyond the scores measured, an exponential tail (its mean excess over the last point:
  exact for an exponential tail, and on the safe side for cosine scores, which cannot exceed 1);
* the **mated** curve, how often a known person's score falls below a value (the false non-match rate, FNMR), from
  scores of faces against their own photos.

From them: the **rarity** of a score, ``-log10(FMR)`` (:meth:`Calibration.rarity`; "a stranger scores this high once
in ``10 ** rarity`` comparisons"), a currency the same in every model and bin, in which the decisions' accept, keep
and margin are counted (:mod:`.decision`); the **accept threshold** for a target FPIR (:meth:`Calibration.
threshold`); and the **reject threshold** below which a known person's score falls only rarely (:meth:`Calibration.
mated_quantile`), the floor of "unknown". A bin where known people's faces score low (turned, tiny) gets a low reject
threshold, so it seldom says "unknown": such faces are "not decided yet".

**Where it comes from.** ``bifold-faces calibrate`` (:mod:`.calibrate`) renders avatars (Rocketbox, MIT) through the
virtual sensor's windows at swept face sizes, poses, camera presets and the fair's light; mated pairs are the same
avatar, non-mated different ones. The result is kept per model in ``<models>/calibration/<model id>.json``; the
package ships one for each profile's recogniser (``recognition/calibrations/``), made from 38 avatars that are not in
the fair. LFW and photos of real people may check a calibration (``docs/research/face_recognition_f2_results.md``),
never make what is shipped. Without any calibration a model gets a :meth:`provisional <Calibration.provisional>` one
from its registered threshold (one false match in a thousand LFW stranger pairs), which never says "unknown".

**Score normalisation** (:class:`Cohort`): a face's score against a person can be measured against how the face
scores against a **cohort** of other faces that are not in the gallery: T-norm (the probe's mean and spread over the
cohort) or AS-norm (the mean and spread of the top ``k`` cohort scores of the probe and of the gallery photo,
averaged). A calibration records which normalisation its scores had and the cohort's :attr:`~Cohort.digest`; the
decisions apply the same normalisation with that cohort and refuse another.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

FORMAT = 1
SIZE_EDGES = (0.0, 10.0, 14.0, 20.0, 28.0, 40.0, 56.0, 80.0, math.inf)  # eye distance turned to the camera, pixels
YAW_EDGES = (0.0, 25.0, 55.0, 180.0)  # degrees, absolute
FMR_LEVELS = (0.5, 0.3, 0.2, 0.1, 0.05, 0.03, 0.02, 0.01, 0.005, 0.003, 0.002, 0.001, 5e-4, 2e-4, 1e-4)
FNMR_LEVELS = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)
MIN_EXCEEDANCES = 10  # a tabled FMR level needs this many non-mated scores at or above it
MIN_NONMATED, MIN_MATED = 200, 20  # a bin with fewer borrows a neighbour's curves
PACKAGED = Path(__file__).with_name("calibrations")

__all__ = ["FMR_LEVELS", "FNMR_LEVELS", "SIZE_EDGES", "YAW_EDGES", "Calibration", "Cohort", "Curve",
           "find_calibration", "find_cohort", "per_person"]


def per_person(fpir: float, people: int) -> float:
    """The false-match rate per known person that gives ``fpir`` per decision with ``people`` known people (see the
    module docs)."""
    fpir = float(np.clip(fpir, 1e-12, 1.0 - 1e-12))
    return 1.0 - (1.0 - fpir) ** (1.0 / max(1, int(people)))


def bin_index(size: float, yaw: float, size_edges=SIZE_EDGES, yaw_edges=YAW_EDGES) -> tuple[int, int]:
    """The bin of a face ``size`` pixels between the eyes (turned to the camera) and ``yaw`` degrees."""
    s = int(np.clip(np.searchsorted(size_edges, size, side="right") - 1, 0, len(size_edges) - 2))
    y = int(np.clip(np.searchsorted(yaw_edges, abs(yaw), side="right") - 1, 0, len(yaw_edges) - 2))
    return s, y


@dataclass(frozen=True)
class Curve:
    """A bin's two curves (see the module docs). Non-mated: ``scores`` ascending with their ``log_fmr`` (log10 of the
    share of stranger scores at or above), then an exponential tail of mean excess ``beta`` beyond the last point.
    Mated: ``mated`` scores ascending at the false non-match rates ``fnmr``. ``counts``: (non-mated, mated, people)
    measured; ``borrowed``: the bin the curves come from when this one had too few scores (else ``None``)."""

    scores: tuple
    log_fmr: tuple
    beta: float
    mated: tuple = ()
    fnmr: tuple = ()
    counts: tuple = (0, 0, 0)
    borrowed: tuple | None = None

    @classmethod
    def fit(cls, nonmated, mated=(), people: int = 0) -> Curve:
        """The curves of measured ``nonmated`` and ``mated`` scores."""
        nonmated = np.sort(np.asarray(nonmated, dtype=np.float64))
        mated = np.sort(np.asarray(mated, dtype=np.float64))
        n = len(nonmated)
        if n < MIN_EXCEEDANCES:
            raise ValueError(f"{n} non-mated scores are too few for a curve")
        levels = [lv for lv in FMR_LEVELS if lv * n >= MIN_EXCEEDANCES]
        points = [(float(np.quantile(nonmated, 1.0 - lv, method="linear")), math.log10(lv)) for lv in levels]
        points.insert(0, (float(nonmated[0]), 0.0))
        scores, logs = [], []
        for s, lf in points:  # strictly increasing scores
            if scores and s <= scores[-1]:
                s = scores[-1] + 1e-6
            scores.append(s)
            logs.append(lf)
        top = nonmated[nonmated >= scores[-1]]
        beta = float(max(np.mean(top - scores[-1]) if len(top) else 0.0, 1e-3))
        q, f = (), ()
        if len(mated) >= 2:
            q = tuple(float(np.quantile(mated, lv, method="linear")) for lv in FNMR_LEVELS)
            f = FNMR_LEVELS
        return cls(tuple(scores), tuple(logs), beta, q, f, (n, len(mated), int(people)))

    def rarity(self, score: float) -> float:
        """``-log10`` of the share of stranger scores at or above ``score`` (0 at or below the lowest measured)."""
        scores, logs = self.scores, self.log_fmr
        if score <= scores[0]:
            return 0.0
        if score <= scores[-1]:
            return float(-np.interp(score, scores, logs))
        return float(-logs[-1] + (score - scores[-1]) / (self.beta * math.log(10.0)))

    def threshold(self, fmr: float) -> float:
        """The score whose rarity is ``-log10(fmr)``: strangers reach it with a rate ``fmr``."""
        target = math.log10(float(np.clip(fmr, 1e-300, 1.0)))
        logs = self.log_fmr
        if target >= logs[0]:
            return float(self.scores[0])
        if target >= logs[-1]:
            return float(np.interp(-target, [-v for v in logs], self.scores))
        return float(self.scores[-1] + (logs[-1] - target) * self.beta * math.log(10.0))

    def mated_quantile(self, fnmr: float) -> float | None:
        """The score a known person's face falls below with a rate ``fnmr`` (``None`` without mated scores)."""
        if not self.mated:
            return None
        return float(np.interp(fnmr, self.fnmr, self.mated))

    def fnmr_at(self, score: float) -> float | None:
        """How often a known person's face scores below ``score`` (``None`` without mated scores)."""
        if not self.mated:
            return None
        return float(np.interp(score, self.mated, self.fnmr, left=0.0, right=1.0))

    def to_json(self) -> dict:
        return {"scores": [round(v, 6) for v in self.scores], "log_fmr": [round(v, 6) for v in self.log_fmr],
                "beta": round(self.beta, 6), "mated": [round(v, 6) for v in self.mated], "fnmr": list(self.fnmr),
                "counts": list(self.counts), "borrowed": None if self.borrowed is None else list(self.borrowed)}

    @classmethod
    def from_json(cls, data: dict) -> Curve:
        return cls(tuple(data["scores"]), tuple(data["log_fmr"]), float(data["beta"]), tuple(data.get("mated", ())),
                   tuple(data.get("fnmr", ())), tuple(data.get("counts", (0, 0, 0))),
                   None if data.get("borrowed") is None else tuple(data["borrowed"]))


@dataclass(frozen=True)
class Calibration:
    """A model's calibration (see the module docs): ``curves[size bin][yaw bin]``, the bins' edges, the normalisation
    its scores had (``"none"``, ``"tnorm"`` or ``"asnorm"``), and ``made``: how it was made (the avatars, renders,
    presets; ``provisional`` when it comes from the registered threshold alone)."""

    model: str
    curves: tuple
    normalisation: str = "none"
    size_edges: tuple = SIZE_EDGES
    yaw_edges: tuple = YAW_EDGES
    made: dict = field(default_factory=dict)
    path: str | None = None

    # -- using it ---------------------------------------------------------------------------------------------------

    def bin(self, size: float, yaw: float) -> tuple[int, int]:
        return bin_index(size, yaw, self.size_edges, self.yaw_edges)

    def curve(self, size: float, yaw: float) -> Curve:
        s, y = self.bin(size, yaw)
        return self.curves[s][y]

    def bin_label(self, size: float, yaw: float) -> str:
        s, y = self.bin(size, yaw)
        lo, hi = self.size_edges[s], self.size_edges[s + 1]
        ylo, yhi = self.yaw_edges[y], self.yaw_edges[y + 1]
        return f"{lo:g}-{hi:g} px, {ylo:g}-{yhi:g} deg"

    @property
    def is_provisional(self) -> bool:
        """Whether this is a stand-in from the registered threshold (:meth:`provisional`), not a calibration."""
        return bool(self.made.get("provisional"))

    def rarity(self, score: float, size: float, yaw: float) -> float:
        """``-log10`` of the share of strangers' scores reaching ``score`` for a face of this ``size`` and ``yaw``."""
        return self.curve(size, yaw).rarity(score)

    def threshold(self, fpir: float, people: int, size: float, yaw: float) -> float:
        """The score that names a face of this ``size`` and ``yaw`` at a false-alarm rate ``fpir`` per decision with
        ``people`` known people."""
        return self.curve(size, yaw).threshold(per_person(fpir, people))

    def mated_quantile(self, fnmr: float, size: float, yaw: float) -> float | None:
        return self.curve(size, yaw).mated_quantile(fnmr)

    def fnmr_at(self, score: float, size: float, yaw: float) -> float | None:
        return self.curve(size, yaw).fnmr_at(score)

    # -- making and keeping it -------------------------------------------------------------------------------------

    @classmethod
    def fit(cls, model: str, scores, mated, sizes, yaws, people=None, normalisation: str = "none", made=None,
            size_edges=SIZE_EDGES, yaw_edges=YAW_EDGES) -> Calibration:
        """A calibration from measured person-level ``scores`` (one per comparison of a face with a known person),
        whether each is ``mated``, and the face's ``size`` and ``yaw``; ``people``: who each face shows (to count
        identities per bin). Bins with fewer than :data:`MIN_NONMATED` non-mated or :data:`MIN_MATED` mated scores
        borrow the curves of the nearest bin that has them, a smaller face first (on the safe side)."""
        scores = np.asarray(scores, dtype=np.float64)
        mated = np.asarray(mated, dtype=bool)
        sizes, yaws = np.asarray(sizes, dtype=np.float64), np.abs(np.asarray(yaws, dtype=np.float64))
        people = np.asarray(people if people is not None else np.zeros(len(scores)))
        bins = np.array([bin_index(s, y, size_edges, yaw_edges) for s, y in zip(sizes, yaws)]).reshape(-1, 2)
        n_size, n_yaw = len(size_edges) - 1, len(yaw_edges) - 1
        own: dict[tuple[int, int], Curve] = {}
        for s in range(n_size):
            for y in range(n_yaw):
                here = (bins[:, 0] == s) & (bins[:, 1] == y) if len(bins) else np.zeros(0, bool)
                non, mat = scores[here & ~mated], scores[here & mated]
                if len(non) >= MIN_NONMATED and len(mat) >= MIN_MATED:
                    own[(s, y)] = Curve.fit(non, mat, len(set(people[here].tolist())))
        if not own:
            raise ValueError(f"too few scores to calibrate {model}: no bin has {MIN_NONMATED} non-mated and "
                             f"{MIN_MATED} mated scores")
        curves = []
        for s in range(n_size):
            row = []
            for y in range(n_yaw):
                if (s, y) in own:
                    row.append(own[(s, y)])
                    continue
                # nearest bin with data: the same yaw first, smaller faces before larger, then more turned ones
                order = sorted(own, key=lambda k: (abs(k[1] - y) + (0 if k[1] >= y else 0.5),
                                                   abs(k[0] - s) + (0 if k[0] <= s else 0.5)))
                donor = order[0]
                curve = own[donor]
                row.append(Curve(curve.scores, curve.log_fmr, curve.beta, curve.mated, curve.fnmr, curve.counts,
                                 donor))
            curves.append(tuple(row))
        return cls(model, tuple(curves), normalisation, tuple(size_edges), tuple(yaw_edges), dict(made or {}))

    @classmethod
    def provisional(cls, model: str, threshold: float, decade: float = 0.04) -> Calibration:
        """A stand-in from a model's registered ``threshold`` (one false match in a thousand LFW stranger pairs): a
        non-mated curve one decade per ``decade`` of score around it, the same for every bin, and no mated curve. It
        only puts the scores of the decision log on a scale: the decisions name nobody with it (on avatars that
        threshold let a third of strangers through), until a calibration is made (``bifold-faces calibrate``)."""
        scores = tuple(threshold + decade * (k - 3) for k in range(7))
        curve = Curve(scores, tuple(-float(k) for k in range(7)), decade / math.log(10.0))
        rows = tuple(tuple(curve for _ in range(len(YAW_EDGES) - 1)) for _ in range(len(SIZE_EDGES) - 1))
        return cls(model, rows, "none", made={"provisional": True, "from": f"the registered threshold {threshold}"})

    def to_json(self) -> dict:
        return {"format": FORMAT, "model": self.model, "normalisation": self.normalisation,
                "size_edges": [None if math.isinf(v) else v for v in self.size_edges],
                "yaw_edges": list(self.yaw_edges), "made": self.made,
                "bins": [[c.to_json() for c in row] for row in self.curves]}

    @classmethod
    def from_json(cls, data: dict, path=None) -> Calibration:
        if int(data.get("format", 0)) != FORMAT:
            raise ValueError(f"calibration format {data.get('format')} is not {FORMAT}")
        edges = tuple(math.inf if v is None else float(v) for v in data["size_edges"])
        curves = tuple(tuple(Curve.from_json(c) for c in row) for row in data["bins"])
        return cls(str(data["model"]), curves, str(data.get("normalisation", "none")), edges,
                   tuple(float(v) for v in data["yaw_edges"]), dict(data.get("made", {})),
                   None if path is None else str(path))

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.to_json(), indent=1))
        tmp.replace(path)
        return path

    @classmethod
    def load(cls, path) -> Calibration:
        return cls.from_json(json.loads(Path(path).read_text()), path)


def calibration_path(model_id: str, models_folder=None) -> Path:
    """Where ``bifold-faces calibrate`` keeps ``model_id``'s calibration: ``<models>/calibration/<model id>.json``."""
    from .models import models_dir

    return models_dir(models_folder) / "calibration" / f"{model_id}.json"


def has_calibration(model_id: str, models_folder=None) -> bool:
    """Whether ``model_id`` has a calibration made for it (in the models folder, or the package's own)."""
    return calibration_path(model_id, models_folder).is_file() or (PACKAGED / f"{model_id}.json").is_file()


def find_calibration(model_id: str, models_folder=None, threshold: float | None = None) -> Calibration:
    """``model_id``'s calibration: ``<models>/calibration/<model id>.json``, else the one the package ships, else a
    :meth:`provisional <Calibration.provisional>` one from ``threshold`` (default: the registry's)."""
    for path in (calibration_path(model_id, models_folder), PACKAGED / f"{model_id}.json"):
        if path.is_file():
            calibration = Calibration.load(path)
            if calibration.model != model_id:
                raise ValueError(f"{path} calibrates {calibration.model}, not {model_id}")
            return calibration
    if threshold is None:
        from . import registry

        threshold = registry.model(model_id).threshold
    return Calibration.provisional(model_id, float(threshold))


# -- score normalisation ----------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Cohort:
    """Unit embeddings of faces that are not in the gallery (``vectors``, ``(n, dim)``) of one ``model``, with who
    each shows (``people``, to leave a person's own faces out when calibrating), for score normalisation (see the
    module docs). ``top``: the cohort scores AS-norm averages (the highest ``top``)."""

    model: str
    vectors: np.ndarray = field(repr=False)
    people: tuple = ()
    top: int = 50

    def __post_init__(self):
        vectors = np.asarray(self.vectors, dtype=np.float64)
        object.__setattr__(self, "vectors", vectors.reshape(len(vectors), -1))
        if self.people and len(self.people) != len(vectors):
            raise ValueError(f"{len(vectors)} cohort faces but {len(self.people)} names")

    def __len__(self) -> int:
        return len(self.vectors)

    @property
    def digest(self) -> str:
        """A short hash of the cohort as it is saved (half precision), recorded by the calibration made with it
        (``made["cohort"]``)."""
        return hashlib.sha256(np.ascontiguousarray(self.vectors.astype(np.float16)).tobytes()).hexdigest()[:16]

    def _rows(self, leave_out=()) -> np.ndarray:
        if not leave_out or not self.people:
            return self.vectors
        keep = np.array([p not in leave_out for p in self.people], bool)
        return self.vectors[keep]

    def stats(self, vectors, top: int | None = None, leave_out=()) -> tuple[np.ndarray, np.ndarray]:
        """``(means, spreads)`` of each of ``vectors``' (``(m, dim)``) scores against the cohort: of all of them
        (``top`` 0) or of the highest ``top`` (default: :attr:`top`); ``leave_out``: people whose faces do not count."""
        vectors = np.asarray(vectors, dtype=np.float64).reshape(-1, self.vectors.shape[1])
        rows = self._rows(leave_out)
        scores = vectors @ rows.T
        k = self.top if top is None else int(top)
        if k and k < scores.shape[1]:
            scores = -np.partition(-scores, k - 1, axis=1)[:, :k]
        return scores.mean(axis=1), np.maximum(scores.std(axis=1), 1e-3)

    def normalise(self, raw: np.ndarray, probe: np.ndarray, gallery: np.ndarray, kind: str,
                  leave_out=(), gallery_stats=None) -> np.ndarray:
        """``raw`` scores of ``probe`` (one unit embedding) against the ``gallery``'s rows, normalised by ``kind``:
        ``"tnorm"`` or ``"asnorm"`` (``"none"``: as they are). ``gallery_stats``: the gallery rows' (means, spreads)
        when known (AS-norm)."""
        raw = np.asarray(raw, dtype=np.float64)
        if kind == "none":
            return raw
        if kind == "tnorm":
            mean, spread = self.stats(probe[None], top=0, leave_out=leave_out)
            return (raw - mean[0]) / spread[0]
        if kind == "asnorm":
            mean, spread = self.stats(probe[None], leave_out=leave_out)
            g_mean, g_spread = gallery_stats if gallery_stats is not None else self.stats(gallery, leave_out=leave_out)
            return 0.5 * ((raw - mean[0]) / spread[0] + (raw - g_mean) / g_spread)
        raise ValueError(f"unknown normalisation {kind!r} (none, tnorm or asnorm)")

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, model=np.array(self.model), vectors=self.vectors.astype(np.float16),
                            people=np.array(list(self.people), str), top=np.array(self.top))
        return path

    @classmethod
    def load(cls, path) -> Cohort:
        with np.load(path, allow_pickle=False) as data:
            return cls(str(data["model"]), data["vectors"].astype(np.float64),
                       tuple(str(p) for p in data["people"]), int(data["top"]))


def cohort_path(model_id: str, models_folder=None) -> Path:
    from .models import models_dir

    return models_dir(models_folder) / "calibration" / f"{model_id}.cohort.npz"


def find_cohort(model_id: str, models_folder=None, calibration: Calibration | None = None) -> Cohort | None:
    """``model_id``'s cohort: the one beside ``calibration``'s file when given (a cohort belongs to the calibration
    made with it), else ``<models>/calibration/<model id>.cohort.npz``, else the package's; ``None`` when there is
    none."""
    if calibration is not None and calibration.path:
        paths = (Path(calibration.path).with_name(f"{model_id}.cohort.npz"),)
    else:
        paths = (cohort_path(model_id, models_folder), PACKAGED / f"{model_id}.cohort.npz")
    for path in paths:
        if path.is_file():
            cohort = Cohort.load(path)
            if cohort.model != model_id:
                raise ValueError(f"{path} is a cohort of {cohort.model}, not {model_id}")
            return cohort
    return None
