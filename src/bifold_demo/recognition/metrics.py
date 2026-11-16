"""Measuring the face recognition (step E1): verification, open-set identification, track-level outcomes,
calibration and cost, with intervals that count people, not frames.

Numpy only: the curves are short enough to write here, and ``tests/test_metrics.py`` checks them against exact
hand-computed cases and against scikit-learn's ``roc_curve`` where scikit-learn is installed (it is not a dependency).

**Conventions.** A score is a similarity (higher: more alike). A comparison is *accepted* when its score is at or
above the threshold, ``score >= t``, as the decisions accept. Rates are shares of comparisons, searches or tracks;
operating points are *counted* on the scores (the threshold that lets at most the asked share through), never
interpolated between them, so a rate is never claimed below what the counts can show: with ``n`` impostor scores a
false-accept rate under ``1 / n`` is not resolvable (:func:`far_levels`).

* **Pairs** (verification): ``genuine`` (mated: one person) and ``impostor`` (non-mated) scores. :func:`roc` gives the
  false-accept rate (FAR) and true-accept rate (TAR) at every threshold, :func:`det` the FAR and the false-reject
  rate (FRR) for a DET plot on normal-deviate axes (:func:`probit`), :func:`tar_at_far` the TAR at the strictest
  threshold whose FAR is at most the asked one, :func:`eer` the equal error rate and :func:`d_prime` the separation.
  :func:`pairs_report` puts them together, with intervals.
* **Open-set 1:N** (the demo's case; NIST FRTE's measures): searches scored against every enrolled person
  (``scores``: searches x people, a person's score the best of their photos); ``mate``: the column of a search's own
  person, ``-1`` for a stranger's (non-mated) search. :func:`open_set` gives, per threshold, the false positive
  identification rate (FPIR: non-mated searches whose best score reaches it), the false negative identification rate
  (FNIR: mated searches whose mate is not the best candidate at or above it) and the detection and identification
  rate DIR = 1 - FNIR; :func:`fnir_at_fpir` the FNIR at the strictest threshold whose FPIR is at most the asked one.
  :func:`cmc` and the rank-1 rate are the closed-set sanity check (they say nothing about strangers). Ties count
  against the search: a mate tied with another person is not ranked first. :func:`open_set_report` puts them
  together, with intervals.
* **Tracks** (what a visitor sees): :func:`track_summary` adds intervals and rates per hour to F2's
  :class:`~.truth.Outcomes`; :class:`NameEvents` counts what :class:`~.truth.Outcomes` does not: a track's name
  switching to another name, and one name on two tracks at once.
* **Calibration**: :func:`achieved`, the false-alarm rate measured against the one asked, per group (a profile, a
  face-size bin), with intervals and whether it is within a factor of the target.
* **Acceptance** (``docs/research/face_recognition_sota.md``, section 4.4): :func:`acceptance` judges a run's counts
  against :class:`Acceptance` (at least 95 % of enrolled people's tracks named correctly within 2 s, at 1 % FPIR per
  track, achieved within a factor of 2, no identity on two tracks at once).
* **Cost**: :func:`cost_summary` (milliseconds per stage, embeddings a second) and :func:`peak_memory_mb`.

**Statistics.** Frames of one person, tracks of one person and searches of one person are not independent: the
intervals come from a **bootstrap over clusters** (:func:`cluster_counts`): people (or identity pairs) are drawn with
replacement and every comparison, search or track of a drawn cluster counts as often as it was drawn. For pairs the
clusters are people and an impostor pair counts with the product of its two people's counts (:func:`pairs_report`).
Percentile intervals, two-sided at ``level`` (0.95 unless said). :func:`binomial_interval` (Clopper-Pearson, exact)
is for counts of independent units (people, identity pairs); with zero events it gives the exact upper bound (about
three over ``n`` at 95 %, the "rule of three").
"""

from __future__ import annotations

import math
import resource
import sys
from dataclasses import asdict, dataclass

import numpy as np

LEVEL = 0.95  # two-sided level of the intervals
DRAWS = 1000  # bootstrap draws
FAR_LEVELS = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6)  # asked false-accept rates (those the counts resolve are reported)
FPIR_LEVELS = (1e-1, 1e-2, 1e-3, 1e-4)

__all__ = ["DRAWS", "FAR_LEVELS", "FPIR_LEVELS", "LEVEL", "Acceptance", "NameEvents", "acceptance", "achieved",
           "binomial_interval", "bootstrap_rate", "cluster_counts", "cmc", "cost_summary", "d_prime", "det", "eer",
           "far_levels", "fnir_at_fpir", "open_set", "open_set_report", "pair_weights", "pairs_report",
           "peak_memory_mb", "probit", "ranks", "roc", "tar_at_far", "track_summary"]


# -- helpers ----------------------------------------------------------------------------------------------------------


def _scores(values) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).ravel()
    if np.isnan(values).any():
        raise ValueError("scores must not be NaN")
    return values


def _weights(weights, n: int) -> np.ndarray:
    if weights is None:
        return np.ones(n)
    weights = np.asarray(weights, dtype=np.float64).ravel()
    if len(weights) != n:
        raise ValueError(f"{len(weights)} weights for {n} scores")
    if (weights < 0).any():
        raise ValueError("weights must not be negative")
    return weights


def _share_at_or_above(scores: np.ndarray, weights: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """The (weighted) share of ``scores`` at or above each of ``thresholds``."""
    order = np.argsort(scores, kind="stable")
    s, w = scores[order], weights[order]
    below = np.concatenate([[0.0], np.cumsum(w)])  # weight strictly before index i
    total = below[-1]
    if total <= 0:
        return np.full(len(thresholds), np.nan)
    first = np.searchsorted(s, thresholds, side="left")  # the first score >= t
    return (total - below[first]) / total


def _interval(draws: np.ndarray, level: float) -> list[float | None]:
    draws = np.asarray(draws, dtype=np.float64)
    draws = draws[np.isfinite(draws)]
    if not len(draws):
        return [None, None]
    tail = (1.0 - level) / 2.0
    return [float(np.quantile(draws, tail)), float(np.quantile(draws, 1.0 - tail))]


def _clusters(clusters, n: int) -> tuple[np.ndarray, int]:
    """``(index of each item's cluster, number of clusters)``; ``None``: every item its own cluster."""
    if clusters is None:
        return np.arange(n), n
    clusters = np.asarray(clusters)
    if len(clusters) != n:
        raise ValueError(f"{len(clusters)} cluster labels for {n} items")
    _, index = np.unique(clusters.astype(str), return_inverse=True)
    return index.ravel(), int(index.max()) + 1 if n else 0


def cluster_counts(clusters: int, draws: int = DRAWS, seed: int = 0) -> np.ndarray:
    """``(draws, clusters)``: how often each cluster is drawn in each bootstrap draw (``clusters`` drawn with
    replacement from ``clusters``)."""
    rng = np.random.default_rng(seed)
    if clusters <= 0:
        return np.zeros((draws, 0))
    return rng.multinomial(clusters, np.full(clusters, 1.0 / clusters), size=draws).astype(np.float64)


def bootstrap_rate(flags, clusters=None, draws: int = DRAWS, seed: int = 0, level: float = LEVEL) -> dict:
    """The share of ``flags`` that are true, with an interval from a bootstrap over ``clusters`` (labels, one per
    flag; default: each flag its own): ``{rate, interval, n, k, clusters, kind}``. When no flag (or every flag) is
    true the bootstrap's interval would be a point; the interval is then the exact bound with the clusters as the
    independent units (``kind`` ``"exact, clusters"``): ``[0, 1 - ((1 - level) / 2) ** (1 / clusters)]``, about three
    over the clusters at 95 %, valid for the share of flags too (a share of flags is at most the share of clusters
    with any)."""
    flags = np.asarray(flags, dtype=bool).ravel()
    n = len(flags)
    if not n:
        return {"rate": None, "interval": [None, None], "n": 0, "k": 0, "clusters": 0, "kind": None}
    index, count = _clusters(clusters, n)
    k = int(flags.sum())
    out = {"rate": k / n, "n": n, "k": k, "clusters": count}
    tail = (1.0 - level) / 2.0
    if k == 0 or k == n:
        bound = tail ** (1.0 / count)
        out["interval"] = [0.0, 1.0 - bound] if k == 0 else [bound, 1.0]
        out["kind"] = "exact, clusters"
        return out
    hits = np.bincount(index, weights=flags.astype(np.float64), minlength=count)
    trials = np.bincount(index, minlength=count).astype(np.float64)
    boot = cluster_counts(count, draws, seed)
    with np.errstate(invalid="ignore", divide="ignore"):
        rates = (boot @ hits) / (boot @ trials)
    out["interval"] = _interval(rates, level)
    out["kind"] = "bootstrap"
    return out


def _binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p)."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    logs = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * math.log(p)
            + (n - i) * math.log1p(-p) for i in range(k + 1)]
    top = max(logs)
    return float(min(1.0, math.exp(top) * sum(math.exp(v - top) for v in logs)))


def binomial_interval(k: int, n: int, level: float = LEVEL) -> list[float]:
    """The exact (Clopper-Pearson) two-sided ``level`` interval of a rate with ``k`` events in ``n`` independent
    trials. With ``k = 0`` the upper end is ``1 - ((1 - level) / 2) ** (1 / n)``."""
    k, n = int(k), int(n)
    if n <= 0:
        return [0.0, 1.0]
    if not 0 <= k <= n:
        raise ValueError(f"{k} events in {n} trials")
    tail = (1.0 - level) / 2.0

    def solve(target, cdf_k, rising):
        lo, hi = 0.0, 1.0
        for _ in range(80):
            mid = (lo + hi) / 2.0
            value = _binom_cdf(cdf_k, n, mid)
            if (value > target) == rising:
                hi = mid
            else:
                lo = mid
        return (lo + hi) / 2.0

    lower = 0.0 if k == 0 else solve(1.0 - tail, k - 1, False)  # P(X >= k | p) = tail
    upper = 1.0 if k == n else solve(tail, k, False)  # P(X <= k | p) = tail
    return [lower, upper]


# -- pairs (verification) ---------------------------------------------------------------------------------------------


def roc(genuine, impostor, genuine_weights=None, impostor_weights=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(far, tar, thresholds)`` at every distinct score, thresholds falling, starting at ``+inf`` (nothing
    accepted): the false-accept rate (impostor scores at or above the threshold) and the true-accept rate (genuine
    ones). ``*_weights``: per score (a bootstrap draw's counts)."""
    genuine, impostor = _scores(genuine), _scores(impostor)
    gw, iw = _weights(genuine_weights, len(genuine)), _weights(impostor_weights, len(impostor))
    thresholds = np.r_[np.inf, np.unique(np.r_[genuine, impostor])[::-1]]
    return (_share_at_or_above(impostor, iw, thresholds), _share_at_or_above(genuine, gw, thresholds), thresholds)


def det(genuine, impostor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(far, frr, thresholds)``: the detection error trade-off, the false-reject rate ``1 - tar`` against the
    false-accept rate (plot both through :func:`probit`)."""
    far, tar, thresholds = roc(genuine, impostor)
    return far, 1.0 - tar, thresholds


def tar_at_far(genuine, impostor, far: float, genuine_weights=None, impostor_weights=None) -> dict:
    """The true-accept rate at the strictest threshold whose false-accept rate is at most ``far`` (the most genuine
    pairs accepted while at most ``far`` of the impostor pairs are): ``{tar, far (achieved), threshold,
    impostors_allowed, resolvable}``. ``resolvable``: the impostor pairs are enough for the asked rate to let at
    least one through (``far * n >= 1``); otherwise the threshold lets none through and the TAR is that at a
    false-accept rate of zero (a bound, not the asked operating point)."""
    rates, tars, thresholds = roc(genuine, impostor, genuine_weights, impostor_weights)
    ok = np.flatnonzero(rates <= far + 1e-12)
    k = ok[np.argmax(tars[ok])]  # (the first point, +inf, is always ok; of equal TARs the strictest threshold)
    threshold = float(thresholds[k])
    n = len(_scores(impostor)) if impostor_weights is None else float(np.sum(impostor_weights))
    return {"tar": float(tars[k]), "far": float(rates[k]), "threshold": threshold,
            "impostors_allowed": math.floor(far * n + 1e-9), "resolvable": bool(far * n >= 1.0 - 1e-9)}


def far_levels(impostors: float, levels=FAR_LEVELS, min_count: float = 1.0) -> list[float]:
    """The ``levels`` that ``impostors`` comparisons resolve: those letting at least ``min_count`` through."""
    return [lv for lv in levels if lv * impostors >= min_count - 1e-9]


def eer(genuine, impostor, genuine_weights=None, impostor_weights=None) -> dict:
    """The equal error rate ``{eer, threshold}``: where the false-reject rate (genuine scores below the threshold)
    meets the false-accept rate, interpolated linearly between the two thresholds where they cross."""
    genuine, impostor = _scores(genuine), _scores(impostor)
    if not len(genuine) or not len(impostor):
        raise ValueError("the equal error rate needs genuine and impostor scores")
    gw, iw = _weights(genuine_weights, len(genuine)), _weights(impostor_weights, len(impostor))
    thresholds = np.unique(np.r_[genuine, impostor, np.inf])
    far = _share_at_or_above(impostor, iw, thresholds)
    frr = 1.0 - _share_at_or_above(genuine, gw, thresholds)
    gap = frr - far  # rises with the threshold, from <= 0 to >= 0
    k = int(np.argmax(gap >= 0))
    if k == 0 or gap[k] == 0:
        return {"eer": float((far[k] + frr[k]) / 2.0), "threshold": float(thresholds[k])}
    a, b = gap[k - 1], gap[k]  # a < 0 < b
    u = -a / (b - a)
    rate = (1 - u) * (far[k - 1] + frr[k - 1]) / 2.0 + u * (far[k] + frr[k]) / 2.0
    t0, t1 = thresholds[k - 1], thresholds[k]
    return {"eer": float(rate), "threshold": float(t0 + u * (t1 - t0)) if np.isfinite(t1) else float(t0)}


def d_prime(genuine, impostor, genuine_weights=None, impostor_weights=None) -> float:
    """How far apart the means are, in root mean variances: ``(mean_g - mean_i) / sqrt((var_g + var_i) / 2)``."""
    genuine, impostor = _scores(genuine), _scores(impostor)
    gw, iw = _weights(genuine_weights, len(genuine)), _weights(impostor_weights, len(impostor))
    mg, mi = np.average(genuine, weights=gw), np.average(impostor, weights=iw)
    vg, vi = np.average((genuine - mg) ** 2, weights=gw), np.average((impostor - mi) ** 2, weights=iw)
    spread = math.sqrt((vg + vi) / 2.0)
    return float((mg - mi) / spread) if spread > 0 else math.inf


def probit(p) -> np.ndarray:
    """The standard normal quantile of ``p`` (0-1), for DET axes (Acklam's approximation with one Newton step:
    about 1e-9 accurate; 0 and 1 give infinities)."""
    p = np.asarray(p, dtype=np.float64)
    a = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02, 1.383577518672690e02,
         -3.066479806614716e01, 2.506628277459239e00)
    b = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02, 6.680131188771972e01,
         -1.328068155288572e01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00, -2.549732539343734e00,
         4.374664141464968e00, 2.938163982698783e00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00)
    out = np.full(p.shape, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        low, high = p < 0.02425, p > 1 - 0.02425
        mid = ~low & ~high & (p > 0) & (p < 1)
        q = np.sqrt(-2 * np.log(np.where(low, p, 0.5)))
        out = np.where(low, (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                       / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1), out)
        q = np.sqrt(-2 * np.log(np.where(high, 1 - p, 0.5)))
        out = np.where(high, -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                       / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1), out)
        q = np.where(mid, p - 0.5, 0.0)
        r = q * q
        out = np.where(mid, (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
                       / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1), out)
        inside = (p > 0) & (p < 1)
        x = np.where(inside, out, 0.0)
        error = 0.5 * np.array([math.erfc(-v / math.sqrt(2)) for v in x.ravel()]).reshape(x.shape) - p
        step = error * math.sqrt(2 * math.pi) * np.exp(x * x / 2)
        x = x - step / (1 + x * step / 2)
        out = np.where(inside, x, np.where(p <= 0, -np.inf, np.where(p >= 1, np.inf, np.nan)))
    return out if out.shape else float(out)


def pair_weights(counts, genuine_index, impostor_index) -> tuple[np.ndarray, np.ndarray]:
    """A bootstrap draw's weights of the pairs (``counts``: how often each person was drawn): a genuine pair counts
    as often as its person (``genuine_index``: the person of each), an impostor pair as the product of its two
    people's counts (``impostor_index``, ``(n, 2)``; a second index of ``-1``: the pair is a cluster of its own)."""
    counts = np.asarray(counts, dtype=np.float64)
    impostor_index = np.asarray(impostor_index).reshape(-1, 2)
    second = np.where(impostor_index[:, 1] >= 0, counts[np.maximum(impostor_index[:, 1], 0)], 1.0)
    return counts[np.asarray(genuine_index)], counts[impostor_index[:, 0]] * second


def pairs_report(genuine, impostor, genuine_people=None, impostor_people=None, far=FAR_LEVELS,
                 draws: int = DRAWS, seed: int = 0, level: float = LEVEL) -> dict:
    """The verification measures of ``genuine`` and ``impostor`` scores with intervals from a bootstrap over people:
    ``genuine_people`` (one person per genuine pair) and ``impostor_people`` (``(n, 2)``: the two people of each
    impostor pair); a drawn person's pairs count as often as they were drawn, an impostor pair with the product of
    its two people's counts. Without people the pairs are taken as independent (an interval over pairs). TAR at each
    of ``far`` that the impostor pairs resolve (:func:`far_levels`), the EER and d' (with intervals), and the counts."""
    genuine, impostor = _scores(genuine), _scores(impostor)
    levels = far_levels(len(impostor), far)
    out = {"genuine": len(genuine), "impostor": len(impostor), "d_prime": d_prime(genuine, impostor),
           "eer": eer(genuine, impostor)["eer"],
           "tar_at_far": {f"{f:g}": tar_at_far(genuine, impostor, f) for f in levels}}
    if genuine_people is None or impostor_people is None:  # every pair its own cluster
        people = len(genuine) + len(impostor)
        g_index = np.arange(len(genuine))
        i_index = np.stack([np.arange(len(impostor)) + len(genuine), np.full(len(impostor), -1)], axis=1)
        out["people"] = None
    else:
        names = np.unique(np.r_[np.asarray(genuine_people, str), np.asarray(impostor_people, str).ravel()])
        g_index = np.searchsorted(names, np.asarray(genuine_people, str))
        i_index = np.searchsorted(names, np.asarray(impostor_people, str).reshape(-1, 2))
        people = len(names)
        out["people"] = people
    boot = {"d_prime": [], "eer": [], **{f"{f:g}": [] for f in levels}}
    for row in cluster_counts(people, draws, seed):
        gw, iw = pair_weights(row, g_index, i_index)
        if gw.sum() <= 0 or iw.sum() <= 0:
            continue
        boot["d_prime"].append(d_prime(genuine, impostor, gw, iw))
        boot["eer"].append(eer(genuine, impostor, gw, iw)["eer"])
        for f in levels:
            boot[f"{f:g}"].append(tar_at_far(genuine, impostor, f, gw, iw)["tar"])
    out["d_prime_interval"] = _interval(np.array(boot["d_prime"]), level)
    out["eer_interval"] = _interval(np.array(boot["eer"]), level)
    for f in levels:
        out["tar_at_far"][f"{f:g}"]["interval"] = _interval(np.array(boot[f"{f:g}"]), level)
    out["level"], out["draws"] = level, draws
    return out


# -- open-set identification ------------------------------------------------------------------------------------------


def _searches(scores, mate) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2:
        raise ValueError("scores must be searches x enrolled people")
    if np.isnan(scores).any():
        raise ValueError("scores must not be NaN (use -inf where a person has no score)")
    mate = np.asarray(mate, dtype=np.int64).ravel()
    if len(mate) != len(scores):
        raise ValueError(f"{len(mate)} mates for {len(scores)} searches")
    if (mate >= scores.shape[1]).any() or (mate < -1).any():
        raise ValueError("a mate must be a column of the scores, or -1 for a stranger")
    return scores, mate


def ranks(scores, mate) -> np.ndarray:
    """The rank of each mated search's mate (1: the best; ties count against it: 1 + the people scoring at least as
    high); 0 for non-mated searches."""
    scores, mate = _searches(scores, mate)
    out = np.zeros(len(mate), np.int64)
    rows = np.flatnonzero(mate >= 0)
    if len(rows):
        own = scores[rows, mate[rows]]
        out[rows] = (scores[rows] >= own[:, None]).sum(axis=1)  # (the mate itself counts once: rank 1 alone)
    return out


def cmc(scores, mate, top: int = 20) -> np.ndarray:
    """The cumulative match characteristic: the share of mated searches whose mate is within rank ``k``, for ``k`` =
    1..``top`` (closed set: a sanity check)."""
    r = ranks(scores, mate)
    r = r[np.asarray(mate) >= 0]
    if not len(r):
        return np.full(top, np.nan)
    return np.array([(r <= k).mean() for k in range(1, top + 1)])


def open_set(scores, mate, weights=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(fpir, fnir, thresholds)`` at every distinct score (thresholds falling, from ``+inf``): the share of
    non-mated searches whose best score reaches the threshold, and the share of mated searches whose mate is not the
    best candidate at or above it (the detection and identification rate is ``1 - fnir``). ``weights``: per search."""
    scores, mate = _searches(scores, mate)
    w = _weights(weights, len(mate))
    best = scores.max(axis=1) if scores.shape[1] else np.full(len(mate), -np.inf)
    first = ranks(scores, mate) == 1
    stranger = mate < 0
    found = np.where(~stranger & first, scores[np.arange(len(mate)), np.maximum(mate, 0)], -np.inf)
    values = np.r_[best[stranger], found[~stranger]]
    thresholds = np.r_[np.inf, np.unique(values[np.isfinite(values)])[::-1]]
    fpir = _share_at_or_above(best[stranger], w[stranger], thresholds) if stranger.any() else \
        np.full(len(thresholds), np.nan)
    dir_ = _share_at_or_above(found[~stranger], w[~stranger], thresholds) if (~stranger).any() else \
        np.full(len(thresholds), np.nan)
    return fpir, 1.0 - dir_, thresholds


def fnir_at_fpir(scores, mate, fpir: float, weights=None) -> dict:
    """The FNIR at the strictest threshold whose FPIR is at most ``fpir``: ``{fnir, dir, fpir (achieved), threshold,
    strangers_allowed, resolvable}`` (``resolvable``: the non-mated searches let at least one through at ``fpir``)."""
    rates, misses, thresholds = open_set(scores, mate, weights)
    ok = np.flatnonzero(rates <= fpir + 1e-12)
    k = ok[np.argmin(misses[ok])]
    _, m = _searches(scores, mate)
    w = _weights(weights, len(m))
    n = float(w[m < 0].sum())
    return {"fnir": float(misses[k]), "dir": float(1.0 - misses[k]), "fpir": float(rates[k]),
            "threshold": float(thresholds[k]), "strangers_allowed": math.floor(fpir * n + 1e-9),
            "resolvable": bool(fpir * n >= 1.0 - 1e-9)}


def open_set_report(scores, mate, people=None, levels=FPIR_LEVELS, draws: int = DRAWS, seed: int = 0,
                    level: float = LEVEL, top: int = 10) -> dict:
    """The open-set measures with intervals from a bootstrap over ``people`` (one label per search: whose face it is;
    default: every search its own): FNIR and DIR at each FPIR of ``levels`` the non-mated searches resolve, the rank-1
    rate and the CMC to ``top``, and the counts (searches, mated and not, people)."""
    scores, mate = _searches(scores, mate)
    index, count = _clusters(people, len(mate))
    strangers = int((mate < 0).sum())
    resolved = [lv for lv in levels if lv * strangers >= 1.0 - 1e-9]
    r = ranks(scores, mate)
    out = {"searches": len(mate), "mated": int((mate >= 0).sum()), "nonmated": strangers,
           "people": count if people is not None else None, "enrolled": int(scores.shape[1]),
           "rank1": float((r[mate >= 0] == 1).mean()) if (mate >= 0).any() else None,
           "cmc": [round(float(v), 6) for v in cmc(scores, mate, top)],
           "fnir_at_fpir": {f"{lv:g}": fnir_at_fpir(scores, mate, lv) for lv in resolved}}
    boot = {f"{lv:g}": [] for lv in resolved}
    boot["rank1"] = []
    for row in cluster_counts(count, draws, seed):
        w = row[index]
        mated_w = w[mate >= 0]
        if mated_w.sum() > 0:
            boot["rank1"].append(float((mated_w * (r[mate >= 0] == 1)).sum() / mated_w.sum()))
        if w[mate < 0].sum() <= 0 or mated_w.sum() <= 0:
            continue
        for lv in resolved:
            boot[f"{lv:g}"].append(fnir_at_fpir(scores, mate, lv, w)["fnir"])
    out["rank1_interval"] = _interval(np.array(boot["rank1"]), level)
    for lv in resolved:
        out["fnir_at_fpir"][f"{lv:g}"]["interval"] = _interval(np.array(boot[f"{lv:g}"]), level)
    out["level"], out["draws"] = level, draws
    return out


# -- tracks -----------------------------------------------------------------------------------------------------------


class NameEvents:
    """What a visitor sees of the names over a run, beyond :class:`~.truth.Outcomes`: per frame (:meth:`update`),
    the name each track shows. Counts ``switches`` (a track that showed one name shows another later, whatever it
    showed between: ``{track: [names in order]}`` in ``names``) and **one name on two tracks** at once: the frames and
    seconds a name was on more than one track (``shared_frames``, ``shared_seconds``, ``shared``: ``{name: [track
    ids]}``). Tracks are the recogniser's numbers; the frames' ``now`` is in seconds."""

    def __init__(self):
        self.names: dict[int, list[str]] = {}
        self.shared: dict[str, set] = {}
        self.shared_frames = 0
        self.shared_seconds = 0.0
        self.frames = 0
        self.seconds = 0.0
        self._last: float | None = None

    def update(self, now: float, decisions: dict) -> None:
        """Count a frame: ``decisions`` maps a track's id to its :class:`~.decision.Decision` (or ``None``)."""
        dt = 0.0 if self._last is None else max(0.0, float(now) - self._last)
        self._last = float(now)
        self.frames += 1
        self.seconds += dt
        holders: dict[str, list[int]] = {}
        for track, decision in decisions.items():
            if decision is None or getattr(decision, "state", None) != "named" or not decision.name:
                continue
            holders.setdefault(decision.name, []).append(int(track))
            shown = self.names.setdefault(int(track), [])
            if not shown or shown[-1] != decision.name:
                shown.append(decision.name)
        twice = {name: tracks for name, tracks in holders.items() if len(tracks) > 1}
        if twice:
            self.shared_frames += 1
            self.shared_seconds += dt
            for name, tracks in twice.items():
                self.shared.setdefault(name, set()).update(tracks)

    @property
    def switches(self) -> int:
        """Name changes within tracks (A to B counts one, A to B to A two)."""
        return sum(max(0, len(v) - 1) for v in self.names.values())

    def summary(self) -> dict:
        return {"frames": self.frames, "seconds": round(self.seconds, 3), "named_tracks": len(self.names),
                "switches": self.switches, "tracks_switching": sum(len(v) > 1 for v in self.names.values()),
                "shared_frames": self.shared_frames, "shared_seconds": round(self.shared_seconds, 3),
                "shared": {name: sorted(tracks) for name, tracks in sorted(self.shared.items())}}


def track_summary(segments, shown: float = 1.0, within=(2.0, 5.0), identity=None, draws: int = DRAWS,
                  seed: int = 0, level: float = LEVEL) -> dict:
    """Track-level measures of :class:`~.truth.Segment` items (``Outcomes.segments.values()``, of one run or
    several) with intervals from a bootstrap over people (``identity``: a segment's person, default its ``avatar``,
    the identity scored against): over the known people's segments whose face was shown at least ``shown`` seconds,
    named correctly within each of ``within`` seconds and at all, the time to the first correct name (median, 90th
    percentile), "unknown" said to them, wrong names (also over every known segment); over the strangers' eligible
    segments, named at all (the false-alarm rate per track) and called unknown, and over all of theirs; wrong names
    per track and per hour of followed people (each segment's span) and the people and segments counted."""
    identity = identity or (lambda g: g.avatar)
    segs = list(segments)
    known = [g for g in segs if g.known and g.eligible(shown)]
    all_known = [g for g in segs if g.known]
    strangers = [g for g in segs if not g.known and g.eligible(shown)]
    all_strangers = [g for g in segs if not g.known]

    def rate(items, test):
        return bootstrap_rate([bool(test(g)) for g in items], [identity(g) for g in items], draws, seed, level)

    times = [g.correct_at - g.first_shown for g in known
             if g.correct_at is not None and g.first_shown is not None and g.correct_at >= g.first_shown]
    hours = sum(max(0.0, g.end - g.start) for g in segs) / 3600.0
    wrong = sum(1 for g in all_known if g.wrong) + sum(1 for g in all_strangers if g.named_at is not None)
    out = {
        "segments": len(segs), "people": len({identity(g) for g in segs}),
        "known": {"eligible": len(known), "people": len({identity(g) for g in known}), "all": len(all_known),
                  "named_within": {f"{t:g}": rate(known, lambda g, t=t: g.named_within(t)) for t in within},
                  "named": rate(known, lambda g: g.correct_at is not None),
                  "time_to_name": {"median": float(np.median(times)) if times else None,
                                   "p90": float(np.quantile(times, 0.9)) if times else None, "n": len(times)},
                  "unknown": rate(known, lambda g: g.unknown_at is not None),
                  "wrong_eligible": rate(known, lambda g: bool(g.wrong)),
                  "wrong_all": rate(all_known, lambda g: bool(g.wrong))},
        "strangers": {"eligible": len(strangers), "people": len({identity(g) for g in strangers}),
                      "all": len(all_strangers),
                      "fpir": rate(strangers, lambda g: g.named_at is not None),
                      "called_unknown": rate(strangers, lambda g: g.unknown_at is not None),
                      "fpir_all": rate(all_strangers, lambda g: g.named_at is not None)},
        "wrong_names": wrong, "hours_followed": round(hours, 4),
        "wrong_per_track": wrong / len(segs) if segs else None,
        "wrong_per_hour": wrong / hours if hours > 0 else None,
    }
    if not strangers and all_strangers:
        out["strangers"]["note"] = "no stranger's face was shown long enough"
    return out


# -- calibration ------------------------------------------------------------------------------------------------------


def achieved(named, asked: float, groups=None, clusters=None, factor: float = 2.0, draws: int = DRAWS, seed: int = 0,
             level: float = LEVEL) -> dict:
    """The false-alarm rate measured against the one ``asked``, per group (``groups``: a label per stranger's search
    or track, e.g. its face-size bin; ``None``: all together): ``{group: {n, k, rate, interval, ratio, within}}``,
    the interval from a bootstrap over ``clusters`` (people or identity pairs), ``ratio`` the rate over the asked
    one, ``within``: ``"yes"`` when the rate is within ``factor`` of the asked one either way, ``"consistent"`` when
    only its interval reaches that band, else ``"no"``."""
    named = np.asarray(named, bool).ravel()
    groups = np.full(len(named), "all") if groups is None else np.asarray(groups).astype(str)
    clusters = None if clusters is None else np.asarray(clusters)
    out = {}
    low, high = asked / factor, asked * factor
    for group in sorted(set(groups.tolist())):
        here = groups == group
        r = bootstrap_rate(named[here], None if clusters is None else clusters[here], draws, seed, level)
        rate, (lo, hi) = r["rate"], r["interval"]
        if rate is not None and low <= rate <= high:
            within = "yes"
        elif lo is not None and hi is not None and hi >= low and lo <= high:
            within = "consistent"
        else:
            within = "no"
        out[group] = {"n": r["n"], "k": r["k"], "clusters": r["clusters"], "rate": rate, "interval": r["interval"],
                      "kind": r["kind"], "ratio": None if rate is None else rate / asked, "within": within}
    return out


# -- acceptance -------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Acceptance:
    """The acceptance test of ``docs/research/face_recognition_sota.md`` (section 4.4): at least ``named`` of the
    enrolled people's tracks whose face was shown (yaw under 45 degrees) at least ``shown`` seconds are named
    correctly within ``seconds``, at a false-alarm rate per stranger track of ``fpir``; the rate achieved within
    ``factor`` of it either way; no identity on two tracks at once."""

    named: float = 0.95
    seconds: float = 2.0
    shown: float = 1.0
    fpir: float = 0.01
    factor: float = 2.0


def acceptance(summary: dict, names: dict | None = None, rule: Acceptance | None = None) -> dict:
    """Judge a run (or runs) against ``rule``: ``summary`` from :func:`track_summary` (with ``within`` holding
    ``rule.seconds`` and ``shown`` = ``rule.shown``), ``names`` from :meth:`NameEvents.summary` (summed over the runs
    by the caller; ``None``: not counted). Each criterion: the measured value, its interval and the counts, a
    ``status`` and ``confident`` (whether its interval also meets it):

    * ``named_within``: ``"pass"`` when the share named correctly within ``rule.seconds`` is at least ``rule.named``;
    * ``fpir_within_factor``: ``"pass"`` when the false-alarm rate per stranger track is within ``rule.factor`` of
      ``rule.fpir`` either way, ``"consistent"`` when only its interval reaches that band (no stranger named among a
      few: the counts cannot tell), ``"fail"`` when even the interval misses it (far too many false alarms, or none
      over so many strangers that the calibration is far too strict);
    * ``one_identity_per_track``: ``"pass"`` when no name was on two tracks at once.

    The ``verdict`` passes when ``named_within`` and ``one_identity_per_track`` pass and the false-alarm rate is not
    ``"fail"``; it is ``confident`` when every interval meets its criterion too. Whether the false-alarm rate is at
    most the target (``fpir``) is reported, not judged: the research's criterion is the factor. ``rule``: default
    :class:`Acceptance`'s."""
    rule = rule or Acceptance()
    known, strangers = summary["known"], summary["strangers"]
    named = known["named_within"].get(f"{rule.seconds:g}") or {}
    fpir = strangers["fpir"]
    rate, (lo, hi) = named.get("rate"), named.get("interval") or [None, None]
    criteria = {"named_within": {
        "value": rate, "interval": [lo, hi], "tracks": known["eligible"], "people": known["people"],
        "status": "pass" if rate is not None and rate >= rule.named - 1e-12 else "fail",
        "confident": lo is not None and lo >= rule.named - 1e-12}}
    rate_f, (lo_f, hi_f) = fpir.get("rate"), fpir.get("interval") or [None, None]
    low, high = rule.fpir / rule.factor, rule.fpir * rule.factor
    if rate_f is not None and low <= rate_f <= high:
        within = "pass"
    elif lo_f is not None and hi_f is not None and hi_f >= low and lo_f <= high:
        within = "consistent"
    else:
        within = "fail"
    criteria["fpir_within_factor"] = {
        "value": rate_f, "interval": [lo_f, hi_f], "band": [low, high], "tracks": strangers["eligible"],
        "people": strangers["people"], "status": within,
        "confident": within == "pass" and lo_f is not None and lo_f >= low and hi_f <= high}
    if names is not None:
        criteria["one_identity_per_track"] = {
            "value": int(names.get("shared_frames", 0)), "seconds": float(names.get("shared_seconds", 0.0)),
            "status": "fail" if names.get("shared_frames") else "pass", "confident": not names.get("shared_frames")}
    verdict = all(c["status"] == "pass" for key, c in criteria.items() if key != "fpir_within_factor") and \
        criteria["fpir_within_factor"]["status"] != "fail"
    return {"rule": asdict(rule), "criteria": criteria,
            "fpir": {"value": rate_f, "asked": rule.fpir,
                     "at_most_asked": rate_f is not None and rate_f <= rule.fpir + 1e-12},
            "verdict": "pass" if verdict else "fail",
            "confident": verdict and all(c["confident"] for c in criteria.values())}


# -- cost -------------------------------------------------------------------------------------------------------------


def cost_summary(timings: dict, embeddings: int = 0, seconds: float = 0.0, frames: int = 0) -> dict:
    """Milliseconds per stage (``timings``: ``{stage: [ms per frame]}``; median, 90th percentile, mean), embeddings a
    second of the run's clock and frames analysed."""
    out = {"stages": {}}
    for stage, values in timings.items():
        v = np.asarray(values, dtype=np.float64)
        if len(v):
            out["stages"][stage] = {"median": float(np.median(v)), "p90": float(np.quantile(v, 0.9)),
                                    "mean": float(v.mean()), "n": len(v)}
    out["embeddings"] = int(embeddings)
    out["embeddings_per_s"] = embeddings / seconds if seconds > 0 else None
    out["frames"] = int(frames)
    return out


def peak_memory_mb() -> float:
    """The process's peak resident memory so far (MB)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(peak) / (1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0)
