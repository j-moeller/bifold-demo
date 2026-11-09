"""The evidence of who a person track shows: which faces are worth embedding, which gets the next embedding, and what
a track's embeddings add up to.

Embedding a face is the expensive step (on four CPU cores AdaFace IR-101 takes 0.5-1 s, the ResNet-50 0.1-0.2 s, SFace
15-30 ms), so only faces worth it are embedded, at a rate the machine can keep up:

* :class:`QualityGate`: a face is embedded only when it is big enough (its eye distance turned to the camera, from
  its box: :func:`~.heads.face_size`, in the pixels it is aligned from), turned less than ``max_yaw``, sharp, mostly in
  the picture, confidently found and, where the profile has eDifFIQA-T, recognisable enough by it. A turned or tiny
  face adds nothing: the track waits for a better one.
* :class:`EmbeddingBudget`: embeddings a second for the whole system, a token bucket filled by the frames' clock
  (scene time for a scene, so that runs are deterministic; the camera's for a webcam). :data:`RATES` gives each
  model's rate on a CPU like the development machine's (four shared cores); the Mac's CoreML is expected to allow ten
  times more.
* :class:`EmbeddingScheduler`: which tracks' faces get the budget of a frame. Tracks not decided yet come before
  decided ones, better faces first; a decided track is re-checked now and then (every :attr:`~EmbeddingScheduler.
  recheck` seconds), so that a wrong or outdated name does not stand forever. A track is embedded at most
  :attr:`~EmbeddingScheduler.per_track` times a second, and once it has a few frames only faces better than its
  median so far (the research's "embed a track at most twice a second, and only frames better than its median").

:class:`TrackEvidence` keeps a track's embeddings of one model (embeddings of different models cannot be mixed): the
best :attr:`~TrackEvidence.keep` frames by their quality weight (:attr:`~.quality.Quality.weight`) with a time decay
(half the weight every :attr:`~TrackEvidence.half_life` seconds), and their **quality-weighted mean of unit
embeddings** (:meth:`TrackEvidence.aggregate`). Consecutive frames of one face are nearly the same picture, so they
are not independent evidence: the decision scores the aggregate once (:mod:`.decision`), it never sums per-frame
scores. The aggregate also says how good its frames were (their weighted face size and yaw): the calibration's bin.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# embeddings a second for the whole system, per model, on four shared x86 cores (measured: AdaFace IR-101 0.5-1.0 s a
# face under load, the ResNet-50 0.1-0.2 s, SFace 15-30 ms; see docs/research/face_recognition_f2_results.md)
RATES = {"adaface_ir101_wf12m": 1.0, "antelopev2_r100": 1.0, "auraface_v1": 1.0, "buffalo_l_r50": 4.0,
         "sface_2021dec": 20.0}
DEFAULT_RATE = 2.0  # a model not listed

__all__ = ["DEFAULT_RATE", "RATES", "Aggregate", "Candidate", "EmbeddingBudget", "EmbeddingScheduler",
           "FrameEvidence", "QualityGate", "TrackEvidence", "rate_for"]


def rate_for(model_id: str) -> float:
    """The embeddings a second :data:`RATES` gives ``model_id`` (:data:`DEFAULT_RATE` for others)."""
    return float(RATES.get(model_id, DEFAULT_RATE))


@dataclass(frozen=True)
class QualityGate:
    """Which faces are worth embedding (see the module docs). ``min_size``: the face's eye distance turned to the
    camera, in the pixels it is aligned from; ``max_yaw``: degrees; ``min_sharpness``: of the aligned face
    (:func:`~.quality.image_measures`; sharp faces 1 and more, a blur of 3 crop pixels 0.02-0.09); ``max_outside``: the
    share of the aligned crop beyond the picture; ``min_score``: the detector's; ``min_fiqa``: eDifFIQA-T's (renders
    0.5 at 12 pixels between the eyes, 0.64 at 32; profile views 0.37-0.59)."""

    min_size: float = 8.0
    max_yaw: float = 55.0
    min_sharpness: float = 0.05
    max_outside: float = 0.2
    min_score: float = 0.5
    min_fiqa: float = 0.3

    def check(self, quality, size: float, score: float | None = None) -> str | None:
        """``None`` when a face of this :class:`~.quality.Quality`, ``size`` (eye distance turned to the camera) and
        detector ``score`` (default: the quality's) is worth embedding, else why not (``"small"``, ``"turned"``,
        ``"blurred"``, ``"cut off"``, ``"unsure"`` or ``"hard"``)."""
        score = quality.detector if score is None else score
        if size < self.min_size:
            return "small"
        if abs(quality.yaw) > self.max_yaw:
            return "turned"
        if quality.sharpness < self.min_sharpness:
            return "blurred"
        if quality.outside > self.max_outside:
            return "cut off"
        if score < self.min_score:
            return "unsure"
        if quality.fiqa is not None and quality.fiqa < self.min_fiqa:
            return "hard"
        return None


class EmbeddingBudget:
    """``rate`` embeddings a second at most, for the whole system: a bucket of tokens filled by the frames' clock up to
    ``burst`` (default: one second's worth, at least one). Deterministic: it only reads the times it is given."""

    def __init__(self, rate: float, burst: float | None = None):
        if rate <= 0:
            raise ValueError(f"an embedding budget needs a rate above 0, not {rate}")
        self.rate = float(rate)
        self.burst = max(1.0, self.rate if burst is None else float(burst))
        self.tokens = 1.0  # one at once: the first good face need not wait
        self.time: float | None = None
        self.spent = 0

    def available(self, now: float) -> int:
        """Whole embeddings that may be made at ``now`` (the bucket filled since the last call)."""
        now = float(now)
        if self.time is not None and now > self.time:
            self.tokens = min(self.burst, self.tokens + (now - self.time) * self.rate)
        if self.time is None or now > self.time:
            self.time = now
        return math.floor(self.tokens + 1e-9)

    def take(self, count: int = 1) -> None:
        """Spend ``count`` embeddings (call :meth:`available` first)."""
        self.tokens = max(0.0, self.tokens - count)
        self.spent += count


@dataclass(frozen=True)
class Candidate:
    """A face that passed the gate, waiting for an embedding: the ``track``'s id, the face's quality ``weight``, the
    ``key`` the caller finds it by (any object), whether the track is ``decided`` (named or unknown), when the track
    was last embedded (``last``; ``None``: never) and its frames' median weight so far (``median``; ``None``: fewer
    than :attr:`EmbeddingScheduler.median_after` frames)."""

    track: int
    weight: float
    key: object = None
    decided: bool = False
    last: float | None = None
    median: float | None = None


@dataclass
class EmbeddingScheduler:
    """Which candidates get the frame's embeddings (see the module docs). ``recheck``: seconds after which a decided
    track's face is embedded again; ``per_track``: the most embeddings a track gets a second; ``median_after``: frames
    a track needs before only faces better than its median count."""

    recheck: float = 5.0
    per_track: float = 2.0
    median_after: int = 4

    def eligible(self, candidate: Candidate, now: float) -> bool:
        """Whether ``candidate`` may be embedded at ``now`` at all."""
        if candidate.last is not None and now - candidate.last < 1.0 / self.per_track - 1e-9:
            return False
        if candidate.decided and candidate.last is not None and now - candidate.last < self.recheck:
            return False
        return candidate.median is None or candidate.weight > candidate.median

    def choose(self, candidates, count: int, now: float) -> list[Candidate]:
        """At most ``count`` of ``candidates`` (one per track), in the order they should be embedded: undecided
        tracks first, then decided ones due for a check; better faces first, then the track waiting longest."""
        best: dict[int, Candidate] = {}
        for c in candidates:
            if self.eligible(c, now) and (c.track not in best or c.weight > best[c.track].weight):
                best[c.track] = c
        ranked = sorted(best.values(), key=lambda c: (c.decided, -c.weight, -math.inf if c.last is None else c.last,
                                                      c.track))
        return ranked[:max(0, int(count))]


@dataclass(frozen=True)
class FrameEvidence:
    """One embedded face of a track: its unit ``vector``, quality ``weight`` (0-1), ``time``, ``size`` (eye distance
    turned to the camera, in the pixels it was aligned from), ``yaw`` (degrees), the ``model`` that embedded it, its
    full :class:`~.quality.Quality` and where it came from (``source``: ``"window"`` or ``"frame"``; ``frame``: the
    frame's index)."""

    vector: np.ndarray = field(repr=False)
    weight: float
    time: float
    size: float
    yaw: float
    model: str = ""
    quality: object = field(default=None, repr=False)
    source: str = "frame"
    frame: int | None = None

    def decayed(self, now: float, half_life: float) -> float:
        age = max(0.0, now - self.time)
        return self.weight * 0.5 ** (age / half_life) if half_life > 0 else self.weight


@dataclass(frozen=True)
class Aggregate:
    """What a track's frames add up to: the unit ``vector`` (their weighted mean), the summed decayed ``weight``, the
    weighted face ``size`` and ``yaw`` (absolute), the number of ``frames``, the ``model``, and the ``best`` weight of a
    single frame."""

    vector: np.ndarray = field(repr=False)
    weight: float
    size: float
    yaw: float
    frames: int
    model: str
    best: float


@dataclass
class TrackEvidence:
    """A track's embeddings of one ``model`` (see the module docs): the best ``keep`` frames by their weight decayed
    by ``half_life`` seconds."""

    model: str
    keep: int = 16
    half_life: float = 20.0
    frames: list[FrameEvidence] = field(default_factory=list)
    started: float | None = None  # when the current evidence began (after a reset: then)

    def add(self, frame: FrameEvidence) -> None:
        """Add an embedded face (of this model) and keep the best ``keep``."""
        if frame.model and self.model and frame.model != self.model:
            raise ValueError(f"an embedding of {frame.model} for evidence of {self.model}: models cannot be mixed")
        if self.started is None:
            self.started = frame.time
        self.frames.append(frame)
        if len(self.frames) > self.keep:
            now = max(f.time for f in self.frames)
            self.frames.sort(key=lambda f: (-f.decayed(now, self.half_life), -f.time))
            del self.frames[self.keep:]
            self.frames.sort(key=lambda f: f.time)

    def reset(self, now: float, keep_after: float | None = None) -> None:
        """Forget the frames (those before ``keep_after`` only, when given): the track turned out to show someone
        else (see :mod:`.decision`)."""
        self.frames = [f for f in self.frames if keep_after is not None and f.time >= keep_after]
        self.started = min((f.time for f in self.frames), default=now)

    @property
    def count(self) -> int:
        return len(self.frames)

    def median_weight(self, after: int = 4) -> float | None:
        """The median weight of the frames so far, ``None`` with fewer than ``after``."""
        if len(self.frames) < after:
            return None
        return float(np.median([f.weight for f in self.frames]))

    def last_time(self) -> float | None:
        return max((f.time for f in self.frames), default=None)

    def aggregate(self, now: float | None = None, without: FrameEvidence | None = None) -> Aggregate | None:
        """The quality-weighted mean of the frames' unit embeddings at ``now`` (default: the last frame's time), made
        unit length again, or ``None`` without frames; ``without``: leave that frame out (to compare it with the
        rest)."""
        frames = [f for f in self.frames if f is not without]
        if not frames:
            return None
        now = max(f.time for f in frames) if now is None else float(now)
        weights = np.array([max(f.decayed(now, self.half_life), 1e-6) for f in frames])
        vectors = np.array([f.vector for f in frames], dtype=np.float64)
        mean = (weights[:, None] * vectors).sum(axis=0)
        norm = float(np.linalg.norm(mean))
        if norm < 1e-12:
            return None
        share = weights / weights.sum()
        return Aggregate(mean / norm, float(weights.sum()), float(share @ [f.size for f in frames]),
                         float(share @ [abs(f.yaw) for f in frames]), len(frames), self.model,
                         float(max(f.weight for f in frames)))
