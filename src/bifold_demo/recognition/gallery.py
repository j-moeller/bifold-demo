"""The people the recogniser knows: a folder of photos per person, their embeddings, and matching a face to them.

Layout of the faces folder (``faces/`` by default)::

    faces/
      Ada Lovelace/        a folder per person, named as they should be called;
        portrait.jpg       as many photos as you like (different days, light, angles: the more, the better)
        holiday.png
      Grace Hopper.jpg     or a single photo named after the person

The folder is looked at again at most once a second, so photos dropped in while the demo runs are known a moment
later. New photos are enrolled outside the gallery's lock: looking up who is known (:meth:`Gallery.people`,
:meth:`Gallery.scores`, ...) never waits for the models, and the new photos count from the moment they are all
enrolled (``refresh(wait=False)``, the render loop's call, enrols them on a worker thread). Each photo is embedded
with exactly the engine's detector, alignment and model, the ones the faces it will be compared with go through, and
checked (**enrolment**):

* **one dominant face**: a second face nearly as big (at least :data:`DOMINANT` of the biggest's size) makes the
  photo ambiguous, so it is rejected;
* **the eye distance**: rejected under :data:`MIN_EYES` pixels, a warning under :data:`GOOD_EYES` (the standards ask
  for 90 in reference photos);
* **blur**: rejected when the aligned face's sharpness is under :data:`BLURRED`, a warning under :data:`SOFT`
  (measured on the test photos: sharp ones 1.2-4.8; blurred by a Gaussian of 1 crop pixel 0.23-1.15, of 2 pixels
  0.04-0.26, of 3 pixels 0.02-0.09; avatar renders turned 60 degrees 0.38 and more);
* **how recognisable** (the profiles with eDifFIQA-T): a warning when its score is under :data:`HARD` (sharp photos
  0.61-0.83, blurred by 2 crop pixels 0.39-0.61, by 3 pixels 0.16-0.47), except for profile views, which score lower
  anyway (renders turned 60 degrees: 0.37-0.59); a warning only, as such scores can differ between groups of people;
* **light**: a warning when the face is very dark, very bright or clipped;
* **duplicates**: a photo whose embedding is within :data:`DUPLICATE` (cosine, InsightFace's default) of a better
  photo of the same person adds nothing and is left out (it stays in the folder);
* **the pose bin** of each photo (frontal, left, right, profile; see :mod:`.quality`);
* **near-twins**: two people whose photos reach the threshold against each other get a warning: the model finds them
  alike, and deciding between them needs a margin.

A person is matched by every photo of theirs and by a **template per pose bin** (the quality-weighted mean of the
bin's embeddings, made unit length again): :meth:`Gallery.scores` gives each person's score, the best of their
templates and photos. :meth:`Gallery.match` turns the scores into a name with :attr:`Gallery.decide`, a callable
``decide(scores, quality)`` (the face's :class:`~.quality.Quality` when the caller has it, else ``None``), by default
:class:`ThresholdDecision` (the best person, if their score reaches the threshold). The open-set decision with
calibration per model and face size (step F2) plugs in there.

**Caches.** Embeddings of different models cannot be compared, so they are cached per model in
``faces/.embeddings.<model id>.npz``, with the execution provider that made them (CoreML's numbers differ slightly
from the CPU's: a cache made by another provider is not used); only new or changed photos are looked at again. The
embeddings are biometric data like the photos: when a photo leaves the folder, or is replaced by another under the
same name, its entry leaves every model's cache.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ..scene.textures import IMAGE_EXTENSIONS

THRESHOLD = 0.363  # cosine similarity: OpenCV's recommended cut for SFace (99.6 % on LFW); other models: their own
CACHE = ".embeddings.npz"  # the cache of the demo before the per-model caches (removed when a new one is written)
CACHE_VERSION = 2  # raised when what is cached changes (the alignment, the checks): older caches are not used
RESCAN = 1.0  # seconds between looks at the folder
DOMINANT = 0.7  # a second face this big (relative to the biggest) makes a photo ambiguous
MIN_EYES, GOOD_EYES = 40.0, 90.0  # pixels between the eyes: rejected under, warned under
BLURRED, SOFT = 0.10, 0.30  # the aligned face's sharpness (see quality.image_measures): rejected under, warned under
HARD = 0.45  # eDifFIQA-T's score: a warning under it (not for profile views; see the module docs)
DARK, BRIGHT, CLIPPED = 50.0, 215.0, 0.15  # mean grey level of the face, share of clipped pixels: warned beyond
DUPLICATE = 0.98  # cosine: a photo this close to a better one of the same person adds nothing


def cache_name(model_id: str) -> str:
    """The cache file of ``model_id``'s embeddings in the faces folder."""
    return f".embeddings.{model_id}.npz"


@dataclass(frozen=True)
class Match:
    """Who a face is: ``name`` (``None``: nobody known), the ``score`` of the closest photo of them and the score of
    the closest other person (``runner_up``)."""

    name: str | None
    score: float
    runner_up: float = -1.0


@dataclass(frozen=True)
class PersonScore:
    """How much a face looks like a known person: ``score``, the best of the person's ``template`` score (the best
    pose bin's) and ``photo`` score (the best single photo's; -1 where there is none), and the ``pose`` bin of the
    template or photo that scored best."""

    name: str
    score: float
    template: float
    photo: float
    pose: str


class ThresholdDecision:
    """The default decision: the best person if their score reaches ``gallery.threshold``, else nobody."""

    def __init__(self, gallery):
        self.gallery = gallery

    def __call__(self, scores: list[PersonScore], quality=None) -> Match:
        if not scores:
            return Match(None, -1.0)
        best = scores[0]
        runner_up = scores[1].score if len(scores) > 1 else -1.0
        return Match(best.name if best.score >= self.gallery.threshold else None, best.score, runner_up)


@dataclass
class Photo:
    """A photo of a known person and what enrolment found in it (see the module docs)."""

    path: Path
    name: str
    stamp: tuple[int, int]  # (size, mtime_ns): looked at again when it changes
    vector: np.ndarray | None  # the unit embedding of its face (None: no usable face)
    face_size: float = 0.0
    status: str = "ok"  # "ok" or "rejected"
    problems: tuple[str, ...] = ()  # why it was rejected, or warnings
    faces: int = 0  # how many faces the detector found
    norm: float = math.nan  # the embedding's raw length
    ied: float = math.nan  # quality (nan: not measured, e.g. with a stand-in engine)
    yaw: float = math.nan
    pitch: float = math.nan
    sharpness: float = math.nan
    fiqa: float = math.nan
    weight: float = 1.0  # how much it counts in its pose bin's template
    pose: str = ""
    duplicate_of: Path | None = field(default=None, compare=False)  # set when the gallery is rebuilt

    @property
    def usable(self) -> bool:
        return self.vector is not None and self.status == "ok"


_NUMBERS = ("face_size", "norm", "ied", "yaw", "pitch", "sharpness", "fiqa", "weight")


def _person(path: Path, folder: Path) -> str | None:
    relative = path.relative_to(folder)
    if len(relative.parts) == 1:
        return path.stem.replace("_", " ").strip() or None
    if len(relative.parts) == 2:
        return relative.parts[0].strip() or None
    return None


def _photos(folder: Path) -> list[Path]:
    found = []
    for root, dirs, files in os.walk(folder):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for name in sorted(files):
            path = Path(root) / name
            if not name.startswith(".") and path.suffix.lower() in IMAGE_EXTENSIONS:
                found.append(path)
    return found


def has_photos(folder) -> bool:
    """Whether ``folder`` has any photos of people (cheap: nothing is loaded)."""
    folder = Path(folder).expanduser()
    return folder.is_dir() and any(_person(p, folder) is not None for p in _photos(folder))


def enrol(engine, image: np.ndarray) -> dict:
    """What enrolment finds in ``image`` (BGR) with ``engine``: the fields of a :class:`Photo` (see the module docs).
    An engine without ``analyse`` (a stand-in) gives no quality: only the face checks are made."""
    if image is None:
        return {"vector": None, "status": "rejected", "problems": ("not an image",)}
    faces = engine.detect(image, min_size=0)
    if not faces:
        return {"vector": None, "status": "rejected", "problems": ("no face found",)}
    face = faces[0]  # the biggest
    out = {"faces": len(faces), "face_size": float(face.size)}
    if len(faces) > 1 and faces[1].size >= DOMINANT * face.size:
        return {**out, "vector": None, "status": "rejected",
                "problems": (f"{len(faces)} faces of about the same size: which one is the person?",)}
    analyse = getattr(engine, "analyse", None)
    if analyse is None:
        return {**out, "vector": np.asarray(engine.embed(image, face), np.float64)}
    record = analyse(image, [face])[0]
    q = record.quality
    out.update(vector=np.asarray(record.vector, np.float64), norm=record.norm, ied=q.ied, yaw=q.yaw, pitch=q.pitch,
               sharpness=q.sharpness, fiqa=math.nan if q.fiqa is None else q.fiqa, weight=q.weight, pose=q.pose)
    rejected, warnings = [], []
    if q.ied < MIN_EYES:
        rejected.append(f"the face is too small: {q.ied:.0f} pixels between the eyes (at least {MIN_EYES:.0f})")
    elif q.ied < GOOD_EYES:
        warnings.append(f"a small face: {q.ied:.0f} pixels between the eyes ({GOOD_EYES:.0f} or more is best)")
    if q.sharpness < BLURRED:
        rejected.append(f"the face is blurred (sharpness {q.sharpness:.2f}, at least {BLURRED}; "
                        "sharp photos 1 or more)")
    elif q.sharpness < SOFT:
        warnings.append(f"the face is a little blurred (sharpness {q.sharpness:.2f}; sharp photos 1 or more)")
    if q.fiqa is not None and q.fiqa < HARD and q.pose != "profile":
        warnings.append(f"the model finds the face hard to recognise (eDifFIQA {q.fiqa:.2f}; good photos 0.6-0.8): "
                        "blurred, badly lit or partly covered?")
    if q.brightness < DARK or q.brightness > BRIGHT or q.clipped > CLIPPED:
        warnings.append(f"the face is badly lit (brightness {q.brightness:.0f} of 255, {q.clipped:.0%} clipped)")
    if q.pose == "profile":
        warnings.append(f"a profile view ({q.yaw:+.0f} degrees): it helps with turned faces, add frontal photos too")
    if rejected:
        return {**out, "status": "rejected", "problems": tuple(rejected + warnings)}
    return {**out, "problems": tuple(warnings)}


class Gallery:
    """Known people from ``folder`` (see the module docs), embedded with ``engine`` (a :class:`~.engine.FaceEngine`
    of ``profile``, made when first needed). ``threshold``: the score a face needs to be someone (default: the
    engine's model's own); ``decide``: what turns the scores into a :class:`Match` (default:
    :class:`ThresholdDecision`). Thread safe. ``trouble``: why the last enrolment on a worker thread failed (``None``
    when it did not)."""

    def __init__(self, folder="faces", engine=None, threshold: float | None = None, profile: str | None = None,
                 decide=None):
        self.folder = Path(folder).expanduser()
        self._threshold = None if threshold is None else float(threshold)
        self.profile = profile
        self._engine = engine
        self.decide = decide or ThresholdDecision(self)
        self._photos: dict[Path, Photo] = {}
        self._checked = -math.inf
        self._lock = threading.RLock()  # the state below; held only briefly (never while the models run)
        self._busy = threading.Lock()  # one look at the folder and its enrolment at a time
        self.trouble: str | None = None
        self._names: list[str] = []
        self._matrix = np.zeros((0, 0))
        self._owners = np.zeros(0, int)
        self._poses: list[str] = []
        self._templates = np.zeros((0, 0))
        self._template_owners = np.zeros(0, int)
        self._template_poses: list[str] = []
        self._twins: dict[str, list[tuple[str, float]]] = {}
        self._loaded = False

    @property
    def engine(self):
        if self._engine is None:
            with self._lock:  # (made once, whichever thread asks first)
                if self._engine is None:
                    from .engine import FaceEngine

                    self._engine = FaceEngine(profile=self.profile)
        return self._engine

    @property
    def threshold(self) -> float:
        """The score a face needs to be someone: as given, else the engine's model's own (:data:`THRESHOLD` for a
        stand-in engine without one)."""
        if self._threshold is not None:
            return self._threshold
        return float(getattr(self.engine, "threshold", THRESHOLD))

    @threshold.setter
    def threshold(self, value: float | None) -> None:
        self._threshold = None if value is None else float(value)

    @property
    def model_id(self) -> str:
        return str(getattr(self.engine, "model_id", "unknown"))

    def _provider(self) -> str:
        return str(getattr(self.engine, "provider_name", ""))

    # -- the folder -------------------------------------------------------------------------------------------

    def refresh(self, now: float | None = None, force: bool = False, wait: bool = True) -> bool:
        """Look at the folder again (at most every ``RESCAN`` seconds unless ``force``); True if anything changed.

        New and changed photos are enrolled outside the lock, then swapped in at once (see the module docs).
        ``wait=False`` returns at once: photos to enrol go to a worker thread (the result: False, the change shows
        when they are enrolled), and while photos are being enrolled the call does nothing."""
        now = time.monotonic() if now is None else now
        with self._lock:
            if not force and now - self._checked < RESCAN:
                return False
            self._checked = now
        if not self._busy.acquire(blocking=wait):
            return False  # photos are being enrolled: the next look sees what changed since
        try:
            plan = self._plan()
        except BaseException:
            self._busy.release()
            raise
        if plan[-1] and not wait:
            threading.Thread(target=self._enrol_new, args=(*plan, True), name="faces enrolment", daemon=True).start()
            return False
        return self._enrol_new(*plan)

    def _plan(self):
        """``(first look, the photos known before, the folder's photos in order, those unchanged, those to enrol)``
        (called holding ``_busy``: nothing else changes the photos meanwhile)."""
        first = not self._loaded
        known = self._read_cache() if first else self._photos
        order: list[Path] = []
        keep: dict[Path, Photo] = {}
        todo: list[tuple[Path, str, tuple[int, int]]] = []
        for path in _photos(self.folder) if self.folder.is_dir() else []:
            name = _person(path, self.folder)
            if name is None:
                continue
            try:
                st = path.stat()
            except OSError:
                continue
            stamp = (st.st_size, st.st_mtime_ns)
            order.append(path)
            old = known.get(path)
            if old is not None and old.stamp == stamp:
                keep[path] = old if old.name == name else dataclasses.replace(old, name=name)
            else:
                todo.append((path, name, stamp))
        return first, known, order, keep, todo

    def _enrol_new(self, first, known, order, keep, todo, quietly: bool = False) -> bool:
        """Enrol ``todo`` (outside the lock), then swap the folder's photos in; releases ``_busy``. ``quietly`` (on
        the worker thread): a failure goes to :attr:`trouble`, the photos are tried again at the next look."""
        try:
            for path, name, stamp in todo:
                keep[path] = self._embed(path, name, stamp)
            photos = {path: keep[path] for path in order}
            changed = first or bool(todo) or set(photos) != set(known)
            with self._lock:
                self._photos = photos
                self._loaded = True
                if changed:
                    self._rebuild()
            if changed:
                self._save_cache()
            self.trouble = None
            return changed
        except Exception as exc:
            if not quietly:
                raise
            self.trouble = f"cannot enrol the new photos in {self.folder}: {type(exc).__name__}: {exc}"
            return False
        finally:
            self._busy.release()

    def _embed(self, path: Path, name: str, stamp) -> Photo:
        found = enrol(self.engine, cv2.imread(str(path), cv2.IMREAD_COLOR))
        return Photo(path, name, stamp, **found)

    def _rebuild(self) -> None:
        good = [p for p in self._photos.values() if p.usable]
        # duplicates: within a person, a photo nearly the same as a better one is left out
        for p in self._photos.values():
            p.duplicate_of = None
        by_person: dict[str, list[Photo]] = {}
        for p in good:
            by_person.setdefault(p.name, []).append(p)
        for photos in by_person.values():
            photos.sort(key=lambda p: (-p.weight, -p.face_size, str(p.path)))
            kept: list[Photo] = []
            for p in photos:
                twin = next((k for k in kept if float(k.vector @ p.vector) > DUPLICATE), None)
                if twin is None:
                    kept.append(p)
                else:
                    p.duplicate_of = twin.path
        good = [p for p in good if p.duplicate_of is None]
        self._names = sorted({p.name for p in good})
        index = {n: k for k, n in enumerate(self._names)}
        dim = len(good[0].vector) if good else 0
        self._matrix = np.array([p.vector for p in good]).reshape(len(good), dim)
        self._owners = np.array([index[p.name] for p in good], int)
        self._poses = [p.pose for p in good]
        templates, owners, poses = [], [], []
        for name in self._names:
            mine = [p for p in good if p.name == name]
            for pose in sorted({p.pose for p in mine}):
                members = [p for p in mine if p.pose == pose]
                weights = np.array([max(p.weight, 1e-3) for p in members])
                mean = (weights[:, None] * np.array([p.vector for p in members])).sum(axis=0)
                templates.append(mean / max(np.linalg.norm(mean), 1e-12))
                owners.append(index[name])
                poses.append(pose)
        self._templates = np.array(templates).reshape(len(templates), dim)
        self._template_owners = np.array(owners, int)
        self._template_poses = poses
        self._twins = self._near_twins()

    def _near_twins(self) -> dict[str, list[tuple[str, float]]]:
        """``{name: [(other, score)]}`` for people whose photos score at least the threshold against each other."""
        twins: dict[str, list[tuple[str, float]]] = {}
        if len(self._names) < 2:
            return twins
        scores = self._matrix @ self._matrix.T
        n = len(self._names)
        best = np.full((n, n), -1.0)
        for i, a in enumerate(self._owners):
            np.maximum.at(best[a], self._owners, scores[i])
        threshold = self.threshold
        for a in range(n):
            for b in range(n):
                if a != b and best[a, b] >= threshold:
                    twins.setdefault(self._names[a], []).append((self._names[b], float(best[a, b])))
        return twins

    # -- caches -----------------------------------------------------------------------------------------------

    def _cache_path(self) -> Path:
        return self.folder / cache_name(self.model_id)

    def _read_cache(self) -> dict[Path, Photo]:
        """The photos this model's cache knows (none when it was made differently: all are looked at again)."""
        photos: dict[Path, Photo] = {}
        try:
            data = np.load(self._cache_path(), allow_pickle=False)
            if int(data["version"]) != CACHE_VERSION or str(data["provider"]) != self._provider():
                return photos  # made differently: everything is looked at again
            rows = {key: data[key] for key in data.files}
        except (OSError, KeyError, ValueError):
            return photos
        for k, rel in enumerate(rows["paths"]):
            path = self.folder / str(rel)
            name = _person(path, self.folder) if path.is_relative_to(self.folder) else None
            if name is None:
                continue
            vector = rows["vectors"][k].astype(np.float64)
            numbers = {key: float(rows[key][k]) for key in _NUMBERS}
            photos[path] = Photo(
                path, name, (int(rows["stamps"][k][0]), int(rows["stamps"][k][1])),
                vector if np.isfinite(vector).all() else None, status=str(rows["status"][k]),
                problems=tuple(json.loads(str(rows["problems"][k]))), faces=int(rows["faces"][k]),
                pose=str(rows["poses"][k]), **numbers)
        return photos

    def _save_cache(self) -> None:
        if not self.folder.is_dir():
            return
        photos = list(self._photos.values())
        dim = next((len(p.vector) for p in photos if p.vector is not None), 1)
        rows = {
            "paths": np.array([str(p.path.relative_to(self.folder)) for p in photos], str),
            "stamps": np.array([p.stamp for p in photos], np.int64).reshape(-1, 2),
            "vectors": np.array([p.vector if p.vector is not None else np.full(dim, np.nan) for p in photos],
                                np.float64).reshape(-1, dim),
            "status": np.array([p.status for p in photos], str),
            "problems": np.array([json.dumps(list(p.problems)) for p in photos], str),
            "faces": np.array([p.faces for p in photos], np.int64),
            "poses": np.array([p.pose for p in photos], str),
            **{key: np.array([getattr(p, key) for p in photos], np.float64) for key in _NUMBERS},
            "model": np.array(self.model_id), "provider": np.array(self._provider()),
            "version": np.array(CACHE_VERSION),
        }
        _write(self._cache_path(), rows)
        self._forget_elsewhere({str(p.path.relative_to(self.folder)): p.stamp for p in photos})

    def _forget_elsewhere(self, kept: dict[str, tuple[int, int]]) -> None:
        """Remove from the other models' caches (and the old shared cache) the photos no longer in the folder and
        those changed since they were cached there (a photo replaced under the same name): ``kept`` maps the
        folder's photos to their stamps."""
        (self.folder / CACHE).unlink(missing_ok=True)
        mine = self._cache_path().name
        for other in self.folder.glob(".embeddings.*.npz"):
            if other.name == mine:
                continue
            try:
                with np.load(other, allow_pickle=False) as data:
                    rows = {key: data[key] for key in data.files}
                stamps = rows["stamps"].reshape(-1, 2)
                keep = np.array([kept.get(str(p)) == (int(a), int(b)) for p, (a, b) in zip(rows["paths"], stamps)],
                                bool)
                if keep.all():
                    continue
                for key, value in rows.items():
                    if value.ndim and len(value) == len(keep):
                        rows[key] = value[keep]
                _write(other, rows)
            except (OSError, KeyError, ValueError):
                other.unlink(missing_ok=True)  # unreadable: it goes (it is only a cache)

    # -- who is it --------------------------------------------------------------------------------------------

    def people(self) -> dict[str, int]:
        """``{name: number of usable photos}`` (duplicates not counted)."""
        with self._lock:
            counts: dict[str, int] = {}
            for p in self._photos.values():
                if p.usable and p.duplicate_of is None:
                    counts[p.name] = counts.get(p.name, 0) + 1
            return dict(sorted(counts.items()))

    def problems(self) -> list[Path]:
        """Photos that cannot be used (no face, two faces, too small, blurred; see :meth:`report`)."""
        with self._lock:
            return sorted(p.path for p in self._photos.values() if not p.usable)

    def report(self) -> list[tuple[Path, str, tuple[str, ...]]]:
        """``(photo, status, problems)`` for every photo with something to say: ``rejected``, ``duplicate`` (of
        another photo) or ``ok`` with warnings, and ``(name, "near-twin", ...)`` lines per person who looks like
        another to the model."""
        with self._lock:
            out = []
            for p in sorted(self._photos.values(), key=lambda p: str(p.path)):
                if p.duplicate_of is not None:
                    out.append((p.path, "duplicate", (f"nearly the same as {p.duplicate_of.name}",)))
                elif p.status != "ok" or p.problems:
                    out.append((p.path, p.status, p.problems))
            for name, others in sorted(self._twins.items()):
                lines = tuple(f"looks like {other} to the model ({score:.2f})" for other, score in others)
                out.append((Path(name), "near-twin", lines))
            return out

    def near_twins(self) -> dict[str, list[tuple[str, float]]]:
        """``{name: [(other person, best score between their photos)]}`` (see the module docs)."""
        with self._lock:
            return {name: list(others) for name, others in self._twins.items()}

    def photos(self, name: str) -> list[Path]:
        """The usable photos of ``name``, the one with the biggest face first."""
        with self._lock:
            mine = [p for p in self._photos.values() if p.name == name and p.usable and p.duplicate_of is None]
            return [p.path for p in sorted(mine, key=lambda p: -p.face_size)]

    def poses(self, name: str) -> dict[str, int]:
        """``{pose bin: number of photos}`` of ``name`` (the bins its templates cover)."""
        with self._lock:
            counts: dict[str, int] = {}
            for p in self._photos.values():
                if p.name == name and p.usable and p.duplicate_of is None:
                    counts[p.pose or "unknown"] = counts.get(p.pose or "unknown", 0) + 1
            return counts

    def scores(self, vector: np.ndarray) -> list[PersonScore]:
        """Every known person's score for the face with embedding ``vector``, best first (see the module docs)."""
        with self._lock:
            names, matrix, owners, poses = self._names, self._matrix, self._owners, self._poses
            templates, t_owners, t_poses = self._templates, self._template_owners, self._template_poses
        if not names:
            return []
        vector = np.asarray(vector, np.float64)
        if matrix.shape[1] != len(vector):
            raise ValueError(f"an embedding of length {len(vector)} for a gallery of {matrix.shape[1]}: another model?")
        photo = matrix @ vector
        template = templates @ vector if len(templates) else np.zeros(0)
        out = []
        for k, name in enumerate(names):
            mine, theirs = photo[owners == k], template[t_owners == k] if len(template) else np.zeros(0)
            best_photo = float(mine.max()) if len(mine) else -1.0
            best_template = float(theirs.max()) if len(theirs) else -1.0
            if best_template >= best_photo:
                pose = t_poses[int(np.flatnonzero(t_owners == k)[int(theirs.argmax())])]
            else:
                pose = poses[int(np.flatnonzero(owners == k)[int(mine.argmax())])]
            out.append(PersonScore(name, max(best_photo, best_template), best_template, best_photo, pose))
        out.sort(key=lambda s: -s.score)
        return out

    def match(self, vector: np.ndarray, quality=None) -> Match:
        """Who the face with embedding ``vector`` (and ``quality``, when known) is, decided by :attr:`decide`."""
        return self.decide(self.scores(vector), quality)

    def score(self, vector: np.ndarray, name: str) -> float:
        """How much the face with embedding ``vector`` looks like ``name`` (their score; -1: not known)."""
        return next((s.score for s in self.scores(vector) if s.name == name), -1.0)

    # -- adding people ----------------------------------------------------------------------------------------

    def add(self, name: str, photos) -> list[str]:
        """Copy ``photos`` of ``name`` into their folder (``faces/<name>/``); returns a problem per photo enrolment
        rejects (those are not copied). Warnings about the photos copied: :meth:`report`."""
        name = name.strip()
        if not name or any(c in name for c in "/\\") or name.startswith("."):
            raise ValueError(f"not a usable name for a person: {name!r}")
        target = self.folder / name
        problems = []
        for photo in photos:
            photo = Path(photo).expanduser()
            found = enrol(self.engine, cv2.imread(str(photo), cv2.IMREAD_COLOR))
            if found.get("status") == "rejected":
                problems.append(f"{photo}: {'; '.join(found['problems'])}")
                continue
            target.mkdir(parents=True, exist_ok=True)
            dest = target / photo.name
            k = 2
            while dest.exists():
                dest = target / f"{photo.stem}_{k}{photo.suffix}"
                k += 1
            shutil.copy2(photo, dest)
        self.refresh(force=True)
        return problems


def _write(path: Path, rows: dict) -> None:
    """Write ``rows`` to the npz ``path`` atomically (a temporary file first)."""
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp.npz")
    try:
        np.savez(tmp, **rows)
        os.replace(tmp, path)
    except OSError:
        tmp.unlink(missing_ok=True)
