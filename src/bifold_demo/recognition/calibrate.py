"""``bifold-faces calibrate``: each model's calibration (:mod:`.calibration`), made from renders of avatars.

Calibrations must be made from data that may be shipped: renders of the Rocketbox avatars (MIT). Photos of real
people and LFW may check a calibration, never make one. The data:

* **Who.** The avatars of a character library (``--characters``; by default every prepared avatar). Each avatar is
  one identity: mated pairs are the same avatar, non-mated pairs different avatars. The shipped calibrations come
  from 38 avatars that are *not* in the fair's library, so that the fair can check them (its people are strangers to
  the calibration).
* **Their gallery.** Each avatar's "photos": windows of the virtual sensor on its head in the fair's hall, 2 m from
  the camera, 100 pixels between the eyes, the head turned -25, 0 and 25 degrees, enrolled by the
  :class:`~.gallery.Gallery` as photos are (the same checks, templates per pose).
* **Their faces as the live pipeline sees them.** Windows (:class:`~bifold_demo.sensor.WindowRequest` with
  ``eye_px``: an exact eye distance) at swept face sizes (:data:`EYE_PX`, 8 to 64 pixels between the eyes), the head
  turned (:data:`YAWS`), through camera presets (:data:`PRESETS`: an ideal camera and a budget 1080p one with its
  blur, noise and JPEG), at a few places in the hall (its light varies), found by the profile's face detector as the
  live pipeline finds faces in windows, aligned and embedded. Each face is binned by what the pipeline measures: the
  face's size from its box and its yaw from its landmarks.
* **Evidence of several faces.** The decisions score a track's averaged faces, so the calibration also scores
  averages of two and three faces of one avatar at one size (neighbouring yaws), weighted by quality as the tracks
  are (:data:`AGGREGATES`).

Each face (and average) is scored against every avatar's gallery (a person's score: the best of their photos and pose
templates); the scores give each bin's non-mated and mated curves (:meth:`.calibration.Calibration.fit`).

**Normalisation and the check.** The avatars are split in halves, many times over: one half calibrates; in the other,
half the avatars are known (their gallery) and half strangers. On those held-out searches the false-alarm rate asked
for (0.1 %, 1 %, 10 % per decision) is compared with the one measured, and the share of known faces named correctly
at it. Done without normalisation and with T-norm and AS-norm against a cohort of the calibrating half's faces (up to
:data:`COHORT_KEEP` single faces of each avatar, each face's own avatar and the gallery person left out); a
normalisation is kept only when it names more at the same false-alarm rate (``--normalisation auto``). The final
calibration uses every avatar; with a normalisation, its cohort is kept beside it (``<model id>.cohort.npz``), the
very faces its scores were normalised with.

The renders, the aligned faces and their embeddings are kept in ``--data`` (default ``<models>/calibration/data``),
so that a calibration can be made again without rendering (``--refit``).
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .calibration import Calibration, Cohort, bin_index, calibration_path, cohort_path, per_person

EYE_PX = (8.0, 11.0, 14.0, 18.0, 24.0, 32.0, 44.0, 64.0)  # pixels between the eyes asked of the windows
YAWS = (-40.0, -20.0, 0.0, 20.0, 40.0)  # degrees the head is turned (to its own left: positive)
PRESETS = ("ideal", "budget 1080p")
DISTANCES = (3.5, 5.0, 6.5)  # metres from the camera; each face is at one of them (the light varies)
GALLERY_YAWS = (-25.0, 0.0, 25.0)
GALLERY_EYE_PX = 100.0
AGGREGATES = ((1, 2), (2, 3), (0, 1), (3, 4), (1, 2, 3))  # indices into YAWS of faces averaged together
TARGETS = (0.001, 0.01, 0.1)  # false-alarm rates per decision checked
SPLITS = 10  # random halvings of the avatars for the check
KINDS = ("none", "tnorm", "asnorm")
COHORT_KEEP = 16  # faces per avatar kept in a shipped cohort

__all__ = ["AGGREGATES", "DISTANCES", "EYE_PX", "PRESETS", "YAWS", "Probes", "check", "fit", "render", "scores"]


@dataclass
class Probes:
    """The faces of the avatars as one face detector found them in the windows: per window the ``avatar``, the
    ``eye_px`` asked, the ``yaw`` asked, the ``preset``, the ``distance``; ``found``: whether the face was found;
    ``crops``: the aligned faces (``(n, 112, 112, 3)``, black where none was found); ``sizes``: the face's eye distance
    turned to the camera (from its box, window pixels); ``yaws``: the yaw its landmarks give; ``weights``: its
    quality weight without eDifFIQA; ``detector``: the detector's id."""

    detector: str
    avatar: np.ndarray
    eye_px: np.ndarray
    yaw: np.ndarray
    preset: np.ndarray
    distance: np.ndarray
    found: np.ndarray
    crops: np.ndarray = field(repr=False)
    sizes: np.ndarray = field(repr=False)
    yaws: np.ndarray = field(repr=False)
    sharpness: np.ndarray = field(repr=False)
    outside: np.ndarray = field(repr=False)
    ied: np.ndarray = field(repr=False)
    score: np.ndarray = field(repr=False)

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **{k: getattr(self, k) for k in self.__dataclass_fields__})
        return path

    @classmethod
    def load(cls, path) -> Probes:
        with np.load(path, allow_pickle=False) as data:
            values = {k: data[k] for k in data.files}
        values["detector"] = str(values["detector"])
        return cls(**values)

    def weights(self, fiqa=None) -> np.ndarray:
        """The quality weights (:attr:`~.quality.Quality.weight`), with eDifFIQA-T's scores when given."""
        from .quality import Quality

        out = np.zeros(len(self.found))
        for k in np.flatnonzero(self.found):
            q = Quality(float(self.ied[k]), float(self.yaws[k]), 0.0, 0.0, float(self.sharpness[k]), 128.0, 50.0,
                        0.0, float(self.outside[k]), float(self.score[k]),
                        None if fiqa is None else float(fiqa[k]))
            out[k] = q.weight
        return out


# -- rendering -----------------------------------------------------------------------------------------------------


def render(library, names, detectors: dict, gallery_folder, eye_px=EYE_PX, yaws=YAWS, presets=PRESETS,
           distances=DISTANCES, progress=print, seed: int = 0) -> dict:
    """Render ``names`` (avatars of ``library``) in the fair's hall: their gallery photos (written as PNG files into
    ``gallery_folder/<avatar>/``) and their faces through the windows, found by each of ``detectors`` (``{id:
    detector}``). Returns ``{detector id: Probes}``. Needs OpenGL."""
    import cv2

    from ..characters.character import Character
    from ..characters.model import CharacterState
    from ..scene.renderer import Renderer
    from ..scenes import fair
    from ..sensor.geometry import Sensor
    from ..sensor.requests import WindowRequest
    from ..sensor.service import eye_points, render_windows
    from .align import align
    from .heads import WINDOW_CANVAS, face_size
    from .quality import assess

    gallery_folder = Path(gallery_folder)
    rng = np.random.default_rng(seed)
    scene = fair.build_scene(people=[])  # the hall, its light and haze, nobody in it
    camera = scene.camera
    position = np.asarray(camera.current_position, dtype=np.float64)
    forward = np.asarray(camera.current_target, dtype=np.float64) - position
    forward[1] = 0.0
    forward /= np.linalg.norm(forward)
    across = np.array([-forward[2], 0.0, forward[0]])
    heading = math.degrees(math.atan2(-forward[0], -forward[2]))  # facing the camera
    view = (1280, 720)
    renderer = Renderer(*view)
    renderer.detail_wait = True
    sensor = Sensor.scaled(6, view)
    rows = {key: {k: [] for k in ("avatar", "eye_px", "yaw", "preset", "distance", "found", "crops", "sizes",
                                  "yaws", "sharpness", "outside", "ied", "score")} for key in detectors}
    started = time.perf_counter()
    try:
        for n, name in enumerate(names):
            person = Character(library.load(name), name=name, shadow=False)
            scene.add(person)
            poses = [(2.0, 0.0, yaw, "gallery") for yaw in GALLERY_YAWS]
            poses += [(float(rng.choice(distances)), float(rng.uniform(-0.8, 0.8)), yaw, "probe") for yaw in yaws]
            for distance, side, yaw, kind in poses:
                spot = position + forward * distance + across * side
                spot[1] = 0.0
                person.behaviour = (lambda _t, s=tuple(spot), y=yaw: CharacterState(s, heading, [], head_yaw=y))
                scene.update(0.0, 0.0)
                renderer.render(scene)  # (the windows reuse what the frame uploaded)
                head = tuple(eye_points(person).mean(axis=0))
                if kind == "gallery":
                    request = WindowRequest((name, "gallery", yaw), head=head, head_size=0.45, subject=person,
                                            eye_px=GALLERY_EYE_PX)
                    [window] = render_windows(scene, renderer, [request], sensor, truth=False)
                    folder = gallery_folder / name
                    folder.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(folder / f"yaw{yaw:+.0f}.png"), window.image)
                    continue
                requests = [WindowRequest((name, px, preset), head=head, head_size=0.5, subject=person, eye_px=px,
                                          preset=preset) for px in eye_px for preset in presets]
                for window in render_windows(scene, renderer, requests, sensor, truth=False):
                    _, px, preset = window.tag
                    image = window.image
                    side_px = max(image.shape[:2])
                    for key, detector in detectors.items():
                        faces = detector.detect(image, 0.0, WINDOW_CANVAS if side_px > WINDOW_CANVAS else None)
                        middle = np.array(image.shape[1::-1]) / 2.0
                        face = min(faces, key=lambda f: np.linalg.norm(f.points[:2].mean(axis=0) - middle)) \
                            if faces else None
                        r = rows[key]
                        r["avatar"].append(name)
                        r["eye_px"].append(px)
                        r["yaw"].append(yaw)
                        r["preset"].append(preset)
                        r["distance"].append(distance)
                        r["found"].append(face is not None)
                        if face is None:
                            r["crops"].append(np.zeros((112, 112, 3), np.uint8))
                            for k in ("sizes", "yaws", "sharpness", "outside", "ied", "score"):
                                r[k].append(0.0)
                            continue
                        aligned = align(image, face.landmarks)
                        q = assess(face, aligned)
                        r["crops"].append(aligned.crop)
                        r["sizes"].append(face_size(face))
                        r["yaws"].append(q.yaw)
                        r["sharpness"].append(q.sharpness)
                        r["outside"].append(q.outside)
                        r["ied"].append(q.ied)
                        r["score"].append(face.score)
            scene.remove(person)
            if progress:
                progress(f"[calibrate] {n + 1}/{len(names)} {name} ({time.perf_counter() - started:.0f} s)")
    finally:
        renderer.release()
    out = {}
    for key, r in rows.items():
        out[key] = Probes(key, np.array(r["avatar"], str), np.array(r["eye_px"]), np.array(r["yaw"]),
                          np.array(r["preset"], str), np.array(r["distance"]), np.array(r["found"], bool),
                          np.array(r["crops"], np.uint8).reshape(-1, 112, 112, 3), np.array(r["sizes"]),
                          np.array(r["yaws"]), np.array(r["sharpness"]), np.array(r["outside"]), np.array(r["ied"]),
                          np.array(r["score"]))
    return out


def gallery_photos(library, names, folder, progress=None) -> list[Path]:
    """Each of ``names``' gallery photos, rendered as :func:`render` renders them (2 m from the camera, 100 pixels
    between the eyes, turned -25, 0 and 25 degrees) into ``folder/<avatar>/``: a gallery of avatars, which may be
    shipped and shown (MIT), for the fair's known people. Avatars whose photos are there already are left as they
    are. Needs OpenGL."""
    folder = Path(folder)
    todo = [n for n in names if not (folder / n).is_dir() or len(list((folder / n).glob("*.png"))) < len(GALLERY_YAWS)]
    if todo:
        render(library, todo, {}, folder, yaws=(), progress=progress)
    return sorted(p for n in names for p in (folder / n).glob("*.png"))


# -- scores --------------------------------------------------------------------------------------------------------


@dataclass
class Faces:
    """Faces (single and averaged) of one model: unit ``vectors``, who they show (``people``), their ``sizes`` and
    ``yaws`` (absolute) for the bins, and whether each is a ``single`` face (the cohort's)."""

    vectors: np.ndarray
    people: np.ndarray
    sizes: np.ndarray
    yaws: np.ndarray
    single: np.ndarray
    eye_px: np.ndarray


def faces(probes: Probes, vectors: np.ndarray, weights: np.ndarray) -> Faces:
    """The single faces found, and averages of two and three faces of one avatar at one size, preset and place
    (:data:`AGGREGATES`, neighbouring yaws), weighted by quality as a track's evidence is."""
    found = probes.found & (weights > 0)
    out_v, out_p, out_s, out_y, out_single, out_px = [], [], [], [], [], []
    for k in np.flatnonzero(found):
        out_v.append(vectors[k])
        out_p.append(probes.avatar[k])
        out_s.append(probes.sizes[k])
        out_y.append(abs(probes.yaws[k]))
        out_single.append(True)
        out_px.append(probes.eye_px[k])
    groups: dict = {}
    for k in np.flatnonzero(found):
        groups.setdefault((probes.avatar[k], probes.eye_px[k], probes.preset[k]), {})[float(probes.yaw[k])] = k
    for (avatar, px, _preset), by_yaw in groups.items():
        for combo in AGGREGATES:
            members = [by_yaw.get(YAWS[i]) for i in combo if i < len(YAWS)]
            if len(members) != len(combo) or any(m is None for m in members):
                continue
            w = np.array([weights[m] for m in members])
            mean = (w[:, None] * vectors[members]).sum(axis=0)
            mean /= max(np.linalg.norm(mean), 1e-12)
            share = w / w.sum()
            out_v.append(mean)
            out_p.append(avatar)
            out_s.append(float(share @ probes.sizes[members]))
            out_y.append(float(share @ np.abs(probes.yaws[members])))
            out_single.append(False)
            out_px.append(px)
    return Faces(np.array(out_v), np.array(out_p, str), np.array(out_s), np.array(out_y), np.array(out_single, bool),
                 np.array(out_px))


@dataclass
class GalleryRows:
    """A gallery's photos and pose templates (unit ``rows``) and whose each is (``owners``: avatar names)."""

    rows: np.ndarray
    owners: np.ndarray

    @classmethod
    def of(cls, gallery) -> GalleryRows:
        g = gallery
        with g._lock:
            names, matrix, owners = list(g._names), g._matrix, g._owners
            templates, t_owners = g._templates, g._template_owners
        rows = np.vstack([m for m in (matrix, templates) if len(m)])
        who = np.concatenate([owners, t_owners]).astype(int)
        return cls(rows, np.array([names[k] for k in who], str))

    def subset(self, people) -> GalleryRows:
        keep = np.isin(self.owners, list(people))
        return GalleryRows(self.rows[keep], self.owners[keep])


def scores(probe: Faces, gallery: GalleryRows, kind: str = "none", cohort: Faces | None = None,
           top: int = 50) -> tuple[np.ndarray, list]:
    """``(person scores (faces x people), people)``: each face against each gallery person (the best of their rows),
    normalised by ``kind`` against ``cohort``'s single faces with the face's own avatar and the gallery person left out
    (see the module docs)."""
    people = sorted(set(gallery.owners.tolist()))
    raw = probe.vectors @ gallery.rows.T  # faces x rows
    if kind != "none":
        raw = _normalise(raw, probe, gallery, kind, cohort, top)
    out = np.full((len(probe.vectors), len(people)), -np.inf)
    for j, person in enumerate(people):
        mine = gallery.owners == person
        out[:, j] = raw[:, mine].max(axis=1)
    return out, people


def _normalise(raw, probe: Faces, gallery: GalleryRows, kind: str, cohort: Faces, top: int) -> np.ndarray:
    """T-norm or AS-norm of ``raw`` (faces x gallery rows), leaving out of the cohort each face's own avatar and each
    row's owner."""
    single = cohort.single
    c_vectors, c_people = cohort.vectors[single], cohort.people[single]
    out = np.empty_like(raw)
    owners = sorted(set(gallery.owners.tolist()))
    p_scores = probe.vectors @ c_vectors.T  # faces x cohort
    g_scores = gallery.rows @ c_vectors.T  # rows x cohort
    for person in owners:
        cols = np.flatnonzero(gallery.owners == person)
        for avatar in sorted(set(probe.people.tolist())):
            rows_ = np.flatnonzero(probe.people == avatar)
            keep = (c_people != avatar) & (c_people != person)
            ps = p_scores[np.ix_(rows_, np.flatnonzero(keep))]
            if kind == "tnorm":
                mean, spread = ps.mean(axis=1), np.maximum(ps.std(axis=1), 1e-3)
                out[np.ix_(rows_, cols)] = (raw[np.ix_(rows_, cols)] - mean[:, None]) / spread[:, None]
                continue
            k = min(top, ps.shape[1])
            pt = -np.partition(-ps, k - 1, axis=1)[:, :k]
            p_mean, p_spread = pt.mean(axis=1), np.maximum(pt.std(axis=1), 1e-3)
            gs = g_scores[np.ix_(cols, np.flatnonzero(keep))]
            gt = -np.partition(-gs, k - 1, axis=1)[:, :k]
            g_mean, g_spread = gt.mean(axis=1), np.maximum(gt.std(axis=1), 1e-3)
            block = raw[np.ix_(rows_, cols)]
            out[np.ix_(rows_, cols)] = 0.5 * ((block - p_mean[:, None]) / p_spread[:, None]
                                              + (block - g_mean[None, :]) / g_spread[None, :])
    return out


def fit(model: str, probe: Faces, gallery: GalleryRows, kind: str = "none", cohort: Faces | None = None,
        made: dict | None = None) -> Calibration:
    """The calibration of ``probe``'s faces against ``gallery`` (every face of an avatar in the gallery is a mated
    search for it and a non-mated one for everyone else)."""
    s, people = scores(probe, gallery, kind, cohort)
    known = np.isin(probe.people, people)
    values, mated, sizes, yaws, who = [], [], [], [], []
    for j, person in enumerate(people):
        values.append(s[known, j])
        mated.append(probe.people[known] == person)
        sizes.append(probe.sizes[known])
        yaws.append(probe.yaws[known])
        who.append(probe.people[known])
    return Calibration.fit(model, np.concatenate(values), np.concatenate(mated), np.concatenate(sizes),
                           np.concatenate(yaws), np.concatenate(who), normalisation=kind, made=made)


def check(model: str, probe: Faces, gallery: GalleryRows, kinds=KINDS, splits: int = SPLITS, seed: int = 0,
          targets=TARGETS, margin: float = 1.0) -> dict:
    """The held-out check (see the module docs): per normalisation and target false-alarm rate, the rate measured on
    strangers' faces (``fpir``, single faces and averages apart) and the share of known people's faces named
    correctly (``tpir``), pooled over ``splits`` random halvings of the avatars."""
    rng = np.random.default_rng(seed)
    avatars = np.array(sorted(set(probe.people.tolist())))
    out = {kind: {str(t): {"strangers": 0, "false": 0, "known": 0, "named": 0, "wrong": 0,
                           "false_single": 0, "strangers_single": 0} for t in targets} for kind in kinds}
    for _ in range(splits):
        order = rng.permutation(avatars)
        half = len(order) // 2
        fitting, rest = order[:half], order[half:]
        known, strangers = rest[: len(rest) // 2], rest[len(rest) // 2:]
        fit_faces = _subset(probe, fitting)
        test_faces = _subset(probe, rest)
        for kind in kinds:
            cohort = cohort_faces(fit_faces, seed=seed) if kind != "none" else None
            calibration = fit(model, fit_faces, gallery.subset(fitting), kind, cohort)
            s, people = scores(test_faces, gallery.subset(known), kind, cohort)
            n = len(people)
            curves = [calibration.curve(z, y) for z, y in zip(test_faces.sizes, test_faces.yaws)]
            rarity = np.array([[c.rarity(v) for v in row] for c, row in zip(curves, s)])
            order_ = np.argsort(-rarity, axis=1)
            best = rarity[np.arange(len(rarity)), order_[:, 0]]
            second = rarity[np.arange(len(rarity)), order_[:, 1]] if n > 1 else np.zeros(len(rarity))
            for t in targets:
                accept = -math.log10(per_person(t, n))
                named = (best >= accept) & (best - second >= margin)
                entry = out[kind][str(t)]
                is_stranger = np.isin(test_faces.people, strangers)
                entry["strangers"] += int(is_stranger.sum())
                entry["false"] += int((named & is_stranger).sum())
                entry["strangers_single"] += int((is_stranger & test_faces.single).sum())
                entry["false_single"] += int((named & is_stranger & test_faces.single).sum())
                is_known = ~is_stranger
                right = np.array([people[i] for i in order_[:, 0]]) == test_faces.people
                entry["known"] += int(is_known.sum())
                entry["named"] += int((named & is_known & right).sum())
                entry["wrong"] += int((named & is_known & ~right).sum())
    for kind in kinds:
        for t in targets:
            e = out[kind][str(t)]
            e["fpir"] = e["false"] / max(1, e["strangers"])
            e["fpir_single"] = e["false_single"] / max(1, e["strangers_single"])
            e["tpir"] = e["named"] / max(1, e["known"])
    return out


def _subset(faces_: Faces, people) -> Faces:
    keep = np.isin(faces_.people, list(people))
    return Faces(faces_.vectors[keep], faces_.people[keep], faces_.sizes[keep], faces_.yaws[keep],
                 faces_.single[keep], faces_.eye_px[keep])


def choose(results: dict, target: float = 0.01, kinds=KINDS) -> str:
    """The normalisation to keep: the one naming most known faces at ``target`` whose measured false-alarm rate stays
    within twice the target; ``none`` unless another names at least one point more."""
    t = str(target)
    usable = [k for k in kinds if results[k][t]["fpir"] <= 2 * target] or ["none"]
    best = max(usable, key=lambda k: results[k][t]["tpir"])
    if best != "none" and "none" in usable and results[best][t]["tpir"] < results["none"][t]["tpir"] + 0.01:
        return "none"
    return best


def cohort_faces(faces_: Faces, keep: int = COHORT_KEEP, seed: int = 0) -> Faces:
    """Up to ``keep`` single faces of each avatar: a cohort (the same for fitting a calibration and for shipping it)."""
    rng = np.random.default_rng(seed)
    rows = []
    for avatar in sorted(set(faces_.people.tolist())):
        mine = np.flatnonzero((faces_.people == avatar) & faces_.single)
        rows += sorted(rng.choice(mine, size=min(keep, len(mine)), replace=False).tolist())
    rows = np.array(rows, int)
    return Faces(faces_.vectors[rows], faces_.people[rows], faces_.sizes[rows], faces_.yaws[rows],
                 faces_.single[rows], faces_.eye_px[rows])


def cohort_of(model: str, faces_: Faces) -> Cohort:
    """The :class:`~.calibration.Cohort` of the cohort ``faces_`` (:func:`cohort_faces`), for shipping."""
    return Cohort(model, faces_.vectors, tuple(faces_.people.tolist()))


def save_report(path, report: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=1, default=str))
    return path


def write(model: str, calibration: Calibration, cohort: Cohort | None, models_folder=None) -> list[Path]:
    """Save ``calibration`` (and ``cohort``) where :func:`~.calibration.find_calibration` looks first."""
    paths = [calibration.save(calibration_path(model, models_folder))]
    if cohort is not None:
        paths.append(cohort.save(cohort_path(model, models_folder)))
    else:
        cohort_path(model, models_folder).unlink(missing_ok=True)
    return paths


def bins_table(calibration: Calibration, fpir: float = 0.01, people: int = 10) -> list[str]:
    """Lines for the user: per bin the accept and reject thresholds and the scores counted."""
    lines = []
    for s in range(len(calibration.size_edges) - 1):
        for y in range(len(calibration.yaw_edges) - 1):
            curve = calibration.curves[s][y]
            if curve.borrowed is not None:
                continue
            lo, hi = calibration.size_edges[s], calibration.size_edges[s + 1]
            accept = curve.threshold(per_person(fpir, people))
            reject = curve.mated_quantile(0.02)
            lines.append(f"  {lo:>4g}-{hi:<4g} px, yaw {calibration.yaw_edges[y]:g}-{calibration.yaw_edges[y + 1]:g}:"
                         f" accept {accept:.3f}, reject {reject if reject is None else round(reject, 3)}, "
                         f"non-mated {curve.counts[0]}, mated {curve.counts[1]}, avatars {curve.counts[2]}")
    return lines


def bin_of(size: float, yaw: float) -> tuple[int, int]:
    return bin_index(size, yaw)
