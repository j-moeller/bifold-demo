"""``bifold-faces calibrate``: each model's calibration (:mod:`.calibration`), made from renders of avatars.

Calibrations must be made from data that may be shipped: renders of the Rocketbox avatars (MIT). Photos of real
people and LFW may check a calibration, never make one. The data:

* **Who.** The avatars of a character library (``--characters``; by default every prepared avatar). Each avatar is
  one identity: mated pairs are the same avatar, non-mated pairs different avatars; except that avatars showing the
  same face count as one (:func:`same_faces`: Rocketbox reuses a face, Security_Male_01 wears Male_Adult_03's; the
  first profile's model decides, the others take its say, so that their own lookalikes stay strangers), and that
  avatars wearing a face of another library's avatars (``--exclude-faces-of``, by default the fair's) are left out
  (:func:`shared_faces`: Wood_Male_01 wears the fair's Male_Adult_17's), so that the fair's people are strangers to
  the calibration and can check it.
* **Their gallery.** Each avatar's "photos": windows of the virtual sensor on its head in the fair's hall, 2 m from
  the camera, 100 pixels between the eyes, the head turned -25, 0 and 25 degrees, enrolled by the
  :class:`~.gallery.Gallery` as photos are (the same checks, templates per pose).
* **Their faces as the live pipeline sees them.** Windows (:class:`~bifold_demo.sensor.WindowRequest` with
  ``eye_px``: an exact eye distance) at swept face sizes (:data:`EYE_PX`, 8 to 90 pixels between the eyes: 90 for
  the bin of a webcam's near faces), the head turned (:data:`YAWS`), through camera presets (:data:`PRESETS`: an
  ideal camera and a budget 1080p one with its blur, noise and JPEG), at a few places in the hall (its light varies;
  each avatar's own random places), found by the profile's face detector as the live pipeline finds faces in
  windows, aligned, gated as live faces are (:meth:`Probes.passed`: those the live gate would not embed do not
  count) and embedded. Each face is binned by what the pipeline measures: the face's size from its box and its yaw
  from its landmarks. Renders and embeddings are kept per window, so new avatars or sizes add to what was done.
* **Evidence of several faces.** The decisions score a track's averaged faces, so the calibration also scores
  averages of two and three faces of one avatar at one size (neighbouring yaws), weighted by quality as the tracks
  are (:data:`AGGREGATES`).

Each face (and average) is scored against every avatar's gallery (a person's score: the best of their photos and pose
templates); the scores give each bin's non-mated and mated curves (:meth:`.calibration.Calibration.fit`).

**Normalisation and the check.** The avatars are split in halves, many times over: one half calibrates; in the other,
half the avatars are known (their gallery) and half strangers. On those held-out searches the false-alarm rate asked
for (0.1 %, 1 %, 10 % per decision) is compared with the one measured, and the share of known faces named correctly
at it. Done without normalisation and with T-norm and AS-norm against a cohort of the calibrating half's faces (up to
:data:`COHORT_KEEP` single faces of each avatar, each face's own avatar and the gallery person left out). Kept
(``--normalisation auto``, :func:`choose`) is the one naming the most among those whose measured rate at 1 % stays
within twice the target: no normalisation unless another names at least a point more. The final
calibration uses every avatar; with a normalisation, its cohort is kept beside it (``<model id>.cohort.npz``), the
very faces its scores were normalised with.

**The target per stranger track.** A live track is decided again after every new face, so a stranger's *track* is
named more often than a single decision is. :func:`track_check` simulates strangers' tracks on more halvings
(:data:`TRACK_SPLITS`): each stranger's faces at one camera and one class of sizes arrive one by one, and the running
quality-weighted mean is decided after each, as live (enough evidence, the accept threshold, the margin). The offsets
(:func:`track_offsets`: decades added to every accept threshold, per target of :data:`TRACK_TARGETS`) are the
smallest whose upper bound (one-sided 95 %, a bootstrap over identity pairs: a stranger's false alarms come from the
few people they resemble) keeps the share of stranger tracks named at the target; :func:`track_rates` then measures
the rate per track with them on as many other halvings. The calibration keeps them (``track``), with that check.

The renders, the aligned faces and their embeddings are kept in ``--data`` (default ``<models>/calibration/data``),
so that a calibration can be made again without rendering (``--refit``).
"""

from __future__ import annotations

import dataclasses
import json
import math
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .calibration import Calibration, Cohort, bin_index, calibration_path, cohort_path, per_person

# pixels between the eyes asked of the windows (90: the bin of faces 80 pixels and more apart, a webcam's near faces)
EYE_PX = (8.0, 11.0, 14.0, 18.0, 24.0, 32.0, 44.0, 64.0, 90.0)
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
# two avatars show the same face when one's faces score at least this share of their own median against the other's
# photos (AdaFace and the ResNet-50: Security_Male_01 and Male_Adult_03 0.80-0.85 of it, the next most alike pair
# 0.52-0.55; SFace tells faces apart less well: 0.75 for them, 0.69-0.72 for two different men)
SAME_FACE = 0.8
# Rocketbox avatars that wear another's face (their renders and head textures show it), counted as one (or, when the
# other is in another library, left out) whatever the model's scores say. Wood_Male_01 wears the face of the fair's
# Male_Adult_17 (SFace scores their photos 0.89, any two different avatars at most 0.68)
KNOWN_SAME_FACES = {"Security_Male_01": "Male_Adult_03", "Wood_Male_01": "Male_Adult_17"}
# stranger tracks (:func:`stranger_tracks`): a track's faces come from one of these classes of eye distances asked
TRACK_SIZES = ((8.0, 11.0), (14.0, 18.0), (24.0, 32.0), (44.0, 64.0), (90.0,))
TRACK_TARGETS = (0.1, 0.03, 0.01, 0.003, 0.001)  # false-alarm rates per stranger track the offsets are measured at
TRACK_SPLITS = 60  # halvings for the offsets (and as many others for the check of them)
CONFIDENCE = 0.95  # the offsets keep the upper bound of this (one-sided) interval at the target
OFFSET_STEP, OFFSET_MOST = 0.02, 3.0  # decades

__all__ = ["AGGREGATES", "DISTANCES", "EYE_PX", "PRESETS", "TRACK_SIZES", "TRACK_TARGETS", "YAWS", "Probes",
           "StrangerTracks", "as_one", "check", "fit", "render", "same_faces", "scores", "shared_faces",
           "stranger_tracks", "track_check", "track_offsets", "track_rates", "window_key", "window_keys"]


@dataclass
class Probes:
    """The faces of the avatars as one face detector found them in the windows: per window the ``avatar``, the
    ``eye_px`` asked, the ``yaw`` asked, the ``preset``, the ``distance``; ``found``: whether the face was found;
    ``crops``: the aligned faces (``(n, 112, 112, 3)``, black where none was found); ``sizes``: the face's eye distance
    turned to the camera (from its box, window pixels); ``yaws``: the yaw its landmarks give; ``weights``: its
    quality weight without eDifFIQA; ``detector``: the detector's id; ``rendered``: how the windows were drawn (JSON:
    the renderer's quality and light model, the hall's hemisphere light, the heads' textures), recorded with every
    calibration made from them (scores depend on it)."""

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
    rendered: str = "{}"

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
        values["rendered"] = str(values.get("rendered", "{}"))
        return cls(**values)

    def rendering(self) -> dict:
        """How the windows were drawn (see the class docs; ``{}`` when not recorded)."""
        return json.loads(self.rendered)

    def keys(self) -> np.ndarray:
        """Each window's key (:func:`window_key`): which avatar, size, yaw and camera it shows."""
        return np.array([window_key(a, px, y, p) for a, px, y, p in zip(self.avatar.tolist(), self.eye_px.tolist(),
                                                                         self.yaw.tolist(), self.preset.tolist())],
                        str)

    def subset(self, mask) -> Probes:
        """The windows where ``mask`` is true."""
        mask = np.asarray(mask)
        values = {k: getattr(self, k) for k in self.__dataclass_fields__}
        return Probes(**{k: (v[mask] if isinstance(v, np.ndarray) else v) for k, v in values.items()})

    def concat(self, other: Probes) -> Probes:
        """These windows, then ``other``'s (of the same detector, drawn the same way)."""
        if other.detector != self.detector:
            raise ValueError(f"probes of {self.detector} and {other.detector} cannot be joined")
        values = {k: getattr(self, k) for k in self.__dataclass_fields__}
        return Probes(**{k: (np.concatenate([v, getattr(other, k)]) if isinstance(v, np.ndarray) else v)
                         for k, v in values.items()})

    def _quality(self, k: int, fiqa=None):
        from .quality import Quality

        return Quality(float(self.ied[k]), float(self.yaws[k]), 0.0, 0.0, float(self.sharpness[k]), 128.0, 50.0, 0.0,
                       float(self.outside[k]), float(self.score[k]), None if fiqa is None else float(fiqa[k]))

    def weights(self, fiqa=None) -> np.ndarray:
        """The quality weights (:attr:`~.quality.Quality.weight`), with eDifFIQA-T's scores when given."""
        out = np.zeros(len(self.found))
        for k in np.flatnonzero(self.found):
            out[k] = self._quality(k, fiqa).weight
        return out

    def passed(self, fiqa=None, gate=None) -> np.ndarray:
        """Which faces were found and pass the live pipeline's quality gate (``gate``, default
        :class:`~.evidence.QualityGate`; with eDifFIQA-T's scores when given), as live faces must to be embedded."""
        from .evidence import QualityGate

        gate = gate or QualityGate()
        out = np.zeros(len(self.found), bool)
        for k in np.flatnonzero(self.found):
            out[k] = gate.check(self._quality(k, fiqa), float(self.sizes[k]), float(self.score[k])) is None
        return out


# -- rendering -----------------------------------------------------------------------------------------------------


def window_key(avatar: str, eye_px: float, yaw: float, preset: str) -> str:
    """The key of a calibration window: ``avatar|eye_px|yaw|preset``."""
    return f"{avatar}|{float(eye_px):g}|{float(yaw):g}|{preset}"


def window_keys(names, eye_px=EYE_PX, yaws=YAWS, presets=PRESETS) -> list[str]:
    """Every window :func:`render` draws of ``names``."""
    return [window_key(n, px, y, p) for n in names for y in yaws for px in eye_px for p in presets]


def render(library, names, detectors: dict, gallery_folder, eye_px=EYE_PX, yaws=YAWS, presets=PRESETS,
           distances=DISTANCES, progress=print, seed: int = 0, skip=frozenset()) -> dict:
    """Render ``names`` (avatars of ``library``) in the fair's hall: their gallery photos (written as PNG files into
    ``gallery_folder/<avatar>/``, unless there already) and their faces through the windows, found by each of
    ``detectors`` (``{id: detector}``), except the windows whose :func:`window_key` is in ``skip`` (rendered before).
    Each avatar's places in the hall are drawn from its own random numbers (``seed`` and its name), so an avatar's
    renders do not depend on which others are rendered with it. Returns ``{detector id: Probes}`` of the windows
    rendered. Needs OpenGL."""
    import cv2

    from ..characters.character import Character
    from ..characters.model import CharacterState
    from ..characters.rocketbox import FORMAT
    from ..scene.renderer import Renderer
    from ..scenes import fair
    from ..sensor.geometry import Sensor
    from ..sensor.requests import WindowRequest
    from ..sensor.service import eye_points, render_windows
    from .align import align
    from .heads import WINDOW_CANVAS, face_size
    from .quality import assess

    gallery_folder = Path(gallery_folder)
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
    available = library.available()
    rendered = json.dumps({"quality": renderer.window_quality, "linear": renderer.linear,
                           "hemisphere": scene.hemisphere, "specular": renderer.specular,
                           "normal_strength": renderer.normal_strength,
                           "heads": "2048 textures and maps" if all(available.get(n, {}).get("format") == FORMAT
                                                                     for n in names) else "some without maps",
                           "sensor": f"{sensor.width}x{sensor.height}", "presets": list(presets)})
    rows = {key: {k: [] for k in ("avatar", "eye_px", "yaw", "preset", "distance", "found", "crops", "sizes",
                                  "yaws", "sharpness", "outside", "ied", "score")} for key in detectors}
    started = time.perf_counter()
    try:
        for n, name in enumerate(names):
            rng = np.random.default_rng([seed, zlib.crc32(name.encode())])
            poses = [(2.0, 0.0, yaw, "gallery") for yaw in GALLERY_YAWS
                     if not (gallery_folder / name / f"yaw{yaw:+.0f}.png").is_file()]
            poses += [(float(rng.choice(distances)), float(rng.uniform(-0.8, 0.8)), yaw, "probe") for yaw in yaws]
            wanted = {(yaw, px, preset) for yaw in yaws for px in eye_px for preset in presets
                      if window_key(name, px, yaw, preset) not in skip}
            poses = [p for p in poses if p[3] == "gallery" or any(w[0] == p[2] for w in wanted)]
            if not poses:
                continue
            person = Character(library.load(name), name=name, shadow=False)
            scene.add(person)
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
                                          preset=preset) for px in eye_px for preset in presets
                            if (yaw, px, preset) in wanted]
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
                          np.array(r["score"]), rendered)
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
    ``yaws`` (absolute) for the bins, whether each is a ``single`` face (the cohort's), the eye distance asked
    (``eye_px``), and where known their quality ``weights`` (an average's: the sum) and camera ``presets``."""

    vectors: np.ndarray
    people: np.ndarray
    sizes: np.ndarray
    yaws: np.ndarray
    single: np.ndarray
    eye_px: np.ndarray
    weights: np.ndarray | None = None
    presets: np.ndarray | None = None

    def take(self, keep) -> Faces:
        """The faces where ``keep`` (a mask or indices) says."""
        return Faces(*(None if v is None else v[keep] for v in (self.vectors, self.people, self.sizes, self.yaws,
                                                                 self.single, self.eye_px, self.weights,
                                                                 self.presets)))


def faces(probes: Probes, vectors: np.ndarray, weights: np.ndarray, passed=None) -> Faces:
    """The single faces found, and averages of two and three faces of one avatar at one size, preset and place
    (:data:`AGGREGATES`, neighbouring yaws), weighted by quality as a track's evidence is; ``passed``: only the faces
    that pass the live pipeline's quality gate (:meth:`Probes.passed`), as live ones must."""
    found = probes.found & (weights > 0)
    if passed is not None:
        found &= np.asarray(passed, bool)
    out_v, out_p, out_s, out_y, out_single, out_px, out_w, out_pre = [], [], [], [], [], [], [], []
    for k in np.flatnonzero(found):
        out_v.append(vectors[k])
        out_p.append(probes.avatar[k])
        out_s.append(probes.sizes[k])
        out_y.append(abs(probes.yaws[k]))
        out_single.append(True)
        out_px.append(probes.eye_px[k])
        out_w.append(weights[k])
        out_pre.append(probes.preset[k])
    groups: dict = {}
    for k in np.flatnonzero(found):
        groups.setdefault((probes.avatar[k], probes.eye_px[k], probes.preset[k]), {})[float(probes.yaw[k])] = k
    for (avatar, px, preset), by_yaw in groups.items():
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
            out_w.append(float(w.sum()))
            out_pre.append(preset)
    return Faces(np.array(out_v), np.array(out_p, str), np.array(out_s), np.array(out_y), np.array(out_single, bool),
                 np.array(out_px), np.array(out_w), np.array(out_pre, str))


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
    correctly (``tpir``), pooled over ``splits`` random halvings of the avatars; ``fpir_by_split``: each halving's
    own rate (they differ much: a stranger alike to a known person names many of their faces)."""
    rng = np.random.default_rng(seed)
    avatars = np.array(sorted(set(probe.people.tolist())))
    out = {kind: {str(t): {"strangers": 0, "false": 0, "known": 0, "named": 0, "wrong": 0,
                           "false_single": 0, "strangers_single": 0, "fpir_by_split": []} for t in targets}
           for kind in kinds}
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
            rarity = rarities(calibration, s, test_faces.sizes, test_faces.yaws)
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
                entry["fpir_by_split"].append(round(float((named & is_stranger).sum() / max(1, is_stranger.sum())), 5))
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


def rarities(calibration: Calibration, values: np.ndarray, sizes, yaws) -> np.ndarray:
    """The rarities of ``values`` (one row per face: its scores against the people) under ``calibration``, each row in
    its face's bin (``sizes``, ``yaws``)."""
    values = np.asarray(values, dtype=np.float64)
    out = np.zeros_like(values)
    bins = np.array([calibration.bin(z, y) for z, y in zip(sizes, yaws)], int).reshape(-1, 2)
    for s, y in {tuple(b) for b in bins.tolist()}:
        here = (bins[:, 0] == s) & (bins[:, 1] == y)
        out[here] = calibration.curves[s][y].rarities(values[here])
    return out


# -- stranger tracks: the target per track ----------------------------------------------------------------------------


@dataclass
class StrangerTracks:
    """Simulated stranger tracks (:func:`stranger_tracks`), one entry per track: the stranger (``avatar``), the halving
    (``split``), the gallery's size (``people``), the highest rarity of the best person among its decisions whose
    margin held (``best``; ``-inf``: none did) and whom that decision would name (``named``; where none held, the best
    person of its most alike decision): the identity pair the track's false alarms come from. Per decision:
    ``decision_track`` (its track) and ``decision_best`` (the best person's rarity where the margin held, else
    ``-inf``)."""

    avatar: np.ndarray
    split: np.ndarray
    people: np.ndarray
    best: np.ndarray
    named: np.ndarray
    decision_track: np.ndarray
    decision_best: np.ndarray

    def __len__(self) -> int:
        return len(self.avatar)

    @classmethod
    def join(cls, parts) -> StrangerTracks:
        parts = [p for p in parts if len(p)]
        if not parts:
            return cls(*(np.zeros(0, t) for t in (str, int, int, float, str, int, float)))
        offsets = np.cumsum([0] + [len(p) for p in parts[:-1]])
        return cls(np.concatenate([p.avatar for p in parts]), np.concatenate([p.split for p in parts]),
                   np.concatenate([p.people for p in parts]), np.concatenate([p.best for p in parts]),
                   np.concatenate([p.named for p in parts]),
                   np.concatenate([p.decision_track + o for p, o in zip(parts, offsets)]),
                   np.concatenate([p.decision_best for p in parts]))

    def pairs(self) -> np.ndarray:
        """Each track's identity pair (the stranger and the person it would be named as, in either order)."""
        return np.array(["|".join(sorted((a, b))) for a, b in zip(self.avatar.tolist(), self.named.tolist())], str)

    def excess(self, target: float, offset: float = 0.0) -> np.ndarray:
        """Per track, how far (decades) its best decision passes the accept threshold for ``target`` plus ``offset``
        (a track is named where this is at least 0)."""
        accept = np.array([-math.log10(per_person(target, int(n))) for n in self.people])
        return self.best - accept - offset

    def decisions_named(self, target: float, offset: float = 0.0) -> float:
        """The share of decisions (not tracks) that name someone at ``target`` plus ``offset``."""
        accept = np.array([-math.log10(per_person(target, int(n))) for n in self.people])
        return float(np.mean(self.decision_best >= accept[self.decision_track] + offset - 1e-12)) \
            if len(self.decision_best) else 0.0


def stranger_tracks(calibration: Calibration, probe: Faces, gallery: GalleryRows, strangers, kind: str = "none",
                    cohort: Faces | None = None, seed: int = 0, split: int = 0, margin: float = 1.0,
                    min_weight: float = 0.5, min_frames: int = 2, top: int = 50, sizes=TRACK_SIZES,
                    single_face: float = 40.0, single_yaw: float = 25.0) -> StrangerTracks:
    """Strangers' tracks as the live decisions see them: each stranger's single faces at one camera preset and one
    class of sizes (:data:`TRACK_SIZES`; all their yaws) arrive one by one in a random order; after each (from the
    ``min_frames``-th, or from the first while their mean face is ``single_face`` pixels between the eyes or more and
    turned less than ``single_yaw``; once their quality weights reach ``min_weight``: the live decisions' settings,
    :class:`~.decision.DecisionSettings`) the running quality-weighted mean of the faces
    so far is scored against ``gallery`` (normalised by ``kind`` against ``cohort``'s single faces, which hold none of
    these people) and decided as live: the best person's rarity in the mean's bin, and whether it beats the
    runner-up's by ``margin``. ``probe`` must carry its faces' ``weights`` and ``presets``; ``sizes``: the classes."""
    people = sorted(set(gallery.owners.tolist()))
    owner = np.searchsorted(np.array(people), gallery.owners)
    order = np.argsort(owner, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(owner[order])) + 1]
    rows = gallery.rows[order]
    c_vectors = None if kind == "none" else cohort.vectors[cohort.single]
    g_stats = None
    if kind == "asnorm":
        g = rows @ c_vectors.T
        k = min(top, g.shape[1])
        g = -np.partition(-g, k - 1, axis=1)[:, :k]
        g_stats = (g.mean(axis=1), np.maximum(g.std(axis=1), 1e-3))
    rng = np.random.default_rng([seed, split])
    single = probe.single & np.isin(probe.people, list(strangers))
    out = {"avatar": [], "best": [], "named": [], "decision_track": [], "decision_best": []}
    for avatar in sorted(strangers):
        mine = single & (probe.people == avatar)
        for preset in sorted(set(probe.presets[mine].tolist())):
            for group in sizes:
                idx = np.flatnonzero(mine & (probe.presets == preset) & np.isin(probe.eye_px, group))
                if not len(idx):
                    continue
                idx = rng.permutation(idx)
                w = np.maximum(probe.weights[idx], 1e-6)
                total = np.cumsum(w)
                mean = np.cumsum(w[:, None] * probe.vectors[idx], axis=0)
                mean /= np.maximum(np.linalg.norm(mean, axis=1, keepdims=True), 1e-12)
                size = np.cumsum(w * probe.sizes[idx]) / total
                yaw = np.cumsum(w * probe.yaws[idx]) / total
                one = (single_face > 0) & (size >= single_face) & (yaw < single_yaw)
                decide = ((np.arange(1, len(idx) + 1) >= min_frames) | one) & (total >= min_weight - 1e-9)
                if not decide.any():
                    continue
                mean, size, yaw = mean[decide], size[decide], yaw[decide]
                raw = mean @ rows.T
                if kind == "tnorm":
                    c = mean @ c_vectors.T
                    raw = (raw - c.mean(axis=1, keepdims=True)) / np.maximum(c.std(axis=1, keepdims=True), 1e-3)
                elif kind == "asnorm":
                    c = mean @ c_vectors.T
                    k = min(top, c.shape[1])
                    c = -np.partition(-c, k - 1, axis=1)[:, :k]
                    p_mean, p_spread = c.mean(axis=1, keepdims=True), np.maximum(c.std(axis=1, keepdims=True), 1e-3)
                    raw = 0.5 * ((raw - p_mean) / p_spread + (raw - g_stats[0][None]) / g_stats[1][None])
                person = np.maximum.reduceat(raw, starts, axis=1)
                rarity = rarities(calibration, person, size, yaw)
                ranked = np.argsort(-rarity, axis=1, kind="stable")
                best = rarity[np.arange(len(rarity)), ranked[:, 0]]
                second = rarity[np.arange(len(rarity)), ranked[:, 1]] if len(people) > 1 else np.zeros(len(rarity))
                held = np.where(best - second >= margin, best, -np.inf)
                j = int(np.argmax(held)) if np.isfinite(held).any() else int(np.argmax(best))
                track = len(out["avatar"])
                out["avatar"].append(avatar)
                out["best"].append(float(held.max()))
                out["named"].append(people[ranked[j, 0]])
                out["decision_track"] += [track] * len(held)
                out["decision_best"] += held.tolist()
    n = len(out["avatar"])
    return StrangerTracks(np.array(out["avatar"], str), np.full(n, split), np.full(n, len(people)),
                          np.array(out["best"]), np.array(out["named"], str), np.array(out["decision_track"], int),
                          np.array(out["decision_best"]))


def track_check(model: str, probe: Faces, gallery: GalleryRows, kind: str = "none", splits: int = TRACK_SPLITS,
                seed: int = 0, progress=None) -> StrangerTracks:
    """Stranger tracks (:func:`stranger_tracks`) over ``splits`` random halvings of the avatars, as the held-out check
    halves them (:func:`check`): one half calibrates (with ``kind``'s cohort of its faces), the other is split into
    known people and strangers; the strangers' tracks are decided against the known people's gallery."""
    rng = np.random.default_rng(seed)
    avatars = np.array(sorted(set(probe.people.tolist())))
    parts = []
    for split in range(splits):
        order = rng.permutation(avatars)
        half = len(order) // 2
        fitting, rest = order[:half], order[half:]
        known, strangers = rest[: len(rest) // 2], rest[len(rest) // 2:]
        fit_faces = _subset(probe, fitting)
        cohort = cohort_faces(fit_faces, seed=seed) if kind != "none" else None
        calibration = fit(model, fit_faces, gallery.subset(fitting), kind, cohort)
        parts.append(stranger_tracks(calibration, _subset(probe, strangers), gallery.subset(known), strangers, kind,
                                     cohort, seed, split))
        if progress and (split + 1) % 10 == 0:
            progress(f"[calibrate] {model}: stranger tracks of {split + 1} of {splits} halvings")
    return StrangerTracks.join(parts)


def _bootstrap(tracks: StrangerTracks, draws: int, seed: int):
    """``(cluster of each track, draws x clusters counts)``: a bootstrap over identity pairs (clusters drawn with
    replacement), for intervals that allow for the false alarms coming from few pairs."""
    _, cluster = np.unique(tracks.pairs(), return_inverse=True)
    clusters = int(cluster.max()) + 1 if len(cluster) else 0
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(clusters, np.full(clusters, 1.0 / max(clusters, 1)), size=draws).astype(np.float32)
    return cluster, clusters, counts


def track_offsets(tracks: StrangerTracks, targets=TRACK_TARGETS, confidence: float = CONFIDENCE, draws: int = 2000,
                  seed: int = 0, step: float = OFFSET_STEP, most: float = OFFSET_MOST) -> dict:
    """The offsets (decades added to a single decision's accept threshold) that keep the false-alarm rate per
    stranger track at each of ``targets``: the smallest (from 0, in steps of ``step``) whose upper bound (one-sided,
    ``confidence``; a bootstrap over identity pairs) is at most the target; ``most`` where none is. Each target's
    accept threshold with its offset stays at least as strict as a looser target's. Returns the calibration's
    ``track`` (:meth:`~.calibration.Calibration.track_offset`) with how they were measured."""
    cluster, clusters, counts = _bootstrap(tracks, draws, seed)
    sizes = np.bincount(cluster, minlength=clusters).astype(np.float32)
    grid = np.arange(0.0, most + step / 2, step)
    totals = counts @ sizes
    out = {"targets": [], "offsets": [], "upper": [], "rate": [], "rate_without": [], "upper_without": []}
    previous = None
    for target in sorted(targets, reverse=True):
        excess = tracks.excess(target)
        named = np.stack([np.bincount(cluster, weights=(excess >= g - 1e-12), minlength=clusters) for g in grid],
                         axis=1).astype(np.float32)
        upper = np.quantile((counts @ named) / totals[:, None], confidence, axis=0)
        rate = named.sum(axis=0) / sizes.sum()
        ok = np.flatnonzero(upper <= target)
        k = int(ok[0]) if len(ok) else len(grid) - 1
        offset = float(grid[k])
        if previous is not None:  # (no looser than the looser target's threshold)
            loose_target, loose_offset = previous
            n = int(np.median(tracks.people))
            floor = loose_offset - (math.log10(per_person(loose_target, n)) - math.log10(per_person(target, n)))
            offset = max(offset, round(floor, 4))
        previous = (target, offset)
        k = min(round(offset / step), len(grid) - 1)
        out["targets"].append(target)
        out["offsets"].append(round(offset, 4))
        out["upper"].append(round(float(upper[k]), 5))
        out["rate"].append(round(float(rate[k]), 5))
        out["rate_without"].append(round(float(rate[0]), 5))
        out["upper_without"].append(round(float(upper[0]), 5))
    out.update(confidence=confidence, tracks=len(tracks), pairs=clusters, splits=len(set(tracks.split.tolist())),
               draws=int(draws), step=step)
    return out


def track_rates(tracks: StrangerTracks, track: dict, interval: float = 0.9, draws: int = 2000, seed: int = 1) -> dict:
    """The false-alarm rate per stranger track measured at each target of ``track`` (offsets,
    :func:`track_offsets`) and without them: the share of tracks named, with an ``interval`` (two-sided) from a
    bootstrap over identity pairs; the share of single decisions naming; how many tracks were named and from how
    many pairs."""
    cluster, clusters, counts = _bootstrap(tracks, draws, seed)
    sizes = np.bincount(cluster, minlength=clusters).astype(np.float32)
    totals = counts @ sizes
    lo, hi = (1.0 - interval) / 2, 1.0 - (1.0 - interval) / 2
    pairs = tracks.pairs()
    out = {}
    for target, offset in zip(track["targets"], track["offsets"]):
        entry = {}
        for label, used in (("with", offset), ("without", 0.0)):
            named = tracks.excess(target, used) >= -1e-12
            per_cluster = np.bincount(cluster, weights=named, minlength=clusters).astype(np.float32)
            boot = (counts @ per_cluster) / totals
            entry[label] = {"offset": used, "per_track": round(float(named.mean()), 5),
                            "interval": [round(float(np.quantile(boot, lo)), 5), round(float(np.quantile(boot, hi)),
                                                                                       5)],
                            "named": int(named.sum()), "pairs": len(set(pairs[named].tolist())),
                            "per_decision": round(tracks.decisions_named(target, used), 5)}
        out[str(target)] = entry
    out["tracks"], out["pairs"], out["interval"] = len(tracks), clusters, interval
    out["splits"] = len(set(tracks.split.tolist()))
    return out


def same_faces(probe: Faces, gallery: GalleryRows, share: float = SAME_FACE, known=None) -> dict[str, str]:
    """Avatars that show the same face as another, ``{avatar: the avatar it counts as}`` (the first by name of each
    group): those of :data:`KNOWN_SAME_FACES` and ``known`` (pairs found before, by another model), and those the
    scores show, judged on single faces of at least 20 pixels between the eyes turned less than 25 degrees: the median
    of an avatar's scores against the other's photos reaches ``share`` of the lower of the two avatars' medians
    against their own."""
    s, people = scores(probe, gallery)
    big = (probe.sizes >= 20) & (probe.yaws < 25) & probe.single
    own = {}
    for j, person in enumerate(people):
        mine = big & (probe.people == person)
        if mine.any():
            own[person] = float(np.median(s[mine, j]))
    parent = {person: person for person in people}

    def root(person):
        while parent[person] != person:
            person = parent[person]
        return person

    for a, b in {**KNOWN_SAME_FACES, **(known or {})}.items():
        if a in parent and b in parent:
            first, second = sorted((root(a), root(b)))
            parent[second] = first
    for a in sorted(own):
        mine = big & (probe.people == a)
        for j, b in enumerate(people):
            if b != a and b in own and float(np.median(s[mine, j])) >= share * min(own[a], own[b]):
                first, second = sorted((root(a), root(b)))
                parent[second] = first
    return {person: root(person) for person in people if root(person) != person}


def shared_faces(probe: Faces, gallery: GalleryRows, others: GalleryRows, share: float = SAME_FACE,
                 known=None) -> dict[str, str]:
    """Avatars of ``probe`` that wear a face of another library's avatars (``others``: their gallery photos), ``{avatar:
    the other avatar}``: those of :data:`KNOWN_SAME_FACES` and ``known`` whose other avatar is among ``others``, and
    those whose single faces of at least 20 pixels between the eyes turned less than 25 degrees score against an
    other avatar's photos a median of at least ``share`` of their median against their own photos. Such an avatar is
    left out of a calibration made to be checked on the other library (the fair's)."""
    s, people = scores(probe, gallery)
    o, other_people = scores(probe, others)
    big = (probe.sizes >= 20) & (probe.yaws < 25) & probe.single
    out = {}
    for j, person in enumerate(people):
        mine = big & (probe.people == person)
        if not mine.any():
            continue
        own = float(np.median(s[mine, j]))
        medians = np.median(o[mine], axis=0)
        k = int(np.argmax(medians))
        if medians[k] >= share * own:
            out[person] = other_people[k]
    for a, b in {**KNOWN_SAME_FACES, **(known or {})}.items():
        if a in people and b in other_people:
            out.setdefault(a, b)
    return out


def as_one(faces_: Faces, gallery: GalleryRows, alias: dict) -> tuple[Faces, GalleryRows]:
    """``faces_`` and ``gallery`` with each avatar of ``alias`` (:func:`same_faces`) counted as the one it shows."""
    if not alias:
        return faces_, gallery

    def renamed(names):
        return np.array([alias.get(n, n) for n in names.tolist()], str)

    return (dataclasses.replace(faces_, people=renamed(faces_.people)),
            GalleryRows(gallery.rows, renamed(gallery.owners)))


def _subset(faces_: Faces, people) -> Faces:
    return faces_.take(np.isin(faces_.people, list(people)))


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
    chosen = faces_.take(rows)
    return dataclasses.replace(chosen, vectors=chosen.vectors.astype(np.float16).astype(np.float64))  # (as saved)


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
