"""The evaluation's protocols, as manifests for ``bifold-faces eval`` (``--build``; :mod:`.evaluate`).

The protocols are those of ``docs/research/face_recognition_sota.md`` (section 4.3) that can be built from what the
repository has: rendered stills (P2), the ladder of baselines and ablations (``docs/research/high_res_faces.md``,
section 4.5), a public test pack (P6, LFW) and, on a machine with consented people, a folder of photos (P4). The scene
runs (P3) and webcam recordings (P5) are built by :mod:`.evalscene`.

**Who is on which side.** The identities measured are the fair's library (its 20 avatars); the shipped calibrations
were made from other avatars (:func:`sides` reads which, from the calibrations themselves), so every identity here is
held out from them. In the stills an identity is **known** (enrolled) or a **stranger** by the splits of
:func:`~.evaluate.splits` (half and half, many times): every identity is measured both ways.

**P2, rendered stills** (:func:`p2`). Per identity:

* **the gallery, set B**: three windows of its own avatar in the hall (2 m, 100 pixels between the eyes, turned -25,
  0 and 25 degrees, smiling a little, the fair's light);
* **probes of three kinds**: its own avatar with its own face (``face: own``; the ceiling, and the stranger whose
  face is the avatar's own), and its face as a person's: textures made from **set A** (other renders of it, neutral,
  under a studio light: S3's six) worn on another avatar of its sex and age (:func:`bodies`), on that avatar's own
  head (``head: own``) and on the head fitted to set A (``head: fitted``);
* **conditions** (windows of the 8K sensor, 4 m from the camera unless said): the eye distance over the research's
  bins (8 to 100 pixels) through each camera preset (ideal, good 4K, budget 1080p); at 38 pixels the head turned 15
  to 90 degrees (to the left for half the identities, to the right for the others), nodded -30 to 30 degrees, three
  more lights (side, warm, top), two occluders (the lower face, one side of the face); small faces turned (12, 18 and
  26 pixels at 30 and 60 degrees).

**The ladder** (:func:`ladder`): one change at a time, at matched distance and pose (the person where the 720p view
shows the face 10, 14 or 20 pixels between the eyes, frontal and turned 30 degrees), each rung adding to the one
before:

* B0: the 720p view, the old shading (``quality: low``, ``linear: false``), 1024 textures;
* B1: + the sensor's window (six times the view: 60, 84 and 120 pixels), the ideal camera; ``B1=``, the window at
  the view's own eye distance (a control: the window's pixels alone);
* B2: + the camera presets (good 4K and budget 1080p);
* B3: + 2048 textures (they show only where the eyes are at least 64 pixels apart);
* B4: + linear light and the heads' normal and specular maps (a face from photos is drawn without the avatar's maps:
  S2's fix);
* B5: + the head fitted to set A, with the textures made on it (known people only).

From B3 the good 4K camera is used. Every rung is measured for the avatars' own faces and for the faces made from
set A.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from .evaluate import FORMAT

GALLERY_YAWS = (-25.0, 0.0, 25.0)
GALLERY = {"view": "window", "ied": 100.0, "distance": 2.0, "light": "fair", "expression": "smile",
           "expression_weight": 0.5, "preset": "ideal", "face": "own", "head": "own"}
IEDS = (8.0, 12.0, 18.0, 26.0, 38.0, 54.0, 76.0, 100.0)  # one in each of the research's bins
BASE_IED = 38.0
PRESETS = ("ideal", "good 4K", "budget 1080p")
YAWS = (15.0, 30.0, 45.0, 60.0, 75.0, 90.0)
PITCHES = (-30.0, -15.0, 15.0, 30.0)
LIGHTS = ("side", "warm", "top")
OCCLUDERS = ("low", "side")
SMALL_TURNED = ((12.0, 30.0), (18.0, 30.0), (26.0, 30.0), (12.0, 60.0), (18.0, 60.0), (26.0, 60.0))
VARIANTS = {"own": {"face": "own", "head": "own"}, "renders": {"face": "renders", "head": "own"},
            "fitted": {"face": "renders", "head": "fitted"}}
LADDER_VIEW_IED = (10.0, 14.0, 20.0)  # pixels between the eyes in the 720p view: the fair's faces are 11-19
LADDER_YAWS = (0.0, 30.0)
SENSOR_SCALE = 6.0  # the 8K sensor against the 720p view

__all__ = ["bodies", "fair_identities", "gallery_items", "halves", "ladder", "lfw", "p2", "p3", "photos", "sides"]


def fair_identities(library=None) -> list[str]:
    """The fair's library's avatars (default: ``$BIFOLD_CHARACTERS``)."""
    from ..characters.library import CharacterLibrary

    return sorted((CharacterLibrary(library) if library else CharacterLibrary()).available())


def sides(identities) -> dict:
    """Which avatars are on which side: the ones measured, and per shipped calibration the ones it was made from
    (and those it left out because they wear a face of the fair's)."""
    from .calibration import PACKAGED

    out = {"evaluation": sorted(identities), "calibrations": {}}
    for path in sorted(PACKAGED.glob("*.json")):
        made = json.loads(path.read_text()).get("made", {})
        out["calibrations"][path.stem] = {"avatars": sorted(made.get("avatars", [])),
                                          "left_out": made.get("left_out", {})}
        overlap = sorted(set(made.get("avatars", [])) & set(identities))
        if overlap:
            out["calibrations"][path.stem]["overlap_with_evaluation"] = overlap
    return out


def _group(name: str) -> tuple[str, str]:
    sex = "female" if "female" in name.lower() else "male"
    return sex, "child" if "child" in name.lower() else "adult"


def bodies(identities) -> dict[str, str]:
    """``{identity: the avatar that wears its face}``: another avatar of the same sex and age group (the next in
    name order, round), or of the same age group when it is the only one of its sex (the two children)."""
    names = sorted(identities)
    out = {}
    for name in names:
        group = [n for n in names if _group(n) == _group(name)]
        if len(group) < 2:
            group = [n for n in names if _group(n)[1] == _group(name)[1]]
        if len(group) < 2:
            group = names
        out[name] = group[(group.index(name) + 1) % len(group)]
    return out


def _still(item_id, role, identity, conditions) -> dict:
    return {"id": item_id, "kind": "still", "role": role, "identity": identity, "source": "render",
            "conditions": conditions}


def gallery_items(identities) -> list[dict]:
    """Set B of ``identities``: the gallery's items (see the module docs)."""
    return [_still(f"g|{n}|{y:+.0f}", "gallery", n, {**GALLERY, "yaw": y}) for n in identities for y in GALLERY_YAWS]


_gallery = gallery_items


def _variant(name, identity, body) -> dict:
    c = dict(VARIANTS[name])
    c["body"] = identity if name == "own" else body
    return c


def p2(identities, size: str = "full", variants=tuple(VARIANTS)) -> dict:
    """The P2 manifest (see the module docs) for ``identities``; ``size``: ``full``, ``small`` (fewer conditions)
    or ``tiny`` (four, for tests)."""
    identities = sorted(identities)
    wearers = bodies(identities)
    items = _gallery(identities)
    for k, identity in enumerate(identities):
        sign = 1.0 if k % 2 == 0 else -1.0
        conditions = []
        if size == "tiny":
            conditions = [{"ied": 38.0}, {"ied": 12.0}, {"ied": 38.0, "yaw": 45.0 * sign},
                          {"ied": 26.0, "preset": "budget 1080p"}]
        else:
            presets = PRESETS if size == "full" else ("ideal",)
            conditions += [{"ied": ied, "preset": p} for ied in IEDS for p in presets]
            yaws = YAWS if size == "full" else (30.0, 60.0, 90.0)
            conditions += [{"ied": BASE_IED, "yaw": sign * y} for y in yaws]
            if size == "full":
                conditions += [{"ied": BASE_IED, "pitch": p} for p in PITCHES]
                conditions += [{"ied": ied, "yaw": sign * y} for ied, y in SMALL_TURNED]
            conditions += [{"ied": BASE_IED, "light": light} for light in (LIGHTS if size == "full" else ("side",))]
            conditions += [{"ied": BASE_IED, "occluder": o} for o in (OCCLUDERS if size == "full" else ("low",))]
        for variant in variants:
            base = _variant(variant, identity, wearers[identity])
            for c in conditions:
                full = {"view": "window", "distance": 4.0, "light": "fair", "preset": "ideal", **base, **c}
                tag = "|".join(f"{key}={full[key]}" for key in ("ied", "yaw", "pitch", "light", "preset", "occluder")
                               if key in c)
                item = _still(f"p|{identity}|{variant}|{tag}", "probe", identity, full)
                item["labels"] = {"variant": variant, "sweep": _sweep(c), "turn": abs(float(full.get("yaw", 0.0)))}
                items.append(item)
    breakdowns = [
        {"name": "variant", "by": ["variant"]},
        {"name": "ied by preset", "where": {"sweep": "ied"}, "by": ["variant", "preset", "ied"]},
        {"name": "turn", "where": {"sweep": "yaw"}, "by": ["variant", "turn"]},
        {"name": "pitch", "where": {"sweep": "pitch"}, "by": ["variant", "pitch"]},
        {"name": "light", "where": {"sweep": "light"}, "by": ["variant", "light"]},
        {"name": "occluder", "where": {"sweep": "occluder"}, "by": ["variant", "occluder"]},
        {"name": "small turned", "where": {"sweep": "small turned"}, "by": ["variant", "ied", "turn"]},
        {"name": "base", "where": {"sweep": "ied", "preset": "ideal", "ied": BASE_IED}, "by": ["variant"]},
    ]
    return {"format": FORMAT, "name": f"p2-{size}", "protocol": "P2",
            "description": "Rendered stills: the fair's avatars as people, the gallery set B, faces from set A worn by "
                           "other avatars, own and fitted heads, through the sensor's windows (protocols.p2)",
            "sides": {**sides(identities), "bodies": wearers},
            "settings": {"splits": 20, "seed": 0, "breakdowns": breakdowns}, "items": items}


def _sweep(c: dict) -> str:
    """Which of P2's sweeps a condition belongs to (the reports compare like with like)."""
    if "occluder" in c:
        return "occluder"
    if "light" in c:
        return "light"
    if "pitch" in c:
        return "pitch"
    if "yaw" in c:
        return "yaw" if c.get("ied") == BASE_IED else "small turned"
    return "ied"


def ladder(identities, variants=("own", "renders")) -> dict:
    """The ladder B0-B5 (see the module docs) for ``identities``."""
    identities = sorted(identities)
    wearers = bodies(identities)
    items = _gallery(identities)
    old = {"quality": "low", "linear": False, "detail": 1024}
    rungs = [("B0", {"view": "frame", **old, "preset": "ideal"}, "view"),
             ("B1", {"view": "window", **old, "preset": "ideal"}, "sensor"),
             ("B1=", {"view": "window", **old, "preset": "ideal"}, "view"),
             ("B2 good 4K", {"view": "window", **old, "preset": "good 4K"}, "sensor"),
             ("B2 budget 1080p", {"view": "window", **old, "preset": "budget 1080p"}, "sensor"),
             ("B3", {"view": "window", "quality": "low", "linear": False, "detail": 2048, "preset": "good 4K"},
              "sensor"),
             ("B4", {"view": "window", "quality": "high", "linear": True, "detail": 2048, "preset": "good 4K"},
              "sensor")]
    for k, identity in enumerate(identities):
        sign = 1.0 if k % 2 == 0 else -1.0
        for variant in (*variants, "fitted"):
            base = _variant(variant, identity, wearers[identity])
            mine = rungs if variant != "fitted" else [("B5", rungs[-1][1], "sensor")]
            for rung, settings, scale in mine:
                for view_ied in LADDER_VIEW_IED:
                    for yaw in LADDER_YAWS:
                        ied = view_ied * (SENSOR_SCALE if scale == "sensor" else 1.0)
                        c = {"light": "fair", **settings, **base, "ied": ied, "yaw": sign * yaw,
                             "distance": _distance(view_ied)}
                        item = _still(f"p|{identity}|{variant}|{rung}|v{view_ied:g}|y{sign * yaw:+.0f}", "probe",
                                      identity, c)
                        item["labels"] = {"rung": rung, "view_ied": view_ied, "variant": variant}
                        items.append(item)
    breakdowns = [{"name": "rung", "by": ["variant", "rung"]},
                  {"name": "rung by distance", "by": ["variant", "rung", "view_ied"]}]
    return {"format": FORMAT, "name": "ladder", "protocol": "B0-B5",
            "description": "Baselines and ablations at matched distance and pose (protocols.ladder)",
            "sides": {**sides(identities), "bodies": wearers},
            "settings": {"splits": 20, "seed": 0, "breakdowns": breakdowns}, "items": items}


def halves(identities) -> tuple[list[str], list[str]]:
    """Two complementary halves of ``identities`` to enrol in turn, such that no enrolled identity's carrier
    (:func:`bodies`) is enrolled with it: along each cycle of carriers every other identity (a carrier's head and
    hair show through the face it wears, so it must not be a known person of the same run). Identities that cannot
    be placed so (an odd cycle) are strangers in both."""
    wearers = bodies(identities)
    first, seen = [], set()
    for start in sorted(identities):
        if start in seen:
            continue
        cycle, name = [], start
        while name not in seen:
            seen.add(name)
            cycle.append(name)
            name = wearers[name]
        first += cycle[0:len(cycle) - len(cycle) % 2:2]
    second = [n for n in sorted(identities) if n not in first]
    second = [n for n in second if wearers[n] not in second]
    return sorted(first), sorted(second)


def p3(identities, runs: int = 4, seconds: float = 60.0, fps: float = 5.0, lanes: int = 8, crossers: int = 4,
       fpir: float = 0.01, detector: str = "profile", windows: bool = True, preset: str = "ideal") -> dict:
    """P3, rendered video (``evalscene``): ``runs`` parades, enrolling the two :func:`halves` in turn; in each, half
    the ``lanes`` known people (their faces from set A on the fitted head, worn by their carriers) and half strangers
    (avatars with their own faces), and ``crossers`` strangers walking across; seeded."""
    identities = sorted(identities)
    wearers = bodies(identities)
    a, b = halves(identities)
    items = []
    for r in range(runs):
        known = a if r % 2 == 0 else b
        strangers = [n for n in identities if n not in known]
        rng = random.Random(f"p3 {r}")
        lane_known = rng.sample(known, min(len(known), lanes // 2))
        lane_strangers = rng.sample(strangers, min(len(strangers), lanes - len(lane_known)))
        rest = [n for n in strangers if n not in lane_strangers]
        walking = rng.sample(rest, min(len(rest), crossers))
        people = [{"identity": n, "body": wearers[n], "face": "renders", "head": "fitted", "role": "lane"}
                  for n in lane_known]
        people += [{"identity": n, "body": n, "face": "own", "head": "own", "role": "lane"} for n in lane_strangers]
        rng.shuffle(people)
        people += [{"identity": n, "body": n, "face": "own", "head": "own", "role": "crosser"} for n in walking]
        items.append({"id": f"p3-{r + 1}", "kind": "scene", "scene": "parade", "seconds": float(seconds),
                      "fps": float(fps), "seed": r + 1, "people": people, "known": known, "windows": windows,
                      "preset": preset, "detector": detector, "fpir": fpir})
    return {"format": FORMAT, "name": f"p3-{runs}x{seconds:g}s", "protocol": "P3",
            "description": "Rendered video: parades of known people (faces from set A, fitted) and strangers through "
                           "the whole pipeline (protocols.p3, evalscene)",
            "sides": {**sides(identities), "bodies": wearers, "halves": [a, b]}, "items": items}


def _distance(view_ied: float) -> float:
    """About how far from the fair's camera a face has ``view_ied`` pixels between the eyes in the 720p view (a
    63 mm eye distance; windows of the ladder stand there, frames are placed exactly by the renderer)."""
    import math

    return round(360.0 / math.tan(math.radians(27.5)) * 0.063 / view_ied, 3)


def lfw(pack, per_fold=(75, 75)) -> dict:
    """P6: LFW's pairs from InsightFace's ``lfw.bin`` (``pack``): F1's subset (the first 75 mated and 75 non-mated
    pairs of each fold), or every pair (``per_fold`` ``None``)."""
    item = {"id": "lfw" if per_fold is None else f"lfw-{per_fold[0]}-{per_fold[1]}", "kind": "pack",
            "path": str(Path(pack).resolve()), "tracks": True}
    if per_fold is not None:
        item["per_fold"] = list(per_fold)
    return {"format": FORMAT, "name": item["id"], "protocol": "P6",
            "description": "LFW's verification pairs (checks only: nothing made from it is kept but numbers)",
            "items": [item]}


PHOTO_LAYOUT = """\
<folder>/
  enrol/<name>/*.jpg       enrolment photos of each consenting, enrolled person (the gallery)
  probe/<name>/*.jpg       other photos of the same people: other days, light and poses (never the enrolment ones)
  strangers/<name>/*.jpg   photos of consenting people who are not enrolled
  conditions.csv           optional: file,<label>,<label>... (a file's path below <folder>, then free labels such as
                           distance, light, pose, session, camera: each becomes a condition to group by)"""


def photos(folder, scales=(1.0, 0.5, 0.25, 0.125)) -> dict:
    """P4: a manifest of a folder of photos laid out as :data:`PHOTO_LAYOUT`; each probe photo also at ``scales``
    (shrunk, the size curve: downsampled photos are not native low resolution, so P5 is the arbiter for small
    faces). Paths are relative to ``folder``, where the manifest belongs."""
    from .evaluate import IMAGE_SUFFIXES

    folder = Path(folder)
    labels = {}
    table = folder / "conditions.csv"
    if table.is_file():
        import csv

        with table.open(newline="") as handle:
            for row in csv.DictReader(handle):
                path = (row.pop("file", "") or "").strip()
                if path:
                    labels[path] = {k: v for k, v in row.items() if k and v not in (None, "")}

    def pictures(where):
        return sorted(p for p in where.glob("*") if p.suffix.lower() in IMAGE_SUFFIXES and p.is_file())

    items = []
    for kind, role in (("enrol", "gallery"), ("probe", "probe"), ("strangers", "probe")):
        root = folder / kind
        if not root.is_dir():
            continue
        for person in sorted(p for p in root.iterdir() if p.is_dir()):
            identity = person.name if kind != "strangers" else f"stranger {person.name}"
            for picture in pictures(person):
                rel = picture.relative_to(folder).as_posix()
                base = {"id": f"{kind}|{person.name}|{picture.name}", "kind": "still", "role": role,
                        "identity": identity, "source": "image", "path": rel,
                        "conditions": {"set": kind, **labels.get(rel, {})}}
                if role == "gallery":
                    items.append(base)
                    continue
                for scale in scales:
                    items.append({**base, "id": f"{base['id']}|{scale:g}",
                                  "conditions": {**base["conditions"], "scale": scale}})
    return {"format": FORMAT, "name": f"photos-{folder.name}", "protocol": "P4",
            "description": "Photos of consenting people (P4): kept in git-ignored folders, never committed",
            "settings": {"splits": 0}, "items": items}
