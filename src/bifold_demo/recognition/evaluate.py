"""``bifold-faces eval``: the face recognition measured on manifests of stills, renders, test packs and scene runs
(step E1), with the metrics of :mod:`.metrics` and intervals over people.

**Manifests** (JSON, or CSV for stills) list items with their conditions::

    {"format": 1, "name": "p2", "protocol": "P2", "description": "...",
     "sides": {"evaluation": [...], "calibration": [...]},       # which avatars or people are on which side
     "settings": {"splits": 20, "seed": 0},                     # how the identities are split (see below)
     "defaults": {"light": "fair", "preset": "ideal"},          # conditions every rendered item gets unless it says
     "items": [
       {"id": "g-Ada-0", "kind": "still", "role": "gallery", "identity": "Ada", "source": "image",
        "path": "people/Ada/enrol/1.jpg", "conditions": {"session": "1"}},
       {"id": "p-Male_Adult_05-...", "kind": "still", "role": "probe", "identity": "Male_Adult_05",
        "source": "render", "conditions": {"body": "Male_Adult_17", "face": "renders", "head": "fitted",
                                           "ied": 24, "yaw": 30, "pitch": 0, "light": "side", "preset": "good 4K"}},
       {"id": "lfw", "kind": "pack", "path": ".../lfw.bin", "per_fold": [75, 75]},
       {"id": "run-1", "kind": "scene", "scene": "parade", "seconds": 60, ...}]}

A CSV manifest has one still per row: the columns ``id, kind, role, identity, source, path`` and one column per
condition. Paths are relative to the manifest's folder. :func:`load` reads a manifest and refuses a bad one with every
problem it finds (:class:`ManifestError`): unknown kinds, roles or conditions, values out of range, missing files,
duplicate ids, the same image (or the same render) both enrolled and probed (leakage), probes without any gallery.

* **Stills** (``kind: still``) are a photo (``source: image``; its conditions are free labels to group by) or a render
  (``source: render``; :data:`RENDER`: who is drawn, with whose face and head, through which view, how big, turned,
  nodded, lit, through which camera, occluded, with which renderer settings; drawn by :mod:`.evalrender`). An
  identity is the person a face shows: an avatar with its own face is that avatar; an avatar wearing textures made
  from renders of another (``face: renders``) shows that other avatar's identity, on its own head or on the head
  fitted to those renders (``head: fitted``, S3).
* **Packs** (``kind: pack``): an InsightFace verification pack (``lfw.bin``), its pairs embedded from the aligned crops;
  ``per_fold``: the first so many mated and non-mated pairs of each of its ten folds (F1's subset: 75 and 75). The
  folds hold disjoint people, so the intervals come from a bootstrap over folds.
* **Scenes** (``kind: scene``): a scripted run of the fair through the whole pipeline, scored per track
  (:mod:`.evalscene`).

**Caches.** Re-runs reuse what was computed: a still's aligned face per detector (``cache/crops.<detector>.npz``) and
its embedding per model (``cache/embeddings.<model>.npz``), each keyed by the item's content (:func:`item_key`: an
image file's SHA-256, or a render's every condition and the avatars' files), the detector's or model's id and file
SHA-256 and the code's versions (:data:`RENDER_VERSION`, :data:`ACQUIRE_VERSION`). A changed condition, model file or
avatar makes a new entry; nothing stale is used. The caches are saved as they grow, so a long run that stops resumes.
Runs are deterministic: renders have fixed seeds, the models are deterministic, the splits and bootstraps are
seeded.

**What is measured** on stills, per profile (:func:`score_stills`), over the probes and with the gallery enrolled as
the live gallery enrols photos (:class:`~.gallery.Gallery`, photos and pose templates):

* *verification* (every probe against every identity's gallery, raw scores): TAR at FAR, EER and d' with intervals
  over people (an impostor pair weighted by both its people);
* *open-set identification*: the identities with a gallery are split many times (``settings.splits``, seeded) into
  an enrolled half and a stranger half; every probe is a mated search when its identity is enrolled, else a
  non-mated one; FNIR and DIR at FPIR, rank-1 and CMC, pooled over the splits, with a bootstrap over people
  (identities without a gallery are always strangers; ``splits: 0``: every identity with a gallery enrolled);
* *the system's decision* on a single face, as live (:mod:`.decision`): scores normalised as the calibration says,
  turned into rarities in the face's bin, named when the best reaches the accept rarity of the target false-alarm
  rate per decision (no track offset: a still is one decision) and beats the runner-up by the margin: the share of
  known people's probes named rightly and wrongly, and of strangers' probes named, with the rate achieved against the
  one asked per face-size bin (:func:`~.metrics.achieved`, intervals over identity pairs);
* *failures to acquire*: no face found, or a face the live quality gate would not embed (those count as misses, and
  name nobody);
* all of it **per condition** (each condition that varies, and the research's bins of the measured eye distance),
  each with its count of probes and people.

Faces not found are misses in FNIR, never left out. Every report holds its counts.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import metrics

FORMAT = 1
KINDS = ("still", "pack", "scene", "video")
ROLES = ("gallery", "probe")
SOURCES = ("image", "render")
RENDER_VERSION = 1  # raised when rendering changes: renders keyed with another version are not reused
ACQUIRE_VERSION = 1  # raised when finding and aligning faces changes
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
LIGHTS = ("fair", "side", "warm", "top")
OCCLUDERS = ("none", "low", "side")
EXPRESSIONS = ("none", "smile", "grin", "tongue", "surprise", "scrunch", "roar")
PRESET_NAMES = ("ideal", "good 8K", "good 4K", "budget 1080p")
SHOTS = ("three-quarter", "fronts", "profiles", "front")
# the research's bins of the eye distance (docs/research/face_recognition_sota.md, section 4.2)
IED_BINS = (0.0, 10.0, 15.0, 22.0, 32.0, 45.0, 64.0, 90.0, math.inf)
TARGETS = (0.001, 0.01, 0.1)  # false-alarm rates per decision the system's operating points are measured at
SPLITS = 20
DEFAULT_OUT = "eval"

__all__ = ["ACQUIRE_VERSION", "IED_BINS", "RENDER", "RENDER_VERSION", "Condition", "Item", "Manifest",
           "ManifestError", "Store", "dump", "item_key", "load", "parse", "problems", "score_stills", "splits"]


# -- conditions --------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Condition:
    """A condition of a rendered still: its type, default and what it may be (``choices``, or ``low``-``high``)."""

    kind: type
    default: object
    choices: tuple | None = None
    low: float | None = None
    high: float | None = None
    help: str = ""

    def parse(self, value):
        """``value`` (from JSON or a CSV cell) as this condition's type; ValueError when it may not be."""
        if self.kind is bool:
            if isinstance(value, str):
                if value.strip().lower() not in ("true", "false", "1", "0", "yes", "no"):
                    raise ValueError(f"{value!r} is not true or false")
                value = value.strip().lower() in ("true", "1", "yes")
            elif not isinstance(value, (bool, int)):
                raise ValueError(f"{value!r} is not true or false")
            return bool(value)
        if self.kind in (int, float):
            if isinstance(value, bool):
                raise ValueError(f"{value!r} is not a number")
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{value!r} is not a number") from exc
            if not math.isfinite(number):
                raise ValueError(f"{value!r} is not a finite number")
            if self.kind is int:
                if number != int(number):
                    raise ValueError(f"{value!r} is not a whole number")
                number = int(number)
            if self.low is not None and number < self.low or self.high is not None and number > self.high:
                raise ValueError(f"{value!r} is outside {self.low:g}-{self.high:g}")
            if self.choices is not None and number not in self.choices:
                raise ValueError(f"{value!r} is not one of {', '.join(map(str, self.choices))}")
            return number
        if value is None:
            if self.default is None:
                return None
            raise ValueError("a value is needed")
        value = str(value).strip()
        if self.choices is not None and value not in self.choices:
            raise ValueError(f"{value!r} is not one of {', '.join(self.choices)}")
        return value


RENDER: dict[str, Condition] = {
    "body": Condition(str, None, help="the avatar drawn (default: the identity's own)"),
    "face": Condition(str, "own", ("own", "renders"),
                      help="own: the body's own textures; renders: textures made from the identity's renders"),
    "head": Condition(str, "own", ("own", "fitted"), help="fitted: the head fitted to the identity's renders (S3)"),
    "shots": Condition(str, "three-quarter", SHOTS, help="the identity's renders the face textures are made from"),
    "view": Condition(str, "window", ("window", "frame"),
                      help="window: the virtual sensor's window; frame: the 1280x720 view's head region"),
    "ied": Condition(float, 32.0, low=4.0, high=400.0,
                     help="pixels between the eyes of the face turned to the camera (the window's, or the frame's)"),
    "distance": Condition(float, 4.0, low=0.8, high=40.0, help="metres from the camera (windows; frames: from ied)"),
    "side": Condition(float, 0.0, low=-3.0, high=3.0, help="metres to the side of the camera's axis"),
    "yaw": Condition(float, 0.0, low=-90.0, high=90.0, help="degrees the head is turned (to its left: positive)"),
    "pitch": Condition(float, 0.0, low=-45.0, high=45.0, help="degrees the head is nodded (down: positive)"),
    "light": Condition(str, "fair", LIGHTS, help="the hall's light: fair (as built), side, warm, top"),
    "preset": Condition(str, "ideal", PRESET_NAMES, help="the camera model's preset"),
    "occluder": Condition(str, "none", OCCLUDERS, help="low: the lower face covered; side: one side up to the nose"),
    "expression": Condition(str, "none", EXPRESSIONS, help="a face made (characters.faces)"),
    "expression_weight": Condition(float, 0.6, low=0.0, high=1.0),
    "quality": Condition(str, "high", ("high", "low"), help="the renderer's shading (low: before the maps)"),
    "linear": Condition(bool, True, help="the renderer's linear light (false: the old shading on stored values)"),
    "detail": Condition(int, 2048, (1024, 2048), help="the heads' textures: 1024 only, or 2048 where big enough"),
    "seed": Condition(int, 0, low=0, high=2 ** 31 - 1, help="the camera model's noise"),
}
SCENE_KEYS = {"scene": str, "seconds": float, "fps": float, "seed": int, "people": list, "known": list,
              "windows": bool, "preset": str, "detector": str, "rate": float, "fpir": float}


# -- manifests ---------------------------------------------------------------------------------------------------------


class ManifestError(ValueError):
    """A manifest that cannot be used: ``problems`` lists everything wrong with it."""

    def __init__(self, source, found: list[str]):
        self.problems = list(found)
        shown = "\n  ".join(found[:30]) + (f"\n  ... and {len(found) - 30} more" if len(found) > 30 else "")
        super().__init__(f"{source}: {len(found)} problem(s):\n  {shown}")


@dataclass(frozen=True)
class Item:
    """One item of a manifest (see the module docs); a render's ``conditions`` hold every :data:`RENDER` condition
    (defaults filled in)."""

    id: str
    kind: str
    role: str | None = None
    identity: str | None = None
    source: str | None = None
    path: str | None = None
    conditions: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)  # a pack's or a scene's own fields
    labels: dict = field(default_factory=dict)  # free labels to group by (not drawn): a render's rung, say

    def file(self, folder) -> Path | None:
        return None if self.path is None else (Path(folder) / self.path)

    def groups(self) -> dict:
        """What the reports group this still by: its conditions and labels."""
        return {**self.conditions, **self.labels}

    def to_json(self) -> dict:
        out = {"id": self.id, "kind": self.kind}
        for key in ("role", "identity", "source", "path"):
            if getattr(self, key) is not None:
                out[key] = getattr(self, key)
        if self.conditions:
            out["conditions"] = dict(self.conditions)
        if self.labels:
            out["labels"] = dict(self.labels)
        out.update(self.extra)
        return out


@dataclass(frozen=True)
class Manifest:
    """A checked manifest (:func:`load`, :func:`parse`)."""

    name: str
    items: tuple
    folder: str = "."
    protocol: str = ""
    description: str = ""
    sides: dict = field(default_factory=dict)
    settings: dict = field(default_factory=dict)

    def stills(self, role: str | None = None) -> list[Item]:
        return [i for i in self.items if i.kind == "still" and (role is None or i.role == role)]

    def of_kind(self, kind: str) -> list[Item]:
        return [i for i in self.items if i.kind == kind]

    def identities(self, role: str | None = None) -> list[str]:
        return sorted({i.identity for i in self.stills(role)})

    def to_json(self) -> dict:
        return {"format": FORMAT, "name": self.name, "protocol": self.protocol, "description": self.description,
                "sides": self.sides, "settings": self.settings, "items": [i.to_json() for i in self.items]}


def dump(manifest: Manifest, path) -> Path:
    """Write ``manifest`` as JSON (renders with every condition, so the file says exactly what is drawn)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest.to_json(), indent=1))
    tmp.replace(path)
    return path


def load(path) -> Manifest:
    """The manifest in ``path`` (``.json``, or ``.csv`` of stills), checked (:class:`ManifestError` lists every
    problem)."""
    path = Path(path)
    if not path.is_file():
        raise ManifestError(path, [f"no such file: {path}"])
    if path.suffix.lower() == ".csv":
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        fixed = ("id", "kind", "role", "identity", "source", "path")
        items = []
        for row in rows:
            entry = {k: (row.get(k) or "").strip() or None for k in fixed}
            entry["kind"] = entry["kind"] or "still"
            entry["conditions"] = {k: v for k, v in row.items()
                                   if k not in fixed and k is not None and v not in (None, "")}
            items.append(entry)
        raw = {"format": FORMAT, "name": path.stem, "items": items}
    else:
        try:
            raw = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise ManifestError(path, [f"not JSON: {exc}"]) from exc
    return parse(raw, path.parent, source=path)


def parse(raw: dict, folder=".", source="manifest") -> Manifest:
    """A :class:`Manifest` from its JSON form ``raw`` (paths relative to ``folder``), or :class:`ManifestError`."""
    found = problems(raw, folder)
    if found:
        raise ManifestError(source, found)
    defaults = dict(raw.get("defaults") or {})
    items = []
    for entry in raw["items"]:
        kind = entry.get("kind", "still")
        conditions = dict(entry.get("conditions") or {})
        if kind == "still" and entry.get("source") == "render":
            merged = {**defaults, **conditions}
            conditions = {name: c.parse(merged[name]) if name in merged else c.default for name, c in RENDER.items()}
            conditions["body"] = conditions["body"] or entry["identity"]
        elif kind == "still":
            conditions = {str(k): v for k, v in conditions.items()}
            if "scale" in conditions:
                conditions["scale"] = float(conditions["scale"])
        own = ("id", "kind", "role", "identity", "source", "path", "conditions") + (("labels",) if kind == "still"
                                                                                   else ())
        extra = {k: v for k, v in entry.items() if k not in own}
        labels = {str(k): v for k, v in (entry.get("labels") or {}).items()} if kind == "still" else {}
        items.append(Item(str(entry["id"]), kind, entry.get("role"), entry.get("identity"), entry.get("source"),
                          entry.get("path"), conditions, extra, labels))
    return Manifest(str(raw.get("name") or "manifest"), tuple(items), str(folder), str(raw.get("protocol") or ""),
                    str(raw.get("description") or ""), dict(raw.get("sides") or {}), dict(raw.get("settings") or {}))


def problems(raw, folder=".") -> list[str]:
    """Everything wrong with a manifest in its JSON form (``[]``: none); see the module docs."""
    out = []
    if not isinstance(raw, dict):
        return ["a manifest is a JSON object with its items"]
    if int(raw.get("format", 0) or 0) != FORMAT:
        out.append(f"format {raw.get('format')!r} is not {FORMAT}")
    items = raw.get("items")
    if not isinstance(items, list) or not items:
        return out + ["no items"]
    defaults = raw.get("defaults") or {}
    if not isinstance(defaults, dict):
        out.append("defaults must be an object of conditions")
        defaults = {}
    for name, value in defaults.items():
        if name not in RENDER:
            out.append(f"defaults: unknown condition {name!r}")
        else:
            try:
                RENDER[name].parse(value)
            except ValueError as exc:
                out.append(f"defaults: {name}: {exc}")
    settings = raw.get("settings") or {}
    if not isinstance(settings, dict):
        out.append("settings must be an object")
    else:
        if "splits" in settings and (not isinstance(settings["splits"], int) or settings["splits"] < 0):
            out.append("settings.splits must be a whole number, 0 or more")
        for k, b in enumerate(settings.get("breakdowns") or []):
            if not isinstance(b, dict) or not isinstance(b.get("by"), list) or not b["by"] \
                    or not all(isinstance(v, str) for v in b["by"]) or not isinstance(b.get("where", {}), dict):
                out.append(f"settings.breakdowns[{k}]: an object with 'by' (a list of conditions or labels) and "
                           "optionally 'where' ({label: value}) and 'name'")
    seen: dict[str, int] = {}
    uses: dict[tuple, set] = {}  # what an image or a render is used as: leakage if both
    probes, gallery = set(), set()
    for k, entry in enumerate(items):
        where = f"item {k + 1}"
        if not isinstance(entry, dict):
            out.append(f"{where}: not an object")
            continue
        ident = entry.get("id")
        if not ident or not isinstance(ident, (str, int)):
            out.append(f"{where}: no id")
        else:
            where = f"item {ident!r}"
            if str(ident) in seen:
                out.append(f"{where}: the id is used twice (items {seen[str(ident)]} and {k + 1})")
            seen[str(ident)] = k + 1
        kind = entry.get("kind", "still")
        if kind not in KINDS:
            out.append(f"{where}: unknown kind {kind!r} (one of {', '.join(KINDS)})")
            continue
        if kind == "pack":
            out += _pack_problems(entry, folder, where)
            continue
        if kind == "scene":
            out += _scene_problems(entry, where)
            continue
        if kind == "video":
            path = entry.get("path")
            if not path or not (Path(folder) / str(path)).is_file():
                out.append(f"{where}: no such recording: {path}")
            if not (Path(folder) / str(entry.get("faces", "faces"))).is_dir():
                out.append(f"{where}: no faces folder (the enrolled people): {entry.get('faces', 'faces')}")
            continue
        role, identity, source = entry.get("role"), entry.get("identity"), entry.get("source")
        if role not in ROLES:
            out.append(f"{where}: role {role!r} is not gallery or probe")
        if not identity or not isinstance(identity, str):
            out.append(f"{where}: no identity (whose face it shows)")
        if source not in SOURCES:
            out.append(f"{where}: source {source!r} is not image or render")
            continue
        conditions = entry.get("conditions") or {}
        if not isinstance(conditions, dict):
            out.append(f"{where}: conditions must be an object")
            continue
        labels = entry.get("labels") or {}
        if not isinstance(labels, dict) or not all(isinstance(v, (str, int, float, bool)) for v in labels.values()):
            out.append(f"{where}: labels must be an object of plain values")
        elif set(labels) & set(conditions):
            out.append(f"{where}: {', '.join(sorted(set(labels) & set(conditions)))} both a label and a condition")
        if source == "image":
            path = entry.get("path")
            file = Path(folder) / str(path) if path else None
            if not path:
                out.append(f"{where}: an image needs a path")
            elif not file.is_file():
                out.append(f"{where}: no such image: {file}")
            elif file.suffix.lower() not in IMAGE_SUFFIXES:
                out.append(f"{where}: {file.name} is not an image file ({', '.join(IMAGE_SUFFIXES)})")
            else:
                scale = conditions.get("scale", 1.0)
                try:
                    scale = float(scale)
                    if not 0.02 <= scale <= 1.0:
                        raise ValueError
                except (TypeError, ValueError):
                    out.append(f"{where}: scale {conditions.get('scale')!r} is not a factor of 0.02-1 (the photo "
                               "shrunk)")
                    scale = 1.0
                if role == "gallery" and scale != 1.0:
                    out.append(f"{where}: the gallery's photos are enrolled as they are (no scale)")
                uses.setdefault(("image", str(file.resolve())), set()).add(role)
        else:
            merged = {**defaults, **conditions}
            parsed = {}
            for name, value in merged.items():
                if name not in RENDER:
                    out.append(f"{where}: unknown condition {name!r} (known: {', '.join(RENDER)})")
                    continue
                try:
                    parsed[name] = RENDER[name].parse(value)
                except ValueError as exc:
                    out.append(f"{where}: {name}: {exc}")
            body = parsed.get("body") or identity
            if parsed.get("face", "own") == "own" and body != identity:
                out.append(f"{where}: an avatar's own face is that avatar's ({body}), not {identity!r}: face 'renders'"
                           " puts another identity's face on it")
            if parsed.get("head", "own") == "fitted" and parsed.get("face", "own") != "renders":
                out.append(f"{where}: a fitted head wears the textures made on it: face must be 'renders'")
            full = {n: parsed.get(n, c.default) for n, c in RENDER.items()}
            full["body"] = body
            uses.setdefault(("render", identity, json.dumps(full, sort_keys=True, default=str)), set()).add(role)
        (probes if role == "probe" else gallery).add(identity)
    for key, roles in uses.items():
        if len(roles) > 1:
            what = key[1] if key[0] == "image" else f"a render of {key[1]}"
            out.append(f"{what} is both enrolled and probed (leakage: the gallery and the probes must be disjoint)")
    if probes and not gallery and any(e.get("kind", "still") == "still" for e in items if isinstance(e, dict)):
        out.append("probes but no gallery: nobody is enrolled")
    return out


def _pack_problems(entry, folder, where) -> list[str]:
    out = []
    path = entry.get("path")
    if not path or not (Path(folder) / str(path)).is_file():
        out.append(f"{where}: no such pack: {path}")
    per_fold = entry.get("per_fold")
    if per_fold is not None and (not isinstance(per_fold, list) or len(per_fold) != 2
                                 or not all(isinstance(v, int) and v > 0 for v in per_fold)):
        out.append(f"{where}: per_fold is [mated, non-mated] pairs per fold, whole numbers above 0")
    return out


def _scene_problems(entry, where) -> list[str]:
    out = []
    for key, kind in SCENE_KEYS.items():
        if key in entry and not isinstance(entry[key], kind if kind is not float else (int, float)):
            out.append(f"{where}: {key} must be a {kind.__name__}")
    for key in ("scene", "seconds", "fps", "people"):
        if key not in entry:
            out.append(f"{where}: a scene needs {key}")
    if isinstance(entry.get("seconds"), (int, float)) and entry["seconds"] <= 0:
        out.append(f"{where}: seconds must be above 0")
    if isinstance(entry.get("fps"), (int, float)) and entry["fps"] <= 0:
        out.append(f"{where}: fps must be above 0")
    for k, person in enumerate(entry.get("people") or []):
        if not isinstance(person, dict) or not person.get("identity"):
            out.append(f"{where}: person {k + 1} needs an identity")
            continue
        face = person.get("face", "own")
        if face not in ("own", "renders") or person.get("head", "own") not in ("own", "fitted"):
            out.append(f"{where}: person {k + 1}: face is own or renders, head own or fitted")
        elif face == "own" and person.get("body", person["identity"]) != person["identity"]:
            out.append(f"{where}: person {k + 1}: an avatar's own face is that avatar's")
    return out


# -- keys and caches ---------------------------------------------------------------------------------------------------


def file_digest(path, chunk: int = 1 << 20) -> str:
    """The SHA-256 of a file's contents."""
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def avatar_digest(library_folder, name: str) -> str:
    """What an avatar's prepared files are (their contents for the model and its description, the textures by name
    and size), so that a re-prepared avatar makes its renders anew."""
    folder = Path(library_folder) / "avatars"
    h = hashlib.sha256(name.encode())
    for path in sorted(folder.glob(f"{name}.*")):
        if path.suffix in (".json", ".npz"):
            h.update(path.name.encode() + bytes.fromhex(file_digest(path)))
        else:
            h.update(f"{path.name}:{path.stat().st_size}".encode())
    return h.hexdigest()[:24]


def item_key(item: Item, folder=".", avatars: dict | None = None) -> str:
    """The key a still's cached face and embeddings are kept under: an image's contents, or a render's identity,
    every condition, the avatars it draws (``avatars``: ``{name: avatar_digest}``) and :data:`RENDER_VERSION`."""
    if item.source == "image":
        scale = float(item.conditions.get("scale", 1.0))
        return f"image-{file_digest(item.file(folder))[:40]}" + ("" if scale == 1.0 else f"-x{scale:g}")
    names = sorted({item.identity, item.conditions.get("body") or item.identity})
    content = {"v": RENDER_VERSION, "identity": item.identity, "conditions": item.conditions,
               "avatars": {n: (avatars or {}).get(n, "") for n in names}}
    return "render-" + hashlib.sha256(json.dumps(content, sort_keys=True, default=str).encode()).hexdigest()[:40]


def detector_key(engine) -> str:
    """The key of what a profile's face detector found (its id, file and :data:`ACQUIRE_VERSION`)."""
    spec = engine.detector_spec
    return f"{spec.id}-{str(getattr(spec.file, 'sha256', ''))[:12]}-a{ACQUIRE_VERSION}"


def model_key(engine) -> str:
    """The key of a model's embeddings: its id, its file's SHA-256 and what ran it (embeddings of two models, or of
    one model's two files, are never mixed)."""
    spec = engine.embedder_spec
    return f"{spec.id}-{str(getattr(spec.file, 'sha256', ''))[:12]}-{getattr(engine, 'provider_name', '')}"


class Store:
    """Records (``{field: array or number}``) by key in one ``.npz`` file: loaded at start, saved atomically every
    ``every`` new records and on :meth:`save` (a long run that stops keeps what it did). Every record of a store has
    the same fields."""

    def __init__(self, path, every: int = 64):
        self.path = Path(path)
        self.every = int(every)
        self._index: dict[str, int] = {}
        self._fields: dict[str, np.ndarray] = {}
        self._new: list[tuple[str, dict]] = []
        if self.path.is_file():
            with np.load(self.path, allow_pickle=False) as data:
                keys = data["keys"].tolist()
                self._fields = {k: data[k] for k in data.files if k != "keys"}
            self._index = {k: i for i, k in enumerate(keys)}

    def __len__(self) -> int:
        return len(self._index) + len(self._new)

    def __contains__(self, key) -> bool:
        return key in self._index or any(k == key for k, _ in self._new)

    def get(self, key: str) -> dict | None:
        i = self._index.get(key)
        if i is not None:
            return {name: values[i] for name, values in self._fields.items()}
        for k, record in reversed(self._new):
            if k == key:
                return record
        return None

    def put(self, key: str, record: dict) -> None:
        self._new.append((key, {name: np.asarray(v) for name, v in record.items()}))
        if len(self._new) >= self.every:
            self.save()

    def save(self) -> None:
        if not self._new:
            return
        keys = [None] * len(self._index)
        for k, i in self._index.items():
            keys[i] = k
        names = sorted(self._new[0][1])
        fresh = {name: np.stack([r[name] for _, r in self._new]) for name in names}
        if self._fields:
            if sorted(self._fields) != names:
                raise ValueError(f"{self.path}: records with fields {names}, the store has {sorted(self._fields)}")
            fresh = {name: np.concatenate([self._fields[name], fresh[name]]) for name in names}
        keys += [k for k, _ in self._new]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp.npz")
        np.savez(tmp, keys=np.array(keys, str), **fresh)
        tmp.replace(self.path)
        self._fields, self._index, self._new = fresh, {k: i for i, k in enumerate(keys)}, []


# -- faces of the stills ---------------------------------------------------------------------------------------------


CROP_FIELDS = ("found", "crop", "size", "yaw", "ied", "sharpness", "outside", "score")


def acquire(image: np.ndarray, engine, where: str = "image", region=None, near=None) -> dict:
    """The face of a still as the live pipeline would find it, with ``engine``'s detector: in a photo the biggest face
    (``where="image"``); in a window the face nearest its middle, the window scaled to the head's canvas as live
    (``"window"``); in a frame the face in the head ``region`` ``(x0, y0, x1, y1)`` nearest ``near`` (the eyes'
    middle; ``"frame"``). Aligned from those pixels and measured: :data:`CROP_FIELDS`."""
    from .heads import WINDOW_CANVAS, face_size
    from .quality import assess

    faces = []
    if where == "window":
        side = max(image.shape[:2])
        faces = engine.detect(image, min_size=0.0, canvas=WINDOW_CANVAS if side > WINDOW_CANVAS else None)
        target = np.array(image.shape[1::-1], np.float64) / 2.0
    elif where == "frame":
        x0, y0, x1, y1 = (float(v) for v in region)
        h, w = image.shape[:2]
        x0, y0, x1, y1 = max(0.0, x0), max(0.0, y0), min(float(w), x1), min(float(h), y1)
        if x1 - x0 >= 8 and y1 - y0 >= 8:
            faces = engine.detect(image, min_size=0.0, region=(x0, y0, x1 - x0, y1 - y0))
        target = np.asarray(near if near is not None else ((x0 + x1) / 2, (y0 + y1) / 2), np.float64)
    else:
        faces = engine.detect(image, min_size=0.0)
        target = None
    if not faces:
        return {"found": False, "crop": np.zeros((112, 112, 3), np.uint8), "size": 0.0, "yaw": 0.0, "ied": 0.0,
                "sharpness": 0.0, "outside": 0.0, "score": 0.0}
    if target is None:
        face = max(faces, key=lambda f: f.size)
    else:
        face = min(faces, key=lambda f: float(np.linalg.norm(f.points[:2].mean(axis=0) - target)))
    aligned = engine.align(image, face)
    q = assess(face, aligned)
    return {"found": True, "crop": aligned.crop, "size": face_size(face), "yaw": q.yaw, "ied": q.ied,
            "sharpness": q.sharpness, "outside": q.outside, "score": float(face.score)}


@dataclass
class Stills:
    """The probes of a manifest as one profile sees them: per probe its item, identity, conditions, whether a face was
    found and passes the live gate, the face's measured size (eye distance turned to the camera) and yaw, its quality
    weight, its unit embedding (zeros when not found) and eDifFIQA-T's score (NaN without it)."""

    items: list
    identity: np.ndarray
    found: np.ndarray
    passed: np.ndarray
    size: np.ndarray
    yaw: np.ndarray
    weight: np.ndarray
    vectors: np.ndarray
    fiqa: np.ndarray
    truth: np.ndarray | None = None  # (n, 3): what a render's scene measured, yaw, pitch and eye distance (NaN: photos)

    def __len__(self) -> int:
        return len(self.items)

    def take(self, mask) -> Stills:
        mask = np.asarray(mask)
        index = np.flatnonzero(mask) if mask.dtype == bool else mask
        return Stills([self.items[i] for i in index], self.identity[index], self.found[index], self.passed[index],
                      self.size[index], self.yaw[index], self.weight[index], self.vectors[index], self.fiqa[index],
                      None if self.truth is None else self.truth[index])


def _quality(record: dict, fiqa):
    from .quality import Quality

    return Quality(float(record["ied"]), float(record["yaw"]), 0.0, 0.0, float(record["sharpness"]), 128.0, 50.0, 0.0,
                   float(record["outside"]), float(record["score"]), None if fiqa is None else float(fiqa))


def stills_of(items: list, crops: dict, embeddings: dict, dim: int, gate=None) -> Stills:
    """:class:`Stills` from the cached records (``crops``, ``embeddings``: ``{item id: record}``)."""
    from .evidence import QualityGate

    gate = gate or QualityGate()
    n = len(items)
    found, passed = np.zeros(n, bool), np.zeros(n, bool)
    size, yaw, weight, fiqa = np.zeros(n), np.zeros(n), np.zeros(n), np.full(n, np.nan)
    vectors = np.zeros((n, dim))
    truth = np.full((n, 3), np.nan)
    for k, item in enumerate(items):
        crop, emb = crops[item.id], embeddings.get(item.id)
        truth[k] = [float(crop.get(f"truth_{f}", np.nan)) for f in ("yaw", "pitch", "ied")]
        if not bool(crop["found"]):
            continue
        found[k] = True
        size[k], yaw[k] = float(crop["size"]), float(crop["yaw"])
        f = None if emb is None or not np.isfinite(float(emb["fiqa"])) else float(emb["fiqa"])
        fiqa[k] = np.nan if f is None else f
        q = _quality(crop, f)
        weight[k] = q.weight
        passed[k] = gate.check(q, size[k], float(crop["score"])) is None
        if emb is not None:
            vectors[k] = np.asarray(emb["vector"], np.float64)
    return Stills(list(items), np.array([i.identity for i in items], str), found, passed, size, yaw, weight, vectors,
                  fiqa, truth)


# -- scoring -----------------------------------------------------------------------------------------------------------


def splits(identities, count: int, seed: int = 0, share: float = 0.5) -> list[tuple[list, list]]:
    """``count`` random halvings of ``identities`` (sorted first, so the same names give the same splits):
    ``(enrolled, strangers)``, ``share`` of them enrolled; ``count`` 0: one split, every identity enrolled."""
    names = sorted(identities)
    if count <= 0:
        return [(names, [])]
    rng = np.random.default_rng(seed)
    k = max(1, round(share * len(names)))
    out = []
    for _ in range(count):
        order = rng.permutation(len(names))
        out.append((sorted(names[i] for i in order[:k]), sorted(names[i] for i in order[k:])))
    return out


def person_scores(vectors: np.ndarray, rows: np.ndarray, owners: np.ndarray, people: list,
                  cohort=None, kind: str = "none") -> np.ndarray:
    """``(probes, people)``: each probe's score against each person, the best of their gallery ``rows`` (photos and
    templates, owned by ``owners``), normalised as the decisions do (``kind``: ``"tnorm"`` with ``cohort``, a
    :class:`~.calibration.Cohort`; per probe a monotone map, so the best row is the best raw row); ``-inf`` where a
    person has no row."""
    raw = vectors @ rows.T
    out = np.full((len(vectors), len(people)), -np.inf)
    for j, person in enumerate(people):
        mine = owners == person
        if mine.any():
            out[:, j] = raw[:, mine].max(axis=1)
    if kind == "tnorm" and cohort is not None and len(vectors):
        mean, spread = cohort.stats(vectors, top=0)
        out = (out - mean[:, None]) / spread[:, None]
    elif kind not in ("none", "tnorm"):
        raise ValueError(f"normalisation {kind!r} is not measured here (none or tnorm)")
    return out


def _decisions(scores: np.ndarray, stills: Stills, calibration, people: list, target: float, margin: float):
    """Per probe ``(named, who)``: the single-face decision at ``target`` per decision (see the module docs)."""
    from .calibrate import rarities
    from .calibration import per_person

    usable = stills.found & stills.passed
    rarity = rarities(calibration, np.where(np.isfinite(scores), scores, -np.inf), stills.size, stills.yaw)
    order = np.argsort(-rarity, axis=1, kind="stable")
    rows = np.arange(len(scores))
    best = rarity[rows, order[:, 0]] if len(people) else np.zeros(len(scores))
    second = rarity[rows, order[:, 1]] if len(people) > 1 else np.zeros(len(scores))
    accept = -math.log10(per_person(target, max(1, len(people))))
    named = usable & (best >= accept - 1e-12) & (best - second >= margin - 1e-12)
    who = np.array([people[j] for j in order[:, 0]], str) if len(people) else np.full(len(scores), "", str)
    return named, who, best


def _size_bin(size: float) -> str:
    k = int(np.clip(np.searchsorted(IED_BINS, size, side="right") - 1, 0, len(IED_BINS) - 2))
    lo, hi = IED_BINS[k], IED_BINS[k + 1]
    return f"{lo:g}-{hi:g}" if math.isfinite(hi) else f"{lo:g}+"


def score_stills(stills: Stills, rows: np.ndarray, owners: np.ndarray, calibration=None, cohort=None,
                 count: int = SPLITS, seed: int = 0, draws: int = metrics.DRAWS, margin: float = 1.0,
                 targets=TARGETS, conditions: bool = True, level: float = metrics.LEVEL, breakdowns=None) -> dict:
    """The measures of one profile on ``stills`` (the probes) against a gallery (``rows``: unit embeddings of its
    photos and templates, ``owners``: whose each is), see the module docs; ``calibration`` (and ``cohort``): the
    profile's, for the system's decisions (``None``: those are left out); ``count`` splits, ``draws`` bootstrap
    draws; ``conditions``: also per condition (``breakdowns``: the manifest's, see :func:`_condition_groups`)."""
    enrollable = sorted(set(owners.tolist()))
    everyone = sorted(set(stills.identity.tolist()) | set(enrollable))
    unmasked = person_scores(stills.vectors, rows, owners, everyone)
    unmasked[~stills.found] = -np.inf  # (no face: nobody's score)
    raw_all = carriers_left_out(unmasked, everyone, stills)
    out = {"counts": {"probes": len(stills), "people": len(set(stills.identity.tolist())),
                      "enrollable": len(enrollable), "found": int(stills.found.sum()),
                      "passed": int((stills.found & stills.passed).sum()),
                      "worn": int((carriers(stills) != "").sum())},
           "acquisition": _acquisition(stills, draws, seed, level)}
    out["verification"] = _verification(stills, raw_all, everyone, enrollable, draws, seed, level)
    parts = splits(enrollable, count, seed)
    out["counts"]["splits"] = len(parts)
    out["counts"]["enrolled_per_split"] = len(parts[0][0]) if parts else 0
    out["open_set"] = _open_set(stills, raw_all, everyone, parts, draws, seed, level)
    if calibration is not None:
        out["system"] = _system(stills, rows, owners, calibration, cohort, parts, margin, targets, draws, seed, level)
    if out["counts"]["worn"]:
        out["carrier"] = _carrier_leak(stills, unmasked, everyone, rows, owners, calibration, cohort, margin, draws,
                                       seed, level)
    if conditions:
        out["conditions"] = {}
        for name, groups in _condition_groups(stills, breakdowns).items():
            out["conditions"][name] = {}
            for value, mask in groups.items():
                sub = stills.take(mask)
                entry = {"probes": int(mask.sum()), "people": len(set(sub.identity.tolist())),
                         "drawn": _drawn(sub), "acquisition": _acquisition(sub, draws, seed, level, short=True)}
                entry["verification"] = _verification(sub, raw_all[mask], everyone, enrollable, draws, seed, level,
                                                      short=True)
                entry["open_set"] = _open_set(sub, raw_all[mask], everyone, parts, draws, seed, level, short=True)
                if calibration is not None:
                    entry["system"] = _system(sub, rows, owners, calibration, cohort, parts, margin, (0.01,), draws,
                                              seed, level, short=True)
                out["conditions"][name][value] = entry
    return out


def carriers(stills: Stills) -> np.ndarray:
    """Per probe the avatar that wears its face when it is not the identity's own (``face: renders``), else ``""``."""
    return np.array([str(i.conditions.get("body") or "") if i.conditions.get("face") == "renders" else ""
                     for i in stills.items], str)


def carriers_left_out(scores: np.ndarray, people: list, stills: Stills) -> np.ndarray:
    """``scores`` (probes x ``people``) with each worn face's score against its carrier set to NaN: a person
    wearing a face is not the avatar under it, so that avatar is never enrolled for that probe (the carrier's head,
    hair and skin show through: how much is measured on its own, :func:`_carrier_leak`)."""
    out = np.array(scores, dtype=np.float64, copy=True)
    index = {p: j for j, p in enumerate(people)}
    for k, carrier in enumerate(carriers(stills)):
        if carrier and carrier in index:
            out[k, index[carrier]] = np.nan
    return out


def _carrier_leak(stills: Stills, unmasked, everyone, rows, owners, calibration, cohort, margin, draws, seed,
                  level) -> dict:
    """How much of the carrier a worn face shows, with every identity enrolled (no split): the share of worn faces
    scoring higher against their carrier's gallery than against their own identity's, and (with a calibration) named
    as the carrier at 1 % per decision; per head (own or fitted)."""
    worn = carriers(stills)
    out = {}
    for head in sorted({str(i.conditions.get("head")) for i, c in zip(stills.items, worn) if c}):
        mask = (worn != "") & np.array([str(i.conditions.get("head")) == head for i in stills.items])
        sub, scores = stills.take(mask), unmasked[mask]
        own = np.array([scores[k, everyone.index(w)] for k, w in enumerate(sub.identity)])
        theirs = np.array([scores[k, everyone.index(c)] if c in everyone else -np.inf
                           for k, c in enumerate(worn[mask])])
        entry = {"probes": int(mask.sum()), "people": len(set(sub.identity.tolist())),
                 "carrier_above_own": metrics.bootstrap_rate(sub.found & (theirs > own), sub.identity, draws, seed,
                                                             level)}
        if calibration is not None:
            enrolled = sorted(set(owners.tolist()))
            s = person_scores(sub.vectors, rows, owners, enrolled, cohort, calibration.normalisation)
            named, who, _ = _decisions(s, sub, calibration, enrolled, 0.01, margin)
            entry["named_as_carrier"] = metrics.bootstrap_rate(named & (who == worn[mask]), sub.identity, draws, seed,
                                                               level)
        out[f"head={head}"] = entry
    return out


def _condition_groups(stills: Stills, breakdowns=None) -> dict:
    """``{breakdown: {value: mask}}``. With ``breakdowns`` (a manifest's: ``[{"name", "where": {label: value},
    "by": [conditions or labels]}]``), the probes matching ``where`` grouped by the values of ``by`` together
    (``"a=1, b=2"``); without, every condition or label with more than one value among ``stills``, alone. Always
    also the research's bins of the measured eye distance (``ied_bin``)."""
    out = {}
    groups = [item.groups() for item in stills.items]
    if breakdowns:
        for b in breakdowns:
            where, by = dict(b.get("where") or {}), list(b.get("by") or [])
            keys = []
            for g in groups:
                matches = all(str(g.get(k)) == str(v) for k, v in where.items())
                keys.append(", ".join(f"{k}={g.get(k)}" for k in by) if matches else None)
            distinct = sorted({k for k in keys if k is not None}, key=_natural_key)
            out[str(b.get("name") or "+".join(by))] = {v: np.array([k == v for k in keys]) for v in distinct}
    else:
        for name in sorted({k for g in groups for k in g}):
            values = [str(g.get(name, "")) for g in groups]
            distinct = sorted(set(values), key=_natural)
            if len(distinct) > 1:
                out[name] = {v: np.array([x == v for x in values]) for v in distinct}
    bins = np.array([_size_bin(s) if f else "no face" for s, f in zip(stills.size, stills.found)])
    order = [_size_bin(v + 0.5) for v in IED_BINS[:-1]] + ["no face"]
    out["ied_bin"] = {b: bins == b for b in order if (bins == b).any()}
    return out


def _natural_key(text: str):
    """``"variant=own, ied=8.0"`` sorted part by part, numbers as numbers."""
    return tuple(_natural(part.split("=", 1)[-1]) for part in text.split(", "))


def _drawn(stills: Stills) -> dict:
    """What the scene measured of the probes' heads (medians): their yaw, pitch and eye distance, as drawn."""
    truth = stills.truth
    if truth is None or not np.isfinite(truth).any():
        return {}
    med = np.nanmedian(truth, axis=0)
    return {"yaw": round(float(med[0]), 2), "pitch": round(float(med[1]), 2), "ied": round(float(med[2]), 2)}


def _natural(value: str):
    try:
        return (0, float(value), "")
    except ValueError:
        return (1, 0.0, value)


def _acquisition(stills: Stills, draws, seed, level, short=False) -> dict:
    people = stills.identity
    found = metrics.bootstrap_rate(stills.found, people, draws, seed, level)
    usable = metrics.bootstrap_rate(stills.found & stills.passed, people, draws, seed, level)
    return {"found": _rate(found, short), "usable": _rate(usable, short)}


def _rate(r: dict, short=False) -> dict:
    if short:
        return {"rate": r["rate"], "interval": r["interval"], "n": r["n"]}
    return r


def _verification(stills: Stills, raw: np.ndarray, everyone, enrollable, draws, seed, level, short=False) -> dict:
    """Every probe whose identity has a gallery against every identity with a gallery: its own is a genuine pair, the
    others impostor pairs; a probe without a face scores -1 against everyone (a miss, never left out); a worn face
    is not compared with its carrier (NaN: :func:`carriers_left_out`)."""
    columns = [everyone.index(p) for p in enrollable]
    genuine, g_people, impostor, i_people = [], [], [], []
    for k in range(len(stills)):
        who = stills.identity[k]
        row = raw[k]
        for j, person in zip(columns, enrollable):
            if np.isnan(row[j]):
                continue
            score = float(row[j]) if np.isfinite(row[j]) else -1.0
            if person == who:
                genuine.append(score)
                g_people.append(who)
            else:
                impostor.append(score)
                i_people.append((who, person))
    if not genuine or not impostor:
        return {"genuine": len(genuine), "impostor": len(impostor)}
    report = metrics.pairs_report(genuine, impostor, g_people, i_people, draws=draws if not short else
                                  max(100, draws // 4), seed=seed, level=level)
    if short:
        keep = ("genuine", "impostor", "people", "d_prime", "d_prime_interval", "eer", "eer_interval")
        report = {k: report[k] for k in keep} | {"tar_at_far": {
            f: {"tar": v["tar"], "interval": v.get("interval")} for f, v in report["tar_at_far"].items()}}
    far, tar, _ = metrics.roc(genuine, impostor)
    report["curve"] = _sampled(far, tar, len(impostor), "tar")
    return report


CURVE_GRID = np.logspace(-6, 0, 31)


def _sampled(x: np.ndarray, y: np.ndarray, n: int, name: str) -> dict:
    """A curve's points at :data:`CURVE_GRID` rates of ``x`` (a non-decreasing rate along the curve: FAR or FPIR) that
    ``n`` comparisons resolve: ``y`` at the last point whose ``x`` is at most the grid's (counted, a step curve)."""
    grid = CURVE_GRID[CURVE_GRID * n >= 1.0 - 1e-9]
    index = np.searchsorted(x, grid + 1e-12, side="right") - 1
    return {"x": [float(g) for g in grid], name: [float(y[max(0, k)]) for k in index]}


def _open_set(stills: Stills, raw: np.ndarray, everyone, parts, draws, seed, level, short=False) -> dict:
    """FNIR at FPIR pooled over the splits (each probe once per split; the bootstrap draws people)."""
    blocks, mates, people = [], [], []
    for enrolled, _strangers in parts:
        columns = [everyone.index(p) for p in enrolled]
        block = raw[:, columns]
        block = np.where(np.isnan(block), -np.inf, block)  # (a carrier is no candidate)
        mate = np.array([enrolled.index(w) if w in enrolled else -1 for w in stills.identity.tolist()])
        width = len(enrolled)
        blocks.append(block)
        mates.append(mate)
        people.append(stills.identity)
        if any(b.shape[1] != width for b in blocks):
            raise ValueError("splits of different sizes cannot be pooled")
    if not blocks:
        return {}
    scores, mate, who = np.vstack(blocks), np.concatenate(mates), np.concatenate(people)
    report = metrics.open_set_report(scores, mate, who, draws=draws if not short else max(100, draws // 4),
                                     seed=seed, level=level, levels=(0.1, 0.01) if short else metrics.FPIR_LEVELS)
    if short:
        report = {k: report[k] for k in ("searches", "mated", "nonmated", "people", "rank1", "rank1_interval",
                                         "fnir_at_fpir")}
    report["splits"] = len(parts)
    if (mate < 0).any() and (mate >= 0).any():
        fpir, fnir, _ = metrics.open_set(scores, mate)
        report["curve"] = _sampled(fpir, fnir, int((mate < 0).sum()), "fnir")
    return report


def _system(stills: Stills, rows, owners, calibration, cohort, parts, margin, targets, draws, seed, level,
            short=False) -> dict:
    """The single-face decisions, pooled over the splits (see the module docs)."""
    out = {"calibration": {"model": calibration.model, "digest": calibration.digest,
                           "normalisation": calibration.normalisation, "path": calibration.path},
           "margin": margin, "targets": {}}
    for target in targets:
        known_named, known_right, known_people = [], [], []
        stranger_named, stranger_pairs, stranger_people, stranger_bins = [], [], [], []
        for enrolled, _ in parts:
            mask = np.isin(owners, enrolled)
            scores = person_scores(stills.vectors, rows[mask], owners[mask], enrolled, cohort,
                                   calibration.normalisation)
            scores = carriers_left_out(scores, enrolled, stills)
            named, who, _best = _decisions(scores, stills, calibration, enrolled, target, margin)
            is_known = np.isin(stills.identity, enrolled)
            known_named += named[is_known].tolist()
            known_right += (named & (who == stills.identity))[is_known].tolist()
            known_people += stills.identity[is_known].tolist()
            stranger_named += named[~is_known].tolist()
            stranger_people += stills.identity[~is_known].tolist()
            stranger_pairs += ["|".join(sorted((a, b))) for a, b in zip(stills.identity[~is_known], who[~is_known])]
            stranger_bins += [calibration.bin_label(s, y) if f else "no face"
                              for s, y, f in zip(stills.size[~is_known], stills.yaw[~is_known],
                                                 stills.found[~is_known])]
        known_named, known_right = np.array(known_named, bool), np.array(known_right, bool)
        entry = {"known": {"decisions": len(known_named), "people": len(set(known_people)),
                           "named_right": _rate(metrics.bootstrap_rate(known_right, known_people, draws, seed, level),
                                                short),
                           "named_wrong": _rate(metrics.bootstrap_rate(known_named & ~known_right, known_people,
                                                                       draws, seed, level), short)},
                 "strangers": {"decisions": len(stranger_named), "people": len(set(stranger_people)),
                               "named": _rate(metrics.bootstrap_rate(stranger_named, stranger_pairs, draws, seed,
                                                                     level), short)}}
        if not short and stranger_named:
            entry["strangers"]["achieved"] = metrics.achieved(stranger_named, target, stranger_bins, stranger_pairs,
                                                              draws=draws, seed=seed, level=level)
            entry["strangers"]["achieved_all"] = metrics.achieved(stranger_named, target, None, stranger_pairs,
                                                                  draws=draws, seed=seed, level=level)["all"]
        out["targets"][f"{target:g}"] = entry
    return out


# -- packs -------------------------------------------------------------------------------------------------------------


def pack_pairs(pack_path, per_fold=None, folds: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(pair indices, same, fold)`` of a verification pack: every pair, or the first ``per_fold`` = ``[mated,
    non-mated]`` of each fold (the folds are the pack's consecutive tenths)."""
    from .checks import read_pack

    _images, same = read_pack(pack_path)
    n = len(same)
    fold = np.minimum(np.arange(n) * folds // max(n, 1), folds - 1)
    if per_fold is None:
        return np.arange(n), same, fold
    keep = []
    for f in range(folds):
        block = np.flatnonzero(fold == f)
        keep += block[same[block]][: per_fold[0]].tolist() + block[~same[block]][: per_fold[1]].tolist()
    keep = np.array(sorted(keep))
    return keep, same[keep], fold[keep]


def score_pack(vectors_a: np.ndarray, vectors_b: np.ndarray, same: np.ndarray, fold: np.ndarray,
               draws: int = metrics.DRAWS, seed: int = 0, level: float = metrics.LEVEL) -> dict:
    """Verification measures of a pack's pairs (cosine of the two faces), with intervals from a bootstrap over its
    folds (disjoint people), and the benchmark's own 10-fold accuracy."""
    from .checks import verification

    scores = np.einsum("ij,ij->i", vectors_a, vectors_b)
    genuine, impostor = scores[same], scores[~same]
    g_fold, i_fold = fold[same].astype(str), fold[~same].astype(str)
    report = metrics.pairs_report(genuine, impostor, g_fold, i_fold, draws=draws, seed=seed, level=level)
    report["people"] = None  # (the bootstrap drew folds, disjoint in people)
    report["folds"] = len(set(fold.tolist()))
    report["accuracy"] = verification(scores, same)
    far, tar, _ = metrics.roc(genuine, impostor)
    report["curve"] = _sampled(far, tar, len(impostor), "tar")
    return report


# -- results -----------------------------------------------------------------------------------------------------------


def write_json(path, data) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, default=_plain))
    tmp.replace(path)
    return path


def _plain(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (set, tuple)):
        return list(value)
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _pct(value, interval=None) -> str:
    if value is None:
        return "-"
    text = f"{100 * value:.1f} %"
    if interval and interval[0] is not None:
        text += f" [{100 * interval[0]:.1f}-{100 * interval[1]:.1f}]"
    return text


def summary_lines(results: dict) -> list[str]:
    """A few lines per profile for the terminal: the counts and the main measures with their intervals."""
    lines = [f"{results.get('manifest')} ({results.get('protocol') or 'manifest'})"]
    for profile, r in (results.get("stills") or {}).items():
        c = r["counts"]
        lines.append(f"  {profile} ({r['model']}): {c['probes']} probes of {c['people']} people, "
                     f"{c['enrollable']} with a gallery, {c['splits']} splits; faces usable "
                     f"{_pct(r['acquisition']['usable']['rate'], r['acquisition']['usable']['interval'])}")
        v = r.get("verification") or {}
        if "d_prime" in v:
            tars = ", ".join(f"TAR@{f} {_pct(t['tar'], t.get('interval'))}" for f, t in v["tar_at_far"].items()
                             if float(f) <= 0.01)
            lines.append(f"    verification: d' {v['d_prime']:.2f}, EER {_pct(v['eer'], v.get('eer_interval'))}, "
                         f"{tars}")
        o = r.get("open_set") or {}
        if o:
            fn = ", ".join(f"FNIR@FPIR {f} {_pct(e['fnir'], e.get('interval'))}" for f, e in o["fnir_at_fpir"].items())
            lines.append(f"    open set: rank-1 {_pct(o.get('rank1'), o.get('rank1_interval'))}, {fn}")
        s = (r.get("system") or {}).get("targets", {}).get("0.01")
        if s:
            lines.append(f"    decisions at 1 % per decision: known named "
                         f"{_pct(s['known']['named_right']['rate'], s['known']['named_right']['interval'])}, "
                         f"wrongly {_pct(s['known']['named_wrong']['rate'])}; strangers named "
                         f"{_pct(s['strangers']['named']['rate'], s['strangers']['named']['interval'])}")
    for pack, by_profile in (results.get("packs") or {}).items():
        for profile, r in by_profile.items():
            tars = ", ".join(f"TAR@{f} {_pct(t['tar'], t.get('interval'))}" for f, t in r["tar_at_far"].items())
            lines.append(f"  {pack} {profile} ({r['model']}): {r['pairs']} pairs, accuracy "
                         f"{r['accuracy']['accuracy']:.2f} %, d' {r['d_prime']:.2f}, EER {_pct(r['eer'])}, {tars}")
    for profile, r in (results.get("scenes") or {}).items():
        score = r.get("score")
        if not score:
            continue
        s, acc = score["summary"], score["acceptance"]
        within = s["known"]["named_within"].get("2") or {}
        fpir = s["strangers"]["fpir"]
        lines.append(f"  {profile}: {score['runs']} runs, {s['segments']} tracks of {s['people']} people; known named "
                     f"within 2 s {_pct(within.get('rate'), within.get('interval'))} of {s['known']['eligible']}, "
                     f"strangers named {_pct(fpir.get('rate'), fpir.get('interval'))} of {s['strangers']['eligible']},"
                     f" wrong names {s['wrong_names']}; acceptance: {acc['verdict']}"
                     + (" (confident)" if acc["confident"] else " (not with confidence)"))
    for video, by_profile in (results.get("videos") or {}).items():
        if not by_profile:
            lines.append(f"  {video}: label its tracks, then run it again")
        for profile, r in by_profile.items():
            s, acc = r["summary"], r["acceptance"]
            within = s["known"]["named_within"].get("2") or {}
            fpir = s["strangers"]["fpir"]
            lines.append(f"  {video} {profile}: {r['labelled']} of {r['tracks']} tracks labelled; known named within "
                         f"2 s {_pct(within.get('rate'), within.get('interval'))} of {s['known']['eligible']}, "
                         f"strangers named {_pct(fpir.get('rate'), fpir.get('interval'))} of "
                         f"{s['strangers']['eligible']}, wrong names {s['wrong_names']}; acceptance: {acc['verdict']}")
    return lines


def environment() -> dict:
    """What the numbers were measured on (for the results' JSON)."""
    import platform

    return {"date": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "python": platform.python_version(),
            "machine": platform.machine(), "system": platform.system(), "cpus": os.cpu_count()}


# -- running -----------------------------------------------------------------------------------------------------------


class Run:
    """One ``bifold-faces eval`` run: manifests measured with each of ``engines`` (``{profile: FaceEngine}``), results
    and caches in ``out``. ``library``: the avatars' folder (renders); ``renderer``: a callable making the still
    renderer (default: :class:`.evalrender.StillRenderer`, made on first need); ``progress``: where messages go
    (``None``: nowhere)."""

    def __init__(self, engines: dict, out=DEFAULT_OUT, library=None, renderer=None, progress=print,
                 draws: int = metrics.DRAWS, seed: int = 0, models_folder=None):
        self.engines = dict(engines)
        self.out = Path(out)
        self.library = library
        self._renderer_factory = renderer
        self._renderer = None
        self.progress = progress or (lambda *_: None)
        self.draws, self.seed = int(draws), int(seed)
        self.models_folder = models_folder
        self._avatars: dict[str, str] = {}
        self._keys: dict[tuple, str] = {}

    # -- caches ----------------------------------------------------------------------------------------------------

    def _store(self, kind: str, key: str) -> Store:
        return Store(self.out / "cache" / f"{kind}.{key}.npz")

    def avatar_digests(self, names) -> dict:
        folder = self.library or os.environ.get("BIFOLD_CHARACTERS")
        for name in names:
            if name not in self._avatars:
                self._avatars[name] = avatar_digest(folder, name) if folder else ""
        return {n: self._avatars[n] for n in names}

    def key(self, manifest: Manifest, item: Item) -> str:
        """:func:`item_key`, worked out once per item and file state (a photo that changes is hashed again)."""
        stamp = None
        if item.source == "image":
            info = item.file(manifest.folder).stat()
            stamp = (info.st_size, info.st_mtime_ns)
        memo = (manifest.folder, item.id, item.path, stamp, json.dumps(item.conditions, sort_keys=True, default=str))
        cached = self._keys.get(memo)
        if cached is None:
            avatars = None
            if item.source == "render":
                avatars = self.avatar_digests(sorted({item.identity, item.conditions.get("body") or item.identity}))
            cached = self._keys[memo] = item_key(item, manifest.folder, avatars)
        return cached

    @property
    def renderer(self):
        if self._renderer is None:
            if self._renderer_factory is not None:
                self._renderer = self._renderer_factory()
            else:
                from .evalrender import StillRenderer

                self._renderer = StillRenderer(self.library, progress=self.progress)
        return self._renderer

    def close(self) -> None:
        if self._renderer is not None and hasattr(self._renderer, "close"):
            self._renderer.close()
        self._renderer = None

    # -- the steps ---------------------------------------------------------------------------------------------------

    def pixels(self, manifest: Manifest, item: Item) -> dict:
        """``{image, where, region, near, truth}`` of a still (see :func:`acquire`; ``truth``: what a render's scene
        measured of the head, empty for a photo)."""
        if item.source == "image":
            import cv2

            image = cv2.imread(str(item.file(manifest.folder)), cv2.IMREAD_COLOR)
            if image is None:
                raise OSError(f"cannot read {item.file(manifest.folder)}")
            scale = float(item.conditions.get("scale", 1.0))
            if scale != 1.0:  # (shrunk as a smaller sensor would see it: area averaging)
                h, w = image.shape[:2]
                size = (max(1, round(w * scale)), max(1, round(h * scale)))
                image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
            return {"image": image, "where": "image", "region": None, "near": None, "truth": {}}
        return self.renderer.still(item)

    def acquire(self, manifest: Manifest, items: list) -> dict:
        """The faces of ``items`` for every profile's detector, from the caches or found now: ``{detector key:
        {item id: record}}`` (:data:`CROP_FIELDS` and the scene's measured ``truth_yaw``, ``truth_pitch`` and
        ``truth_ied``, NaN for photos)."""
        detectors = {}
        for engine in self.engines.values():
            detectors.setdefault(detector_key(engine), engine)
        stores = {key: self._store("crops", key) for key in detectors}
        todo = [i for i in items if any(self.key(manifest, i) not in s for s in stores.values())]
        todo.sort(key=_render_order)  # (a person's face textures are made once and reused while they are drawn)
        if todo:
            self.progress(f"[eval] {manifest.name}: finding the faces of {len(todo)} of {len(items)} stills")
        started = time.perf_counter()
        for n, item in enumerate(todo):
            key = self.key(manifest, item)
            got = self.pixels(manifest, item)
            truth = got.get("truth") or {}
            for dkey, engine in detectors.items():
                if key not in stores[dkey]:
                    record = acquire(got["image"], engine, got["where"], got["region"], got["near"])
                    record.update({f"truth_{k}": float(truth.get(k, math.nan)) for k in ("yaw", "pitch", "ied")})
                    stores[dkey].put(key, record)
            if (n + 1) % 50 == 0:
                self.progress(f"[eval] {n + 1}/{len(todo)} stills ({time.perf_counter() - started:.0f} s, peak "
                              f"memory {metrics.peak_memory_mb():.0f} MB)")
        for store in stores.values():
            store.save()
        return {dkey: {i.id: stores[dkey].get(self.key(manifest, i)) for i in items} for dkey in detectors}

    def embed(self, manifest: Manifest, items: list, engine, crops: dict) -> dict:
        """The embeddings (and eDifFIQA-T's scores) of ``items``' faces with ``engine``: ``{item id: record}``."""
        store = self._store("embeddings", f"{model_key(engine)}.{detector_key(engine)}")
        todo = [i for i in items if bool(crops[i.id]["found"]) and self.key(manifest, i) not in store]
        if todo:
            self.progress(f"[eval] {manifest.name}: {engine.model_id}: embedding {len(todo)} faces")
        started = time.perf_counter()
        fiqa = getattr(engine, "fiqa", None)
        for start in range(0, len(todo), 16):
            part = todo[start:start + 16]
            stack = np.stack([np.asarray(crops[i.id]["crop"], np.uint8) for i in part])
            vectors = engine.embedder.embed(stack).vectors
            scores = fiqa.score(stack) if fiqa is not None else [math.nan] * len(part)
            for item, vector, score in zip(part, vectors, scores):
                store.put(self.key(manifest, item), {"vector": np.asarray(vector, np.float32), "fiqa": float(score)})
            if (start // 16) % 8 == 7:
                store.save()
                self.progress(f"[eval] {engine.model_id}: {start + len(part)}/{len(todo)} embedded "
                              f"({time.perf_counter() - started:.0f} s)")
        store.save()
        return {i.id: store.get(self.key(manifest, i)) for i in items if bool(crops[i.id]["found"])}

    def gallery_folder(self, manifest: Manifest) -> Path:
        """The manifest's gallery as a faces folder (``<identity>/<item id>.<ext>``): photos linked (or copied),
        renders drawn once (a key file beside each says what it shows)."""
        import cv2

        folder = self.out / "galleries" / manifest.name
        wanted = set()
        for item in manifest.stills("gallery"):
            person = folder / _folder_name(item.identity)
            person.mkdir(parents=True, exist_ok=True)
            key = self.key(manifest, item)
            if item.source == "image":
                source = item.file(manifest.folder)
                target = person / f"{_folder_name(item.id)}{source.suffix.lower()}"
                if not target.exists():
                    try:
                        os.symlink(source.resolve(), target)
                    except OSError:
                        import shutil

                        shutil.copyfile(source, target)
            else:
                target = person / f"{_folder_name(item.id)}.png"
                stamp = target.with_suffix(".key")
                if not (target.is_file() and stamp.is_file() and stamp.read_text() == key):
                    cv2.imwrite(str(target), self.pixels(manifest, item)["image"])
                    stamp.write_text(key)
            wanted.add(target)
        if folder.is_dir():  # (items no longer in the manifest leave its gallery)
            for path in folder.glob("*/*"):
                if path.suffix != ".key" and path not in wanted and not path.name.startswith("."):
                    path.unlink()
                    path.with_suffix(".key").unlink(missing_ok=True)
        return folder

    def gallery_rows(self, manifest: Manifest, engine) -> tuple[np.ndarray, np.ndarray, dict]:
        """``(rows, owners, report)``: the gallery enrolled by ``engine`` as live (photos and pose templates)."""
        from .calibrate import GalleryRows
        from .gallery import Gallery

        gallery = Gallery(self.gallery_folder(manifest), engine)
        gallery.refresh(force=True)
        if not gallery.people():
            raise ValueError(f"{manifest.name}: nobody could be enrolled with {engine.model_id}")
        rows = GalleryRows.of(gallery)
        report = {"people": len(gallery.people()), "photos": sum(gallery.people().values()), "rows": len(rows.rows),
                  "rejected": [str(p.relative_to(gallery.folder)) for p, s, _ in gallery.report() if s == "rejected"]}
        return rows.rows, rows.owners, report

    def calibration(self, engine):
        """The profile's calibration and cohort (``(None, None)`` without one made for its model)."""
        from .calibration import find_calibration, find_cohort

        calibration = find_calibration(engine.model_id, self.models_folder, getattr(engine, "threshold", None))
        if calibration.is_provisional:
            return None, None
        cohort = find_cohort(engine.model_id, self.models_folder, calibration) \
            if calibration.normalisation != "none" else None
        return calibration, cohort

    # -- a manifest ----------------------------------------------------------------------------------------------------

    def stills(self, manifest: Manifest, count: int | None = None, conditions: bool = True, done=None) -> dict:
        """Every profile's measures on ``manifest``'s stills (see the module docs): ``{profile: report}``; ``done``:
        called with the reports so far after each profile (a long run's results appear as they come)."""
        probes = manifest.stills("probe")
        if not probes:
            return {}
        crops = self.acquire(manifest, probes)
        count = int(manifest.settings.get("splits", SPLITS) if count is None else count)
        out = {}
        for profile, engine in self.engines.items():
            started = time.perf_counter()
            mine = crops[detector_key(engine)]
            embedded = self.embed(manifest, probes, engine, mine)
            stills = stills_of(probes, mine, embedded, engine.dim)
            rows, owners, enrolled = self.gallery_rows(manifest, engine)
            calibration, cohort = self.calibration(engine)
            report = score_stills(stills, rows, owners, calibration, cohort, count, self.seed, self.draws,
                                  conditions=conditions, breakdowns=manifest.settings.get("breakdowns"))
            report.update(profile=profile, model=engine.model_id, detector=engine.detector_spec.id,
                          gallery=enrolled, seconds=round(time.perf_counter() - started, 1))
            out[profile] = report
            self.progress(f"[eval] {manifest.name}: {profile} ({engine.model_id}) scored in "
                          f"{time.perf_counter() - started:.0f} s")
            if done is not None:
                done(dict(out))
        return out

    def packs(self, manifest: Manifest) -> dict:
        """Every profile's verification measures on ``manifest``'s packs: ``{pack id: {profile: report}}``."""
        from .checks import decode, read_pack

        out = {}
        for item in manifest.of_kind("pack"):
            path = item.file(manifest.folder)
            index, same, fold = pack_pairs(path, item.extra.get("per_fold"))
            images, _ = read_pack(path)
            digest = file_digest(path)[:24]
            out[item.id] = {}
            for profile, engine in self.engines.items():
                store = self._store("embeddings", f"{model_key(engine)}.pack-{digest}")
                wanted = sorted({2 * int(i) for i in index} | {2 * int(i) + 1 for i in index})
                todo = [w for w in wanted if f"{w}" not in store]
                if todo:
                    self.progress(f"[eval] {item.id}: {engine.model_id}: embedding {len(todo)} faces")
                started = time.perf_counter()
                for start in range(0, len(todo), 32):
                    part = todo[start:start + 32]
                    vectors = engine.embedder.embed(decode([images[w] for w in part])).vectors
                    for w, v in zip(part, vectors):
                        store.put(f"{w}", {"vector": np.asarray(v, np.float32)})
                    if (start // 32) % 10 == 9:
                        store.save()
                        self.progress(f"[eval] {engine.model_id}: {start + len(part)}/{len(todo)} "
                                      f"({time.perf_counter() - started:.0f} s)")
                store.save()
                a = np.stack([np.asarray(store.get(f"{2 * int(i)}")["vector"], np.float64) for i in index])
                b = np.stack([np.asarray(store.get(f"{2 * int(i) + 1}")["vector"], np.float64) for i in index])
                report = score_pack(a, b, same, fold, self.draws, self.seed)
                report.update(profile=profile, model=engine.model_id, pairs=len(index))
                out[item.id][profile] = report
        return out


def _folder_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_ ." else "_" for c in str(name)).strip() or "_"


def _render_order(item: Item) -> tuple:
    c = item.conditions
    return (item.source != "render", str(item.identity), str(c.get("body")), str(c.get("face")), str(c.get("head")),
            str(c.get("shots")), item.id)
