"""Microsoft Rocketbox avatars: what to download, how to fetch it and how to prepare a fast cache.

The `Microsoft Rocketbox <https://github.com/microsoft/Microsoft-Rocketbox>`_ library (MIT licence) has 115
rigged human avatars and hundreds of shared animation clips. This module knows which files we need
(:data:`AVATARS`, :data:`CLIPS`, :data:`ROLE_CLIPS`), downloads them (:func:`fetch`) and converts them once
(:func:`prepare`, slow, needs the FBX code) into a cache that :mod:`.library` loads in milliseconds:

* ``<cache>/avatars/<Name>.npz``: skeleton and skinned mesh arrays; ``<Name>.tex<k>.png``: RGBA textures;
  ``<Name>.json``: metadata (catalog entry, height, triangles, primitives and their textures);
* ``<cache>/clips/<name>.npz``: an in-place animation clip with its metadata.
"""

from __future__ import annotations

import difflib
import importlib
import json
import os
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from http.client import HTTPException
from pathlib import Path

import numpy as np

from .model import AnimationClip, CharacterModel, Skeleton, SkinnedPrimitive

BASE_URL = "https://raw.githubusercontent.com/microsoft/Microsoft-Rocketbox/master/"
CATALOG_PATH = Path(__file__).with_name("rocketbox_catalog.json")
FORMAT = 1
LICENSE = "MIT, Microsoft Rocketbox (https://github.com/microsoft/Microsoft-Rocketbox)"
FBX_MAGIC = b"Kaydara FBX Binary  \x00"


# -- manifest --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Avatar:
    """One avatar of the library: ``Assets/Avatars/<category>/<name>``."""

    name: str
    category: str  # "Adults", "Children" or "Professions"
    textures: tuple[str, ...]  # colour texture file names (incl. *_opacity_color.tga)

    @property
    def path(self) -> str:
        """``<category>/<name>``, as in the catalog."""
        return f"{self.category}/{self.name}"

    @property
    def fbx(self) -> str:
        return f"Assets/Avatars/{self.path}/Export/{self.name}.fbx"

    @property
    def files(self) -> list[str]:
        """Repository paths to download: the FBX and its colour textures."""
        return [self.fbx] + [f"Assets/Avatars/{self.path}/Textures/{t}" for t in self.textures]

    @property
    def sex(self) -> str:
        return "female" if "Female" in self.name else "male"

    @property
    def age_group(self) -> str:
        return "child" if self.category == "Children" else "adult"

    @property
    def outfit(self) -> str:
        if self.name.startswith("Business"):
            return "business"
        if self.name.startswith("Sports"):
            return "sportswear"
        if "Party" in self.name:
            return "party"
        return "uniform" if self.category == "Professions" else "casual"


@dataclass(frozen=True)
class Clip:
    """One animation clip: ``Assets/Animations/all_animations_max_motextr_<folder>/<name>.max.fbx``."""

    name: str
    folder: str  # "xy" (walks), "static" (in place) or "xyz" (turns)
    role: str

    @property
    def path(self) -> str:
        return f"Assets/Animations/all_animations_max_motextr_{self.folder}/{self.name}.max.fbx"

    @property
    def sex(self) -> str:
        return "female" if self.name.startswith("f_") else "male"


# Per avatar the colour textures in its Textures/ folder: "x" means x_color.tga, names containing "color" are kept.
# (Professions/Female_Party_0x only hold a facial FBX; the real avatars are Adults/Female_Party_0x.)
_AVATAR_MANIFEST = {
    "Adults": """
Female_Adult_01: f001_body f001_head f001_opacity
Female_Adult_02: f002_body f002_head f002_opacity
Female_Adult_03: f003_body f003_head f003_opacity
Female_Adult_04: f004_body f004_head f004_opacity
Female_Adult_05: f005_body f005_head f005_opacity
Female_Adult_06: f202_body f202_head f202_opacity
Female_Adult_07: f007_body f007_head f007_opacity
Female_Adult_08: f008_body f008_head f008_opacity
Female_Adult_09: f009_body f009_head f009_opacity
Female_Adult_10: f201_body f201_head f201_opacity
Female_Adult_11: f011_body f011_head f011_opacity
Female_Adult_12: f012_body f012_head f012_opacity
Female_Adult_13: f019_body f019_head f019_opacity
Female_Adult_14: f017_body f017_head f017_opacity
Female_Adult_15: f018_body f018_head f018_opacity
Female_Adult_16: f204
Female_Adult_17: f006_body f006_head f006_opacity
Female_Party_01: f010_body f010_head f010_opacity
Female_Party_02: f022_body f022_head f022_opacity
Male_Adult_01: m002_body m002_head m002_opacity
Male_Adult_02: m003_body m003_head m003_opacity
Male_Adult_03: m004_body m004_head m004_opacity
Male_Adult_04: m006_body m006_head m006_opacity
Male_Adult_05: m009_body m009_head m009_opacity
Male_Adult_06: m011_body m011_head m011_opacity
Male_Adult_07: m013_body m013_head m013_opacity
Male_Adult_08: m014_body m014_head m014_opacity
Male_Adult_09: m017_body m017_head m017_opacity
Male_Adult_10: m024_body m024_head m024_opacity
Male_Adult_11: m001_body m001_head
Male_Adult_12: m007_body m007_head
Male_Adult_13: m010_body m010_head
Male_Adult_14: m012_body m012_head
Male_Adult_15: m204_body m204_head
Male_Adult_16: m019_body m019_head
Male_Adult_17: m022_body m022_head
Male_Adult_18: m023_body m023_head
Male_Adult_19: m250_body m250_head m250_keffiyeh
Male_Adult_20: m027_body m027_head
Male_Adult_21: m251_body m251_head m251_keffiyeh
""",
    "Children": """
Female_Child_01: cf001_body cf001_head
Female_Child_02: cf011_body cf011_head
Male_Child_01: cm001_body cm001_head
Male_Child_02: cm012_body cm012_head
""",
    "Professions": """
Business_Female_01: f014_body f014_head f014_opacity
Business_Female_02: f015_body f015_glasses_opacity f015_head f015_opacity
Business_Female_03: f016_body f016_head f016_opacity
Business_Female_04: f020_body f020_head f020_opacity
Business_Male_01: m005_body m005_head m005_opacity
Business_Male_02: m008_body m008_head m008_opacity
Business_Male_03: m020_body m020_head m020_opacity
Business_Male_04: m015_body m015_head
Business_Male_05: m016_body m016_head
Business_Male_06: m025_body m025_head
Business_Male_07: m018_body m018_head
Chef_Female_01: f110_body f110_head_color_a
Construction_Female_01: f101_body f101_head f101_helmet
Construction_Male_01: m111_body m111_head m111_helmet m111_opacity m111_tools
Construction_Male_02: m101_body m101_head m101_helmet
Construction_Male_03: m102_body m102_head m102_helmet m102_tools
Construction_Male_04: m103_body m103_head m103_helmet m103_tools
Construction_Male_05: m104_body m104_head m104_helmet
Construction_Male_06: m105_body m105_head m105_helmet m105_tools
Construction_Male_07: m107_body m107_head m107_helmet
Construction_Male_08: m108_body m108_head m108_helmet
Delivery_Male_01: m112_body m112_head
Fire_Female_01: m131_body m131_flashlight m131_head m131_helmet m131_mask m131_opacity
Fire_Female_02: m133_body m133_flashlight m133_head m133_helmet m133_mask m133_opacity
Fire_Female_03: m137_body m137_flashlight m137_head m137_helmet m137_mask m137_opacity
Fire_Male_01: m138_body m138_head m138_helmet
Fire_Male_02: m132_body m132_head m132_helmet m132_opacity
Fire_Male_03: m134_body m134_head m134_helmet
Fire_Male_04: m135_body m135_flashlight m135_head m135_helmet m135_mask m135_opacity
Fire_Male_05: m136_body m136_head m136_helmet m136_opacity
Fire_Male_06: m130_body m130_head m130_helmet
Fire_Male_07: m139_body m139_flashlight m139_head m139_helmet m139_mask m139_opacity
Gardener_Male_01: m106_body m106_head m106_opacity
Medical_Female_01: f152_body f152_head f152_opacity
Medical_Female_02: f153_body f153_head f153_opacity f153_stetoskop
Medical_Female_03: f154_body_color_blue f154_head f154_opacity
Medical_Male_01: m152_body m152_head m152_opacity
Medical_Male_02: m154_body_color_blue m154_head m154_opacity
Medical_Male_03: m153_body m153_head m153_stetoskop
Medical_Male_04: m162_body m162_head m162_protection
Medical_Male_05: m163_body m163_head m163_protection
Military_Female_01: sf003_body_color_acu sf003_equipment_color_acu sf003_head sf003_helmet_color_acu
Military_Female_02: sf004_body_color_acu sf004_equipment_color_acu sf004_head sf004_helmet_color_acu
Military_Male_01: sm002_body_color_acu sm002_combat_knife sm002_equipment_color_acu sm002_head sm002_helmet_color_acu
Military_Male_02: sm024_body_color_acu sm024_head_color_acu
Military_Male_03: sm004_body_color_acu sm004_equipment_color_acu sm004_head sm004_helmet_color_acu sm004_pistol
Military_Male_04: sm005_body_color_acu sm005_equipment_color_acu sm005_head sm005_helmet_color_acu
Military_Male_05: sm022_body_color_acu sm022_head_color_acu
Military_Male_06: sm021_body_color_acu sm021_head_color_acu
Pilot_Female_01: f168_body f168_hat f168_head
Pilot_Female_02: f169_body f169_hat f169_head
Pilot_Male_01: m150_body m150_head
Pilot_Male_02: m151_body m151_head
Pilot_Male_03: m167_body m167_head
Police_Female_01: m157_body m157_equipment m157_head m157_helmet m157_machinegun m157_opacity m157_pistol m157_shotgun
Police_Male_01: m220_body m220_equipment m220_head m220_pistol
Police_Male_02: m158_body m158_equipment m158_head m158_helmet m158_opacity m158_pistol
Police_Male_03: m156_body m156_equipment m156_head m156_pistol m172_body m172_equipment m172_head m172_pistol
Police_Male_04: m173_body m173_equipment m173_head m173_pistol
Police_Male_05: m169_body m169_head
Police_Male_06: m175_body m175_head
Police_Male_07: m176_body m176_head
Security_Female_01: f170_body f170_hat f170_head f170_pistol
Security_Male_01: m171_body m171_equipment m171_head
Sports_Female_01: f021_body f021_head
Sports_Female_02: f013_body f013_head f013_opacity
Sports_Male_01: m021_body m021_head
Sports_Male_02: m301_body m301_head
Sports_Male_03: m300_body m300_head m300_opacity
Sports_Male_04: m026_body m026_head
Wood_Male_01: m110_body m110_head
""",
}

# role -> (folder, clip names without the m_/f_ prefix); every clip exists for men (m_) and women (f_).
_CLIP_MANIFEST = {
    "walk": ("xy", "walk_neutral_01 walk_neutral_02 walk_neutral_03"),
    "walk_slow": ("xy", "walk_slow_01 walk_slow_02"),
    "stroll": ("xy", "walk_stroll_01 walk_stroll_02"),
    "idle": ("static", " ".join(f"idle_neutral_{i:02d}" for i in range(1, 10))),
    "look_around": ("static", "idle_look_around_01 idle_look_around_02 idle_look_around_03"),
    "talk": ("static", "gestic_talk_neutral_01 gestic_talk_neutral_02 gestic_talk_neutral_03 gestic_talk_relaxed_01 "
                       + "gestic_talk_relaxed_02"),
    "listen": ("static", "gestic_listen_neutral_01 gestic_listen_neutral_02 gestic_listen_neutral_03 "
                         + " ".join(f"gestic_listen_accept_{i:02d}" for i in range(1, 6))),
    "phone": ("static", "cell_phone_textmessage"),
    "turn_left_90": ("xyz", "turn_left_90"),
    "turn_right_90": ("xyz", "turn_right_90"),
    "turn_left_180": ("xyz", "turn_left_180"),
    "turn_right_180": ("xyz", "turn_right_180"),
}


def _parse_avatars() -> dict[str, Avatar]:
    avatars = {}
    for category, text in _AVATAR_MANIFEST.items():
        for line in text.strip().splitlines():
            name, tokens = line.split(":")
            textures = tuple(t + ".tga" if "color" in t else t + "_color.tga" for t in tokens.split())
            avatars[name] = Avatar(name, category, textures)
    return avatars


def _parse_clips() -> dict[str, Clip]:
    return {f"{sex}_{name}": Clip(f"{sex}_{name}", folder, role)
            for role, (folder, names) in _CLIP_MANIFEST.items() for name in names.split() for sex in "mf"}


AVATARS: dict[str, Avatar] = _parse_avatars()
CLIPS: dict[str, Clip] = _parse_clips()
ROLE_CLIPS: dict[str, tuple[str, ...]] = {role: tuple(c.name for c in CLIPS.values() if c.role == role)
                                          for role in _CLIP_MANIFEST}

# A quick start: women and men, a girl and a boy, older people, several ethnicities, mostly conference outfits.
SMALL_SET = ("Business_Female_01", "Business_Male_02", "Male_Adult_05", "Female_Child_01", "Male_Child_02",
             "Business_Female_02", "Male_Adult_12", "Female_Adult_03", "Business_Male_05", "Female_Adult_06",
             "Female_Adult_15", "Male_Adult_21")
AVATAR_SETS = ("conference", "all", "small")
CLIP_SETS = ("all", "basic")


def load_catalog(path=None) -> dict[str, dict]:
    """The avatar classification (sex, age group, skin tone, outfit, conference fit); ``{}`` if there is none."""
    path = Path(path) if path is not None else CATALOG_PATH
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return {}
    key = (str(path), mtime)
    if _catalog_cache.get("key") != key:
        try:
            catalog = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"cannot read the avatar catalog {path}: {exc}") from exc
        if not isinstance(catalog, dict):
            raise TypeError(f"the avatar catalog {path} must map avatar names to entries")
        _catalog_cache.update(key=key, value={k: v for k, v in catalog.items() if isinstance(v, dict)})
    return _catalog_cache["value"]


_catalog_cache: dict = {}


def avatar_metadata(name: str, catalog: dict | None = None) -> dict:
    """Facts about an avatar from its name, overridden by its catalog entry when there is one."""
    catalog = load_catalog() if catalog is None else catalog
    avatar = AVATARS.get(name)
    meta = {"name": name}
    if avatar is not None:
        meta.update(path=avatar.path, category=avatar.category, sex=avatar.sex, age_group=avatar.age_group,
                    outfit=avatar.outfit)
    meta.update(catalog.get(name, {}))
    return meta


def avatar_set(name: str, catalog: dict | None = None) -> list[str]:
    """Avatar names of a named set: ``"conference"``, ``"all"`` or ``"small"``.

    ``conference`` uses the catalog (``conference_fit >= 2``) when it exists, else the Adults, the Children
    and the Business people.
    """
    if name == "all":
        return list(AVATARS)
    if name == "small":
        return list(SMALL_SET)
    if name == "conference":
        catalog = load_catalog() if catalog is None else catalog
        if catalog:
            return [n for n in AVATARS if float(catalog.get(n, {}).get("conference_fit", 0) or 0) >= 2]
        return [n for n, a in AVATARS.items() if a.category != "Professions" or n.startswith("Business_")]
    raise ValueError(f"unknown avatar set {name!r}; choose one of {', '.join(AVATAR_SETS)}")


def clip_set(name: str) -> list[str]:
    """Clip names of a named set: ``"all"`` or ``"basic"`` (the first clip of every role, for men and women)."""
    if name == "all":
        return list(CLIPS)
    if name == "basic":
        return [n for names in ROLE_CLIPS.values() for n in names[:2]]  # m_ and f_ of the first variant
    raise ValueError(f"unknown clip set {name!r}; choose one of {', '.join(CLIP_SETS)}")


def check_names(names, known, kind: str) -> list[str]:
    """``names`` as a list; ``ValueError`` (with suggestions) for the first unknown one."""
    names = list(names)
    for name in names:
        if name not in known:
            close = difflib.get_close_matches(name, list(known), n=3)
            hint = f"; did you mean {' or '.join(close)}?" if close else ""
            raise ValueError(f"unknown {kind} {name!r}{hint}")
    return names


def files_for(avatars=(), clips=()) -> list[str]:
    """Repository paths needed for these avatars and clips (in a stable order, no duplicates)."""
    paths = [p for n in check_names(avatars, AVATARS, "avatar") for p in AVATARS[n].files]
    paths += [CLIPS[n].path for n in check_names(clips, CLIPS, "clip")]
    return list(dict.fromkeys(paths))


def estimate_size(path: str) -> int:
    """Rough download size of a repository file in bytes (textures are 2048x2048 TGA files)."""
    lower = path.lower()
    if lower.endswith(".tga"):
        return 16_777_260 if "opacity" in lower else 12_582_956
    if "/animations/" in lower:
        return 3_500_000 if "_static/" in lower else 1_200_000
    return 550_000


def url_for(path: str, base_url: str = BASE_URL) -> str:
    return base_url.rstrip("/") + "/" + urllib.parse.quote(path)


# -- download --------------------------------------------------------------------------------------


class FetchError(RuntimeError):
    """Some files could not be downloaded; ``report.failed`` lists them (the rest was downloaded)."""

    def __init__(self, report: FetchReport):
        self.report = report
        lines = [f"  {path}: {reason}" for path, reason in report.failed[:5]]
        more = f"\n  ... and {len(report.failed) - 5} more" if len(report.failed) > 5 else ""
        super().__init__(f"{len(report.failed)} of {len(report.files)} files could not be downloaded:\n"
                         + "\n".join(lines) + more + "\nRun the same command again to resume.")


@dataclass
class FetchReport:
    files: list[Path] = field(default_factory=list)  # every requested file (local paths)
    downloaded: int = 0
    skipped: int = 0  # already complete
    bytes: int = 0  # downloaded bytes
    failed: list[tuple[str, str]] = field(default_factory=list)  # (repository path, reason)


class _Permanent(Exception):
    """A download error that retrying cannot fix."""


def _default_opener():
    context = ssl.create_default_context()
    if not os.environ.get("SSL_CERT_FILE"):
        try:  # e.g. python.org builds on macOS have no system certificates
            import certifi

            context.load_verify_locations(certifi.where())
        except (ImportError, OSError):
            pass

    def opener(request, timeout=60.0):
        return urllib.request.urlopen(request, timeout=timeout, context=context)

    return opener


def _download(url: str, target: Path, opener, retries: int, backoff: float, timeout: float) -> int:
    """Download ``url`` to ``target`` via ``target.part`` (resumed with a Range request); returns the size."""
    part = target.with_name(target.name + ".part")
    error: Exception | None = None
    for attempt in range(retries + 1):
        if attempt:
            time.sleep(backoff * 2 ** (attempt - 1))
        try:
            offset = part.stat().st_size if part.exists() else 0
            headers = {"User-Agent": "bifold-demo"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            with opener(urllib.request.Request(url, headers=headers), timeout=timeout) as response:
                status = getattr(response, "status", None) or response.getcode()
                if offset and status != 206:
                    offset = 0  # the server sent the whole file
                length = response.headers.get("Content-Length")
                expected = offset + int(length) if length is not None else None
                with open(part, "ab" if offset else "wb") as out:
                    while block := response.read(1 << 20):
                        out.write(block)
            size = part.stat().st_size
            if expected is not None and size != expected:
                raise OSError(f"incomplete download ({size} of {expected} bytes)")
            if target.suffix.lower() == ".fbx":
                with open(part, "rb") as f:
                    head = f.read(len(FBX_MAGIC))
                if head != FBX_MAGIC:
                    part.unlink()
                    what = "a Git LFS pointer" if head.startswith(b"version http") else "something else"
                    raise _Permanent(f"not a binary FBX file (the server sent {what})")
            os.replace(part, target)
            return size
        except urllib.error.HTTPError as exc:
            if exc.code == 416:  # stale .part: start over
                part.unlink(missing_ok=True)
            elif exc.code in (408, 429) or exc.code >= 500:
                pass
            else:
                raise _Permanent(f"HTTP {exc.code} {exc.reason} for {url}") from None
            error = exc
        except (urllib.error.URLError, HTTPException, OSError, ValueError) as exc:
            if isinstance(getattr(exc, "reason", None), ssl.SSLCertVerificationError):
                raise _Permanent(f"TLS certificate check failed ({exc.reason}); install certificates for this "
                                 f"Python (macOS: run 'Install Certificates.command'; or: pip install certifi)")
            error = exc
    if isinstance(error, urllib.error.HTTPError):
        reason = f"HTTP {error.code} {error.reason}"
    else:
        reason = str(getattr(error, "reason", None) or error) or type(error).__name__
    raise RuntimeError(f"{reason} (after {retries + 1} attempts)")


def fetch(dest, avatars, clips, jobs: int = 8, progress=None, *, retries: int = 3, opener=None,
          base_url: str = BASE_URL, backoff: float = 1.0, timeout: float = 60.0) -> FetchReport:
    """Download the files of ``avatars`` and ``clips`` (names; ``None`` = all) into ``dest`` (repository layout).

    Files that are already there are skipped, partial downloads (``*.part``) are resumed, each file appears
    under its final name only when complete, and ``jobs`` files download in parallel with ``retries``
    retries each. ``progress(done, total, path, status)`` is called after every file. Raises
    :class:`FetchError` after the others finished when some files failed. ``opener(request, timeout=...)``
    replaces :func:`urllib.request.urlopen` (for tests).
    """
    if jobs < 1 or retries < 0:
        raise ValueError("jobs must be at least 1 and retries at least 0")
    dest = Path(dest).expanduser()
    paths = files_for(AVATARS if avatars is None else avatars, CLIPS if clips is None else clips)
    report = FetchReport(files=[dest / p for p in paths])
    opener = opener or _default_opener()
    todo = []
    for path, target in zip(paths, report.files):
        if target.is_file():
            report.skipped += 1
            if progress:
                progress(report.skipped, len(paths), path, "already there")
        else:
            todo.append((path, target))
    done = report.skipped
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {}
        for path, target in todo:
            target.parent.mkdir(parents=True, exist_ok=True)
            futures[pool.submit(_download, url_for(path, base_url), target, opener, retries, backoff,
                                timeout)] = path
        for future in as_completed(futures):
            path, done = futures[future], done + 1
            try:
                size = future.result()
            except (_Permanent, RuntimeError) as exc:
                report.failed.append((path, str(exc)))
                status = f"failed: {exc}"
            else:
                report.downloaded += 1
                report.bytes += size
                status = f"downloaded {size / 1e6:.1f} MB"
            if progress:
                progress(done, len(paths), path, status)
    if report.failed:
        report.failed.sort()
        raise FetchError(report)
    return report


# -- prepared cache --------------------------------------------------------------------------------


def _write_atomic(path: Path, write) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        write(f)
    os.replace(tmp, path)


def _json(data, **kw) -> str:
    return json.dumps(data, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o), **kw)


def _write_json(path: Path, data: dict) -> None:
    _write_atomic(path, lambda f: f.write(_json(data, indent=1, sort_keys=True).encode("utf-8")))


def save_character(model: CharacterModel, cache_dir, metadata: dict | None = None) -> Path:
    """Write ``model`` into ``<cache_dir>/avatars/`` (arrays, PNG textures, then the JSON); returns the JSON path."""
    import cv2

    folder = Path(cache_dir).expanduser() / "avatars"
    folder.mkdir(parents=True, exist_ok=True)
    name = model.name
    arrays = {"names": np.array(model.skeleton.names), "parents": model.skeleton.parents,
              "rest_local": model.skeleton.rest_local, "inverse_bind": np.asarray(model.inverse_bind)}
    textures: dict[int, str] = {}
    primitives = []
    for i, p in enumerate(model.primitives):
        for key in ("positions", "normals", "uvs", "indices", "joints", "weights"):
            arrays[f"p{i}_{key}"] = np.asarray(getattr(p, key))
        file = None
        if isinstance(p.texture, np.ndarray):
            if id(p.texture) not in textures:
                file = f"{name}.tex{len(textures)}.png"
                rgba = p.texture if p.texture.ndim == 3 else p.texture[..., None]
                bgra = cv2.cvtColor(rgba, {1: cv2.COLOR_GRAY2BGRA, 3: cv2.COLOR_RGB2BGRA,
                                           4: cv2.COLOR_RGBA2BGRA}[rgba.shape[2]])
                ok, png = cv2.imencode(".png", bgra, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                if not ok:
                    raise ValueError(f"cannot encode the texture of {name}/{p.name}")
                _write_atomic(folder / file, lambda f, data=png: f.write(data.tobytes()))
                textures[id(p.texture)] = file
            file = textures[id(p.texture)]
        primitives.append({"name": p.name, "texture": file, "vertices": len(p.positions),
                           "triangles": len(p.indices)})
    _write_atomic(folder / f"{name}.npz", lambda f: np.savez(f, **arrays))
    meta = {**model.metadata, **(metadata or {})}
    meta.update(name=name, format=FORMAT, height=round(model.height, 4), bones=len(model.skeleton),
                triangles=int(sum(len(p.indices) for p in model.primitives)), primitives=primitives)
    path = folder / f"{name}.json"
    _write_json(path, meta)
    return path


def _read_png(path: Path) -> np.ndarray:
    import cv2

    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"cannot read texture {path}")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGBA)
    return cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA if image.shape[2] == 4 else cv2.COLOR_BGR2RGBA)


def read_character(cache_dir, name: str, textures: str = "path", metadata: dict | None = None) -> CharacterModel:
    """Load a prepared avatar (fast: no FBX parsing).

    ``textures="path"`` puts the PNG file paths into the primitives (the renderer decodes each file once and
    shares it); ``"array"`` decodes them into RGBA ``uint8`` arrays (row 0 = top), like the FBX loader;
    ``"none"`` leaves them out. ``metadata`` replaces the stored JSON metadata.
    """
    if textures not in ("path", "array", "none"):
        raise ValueError("textures must be 'path', 'array' or 'none'")
    folder = Path(cache_dir).expanduser() / "avatars"
    meta = metadata if metadata is not None else json.loads((folder / f"{name}.json").read_text(encoding="utf-8"))
    with np.load(folder / f"{name}.npz") as z:
        arrays = {k: z[k] for k in z.files}
    skeleton = Skeleton([str(n) for n in arrays["names"]], arrays["parents"], arrays["rest_local"])
    files = sorted({p["texture"] for p in meta["primitives"] if p.get("texture")})
    if textures == "array" and files:
        with ThreadPoolExecutor(max_workers=min(4, len(files))) as pool:
            images = dict(zip(files, pool.map(lambda f: _read_png(folder / f), files)))
    else:
        images = {f: (str(folder / f) if textures == "path" else None) for f in files}
    primitives = []
    for i, p in enumerate(meta["primitives"]):
        primitives.append(SkinnedPrimitive(p["name"], *(arrays[f"p{i}_{k}"] for k in (
            "positions", "normals", "uvs", "indices", "joints", "weights")), texture=images.get(p.get("texture"))))
    return CharacterModel(name, skeleton, primitives, arrays["inverse_bind"], dict(meta))


def save_clip(clip: AnimationClip, cache_dir, metadata: dict | None = None) -> Path:
    """Write ``clip`` to ``<cache_dir>/clips/<name>.npz`` (float32 arrays plus JSON info)."""
    folder = Path(cache_dir).expanduser() / "clips"
    folder.mkdir(parents=True, exist_ok=True)
    info = {"name": clip.name, "fps": float(clip.fps), "loop": bool(clip.loop), "root": clip.root,
            "format": FORMAT, "metadata": {**clip.metadata, **(metadata or {})}}
    path = folder / f"{clip.name}.npz"
    _write_atomic(path, lambda f: np.savez(
        f, bone_names=np.array(clip.bone_names), rotations=np.asarray(clip.rotations, np.float32),
        translations=np.asarray(clip.translations, np.float32), info=np.array(_json(info))))
    return path


def read_clip(path) -> AnimationClip:
    """Load a clip written by :func:`save_clip`."""
    with np.load(Path(path)) as z:
        info = json.loads(str(z["info"]))
        return AnimationClip(info["name"], info["fps"], [str(n) for n in z["bone_names"]],
                             z["rotations"].astype(np.float64), z["translations"].astype(np.float64),
                             loop=info["loop"], root=info["root"], metadata=info["metadata"])


# -- conversion ------------------------------------------------------------------------------------


@dataclass
class PrepareReport:
    avatars: list[str] = field(default_factory=list)  # converted now
    clips: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)  # already prepared
    failed: list[tuple[str, str]] = field(default_factory=list)  # (name, reason)


def _assets_root(raw_dir) -> Path:
    """The folder that contains ``Assets/`` (``raw_dir`` may be that folder or ``Assets`` itself)."""
    raw = Path(raw_dir).expanduser()
    return raw.parent if raw.name == "Assets" and not (raw / "Assets").is_dir() else raw


def _delete(root: Path, paths: list[str]) -> None:
    for path in paths:
        file = root / path
        file.unlink(missing_ok=True)
        folder = file.parent
        while folder != root and folder.is_dir() and not any(folder.iterdir()):
            folder.rmdir()
            folder = folder.parent


def prepare(raw_dir, cache_dir, avatars=None, clips=None, delete_raw: bool = False, *, force: bool = False,
            max_texture_size: int = 1024, progress=None) -> PrepareReport:
    """Convert downloaded avatars and clips (names; ``None`` = all that are in ``raw_dir``) into the cache.

    Needs :mod:`.fbx_convert` (imported here only). Items already in the cache are skipped unless ``force``.
    ``delete_raw`` removes each item's downloaded files once it is converted. Failures are collected in the
    report instead of stopping the run. ``progress(done, total, name, status)`` is called after every item.
    """
    root, cache = _assets_root(raw_dir), Path(cache_dir).expanduser()
    if avatars is None:
        avatars = [n for n, a in AVATARS.items() if (root / a.fbx).is_file()]
    if clips is None:
        clips = [n for n, c in CLIPS.items() if (root / c.path).is_file()]
    items = [("avatar", n) for n in check_names(avatars, AVATARS, "avatar")]
    items += [("clip", n) for n in check_names(clips, CLIPS, "clip")]
    report, converter, catalog = PrepareReport(), None, load_catalog()
    for done, (kind, name) in enumerate(items, 1):
        files = AVATARS[name].files if kind == "avatar" else [CLIPS[name].path]
        out = cache / "avatars" / f"{name}.json" if kind == "avatar" else cache / "clips" / f"{name}.npz"
        missing = [p for p in files if not (root / p).is_file()]
        ok = False
        if out.is_file() and not force:
            report.skipped.append(name)
            status, ok = "already prepared", True
        elif missing:
            report.failed.append((name, f"missing {missing[0]}"))
            status = f"failed: missing {missing[0]} (download it with: bifold-characters fetch)"
        else:
            if converter is None:
                try:
                    converter = importlib.import_module(".fbx_convert", __package__)
                except ImportError as exc:
                    raise ImportError(f"converting Rocketbox files needs {__package__}.fbx_convert: {exc}") from exc
            try:
                if kind == "avatar":
                    model = converter.load_character(root / files[0], max_texture_size=max_texture_size)
                    model.name = name
                    meta = {**avatar_metadata(name, catalog), "source": files[0], "license": LICENSE}
                    save_character(model, cache, meta)
                    report.avatars.append(name)
                    status = (f"prepared ({meta.get('sex', '?')}, {model.height:.2f} m, "
                              f"{sum(len(p.indices) for p in model.primitives)} triangles)")
                else:
                    clip = converter.load_clip(root / files[0], name=name)
                    save_clip(clip, cache, {"role": CLIPS[name].role, "sex": CLIPS[name].sex, "source": files[0]})
                    report.clips.append(name)
                    status = f"prepared ({clip.frames} frames, {clip.duration:.1f} s)"
                ok = True
            except Exception as exc:  # noqa: BLE001 - one broken file must not stop the others
                report.failed.append((name, f"{type(exc).__name__}: {exc}"))
                status = f"failed: {type(exc).__name__}: {exc}"
        if delete_raw and ok:
            _delete(root, files)
        if progress:
            progress(done, len(items), name, status)
    return report
