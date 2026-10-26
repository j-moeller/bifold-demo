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

import collections
import difflib
import http.client
import importlib
import json
import os
import queue
import ssl
import threading
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
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

# A quick start: 6 women and 6 men, a girl and a boy, an older man (Male_Adult_03, about 58: the catalog's oldest),
# all three skin-tone bands, mostly conference outfits.
SMALL_SET = ("Business_Female_01", "Business_Male_02", "Male_Adult_03", "Female_Child_01", "Male_Child_02",
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

    ``conference`` uses the catalog when it exists: people that fit a conference (``conference_fit >= 2``)
    plus all children (families visit fairs too; the catalog rates children as unusual), else the Adults,
    the Children and the Business people.
    """
    if name == "all":
        return list(AVATARS)
    if name == "small":
        return list(SMALL_SET)
    if name == "conference":
        catalog = load_catalog() if catalog is None else catalog
        if catalog:
            return [n for n in AVATARS if float(catalog.get(n, {}).get("conference_fit", 0) or 0) >= 2
                    or catalog.get(n, {}).get("age_group", AVATARS[n].age_group) == "child"]
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
    """Some files could not be downloaded; ``report.failed`` lists them (the rest was downloaded).

    When the fetch gave up early (the server cannot be reached, or its TLS certificate cannot be verified),
    ``report.stopped`` says why and what to do, and the other files were not tried.
    """

    def __init__(self, report: FetchReport):
        self.report = report
        if report.stopped:
            missing = len(report.files) - report.downloaded - report.skipped
            super().__init__(f"{report.stopped}, then run the same command again ({missing} of {len(report.files)} "
                             f"files are still missing)")
            return
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
    stopped: str = ""  # why the fetch gave up before trying every file, and what to do; else ""


UNREACHABLE_AFTER = 3  # files in a row that got no answer before a fetch gives up on the rest
CONNECT_TIMEOUT = 15.0  # seconds for connecting (through a proxy, with TLS) and again for the answer to start


class _Permanent(Exception):
    """A download error that retrying cannot fix."""


class _Untrusted(_Permanent):
    """The TLS certificate of the server cannot be verified: every file of the fetch would fail the same way."""


class _Unreachable(RuntimeError):
    """No attempt got an answer (no network, unknown host, refused connection, proxy error, no answer in time)."""


class _Cancelled(Exception):
    """The download was stopped from outside (Ctrl-C, or the fetch gave up); its ``.part`` file is kept."""


class _QuickToAnswer:
    """For an ``http.client`` connection: at most :data:`CONNECT_TIMEOUT` until the answer starts.

    Connecting (through a proxy, with the TLS handshake) and receiving the status line and headers each wait at
    most :data:`CONNECT_TIMEOUT`; the request's own (longer) timeout applies to reading the body.
    """

    def connect(self):
        timeout = self.timeout
        if isinstance(timeout, (int, float)):
            self.timeout = min(timeout, CONNECT_TIMEOUT)
        try:
            super().connect()
        finally:
            self.timeout = timeout

    def getresponse(self):
        sock = self.sock
        response = super().getresponse()
        if sock is not None and isinstance(self.timeout, (int, float)):
            sock.settimeout(self.timeout)  # the response reads the body from this socket
        return response


class _HTTPConnection(_QuickToAnswer, http.client.HTTPConnection):
    pass


class _HTTPSConnection(_QuickToAnswer, http.client.HTTPSConnection):
    pass


class _HTTPHandler(urllib.request.HTTPHandler):
    def http_open(self, req):
        return self.do_open(_HTTPConnection, req)


class _HTTPSHandler(urllib.request.HTTPSHandler):
    def https_open(self, req):
        return self.do_open(_HTTPSConnection, req, context=self._context)


def _default_opener():
    context = ssl.create_default_context()
    if not os.environ.get("SSL_CERT_FILE"):
        try:  # e.g. python.org builds on macOS have no system certificates
            import certifi

            context.load_verify_locations(certifi.where())
        except (ImportError, OSError):
            pass
    director = urllib.request.build_opener(_HTTPHandler(), _HTTPSHandler(context=context))  # proxies from the env

    def opener(request, timeout=60.0):
        return director.open(request, timeout=timeout)

    return opener


def _download(url: str, target: Path, opener, retries: int, backoff: float, timeout: float,
              stop: threading.Event | None = None, answered: threading.Event | None = None) -> int:
    """Download ``url`` to ``target`` via ``target.part`` (resumed with a Range request); returns the size.

    ``answered`` is set by the first answer of the server (it is shared by the files of a fetch). Raises
    :class:`_Cancelled` soon after ``stop`` is set, :class:`_Untrusted` when the TLS certificate of the server
    cannot be verified and :class:`_Unreachable` when no attempt got an answer. A timeout before ``answered`` is
    set is not retried: nothing has answered yet (most likely there is no network) and every retry would wait as long.
    """
    stop = stop or threading.Event()
    answered = answered or threading.Event()
    part = target.with_name(target.name + ".part")
    error: Exception | None = None
    silent = False  # the latest attempt got no answer in time
    for attempt in range(retries + 1):
        if stop.wait(backoff * 2 ** (attempt - 1) if attempt else 0):
            raise _Cancelled(url)
        response = None
        try:
            offset = part.stat().st_size if part.exists() else 0
            headers = {"User-Agent": "bifold-demo"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            with opener(urllib.request.Request(url, headers=headers), timeout=timeout) as response:
                answered.set()
                status = getattr(response, "status", None) or response.getcode()
                if offset and status != 206:
                    offset = 0  # the server sent the whole file
                length = response.headers.get("Content-Length")
                expected = offset + int(length) if length is not None else None
                with open(part, "ab" if offset else "wb") as out:
                    while block := response.read(1 << 16):
                        out.write(block)
                        if stop.is_set():
                            raise _Cancelled(url)
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
            answered.set()
            if exc.code == 416:  # stale .part: start over
                part.unlink(missing_ok=True)
            elif exc.code in (408, 429) or exc.code >= 500:
                pass
            else:
                raise _Permanent(f"HTTP {exc.code} {exc.reason} for {url}") from None
            error, silent = exc, False
        except (urllib.error.URLError, HTTPException, OSError, ValueError) as exc:
            reason = exc.reason if isinstance(exc, urllib.error.URLError) else exc
            if isinstance(reason, ssl.SSLCertVerificationError):
                raise _Untrusted(f"TLS certificate check failed ({getattr(reason, 'verify_message', '') or reason})") \
                    from None
            error, silent = exc, response is None and isinstance(reason, TimeoutError)
            if silent and not answered.is_set():
                break
    attempts = attempt + 1
    if silent:
        reason = "no answer in time"
    elif isinstance(error, urllib.error.HTTPError):
        reason = f"HTTP {error.code} {error.reason}"
    else:
        reason = str(getattr(error, "reason", None) or error) or type(error).__name__
    # urlopen wraps the errors of connecting (not those of reading a response) in a URLError
    unreachable = silent or (isinstance(error, urllib.error.URLError) and not isinstance(error, urllib.error.HTTPError))
    raise (_Unreachable if unreachable else RuntimeError)(f"{reason} (after {attempts} attempt"
                                                          f"{'s' if attempts > 1 else ''})")


def fetch(dest, avatars, clips, jobs: int = 8, progress=None, *, retries: int = 3, opener=None,
          base_url: str = BASE_URL, backoff: float = 1.0, timeout: float = 60.0) -> FetchReport:
    """Download the files of ``avatars`` and ``clips`` (names; ``None`` = all) into ``dest``.

    ``dest`` gets the repository layout (``Assets/...``); an ``Assets`` folder (one that holds ``Avatars/`` or
    ``Animations/``, under any name) gets the layout inside it, like :func:`prepare` reads it. Files that are
    already there are skipped, partial downloads (``*.part``) are resumed, each file appears under its final name
    only when complete, and ``jobs`` files download in parallel with ``retries`` retries each.
    ``progress(done, total, path, status)`` is called after every file.

    Raises :class:`FetchError` after the others finished when some files failed, or at once (without waiting for
    the running downloads) when :data:`UNREACHABLE_AFTER` files in a row got no answer (no network) or when the TLS
    certificate of the server cannot be verified. Connecting and waiting for an answer to start take at most
    :data:`CONNECT_TIMEOUT` seconds each, ``timeout`` is for reading. On ``KeyboardInterrupt`` the queued files are
    dropped and the running downloads stop at their next block, so a rerun resumes them; the downloads run in daemon
    threads, so one that waits for the network does not keep the program alive. ``opener(request, timeout=...)``
    replaces the default opener (for tests).
    """
    if jobs < 1 or retries < 0:
        raise ValueError("jobs must be at least 1 and retries at least 0")
    _, local = raw_files(dest)
    paths = files_for(AVATARS if avatars is None else avatars, CLIPS if clips is None else clips)
    report = FetchReport(files=[local(p) for p in paths])
    opener = opener or _default_opener()
    todo = []
    for path, target in zip(paths, report.files):
        if target.is_file():
            report.skipped += 1
            if progress:
                progress(report.skipped, len(paths), path, "already there")
        else:
            todo.append((path, target))
    host = urllib.parse.urlsplit(base_url).netloc
    stop, answered, lock = threading.Event(), threading.Event(), threading.Lock()
    work, results = collections.deque(todo), queue.SimpleQueue()
    unreachable: list[str] = []  # the latest files in a row that got no answer
    stopped_by: list[str] = []  # the file whose failure made the fetch give up

    def give_up(path: str, why: str) -> None:  # with the lock held
        if not stop.is_set():
            report.stopped, stopped_by[:] = why, [path]
            stop.set()  # the workers take no more files and the running downloads end (as cancelled)

    def run(path: str, target: Path) -> None:
        try:
            outcome = _download(url_for(path, base_url), target, opener, retries, backoff, timeout, stop, answered)
        except _Untrusted as exc:  # the same for every file: give up at once
            with lock:  # (put with the lock held: the file that makes the fetch give up is the last one counted)
                give_up(path, f"cannot download from {host}: {exc}; install certificates for this Python (macOS: "
                              f"run 'Install Certificates.command'; or: pip install certifi), or, if this network "
                              f"intercepts HTTPS (a Wi-Fi login page or a company proxy), log in to it first or set "
                              f"SSL_CERT_FILE to its certificate")
                results.put((path, exc))
            return
        except _Unreachable as exc:
            with lock:
                unreachable.append(path)
                if len(unreachable) >= UNREACHABLE_AFTER:
                    give_up(path, f"cannot reach {host}: {exc}; check the network connection (and proxy settings)")
                results.put((path, exc))
            return
        except BaseException as exc:  # noqa: BLE001 - the main thread reports it
            outcome = exc
        if not isinstance(outcome, _Cancelled):
            with lock:
                unreachable.clear()
        results.put((path, outcome))

    def worker() -> None:
        while not stop.is_set():
            try:
                path, target = work.popleft()
            except IndexError:
                return
            run(path, target)

    for folder in dict.fromkeys(target.parent for _, target in todo):
        folder.mkdir(parents=True, exist_ok=True)
    threads = [threading.Thread(target=worker, name=f"fetch-{i + 1}", daemon=True) for i in range(min(jobs, len(todo)))]
    done, received = report.skipped, 0
    try:
        for thread in threads:
            thread.start()
        while received < len(todo):
            try:
                path, outcome = results.get(timeout=0.25)  # short waits: Ctrl-C gets through at once on every system
            except queue.Empty:
                continue
            received += 1
            if isinstance(outcome, _Cancelled):
                continue
            if isinstance(outcome, (_Permanent, RuntimeError)):
                report.failed.append((path, str(outcome)))
                status = f"failed: {outcome}"
            elif isinstance(outcome, BaseException):
                raise outcome
            else:
                report.downloaded += 1
                report.bytes += outcome
                status = f"downloaded {outcome / 1e6:.1f} MB"
            done += 1
            if progress:
                progress(done, len(paths), path, status)
            if stopped_by and path == stopped_by[0]:
                break  # given up: the running downloads end by themselves (cancelled, or at their timeout)
    except BaseException:  # e.g. Ctrl-C: the workers take no more files, the running ones stop at their next block
        stop.set()
        raise
    if not stop.is_set():
        for thread in threads:
            thread.join()
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
        for key in _PRIMITIVE_ARRAYS:
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
        primitives.append(SkinnedPrimitive(p["name"], *(arrays[f"p{i}_{k}"] for k in _PRIMITIVE_ARRAYS),
                                           texture=images.get(p.get("texture"))))
    return CharacterModel(name, skeleton, primitives, arrays["inverse_bind"], dict(meta))


def save_clip(clip: AnimationClip, cache_dir, metadata: dict | None = None, path=None) -> Path:
    """Write ``clip`` to ``<cache_dir>/clips/<name>.npz`` (float32 arrays plus JSON info), or to ``path``."""
    folder = Path(cache_dir).expanduser() / "clips" if path is None else Path(path).parent
    folder.mkdir(parents=True, exist_ok=True)
    info = {"name": clip.name, "fps": float(clip.fps), "loop": bool(clip.loop), "root": clip.root,
            "format": FORMAT, "metadata": {**clip.metadata, **(metadata or {})}}
    path = folder / f"{clip.name}.npz" if path is None else Path(path)
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


_SKELETON_ARRAYS = ("names", "parents", "rest_local", "inverse_bind")
_PRIMITIVE_ARRAYS = ("positions", "normals", "uvs", "indices", "joints", "weights")
_CLIP_ARRAYS = ("bone_names", "rotations", "translations", "info")
_PNG_START, _PNG_END = b"\x89PNG\r\n\x1a\n", b"IEND\xaeB`\x82"


def _check_npz(path: Path, arrays) -> None:
    try:
        with zipfile.ZipFile(path) as z:  # the directory is at the end: a cut-off file fails here
            names = {n.removesuffix(".npy") for n in z.namelist()}
            missing = [a for a in arrays if a not in names]
            bad = None if missing else z.testzip()  # reads every array (decodes nothing) and checks its CRC
    except FileNotFoundError:
        raise ValueError(f"{path.name} is missing") from None
    except Exception as exc:  # noqa: BLE001 - zipfile raises several kinds of errors for a damaged file
        raise ValueError(f"{path.name} is damaged ({type(exc).__name__}: {exc})") from None
    if missing:
        raise ValueError(f"{path.name} is damaged (no {missing[0]} array)")
    if bad is not None:
        raise ValueError(f"{path.name} is damaged (bad data in {bad})")


def _check_png(path: Path) -> None:
    try:
        with open(path, "rb") as f:
            start = f.read(len(_PNG_START))
            f.seek(max(path.stat().st_size - len(_PNG_END), 0))
            end = f.read()
    except FileNotFoundError:
        raise ValueError(f"{path.name} is missing") from None
    except OSError as exc:
        raise ValueError(f"cannot read {path.name} ({exc})") from None
    if start != _PNG_START or end != _PNG_END:
        raise ValueError(f"{path.name} is damaged (cut off, or not a PNG file)")


def check_avatar(cache_dir, name: str) -> dict:
    """The stored metadata of the prepared avatar ``name`` after a quick check of its files (nothing is decoded).

    The arrays are read to compare them with their checksums; of the textures only the start and the end are read.
    Raises ``FileNotFoundError`` when it is not prepared, ``ValueError`` saying what is wrong when one of its files
    is missing, damaged (e.g. cut off, or changed inside) or was written by another version.
    """
    folder = Path(cache_dir).expanduser() / "avatars"
    path = folder / f"{name}.json"
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (OSError, ValueError) as exc:
        raise ValueError(f"{path.name} is damaged ({exc})") from None
    primitives = meta.get("primitives") if isinstance(meta, dict) else None
    if not isinstance(primitives, list) or not all(isinstance(p, dict) for p in primitives):
        raise ValueError(f"{path.name} is damaged (no list of primitives)")
    if meta.get("format") != FORMAT:
        raise ValueError(f"prepared by another version (format {meta.get('format')!r}, this version reads {FORMAT})")
    _check_npz(folder / f"{name}.npz",
               [*_SKELETON_ARRAYS, *(f"p{i}_{k}" for i in range(len(primitives)) for k in _PRIMITIVE_ARRAYS)])
    for file in sorted({p["texture"] for p in primitives if p.get("texture")}):
        _check_png(folder / file)
    return meta


def check_clip(path) -> None:
    """A quick check of a prepared clip file (its arrays are compared with their checksums; nothing is decoded).

    Raises ``FileNotFoundError`` when it does not exist, ``ValueError`` saying what is wrong when it is damaged.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no clip file {path}")
    _check_npz(path, _CLIP_ARRAYS)


def is_prepared(cache_dir, kind: str, name: str) -> bool:
    """Whether the ``kind`` (``"avatar"`` or ``"clip"``) ``name`` is in the cache and passes the quick check.

    The one test for "prepared" that :func:`prepare`, ``bifold-characters fetch`` and :mod:`.library` share: a
    damaged item counts as not prepared, so preparing (or fetching) it again repairs it.
    """
    cache = Path(cache_dir).expanduser()
    try:
        if kind == "avatar":
            check_avatar(cache, name)
        else:
            check_clip(cache / "clips" / f"{name}.npz")
    except (OSError, ValueError):
        return False
    return True


# -- conversion ------------------------------------------------------------------------------------


@dataclass
class PrepareReport:
    avatars: list[str] = field(default_factory=list)  # converted now
    clips: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)  # already prepared
    failed: list[tuple[str, str]] = field(default_factory=list)  # (name, reason)


def raw_files(raw_dir):
    """``(root, local)``: ``local(path)`` is the file of a repository path (``Assets/...``) in ``raw_dir``.

    ``raw_dir`` may hold ``Assets/`` (a clone of the repository, or the fetch folder) or be an ``Assets``
    folder itself, under any name (it holds ``Avatars/`` or ``Animations/``). :func:`fetch` downloads to these
    files and :func:`prepare` reads them, so both treat a folder the same way.
    """
    raw = Path(raw_dir).expanduser()
    if (raw / "Assets").is_dir() or not ((raw / "Avatars").is_dir() or (raw / "Animations").is_dir()):
        return raw, lambda path: raw / path
    return raw, lambda path: raw / path.removeprefix("Assets/")


def _delete(root: Path, files: list[Path]) -> None:
    for file in files:
        file.unlink(missing_ok=True)
        folder = file.parent
        while folder != root and folder.is_dir() and not any(folder.iterdir()):
            folder.rmdir()
            folder = folder.parent


def prepare(raw_dir, cache_dir, avatars=None, clips=None, delete_raw: bool = False, *, force: bool = False,
            max_texture_size: int = 1024, progress=None) -> PrepareReport:
    """Convert downloaded avatars and clips (names; ``None`` = all that are in ``raw_dir``) into the cache.

    Needs :mod:`.fbx_convert` (imported here only). Items already in the cache (see :func:`is_prepared`; damaged
    ones are converted again) are skipped unless ``force``. ``delete_raw`` removes each item's downloaded files
    once it is converted. Failures are collected in the report instead of stopping the run.
    ``progress(done, total, name, status)`` is called after every item.
    """
    (root, local), cache = raw_files(raw_dir), Path(cache_dir).expanduser()
    if avatars is None:
        avatars = [n for n, a in AVATARS.items() if local(a.fbx).is_file()]
    if clips is None:
        clips = [n for n, c in CLIPS.items() if local(c.path).is_file()]
    items = [("avatar", n) for n in check_names(avatars, AVATARS, "avatar")]
    items += [("clip", n) for n in check_names(clips, CLIPS, "clip")]
    report, converter, catalog = PrepareReport(), None, load_catalog()
    for done, (kind, name) in enumerate(items, 1):
        files = AVATARS[name].files if kind == "avatar" else [CLIPS[name].path]
        missing = [p for p in files if not local(p).is_file()]
        ok = False
        if not force and is_prepared(cache, kind, name):
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
                    model = converter.load_character(local(files[0]), max_texture_size=max_texture_size)
                    model.name = name
                    meta = {**avatar_metadata(name, catalog), "source": files[0], "license": LICENSE}
                    save_character(model, cache, meta)
                    report.avatars.append(name)
                    status = (f"prepared ({meta.get('sex', '?')}, {model.height:.2f} m, "
                              f"{sum(len(p.indices) for p in model.primitives)} triangles)")
                else:
                    clip = converter.load_clip(local(files[0]), name=name)
                    save_clip(clip, cache, {"role": CLIPS[name].role, "sex": CLIPS[name].sex, "source": files[0]})
                    report.clips.append(name)
                    status = f"prepared ({clip.frames} frames, {clip.duration:.1f} s)"
                ok = True
            except Exception as exc:  # noqa: BLE001 - one broken file must not stop the others
                report.failed.append((name, f"{type(exc).__name__}: {exc}"))
                status = f"failed: {type(exc).__name__}: {exc}"
        if delete_raw and ok:
            _delete(root, [local(p) for p in files])
        if progress:
            progress(done, len(items), name, status)
    return report
