import importlib
import io
import json
import os
import subprocess
import sys
import threading
import time
import types
import urllib.error
from pathlib import Path

import cv2
import numpy as np
import pytest

from bifold_demo.characters import cli, rocketbox
from bifold_demo.characters.library import CharacterLibrary, default_folder
from bifold_demo.characters.model import AnimationClip, CharacterModel, Skeleton, SkinnedPrimitive, quat_from_axis_angle

SCRATCH = "/tmp/claude-0/-home-user-bifold-demo/44ad5b06-773f-5ffd-8306-515788fc8497/scratchpad"
FILE_LIST = Path(os.environ.get("BIFOLD_ROCKETBOX_FILES", f"{SCRATCH}/rocketbox_files.txt"))
ROCKETBOX = Path(os.environ.get("BIFOLD_ROCKETBOX_TEST_DIR", f"{SCRATCH}/rb/Assets"))
CONTRACT_ROLES = ("walk", "walk_slow", "stroll", "idle", "look_around", "talk", "listen", "phone",
                  "turn_left_90", "turn_right_90", "turn_left_180", "turn_right_180")
BONES = ["Bip01", "Bip01 Pelvis", "Bip01 Spine", "Bip01 Head"]


# -- manifest --------------------------------------------------------------------------------------


def test_every_role_has_clips_for_men_and_women():
    assert tuple(rocketbox.ROLE_CLIPS) == CONTRACT_ROLES
    for role, names in rocketbox.ROLE_CLIPS.items():
        assert any(n.startswith("m_") for n in names) and any(n.startswith("f_") for n in names), role
        assert all(rocketbox.CLIPS[n].role == role for n in names)
    try:
        from bifold_demo.characters.behaviour import ROLES
    except ImportError:
        return
    assert tuple(ROLES) == CONTRACT_ROLES


def test_manifest_matches_the_repository_listing():
    if not FILE_LIST.is_file():
        pytest.skip(f"no Rocketbox file list at {FILE_LIST}")
    listing = {line.strip() for line in FILE_LIST.read_text().splitlines()}
    paths = rocketbox.files_for(rocketbox.AVATARS, rocketbox.CLIPS)
    assert not [p for p in paths if p not in listing]
    exported = {p.split("/")[3] for p in listing
                if p.startswith("Assets/Avatars/") and p.endswith(".fbx") and p.split("/")[4:5] == ["Export"]
                and p.split("/")[5] == p.split("/")[3] + ".fbx"}
    assert set(rocketbox.AVATARS) == exported and len(exported) == 115  # Female_Party_0x only once
    for avatar in rocketbox.AVATARS.values():
        colours = {p.rsplit("/", 1)[1] for p in listing if p.startswith(f"Assets/Avatars/{avatar.path}/Textures/")
                   and "color" in p.lower() and p.lower().endswith(".tga")}
        assert set(avatar.textures) == colours, avatar.name


def test_avatar_sets():
    small = [rocketbox.AVATARS[n] for n in rocketbox.avatar_set("small")]
    assert 10 <= len(small) <= 14
    assert {a.sex for a in small} == {"female", "male"} and any(a.age_group == "child" for a in small)
    fallback = rocketbox.avatar_set("conference", catalog={})
    assert "Business_Male_02" in fallback and "Female_Child_01" in fallback and "Fire_Male_01" not in fallback
    assert len(fallback) == 40 + 4 + 11
    catalog = {"Male_Adult_01": {"conference_fit": 3}, "Fire_Male_01": {"conference_fit": 1},
               "Female_Child_01": {"conference_fit": 1, "age_group": "child"}}
    chosen = rocketbox.avatar_set("conference", catalog=catalog)
    children = [n for n, a in rocketbox.AVATARS.items() if a.age_group == "child"]
    assert chosen == ["Male_Adult_01"] + children or sorted(chosen) == sorted(["Male_Adult_01"] + children)
    assert "Fire_Male_01" not in chosen  # children always come along, other poor fits do not
    assert len(rocketbox.avatar_set("all")) == 115
    with pytest.raises(ValueError, match="unknown avatar set"):
        rocketbox.avatar_set("everyone")
    assert set(rocketbox.clip_set("basic")) <= set(rocketbox.clip_set("all"))
    assert len(rocketbox.clip_set("basic")) == 2 * len(CONTRACT_ROLES)


def test_names_sizes_and_urls():
    with pytest.raises(ValueError, match="Business_Female_01"):
        rocketbox.files_for(["Business_Femal_01"])
    files = rocketbox.files_for(["Business_Female_01"], ["m_walk_neutral_01"])
    assert files[0].endswith("Export/Business_Female_01.fbx") and files[-1].endswith("m_walk_neutral_01.max.fbx")
    sizes = [rocketbox.estimate_size(p) for p in files]
    assert sizes[0] < 1e6 and all(12e6 <= s <= 17e6 for s in sizes[1:-1]) and 1e6 <= sizes[-1] <= 6e6
    assert rocketbox.url_for("Assets/a b.tga") == rocketbox.BASE_URL + "Assets/a%20b.tga"


# -- download --------------------------------------------------------------------------------------


class FakeResponse:
    def __init__(self, data: bytes, status=200, length=None):
        self._data, self.status = io.BytesIO(data), status
        self.headers = {"Content-Length": str(len(data) if length is None else length)}

    def read(self, n=-1):
        return self._data.read(n)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeOpener:
    """Serves ``files`` (url -> bytes); ``failures[url]`` connection errors first; ``truncate`` halves bodies."""

    def __init__(self, files, failures=None, truncate=False):
        self.files, self.failures, self.truncate = files, dict(failures or {}), truncate
        self.requests: list[tuple[str, str | None]] = []
        self._lock = threading.Lock()

    def __call__(self, request, timeout=None):
        url, byte_range = request.full_url, request.get_header("Range")
        with self._lock:
            self.requests.append((url, byte_range))
            if self.failures.get(url, 0) > 0:
                self.failures[url] -= 1
                raise urllib.error.URLError("connection reset by peer")
        if url not in self.files:
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)
        data, status = self.files[url], 200
        if byte_range:
            data, status = data[int(byte_range.split("=")[1].rstrip("-")):], 206
        if self.truncate:
            return FakeResponse(data[: len(data) // 2], status, length=len(data))
        return FakeResponse(data, status)

    def count(self, url):
        return sum(u == url for u, _ in self.requests)


def _remote(avatars=("Female_Child_01",), clips=("m_walk_neutral_01",)):
    files = {}
    for i, path in enumerate(rocketbox.files_for(avatars, clips)):
        body = bytes(np.random.default_rng(i).integers(0, 256, 3000 + 100 * i, dtype=np.uint8))
        files[rocketbox.url_for(path)] = (rocketbox.FBX_MAGIC + body) if path.endswith(".fbx") else body
    return files


def _fetch(dest, opener, avatars=("Female_Child_01",), clips=("m_walk_neutral_01",), **kw):
    return rocketbox.fetch(dest, list(avatars), list(clips), opener=opener, backoff=0, **kw)


def test_fetch_downloads_in_parallel_with_progress(tmp_path):
    remote = _remote()
    calls = []
    report = _fetch(tmp_path, FakeOpener(remote), jobs=3, progress=lambda *a: calls.append(a))
    assert report.downloaded == 4 and report.skipped == 0 and report.bytes == sum(map(len, remote.values()))
    for path, local in zip(rocketbox.files_for(["Female_Child_01"], ["m_walk_neutral_01"]), report.files):
        assert local == tmp_path / path and local.read_bytes() == remote[rocketbox.url_for(path)]
    assert not list(tmp_path.rglob("*.part"))
    assert sorted(c[0] for c in calls) == [1, 2, 3, 4] and all(c[1] == 4 and "downloaded" in c[3] for c in calls)


def test_fetch_retries_and_gives_clear_errors(tmp_path):
    remote = _remote()
    urls = list(remote)
    opener = FakeOpener(remote, failures={urls[1]: 2})
    assert _fetch(tmp_path / "a", opener, retries=2).downloaded == 4
    assert opener.count(urls[1]) == 3

    missing = dict(remote)
    del missing[urls[2]]
    opener = FakeOpener(missing, failures={urls[1]: 5})
    with pytest.raises(rocketbox.FetchError, match="Run the same command again") as info:
        _fetch(tmp_path / "b", opener, retries=2)
    failed = dict(info.value.report.failed)
    assert len(failed) == 2 and "404" in failed[rocketbox.files_for(["Female_Child_01"])[2]]
    assert opener.count(urls[2]) == 1  # not found: no retries
    assert opener.count(urls[1]) == 3
    assert info.value.report.downloaded == 2


def test_fetch_resumes_and_skips_complete_files(tmp_path):
    remote = _remote()
    paths = rocketbox.files_for(["Female_Child_01"], ["m_walk_neutral_01"])
    (tmp_path / paths[0]).parent.mkdir(parents=True)
    (tmp_path / paths[0]).write_bytes(remote[rocketbox.url_for(paths[0])])
    (tmp_path / paths[1]).parent.mkdir(parents=True)
    (tmp_path / (paths[1] + ".part")).write_bytes(remote[rocketbox.url_for(paths[1])][:1000])
    opener = FakeOpener(remote)
    report = _fetch(tmp_path, opener)
    assert report.skipped == 1 and report.downloaded == 3
    assert opener.count(rocketbox.url_for(paths[0])) == 0
    assert (rocketbox.url_for(paths[1]), "bytes=1000-") in opener.requests
    assert (tmp_path / paths[1]).read_bytes() == remote[rocketbox.url_for(paths[1])]


def test_incomplete_downloads_never_appear_under_the_final_name(tmp_path):
    remote = _remote(clips=())
    with pytest.raises(rocketbox.FetchError, match="incomplete"):
        _fetch(tmp_path, FakeOpener(remote, truncate=True), clips=(), retries=1)
    paths = rocketbox.files_for(["Female_Child_01"])
    assert not any((tmp_path / p).exists() for p in paths)
    assert all((tmp_path / (p + ".part")).stat().st_size > 0 for p in paths)
    opener = FakeOpener(remote)
    assert _fetch(tmp_path, opener, clips=()).downloaded == 3  # resumes the .part files
    assert all(r for _, r in opener.requests)
    assert all((tmp_path / p).read_bytes() == remote[rocketbox.url_for(p)] for p in paths)

    html = {url: b"<html>rate limited</html>" for url in remote}
    with pytest.raises(rocketbox.FetchError, match="not a binary FBX"):
        _fetch(tmp_path / "html", FakeOpener(html), clips=())


# -- prepare and load ------------------------------------------------------------------------------


def _model(name: str) -> CharacterModel:
    rest = np.tile(np.eye(4), (4, 1, 1))
    rest[1, 1, 3], rest[2, 1, 3], rest[3, 1, 3] = 0.9, 0.2, 0.5
    skeleton = Skeleton(BONES, [-1, 0, 1, 2], rest)
    rng = np.random.default_rng(len(name))
    shared, own = (rng.integers(0, 256, (16, 8, 4), dtype=np.uint8) for _ in range(2))
    primitives = []
    for i, texture in enumerate([shared, shared, own, None]):
        n = 6 + i
        positions = rng.random((n, 3), dtype=np.float32) * [0.4, 1.6 + 0.1 * len(name) / 20, 0.3]
        primitives.append(SkinnedPrimitive(f"part{i}", positions.astype(np.float32),
                                           rng.random((n, 3), dtype=np.float32), rng.random((n, 2), dtype=np.float32),
                                           rng.integers(0, n, (n - 2, 3)).astype(np.uint32),
                                           rng.integers(0, 4, (n, 4)).astype(np.int32),
                                           np.full((n, 4), 0.25, np.float32), texture))
    return CharacterModel(name, skeleton, primitives, np.linalg.inv(skeleton.rest_global),
                          {"lod": "hipoly", "triangles": sum(len(p.indices) for p in primitives)})


def _clip(name: str) -> AnimationClip:
    kind = "walk" if "walk" in name else "turn" if "turn" in name else "idle"
    frames = 5 + len(name) % 7
    angles = np.linspace(0, 30, frames)[:, None].repeat(len(BONES), 1)
    rotations = quat_from_axis_angle([0.0, 1.0, 0.0], angles)
    translations = np.zeros((frames, len(BONES), 3))
    translations[:, 0, 1] = 0.95 + 0.01 * np.sin(np.arange(frames))
    return AnimationClip(name, 30.0, list(BONES), rotations, translations, loop=kind != "turn", root="Bip01",
                         metadata={"kind": kind, "speed": 1.2 if kind == "walk" else 0.0,
                                   "turn": 90.0 if "left_90" in name else 0.0})


@pytest.fixture
def stub_convert(monkeypatch):
    """A fake ``fbx_convert`` (no FBX parsing) that records what it converts."""
    calls = []

    def load_character(fbx_path, texture_dir=None, lod="auto", max_texture_size=1024):
        calls.append(Path(fbx_path).name)
        model = _model(Path(fbx_path).stem)
        model.metadata["source"] = str(fbx_path)
        return model

    def load_clip(fbx_path, name=None):
        calls.append(Path(fbx_path).name)
        return _clip(name or Path(fbx_path).name.split(".")[0])

    module = types.ModuleType("bifold_demo.characters.fbx_convert")
    module.load_character, module.load_clip = load_character, load_clip
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(rocketbox, "CATALOG_PATH", Path("/nonexistent/catalog.json"))
    return calls


AVATARS = ("Female_Child_01", "Business_Male_02")


def _raw_tree(root: Path, avatars=AVATARS, clips=None) -> Path:
    for path in rocketbox.files_for(avatars, rocketbox.clip_set("basic") if clips is None else clips):
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_bytes(b"raw")
    return root


def test_prepare_and_load_round_trip(tmp_path, stub_convert):
    raw, cache = _raw_tree(tmp_path / "raw"), tmp_path / "cache"
    report = rocketbox.prepare(raw / "Assets", cache)
    assert sorted(report.avatars) == sorted(AVATARS) and len(report.clips) == 24 and not report.failed
    library = CharacterLibrary(cache)
    available = library.available()
    assert sorted(available) == sorted(AVATARS)
    meta = available["Female_Child_01"]
    assert (meta["sex"], meta["age_group"], meta["category"]) == ("female", "child", "Children")
    assert meta["source"] == rocketbox.AVATARS["Female_Child_01"].fbx and meta["lod"] == "hipoly"
    assert len(list((cache / "avatars").glob("Female_Child_01.*.png"))) == 2  # shared texture written once

    original, model = _model("Female_Child_01"), library.load("Female_Child_01", textures="array")
    assert model.skeleton.names == BONES and np.array_equal(model.skeleton.parents, original.skeleton.parents)
    assert np.allclose(model.skeleton.rest_local, original.skeleton.rest_local)
    assert np.allclose(model.inverse_bind, original.inverse_bind)
    assert abs(model.height - original.height) < 1e-6 and model.metadata["height"] == pytest.approx(model.height, 1e-3)
    for a, b in zip(original.primitives, model.primitives):
        for key in ("positions", "normals", "uvs", "indices", "joints", "weights"):
            assert np.array_equal(getattr(a, key), getattr(b, key)) and getattr(b, key).dtype == getattr(a, key).dtype
        assert (b.texture is None) if a.texture is None else np.array_equal(a.texture, b.texture)
    assert model.primitives[0].texture is model.primitives[1].texture
    assert library.load("Female_Child_01", textures="array") is model

    paths = library.load("Female_Child_01").primitives
    assert isinstance(paths[0].texture, str) and paths[3].texture is None
    rgba = cv2.cvtColor(cv2.imread(paths[2].texture, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA)
    assert np.array_equal(rgba, original.primitives[2].texture)
    with pytest.raises(KeyError, match="not prepared yet"):
        library.load("Male_Adult_01")

    clips = library.clips()
    assert len(clips) == 24
    clip, source = clips["m_turn_left_90"], _clip("m_turn_left_90")
    assert clip.bone_names == BONES and clip.fps == 30.0 and clip.loop is False and clip.root == "Bip01"
    assert np.allclose(clip.rotations, source.rotations, atol=1e-6) and clip.rotations.dtype == np.float64
    assert np.allclose(clip.translations, source.translations, atol=1e-6)
    assert clip.metadata["kind"] == "turn" and clip.metadata["turn"] == 90.0
    assert (clip.metadata["role"], clip.metadata["sex"]) == ("turn_left_90", "male")
    assert library.is_ready() and library.describe().endswith("ready")


def test_prepare_skips_prepared_items_and_reports_missing_files(tmp_path, stub_convert):
    raw, cache = _raw_tree(tmp_path / "raw"), tmp_path / "cache"
    rocketbox.prepare(raw, cache, avatars=["Female_Child_01"], clips=["m_walk_neutral_01"])
    stub_convert.clear()
    report = rocketbox.prepare(raw, cache, avatars=["Female_Child_01"], clips=["m_walk_neutral_01"])
    assert report.skipped == ["Female_Child_01", "m_walk_neutral_01"] and not stub_convert
    report = rocketbox.prepare(raw, cache, avatars=["Female_Child_01"], clips=[], force=True)
    assert report.avatars == ["Female_Child_01"]
    texture = rocketbox.AVATARS["Business_Male_02"].files[1]
    (raw / texture).unlink()
    steps = []
    report = rocketbox.prepare(raw, cache, clips=[], progress=lambda *a: steps.append(a))
    assert report.failed == [("Business_Male_02", f"missing {texture}")]
    assert [s[:3] for s in steps] == [(1, 2, "Female_Child_01"), (2, 2, "Business_Male_02")]
    with pytest.raises(ValueError, match="unknown avatar"):
        rocketbox.prepare(raw, cache, avatars=["Nobody"])


def test_prepare_can_delete_the_raw_files(tmp_path, stub_convert):
    raw, cache = _raw_tree(tmp_path / "raw"), tmp_path / "cache"
    (raw / "Assets" / "keep.txt").write_text("not ours")
    report = rocketbox.prepare(raw, cache, delete_raw=True)
    assert len(report.avatars) == 2 and not report.failed
    assert [p.name for p in raw.rglob("*") if p.is_file()] == ["keep.txt"]
    assert not (raw / "Assets" / "Avatars").exists()
    assert CharacterLibrary(cache).is_ready()


def test_catalog_is_merged_into_the_metadata(tmp_path, stub_convert, monkeypatch):
    raw, cache, catalog = _raw_tree(tmp_path / "raw", clips=[]), tmp_path / "cache", tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"Business_Male_02": {"sex": "male", "age_group": "adult", "skin_tone": 4,
                                                        "outfit": "business", "conference_fit": 3}}))
    monkeypatch.setattr(rocketbox, "CATALOG_PATH", catalog)
    rocketbox.prepare(raw, cache)
    assert json.loads((cache / "avatars" / "Business_Male_02.json").read_text())["skin_tone"] == 4
    catalog.write_text(json.dumps({"Business_Male_02": {"skin_tone": 5}}))
    os.utime(catalog, ns=(time.time_ns(), time.time_ns() + 10**9))
    library = CharacterLibrary(cache)
    assert library.available()["Business_Male_02"]["skin_tone"] == 5
    assert library.load("Business_Male_02").metadata["skin_tone"] == 5
    assert "skin_tone" not in library.available()["Female_Child_01"]


def test_motions_register_the_roles(tmp_path, stub_convert, monkeypatch):
    class MotionLibrary:
        def __init__(self, clips):
            self.clips, self.roles = clips, {}

        def register(self, role, clip_name):
            self.roles.setdefault(role, []).append(clip_name)

    module = types.ModuleType("bifold_demo.characters.behaviour")
    module.MotionLibrary = MotionLibrary
    monkeypatch.setitem(sys.modules, module.__name__, module)
    raw, cache = _raw_tree(tmp_path / "raw", clips=["m_walk_neutral_01", "f_walk_neutral_01", "m_idle_neutral_01",
                                                    "f_turn_left_90"]), tmp_path / "cache"
    rocketbox.prepare(raw, cache)
    library = CharacterLibrary(cache)
    motions = library.motions()
    assert motions is library.motions() and set(motions.clips) == {"m_walk_neutral_01", "f_walk_neutral_01",
                                                                   "m_idle_neutral_01", "f_turn_left_90"}
    assert motions.roles == {"walk": ["m_walk_neutral_01", "f_walk_neutral_01"], "idle": ["m_idle_neutral_01"],
                             "turn_left_90": ["f_turn_left_90"]}
    assert not library.is_ready()
    assert library.missing() == ["turn_right_90", "turn_left_180", "turn_right_180"]
    assert "turn_right_90" in library.describe()


def test_loading_is_fast_and_needs_no_fbx_code(tmp_path, stub_convert):
    raw, cache = _raw_tree(tmp_path / "raw", clips=[]), tmp_path / "cache"
    rocketbox.prepare(raw, cache)
    best = min(_timed_load(cache, "Business_Male_02") for _ in range(3))
    assert best < 0.05
    code = ("import sys; from bifold_demo.characters.library import CharacterLibrary; "
            "lib = CharacterLibrary(sys.argv[1]); lib.load('Business_Male_02'); lib.load('Business_Male_02', 'array'); "
            "bad = [m for m in sys.modules if m.startswith('bifold_demo.characters.fbx')]; "
            "assert not bad, bad")
    subprocess.run([sys.executable, "-c", code, str(cache)], check=True, timeout=60)


def _timed_load(cache, name):
    library = CharacterLibrary(cache)
    start = time.perf_counter()
    library.load(name)
    return time.perf_counter() - start


def test_folder_comes_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("BIFOLD_CHARACTERS", str(tmp_path / "mine"))
    assert default_folder() == tmp_path / "mine" and CharacterLibrary().folder == tmp_path / "mine"
    assert CharacterLibrary(tmp_path / "other").folder == tmp_path / "other"
    monkeypatch.delenv("BIFOLD_CHARACTERS")
    assert default_folder() == Path("~/.cache/bifold-demo/characters").expanduser()
    empty = CharacterLibrary(tmp_path / "empty")
    assert empty.available() == {} and empty.clips() == {} and not empty.is_ready()
    assert "no avatars" in empty.describe() and "--set small" in empty.describe()


# -- command line ----------------------------------------------------------------------------------


def test_cli_parsing():
    parser = cli.build_parser()
    args = parser.parse_args(["fetch"])
    assert (args.set, args.avatars, args.jobs, args.delete_raw, args.dest, args.clips) == \
        ("conference", None, 8, False, None, None)
    args = parser.parse_args(["fetch", "--set", "small", "--avatars", "A", "B", "--dest", "x", "--jobs", "2",
                              "--delete-raw", "--clips", "basic"])
    assert (args.set, args.avatars, args.dest, args.jobs, args.delete_raw, args.clips) == \
        ("small", ["A", "B"], "x", 2, True, "basic")
    assert parser.parse_args(["prepare", "--raw", "r"]).raw == "r"
    assert parser.parse_args(["list", "--all"]).all
    assert parser.parse_args(["info", "Male_Adult_01"]).name == "Male_Adult_01"
    with pytest.raises(SystemExit):
        parser.parse_args(["fetch", "--set", "everyone"])


def test_cli_dry_run_list_and_errors(tmp_path, capsys):
    assert cli.main(["fetch", "--set", "small", "--dry-run", "--dest", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "12 avatars and 24 animation clips" in out and "MB to download" in out
    assert cli.main(["fetch", "--avatars", "Nobody", "--dest", str(tmp_path)]) == 2
    assert "unknown avatar 'Nobody'" in capsys.readouterr().err
    assert cli.main(["list", "--dest", str(tmp_path)]) == 0
    assert "fetch --set small" in capsys.readouterr().out
    assert cli.main(["info", "Business_Female_01", "--dest", str(tmp_path)]) == 1
    assert "not prepared yet" in capsys.readouterr().err
    assert cli.main(["info", "Busines_Female_01", "--dest", str(tmp_path)]) == 1
    assert "did you mean Business_Female_01" in capsys.readouterr().err
    assert cli.main(["prepare", "--dest", str(tmp_path)]) == 1
    assert "bifold-characters fetch" in capsys.readouterr().err


def test_cli_fetch_downloads_prepares_and_lists(tmp_path, stub_convert, monkeypatch, capsys):
    remote = _remote(avatars=["Female_Child_01"], clips=rocketbox.clip_set("basic"))
    monkeypatch.setattr(rocketbox, "_default_opener", lambda: FakeOpener(remote))
    dest = tmp_path / "characters"
    assert cli.main(["fetch", "--avatars", "Female_Child_01", "--clips", "basic", "--dest", str(dest),
                     "--delete-raw"]) == 0
    out = capsys.readouterr().out
    assert "27 files" in out and "[prepare]" in out and out.rstrip().endswith("ready")
    assert not [p for p in (dest / "raw").rglob("*") if p.is_file()]
    assert cli.main(["fetch", "--avatars", "Female_Child_01", "--clips", "basic", "--dest", str(dest)]) == 0
    assert "25 already prepared" in capsys.readouterr().out
    assert cli.main(["list", "--dest", str(dest)]) == 0
    assert "Female_Child_01" in capsys.readouterr().out
    assert cli.main(["info", "Female_Child_01", "--dest", str(dest)]) == 0
    assert "child" in capsys.readouterr().out
    assert cli.main(["info", "f_turn_left_90", "--dest", str(dest)]) == 0
    assert "turn_left_90" in capsys.readouterr().out


# -- the real Rocketbox files (when downloaded) ----------------------------------------------------


REAL_AVATARS = ("Business_Female_01", "Business_Male_02", "Male_Adult_05", "Female_Child_01")


@pytest.fixture(scope="module")
def real_cache(tmp_path_factory):
    if not all((ROCKETBOX.parent / rocketbox.AVATARS[n].fbx).is_file() for n in REAL_AVATARS):
        pytest.skip(f"no local Rocketbox avatars in {ROCKETBOX}")
    try:
        converter = importlib.import_module("bifold_demo.characters.fbx_convert")
    except ImportError as exc:
        pytest.skip(f"no FBX converter: {exc}")
    if not hasattr(converter, "load_character") or not hasattr(converter, "load_clip"):
        pytest.skip("the FBX converter has no load_character / load_clip")
    cache = tmp_path_factory.mktemp("characters")
    report = rocketbox.prepare(ROCKETBOX, cache, avatars=REAL_AVATARS)
    assert not report.failed, report.failed
    assert sorted(report.avatars) == sorted(REAL_AVATARS) and len(report.clips) >= 2
    return cache


def test_real_avatars_load_fast(real_cache):
    library = CharacterLibrary(real_cache)
    assert sorted(library.available()) == sorted(REAL_AVATARS)
    heights = {}
    for name in REAL_AVATARS:
        assert min(_timed_load(real_cache, name) for _ in range(2)) < 0.05
        model = library.load(name)
        assert model.skeleton.find("Bip01 Head") is not None and model.metadata["triangles"] > 1000
        assert all(isinstance(p.texture, str) and Path(p.texture).is_file() for p in model.primitives)
        heights[name] = model.height
    assert 1.0 < heights["Female_Child_01"] < min(heights[n] for n in REAL_AVATARS[:3]) < 2.0
    arrays = library.load("Business_Female_01", textures="array")
    assert all(p.texture.dtype == np.uint8 and p.texture.shape[2] == 4 for p in arrays.primitives)


def test_real_clips_keep_their_metadata(real_cache):
    library = CharacterLibrary(real_cache)
    clips = library.clips()
    walk = clips["m_walk_neutral_01"]
    assert walk.metadata["kind"] == "walk" and 0.5 < walk.metadata["speed"] < 2.0 and walk.root == "Bip01"
    if "m_turn_left_180" in clips:
        assert clips["m_turn_left_180"].metadata["kind"] == "turn" and clips["m_turn_left_180"].loop is False
    roles = library.roles()
    assert "m_walk_neutral_01" in roles["walk"] and "f_idle_neutral_01" in roles["idle"]
    try:
        from bifold_demo.characters.behaviour import MotionLibrary  # noqa: F401
    except ImportError:
        return
    assert library.motions().role("walk", sex="female").name.startswith("f_walk")
