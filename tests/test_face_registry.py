"""The face models: checked downloads (zip packs, fallbacks), the registry and its profiles, ONNX Runtime sessions.

Everything here runs offline: downloads come from ``file://`` URLs and the ONNX models are tiny ones made in the
test (``face_models``)."""

import hashlib
import io
import os
import re
import zipfile

import numpy as np
import pytest

from bifold_demo import downloads
from bifold_demo.downloads import ModelFile, Source, member_of
from bifold_demo.recognition import registry, runtime


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _zip(path, members: dict) -> Source:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    path.write_bytes(buffer.getvalue())
    return Source(path.as_uri(), sha256=_sha(buffer.getvalue()), size=len(buffer.getvalue()))


@pytest.fixture
def counted(monkeypatch):
    """Counts the downloads (``urlopen`` calls) by URL."""
    counts = {}
    real = downloads.urllib.request.urlopen

    def urlopen(url, *args, **kwargs):
        counts[url] = counts.get(url, 0) + 1
        return real(url, *args, **kwargs)

    monkeypatch.setattr(downloads.urllib.request, "urlopen", urlopen)
    return counts


# -- downloads ------------------------------------------------------------------------------------------------------


def test_a_direct_download_is_checked_and_kept(tmp_path, counted):
    data = b"weights" * 100
    (tmp_path / "remote.onnx").write_bytes(data)
    model = ModelFile("model.onnx", (tmp_path / "remote.onnx").as_uri(), _sha(data), len(data))
    folder = tmp_path / "models"
    with pytest.raises(FileNotFoundError, match="run: bifold-faces fetch"):
        downloads.ensure(model, folder, download=False, fetch_hint="bifold-faces fetch")
    path = downloads.ensure(model, folder, progress=None)
    assert path.read_bytes() == data and downloads.ensure(model, folder, progress=None) == path
    assert sum(counted.values()) == 1  # the second time from the folder
    wrong = ModelFile("other.onnx", model.url, "0" * 64, len(data))
    with pytest.raises(downloads.DownloadError, match="checksum") as caught:
        downloads.ensure(wrong, folder, progress=None)
    assert "other.onnx" in caught.value.errors
    assert sorted(p.name for p in folder.iterdir()) == ["model.onnx"]  # nothing half-written is left behind


def test_members_of_a_zip_pack_come_out_of_one_download(tmp_path, counted):
    det, rec = b"detector" * 50, b"recogniser" * 80
    pack = _zip(tmp_path / "pack.zip", {"pack/det.onnx": det, "pack/rec.onnx": rec, "pack/other.onnx": b"x"})
    files = [member_of(pack, "det.onnx", _sha(det), len(det), member="pack/det.onnx"),
             member_of(pack, "my_rec.onnx", _sha(rec), len(rec), member="pack/rec.onnx")]
    folder = tmp_path / "models"
    found = downloads.ensure_many(files, folder, progress=None)
    assert found["det.onnx"].read_bytes() == det and found["my_rec.onnx"].read_bytes() == rec
    assert counted == {pack.url: 1}  # the pack once, for both
    assert sorted(p.name for p in folder.iterdir()) == ["det.onnx", "my_rec.onnx"]  # the pack itself is gone


def test_a_bad_member_falls_back_to_a_mirror_and_a_changed_pack_is_refused(tmp_path, counted):
    good = b"the real weights" * 30
    pack = _zip(tmp_path / "pack.zip", {"m.onnx": b"something else" * 30})
    (tmp_path / "mirror.onnx").write_bytes(good)
    model = member_of(pack, "m.onnx", _sha(good), len(good), mirrors=((tmp_path / "mirror.onnx").as_uri(),))
    path = downloads.ensure(model, tmp_path / "a", progress=None)
    assert path.read_bytes() == good and counted == {pack.url: 1, (tmp_path / "mirror.onnx").as_uri(): 1}
    changed = Source(pack.url, sha256="1" * 64, size=pack.size)  # the pack is not the one pinned
    with pytest.raises(downloads.DownloadError, match="checksum"):
        downloads.ensure(member_of(changed, "m.onnx", _sha(good), len(good)), tmp_path / "b", progress=None)
    missing = member_of(pack, "absent.onnx", _sha(good), len(good))
    with pytest.raises(downloads.DownloadError, match="absent.onnx"):
        downloads.ensure(missing, tmp_path / "c", progress=None)
    assert not any((tmp_path / "b").iterdir()) and not any((tmp_path / "c").iterdir())


def test_a_member_is_written_under_its_own_name_only(tmp_path):
    data = b"payload" * 20
    pack = _zip(tmp_path / "pack.zip", {"../../escape.onnx": data})
    model = member_of(pack, "safe.onnx", _sha(data), len(data), member="../../escape.onnx")
    path = downloads.ensure(model, tmp_path / "models", progress=None)
    assert path == tmp_path / "models" / "safe.onnx" and not (tmp_path / "escape.onnx").exists()


def test_a_file_of_unknown_size_is_known_by_its_hash(tmp_path):
    data = b"auraface" * 40
    (tmp_path / "remote.onnx").write_bytes(data)
    model = ModelFile("model.onnx", (tmp_path / "remote.onnx").as_uri(), _sha(data), None)
    folder = tmp_path / "models"
    folder.mkdir()
    (folder / "model.onnx").write_bytes(b"damaged")  # there, but not the file: downloaded again
    assert downloads.present(model, folder) is None
    assert downloads.ensure(model, folder, progress=None).read_bytes() == data
    assert downloads.present(model, folder) == folder / "model.onnx"


# -- the registry -------------------------------------------------------------------------------------------------


def test_every_model_is_pinned_and_described():
    for spec in registry.MODELS.values():
        assert re.fullmatch(r"[0-9a-f]{64}", spec.file.sha256), spec.id
        assert spec.file.size is None or spec.file.size > 0
        assert spec.file.size is not None or spec.needs, spec.id  # only a file we cannot see here lacks its size
        for source in spec.file.places():
            assert source.url.startswith("https://") and (source.member is None or source.size)
        assert spec.task in ("detector", "embedder", "quality") and spec.form in ("onnxruntime", "opencv")
        assert spec.licence and spec.data and spec.origin and spec.size[0] > 0
        assert (spec.dim > 0) == (spec.task != "detector")
        if spec.norm_output:
            assert spec.form == "onnxruntime" and spec.task == "embedder"
        assert (0.0 < spec.threshold < 1.0) == (spec.task == "embedder"), spec.id  # a provisional cut each
    assert len({spec.file.name for spec in registry.MODELS.values()}) == len(registry.MODELS)  # no clashes
    assert registry.MODELS["adaface_ir101_wf12m"].bgr and not registry.MODELS["buffalo_l_r50"].bgr


def test_the_profiles():
    assert set(registry.PROFILES) == {"best", "licensed", "commercial"}
    for p in registry.PROFILES.values():
        assert registry.model(p.detector).task == "detector"
        assert p.embedders and all(registry.model(e).task == "embedder" for e in p.embedders)
        assert p.quality is None or registry.model(p.quality).task == "quality"
    best = registry.PROFILES["best"]
    assert best.detector == "scrfd_10g_kps" and best.embedders == ("adaface_ir101_wf12m", "buffalo_l_r50")
    assert registry.PROFILES["licensed"].embedders == ("buffalo_l_r50",)
    commercial = registry.PROFILES["commercial"]
    assert commercial.embedders == ("auraface_v1", "sface_2021dec") and commercial.quality is None
    assert all(registry.model(m).commercial for m in commercial.models())  # by the weights' licences
    assert registry.model("buffalo_l_r50").profiles == ("best", "licensed")
    assert registry.model("antelopev2_r100").profiles == () and registry.model("antelopev2_r100").optional


def test_the_profile_is_chosen_by_name_or_environment(monkeypatch):
    monkeypatch.delenv(registry.PROFILE_ENV, raising=False)
    assert registry.profile_name() == "best" and registry.profile().name == "best"
    monkeypatch.setenv(registry.PROFILE_ENV, "Commercial")
    assert registry.profile_name() == "commercial" and registry.profile_name("licensed") == "licensed"
    with pytest.raises(ValueError, match="unknown face profile"):
        registry.profile_name("fastest")
    with pytest.raises(ValueError, match="unknown face model"):
        registry.model("nothing")


def test_fetching_a_profile_takes_the_fallback_when_the_first_choice_cannot_be_had(tmp_path, monkeypatch):
    import dataclasses

    def fake(spec, data, reachable=True):
        remote = tmp_path / "remote" / spec.file.name
        remote.parent.mkdir(exist_ok=True)
        if reachable:
            remote.write_bytes(data)
        return dataclasses.replace(spec, file=ModelFile(spec.file.name, remote.as_uri(), _sha(data), len(data)))

    specs = {"yunet_2023mar": fake(registry.YUNET, b"det"), "auraface_v1": fake(registry.AURAFACE, b"a", False),
             "sface_2021dec": fake(registry.SFACE, b"sface")}
    monkeypatch.setattr(registry, "MODELS", {**registry.MODELS, **specs})
    found = registry.fetch("commercial", tmp_path / "models", progress=None)
    assert found["sface_2021dec"] == tmp_path / "models" / registry.SFACE.file.name
    assert isinstance(found["auraface_v1"], OSError) and "huggingface.co" in str(found["auraface_v1"])
    for gone in (tmp_path / "remote", tmp_path / "models"):  # now SFace cannot be had either
        (gone / registry.SFACE.file.name).unlink()
    with pytest.raises(OSError, match="auraface_v1"):  # no embedder at all
        registry.fetch("commercial", tmp_path / "models", progress=None)


def test_bifold_faces_list_shows_the_models_with_their_licences(tmp_path, capsys, monkeypatch):
    from bifold_demo.recognition import cli

    monkeypatch.delenv(registry.PROFILE_ENV, raising=False)
    assert cli.main(["--models", str(tmp_path), "--face-profile", "commercial", "list", "--only", "models"]) == 0
    out = capsys.readouterr().out
    assert "profile: commercial" in out and "* yunet_2023mar" in out and "  scrfd_10g_kps" in out
    assert "Apache-2.0; commercial use allowed by the licence" in out and "research use only" in out
    assert "not fetched, needs huggingface.co" in out


# -- ONNX Runtime -------------------------------------------------------------------------------------------------


def test_importing_the_recognition_switches_onnx_runtime_telemetry_off():
    import bifold_demo.recognition  # noqa: F401

    assert os.environ.get("ORT_DISABLE_TELEMETRY") == "1"
    assert runtime.ort() is runtime.ort()


def test_batches_are_cut_into_fixed_sizes():
    assert [s.stop - s.start for s in runtime.chunks(11)] == [8, 1, 1, 1]
    assert [s.stop - s.start for s in runtime.chunks(16)] == [8, 8] and runtime.chunks(0) == []
    assert [s.stop - s.start for s in runtime.chunks(5, sizes=(4,))] == [4, 1]


def test_an_onnx_model_runs_with_extra_outputs_and_fixed_shapes(tmp_path, monkeypatch):
    pytest.importorskip("onnx")
    from face_models import doubler

    monkeypatch.setenv(runtime.PROVIDER_ENV, "cpu")
    path = doubler(tmp_path / "doubler.onnx")
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    model = runtime.OnnxModel(path, (2, 3))
    assert model.output_names == ["y"] and not model.fixed and model.providers_used == ["CPUExecutionProvider"]
    assert np.allclose(model.run(x)[0], 2 * x + 1) and np.allclose(model.run(x[:1])[0], 2 * x[:1] + 1)
    inside = runtime.OnnxModel(path, (2, 3), extra_outputs=("twice",), fixed=True)
    y, twice = inside.run(x)
    assert np.allclose(twice, 2 * x) and np.allclose(y, 2 * x + 1)
    inside.run(x[:1])
    assert sorted(inside._sessions) == [(1, 3), (2, 3)]  # one session per shape
    with pytest.raises(ValueError, match="no tensor"):
        runtime.OnnxModel(path, (2, 3), extra_outputs=("nothing",))
    with pytest.raises(ValueError, match="BIFOLD_FACE_PROVIDER"):
        runtime.provider_choice("gpu")
