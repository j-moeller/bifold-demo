"""The model checks: reading test packs safely, the verification protocol, separation, and the file and graph checks
(with tiny ONNX models; no real model needed)."""

import dataclasses
import hashlib
import pickle

import cv2
import numpy as np
import pytest

from bifold_demo.downloads import ModelFile
from bifold_demo.recognition import checks, registry


def _pack(path, images, same):
    path.write_bytes(pickle.dumps((images, list(same)), protocol=4))
    return path


def test_a_pack_is_read_as_plain_data_and_anything_else_is_refused(tmp_path):
    jpeg = cv2.imencode(".jpg", np.full((112, 112, 3), 90, np.uint8))[1].tobytes()
    small = cv2.imencode(".png", np.full((56, 56, 3), 200, np.uint8))[1].tobytes()
    images, same = checks.read_pack(_pack(tmp_path / "lfw.bin", [jpeg, small, jpeg, jpeg], [True, False]))
    assert len(images) == 4 and same.tolist() == [True, False]
    crops = checks.decode(images)
    assert crops.shape == (4, 112, 112, 3) and abs(int(crops[1].mean()) - 200) <= 1

    class Evil:
        def __reduce__(self):  # unpickling this would call a function
            return (print, ("ran code",))

    (tmp_path / "evil.bin").write_bytes(pickle.dumps(([Evil()], [True]), protocol=2))
    with pytest.raises(ValueError, match="refused"):
        checks.read_pack(tmp_path / "evil.bin")
    with pytest.raises(ValueError, match="pairs"):
        checks.read_pack(_pack(tmp_path / "odd.bin", [jpeg] * 3, [True]))


def test_the_verification_protocol_and_separation():
    rng = np.random.default_rng(0)
    same = np.tile([True, False], 300)
    scores = np.where(same, rng.normal(0.7, 0.1, 600), rng.normal(0.1, 0.1, 600))
    result = checks.verification(scores, same)
    assert 99.0 < result["accuracy"] <= 100.0 and 0.3 < result["threshold"] < 0.5 and result["pairs"] == 600
    assert checks.verification(np.where(same, 1.0, 0.0), same)["accuracy"] == 100.0
    sep = checks.separation(scores[same], scores[~same])
    assert sep["d_prime"] == pytest.approx(6.0, abs=0.6) and sep["eer"] < 0.01
    assert sep["mated"] == 300 and sep["nonmated"] == 300
    overlap = checks.separation([0.1, 0.2, 0.3, 0.4], [0.15, 0.25, 0.35, 0.45])
    assert 0.2 <= overlap["eer"] <= 0.6
    nonmated = np.linspace(0.0, 0.999, 1000)
    t = checks.threshold_at(nonmated, 0.01)
    assert np.mean(nonmated >= t) <= 0.01 and np.mean(nonmated >= t - 0.002) > 0.01
    assert checks.pair_scores(np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8], [0.6, 0.8]])).tolist() == [0.0, 1.0]


def _registered(path, spec, **changes):
    data = path.read_bytes()
    file = ModelFile(path.name, "file:///nowhere", hashlib.sha256(data).hexdigest(), len(data))
    return dataclasses.replace(spec, file=file, **changes)


def test_file_and_graph_checks(tmp_path, monkeypatch):
    pytest.importorskip("onnx")
    import onnx
    from face_models import embedder

    monkeypatch.setenv("BIFOLD_FACE_PROVIDER", "cpu")
    path, _ = embedder(tmp_path / "tiny.onnx", dim=8, normalise=True)
    spec = _registered(path, registry.ADAFACE_IR101, dim=8, opset=13, norm_output="norm",
                       inputs=(registry.Tensor("input", ("batch_size", 3, 112, 112)),),
                       outputs=(registry.Tensor("output", ("batch_size", 8)),))
    assert checks.check_file(spec, tmp_path).ok
    report = checks.check_graph(spec, path)
    assert report.ok, str(report)
    assert "the same alone as in a batch" in str(report) and report.values["opset"] == 13
    wrong = dataclasses.replace(spec, opset=11, outputs=(registry.Tensor("output", ("batch_size", 512)),),
                                norm_output="nothing")
    lines = str(checks.check_graph(wrong, path))
    assert "opset 13, not 11" in lines and "outputs" in lines and "no tensor nothing" in lines
    model = onnx.load(str(path))
    model.graph.node[0].domain = "com.example"  # a custom operator
    onnx.save(model, str(tmp_path / "custom.onnx"))
    custom = checks.check_graph(spec, tmp_path / "custom.onnx")
    assert not custom.ok and "custom operators: ['com.example']" in str(custom)
    (tmp_path / "tiny.onnx").write_bytes(path.read_bytes()[:-1] + b"x")  # changed by a byte
    assert "SHA-256" in str(checks.check_file(spec, tmp_path)) and not checks.check_file(spec, tmp_path).ok
    assert "is missing" in str(checks.check_file(spec, tmp_path / "nowhere"))


def test_the_colour_order_and_published_accuracy_checks(tmp_path, monkeypatch):
    pytest.importorskip("onnx")
    from face_models import embedder

    monkeypatch.setenv("BIFOLD_FACE_PROVIDER", "cpu")
    path, _ = embedder(tmp_path / "tiny.onnx", dim=8, seed=4)
    spec = _registered(path, registry.BUFFALO_L_R50, dim=8)
    rng = np.random.default_rng(3)
    # "people" are patterns in the red channel only: in the registered order (red first) they are told apart
    base = [rng.integers(0, 256, (112, 112), dtype=np.uint8) for _ in range(4)]
    crops, labels = [], []
    for k, pattern in enumerate(base):
        for _ in range(3):
            crop = rng.integers(100, 110, (112, 112, 3), dtype=np.uint8)
            crop[..., 2] = np.clip(pattern + rng.integers(-5, 6, (112, 112)), 0, 255)
            crops.append(crop)
            labels.append(f"p{k}")
    report = checks.colour_order(spec, path, np.array(crops), np.array(labels))
    assert report.values["RGB"]["d_prime"] != report.values["BGR"]["d_prime"]
    encoded = [cv2.imencode(".png", c)[1].tobytes() for c in crops[:8]]
    pack = _pack(tmp_path / "lfw.bin", encoded, [True, False, True, False])
    accuracy = checks.published_accuracy(spec, path, pack)
    assert accuracy.name == "buffalo_l_r50: LFW" and 0.0 <= accuracy.values["accuracy"] <= 100.0


def test_bifold_faces_check_reports_a_missing_model(tmp_path, capsys):
    from bifold_demo.recognition import cli

    assert cli.main(["--models", str(tmp_path), "check", "--model", "sface_2021dec"]) == 1
    out = capsys.readouterr().out
    assert "sface_2021dec: file: FAILED" in out and "is missing" in out and "0 of 1 checks passed" in out
    assert not (tmp_path / checks.RECORDS).exists()  # nothing recorded of a missing file
    assert cli.main(["--models", str(tmp_path), "check", "--model", "auraface_v1"]) == 1
    assert "auraface_v1: not used until a recorded check shows it beats SFace" in capsys.readouterr().out
