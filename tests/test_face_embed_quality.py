"""Embedders and quality measures, with tiny ONNX models made in the test (no real model needed)."""

import dataclasses
import math

import cv2
import numpy as np
import pytest

from bifold_demo.recognition import embed, quality, registry

pytest.importorskip("onnx")
from face_models import embedder as tiny_embedder  # (after the check: it needs onnx)


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    monkeypatch.setenv("BIFOLD_FACE_PROVIDER", "cpu")


def _spec(**changes):
    base = dataclasses.replace(registry.BUFFALO_L_R50, id="tiny", dim=8, norm_output=None, bgr=False)
    return dataclasses.replace(base, **changes)


def _crops(n=11, seed=1):
    return np.random.default_rng(seed).integers(0, 256, (n, 112, 112, 3), dtype=np.uint8)


def test_an_onnx_embedder_gives_unit_vectors_and_raw_norms_in_any_batch(tmp_path):
    path, weights = tiny_embedder(tmp_path / "e.onnx")
    model = embed.OnnxEmbedder(path, _spec())
    crops = _crops()
    out = model.embed(crops)
    assert out.vectors.shape == (11, 8) and out.vectors.dtype == np.float32 and len(out) == 11
    assert np.allclose(np.linalg.norm(out.vectors, axis=1), 1.0, atol=1e-6)
    # by hand: RGB, (x - 127.5) / 127.5, the mean of each 28 x 28 cell per channel, times the weights
    x = (crops[..., ::-1].astype(np.float64) - 127.5) / 127.5
    cells = x.reshape(11, 4, 28, 4, 28, 3).mean(axis=(2, 4)).transpose(0, 3, 1, 2).reshape(11, 48)
    raw = cells @ weights
    assert np.allclose(out.norms, np.linalg.norm(raw, axis=1), rtol=1e-4)
    assert np.allclose(out.vectors, raw / np.linalg.norm(raw, axis=1, keepdims=True), atol=1e-5)
    alone = model.embed(crops[3])  # one crop gives what it gives in a batch of 11
    assert np.allclose(alone.vectors[0], out.vectors[3], atol=1e-6)
    with pytest.raises(ValueError, match="aligned crops"):
        model.embed(np.zeros((2, 100, 100, 3), np.uint8))


def test_on_the_cpu_a_call_is_one_run_and_with_fixed_shapes_batches_of_8_and_1(tmp_path):
    """Review M6: CPU sessions take any batch size, so the crops of a call go in one run (five crops in one run cost
    a quarter to a third less than five runs); fixed shapes (CoreML) keep batches of 8 and 1. The results are the
    same."""
    from bifold_demo.recognition import runtime

    assert runtime.batches(5, fixed=False) == [slice(0, 5)]
    assert runtime.batches(130, fixed=False) == [slice(0, 64), slice(64, 128), slice(128, 130)]
    assert runtime.batches(11, fixed=True) == [slice(0, 8), slice(8, 9), slice(9, 10), slice(10, 11)]
    path, _ = tiny_embedder(tmp_path / "e.onnx")
    fiqa_path, _ = tiny_embedder(tmp_path / "q.onnx", dim=1)
    embedder = embed.OnnxEmbedder(path, _spec())
    fiqa = quality.FiqaModel(fiqa_path, dataclasses.replace(registry.EDIFFIQA_T, id="tiny_fiqa"))
    sent = []
    for model in (embedder.model, fiqa.model):
        model.run = lambda x, run=model.run: sent.append(len(x)) or run(x)
    crops = _crops(11)
    free = embedder.embed(crops).vectors, fiqa.score(crops)
    assert sent == [11, 11]
    sent.clear()
    embedder.model.fixed = fiqa.model.fixed = True  # as with CoreML
    fixed = embedder.embed(crops).vectors, fiqa.score(crops)
    assert sent == [8, 1, 1, 1] * 2
    assert np.allclose(fixed[0], free[0], atol=1e-6) and np.allclose(fixed[1], free[1], atol=1e-5)


def test_the_colour_order_and_the_norm_read_out_of_the_graph(tmp_path):
    plain, _ = tiny_embedder(tmp_path / "plain.onnx", seed=3)
    normalised, _ = tiny_embedder(tmp_path / "normed.onnx", seed=3, normalise=True)
    crops = _crops(4)
    rgb = embed.OnnxEmbedder(plain, _spec())
    ada = embed.OnnxEmbedder(normalised, _spec(norm_output="norm"))
    assert np.allclose(ada.embed(crops).norms, rgb.embed(crops).norms, rtol=1e-5)  # the length before dividing
    assert np.allclose(ada.raw(crops), rgb.raw(crops), rtol=1e-4, atol=1e-5)
    bgr = embed.OnnxEmbedder(plain, _spec(bgr=True))
    assert np.allclose(bgr.raw(crops), rgb.raw(crops[..., ::-1]), atol=1e-5)
    assert np.allclose(embed.OnnxEmbedder(plain, _spec(), bgr=True).raw(crops), bgr.raw(crops))


def test_flipping_sums_the_raw_embeddings_of_the_crop_and_its_mirror(tmp_path):
    path, _ = tiny_embedder(tmp_path / "e.onnx")
    model = embed.OnnxEmbedder(path, _spec())
    crops = _crops(3)
    both = model.raw(crops) + model.raw(crops[:, :, ::-1])
    out = model.embed(crops, flip=True)
    assert np.allclose(out.vectors, both / np.linalg.norm(both, axis=1, keepdims=True), atol=1e-5)
    assert np.allclose(out.norms, model.embed(crops).norms)  # the crop's own length
    with pytest.raises(ValueError, match="not an embedder"):
        embed.embedder(registry.SCRFD, path)


# -- quality ------------------------------------------------------------------------------------------------------


def _project(yaw, pitch, roll=0.0, scale=3.0, shift=(320.0, 240.0)):
    """The five points of the mean face turned by ``yaw`` and ``pitch`` (the yaw applied last) and leaning by
    ``roll``, seen by a scaled orthographic camera."""
    t, p, r = (math.radians(a) for a in (yaw, pitch, roll))
    elevation = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    azimuth = np.array([[math.cos(t), 0, -math.sin(t)], [0, 1, 0], [math.sin(t), 0, math.cos(t)]])
    model = quality.FACE_3D - quality.FACE_3D.mean(axis=0)
    flat = (model @ (azimuth @ elevation).T)[:, :2] * scale
    lean = np.array([[math.cos(r), -math.sin(r)], [math.sin(r), math.cos(r)]])
    return flat @ lean.T + shift


@pytest.mark.parametrize("yaw, pitch, roll", [(0, 0, 0), (30, 0, 0), (-45, 10, 0), (20, -15, 12), (60, 5, -20)])
def test_the_head_pose_comes_back_from_the_five_points(yaw, pitch, roll):
    template_roll = math.degrees(math.atan2(*(quality.TEMPLATE[1] - quality.TEMPLATE[0])[::-1]))
    est_yaw, est_pitch, est_roll = quality.head_pose(_project(yaw, pitch, roll))
    assert est_yaw == pytest.approx(yaw, abs=0.6) and est_pitch == pytest.approx(pitch, abs=0.6)
    assert est_roll == pytest.approx(roll + template_roll, abs=0.6)


def test_pose_bins():
    assert [quality.pose_bin(y) for y in (0, 19, 25, -25, -54, 55, -70)] == \
        ["frontal", "frontal", "left", "right", "right", "profile", "profile"]


def test_sharpness_drops_with_blur_but_not_with_contrast():
    rng = np.random.default_rng(0)
    texture = cv2.GaussianBlur(rng.normal(128, 40, (112, 112)).astype(np.float32), (0, 0), 1.0)
    crop = np.repeat(np.clip(texture, 0, 255)[..., None], 3, axis=2).astype(np.uint8)
    sharp, brightness, contrast, clipped = quality.image_measures(crop)
    blurred = quality.image_measures(cv2.GaussianBlur(crop, (0, 0), 2.0))[0]
    flat = quality.image_measures((crop.astype(float) * 0.5 + 64).astype(np.uint8))
    assert blurred < sharp / 3 and abs(flat[0] - sharp) < 0.1 * sharp and abs(flat[2] - contrast / 2) < 1.5
    assert 115 < brightness < 140 and clipped < 0.05
    dark = quality.image_measures(np.zeros((112, 112, 3), np.uint8))
    assert dark[3] == 1.0 and dark[0] == 0.0


def test_a_quality_record_and_its_weight():
    from bifold_demo.recognition.align import align
    from bifold_demo.recognition.detect import Face

    marks = _project(25, 0, 0, scale=2.0)
    image = np.random.default_rng(0).integers(0, 256, (480, 640, 3), dtype=np.uint8)
    face = Face((0.0, 0.0, 100.0, 120.0), tuple(map(tuple, marks)), 0.9)
    q = quality.assess(face, align(image, marks), fiqa=0.5, norm=21.0)
    assert q.pose == "left" and q.yaw == pytest.approx(25, abs=0.6) and q.ied == pytest.approx(face.eye_distance)
    assert q.detector == 0.9 and q.fiqa == 0.5 and q.norm == 21.0 and 0 < q.weight < 1
    worse = dataclasses.replace(q, ied=12.0)
    assert worse.weight < q.weight and dataclasses.replace(q, yaw=80.0).weight == 0.0
    assert dataclasses.replace(q, fiqa=None).weight >= q.weight


def test_the_quality_model_scores_each_crop(tmp_path):
    path, weights = tiny_embedder(tmp_path / "q.onnx", dim=1)
    spec = dataclasses.replace(registry.EDIFFIQA_T, id="tiny_fiqa")
    model = quality.FiqaModel(path, spec)
    crops = _crops(9)
    scores = model.score(crops)
    x = (crops[..., ::-1].astype(np.float64) - 127.5) / 127.5
    cells = x.reshape(9, 4, 28, 4, 28, 3).mean(axis=(2, 4)).transpose(0, 3, 1, 2).reshape(9, 48)
    assert scores.shape == (9,) and np.allclose(scores, (cells @ weights).ravel(), atol=1e-4)
