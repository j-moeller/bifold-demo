"""The FaceEngine built from a profile: which models it takes (and falls back to), and the records it gives, with tiny
ONNX models standing in for the registry's files (no real model needed)."""

import dataclasses
import hashlib

import numpy as np
import pytest

from bifold_demo.downloads import ModelFile
from bifold_demo.recognition import engine as E
from bifold_demo.recognition import registry
from bifold_demo.recognition.detect import Face

pytest.importorskip("onnx")
from face_models import embedder as tiny_embedder  # (after the check: it needs onnx)


@pytest.fixture
def tiny(tmp_path, monkeypatch):
    """A models folder with tiny stand-ins for SCRFD, buffalo_l's ResNet-50 and eDifFIQA-T (AdaFace is missing), the
    registry pointing at them."""
    monkeypatch.setenv("BIFOLD_FACE_PROVIDER", "cpu")
    monkeypatch.delenv(registry.PROFILE_ENV, raising=False)
    folder = tmp_path / "models"
    folder.mkdir()
    specs = dict(registry.MODELS)
    for spec, dim in ((registry.SCRFD, 4), (registry.BUFFALO_L_R50, 8), (registry.EDIFFIQA_T, 1)):
        path, _ = tiny_embedder(folder / spec.file.name, dim=dim)
        data = path.read_bytes()
        file = ModelFile(spec.file.name, "file:///nowhere", hashlib.sha256(data).hexdigest(), len(data))
        specs[spec.id] = dataclasses.replace(spec, file=file, dim=dim if spec.task != "detector" else 0,
                                             threshold=0.42)
    monkeypatch.setattr(registry, "MODELS", specs)
    return folder


class Faces:
    """A detector that finds the faces it is given."""

    def __init__(self, faces):
        self.faces = faces

    def detect(self, image, min_size=0.0, canvas=None):
        return [f for f in self.faces if f.size >= min_size]


def _face(x0=200.0, eyes=60.0):
    marks = ((x0, 200.0), (x0 + eyes, 200.0), (x0 + eyes / 2, 235.0), (x0 + 5, 265.0), (x0 + eyes - 5, 265.0))
    return Face((x0 - 30, 160.0, eyes + 60, 140.0), marks, 0.9)


def test_the_best_profile_falls_back_when_adaface_is_missing(tiny):
    engine = E.FaceEngine(tiny, download=False, progress=None)
    assert engine.profile.name == "best" and engine.model_id == "buffalo_l_r50" and engine.dim == 8
    assert engine.threshold == 0.42 and engine.provider_name == "cpu"
    assert len(engine.notes) == 1 and "using ArcFace ResNet-50" in engine.notes[0] and "AdaFace" in engine.notes[0]
    assert engine.describe().startswith("best: SCRFD-10G") and "eDifFIQA-T" in engine.describe()
    assert engine._embedder is None  # loaded when first needed
    with pytest.raises(FileNotFoundError, match="face_detection_yunet_2023mar.onnx is missing"):
        E.FaceEngine(tiny, download=False, profile="commercial", progress=None)  # its detector is not there
    (tiny / registry.BUFFALO_L_R50.file.name).unlink()
    with pytest.raises(FileNotFoundError, match="no face recogniser of the profile licensed"):
        E.FaceEngine(tiny, download=False, profile="licensed", progress=None)


def test_records_embed_all_faces_of_an_image_in_one_batch(tiny):
    engine = E.FaceEngine(tiny, download=False, progress=None)
    faces = [_face(200.0), _face(400.0, eyes=30.0)]
    engine.detector = Faces(faces)
    image = np.random.default_rng(0).integers(0, 256, (480, 640, 3), dtype=np.uint8)
    records = engine.analyse(image)
    assert [r.face for r in records] == faces
    for record in records:
        assert record.vector.shape == (8,) and abs(np.linalg.norm(record.vector) - 1.0) < 1e-5
        assert record.aligned.crop.shape == (112, 112, 3) and record.norm > 0
        assert record.quality.ied == pytest.approx(record.face.eye_distance) and record.quality.fiqa is not None
        assert record.quality.norm is None  # a ResNet-50's norm is not a quality signal
    alone = engine.embed(image, faces[1])
    assert alone.dtype == np.float64 and np.allclose(alone, records[1].vector, atol=1e-5)
    pairs = engine.faces(image, min_size=100)
    assert [f for f, _ in pairs] == [faces[0]] and np.allclose(pairs[0][1], records[0].vector, atol=1e-5)
    assert engine.analyse(image, faces=[]) == [] and engine.faces(np.zeros((10, 10, 3), np.uint8), 500) == []
    plain = E.FaceEngine(tiny, download=False, progress=None, quality=False)
    plain.detector = Faces(faces)
    assert plain.analyse(image)[0].quality.fiqa is None and plain.quality_spec is None


def test_a_region_is_searched_alone_and_its_faces_come_back_in_the_frame(tiny):
    engine = E.FaceEngine(tiny, download=False, progress=None)
    seen = []

    class Region(Faces):
        def detect(self, image, min_size=0.0, canvas=None):
            seen.append((image.shape, canvas))
            return super().detect(image, min_size, canvas)

    engine.detector = Region([_face(20.0, eyes=40.0)])  # in the region's pixels
    frame = np.random.default_rng(1).integers(0, 256, (480, 640, 3), dtype=np.uint8)
    (record,) = engine.analyse(frame, region=(300.0, 100.0, 120.0, 150.0))
    assert seen == [((150, 120, 3), E.HEAD_CANVAS)]
    assert record.face.landmarks[0] == (320.0, 300.0) and record.face.box[0] == 290.0  # moved into the frame
    assert record.quality.ied == pytest.approx(40.0)
    assert engine.detect(frame, region=(630.0, 470.0, 50.0, 50.0)) == []  # (only 10 x 10 pixels inside the frame)
