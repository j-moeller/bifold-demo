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
    assert engine._embedder is None and engine.notes == []  # the recogniser is found when first needed
    assert engine.profile.name == "best" and engine.model_id == "buffalo_l_r50" and engine.dim == 8
    assert engine.threshold == 0.42 and engine.provider_name == "cpu"
    assert len(engine.notes) == 1 and "using ArcFace ResNet-50" in engine.notes[0] and "AdaFace" in engine.notes[0]
    assert engine.describe().startswith("best: SCRFD-10G") and "eDifFIQA-T" in engine.describe()
    assert engine.embedder_path == tiny / registry.BUFFALO_L_R50.file.name
    with pytest.raises(FileNotFoundError, match="face_detection_yunet_2023mar.onnx is missing"):
        E.FaceEngine(tiny, download=False, profile="commercial", progress=None)  # its detector is not there
    (tiny / registry.BUFFALO_L_R50.file.name).unlink()
    licensed = E.FaceEngine(tiny, download=False, profile="licensed", progress=None)  # finding faces still works
    hint = "no face recogniser of the profile licensed.*--face-profile commercial"
    with pytest.raises(FileNotFoundError, match=hint):
        licensed.describe()


def _register(folder, spec, data: bytes, sha=None):
    """Put ``data`` in ``folder`` as ``spec``'s file, registered with its size and ``sha`` (default: its own)."""
    (folder / spec.file.name).write_bytes(data)
    file = ModelFile(spec.file.name, "file:///nowhere", sha or hashlib.sha256(data).hexdigest(), len(data))
    registry.MODELS[spec.id] = dataclasses.replace(spec, file=file, threshold=0.33)


def test_a_damaged_or_unloadable_adaface_falls_back_to_the_resnet(tiny):
    good = (tiny / registry.BUFFALO_L_R50.file.name).read_bytes()
    _register(tiny, registry.ADAFACE_IR101, bytes(len(good)), sha=hashlib.sha256(good).hexdigest())  # zeros
    engine = E.FaceEngine(tiny, download=False, progress=None)
    assert engine.model_id == "buffalo_l_r50" and "is damaged: its SHA-256" in engine.notes[0]
    _register(tiny, registry.ADAFACE_IR101, b"not a model at all")  # the right hash, but it does not load
    engine = E.FaceEngine(tiny, download=False, progress=None)
    assert engine.model_id == "buffalo_l_r50" and "AdaFace IR-101" in engine.notes[0]
    assert len(engine.analyse(np.zeros((200, 200, 3), np.uint8), faces=[_face(60.0, 40.0)])) == 1  # it works


def test_a_gated_model_is_used_only_once_a_recorded_check_shows_it_beats_its_gate(tiny, monkeypatch):
    """Review L9: AuraFace (colour order and normalisation assumed) is used by the commercial profile only once
    ``bifold-faces check`` has recorded that it passed and beats SFace on the same test packs. Here a tiny AuraFace is
    gated on the tiny ResNet-50."""
    from bifold_demo.recognition import checks

    data = (tiny / registry.BUFFALO_L_R50.file.name).read_bytes()
    _register(tiny, registry.AURAFACE, data)
    aura = registry.MODELS[registry.AURAFACE.id] = dataclasses.replace(registry.MODELS[registry.AURAFACE.id],
                                                                       gate=registry.BUFFALO_L_R50.id, dim=8)
    r50 = registry.MODELS[registry.BUFFALO_L_R50.id]
    gated = registry.Profile("gated", registry.SCRFD.id, (aura.id, r50.id), None, "a test profile")
    monkeypatch.setitem(registry.PROFILES, "gated", gated)

    def used():
        engine = E.FaceEngine(tiny, download=False, profile="gated", progress=None)
        engine.describe()  # (the recogniser is found when first needed)
        return engine

    engine = used()
    assert engine.model_id == r50.id and "not used until a recorded check shows it beats" in engine.notes[0]
    assert f"bifold-faces check --model {aura.id} {r50.id} --pack lfw.bin" in engine.notes[0]

    def accuracy(spec, value, ok=True):
        return checks.Report(f"{spec.id}: LFW", ok=ok, values={"accuracy": value})

    checks.record(aura, tiny, [checks.Report(f"{aura.id}: file"), accuracy(aura, 99.70)])
    assert "on a test pack" in used().notes[0]  # nothing of the gate model's to compare with
    checks.record(r50, tiny, [accuracy(r50, 99.80)])
    assert used().model_id == r50.id and "99.70 % against" in checks.gate(aura, tiny)
    checks.record(r50, tiny, [accuracy(r50, 99.60)])  # (a later check replaces the same report)
    assert checks.gate(aura, tiny) is None and used().model_id == aura.id and used().notes == []
    checks.record(aura, tiny, [checks.Report(f"{aura.id}: colour order", ok=False)])  # added to the record
    assert "its recorded check failed (auraface_v1: colour order)" in checks.gate(aura, tiny)
    assert used().model_id == r50.id
    registry.MODELS[aura.id] = dataclasses.replace(aura, file=dataclasses.replace(aura.file, sha256="0" * 64))
    assert "not used until" in checks.gate(registry.MODELS[aura.id], tiny)  # a record of another file counts not


def test_finding_faces_needs_no_recogniser_and_a_missing_detector_names_the_commercial_profile(tiny):
    for spec in (registry.BUFFALO_L_R50, registry.EDIFFIQA_T):
        (tiny / spec.file.name).unlink()
    engine = E.FaceEngine(tiny, download=False, progress=None)  # SCRFD alone: enough to find faces
    engine.detector = Faces([_face()])
    assert len(engine.detect(np.zeros((400, 400, 3), np.uint8))) == 1 and engine._embedder is None
    assert engine.quality_spec is None and "no eDifFIQA-T" in engine.notes[0]
    (tiny / registry.SCRFD.file.name).unlink()
    with pytest.raises(FileNotFoundError, match="face detector of the profile best .*--face-profile commercial"):
        E.FaceEngine(tiny, download=False, progress=None)
    assert registry.SCRFD.file.places()[0].member is None  # the 17 MB file first, the 289 MB pack second


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


def test_several_images_and_regions_are_analysed_in_one_batch(tiny):
    """Review L11: the faces of several images (or of several regions of one frame) are embedded in one batch, and
    every record names its model."""
    engine = E.FaceEngine(tiny, download=False, progress=None)
    faces = [_face(200.0), _face(400.0, eyes=30.0)]
    engine.detector = Faces(faces)
    rng = np.random.default_rng(2)
    one, two = (rng.integers(0, 256, (480, 640, 3), dtype=np.uint8) for _ in range(2))
    sent = []
    embed = engine.embedder.embed
    engine.embedder.embed = lambda crops, **kw: sent.append(len(crops)) or embed(crops, **kw)
    many = engine.analyse_many([one, two, one], faces=[faces, faces[:1], None], regions=[None, None, None])
    assert sent == [5] and [len(records) for records in many] == [2, 1, 2]  # (the third: found by the detector)
    alone = engine.analyse(two, faces=faces[:1])
    assert np.allclose(many[1][0].vector, alone[0].vector, atol=1e-5) and many[1][0].model == "buffalo_l_r50"
    assert all(r.model == engine.model_id for records in many for r in records)
    assert engine.analyse_many([one, two], faces=[[], []]) == [[], []] and engine.analyse_many([]) == []
    with pytest.raises(ValueError, match="one each"):
        engine.analyse_many([one, two], faces=[faces])


def test_a_close_up_that_fills_the_photo_is_looked_for_on_smaller_canvases(tiny):
    """Review L2: SCRFD misses a face filling most of a 640 canvas (a tight close-up); when the image's own canvas
    finds nothing the engine tries 640, then 320 (each only if the image is much bigger)."""
    engine = E.FaceEngine(tiny, download=False, progress=None)
    seen = []

    def found_at_320(image, min_size=0.0, canvas=None):
        seen.append(canvas)
        return [_face()] if canvas == 320 else []

    engine.detector.detect = found_at_320
    assert len(engine.detect(np.zeros((1600, 1200, 3), np.uint8))) == 1 and seen == [None, 640, 320]
    seen.clear()
    assert len(engine.detect(np.zeros((700, 500, 3), np.uint8))) == 1 and seen == [None, 320]  # (640: much the same)
    seen.clear()
    assert engine.detect(np.zeros((300, 300, 3), np.uint8)) == [] and seen == [None]
    seen.clear()
    assert engine.detect(np.zeros((1600, 1200, 3), np.uint8), canvas=640) == [] and seen == [640]  # as asked
    seen.clear()
    assert engine.detect(np.zeros((1600, 1200, 3), np.uint8), region=(0, 0, 400, 400)) == []
    assert seen == [E.HEAD_CANVAS]  # a head region: no fallback


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
