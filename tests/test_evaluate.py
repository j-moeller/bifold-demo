"""The evaluation's manifests, cache keys, stores and scoring (recognition/evaluate.py), with photos and a scripted
stand-in engine (no model, no OpenGL): manifests parsed, checked and refused, every render condition in the key,
caches reused and not stale, splits, the measures of a run, a test pack, the protocols' manifests."""

import json
import pickle
import types

import cv2
import numpy as np
import pytest

from bifold_demo.recognition import evaluate as E
from bifold_demo.recognition import protocols
from bifold_demo.recognition.align import align
from bifold_demo.recognition.detect import Face
from bifold_demo.recognition.embed import Embeddings
from bifold_demo.recognition.engine import FaceRecord
from bifold_demo.recognition.quality import assess

DIM = 32
MARKS = ((80.0, 90.0), (120.0, 90.0), (100.0, 112.0), (84.0, 130.0), (116.0, 130.0))


def _unit(v):
    v = np.asarray(v, np.float64)
    return v / np.linalg.norm(v)


class Embedder:
    """Reads a crop's middle colour: the identity is its red channel, the photo its green (noise)."""

    def __init__(self, people):
        self.people, self.calls = people, 0

    def embed(self, crops):
        self.calls += len(crops)
        out = []
        for crop in np.asarray(crops):
            _b, g, r = (int(v) for v in np.median(crop[40:72, 40:72].reshape(-1, 3), axis=0))
            base = self.people.get(r)
            if base is None:  # (someone nobody enrolled: a face of their own)
                base = _unit(np.random.default_rng(1000 + r).normal(size=DIM))
            noise = np.random.default_rng(g).normal(0, 0.25, DIM)
            out.append(_unit(base + noise))
        out = np.array(out)
        return Embeddings(out, np.ones(len(out)))


class Engine:
    """A stand-in engine: one face in the middle of every picture that is not black."""

    def __init__(self, people, model="tiny", sha="ab" * 32):
        self.embedder = Embedder(people)
        self.model_id = model
        self.dim = DIM
        self.fiqa = None
        self.threshold = 0.5
        self.provider_name = "cpu"
        self.detector_spec = types.SimpleNamespace(id="stand-in", file=types.SimpleNamespace(sha256="cd" * 32))
        self.embedder_spec = types.SimpleNamespace(id=model, file=types.SimpleNamespace(sha256=sha),
                                                   norm_quality=False)

    def detect(self, image, min_size=0.0, canvas=None, region=None):
        if not np.asarray(image).any():
            return []
        s = image.shape[1] / 200.0
        return [Face((60 * s, 60 * s, 80 * s, 100 * s), tuple((x * s, y * s) for x, y in MARKS), 0.9)]

    def align(self, image, face):
        return align(image, face.landmarks)

    def analyse(self, image, faces):
        aligned = self.align(image, faces[0])
        vector = self.embedder.embed(aligned.crop[None]).vectors[0]
        return [FaceRecord(faces[0], aligned, vector, 20.0, assess(faces[0], aligned))]


def _people(n=6, seed=0):
    rng = np.random.default_rng(seed)
    return {40 + 20 * k: _unit(rng.normal(size=DIM)) for k in range(n)}


def _photo(path, red, green, size=200):
    """A photo the stand-ins read: red says who, green which photo; noise in blue gives it detail (a flat picture is
    refused as blurred)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((size, size, 3), np.uint8)
    image[:] = (0, green, red)
    image[..., 0] = np.random.default_rng(red * 256 + green).integers(0, 255, (size, size))
    cv2.imwrite(str(path), image)
    return path


def _photos_manifest(folder, people, probes_each=3, strangers=0):
    """enrol/<p>/ and probe/<p>/ photos of each person (red = who), as protocols.photos lays them out."""
    for k, red in enumerate(sorted(people)):
        name = f"p{k}"
        _photo(folder / "enrol" / name / "a.png", red, 1)
        _photo(folder / "enrol" / name / "b.png", red, 2)
        for j in range(probes_each):
            _photo(folder / "probe" / name / f"{j}.png", red, 10 + j)
    for k in range(strangers):
        _photo(folder / "strangers" / f"s{k}" / "0.png", 250 - k, 5)
    return protocols.photos(folder, scales=(1.0,))


# -- manifests -----------------------------------------------------------------------------------------------------


def _render(item_id, role, identity, **conditions):
    return {"id": item_id, "kind": "still", "role": role, "identity": identity, "source": "render",
            "conditions": conditions}


def test_a_render_gets_every_condition_with_its_default(tmp_path):
    raw = {"format": 1, "name": "m", "defaults": {"light": "side"},
           "items": [_render("g", "gallery", "Ada", ied=100), _render("p", "probe", "Ada", ied=24, yaw=30)]}
    manifest = E.parse(raw, tmp_path)
    probe = manifest.stills("probe")[0]
    assert set(probe.conditions) == set(E.RENDER)
    assert probe.conditions["body"] == "Ada" and probe.conditions["light"] == "side"
    assert probe.conditions["ied"] == 24.0 and probe.conditions["yaw"] == 30.0 and probe.conditions["preset"] == "ideal"
    assert manifest.identities() == ["Ada"]


def test_bad_manifests_are_refused_with_every_problem(tmp_path):
    _photo(tmp_path / "a.png", 40, 1)
    raw = {"format": 1, "items": [
        _render("x", "probe", "Ada", ied=2),  # out of range
        _render("x", "probe", "Ada", yaw=20, colour="red"),  # the id twice, an unknown condition
        _render("y", "probe", "Ada", face="own", body="Bob"),  # Ada's face is not Bob's own
        _render("z", "probe", "Ada", head="fitted"),  # a fitted head without the face made on it
        {"id": "w", "kind": "still", "role": "gallery", "identity": "Ada", "source": "image", "path": "a.png"},
        {"id": "v", "kind": "still", "role": "probe", "identity": "Ada", "source": "image", "path": "a.png"},
        {"id": "u", "kind": "still", "role": "probe", "identity": "Ada", "source": "image", "path": "none.png"},
        {"id": "t", "kind": "still", "role": "visitor", "identity": "", "source": "image", "path": "a.png"},
        {"id": "s", "kind": "movie"},
        _render("r1", "gallery", "Cy", ied=100), _render("r2", "probe", "Cy", ied=100)]}  # the same render twice
    with pytest.raises(E.ManifestError) as caught:
        E.parse(raw, tmp_path)
    text = "\n".join(caught.value.problems)
    for expected in ("ied: 2 is outside", "used twice", "unknown condition 'colour'", "own face is that avatar's",
                     "fitted head wears", "both enrolled and probed", "no such image", "role 'visitor'",
                     "no identity", "unknown kind 'movie'", "a render of Cy is both enrolled and probed"):
        assert expected in text, expected
    assert len(caught.value.problems) >= 11


def test_probes_without_a_gallery_and_bad_settings_are_refused(tmp_path):
    with pytest.raises(E.ManifestError, match="no gallery"):
        E.parse({"format": 1, "items": [_render("p", "probe", "Ada")]}, tmp_path)
    with pytest.raises(E.ManifestError, match="breakdowns"):
        E.parse({"format": 1, "settings": {"breakdowns": [{"by": "ied"}]},
                 "items": [_render("g", "gallery", "Ada"), _render("p", "probe", "Ada", ied=20)]}, tmp_path)
    with pytest.raises(E.ManifestError, match="format"):
        E.parse({"format": 2, "items": [_render("g", "gallery", "Ada")]}, tmp_path)


def test_a_csv_manifest(tmp_path):
    for name in ("g.png", "p.png"):
        _photo(tmp_path / name, 40, 1 if name == "g.png" else 2)
    (tmp_path / "m.csv").write_text("id,kind,role,identity,source,path,light,scale\n"
                                    "g,still,gallery,Ada,image,g.png,window,\n"
                                    "p,still,probe,Ada,image,p.png,window,0.5\n")
    manifest = E.load(tmp_path / "m.csv")
    assert [i.id for i in manifest.items] == ["g", "p"]
    assert manifest.items[1].conditions == {"light": "window", "scale": 0.5}
    with pytest.raises(E.ManifestError, match="scale"):
        (tmp_path / "bad.csv").write_text("id,kind,role,identity,source,path,scale\np,still,probe,Ada,image,p.png,3\n"
                                          "g,still,gallery,Ada,image,g.png,\n")
        E.load(tmp_path / "bad.csv")


# -- keys and stores -----------------------------------------------------------------------------------------------


def test_every_render_condition_is_in_the_key(tmp_path):
    """A condition left out of the key would let a changed still reuse a stale face and embedding."""
    base = E.parse({"format": 1, "items": [_render("g", "gallery", "Ada", ied=100), _render("p", "probe", "Ada")]},
                   tmp_path).stills("probe")[0]
    key = E.item_key(base, tmp_path, {"Ada": "1"})
    other = {"body": "Bob", "face": "renders", "head": "fitted", "shots": "fronts", "view": "frame", "ied": 40.0,
             "distance": 5.0, "side": 0.5, "yaw": 10.0, "pitch": 5.0, "light": "warm", "preset": "good 4K",
             "occluder": "low", "expression": "smile", "expression_weight": 0.3, "quality": "low", "linear": False,
             "detail": 1024, "seed": 3}
    assert set(other) == set(E.RENDER)
    for name, value in other.items():
        changed = E.Item(base.id, base.kind, base.role, base.identity, base.source, None,
                         {**base.conditions, name: value})
        assert E.item_key(changed, tmp_path, {"Ada": "1", "Bob": "2"}) != key, name
    assert E.item_key(base, tmp_path, {"Ada": "2"}) != key  # a re-prepared avatar
    assert E.item_key(base, tmp_path, {"Ada": "1"}) == key


def test_an_image_key_is_its_contents_and_scale(tmp_path):
    path = _photo(tmp_path / "a.png", 40, 1)
    item = E.Item("a", "still", "probe", "Ada", "image", "a.png", {"light": "x"})
    key = E.item_key(item, tmp_path)
    assert E.item_key(E.Item("b", "still", "probe", "Ada", "image", "a.png", {"light": "y"}), tmp_path) == key
    assert E.item_key(E.Item("a", "still", "probe", "Ada", "image", "a.png", {"scale": 0.5}), tmp_path) != key
    _photo(path, 41, 1)
    assert E.item_key(item, tmp_path) != key


def test_model_and_detector_keys_follow_the_files():
    a, b = Engine(_people(), sha="ab" * 32), Engine(_people(), sha="ef" * 32)
    assert E.model_key(a) != E.model_key(b) and E.detector_key(a) == E.detector_key(b)


def test_store_saves_and_resumes(tmp_path):
    store = E.Store(tmp_path / "s.npz", every=2)
    store.put("a", {"v": np.arange(3.0), "f": 1.0})
    assert not (tmp_path / "s.npz").exists() and store.get("a")["f"] == 1.0
    store.put("b", {"v": np.ones(3), "f": 2.0})  # (saved: two new records)
    again = E.Store(tmp_path / "s.npz")
    assert len(again) == 2 and again.get("b")["f"] == 2.0 and "a" in again and again.get("c") is None
    again.put("c", {"v": np.zeros(3), "f": 3.0})
    again.save()
    assert len(E.Store(tmp_path / "s.npz")) == 3
    again.put("d", {"v": np.zeros(3)})
    with pytest.raises(ValueError, match="fields"):
        again.save()


# -- scoring -------------------------------------------------------------------------------------------------------


def test_splits_are_seeded_halvings():
    names = [f"p{k}" for k in range(10)]
    a, b = E.splits(names, 5, seed=1), E.splits(list(reversed(names)), 5, seed=1)
    assert a == b and len(a) == 5
    for enrolled, strangers in a:
        assert len(enrolled) == 5 and not set(enrolled) & set(strangers) and set(enrolled) | set(strangers) == \
            set(names)
    assert E.splits(names, 0) == [(sorted(names), [])]


def _run(tmp_path, people, raw, **options):
    folder = tmp_path / "photos"
    (folder / "manifest.json").write_text(json.dumps(raw))
    manifest = E.load(folder / "manifest.json")
    engine = Engine(people)
    run = E.Run({"tiny": engine}, tmp_path / "out", progress=None, draws=100, **options)
    return run, manifest, engine


def test_a_run_on_photos_and_its_caches(tmp_path):
    people = _people(6)
    raw = _photos_manifest(tmp_path / "photos", people, probes_each=3, strangers=2)
    raw["settings"] = {"splits": 4}
    run, manifest, engine = _run(tmp_path, people, raw)
    first = run.stills(manifest)["tiny"]
    assert first["counts"]["probes"] == 20 and first["counts"]["people"] == 8 and first["counts"]["enrollable"] == 6
    assert first["counts"]["splits"] == 4 and first["counts"]["enrolled_per_split"] == 3
    assert first["acquisition"]["found"]["rate"] == 1.0
    assert first["verification"]["d_prime"] > 3 and first["verification"]["people"] == 8
    assert first["open_set"]["rank1"] == 1.0 and first["open_set"]["people"] == 8
    assert first["open_set"]["searches"] == 4 * 20
    assert "system" not in first  # (no calibration made for the stand-in's model)
    assert set(first["conditions"]) >= {"set", "ied_bin"}
    calls = engine.embedder.calls
    second = run.stills(manifest)["tiny"]  # from the caches: nothing embedded again
    assert engine.embedder.calls == calls

    def same(report):
        return json.dumps({k: v for k, v in report.items() if k != "seconds"}, sort_keys=True, default=str)

    assert same(second) == same(first)  # (deterministic)
    _photo(tmp_path / "photos" / "probe" / "p0" / "0.png", sorted(people)[1], 99)  # p0's probe now shows p1
    third = run.stills(manifest)["tiny"]
    assert engine.embedder.calls == calls + 1 + 0  # (only the changed photo; the gallery's own cache is the gallery's)
    assert third["open_set"]["rank1"] < 1.0


def test_faces_not_found_are_misses_not_left_out(tmp_path):
    people = _people(4)
    raw = _photos_manifest(tmp_path / "photos", people, probes_each=2)
    black = tmp_path / "photos" / "probe" / "p0" / "black.png"
    black.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(black), np.zeros((200, 200, 3), np.uint8))
    raw["items"].append({"id": "black", "kind": "still", "role": "probe", "identity": "p0", "source": "image",
                         "path": "probe/p0/black.png"})
    raw["settings"] = {"splits": 0}
    run, manifest, _ = _run(tmp_path, people, raw)
    report = run.stills(manifest)["tiny"]
    assert report["counts"]["probes"] == 9 and report["counts"]["found"] == 8
    assert report["open_set"]["rank1"] == pytest.approx(8 / 9)  # (the black photo's search misses)
    assert report["conditions"]["ied_bin"]["no face"]["probes"] == 1


def test_breakdowns_group_what_they_say(tmp_path):
    people = _people(4)
    raw = _photos_manifest(tmp_path / "photos", people, probes_each=2)
    for k, item in enumerate(i for i in raw["items"] if i["role"] == "probe"):
        item["conditions"]["light"] = "dim" if k % 2 else "bright"
    raw["settings"] = {"splits": 2, "breakdowns": [{"name": "light", "where": {"set": "probe"}, "by": ["light"]}]}
    run, manifest, _ = _run(tmp_path, people, raw)
    groups = run.stills(manifest)["tiny"]["conditions"]["light"]
    assert sorted(groups) == ["light=bright", "light=dim"]
    assert groups["light=bright"]["probes"] == groups["light=dim"]["probes"] == 4


def test_score_stills_system_decisions():
    """With a calibration: known people's single faces named, strangers' not (clear, synthetic identities)."""
    from bifold_demo.recognition.calibration import Calibration

    rng = np.random.default_rng(0)
    dim = 512  # (random identities' cosines spread about 0.044: inside the calibration's non-mated curve below)
    centres = {f"p{k}": _unit(rng.normal(size=dim)) for k in range(8)}
    items, vectors = [], []
    for name, centre in centres.items():
        for j in range(5):
            items.append(E.Item(f"{name}-{j}", "still", "probe", name, "image", None, {"j": j}))
            vectors.append(_unit(centre + rng.normal(0, 0.02, dim)))
    vectors = np.array(vectors)
    n = len(items)
    stills = E.Stills(items, np.array([i.identity for i in items]), np.ones(n, bool), np.ones(n, bool),
                      np.full(n, 50.0), np.zeros(n), np.ones(n), vectors, np.full(n, np.nan))
    rows = np.array(list(centres.values()))
    owners = np.array(list(centres))
    nonmated = rng.normal(0.0, 0.12, 4000)
    mated = rng.normal(0.85, 0.05, 400)
    calibration = Calibration.fit("tiny", np.r_[nonmated, mated], np.r_[np.zeros(4000, bool), np.ones(400, bool)],
                                  np.full(4400, 50.0), np.zeros(4400), np.arange(4400) % 20)
    report = E.score_stills(stills, rows, owners, calibration, count=6, draws=50, conditions=False)
    at = report["system"]["targets"]["0.01"]
    assert at["known"]["named_right"]["rate"] == 1.0 and at["known"]["named_wrong"]["rate"] == 0.0
    assert at["strangers"]["named"]["rate"] == 0.0
    assert at["strangers"]["achieved_all"]["within"] in ("consistent", "no")
    assert report["counts"]["splits"] == 6 and at["known"]["decisions"] == 6 * 20
    gated = E.Stills(stills.items, stills.identity, stills.found, np.arange(n) % 5 != 0, stills.size, stills.yaw,
                     stills.weight, stills.vectors, stills.fiqa)  # (a fifth of the faces refused by the live gate)
    again = E.score_stills(gated, rows, owners, calibration, count=6, draws=50, conditions=False)
    assert again["system"]["targets"]["0.01"]["known"]["named_right"]["rate"] == pytest.approx(0.8)
    assert again["open_set"]["rank1"] == report["open_set"]["rank1"]  # (the model's own measures: every face found)

    # a face worn by another avatar (its carrier) that shows the carrier's head: never named as the carrier, which is
    # not a person; how much it shows is measured on its own
    worn = [E.Item(f"w{k}", "still", "probe", "p0", "render", None, {"face": "renders", "body": "p1"})
            for k in range(4)]
    vectors_worn = np.array([_unit(0.6 * centres["p1"] + 0.4 * centres["p0"] + rng.normal(0, 0.005, dim))
                             for _ in worn])
    both = E.Stills(items + worn, np.array([i.identity for i in items + worn]), np.ones(n + 4, bool),
                    np.ones(n + 4, bool), np.full(n + 4, 50.0), np.zeros(n + 4), np.ones(n + 4),
                    np.vstack([vectors, vectors_worn]), np.full(n + 4, np.nan))
    mixed = E.score_stills(both, rows, owners, calibration, count=0, draws=50, conditions=False)
    assert mixed["counts"]["worn"] == 4
    assert mixed["carrier"]["head=None"]["carrier_above_own"]["rate"] == 1.0
    assert mixed["carrier"]["head=None"]["named_as_carrier"]["rate"] == 1.0
    assert mixed["system"]["targets"]["0.01"]["known"]["named_wrong"]["rate"] == 0.0
    assert mixed["open_set"]["rank1"] == 1.0


def test_a_face_between_two_lookalikes_is_not_named():
    """Two enrolled people nearly alike and a face between them: both reach the accept threshold, but neither beats
    the other by the margin, so the decision waits (no name), as live."""
    from bifold_demo.recognition.calibration import Calibration

    rng = np.random.default_rng(3)
    dim = 512
    a = _unit(rng.normal(size=dim))
    twin = _unit(a + rng.normal(0, 0.012, dim))  # (cosine about 0.96 with a)
    other = _unit(rng.normal(size=dim))
    rows, owners = np.array([a, twin, other]), np.array(["a", "a twin", "other"])
    probes = [_unit(a + twin + rng.normal(0, 0.01, dim)) for _ in range(4)]
    items = [E.Item(f"p{k}", "still", "probe", "a", "image", None, {}) for k in range(4)]
    stills = E.Stills(items, np.array(["a"] * 4), np.ones(4, bool), np.ones(4, bool), np.full(4, 50.0), np.zeros(4),
                      np.ones(4), np.array(probes), np.full(4, np.nan))
    nonmated = rng.normal(0.0, 0.05, 4000)
    calibration = Calibration.fit("tiny", np.r_[nonmated, rng.normal(0.9, 0.03, 400)],
                                  np.r_[np.zeros(4000, bool), np.ones(400, bool)], np.full(4400, 50.0),
                                  np.zeros(4400), np.arange(4400) % 20)
    report = E.score_stills(stills, rows, owners, calibration, count=0, draws=20, conditions=False)
    at = report["system"]["targets"]["0.01"]["known"]
    assert at["decisions"] == 4 and at["named_right"]["rate"] == 0.0 and at["named_wrong"]["rate"] == 0.0


# -- packs ---------------------------------------------------------------------------------------------------------


def test_a_pack_is_scored_with_a_bootstrap_over_folds(tmp_path):
    people = _people(5)
    reds = sorted(people)
    images, same = [], []
    rng = np.random.default_rng(1)
    for fold in range(10):
        for k in range(4):
            a = reds[(fold + k) % 5]
            b = a if k % 2 == 0 else reds[(fold + k + 1) % 5]
            for red in (a, b):
                image = np.zeros((112, 112, 3), np.uint8)
                image[:] = (30, int(rng.integers(0, 200)), red)
                images.append(cv2.imencode(".png", image)[1].tobytes())
            same.append(a == b)
    pack = tmp_path / "tiny.bin"
    pack.write_bytes(pickle.dumps((images, same)))
    raw = {"format": 1, "name": "pack", "items": [{"id": "tiny", "kind": "pack", "path": "tiny.bin",
                                                   "per_fold": [1, 1]}]}
    manifest = E.parse(raw, tmp_path)
    engine = Engine(people)
    run = E.Run({"tiny": engine}, tmp_path / "out", progress=None, draws=50)
    report = run.packs(manifest)["tiny"]["tiny"]
    assert report["pairs"] == 20 and report["folds"] == 10 and report["genuine"] == 10
    assert report["accuracy"]["accuracy"] > 70 and report["d_prime"] > 1  # (noisy stand-in faces)
    lo, hi = report["d_prime_interval"]
    assert lo < report["d_prime"] < hi
    calls = engine.embedder.calls
    run.packs(manifest)
    assert engine.embedder.calls == calls  # (cached)


# -- protocols -----------------------------------------------------------------------------------------------------


NAMES = ["Business_Female_01", "Female_Adult_04", "Female_Child_01", "Male_Adult_05", "Male_Adult_12",
         "Male_Child_02"]


def test_bodies_keep_sex_and_age():
    wearers = protocols.bodies(NAMES)
    assert all(a != b for a, b in wearers.items())
    assert wearers["Female_Adult_04"] == "Business_Female_01" and wearers["Male_Adult_05"] == "Male_Adult_12"
    assert {wearers["Female_Child_01"], wearers["Male_Child_02"]} == {"Male_Child_02", "Female_Child_01"}


def test_the_protocols_manifests_parse(tmp_path):
    for raw in (protocols.p2(NAMES, "tiny"), protocols.p2(NAMES, "small"), protocols.p2(NAMES),
                protocols.ladder(NAMES)):
        manifest = E.parse(raw, tmp_path)
        assert manifest.identities("gallery") == sorted(NAMES)
        assert set(manifest.identities("probe")) == set(NAMES)
        assert manifest.sides["evaluation"] == sorted(NAMES)
        for item in manifest.stills("probe"):  # set B (the gallery) is never what a probe draws
            assert item.conditions["face"] == "renders" or item.conditions["expression"] == "none"
    full = E.parse(protocols.p2(NAMES), tmp_path)
    per_identity = len(full.stills("probe")) // len(NAMES)
    assert per_identity == 3 * (8 * 3 + 6 + 4 + 6 + 3 + 2)
    ladder = E.parse(protocols.ladder(NAMES), tmp_path)
    rungs = {i.labels["rung"] for i in ladder.stills("probe")}
    assert rungs == {"B0", "B1", "B1=", "B2 good 4K", "B2 budget 1080p", "B3", "B4", "B5"}
    b0 = next(i for i in ladder.stills("probe") if i.labels["rung"] == "B0")
    assert b0.conditions["view"] == "frame" and not b0.conditions["linear"] and b0.conditions["detail"] == 1024


def test_halves_never_enrol_a_carrier_with_the_face_it_wears():
    fair = ["Business_Female_01", "Business_Female_02", "Business_Male_02", "Business_Male_05", "Business_Male_07",
            "Female_Adult_01", "Female_Adult_04", "Female_Adult_06", "Female_Adult_09", "Female_Adult_13",
            "Female_Child_01", "Female_Party_02", "Male_Adult_04", "Male_Adult_05", "Male_Adult_10", "Male_Adult_12",
            "Male_Adult_14", "Male_Adult_17", "Male_Adult_21", "Male_Child_02"]
    wearers = protocols.bodies(fair)
    a, b = protocols.halves(fair)
    assert len(a) == len(b) == 10 and not set(a) & set(b)
    for half in (a, b):
        assert not any(wearers[n] in half for n in half)
    odd = protocols.halves(["Male_Adult_04", "Male_Adult_05", "Male_Adult_10"])  # a cycle of three: one left out
    for half in odd:
        assert not any(protocols.bodies(["Male_Adult_04", "Male_Adult_05", "Male_Adult_10"])[n] in half for n in half)


def test_the_p3_manifest(tmp_path):
    raw = protocols.p3(NAMES + ["Female_Adult_09", "Male_Adult_17"], runs=2, seconds=10)
    manifest = E.parse(raw, tmp_path)
    runs = manifest.of_kind("scene")
    assert len(runs) == 2 and runs[0].extra["known"] != runs[1].extra["known"]
    for run in runs:
        known = set(run.extra["known"])
        lanes = [p for p in run.extra["people"] if p["role"] == "lane"]
        assert lanes and all((p["identity"] in known) == (p["face"] == "renders") for p in lanes)
        assert all(p["body"] not in known for p in run.extra["people"] if p["face"] == "renders")
        assert all(p["identity"] not in known for p in run.extra["people"] if p["role"] == "crosser")
    with pytest.raises(E.ManifestError, match="needs fps"):
        E.parse({"format": 1, "items": [{"id": "s", "kind": "scene", "scene": "parade", "seconds": 5,
                                         "people": []}]}, tmp_path)


def test_a_replay_with_its_own_margin_and_looser_ones():
    from bifold_demo.recognition import evalscene

    rows = [[0.0, False, "Ada", None, None, None, 0.0, "undecided", None, 2.0],
            [0.2, True, "Ada", 3.1, 1.0, 4.0, 1.0, "undecided", None, 9.0],
            [0.4, True, "Ada", 3.6, 1.0, 4.0, 1.0, "undecided", None, 9.0],
            [0.6, True, "Ada", 4.2, 3.5, 4.0, 1.0, "undecided", None, 9.0],  # (the runner-up too close)
            [0.8, True, "Ada", 4.5, 1.0, 4.0, 1.0, "named", "Ada", 9.0]]
    assert evalscene.replay(rows, 0.0)["named_at"] == 0.8
    assert evalscene.replay(rows, -0.5)["named_at"] == 0.4
    assert evalscene.replay(rows, -1.0)["named_at"] == 0.2
    assert evalscene.replay(rows, 1.0)["named_at"] is None
    got = evalscene.replay(rows, 0.0)
    assert got["first_shown"] == 0.2 and got["shown"] == pytest.approx(0.6)


def test_the_margins_offsets():
    """Simulated stranger tracks whose best rarities pass a single decision's threshold for a few pairs: no offset
    lets them through; the point estimate's offset keeps their share at the target; the bootstrap's median about
    there; the 95 % bound further (the few pairs make the share uncertain)."""
    from bifold_demo.recognition import calibrate as cal
    from bifold_demo.recognition import margins
    from bifold_demo.recognition.calibration import per_person

    rng = np.random.default_rng(5)
    n_pairs, per_pair = 60, 20
    accept = -np.log10(per_person(0.01, 10))
    lookalike = np.repeat(np.arange(n_pairs) < 6, per_pair)  # six pairs of lookalikes make the false alarms
    best = accept + np.where(lookalike, rng.uniform(-0.2, 1.5, n_pairs * per_pair), rng.uniform(-4, -1, n_pairs *
                                                                                                    per_pair))
    avatars = np.repeat([f"s{k}" for k in range(n_pairs)], per_pair)
    tracks = cal.StrangerTracks(avatars, np.zeros(len(best), int), np.full(len(best), 10), best,
                                np.repeat([f"k{k}" for k in range(n_pairs)], per_pair), np.arange(len(best)), best)
    chosen = {c: margins.offsets(tracks, c, targets=(0.1, 0.01)) for c in margins.CHOICES}
    at = {c: dict(zip(v["targets"], v["offsets"]))[0.01] for c, v in chosen.items()}
    assert at["none"] == 0.0 and 0.0 < at["point"] <= at["median"] + 0.02 <= at["bound"] + 0.02
    assert at["point"] < at["bound"]
    assert np.mean(tracks.excess(0.01, at["point"]) >= -1e-12) <= 0.01
    assert np.mean(tracks.excess(0.01, at["point"] - 0.02) >= -1e-12) > 0.01
    checked = margins.check(tracks, chosen, draws=200)
    assert checked["none"]["0.01"]["with"]["per_track"] > checked["bound"]["0.01"]["with"]["per_track"]


def test_labels_carried_to_another_profiles_tracks_and_scored(tmp_path):
    from bifold_demo.recognition import evalvideo

    sheet = tmp_path / "s.labels.csv"
    sheet.write_text("track,first,last,frames,faces,shown_by_the_system,label\n1,0,2,3,3,,Ada\n2,0,2,3,3,,stranger\n"
                     "3,0,1,2,0,,\n4,0,1,2,1,,skip\n")
    labels = evalvideo.read_labels(sheet)
    assert labels == {"1": "Ada", "2": "stranger", "4": "skip"}

    def run(tracks):
        return {"tracks": {t: {"boxes": [[k * 0.5, *box] for k in range(3)], "rows": rows, "faces": []}
                           for t, (box, rows) in tracks.items()}}

    named = [[0.0, True, "Ada", 1.0, 0.1, 4.0, 0.5, "undecided", None], [0.5, True, "Ada", 5.0, 0.2, 4.0, 1.0,
                                                                         "named", "Ada"],
             [1.0, True, "Ada", 5.0, 0.2, 4.0, 1.0, "named", "Ada"]]
    waits = [[0.0, True, "Ada", 1.0, 0.1, 4.0, 1.0, "undecided", None], [0.5, True, None, None, None, None, 1.0,
                                                                          "unknown", None],
             [1.5, True, None, None, None, None, 1.0, "unknown", None]]
    reference = run({"1": ((0, 0, 50, 150), named), "2": ((200, 0, 250, 150), waits), "4": ((400, 0, 450, 150), waits)})
    other = run({"7": ((2, 1, 52, 151), named), "8": ((198, 0, 251, 149), waits), "9": ((600, 0, 650, 150), waits),
                 "10": ((40, 0, 90, 150), waits)})
    carried = evalvideo.carry_labels(reference, labels, other)
    assert carried == {"7": "Ada", "8": "stranger"}  # (9 overlaps nobody labelled, 10 Ada's track too little)
    segments = evalvideo.score_video(other | {"tracks": other["tracks"]}, carried, known=["Ada"])
    by = {g.avatar: g for g in segments}
    assert by["Ada"].known and by["Ada"].correct_at == 0.5 and by["Ada"].first_shown == 0.0
    assert not by["stranger"].known and by["stranger"].named_at is None and by["stranger"].unknown_at == 0.5


def test_simulated_tracks_of_held_out_people():
    """Stills of clear identities as tracks: the enrolled named by their first decision (the second face), strangers
    never; an offset out of reach names nobody; a worn face is never named as its carrier."""
    from bifold_demo.recognition import margins
    from bifold_demo.recognition.calibration import Calibration

    rng = np.random.default_rng(4)
    dim = 512
    centres = {f"p{k}": _unit(rng.normal(size=dim)) for k in range(8)}
    items, vectors = [], []
    for name, centre in centres.items():
        for j in range(4):
            items.append(E.Item(f"{name}-{j}", "still", "probe", name, "render", None,
                                {"face": "own", "body": name, "preset": "ideal"}))
            vectors.append(_unit(centre + rng.normal(0, 0.02, dim)))
    n = len(items)
    stills = E.Stills(items, np.array([i.identity for i in items]), np.ones(n, bool), np.ones(n, bool),
                      np.full(n, 30.0), np.zeros(n), np.ones(n), np.array(vectors), np.full(n, np.nan))
    calibration = Calibration.fit("tiny", np.r_[rng.normal(0.0, 0.05, 4000), rng.normal(0.9, 0.03, 400)],
                                  np.r_[np.zeros(4000, bool), np.ones(400, bool)], np.full(4400, 30.0),
                                  np.zeros(4400), np.arange(4400) % 20)
    rows, owners = np.array(list(centres.values())), np.array(list(centres))
    out = margins.simulated_tracks(stills, rows, owners, calibration, None, lambda i: i.conditions["preset"],
                                   {"none": 0.0, "far": 40.0}, count=5, draws=50)
    assert out["none"]["known_named"]["rate"] == 1.0 and out["none"]["faces_to_name"] == 2
    assert out["none"]["strangers_named"]["rate"] == 0.0 and out["none"]["known_tracks"] == 5 * 4
    assert out["far"]["known_named"]["rate"] == 0.0
    # two lookalikes (p0 and its twin) enrolled together, p0's faces between them: never named, by the margin
    twin = _unit(centres["p0"] + rng.normal(0, 0.012, dim))
    between = [_unit(centres["p0"] + twin + rng.normal(0, 0.005, dim)) for _ in range(4)]
    twins = E.Stills(items[:4], stills.identity[:4], np.ones(4, bool), np.ones(4, bool), np.full(4, 30.0),
                     np.zeros(4), np.ones(4), np.array(between), np.full(4, np.nan))
    few = np.array([centres["p0"], twin, centres["p1"], centres["p2"]])
    both = margins.simulated_tracks(twins, few, np.array(["p0", "p0 twin", "p1", "p2"]), calibration, None,
                                    lambda i: i.conditions["preset"], {"none": 0.0}, count=40, draws=50)
    assert both["none"]["known_wrong"]["rate"] == 0.0 and both["none"]["known_named"]["rate"] < 1.0


def test_a_packs_people_from_duplicates_and_mated_pairs():
    images = [b"a1", b"b1", b"a1", b"c1", b"a2", b"a1", b"d1", b"d2"]  # pairs (a1,b1) (a1,c1) (a2,a1) (d1,d2)
    same = [False, False, True, True]
    people = E.pack_people(images, same)
    assert people[0] == people[2] == people[4] == people[5]  # (a1 three times; a2 mated with a1)
    assert people[6] == people[7] and len({people[0], people[1], people[3], people[6]}) == 4


def test_real_faces_per_track():
    """Tracks of a pack's people decided as live: the enrolled people named by their running mean, strangers not; an
    offset beyond reach names nobody."""
    from bifold_demo.recognition.calibration import Calibration

    rng = np.random.default_rng(2)
    dim, vectors, people, digests = 512, {}, [], []
    for person in range(40):
        centre = _unit(rng.normal(size=dim))
        for k in range(3 if person < 30 else 2):
            vectors[len(people)] = _unit(centre + rng.normal(0, 0.03, dim))
            people.append(person)
            digests.append(f"{person}-{k}")
    calibration = Calibration.fit("tiny", np.r_[rng.normal(0.0, 0.05, 4000), rng.normal(0.85, 0.03, 400)],
                                  np.r_[np.zeros(4000, bool), np.ones(400, bool)], np.full(4400, 35.0),
                                  np.zeros(4400), np.arange(4400) % 20)
    out = E.pack_tracks(vectors, np.array(people), digests, calibration, offsets={"none": 0.0, "far": 30.0},
                        count=4, draws=50)
    assert out["people"] == {"three_or_more": 30, "two": 10}
    assert out["none"]["known_named"]["rate"] == 1.0 and out["none"]["strangers_named"]["rate"] == 0.0
    assert out["none"]["known_named"]["n"] == 4 * 15 and out["none"]["strangers_named"]["n"] == 4 * 25
    assert out["far"]["known_named"]["rate"] == 0.0


def test_the_calibrations_hold_none_of_the_fairs_avatars():
    """The shipped calibrations were made from avatars that are not in the fair's library."""
    fair = ["Business_Female_01", "Business_Female_02", "Business_Male_02", "Business_Male_05", "Business_Male_07",
            "Female_Adult_01", "Female_Adult_04", "Female_Adult_06", "Female_Adult_09", "Female_Adult_13",
            "Female_Child_01", "Female_Party_02", "Male_Adult_04", "Male_Adult_05", "Male_Adult_10", "Male_Adult_12",
            "Male_Adult_14", "Male_Adult_17", "Male_Adult_21", "Male_Child_02"]
    sides = protocols.sides(fair)
    assert len(sides["calibrations"]) == 3
    for made in sides["calibrations"].values():
        assert len(made["avatars"]) > 80 and "overlap_with_evaluation" not in made
