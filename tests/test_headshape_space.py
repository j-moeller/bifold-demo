"""The identity space for fitting heads (ICT-FaceKit Light): fetching, checking, converting and storing it, and the
landmarks on it that ship with the code.

Synthetic tests run on a tiny fake ICT folder served from ``file://`` URLs. The checks of the real space are skipped
without it (``BIFOLD_IDENTITY_SPACE``, see ``bifold-characters identity-space``).
"""

import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

from bifold_demo import downloads
from bifold_demo.characters import cli
from bifold_demo.characters.headshape import space

REAL = Path(os.environ.get("BIFOLD_IDENTITY_SPACE", "/nonexistent")).expanduser()


def _obj(points, faces=()) -> bytes:
    lines = ["# a fake ICT-FaceKit file"] + [f"v {x:.5f} {y:.5f} {z:.5f}" for x, y, z in points]
    lines += ["f " + " ".join(f"{i + 1}/{i + 1}" for i in face) for face in faces]
    return ("\n".join(lines) + "\n").encode()


NEUTRAL = np.array([[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1], [0, 0, 0], [1, 0, 0], [5, 5, 5], [6, 6, 6]], float)
FACES = [(0, 1, 2, 3), (0, 4, 5, 1), (6, 7, 0)]  # a quad, a quad on the head, and one into the dropped parts


@pytest.fixture
def tiny(monkeypatch, tmp_path):
    """A fake ICT-FaceKit of 8 vertices (6 kept: 4 face, 2 head), 2 identity modes and 2 expressions, pinned and
    served from a folder; returns ``(source folder, the files' contents by name)``."""
    rng = np.random.default_rng(3)
    source = tmp_path / "ict" / "FaceXModel"
    source.mkdir(parents=True)
    contents = {"LICENSE": b"MIT License\nCopyright (c) 2020 USC Institute for Creative Technologies\n",
                "generic_neutral_mesh.obj": _obj(NEUTRAL, FACES)}
    shapes = {}
    for name in ("identity000", "identity001", "jawOpen", "eyeBlink_L"):
        shapes[name] = NEUTRAL + rng.normal(0, 0.3, NEUTRAL.shape)
        contents[f"{name}.obj"] = _obj(shapes[name])
    table = []
    for name, data in contents.items():
        rel = name if name == "LICENSE" else f"FaceXModel/{name}"
        (tmp_path / "ict" / rel).write_bytes(data)
        table.append([rel, hashlib.sha256(data).hexdigest(), len(data)])
    monkeypatch.setattr(space, "files", lambda: {"repository": "fake", "commit": "abc", "licence": "MIT",
                                                 "files": table})
    monkeypatch.setattr(space, "VERTICES", 6)
    monkeypatch.setattr(space, "FACE_VERTICES", 4)
    monkeypatch.setattr(space, "ALL_VERTICES", 8)
    monkeypatch.setattr(space, "RAW", (tmp_path / "ict").as_uri() + "/{path}")
    return tmp_path / "ict", shapes


def test_obj_files_are_read_and_their_quads_split_in_two():
    data = _obj(NEUTRAL, FACES)
    assert np.allclose(space.obj_vertices(data), NEUTRAL)
    assert space.obj_faces(data) == [list(f) for f in FACES]
    triangles = space.triangulate(space.obj_faces(data), 6)
    assert triangles.tolist() == [[0, 1, 2], [0, 2, 3], [0, 4, 5], [0, 5, 1]]  # the order kept; the third dropped


def test_the_conversion_keeps_the_face_and_head_in_metres(tiny, tmp_path):
    source, shapes = tiny
    converted = space.convert(_flat(source, tmp_path), tmp_path / "out.npz", progress=None)
    assert converted.vertices == 6 and converted.modes == 2
    assert np.allclose(converted.neutral, NEUTRAL[:6] * 0.01, atol=1e-6)
    assert np.allclose(converted.identity[1], (shapes["identity001"] - NEUTRAL)[:6] * 0.01, atol=1e-6)
    assert converted.expression_names == ("jawOpen", "eyeBlink_L")
    assert np.allclose(converted.expressions[0], (shapes["jawOpen"] - NEUTRAL)[:6] * 0.01, atol=1e-6)
    assert converted.triangles.max() < 6 and "MIT License" in converted.license and converted.commit == "abc"
    again = space.IdentitySpace.read(tmp_path / "out.npz")
    assert np.array_equal(again.identity, converted.identity) and again.expression_names == converted.expression_names
    beta, expression = np.array([1.0, -0.5]), {"jawOpen": 0.25}
    expected = converted.neutral + converted.identity[0] - 0.5 * converted.identity[1] + 0.25 * \
        converted.expressions[0]
    assert np.allclose(again.shape(beta, expression), expected, atol=1e-6)


def _flat(source, tmp_path):
    """ICT's files in one folder by their names (as :func:`space.fetch` keeps them)."""
    flat = tmp_path / "flat"
    flat.mkdir(exist_ok=True)
    for path in source.rglob("*"):
        if path.is_file():
            (flat / path.name).write_bytes(path.read_bytes())
    return flat


def test_fetching_checks_every_file_converts_and_deletes_the_downloads(tiny, tmp_path, monkeypatch):
    dest = tmp_path / "characters" / "identity"
    messages = []
    path = space.fetch(dest, progress=messages.append, jobs=2)
    assert path == dest / space.FILE and path.is_file()
    assert not (dest / "raw").exists(), "the downloaded OBJ files are deleted"
    assert "MIT License" in (dest / space.LICENSE_FILE).read_text()
    assert space.load(dest).modes == 2
    assert any("6 files" in m for m in messages)

    def no_downloads(*args, **kwargs):
        raise AssertionError("nothing is downloaded again")

    monkeypatch.setattr(downloads, "ensure", no_downloads)
    assert space.fetch(dest, progress=None) == path


def test_a_changed_file_upstream_is_refused(tiny, tmp_path):
    source, _shapes = tiny
    path = source / "FaceXModel" / "identity001.obj"
    data = bytearray(path.read_bytes())
    data[-3] = ord("7") if data[-3] != ord("7") else ord("8")  # one digit changed: the size as pinned, not the hash
    path.write_bytes(bytes(data))
    dest = tmp_path / "identity"
    with pytest.raises(downloads.DownloadError):
        space.fetch(dest, progress=None)
    assert not (dest / space.FILE).exists()


def test_a_missing_space_names_the_command_that_fetches_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="bifold-characters identity-space"):
        space.load(tmp_path / "nothing")


def test_a_file_that_is_not_a_space_is_refused(tiny, tmp_path):
    converted = space.convert(_flat(tiny[0], tmp_path), tmp_path / "out.npz", progress=None)
    np.savez(tmp_path / "bad.npz", format=np.array(99), neutral=converted.neutral)
    with pytest.raises(ValueError, match="format 99"):
        space.IdentitySpace.read(tmp_path / "bad.npz")


def test_the_space_is_kept_in_the_characters_folder_unless_told_otherwise(monkeypatch, tmp_path):
    monkeypatch.delenv(space.ENV, raising=False)
    monkeypatch.setenv("BIFOLD_CHARACTERS", str(tmp_path / "chars"))
    assert space.folder() == tmp_path / "chars" / "identity"
    monkeypatch.setenv(space.ENV, str(tmp_path / "elsewhere"))
    assert space.folder() == tmp_path / "elsewhere"
    assert space.folder(tmp_path / "given") == tmp_path / "given"


def test_the_command_line_fetches_into_the_characters_folder(monkeypatch, tmp_path):
    called = {}

    def fake_fetch(path=None, **options):
        called["path"], called["options"] = path, options
        return Path(path) / space.FILE

    monkeypatch.setattr(space, "fetch", fake_fetch)
    assert cli.main(["identity-space", "--dest", str(tmp_path), "--jobs", "3"]) == 0
    assert called["path"] == tmp_path / "identity" and called["options"]["jobs"] == 3


def test_the_landmarks_on_icts_head_ship_with_the_code():
    marks = space.space_landmarks()
    assert marks.vertices.shape == (478, 3) and marks.weights.shape == (478, 3)
    assert (marks.vertices >= 0).all() and (marks.vertices < space.VERTICES).all()
    assert np.allclose(marks.weights.sum(axis=1), 1.0, atol=1e-3) and (marks.weights >= -1e-6).all()
    assert ((marks.reliability >= 0) & (marks.reliability <= 1)).all() and marks.reliability.mean() > 0.6


def test_the_pinned_files_are_icts_at_one_commit():
    table = space.files()
    assert table["licence"] == "MIT" and len(table["commit"]) == 40
    names = [Path(rel).name for rel, _sha, _size in table["files"]]
    assert "generic_neutral_mesh.obj" in names and "LICENSE" in names
    assert sum(n.startswith("identity") for n in names) == 100 and len(space.expression_names()) == 53
    assert all(len(sha) == 64 and size > 0 for _rel, sha, size in table["files"])


# -- the real space ------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real():
    if not (REAL / space.FILE).is_file():
        pytest.skip("set BIFOLD_IDENTITY_SPACE to a folder with the identity space (bifold-characters "
                    "identity-space)")
    return space.load(REAL)


def test_the_real_space_is_icts_light_model(real):
    assert real.vertices == space.VERTICES and real.modes == 100 and len(real.expressions) == 53
    assert real.commit == space.files()["commit"] and "MIT License" in real.license
    rms = np.sqrt((real.identity ** 2).sum(axis=2).mean(axis=1)) * 1000.0  # mm per unit coefficient
    assert 3.0 < rms[0] < 8.0 and rms[-1] < rms[0] / 3  # principal components: unit variance, falling
    assert 0.12 < np.ptp(real.neutral[:, 1]) < 0.4  # a head and neck in metres


def test_the_shipped_landmarks_agree_with_icts_own(real):
    """MediaPipe's landmarks found on ICT's renders lie where ICT's 68 Multi-PIE landmarks are at the eye corners,
    the nose tip and the mouth corners (measured: 0.9-2.4 mm)."""
    marks = space.space_landmarks().points(real.neutral)
    pairs = [(33, 36), (133, 39), (362, 42), (263, 45), (4, 30), (61, 48), (291, 54)]  # MediaPipe, Multi-PIE
    for ours, theirs in pairs:
        assert np.linalg.norm(marks[ours] - real.neutral[space.MULTI_PIE[theirs]]) < 0.0035
