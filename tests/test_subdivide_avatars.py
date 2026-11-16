"""Subdivided heads (step S4) on the real avatars: the S3 rig checks in the 12 faces, the closed mouth, the neck's
edge, a fitted head, the face rig.

Skipped without a prepared character library (``BIFOLD_CHARACTERS_TEST_DIR``). No OpenGL, MediaPipe or identity space
needed.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from bifold_demo.characters import faces
from bifold_demo.characters import subdivide as sd
from bifold_demo.characters.character import skin_vertices
from bifold_demo.characters.headshape import head as s0
from bifold_demo.characters.headshape import mesh, rig

CHARACTERS = Path(os.environ.get("BIFOLD_CHARACTERS_TEST_DIR", "/nonexistent")).expanduser()
ADULT, CHILD = "Male_Adult_05", "Male_Child_02"


@pytest.fixture(scope="module")
def library():
    if not CHARACTERS.is_dir():
        pytest.skip("set BIFOLD_CHARACTERS_TEST_DIR to a prepared character library")
    from bifold_demo.characters.library import CharacterLibrary

    library = CharacterLibrary(CHARACTERS)
    if any(name not in library.available() for name in (ADULT, CHILD)):
        pytest.skip(f"the character library lacks {ADULT} or {CHILD}")
    return library


@pytest.fixture(scope="module")
def heads(library, tmp_path_factory):
    os.environ.setdefault("BIFOLD_HEADTEX_CACHE", str(tmp_path_factory.mktemp("headtex")))
    out = {}
    for name in (ADULT, CHILD):
        model = library.load(name)
        out[name] = (model, sd.head_labels(model))
    return out


def test_the_labels_of_a_real_head(heads):
    for name, (model, labels) in heads.items():
        assert (labels.eye == 1).sum() == 73 and (labels.eye == 2).sum() == 73
        assert (labels.inner > 0).sum() > 50 and (labels.parts == s0.SKIN).sum() > 1000
        assert (labels.tie >= 0).sum() > 20  # the closed lips and the lids' corners
        made = sd.subdivision(model, 1, labels)
        counts = made.counts()
        assert counts["crease"] > 60 and counts["sphere"] > 300 and counts["curved"] > 3000, (name, counts)


@pytest.mark.parametrize("level", [1, 2])
def test_a_subdivided_head_passes_the_rigs_checks_in_the_12_faces(heads, level):
    """No eyeball through a lid, no teeth through the lips, no mouth opened that was shut, the neck's edge where it
    was and no triangle folded, in each face of the rig seen from the front and the side, against the coarse head's
    own surface (measured: 1.5-4.6 s per head and level)."""
    for name, (model, labels) in heads.items():
        check = sd.check(model, level, labels)
        assert check.ok, (name, check.problems)
        assert len(check.shown) == 2 * (1 + len(faces.EXPRESSIONS))


def test_the_closed_mouth_stays_closed_and_the_neck_does_not_move(heads):
    for name, (model, labels) in heads.items():
        for level in (1, 2):
            curved, flat = sd.subdivision(model, level, labels), sd.subdivision(model, level, labels, smooth=False)
            q, f = curved.primitive.positions.astype(np.float64), flat.primitive.positions.astype(np.float64)
            tie = curved.labels.tie
            coarse = labels.tie
            p = model.primitives[labels.primitive].positions.astype(np.float64)
            old = max(np.ptp(p[coarse == g], axis=0).max() for g in np.unique(coarse[coarse >= 0]))
            new = max(np.ptp(q[tie == g], axis=0).max() for g in np.unique(tie[tie >= 0]))
            assert new <= old + 1e-6, (name, level, new, old)  # the lips touch where they touched, no further apart
            assert (tie[len(p):] >= 0).sum() > 10
            neck = curved.labels.fade <= 0
            assert neck.sum() > (labels.fade <= 0).sum() and np.abs(q[neck] - f[neck]).max() == 0.0
            assert _off_open_edges(model.primitives[labels.primitive], labels, curved.primitive,
                                   curved.labels) < 1e-6


def _off_open_edges(coarse, coarse_labels, fine, fine_labels) -> float:
    """How far the subdivided head's open edges lie from the coarse head's (metres, at most), leaving out the
    eyeballs' open backs (on their spheres): the neck's new points on its edges."""
    def open_points(prim, labels):
        node = mesh.weld(prim.positions)[0]
        tri = np.asarray(prim.indices, np.int64)
        tri = tri[(labels.eye[tri] == 0).all(axis=1)]
        edges = mesh.open_edges(node[tri])
        where = np.zeros((node.max() + 1, 3))
        where[node] = prim.positions
        return where, edges

    where, edges = open_points(coarse, coarse_labels)
    fine_where, fine_edges = open_points(fine, fine_labels)
    points = fine_where[np.unique(fine_edges)]
    a, b = where[edges[:, 0]][None], where[edges[:, 1]][None]
    ab = b - a
    t = np.clip(np.einsum("pek,pek->pe", points[:, None] - a, ab) / np.einsum("pek,pek->pe", ab, ab), 0.0, 1.0)
    return float(np.linalg.norm(points[:, None] - (a + t[..., None] * ab), axis=2).min(axis=1).max())


def test_a_fitted_head_is_subdivided_after_its_shape(heads):
    """A head given a shape (here: the nose and the cheeks moved by a few millimetres, faded to the neck) and put on
    the rig is subdivided like the avatar's own: its own vertices kept, the checks passed."""
    model, labels = heads[ADULT]
    prim = model.primitives[labels.primitive]
    points = np.asarray(prim.positions, np.float64)
    from bifold_demo.characters.headtex.landmarks import face_frame

    frame = face_frame(model)
    bump = np.exp(-np.sum((points - frame.nose) ** 2, axis=1) / 0.015 ** 2)
    offsets = np.zeros_like(points)
    offsets[:, 2] = 0.003 * bump * labels.fade
    offsets[:, 0] = 0.002 * np.sign(points[:, 0] - frame.nose[0]) * np.exp(
        -((points[:, 1] - frame.nose[1]) / 0.03) ** 2) * labels.fade
    offsets[labels.parts != s0.SKIN] = 0.0
    shape = rig.Shape(model.name, rig.base_key(model), prim.name, offsets.astype(np.float32))
    fitted = rig.apply_shape(model, shape)
    made = sd.subdivide(fitted, 1, labels)
    head = made.primitives[labels.primitive]
    assert np.array_equal(head.positions[:len(points)], fitted.primitives[labels.primitive].positions)
    check = sd.check(fitted, 1, labels)
    assert check.ok, check.problems


def test_the_subdivided_model_poses_as_the_model(heads):
    """The same skeleton and face rig: in a face of the rig the subdivided head's own vertices are where the coarse
    head's are, bit for bit, and its new ones lie on the posed coarse surface's curved version."""
    model, labels = heads[CHILD]
    made = sd.subdivide(model, 2, labels)
    assert faces.face_rig(made) is faces.face_rig(model)
    from bifold_demo.characters.model import Poser

    local = Poser(model).local_pose([])
    faces.apply_face(local, faces.face_rig(model), {"surprise": 1.0})
    palette = model.skinning_matrices(model.skeleton.global_transforms(local))
    coarse = skin_vertices(model, palette)[labels.primitive]
    fine = skin_vertices(made, palette)[labels.primitive]
    assert np.array_equal(fine[:len(coarse)], coarse)
    flat = skin_vertices(sd.subdivide(model, 2, labels, smooth=False), palette)[labels.primitive]
    gap = np.linalg.norm(fine[len(coarse):] - flat[len(coarse):], axis=1)
    print("posed: new points off the flat surface, median and most (mm):", np.median(gap) * 1000, gap.max() * 1000)
    assert np.median(gap) < 0.0005 and gap.max() < 0.004
