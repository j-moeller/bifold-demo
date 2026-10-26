"""Funny faces: the expressions on a synthetic head with the Rocketbox face bones, and on every real avatar."""

import gc
import itertools
import math
import os
import time
from pathlib import Path

import numpy as np
import pytest
from test_yoga import synthetic_biped

from bifold_demo.characters import faces, rocketbox
from bifold_demo.characters.character import _FastPoser, skin_vertices
from bifold_demo.characters.model import CharacterModel, Skeleton, SkinnedPrimitive
from bifold_demo.characters.yoga import axis_rotation

# a prepared character library (bifold-characters fetch --dest DIR) for the real-avatar tests; they skip without
CHARACTERS = Path(os.environ.get("BIFOLD_CHARACTERS_TEST_DIR", "/nonexistent")).expanduser()

# rest positions (character space, metres) of the face bones of a Rocketbox woman, rounded: (name, parent, position)
FACE = [("MJaw", "Head", (0, 1.551, -0.04)), ("MBottomLip", "MJaw", (0, 1.539, 0.067)),
        ("MTongue", "MJaw", (0, 1.542, 0.034)), ("MUpperLip", "Head", (0, 1.552, 0.06)),
        ("MMiddleEyebrow", "Head", (0, 1.624, 0.054)), ("MNose", "Head", (0, 1.584, 0.068))]
FACE_SIDES = [("Eye", "Head", (0.03, 1.612, 0.029)), ("MouthBottom", "MJaw", (0.014, 1.541, 0.066)),
              ("Masseter", "Head", (0.055, 1.553, 0.021)), ("Caninus", "Head", (0.028, 1.564, 0.052)),
              ("EyeBlinkBottom", "Head", (0.033, 1.605, 0.039)), ("Upperlip", "Head", (0.014, 1.551, 0.058)),
              ("MouthCorner", "Head", (0.026, 1.544, 0.048)), ("Cheek", "Head", (0.04, 1.586, 0.045)),
              ("EyeBlinkTop", "Head", (0.032, 1.619, 0.039)), ("InnerEyebrow", "Head", (0.025, 1.628, 0.05)),
              ("OuterEyebrow", "Head", (0.047, 1.625, 0.038))]
SYMMETRIC = [e for e in faces.EXPRESSIONS if e not in ("wink", "sideways")]


def synthetic_head(turned: bool = False, scale: float = 1.0) -> CharacterModel:
    """:func:`synthetic_biped` with the Rocketbox face bones added below its head (no mesh on them).

    ``turned`` gives every face bone (and the head) a different, random rest rotation, as real skeletons have;
    ``scale`` shrinks the whole person (a child).
    """
    body = synthetic_biped()
    sk = body.skeleton
    names = list(sk.names)
    parents = list(sk.parents)
    positions = [sk.rest_global[b, :3, 3] for b in range(len(sk))]
    for name, parent, position in FACE:
        names.append("Bip01 " + name)
        parents.append(names.index("Bip01 " + parent))
        positions.append(np.array(position, float))
    for name, parent, (x, y, z) in FACE_SIDES:
        for side, sign in (("L", 1), ("R", -1)):
            names.append(f"Bip01 {side}{name}")
            parent_name = "Bip01 " + parent if parent in ("Head", "MJaw") else f"Bip01 {side}{parent}"
            parents.append(names.index(parent_name))
            positions.append(np.array((sign * x, y, z), float))
    rng = np.random.default_rng(7)
    head = names.index("Bip01 Head")
    glob = np.repeat(np.eye(4)[None], len(names), axis=0)
    for k, position in enumerate(positions):
        glob[k, :3, 3] = np.asarray(position) * scale
        if turned and k >= head:
            glob[k, :3, :3] = axis_rotation(rng.normal(size=3), rng.uniform(20, 160))
    rest = np.array([glob[k] if p < 0 else np.linalg.inv(glob[p]) @ glob[k] for k, p in enumerate(parents)])
    skeleton = Skeleton(names, parents, rest)
    primitives = []
    for prim in body.primitives:  # the boxes of the body (their bones keep their indices)
        primitives.append(SkinnedPrimitive(prim.name, prim.positions * np.float32(scale), prim.normals, prim.uvs,
                                           prim.indices, prim.joints, prim.weights, prim.texture))
    return CharacterModel("synthetic_head", skeleton, primitives, np.linalg.inv(skeleton.rest_global))


@pytest.fixture(scope="module")
def head():
    return synthetic_head()


def _local(model):
    return _FastPoser(model).local_pose([])


def _posed(model, weights, local=None):
    """Character-space bone transforms of ``model`` making ``weights``."""
    poser = _FastPoser(model)
    local = poser.local_pose([]) if local is None else local.copy()
    faces.apply_face(local, faces.face_rig(model), weights)
    return poser.forward_kinematics(local)


def _at(model, glob, name):
    return glob[model.skeleton.index("Bip01 " + name), :3, 3]


def _moved(model, weights, name):
    """How far bone ``name`` moved (character space) for ``weights``."""
    return _at(model, _posed(model, weights), name) - _at(model, _posed(model, {}), name)


def _turned(model, weights, name) -> float:
    """Degrees bone ``name`` turned (character space) for ``weights``."""
    b = model.skeleton.index("Bip01 " + name)
    r = _posed(model, weights)[b, :3, :3] @ _posed(model, {})[b, :3, :3].T
    return math.degrees(math.acos(np.clip((np.trace(r) - 1) / 2, -1, 1)))


# -- the rig ---------------------------------------------------------------------------------------------------


def test_expressions_are_described_and_include_the_basic_faces():
    required = {"smile", "grin", "tongue", "puff", "surprise", "scrunch", "fish", "wink"}
    assert required <= set(faces.EXPRESSIONS)
    assert all(isinstance(text, str) and len(text) > 20 for text in faces.EXPRESSIONS.values())
    assert set(faces.FAR_FACES) <= set(faces.EXPRESSIONS)
    assert "tongue" in faces.FAR_FACES and "puff" not in faces.FAR_FACES  # puffed cheeks only show close up


def test_face_rig_needs_the_face_bones(head):
    assert faces.face_rig(synthetic_biped()) is None
    rig = faces.face_rig(head)
    assert rig is not None and faces.face_rig(head) is rig  # worked out once
    assert len(rig.bones) == len(faces.FACE_BONES) == 28
    assert rig.size == pytest.approx(0.06 / faces.ADULT_EYES)
    assert faces.face_rig(synthetic_head(scale=0.8)).size == pytest.approx(0.8 * rig.size)


def test_no_weights_leave_the_pose_untouched(head):
    rig = faces.face_rig(head)
    local = _local(head)
    local[5, :3, :3] = axis_rotation((1, 2, 3), 30)  # some pose
    before = local.copy()
    for weights in ({}, {"smile": 0.0}, {"grin": 0, "tongue": 0.0, "wink": -1.0}, {"dance": 1.0}, {"puff": float("nan")}):
        faces.apply_face(local, rig, weights)
        assert local.tobytes() == before.tobytes(), weights


def test_unknown_names_are_ignored_and_weights_clipped(head):
    np.testing.assert_array_equal(_posed(head, {"smile": 0.7, "moonwalk": 1.0}), _posed(head, {"smile": 0.7}))
    np.testing.assert_array_equal(_posed(head, {"surprise": 3.0}), _posed(head, {"surprise": 1.0}))


@pytest.mark.parametrize("name", list(faces.EXPRESSIONS))
def test_every_expression_moves_the_face(head, name):
    moves = [np.linalg.norm(_moved(head, {name: 1.0}, bone)) for bone in faces.FACE_BONES]
    turns = [_turned(head, {name: 1.0}, bone) for bone in ("MJaw", "MTongue", "LEye", "REye")]
    assert max(moves) > 0.006 or max(turns) > 10, (moves, turns)
    assert max(moves) < 0.07  # nothing leaves the head (the chin moves most, with the jaw)
    for bone in faces.FACE_BONES:  # bones only slide and turn: the transforms stay rigid ...
        b = head.skeleton.index("Bip01 " + bone)
        r = _posed(head, {name: 1.0})[b, :3, :3]
        stretch = faces._STRETCH.get(name, {}).get(bone)
        if stretch is None:
            np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-12)
        else:  # ... but the stuck-out tongue, which also gets longer and narrower
            np.testing.assert_allclose(np.linalg.svd(r, compute_uv=False), sorted(stretch, reverse=True), atol=1e-12)


def test_the_jaw_opens_for_the_open_mouthed_faces(head):
    for name, least in (("surprise", 20), ("roar", 25), ("tongue", 10), ("grin", 5), ("cross_eyed", 5)):
        assert _turned(head, {name: 1.0}, "MJaw") > least, name
        assert _moved(head, {name: 1.0}, "MBottomLip")[1] < -0.01 * least / 20, name  # the chin goes down
    for name in ("smile", "puff", "wink", "pout"):
        assert _turned(head, {name: 1.0}, "MJaw") == 0.0, name


def test_the_tongue_comes_out_late_over_the_lower_lip(head):
    glob = _posed(head, {"tongue": 1.0})
    sk = head.skeleton
    tongue = sk.index("Bip01 MTongue")
    tip = glob[tongue] @ head.inverse_bind[tongue] @ np.array([0, 1.5424, 0.0426, 1.0])  # its tip vertex
    lip = _at(head, glob, "MBottomLip")
    assert tip[2] > lip[2] + 0.003  # out in front of the lower lip (whose skin is behind its bone) ...
    assert tip[1] < lip[1] - 0.002 and tip[1] < _at(head, glob, "MTongue")[1] - 0.01  # ... hanging down over it
    rest_tip = np.linalg.norm(np.array([0, 1.5424, 0.0426]) - head.skeleton.rest_global[tongue, :3, 3])
    assert np.linalg.norm(tip[:3] - _at(head, glob, "MTongue")) == pytest.approx(1.65 * rest_tip, rel=0.01)  # longer
    out = {w: _moved(head, {"tongue": w}, "MTongue") for w in (0.25, 0.5, 1.0)}
    assert out[0.25][2] < 0.15 * out[1.0][2] and out[0.5][2] < 0.4 * out[1.0][2]  # comes out once the jaw is open
    jaw = sk.index("Bip01 MJaw")

    def lift(w):  # the tongue's height over its rest place on the jaw
        glob = _posed(head, {"tongue": w})
        return (np.linalg.inv(glob[jaw]) @ glob[tongue])[1, 3] - head.skeleton.rest_local[tongue, 1, 3]

    assert lift(0.5) > 0.01  # it goes over the lower teeth on the way


def test_the_eyelids_close_for_scrunch_and_the_left_one_for_wink(head):
    for side in "LR":
        assert _moved(head, {"scrunch": 1.0}, side + "EyeBlinkTop")[1] < -0.011
        assert _moved(head, {"scrunch": 1.0}, side + "EyeBlinkBottom")[1] > 0.005
        assert _moved(head, {"surprise": 1.0}, side + "EyeBlinkTop")[1] > 0.002  # eyes wide
    assert _moved(head, {"wink": 1.0}, "LEyeBlinkTop")[1] < -0.013
    assert _moved(head, {"wink": 1.0}, "REyeBlinkTop")[1] == 0.0
    assert _moved(head, {"wink": 1.0}, "LMouthCorner")[1] > 0.01  # that mouth corner goes up


def test_brows_and_corners_go_where_the_faces_say(head):
    for side, sign in (("L", 1), ("R", -1)):
        smile = _moved(head, {"smile": 1.0}, side + "MouthCorner")
        assert smile[1] > 0.012 and sign * smile[0] > 0.012  # up and out
        assert _moved(head, {"surprise": 1.0}, side + "InnerEyebrow")[1] > 0.008
        assert _moved(head, {"scrunch": 1.0}, side + "InnerEyebrow")[1] < -0.008
        assert sign * _moved(head, {"puff": 1.0}, side + "Masseter")[0] > 0.03  # cheeks blown out ...
        for bone in ("Masseter", "Cheek", "Caninus", "MouthCorner", "MouthBottom"):  # ... all of them beside the mouth,
            moved = _moved(head, {"puff": 1.0}, side + bone)
            assert sign * moved[0] > 0.008 and moved[2] > 0.005 and np.linalg.norm(moved) < 0.04, bone  # none far
        assert sign * _moved(head, {"fish": 1.0}, side + "MouthCorner")[0] < -0.01  # lips pursed ...
    assert _moved(head, {"fish": 1.0}, "MUpperLip")[2] > 0.012  # ... forward
    assert _moved(head, {"scrunch": 1.0}, "MNose")[1] > 0.008


def test_raised_brows_slide_up_the_forehead(head):
    for name in ("surprise", "fish", "tongue", "pout"):
        for bone in ("LInnerEyebrow", "RInnerEyebrow"):
            moved = _moved(head, {name: 1.0}, bone)
            assert moved[1] > 0.003 and moved[2] < faces.FOREHEAD * moved[1] + 0.0011, (name, bone)  # up and back
    down = _moved(head, {"scrunch": 1.0}, "LInnerEyebrow")
    assert down[1] < -0.008 and down[2] > 0.0  # lowered brows come forward over the brow ridge


@pytest.mark.parametrize("name", faces.EXPRESSIONS)
def test_symmetric_faces_are_mirror_images(head, name):
    glob = _posed(head, {name: 1.0})
    mirror = np.diag([-1.0, 1.0, 1.0])
    pairs = [(f"L{n}", f"R{n}") for n, _, _ in FACE_SIDES]
    gaps = [np.abs(mirror @ _at(head, glob, left) - _at(head, glob, right)).max() for left, right in pairs]
    left_eye, right_eye = (glob[head.skeleton.index(f"Bip01 {s}Eye"), :3, 2] for s in "LR")  # where they look
    gaps.append(np.abs(mirror @ left_eye - right_eye).max())
    if name in SYMMETRIC:
        assert max(gaps) < 1e-9
    else:
        assert max(gaps) > 0.005


def test_the_rest_frames_of_the_bones_do_not_matter(head):
    turned = synthetic_head(turned=True)
    weights = {"grin": 0.6, "tongue": 0.7, "cross_eyed": 0.5, "wink": 0.4}
    for bone in faces.FACE_BONES:
        np.testing.assert_allclose(_moved(turned, weights, bone), _moved(head, weights, bone), atol=1e-9)
        assert _turned(turned, weights, bone) == pytest.approx(_turned(head, weights, bone), abs=1e-4)


def test_children_make_smaller_faces(head):
    child = synthetic_head(scale=0.8)
    for bone in ("LMouthCorner", "MUpperLip", "LEyeBlinkTop", "MTongue"):
        np.testing.assert_allclose(_moved(child, {"smile": 1.0, "tongue": 1.0}, bone),
                                   0.8 * _moved(head, {"smile": 1.0, "tongue": 1.0}, bone), atol=1e-9)
    assert _turned(child, {"surprise": 1.0}, "MJaw") == pytest.approx(_turned(head, {"surprise": 1.0}, "MJaw"))


def test_two_faces_at_half_weight_mix_halfway(head):
    rig = faces.face_rig(head)
    mix = rig.offsets({"smile": 0.5, "grin": 0.5})
    smile, grin = rig.offsets({"smile": 1.0}), rig.offsets({"grin": 1.0})
    for k in range(2):
        np.testing.assert_allclose(mix[k], 0.5 * smile[k] + 0.5 * grin[k], atol=1e-15)
    for name in faces.EXPRESSIONS:  # half a face is half way, except the tongue's late and midway parts
        if name != "tongue":
            np.testing.assert_allclose(rig.offsets({name: 0.5})[0], 0.5 * rig.offsets({name: 1.0})[0], atol=1e-15)


def test_the_tongue_out_holds_the_lower_lip(head):
    rig = faces.face_rig(head)
    lip = [list(rig.bones).index(head.skeleton.index("Bip01 " + n)) for n in ("MBottomLip", "LMouthBottom")]

    def slides(weights):
        return rig.offsets(weights)[0][lip]

    for other in ("fish", "pout", "smile"):  # the tongue's own lower lip, not pushed up or forward into it
        np.testing.assert_allclose(slides({"tongue": 1.0, other: 1.0}), slides({"tongue": 1.0}), atol=1e-15)
        half = slides({"tongue": 0.5, other: 1.0}) - slides({"tongue": 0.5})
        np.testing.assert_allclose(half, 0.75 * slides({other: 1.0}), atol=1e-15)  # it gives way as the tongue's out


def test_faces_follow_the_pose_of_the_head(head):
    poser = _FastPoser(head)
    local = poser.local_pose([])
    neck = head.skeleton.index("Bip01 Neck")
    local[neck, :3, :3] = axis_rotation((0, 1, 0), 90) @ local[neck, :3, :3]  # looking to the left
    glob_turned, glob_rest = _posed(head, {"smile": 1.0}, local), _posed(head, {"smile": 1.0})
    corner = head.skeleton.index("Bip01 LMouthCorner")
    turned_move = glob_turned[corner, :3, 3] - poser.forward_kinematics(local)[corner, :3, 3]
    rest_move = glob_rest[corner, :3, 3] - _posed(head, {})[corner, :3, 3]
    np.testing.assert_allclose(turned_move, axis_rotation((0, 1, 0), 90) @ rest_move, atol=1e-12)


def test_apply_face_is_cheap(head):
    rig = faces.face_rig(head)
    poses = [_local(head) for _ in range(200)]  # a fresh pose each time, as every frame (the tongue's stretch adds up)
    weights = {"grin": 0.4, "tongue": 0.6, "wink": 0.3, "cross_eyed": 0.2}
    best = math.inf
    for _ in range(5):
        start = time.perf_counter()
        for local in poses:
            faces.apply_face(local, rig, weights)
        best = min(best, (time.perf_counter() - start) / len(poses))
        poses = [_local(head) for _ in range(200)]
    assert best < 1e-4  # seconds per call; it runs every frame for every face-making person


def _with_stray_gum(model) -> tuple[CharacterModel, SkinnedPrimitive]:
    """``model`` with a mouth primitive: a tongue of two triangles and a bit of gum skinned to the tongue by mistake."""
    sk = model.skeleton
    tongue, jaw = sk.index("Bip01 MTongue"), sk.index("Bip01 MJaw")
    positions = np.array([[0, 0, 0], [0.01, 0, 0], [0, 0, 0.01], [0.01, 0, 0.01],  # the tongue: two triangles
                          [0.03, 0, 0], [0.04, 0, 0], [0.03, 0, 0.01]], np.float32)  # a stray bit of gum
    joints = np.zeros((7, 4), np.int64)
    weights = np.zeros((7, 4), np.float32)
    joints[:, 0], weights[:, 0] = tongue, 1.0
    joints[5, 0] = jaw
    prim = SkinnedPrimitive("mouth", positions, np.zeros_like(positions), np.zeros((7, 2), np.float32),
                            np.array([[0, 1, 2], [1, 3, 2], [4, 5, 6]], np.uint32), joints, weights)
    return CharacterModel("mouth", sk, [*model.primitives, prim], model.inverse_bind), prim


def test_mend_tongue_gives_stray_tongue_vertices_to_the_jaw(head):
    sk = head.skeleton
    tongue, jaw = sk.index("Bip01 MTongue"), sk.index("Bip01 MJaw")
    model, prim = _with_stray_gum(head)
    assert rocketbox.mend_tongue(model) == 2
    assert prim.joints[:4, 0].tolist() == [tongue] * 4 and prim.joints[4:, 0].tolist() == [jaw] * 3
    assert rocketbox.mend_tongue(model) == 0
    assert rocketbox.mend_tongue(synthetic_biped()) == 0  # no face bones: nothing to do


def test_avatars_are_mended_when_read_and_face_rig_changes_nothing(head, tmp_path):
    model, _ = _with_stray_gum(head)
    rocketbox.save_character(model, tmp_path)
    read = rocketbox.read_character(tmp_path, "mouth", textures="none")
    assert read.primitives[-1].joints[4:, 0].tolist() == [head.skeleton.index("Bip01 MJaw")] * 3  # before any use
    joints = [p.joints.copy() for p in read.primitives]
    assert faces.face_rig(read) is not None
    assert all(np.array_equal(a, p.joints) for a, p in zip(joints, read.primitives))  # it only reads the model


def test_face_rigs_go_with_their_models():
    model = synthetic_head()
    assert faces.face_rig(model) is not None
    key = id(model)
    assert key in faces._RIGS
    del model
    gc.collect()
    assert key not in faces._RIGS  # a rig does not keep its model alive


def _with_cheeks(model, scarf: float | None) -> CharacterModel:
    """``model`` with a patch of skin on each Masseter bone and, ``scarf`` metres beside it, a still vertex."""
    sk = model.skeleton
    positions, joints = [], []
    for side, sign in (("L", 1), ("R", -1)):
        bone = sk.index(f"Bip01 {side}Masseter")
        for dy, dz in ((0, 0), (0.01, 0), (0, 0.01), (0.01, 0.01)):
            positions.append((sign * 0.06, 1.55 + dy, 0.02 + dz))
            joints.append(bone)
        if scarf is not None:
            positions.append((sign * (0.06 + scarf), 1.555, 0.025))
            joints.append(sk.index("Bip01 Head"))
    n = len(positions)
    weights = np.zeros((n, 4), np.float32)
    weights[:, 0] = 1.0
    prim = SkinnedPrimitive("cheeks", np.array(positions, np.float32), np.zeros((n, 3), np.float32),
                            np.zeros((n, 2), np.float32), np.array([[0, 1, 2], [1, 3, 2]], np.uint32),
                            np.c_[joints, np.zeros((n, 3), np.int64)], weights)
    return CharacterModel("cheeks", sk, [*model.primitives, prim], model.inverse_bind)


def test_cheeks_do_not_bulge_through_a_headscarf(head):
    free, scarf = _with_cheeks(head, 0.01), _with_cheeks(head, 0.0003)
    bones = {name: head.skeleton.index("Bip01 " + name) for name in faces.FACE_BONES}
    assert faces.cheek_room(free, bones) == pytest.approx(0.01, abs=1e-6)
    assert faces.cheek_room(_with_cheeks(head, None), bones) == math.inf
    assert not faces.face_rig(free).covered and faces.face_rig(scarf).covered
    for side, sign in (("L", 1), ("R", -1)):
        wide, held = _moved(free, {"puff": 1.0}, side + "Masseter"), _moved(scarf, {"puff": 1.0}, side + "Masseter")
        assert sign * held[0] == pytest.approx(faces.COVERED_SHARE * sign * wide[0])  # less out sideways
        assert held[2] == pytest.approx(wide[2])  # as far forward


# -- every face on real avatars --------------------------------------------------------------------------------


def _real_avatars():
    from bifold_demo.characters.library import CharacterLibrary

    if not (CHARACTERS / "avatars").is_dir():
        pytest.skip("set BIFOLD_CHARACTERS_TEST_DIR to a prepared character library")
    library = CharacterLibrary(CHARACTERS)
    names = sorted(library.available())
    if not names:
        pytest.skip("the prepared library has no avatars")
    return [library.load(name, textures="none") for name in names]


@pytest.fixture(scope="module")
def avatars():
    return _real_avatars()


class _Head:
    """The head mesh of a real avatar and the vertex groups the checks look at."""

    def __init__(self, model):
        self.model = model
        sk = model.skeleton
        self.rig = faces.face_rig(model)
        self.poser = _FastPoser(model)
        self.bone = {name: sk.index("Bip01 " + name) for name in faces.FACE_BONES}
        self.index = max(range(len(model.primitives)), key=lambda i: self.weight(model.primitives[i], "MJaw").sum())
        prim = model.primitives[self.index]
        rest = self.rest = prim.positions.astype(np.float64)
        s = self.rig.size
        w = {name: self.weight(prim, name) for name in faces.FACE_BONES}
        head_weight = self.weight(prim, None, sk.index("Bip01 Head"))
        mid = np.abs(rest[:, 0]) < 0.004 * s
        self.upper_mid, self.lower_mid = mid & (w["MUpperLip"] > 0.5), mid & (w["MBottomLip"] > 0.5)
        self.upper_lip = w["MUpperLip"] + w["LUpperlip"] + w["RUpperlip"] > 0.3
        self.lower_lip = w["MBottomLip"] + w["LMouthBottom"] + w["RMouthBottom"] > 0.3
        self.tongue = w["MTongue"] > 0.5
        self.eyes = {}
        for side in "LR":
            near = np.abs(rest[:, 0] - sk.rest_global[self.bone[side + "Eye"], 0, 3]) < 0.006 * s
            self.eyes[side] = near & (w[side + "EyeBlinkTop"] > 0.5), near & (w[side + "EyeBlinkBottom"] > 0.25)
        lip_y = 0.5 * (sk.rest_global[self.bone["MUpperLip"], 1, 3] + sk.rest_global[self.bone["MBottomLip"], 1, 3])
        lip_z = sk.rest_global[self.bone["MBottomLip"], 2, 3]
        mouth = (np.abs(rest[:, 1] - lip_y) < 0.012 * s) & (rest[:, 2] < lip_z + 0.01 * s) & \
            (np.abs(rest[:, 0]) < 0.02 * s)
        self.lower_teeth = mouth & (w["MJaw"] > 0.99) & (rest[:, 2] > rest[self.tongue, 2].max())
        self.upper_teeth = mouth & (head_weight > 0.99) & (rest[:, 2] > rest[self.tongue, 2].max())
        self.gum_teeth = (w["MJaw"] > 0.99) & (np.abs(rest[:, 1] - lip_y) < 0.015 * s) & \
            (rest[:, 2] > lip_z - 0.04 * s) & (rest[:, 2] < lip_z) & (np.abs(rest[:, 0]) < 0.025 * s)
        jaw_side = w["MJaw"] + w["MBottomLip"] + w["LMouthBottom"] + w["RMouthBottom"]  # the skin on the jaw, in front
        self.chin = (jaw_side > 0.9) & ~self.tongue & (rest[:, 2] > rest[self.lower_teeth, 2].max())  # of the teeth

    def weight(self, prim, name, bone=None):
        bone = self.bone[name] if bone is None else bone
        return np.where(prim.joints == bone, prim.weights, 0.0).sum(axis=1)

    def skin(self, weights):
        """``(head vertices in character space, the same in the jaw's rest frame, the pose)``."""
        local = self.poser.local_pose([])
        faces.apply_face(local, self.rig, weights)
        glob = self.poser.forward_kinematics(local)
        points = skin_vertices(self.model, self.model.skinning_matrices(glob))[self.index]
        jaw = self.bone["MJaw"]
        back = np.linalg.inv(glob[jaw] @ self.model.inverse_bind[jaw])
        return points, points @ back[:3, :3].T + back[:3, 3]

    def mouth_gap(self, points):
        return points[self.upper_mid, 1].min() - points[self.lower_mid, 1].max()

    def eye_gap(self, points, side):
        top, bottom = self.eyes[side]
        return points[top, 1].min() - points[bottom, 1].max()

    def lips_ahead_of_teeth(self, points, in_jaw):
        """Least margin (metres) by which the lips stay in front of the front teeth, in 4 mm strips across."""
        least = math.inf
        edges = np.arange(-0.02, 0.0201, 0.004) * self.rig.size
        for teeth, lip, p in ((self.lower_teeth, self.lower_lip, in_jaw), (self.upper_teeth, self.upper_lip, points)):
            for a, b in itertools.pairwise(edges):
                strip = (p[:, 0] >= a) & (p[:, 0] < b)
                if (teeth & strip).any() and (lip & strip).any():
                    least = min(least, p[lip & strip, 2].max() - p[teeth & strip, 2].max())
        return least

    def tongue_in_front(self, points):
        """Least margin (metres) by which the tongue lies in front of the lower lip and chin skin next to it."""
        tongue, skin = points[self.tongue], points[self.chin]
        near = np.hypot(tongue[:, None, 0] - skin[None, :, 0], tongue[:, None, 1] - skin[None, :, 1]) < 0.0025
        return np.where(near, tongue[:, None, 2] - skin[None, :, 2], np.inf).min()

    def tongue_over_teeth(self, in_jaw):
        """Least height (metres) of the tongue over the lower teeth and gums right under it (jaw frame)."""
        tongue, teeth = in_jaw[self.tongue], self.rest[self.gum_teeth]
        near = np.hypot(tongue[:, None, 0] - teeth[None, :, 0], tongue[:, None, 2] - teeth[None, :, 2])
        return np.where(near < 0.002 * self.rig.size, tongue[:, None, 1] - teeth[None, :, 1], np.inf).min()


OPEN = {"grin": 1.0, "tongue": 2.0, "surprise": 3.5, "roar": 4.0, "cross_eyed": 1.0}  # least mouth gap, cm
CLOSED = ("smile", "puff", "wink", "pout")


def test_real_avatars_have_face_rigs(avatars):
    for model in avatars:
        rig = faces.face_rig(model)
        assert rig is not None, model.name
        assert (0.7 < rig.size < 0.85) if "Child" in model.name else (0.9 < rig.size < 1.1), (model.name, rig.size)
        assert rocketbox.mend_tongue(model) == 0  # mended when read


def test_real_faces_open_and_close_what_they_should(avatars):
    for model in avatars:
        h = _Head(model)
        s = h.rig.size
        rest, rest_in_jaw = h.skin({})
        assert abs(h.mouth_gap(rest)) < 0.002 and min(h.eye_gap(rest, side) for side in "LR") > 0.005, model.name
        for name in faces.EXPRESSIONS:
            points, in_jaw = h.skin({name: 1.0})
            label = (model.name, name)
            gap = h.mouth_gap(points) / s
            if name in OPEN:
                assert gap > 0.01 * OPEN[name], (label, gap)
            elif name in CLOSED:
                assert gap < 0.003, (label, gap)
            eyes = {side: h.eye_gap(points, side) / s for side in "LR"}
            if name == "scrunch":
                assert max(eyes.values()) < 0.001, (label, eyes)  # squeezed shut
            elif name == "wink":
                assert eyes["L"] < 0.0 and eyes["R"] > 0.005, (label, eyes)
            else:
                assert min(eyes.values()) > 0.004, (label, eyes)
            moved = np.linalg.norm(points - rest, axis=1) / s
            assert moved[~h.tongue].max() < 0.065 and moved.max() < 0.08, label  # nothing leaves the head
            upper = np.linalg.norm(points - rest, axis=1)[h.upper_lip].max() / s  # the head does not move here
            lower = np.linalg.norm(in_jaw - rest_in_jaw, axis=1)[h.lower_lip].max() / s  # on the jaw
            assert upper < 0.03 and lower < 0.03, (label, upper, lower)  # lips stretch, they do not tear off
            assert h.lips_ahead_of_teeth(points, in_jaw) / s > 0.0008, label  # no teeth through the lips


def test_real_tongues_come_out_over_the_teeth(avatars):
    for model in avatars:
        h = _Head(model)
        s = h.rig.size
        _, rest_in_jaw = h.skin({})
        start = h.tongue_over_teeth(rest_in_jaw)
        for w in np.linspace(0.05, 1.0, 20):
            points, in_jaw = h.skin({"tongue": w})
            assert h.tongue_over_teeth(in_jaw) > min(start, 0.0) - 0.0005 * s, (model.name, w)
        tip = points[h.tongue, 2].max() - points[h.lower_lip, 2].max()
        assert tip > 0.012 * s, (model.name, tip)  # far out in front of the lower lip ...
        hang = points[h.lower_mid, 1].min() - points[h.tongue, 1].min()
        assert hang > 0.004 * s, (model.name, hang)  # ... and hanging down below it, so it shows from the front
        assert h.tongue_in_front(points) > 0.0005 * s, model.name  # not into the lip or the chin
        for other in faces.EXPRESSIONS:  # not even when mixed with pursed or pressed lips
            points, _ = h.skin({"tongue": 1.0, other: 1.0})
            assert h.tongue_in_front(points) > 0.0, (model.name, other)
