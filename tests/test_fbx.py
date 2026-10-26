import os
import struct
import zlib
from pathlib import Path

import numpy as np
import pytest

from bifold_demo.characters.fbx import MAGIC, FBXNode, parse_fbx, read_fbx
from bifold_demo.characters.fbx_convert import KTIME_PER_SECOND, load_character, load_clip
from bifold_demo.characters.model import quat_to_matrix

# -- a minimal binary FBX writer ------------------------------------------------------------------


class I16(int):
    """Written as ``Y`` (int16)."""


class I64(int):
    """Written as ``L`` (int64)."""


class F32(float):
    """Written as ``F`` (float32)."""


_ARRAY_CODES = {np.dtype("float32"): b"f", np.dtype("float64"): b"d", np.dtype("int64"): b"l",
                np.dtype("int32"): b"i", np.dtype("bool"): b"b"}


class Packed:
    """An array written zlib-compressed."""

    def __init__(self, array):
        self.array = np.asarray(array)


def _prop(value) -> bytes:
    if isinstance(value, (np.ndarray, Packed)):
        array = value.array if isinstance(value, Packed) else value
        raw = np.ascontiguousarray(array).astype(array.dtype.newbyteorder("<")).tobytes()
        payload = zlib.compress(raw) if isinstance(value, Packed) else raw
        return _ARRAY_CODES[array.dtype] + struct.pack("<III", len(array), int(isinstance(value, Packed)),
                                                       len(payload)) + payload
    if isinstance(value, bool):
        return b"C" + struct.pack("<?", value)
    if isinstance(value, I16):
        return b"Y" + struct.pack("<h", value)
    if isinstance(value, I64):
        return b"L" + struct.pack("<q", value)
    if isinstance(value, int):
        return b"I" + struct.pack("<i", value)
    if isinstance(value, F32):
        return b"F" + struct.pack("<f", value)
    if isinstance(value, float):
        return b"D" + struct.pack("<d", value)
    if isinstance(value, str):
        return b"S" + struct.pack("<I", len(value.encode())) + value.encode()
    if isinstance(value, bytes):
        return b"R" + struct.pack("<I", len(value)) + value
    raise TypeError(value)


def _node(node, offset: int, wide: bool) -> bytes:
    name, props, children = node
    body = b"".join(_prop(p) for p in props)
    header = (24 if wide else 12) + 1 + len(name)
    nested = b""
    for child in children:
        nested += _node(child, offset + header + len(body) + len(nested), wide)
    if children:
        nested += b"\0" * (25 if wide else 13)
    end = offset + header + len(body) + len(nested)
    counts = struct.pack("<QQQ" if wide else "<III", end, len(props), len(body))
    return counts + bytes([len(name)]) + name.encode() + body + nested


def write_fbx(nodes, version: int = 7400) -> bytes:
    """Binary FBX bytes for ``nodes`` given as ``(name, props, children)`` tuples."""
    wide = version >= 7500
    data = MAGIC + struct.pack("<I", version)
    for node in nodes:
        data += _node(node, len(data), wide)
    return data + b"\0" * (25 if wide else 13) + b"\0" * 16


def N(name, *props, children=()):
    return (name, list(props), list(children))


# -- parser ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("version", [7400, 7500])
def test_parser_round_trips_every_property_type(version):
    floats = np.linspace(-1, 1, 50, dtype=np.float32)
    doubles = np.arange(7, dtype=np.float64) / 3
    longs = np.array([0, -3, 2**40], dtype=np.int64)
    ints = np.arange(-5, 300, dtype=np.int32)
    flags = np.array([True, False, True])
    tree = [
        N("Header", I16(-7), True, 123456, F32(1.5), 2.25, I64(-(2**40)), "text\x00\x01Class", b"\x00\xffraw",
          children=[N("Empty"), N("Deep", children=[N("Deeper", "x", children=[N("Leaf", 1)])])]),
        N("Arrays", floats, Packed(doubles), longs, Packed(ints), flags, Packed(flags)),
        N("Arrays"),
    ]
    root = parse_fbx(write_fbx(tree, version))
    assert isinstance(root, FBXNode) and root.props == [version]
    assert [c.name for c in root.children] == ["Header", "Arrays", "Arrays"]
    header = root.find("Header")
    assert header.props == [-7, True, 123456, 1.5, 2.25, -(2**40), "text\x00\x01Class", b"\x00\xffraw"]
    assert header.find("Empty").props == [] and header.find("Empty").children == []
    assert header.find("Deep").find("Deeper").find("Leaf").props == [1]
    assert header.find("Missing") is None
    assert len(root.find_all("Arrays")) == 2
    arrays = root.find("Arrays").props
    for got, want in zip(arrays, [floats, doubles, longs, ints, flags, flags]):
        assert got.dtype == want.dtype
        np.testing.assert_array_equal(got, want)


def test_parser_rejects_other_files(tmp_path):
    path = tmp_path / "ascii.fbx"
    path.write_text("; FBX 7.4.0 project file\n")
    with pytest.raises(ValueError):
        read_fbx(path)


# -- converter on a synthetic scene ---------------------------------------------------------------


def P(name, *values, kind=""):
    return N("P", name, kind, "", "A", *values)


def _model(node_id, name, subtype, translation=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0), pre=None):
    props = [P("Lcl Translation", *map(float, translation)), P("Lcl Rotation", *map(float, rotation))]
    if pre is not None:
        props.append(P("PreRotation", *map(float, pre)))
    return N("Model", I64(node_id), f"{name}\x00\x01Model", subtype, children=[N("Properties70", children=props)])


def _curve(node_id, times, values):
    return N("AnimationCurve", I64(node_id), "\x00\x01AnimCurve", "",
             children=[N("Default", 0.0), N("KeyTime", np.asarray(times, np.int64)),
                       N("KeyValueFloat", np.asarray(values, np.float32))])


def _column_major(matrix):
    return np.asarray(matrix, np.float64).T.ravel()


def _translation(x, y, z):
    m = np.eye(4)
    m[:3, 3] = (x, y, z)
    return m


def _synthetic_scene(path: Path, animation=None, version=7400):
    """A Z-up, centimetre (3ds Max style) scene: a root, two legs, and a skinned quad plus a triangle.

    The figure faces -Y in the file (+Z after conversion). ``animation`` maps (model id, property) to
    ``{axis: (key_times, values)}``.
    """
    bones = {  # id: (name, subtype, parent, local translation (cm), pre-rotation)
        10: ("Bip01", "Root", 0, (0, 0, 90), None),
        11: ("Bip01 L Thigh", "LimbNode", 10, (10, 0, 0), (0, 0, 90)),
        12: ("Bip01 L Foot", "LimbNode", 11, (0, 0, -80), None),
        13: ("Bip01 L Toe0", "LimbNode", 12, (0, -15, -8), None),
        14: ("Bip01 R Thigh", "LimbNode", 10, (-10, 0, 0), None),
    }
    globals_ = {}
    objects, connections = [], []
    for node_id, (name, subtype, parent, translation, pre) in bones.items():
        rotation = (0, 0, -90) if pre is not None else (0, 0, 0)  # cancels the pre-rotation
        objects.append(_model(node_id, name, subtype, translation, rotation, pre))
        connections.append(N("C", "OO", I64(node_id), I64(parent)))
        globals_[node_id] = (globals_[parent] if parent else np.eye(4)) @ _translation(*translation)
    objects.append(_model(20, "test_hipoly_mesh", "Mesh"))
    connections.append(N("C", "OO", I64(20), I64(0)))
    vertices = np.array([[0, 0, 0], [20, 0, 0], [20, 0, 100], [0, 0, 100], [0, -10, 170]], np.float64)
    polygons = np.array([0, 1, 2, ~3, 3, 2, ~4], np.int32)
    objects.append(N("Geometry", I64(30), "\x00\x01Geometry", "Mesh", children=[
        N("Vertices", Packed(vertices.ravel())),
        N("PolygonVertexIndex", polygons),
        N("LayerElementNormal", 0, children=[
            N("MappingInformationType", "ByPolygonVertex"), N("ReferenceInformationType", "Direct"),
            N("Normals", np.tile([0.0, -1.0, 0.0], 7))]),
        N("LayerElementUV", 0, children=[
            N("MappingInformationType", "ByPolygonVertex"), N("ReferenceInformationType", "IndexToDirect"),
            N("UV", np.array([0, 0, 1, 0, 1, 1, 0, 1, 0.5, 0.5], np.float64)),
            N("UVIndex", np.array([0, 1, 2, 3, 3, 2, 4], np.int32))]),
        N("LayerElementMaterial", 0, children=[
            N("MappingInformationType", "ByPolygon"), N("ReferenceInformationType", "IndexToDirect"),
            N("Materials", np.array([0, 1], np.int32))]),
    ]))
    connections.append(N("C", "OO", I64(30), I64(20)))
    objects.append(N("Deformer", I64(40), "\x00\x01Deformer", "Skin"))
    connections.append(N("C", "OO", I64(40), I64(30)))
    for cluster_id, bone, indexes, weights in ((41, 10, [0, 1, 2, 3], [1, 1, 0.5, 0.5]), (42, 11, [2, 3], [0.5, 0.5])):
        link = globals_[bone]
        objects.append(N("Deformer", I64(cluster_id), "\x00\x01SubDeformer", "Cluster", children=[
            N("Indexes", np.array(indexes, np.int32)), N("Weights", np.array(weights, np.float64)),
            N("Transform", _column_major(np.linalg.inv(link))), N("TransformLink", _column_major(link))]))
        connections += [N("C", "OO", I64(cluster_id), I64(40)), N("C", "OO", I64(bone), I64(cluster_id))]
    for material_id, name in ((50, "skin"), (51, "hair")):
        objects.append(N("Material", I64(material_id), f"{name}\x00\x01Material", ""))
        connections.append(N("C", "OO", I64(material_id), I64(20)))
    for texture_id, file_name, material_id, slot in ((60, "C:\\art\\SKIN_color.TGA", 50, "DiffuseColor"),
                                                     (61, "D:/x/hair_opacity.tga", 51, "DiffuseColor"),
                                                     (62, "hair_opacity.tga", 51, "TransparentColor")):
        objects.append(N("Texture", I64(texture_id), "t\x00\x01Texture", "", children=[N("FileName", file_name)]))
        connections.append(N("C", "OP", I64(texture_id), I64(material_id), slot))
    if animation:
        objects += [N("AnimationStack", I64(70), "Take\x00\x01AnimStack", "", children=[N("Properties70", children=[
                        P("LocalStop", I64(KTIME_PER_SECOND))])]),
                    N("AnimationLayer", I64(71), "Base\x00\x01AnimLayer", "")]
        connections.append(N("C", "OO", I64(71), I64(70)))
        next_id = 100
        for (model_id, prop), curves in animation.items():
            objects.append(N("AnimationCurveNode", I64(next_id), "T\x00\x01AnimCurveNode", ""))
            connections += [N("C", "OO", I64(next_id), I64(71)), N("C", "OP", I64(next_id), I64(model_id), prop)]
            for axis, (times, values) in curves.items():
                objects.append(_curve(next_id + 1 + axis, times, values))
                connections.append(N("C", "OP", I64(next_id + 1 + axis), I64(next_id), "d|" + "XYZ"[axis]))
            next_id += 10
    settings = N("GlobalSettings", children=[N("Properties70", children=[
        P("UpAxis", 2), P("UpAxisSign", 1), P("FrontAxis", 1), P("FrontAxisSign", -1), P("CoordAxis", 0),
        P("CoordAxisSign", 1), P("UnitScaleFactor", 1.0), P("TimeMode", 6)])])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(write_fbx([settings, N("Objects", children=objects), N("Connections", children=connections)],
                               version))
    return path


@pytest.fixture
def synthetic_avatar(tmp_path):
    from PIL import Image

    textures = tmp_path / "Textures"
    textures.mkdir()
    skin = np.zeros((8, 4, 3), np.uint8)
    skin[:4] = (255, 0, 0)  # red on top
    skin[4:] = (0, 0, 255)
    Image.fromarray(skin).save(textures / "skin_color.tga")
    hair = np.zeros((4, 4, 4), np.uint8)
    hair[..., :3] = 90
    hair[:, 2:, 3] = 255  # opaque on the right
    Image.fromarray(hair, "RGBA").save(textures / "Hair_Opacity.tga")
    return _synthetic_scene(tmp_path / "Export" / "Test.fbx", version=7500)


def test_load_character_converts_axes_units_and_skin(synthetic_avatar):
    model = load_character(synthetic_avatar)
    skeleton = model.skeleton
    assert skeleton.names == ["Bip01", "Bip01 L Thigh", "Bip01 L Foot", "Bip01 L Toe0", "Bip01 R Thigh"]
    np.testing.assert_allclose(skeleton.rest_global[skeleton.index("Bip01 L Toe0"), :3, 3], [0.1, 0.02, 0.15],
                               atol=1e-9)  # toes in front (+Z) of the foot, metres, Y up
    np.testing.assert_allclose(model.skinning_matrices(skeleton.rest_global), np.tile(np.eye(4), (5, 1, 1)),
                               atol=1e-9)
    assert model.metadata["lod"] == "hipoly" and model.metadata["triangles"] == 3
    assert model.height == pytest.approx(1.7)
    skin, hair = model.primitives
    assert (skin.name, hair.name) == ("skin", "hair")
    assert len(skin.positions) == 4 and skin.indices.shape == (2, 3) and len(hair.positions) == 3
    corners = {tuple(np.round(p.astype(float), 4).tolist()) for p in skin.positions}
    assert corners == {(0.0, 0.0, 0.0), (0.2, 0.0, 0.0), (0.2, 1.0, 0.0), (0.0, 1.0, 0.0)}
    np.testing.assert_allclose(skin.normals, np.tile([0, 0, 1], (4, 1)), atol=1e-6)  # file -Y = forward
    a, b, c = skin.positions[skin.indices[0]]
    assert np.cross(b - a, c - a)[2] > 0  # counter-clockwise seen from the normal side
    top = np.argmax(skin.positions[:, 1] + skin.positions[:, 0])
    np.testing.assert_allclose(skin.weights[top], [0.5, 0.5, 0, 0])
    assert set(skin.joints[top, :2]) == {0, 1}
    apex = np.argmax(hair.positions[:, 1])
    assert hair.weights[apex, 0] == 1 and hair.joints[apex, 0] == 0  # unweighted: nearest bone
    np.testing.assert_allclose(hair.uvs[apex], [0.5, 0.5])
    assert skin.texture.shape == (8, 4, 4) and tuple(skin.texture[0, 0]) == (255, 0, 0, 255)
    assert tuple(skin.texture[-1, 0]) == (0, 0, 255, 255)  # row 0 = top of the image
    assert hair.texture[0, 0, 3] == 0 and hair.texture[0, 3, 3] == 255  # alpha from the opacity map


def test_load_character_lod_and_texture_options(synthetic_avatar):
    model = load_character(synthetic_avatar, lod="hipoly", texture_dir=synthetic_avatar.parent, max_texture_size=2)
    assert all(p.texture is None for p in model.primitives)  # nothing to find there
    with pytest.raises(ValueError, match="hipoly"):
        load_character(synthetic_avatar, lod="lowpoly")
    by_name = load_character(synthetic_avatar, lod="Test_HiPoly_Mesh", max_texture_size=2)  # a mesh name
    assert by_name.metadata["lod"] == "hipoly" and by_name.metadata["meshes"] == ["test_hipoly_mesh"]
    with pytest.raises(ValueError, match="no LOD"):
        load_character(synthetic_avatar, lod="20")  # the mesh's FBX object id is not its name
    small = load_character(synthetic_avatar, max_texture_size=2)
    assert max(small.primitives[0].texture.shape[:2]) == 2


def test_load_clip_makes_walks_in_place(tmp_path):
    second = KTIME_PER_SECOND
    walk = {(10, "Lcl Translation"): {0: ([0, second], [0, 0]), 1: ([0, second], [0, -120]), 2: ([0, second], [90, 90])},
            (11, "Lcl Rotation"): {0: ([0, second], [0, 30])}}
    clip = load_clip(_synthetic_scene(tmp_path / "test_walk_01.fbx", walk))
    assert clip.name == "test_walk_01" and clip.fps == 30 and clip.frames == 31 and clip.loop
    assert clip.root == "Bip01" and {"Bip01", "Bip01 L Thigh"} <= set(clip.bone_names)
    assert clip.metadata["kind"] == "walk" and clip.metadata["speed"] == pytest.approx(1.2)
    root = clip.translations[:, clip.bone_names.index("Bip01")]
    np.testing.assert_allclose(root[:, [0, 2]], 0, atol=1e-9)  # travel removed
    np.testing.assert_allclose(root[:, 1], 0.9)
    model = load_character(_synthetic_scene(tmp_path / "Export" / "Test.fbx"))
    assert set(clip.bone_names) <= set(model.skeleton.names)


def test_load_clip_measures_turns(tmp_path):
    second = KTIME_PER_SECOND
    times = np.linspace(0, second, 5).astype(np.int64)
    turn = {(10, "Lcl Rotation"): {2: (times, np.linspace(30, 120, 5))}}  # about the up axis, to the left
    clip = load_clip(_synthetic_scene(tmp_path / "x_turn_left_90.fbx", turn))
    assert clip.metadata["kind"] == "turn" and not clip.loop
    assert clip.metadata["turn"] == pytest.approx(90.0, abs=1e-6) and clip.metadata["speed"] == 0
    root = quat_to_matrix(clip.rotations[:, clip.bone_names.index("Bip01")])
    np.testing.assert_allclose(root[0] @ [0, -1, 0], [0, 0, 1], atol=1e-9)  # the file's front starts at +Z
    np.testing.assert_allclose(root[-1] @ [0, -1, 0], [1, 0, 0], atol=1e-9)  # ... and ends at +X (left turn)


# -- real Rocketbox files (skipped when missing) ----------------------------------------------------

ROCKETBOX = Path(os.environ.get(
    "BIFOLD_ROCKETBOX_TEST_DIR",
    "/tmp/claude-0/-home-user-bifold-demo/44ad5b06-773f-5ffd-8306-515788fc8497/scratchpad/rb/Assets"))
AVATARS = {"Male_Adult_05": ("Adults", (1.6, 1.95)), "Female_Child_01": ("Children", (1.1, 1.45))}
CLIPS = {
    "m_walk_neutral_01": "all_animations_max_motextr_xy",
    "m_turn_left_180": "all_animations_max_motextr_xyz",
    "f_turn_right_90": "all_animations_max_motextr_xyz",
    "m_idle_neutral_01": "all_animations_max_motextr_static",
    "f_walk_neutral_01": "all_animations_max_motextr_xy",
    "f_idle_neutral_01": "all_animations_max_motextr_static",
}
SEX = {"Male_Adult_05": "m", "Female_Child_01": "f"}


def _avatar_path(name):
    return ROCKETBOX / "Avatars" / AVATARS[name][0] / name / "Export" / f"{name}.fbx"


def _clip_path(name):
    return ROCKETBOX / "Animations" / CLIPS[name] / f"{name}.max.fbx"


def _require(path):
    if not path.is_file():
        pytest.skip(f"Rocketbox test file missing: {path}")
    return path


@pytest.fixture(scope="module", params=sorted(AVATARS))
def avatar(request):
    return request.param, load_character(_require(_avatar_path(request.param)), max_texture_size=64)


@pytest.fixture(scope="module")
def clips():
    return {name: load_clip(_require(_clip_path(name))) for name in CLIPS}


def test_rocketbox_avatar(avatar):
    name, model = avatar
    low, high = AVATARS[name][1]
    assert low < model.height < high
    skeleton = model.skeleton
    assert len(skeleton) >= 60 and skeleton.names[0] == "Bip01"
    assert all(p < i for i, p in enumerate(skeleton.parents))
    for side in "LR":
        toe = skeleton.rest_global[skeleton.index(f"Bip01 {side} Toe0"), :3, 3]
        foot = skeleton.rest_global[skeleton.index(f"Bip01 {side} Foot"), :3, 3]
        assert toe[2] > foot[2] + 0.05  # facing +Z
    positions = np.concatenate([p.positions for p in model.primitives])
    assert abs(positions[:, 1].min()) < 0.02  # feet on the floor
    head = skeleton.rest_global[skeleton.index("Bip01 Head"), :3, 3]
    assert abs(head[0]) < 0.02 and head[1] > 0.8 * model.height
    np.testing.assert_allclose(model.skinning_matrices(skeleton.rest_global), np.tile(np.eye(4), (len(skeleton), 1, 1)),
                               atol=1e-9)
    assert model.metadata["triangles"] > 4000 and model.metadata["lod"] == "hipoly"
    for p in model.primitives:
        assert p.texture is not None and p.texture.dtype == np.uint8 and p.texture.shape[2] == 4
        np.testing.assert_allclose(p.weights.sum(axis=1), 1, atol=1e-5)
        assert p.joints.max() < len(skeleton)


def test_rocketbox_clips(clips, avatar):
    _, model = avatar
    walk, left, right, idle = (clips[n] for n in ("m_walk_neutral_01", "m_turn_left_180", "f_turn_right_90",
                                                  "m_idle_neutral_01"))
    assert walk.metadata["kind"] == "walk" and 0.5 < walk.metadata["speed"] < 1.8 and walk.loop
    assert left.metadata["kind"] == "turn" and left.metadata["turn"] == pytest.approx(180, abs=15) and not left.loop
    assert right.metadata["turn"] == pytest.approx(-90, abs=15)
    assert idle.metadata["kind"] == "idle" and idle.metadata["speed"] == 0
    for clip in clips.values():
        assert clip.fps == 30 and clip.root == "Bip01"
        assert set(model.skeleton.names) <= set(clip.bone_names)  # every avatar bone is animated
        root = clip.translations[:, clip.bone_names.index("Bip01")]
        assert np.abs(root[:, [0, 2]]).max() < 0.2 and 0.7 < root[:, 1].mean() < 1.0  # in place


def _lowest_vertex(model, pose, keep) -> float:
    """Lowest y of the vertices selected by ``keep`` (one mask per primitive) after CPU skinning."""
    palette = model.skinning_matrices(pose)
    lowest = np.inf
    for prim, mask in zip(model.primitives, keep):
        if not mask.any():
            continue
        m = np.einsum("nk,nkij->nij", prim.weights[mask].astype(np.float64), palette[prim.joints[mask]])
        y = np.einsum("nj,nj->n", m[:, 1, :3], prim.positions[mask].astype(np.float64)) + m[:, 1, 3]
        lowest = min(lowest, float(y.min()))
    return lowest


def test_rocketbox_walk_and_idle_keep_the_soles_on_the_floor(clips, avatar):
    """Adults and children alike: the root scale comes from the legs, not the walk's lowered mean pelvis height."""
    from bifold_demo.characters.behaviour import stride_scale
    from bifold_demo.characters.model import Layer, Poser

    name, model = avatar
    poser = Poser(model)
    feet = [p.positions[:, 1] < 0.15 for p in model.primitives]
    for kind in ("walk", "idle"):
        clip = clips[f"{SEX[name]}_{kind}_neutral_01"]
        lows = [_lowest_vertex(model, poser.global_pose([Layer(clip, t)]), feet)
                for t in np.linspace(0.0, min(clip.duration, 4.0), 24, endpoint=False)]
        assert -0.015 < min(lows) < 0.005, (kind, min(lows))  # a sole reaches the floor (walks floated 1.4-2.9 cm)
        assert max(lows) < (0.015 if kind == "walk" else 0.005), (kind, max(lows))
        assert stride_scale(model, clip) == pytest.approx(poser._map(clip)[-1])  # walking speed matches body size
