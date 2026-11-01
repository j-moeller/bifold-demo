"""Tangent frames, the normal maps' green convention and the packed maps (characters/tangents.py)."""

import numpy as np
import pytest

from bifold_demo.characters.tangents import GREEN, normal_green, pack_maps, vertex_tangents


def _grid(width=4, height=3, uv=lambda x, y: (x, y)):
    """A flat grid in the z = 0 plane facing +z: ``(positions, normals, uvs, indices)``."""
    xs, ys = np.meshgrid(np.linspace(-1, 1, width + 1), np.linspace(-1, 1, height + 1))
    positions = np.stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)], axis=1)
    normals = np.tile([0.0, 0.0, 1.0], (len(positions), 1))
    uvs = np.array([uv(x, y) for x, y in positions[:, :2]])
    quads = []
    for j in range(height):
        for i in range(width):
            a = j * (width + 1) + i
            b, c, d = a + 1, a + width + 2, a + width + 1
            quads += [(a, b, c), (a, c, d)]  # counter-clockwise seen from +z
    return positions, normals, uvs, np.array(quads)


def test_a_plane_gets_u_as_its_tangent_and_right_handedness():
    t = vertex_tangents(*_grid())
    assert t.shape == (20, 4) and t.dtype == np.float32
    np.testing.assert_allclose(t, np.tile([1.0, 0.0, 0.0, 1.0], (20, 1)), atol=1e-6)


def test_the_tangent_follows_the_texture_not_the_mesh():
    turned = vertex_tangents(*_grid(uv=lambda x, y: (y, -x)))  # u grows along +y on the mesh
    np.testing.assert_allclose(turned[:, :3], np.tile([0.0, 1.0, 0.0], (20, 1)), atol=1e-6)
    assert (turned[:, 3] == 1).all()
    scaled = vertex_tangents(*_grid(uv=lambda x, y: (0.01 * x, 30 * y)))  # the texture's scale does not matter
    np.testing.assert_allclose(scaled[:, :3], np.tile([1.0, 0.0, 0.0], (20, 1)), atol=1e-6)


def test_mirrored_uvs_turn_the_handedness_round():
    positions, normals, uvs, indices = _grid(uv=lambda x, y: (abs(x), y))  # the left half mirrors the right
    t = vertex_tangents(positions, normals, uvs, indices)
    x = positions[:, 0]
    left, right = x < -1e-9, x > 1e-9
    np.testing.assert_allclose(t[left, :3], np.tile([-1.0, 0.0, 0.0], (left.sum(), 1)), atol=1e-6)
    assert (t[left, 3] == -1).all()  # bitangent = w * cross(n, t) = +y, the way v grows
    np.testing.assert_allclose(t[right, :3], np.tile([1.0, 0.0, 0.0], (right.sum(), 1)), atol=1e-6)
    assert (t[right, 3] == 1).all()
    seam = ~(left | right)  # shared by both halves: one side decides, the frame stays a frame
    assert np.allclose(np.linalg.norm(t[seam, :3], axis=1), 1) and np.allclose(t[seam, 2], 0)
    assert set(np.abs(t[seam, 0]).round(6)) == {1.0} and set(t[seam, 3]) <= {-1.0, 1.0}
    bitangent = t[:, 3:] * np.cross(normals, t[:, :3])
    np.testing.assert_allclose(bitangent[left | right], np.tile([0.0, 1.0, 0.0], ((left | right).sum(), 1)), atol=1e-6)


def _creased(welded: bool):
    """A grid bent up by 60 degrees along x = 0, its UVs at 45 degrees to the crease; the vertices on the crease are
    split by their normals (and, unless ``welded``, by a tiny UV offset too). Returns the tangents and normals of the
    crease's two copies."""
    positions, normals, uvs, indices = _grid(2, 1, uv=lambda x, y: ((x + y) / 2, (y - x) / 2))
    right = np.nonzero(positions[:, 0] > 1e-9)[0]
    angle = np.radians(60)
    positions[right, 2] = positions[right, 0] * np.sin(angle)
    positions[right, 0] *= np.cos(angle)
    crease = np.nonzero(np.abs(positions[:, 0]) < 1e-9)[0]
    split = len(positions) + np.arange(len(crease))
    tilted = np.array([-np.sin(angle), 0.0, np.cos(angle)])
    positions = np.vstack([positions, positions[crease]])
    uvs = np.vstack([uvs, uvs[crease] + (0.0 if welded else 1e-3)])
    normals = np.vstack([normals, np.tile(tilted, (len(crease), 1))])
    normals[right] = tilted
    remap = dict(zip(crease, split))
    for k, tri in enumerate(indices):
        if positions[tri, 0].max() > 1e-9:  # a triangle of the bent half uses the split copies
            indices[k] = [remap.get(v, v) for v in tri]
    t = vertex_tangents(positions, normals, uvs, indices)
    return t[crease, :3], t[split, :3], normals[crease], normals[split]


def test_vertices_split_only_by_their_normals_share_one_tangent():
    turn = {}
    for welded in (True, False):
        flat, bent, flat_normal, bent_normal = _creased(welded)
        assert np.allclose(np.einsum("ij,ij->i", flat, flat_normal), 0, atol=1e-6)  # each in its own plane
        assert np.allclose(np.einsum("ij,ij->i", bent, bent_normal), 0, atol=1e-6)
        turn[welded] = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", flat, bent), -1, 1))).max()
    assert turn[False] == pytest.approx(41.4, abs=0.5)  # apart, each copy keeps its own side's tangent
    # one sum for both sides, (1.06, 1.41, 0.61), laid into each plane: (0.6, 0.8, 0) and (0.3, 0.8, 0.52)
    assert turn[True] == pytest.approx(34.9, abs=0.5)


def test_a_cylinder_keeps_its_tangents_round_the_uv_seam():
    sides, rows = 12, 3
    angles = np.linspace(0, 2 * np.pi, sides + 1)  # the last column repeats the first: a UV seam
    heights = np.linspace(0, 1, rows + 1)
    positions = np.array([(np.cos(a), h, -np.sin(a)) for h in heights for a in angles])
    normals = positions * [1, 0, 1]
    uvs = np.array([(k / sides, h) for h in heights for k in range(sides + 1)])
    indices = []
    for j in range(rows):
        for i in range(sides):
            a, b = j * (sides + 1) + i, j * (sides + 1) + i + 1
            c, d = b + sides + 1, a + sides + 1
            indices += [(a, b, c), (a, c, d)]
    t = vertex_tangents(positions, normals, uvs, np.array(indices))
    expected = np.stack([-np.sin(angles), np.zeros_like(angles), -np.cos(angles)], axis=1)  # d(position)/du
    np.testing.assert_allclose(t[:, :3], np.tile(expected, (rows + 1, 1)), atol=0.02)
    assert (t[:, 3] == 1).all()


def test_vertices_without_a_usable_triangle_get_some_frame():
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [5, 5, 5]], float)
    normals = np.tile([0.0, 1.0, 0.0], (4, 1))
    uvs = np.zeros((4, 2))  # no UV area at all
    t = vertex_tangents(positions, normals, uvs, np.array([[0, 1, 2]]))
    assert np.allclose(np.linalg.norm(t[:, :3], axis=1), 1) and np.allclose(t[:, 1], 0) and (t[:, 3] == 1).all()
    assert vertex_tangents(np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 2)), np.zeros((0, 3))).shape == (0, 4)


def _bumps(size=96, green=GREEN):
    """A normal map (``uint8`` RGB, row 0 at the top) of a field of round bumps, with green towards ``green``."""
    rows, cols = np.mgrid[0:size, 0:size].astype(float)
    u, v = cols / size, 1.0 - rows / size
    height = 0.02 * (np.sin(9 * np.pi * u) * np.sin(7 * np.pi * v) + 0.5 * np.cos(13 * np.pi * u + 3 * v))
    dh_du = np.gradient(height, axis=1) * size
    dh_dv = -np.gradient(height, axis=0) * size
    n = np.stack([-dh_du, -dh_dv, np.ones_like(height)], axis=2)
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    if green != GREEN:
        n[..., 1] *= -1
    return np.clip(np.round((n + 1) * 127.5), 0, 255).astype(np.uint8)


def test_the_green_channel_direction_is_found_by_the_curl():
    found, up, down = normal_green(_bumps())
    assert found == "+v" and up < 0.35 and down > 0.7
    found, up, down = normal_green(_bumps(green="-v"))
    assert found == "-v" and down < 0.35 and up > 0.7
    flat = np.zeros((32, 32, 3), np.uint8) + (128, 128, 255)
    assert normal_green(flat)[0] == GREEN  # nothing to tell: the library's convention


def test_packed_maps_hold_the_normal_the_specular_level_and_room_for_occlusion():
    normal = _bumps(16)
    spec = np.zeros((16, 16, 3), np.uint8)
    spec[..., 0], spec[..., 1], spec[..., 2] = 30, 33, 36
    packed = pack_maps(normal, spec)
    assert packed.shape == (16, 16, 4) and packed.dtype == np.uint8
    assert np.array_equal(packed[..., :2], normal[..., :2]) and (packed[..., 2] == 33).all()
    assert (packed[..., 3] == 255).all()
    flipped = pack_maps(normal, spec[..., 0], green="-v", occlusion=np.full((16, 16), 90, np.uint8))
    assert np.array_equal(flipped[..., 1], 255 - normal[..., 1]) and (flipped[..., 2] == 30).all()
    assert (flipped[..., 3] == 90).all()
    assert (pack_maps(normal)[..., 2] == 0).all()
    with pytest.raises(ValueError, match="specular map"):
        pack_maps(normal, np.zeros((8, 8), np.uint8))
