"""Baking a primitive's texture layout: each texel's point on the body, normal, triangle, bone and island."""

import numpy as np

from bifold_demo.characters.model import SkinnedPrimitive
from bifold_demo.characters.uvbake import bake, cached_bake, islands


class _Model:
    def __init__(self, primitives):
        self.primitives = primitives


def _square(uv0=(0.0, 0.0), uv1=(1.0, 1.0), z=0.0, bones=(3, 3, 5, 5)):
    """A unit square in the XY plane at ``z`` (facing +Z), mapped onto the UV rectangle ``uv0``-``uv1``."""
    (u0, v0), (u1, v1) = uv0, uv1
    positions = np.array([[0, 0, z], [1, 0, z], [1, 1, z], [0, 1, z]], np.float32)
    uvs = np.array([[u0, v0], [u1, v0], [u1, v1], [u0, v1]], np.float32)
    joints = np.array([[b, 0, 0, 0] for b in bones], np.int32)
    weights = np.tile(np.array([1, 0, 0, 0], np.float32), (4, 1))
    return SkinnedPrimitive("square", positions, np.tile(np.float32([0, 0, 1]), (4, 1)), uvs,
                            np.array([[0, 1, 2], [0, 2, 3]], np.uint32), joints, weights)


def test_a_texel_knows_its_point_normal_and_bone():
    b = bake(_Model([_square()]), 0, size=64)
    assert b.covered.mean() > 0.97 and not b.overlap.any()
    row, col = b.texel((0.25, 0.75))  # a quarter across, three quarters up
    assert np.allclose(b.position[row, col], (0.25, 0.75, 0.0), atol=1.5 / 64)
    assert np.allclose(b.normal[row, col], (0, 0, 1))
    assert b.bone[b.texel((0.1, 0.05))] == 3 and b.bone[b.texel((0.9, 0.95))] == 5  # the nearest corner's bone
    assert set(np.unique(b.triangle[b.covered])) == {0, 1} and set(np.unique(b.island[b.covered])) == {0}


def test_islands_and_overlaps():
    a, c = _square((0.0, 0.0), (0.4, 0.4)), _square((0.6, 0.6), (1.0, 1.0), z=1.0)
    stacked = _square((0.0, 0.0), (0.4, 0.4), z=2.0)  # on top of the first in UV
    parts = (a, c, stacked)
    merged = SkinnedPrimitive("three", *(np.concatenate([getattr(q, f) for q in parts])
                                         for f in ("positions", "normals", "uvs")),
                              np.r_[a.indices, c.indices + 4, stacked.indices + 8],
                              *(np.concatenate([getattr(q, f) for q in parts]) for f in ("joints", "weights")))
    assert list(islands(merged)) == [0, 0, 1, 1, 0, 0]  # the stacked square shares the first one's UV corners
    b = bake(_Model([merged]), 0, size=100)
    assert b.overlap[b.texel((0.2, 0.2))] and not b.overlap[b.texel((0.8, 0.8))]
    assert b.position[b.texel((0.2, 0.2))][2] == 2.0  # the last drawn wins
    assert not b.covered[b.texel((0.5, 0.5))]
    assert np.isnan(b.position[b.texel((0.5, 0.5))]).all() and b.triangle[b.texel((0.5, 0.5))] == -1


def test_bakes_are_kept_while_the_model_lives():
    model = _Model([_square()])
    assert cached_bake(model, 0, 32) is cached_bake(model, 0, 32)
    assert cached_bake(model, 0, 16) is not cached_bake(model, 0, 32)


def test_a_bake_comes_back_from_its_triangle_map():
    from bifold_demo.characters.uvbake import from_triangles, rasterise

    model = _Model([_square(bones=(3, 4, 5, 6))])
    full = bake(model, 0, size=48)
    triangle, overlap = rasterise(model.primitives[0], 48)
    again = from_triangles(model, 0, triangle.astype(np.int16), overlap)
    for name in ("covered", "triangle", "bone", "island", "overlap"):
        assert np.array_equal(getattr(full, name), getattr(again, name)), name
    assert np.allclose(full.position, again.position, equal_nan=True)
    assert np.allclose(full.normal, again.normal, equal_nan=True)
