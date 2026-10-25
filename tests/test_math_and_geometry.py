import numpy as np
import pytest

from bifold_demo.scene.geometry import Mesh, box_mesh, cylinder_mesh, plane_mesh, quad_mesh, sphere_mesh
from bifold_demo.scene.transforms import euler_rotation, look_at, perspective, rotation_y, translation


def test_rotation_y_turns_x_axis_towards_minus_z():
    x_axis = np.array([1.0, 0, 0, 0])
    np.testing.assert_allclose(rotation_y(90) @ x_axis, [0, 0, -1, 0], atol=1e-12)


def test_euler_order_is_x_then_y_then_z():
    expected = euler_rotation((0, 0, 30)) @ euler_rotation((0, 20, 0)) @ euler_rotation((10, 0, 0))
    np.testing.assert_allclose(euler_rotation((10, 20, 30)), expected)


def test_camera_projects_target_to_screen_centre():
    view = look_at((1, 2, 5), (0.5, 1, 0))
    clip = perspective(60, 16 / 9, 0.1, 100) @ view @ np.array([0.5, 1, 0, 1.0])
    ndc = clip[:3] / clip[3]
    np.testing.assert_allclose(ndc[:2], [0, 0], atol=1e-9)
    assert -1 < ndc[2] < 1


def test_look_at_straight_down_is_well_defined():
    view = look_at((0, 5, 0), (0, 0, 0))
    assert np.all(np.isfinite(view))


def test_translation():
    np.testing.assert_allclose(translation((1, 2, 3)) @ [0, 0, 0, 1], [1, 2, 3, 1])


@pytest.mark.parametrize(
    "mesh",
    [quad_mesh(2, 1), plane_mesh(3, 2), box_mesh(1, 2, 3), sphere_mesh(1.0, 16, 8), cylinder_mesh(0.5, 2.0, 16)],
    ids=["quad", "plane", "box", "sphere", "cylinder"],
)
def test_faces_wind_counter_clockwise_around_their_normals(mesh):
    p = mesh.positions.astype(np.float64)
    a, b, c = (p[mesh.indices[:, k]] for k in range(3))
    face = np.cross(b - a, c - a)
    area = np.linalg.norm(face, axis=1)
    keep = area > 1e-9  # sphere poles have degenerate triangles
    vertex_normals = mesh.normals[mesh.indices[keep]].mean(axis=1)
    assert np.all(np.einsum("ij,ij->i", face[keep], vertex_normals) > 0)


def test_box_dimensions_and_uvs():
    mesh = box_mesh(1, 2, 3)
    np.testing.assert_allclose(mesh.positions.min(axis=0), [-0.5, -1, -1.5])
    np.testing.assert_allclose(mesh.positions.max(axis=0), [0.5, 1, 1.5])
    assert mesh.uvs.min() == 0 and mesh.uvs.max() == 1
    assert len(mesh.indices) == 12


def test_custom_mesh_computes_normals():
    mesh = Mesh([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]])
    np.testing.assert_allclose(mesh.normals, [[0, 0, 1]] * 3)
    assert mesh.interleaved().shape == (3, 8)


def test_mesh_rejects_bad_indices():
    with pytest.raises(ValueError):
        Mesh([[0, 0, 0]], [[0, 1, 2]])
