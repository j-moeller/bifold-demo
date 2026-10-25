"""Custom geometry from raw vertex arrays, and scaling objects over time.

Run:  bifold-demo --scene examples/custom_geometry.py
"""

import math

from bifold_demo import Camera, Mesh, Plane, Scene, SceneObject, checkerboard, oscillate, run, sample_image, spin


def pyramid(size: float = 1.0, height: float = 1.2) -> Mesh:
    """A square pyramid standing on the origin; each side shows the whole texture."""
    s = size / 2
    apex = (0, height, 0)
    corners = [(-s, 0, s), (s, 0, s), (s, 0, -s), (-s, 0, -s)]  # counter-clockwise seen from above
    positions, uvs, indices = [], [], []
    for i in range(4):  # one triangle per side, counter-clockwise seen from outside
        a, b = corners[i], corners[(i + 1) % 4]
        base = len(positions)
        positions += [a, b, apex]
        uvs += [(0, 0), (1, 0), (0.5, 1)]
        indices.append((base, base + 1, base + 2))
    base = len(positions)  # bottom square, facing down
    positions += corners
    uvs += [(0, 0), (1, 0), (1, 1), (0, 1)]
    indices += [(base, base + 2, base + 1), (base, base + 3, base + 2)]
    return Mesh(positions, indices, uvs=uvs)  # normals are computed automatically


def build_scene() -> Scene:
    scene = Scene(background=(36, 38, 46))
    scene.camera = Camera(position=(0, 2.0, 5.5), target=(0, 0.9, 0), fov=50)
    scene.add(Plane(size=20, uv_repeat=8, texture=checkerboard()), name="floor")

    scene.add(SceneObject(pyramid(2.0, 2.0), texture=sample_image("bus.jpg"), rotation=spin(25)), name="pyramid")
    scene.add(
        SceneObject(
            pyramid(0.8, 0.8),
            texture=sample_image("zidane.jpg"),
            position=(2.2, 0.0, 0.5),
            scale=oscillate(center=(1, 1, 1), amplitude=(0.3, 0.6, 0.3), period=3),  # breathing
        ),
        name="small",
    )
    scene.add(
        SceneObject(pyramid(0.8, 0.8), texture="#f4a261", position=(-2.2, 0.8, 0.5),
                    rotation=lambda t: (180 * math.sin(t), 0, 0)),
        name="flipper",
    )
    return scene


if __name__ == "__main__":
    run(build_scene())
