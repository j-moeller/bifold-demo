"""Simple physics: boxes and balls with velocity, spin, gravity and bouncy walls.

Run:  bifold-demo --scene examples/physics.py      (press R to restart the simulation)
"""

import random

from bifold_demo import Body, Box, Camera, Plane, Scene, Sphere, checkerboard, run, sample_image

ROOM = ((-3.0, 0.0, -2.5), (3.0, 4.0, 1.5))  # (min corner, max corner) the objects stay inside


def build_scene() -> Scene:
    rng = random.Random(7)  # fixed seed: the same motion after every reset
    scene = Scene(background=(32, 34, 40))
    scene.camera = Camera(position=(0, 2.2, 6.5), target=(0, 1.2, 0), fov=55)
    scene.add(Plane(size=20, uv_repeat=8, texture=checkerboard()), name="floor")

    textures = [sample_image("bus.jpg"), sample_image("zidane.jpg")]
    for i in range(4):
        scene.add(
            Box(
                size=0.9,
                texture=textures[i % 2],
                position=(rng.uniform(-2, 2), rng.uniform(1.5, 3.5), rng.uniform(-1.5, 0.5)),
                body=Body(
                    velocity=(rng.uniform(-2, 2), 0, rng.uniform(-1, 1)),
                    angular_velocity=(rng.uniform(-60, 60), rng.uniform(-90, 90), 0),
                    gravity=-9.81,
                    bounce=0.9,
                    floor=0.0,
                    bounds=ROOM,
                ),
            ),
            name=f"box{i + 1}",
        )

    for i in range(3):
        scene.add(
            Sphere(
                radius=0.3,
                texture=checkerboard(4, color_a="#ffffff", color_b=("#e63946", "#2a9d8f", "#f4a261")[i]),
                position=(-1.5 + 1.5 * i, 3.5, 1.0),
                body=Body(velocity=(rng.uniform(-1, 1), 0, 0), gravity=-9.81, bounce=0.8, floor=0.0, bounds=ROOM),
            ),
            name=f"ball{i + 1}",
        )

    # Without gravity an object just drifts and spins, bouncing off the walls forever.
    scene.add(
        Box(size=0.6, texture="#8ecae6", position=(0, 3.2, -1.0),
            body=Body(velocity=(1.5, 0.7, 0.4), angular_velocity=(45, 90, 0), bounce=1.0, bounds=ROOM)),
        name="drifter",
    )
    return scene


if __name__ == "__main__":
    run(build_scene())
