"""Animating position, rotation, texture and visibility over simulation time.

Run:  bifold-demo --scene examples/animation.py
"""

import math

from bifold_demo import Box, Camera, Keyframes, Plane, Quad, Scene, checkerboard, orbit, run, sample_image, spin

BUS = sample_image("bus.jpg")
PEOPLE = sample_image("zidane.jpg")


def build_scene() -> Scene:
    scene = Scene(background=(40, 44, 52))
    scene.camera = Camera(position=(0, 2.0, 7.0), target=(0, 1.0, 0), fov=50)
    scene.add(Plane(size=20, uv_repeat=6, texture=checkerboard()), name="floor")

    # 1) Keyframes: move along a path (smoothly), returning to the start every 8 seconds.
    scene.add(
        Box(
            size=1.2,
            texture=PEOPLE,
            position=Keyframes(
                {0: (-2.5, 0.6, 0), 2: (0, 0.6, 1.5), 4: (2.5, 0.6, 0), 6: (0, 0.6, -1.5), 8: (-2.5, 0.6, 0)},
                loop=True,
                easing="smooth",
            ),
            rotation=spin(60),  # 60 degrees per second around the vertical axis
        ),
        name="walker",
    )

    # 2) A plain function of time works for any property.
    scene.add(
        Quad(
            texture=BUS,
            height=2.0,
            position=lambda t: (0.0, 1.6 + 0.3 * math.sin(t * 2.0), -2.5),
            rotation=lambda t: (0.0, 25.0 * math.sin(t * 0.7), 0.0),
            lit=False,
        ),
        name="poster",
    )

    # 3) Textures can be keyframed too (values that are not numbers switch instead of blending).
    scene.add(
        Box(
            size=0.8,
            position=orbit(center=(0, 2.6, 0), radius=3.0, period=10),
            texture=Keyframes({0: BUS, 3: PEOPLE, 6: "#ffcc00", 9: BUS}, loop=True),  # 9 s loop
            rotation=spin(90, axis="x"),
        ),
        name="satellite",
    )

    # 4) Blinking: visibility as a function of time.
    scene.add(
        Quad(texture=PEOPLE, height=1.2, position=(3.2, 1.0, -1.0), face_camera=True,
             visible=lambda t: int(t) % 2 == 0),
        name="blinker",
    )

    # 5) Per-object callback for anything else, here: tint the walker red while it is on the right.
    def tint(obj, t, dt):
        obj.color = (255, 120, 120) if obj.current_position[0] > 1.5 else (255, 255, 255)

    scene["walker"].on_update = tint
    return scene


if __name__ == "__main__":
    run(build_scene())
