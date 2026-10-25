"""Default demo scene: a photo poster, a photo cube and a bouncing ball under a slowly drifting camera.

Objects (drop ``uploads/<name>.jpg`` to replace a texture): ``poster``, ``cube``, ``ball``, ``floor``.
"""

from bifold_demo.scene import (
    Body,
    Box,
    Camera,
    Plane,
    Quad,
    Scene,
    Sphere,
    checkerboard,
    gradient,
    oscillate,
    sample_image,
    spin,
)


def build_scene() -> Scene:
    scene = Scene(background=gradient(top=(110, 135, 165), bottom=(30, 32, 40)), ambient=0.55)
    scene.camera = Camera(
        position=oscillate(center=(0.0, 1.6, 5.4), amplitude=(1.0, 0.15, 0.0), period=14),
        target=(0.0, 1.15, 0.0),
        fov=50,
    )

    scene.add(Plane(size=(40, 40), uv_repeat=12, texture=checkerboard()), name="floor")

    # Sample photos shipped with Ultralytics (a bus with pedestrians, two people); colours if unavailable.
    scene.add(
        Quad(
            height=2.6,
            texture=sample_image("bus.jpg") or "#4a7bd0",
            position=(-1.7, 1.35, -0.3),
            rotation=oscillate(center=(0, 20, 0), amplitude=(0, 15, 0), period=9),
            lit=False,  # show the photo with its original colours
        ),
        name="poster",
    )
    scene.add(
        Box(
            size=1.7,
            texture=sample_image("zidane.jpg") or "#d06a4a",
            position=(1.8, 0.85, 0.2),
            rotation=spin(30, start=(0, -30, 0)),
        ),
        name="cube",
    )
    scene.add(
        Sphere(
            radius=0.35,
            texture=checkerboard(4, color_a="#ffffff", color_b="#ff7a1a"),
            position=(-0.4, 2.8, 1.4),
            body=Body(velocity=(0.8, 0, 0), angular_velocity=(0, 0, -150), gravity=-9.81, bounce=0.85,
                      floor=0.0, bounds=((-2.5, 0, -2), (2.5, 10, 3))),
        ),
        name="ball",
    )

    @scene.on_update
    def relaunch_ball(scene, t, dt):
        if int(t / 12) != int((t - dt) / 12):  # every 12 s
            ball = scene["ball"]
            ball.body.reset(ball.position, ball.rotation)

    return scene
