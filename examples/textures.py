"""All the ways to put a texture on an object.

Run:  bifold-demo --scene examples/textures.py
Then drop an image named `screen.jpg`, `photo.png`, ... into the `uploads/` folder.
"""

import time

import cv2
import numpy as np

from bifold_demo import Box, Camera, Cylinder, Plane, Quad, Scene, Sphere, Texture, checkerboard, run, sample_image


def build_scene() -> Scene:
    scene = Scene(background=sample_image("bus.jpg") or (30, 30, 30), ambient=0.6)  # images work as backgrounds too
    scene.camera = Camera(position=(0, 1.8, 6.0), target=(0, 1.2, 0), fov=55)
    scene.add(Plane(size=20, uv_repeat=8, texture=checkerboard()), name="floor")

    # 1) An image file (jpg, png with transparency, webp, ...). Edit the file on disk and the object updates.
    scene.add(Quad(texture=sample_image("zidane.jpg"), height=1.6, position=(-2.2, 1.2, 0), lit=False), name="photo")

    # 2) A colour: (r, g, b) or "#rrggbb".
    scene.add(Sphere(radius=0.4, texture="#2a9d8f", position=(0, 0.4, 1.5)), name="ball")

    # 3) A NumPy array in OpenCV's BGR order (e.g. from cv2.imread, or generated).
    stripes = np.full((256, 256, 3), (255, 120, 0), np.uint8)  # blue
    stripes[:, (np.arange(256) // 32) % 2 == 0] = (0, 200, 255)  # orange stripes
    scene.add(Box(size=1.0, texture=stripes, position=(2.2, 0.5, 0.5)), name="crate")

    # 4) A Texture you redraw yourself: here a live clock, updated from a scene callback.
    screen = Texture.from_array(np.zeros((360, 640, 3), np.uint8))
    scene.add(Quad(texture=screen, height=1.2, position=(0, 2.2, -0.5), lit=False), name="screen")

    @scene.on_update
    def draw_clock(scene, t, dt):
        image = np.full((360, 640, 3), (60, 30, 20), np.uint8)
        cv2.putText(image, time.strftime("%H:%M:%S"), (90, 200), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255, 255, 255), 6)
        cv2.putText(image, f"t = {t:6.2f} s", (200, 290), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (180, 220, 255), 2)
        screen.update(image)  # BGR array, like everything from OpenCV

    # 5) A video file plays in simulation time:  texture="clip.mp4"  (paused with Space, restarted with R).
    scene.add(Cylinder(radius=0.4, height=1.2, texture=checkerboard(6), position=(0.9, 0.6, 0.8)), name="can")

    # 6) Swap any texture at any time, e.g. from a callback, another module, or the uploads folder:
    #      scene["photo"].texture = "dog.jpg"
    #      scene.set_texture("crate", some_numpy_array)
    scene.watch_folder("uploads")  # uploads/<object name>.<ext> -> texture of that object
    return scene


if __name__ == "__main__":
    run(build_scene())
