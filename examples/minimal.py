"""The smallest useful scene: one picture standing in front of the camera.

Run:  python examples/minimal.py
 or:  bifold-demo --scene examples/minimal.py      (reloads when you save this file)
"""

from bifold_demo import Quad, Scene, run, sample_image


def build_scene() -> Scene:
    scene = Scene()
    # Any image path works here, e.g. texture="~/Pictures/dog.jpg"
    scene.add(Quad(texture=sample_image("bus.jpg"), height=2.5, position=(0, 1.2, 0)), name="photo")
    return scene


if __name__ == "__main__":
    run(build_scene())
