"""bifold-demo: real-time object detection on a webcam or a rendered 3D scene.

Quick start::

    from bifold_demo import Scene, Quad, Box, spin, run

    scene = Scene()
    scene.add(Quad(texture="my_photo.jpg", height=2, position=(0, 1, 0)), name="photo")
    scene.add(Box(texture="cat.jpg", position=(2, 0.5, 0), rotation=spin(45)), name="cube")
    scene["photo"].texture = "dog.jpg"   # swap textures at any time
    run(scene)
"""

from .app import DemoConfig, run
from .scene import (
    Body,
    Box,
    Camera,
    Cylinder,
    Keyframes,
    Mesh,
    Plane,
    Quad,
    Scene,
    SceneObject,
    Sphere,
    Texture,
    VideoTexture,
    checkerboard,
    gradient,
    linear,
    orbit,
    oscillate,
    sample_image,
    spin,
)

__version__ = "0.1.0"

__all__ = [
    "Body",
    "Box",
    "Camera",
    "Cylinder",
    "DemoConfig",
    "Keyframes",
    "Mesh",
    "Plane",
    "Quad",
    "Scene",
    "SceneObject",
    "Sphere",
    "Texture",
    "VideoTexture",
    "checkerboard",
    "gradient",
    "linear",
    "orbit",
    "oscillate",
    "run",
    "sample_image",
    "spin",
]
