"""bifold-demo: real-time object detection on a webcam or a rendered 3D scene.

Quick start::

    from bifold_demo import Scene, Quad, Box, spin, run

    scene = Scene()
    scene.add(Quad(texture="my_photo.jpg", height=2, position=(0, 1, 0)), name="photo")
    scene.add(Box(texture="cat.jpg", position=(2, 0.5, 0), rotation=spin(45)), name="cube")
    scene["photo"].texture = "dog.jpg"   # swap textures at any time
    run(scene)

Importing the package switches ONNX Runtime's usage telemetry off (``ORT_DISABLE_TELEMETRY=1``, unless set already)
before anything can load ONNX Runtime: the face models, or Ultralytics running an ``.onnx`` YOLO model (see
:mod:`bifold_demo.recognition.runtime`).
"""

import os

os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")  # before ONNX Runtime is first loaded (see above)

from .app import DemoConfig, run  # (after the telemetry switch)
from .scene import (
    Body,
    Box,
    Camera,
    Cylinder,
    Fog,
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
    "Fog",
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
