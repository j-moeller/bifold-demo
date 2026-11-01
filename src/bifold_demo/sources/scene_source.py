"""Frame source that renders a 3D scene (its 2D projection through the scene camera)."""

from __future__ import annotations

import time

from ..scene import Scene
from .base import Frame, FrameSource
from .clock import FixedStepClock, SimulationClock


class SceneSource(FrameSource):
    """Renders ``scene`` at ``width`` x ``height``, advancing it with ``clock``.

    ``windows``: a :class:`bifold_demo.sensor.WindowService`; its windows are rendered right after each frame, of the
    same moment, and delivered on the frame (``frame.windows``). It can be set later too; setting it tells the
    service the frames' size (and fails there when the service's sensor is for another view).
    """

    @property
    def windows(self):
        return self._windows

    @windows.setter
    def windows(self, service) -> None:
        if service is not None:
            service.bind((self.width, self.height))
        self._windows = service

    name = "scene"

    def __init__(self, scene: Scene, width: int = 1280, height: int = 720, fps: float = 30.0,
                 clock: SimulationClock | None = None, samples: int = 4, windows=None):
        self.scene = scene
        self.width, self.height, self.fps = width, height, fps
        self.clock = clock or SimulationClock()
        self.samples = samples
        self.windows = windows
        self.renderer = None
        self._index = 0
        self._last_t = None
        self._next_due = None

    def start(self) -> None:
        if self.renderer is None:
            from ..scene.renderer import Renderer

            self.renderer = Renderer(self.width, self.height, self.samples)

    def set_scene(self, scene: Scene) -> None:
        """Swap in a different scene (e.g. after editing the scene file); time restarts at 0."""
        self.scene = scene
        if self.renderer is not None:  # free the old scene's meshes, textures and joint palettes right away
            self.renderer._collect_garbage()
        self.reset()

    def reset(self) -> None:
        self.clock.reset()
        self.scene.reset()
        self._last_t = None

    def read(self, timeout: float = 1.0) -> Frame | None:
        if self.renderer is None:
            self.start()
        if not isinstance(self.clock, FixedStepClock):  # pace rendering to the target frame rate
            start = time.perf_counter()
            if self._next_due is None:
                self._next_due = start
            if start < self._next_due:
                time.sleep(min(self._next_due - start, timeout))
            self._next_due = max(self._next_due, start) + 1.0 / self.fps

        t = self.clock.now()
        dt = 0.0 if self._last_t is None else max(0.0, min(t - self._last_t, 0.25))
        self._last_t = t
        self.scene.update(t, dt)
        image = self.renderer.render(self.scene)
        self._index += 1
        frame = Frame(image=image, index=self._index, timestamp=time.perf_counter(), source=self.name, time=t)
        if self.windows is not None:  # before anything moves on: the windows show the frame's moment
            frame.windows = self.windows.for_scene(self.scene, self.renderer, frame)
        if isinstance(self.clock, FixedStepClock):
            self.clock.tick()
        return frame

    def stop(self) -> None:
        if self.renderer is not None:
            self.renderer.release()
            self.renderer = None
