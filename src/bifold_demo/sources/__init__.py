"""Video sources: a webcam or a rendered 3D scene, with switching between them."""

from __future__ import annotations

from typing import Callable

from .base import Frame, FrameSource
from .clock import FixedStepClock, SimulationClock
from .scene_source import SceneSource
from .webcam import WebcamSource


class SourceSwitcher:
    """Holds several named sources and toggles between them.

    Sources are created and started lazily the first time they are selected and
    kept open afterwards so switching back is instant.
    """

    def __init__(self, factories: dict[str, Callable[[], FrameSource]], initial: str):
        if initial not in factories:
            raise ValueError(f"unknown source {initial!r}; choose from {list(factories)}")
        self._factories = dict(factories)
        self._sources: dict[str, FrameSource] = {}
        self.active_name = initial
        self.last_error: str | None = None

    @property
    def names(self) -> list[str]:
        return list(self._factories)

    def get(self, name: str) -> FrameSource:
        """Return source ``name``, creating and starting it if needed (may raise)."""
        if name not in self._sources:
            source = self._factories[name]()
            source.start()
            self._sources[name] = source
        return self._sources[name]

    @property
    def active(self) -> FrameSource:
        return self.get(self.active_name)

    def switch(self, name: str) -> bool:
        """Make ``name`` the active source; on failure keep the current one and set ``last_error``."""
        try:
            self.get(name)
        except Exception as exc:  # noqa: BLE001 - surface any device error to the user
            self.last_error = f"{name}: {exc}"
            return False
        self.active_name = name
        self.last_error = None
        return True

    def toggle(self) -> bool:
        names = self.names
        return self.switch(names[(names.index(self.active_name) + 1) % len(names)])

    def stop(self) -> None:
        for source in self._sources.values():
            source.stop()
        self._sources.clear()


__all__ = ["FixedStepClock", "Frame", "FrameSource", "SceneSource", "SimulationClock", "SourceSwitcher", "WebcamSource"]
