"""Simulation time for scenes."""

from __future__ import annotations

import time


class SimulationClock:
    """Wall-clock based simulation time that can be paused, sped up and reset."""

    def __init__(self, speed: float = 1.0):
        self.speed = speed
        self._elapsed = 0.0
        self._resumed_at: float | None = time.perf_counter()

    @property
    def paused(self) -> bool:
        return self._resumed_at is None

    def now(self) -> float:
        if self._resumed_at is None:
            return self._elapsed
        return self._elapsed + (time.perf_counter() - self._resumed_at) * self.speed

    def pause(self) -> None:
        if self._resumed_at is not None:
            self._elapsed = self.now()
            self._resumed_at = None

    def resume(self) -> None:
        if self._resumed_at is None:
            self._resumed_at = time.perf_counter()

    def toggle_pause(self) -> bool:
        """Pause or resume; returns ``True`` if the clock is now paused."""
        self.resume() if self.paused else self.pause()
        return self.paused

    def reset(self) -> None:
        self._elapsed = 0.0
        if self._resumed_at is not None:
            self._resumed_at = time.perf_counter()


class FixedStepClock(SimulationClock):
    """Advances by exactly ``1 / fps`` per frame, independent of how long rendering takes.

    Used for headless recording so the output video plays back at the right speed.
    """

    def __init__(self, fps: float = 30.0, speed: float = 1.0):
        super().__init__(speed)
        self.step = 1.0 / fps
        self._time = 0.0
        self._paused = False

    @property
    def paused(self) -> bool:
        return self._paused

    def now(self) -> float:
        return self._time

    def tick(self) -> None:
        if not self._paused:
            self._time += self.step * self.speed

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def reset(self) -> None:
        self._time = 0.0
