"""Time-varying properties.

Any object property (``position``, ``rotation``, ``scale``, ``color``,
``texture``, ``visible``) and any camera property accepts either a constant or a
function of simulation time ``t`` (seconds)::

    box.position = lambda t: (math.sin(t), 0.5, 0)
    box.rotation = spin(90)                             # 90 degrees per second about Y
    box.texture = Keyframes({0: "cat.jpg", 5: "dog.jpg"}, loop=True)
"""

from __future__ import annotations

import bisect
import math

import numpy as np

from .transforms import vec3


def resolve(value, t: float):
    """Evaluate a property: call it with ``t`` if it is a function, else return it as is."""
    return value(t) if callable(value) else value


_EASINGS = {
    "linear": lambda a: a,
    "smooth": lambda a: a * a * (3 - 2 * a),
    "ease_in": lambda a: a * a,
    "ease_out": lambda a: 1 - (1 - a) * (1 - a),
}


class Keyframes:
    """Interpolate between values at given times.

    ``keys`` maps time (seconds) to a value. Numbers and vectors are interpolated
    (``easing``: ``"linear"``, ``"smooth"``, ``"ease_in"``, ``"ease_out"``); anything
    else (textures, booleans) switches at each key time.

    ``loop=True`` repeats the animation from the first key once the last key time is
    reached (so the last key is only a loop marker: give it the first key's value);
    ``pingpong=True`` plays it back and forth. Otherwise the first value is held
    before the first key and the last value after the last key.
    """

    def __init__(self, keys, *, loop: bool = False, pingpong: bool = False, easing: str = "linear"):
        items = sorted(dict(keys).items()) if isinstance(keys, dict) else sorted(keys, key=lambda kv: kv[0])
        if not items:
            raise ValueError("Keyframes needs at least one key")
        if easing not in _EASINGS:
            raise ValueError(f"unknown easing {easing!r}; choose from {sorted(_EASINGS)}")
        self.times = [float(t) for t, _ in items]
        self.values = [v for _, v in items]
        self.loop, self.pingpong, self.easing = loop, pingpong, _EASINGS[easing]
        self._numeric = all(_is_numeric(v) for v in self.values)
        if self._numeric:
            self._arrays = [np.asarray(v, dtype=np.float64) for v in self.values]

    @property
    def duration(self) -> float:
        return self.times[-1] - self.times[0]

    def _local_time(self, t: float) -> float:
        start, span = self.times[0], self.duration
        if span <= 0 or t <= start:
            return start
        if self.pingpong:
            phase = (t - start) % (2 * span)
            return start + (phase if phase <= span else 2 * span - phase)
        if self.loop:
            return start + (t - start) % span
        return min(t, self.times[-1])

    def __call__(self, t: float):
        t = self._local_time(t)
        i = bisect.bisect_right(self.times, t) - 1
        if i >= len(self.times) - 1:
            i = len(self.times) - 1
            return self._arrays[i] if self._numeric else self.values[i]
        i = max(i, 0)
        if not self._numeric:
            return self.values[i]
        t0, t1 = self.times[i], self.times[i + 1]
        a = self.easing((t - t0) / (t1 - t0)) if t1 > t0 else 1.0
        return self._arrays[i] + (self._arrays[i + 1] - self._arrays[i]) * a


def _is_numeric(value) -> bool:
    if isinstance(value, (bool, str)):
        return False
    try:
        return np.asarray(value).dtype.kind in "iuf"
    except (TypeError, ValueError):
        return False


def oscillate(center=(0.0, 0.0, 0.0), amplitude=(1.0, 0.0, 0.0), period: float = 2.0, phase: float = 0.0):
    """Sine-wave motion around ``center``: ``center + amplitude * sin(2*pi*(t/period + phase))``."""
    c, a = np.asarray(center, np.float64), np.asarray(amplitude, np.float64)
    return lambda t: c + a * math.sin(2 * math.pi * (t / period + phase))


def spin(degrees_per_second: float = 90.0, axis: str | tuple = "y", start=(0.0, 0.0, 0.0)):
    """Constant rotation (use for ``rotation``); ``axis`` is ``"x"``, ``"y"``, ``"z"`` or a 3-vector."""
    if isinstance(axis, str):
        axis = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis.lower()]
    rate, start = vec3(axis) * degrees_per_second, vec3(start)
    return lambda t: start + rate * t


def orbit(center=(0.0, 0.0, 0.0), radius: float = 3.0, period: float = 10.0, height: float = 0.0, phase: float = 0.0):
    """Circular motion in the horizontal plane around ``center`` (counter-clockwise seen from above)."""
    c = vec3(center)

    def position(t):
        angle = 2 * math.pi * (t / period + phase)
        return c + np.array([radius * math.sin(angle), height, radius * math.cos(angle)])

    return position


def linear(start=(0.0, 0.0, 0.0), velocity=(1.0, 0.0, 0.0)):
    """Straight-line motion: ``start + velocity * t``."""
    s, v = vec3(start), vec3(velocity)
    return lambda t: s + v * t
