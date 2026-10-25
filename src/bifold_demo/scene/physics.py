"""Deliberately simple physics: velocity, spin, gravity and bouncing off a floor or box walls."""

from __future__ import annotations

import numpy as np

from .transforms import vec3


class Body:
    """Attach to an object (``obj.body = Body(...)``) to simulate its motion.

    While an object has a body, the body owns its position and rotation: they start
    from the object's ``position``/``rotation`` (at time 0 and on every reset) and
    then evolve over simulation time.

    Args:
        velocity: initial velocity in units per second.
        angular_velocity: spin in degrees per second around the x, y and z axes.
        gravity: acceleration; a number is taken as the y component (e.g. ``-9.81``).
        bounce: fraction of speed kept when hitting the floor or a wall (0 = stop, 1 = perfectly elastic).
        floor: height of a floor to bounce on, or ``None``.
        bounds: ``((xmin, ymin, zmin), (xmax, ymax, zmax))`` box to keep the object inside, or ``None``.
        drag: velocity damping per second (0 = none).
    """

    def __init__(
        self,
        velocity=(0.0, 0.0, 0.0),
        angular_velocity=(0.0, 0.0, 0.0),
        gravity=0.0,
        bounce: float = 0.7,
        floor: float | None = None,
        bounds=None,
        drag: float = 0.0,
    ):
        self.initial_velocity = vec3(velocity, "velocity")
        self.angular_velocity = vec3(angular_velocity, "angular_velocity")
        self.gravity = np.array([0.0, float(gravity), 0.0]) if np.isscalar(gravity) else vec3(gravity, "gravity")
        self.bounce = float(bounce)
        self.floor = floor
        self.bounds = None if bounds is None else (vec3(bounds[0], "bounds"), vec3(bounds[1], "bounds"))
        self.drag = float(drag)
        self.position = np.zeros(3)
        self.rotation = np.zeros(3)
        self.velocity = self.initial_velocity.copy()

    def reset(self, position, rotation) -> None:
        self.position = vec3(position, "position")
        self.rotation = vec3(rotation, "rotation")
        self.velocity = self.initial_velocity.copy()

    def step(self, dt: float, extents) -> None:
        """Advance by ``dt`` seconds.

        ``extents(rotation)`` returns the ``(low, high)`` offsets of the object's
        bounding box relative to its position; it is only called when there is
        something to collide with.
        """
        self.velocity += self.gravity * dt
        if self.drag:
            self.velocity *= max(0.0, 1.0 - self.drag * dt)
        self.position += self.velocity * dt
        self.rotation += self.angular_velocity * dt
        if self.floor is None and self.bounds is None:
            return

        lower, upper = np.full(3, -np.inf), np.full(3, np.inf)
        if self.bounds is not None:
            lower, upper = self.bounds[0].copy(), self.bounds[1].copy()
        if self.floor is not None:
            lower[1] = max(lower[1], float(self.floor))
        low, high = extents(self.rotation)
        for axis in range(3):
            if self.position[axis] + low[axis] < lower[axis]:
                self.position[axis] = lower[axis] - low[axis]
                if self.velocity[axis] < 0:
                    self._rebound(axis)
            elif self.position[axis] + high[axis] > upper[axis]:
                self.position[axis] = upper[axis] - high[axis]
                if self.velocity[axis] > 0:
                    self._rebound(axis)

    def _rebound(self, axis: int) -> None:
        self.velocity[axis] *= -self.bounce
        # come to rest instead of jittering when gravity pins the object to a surface
        if self.gravity[axis] != 0 and abs(self.velocity[axis]) < 0.15:
            self.velocity[axis] = 0.0

    def __repr__(self) -> str:
        return f"Body(position={self.position.round(3).tolist()}, velocity={self.velocity.round(3).tolist()})"
