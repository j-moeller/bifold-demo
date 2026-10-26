"""The plain data the window and the editor pass each other (no PyTorch here)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

AGE_END = 40.0  # years: the age slider goes from -AGE_END to +AGE_END
GENDER_END = 1.25  # the gender slider's ends move a face this many female-to-male gaps


@dataclass(frozen=True)
class AgeGender:
    age: float  # years
    male: float  # probability the face is male, 0..1

    @property
    def gender(self) -> str:
        return "male" if self.male >= 0.5 else "female"

    @property
    def confidence(self) -> float:
        return max(self.male, 1 - self.male)


@dataclass
class Settings:
    """What the sliders and switches say."""

    age: float = 0.0  # years older (negative: younger)
    gender: float = 0.0  # -1 towards female ... +1 towards male
    enabled: bool = True  # False: show the camera picture as it is
    readout: bool = True  # let the judges read the faces (if they are loaded)

    @property
    def edits(self) -> bool:
        return self.enabled and (abs(self.age) >= 0.5 or abs(self.gender) >= 0.01)


@dataclass
class FaceResult:
    track_id: int
    box: tuple[float, float, float, float]
    quad: np.ndarray
    before: AgeGender | None = None
    after: AgeGender | None = None


@dataclass
class FrameResult:
    image: np.ndarray
    faces: list[FaceResult]
    timings: dict = field(default_factory=dict)  # seconds per step
