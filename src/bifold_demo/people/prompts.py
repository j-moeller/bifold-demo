"""Random but reproducible descriptions of trade-fair visitors, turned into text-to-image prompts."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass

AGES = [("in their twenties", 0.3), ("in their thirties", 0.3), ("in their forties", 0.2),
        ("in their fifties", 0.13), ("in their sixties", 0.07)]
APPEARANCES = ["East Asian", "South Asian", "Southeast Asian", "Black", "White", "Latino", "Middle Eastern"]
OUTFITS = {
    "man": [
        "a navy business suit and a white shirt",
        "a grey blazer over a black t-shirt and dark jeans",
        "a company polo shirt and beige chinos",
        "a quarter-zip sweater and dark trousers",
        "a hoodie, jeans and white sneakers",
        "a light blue shirt with rolled-up sleeves and dark trousers",
        "a bomber jacket, a t-shirt and jeans",
        "a knitted cardigan, a shirt and corduroy trousers",
    ],
    "woman": [
        "a dark trouser suit",
        "a blazer over a blouse and tailored trousers",
        "a knee-length dress with a cardigan",
        "a company polo shirt and jeans",
        "a knit sweater, jeans and white sneakers",
        "a trench coat over business-casual clothes",
        "a blouse, a midi skirt and flat shoes",
        "a denim jacket, a t-shirt and black trousers",
    ],
}
EXTRAS = [
    "wearing a conference lanyard with a visitor badge",
    "wearing a lanyard badge and carrying a tote bag",
    "wearing a backpack and a lanyard badge",
    "holding a paper coffee cup",
    "holding a smartphone",
    "holding a folded brochure",
    "carrying a shoulder bag",
]
HEIGHT = {"man": (1.77, 0.07), "woman": (1.64, 0.065)}  # mean and spread in metres
POSES = ("standing", "walking")


@dataclass
class PersonSpec:
    """What one generated person should look like."""

    seed: int
    pose: str  # "standing" (facing the camera) or "walking" (side view, facing left)
    gender: str
    age: str
    appearance: str
    outfit: str
    extra: str
    height: float  # metres, used to size the person in the 3D scene

    @property
    def facing(self) -> str:
        return "front" if self.pose == "standing" else "left"

    def prompt(self) -> str:
        if self.pose == "walking":
            action = "walking to the left in mid-stride, seen from the side in full profile"
        else:
            action = "standing upright and facing the camera, relaxed natural pose, arms by the sides"
        return (
            f"Full-body photograph of a {self.appearance} {self.gender} {self.age}, wearing {self.outfit}, "
            f"{self.extra}, {action}. The entire body is visible from the top of the head to the shoes, "
            "with space above the head and below the feet. A single person, centred, on a plain light grey "
            "studio background, soft even studio lighting, sharp focus, realistic photo, 50mm lens."
        )

    def to_dict(self) -> dict:
        return asdict(self)


def random_spec(seed: int, pose: str | None = None, walking_share: float = 0.4) -> PersonSpec:
    """A reproducible random visitor; ``pose=None`` picks walking with probability ``walking_share``."""
    rng = random.Random(seed)
    gender = rng.choice(["man", "woman"])
    ages, weights = zip(*AGES)
    mean, spread = HEIGHT[gender]
    if pose is None:
        pose = "walking" if rng.random() < walking_share else "standing"
    if pose not in POSES:
        raise ValueError(f"pose must be one of {POSES}, got {pose!r}")
    return PersonSpec(
        seed=seed,
        pose=pose,
        gender=gender,
        age=rng.choices(ages, weights)[0],
        appearance=rng.choice(APPEARANCES),
        outfit=rng.choice(OUTFITS[gender]),
        extra=rng.choice(EXTRAS),
        height=round(min(max(rng.gauss(mean, spread), mean - 0.2), mean + 0.2), 2),
    )
