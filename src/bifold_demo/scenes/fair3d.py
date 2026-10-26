"""The trade-fair hall with 3D people frozen in natural poses (a still life of ``bifold_demo.scenes.fair``).

The default scene (``fair``) animates its people; this one freezes everyone in one frame of an idle, talk,
listen or walk clip: groups chatting near the camera, exhibitors behind their counters, visitors at the
booths, people caught mid-stride in the aisles and a speaker on the stage. Nobody faces the camera. Useful
for checking avatars and detections on a still scene.

Prepare the people once (downloads and converts about 900 MB of Rocketbox files)::

    bifold-characters fetch --avatars Business_Female_01 Business_Female_02 ... --clips basic

(the names are in ``CAST``; ``bifold-characters fetch --set small`` also works), then run::

    bifold-demo --scene bifold_demo.scenes.fair3d

Without prepared people the hall is shown empty and the terminal says how to prepare them.
"""

from __future__ import annotations

import math
import random

from bifold_demo.characters.character import Character
from bifold_demo.characters.library import CharacterLibrary
from bifold_demo.characters.model import CharacterState, Layer
from bifold_demo.scenes import fair

SEED = 11
# a mix of women and men of different ages and backgrounds, two children (skin tones and ages: see the catalog)
CAST = [
    "Business_Female_01", "Business_Female_02", "Female_Adult_01", "Female_Adult_04", "Female_Adult_06",
    "Female_Adult_09", "Female_Adult_13", "Female_Party_02", "Business_Male_02", "Business_Male_05",
    "Business_Male_07", "Male_Adult_04", "Male_Adult_05", "Male_Adult_10", "Male_Adult_12", "Male_Adult_14",
    "Male_Adult_17", "Male_Adult_21", "Female_Child_01", "Male_Child_02",
]
MIN_ANGLE_TO_CAMERA = 40.0  # degrees between where a person faces and the direction to the camera
# chatting groups near the camera: (centre on the floor, number of people, include a child)
GROUPS = [((-1.9, 0.0, 1.4), 3, False), ((2.1, 0.0, 0.4), 2, False), ((-0.4, 0.0, -3.2), 3, True),
          ((3.4, 0.0, -4.6), 2, False), ((-4.2, 0.0, -3.8), 2, False)]
# people caught mid-stride: (position, heading)
WALKERS = [((-6.0, 0.0, fair.WALK_AISLES[0]), 90.0), ((4.5, 0.0, fair.WALK_AISLES[0] + 0.6), -90.0),
           ((-2.5, 0.0, fair.WALK_AISLES[1]), 90.0), ((1.2, 0.0, -10.0), 180.0), ((10.0, 0.0, fair.WALK_AISLES[1]), -90.0)]


def build_scene():
    scene = fair.build_scene(people=[])
    library = CharacterLibrary()
    available = library.available()
    names = [n for n in CAST if n in available] or sorted(available)
    if not names:
        print("[fair3d] no 3D people prepared yet; run: bifold-characters fetch --set small")
        return scene
    cast = Cast(library, names, random.Random(SEED))
    add_groups(scene, cast)
    add_walkers(scene, cast)
    add_booth_people(scene, cast)
    add_speaker(scene, cast)
    return scene


class Cast:
    """Hands out avatars (unique until all are used) and frozen poses that suit them."""

    def __init__(self, library: CharacterLibrary, names: list[str], rng: random.Random):
        self.library, self.rng = library, rng
        meta = library.available()
        self.sex = {n: meta[n].get("sex", "male") for n in names}
        self.children = [n for n in names if meta[n].get("age_group") == "child"]
        self.adults = [n for n in names if n not in self.children] or list(names)
        rng.shuffle(self.adults)
        self._next = 0
        self.clips = library.clips()
        self.roles = library.roles()

    def adult(self) -> str:
        name = self.adults[self._next % len(self.adults)]
        self._next += 1
        return name

    def child(self) -> str | None:
        return self.rng.choice(self.children) if self.children else None

    def pose(self, name: str, *roles: str) -> Layer:
        """One frame of a clip of the first available role, preferring the avatar's sex."""
        prefix = "f_" if self.sex.get(name) == "female" else "m_"
        for role in roles:
            clips = [self.clips[c] for c in self.roles.get(role, []) if c in self.clips]
            if clips:
                matching = [c for c in clips if c.name.startswith(prefix)] or clips
                clip = self.rng.choice(matching)
                return Layer(clip, self.rng.uniform(0, clip.duration))
        raise LookupError(f"no clip for any of {roles}")

    def place(self, scene, name: str, label: str, position, heading: float, *roles: str) -> None:
        heading = away_from_camera(position, heading)
        state = CharacterState(position=tuple(position), heading=heading, layers=[self.pose(name, *roles)])
        scene.add(Character(self.library.load(name), behaviour=lambda t, s=state: s), name=label)


def away_from_camera(position, heading: float, camera=fair.CAMERA_POSITION) -> float:
    """``heading`` turned just enough that the person faces at least MIN_ANGLE_TO_CAMERA away from the camera."""
    to_camera = math.degrees(math.atan2(camera[0] - position[0], camera[2] - position[2]))
    off = (heading - to_camera + 180) % 360 - 180
    if abs(off) >= MIN_ANGLE_TO_CAMERA:
        return heading
    return to_camera + math.copysign(MIN_ANGLE_TO_CAMERA, off if off else 1.0)


def _facing(position, target) -> float:
    return math.degrees(math.atan2(target[0] - position[0], target[2] - position[2]))


def add_groups(scene, cast: Cast) -> None:
    """Small groups talking, arranged so that nobody stands on the far side facing the camera."""
    for g, (centre, size, with_child) in enumerate(GROUPS):
        to_camera = _facing(centre, fair.CAMERA_POSITION)
        # angles around the centre measured from the camera direction: 0 = back to the camera, +-100 = side-on
        angles = {2: (95, -95), 3: (0, 105, -105)}[size]
        for k, angle in enumerate(angles):
            is_child = with_child and k == size - 1
            name = cast.child() if is_child else cast.adult()
            if name is None:
                name, is_child = cast.adult(), False
            radius = 0.55 if is_child else 0.7
            a = math.radians(to_camera + angle)
            position = (centre[0] + radius * math.sin(a), 0.0, centre[2] + radius * math.cos(a))
            roles = ("idle", "look_around") if is_child else (("talk", "listen", "idle") if k == 0 else ("listen", "talk", "idle"))
            cast.place(scene, name, f"group{g + 1}_{k + 1}", position, _facing(position, centre), *roles)


def add_walkers(scene, cast: Cast) -> None:
    for i, (position, heading) in enumerate(WALKERS):
        name = cast.adult()
        cast.place(scene, name, f"walker{i + 1}", position, heading, "walk", "stroll", "walk_slow")


def add_booth_people(scene, cast: Cast) -> None:
    """An exhibitor behind every counter, and a visitor at every second booth."""
    d = fair.BOOTH_DEPTH
    for index, origin, yaw in fair.booth_layout():
        at = fair._placer(origin, yaw)
        cast.place(scene, cast.adult(), f"staff{index}", at((0.9, fair.BOOTH_FLOOR, d / 2 - 1.75)), yaw, "idle", "talk")
        if index % 2:
            cast.place(scene, cast.adult(), f"guest{index}", at((0.7, 0.0, d / 2 + 0.45)), yaw + 180,
                       "listen", "idle")


def add_speaker(scene, cast: Cast) -> None:
    cast.place(scene, cast.adult(), "speaker", (2.5, fair.STAGE_HEIGHT, fair.STAGE_Z + 1.0), 30.0, "talk", "idle")
