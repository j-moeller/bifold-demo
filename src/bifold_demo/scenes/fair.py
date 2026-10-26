"""Trade-fair hall with exhibitor booths and a keynote stage, seen by a fixed camera, full of people.

People are animated 3D characters (Microsoft Rocketbox avatars, prepared once with
``bifold-characters fetch --set conference``): some walk down the aisle towards the camera, stop and
turn around, others walk by along the cross aisles or wander about, small groups chat, exhibitors
stand behind their counters talking to visitors, and a speaker is on the stage. Nobody looks at
the camera. The crowd mixes women and men, skin tones and ages (see ``characters.crowd``).

Until the characters are prepared, 2D cut-out people stand in: from the ``people/`` folder
(``bifold-people generate -n 30``) or cut out of the Ultralytics sample photos.

Everything is laid out by the constants below. Objects you can re-texture by dropping
``uploads/<name>.jpg``: ``booth1_wall`` ... ``booth8_wall``, ``booth1_sign``, ``booth1_counter``,
``booth1_screen``, ``stage_screen``, ``banner``, ``floor``, and the people ``visitor1`` ...,
``walker1`` ..., ``staff1`` ..., ``speaker``.
"""

from __future__ import annotations

import math
import random
import zlib
from dataclasses import replace
from functools import partial

import cv2
import numpy as np

from bifold_demo.characters.behaviour import approach, chat, fits_period, stand, walk_by, wander
from bifold_demo.characters.crowd import age_group, describe_crowd, populate, sample_crowd
from bifold_demo.characters.library import CharacterLibrary, fetch_command
from bifold_demo.people import people_for_scene
from bifold_demo.scene import (
    Box,
    Camera,
    Cylinder,
    Fog,
    Keyframes,
    Plane,
    Quad,
    Scene,
    Sphere,
    Texture,
    oscillate,
    spin,
)
from bifold_demo.scene.transforms import rotation_y

# -- layout (metres; Y is up and the camera looks down the hall towards -Z) ----------------------
SEED = 7  # change for a different arrangement of people
CAMERA_POSITION = (0.0, 1.6, 6.0)  # the camera is fixed, like a mounted camera
CAMERA_TARGET = (0.0, 1.5, -12.0)
CAMERA_FOV = 55.0  # vertical field of view, degrees
HALL_WIDTH, HALL_LENGTH, HALL_HEIGHT = 44.0, 72.0, 12.0
HALL_FRONT = 12.0  # z of the wall behind the camera; the hall extends to HALL_FRONT - HALL_LENGTH
VISITORS = 6  # people standing in a half circle in front of the camera, open towards it
VISITOR_CIRCLE_CENTER = 3.0  # metres in front of the camera (0 = the camera is the circle's centre)
VISITOR_RADIUS = 2.6
VISITOR_SPREAD = 140.0  # degrees of the circle covered, centred on the far side
WALKERS = 11  # people walking by along the cross aisles
WALK_AISLES = (-7.0, -20.5, -34.0)  # z of the cross aisles
BOOTH_ROWS = (-11.5, -16.5, -24.5, -29.5)  # z of booth centres; one booth on each side per row
BOOTH_OFFSET = 7.0  # distance of booth centres from the centre line
STAGE_Z = -48.0
STAGE_HEIGHT = 1.0
PEOPLE_FOLDER = "people"
EVENT = "TECH EXPO 2026"
COMPANIES = [  # fictional exhibitors: name, tagline, main colour, accent colour (RGB)
    ("NOVASENSE", "Sensors for every edge", (24, 72, 160), (255, 184, 28)),
    ("KESTREL VISION", "See further", (18, 18, 24), (0, 200, 170)),
    ("BLUEWAVE ROBOTICS", "Automation made simple", (0, 110, 190), (255, 255, 255)),
    ("HELIOLAB", "Clean energy research", (240, 120, 20), (255, 236, 200)),
    ("ORBITA", "Satellite data platform", (60, 20, 110), (230, 90, 220)),
    ("LUMENIX", "Light. Measured.", (250, 250, 250), (230, 40, 60)),
    ("ARCFLOW", "Industrial AI pipelines", (20, 90, 60), (160, 230, 90)),
    ("TERRAMAP", "3D mapping at scale", (90, 60, 40), (250, 200, 120)),
]


def build_scene(people=None) -> Scene:
    """The hall with animated 3D people when they are prepared, otherwise 2D cut-outs.

    ``people``: use these 2D cut-outs instead (``[]`` = an empty hall).
    """
    rng = random.Random(SEED)
    scene = Scene(
        background=(205, 207, 212),
        ambient=0.62,
        light_direction=(-0.25, -1.0, -0.35),
        light_color=(255, 250, 240),
        fog=Fog(start=18, end=95, color=(214, 216, 222), strength=0.55),
    )
    scene.camera = Camera(position=CAMERA_POSITION, target=CAMERA_TARGET, fov=CAMERA_FOV, near=0.1)
    add_hall(scene)
    add_stage(scene)
    if people is None and PEOPLE_3D and _add_3d_people_if_ready(scene):
        for index, origin, yaw in booth_layout():
            add_booth(scene, index, origin, yaw, COMPANIES[(index - 1) % len(COMPANIES)], Crowd([], rng), rng)
        return scene
    people = Crowd(people_for_scene(PEOPLE_FOLDER) if people is None else people, rng)
    for index, origin, yaw in booth_layout():
        add_booth(scene, index, origin, yaw, COMPANIES[(index - 1) % len(COMPANIES)], people, rng)
    add_speaker(scene, people)
    add_visitors(scene, people, rng)
    add_walkers(scene, people, rng)
    return scene


def booth_layout() -> list[tuple[int, tuple, float]]:
    """``(number, centre, yaw)`` of every booth; left booths face +x, right booths face -x."""
    sides = [(-BOOTH_OFFSET, 90.0), (BOOTH_OFFSET, -90.0)]
    rows = ((z, side) for z in BOOTH_ROWS for side in sides)
    return [(i + 1, (x, 0.0, z), yaw) for i, (z, (x, yaw)) in enumerate(rows)]


# -- people -------------------------------------------------------------------------------------


class Crowd:
    """Hands out people from the library, cycling (and mirroring repeats) when there are few."""

    def __init__(self, people, rng: random.Random):
        self.rng = rng
        self.standing = [p for p in people if p.full_body and p.pose == "standing"]
        self.walking = [p for p in people if p.full_body and p.pose == "walking"]
        self.busts = [p for p in people if not p.full_body]
        self.full = self.standing + self.walking
        self._next: dict[str, int] = {}
        for group in (self.standing, self.walking, self.busts):
            rng.shuffle(group)

    def take(self, kind: str):
        """(person, mirror) of ``kind`` ("standing", "walking" or "bust"), falling back to other kinds."""
        group = {"standing": self.standing or self.full, "walking": self.walking or self.standing,
                 "bust": self.busts or self.standing or self.full}[kind]
        if not group:
            return None, False
        n = self._next.get(kind, 0)
        self._next[kind] = n + 1
        return group[n % len(group)], (n // len(group)) % 2 == 1  # every second round is mirrored


def _person_quad(person, mirror: bool, position, **kwargs) -> Quad:
    tint = kwargs.pop("tint", (255, 255, 255))
    return Quad(height=person.height, texture=person.texture(mirror), position=position, face_camera=True,
                lit=False, color=tint, **kwargs)


def add_visitors(scene: Scene, people: Crowd, rng: random.Random) -> None:
    """People standing in a half circle in front of the camera, swaying a little."""
    cx, _, cz = CAMERA_POSITION
    cz -= VISITOR_CIRCLE_CENTER
    for i in range(VISITORS):
        person, mirror = people.take("standing")
        if person is None:
            return
        angle = math.radians(-VISITOR_SPREAD / 2 + VISITOR_SPREAD * i / max(1, VISITORS - 1) + rng.uniform(-4, 4))
        r = VISITOR_RADIUS + rng.uniform(-0.25, 0.25)
        base = (cx + r * math.sin(angle), person.height / 2, cz - r * math.cos(angle))
        sway = oscillate(center=base, amplitude=(0.03, 0.005, 0.0), period=rng.uniform(3.0, 5.0), phase=rng.random())
        scene.add(_person_quad(person, mirror, sway), name=f"visitor{i + 1}")


def add_walkers(scene: Scene, people: Crowd, rng: random.Random) -> None:
    """People walking by along the cross aisles: wrapping around at the hall walls where the camera cannot see the
    walls, walking back and forth elsewhere."""
    span = HALL_WIDTH - 2.0
    for i in range(WALKERS):
        person, _ = people.take("walking")
        if person is None:
            return
        direction = 1 if i % 2 == 0 else -1
        z = WALK_AISLES[i % len(WALK_AISLES)] + rng.uniform(-0.5, 0.5)
        speed = rng.uniform(0.9, 1.5)
        start = rng.uniform(0, span)
        cadence = 1.8 * speed  # steps per second
        wrap = not (in_view((-span / 2, z)) or in_view((span / 2, z)))
        ahead = person.facing_texture("right" if direction > 0 else "left")

        def position(t, start=start, speed=speed, direction=direction, z=z, h=person.height, cadence=cadence,
                     wrap=wrap):
            if wrap:
                x = (start + direction * speed * t) % span - span / 2
            else:  # out and back: u runs over twice the span
                u = (start + speed * t) % (2 * span)
                x = direction * ((u if u < span else 2 * span - u) - span / 2)
            return (x, h / 2 + 0.025 * abs(math.sin(math.pi * cadence * t)), z)

        if wrap:
            texture = ahead
        else:  # facing the other way while walking back
            back = person.facing_texture("left" if direction > 0 else "right")
            turn, again, period = (span - start) / speed, (2 * span - start) / speed, 2 * span / speed
            texture = Keyframes({0.0: ahead, turn: back, again: ahead, period: ahead}, loop=True)
        scene.add(Quad(height=person.height, texture=texture, position=position, face_camera=True, lit=False),
                  name=f"walker{i + 1}")


# -- animated 3D people -------------------------------------------------------------------------

PEOPLE_3D = True  # False: always use the 2D cut-outs
CROWD_SEED = 5  # change for a different crowd and different timings
# walk from far away towards the camera, stop (3.5-5 m from it, so the feet stay in the picture), turn around,
# walk back: (start, stop) on the floor
APPROACHES = [((-0.8, -16.0), (-1.7, 2.4)), ((1.4, -23.5), (1.6, 1.2)), ((0.3, -12.5), (0.6, 2.0))]
# walk across the hall: (z, direction); people vanish at the side walls where the camera cannot see the wall,
# elsewhere they turn around there and walk back
CROSSINGS = [(-7.0, 1), (-6.2, -1), (-7.8, 1), (-20.5, -1), (-19.8, 1), (-21.3, -1), (-34.0, 1)]
WANDER_AREAS = [((-8.0, -4.5), (-4.0, 1.5)), ((5.8, -5.4), (8.2, -3.4)), ((-3.0, -42.0), (3.0, -36.0))]
CHAT_GROUPS = [((4.6, -2.4), 2), ((3.8, -12.5), 3), ((-3.6, -15.6), 2)]  # (centre, size); a child may join
MIN_DISTANCE = 0.6  # metres people keep from each other (walk timings are shifted until they do)
LOOP_PERIOD = 360.0  # seconds; moving people's loops are stretched to fit it, so the hall repeats every 6 min
FRAME_ASPECT = 16 / 9  # the widest picture the camera is used with (for the "out of view" checks)
MOVING = ("approach", "walker", "wanderer")  # kinds of people whose timings keep_apart may shift


def shared_library():
    """The character library, kept across scene reloads when it can be (loading avatars takes a while)."""
    shared = getattr(CharacterLibrary, "shared", None)
    return shared() if shared is not None else CharacterLibrary()


def _add_3d_people_if_ready(scene: Scene) -> bool:
    """Add the animated people, or say why not (then the hall gets 2D cut-outs)."""
    for attempt in (1, 2):
        try:
            library = shared_library()
            if not crowd_ready(library):
                missing = library.missing() if hasattr(library, "missing") else ["avatars"]
                if "avatars" in missing:
                    print("[fair] no 3D people prepared yet (run: bifold-characters fetch --set conference, or set "
                          "BIFOLD_CHARACTERS to the folder you prepared them in); showing 2D cut-outs instead")
                else:  # avatars, but their walk or idle clips are missing or damaged
                    print(f"[fair] the prepared 3D people have no {' or '.join(missing)} clips (run: "
                          f"{fetch_command('--avatars --clips basic', library.folder)}); showing 2D cut-outs instead")
                return False
            add_3d_people(scene, library, random.Random(CROWD_SEED))
            return True
        except Exception as error:  # noqa: BLE001 - e.g. a damaged file in the character cache: the demo still runs
            if attempt == 1:  # the library leaves out an avatar whose files turn out to be damaged: try without it
                print(f"[fair] trouble setting up the 3D people ({type(error).__name__}: {error}); trying again")
            else:
                print(f"[fair] could not set up the 3D people ({type(error).__name__}: {error}); "
                      "showing 2D cut-outs instead")
    return False


def crowd_ready(library) -> bool:
    """Prepared avatars plus walk and idle clips (turns without clips are done by hand)."""
    roles = library.roles()
    return bool(library.available()) and bool(roles.get("walk")) and bool(roles.get("idle"))


def in_view(point, margin: float = 0.5, height: float = 1.9, aspect: float = FRAME_ASPECT) -> bool:
    """Whether someone standing at ``point`` (``(x, z)`` on the floor) can show up in the fixed camera's picture.

    ``margin``: how far the body reaches around the point (metres). Walls and booths in the way are ignored.
    """
    camera = Camera(position=CAMERA_POSITION, target=CAMERA_TARGET, fov=CAMERA_FOV, near=0.1)
    clip_from_world = camera.projection_matrix(aspect) @ camera.view_matrix()
    x, z = point
    corners = np.array([(x + dx, y, z + dz, 1.0) for dx in (-margin, margin) for dz in (-margin, margin)
                        for y in (0.0, height)])
    clip = corners @ clip_from_world.T
    behind = clip[:, 3] <= 1e-6
    if behind.any():  # all behind the camera, or right next to it
        return not behind.all()
    ndc = clip[:, :2] / clip[:, 3:]
    return bool(np.all(ndc.min(axis=0) <= 1.0) and np.all(ndc.max(axis=0) >= -1.0))


def _bearing(frm, to) -> float:
    return math.degrees(math.atan2(to[0] - frm[0], to[1] - frm[1]))


class Raised:
    """A behaviour whose person stands ``height`` metres above the floor (on the stage, on a booth platform).

    Everything else (``duration``, ``offset``, ``face_heading`` ...) is the ``inner`` behaviour's.
    """

    def __init__(self, inner, height: float):
        self.__dict__.update(inner=inner, height=float(height))

    def __call__(self, t: float):
        state = self.inner(t)
        x, y, z = state.position
        return replace(state, position=(x, y + self.height, z))

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def __setattr__(self, name, value):
        setattr(self.inner, name, value)


def _raised(make, height: float):
    """A behaviour factory whose people stand ``height`` metres above the floor."""
    return lambda **kwargs: Raised(make(**kwargs), height)


def people_slots():
    """``(kind, behaviour factory, child allowed, prefers business clothes)`` for everyone in the hall."""
    cam = CAMERA_POSITION
    booths = booth_layout()
    booth_points = [(o[0], 1.5, o[2]) for _, o, _ in booths]
    slots = []
    moving = {"camera": cam, "look_at": booth_points, "period": LOOP_PERIOD}  # all loops fit LOOP_PERIOD
    for (x0, z0), (x1, z1) in APPROACHES:  # straight, in separate lanes (their timings cannot keep them apart)
        slots.append(("approach", partial(approach, (x0, 0.0, z0), (x1, 0.0, z1), stop_distance=(3.5, 5.0), bow=0.0,
                                          **moving), False, False))
    half = HALL_WIDTH / 2 - 1.0
    for z, direction in CROSSINGS:
        if in_view((-half, z)) or in_view((half, z)):  # they would be seen vanishing: turn round earlier instead
            ends = [(-(half - 4.0) * direction, 0.0, z), ((half - 4.0) * direction, 0.0, z)]
            make = partial(walk_by, ends, return_mode="turn", pause=(0.5, 2.5), **moving)
        else:
            ends = [(-half * direction, 0.0, z), (half * direction, 0.0, z)]
            make = partial(walk_by, ends, return_mode="teleport", **moving)
        slots.append(("walker", make, False, False))
    for area in WANDER_AREAS:
        slots.append(("wanderer", partial(wander, area, **moving), True, False))
    for g, ((cx, cz), size) in enumerate(CHAT_GROUPS):
        to_camera = _bearing((cx, cz), (cam[0], cam[2]))
        angles = {2: (95, -95), 3: (0, 105, -105)}[size]  # nobody on the far side, facing the camera
        spots = [(cx + 0.7 * math.sin(math.radians(to_camera + a)), cz + 0.7 * math.cos(math.radians(to_camera + a)))
                 for a in angles]
        for k, (x, z) in enumerate(spots):
            partners = [(px, 1.6, pz) for j, (px, pz) in enumerate(spots) if j != k]
            slots.append((f"chat{g + 1}_", partial(chat, (x, 0.0, z), (cx, 0.0, cz), partners=partners, camera=cam),
                          k == size - 1, False))
    d = BOOTH_DEPTH
    for index, origin, yaw in booths:
        at = _placer(origin, yaw)
        staff = at((0.9, 0.0, d / 2 - 1.75))  # on the platform, far enough behind the counter for gestures
        make = partial(stand, staff, heading=yaw, roles=("idle", "look_around", "talk"), camera=cam)
        slots.append(("staff", _raised(make, BOOTH_FLOOR), False, True))
        if index % 2:
            guest = at((0.7, 0.0, d / 2 + 0.45))
            slots.append(("guest", partial(chat, guest, staff, partners=[(staff[0], 1.6, staff[2])],
                                           roles=("listen", "talk"), camera=cam), False, False))
    speaker = partial(stand, (2.5, 0.0, STAGE_Z + 1.0), heading=20.0, roles=("talk", "idle"), camera=cam)
    slots.append(("speaker", _raised(speaker, STAGE_HEIGHT), False, True))
    return slots


def add_3d_people(scene: Scene, library, rng: random.Random) -> list:
    """Fill the hall with animated people chosen by :func:`~bifold_demo.characters.crowd.sample_crowd`.

    Nothing is added to ``scene`` when this fails half way.
    """
    slots = people_slots()
    catalog = library.available()
    names = assign_people(sample_crowd(catalog, len(slots), rng), slots, catalog, rng)
    people = populate(library, names, [make for _, make, _, _ in slots], rng=rng)
    left = keep_apart([p.behaviour for p in people], [kind for kind, *_ in slots], rng)
    if left:
        print(f"[fair] {left} pair(s) of people still walk through each other now and then "
              f"(try another CROWD_SEED)")
    counts: dict[str, int] = {}
    for (kind, *_), person in zip(slots, people):
        counts[kind] = counts.get(kind, 0) + 1
        scene.add(person, name=f"{kind}{counts[kind]}" if kind != "speaker" else "speaker")
    summary = describe_crowd(catalog, names)
    print(f"[fair] {len(people)} animated people ({summary['unique']} different avatars): "
          + "; ".join(f"{key.replace('_', ' ')} " + ", ".join(f"{v} {k}" for k, v in summary[key].items())
                      for key in ("sex", "age_group", "skin_band")))
    return people


def assign_people(names: list[str], slots, catalog: dict, rng: random.Random) -> list[str]:
    """Order ``names`` to fit the slots: children join chat groups (or wander), business clothes behind counters,
    and nobody talks to their own double. Otherwise the order of ``names`` is kept (``sample_crowd`` spreads the
    repeats of an avatar out over it)."""
    def is_child(name):
        return age_group(catalog.get(name, {})) == "child"

    pool = list(names)
    result: list[str | None] = [None] * len(slots)
    kids = [n for n in pool if is_child(n)]
    child_slots = sorted((i for i, slot in enumerate(slots) if slot[2]),
                         key=lambda i: not slots[i][0].startswith("chat"))  # with grown-ups first, then wandering
    for i in child_slots[:len(kids)]:
        result[i] = kids.pop(0)
        pool.remove(result[i])
    formal = [n for n in pool if not is_child(n) and catalog.get(n, {}).get("outfit") in ("business", "smart_casual")]
    for i, (_, _, _, business) in enumerate(slots):
        if business and result[i] is None:
            options = formal or [n for n in pool if not is_child(n)]
            if options:
                result[i] = options[rng.randrange(len(options))]
                pool.remove(result[i])
                if result[i] in formal:
                    formal.remove(result[i])
    rest = [n for n in pool if not is_child(n)]
    grown_ups = [n for n in names if not is_child(n)] or list(names)
    for i in range(len(slots)):
        if result[i] is None:  # children left over (more than child places) are replaced by grown-ups
            result[i] = rest.pop(0) if rest else rng.choice(grown_ups)
    _separate_doubles(result, slots, is_child, rng)
    return result


def conversation_groups(slots) -> list[list[int]]:
    """Slot numbers of people who talk to each other: the chat groups, and each exhibitor with their visitor."""
    groups: list[list[int]] = []
    for i, (kind, *_) in enumerate(slots):
        if kind.startswith("chat"):
            if i and slots[i - 1][0] == kind:
                groups[-1].append(i)
            else:
                groups.append([i])
        elif kind == "guest" and i and slots[i - 1][0] == "staff":
            groups.append([i - 1, i])
    return groups


def _separate_doubles(result: list[str], slots, is_child, rng: random.Random) -> None:
    """Swap people between slots so that no conversation has the same avatar twice (where the crowd allows)."""
    groups = conversation_groups(slots)
    group_of = {i: g for g, members in enumerate(groups) for i in members}

    def others(i):
        g = group_of.get(i)
        return {result[j] for j in groups[g] if j != i} if g is not None else set()

    for members in groups:
        for i in sorted(members, key=lambda i: slots[i][3]):  # keep the formally dressed exhibitor
            if result[i] not in others(i):
                continue
            options = [j for j in range(len(slots)) if group_of.get(j, -1) != group_of[i] and not slots[j][3]
                       and is_child(result[j]) == is_child(result[i]) and result[j] not in others(i)
                       and result[i] not in others(j)]
            if options:
                j = rng.choice(options)
                result[i], result[j] = result[j], result[i]


def _track(behaviour, times) -> np.ndarray:
    """``(x, z)`` of a person at ``times``; nan while hidden."""
    if hasattr(behaviour, "positions"):
        return behaviour.positions(times)
    track = np.full((len(times), 2), np.nan)
    for k, t in enumerate(times):
        state = behaviour(float(t))
        if state.visible:
            track[k] = state.position[0], state.position[2]
    return track


def _closest(track, tracks) -> np.ndarray:
    """Smallest distance over time between ``track`` (..., samples, 2) and each of ``tracks`` (people, samples, 2)."""
    distance = np.hypot(tracks[..., 0] - track[..., 0], tracks[..., 1] - track[..., 1])
    return np.where(np.isnan(distance), np.inf, distance).min(axis=-1)  # nan: someone is out of sight


def keep_apart(behaviours, kinds, rng: random.Random, period: float = LOOP_PERIOD, step: float = 0.1,
               rounds: int = 30) -> int:
    """Shift the timing of moving people until nobody ever comes closer than ``MIN_DISTANCE`` to anyone else.

    Moving people (``MOVING`` kinds) loop in a whole fraction of ``period`` (their behaviours' ``period``), so after
    ``period`` seconds everyone is back where they started and checking one period covers all time; a loop that
    does not fit is only checked for one period. Two people whose paths cross, looping in ``period / k1`` and
    ``period / k2`` seconds, meet at the crossing every ``period / lcm(k1, k2)`` seconds of timing difference: they
    can be kept apart when that is longer than the time they take to pass it (true for the hall's 20-90 s loops on
    crossing paths). People walking along nearly the same line for a while need separate lanes instead. Standing
    people never move out of the way. Returns the number of pairs of people who still come too close (0 = none).
    """
    apart = MIN_DISTANCE + 0.05  # aim a little further apart, for passes between the samples
    samples = round(period / step)
    times = np.arange(samples) * step
    moving = [kind in MOVING and hasattr(b, "offset") for b, kind in zip(behaviours, kinds)]
    movers = [i for i, m in enumerate(moving) if m]
    if not movers:
        return 0
    tracks = np.stack([_track(b, times) if m else np.repeat(_track(b, times[:1]), samples, axis=0)
                       for b, m in zip(behaviours, moving)])
    seen = ~np.isnan(tracks)
    lows = np.where(seen, tracks, np.inf).min(axis=1) - apart  # the area each person ever covers, plus a margin
    highs = np.where(seen, tracks, -np.inf).max(axis=1) + apart

    def near(k):  # everyone k could ever meet
        return [j for j in range(len(tracks)) if j != k and np.all(highs[j] >= lows[k]) and np.all(lows[j] <= highs[k])]

    def conflicts(limit):
        closest = _closest(tracks[movers][:, None], tracks[None])  # (movers, people)
        return [(i, j) for row, i in enumerate(movers) for j in np.nonzero(closest[row] < limit)[0]
                if j != i and (not moving[j] or j > i)]

    tries: dict[tuple, int] = {}  # a pair no shift keeps apart (a walkway too close to someone standing) gives up
    for _ in range(rounds):
        pairs = [pair for pair in conflicts(apart) if tries.get(pair, 0) < 4]
        if not pairs:
            break
        for i, j in pairs:
            tries[i, j] = tries.get((i, j), 0) + 1
            k = [x for x in (i, j) if moving[x]][tries[i, j] % 2 if moving[i] and moving[j] else 0]
            b = behaviours[k]
            if fits_period(b.duration, period):  # try shifts across the loop; shifting by whole steps rolls the track
                own = max(1, round(b.duration / step))
                shifts = np.unique(np.linspace(0, own, min(own, 64), endpoint=False).astype(int))
                rolled = tracks[k][(np.arange(samples)[None, :] + shifts[:, None]) % samples]
                others = near(k)
                closest = np.stack([_closest(rolled, tracks[o]) for o in others] or [np.full(len(shifts), np.inf)])
                # fewest people too close first, then the most room
                best = int(np.lexsort((-closest.min(axis=0), (closest < apart).sum(axis=0)))[0])
                b.offset = (b.offset + shifts[best] * step) % b.duration
                tracks[k] = rolled[best]
            else:
                b.offset = rng.uniform(0.0, b.duration)
                tracks[k] = _track(b, times)
    return len(conflicts(MIN_DISTANCE))


def close_encounters(behaviours, seconds: float, step: float = 0.1, distance: float = MIN_DISTANCE):
    """``(first, second, time, distance)`` for every pair of people closer than ``distance`` within ``seconds``."""
    times = np.arange(0.0, seconds, step)
    tracks = np.stack([_track(b, times) for b in behaviours])
    found = []
    for i in range(len(tracks) - 1):
        d = np.hypot(tracks[i + 1:, :, 0] - tracks[i, :, 0], tracks[i + 1:, :, 1] - tracks[i, :, 1])
        d = np.where(np.isnan(d), np.inf, d)
        for j in np.nonzero(d.min(axis=1) < distance)[0]:
            k = int(np.argmin(d[j]))
            found.append((i, i + 1 + int(j), float(times[k]), float(d[j, k])))
    return found


# -- booths -------------------------------------------------------------------------------------

BOOTH_WIDTH, BOOTH_DEPTH, BOOTH_WALL_HEIGHT = 5.0, 4.0, 3.0
BOOTH_FLOOR = 0.08  # height of the booth platform


def _placer(origin, yaw: float):
    """Maps booth-local positions (the booth opens towards local +Z) to the hall."""
    rotation = rotation_y(yaw)[:3, :3]
    origin = np.asarray(origin, dtype=np.float64)
    return lambda local: tuple(origin + rotation @ np.asarray(local, dtype=np.float64))


def add_booth(scene: Scene, index: int, origin, yaw: float, company, people: Crowd, rng: random.Random) -> None:
    name, tagline, main, accent = company
    at = _placer(origin, yaw)
    turn = (0.0, yaw, 0.0)
    w, d, h = BOOTH_WIDTH, BOOTH_DEPTH, BOOTH_WALL_HEIGHT
    prefix = f"booth{index}"

    scene.add(Box(size=(w, BOOTH_FLOOR, d), texture="#cfc8bd", position=at((0, BOOTH_FLOOR / 2, 0)), rotation=turn),
              name=f"{prefix}_floor")
    scene.add(Box(size=(w, h, 0.15), texture=booth_wall_texture(name, tagline, main, accent),
                  position=at((0, h / 2, -d / 2)), rotation=turn), name=f"{prefix}_wall")
    # a side wall on the far side, so the booth opens towards the camera; it covers the back half only, so it does
    # not hide the exhibitor of the booth behind
    far_side = w / 2 if yaw > 0 else -w / 2
    scene.add(Box(size=(0.15, h, d / 2), texture=main, position=at((far_side, h / 2, -d / 4)), rotation=turn),
              name=f"{prefix}_side")
    scene.add(Box(size=(1.8, 1.05, 0.6), texture=logo_texture(name, main, accent), position=at((0.9, 0.525, d / 2 - 0.9)),
                  rotation=turn), name=f"{prefix}_counter")
    scene.add(Box(size=(1.75, 1.05, 0.08), texture="#202024", position=at((-0.8, 1.9, -d / 2 + 0.13)), rotation=turn),
              name=f"{prefix}_bezel")
    slides = [slide_texture(name, title, main, accent, k) for k, title in enumerate(
        ("Product overview", "Live demo", tagline))]
    scene.add(Quad(size=(1.6, 0.9), position=at((-0.8, 1.9, -d / 2 + 0.18)), rotation=turn, lit=False,
                   texture=Keyframes({0: slides[0], 6: slides[1], 12: slides[2], 18: slides[0]}, loop=True)),
              name=f"{prefix}_screen")
    scene.add(Box(size=(2.6, 0.9, 2.6), texture=logo_texture(name, main, accent), position=at((0, 5.2, 0)),
                  rotation=spin(rng.uniform(8, 14) * rng.choice((-1, 1)), start=turn)), name=f"{prefix}_sign")
    scene.add(Cylinder(radius=0.25, height=0.5, texture="#e8e4dc", position=at((w / 2 - 0.45, 0.33, d / 2 - 0.45))),
              name=f"{prefix}_pot")
    scene.add(Sphere(radius=0.45, texture=foliage_texture(), position=at((w / 2 - 0.45, 0.98, d / 2 - 0.45)),
                     rotation=(0, rng.uniform(0, 360), 0)), name=f"{prefix}_plant")

    # exhibitor behind the counter (legs hidden by it) and sometimes a visitor in front of it
    staff, mirror = people.take("bust")
    if staff is not None:
        top = 1.72 + rng.uniform(-0.08, 0.08)
        y = top - staff.height / 2 if not staff.full_body else staff.height / 2
        scene.add(_person_quad(staff, mirror, at((0.9 + rng.uniform(-0.3, 0.3), y, d / 2 - 1.55))),
                  name=f"staff{index}")
    if rng.random() < 0.5:
        guest, mirror = people.take("standing")
        if guest is not None:
            scene.add(_person_quad(guest, mirror, at((rng.uniform(-0.6, 1.4), guest.height / 2, d / 2 + 0.25))),
                      name=f"guest{index}")


# -- hall and stage -----------------------------------------------------------------------------


def add_hall(scene: Scene) -> None:
    w, length, h = HALL_WIDTH, HALL_LENGTH, HALL_HEIGHT
    mid = HALL_FRONT - length / 2
    back = HALL_FRONT - length
    scene.add(Plane(size=(w, length), uv_repeat=(w / 4, length / 4), texture=floor_texture(), position=(0, 0, mid)),
              name="floor")
    # carpets lie flat on the floor, below the people's blob shadows (renderer.SHADOW_LIFT above their feet)
    carpet = carpet_texture((150, 32, 40))
    scene.add(Plane(size=(5.0, length - 16), uv_repeat=(1, (length - 16) / 5), texture=carpet,
                    position=(0, 0.004, HALL_FRONT - (length - 16) / 2)), name="carpet_main")
    for i, z in enumerate(WALK_AISLES):
        scene.add(Plane(size=(w - 2, 3.2), uv_repeat=((w - 2) / 5, 1), texture=carpet_texture((40, 60, 120)),
                        position=(0, 0.008, z)), name=f"carpet_aisle{i + 1}")

    wall = wall_texture()
    scene.add(Plane(size=(w, h), uv_repeat=(w / 6, h / 6), texture=wall, position=(0, h / 2, back), rotation=(90, 0, 0)),
              name="wall_back")
    for side, sign in (("left", -1), ("right", 1)):
        scene.add(Plane(size=(length, h), uv_repeat=(length / 6, h / 6), texture=wall,
                        position=(sign * w / 2, h / 2, mid), rotation=(90, -90 * sign, 0)), name=f"wall_{side}")
        for k in range(6):  # high windows
            z = HALL_FRONT - 8 - k * 11
            scene.add(Quad(size=(7.0, 2.6), texture=window_texture(), lit=False,
                           position=(sign * (w / 2 - 0.05), h - 2.4, z), rotation=(0, -90 * sign, 0)),
                      name=f"window_{side}{k + 1}")
        for k in range(8):  # pillars along the walls
            scene.add(Box(size=(0.9, h, 0.9), texture="#bfc0c2", position=(sign * (w / 2 - 0.45), h / 2, HALL_FRONT - 4 - k * 9)),
                      name=f"pillar_{side}{k + 1}")
    scene.add(Plane(size=(w, length), uv_repeat=(w / 8, length / 8), texture=ceiling_texture(), lit=False,
                    position=(0, h, mid), rotation=(180, 0, 0)), name="ceiling")
    for k in range(8):  # roof trusses
        scene.add(Box(size=(w, 0.5, 0.5), texture="#3a3d42", position=(0, h - 1.2, HALL_FRONT - 6 - k * 8.5)),
                  name=f"truss{k + 1}")


def add_stage(scene: Scene) -> None:
    z = STAGE_Z
    scene.add(Box(size=(16, STAGE_HEIGHT, 7), texture="#26272c", position=(0, STAGE_HEIGHT / 2, z)), name="stage")
    keynote = [slide_texture(EVENT, title, (16, 24, 48), (80, 200, 255), k)
               for k, title in enumerate(("Keynote: the next decade of AI", "Live demos all day", "Welcome!"))]
    scene.add(Quad(size=(14, 6.2), position=(0, 4.6, z - 3.2), lit=False,
                   texture=Keyframes({0: keynote[0], 8: keynote[1], 16: keynote[2], 24: keynote[0]}, loop=True)),
              name="stage_screen")
    for x in (-7.6, 7.6):
        scene.add(Box(size=(0.4, 8.6, 0.4), texture="#5a5d63", position=(x, 4.3, z - 3.0)), name=f"stage_truss{'lr'[x > 0]}")
    scene.add(Box(size=(15.6, 0.4, 0.4), texture="#5a5d63", position=(0, 8.4, z - 3.0)), name="stage_truss_top")
    scene.add(Quad(size=(20, 2.6), texture=banner_texture(EVENT), lit=False, position=(0, 10.0, z - 3.3)), name="banner")
    scene.add(Box(size=(0.7, 1.15, 0.5), texture=logo_texture("EXPO", (16, 24, 48), (80, 200, 255)),
                  position=(-4.5, 1.575, z + 1.5)), name="lectern")


def add_speaker(scene: Scene, people: Crowd) -> None:
    speaker, mirror = people.take("standing")
    if speaker is not None:
        scene.add(_person_quad(speaker, mirror, (2.5, STAGE_HEIGHT + speaker.height / 2, STAGE_Z + 1.0)), name="speaker")


# -- procedural textures (drawn with OpenCV in BGR) -----------------------------------------------


def _bgr(rgb) -> tuple[int, int, int]:
    return int(rgb[2]), int(rgb[1]), int(rgb[0])


def _text(image, text, center, max_width, max_height, color, font=cv2.FONT_HERSHEY_DUPLEX, thickness=None):
    """Draw ``text`` centred at ``center``, scaled to fit the box."""
    base_thickness = thickness or 2
    (tw, th), _ = cv2.getTextSize(text, font, 1.0, base_thickness)
    scale = min(max_width / max(tw, 1), max_height / max(th, 1))
    thickness = thickness or max(1, round(scale * 1.6))
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    org = (round(center[0] - tw / 2), round(center[1] + th / 2))
    cv2.putText(image, text, org, font, scale, color, thickness, cv2.LINE_AA)


def booth_wall_texture(name, tagline, main, accent) -> Texture:
    img = np.zeros((600, 1000, 3), np.uint8)
    img[:] = _bgr(main)
    light = sum(main) / 3 > 160
    ink = (30, 30, 30) if light else (255, 255, 255)
    cv2.fillPoly(img, [np.array([[620, 0], [1000, 0], [1000, 600], [380, 600]])], _bgr(accent))
    cv2.circle(img, (840, 170), 90, _bgr(main), -1, cv2.LINE_AA)
    _text(img, name, (330, 250), 560, 90, ink)
    _text(img, tagline, (330, 350), 480, 36, ink, font=cv2.FONT_HERSHEY_SIMPLEX)
    _text(img, f"Hall 3  -  {EVENT}", (330, 520), 380, 24, ink, font=cv2.FONT_HERSHEY_SIMPLEX)
    return Texture.from_array(img, name=f"{name} wall")


def logo_texture(name, main, accent) -> Texture:
    img = np.zeros((300, 560, 3), np.uint8)
    img[:] = _bgr(accent)
    cv2.rectangle(img, (14, 14), (546, 286), _bgr(main), -1)
    ink = (30, 30, 30) if sum(main) / 3 > 160 else (255, 255, 255)
    _text(img, name, (280, 150), 480, 90, ink)
    return Texture.from_array(img, name=f"{name} logo")


def slide_texture(name, title, main, accent, kind: int) -> Texture:
    img = np.zeros((360, 640, 3), np.uint8)
    img[:] = (28, 24, 22)
    cv2.rectangle(img, (0, 0), (640, 56), _bgr(main), -1)
    _text(img, name, (320, 28), 420, 26, (255, 255, 255))
    _text(img, title, (320, 100), 560, 34, (255, 255, 255), font=cv2.FONT_HERSHEY_SIMPLEX)
    rng = np.random.default_rng(zlib.crc32(f"{name}/{kind}".encode()))
    if kind % 2 == 0:  # bar chart
        for i, v in enumerate(rng.uniform(0.3, 1.0, 6)):
            x = 90 + i * 80
            cv2.rectangle(img, (x, 320 - int(170 * v)), (x + 50, 320), _bgr(accent), -1)
    else:  # bullet points
        for i in range(4):
            y = 160 + i * 42
            cv2.circle(img, (90, y), 7, _bgr(accent), -1, cv2.LINE_AA)
            cv2.rectangle(img, (115, y - 6), (115 + int(rng.uniform(200, 420)), y + 6), (200, 200, 200), -1)
    return Texture.from_array(img, name=f"{name}: {title}")


def banner_texture(text) -> Texture:
    img = np.zeros((260, 2000, 3), np.uint8)
    img[:] = (60, 30, 16)
    cv2.rectangle(img, (0, 220), (2000, 260), (255, 200, 80), -1)
    _text(img, f"WELCOME TO {text}", (1000, 110), 1700, 120, (255, 255, 255))
    return Texture.from_array(img, name="banner")


def _noise(size: int, sigma: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = cv2.GaussianBlur(rng.normal(0, 1, (size, size)).astype(np.float32), (0, 0), sigma)
    return n / (np.abs(n).max() + 1e-6)


def floor_texture() -> Texture:
    size = 256  # one 4 x 4 m texture repeat = 2 x 2 tiles
    img = np.full((size, size, 3), 196, np.float32) + _noise(size, 2.0, 1)[..., None] * 8
    img[:, :, 0] += 4  # slightly cool grey
    for k in (0, size // 2):
        img[k : k + 2, :] = img[:, k : k + 2] = 150
    return Texture.from_array(np.clip(img, 0, 255).astype(np.uint8), name="floor")


def carpet_texture(rgb) -> Texture:
    size = 128
    base = np.array(_bgr(rgb), np.float32)
    img = base[None, None, :] * (1 + 0.12 * _noise(size, 1.0, 2)[..., None])
    return Texture.from_array(np.clip(img, 0, 255).astype(np.uint8), name="carpet")


def wall_texture() -> Texture:
    size = 256  # one 6 x 6 m repeat = 3 panels
    img = np.full((size, size, 3), 226, np.float32) + _noise(size, 3.0, 3)[..., None] * 4
    for k in range(0, size, size // 3):
        img[:, k : k + 2] = 196
    img[size - 30 :, :] = 170  # skirting
    return Texture.from_array(np.clip(img, 0, 255).astype(np.uint8), name="wall")


def ceiling_texture() -> Texture:
    size = 256  # one 8 x 8 m repeat with a light panel
    img = np.full((size, size, 3), 58, np.uint8)
    cv2.rectangle(img, (40, 100), (216, 156), (236, 244, 250), -1)
    img[:, ::32] = 48
    return Texture.from_array(img, name="ceiling")


def window_texture() -> Texture:
    img = np.zeros((130, 350, 3), np.uint8)
    t = np.linspace(0, 1, 130)[:, None]
    sky = (np.array([250, 225, 190]) * (1 - t) + np.array([255, 245, 230]) * t)
    img[:] = sky[:, None, :].astype(np.uint8)
    img[:, ::70] = img[::65, :] = (90, 90, 95)  # window frames
    return Texture.from_array(img, name="window")


def foliage_texture() -> Texture:
    size = 128
    n = _noise(size, 1.5, 4)
    img = np.stack([40 + 25 * n, 120 + 50 * n, 50 + 25 * n], -1)
    return Texture.from_array(np.clip(img, 0, 255).astype(np.uint8), name="foliage")
