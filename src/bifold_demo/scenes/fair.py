"""Trade-fair hall: visitors standing in a half circle in front of the camera, people walking by in
the background, exhibitor booths and a keynote stage, all inside a large exhibition hall.

People come from the ``people/`` folder (fill it with ``bifold-people generate -n 30``); until it
has people, default people cut out of the Ultralytics sample photos stand in.

Everything is laid out by the constants below. Objects you can re-texture by dropping
``uploads/<name>.jpg``: ``booth1_wall`` ... ``booth8_wall``, ``booth1_sign``, ``booth1_counter``,
``booth1_screen``, ``stage_screen``, ``banner``, ``floor``, and the people ``visitor1`` ...,
``walker1`` ..., ``staff1`` ..., ``speaker``.
"""

from __future__ import annotations

import math
import random
import zlib

import cv2
import numpy as np

from bifold_demo.people import people_for_scene
from bifold_demo.scene import Box, Camera, Cylinder, Fog, Keyframes, Plane, Quad, Scene, Sphere, Texture, oscillate, spin
from bifold_demo.scene.transforms import rotation_y

# -- layout (metres; Y is up and the camera looks down the hall towards -Z) ----------------------
SEED = 7  # change for a different arrangement of people
CAMERA_POSITION = (0.0, 1.6, 6.0)  # the camera is fixed, like a mounted camera
CAMERA_TARGET = (0.0, 1.5, -12.0)
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
    """The hall; ``people`` are 2D cut-outs (default: ``PEOPLE_FOLDER`` or default people; ``[]`` = an empty hall)."""
    rng = random.Random(SEED)
    scene = Scene(
        background=(205, 207, 212),
        ambient=0.62,
        light_direction=(-0.25, -1.0, -0.35),
        light_color=(255, 250, 240),
        fog=Fog(start=18, end=95, color=(214, 216, 222), strength=0.55),
    )
    scene.camera = Camera(position=CAMERA_POSITION, target=CAMERA_TARGET, fov=55, near=0.1)
    add_hall(scene)
    add_stage(scene)
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
    """People walking by along the cross aisles, wrapping around at the hall walls."""
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

        def position(t, start=start, speed=speed, direction=direction, z=z, h=person.height, cadence=cadence):
            x = (start + direction * speed * t) % span - span / 2
            return (x, h / 2 + 0.025 * abs(math.sin(math.pi * cadence * t)), z)

        texture = person.facing_texture("right" if direction > 0 else "left")
        scene.add(Quad(height=person.height, texture=texture, position=position, face_camera=True, lit=False),
                  name=f"walker{i + 1}")


# -- booths -------------------------------------------------------------------------------------

BOOTH_WIDTH, BOOTH_DEPTH, BOOTH_WALL_HEIGHT = 5.0, 4.0, 3.0


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

    scene.add(Box(size=(w, 0.08, d), texture="#cfc8bd", position=at((0, 0.04, 0)), rotation=turn), name=f"{prefix}_floor")
    scene.add(Box(size=(w, h, 0.15), texture=booth_wall_texture(name, tagline, main, accent),
                  position=at((0, h / 2, -d / 2)), rotation=turn), name=f"{prefix}_wall")
    far_side = w / 2 if yaw > 0 else -w / 2  # side wall on the far side, so the booth opens towards the camera
    scene.add(Box(size=(0.15, h, d), texture=main, position=at((far_side, h / 2, 0)), rotation=turn), name=f"{prefix}_side")
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
    carpet = carpet_texture((150, 32, 40))
    scene.add(Plane(size=(5.0, length - 16), uv_repeat=(1, (length - 16) / 5), texture=carpet,
                    position=(0, 0.01, HALL_FRONT - (length - 16) / 2)), name="carpet_main")
    for i, z in enumerate(WALK_AISLES):
        scene.add(Plane(size=(w - 2, 3.2), uv_repeat=((w - 2) / 5, 1), texture=carpet_texture((40, 60, 120)),
                        position=(0, 0.02, z)), name=f"carpet_aisle{i + 1}")

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
    scene.add(Box(size=(16, 1.0, 7), texture="#26272c", position=(0, 0.5, z)), name="stage")
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
        scene.add(_person_quad(speaker, mirror, (2.5, 1.0 + speaker.height / 2, STAGE_Z + 1.0)), name="speaker")


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
