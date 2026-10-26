"""Things the 3D people wear and carry: a hat, glasses, a sign and a t-shirt, in textures of your choice.

:func:`dress` gives a :class:`~.character.Character` one of the :data:`ITEMS` (or takes it off again)::

    from bifold_demo.characters.accessories import dress

    dress(person, "hat", "#24406a")          # any texture value: a colour, an image file, an array, a Texture
    dress(person, "glasses", "sunglasses")   # or the name of a built-in look (see BUILT_IN)
    dress(person, "sign", "my_poster.png")   # held up on a stick in the right hand
    dress(person, "tshirt", "logo.png")      # worn over their clothes
    dress(person, "hat", None)               # off again

The items fit each avatar: the hat sits on the head over the hair, the glasses in front of the eyes, the t-shirt is
the avatar's own upper body (skinned like it, so it moves with every clip) lifted a few millimetres off it and cut to
short sleeves. The sign is held up in the right fist (an arm pose blended over the animation); it is put away while the
person does a yoga pose. Only avatars with the Biped skeleton of the Rocketbox people can wear items.

Texture layouts: the hat's texture wraps once around the crown (bottom rows at the brim, so a dark bottom stripe is a
hat band), and also covers the brim and the top. The glasses' texture is split in two: the left half colours the frame,
the right half the lenses (transparent pixels are clear glass). The sign shows the whole texture on both sides (the back
reads the right way round too). The t-shirt's texture wraps around the body: its middle is the centre of the chest, its
left and right edges meet in the middle of the back, its bottom row is the hem and its top row the neckline.

A :class:`Wardrobe` holds the looks each item comes in: textures you give it, image files in a folder (``hat.png``,
``hat_red.jpg``, ``items/sign/*.png``: a file belongs to an item when its name, or its subfolder, starts with the
item's name), then the built-in ones. Replacing a file changes what everybody wearing it looks like at once.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ..scene.geometry import Mesh, _merge, box_mesh, cylinder_mesh
from ..scene.scene import SceneObject
from ..scene.textures import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, Texture, as_texture
from .model import CharacterModel, SkinnedPrimitive, quat_from_matrix, quat_nlerp, quat_to_matrix

ITEMS = ("hat", "glasses", "sign", "tshirt")
TITLES = {"hat": "hat", "glasses": "glasses", "sign": "sign", "tshirt": "t-shirt"}
ALIASES = {"t-shirt": "tshirt", "t_shirt": "tshirt", "shirt": "tshirt", "cap": "hat", "board": "sign"}
FADE = 0.5  # seconds the arm takes to lift a sign or to let it go
SHIRT_LIFT = 0.006  # metres the t-shirt stands off the clothes under it
SLEEVE = 0.42  # the t-shirt's sleeves cover this share of the upper arm


def item_kind(name: str) -> str:
    """``"tshirt"`` for ``"T-Shirt"`` (case and the aliases in :data:`ALIASES` do not matter)."""
    key = str(name).strip().lower()
    key = ALIASES.get(key, key)
    if key not in ITEMS:
        raise KeyError(f"unknown item {name!r}; known: {', '.join(ITEMS)}")
    return key


def supports(model: CharacterModel) -> bool:
    """Whether ``model`` can wear items (the Biped skeleton of the Rocketbox people)."""
    from . import yoga

    return yoga.supports(model) and model.skeleton.find("Bip01 Head") is not None


# -- built-in looks ------------------------------------------------------------------------------------------


def _fabric(size, rgb, seed: int, grain: float = 10.0) -> np.ndarray:
    """An RGB image of ``rgb`` with a little woven noise."""
    rng = np.random.default_rng(seed)
    h, w = size
    noise = rng.normal(0.0, grain, (h, w, 1))
    weave = 4.0 * (np.sin(np.arange(w) * 1.7)[None, :, None] + np.sin(np.arange(h) * 1.7)[:, None, None])
    return np.clip(np.asarray(rgb, float) + noise + weave, 0, 255).astype(np.uint8)


def _hat(rgb, band, seed) -> np.ndarray:
    img = _fabric((128, 256), rgb, seed)
    img[-20:-6] = np.clip(np.asarray(band, float) + np.random.default_rng(seed + 1).normal(0, 6, (14, 256, 1)),
                          0, 255).astype(np.uint8)
    return img


def _straw(seed) -> np.ndarray:
    img = _fabric((128, 256), (214, 186, 128), seed, grain=14.0)
    stripes = (np.sin(np.arange(128)[:, None] * 0.9) * np.sin(np.arange(256)[None, :] * 0.35)) > 0.3
    img[stripes] = np.clip(img[stripes].astype(int) - 28, 0, 255).astype(np.uint8)
    img[-20:-6] = (120, 60, 40)
    return img


def _glasses(frame, lens, alpha_lens: int, pattern: bool = False) -> np.ndarray:
    img = np.zeros((32, 64, 4), np.uint8)
    img[:, :32, :3] = frame
    if pattern:  # tortoiseshell
        rng = np.random.default_rng(3)
        spots = cv2.GaussianBlur(rng.random((32, 32)).astype(np.float32), (0, 0), 2.0)
        spots = (spots - spots.min()) / max(float(np.ptp(spots)), 1e-6)
        img[:, :32, :3] = (np.asarray((60, 32, 14)) + spots[..., None] * np.asarray((110, 70, 30))).astype(np.uint8)
    img[:, :32, 3] = 255
    img[:, 32:, :3] = lens
    img[:, 32:, 3] = alpha_lens
    return img


def _sign(text: str, ink, board, accent) -> np.ndarray:
    img = np.zeros((420, 600, 3), np.uint8)
    img[:] = accent
    cv2.rectangle(img, (14, 14), (586, 406), board, -1)
    font, scale, thick = cv2.FONT_HERSHEY_DUPLEX, 3.0, 7
    lines = text.split("\n")
    for k, line in enumerate(lines):
        (tw, th), _ = cv2.getTextSize(line, font, scale, thick)
        s = min(scale, scale * 520 / max(tw, 1))
        (tw, th), _ = cv2.getTextSize(line, font, s, thick)
        y = int(210 + (k - (len(lines) - 1) / 2) * 1.5 * th + th / 2)
        cv2.putText(img, line, ((600 - tw) // 2, y), font, s, ink, thick, cv2.LINE_AA)
    return img  # RGB values written straight in: the colours given here are RGB


def _tshirt(rgb, seed, emblem=None) -> np.ndarray:
    img = _fabric((256, 512), rgb, seed, grain=6.0)
    img[-6:] = np.clip(img[-6:].astype(int) - 25, 0, 255)  # the neckline seam
    img[:5] = np.clip(img[:5].astype(int) - 20, 0, 255)  # the hem
    if emblem is not None:
        colour, text = emblem
        cv2.circle(img, (256, 118), 62, colour, 9, cv2.LINE_AA)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1.4, 3)
        cv2.putText(img, text, (256 - tw // 2, 118 + th // 2), cv2.FONT_HERSHEY_DUPLEX, 1.4, colour, 3, cv2.LINE_AA)
    return img


def _built_in() -> dict[str, dict[str, Texture]]:
    def rgb(image, name):
        return Texture.from_array(image, order="rgb", name=name)

    return {
        "hat": {"navy": rgb(_hat((36, 52, 92), (20, 22, 30), 1), "navy hat"),
                "straw": rgb(_straw(2), "straw hat"),
                "red": rgb(_hat((176, 32, 36), (240, 240, 240), 3), "red hat"),
                "grey": rgb(_hat((120, 122, 126), (30, 30, 34), 4), "grey hat")},
        "glasses": {"black": rgb(_glasses((18, 18, 20), (0, 0, 0), 0), "black glasses"),
                    "tortoise": rgb(_glasses((90, 50, 20), (0, 0, 0), 0, pattern=True), "tortoiseshell glasses"),
                    "sunglasses": rgb(_glasses((20, 20, 22), (24, 26, 30), 255), "sunglasses"),
                    "gold": rgb(_glasses((196, 160, 72), (0, 0, 0), 0), "gold glasses")},
        "sign": {"welcome": rgb(_sign("WELCOME", (30, 30, 30), (250, 250, 246), (36, 90, 170)), "welcome sign"),
                 "ask me": rgb(_sign("ASK ME!", (255, 255, 255), (220, 60, 40), (250, 250, 246)), "ask me sign"),
                 "yoga": rgb(_sign("YOGA\nTODAY", (20, 80, 40), (236, 246, 230), (40, 140, 80)), "yoga sign")},
        "tshirt": {"white": rgb(_tshirt((236, 236, 232), 5), "white t-shirt"),
                   "black": rgb(_tshirt((28, 28, 32), 6), "black t-shirt"),
                   "red": rgb(_tshirt((190, 34, 40), 7), "red t-shirt"),
                   "logo": rgb(_tshirt((246, 246, 242), 8, emblem=((40, 120, 200), "YOGA")), "logo t-shirt")},
    }


_BUILT_IN: dict | None = None


def built_in() -> dict[str, dict[str, Texture]]:
    """``{item: {look: Texture}}``: the looks every item comes in."""
    global _BUILT_IN
    if _BUILT_IN is None:
        _BUILT_IN = _built_in()
    return _BUILT_IN


class Wardrobe:
    """The looks each item comes in (see the module docs): ``textures`` (``{item: [texture value, ...]}``) first,
    then the image or video files for it in ``folder`` (checked again every second), then the built-in looks
    (unless ``built_in=False``)."""

    def __init__(self, folder=None, textures: dict | None = None, built_in: bool = True):
        self.folder = Path(folder).expanduser() if folder else None
        self.textures = {item_kind(k): list(v) if isinstance(v, (list, tuple)) else [v]
                         for k, v in (textures or {}).items()}
        self.use_built_in = built_in
        self._files: dict[str, list[Path]] = {}
        self._checked = -math.inf

    def looks(self, item: str, now: float | None = None) -> list[tuple[str, object]]:
        """``[(name, texture value)]`` for ``item``, in the order a key steps through them."""
        item = item_kind(item)
        given = [(_describe(spec), spec) for spec in self.textures.get(item, [])]
        files = [(path.stem, path) for path in self._folder_files(now).get(item, [])]
        ours = list(built_in()[item].items()) if self.use_built_in else []
        return given + files + ours

    def look(self, item: str, name_or_spec):
        """A texture value: a built-in look's name, a file of the folder by its name, or any texture value."""
        if isinstance(name_or_spec, str) and not name_or_spec.startswith("#"):
            for name, spec in self.looks(item):
                if name.lower() == name_or_spec.lower():
                    return spec
        return name_or_spec

    def _folder_files(self, now: float | None) -> dict[str, list[Path]]:
        import time

        now = time.monotonic() if now is None else now
        if self.folder is None or now - self._checked < 1.0:
            return self._files
        self._checked = now
        found: dict[str, list[Path]] = {}
        try:
            paths = sorted(p for p in self.folder.rglob("*") if p.is_file())
        except OSError:
            paths = []
        for path in paths:
            if path.name.startswith(".") or path.suffix.lower() not in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS:
                continue
            names = [part.lower() for part in path.relative_to(self.folder).parts]
            for item in ITEMS:
                keys = [item] + [a for a, k in ALIASES.items() if k == item]
                if any(_named(n, k) for n in names for k in keys):
                    found.setdefault(item, []).append(path)
                    break
        self._files = found
        return found


def _named(part: str, key: str) -> bool:
    """Whether a file or folder name is about ``key``: ``hat``, ``hat.png``, ``hat_red.jpg``, ``hat-2.png``."""
    stem = Path(part).stem
    return stem == key or (stem.startswith(key) and stem[len(key)] in "_- 0123456789")


def _describe(spec) -> str:
    if isinstance(spec, (str, Path)):
        return Path(spec).stem if not str(spec).startswith("#") else str(spec)
    return getattr(spec, "name", None) or type(spec).__name__


# -- fitting the items to an avatar ----------------------------------------------------------------------


@dataclass
class Fit:
    """Where the items go on one avatar: the rest pose, the bones they hang from and their meshes."""

    rest: np.ndarray  # (B, 4, 4) rest pose as the Poser builds it
    head: int
    hand: int
    hat: Mesh
    hat_offset: np.ndarray  # head bone space <- hat space
    glasses: Mesh
    glasses_offset: np.ndarray
    board: Mesh
    stick: Mesh
    board_offset: np.ndarray  # hand bone space <- board space (in the holding pose)
    stick_offset: np.ndarray
    grip: dict = field(default_factory=dict)  # bone -> local rotation (3x3) holding the sign
    shirt: SkinnedPrimitive | None = None


_FITS: dict[int, tuple[CharacterModel, Fit]] = {}


def fit(model: CharacterModel, poser=None) -> Fit:
    """How the items fit ``model`` (worked out once per model)."""
    known = _FITS.get(id(model))
    if known is not None and known[0] is model:
        return known[1]
    from .character import _FastPoser

    poser = poser or _FastPoser(model)
    rest = poser.forward_kinematics(poser.local_pose([]))
    sk = model.skeleton
    head = sk.index("Bip01 Head")
    points, owners = _rest_vertices(model)
    head_bones = {int(b) for b in sk.descendants(head)} | {head}
    on_head = np.isin(owners, list(head_bones))
    hat_mesh, hat_place = _fit_hat(model, rest, points[on_head])
    glasses_mesh, glasses_place = _fit_glasses(model, rest, points[on_head])
    grip, hand, board, stick, board_place, stick_place, held_hand = _fit_sign(model, poser)
    to_hand = np.linalg.inv(held_hand)
    result = Fit(rest, head, hand, hat_mesh, np.linalg.inv(rest[head]) @ hat_place, glasses_mesh,
                 np.linalg.inv(rest[head]) @ glasses_place, board, stick, to_hand @ board_place,
                 to_hand @ stick_place, grip, _fit_shirt(model, rest))
    _FITS[id(model)] = (model, result)
    return result


def _rest_vertices(model):
    points = np.concatenate([p.positions for p in model.primitives]).astype(np.float64)
    joints = np.concatenate([p.joints for p in model.primitives]).astype(np.int64)
    weights = np.concatenate([p.weights for p in model.primitives])
    return points, joints[np.arange(len(joints)), weights.argmax(axis=1)]


def _translation(x, y, z) -> np.ndarray:
    m = np.eye(4)
    m[:3, 3] = (x, y, z)
    return m


def _eyes(model, rest, head_points):
    """The left and right eyeballs' centres (the eye bones; else halfway up the head, near its front)."""
    sk = model.skeleton
    eyes = [sk.find(n) for n in ("Bip01 LEye", "Bip01 REye")]
    if None not in eyes:
        return rest[eyes[0], :3, 3], rest[eyes[1], :3, 3]
    head = rest[sk.index("Bip01 Head"), :3, 3]
    top = float(head_points[:, 1].max()) if len(head_points) else head[1] + 0.2
    front = float(head_points[:, 2].max()) - 0.02 if len(head_points) else head[2] + 0.07
    y = head[1] + 0.45 * (top - head[1])
    return np.array([head[0] + 0.032, y, front]), np.array([head[0] - 0.032, y, front])


def _fit_hat(model, rest, head_points):
    left, right = _eyes(model, rest, head_points)
    eye_y = 0.5 * (left[1] + right[1])
    top = float(np.percentile(head_points[:, 1], 99.7))
    brow = eye_y + 0.27 * (top - eye_y)  # about 3.5 cm over the eyes
    band = head_points[(head_points[:, 1] > brow) & (head_points[:, 1] < top)]
    cx = 0.5 * (left[0] + right[0])
    centre = np.array([cx, 0.0, float(np.median(band[:, 2]))]) if len(band) else np.array([cx, 0.0, left[2] - 0.08])
    dx, dz = np.abs(band[:, 0] - centre[0]), np.abs(band[:, 2] - centre[2])
    near = np.hypot(dx, dz) < 0.17  # leave out a ponytail or long hair; the ears, a scarf or a bun go in
    rx = float(dx[near].max()) + 0.008 if near.any() else 0.095
    rz = float(dz[near].max()) + 0.008 if near.any() else 0.11
    crown = top - brow + 0.02
    mesh = _hat_mesh(rx, rz, crown, brim=0.055)
    return mesh, _translation(centre[0], brow, centre[2])


def _hat_mesh(rx, rz, height, brim, segments=40) -> Mesh:
    """A crown (a little narrower towards its domed top) on a brim, the base of the crown at y = 0 and its
    ``rx`` / ``rz`` the half widths there; the texture wraps around the crown (bottom rows at the brim)."""
    lon = np.linspace(0, 2 * math.pi, segments + 1)
    sx, sz, u = np.sin(lon), np.cos(lon), lon / (2 * math.pi)
    rows = [(0.0, 1.0), (0.7, 0.97), (0.88, 0.9), (0.97, 0.7), (1.0, 0.4)]
    crown = np.vstack([np.stack([sx * rx * k, np.full_like(sx, f * height), sz * rz * k], 1) for f, k in rows])
    crown_uv = np.vstack([np.stack([u, np.full_like(u, 0.1 + 0.6 * f)], 1) for f, _k in rows])
    j = np.arange(1, segments + 1)
    top = np.vstack([[0.0, height, 0.0], crown[-(segments + 1):]])
    top_uv = np.vstack([[0.5, 0.85], np.stack([0.5 + 0.12 * sx, 0.85 + 0.1 * sz], 1)])
    brim_in = np.stack([sx * rx, np.zeros_like(sx), sz * rz], 1)
    brim_out = np.stack([sx * (rx + brim), np.zeros_like(sx), sz * (rz + brim)], 1)
    thick = np.array([0.0, 0.003, 0.0])
    ring = np.vstack([brim_in, brim_out])
    ring_uv = np.vstack([np.stack([u, np.full_like(u, 0.07)], 1), np.stack([u, np.full_like(u, 0.0)], 1)])
    flat = _rows_indices(2, segments)
    parts = [(crown, None, crown_uv, _rows_indices(len(rows), segments)),
             (top, None, top_uv, np.stack([np.zeros_like(j), j, j + 1], 1)),
             (ring + thick, None, ring_uv, flat[:, ::-1]),  # the brim's top (its rings go outwards)
             (ring - thick, None, ring_uv, flat),  # and underside
             (np.vstack([brim_out + thick, brim_out - thick]), None, np.vstack([ring_uv[segments + 1:]] * 2),
              _rows_indices(2, segments)[:, ::-1])]  # its edge
    mesh = _merge([(pos, np.zeros_like(pos), uv, tris) for pos, _n, uv, tris in parts])
    return Mesh(mesh.positions, mesh.indices, uvs=mesh.uvs)  # normals from the faces


def _rows_indices(rows: int, segments: int) -> np.ndarray:
    """Quads between consecutive rings of ``segments + 1`` vertices, facing outwards for rings going up."""
    tris = []
    for r in range(rows - 1):
        a = r * (segments + 1) + np.arange(segments)
        b = a + segments + 1
        tris.append(np.stack([a, a + 1, b + 1], 1))
        tris.append(np.stack([a, b + 1, b], 1))
    return np.vstack(tris)


def _fit_glasses(model, rest, head_points):
    left, right = _eyes(model, rest, head_points)
    ipd = float(np.linalg.norm(left - right))
    mid = 0.5 * (left + right)
    width, height = 0.82 * ipd, 0.5 * ipd
    region = head_points[(np.abs(head_points[:, 1] - mid[1]) < 0.018) & (np.abs(head_points[:, 0] - mid[0]) > 0.012)
                         & (np.abs(head_points[:, 0] - mid[0]) < 0.5 * ipd + 0.02)]
    front = float(region[:, 2].max()) + 0.008 if len(region) else float(mid[2]) + 0.028
    sides = head_points[(np.abs(head_points[:, 1] - mid[1]) < 0.02) & (head_points[:, 2] < front)
                        & (head_points[:, 2] > front - 0.1)]
    half = float(np.abs(sides[:, 0] - mid[0]).max()) + 0.004 if len(sides) else 0.5 * ipd + 0.035
    ear_z = front - 0.095
    mesh = _glasses_mesh(ipd, width, height, max(half, 0.5 * ipd + 0.5 * width + 0.006), ear_z - front)
    return mesh, _translation(mid[0], mid[1] + 0.003, front)


def _outline(w, h, n=24) -> np.ndarray:
    """A rounded rectangle (superellipse) around the origin in the xy plane."""
    a = np.linspace(0, 2 * math.pi, n, endpoint=False)
    c, s = np.cos(a), np.sin(a)
    return np.stack([0.5 * w * np.sign(c) * np.abs(c) ** 0.45, 0.5 * h * np.sign(s) * np.abs(s) ** 0.6], 1)


def _glasses_mesh(ipd, width, height, half_head, back, rim=0.0035, depth=0.003) -> Mesh:
    """Two rims with lenses, a bridge and two temples; the lens plane at z = 0, the temples going back to ``back``.
    Frame parts take their colour from the left half of the texture, the lenses from the right half."""
    parts = []
    for side in (1.0, -1.0):
        cx = side * 0.5 * ipd
        inner = _outline(width, height)
        outer = _outline(width + 2 * rim, height + 2 * rim)
        n = len(inner)
        idx = np.arange(n)
        nxt = (idx + 1) % n
        for z, normal in ((0.0, 1.0), (-depth, -1.0)):  # front and back of the rim
            pos = np.vstack([np.c_[inner[:, 0] + cx, inner[:, 1], np.full(n, z)],
                             np.c_[outer[:, 0] + cx, outer[:, 1], np.full(n, z)]])
            tris = np.vstack([np.stack([idx, n + idx, n + nxt], 1), np.stack([idx, n + nxt, nxt], 1)])
            if normal < 0:
                tris = tris[:, ::-1]
            parts.append((pos, np.tile([0.0, 0.0, normal], (2 * n, 1)), np.full((2 * n, 2), 0.25), tris))
        for ring, outward in ((outer, 1.0), (inner, -1.0)):  # the rim's outer and inner edges
            pos = np.vstack([np.c_[ring[:, 0] + cx, ring[:, 1], np.zeros(n)],
                             np.c_[ring[:, 0] + cx, ring[:, 1], np.full(n, -depth)]])
            normals = np.c_[ring / np.linalg.norm(ring, axis=1, keepdims=True) * outward, np.zeros(n)]
            tris = np.vstack([np.stack([idx, n + idx, n + nxt], 1), np.stack([idx, n + nxt, nxt], 1)])
            if outward < 0:
                tris = tris[:, ::-1]
            parts.append((pos, np.vstack([normals, normals]), np.full((2 * n, 2), 0.25), tris))
        lens = np.c_[inner[:, 0] * 1.01 + cx, inner[:, 1] * 1.01, np.full(n, -0.5 * depth)]
        centre = np.array([[cx, 0.0, -0.5 * depth]])
        lens_uv = np.c_[0.75 + 0.2 * inner[:, 0] / width, 0.5 + 0.4 * inner[:, 1] / height]
        fan = np.stack([np.zeros(n, int), idx + 1, nxt + 1], 1)
        for normal in (1.0, -1.0):
            parts.append((np.vstack([centre, lens]), np.tile([0.0, 0.0, normal], (n + 1, 1)),
                          np.vstack([[0.75, 0.5], lens_uv]), fan if normal > 0 else fan[:, ::-1]))
        hinge = np.array([side * (0.5 * ipd + 0.5 * width + rim * 0.5), 0.25 * height, -0.5 * depth])
        ear = np.array([side * half_head, 0.15 * height, back])
        parts.append(_bar(hinge, ear, 0.0025, 0.004))
    gap = ipd - width - 2 * rim
    parts.append(_bar(np.array([-0.5 * gap - rim, 0.18 * height, -0.5 * depth]),
                      np.array([0.5 * gap + rim, 0.18 * height, -0.5 * depth]), 0.003, 0.0025))
    return _merge(parts)


def _bar(a, b, thick, tall):
    """A thin box from ``a`` to ``b`` (frame colour)."""
    length = float(np.linalg.norm(b - a))
    box = box_mesh(thick, tall, length)
    direction = (b - a) / max(length, 1e-9)
    up = np.array([0.0, 1.0, 0.0])
    x = np.cross(up, direction)
    x = x / max(np.linalg.norm(x), 1e-9)
    y = np.cross(direction, x)
    rot = np.stack([x, y, direction], 1)
    pos = box.positions.astype(np.float64) @ rot.T + 0.5 * (a + b)
    return pos, box.normals.astype(np.float64) @ rot.T, np.full((len(pos), 2), 0.25), box.indices


def _fit_sign(model, poser):
    """The arm pose holding a sign up in the right fist, and where the sign is then: ``(grip, hand bone, board mesh,
    stick mesh, board placement, stick placement, the hand's transform in the holding pose)``."""
    from . import yoga

    body = yoga.Body(model)
    right = body.sides["right"]
    shoulder = body.rest_pos[right["upper"]]
    arm = body.arm
    wrist = shoulder + arm * np.array([-0.3, -0.3, 0.55])  # out to the side in front, the elbow bent and down
    pose = yoga.mountain(body)
    held = yoga.Limb(tuple(wrist), (-0.3, -1.0, -0.4), (0.0, 0.2, 1.0), (-1.0, 0.0, 0.0))
    from dataclasses import replace

    local, glob = yoga.solve(body, replace(pose, right_arm=held))
    grip = {b: local[b, :3, :3].copy() for b in (right["upper"], right["fore"], right["hand"])}
    _curl_fingers(body, right, glob, grip)
    held_local = poser.local_pose([])
    for bone, rotation in grip.items():
        held_local[bone, :3, :3] = rotation
    pose_global = poser.forward_kinematics(held_local)
    hand = right["hand"]
    h = pose_global[hand]
    knuckles = pose_global[right["fingers"][2][0], :3, 3] if right["fingers"][2] else h[:3, 3]
    inward = -(h[:3, :3] @ body.rest_rot[hand].T @ right["hand_back"])  # towards the palm
    fist = 0.5 * (h[:3, 3] + knuckles) + 0.018 * inward
    scale = body.height / 1.75
    board_w, board_h = 0.6 * scale ** 0.5, 0.42 * scale ** 0.5
    eyes = 0.5 * (body.rest_pos[body.skeleton.index("Bip01 LEye"), 1] + body.rest_pos[body.skeleton.index("Bip01 REye"), 1]) \
        if body.skeleton.find("Bip01 LEye") is not None else body.height * 0.93
    bottom = max(eyes - 0.02, fist[1] + 0.2)  # the board's lower edge about at eye level, beside the head
    stick_below, stick_above = 0.2 * scale, bottom - fist[1] + 0.75 * board_h
    board = _board_mesh(board_w, board_h, 0.012)
    board_place = _translation(fist[0] - 0.12 * board_w, bottom + 0.5 * board_h, fist[2] + 0.014)
    stick = cylinder_mesh(0.012, stick_below + stick_above, segments=12)
    stick.positions[:, 1] += 0.5 * (stick_above - stick_below)
    return grip, hand, board, stick, board_place, _translation(*fist), h


def _curl_fingers(body, side, glob, grip) -> None:
    """Close the fingers of the solved hand ``glob`` around a stick, adding their local rotations to ``grip``."""
    from . import yoga

    sk = body.skeleton
    hand = side["hand"]
    G = {b: glob[b, :3, :3].copy() for b in range(len(sk.names))}
    hand_rot = G[hand]
    forward = hand_rot @ body.rest_rot[hand].T @ side["hand_forward"]
    back = hand_rot @ body.rest_rot[hand].T @ side["hand_back"]
    across = np.cross(back, forward)  # about which the fingers curl towards the palm
    curls = (70.0, 85.0, 60.0)
    for k, chain in enumerate(side["fingers"]):
        for j, bone in enumerate(chain):
            parent = sk.parents[bone]
            G[bone] = G[parent] @ body.local_rot[bone]
            angle = curls[j] * (0.45 if k == 0 else 1.0)
            axis = forward if k == 0 else across
            G[bone] = yoga.axis_rotation(axis, angle * (1.0 if k else -side["sign"])) @ G[bone]
            grip[bone] = G[parent].T @ G[bone]


def _board_mesh(w, h, d) -> Mesh:
    """A sign board facing +Z: the texture on the front and, the right way round, on the back; plain edges."""
    x, y, z = 0.5 * w, 0.5 * h, 0.5 * d
    front = (np.array([[-x, -y, z], [x, -y, z], [x, y, z], [-x, y, z]]), np.tile([0.0, 0.0, 1.0], (4, 1)),
             np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float), np.array([[0, 1, 2], [0, 2, 3]]))
    back = (np.array([[x, -y, -z], [-x, -y, -z], [-x, y, -z], [x, y, -z]]), np.tile([0.0, 0.0, -1.0], (4, 1)),
            np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float), np.array([[0, 1, 2], [0, 2, 3]]))
    edges = box_mesh(w, h, d)
    keep = np.abs(edges.normals[:, 2]) < 0.5
    mapping = np.cumsum(keep) - 1
    tris = edges.indices[keep[edges.indices].all(axis=1)]
    edge_part = (edges.positions[keep].astype(np.float64), edges.normals[keep].astype(np.float64),
                 np.full((int(keep.sum()), 2), 0.01), mapping[tris])
    return _merge([front, back, edge_part])


# -- the t-shirt ---------------------------------------------------------------------------------------------


_TORSO = ("Bip01 Pelvis", "Bip01 Spine", "Bip01 Spine1", "Bip01 Spine2", "Bip01 Neck", "Bip01 L Clavicle",
          "Bip01 R Clavicle", "Bip01 L UpperArm", "Bip01 R UpperArm", "Bip01")


def _fit_shirt(model, rest) -> SkinnedPrimitive | None:
    """The avatar's own upper body as a t-shirt: hips to neckline, short sleeves, lifted off the clothes (from all
    its mesh parts but the head, hair and eyes)."""
    sk = model.skeleton
    torso = {sk.find(n) for n in _TORSO} - {None}
    neck, pelvis = sk.index("Bip01 Neck"), sk.index("Bip01 Pelvis")
    thighs = [sk.index("Bip01 L Thigh"), sk.index("Bip01 R Thigh")]
    uppers = {sk.index("Bip01 L UpperArm"): sk.index("Bip01 L Forearm"),
              sk.index("Bip01 R UpperArm"): sk.index("Bip01 R Forearm")}
    hip = float(np.mean([rest[b, 1, 3] for b in thighs]))
    neck_y = float(rest[neck, 1, 3])
    hem = hip + 0.3 * (float(rest[sk.index("Bip01 Spine1"), 1, 3]) - hip)
    collar = neck_y - 0.02
    axis_x = float(rest[pelvis, 0, 3])
    axis_z = float(0.5 * (rest[pelvis, 2, 3] + rest[sk.index("Bip01 Spine2"), 2, 3]))
    parts = [p for p in model.primitives if not any(k in p.name.lower() for k in ("head", "opacity", "hair", "eye"))]
    parts = parts or list(model.primitives)
    offsets = np.cumsum([0] + [len(p.positions) for p in parts[:-1]])
    pos = np.concatenate([p.positions for p in parts]).astype(np.float64)
    nrm = np.concatenate([p.normals for p in parts]).astype(np.float64)
    weights = np.concatenate([p.weights for p in parts]).astype(np.float64)
    joints = np.concatenate([p.joints for p in parts]).astype(np.int64)
    indices = np.concatenate([p.indices.astype(np.int64) + o for p, o in zip(parts, offsets)])
    torso_share = np.zeros(len(pos))
    for k in range(joints.shape[1]):
        torso_share += np.where(np.isin(joints[:, k], list(torso)), weights[:, k], 0.0)
    main = joints[np.arange(len(joints)), weights.argmax(axis=1)]
    sleeves = []  # (along the upper arm, 0 at the shoulder) for the vertices it moves most
    for upper, fore in uppers.items():
        a, b = rest[upper, :3, 3], rest[fore, :3, 3]
        along = (pos - a) @ (b - a) / max(float(np.dot(b - a, b - a)), 1e-9)
        sleeves.append((main == upper, along, b - a))
    tris = indices[(torso_share[indices] > 0.25).all(axis=1)]  # not the head or hands; the cuts do the rest
    centre = pos[tris].mean(axis=1)
    inside = (centre[:, 1] > hem) & (centre[:, 1] < collar)
    for on_arm, along, _axis in sleeves:  # short sleeves: the first 40 % of the upper arm
        inside &= ~(on_arm[tris].any(axis=1) & (along[tris].mean(axis=1) > SLEEVE))
    tris = tris[inside]
    if len(tris) < 50:
        return None
    used = np.unique(tris)
    remap = np.full(len(pos), -1)
    remap[used] = np.arange(len(used))
    tris = remap[tris]
    p = pos[used] + SHIRT_LIFT * nrm[used]
    edges = np.sort(np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]), axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    border = np.zeros(len(used), bool)
    border[unique[counts == 1].ravel()] = True  # the open edges: hem, neckline, sleeve ends (and any holes)
    p[:, 1] = np.where(border & (p[:, 1] < hem + 0.06), hem, np.clip(p[:, 1], hem, collar))  # a straight hem
    p[:, 1] = np.where(border & (p[:, 1] > collar - 0.05), collar, p[:, 1])  # and neckline
    for on_arm, along, axis in sleeves:  # and straight sleeve ends
        a, d = on_arm[used], along[used]
        end = a & ((d > SLEEVE) | (border & (d > SLEEVE - 0.2)))
        p[end] -= (d[end] - SLEEVE)[:, None] * axis
    normals = nrm[used]
    angle = np.arctan2(p[:, 0] - axis_x, p[:, 2] - axis_z)  # 0 at the front
    u = 0.5 + angle / (2 * math.pi)
    v = np.clip((p[:, 1] - hem) / max(collar - hem, 1e-6), 0.0, 1.0)
    # triangles across the seam in the back (u from about 1 to 0) get copies of their low-u vertices at u + 1,
    # so they show the edge of the texture (it repeats) instead of all of it squeezed
    seam = np.ptp(u[tris], axis=1) > 0.5
    extra: dict[int, int] = {}
    order = len(p)
    for t in np.nonzero(seam)[0]:
        for k in range(3):
            vertex = tris[t, k]
            if u[vertex] < 0.5:
                if vertex not in extra:
                    extra[vertex] = order
                    order += 1
                tris[t, k] = extra[vertex]
    copies = np.array(list(extra), int)
    p = np.vstack([p, p[copies]]) if len(copies) else p
    normals = np.vstack([normals, normals[copies]]) if len(copies) else normals
    u = np.concatenate([u, u[copies] + 1.0])
    v = np.concatenate([v, v[copies]])
    joints_used = np.concatenate([joints[used], joints[used][copies]]) if len(copies) else joints[used]
    weights_used = np.concatenate([weights[used], weights[used][copies]]) if len(copies) else weights[used]
    return SkinnedPrimitive("tshirt", p.astype(np.float32), normals.astype(np.float32),
                            np.c_[u, v].astype(np.float32), tris.astype(np.uint32), joints_used,
                            weights_used.astype(np.float32), None)


# -- wearing them ---------------------------------------------------------------------------------------------


class Garment(SkinnedPrimitive):
    """A skinned item (the t-shirt) drawn with its wearer's bones; ``look`` is its texture value, read again every
    frame so a replaced file shows at once."""

    look: object = None

    def refresh(self, t: float) -> None:
        try:
            self.texture = as_texture(self.look)
        except Exception:  # noqa: BLE001 - a file being copied: keep the last good one
            pass


class Prop(SceneObject):
    """A rigid item piece that moves with one bone of ``owner`` (``offset``: bone space <- piece space); drawn by
    the renderer with its owner, not as a scene object of its own."""

    def __init__(self, owner, bone: int, offset: np.ndarray, mesh: Mesh, texture, name: str):
        super().__init__(mesh, name=name, texture=texture)
        self.owner, self.bone, self.offset = owner, bone, np.asarray(offset, float)
        self.hidden = False

    def update(self, t: float) -> None:
        super().update(t)
        self.current_visible = bool(self.owner.current_visible) and not self.hidden

    def matrices(self, camera_position=None) -> tuple[np.ndarray, np.ndarray]:
        model = self.owner.matrices(camera_position)[0] @ self.owner.pose[self.bone] @ self.offset
        linear = model[:3, :3]
        normal = np.linalg.inv(linear).T if abs(np.linalg.det(linear)) > 1e-12 else np.eye(3)
        return model, normal


class Worn:
    """An item on a person: its rigid :class:`Prop` pieces or its skinned primitive, and its look."""

    def __init__(self, kind: str, look, props=(), primitive: SkinnedPrimitive | None = None, since: float = 0.0):
        self.kind, self.look, self.props, self.primitive = kind, look, list(props), primitive
        self.since = since

    @property
    def name(self) -> str:
        return _describe(self.look)


def dress(character, item: str, look=True, wardrobe: Wardrobe | None = None, t: float | None = None) -> Worn | None:
    """Put ``item`` on ``character`` in ``look`` (a texture value, the name of a look of ``wardrobe``, ``True`` for
    the first look), replacing the one they wear; ``None`` or ``False`` takes it off. Returns what they now wear.
    ``t``: the scene time it happens (a sign is lifted from then on)."""
    kind = item_kind(item)
    if not supports(character.skinned_model):
        raise ValueError(f"{character.skinned_model.name} cannot wear items (not a Biped skeleton)")
    wardrobe = wardrobe or Wardrobe()
    items = _items(character)
    old = items.pop(kind, None)
    if old is not None and kind == "sign":
        _grip(character).release(character, t)
    if look is None or look is False:
        _refresh(character)
        return None
    if look is True:
        look = wardrobe.looks(kind)[0][1]
    else:
        look = wardrobe.look(kind, look)
    as_texture(look)  # fail now on a missing file
    f = fit(character.skinned_model, character.poser)
    name = f"{character.name}.{kind}"
    if kind == "hat":
        worn = Worn(kind, look, [Prop(character, f.head, f.hat_offset, f.hat, look, name)])
    elif kind == "glasses":
        worn = Worn(kind, look, [Prop(character, f.head, f.glasses_offset, f.glasses, look, name)])
    elif kind == "sign":
        worn = Worn(kind, look, [Prop(character, f.hand, f.board_offset, f.board, look, name),
                                 Prop(character, f.hand, f.stick_offset, f.stick, (150, 108, 64), name + "_stick")])
        _grip(character).hold(character, f.grip, t)
    else:
        if f.shirt is None:
            raise ValueError(f"no t-shirt fits {character.skinned_model.name}")
        shirt = Garment(f.shirt.name, f.shirt.positions, f.shirt.normals, f.shirt.uvs, f.shirt.indices,
                        f.shirt.joints, f.shirt.weights, as_texture(look))
        shirt.look = look
        worn = Worn(kind, look, primitive=shirt)
    items[kind] = worn
    _refresh(character)
    return worn


def wearing(character) -> dict[str, Worn]:
    """``{item: Worn}`` of what ``character`` wears."""
    return dict(getattr(character, "items", {}) or {})


def _items(character) -> dict[str, Worn]:
    if getattr(character, "items", None) is None:
        character.items = {}
    return character.items


def _refresh(character) -> None:
    """Tell the character what to draw with it."""
    items = _items(character)
    character.props = [p for worn in items.values() for p in worn.props]
    character.worn = [worn.primitive for worn in items.values() if worn.primitive is not None]


def _grip(character) -> Grip:
    grip = getattr(character, "grip", None)
    if grip is None:
        grip = character.grip = Grip()
        character.pose_adjusters.append(grip)
    return grip


class Grip:
    """The arm pose holding a sign, blended over whatever the person's clips do (and let go of during yoga)."""

    def __init__(self):
        self.rotations: dict[int, np.ndarray] = {}
        self.on: float | None = None
        self.off: float | None = None
        self._quats: dict[int, np.ndarray] = {}

    def hold(self, character, rotations: dict, t: float | None) -> None:
        self.rotations = rotations
        self._quats = {b: quat_from_matrix(r[None])[0] for b, r in rotations.items()}
        self.on, self.off = (t if t is not None else getattr(character, "_time", 0.0)), None

    def release(self, character, t: float | None) -> None:
        if self.on is not None:
            self.off = t if t is not None else getattr(character, "_time", 0.0)

    def weight(self, character, t: float) -> float:
        """How far the arm holds the sign at ``t``: lifting from ``on``, letting go from ``off`` (before them, when the
        scene was restarted, it holds it or has let go)."""
        if self.on is None:
            return 0.0
        if self.off is not None:
            w = 1.0 - _ease((t - self.off) / FADE) if t >= self.off else 0.0
        else:
            w = _ease((t - self.on) / FADE) if t >= self.on else 1.0
        return w * (1.0 - _busy(character, t))

    def __call__(self, local: np.ndarray, character, t: float) -> None:
        w = self.weight(character, t)
        sign = _items(character).get("sign")
        if sign is not None:
            hidden = _busy(character, t) > 0.5
            for prop in sign.props:
                prop.hidden = hidden
        if w <= 0.0:
            return
        for bone, q in self._quats.items():
            current = quat_from_matrix(local[bone : bone + 1, :3, :3])
            local[bone, :3, :3] = quat_to_matrix(quat_nlerp(current, q[None], w))[0]


def _busy(character, t: float) -> float:
    """1 while the person does a yoga clip (0 when not; easing in and out around it)."""
    from .performer import Performer

    performer = character.behaviour
    if not isinstance(performer, Performer):
        return 0.0
    act = performer.current(t)
    if act is None:
        return 0.0
    start, end = act.clip_start, act.clip_end
    return min(_ease((t - (start - FADE)) / FADE), 1.0 - _ease((t - end) / FADE))


def _ease(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


