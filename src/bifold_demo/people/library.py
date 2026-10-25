"""A folder of people cut-outs (PNG with transparency plus a JSON sidecar) for use in scenes."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ..scene.textures import Texture, as_texture
from .cutout import Cutout

DEFAULT_FOLDER = "people"


@dataclass
class Person:
    """One person in a library.

    Attributes:
        name: file stem (``people/<name>.png``).
        height: real-world height in metres of the whole cut-out (head to feet for full-body people).
        full_body: head and feet are both visible.
        pose: ``"standing"`` or ``"walking"``.
        facing: ``"front"``, ``"left"`` or ``"right"`` (the direction a walking person faces).
    """

    name: str
    path: Path
    height: float = 1.72
    full_body: bool = True
    pose: str = "standing"
    facing: str = "front"
    source: str = ""
    prompt: str = ""
    seed: int | None = None
    _textures: dict = field(default_factory=dict, repr=False, compare=False)

    def texture(self, mirror: bool = False) -> Texture:
        """The cut-out as a texture, optionally mirrored left-right."""
        base = as_texture(self.path)
        if not mirror:
            return base
        if "mirrored" not in self._textures or self._textures["mirrored"][0] != base.version:
            self._textures["mirrored"] = (base.version, Texture(base.rgba[:, ::-1].copy(), name=f"{self.name} (mirrored)"))
        return self._textures["mirrored"][1]

    def facing_texture(self, direction: str) -> Texture:
        """The texture of a side-view person turned to face ``"left"`` or ``"right"``."""
        return self.texture(mirror=self.facing in ("left", "right") and self.facing != direction)

    def metadata(self) -> dict:
        data = {k: v for k, v in asdict(self).items() if not k.startswith("_")}
        data.pop("path")
        return data


class PeopleLibrary:
    """People stored in a folder. Any transparent PNG dropped in is picked up too."""

    def __init__(self, folder=DEFAULT_FOLDER):
        self.folder = Path(folder).expanduser()

    def people(self) -> list[Person]:
        if not self.folder.is_dir():
            return []
        people = []
        for png in sorted(self.folder.glob("*.png")):
            if png.name.startswith(("_", ".")):
                continue
            meta = {}
            sidecar = png.with_suffix(".json")
            if sidecar.exists():
                try:
                    meta = json.loads(sidecar.read_text())
                except (OSError, ValueError):
                    meta = {}
            known = {k: meta[k] for k in ("height", "full_body", "pose", "facing", "source", "prompt", "seed") if k in meta}
            people.append(Person(name=png.stem, path=png, **known))
        return people

    def __len__(self) -> int:
        return len(self.people())

    def add(self, cutout: Cutout | np.ndarray, name: str | None = None, **metadata) -> Person:
        """Save a cut-out (or an RGBA array) and its metadata; returns the stored person."""
        rgba = cutout.rgba if isinstance(cutout, Cutout) else cutout
        if isinstance(cutout, Cutout):
            metadata.setdefault("full_body", cutout.full_body)
        self.folder.mkdir(parents=True, exist_ok=True)
        stem = self._unique(_slug(name or "person"))
        path = self.folder / f"{stem}.png"
        if not cv2.imwrite(str(path), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)):
            raise OSError(f"cannot write {path}")
        person = Person(name=stem, path=path, **metadata)
        path.with_suffix(".json").write_text(json.dumps(person.metadata(), indent=2))
        return person

    def _unique(self, stem: str) -> str:
        candidate, n = stem, 1
        while (self.folder / f"{candidate}.png").exists():
            n += 1
            candidate = f"{stem}_{n}"
        return candidate

    def contact_sheet(self, row_height: int = 320, per_row: int = 10) -> np.ndarray | None:
        """All people side by side on grey (BGR), for a quick visual check."""
        tiles = []
        for person in self.people():
            image = cv2.imread(str(person.path), cv2.IMREAD_UNCHANGED)
            if image is None or image.ndim != 3 or image.shape[2] != 4:
                continue
            width = max(1, round(image.shape[1] * row_height / image.shape[0]))
            image = cv2.resize(image, (width, row_height), interpolation=cv2.INTER_AREA)
            alpha = image[..., 3:4] / 255.0
            tile = (image[..., :3] * alpha + 128 * (1 - alpha)).astype(np.uint8)
            cv2.putText(tile, person.name[:18], (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(np.pad(tile, ((4, 4), (4, 4), (0, 0)), constant_values=40))
        if not tiles:
            return None
        rows = [np.hstack(tiles[i : i + per_row]) for i in range(0, len(tiles), per_row)]
        width = max(r.shape[1] for r in rows)
        return np.vstack([np.pad(r, ((0, 0), (0, width - r.shape[1]), (0, 0)), constant_values=40) for r in rows])


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_") or "person"
