"""Characters made in the character creator: saved for reuse, and put into the scene.

A created character is a Rocketbox avatar (its mesh, skeleton and animations) with its own looks: textures painted for
it (a person's face on the head, a new outfit on the body), the accessories it wears and, for editing it again later,
how those were made (the outfit recipe, whose face it is). Everything is kept in a folder per character::

    created/
      Ada at the fair/
        character.json     name, avatar, face, outfit recipe, items, which texture is whose
        f001_head.png      the textures painted for it, by primitive name (only those that differ from the avatar's)
        f001_body.png

The textures are stored ready to draw, so loading a character needs no generator or face model. The folder is looked
at again at most once a second (:meth:`CharacterBook.refresh`), so a character saved by the creator, copied in by
hand or deleted shows up in, or leaves, the scene a moment later. Face photos and the textures made from them are
personal data: the folder stays on this computer (it is ignored by git) and :meth:`CharacterBook.delete` removes
everything of a character.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

FILE = "character.json"
FORMAT = 1
RESCAN = 1.0  # seconds between looks at the folder


@dataclass
class CreatedCharacter:
    """One created character (see the module docs). ``textures`` maps primitive names of the avatar's model to the
    images painted for this character (RGBA arrays in memory; file names once saved)."""

    name: str
    avatar: str
    face: str | None = None  # the known person (recognition gallery) whose face it wears
    outfit: dict = field(default_factory=dict)  # the outfit recipe, for editing it again
    items: dict = field(default_factory=dict)  # accessories: {item: look name, colour or texture file}
    textures: dict = field(default_factory=dict)  # primitive name -> RGBA uint8 image (first row at the top)
    created: float = 0.0
    updated: float = 0.0
    folder: Path | None = None  # where it is saved, once it is

    def describe(self) -> str:
        parts = [self.avatar]
        if self.face:
            parts.append(f"{self.face}'s face")
        if self.outfit.get("name"):
            parts.append(str(self.outfit["name"]))
        return ", ".join(parts)


def slug(name: str) -> str:
    """A folder name for a character called ``name`` (letters, digits, spaces, ``-_.``; never empty or hidden)."""
    cleaned = re.sub(r"[^\w .-]+", "", name, flags=re.UNICODE).strip(" .")
    if not cleaned:
        raise ValueError(f"not a usable name for a character: {name!r}")
    return cleaned[:80]


def save(character: CreatedCharacter, folder) -> Path:
    """Write ``character`` into ``folder``/<its name>/ (replacing what was saved under that name); returns that path."""
    target = Path(folder).expanduser() / slug(character.name)
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    now = time.time()
    files = {}
    for primitive, image in character.textures.items():
        name = re.sub(r"[^\w.-]+", "_", primitive) + ".png"
        array = np.asarray(image)
        bgr = cv2.cvtColor(array, cv2.COLOR_RGBA2BGRA if array.shape[-1] == 4 else cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(tmp / name), bgr):
            raise OSError(f"cannot write {tmp / name}")
        files[primitive] = name
    data = {k: v for k, v in asdict(character).items() if k not in ("textures", "folder")}
    data.update(format=FORMAT, textures=files, created=character.created or now, updated=now)
    (tmp / FILE).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    old = target.with_name(f".{target.name}.{os.getpid()}.old")
    if target.exists():
        os.replace(target, old)
    os.replace(tmp, target)
    if old.exists():
        shutil.rmtree(old, ignore_errors=True)
    character.folder, character.created, character.updated = target, data["created"], now
    return target


def load(path) -> CreatedCharacter:
    """The character saved in the folder ``path``."""
    path = Path(path)
    data = json.loads((path / FILE).read_text(encoding="utf-8"))
    if int(data.get("format", 1)) > FORMAT:
        raise ValueError(f"{path} was saved by a newer version (format {data['format']})")
    textures = {}
    for primitive, name in (data.get("textures") or {}).items():
        image = cv2.imread(str(path / name), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise OSError(f"cannot read {path / name}")
        textures[primitive] = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA if image.shape[-1] == 4 else cv2.COLOR_BGR2RGB)
    return CreatedCharacter(str(data["name"]), str(data["avatar"]), data.get("face"), dict(data.get("outfit") or {}),
                            dict(data.get("items") or {}), textures, float(data.get("created", 0.0)),
                            float(data.get("updated", 0.0)), path)


class CharacterBook:
    """The created characters in ``folder`` (``created/`` by default; see the module docs)."""

    def __init__(self, folder="created"):
        self.folder = Path(folder).expanduser()
        self._known: dict[str, tuple[float, CreatedCharacter]] = {}  # folder name -> (character.json mtime, it)
        self._checked = -math.inf
        self.problems: dict[str, str] = {}  # folder name -> why it could not be read

    def refresh(self, now: float | None = None, force: bool = False) -> bool:
        """Look at the folder again (at most every ``RESCAN`` seconds unless ``force``); True if anything changed."""
        now = time.monotonic() if now is None else now
        if not force and now - self._checked < RESCAN:
            return False
        self._checked = now
        found: dict[str, tuple[float, CreatedCharacter]] = {}
        problems = {}
        if self.folder.is_dir():
            for path in sorted(self.folder.iterdir()):
                if path.name.startswith(".") or not (path / FILE).is_file():
                    continue
                try:
                    stamp = (path / FILE).stat().st_mtime_ns
                except OSError:
                    continue
                old = self._known.get(path.name)
                if old is not None and old[0] == stamp:
                    found[path.name] = old
                    continue
                try:
                    found[path.name] = (stamp, load(path))
                except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                    problems[path.name] = f"{type(exc).__name__}: {exc}"
        changed = {k: v[0] for k, v in found.items()} != {k: v[0] for k, v in self._known.items()}
        self._known, self.problems = found, problems
        return changed

    def characters(self) -> list[CreatedCharacter]:
        """The saved characters, oldest first."""
        return sorted((c for _, c in self._known.values()), key=lambda c: (c.created, c.name))

    def get(self, name: str) -> CreatedCharacter | None:
        entry = self._known.get(slug(name))
        return entry[1] if entry is not None else None

    def save(self, character: CreatedCharacter) -> Path:
        path = save(character, self.folder)
        self._known[path.name] = ((path / FILE).stat().st_mtime_ns, character)
        return path

    def delete(self, name: str) -> bool:
        """Remove the character called ``name`` and all its files; False if there is none."""
        path = self.folder / slug(name)
        if not (path / FILE).is_file():
            return False
        shutil.rmtree(path)
        self._known.pop(path.name, None)
        return True


def textures_of(character, primitives=None) -> dict[str, np.ndarray]:
    """The textures ``character`` (a scene :class:`~.character.Character`) has of its own (``texture_overrides``) as
    RGBA images, by primitive name: what :func:`save` keeps of it."""
    from ..scene.textures import as_texture

    out = {}
    for name, texture in getattr(character, "texture_overrides", {}).items():
        if primitives is None or name in primitives:
            out[name] = as_texture(texture).rgba.copy()
    return out


def apply_looks(person, created: CreatedCharacter) -> None:
    """Give the scene character ``person`` (made from ``created.avatar``) the looks of ``created``: its textures and
    accessories."""
    from ..scene.textures import Texture
    from . import accessories

    names = {p.name for p in person.skinned_model.primitives}
    person.texture_overrides = {name: Texture.from_array(image, order="rgb", name=f"{created.name} {name}")
                                for name, image in created.textures.items() if name in names}
    for item, look in created.items.items():
        try:
            accessories.dress(person, item, look)
        except (KeyError, ValueError, OSError):
            continue  # an accessory this avatar cannot wear, or a file gone: the rest still applies
