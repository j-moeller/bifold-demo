"""Characters made in the character creator: saved for reuse, and put into the scene.

A created character is a Rocketbox avatar (its mesh, skeleton and animations) with its own looks: textures painted for
it (a person's face on the head, a new outfit on the body), the accessories it wears and, for editing it again later,
how those were made (the outfit recipe, whose face it is). Everything is kept in a folder per character::

    created/
      Ada at the fair/
        character.json     name, avatar, face, outfit recipe, items, which texture is whose
        f001_head.png      the textures painted for it, by primitive name (only those that differ from the avatar's)
        f001_body.png
        f001_body.base.png what its outfit was made on, when not the avatar's own texture (a body whose skin was
                           matched to the face), so that the outfit can be changed later and keep that skin

The textures are stored ready to draw, so loading a character needs no generator or face model. The folder is looked
at again at most once a second (:meth:`CharacterBook.refresh`), so a character saved by the creator, copied in by
hand, changed (any of its files) or deleted shows up in, changes in, or leaves, the scene a moment later. That look
is quick: the book reads a character's images only when they are first used (:class:`Textures`), on the thread that
makes its 3D person. Files in a character's folder are hand-editable, so nothing in them is trusted: a
``character.json`` that is not one (not JSON, a field of the wrong kind, half written by a copy) is a problem of the
book (:attr:`CharacterBook.problems`), and a character that could be read before stays as it was until its file
reads well again. Face photos and the textures made from them are personal data: the folder stays on this computer
(it is ignored by git) and :meth:`CharacterBook.delete` removes everything of a character.
"""

from __future__ import annotations

import copy
import json
import math
import os
import re
import shutil
import threading
import time
from collections.abc import MutableMapping
from dataclasses import dataclass, field, fields
from pathlib import Path

import cv2
import numpy as np

FILE = "character.json"
FORMAT = 1
RESCAN = 1.0  # seconds between looks at the folder
MAX_TEXTURE = 1024  # pixels: a larger image is scaled down when it is read (as the avatars' own textures are)
ITEM_FILES = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif", ".tif", ".tiff", ".mp4", ".avi", ".mov", ".mkv",
              ".webm")  # an item look with one of these endings is a file of the character's folder


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
    bases: dict = field(default_factory=dict)  # primitive name -> what its look was made on, if not the avatar's own

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


def _slug_or_none(name: str) -> str | None:
    try:
        return slug(name)
    except ValueError:
        return None


def save(character: CreatedCharacter, folder) -> Path:
    """Write ``character`` into ``folder``/<its name>/ (replacing what was saved under that name); returns that path.
    The folder's other files (an item's picture, say) stay; the textures saved before are replaced. Raises
    ``ValueError`` (and writes nothing) for an item look that :func:`load` would not take, and for a name whose
    folder holds another character already (``"Ada?"`` and ``"Ada"`` both go to ``Ada/``, and so do ``"ada"`` and
    ``"Ada"`` where file names ignore case)."""
    _check_items(character.items)
    target = Path(folder).expanduser() / slug(character.name)
    before = _saved_name(target)
    if before is not None and before != character.name:
        raise ValueError(f"{target} holds another character, {before!r}: delete it first, or choose another name")
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    now = time.time()
    files = _write_images(character.textures, tmp, ".png")
    bases = _write_images(character.bases, tmp, ".base.png")
    for kept in _own_files(target, {*files.values(), *bases.values()}):  # an item's picture: the user's files stay
        shutil.copy2(kept, tmp / kept.name)
    data = {f.name: copy.deepcopy(getattr(character, f.name)) for f in fields(character)
            if f.name not in ("textures", "folder", "bases")}
    data.update(format=FORMAT, textures=files, created=character.created or now, updated=now)
    if bases:
        data["bases"] = bases
    (tmp / FILE).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    old = target.with_name(f".{target.name}.{os.getpid()}.old")
    if target.exists():
        os.replace(target, old)
    os.replace(tmp, target)
    if old.exists():
        shutil.rmtree(old, ignore_errors=True)
    character.folder, character.created, character.updated = target, data["created"], now
    return target


def _write_images(images: dict, folder: Path, ending: str) -> dict:
    """Write ``images`` (primitive name -> RGB(A) image) into ``folder`` as PNG files named after the primitives with
    ``ending``; returns their file names by primitive name."""
    files = {}
    for primitive, image in images.items():
        name = re.sub(r"[^\w.-]+", "_", primitive) + ending
        array = np.asarray(image)
        bgr = cv2.cvtColor(array, cv2.COLOR_RGBA2BGRA if array.shape[-1] == 4 else cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(folder / name), bgr):
            raise OSError(f"cannot write {folder / name}")
        files[primitive] = name
    return files


def _saved_name(folder: Path) -> str | None:
    """The name of the character saved in ``folder``, or ``None`` (nothing there, or not readable)."""
    try:
        data = json.loads((folder / FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError):
        return None
    name = data.get("name") if isinstance(data, dict) else None
    return name if isinstance(name, str) else None


def _own_files(folder: Path, replaced: set[str]) -> list[Path]:
    """The files of the saved character in ``folder`` that are not what :func:`save` writes: not its
    ``character.json``, not the textures it listed, not those it writes now (``replaced``)."""
    try:
        data = json.loads((folder / FILE).read_text(encoding="utf-8"))
        ours = set()
        for key in ("textures", "bases"):
            listed = data.get(key) if isinstance(data, dict) else None
            ours |= {str(v) for v in listed.values()} if isinstance(listed, dict) else set()
        return [p for p in folder.iterdir() if p.is_file() and not p.name.startswith(".")
                and p.name != FILE and p.name not in ours | replaced]
    except (OSError, ValueError, RecursionError):
        return []


class Textures(MutableMapping):
    """A saved character's images by primitive name, each read from its file the first time it is looked at (then
    kept), so that looking through a folder of characters is quick and the images are decoded where they are needed.
    Safe to use from several threads."""

    def __init__(self, files=None):
        self._files: dict[str, Path | None] = {k: Path(v) for k, v in dict(files or {}).items()}
        self._images: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()

    def __getitem__(self, primitive: str) -> np.ndarray:
        with self._lock:
            image = self._images.get(primitive)
            if image is None:
                image = self._images[primitive] = _read(self._files[primitive])
            return image

    def __setitem__(self, primitive: str, image) -> None:
        with self._lock:
            self._files[primitive] = None
            self._images[primitive] = image

    def __delitem__(self, primitive: str) -> None:
        with self._lock:
            del self._files[primitive]
            self._images.pop(primitive, None)

    def __iter__(self):
        return iter(list(self._files))

    def __len__(self) -> int:
        return len(self._files)

    def __repr__(self) -> str:
        return f"Textures({', '.join(self._files)})"


def _read(path: Path) -> np.ndarray:
    """The image file ``path`` as RGBA ``uint8`` (any PNG: grey, a palette, 16 bits per channel), at most
    ``MAX_TEXTURE`` pixels wide and high. Raises ``OSError`` or ``ValueError`` when it cannot be read."""
    from ..scene.textures import _read_image

    try:
        return _read_image(Path(path), MAX_TEXTURE)
    except ValueError as exc:  # not an image (yet: half written, say)
        raise OSError(f"cannot read {path}: {exc}") from None


def _rgba8(image) -> np.ndarray:
    """An image of a character in memory (RGB or RGBA; 8 or 16 bits) as RGBA ``uint8``, at most ``MAX_TEXTURE``
    pixels wide and high."""
    from ..scene.textures import _array_to_rgba, _fit

    image = np.asarray(image)
    if image.dtype == np.uint16:
        image = (image >> 8).astype(np.uint8)
    if not (image.dtype == np.uint8 and image.ndim == 3 and image.shape[2] == 4):  # (read from a file: it is)
        image = _array_to_rgba(image, "rgb")
    return _fit(image, MAX_TEXTURE)


def load(path, lazy: bool = False) -> CreatedCharacter:
    """The character saved in the folder ``path``; ``lazy``: its images are read when first used (:class:`Textures`;
    only whether their files are there is checked now). Raises ``OSError`` for a file it cannot read and
    ``ValueError`` for a ``character.json`` that is not a character's (not JSON, saved by a newer version, a field
    of the wrong kind: a hand-written file, say)."""
    path = Path(path)
    return _load(path, (path / FILE).read_bytes(), lazy)


_JSON_KINDS = {list: "an array", str: "a string", int: "a number", float: "a number", bool: "true or false",
               type(None): "null"}


def _load(path: Path, raw: bytes, lazy: bool) -> CreatedCharacter:
    """:func:`load` of the character in ``path`` whose ``character.json`` holds ``raw``."""
    try:
        data = json.loads(raw.decode("utf-8"))
    except (ValueError, RecursionError) as exc:  # not JSON (text), or nested too deep to read
        raise ValueError(f"not a character file ({type(exc).__name__}: {exc})") from None
    if not isinstance(data, dict):
        kind = _JSON_KINDS.get(type(data), "a value")
        raise ValueError(f"not a character file ({kind}, not an object)")  # noqa: TRY004 - what the file holds
    version = data.get("format", 1)
    if not _a_number(version) or version != int(version):
        raise ValueError(f"the format is not a whole number: {version!r}")
    if int(version) > FORMAT:
        raise ValueError(f"{path} was saved by a newer version (format {int(version)})")
    name, avatar, face = _text(data, "name"), _text(data, "avatar"), data.get("face")
    if face is not None and not isinstance(face, str):
        raise ValueError(f"the face is not a name: {face!r}")
    files, bases = _files(path, data, "textures"), _files(path, data, "bases")
    items = _check_items(data.get("items"))

    def images(found):
        return Textures(found) if lazy else {primitive: _read(file) for primitive, file in found.items()}

    return CreatedCharacter(name, avatar, face, _mapping(data, "outfit"), items, images(files),
                            _seconds(data, "created"), _seconds(data, "updated"), path, images(bases))


def _files(path: Path, data: dict, key: str) -> dict:
    """The image files listed under ``key`` (``"textures"`` or ``"bases"``) of a ``character.json``, by primitive
    name; raises ``ValueError`` for a name that leads out of the folder and ``OSError`` for a file that is not
    there."""
    files = {}
    for primitive, file_name in _mapping(data, key).items():
        if not isinstance(file_name, str) or not _in_folder(file_name):
            what = "texture" if key == "textures" else "base texture"
            raise ValueError(f"the {what} of {primitive} is not a file of the folder: {file_name!r}")
        files[primitive] = path / file_name
    for file in files.values():
        if not file.is_file():
            raise OSError(f"cannot read {file}")
    return files


def _a_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _text(data: dict, key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"no {key}" if value is None else f"the {key} is not a text: {value!r}")
    return value


def _mapping(data: dict, key: str) -> dict:
    value = data.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{key} is not a mapping of names: {value!r}")  # noqa: TRY004 - what the file holds
    return dict(value)


def _seconds(data: dict, key: str) -> float:
    value = data.get(key)
    if value is None:
        return 0.0
    if not _a_number(value):
        raise ValueError(f"{key} is not a time in seconds: {value!r}")
    return float(value)


def _in_folder(name: str) -> bool:
    """Whether the file name ``name`` (from a ``character.json``) stays inside the character's folder."""
    parts = Path(name).parts
    return bool(parts) and not Path(name).is_absolute() and ".." not in parts and name not in (".", "")


def _check_items(items) -> dict:
    """``items`` (item -> look) as a dict; raises ``ValueError`` for something that is not one, or for a look that
    cannot be one (:func:`_a_look`)."""
    if items is None:
        return {}
    if not isinstance(items, dict):
        raise ValueError(f"the items are not a mapping of item to look: {items!r}")  # noqa: TRY004
    items = dict(items)
    for item, look in items.items():
        if not _a_look(look):
            raise ValueError(f"the look of the {item} is not a look name, colour or image file: {look!r}")
    return items


def _a_look(look) -> bool:
    """Whether ``look`` (from a saved ``character.json``) can be an item's look: a look name, colour name or file
    (a string), ``true`` (the item's first look), ``false`` or ``null`` (not worn), an RGB(A) colour (three or four
    numbers 0-255)."""
    if look is None or isinstance(look, (str, bool)):
        return True
    return isinstance(look, (list, tuple)) and len(look) in (3, 4) and \
        all(isinstance(v, (int, float)) and not isinstance(v, bool) and 0 <= v <= 255 for v in look)


def _stamp(path: Path) -> tuple | None:
    """What changes when anything in the character folder ``path`` changes (the names, sizes and times of its files:
    an image copied in after its ``character.json``, say), or ``None`` without a ``character.json``. Raises
    ``OSError`` when the folder cannot be read (or is not one)."""
    files = []
    with os.scandir(path) as entries:
        for entry in entries:
            if not entry.name.startswith(".") and entry.is_file():
                info = entry.stat()
                files.append((entry.name, info.st_mtime_ns, info.st_size))
    return tuple(sorted(files)) if any(name == FILE for name, *_ in files) else None


def _others(stamp: tuple) -> tuple:
    """What of a :func:`_stamp` is not the ``character.json``'s (its images, an item's picture)."""
    return tuple(entry for entry in stamp if entry[0] != FILE)


class CharacterBook:
    """The created characters in ``folder`` (``created/`` by default; see the module docs).

    :attr:`problems` maps a folder name to why its character cannot be read (it is left out, or, when it could be
    read before, kept as it was then); ``trouble`` says why the folder itself could not be looked at, the last time
    (the characters then stay as they were)."""

    def __init__(self, folder="created"):
        self.folder = Path(folder).expanduser()
        # folder name -> (_stamp of its files, the character, what its character.json held)
        self._known: dict[str, tuple[tuple, CreatedCharacter, bytes | None]] = {}
        self._checked = -math.inf
        self.problems: dict[str, str] = {}  # folder name -> why it could not be read
        self.trouble: str | None = None

    def refresh(self, now: float | None = None, force: bool = False) -> bool:
        """Look at the folder again (at most every ``RESCAN`` seconds unless ``force``); True if anything changed.
        Never raises for what is in the folder (see the module docs)."""
        now = time.monotonic() if now is None else now
        if not force and now - self._checked < RESCAN:
            return False
        self._checked = now
        try:
            paths = sorted(self.folder.iterdir()) if self.folder.is_dir() else []
        except OSError as exc:  # it went away (or may not be listed) just now: as it was until the next look
            self.trouble = f"cannot look into {self.folder}: {exc}"
            return False
        self.trouble = None
        found: dict[str, tuple[tuple, CreatedCharacter, bytes | None]] = {}
        problems = {}
        for path in paths:
            if path.name.startswith("."):
                continue
            old = self._known.get(path.name)
            try:
                stamp = _stamp(path)
            except OSError:  # a file, not a folder; or changing just now: as it was, until the next look
                if old is not None:
                    found[path.name] = old
                continue
            if stamp is None:
                continue
            if old is not None and old[0] == stamp:
                found[path.name] = old
                continue
            try:
                raw = (path / FILE).read_bytes()
                if old is not None and old[2] == raw and _others(old[0]) == _others(stamp):
                    found[path.name] = (stamp, old[1], raw)  # written again as it was (an editor): the same
                    continue
                found[path.name] = (stamp, _load(path, raw, lazy=True), raw)
            except Exception as exc:  # noqa: BLE001 - whatever is in the folder, the demo goes on
                problems[path.name] = f"{type(exc).__name__}: {exc}"
                if old is not None:  # e.g. being written just now: as it was, until it reads well
                    found[path.name] = old
                    problems[path.name] += " (shown as it was)"
        folders_of: dict[str, list[str]] = {}
        for folder, (_, character, _) in found.items():
            folders_of.setdefault(character.name, []).append(folder)
        for name, folders in folders_of.items():  # a folder copied by hand: its own (or the first) one counts
            keep = min(folders, key=lambda f, name=name: (f != _slug_or_none(name), f))
            for folder in folders:
                if folder != keep:
                    del found[folder]
                    problems[folder] = f"another folder ({keep}) holds a character called {name!r}"
        changed = {k: id(v[1]) for k, v in found.items()} != {k: id(v[1]) for k, v in self._known.items()}
        self._known, self.problems = found, problems
        return changed

    def characters(self) -> list[CreatedCharacter]:
        """The saved characters, oldest first."""
        return sorted((c for _, c, _ in self._known.values()), key=lambda c: (c.created, c.name))

    def get(self, name: str) -> CreatedCharacter | None:
        entry = self._known.get(_slug_or_none(name) or "")
        return entry[1] if entry is not None else None

    def save(self, character: CreatedCharacter) -> Path:
        path = save(character, self.folder)
        self._known[path.name] = (_stamp(path), character, (path / FILE).read_bytes())
        return path

    def delete(self, name: str) -> bool:
        """Remove the character called ``name`` and all its files; False if there is none (its folder holding
        another one, ``"Ada"`` for ``"Ada?"``, is left alone)."""
        path = self.folder / slug(name)
        if not (path / FILE).is_file() or _saved_name(path) not in (name, None):
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


def textures_for(created: CreatedCharacter, model) -> dict:
    """The textures of ``created`` for the primitives of ``model`` (its avatar), ready to draw
    (:class:`~bifold_demo.scene.textures.Texture`, at most ``MAX_TEXTURE`` pixels wide and high): the slow part of
    :func:`apply_looks` (reading and converting the images), which can run on another thread (it pauses between
    images, see :mod:`bifold_demo.pacing`). Raises ``OSError`` or ``ValueError`` for an image it cannot use."""
    from ..pacing import pause
    from ..scene.textures import Texture, prepare_upload

    names = {p.name for p in model.primitives}
    out = {}
    for name in created.textures:
        if name in names:
            pause()
            out[name] = Texture(_rgba8(created.textures[name]), name=f"{created.name} {name}")
            pause()
            prepare_upload(out[name])
    return out


def item_look(created: CreatedCharacter, look):
    """The look ``look`` of one of ``created``'s items as :func:`~.accessories.dress` takes it: a look ending in
    ``.png`` (an image or video file, :data:`ITEM_FILES`) is a file of the character's folder, never one elsewhere
    (raises ``ValueError`` for a name that leads out of it); anything else is a look name or a colour."""
    if not (isinstance(look, str) and Path(look).suffix.lower() in ITEM_FILES):
        return look
    if not _in_folder(look):
        raise ValueError(f"{look!r} is not a file of the character's folder")
    if created.folder is None:
        raise ValueError(f"{look!r}: the character is not saved, so it has no folder")
    return created.folder / look


@dataclass
class Looks:
    """A created character's looks made ready to put on (:func:`prepare_looks`): ``textures`` by primitive name;
    ``items``: the look of each item that is a file, read (a :class:`~bifold_demo.scene.textures.Texture`), or why it
    cannot be (an exception); ``shared``: the textures of its other item looks (a look of the wardrobe, a colour),
    the scene's shared ones, read already."""

    textures: dict = field(default_factory=dict)
    items: dict = field(default_factory=dict)
    shared: list = field(default_factory=list)

    def gpu(self) -> list:
        """The textures among them, for the GPU (``Renderer.prepare``)."""
        from ..scene.textures import Texture

        return [t for t in (*self.textures.values(), *self.items.values(), *self.shared) if isinstance(t, Texture)]


def prepare_looks(created: CreatedCharacter, model, wardrobe=None) -> Looks:
    """Everything slow of putting ``created``'s looks on a 3D person of ``model``, for another thread (it pauses
    between its steps, see :mod:`bifold_demo.pacing`): its :func:`textures_for`, the pictures of its items read (each
    its own texture, not the scene's shared one of that file: a changed file is new looks, :class:`CharacterBook`),
    the pictures of its looks of ``wardrobe`` read (into the scene's shared textures), and how items fit the avatar
    worked out. :func:`apply_looks` then has nothing slow left to do."""
    from ..pacing import pause
    from ..scene.textures import VIDEO_EXTENSIONS, Texture, VideoTexture, as_texture, prepare_upload
    from . import accessories

    looks = Looks(textures_for(created, model))
    if created.items and accessories.supports(model):
        pause()
        accessories.fit(model)
        for item, look in created.items.items():
            pause()
            try:
                file = item_look(created, look)
                if isinstance(file, Path):
                    if file.suffix.lower() in VIDEO_EXTENSIONS:
                        looks.items[item] = VideoTexture(file)
                    else:
                        looks.items[item] = prepare_upload(Texture(_read(file), name=file.stem))
                elif look is not None and look is not False:  # a look of the wardrobe, a colour: read now (kept)
                    looks.shared.append(as_texture(accessories.look_of(item, look, wardrobe)))
            except Exception as exc:  # noqa: BLE001 - apply_looks leaves the item out and says why
                looks.items[item] = exc
    return looks


def apply_looks(person, created: CreatedCharacter, wardrobe=None, t: float | None = None,
                looks: Looks | None = None) -> list[str]:
    """Give the scene character ``person`` (made from ``created.avatar``) the looks of ``created``: its textures and
    accessories, instead of what it wore before. An item's look is a look name of ``wardrobe`` (an
    :class:`~.accessories.Wardrobe`; the built-in looks without one), a colour, or an image file of the character's
    folder (:func:`item_look`). ``t``: the scene time it happens (a sign is lifted from then on; ``-inf``: held up
    already); ``looks``: :func:`prepare_looks` made already (else it is done now). An item that cannot be worn (this
    avatar has no such item, its file is gone, a look of the wrong kind) is left out, and the rest still applies; so
    are textures for parts the avatar does not have (painted for another avatar): returns what was left out and why
    (``["hat: ..."]``). Raises when the textures cannot be made, before anything changes."""
    from . import accessories

    model = person.skinned_model
    looks = prepare_looks(created, model, wardrobe) if looks is None else looks
    person.texture_overrides = dict(looks.textures)
    names = {p.name for p in model.primitives}
    unknown = [name for name in created.textures if name not in names]
    left_out = [f"textures of {', '.join(unknown)}: {model.name} has no such part"] if unknown else []
    if not accessories.supports(model):
        return left_out + [f"{item}: {model.name} cannot wear items" for item in created.items]
    for item in accessories.wearing(person):
        accessories.dress(person, item, None, t=t)
    for item, look in created.items.items():
        try:
            look = looks.items[item] if item in looks.items else item_look(created, look)
            if isinstance(look, Exception):
                raise look
            accessories.dress(person, item, look, wardrobe, t=t)
        except Exception as exc:  # noqa: BLE001 - one item that cannot be worn: the rest still applies
            left_out.append(f"{item}: {exc.args[0] if isinstance(exc, KeyError) and exc.args else exc}")
    return left_out
