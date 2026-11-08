"""The known people in the scene: each person of the faces folder gets a 3D person who wears their face.

:class:`FaceCast` gives every known person one 3D person (those standing in front of the camera first, so the face
recognition sees them), keeps them while the folder keeps the person, takes the face off when the person goes, and
gives newly added people a 3D person a second later. A face can also be put on someone by hand (:meth:`put`), e.g.
from selection mode. Characters made in the character creator (``person.created``, see ``scenes.created_cast``) have
faces of their own: they never get a known person's.

The face comes from all the person's photos: the head textures from photos (``characters.headtex``, stages 0 to 7:
the photos projected onto the avatar's head, completed, the body's skin and the hair matched to the face) give the 3D
person a new head texture, a body texture with the person's skin and recoloured hair. That takes a few seconds (the
first time for an avatar some more), so it runs on a worker thread, and the result is kept by the photos' contents
in the faces folder (``.headtex/<name>/``), so the next start reuses it. Textures made from the photos are personal
data like the photos: a person's are deleted when the person leaves the folder. Without MediaPipe, or when the photos
cannot be used (no face MediaPipe accepts, say), the face is painted from the first photo onto the avatar's head
texture (``characters.face_photo``), as before, with a message saying why. The face appears when it is ready.
"""

from __future__ import annotations

import shutil
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from ..characters import face_photo

PREFER = ("audience", "approach")  # scene-object name prefixes of the people who get the faces first
CHECK = 1.0  # seconds between looks at the folder and the scene
CACHE = ".headtex"  # the folder in the faces folder that keeps the textures made from the photos


class FaceCast:
    """Puts the faces of ``gallery``'s people on the 3D people of a scene (module docs). ``auto``: give each known
    person a 3D person by itself (else only :meth:`put` does)."""

    def __init__(self, gallery, auto: bool = True, prefer=PREFER):
        self.gallery = gallery
        self.auto = auto
        self.prefer = tuple(prefer)
        self._scene = None
        self._wearing: dict[int, tuple] = {}  # id(character) -> (character, name, photos, by hand, textures worn)
        self._painting: dict[int, tuple] = {}  # id(character) -> (character, name, photos, by hand, future)
        self._pool: ThreadPoolExecutor | None = None
        self._checked = -1e9
        self._told: set[str] = set()  # the fallback's reasons said once

    # -- every frame ------------------------------------------------------------------------------------------

    def update(self, scene, now: float | None = None) -> list[str]:
        """Apply the faces painted since the last call and (once a second) cast the known people; ``scene`` is the
        one shown. Returns messages for the user."""
        if scene is not self._scene:
            if self._scene is not None:  # another scene: its people are gone
                self._forget()
            self._scene = scene
        messages = self._apply()
        now = time.monotonic() if now is None else now
        if now - self._checked >= CHECK:
            self._checked = now
            messages += self._cast()
        return messages

    def _apply(self) -> list[str]:
        messages = []
        for key, (character, name, photos, by_hand, future) in list(self._painting.items()):
            if not future.done():
                continue
            del self._painting[key]
            try:
                textures, note = future.result()
            except Exception as exc:  # noqa: BLE001 - a photo that cannot be used: say so, keep going
                messages.append(f"cannot put {name}'s face on {character.name}: {exc}")
                continue
            self._take_off(character)
            worn = _wear(character, textures)
            self._wearing[key] = (character, name, photos, by_hand, worn)
            messages.append(f"{character.name} has {name}'s face")
            if note and note not in self._told:  # (a reason once: MediaPipe's is the same for everyone)
                self._told.add(note)
                messages.append(f"{name}'s face is {note}")
        return messages

    def _cast(self) -> list[str]:
        self.gallery.refresh(wait=False)  # new photos are enrolled on a worker thread: the frames go on meanwhile
        known = self.gallery.people()
        self._forget_textures(known)
        messages = []
        trouble = getattr(self.gallery, "trouble", None)
        if trouble and trouble not in self._told:
            self._told.add(trouble)
            messages.append(trouble)
        for key, (character, name, photos, by_hand, _worn) in list(self._wearing.items()):
            now = tuple(self.gallery.photos(name)) if name in known else ()
            if not now or self._scene.get(character.name) is not character:
                self._take_off(character)
                if not now:
                    messages.append(f"{name} is no longer known: {character.name} has their own face again")
            elif now != photos and key not in self._painting:  # photos added or changed: make it again
                self._paint(character, name, now, by_hand)
        if not self.auto:
            return messages
        cast = {entry[1] for entry in self._wearing.values()} | {entry[1] for entry in self._painting.values()}
        free = [c for c in self._candidates() if id(c) not in self._wearing and id(c) not in self._painting]
        for name in known:
            if name not in cast and free:
                self._paint(free.pop(0), name, tuple(self.gallery.photos(name)), False)
        return messages

    def _candidates(self) -> list:
        people = [obj for obj in self._scene if getattr(obj, "skinned_model", None) is not None
                  and getattr(obj, "texture_overrides", None) is not None and getattr(obj, "created", None) is None
                  and face_photo.head_layout(obj.skinned_model) is not None]

        def rank(obj):
            name = obj.name or ""
            first = next((k for k, prefix in enumerate(self.prefer) if name.startswith(prefix)), len(self.prefer))
            return first, name

        return sorted(people, key=rank)

    # -- by hand ----------------------------------------------------------------------------------------------

    def known(self) -> list[str]:
        """The known people's names."""
        return list(self.gallery.people())

    def wears(self, character) -> str | None:
        """Whose face ``character`` has (or is about to get), or ``None``."""
        entry = self._wearing.get(id(character)) or self._painting.get(id(character))
        return entry[1] if entry is not None else None

    def put(self, character, name: str | None) -> str:
        """Give ``character`` the face of ``name`` (``None``: their own face back); a person wears one face, so whoever
        had that face before gets their own back."""
        if name is None:
            self._take_off(character)
            self._cancel(character)
            return f"{character.name} has their own face again"
        created = getattr(character, "created", None)
        if created is not None:
            return f"{character.name} is {created.name}, who has a face of their own (from the character creator)"
        photos = self.gallery.photos(name)
        if not photos:
            return f"no photo of {name} with a face"
        for other, other_name, *_ in [*self._wearing.values(), *self._painting.values()]:
            if other_name == name and other is not character:
                self._take_off(other)
                self._cancel(other)
        self._paint(character, name, tuple(photos), True)
        return f"putting {name}'s face on {character.name}"

    # -- helpers ----------------------------------------------------------------------------------------------

    def _paint(self, character, name: str, photos: tuple, by_hand: bool) -> None:
        if self._pool is None:
            self._pool = ThreadPoolExecutor(1, thread_name_prefix="face-paint")
        engine = self.gallery.engine
        root = self._cache_root()
        cache = root / name if root is not None and _folder_name(name) else None
        future: Future = self._pool.submit(_made, character, list(photos), engine, cache)
        self._cancel(character)
        self._painting[id(character)] = (character, name, photos, by_hand, future)

    def _cache_root(self) -> Path | None:
        """Where the textures made from the photos are kept (a folder per person), or ``None``."""
        folder = getattr(self.gallery, "folder", None)
        return Path(folder) / CACHE if folder is not None else None

    def _forget_textures(self, known) -> None:
        """Delete the kept textures of people who are no longer in the faces folder."""
        root = self._cache_root()
        if root is None or not root.is_dir():
            return
        try:
            for path in root.iterdir():
                if path.is_dir() and path.name not in known:
                    shutil.rmtree(path, ignore_errors=True)
        except OSError:  # the folder went away meanwhile
            pass

    def _cancel(self, character) -> None:
        entry = self._painting.pop(id(character), None)
        if entry is not None:
            entry[-1].cancel()

    def _take_off(self, character) -> None:
        entry = self._wearing.pop(id(character), None)
        if entry is not None:
            _take_off(character, entry[-1])

    def _forget(self) -> None:
        """A new scene: the people of the old one are gone."""
        for entry in list(self._painting.values()):
            entry[-1].cancel()
        self._painting.clear()
        self._wearing.clear()
        self._checked = -1e9

    def close(self) -> None:
        self._forget()
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None


def _folder_name(name: str) -> bool:
    """Whether a person's name can be a folder of the cache (the gallery's names are folder or file names)."""
    return bool(name) and not name.startswith(".") and "/" not in name and "\\" not in name


def _made(character, photos, engine, cache) -> tuple[dict, str | None]:
    """``(textures, note)`` for ``character`` from ``photos`` (on the worker thread): the head textures from the
    photos (:func:`_from_photos`), else the face painted from the first photo (:func:`_painted`) and ``note`` says
    why."""
    from ..scene.textures import Texture, prepare_upload

    try:
        made = _from_photos(character, photos, cache)
        return {name: prepare_upload(Texture(rgba, name=f"{character.name} {name}")) for name, rgba in made.items()}, \
            None
    except ImportError:  # MediaPipe is not installed (the message says how to get it)
        why = "MediaPipe is not installed, see the README"
    except Exception as exc:  # noqa: BLE001 - photos the head textures cannot use: the simple painting
        why = f"the head texture from the photos failed: {exc}"
    texture = _painted(character, photos[0], engine)
    head = character.skinned_model.primitives[face_photo.head_layout(character.skinned_model).primitive]
    return {head.name: prepare_upload(texture)}, f"painted from one photo: {why}"


def _from_photos(character, photos, cache) -> dict:
    """The head textures from photos (stages 0 to 7, kept in ``cache``) for ``character``'s avatar: RGBA images by
    primitive name (the head, the body's skin, the hair)."""
    from ..characters.headtex.person import person_textures

    return person_textures(photos, character.skinned_model, cache=cache).textures


def _painted(character, photo, engine):
    """The head texture for ``character`` with the face of ``photo`` painted on (on the worker thread)."""
    import cv2

    from ..scene.textures import Texture

    image = cv2.imread(str(photo), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"cannot read {photo}")
    faces = engine.detect(image, min_size=0)
    if not faces:
        raise ValueError(f"no face in {photo}")
    painted = face_photo.paint_face(character.skinned_model, image, faces[0].landmarks, faces[0].box)
    return Texture.from_array(painted, order="rgb", name=f"{character.name} face")


def _wear(character, textures: dict) -> dict:
    from ..characters.headtex.person import wear_textures

    return wear_textures(character, textures)


def _take_off(character, worn: dict) -> None:
    from ..characters.headtex.person import take_off_textures

    take_off_textures(character, worn)
