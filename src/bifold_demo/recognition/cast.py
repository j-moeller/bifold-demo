"""The known people in the scene: each person of the faces folder gets a 3D person who wears their face.

:class:`FaceCast` gives every known person one 3D person (those standing in front of the camera first, so the face
recognition sees them), keeps them while the folder keeps the person, takes the face off when the person goes, and
gives newly added people a 3D person a second later. A face can also be put on someone by hand (:meth:`put`), e.g.
from selection mode. Painting a face takes a fraction of a second: it runs on a worker thread, and the face appears
when it is ready.
"""

from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor

from ..characters import face_photo

PREFER = ("audience", "approach")  # scene-object name prefixes of the people who get the faces first
CHECK = 1.0  # seconds between looks at the folder and the scene


class FaceCast:
    """Puts the faces of ``gallery``'s people on the 3D people of a scene (module docs). ``auto``: give each known
    person a 3D person by itself (else only :meth:`put` does)."""

    def __init__(self, gallery, auto: bool = True, prefer=PREFER):
        self.gallery = gallery
        self.auto = auto
        self.prefer = tuple(prefer)
        self._scene = None
        self._wearing: dict[int, tuple] = {}  # id(character) -> (character, name, photo path, by hand)
        self._painting: dict[int, tuple] = {}  # id(character) -> (character, name, photo, by hand, future)
        self._pool: ThreadPoolExecutor | None = None
        self._checked = -1e9

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
        for key, (character, name, photo, by_hand, future) in list(self._painting.items()):
            if not future.done():
                continue
            del self._painting[key]
            try:
                texture = future.result()
            except Exception as exc:  # noqa: BLE001 - a photo that cannot be used: say so, keep going
                messages.append(f"cannot put {name}'s face on {character.name}: {exc}")
                continue
            head = character.skinned_model.primitives[face_photo.head_layout(character.skinned_model).primitive]
            character.texture_overrides[head.name] = texture
            self._wearing[key] = (character, name, photo, by_hand)
            messages.append(f"{character.name} has {name}'s face")
        return messages

    def _cast(self) -> list[str]:
        self.gallery.refresh()
        known = self.gallery.people()
        messages = []
        for key, (character, name, photo, by_hand) in list(self._wearing.items()):
            photos = self.gallery.photos(name) if name in known else []
            if not photos or self._scene.get(character.name) is not character:
                self._take_off(character)
                if not photos:
                    messages.append(f"{name} is no longer known: {character.name} has their own face again")
            elif photos[0] != photo and key not in self._painting:  # a better photo came: paint again
                self._paint(character, name, photos[0], by_hand)
        if not self.auto:
            return messages
        cast = {entry[1] for entry in self._wearing.values()} | {entry[1] for entry in self._painting.values()}
        free = [c for c in self._candidates() if id(c) not in self._wearing and id(c) not in self._painting]
        for name in known:
            if name not in cast and free:
                self._paint(free.pop(0), name, self.gallery.photos(name)[0], False)
        return messages

    def _candidates(self) -> list:
        people = [obj for obj in self._scene if getattr(obj, "skinned_model", None) is not None
                  and getattr(obj, "texture_overrides", None) is not None
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
        photos = self.gallery.photos(name)
        if not photos:
            return f"no photo of {name} with a face"
        for other, other_name, *_ in [*self._wearing.values(), *self._painting.values()]:
            if other_name == name and other is not character:
                self._take_off(other)
                self._cancel(other)
        self._paint(character, name, photos[0], True)
        return f"putting {name}'s face on {character.name}"

    # -- helpers ----------------------------------------------------------------------------------------------

    def _paint(self, character, name: str, photo, by_hand: bool) -> None:
        if self._pool is None:
            self._pool = ThreadPoolExecutor(1, thread_name_prefix="face-paint")
        engine = self.gallery.engine
        future: Future = self._pool.submit(_painted, character, photo, engine)
        self._cancel(character)
        self._painting[id(character)] = (character, name, photo, by_hand, future)

    def _cancel(self, character) -> None:
        entry = self._painting.pop(id(character), None)
        if entry is not None:
            entry[-1].cancel()

    def _take_off(self, character) -> None:
        if self._wearing.pop(id(character), None) is not None:
            face_photo.take_face_off(character)

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


def _painted(character, photo, engine):
    """The painted head texture for ``character`` (on the worker thread)."""
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
