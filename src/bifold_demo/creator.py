"""The character creator: a character made from a prepared avatar, dressed, given a known person's face and items, and
saved for the fair.

``C`` opens it in the demo; in selection mode ``C`` opens it for the selected person ("edit as a new character": their
avatar, outfit, items and face, without a name; a created character is loaded to edit it). ``Esc`` closes it. The
screen (:mod:`.display.creator_view`) shows the avatar on a turntable and, beside it, a row per choice:

* the avatar (the prepared ones), the name (typed), the face (none, or a known person of the faces folder);
* the outfit: a preset; per garment (top, shirt, bottoms, shoes) a pattern (the avatar's own clothes, a plain colour,
  a procedural pattern, or a fabric) in a colour and a second colour; a fabric made from a description, whose
  variations come one by one to pick from; the scale of the pattern or fabric;
* the items: a hat, glasses, a sign;
* save, delete, new; and the saved characters, to load one for editing.

:class:`Creator` is the logic, without a window: what is chosen, the work it starts and the keys. ``Up`` / ``Down``
go from row to row, ``Left`` / ``Right`` choose in a row (at once: the avatar, a preset, a colour ...; in the save
row and the saved characters ``Enter`` does it), ``Enter`` starts typing in the name or the fabric row (``Enter``
again ends it; for the fabric it starts the generator), ``,`` and ``.`` turn the avatar, ``F`` and ``B`` show its
front and back, ``Space`` stops or starts the turning. The slow parts never hold up the window: preparing an avatar
(its garment regions: the first time a few seconds), dressing it (0.1-0.5 s, :class:`~.characters.outfit.Tailor`),
its face (:func:`face_texture`: seconds), reading a saved character and saving one run on threads of their own
(quitting waits only for a save); a fabric from a description on the :class:`~.characters.outfit.FabricGenerator`'s
own thread. :meth:`Creator.update` (every frame) picks the results up: the newest choice wins, results made for an
older one are dropped, and :meth:`Creator.busy` says what is going on. A result, a key or a click that goes wrong is
said, never stops the demo.

Saving writes a :class:`~.characters.created.CreatedCharacter` through the demo's
:class:`~.characters.created.CharacterBook` (``created/``): the name, the avatar, whose face it wears, the outfit recipe
(:meth:`~.characters.outfit.Outfit.spec`, fabrics with their images), the items and the textures (the body texture the
tailor made, the head texture of the face). :class:`~.scenes.created_cast.CreatedCast` then puts the character into
the fair, and the demo says so. Loading one restores all of it for editing; its saved textures are used until
something changes.

:func:`face_texture` is the one place that makes a face, so that it can be switched: the head textures from photos
(:mod:`.characters.headtex`, all stages: the head, and the body's skin and the hair matched to the face) when
MediaPipe is installed, else the simple painting of the person's best photo
(:func:`~.characters.face_photo.paint_face`). A body texture the face comes with is what the outfit goes on; it is
saved with the character (:attr:`~.characters.created.CreatedCharacter.bases`), so a loaded character's outfit can
change and keep the skin.
"""

from __future__ import annotations

import math
import threading
import time
import traceback
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .characters import accessories
from .characters.accessories import Wardrobe
from .characters.outfit import PATTERNS, Outfit, Pattern, Solid, Swatch, color
from .characters.outfit.fabric import Fabric
from .display.window import BACKSPACE, BACKTAB, DOWN, LEFT, RIGHT, TAB, UP

GARMENTS = ("top", "shirt", "bottoms", "shoes")
COLOURS = ("black", "white", "grey", "charcoal", "navy", "royal blue", "sky blue", "teal", "green", "olive", "khaki",
           "beige", "brown", "burgundy", "red", "orange", "mustard", "yellow", "pink", "purple")
PATTERN_CHOICES = ("own", "plain", *PATTERNS, "fabric")  # own: the avatar's own clothes; fabric: from a description
ITEMS = ("hat", "glasses", "sign")
ACTIONS = ("save", "delete", "new")
ROWS = ("avatar", "name", "face", "preset", "garment", "pattern", "colour", "colour 2", "fabric", "variation", "scale",
        "hat", "glasses", "sign", "save", "saved")
TITLES = {"avatar": "Avatar", "name": "Name", "face": "Face", "preset": "Outfit", "garment": "Garment",
          "pattern": "Pattern", "colour": "Colour", "colour 2": "2nd colour", "fabric": "Fabric",
          "variation": "Variations", "scale": "Scale", "hat": "Hat", "glasses": "Glasses", "sign": "Sign",
          "save": "Character", "saved": "Saved"}
TEXT_ROWS = ("name", "fabric")
ENTER_ROWS = ("save", "saved")  # rows whose choice is done by Enter (or a click), not by going to it
ENTER = (13, 10)
MAX_TEXT = {"name": 40, "fabric": 100}
MAX_PHOTOS = 6  # photos of a person a face is made from
TURN_SPEED = 24.0  # degrees a second the turntable turns
TURN_STEP = 45.0  # degrees a key turns it
SCALE_STEP = 1.25  # a key makes a pattern's repeat this much larger (or smaller)
SCALE_RANGE = (0.004, 2.0)  # metres per repeat
CONFIRM = 4.0  # seconds a "press again" question waits for its answer
AVATARS_EVERY = 2.0  # seconds between looks at the prepared avatars


# -- the face -------------------------------------------------------------------------------------------------------


@dataclass
class FaceTextures:
    """What :func:`face_texture` made: ``textures`` by primitive name (RGBA ``uint8``, row 0 at the top): the head's,
    and a body texture beside it when the face's skin was matched on the body too (the outfit then goes on that);
    ``method``: how, in a few words; ``notes``: warnings."""

    textures: dict
    method: str
    notes: list = field(default_factory=list)


def _headtex_missing() -> str | None:
    """Why the head-texture pipeline cannot run here (``None`` when it can)."""
    try:
        from .characters.headtex import detect

        detect.mediapipe()
    except ImportError as exc:  # (MediaPipeMissing is one)
        return str(exc)
    return None


def face_texture(photos, model) -> FaceTextures:
    """The textures that put the face of the person in ``photos`` (paths, the best first; at most
    :data:`MAX_PHOTOS` are used) on ``model``. With MediaPipe, the head textures from photos, all their stages
    (:func:`~.characters.headtex.person_textures`: every usable photo fused into the head texture, what no photo
    shows completed, the body texture's skin matched to the face, the hair cards recoloured): the head's, the body's
    and the hair's textures; else, or when that fails, the simple painting (:func:`~.characters.face_photo.paint_face`)
    of the photo with the biggest face. Raises ``ValueError`` when no photo can be used. Slow (seconds; the first time
    for an avatar some more): call it off the main thread."""
    photos = [Path(p) for p in photos][:MAX_PHOTOS]
    if not photos:
        raise ValueError("there are no photos of that person")
    why = _headtex_missing()
    if why is None:
        try:
            from .characters.headtex import person_textures

            person = person_textures(photos, model)
            used = len(person.result.projections) if person.result is not None else len(photos)
            parts = "head and body textures" if person.body is not None else "head texture"
            return FaceTextures(dict(person.textures), f"{parts} from {used} of {len(photos)} photos",
                                list(person.warnings))
        except Exception as exc:  # noqa: BLE001 - the simple painting instead, saying why
            why = f"the head texture failed: {type(exc).__name__}: {exc}"
    return _painted(photos, model, why)


def _painted(photos, model, why: str) -> FaceTextures:
    """The face of the photo with the biggest face painted on ``model``'s head (:func:`face_texture`'s fallback)."""
    import cv2

    from .characters import face_photo
    from .recognition.engine import FaceEngine

    layout = face_photo.head_layout(model)
    if layout is None:
        raise ValueError(f"{model.name} has no face to put a photo on")
    engine = FaceEngine()
    best = None
    for path in photos:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        faces = engine.detect(image, min_size=0) if image is not None else []
        if faces and (best is None or faces[0].size > best[2].size):
            best = (path, image, faces[0])
    if best is None:
        raise ValueError("no face found in the photos")
    path, image, face = best
    painted = face_photo.paint_face(model, image, face.landmarks, face.box)
    return FaceTextures({model.primitives[layout.primitive].name: painted}, f"{path.name} painted on",
                        [f"the simple painting: {why}"])


class KnownFaces:
    """The known people the face row offers: those of the faces folder (a folder of photos per person, or a photo
    named after them; see :mod:`.recognition.gallery`), looked at again at most once a second. ``gallery``: a callable
    giving the face recognition's :class:`~.recognition.gallery.Gallery` when it is loaded (else ``None``): its photos
    with a face come first, the biggest face first."""

    def __init__(self, folder="faces", gallery=None):
        self.folder = Path(folder).expanduser()
        self._gallery = gallery
        self._people: dict[str, list[Path]] = {}
        self._checked = -math.inf

    def _scan(self) -> dict[str, list[Path]]:
        now = time.monotonic()
        if now - self._checked >= 1.0:
            from .recognition.gallery import _person, _photos

            self._checked = now
            found: dict[str, list[Path]] = {}
            try:
                for path in _photos(self.folder) if self.folder.is_dir() else []:
                    name = _person(path, self.folder)
                    if name is not None:
                        found.setdefault(name, []).append(path)
            except OSError:
                return self._people
            self._people = dict(sorted(found.items()))
        return self._people

    def people(self) -> list[str]:
        return list(self._scan())

    def photos(self, name: str) -> list[Path]:
        gallery = self._gallery() if callable(self._gallery) else self._gallery
        best = list(gallery.photos(name)) if gallery is not None else []
        return best + [p for p in self._scan().get(name, []) if p not in best]


# -- an avatar made ready -------------------------------------------------------------------------------------------


@dataclass
class Prepared:
    """An avatar made ready for the creator (on the worker): its model, its :class:`~.characters.outfit.Tailor`
    (``None`` when it cannot be dressed: ``why``), the garments it has, an idle clip for the turntable, its body
    primitive's name and whether it can wear items."""

    name: str
    model: object
    tailor: object | None = None
    why: str = ""
    garments: tuple = ()
    idle: object | None = None
    body: str | None = None
    items: bool = False

    def __repr__(self) -> str:
        return f"Prepared({self.name!r}, garments={self.garments}, items={self.items}, why={self.why!r})"


def prepare(library, name: str) -> Prepared:
    """The avatar ``name`` of ``library`` made ready (slow: its garment regions, the first time a few seconds)."""
    from .characters.crowd import sex
    from .characters.outfit import regions as R
    from .characters.outfit.tailor import tailor_for

    model = library.load(name)
    entry = library.available().get(name, {})
    index = R.body_primitive(model)
    ready = Prepared(name, model, body=model.primitives[index].name if index is not None else None)
    try:
        tailor = tailor_for(model)
        tailor.layers, tailor.surface  # noqa: B018 - made now, on this thread
        counts = tailor.regions.counts()
        ready.garments = tuple(g for g in GARMENTS if counts.get(g))
        ready.tailor = tailor if ready.garments else None
        ready.why = "" if ready.garments else f"{name} has no clothes to dress"
    except Exception as exc:  # noqa: BLE001 - it can still be saved, with its own clothes
        ready.why = f"{name} cannot be dressed ({type(exc).__name__}: {exc})"
    try:
        motions = library.motions()
        ready.idle = motions.role("idle", sex(entry)) if motions.has("idle") else None
    except Exception:  # noqa: BLE001 - it stands still instead
        ready.idle = None
    try:
        ready.items = accessories.supports(model)
        if ready.items:
            accessories.fit(model)
    except Exception:  # noqa: BLE001 - no items for it
        ready.items = False
    return ready


def _dress(tailor, outfit: Outfit, base):
    """The body texture of ``outfit`` over ``base`` (``None``: the avatar's own); ``base`` itself without fills."""
    if tailor is None or not outfit.fills:
        return base
    return tailor.texture(outfit, base=base)


def _read_textures(created, library) -> tuple[str | None, dict, dict]:
    """The name of the body primitive of a saved character's avatar, and the character's images and the bases of its
    looks (read from their files)."""
    from .characters.outfit import regions as R

    model = library.load(created.avatar)
    index = R.body_primitive(model)
    body = model.primitives[index].name if index is not None else None
    bases = getattr(created, "bases", {})
    return (body, {name: np.asarray(created.textures[name]) for name in list(created.textures)},
            {name: np.asarray(bases[name]) for name in list(bases)})


def _save(character, folder) -> tuple[Path, float]:
    """Save ``character`` in ``folder``: where, and when it was first saved."""
    from .characters.created import save

    return save(character, folder), character.created


def _short(text: str, most: int = 110) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= most else text[:most - 3] + "..."


@dataclass
class _FabricRun:
    job: object
    garment: str
    applied: object = None  # the garment's fill the run put on last (or had when it started)
    seen: int = 0  # messages printed so far
    picked: bool = False  # a variation was picked by hand (or the character saved): no more automatic choices
    done: bool = False


# -- the creator ----------------------------------------------------------------------------------------------------


class Creator:
    """The character creator's state and work (see the module docs).

    ``book``: the :class:`~.characters.created.CharacterBook` it saves into (``None``: it cannot save); ``library``:
    the prepared avatars (default the fair's); ``wardrobe``: the item looks; ``faces``: the known people (a
    :class:`KnownFaces`, or anything with ``people()`` and ``photos(name)``); ``fabrics``: the
    :class:`~.characters.outfit.FabricGenerator` (made when first needed, never downloading); ``painter``: what makes a
    face (:func:`face_texture`; tests pass stand-ins); ``threads``: do the slow work on threads (else at once, for
    tests). :attr:`person` is the scene character on the turntable (``None`` until the avatar is ready)."""

    def __init__(self, book=None, library=None, wardrobe: Wardrobe | None = None, faces=None, fabrics=None,
                 painter=None, threads: bool = True):
        self.book = book
        self._library = library
        self.wardrobe = wardrobe if wardrobe is not None else Wardrobe()
        self.faces = faces
        self._fabrics = fabrics
        self.painter = painter or face_texture
        self.threads = threads
        self.active = False
        # what is chosen
        self.avatar: str | None = None
        self.name = ""
        self.face: str | None = None
        self.fills: dict = {}  # garment -> Fill
        self.outfit_name = "custom"
        self.garment = "top"
        self.secondary: dict = {}  # garment -> the second colour chosen for it (for patterns)
        self.items: dict = {}  # item -> look name (or another look of a loaded character)
        self.fabric_text = ""
        self.variations: list = []  # the swatches of the last description, as they came
        self.variation: int | None = None
        # the screen
        self.row = 0
        self.cursor = {row: 0 for row in ENTER_ROWS}
        self.editing: str | None = None
        self._before_typing = ""
        self._heading, self._since, self.turning = 0.0, time.perf_counter(), True
        # the work
        self.prepared: Prepared | None = None
        self.person = None
        self._jobs: dict[str, tuple] = {}  # kind -> (what it is made for, Future)
        self._failed: dict[str, tuple] = {}  # kind -> what failed for (not tried again until that changes)
        self._outfit_version = 0
        self._face_result: tuple | None = None  # ((avatar, face), FaceTextures)
        self._body_result: tuple | None = None  # (body key, RGBA or None)
        self._kept: tuple | None = None  # (avatar, {primitive: RGBA}): a loaded character's other textures
        self._textures: dict = {}  # primitive -> (RGBA, Texture) on the person
        self._fabric_run: _FabricRun | None = None
        self._save_wanted: str | None = None
        self._confirm: tuple | None = None  # (what, name, until)
        self._loaded: str | None = None  # the name of the saved character being edited
        self._created_at = 0.0  # when it was first saved
        self._other_items: dict = {}  # a loaded character's items the creator does not offer (a t-shirt)
        self._changes = self._saved_changes = 0
        self._avatars: tuple[float, list[str]] = (-math.inf, [])
        self._unsaid: list[str] = []  # messages wait() picked up, for the next update()
        self.last_fabric_note = ""

    # -- the parts it uses --------------------------------------------------------------------------------------------

    @property
    def library(self):
        if self._library is None:
            from .scenes.fair import shared_library

            self._library = shared_library()
        return self._library

    @property
    def fabrics(self):
        if self._fabrics is None:
            from .characters.outfit import FabricGenerator

            self._fabrics = FabricGenerator(download=False)
        return self._fabrics

    def avatars(self) -> list[str]:
        """The prepared avatars (looked at again every few seconds)."""
        now = time.monotonic()
        if now - self._avatars[0] >= AVATARS_EVERY:
            try:
                names = sorted(self.library.available())
            except Exception:  # noqa: BLE001 - no library: no avatars
                names = []
            self._avatars = (now, names)
        return self._avatars[1]

    def people(self) -> list[str]:
        if self.faces is None:
            return []
        try:
            return list(self.faces.people())
        except Exception:  # noqa: BLE001 - a faces folder that cannot be read: nobody
            return []

    def saved(self) -> list[str]:
        return [c.name for c in self.book.characters()] if self.book is not None else []

    # -- opening and closing ------------------------------------------------------------------------------------------

    def open(self, person=None, face: str | None = None) -> str:
        """Show the creator; with ``person`` (a scene character) start from them: a created character is loaded to edit
        it, anyone else becomes a new character of their avatar, outfit and items (and ``face``, whose face they
        wear)."""
        self.active = True
        if person is not None:
            created = getattr(person, "created", None)
            if created is not None and self.book is not None and self.book.get(created.name) is not None:
                return self.load(created.name)
            return self.start_from(person, face)
        if self.avatar is None:
            avatars = self.avatars()
            if not avatars:
                return "character creator: no avatars are prepared (bifold-characters fetch)"
            self.set_avatar(avatars[0])
        return "character creator: Up / Down rows, Left / Right choose, Enter types or does, Esc closes"

    def leave(self) -> str:
        self.active = False
        if self.editing is not None:
            self._end_typing(cancel=True)
        return "character creator closed"

    def close(self) -> None:
        """Stop the work going on (for quitting)."""
        if self._fabric_run is not None:
            self._fabric_run.job.cancel()

    # -- every frame ----------------------------------------------------------------------------------------------------

    def update(self) -> list[str]:
        """Pick up the work done since the last call and start what is due; returns messages for the user."""
        messages, self._unsaid = self._unsaid + self._collect(), []
        messages += self._poll_fabric()
        messages += self._schedule()
        if self._confirm is not None and time.monotonic() > self._confirm[2]:
            self._confirm = None
        return messages

    def busy(self) -> list[str]:
        """What is going on, for the status line."""
        out = []
        if self.avatar is not None and (self.prepared is None or self.prepared.name != self.avatar):
            out.append(f"preparing {self.avatar} (the first time a few seconds)")
        for kind, (key, _future) in self._jobs.items():
            if kind == "face":
                out.append(f"making {key[1]}'s face")
            elif kind == "dress":
                out.append("dressing")
            elif kind == "save":
                out.append(f"saving {key}")
            elif kind == "load":
                out.append(f"reading {key}")
        if self._save_wanted and "save" not in self._jobs:
            out.append(f"saving {self._save_wanted} when the work above is done")
        run = self._fabric_run
        if run is not None and not run.done:
            out.append(f"making the fabric: {self.fabric_status()}")
        return out

    def fabric_status(self) -> str:
        """How the fabric of the last description is going (or went)."""
        run = self._fabric_run
        if run is None:
            return ""
        job = run.job
        progress = dict(job.progress)
        if not job.done and progress.get("steps"):
            return f"{progress.get('model')}, seed {progress.get('seed')}: step {progress.get('step')}/" \
                   f"{progress.get('steps')}"
        return job.messages[-1] if job.messages else ("starting" if not job.done else job.status)

    def _start(self, kind: str, key, work, *args) -> None:
        """Run ``work(*args)`` for ``key`` (what it is made for) on a thread of its own (at once without threads).
        Quitting does not wait for it, but for a save (its files are written whole or not at all)."""
        future: Future = Future()

        def run() -> None:
            try:
                future.set_result(work(*args))
            except BaseException as exc:  # noqa: BLE001 - kept on the future, said by update()
                future.set_exception(exc)

        if self.threads:
            threading.Thread(target=run, name=f"creator: {kind}", daemon=kind != "save").start()
        else:
            run()
        self._jobs[kind] = (key, future)

    def wait(self, timeout: float = 60.0) -> bool:
        """Wait (in tests and scripts) until the work and the fabric are done and picked up; True if they are."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            self._unsaid = self.update()
            run = self._fabric_run
            if not self._jobs and not self._due() and (run is None or run.done):
                return True
            time.sleep(0.01)
        return False

    def _due(self) -> bool:
        """Whether work is still to start or its result to be picked up: the avatar, the face, the body texture, a
        save (not what failed: that waits for another choice)."""
        if self.avatar is None:
            return False
        if not self._ready_for(self.avatar):
            return self._failed.get("avatar") != self.avatar
        return self._stale() or self._save_wanted is not None

    def _stale(self) -> bool:
        """Whether the face or the body texture of the current choices is still to be made."""
        face = self._face_key()
        if face is not None and (self._face_result is None or self._face_result[0] != face) \
                and self._failed.get("face") != face:
            return True
        body = self._body_key()
        return (self._body_result is None or self._body_result[0] != body) and self._failed.get("dress") != body

    def _schedule(self) -> list[str]:
        if self.avatar is None:
            return []
        p = self.prepared
        if p is None or p.name != self.avatar:
            if "avatar" not in self._jobs and self._failed.get("avatar") != self.avatar:
                self._start("avatar", self.avatar, prepare, self.library, self.avatar)
            return []
        face = self._face_key()
        if face is not None and "face" not in self._jobs and self._failed.get("face") != face \
                and (self._face_result is None or self._face_result[0] != face):
            photos = self.faces.photos(self.face) if self.faces is not None else []
            self._start("face", face, self.painter, photos, p.model)
        body = self._body_key()
        if "dress" not in self._jobs and self._failed.get("dress") != body \
                and (self._body_result is None or self._body_result[0] != body):
            outfit = self.outfit() if p.tailor is not None else Outfit({})
            self._start("dress", body, _dress, p.tailor, outfit, self._base())
        if self._save_wanted is not None and "save" not in self._jobs and not self._pending():
            return self._save_now()
        return []

    def _pending(self) -> bool:
        """Whether what saving takes is not made yet (the avatar, the face, the body texture)."""
        return any(kind in self._jobs for kind in ("avatar", "face", "dress", "load")) \
            or not self._ready_for(self.avatar) or self._stale()

    def _collect(self) -> list[str]:
        messages = []
        for kind, (key, future) in list(self._jobs.items()):
            if not future.done():
                continue
            del self._jobs[kind]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001 - said, and not tried again until the choice changes
                messages += self._job_failed(kind, key, exc)
                continue
            try:
                messages += self._job_done(kind, key, result)
            except Exception as exc:  # noqa: BLE001 - a result it cannot use never stops the demo
                traceback.print_exc()
                messages += self._job_failed(kind, key, exc)
        return messages

    def _job_done(self, kind: str, key, result) -> list[str]:
        self._failed.pop(kind, None)
        if kind == "avatar":
            if key == self.avatar:
                self.prepared = result
                self._new_person()
            return []
        if kind == "face":
            if key == self._face_key():
                self._face_result = (key, result)
                self._refresh_person()
                notes = f" ({'; '.join(result.notes)})" if result.notes else ""
                print(f"[creator] {key[1]}'s face on {key[0]}: {result.method}{notes}")
                return [f"{key[1]}'s face is on ({result.method})"]
            return []
        if kind == "dress":
            wanted, current = self._body_key(), self._body_result
            if key[0] == self.avatar and (key == wanted or current is None or current[0] != wanted):
                self._body_result = (key, result)  # (an older outfit is still nearer than the avatar's own clothes)
                self._refresh_person()
            return []
        if kind == "load":
            return self._loaded_textures(key, result)
        if kind == "save":
            path, self._created_at = result
            if self.book is not None:
                self.book.refresh(force=True)
            self._loaded, self._saved_changes = key, self._save_token
            return [f"saved {key} in {path}; {key} joins the fair in a moment"]
        return []

    def _job_failed(self, kind: str, key, exc: Exception) -> list[str]:
        why = f"{type(exc).__name__}: {exc}"
        if kind == "avatar":
            self._failed["avatar"] = key
            return [f"cannot prepare {key}: {why}"]
        if kind == "face":
            if key == self._face_key():
                self._failed["face"] = key
                self.face = None
                self._changes += 1
            return [f"cannot make {key[1]}'s face for {key[0]}: {why}"]
        if kind == "dress":
            self._failed["dress"] = key
            return [f"cannot dress {key[0]}: {why}"]
        if kind == "load":
            return [f"cannot read {key[0]}'s textures: {why} (the outfit is made again)"]
        return [f"cannot save {key}: {why}"]

    # -- the person on the turntable ------------------------------------------------------------------------------------

    def heading_at(self, now: float | None = None) -> float:
        """The turntable's angle (degrees; 0: the avatar faces the viewer)."""
        if not self.turning:
            return self._heading
        now = time.perf_counter() if now is None else now
        return (self._heading + TURN_SPEED * (now - self._since)) % 360.0

    def turn(self, degrees: float | None = None, to: float | None = None, turning: bool = False) -> None:
        """Turn the avatar by ``degrees`` or ``to`` an angle; ``turning``: it goes on turning from there."""
        heading = self.heading_at() if to is None else to
        self._heading = (heading + (degrees or 0.0)) % 360.0
        self._since, self.turning = time.perf_counter(), turning

    def _pose(self, t: float):
        from .characters.model import CharacterState, Layer

        p = self.prepared
        layers = [Layer(p.idle, t)] if p is not None and p.idle is not None else []
        return CharacterState((0.0, 0.0, 0.0), self.heading_at(), layers)

    def _new_person(self) -> None:
        from .characters.character import Character

        self.person = Character(self.prepared.model, behaviour=self._pose, name="creator", shadow=False)
        self._textures = {}
        self._refresh_person()
        for item in ITEMS:
            self._wear(item)

    def _refresh_person(self) -> None:
        """Put the textures made so far on the person."""
        from .scene.textures import Texture

        if self.person is None:
            return
        wanted = self._current_textures(exact=False)
        overrides = {}
        for name, rgba in wanted.items():
            known = self._textures.get(name)
            if known is None:
                known = (rgba, Texture.from_array(np.asarray(rgba), order="rgb", name=f"creator {name}"))
            elif known[0] is not rgba:  # new pixels in the same texture: the renderer writes them over the old
                known[1].update(np.asarray(rgba), order="rgb")
                known = (rgba, known[1])
            self._textures[name] = known
            overrides[name] = known[1]
        self._textures = {name: self._textures[name] for name in overrides}
        self.person.texture_overrides = overrides

    def _current_textures(self, exact: bool = True) -> dict:
        """The textures by primitive name: kept from a loaded character, the face's, the body's. ``exact``: only those
        made for the current choices (else an older body texture of the same avatar stands in while dressing)."""
        p = self.prepared
        if p is None or p.name != self.avatar:
            return {}
        out = dict(self._kept[1]) if self._kept is not None and self._kept[0] == self.avatar else {}
        face = self._face()
        if face is not None:
            out.update({name: rgba for name, rgba in face.textures.items() if name != p.body})
        body = self._body_result
        if body is not None and body[0][0] == self.avatar and (not exact or body[0] == self._body_key()):
            if body[1] is not None:
                out[p.body] = body[1]
        elif face is not None and p.body in face.textures and not (self.fills and p.tailor is not None):
            out[p.body] = face.textures[p.body]
        return out

    def _face_key(self):
        return (self.avatar, self.face) if self.face else None

    def _face(self) -> FaceTextures | None:
        """The face made for the current avatar and face, if it is made."""
        key = self._face_key()
        result = self._face_result
        return result[1] if key is not None and result is not None and result[0] == key else None

    def _base(self):
        """The body texture the outfit goes on: the face's (its skin matched), else ``None`` (the avatar's own)."""
        face, p = self._face(), self.prepared
        return face.textures.get(p.body) if face is not None and p is not None and p.body else None

    def _body_key(self) -> tuple:
        p = self.prepared
        base = self._base()
        dressed = p is not None and p.tailor is not None and bool(self.fills)
        return self.avatar, self._outfit_version if dressed else None, id(base) if base is not None else None

    # -- choosing -------------------------------------------------------------------------------------------------------

    def set_avatar(self, name: str) -> str:
        if name == self.avatar:
            return f"avatar: {name}"
        self.avatar = name
        self._failed.pop("avatar", None)
        self._changes += 1
        return f"avatar: {name}" + ("" if self._ready_for(name) else " (preparing it ...)")

    def _ready_for(self, name: str) -> bool:
        return self.prepared is not None and self.prepared.name == name

    def set_name(self, text: str) -> str:
        self.name = str(text).strip()[:MAX_TEXT["name"]]
        self._changes += 1
        return f"name: {self.name}" if self.name else "no name yet"

    def set_face(self, name: str | None) -> str:
        if name == self.face:
            return f"face: {name or 'their own'}"
        self.face = name
        self._failed.pop("face", None)
        self._changes += 1
        if name is None:
            self._refresh_person()
            return "face: the avatar's own"
        return f"making {name}'s face (a few seconds) ..."

    def outfit(self) -> Outfit:
        return Outfit(dict(self.fills), name=self.outfit_name)

    def apply_preset(self, name: str) -> str:
        from .characters.outfit.tailor import preset

        outfit = preset(name)
        self.fills = {garment: fill for garment, fill in outfit.fills.items() if garment in GARMENTS}
        self.secondary = {g: f.secondary for g, f in self.fills.items() if isinstance(f, Pattern)}
        self.outfit_name = outfit.name
        self._outfit_changed()
        return f"outfit: {outfit.name}"

    def _outfit_changed(self, custom: bool = False) -> None:
        if custom:
            self.outfit_name = "custom"
        self._outfit_version += 1
        self._changes += 1
        self._failed.pop("dress", None)

    def select_garment(self, garment: str) -> str:
        if garment not in GARMENTS:
            raise KeyError(f"no garment {garment!r}; garments: {', '.join(GARMENTS)}")
        self.garment = garment
        p = self.prepared
        if p is not None and p.name == self.avatar and garment not in p.garments:
            return f"{self.avatar} has no {garment}" + (f" ({p.why})" if p.why else "")
        return f"garment: {garment}"

    def _set_fill(self, fill, garment: str | None = None) -> None:
        garment = garment or self.garment
        if fill is None:
            self.fills.pop(garment, None)
        else:
            self.fills[garment] = fill
        self._outfit_changed(custom=True)

    def set_pattern(self, kind: str) -> str:
        """``"own"`` (the avatar's own clothes), ``"plain"``, a pattern of :data:`PATTERNS` or ``"fabric"`` (the variation
        chosen) for the garment chosen."""
        fill, garment = self.fills.get(self.garment), self.garment
        if kind == "own":
            self._set_fill(None)
            return f"{garment}: the avatar's own"
        if kind == "fabric":
            if not self.variations:
                return "describe a fabric first (the Fabric row: Enter, type, Enter)"
            return self.pick_variation(self.variation if self.variation is not None else 0)
        primary = primary_of(fill)
        if kind == "plain":
            self._set_fill(Solid(primary if primary is not None else color("navy")))
        elif kind in PATTERNS:
            scale = fill.scale if isinstance(fill, Pattern) and fill.kind == kind else None
            self._set_fill(Pattern(kind, primary, self.secondary.get(garment), scale))
        else:
            raise KeyError(f"no pattern {kind!r}; patterns: {', '.join(PATTERN_CHOICES)}")
        return f"{garment}: {describe_fill(self.fills[garment])}"

    def set_colour(self, name, second: bool = False) -> str:
        """The garment's colour (``second``: its second colour, for patterns)."""
        rgb = color(name)
        fill, garment = self.fills.get(self.garment), self.garment
        if second:
            self.secondary[garment] = rgb
            if not isinstance(fill, Pattern):
                self._changes += 1
                return f"second colour {name} (for a pattern on the {garment})"
            self._set_fill(Pattern(fill.kind, fill.primary, rgb, fill.scale, fill.rotation))
        elif isinstance(fill, Pattern):
            self._set_fill(Pattern(fill.kind, rgb, self.secondary.get(garment, fill.secondary), fill.scale,
                                   fill.rotation))
        else:
            self._set_fill(Solid(rgb))
        return f"{garment}: {describe_fill(self.fills[garment])}"

    def scale_by(self, factor: float) -> str:
        """The pattern's or fabric's repeat on the garment chosen, ``factor`` times as large."""
        fill, garment = self.fills.get(self.garment), self.garment
        if not isinstance(fill, Swatch):
            return f"the {garment} has no pattern to scale"
        scale = float(np.clip(fill.scale * factor, *SCALE_RANGE))
        if isinstance(fill, Pattern):
            new = Pattern(fill.kind, fill.primary, fill.secondary, scale, fill.rotation)
        elif isinstance(fill, Fabric):
            new = Fabric(fill.image, fill.text, scale, fill.rotation, fill.model, fill.seed, fill.method, fill.seam,
                         fill.prompt)
        else:
            new = Swatch(fill.image, scale, fill.rotation, fill.seamless)
        self._set_fill(new)
        return f"{garment}: {scale:.3g} m per repeat"

    def describe_fabric(self, text: str) -> str:
        """Start making fabrics of the description ``text`` for the garment chosen (the variations come as they are
        made, the keyword preview first, which goes on at once; the best goes on when the job ends unless one was
        picked)."""
        from .characters.outfit.fabric import PromptRefused

        self.fabric_text = str(text).strip()[:MAX_TEXT["fabric"]]
        if not self.fabric_text:
            return "describe a fabric, e.g. blue paisley silk"
        try:
            job = self.fabrics.start(self.fabric_text)
        except PromptRefused as exc:
            return exc.message
        except ValueError as exc:
            return str(exc)
        if self._fabric_run is not None:
            self._fabric_run.job.cancel()
        self._fabric_run = _FabricRun(job, self.garment, self.fills.get(self.garment))
        self.variations, self.variation = [], None
        self.last_fabric_note = ""
        return f"making a fabric of {job.text!r} for the {self.garment} ..."

    def pick_variation(self, index: int) -> str:
        if not 0 <= index < len(self.variations):
            return "no such variation"
        if self._fabric_run is not None:
            self._fabric_run.picked = True
        self._use_variation(index, self.garment)
        return f"{self.garment}: {self.variations[index].describe()}"

    def _use_variation(self, index: int, garment: str):
        self.variation = index
        fill = self.variations[index].fill()
        self._set_fill(fill, garment)
        return fill

    def _automatic(self, run: _FabricRun, index: int) -> None:
        """Put variation ``index`` on the run's garment, unless something else was chosen there meanwhile."""
        if not run.picked and self.fills.get(run.garment) is run.applied:
            run.applied = self._use_variation(index, run.garment)

    def _poll_fabric(self) -> list[str]:
        run = self._fabric_run
        if run is None or run.done:
            return []
        job = run.job
        new = job.poll()
        finished = job.done  # (before the variations are read: all of them are there then)
        self.variations = list(job.results)
        for message in job.messages[run.seen:]:
            print(f"[creator] fabric {job.text!r}: {message}")
        run.seen = len(job.messages)
        if new and self.variation is None:
            self._automatic(run, 0)  # the keyword preview, at once
        if not finished:
            return []
        run.done = True
        best = job.best
        if best is not None:
            self._automatic(run, next(k for k, s in enumerate(self.variations) if s is best))
        made = [s for s in job.variations if s.generated]
        if made:
            note = f"fabric {job.text!r}: {len(made)} variation{'s' if len(made) > 1 else ''} to choose from"
        elif best is None:
            note = f"no fabric of {job.text!r}: {job.messages[-1] if job.messages else job.status}"
        else:
            reasons = [m for m in job.messages if not m.startswith(("procedural", "plain colour", "loading"))]
            note = f"fabric {job.text!r}: {best.describe()}" + (f" (no model: {_short(reasons[-1], 80)})"
                                                               if reasons else "")
        self.last_fabric_note = note
        return [note]

    def set_item(self, item: str, look) -> str:
        """Put ``item`` on in ``look`` (a look name of the wardrobe; ``None``: off)."""
        if item not in ITEMS:
            raise KeyError(f"no item {item!r}; items: {', '.join(ITEMS)}")
        if look is None:
            self.items.pop(item, None)
        else:
            self.items[item] = look
        self._changes += 1
        problem = self._wear(item)
        if problem:
            return problem
        return f"{item}: {look}" if look is not None else f"{item}: none"

    def _wear(self, item: str) -> str | None:
        """Put the item chosen on the person (or take it off); says why it cannot be."""
        p, person = self.prepared, self.person
        look = self.items.get(item)
        if person is None or p is None:
            return None
        if not p.items:
            return f"{p.name} cannot wear items" if look is not None else None
        try:
            spec = self._item_spec(item, look)
            accessories.dress(person, item, spec, self.wardrobe, t=-math.inf)
        except Exception as exc:  # noqa: BLE001 - a look that cannot be worn (a missing file): said, the rest stays
            return f"cannot put the {item} on: {exc}"
        return None

    def _item_spec(self, item: str, look):
        if look is None:
            return None
        if self._loaded is not None and self.book is not None:
            from .characters.created import item_look

            created = self.book.get(self._loaded)
            if created is not None:
                return item_look(created, look)
        return look

    # -- saving, loading ------------------------------------------------------------------------------------------------

    def character(self):
        """The :class:`~.characters.created.CreatedCharacter` saving writes (with the textures made so far)."""
        from .characters.created import CreatedCharacter

        items = {**self._other_items, **{k: v for k, v in self.items.items() if v is not None}}
        outfit = self.outfit().spec() if self.fills else {}
        base, p = self._base(), self.prepared
        bases = {p.body: base} if base is not None else {}  # the face's skin on the body: the outfit goes on it
        return CreatedCharacter(self.name.strip(), self.avatar or "", self.face, outfit, items,
                                self._current_textures(), created=self._created_at, bases=bases)

    def save(self) -> str:
        """Save the character as soon as its textures are made (on the worker); :meth:`update` says when it is."""
        from .characters.created import slug

        if self.book is None:
            return "characters cannot be saved: the demo has no folder for them (--created)"
        name = self.name.strip()
        if not name:
            return "type a name first (the Name row: Enter, type, Enter)"
        try:
            slug(name)
        except ValueError as exc:
            return str(exc)
        if self.avatar is None:
            return "choose an avatar first"
        if self._failed.get("avatar") == self.avatar and not self._ready_for(self.avatar):
            return f"{self.avatar} cannot be used: choose another avatar"
        other = self.book.get(name)
        if other is not None and name != self._loaded and not self._confirmed("save", name):
            return f"{other.name} is saved already: save again to replace it"
        self._save_wanted = name
        if self._fabric_run is not None:  # what is saved stays on
            self._fabric_run.picked = True
        return f"saving {name} ..."

    def _confirmed(self, what: str, name: str) -> bool:
        """Whether this is the second press of ``what`` for ``name`` within :data:`CONFIRM` seconds (else it asks)."""
        now = time.monotonic()
        if self._confirm is not None and self._confirm[:2] == (what, name) and now <= self._confirm[2]:
            self._confirm = None
            return True
        self._confirm = (what, name, now + CONFIRM)
        return False

    def _save_now(self) -> list[str]:
        name, self._save_wanted = self._save_wanted, None
        if name != self.name.strip():
            return []  # renamed meanwhile: saved when asked again
        self._save_token = self._changes
        self._start("save", name, _save, self.character(), self.book.folder)
        return []

    _save_token = -1

    def delete(self) -> str:
        """Delete the saved character being edited (or the one of that name), when asked twice."""
        name = self._loaded or self.name.strip()
        if self.book is None or not name or self.book.get(name) is None:
            return "nothing saved under that name to delete"
        if not self._confirmed("delete", name):
            return f"delete {name}? Press delete again (it leaves the fair)"
        if not self.book.delete(name):
            return f"cannot delete {name}"
        self._loaded = None
        self._created_at = 0.0
        self._saved_changes = -1
        return f"deleted {name}; it leaves the fair at the next moment nobody sees"

    def new(self) -> str:
        """Start a new character on the same avatar."""
        self.name, self.face, self.fills, self.items, self.secondary = "", None, {}, {}, {}
        self.outfit_name, self._loaded, self._created_at, self._other_items, self._kept = "custom", None, 0.0, {}, None
        self.variations, self.variation, self.fabric_text = [], None, ""
        self._outfit_changed()
        self._saved_changes = -1
        self._new_person_if_ready()
        return "a new character"

    def _new_person_if_ready(self) -> None:
        if self.prepared is not None and self.prepared.name == self.avatar:
            self._new_person()

    def start_from(self, person, face: str | None = None) -> str:
        """A new character from a scene ``person``: their avatar, outfit and items (``face``: whose face they wear)."""
        from .characters.outfit import outfit_of

        self.new()
        avatar = person.skinned_model.name
        if avatar not in self.avatars():
            return f"{avatar} is not a prepared avatar"
        self.set_avatar(avatar)
        outfit = outfit_of(person)
        if outfit is not None:
            self.fills = {g: f for g, f in outfit.fills.items() if g in GARMENTS}
            self.outfit_name = outfit.name
            self._outfit_changed()
        for item, worn in accessories.wearing(person).items():
            name = self._look_name(item, worn.look)
            if item in ITEMS and name is not None:
                self.items[item] = name
        self.face = face if face in self.people() else None
        self._new_person_if_ready()
        return f"a new character from {getattr(person, 'name', avatar)} ({avatar}): type a name, then save"

    def _look_name(self, item: str, look) -> str | None:
        for name, spec in self.wardrobe.looks(item):
            if spec is look or (isinstance(spec, (str, Path)) and isinstance(look, (str, Path))
                                and str(spec) == str(look)):
                return name
        return look if isinstance(look, str) else None

    def load(self, name: str) -> str:
        """Edit the saved character ``name``: everything it was saved with (its saved textures stand until something
        changes)."""
        if self.book is None:
            return "no saved characters here"
        self.book.refresh(force=True)
        created = self.book.get(name)
        if created is None:
            return f"no saved character called {name!r}"
        try:
            outfit = Outfit.from_spec(created.outfit) if created.outfit else Outfit({})
        except (ValueError, KeyError, TypeError, OSError) as exc:
            return f"cannot read {name}'s outfit: {exc}"
        self.new()
        self.set_avatar(created.avatar)
        self.name, self.face = created.name, created.face
        self.fills = {g: f for g, f in outfit.fills.items() if g in GARMENTS}
        self.outfit_name = outfit.name
        self.secondary = {g: f.secondary for g, f in self.fills.items() if isinstance(f, Pattern)}
        self.fabric_text = next((f.text for f in self.fills.values() if isinstance(f, Fabric)), "")
        self.items = {k: v for k, v in created.items.items() if k in ITEMS and v is not None and v is not False}
        self._other_items = {k: v for k, v in created.items.items() if k not in ITEMS}
        self._loaded, self._created_at = created.name, created.created
        self._outfit_changed()
        self._start("load", (created.name, self.avatar, self.face, self._outfit_version), _read_textures, created,
                    self.library)
        self._new_person_if_ready()
        self._saved_changes = self._changes
        return f"editing {created.name} ({created.describe()})"

    def _loaded_textures(self, key, result) -> list[str]:
        """A loaded character's textures: the body's stands for its outfit, the rest for its face (or as they are)."""
        name, avatar, face, version = key
        if (avatar, version) != (self.avatar, self._outfit_version) or name != self._loaded:
            return []  # something changed meanwhile: made again
        body_name, textures, bases = result
        body = textures.pop(body_name, None)
        base = bases.get(body_name) if face and face == self.face else None  # the face's body: the outfit's base
        if face and face == self.face and textures:
            self._face_result = ((avatar, face), FaceTextures({**textures, **bases}, "as saved"))
        elif textures:
            self._kept = (avatar, textures)
        if body is not None:
            p = self.prepared
            dressed = p is None or p.name != avatar or p.tailor is not None
            self._body_result = ((avatar, version if dressed and self.fills else None,
                                  id(base) if base is not None else None), body)
        self._refresh_person()
        return []

    # -- the rows (keys and clicks) -------------------------------------------------------------------------------------

    def choices(self, row: str) -> list[str]:
        """The labels of the choices of ``row``."""
        if row == "avatar":
            return self.avatars()
        if row == "face":
            return ["none", *self.people()]
        if row == "preset":
            from .characters.outfit.tailor import presets

            return list(presets())
        if row == "garment":
            return list(GARMENTS)
        if row == "pattern":
            return list(PATTERN_CHOICES)
        if row in ("colour", "colour 2"):
            return list(COLOURS)
        if row == "variation":
            return [s.describe() for s in self.variations]
        if row in ITEMS:
            return ["none", *[name for name, _ in self.wardrobe.looks(row)]]
        if row == "save":
            return list(ACTIONS)
        if row == "saved":
            return self.saved()
        return []

    def current(self, row: str) -> int | None:
        """The index of the choice of ``row`` in use (``None``: none of them)."""
        fill = self.fills.get(self.garment)
        if row == "avatar":
            return _index(self.avatars(), self.avatar)
        if row == "face":
            return 0 if self.face is None else _index(self.choices("face"), self.face)
        if row == "preset":
            return _index(self.choices("preset"), self.outfit_name)
        if row == "garment":
            return GARMENTS.index(self.garment)
        if row == "pattern":
            return PATTERN_CHOICES.index(fill_kind(fill))
        if row in ("colour", "colour 2"):
            rgb = primary_of(fill) if row == "colour" else secondary_of(fill)
            if rgb is None and row == "colour 2":
                rgb = self.secondary.get(self.garment)
            return next((k for k, c in enumerate(COLOURS) if rgb is not None and tuple(color(c)) == tuple(rgb)), None)
        if row == "variation":
            return self.variation
        if row in ITEMS:
            return 0 if self.items.get(row) is None else _index(self.choices(row), self.items[row])
        if row in ENTER_ROWS:
            return None
        return None

    def enabled(self, row: str, index: int) -> bool:
        """Whether the choice can be taken (a garment the avatar has, items on an avatar that wears them)."""
        p = self.prepared if self.prepared is not None and self.prepared.name == self.avatar else None
        if p is None:
            return True
        if row == "garment":
            return GARMENTS[index] in p.garments
        if row in ITEMS:
            return p.items or index == 0
        if row in ("pattern", "colour", "colour 2", "preset", "variation"):
            return p.tailor is not None
        return True

    def value(self, row: str) -> str:
        """What ``row`` is set to, in words (beside its chips)."""
        fill = self.fills.get(self.garment)
        if row == "avatar":
            p = self.prepared
            if p is not None and p.name == self.avatar and p.why:
                return p.why
            return self.avatar or "no avatar"
        if row == "face":
            face = self._face()
            return f"{self.face} ({face.method})" if face is not None else (self.face or "the avatar's own")
        if row == "preset":
            return self.outfit_name
        if row == "garment":
            return self.garment
        if row == "pattern":
            return describe_fill(fill)
        if row == "colour":
            rgb = primary_of(fill)
            return _colour_name(rgb) if rgb is not None else "-"
        if row == "colour 2":
            rgb = secondary_of(fill) or self.secondary.get(self.garment)
            return _colour_name(rgb) if rgb is not None else "-"
        if row == "variation":
            return self.variations[self.variation].describe() if self.variation is not None and \
                self.variation < len(self.variations) else ""
        if row == "scale":
            return f"{fill.scale:.3g} m per repeat" if isinstance(fill, Swatch) else "-"
        if row in ITEMS:
            return str(self.items.get(row) or "none")
        if row == "save":
            return self.state()
        if row == "saved":
            saved = self.book.characters() if self.book is not None else []
            if not saved:
                return "none saved yet"
            marked = saved[self.cursor["saved"] % len(saved)]
            return f"{marked.name}: {marked.describe()} (Enter: edit it)"
        return ""

    def state(self) -> str:
        """Saved or not, in a few words."""
        if self._loaded is None:
            return "not saved yet"
        return f"{self._loaded}: saved" if self._saved_changes == self._changes else f"{self._loaded}: changed"

    def pick(self, row: str, index: int) -> str | None:
        """Take choice ``index`` of ``row`` (as going to it with Left / Right does; in the save row and the saved
        characters it is only marked, Enter does it)."""
        choices = self.choices(row)
        if not 0 <= index < len(choices):
            return None
        if row in ENTER_ROWS:
            self.cursor[row] = index
            return None
        if not self.enabled(row, index):
            p = self.prepared
            return f"{p.name} has no {choices[index]}" if row == "garment" else \
                (f"{p.name} cannot wear items" if row in ITEMS else p.why or "not for this avatar")
        if row == "avatar":
            return self.set_avatar(choices[index])
        if row == "face":
            return self.set_face(None if index == 0 else choices[index])
        if row == "preset":
            return self.apply_preset(choices[index])
        if row == "garment":
            return self.select_garment(choices[index])
        if row == "pattern":
            return self.set_pattern(choices[index])
        if row in ("colour", "colour 2"):
            return self.set_colour(choices[index], second=row == "colour 2")
        if row == "variation":
            return self.pick_variation(index)
        if row in ITEMS:
            return self.set_item(row, None if index == 0 else choices[index])
        return None

    def step(self, row: str, direction: int) -> str | None:
        """Left (``-1``) or Right (``1``) in ``row``."""
        if row == "scale":
            return self.scale_by(SCALE_STEP if direction > 0 else 1.0 / SCALE_STEP)
        choices = self.choices(row)
        if row in TEXT_ROWS or not choices:
            return None
        n = len(choices)
        if row in ENTER_ROWS:
            self.cursor[row] = (self.cursor[row] + direction) % n
            return None
        current = self.current(row)
        index = current if current is not None else (-1 if direction > 0 else n)
        for _ in range(n):
            index = (index + direction) % n
            if self.enabled(row, index):
                return self.pick(row, index)
        return self.pick(row, (index + direction) % n)  # (says why not)

    def activate(self, row: str) -> str | None:
        """Enter on ``row``: type in a text row, do the action marked in the save row, load the character marked."""
        if row in TEXT_ROWS:
            self._start_typing(row)
            return None
        if row == "save":
            action = ACTIONS[self.cursor["save"] % len(ACTIONS)]
            return {"save": self.save, "delete": self.delete, "new": self.new}[action]()
        if row == "saved":
            saved = self.saved()
            return self.load(saved[self.cursor["saved"] % len(saved)]) if saved else "no saved characters yet"
        if row == "fabric":
            return None
        return None

    def click(self, row: str, index: int | None = None) -> str | None:
        """A click on choice ``index`` of ``row`` (or on its text box)."""
        try:
            return self._click(row, index)
        except Exception as exc:  # noqa: BLE001 - a choice that goes wrong never stops the demo
            traceback.print_exc()
            return f"cannot do that: {type(exc).__name__}: {exc}"

    def _click(self, row: str, index: int | None) -> str | None:
        self.row = ROWS.index(row)
        if self.editing is not None and self.editing != row:
            self._end_typing()
        if row in TEXT_ROWS:
            self._start_typing(row)
            return None
        if index is None:
            return None
        if row in ENTER_ROWS:
            self.cursor[row] = index
            return self.activate(row)
        return self.pick(row, index)

    def text(self, row: str) -> str:
        return self.name if row == "name" else self.fabric_text if row == "fabric" else ""

    def _start_typing(self, row: str) -> None:
        self.editing, self._before_typing = row, self.text(row)

    def _set_text(self, row: str, text: str) -> None:
        if row == "name":
            self.name = text
        else:
            self.fabric_text = text

    def _end_typing(self, cancel: bool = False) -> str | None:
        row, self.editing = self.editing, None
        if cancel:
            self._set_text(row, self._before_typing)
            return None
        if row == "name":
            if self.name != self._before_typing:
                return self.set_name(self.name)
            return None
        return self.describe_fabric(self.fabric_text) if self.fabric_text.strip() else None

    def handle_key(self, key: int, typed: str | None = None) -> tuple[bool, str | None]:
        """``(handled, message)`` for a key pressed while the creator is shown (it takes every key but ``P``, the
        screenshot); ``typed``: the character the key typed, with its case (see :class:`~.display.window.Window`)."""
        if not self.active:
            return False, None
        try:
            return self._key(key, typed)
        except Exception as exc:  # noqa: BLE001 - a choice that goes wrong never stops the demo
            traceback.print_exc()
            return True, f"cannot do that: {type(exc).__name__}: {exc}"

    def _key(self, key: int, typed: str | None) -> tuple[bool, str | None]:
        if self.editing is not None:
            return True, self._type(key, typed)
        row = ROWS[self.row]
        if key in (27, ord("c")):
            return True, self.leave()
        if key in (UP, BACKTAB, DOWN, TAB):
            self.row = (self.row + (-1 if key in (UP, BACKTAB) else 1)) % len(ROWS)
            return True, None
        if key in (LEFT, RIGHT):
            return True, self.step(row, -1 if key == LEFT else 1)
        if key in ENTER:
            return True, self.activate(row)
        if key in (ord(","), ord(".")):
            self.turn(TURN_STEP if key == ord(".") else -TURN_STEP)
            return True, None
        if key in (ord("f"), ord("b")):
            self.turn(to=0.0 if key == ord("f") else 180.0)
            return True, None
        if key == ord(" "):
            self.turn(turning=not self.turning)
            return True, None
        if key == ord("p"):
            return False, None
        return True, None

    def _type(self, key: int, typed: str | None) -> str | None:
        row = self.editing
        if key in ENTER:
            return self._end_typing()
        if key == 27:
            return self._end_typing(cancel=True)
        if key in (UP, BACKTAB, DOWN, TAB):
            message = self._end_typing()
            self.row = (self.row + (-1 if key in (UP, BACKTAB) else 1)) % len(ROWS)
            return message
        text = self.text(row)
        if key == BACKSPACE:
            self._set_text(row, text[:-1])
            return None
        char = typed if typed else (chr(key) if 32 <= key < 0x110000 and chr(key).isprintable() else None)
        if char and char.isprintable() and len(text) < MAX_TEXT[row]:
            self._set_text(row, text + char)
        return None

    def help(self) -> str:
        if self.editing is not None:
            what = "the name" if self.editing == "name" else "a fabric, e.g. blue paisley silk (Enter: make it)"
            return f"Type {what} | Enter: done | Esc: as it was | Backspace: delete"
        return ("Up / Down: row | Left / Right: choose | Enter: type, do | , . turn  F front  B back  Space turning"
                " | Esc: close")


# -- fills in words -------------------------------------------------------------------------------------------------


def fill_kind(fill) -> str:
    """The choice of the pattern row a fill is: ``"own"``, ``"plain"``, a pattern or ``"fabric"``."""
    if fill is None or fill.keeps():
        return "own"
    if isinstance(fill, Pattern):
        return fill.kind
    if isinstance(fill, Solid):
        return "plain"
    if isinstance(fill, Swatch):
        return "fabric"
    return "own"


def primary_of(fill):
    if isinstance(fill, Pattern):
        return fill.primary
    return fill.rgb if isinstance(fill, Solid) else None


def secondary_of(fill):
    return fill.secondary if isinstance(fill, Pattern) else None


def _colour_name(rgb) -> str:
    from .characters.outfit.fills import COLORS, hex_color

    rgb = tuple(int(v) for v in rgb)
    return next((name for name in COLOURS if tuple(color(name)) == rgb),
                next((name for name, value in COLORS.items() if tuple(value) == rgb), hex_color(rgb)))


def describe_fill(fill) -> str:
    """A garment's fill in words: ``"tartan, red on navy"``, ``"navy"``, ``"fabric 'blue silk' (procedural)"``."""
    kind = fill_kind(fill)
    if kind == "own":
        return "the avatar's own"
    if kind == "plain":
        return _colour_name(fill.rgb)
    if isinstance(fill, Pattern):
        return f"{fill.kind}, {_colour_name(fill.primary)} and {_colour_name(fill.secondary)}"
    if isinstance(fill, Fabric):
        return f"fabric {fill.text!r} ({fill.method})"
    return "a swatch"


def _index(items: list, value) -> int | None:
    try:
        return items.index(value)
    except ValueError:
        return None
