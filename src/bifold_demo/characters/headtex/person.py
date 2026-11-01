"""Stages 5 to 7, and all stages end to end: a person's photos in, the textures of a 3D person out (head, body, hair).

::

    from bifold_demo.characters.headtex import person_textures

    person = person_textures(["front.jpg", "left.jpg"], model, views=[None, "left"])
    print(person.summary())                 # what each stage did, the neck and the hands, warnings
    person.wear(character)                  # on a scene Character of that model: head, body (skin) and hair
    created = person.created("Ada at the fair", face="Ada Lovelace")   # a CreatedCharacter, to save in created/

:func:`finish` takes stages 0 to 4 (a :class:`~.texture.HeadTexture`) the rest of the way:

5. **Complete** what no photo shows (:func:`.completion.complete`): the other half of the face mirrored in, and, with
   an ``inpainter`` (:class:`~.completion.FluxInpainter`), a second-class generative layer.
6. **Match the body** (:func:`.body.match_body`): the body's skin to the face, the neck seam levelled, the hair.
7. **Check** (:attr:`PersonTextures.checks`): the colour difference (Lab delta E) across the neck seam and from the
   face's skin to the hands, before stage 6 and after; with ``check=True`` (OpenGL and the face recognition's
   models), the identity: the face recognition's similarity (its profile's recogniser) between renders of the
   avatar wearing the textures (front and three-quarter) and the photos, and of the avatar's own head as a baseline.
   Since the geometry is the avatar's, not the person's, the similarity is low anyway (with SFace 0.37-0.58 for the
   faces painted on the fair's people from one photo) and only says something relative to that baseline: it is
   reported, never a reason to reject the result. A
   generative layer that lowers it by more than :data:`IDENTITY_DROP` is dropped, with a warning.

The result, :class:`PersonTextures`, holds the textures by primitive name (what a
:class:`~..created.CreatedCharacter` keeps, :meth:`PersonTextures.created`), puts them on a scene character
(:meth:`PersonTextures.wear`, :func:`wear_textures`; a body wearing an outfit keeps it, over the person's skin) and
saves them (:meth:`PersonTextures.save`, :meth:`PersonTextures.load`).

:func:`person_textures` runs stages 0 to 7 from photos. With ``cache`` (a folder, :class:`PersonCache`) the result
is kept by the photos' contents and the avatar, so the same photos are worked on once (the known faces in the fair
keep theirs in ``faces/.headtex/``, next to the photos: textures made from photos are personal data like the photos).
The character creator's face (``bifold_demo.creator.face_texture``) comes from it too.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import light

VERSION = 1  # of what the cache keeps: raise it when the stages change what they make
IDENTITY_YAWS = (0.0, 30.0)  # the renders the identity check compares with the photos
IDENTITY_DROP = 0.02  # a generative layer that lowers the identity by more than this is dropped
CACHE_KEEP = 40  # people's textures the cache keeps (each a few MB); the oldest go first
INFO = "person.json"


@dataclass
class PersonTextures:
    """A person's textures for one avatar (see the module docs): ``textures`` maps primitive names of the avatar to
    RGBA ``uint8`` images (row 0 at the top): the head, the body (its skin matched) and the hair cards."""

    model: str
    textures: dict
    head_primitive: str
    body_primitive: str = ""
    checks: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    timings: dict = field(default_factory=dict)
    result: object = None  # the HeadTexture of stages 0-4 (None when read back from a cache)
    completion: object = None  # stage 5's Completion
    match: object = None  # stage 6's BodyMatch
    cached: bool = False  # read back from a cache

    @property
    def head(self) -> np.ndarray:
        return self.textures[self.head_primitive]

    @property
    def body(self) -> np.ndarray | None:
        return self.textures.get(self.body_primitive)

    def wear(self, character) -> dict:
        """Put the textures on ``character`` (a scene Character of the avatar): :func:`wear_textures`. Returns the
        :class:`~bifold_demo.scene.textures.Texture` per primitive name (what :func:`take_off_textures` takes)."""
        from ...scene.textures import Texture

        return wear_textures(character, {name: Texture(rgba, name=f"{character.name} {name} from photos")
                                         for name, rgba in self.textures.items()})

    def created(self, name: str, face: str | None = None, **fields):
        """A :class:`~..created.CreatedCharacter` called ``name`` wearing these textures (``face``: the known person
        whose face it is; ``fields``: its other fields, such as ``items``); ready for
        :meth:`~..created.CharacterBook.save`."""
        from ..created import CreatedCharacter

        return CreatedCharacter(name, self.model, face, textures=dict(self.textures), **fields)

    def save(self, folder) -> Path:
        """Write the textures (PNG, by primitive name) and what was found (``person.json``) into ``folder``."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        files = {}
        for name, rgba in self.textures.items():
            file = "".join(c if c.isalnum() or c in "-_." else "_" for c in name) + ".png"
            if not cv2.imwrite(str(folder / file), cv2.cvtColor(np.asarray(rgba), cv2.COLOR_RGBA2BGRA)):
                raise OSError(f"cannot write {folder / file}")
            files[name] = file
        info = {"version": VERSION, "model": self.model, "head": self.head_primitive, "body": self.body_primitive,
                "textures": files, "checks": _plain(self.checks), "warnings": list(self.warnings)}
        (folder / INFO).write_text(json.dumps(info, indent=1), encoding="utf-8")
        return folder

    @classmethod
    def load(cls, folder) -> PersonTextures:
        """The textures :meth:`save` wrote into ``folder``. Raises ``OSError`` or ``ValueError`` when they cannot be
        read."""
        folder = Path(folder)
        info = json.loads((folder / INFO).read_text(encoding="utf-8"))
        if not isinstance(info, dict) or info.get("version") != VERSION:
            raise ValueError(f"{folder}: not textures of this version")
        textures = {}
        for name, file in dict(info["textures"]).items():
            image = cv2.imread(str(folder / Path(file).name), cv2.IMREAD_UNCHANGED)
            if image is None or image.ndim != 3 or image.shape[2] != 4:
                raise OSError(f"cannot read {folder / file}")
            textures[name] = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        return cls(str(info["model"]), textures, str(info["head"]), str(info.get("body") or ""),
                   dict(info.get("checks") or {}), list(info.get("warnings") or []), cached=True)

    def summary(self) -> str:
        lines = [f"{self.model}: textures for " + ", ".join(self.textures) + (" (from the cache)" if self.cached
                                                                                 else "")]
        if self.result is not None:  # (its warnings and timings are among this one's)
            lines += [line for line in self.result.summary().splitlines()[1:]
                      if not line.startswith(("  warning:", "  timings:"))]
        if self.completion is not None:
            share = self.completion.share()
            lines.append(f"  stage 5: {self.completion.method}; {100 * share['mirrored']:.0f} % of the texels "
                         f"mirrored, {100 * share['generated']:.0f} % generated")
        if self.match is not None:
            lines.append("  stage 6: the body's skin times " + " ".join(f"{g:.2f}" for g in self.match.gain)
                         + (" (by the neck " + " ".join(f"{g:.2f}" for g in self.match.neck_gain) + ")"
                            if self.match.neck_gain is not None else "")
                         + f"; {self.match.seam.get('pairs', 0)} skin pairs on the neck seam levelled"
                         + (f"; hair recoloured on {', '.join(self.match.hair)}" if self.match.hair else ""))
        for line in describe_checks(self.checks):
            lines.append(f"  {line}")
        lines += [f"  warning: {w}" for w in self.warnings]
        if self.timings:
            lines.append("  timings: " + ", ".join(f"{k} {v:.2f} s" for k, v in self.timings.items()))
        return "\n".join(lines)


def wear_textures(character, textures: dict) -> dict:
    """Put ``textures`` (scene :class:`~bifold_demo.scene.textures.Texture` by primitive name) on ``character``, its
    own (``texture_overrides``: the avatar's other users keep theirs). A body wearing an outfit
    (:func:`~..outfit.tailor.dress`) keeps it, now over the new body texture. Returns what it now wears, by primitive
    name (what :func:`take_off_textures` takes)."""
    from ..outfit.tailor import dress

    for name, texture in textures.items():
        current = character.texture_overrides.get(name)
        if getattr(current, "outfit", None) is not None:
            dress(character, current.outfit, base=texture)
        else:
            character.texture_overrides[name] = texture
    return dict(textures)


def take_off_textures(character, worn: dict) -> None:
    """Give ``character`` back what it wore before :func:`wear_textures` (``worn``: what that returned): the avatar's
    own textures (an outfit stays on, over the avatar's own body texture)."""
    from ..outfit.tailor import dress

    own = {p.name: p.texture for p in character.skinned_model.primitives}
    for name, texture in worn.items():
        current = character.texture_overrides.get(name)
        if current is texture:
            del character.texture_overrides[name]
        elif getattr(current, "outfit", None) is not None and getattr(current, "base", None) is texture:
            dress(character, current.outfit, base=own.get(name))


def describe_checks(checks: dict) -> list[str]:
    """Stage 7's checks in words."""
    out = []

    def pair(name, what):
        before, after = checks.get(f"{name} before"), checks.get(f"{name} after")
        if before is not None and after is not None and np.isfinite([before, after]).all():
            out.append(f"{what}: delta E {before:.1f} before stage 6, {after:.1f} after")

    pair("neck seam", f"across the neck seam ({checks.get('seam pairs', 0)} skin pairs)")
    pair("neck step", "the neck against the body's skin next to it")
    pair("face to hands", "the face's skin against the hands'")
    identity = checks.get("identity")
    if isinstance(identity, dict):
        own = checks.get("identity of the avatar's own head") or {}
        out.append("identity (the face recognition's similarity of renders to the photos): " + ", ".join(
            f"{k} {v:.2f}" + (f" (the avatar's own head {own[k]:.2f})" if k in own else "")
            for k, v in identity.items()))
    return out


def _plain(value):
    """``value`` with numpy numbers as Python ones, NaN as None (for JSON)."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


# -- stages 5 to 7 ----------------------------------------------------------------------------------------------


def finish(result, model, *, inpainter=None, check: bool = False, photos=None, engine=None, renderer=None,
           hair: bool = True) -> PersonTextures:
    """Stages 5 to 7 on ``result`` (a :class:`~.texture.HeadTexture` of ``model``'s head): see the module docs.
    ``inpainter``: stage 5's generative layer (:class:`~.completion.Inpainter`; none: the mirror and the avatar's own
    texture only); ``check``: also the identity check (needs OpenGL and the face recognition's models), against
    ``photos`` (BGR images or paths; default: the registered photos), with ``engine`` (a
    :class:`~bifold_demo.recognition.engine.FaceEngine`) and ``renderer`` when given; ``hair``: recolour the hair."""
    from .body import hands_tone, match_body
    from .completion import complete
    from .texture import neck_seam

    timings = {}
    warnings = list(result.warnings)
    start = time.perf_counter()
    completion = complete(result, model, inpainter=inpainter)
    warnings += completion.warnings
    timings["stage 5"] = time.perf_counter() - start
    checks = {}
    head_name = result.primitive
    if check:
        start = time.perf_counter()
        photos = photos if photos is not None else [v.image for v in result.views if v.image is not None]
        engine = engine or _engine()
        try:
            checks["identity of the avatar's own head"] = identity(model, {}, photos, engine=engine,
                                                                   renderer=renderer)
            if completion.deterministic is not None:
                with_layer = identity(model, {head_name: completion.rgba}, photos, engine=engine, renderer=renderer)
                without = identity(model, {head_name: completion.deterministic}, photos, engine=engine,
                                   renderer=renderer)
                checks["identity without the generated layer"] = without
                if _mean(with_layer) < _mean(without) - IDENTITY_DROP:
                    warnings.append(f"the generated layer lowered the identity check ({_mean(without):.2f} -> "
                                    f"{_mean(with_layer):.2f}): dropped")
                    completion = completion.without_generated()
        except ValueError as exc:  # no face in the photos or the renders
            warnings.append(f"no identity check: {exc}")
            check = False
        timings["stage 7 (identity)"] = time.perf_counter() - start
    start = time.perf_counter()
    match = match_body(result, model, completion.rgba, hair=hair)
    warnings += match.warnings
    timings["stage 6"] = time.perf_counter() - start
    start = time.perf_counter()
    if match.body is not None:
        before = neck_seam(model, completion.rgba, result.atlas)
        after = neck_seam(model, match.head, result.atlas, match.body)
        checks.update({"neck seam before": before["seam"], "neck seam after": after["seam"],
                       "seam pairs": after["pairs"], "neck step before": before["step"],
                       "neck step after": after["step"]})
    if result.skin is not None and match.body is not None:
        for when, body in (("before", None), ("after", match.body)):
            hands = hands_tone(model, body)
            checks[f"face to hands {when}"] = float(light.delta_e(result.skin.lab, hands)) if hands is not None \
                else float("nan")
    textures = match.textures()
    if check:
        try:
            checks["identity"] = identity(model, textures, photos, engine=engine, renderer=renderer)
        except ValueError as exc:
            warnings.append(f"no identity check: {exc}")
    timings["stage 7"] = time.perf_counter() - start
    return PersonTextures(model.name, textures, head_name, match.body_primitive, checks, warnings,
                          {**result.timings, **completion.timings, **timings}, result, completion, match)


def _mean(values: dict) -> float:
    finite = [v for v in values.values() if np.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")


def _engine():
    from ...recognition.engine import FaceEngine

    return FaceEngine(download=os.environ.get("BIFOLD_NO_DOWNLOADS") is None)


def identity(model, textures: dict, photos, *, engine=None, renderer=None, yaws=IDENTITY_YAWS) -> dict:
    """How much renders of ``model`` wearing ``textures`` (by primitive name; ``{}``: its own) look like the person
    in ``photos`` (BGR images or paths) to the face recognition (``engine``, a
    :class:`~bifold_demo.recognition.engine.FaceEngine`): per render (``"front"``, ``"turned 30"``), the
    similarity to the most similar photo; ``nan`` where the detector finds no face on the render. Raises
    ``ValueError`` when no photo has a face. Needs OpenGL."""
    from .texture import preview

    engine = engine or _engine()
    vectors = []
    for photo in photos:
        image = photo if isinstance(photo, np.ndarray) else cv2.imread(str(photo), cv2.IMREAD_COLOR)
        if image is None:
            continue
        faces = engine.detect(image, min_size=0)
        if faces:
            vectors.append(engine.embed(image, faces[0]))
    if not vectors:
        raise ValueError("the face recognition finds no face in the photos")
    renders = np.hsplit(preview(model, textures, renderer=renderer, yaws=yaws), len(yaws))
    out = {}
    for yaw, image in zip(yaws, renders):
        image = np.ascontiguousarray(image)
        faces = engine.detect(image, min_size=0)
        name = "front" if yaw == 0 else f"turned {yaw:.0f}"
        out[name] = float(max(v @ engine.embed(image, faces[0]) for v in vectors)) if faces else float("nan")
    return out


# -- from photos, cached ----------------------------------------------------------------------------------------


class PersonCache:
    """People's textures kept on disk by their photos (see the module docs): a folder per photos-and-avatar key,
    at most ``keep`` of them (the oldest go first). A cache that cannot be written costs time, never a texture."""

    def __init__(self, folder, keep: int = CACHE_KEEP):
        self.folder = Path(folder).expanduser()
        self.keep = keep

    def key(self, photos, model, **options) -> str:
        """The key of ``photos`` (paths or images; their contents count, not their names) for ``model``, with the
        ``options`` that change the result (what each photo shows, the generative model)."""
        from .landmarks import geometry_key

        h = hashlib.blake2b(f"person-textures-{VERSION}-{model.name}-{geometry_key(model)}".encode(), digest_size=12)
        for photo in photos:
            data = np.ascontiguousarray(photo).tobytes() if isinstance(photo, np.ndarray) else \
                Path(photo).read_bytes()
            h.update(hashlib.blake2b(data, digest_size=16).digest())
        h.update(json.dumps(_plain(options), sort_keys=True, default=str).encode())
        return h.hexdigest()

    def get(self, key: str) -> PersonTextures | None:
        folder = self.folder / key
        if not (folder / INFO).is_file():
            return None
        try:
            found = PersonTextures.load(folder)
        except (OSError, ValueError, KeyError, TypeError):
            return None  # damaged: made again
        try:
            os.utime(folder / INFO)  # used: the newest
        except OSError:
            pass
        return found

    def put(self, key: str, person: PersonTextures) -> None:
        target = self.folder / key
        tmp = self.folder / f".{key}.{os.getpid()}.tmp"
        try:
            if tmp.exists():
                shutil.rmtree(tmp)
            person.save(tmp)
            if target.exists():
                shutil.rmtree(target)
            os.replace(tmp, target)
            self._prune()
        except OSError:
            shutil.rmtree(tmp, ignore_errors=True)

    def _prune(self) -> None:
        entries = []
        for path in self.folder.iterdir():
            if path.name.startswith(".") or not (path / INFO).is_file():
                continue
            entries.append(((path / INFO).stat().st_mtime, path))
        for _, path in sorted(entries)[:max(0, len(entries) - self.keep)]:
            shutil.rmtree(path, ignore_errors=True)


def person_textures(photos, model, *, views=None, focals=None, cache=None, inpainter=None, check: bool = False,
                    engine=None, renderer=None, **options) -> PersonTextures:
    """Stages 0 to 7: ``photos`` of a person (paths or BGR images) to the textures of ``model`` (see the module
    docs). ``views``, ``focals`` and ``options`` (``landmarker``, ``segmenter``, ``head``, ``size``, ``cache_dir``)
    as for :func:`~.texture.head_texture`; ``inpainter``, ``check``, ``engine`` and ``renderer`` as for
    :func:`finish`; ``cache``: a folder (or a :class:`PersonCache`) that keeps the result by the photos' contents.
    Raises ``ImportError`` (:class:`~.detect.MediaPipeMissing`) without MediaPipe and ``ValueError`` when no photo
    can be used."""
    from .texture import head_texture

    store = PersonCache(cache) if cache is not None and not isinstance(cache, PersonCache) else cache
    photos = list(photos)
    key = None
    if store is not None:
        key = store.key(photos, model, views=views, inpainter=getattr(inpainter, "name", None),
                        size=options.get("size"))
        found = store.get(key)
        if found is not None:
            return found
    result = head_texture(photos, model, views=views, focals=focals, **options)
    person = finish(result, model, inpainter=inpainter, check=check, engine=engine, renderer=renderer)
    if store is not None:
        store.put(key, person)
    return person



__all__ = ["IDENTITY_DROP", "PersonCache", "PersonTextures", "describe_checks", "finish", "identity",
           "person_textures", "take_off_textures", "wear_textures"]
