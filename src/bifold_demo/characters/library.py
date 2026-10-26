"""The prepared character library: avatars and animation clips ready to use in a scene.

Prepare it once with ``bifold-characters fetch`` (see :mod:`.rocketbox`); afterwards loading an avatar takes
a few milliseconds and needs no FBX code::

    library = CharacterLibrary.shared()  # kept across scene reloads; or CharacterLibrary(folder)
    if library.is_ready():
        model = library.load("Business_Female_01")
        motions = library.motions()

Avatars and clips whose files are damaged (e.g. cut off by a partial copy) are left out with one
:class:`DamagedFileWarning` that names the command that repairs them.
"""

from __future__ import annotations

import difflib
import os
import shlex
import sys
import time
import warnings
from pathlib import Path

from . import rocketbox
from .model import AnimationClip, CharacterModel

DEFAULT_FOLDER = Path("~/.cache/bifold-demo/characters")
ENV_VAR = "BIFOLD_CHARACTERS"
READY_ROLES = ("walk", "idle")  # what a scene needs: the other roles fall back to these, turns are done by hand
_shared: dict[Path, CharacterLibrary] = {}


class DamagedFileWarning(UserWarning):
    """A prepared avatar or clip is left out because one of its files is damaged."""


def default_folder() -> Path:
    """``$BIFOLD_CHARACTERS`` if set, else ``~/.cache/bifold-demo/characters``."""
    return Path(os.environ.get(ENV_VAR) or DEFAULT_FOLDER).expanduser()


def fetch_command(options: str = "", folder=None) -> str:
    """``bifold-characters fetch OPTIONS``, plus ``--dest FOLDER`` when ``folder`` is not :func:`default_folder`."""
    command = f"bifold-characters fetch {options}".rstrip()
    if folder is not None and not _same(folder, default_folder()):
        command += f" --dest {shlex.quote(str(folder))}"
    return command


def _same(a, b) -> bool:
    return Path(a).expanduser().resolve() == Path(b).expanduser().resolve()


def _signature(*paths: Path) -> tuple:
    return tuple((s.st_mtime_ns, s.st_size) for s in (p.stat() for p in paths))


def _clip_option(names) -> str:
    return "--avatars --clips " + ("basic" if set(names) <= set(rocketbox.clip_set("basic")) else "all")


def _avatar_option(names) -> str:
    # --clips basic: --avatars alone means all 74 clips; the basic ones are what a scene needs (and are skipped
    # when they are prepared already)
    return f"--avatars {' '.join(names)} --clips basic"


def _summary(problems: dict[str, str], most: int = 3) -> str:
    text = "; ".join(f"{name}: {problem}" for name, problem in list(problems.items())[:most])
    return text + (f"; and {len(problems) - most} more" if len(problems) > most else "")


class CharacterLibrary:
    """Prepared avatars and clips in ``cache_dir`` (default: :func:`default_folder`).

    Everything is memoised and looked at again when the folder changes. Avatars and clips with damaged files
    are left out (see :meth:`damaged`).
    """

    def __init__(self, cache_dir=None):
        self.folder = Path(cache_dir).expanduser() if cache_dir is not None else default_folder()
        self._scans: dict[str, tuple] = {}  # kind -> (mtime, {name: (signature, metadata or path)}, problems, racy)
        self._broken: dict[tuple[str, str], tuple[tuple, str]] = {}  # failed to load: (kind, name) -> (signature, why)
        self._models: dict[tuple[str, str], tuple[tuple, CharacterModel]] = {}
        self._clips: tuple | None = None  # (scanned clip files, {name: clip}, {name: (signature, clip)})
        self._motions: tuple | None = None  # (clips, MotionLibrary)
        self._warned: set[tuple[str, str, str]] = set()

    @classmethod
    def shared(cls, cache_dir=None) -> CharacterLibrary:
        """The one library of ``cache_dir`` (default: :func:`default_folder`) in this process.

        A scene that uses it keeps the loaded avatars, clips and motions when its module is reloaded.
        """
        folder = (Path(cache_dir).expanduser() if cache_dir is not None else default_folder()).resolve()
        if folder not in _shared:
            _shared[folder] = cls(folder)
        return _shared[folder]

    def _scan(self, kind: str) -> dict[str, tuple]:
        """``{name: (signature, metadata or clip path)}`` of the ``kind`` files that pass the quick check."""
        folder = self.folder / ("avatars" if kind == "avatar" else "clips")
        try:
            mtime = folder.stat().st_mtime_ns
        except OSError:
            self._scans.pop(kind, None)
            return {}
        scan = self._scans.get(kind)
        if scan is None or scan[0] != mtime or scan[3]:
            racy = abs(time.time_ns() - mtime) < 100_000_000  # a change in the same clock tick keeps the mtime
            found, problems = {}, {}
            for path in sorted(folder.glob("*.json" if kind == "avatar" else "*.npz")):
                try:
                    if kind == "avatar":
                        meta = rocketbox.check_avatar(self.folder, path.stem)
                        found[path.stem] = (_signature(path, path.with_suffix(".npz")), meta)
                    else:
                        rocketbox.check_clip(path)
                        found[path.stem] = (_signature(path), path)
                except FileNotFoundError:
                    continue  # removed meanwhile
                except (OSError, ValueError) as exc:
                    problems[path.stem] = str(exc)
            scan = self._scans[kind] = (mtime, found, problems, racy)
            self._warn(kind, problems)
        return {n: v for n, v in scan[1].items() if self._broken.get((kind, n), (None,))[0] != v[0]}

    def available(self) -> dict[str, dict]:
        """``{name: metadata}`` of the prepared avatars; catalog fields (sex, age_group, ...) are current."""
        catalog = rocketbox.load_catalog()
        return {name: {**meta, **catalog.get(name, {})} for name, (_, meta) in self._scan("avatar").items()}

    def load(self, name: str, textures: str = "path") -> CharacterModel:
        """The prepared avatar ``name`` (memoised; share it between characters, do not modify it).

        ``textures``: ``"path"`` (PNG paths the renderer decodes once, the fastest), ``"array"`` (RGBA arrays)
        or ``"none"``; see :func:`.rocketbox.read_character`. Raises ``KeyError`` (saying how to get it) when
        the avatar is not available, ``ValueError`` when its files turn out to be damaged (it is then left out).
        """
        if textures not in ("path", "array", "none"):
            raise ValueError("textures must be 'path', 'array' or 'none'")
        stored = self._scan("avatar")
        if name not in stored:
            raise KeyError(self.explain(name))
        signature = stored[name][0]
        cached = self._models.get((name, textures))
        if cached is None or cached[0] != signature:
            try:
                model = rocketbox.read_character(self.folder, name, textures, metadata=self.available()[name])
            except Exception as exc:  # damage the quick check cannot see
                problem = f"{type(exc).__name__}: {exc}"
                self._broken[("avatar", name)] = (signature, problem)
                self._warn("avatar", {name: problem})
                raise ValueError(self.explain(name)) from exc
            cached = self._models[(name, textures)] = (signature, model)
        return cached[1]

    def explain(self, name: str) -> str:
        """Why the avatar or clip ``name`` is not available (damaged, not prepared yet or unknown) and what fixes it."""
        for kind in ("avatar", "clip"):
            problem = self._problems(kind).get(name)
            if problem:
                return f"the prepared {kind} {name} is damaged ({problem}); to repair it: {self._repair(kind, [name])}"
        if name in rocketbox.AVATARS:
            return f"avatar {name} is not prepared yet; run: {fetch_command(_avatar_option([name]), self.folder)}"
        if name in rocketbox.CLIPS:
            return f"clip {name} is not prepared yet; run: {fetch_command(_clip_option([name]), self.folder)}"
        known = dict.fromkeys([*self._scan("avatar"), *rocketbox.AVATARS, *rocketbox.CLIPS])
        close = difflib.get_close_matches(name, list(known), n=3)
        hint = f"did you mean {' or '.join(close)}?" if close else "see: bifold-characters list --all"
        return f"no avatar or clip {name!r} in {self.folder}; {hint}"

    def damaged(self) -> dict[str, str]:
        """``{name: what is wrong}`` for the avatars and clips that are left out because their files are damaged."""
        return {**self._problems("avatar"), **self._problems("clip")}

    def _problems(self, kind: str) -> dict[str, str]:
        self._scan(kind)
        scan = self._scans.get(kind)
        if scan is None:
            return {}
        problems = dict(scan[2])
        for (k, name), (signature, problem) in self._broken.items():
            if k == kind and name in scan[1] and scan[1][name][0] == signature:
                problems[name] = problem
        return problems

    def _repair(self, kind: str, names) -> str:
        names = list(names)
        option = _avatar_option(names) if kind == "avatar" else _clip_option(names)
        command = fetch_command(option, self.folder)
        broken = [n for n in names if (kind, n) in self._broken]
        if not broken:
            return command  # fetch converts damaged items again
        folder, suffix = ("avatars", ".json") if kind == "avatar" else ("clips", ".npz")
        files = " ".join(shlex.quote(str(self.folder / folder / f"{n}{suffix}")) for n in broken)
        return f"delete {files}, then run: {command}"

    def _warn(self, kind: str, problems: dict[str, str]) -> None:
        new = {n: p for n, p in problems.items() if (kind, n, p) not in self._warned}
        if new:
            self._warned.update((kind, n, p) for n, p in new.items())
            what = f"{len(new)} damaged {kind}s" if len(new) > 1 else f"the damaged {kind}"
            frame, level = sys._getframe(), 1
            while frame is not None and frame.f_code.co_filename == __file__:  # report the caller's line
                frame, level = frame.f_back, level + 1
            warnings.warn(f"{self.folder}: leaving out {what} ({_summary(new)}); to repair: {self._repair(kind, new)}",
                          DamagedFileWarning, stacklevel=level)

    def clips(self) -> dict[str, AnimationClip]:
        """All prepared clips by name (memoised, read again when the folder changes; damaged ones are left out)."""
        files = self._scan("clip")
        if self._clips is None or self._clips[0] != files:
            loaded = self._clips[2] if self._clips else {}
            clips, by_name, problems = {}, {}, {}
            for name, (signature, path) in files.items():
                entry = loaded.get(name)
                if entry is None or entry[0] != signature:
                    try:
                        entry = (signature, rocketbox.read_clip(path))
                    except Exception as exc:  # noqa: BLE001 - damage the quick check cannot see
                        problems[name] = f"{type(exc).__name__}: {exc}"
                        self._broken[("clip", name)] = (signature, problems[name])
                        continue
                by_name[name] = entry
                clips[entry[1].name] = entry[1]
            self._clips = (self._scan("clip"), clips, by_name)
            self._warn("clip", problems)
        return self._clips[1]

    def roles(self) -> dict[str, list[str]]:
        """``{role: [prepared clip names]}`` for every role of :data:`.rocketbox.ROLE_CLIPS` (damaged ones left out)."""
        files = self._scan("clip")
        return {role: [n for n in names if n in files] for role, names in rocketbox.ROLE_CLIPS.items()}

    def motions(self):
        """A :class:`.behaviour.MotionLibrary` of the prepared clips with their roles registered (memoised)."""
        clips = self.clips()
        if self._motions is None or self._motions[0] is not clips:
            from .behaviour import MotionLibrary

            motions = MotionLibrary(clips)
            for role, names in rocketbox.ROLE_CLIPS.items():
                for name in names:
                    if name in clips:
                        motions.register(role, name)
            self._motions = (clips, motions)
        return self._motions[1]

    def missing(self) -> list[str]:
        """What a scene still needs (see :meth:`is_ready`): ``"avatars"`` and/or roles of :data:`READY_ROLES`."""
        roles = self.roles()
        return ([] if self._scan("avatar") else ["avatars"]) + [r for r in READY_ROLES if not roles[r]]

    def is_ready(self) -> bool:
        """At least one avatar and clips for walking and standing: all a scene needs.

        The other roles fall back to these and turns are done by hand; their clips only look more natural.
        """
        return not self.missing()

    def describe(self) -> str:
        """A short human-readable summary of what is prepared, what is damaged and what is missing."""
        avatars = self.available()
        lines = [f"characters folder: {self.folder}"]
        if avatars:
            def count(key):
                values = [str(m.get(key, "?")) for m in avatars.values()]
                return ", ".join(f"{values.count(v)} {v}" for v in sorted(set(values)))

            lines.append(f"  {len(avatars)} avatars: {count('sex')}; {count('age_group')}")
        else:
            lines.append("  no avatars")
        roles = self.roles()
        empty = [r for r, v in roles.items() if not v]
        lines.append(f"  {sum(map(len, roles.values()))} clips for {len(roles) - len(empty)} of {len(roles)} roles"
                     + (f" (none for: {', '.join(empty)})" if empty and len(empty) < len(roles) else ""))
        for kind in ("avatar", "clip"):
            problems = self._problems(kind)
            if problems:
                lines.append(f"  left out, damaged: {_summary(problems)}; to repair: {self._repair(kind, problems)}")
        missing, optional = self.missing(), [r for r in empty if r not in READY_ROLES]
        if missing:
            hint = "--set small" if "avatars" in missing else "--avatars --clips basic"
            lines.append(f"  not ready, missing: {', '.join(missing)}; run: {fetch_command(hint, self.folder)}")
        elif optional:
            lines.append(f"  ready; optional clips that make people look more natural are missing for: "
                         f"{', '.join(optional)}; to add them: {fetch_command('--avatars --clips basic', self.folder)}")
        else:
            lines.append("  ready")
        if not _same(self.folder, default_folder()):
            lines.append(f"  note: bifold-demo uses {default_folder()}; to use this folder, first run: "
                         f"export {ENV_VAR}={shlex.quote(str(self.folder))}")
        return "\n".join(lines)
