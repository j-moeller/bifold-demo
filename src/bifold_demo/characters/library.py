"""The prepared character library: avatars and animation clips ready to use in a scene.

Prepare it once with ``bifold-characters fetch`` (see :mod:`.rocketbox`); afterwards loading an avatar takes
a few milliseconds and needs no FBX code::

    library = CharacterLibrary()
    if library.is_ready():
        model = library.load("Business_Female_01")
        motions = library.motions()
"""

from __future__ import annotations

import difflib
import json
import os
from pathlib import Path

from . import rocketbox
from .model import AnimationClip, CharacterModel

DEFAULT_FOLDER = Path("~/.cache/bifold-demo/characters")
ENV_VAR = "BIFOLD_CHARACTERS"
READY_ROLES = ("walk", "idle", "turn_left_90", "turn_right_90", "turn_left_180", "turn_right_180")


def default_folder() -> Path:
    """``$BIFOLD_CHARACTERS`` if set, else ``~/.cache/bifold-demo/characters``."""
    return Path(os.environ.get(ENV_VAR) or DEFAULT_FOLDER).expanduser()


class CharacterLibrary:
    """Prepared avatars and clips in ``cache_dir`` (default: :func:`default_folder`)."""

    def __init__(self, cache_dir=None):
        self.folder = Path(cache_dir).expanduser() if cache_dir is not None else default_folder()
        self._meta: tuple | None = None  # (avatars folder mtime, {name: stored metadata})
        self._models: dict[tuple[str, str], CharacterModel] = {}
        self._clips: dict[str, AnimationClip] | None = None
        self._motions = None

    def _stored(self) -> dict[str, dict]:
        folder = self.folder / "avatars"
        try:
            mtime = folder.stat().st_mtime_ns
        except OSError:
            return {}
        if self._meta is None or self._meta[0] != mtime:
            stored = {}
            for path in sorted(folder.glob("*.json")):
                try:
                    meta = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                if meta.get("format") == rocketbox.FORMAT and (folder / f"{path.stem}.npz").is_file():
                    stored[path.stem] = meta
            self._meta = (mtime, stored)
        return self._meta[1]

    def available(self) -> dict[str, dict]:
        """``{name: metadata}`` of the prepared avatars; catalog fields (sex, age_group, ...) are current."""
        catalog = rocketbox.load_catalog()
        return {name: {**meta, **catalog.get(name, {})} for name, meta in self._stored().items()}

    def load(self, name: str, textures: str = "path") -> CharacterModel:
        """The prepared avatar ``name`` (memoised; share it between characters, do not modify it).

        ``textures``: ``"path"`` (PNG paths the renderer decodes once, the fastest), ``"array"`` (RGBA arrays)
        or ``"none"``; see :func:`.rocketbox.read_character`.
        """
        key = (name, textures)
        if key not in self._models:
            available = self.available()
            if name not in available:
                raise KeyError(self._unknown(name, available))
            self._models[key] = rocketbox.read_character(self.folder, name, textures, metadata=available[name])
        return self._models[key]

    def _unknown(self, name: str, available: dict) -> str:
        close = difflib.get_close_matches(name, list(available), n=3)
        if close:
            return f"no prepared avatar {name!r}; did you mean {' or '.join(close)}?"
        if name in rocketbox.AVATARS:
            return f"avatar {name!r} is not prepared yet; run: bifold-characters fetch --avatars {name}"
        return f"no avatar {name!r} in {self.folder}; run: bifold-characters list"

    def clips(self) -> dict[str, AnimationClip]:
        """All prepared clips by name (memoised)."""
        if self._clips is None:
            self._clips = {}
            for path in sorted((self.folder / "clips").glob("*.npz")):
                try:
                    clip = rocketbox.read_clip(path)
                except (OSError, ValueError, KeyError):
                    continue
                self._clips[clip.name] = clip
        return self._clips

    def roles(self) -> dict[str, list[str]]:
        """``{role: [prepared clip names]}`` for every role of :data:`.rocketbox.ROLE_CLIPS`."""
        folder = self.folder / "clips"
        return {role: [n for n in names if (folder / f"{n}.npz").is_file()]
                for role, names in rocketbox.ROLE_CLIPS.items()}

    def motions(self):
        """A :class:`.behaviour.MotionLibrary` of the prepared clips with their roles registered (memoised)."""
        if self._motions is None:
            from .behaviour import MotionLibrary

            clips = self.clips()
            motions = MotionLibrary(clips)
            for role, names in rocketbox.ROLE_CLIPS.items():
                for name in names:
                    if name in clips:
                        motions.register(role, name)
            self._motions = motions
        return self._motions

    def missing(self) -> list[str]:
        """What is still needed before :meth:`is_ready`: ``"avatars"`` and/or the missing roles."""
        roles = self.roles()
        return ([] if self._stored() else ["avatars"]) + [r for r in READY_ROLES if not roles[r]]

    def is_ready(self) -> bool:
        """At least one avatar and clips for walking, standing and turning."""
        return not self.missing()

    def describe(self) -> str:
        """A short human-readable summary of what is prepared and what is missing."""
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
        have = sum(len(v) for v in roles.values())
        lines.append(f"  {have} clips for {sum(bool(v) for v in roles.values())} of {len(roles)} roles"
                     + (f" (none for: {', '.join(r for r, v in roles.items() if not v)})" if have else ""))
        missing = self.missing()
        if missing:
            hint = "--set small" if "avatars" in missing else "--avatars --clips basic"
            lines.append(f"  not ready, missing: {', '.join(missing)}; run: bifold-characters fetch {hint}")
        else:
            lines.append("  ready")
        return "\n".join(lines)
