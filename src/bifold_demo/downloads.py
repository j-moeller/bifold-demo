"""Model files downloaded once and checked: the face models of :mod:`.recognition` and of
:mod:`.characters.headtex`.

A :class:`ModelFile` names a file, its size and its SHA-256, and where it comes from: one or more :class:`Source`\\ s,
tried in order. A source is a direct download, or a member of a zip pack (InsightFace publishes its models as zips
such as ``buffalo_l.zip``): the pack is downloaded, checked, and only the wanted member is taken out of it, under the
file's own name; the pack is deleted afterwards. :func:`ensure` gives the file's path in a folder, downloading it
first when it is missing or has the wrong size; :func:`ensure_many` does that for several files and downloads a pack
only once for all its members.

Every download goes to a temporary file next to the target and only replaces it once its checksum matches, so a
cut-off download, a proxy's error page or a changed upstream file never takes the place of a model. A file whose
size is not known in advance (``size=None``: only its SHA-256 was published) is checked by its hash when it is first
used in a process.
"""

from __future__ import annotations

import hashlib
import os
import threading
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

BLOCK = 1 << 20


class DownloadError(OSError):
    """No source gave the files with the right checksums; ``errors``: ``{file name: [what went wrong per source]}``."""

    def __init__(self, message: str, errors: dict[str, list[str]]):
        super().__init__(message)
        self.errors = errors


@dataclass(frozen=True)
class Source:
    """Where a file can be downloaded: ``url`` itself, or the ``member`` (its path inside the zip) of the zip pack at
    ``url``. ``sha256`` and ``size`` are those of the download (the pack's, for a member), checked when given."""

    url: str
    member: str | None = None
    sha256: str | None = None
    size: int | None = None


@dataclass(frozen=True)
class ModelFile:
    name: str  # the file name in the models folder
    url: str  # the usual place to get it (a direct download); see ``sources``
    sha256: str
    size: int | None  # bytes; None: not known in advance (checked by the hash alone)
    sources: tuple[Source, ...] = ()  # where it comes from, tried in order; default: ``url`` alone

    def places(self) -> tuple[Source, ...]:
        """The sources to try, in order."""
        return self.sources or (Source(self.url),)


def member_of(pack: Source, name: str, sha256: str, size: int | None, member: str | None = None,
              mirrors: tuple[str, ...] = ()) -> ModelFile:
    """A file that comes in the zip ``pack`` (as ``member``, default ``name``), with direct-download ``mirrors``
    of the very same file (the same SHA-256) tried after the pack."""
    inside = Source(pack.url, member or name, pack.sha256, pack.size)
    return ModelFile(name, pack.url, sha256, size, (inside, *(Source(url) for url in mirrors)))


def sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


_verified: dict[tuple, bool] = {}  # (path, size, mtime_ns, sha256) -> whether the file has that hash
_verified_lock = threading.Lock()


def verified(path, expected: str) -> bool:
    """Whether the file at ``path`` has the SHA-256 ``expected`` (hashed once per process while it is unchanged)."""
    path = Path(path)
    try:
        st = path.stat()
    except OSError:
        return False
    key = (str(path.resolve()), st.st_size, st.st_mtime_ns, expected)
    with _verified_lock:
        known = _verified.get(key)
    if known is None:
        known = sha256(path) == expected
        with _verified_lock:
            _verified[key] = known
    return known


def present(model: ModelFile, folder) -> Path | None:
    """The path of ``model`` in ``folder`` if it is there (by its size; by its hash when the size is not known)."""
    path = Path(folder).expanduser() / model.name
    if not path.is_file():
        return None
    if model.size is not None:
        return path if path.stat().st_size == model.size else None
    return path if verified(path, model.sha256) else None


def ensure(model: ModelFile, folder, download: bool = True, progress=print, fetch_hint: str = "",
           tag: str = "models") -> Path:
    """The path of ``model`` in ``folder``, downloading it first if it is missing (or damaged). Without
    ``download`` a missing file raises :class:`FileNotFoundError`, naming ``fetch_hint`` (the command that fetches
    it). Raises :class:`DownloadError` when no source gives the file with the right checksum."""
    return ensure_many([model], folder, download, progress, fetch_hint, tag)[model.name]


def ensure_many(models, folder, download: bool = True, progress=print, fetch_hint: str = "",
                tag: str = "models") -> dict[str, Path]:
    """``{name: path}`` for ``models`` in ``folder``, downloading the missing ones; a zip pack is downloaded once for
    all the files taken from it. A file that cannot be had from one source is tried from the next; when some file
    cannot be had at all, :class:`DownloadError` says why (the others are downloaded all the same)."""
    folder = Path(folder).expanduser()
    found: dict[str, Path] = {}
    missing: list[ModelFile] = []
    for model in models:
        path = present(model, folder)
        if path is not None:
            found[model.name] = path
        else:
            missing.append(model)
    if not missing:
        return found
    if not download:
        names = ", ".join(str(folder / m.name) for m in missing)
        raise FileNotFoundError(f"{names} {'is' if len(missing) == 1 else 'are'} missing"
                                + (f" (run: {fetch_hint})" if fetch_hint else ""))
    folder.mkdir(parents=True, exist_ok=True)
    errors: dict[str, list[str]] = {m.name: [] for m in missing}
    attempt = 0
    while missing:  # each round: the next source of every file still missing; members of one pack together
        groups: dict[Source, list[tuple[ModelFile, Source]]] = {}
        for model in missing:
            places = model.places()
            if attempt < len(places):
                source = places[attempt]
                key = Source(source.url, None, source.sha256, source.size) if source.member else source
                groups.setdefault(key, []).append((model, source))
        if not groups:
            break
        for key, wanted in groups.items():
            try:
                if wanted[0][1].member is None:
                    model = wanted[0][0]
                    found[model.name] = _download(model, key, folder, progress, tag)
                else:
                    found.update(_from_pack([m for m, _ in wanted], key, [s.member for _, s in wanted], folder,
                                            progress, tag, errors))
            except OSError as exc:  # (urllib's errors are OSErrors) try the next source
                for model, _ in wanted:
                    errors[model.name].append(f"{key.url}: {exc}")
        missing = [m for m in missing if m.name not in found]
        attempt += 1
    if missing:
        detail = "; ".join(f"{m.name}: {' / '.join(errors[m.name]) or 'no source'}" for m in missing)
        raise DownloadError(f"cannot download {', '.join(m.name for m in missing)} ({detail})",
                            {m.name: errors[m.name] for m in missing})
    return found


def _fetch(url: str, target: Path, sha: str | None, size: int | None, progress, tag: str, what: str) -> None:
    """Download ``url`` into ``target`` (a temporary file), checking ``size`` and ``sha`` when given."""
    if progress:
        megabytes = f" ({size / 1e6:.1f} MB)" if size else ""
        progress(f"[{tag}] downloading {what}{megabytes} from {url}")
    digest = hashlib.sha256()
    count = 0
    with urllib.request.urlopen(url, timeout=60) as response, open(target, "wb") as out:
        while block := response.read(BLOCK):
            count += len(block)
            if size is not None and count > size:
                raise OSError(f"{what}: the download is bigger than expected (a proxy page or a changed file?)")
            out.write(block)
            digest.update(block)
    if size is not None and count != size:
        raise OSError(f"{what}: the download has {count} bytes, not {size} (cut off?)")
    if sha is not None and digest.hexdigest() != sha:
        raise OSError(f"{what}: the download does not match its checksum (a proxy page or a changed file?)")


def _download(model: ModelFile, source: Source, folder: Path, progress, tag: str) -> Path:
    path = folder / model.name
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        _fetch(source.url, tmp, model.sha256, model.size, progress, tag, model.name)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def _from_pack(models: list[ModelFile], pack: Source, members: list[str], folder: Path, progress, tag: str,
               errors: dict[str, list[str]]) -> dict[str, Path]:
    """Download the zip ``pack`` once and take ``members`` out of it as ``models`` (each checked); the zip is
    deleted afterwards. Members that are missing or wrong are noted in ``errors``."""
    name = pack.url.rstrip("/").rsplit("/", 1)[-1] or "pack.zip"
    tmp = folder / f".{name}.{os.getpid()}.{threading.get_ident()}.tmp"
    found = {}
    try:
        _fetch(pack.url, tmp, pack.sha256, pack.size, progress, tag, name)
        try:
            archive = zipfile.ZipFile(tmp)
        except zipfile.BadZipFile as exc:
            raise OSError(f"{name} is not a zip file: {exc}") from exc
        with archive:
            for model, member in zip(models, members):
                try:
                    found[model.name] = _extract(archive, member, model, folder)
                    if progress:
                        progress(f"[{tag}] {model.name}: taken out of {name}")
                except (KeyError, OSError, zipfile.BadZipFile) as exc:
                    errors[model.name].append(f"{member} in {pack.url}: {exc}")
    finally:
        tmp.unlink(missing_ok=True)
    return found


def _extract(archive: zipfile.ZipFile, member: str, model: ModelFile, folder: Path) -> Path:
    """Copy ``member`` of ``archive`` to ``model.name`` in ``folder``, checking its size and SHA-256. Only that
    member is read, under a name chosen here, so paths inside the zip never decide where anything is written."""
    info = archive.getinfo(member)  # KeyError: not in the pack
    if model.size is not None and info.file_size != model.size:
        raise OSError(f"{member} has {info.file_size} bytes, not {model.size}")
    path = folder / model.name
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    digest = hashlib.sha256()
    count = 0
    limit = model.size if model.size is not None else info.file_size
    try:
        with archive.open(info) as source, open(tmp, "wb") as out:
            while block := source.read(BLOCK):
                count += len(block)
                if count > limit:
                    raise OSError(f"{member} is bigger than its entry says")
                out.write(block)
                digest.update(block)
        if digest.hexdigest() != model.sha256:
            raise OSError(f"{member} does not match its checksum")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path
