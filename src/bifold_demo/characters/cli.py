"""``bifold-characters``: download and prepare the 3D people (Microsoft Rocketbox avatars and animations)."""

from __future__ import annotations

import argparse
import difflib
import sys
import time
from pathlib import Path

from . import rocketbox
from .library import CharacterLibrary, default_folder


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB"):
        if n < 1000:
            return f"{n:.1f} MB" if unit == "MB" else f"{n:.0f} {unit}"
        n /= 1000
    return f"{n:.1f} GB"


def _error(message: str, code: int = 2) -> int:
    print(f"error: {message}", file=sys.stderr)
    return code


def _progress(prefix: str):
    def report(done: int, total: int, item: str, status: str) -> None:
        print(f"[{prefix}] [{done:{len(str(total))}d}/{total}] {item}: {status}", flush=True)

    return report


def _folders(args) -> tuple[Path, Path]:
    dest = Path(args.dest).expanduser() if args.dest else default_folder()
    raw = getattr(args, "raw", None)
    return dest, Path(raw).expanduser() if raw else dest / "raw"


def _prepare(raw: Path, dest: Path, avatars, clips, args) -> int:
    try:
        report = rocketbox.prepare(raw, dest, avatars, clips, delete_raw=args.delete_raw, force=args.force,
                                   max_texture_size=args.texture_size, progress=_progress("prepare"))
    except ImportError as exc:
        return _error(str(exc), 1)
    print(f"[prepare] {len(report.avatars)} avatars and {len(report.clips)} clips converted, "
          f"{len(report.skipped)} already prepared, {len(report.failed)} failed")
    for name, reason in report.failed:
        print(f"  {name}: {reason}", file=sys.stderr)
    print(CharacterLibrary(dest).describe())
    return 1 if report.failed else 0


def cmd_fetch(args) -> int:
    dest, raw = _folders(args)
    try:
        chosen = args.avatars is not None
        avatars = rocketbox.check_names(args.avatars, rocketbox.AVATARS, "avatar") if chosen \
            else rocketbox.avatar_set(args.set)
        clips = rocketbox.clip_set(args.clips or ("basic" if args.set == "small" and not chosen else "all"))
    except ValueError as exc:
        return _error(f"{exc} (see: bifold-characters list --all)")
    todo_avatars = [n for n in avatars if args.force or not (dest / "avatars" / f"{n}.json").is_file()]
    todo_clips = [n for n in clips if args.force or not (dest / "clips" / f"{n}.npz").is_file()]
    paths = rocketbox.files_for(todo_avatars, todo_clips)
    present = [p for p in paths if (raw / p).is_file()]
    size = sum(rocketbox.estimate_size(p) for p in paths if not (raw / p).is_file())
    done = len(avatars) - len(todo_avatars) + len(clips) - len(todo_clips)
    print(f"[fetch] {len(avatars)} avatars and {len(clips)} animation clips"
          + (f" ({done} already prepared)" if done else "")
          + f": {len(paths)} files, about {_human(size)} to download"
          + (f" ({len(present)} files already there)" if present else ""))
    print(f"[fetch] from {rocketbox.BASE_URL}\n[fetch] into {raw}, prepared into {dest}")
    if args.dry_run:
        return 0
    if not paths:
        print(CharacterLibrary(dest).describe())
        return 0
    start, failed = time.monotonic(), False
    try:
        report = rocketbox.fetch(raw, todo_avatars, todo_clips, jobs=args.jobs, progress=_progress("fetch"))
    except rocketbox.FetchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        report, failed = exc.report, True
    except ValueError as exc:
        return _error(str(exc))
    print(f"[fetch] downloaded {report.downloaded} files ({_human(report.bytes)}) in "
          f"{time.monotonic() - start:.0f} s, {report.skipped} were already there")
    code = _prepare(raw, dest, todo_avatars, todo_clips, args)
    return 1 if failed else code


def cmd_prepare(args) -> int:
    dest, raw = _folders(args)
    if not raw.is_dir():
        return _error(f"no downloaded files in {raw}; run: bifold-characters fetch (or pass --raw DIR, "
                      f"e.g. a clone of the Microsoft-Rocketbox repository)", 1)
    try:
        avatars = rocketbox.check_names(args.avatars, rocketbox.AVATARS, "avatar") if args.avatars else None
    except ValueError as exc:
        return _error(str(exc))
    return _prepare(raw, dest, avatars, None, args)


def cmd_list(args) -> int:
    dest, _ = _folders(args)
    library = CharacterLibrary(dest)
    available = library.available()
    names = list(rocketbox.AVATARS) if args.all else list(available)
    if not names:
        print(f"{dest} has no prepared avatars yet. Try: bifold-characters fetch --set small")
        return 0
    print(f"{'avatar':22s} {'sex':6s} {'age':6s} {'skin':4s} {'outfit':12s} {'fit':3s} {'height':>6s} {'tris':>6s}")
    for name in names:
        meta = available.get(name) or rocketbox.avatar_metadata(name)
        prepared = name in available
        height = f"{meta['height']:.2f}" if prepared else "-"
        tris = str(meta.get("triangles", "-")) if prepared else "-"
        print(f"{name:22s} {meta.get('sex', '?'):6s} {meta.get('age_group', '?'):6s} "
              f"{str(meta.get('skin_tone', '-')):4s} {meta.get('outfit', '?'):12s} "
              f"{str(meta.get('conference_fit', '-')):3s} {height:>6s} {tris:>6s}"
              + ("" if prepared else "  (not prepared)"))
    print(library.describe())
    return 0


def cmd_info(args) -> int:
    dest, _ = _folders(args)
    library = CharacterLibrary(dest)
    available = library.available()
    if args.name in available:
        meta = available[args.name]
        for key in sorted(k for k in meta if k not in ("primitives", "lods", "meshes")):
            print(f"{key:15s} {meta[key]}")
        for p in meta.get("primitives", []):
            print(f"  part {p['name']}: {p['vertices']} vertices, {p['triangles']} triangles, "
                  f"texture {p['texture'] or '-'}")
        return 0
    clip_path = dest / "clips" / f"{args.name}.npz"
    if clip_path.is_file():
        clip = rocketbox.read_clip(clip_path)
        print(f"clip {clip.name}: {clip.frames} frames at {clip.fps:g} fps ({clip.duration:.2f} s), "
              f"{len(clip.bone_names)} bones, loop {clip.loop}, root {clip.root}")
        for key, value in sorted(clip.metadata.items()):
            print(f"{key:15s} {value}")
        return 0
    if args.name in rocketbox.AVATARS or args.name in rocketbox.CLIPS:
        option = f"--avatars {args.name}" if args.name in rocketbox.AVATARS else "--avatars --clips all"
        return _error(f"{args.name} is not prepared yet; run: bifold-characters fetch {option}", 1)
    known = dict.fromkeys([*available, *rocketbox.AVATARS, *rocketbox.CLIPS])
    close = difflib.get_close_matches(args.name, list(known), n=3)
    hint = f"; did you mean {' or '.join(close)}?" if close else "; see: bifold-characters list --all"
    return _error(f"no avatar or clip {args.name!r}{hint}", 1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bifold-characters", description="Download and prepare 3D people "
                                     "(Microsoft Rocketbox avatars and animations, MIT licence).")
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name, func, help_text):
        p = sub.add_parser(name, help=help_text, description=help_text)
        p.add_argument("--dest", help="characters folder (default: $BIFOLD_CHARACTERS, else "
                                      "~/.cache/bifold-demo/characters)")
        p.set_defaults(func=func)
        return p

    def convert_options(p):
        p.add_argument("--raw", help="folder for the downloaded Rocketbox files (default DEST/raw)")
        p.add_argument("--delete-raw", action="store_true", help="delete the downloaded files once converted")
        p.add_argument("--force", action="store_true", help="convert again even if already prepared")
        p.add_argument("--texture-size", type=int, default=1024,
                       help="largest texture side in pixels (default 1024)")

    f = add("fetch", cmd_fetch, "download avatars and animations, then prepare them")
    f.add_argument("--set", choices=rocketbox.AVATAR_SETS, default="conference",
                   help="conference (default: people that fit a trade fair), all (115 avatars) or small "
                        "(12 diverse avatars, a quick start)")
    f.add_argument("--avatars", nargs="*", metavar="NAME", help="these avatars instead of a set (none: only clips)")
    f.add_argument("--clips", choices=rocketbox.CLIP_SETS, help="animation clips (default: basic for --set small)")
    f.add_argument("--jobs", type=int, default=8, help="parallel downloads (default 8)")
    f.add_argument("--dry-run", action="store_true", help="only show what would be downloaded")
    convert_options(f)

    p = add("prepare", cmd_prepare, "convert already downloaded files (e.g. a clone of the Rocketbox repository)")
    p.add_argument("--avatars", nargs="+", metavar="NAME", help="only these avatars (default: all downloaded)")
    convert_options(p)

    lst = add("list", cmd_list, "list the prepared avatars")
    lst.add_argument("--all", action="store_true", help="also list the avatars that are not prepared yet")

    i = add("info", cmd_info, "show the details of an avatar or animation clip")
    i.add_argument("name", help="avatar or clip name, e.g. Business_Female_01 or m_walk_neutral_01")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\ninterrupted; run the same command again to resume", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
