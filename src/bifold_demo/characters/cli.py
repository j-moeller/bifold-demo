"""``bifold-characters``: download and prepare the 3D people (Microsoft Rocketbox avatars and animations)."""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

from . import rocketbox
from .library import CharacterLibrary, DamagedFileWarning, default_folder


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
    todo_avatars = [n for n in avatars if args.force or not rocketbox.is_prepared(dest, "avatar", n)]
    todo_clips = [n for n in clips if args.force or not rocketbox.is_prepared(dest, "clip", n)]
    paths = rocketbox.files_for(todo_avatars, todo_clips)
    _, local = rocketbox.raw_files(raw)  # what fetch and prepare use: --raw may also be an Assets folder
    present = [p for p in paths if local(p).is_file()]
    size = sum(rocketbox.estimate_size(p) for p in paths if not local(p).is_file())
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
    if failed:  # convert only what is complete; the error above already lists the rest
        todo_avatars = [n for n in todo_avatars if all(local(p).is_file() for p in rocketbox.AVATARS[n].files)]
        todo_clips = [n for n in todo_clips if local(rocketbox.CLIPS[n].path).is_file()]
        if not todo_avatars and not todo_clips:
            return 1
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
        print(f"{dest} has no prepared avatars yet.\n{library.describe()}")
        return 0
    print(f"{'avatar':22s} {'sex':6s} {'age':6s} {'skin':4s} {'outfit':12s} {'fit':3s} {'height':>6s} {'tris':>6s}")
    for name in names:
        meta = available.get(name) or rocketbox.avatar_metadata(name)
        prepared = name in available
        height = f"{meta['height']:.2f}" if prepared else "-"
        tris = str(meta.get("triangles", "-")) if prepared else "-"
        print(f"{name:22s} {meta.get('sex', '?'):6s} {meta.get('age_group', '?'):6s} "
              f"{meta.get('skin_tone', '-')!s:4s} {meta.get('outfit', '?'):12s} "
              f"{meta.get('conference_fit', '-')!s:3s} {height:>6s} {tris:>6s}"
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
    clip = None if args.name in rocketbox.AVATARS else library.clips().get(args.name)
    if clip is not None:
        print(f"clip {clip.name}: {clip.frames} frames at {clip.fps:g} fps ({clip.duration:.2f} s), "
              f"{len(clip.bone_names)} bones, loop {clip.loop}, root {clip.root}")
        for key, value in sorted(clip.metadata.items()):
            print(f"{key:15s} {value}")
        return 0
    return _error(library.explain(args.name), 1)


def cmd_outfit(args) -> int:
    import cv2

    from .outfit import regions as outfit_regions
    from .outfit.tailor import Outfit, presets, tailor_for

    if args.presets:
        for name, fills in presets().items():
            print(f"{name:18s} " + "; ".join(f"{region}: {_describe_fill(spec)}" for region, spec in fills.items()))
        return 0
    if not args.avatar:
        return _error("name an avatar (see: bifold-characters list), or use --presets")
    dest, _ = _folders(args)
    library = CharacterLibrary(dest)
    if args.avatar not in library.available():
        return _error(library.explain(args.avatar), 1)
    model = library.load(args.avatar)
    start = time.perf_counter()
    regions = outfit_regions.body_regions(model)
    tailor = tailor_for(model)
    tailor.layers, tailor.surface  # noqa: B018 - made now, so the timing below is the outfit's alone
    ready = time.perf_counter() - start
    counts = regions.counts()
    total = max(1, sum(counts.values()))
    print(f"{args.avatar}: {regions.size}x{regions.size} texels ready in {ready:.2f} s; "
          + ", ".join(f"{name} {n / total:.0%}" for name, n in counts.items()))
    for flag in ("mirrored", "overlapping"):
        if regions.flags.get(flag):
            print(f"  {flag} islands: {', '.join(map(str, regions.flags[flag]))}")
    for path in regions.flags.get("overrides", []):
        print(f"  overrides from {path}")
    atlas = tailor.original[..., :3]
    if args.regions:
        cv2.imwrite(args.regions, regions.image(atlas))
        print(f"  regions -> {args.regions} (skin, top, shirt, bottoms, shoes, other: peach, red, yellow, blue, "
              f"green, magenta)")
    if args.description is None:
        return 0
    try:
        outfit = Outfit.parse(args.description)
    except (ValueError, KeyError) as exc:
        return _error(str(exc))
    start = time.perf_counter()
    rgba = tailor.texture(outfit)
    print(f"  {outfit.name}: " + "; ".join(f"{r}: {_describe_fill(f.spec())}" for r, f in outfit.fills.items())
          + f" ({(time.perf_counter() - start) * 1000:.0f} ms)")
    if args.output:
        cv2.imwrite(args.output, cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
        print(f"  body texture -> {args.output}")
    if args.views:
        image = _outfit_views(model, tailor.primitive.name, rgba)
        cv2.imwrite(args.views, image)
        print(f"  front and back -> {args.views}")
    return 0


def _describe_fill(spec) -> str:
    if isinstance(spec, dict) and "pattern" in spec:
        colours = " and ".join(spec.get("colors", []))
        return f"{spec['pattern']} ({colours})" if colours else spec["pattern"]
    if isinstance(spec, dict) and ("swatch" in spec or "atlas" in spec):
        kind = "swatch" if "swatch" in spec else "atlas"
        source = str(spec[kind])
        return f"{kind} ({'an image' if source.startswith('data:') else source})"
    return str(spec)


def _outfit_views(model, primitive: str, rgba):
    """The avatar in its rest pose from the front and the back, wearing the body texture ``rgba`` (needs OpenGL)."""
    import numpy as np

    from ..scene import Camera, Scene, Texture
    from ..scene.renderer import Renderer
    from .character import Character
    from .model import CharacterState

    texture = Texture(rgba, name="outfit")
    height = float(model.metadata.get("height", 1.75))
    renderer = Renderer(360, 640)
    views = []
    for heading in (0.0, 180.0):
        person = Character(model, behaviour=lambda t, h=heading: CharacterState((0.0, 0.0, 0.0), h, []), name="p")
        person.texture_overrides[primitive] = texture
        scene = Scene(background=(235, 236, 240), ambient=0.55, light_direction=(-0.3, -0.6, -1.0))
        scene.add(person)
        scene.update(0.0, 0.0)
        scene.camera = Camera(position=(0.0, height * 0.55, height * 1.9), target=(0.0, height * 0.5, 0.0), fov=32)
        views.append(renderer.render(scene))
    return np.hstack(views)


def _pose_names() -> list[str]:
    from . import yoga

    return list(yoga.POSES)


def _ignore_interrupts() -> None:
    """Pool workers leave Ctrl+C to the command, which stops them."""
    import signal

    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _yoga_job(dest: str, name: str, pose: str) -> tuple[str, str, float, str | None]:
    """Solve (or find cached) one avatar's pose; runs in a worker process."""
    from . import yoga_clips

    start = time.perf_counter()
    try:
        model = CharacterLibrary(dest).load(name, textures="none")
        path = yoga_clips.cache_path(Path(dest) / "yoga", model, pose)
        yoga_clips.load_or_build(model, pose, path, yoga_clips.model_key(model), strict=True)
    except Exception as exc:  # noqa: BLE001 - reported per clip
        return name, pose, time.perf_counter() - start, f"{type(exc).__name__}: {exc}"
    return name, pose, time.perf_counter() - start, None


def cmd_yoga(args) -> int:
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from . import yoga

    dest, _ = _folders(args)
    library = CharacterLibrary(dest)
    available = library.available()
    names = args.avatars or sorted(available)
    missing = [n for n in names if n not in available]
    if missing:
        return _error(library.explain(missing[0]), 1)
    poses = args.poses or list(yoga.POSES)
    jobs = [(str(dest), name, pose) for name in names for pose in poses]
    report, failed = _progress("yoga"), 0
    pool = ProcessPoolExecutor(max_workers=max(1, args.jobs), mp_context=multiprocessing.get_context("spawn"),
                               initializer=_ignore_interrupts)  # spawn: like on macOS, and safe with threads
    try:
        futures = [pool.submit(_yoga_job, *job) for job in jobs]
        for done, future in enumerate(as_completed(futures), 1):
            name, pose, seconds, error = future.result()
            failed += error is not None
            report(done, len(jobs), f"{name} {pose}", error or f"{seconds:.1f} s")
    except BaseException:  # Ctrl+C: stop now, not after the queue (the clips done so far are kept)
        processes = list(getattr(pool, "_processes", {}).values())
        pool.shutdown(wait=False, cancel_futures=True)
        for process in processes:
            process.terminate()
        raise
    pool.shutdown()
    ready = len(jobs) - failed
    print(f"[yoga] {ready} {'clip' if ready == 1 else 'clips'} ready in {dest / 'yoga'}"
          + (f", {failed} failed" if failed else ""))
    return 1 if failed else 0


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
        p.add_argument("--raw", help="folder for the downloaded Rocketbox files (default DEST/raw); also a clone of "
                                     "the Rocketbox repository or its Assets folder")
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

    y = add("yoga", cmd_yoga, "solve the yoga poses of the prepared avatars ahead of time, so the demo starts them at "
                              "once (otherwise each is solved the first time it is asked for)")
    y.add_argument("--avatars", nargs="+", metavar="NAME", help="only these avatars (default: all prepared)")
    y.add_argument("--poses", nargs="+", choices=_pose_names(), metavar="POSE",
                   help=f"only these poses (default: all of {', '.join(_pose_names())})")
    y.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                   help="poses solved at the same time (default: one per CPU core but one)")

    o = add("outfit", cmd_outfit, "dress an avatar in an outfit (a preset or a description) and write its new body "
                                  "texture; also shows the garment regions found on its texture")
    o.add_argument("avatar", nargs="?", help="avatar name, e.g. Business_Male_02")
    o.add_argument("description", nargs="?", help='a preset ("business navy") or a description: "red tartan", '
                                                  '"navy pinstripe suit, white shirt, black shoes"')
    o.add_argument("-o", "--output", help="write the body texture here (PNG)")
    o.add_argument("--regions", help="write a picture of the garment regions over the texture here")
    o.add_argument("--views", help="write the avatar from the front and the back here (needs OpenGL)")
    o.add_argument("--presets", action="store_true", help="list the outfit presets")

    i = add("info", cmd_info, "show the details of an avatar or animation clip")
    i.add_argument("name", help="avatar or clip name, e.g. Business_Female_01 or m_walk_neutral_01")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DamagedFileWarning)  # the commands report damaged files themselves
            return args.func(args)
    except KeyboardInterrupt:
        print("\ninterrupted; run the same command again to resume", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
