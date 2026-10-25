"""``bifold-people``: build a library of people cut-outs for the 3D scenes."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2

from .library import DEFAULT_FOLDER, PeopleLibrary


def _size(text: str) -> tuple[int, int]:
    try:
        w, h = text.lower().split("x")
        return int(w), int(h)
    except ValueError:
        raise argparse.ArgumentTypeError("size must look like 768x1344") from None


def cmd_generate(args) -> int:
    from .cutout import PersonSegmenter
    from .generator import ImageGenerator, PeopleGenerator, get_preset, preset_with

    try:
        preset = preset_with(get_preset(args.model), args.steps, args.size)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        images = ImageGenerator(preset, device=args.device)
    except ImportError:
        print('error: image generation needs extra packages: pip install -e ".[people]"', file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - download/auth problems surface as many exception types
        print(f"error: cannot load {preset.repo}: {type(exc).__name__}: {exc}\n"
              f"  - check your internet connection (the weights come from huggingface.co)\n"
              f"  - if the model page says it is gated, accept its terms there and run: hf auth login\n"
              f"  - on a Mac with little memory, try: --model sd-turbo", file=sys.stderr)
        return 1
    seed = args.seed if args.seed is not None else random.randrange(1_000_000)
    print(f"[people] generating {args.count} people with {preset.repo}, seed {seed}")
    library = PeopleLibrary(args.out)
    raw = Path(args.out) / "_raw" if args.keep_raw else None
    generator = PeopleGenerator(images, PersonSegmenter(), library, raw_folder=raw)
    pose = None if args.pose == "mixed" else args.pose
    made = generator.generate(args.count, seed=seed, pose=pose, attempts=args.attempts)
    print(f"[people] added {len(made)} people; the library {args.out}/ now has {len(library)}")
    return 0 if made or args.count == 0 else 1


def cmd_import(args) -> int:
    from .cutout import PersonSegmenter

    segmenter, library, added = PersonSegmenter(), PeopleLibrary(args.out), 0
    for file in args.photos:
        image = cv2.imread(file)
        if image is None:
            print(f"[people] cannot read {file}")
            continue
        cutouts = [c for c in segmenter.cutouts(image) if not ({"left", "right"} & c.touches)]
        if not args.all:
            cutouts = cutouts[:1]
        if not cutouts:
            print(f"[people] no (whole) person found in {file}")
        for cutout in cutouts:
            person = library.add(cutout, name=Path(file).stem, height=args.height if cutout.full_body else 0.95,
                                 pose=args.pose, facing="front" if args.pose == "standing" else args.facing,
                                 source=f"imported:{Path(file).name}")
            kind = "full body" if person.full_body else "cut off at the waist (used behind counters)"
            print(f"[people] {file} -> {person.path} ({kind})")
            added += 1
    return 0 if added else 1


def cmd_list(args) -> int:
    people = PeopleLibrary(args.out).people()
    if not people:
        print(f"{args.out}/ has no people yet. Try: bifold-people generate -n 20")
        return 0
    for p in people:
        body = "full body" if p.full_body else "bust"
        print(f"{p.name:30s} {p.pose:9s} {p.facing:6s} {p.height:.2f} m  {body:9s} {p.source}")
    print(f"{len(people)} people in {args.out}/")
    return 0


def cmd_sheet(args) -> int:
    sheet = PeopleLibrary(args.out).contact_sheet()
    if sheet is None:
        print(f"{args.out}/ has no people yet")
        return 1
    path = Path(args.out) / "_contact_sheet.jpg"
    cv2.imwrite(str(path), sheet)
    print(f"wrote {path}")
    return 0


def cmd_models(args) -> int:
    from .generator import DEFAULT_PRESET, PRESETS

    for name, p in PRESETS.items():
        default = "  (default)" if name == DEFAULT_PRESET else ""
        print(f"{name}{default}\n    {p.repo}, {p.steps} steps, {p.width}x{p.height}\n"
              f"    memory: {p.memory}\n    license: {p.license}")
    print("Any other Hugging Face text-to-image model id works too, e.g. --model owner/name --steps 8")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bifold-people", description="Create people cut-outs for the 3D scenes.")
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name, func, help_text):
        p = sub.add_parser(name, help=help_text, description=help_text)
        p.add_argument("--out", default=DEFAULT_FOLDER, help=f"people folder (default {DEFAULT_FOLDER}/)")
        p.set_defaults(func=func)
        return p

    g = add("generate", cmd_generate, "generate new people with a text-to-image model")
    g.add_argument("-n", "--count", type=int, default=20, help="how many people (default 20)")
    g.add_argument("--model", default="flux2-klein", help="preset (see 'bifold-people models') or Hugging Face id")
    g.add_argument("--pose", choices=["mixed", "standing", "walking"], default="mixed",
                   help="standing = facing the camera, walking = side view (default: a mix)")
    g.add_argument("--seed", type=int, help="start seed, for reproducible people (default: random)")
    g.add_argument("--device", default="auto", help="auto, mps, cuda or cpu")
    g.add_argument("--steps", type=int, help="override the model's number of steps")
    g.add_argument("--size", type=_size, help="override the image size, e.g. 768x1344")
    g.add_argument("--attempts", type=int, default=3, help="images to try per person before giving up (default 3)")
    g.add_argument("--keep-raw", action="store_true", help="also keep the uncut images in <out>/_raw/")

    i = add("import", cmd_import, "cut people out of your own photos and add them")
    i.add_argument("photos", nargs="+", help="image files")
    i.add_argument("--all", action="store_true", help="add every person in a photo, not just the largest")
    i.add_argument("--height", type=float, default=1.72, help="height in metres (default 1.72)")
    i.add_argument("--pose", choices=["standing", "walking"], default="standing")
    i.add_argument("--facing", choices=["left", "right"], default="left", help="direction a walking person faces")

    add("list", cmd_list, "list the people in the library")
    add("sheet", cmd_sheet, "write a contact sheet of all people to <out>/_contact_sheet.jpg")
    add("models", cmd_models, "show the image models you can use")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
