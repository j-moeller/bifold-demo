"""``bifold-morph``: the live demo and its setup.

    bifold-morph setup                    # once: download the models, find the age and gender directions
    bifold-morph                          # the live demo on the webcam (same as `bifold-morph run`)
    bifold-morph run --source clip.mp4    # ... on a video file or a photo
    bifold-morph edit me.jpg -o old.jpg --age 30 --gender -1
    bifold-morph fetch                    # only download the models
    bifold-morph directions --samples 3000
"""

from __future__ import annotations

import argparse
import sys
import time

from . import models

COMMANDS = ("run", "setup", "fetch", "directions", "edit")


def _device_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("--device", default="auto", help="auto (CUDA, else Apple's MPS, else CPU), cpu, mps, cuda, ...")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bifold-morph", description="Live webcam face editing: age and gender "
                                     "sliders (e4e + StyleGAN2, directions found with MiVOLO).")
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="the live demo (the default)")
    run.add_argument("--source", default="0", help="camera number, video file or image file (default 0)")
    _device_arg(run)
    run.add_argument("--size", type=int, default=256, choices=(128, 256, 512, 1024),
                     help="resolution StyleGAN2 draws the faces at (default 256: fast; 1024: sharpest)")
    run.add_argument("--max-faces", type=int, default=2, help="edit at most this many faces, the biggest (2)")
    run.add_argument("--no-mivolo", action="store_true", help="do not load MiVOLO (no age and gender read-out)")
    run.add_argument("--half", action="store_true", help="16-bit floats on the GPU: faster, less tested")
    run.add_argument("--width", type=int, default=1280, help="camera resolution asked for (1280 x 720)")
    run.add_argument("--height", type=int, default=720)
    run.add_argument("--no-mirror", action="store_true", help="do not mirror the webcam")
    run.add_argument("--fullscreen", action="store_true")
    run.add_argument("--age", type=float, default=0.0, help="start with the age slider here (years)")
    run.add_argument("--gender", type=float, default=0.0, help="start with the gender slider here (-1 ... 1)")
    run.add_argument("--replace", action="store_true", help="start in the mode where the generated face replaces "
                     "the real one")

    setup = sub.add_parser("setup", help="download the models and find the directions (once)")
    _device_arg(setup)
    setup.add_argument("--samples", type=int, default=3000, help="random faces to label (3000)")
    setup.add_argument("--force", action="store_true", help="find the directions again even if they exist")

    sub.add_parser("fetch", help="download the models only")

    directions = sub.add_parser("directions", help="find the age and gender directions with MiVOLO")
    _device_arg(directions)
    directions.add_argument("--samples", type=int, default=3000, help="random faces to label (3000)")
    directions.add_argument("--size", type=int, default=1024, choices=(256, 512, 1024),
                            help="resolution the faces are drawn at for MiVOLO (1024)")
    directions.add_argument("--batch", type=int, default=4)
    directions.add_argument("--seed", type=int, default=0)

    edit = sub.add_parser("edit", help="edit the faces in a photo")
    edit.add_argument("image")
    edit.add_argument("-o", "--output", help="where to save it (default: <image>_morph.png)")
    edit.add_argument("--age", type=float, default=0.0, help="years older (negative: younger)")
    edit.add_argument("--gender", type=float, default=0.0, help="-1 towards female ... +1 towards male")
    edit.add_argument("--replace", action="store_true", help="the generated face replaces the real one")
    edit.add_argument("--size", type=int, default=1024, choices=(256, 512, 1024))
    _device_arg(edit)
    return parser


def _fetch() -> None:
    for model in models.FILES:
        path = models.ensure(model)
        print(f"[models] {model.name}: {path} ({path.stat().st_size / 1e6:.1f} MB)")


def _directions(device: str, samples: int, size: int = 1024, batch: int = 4, seed: int = 0) -> None:
    import cv2

    from .app import pick_device
    from .directions import FaceLab, compute, preview
    from .e4e import E4E
    from .faces import FaceDetector
    from .mivolo import MiVOLO

    dev = pick_device(device)
    print(f"[directions] loading e4e + StyleGAN2 and MiVOLO on {dev} ...")
    e4e = E4E.load(models.ensure(models.E4E), dev)
    lab = FaceLab(e4e, MiVOLO.load(models.ensure(models.MIVOLO), dev), FaceDetector(models.ensure(models.YUNET)),
                  dev, size=size, batch=batch)
    start = time.perf_counter()
    result = compute(lab, samples=samples, seed=seed)
    path = result.save(models.directions_path())
    grid = preview(lab, result)
    preview_path = models.models_dir() / models.PREVIEW
    cv2.imwrite(str(preview_path), grid)
    print(f"[directions] saved {path} after {(time.perf_counter() - start) / 60:.1f} min; "
          f"a preview of what they do: {preview_path}")


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or (argv[0] not in COMMANDS and argv[0] not in ("-h", "--help")):
        argv.insert(0, "run")
    args = _parser().parse_args(argv)
    try:
        if args.command == "run":
            from .app import run

            run(source=args.source, device=args.device, size=args.size, max_faces=args.max_faces,
                readout=not args.no_mivolo, half=args.half, width=args.width, height=args.height,
                mirror=False if args.no_mirror else None, fullscreen=args.fullscreen, age=args.age,
                gender=args.gender, replace=args.replace)
        elif args.command == "fetch":
            _fetch()
        elif args.command == "setup":
            _fetch()
            if models.directions_path().is_file() and not args.force:
                print(f"[directions] already there: {models.directions_path()} (--force to find them again)")
            else:
                _directions(args.device, args.samples)
            print("[morph] ready: run `bifold-morph`")
        elif args.command == "directions":
            _directions(args.device, args.samples, args.size, args.batch, args.seed)
        elif args.command == "edit":
            from pathlib import Path

            from .app import edit_file
            from .ui import describe_label

            output = args.output or str(Path(args.image).with_name(Path(args.image).stem + "_morph.png"))
            result = edit_file(args.image, output, args.age, args.gender, args.device, args.size, args.replace)
            for face in result.faces:
                label = describe_label(face.before, face.after, edited=True)
                print(f"[morph] face at {tuple(round(v) for v in face.box)}: {label or 'edited'}")
            print(f"[morph] {len(result.faces)} face(s) edited; saved {output}")
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"bifold-morph: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
