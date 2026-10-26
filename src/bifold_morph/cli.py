"""``bifold-morph``: the live demo, and how its network was made.

    bifold-morph                          # the live demo on the webcam (same as `bifold-morph run`)
    bifold-morph run --source clip.mp4    # ... on a video file or a photo
    bifold-morph edit me.jpg -o her.jpg --gender -1
    bifold-morph fetch                    # download what the app needs (about 90 MB)
    bifold-morph train --work ~/morph     # make the student network again (hours on a CPU, less on a GPU)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import models

COMMANDS = ("run", "fetch", "edit", "train")


def _device_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("--device", default="auto", help="auto (CUDA, else Apple's MPS, else CPU), cpu, mps, cuda, ...")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bifold-morph", description="Live webcam face changing: age and gender "
                                     "sliders (a U-Net distilled from StyleGAN2 edits).")
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="the live demo (the default)")
    run.add_argument("--source", default="0", help="camera number, video file or image file (default 0)")
    _device_arg(run)
    run.add_argument("--max-faces", type=int, default=2, help="change at most this many faces, the biggest (2)")
    run.add_argument("--no-readout", action="store_true", help="do not load the judges (no gender/age read-out)")
    run.add_argument("--half", action="store_true", help="16-bit floats on the GPU")
    run.add_argument("--width", type=int, default=1280, help="camera resolution asked for (1280 x 720)")
    run.add_argument("--height", type=int, default=720)
    run.add_argument("--no-mirror", action="store_true", help="do not mirror the webcam")
    run.add_argument("--fullscreen", action="store_true")
    run.add_argument("--age", type=float, default=0.0, help="start with the age slider here (years)")
    run.add_argument("--gender", type=float, default=0.0, help="start with the gender slider here (-1 female ... 1 male)")
    run.add_argument("--student", type=Path, help="another student network (.pt)")

    sub.add_parser("fetch", help="download the models the app needs")

    edit = sub.add_parser("edit", help="change the faces in a photo")
    edit.add_argument("image")
    edit.add_argument("-o", "--output", help="where to save it (default: <image>_morph.png)")
    edit.add_argument("--age", type=float, default=0.0, help="years older (negative: younger)")
    edit.add_argument("--gender", type=float, default=0.0, help="-1 towards female ... +1 towards male")
    edit.add_argument("--student", type=Path)
    _device_arg(edit)

    train = sub.add_parser("train", help="make the student network: label faces, draw pairs, train")
    train.add_argument("--work", type=Path, required=True, help="folder for the labels, pairs and checkpoints")
    _device_arg(train)
    train.add_argument("--labels", type=int, default=2000, help="random faces to label for the directions (2000)")
    train.add_argument("--pairs", type=int, default=4000, help="before/after pairs to train on (4000)")
    train.add_argument("--steps", type=int, default=4000, help="training steps (4000, batch 8)")
    return parser


def _train(work: Path, device: str, labels: int, pairs: int, steps: int) -> None:
    import numpy as np

    from .app import pick_device
    from .distill import Directions, fit_directions, make_pairs, sample_labels, train
    from .gfpgan import GFPGANv1
    from .nets import FairFace

    dev = pick_device(device)
    work.mkdir(parents=True, exist_ok=True)
    generator = GFPGANv1.load(models.ensure(models.GFPGAN), dev).stylegan_decoder
    fairface_path = models.ensure(models.FAIRFACE)
    if not (work / "directions.npz").is_file():
        w, female, age = sample_labels(generator, FairFace(fairface_path, dev), labels)
        np.savez(work / "labels.npz", w=w, female=female, age=age)
        fit_directions(w, female, age).save(work / "directions.npz")
    directions = Directions.load(work / "directions.npz")
    print(f"[train] directions: female-male gap {directions.gap:.2f} units, {directions.years:.1f} years per unit")
    if not (work / "pairs" / "amounts.npy").is_file():
        make_pairs(generator, directions, pairs, work / "pairs")
    del generator
    train(work / "pairs", fairface_path, work / "student.pt", steps=steps, device=dev)
    print(f"[train] done: {work / 'student.pt'} (use it with --student, or copy it to {models.STUDENT})")


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or (argv[0] not in COMMANDS and argv[0] not in ("-h", "--help")):
        argv.insert(0, "run")
    args = _parser().parse_args(argv)
    try:
        if args.command == "run":
            from .app import run

            run(source=args.source, device=args.device, max_faces=args.max_faces, readout=not args.no_readout,
                half=args.half, width=args.width, height=args.height, mirror=False if args.no_mirror else None,
                fullscreen=args.fullscreen, age=args.age, gender=args.gender, student=args.student)
        elif args.command == "fetch":
            for path in models.ensure_all(models.APP).values():
                print(f"[models] {path} ({path.stat().st_size / 1e6:.1f} MB)")
        elif args.command == "edit":
            from .app import edit_file
            from .ui import describe_label

            output = args.output or str(Path(args.image).with_name(Path(args.image).stem + "_morph.png"))
            result = edit_file(args.image, output, args.age, args.gender, args.device, student=args.student)
            for face in result.faces:
                label = describe_label(face.before, face.after, edited=True)
                print(f"[morph] face at {tuple(round(v) for v in face.box)}: {label or 'changed'}")
            print(f"[morph] {len(result.faces)} face(s) changed; saved {output}")
        elif args.command == "train":
            _train(args.work, args.device, args.labels, args.pairs, args.steps)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"bifold-morph: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
