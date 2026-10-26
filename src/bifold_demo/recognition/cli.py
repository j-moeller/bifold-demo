"""``bifold-faces``: teach the face recogniser who is who, and check it.

    bifold-faces fetch                         download the face models into models/
    bifold-faces add "Ada Lovelace" a.jpg b.jpg   copy photos of a person into faces/Ada Lovelace/
    bifold-faces list                          the known people and how many photos each
    bifold-faces who photo.jpg                 who the faces in a photo are
    bifold-faces landmarks [AVATAR ...]        MediaPipe's face landmarks on avatars' heads (head textures, stage 0)
    bifold-faces register PHOTO[:VIEW] ... --avatar NAME [--out DIR] [--segment]
                                               register photos of a head to an avatar's head (stage 1)

``fetch --headtex`` also downloads MediaPipe's face landmarker and segmenter (MediaPipe itself is optional, see the
README).
"""

from __future__ import annotations

import argparse
import sys

import cv2

from . import models
from .gallery import Gallery


def _gallery(args) -> Gallery:
    from .engine import FaceEngine

    return Gallery(args.faces, FaceEngine(args.models), threshold=args.threshold)


def cmd_fetch(args) -> int:
    for name, path in models.ensure_all(args.models).items():
        print(f"{name}: {path}")
    if args.headtex:
        from ..characters.headtex import detect

        for model in detect.FILES:
            print(f"{model.name}: {detect.model_path(model, args.models)}")
    return 0


def cmd_add(args) -> int:
    gallery = _gallery(args)
    problems = gallery.add(args.name, args.photos)
    for problem in problems:
        print(f"skipped {problem}", file=sys.stderr)
    known = gallery.people().get(args.name.strip(), 0)
    print(f"{args.name.strip()}: {known} photo(s) in {gallery.folder / args.name.strip()}")
    return 1 if problems and len(problems) == len(args.photos) else 0


def cmd_list(args) -> int:
    gallery = _gallery(args)
    gallery.refresh(force=True)
    people = gallery.people()
    if not people:
        print(f"nobody known yet: put photos in {gallery.folder}/<name>/ (or run: bifold-faces add NAME PHOTOS)")
    for name, count in people.items():
        print(f"{name}: {count} photo(s)")
    for path in gallery.problems():
        print(f"no face found in {path}", file=sys.stderr)
    return 0


def cmd_who(args) -> int:
    gallery = _gallery(args)
    gallery.refresh(force=True)
    status = 0
    for photo in args.photos:
        image = cv2.imread(photo, cv2.IMREAD_COLOR)
        if image is None:
            print(f"{photo}: not an image", file=sys.stderr)
            status = 1
            continue
        faces = gallery.engine.faces(image, min_size=0)
        if not faces:
            print(f"{photo}: no face found")
        for face, vector in faces:
            match = gallery.match(vector)
            x, y, w, h = (round(v) for v in face.box)
            who = match.name or "unknown"
            print(f"{photo}: {who} (similarity {match.score:.2f}) at x={x} y={y} {w}x{h}")
    return status


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="bifold-faces", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--faces", default="faces", help="the folder of known people (default: faces)")
    parser.add_argument("--models", default=None, help="where the face models are kept (default: models)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="similarity a face needs to count as someone (default: 0.363)")
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch", help="download the face models")
    fetch.add_argument("--headtex", action="store_true", help="also MediaPipe's models for head textures")
    fetch.set_defaults(run=cmd_fetch)
    add = sub.add_parser("add", help="add photos of a person")
    add.add_argument("name")
    add.add_argument("photos", nargs="+")
    add.set_defaults(run=cmd_add)
    sub.add_parser("list", help="the known people").set_defaults(run=cmd_list)
    who = sub.add_parser("who", help="who is in these photos")
    who.add_argument("photos", nargs="+")
    who.set_defaults(run=cmd_who)
    from ..characters.headtex.cli import add_commands

    add_commands(sub)
    args = parser.parse_args(argv)
    if args.threshold is None:
        from .gallery import THRESHOLD

        args.threshold = THRESHOLD
    try:
        return args.run(args)
    except (OSError, ValueError, ImportError) as exc:  # ImportError: MediaPipe is missing (the message says how)
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
