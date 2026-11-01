"""``bifold-faces``: teach the face recogniser who is who, and check it.

    bifold-faces fetch [--profile P]           download a profile's face models into models/ (default: best)
    bifold-faces fetch --model ID ... | --all  ... or these models, or every model
    bifold-faces add "Ada Lovelace" a.jpg b.jpg   copy photos of a person into faces/Ada Lovelace/
    bifold-faces list [--only models|people]   the face models (licences, profiles, fetched or not) and the known
                                               people with how many photos each
    bifold-faces who photo.jpg                 who the faces in a photo are
    bifold-faces check [--profile P | --model ID ... | --all] [--speed] [--photos DIR] [--pack lfw.bin ...]
                                               check the models: files, graphs, speed, colour order, accuracy
    bifold-faces landmarks [AVATAR ...]        MediaPipe's face landmarks on avatars' heads (head textures, stage 0)
    bifold-faces register PHOTO[:VIEW] ... --avatar NAME [--out DIR] [--segment]
                                               register photos of a head to an avatar's head (stage 1)
    bifold-faces headtex PHOTO[:VIEW] ... --avatar NAME -o head.png [--views render.png] [--debug DIR]
                                               a head texture for an avatar from the photos (stages 0-4)
    bifold-faces headtex PHOTO[:VIEW] ... --avatar NAME -o head.png --full [--check] [--character NAME]
                                               head, body and hair textures (all stages), checked, saved

``fetch --headtex`` also downloads MediaPipe's face landmarker and segmenter (MediaPipe itself is optional, see the
README). ``--face-profile`` (or ``BIFOLD_FACE_PROFILE``) chooses the profile: ``best``, ``licensed`` or
``commercial`` (see :mod:`.registry`).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from . import models, registry
from .gallery import Gallery


def _gallery(args) -> Gallery:
    from .engine import FaceEngine

    engine = FaceEngine(args.models, profile=args.face_profile)
    for note in engine.notes:
        print(f"[faces] {note}", file=sys.stderr)
    return Gallery(args.faces, engine, threshold=args.threshold)


def cmd_fetch(args) -> int:
    if args.all or args.model:
        found = registry.fetch(folder=args.models, ids=args.model or list(registry.MODELS))
    else:
        name = registry.profile_name(args.profile or args.face_profile)
        print(f"profile {name}: {registry.PROFILES[name].summary}")
        found = registry.fetch(name, args.models)
    status = 0
    for model_id, result in found.items():
        if isinstance(result, Exception):
            print(f"{model_id}: not fetched: {result}", file=sys.stderr)
            if args.model:  # asked for by name: a failure is an error (a profile's fallback covers it)
                status = 1
        else:
            print(f"{model_id}: {result}")
    if args.headtex:
        from ..characters.headtex import detect

        print(f"{models.DETECTOR.name}: {models.ensure(models.DETECTOR, args.models)}")
        for model in detect.FILES:
            print(f"{model.name}: {detect.model_path(model, args.models)}")
    return status


def cmd_add(args) -> int:
    gallery = _gallery(args)
    problems = gallery.add(args.name, args.photos)
    for problem in problems:
        print(f"skipped {problem}", file=sys.stderr)
    known = gallery.people().get(args.name.strip(), 0)
    print(f"{args.name.strip()}: {known} photo(s) in {gallery.folder / args.name.strip()}")
    return 1 if problems and len(problems) == len(args.photos) else 0


def cmd_list(args) -> int:
    if args.only != "people":
        _list_models(args)
    if args.only == "models":
        return 0
    gallery = _gallery(args)
    gallery.refresh(force=True)
    people = gallery.people()
    print(f"Known people ({gallery.engine.describe()}; a face is someone from a score of {gallery.threshold:.2f})")
    if not people:
        print(f"nobody known yet: put photos in {gallery.folder}/<name>/ (or run: bifold-faces add NAME PHOTOS)")
    for name, count in people.items():
        poses = ", ".join(f"{n} {pose}" for pose, n in sorted(gallery.poses(name).items()))
        print(f"{name}: {count} photo(s) ({poses})")
    for path, status, problems in gallery.report():
        where = path if status == "near-twin" else path.relative_to(gallery.folder)
        print(f"{status}: {where}: {'; '.join(problems)}", file=sys.stderr)
    return 0


def _list_models(args) -> None:
    from .. import downloads

    name = registry.profile_name(args.face_profile)
    folder = models.models_dir(args.models)
    print(f"Face models in {folder} (profile: {name}; choose with --face-profile or {registry.PROFILE_ENV})")
    for spec in registry.MODELS.values():
        here = downloads.present(spec.file, folder) is not None
        mark = "*" if spec.id in registry.PROFILES[name].models() else " "
        where = "fetched" if here else "not fetched" + (f", needs {spec.needs}" if spec.needs else "")
        print(f"{mark} {spec.id:<20} {spec.task:<9} {spec.title} [{where}]")
        use = "commercial use allowed by the licence" if spec.commercial else "research use only"
        print(f"      licence: {spec.licence}; {use}")
        print(f"      trained on: {spec.data}; profiles: {', '.join(spec.profiles) or 'none (optional)'}")
    print("* the profile's models (an embedder after the first is its fallback)")
    print()


def cmd_check(args) -> int:
    import json

    from .. import downloads
    from . import checks
    from .detect import ScrfdDetector, YuNetDetector

    folder = models.models_dir(args.models)
    if args.model:
        ids = args.model
    elif args.all:
        ids = [i for i, spec in registry.MODELS.items() if downloads.present(spec.file, folder) is not None]
    else:
        ids = list(registry.profile(args.profile or args.face_profile).models())
    crops = labels = None
    if args.photos:
        detector = None
        for spec, make in ((registry.SCRFD, ScrfdDetector), (registry.YUNET, YuNetDetector)):
            if downloads.present(spec.file, folder) is not None:
                detector = make(folder / spec.file.name, spec)
                break
        if detector is None:
            raise FileNotFoundError("the colour-order check needs a face detector: bifold-faces fetch")
        crops, labels = checks.labelled_faces(args.photos, detector)
        print(f"{len(labels)} faces of {len(set(labels.tolist()))} people in {args.photos}")
    reports = []
    for model_id in ids:
        spec = registry.model(model_id)
        mine = [checks.check_file(spec, folder)]
        path = folder / spec.file.name
        if path.is_file():
            mine.append(checks.check_graph(spec, path))
            if args.speed:
                timing = checks.speed(spec, path)
                mine.append(checks.Report(f"{spec.id}: speed", lines=[", ".join(
                    f"{k}: {v:.1f} ms" for k, v in timing.items())], values=timing))
            if spec.task == "embedder" and spec.form == "onnxruntime" and crops is not None:
                mine.append(checks.colour_order(spec, path, crops, labels))
            if spec.task == "embedder":
                mine += [checks.published_accuracy(spec, path, pack) for pack in args.pack or ()]
        for report in mine:
            print(report)
        reports += mine
    if args.json:
        Path(args.json).write_text(json.dumps({r.name: {"ok": r.ok, "lines": r.lines, "values": r.values}
                                               for r in reports}, indent=1, default=str))
    failed = [r.name for r in reports if not r.ok]
    print(f"{len(reports) - len(failed)} of {len(reports)} checks passed" + (f"; failed: {', '.join(failed)}"
                                                                            if failed else ""))
    return 1 if failed else 0


def cmd_who(args) -> int:
    gallery = _gallery(args)
    gallery.refresh(force=True)
    print(f"({gallery.engine.describe()}; a face is someone from a score of {gallery.threshold:.2f})")
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
                        help="similarity a face needs to count as someone (default: the model's own, e.g. 0.363 for "
                             "SFace)")
    parser.add_argument("--face-profile", default=None, choices=tuple(registry.PROFILES),
                        help=f"the face models to use (default: ${registry.PROFILE_ENV}, else best)")
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch", help="download the face models")
    fetch.add_argument("--profile", default=None, choices=tuple(registry.PROFILES),
                       help="the profile whose models to fetch (default: the profile in use)")
    fetch.add_argument("--model", nargs="+", default=None, choices=tuple(registry.MODELS), metavar="ID",
                       help="fetch these models instead (the ids bifold-faces list shows)")
    fetch.add_argument("--all", action="store_true", help="fetch every model (AuraFace needs huggingface.co)")
    fetch.add_argument("--headtex", action="store_true", help="also MediaPipe's models for head textures")
    fetch.set_defaults(run=cmd_fetch)
    add = sub.add_parser("add", help="add photos of a person")
    add.add_argument("name")
    add.add_argument("photos", nargs="+")
    add.set_defaults(run=cmd_add)
    listing = sub.add_parser("list", help="the face models and the known people")
    listing.add_argument("--only", choices=("models", "people"), default=None, help="list only these")
    listing.set_defaults(run=cmd_list)
    who = sub.add_parser("who", help="who is in these photos")
    who.add_argument("photos", nargs="+")
    who.set_defaults(run=cmd_who)
    check = sub.add_parser("check", help="check the face models: files, graphs, speed, colour order, accuracy")
    check.add_argument("--profile", default=None, choices=tuple(registry.PROFILES),
                       help="check this profile's models (default: the profile in use)")
    check.add_argument("--model", nargs="+", default=None, choices=tuple(registry.MODELS), metavar="ID",
                       help="check these models instead")
    check.add_argument("--all", action="store_true", help="check every model that is fetched")
    check.add_argument("--speed", action="store_true", help="also time them (ms per crop at batch 1 and 8)")
    check.add_argument("--photos", default=None, metavar="DIR",
                       help="a folder of people (as faces/) for the colour-order check")
    check.add_argument("--pack", nargs="+", default=None, metavar="FILE",
                       help="test packs (lfw.bin, cfp_fp.bin, agedb_30.bin) for the published-accuracy check")
    check.add_argument("--json", default=None, metavar="FILE", help="also write the results as JSON")
    check.set_defaults(run=cmd_check)
    from ..characters.headtex.cli import add_commands

    add_commands(sub)
    args = parser.parse_args(argv)
    try:
        return args.run(args)
    except (OSError, ValueError, ImportError) as exc:  # ImportError: MediaPipe is missing (the message says how)
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
