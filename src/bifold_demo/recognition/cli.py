"""``bifold-faces``: teach the face recogniser who is who, and check it.

    bifold-faces fetch [--profile P]           download a profile's face models into models/ (default: best)
    bifold-faces fetch --model ID ... | --all  ... or these models, or every model
    bifold-faces add "Ada Lovelace" a.jpg b.jpg   copy photos of a person into faces/Ada Lovelace/
    bifold-faces list [--only models|people]   the face models (licences, profiles, fetched or not) and the known
                                               people with how many photos each
    bifold-faces who photo.jpg                 who the faces in a photo are
    bifold-faces check [--profile P | --model ID ... | --all] [--speed] [--photos DIR] [--pack lfw.bin ...]
                                               check the models: files, graphs, speed, colour order, accuracy
    bifold-faces calibrate [--profiles P ...] [--characters DIR] [--avatars NAME ...] [--refit]
                                               calibrate the decisions of each profile's recogniser from renders of
                                               avatars (writes models/calibration/<model>.json)
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
            # asked for by name: any failure is an error; --all: only a model that needs no other site (AuraFace
            # needs Hugging Face, which may be out of reach); a profile: its fallback covers it
            if args.model or (args.all and not registry.model(model_id).needs):
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
        if spec.gate and here:
            from .checks import gate

            blocked = gate(spec, folder)
            where += f"; {blocked}" if blocked else f"; checked: beats {registry.model(spec.gate).title}"
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
        if path.is_file():
            print(f"{spec.id}: recorded in {checks.record(spec, folder, mine)}")
        reports += mine
    for spec in map(registry.model, dict.fromkeys(ids)):
        if spec.gate:
            blocked = checks.gate(spec, folder)
            print(f"{spec.id}: {blocked or f'used by its profiles (it beats {registry.model(spec.gate).title})'}")
    if args.json:
        Path(args.json).write_text(json.dumps({r.name: {"ok": r.ok, "lines": r.lines, "values": r.values}
                                               for r in reports}, indent=1, default=str))
    failed = [r.name for r in reports if not r.ok]
    print(f"{len(reports) - len(failed)} of {len(reports)} checks passed" + (f"; failed: {', '.join(failed)}"
                                                                            if failed else ""))
    return 1 if failed else 0


def cmd_calibrate(args) -> int:
    """Make each profile's recogniser's calibration from renders of avatars (see :mod:`.calibrate`)."""
    import datetime

    from ..characters.library import CharacterLibrary
    from . import calibrate as cal
    from .engine import FaceEngine
    from .gallery import Gallery

    folder = models.models_dir(args.models)
    data = Path(args.data) if args.data else folder / "calibration" / "data"
    engines: dict[str, FaceEngine] = {}
    for name in args.profiles or list(registry.PROFILES):
        engine = FaceEngine(args.models, download=not args.offline, profile=name)
        for note in engine.notes:
            print(f"[faces] {note}", file=sys.stderr)
        engines.setdefault(engine.model_id, engine)
    detectors = {engine.detector_spec.id: engine.detector for engine in engines.values()}
    paths = {key: data / f"probes.{key}.npz" for key in detectors}
    if args.refit and all(path.is_file() for path in paths.values()):
        probes = {key: cal.Probes.load(path) for key, path in paths.items()}
        print(f"[calibrate] the faces rendered before, from {data}")
    else:
        library = CharacterLibrary(args.characters) if args.characters else CharacterLibrary()
        names = args.avatars or sorted(library.available())
        if len(names) < 6:
            raise ValueError(f"{len(names)} avatars are too few to calibrate (prepare more: bifold-characters fetch)")
        print(f"[calibrate] {len(names)} avatars from {library.folder}: their gallery and "
              f"{len(cal.EYE_PX) * len(cal.YAWS) * len(cal.PRESETS)} windows each")
        probes = cal.render(library, names, detectors, data / "gallery", seed=args.seed)
        for key, found in probes.items():
            found.save(paths[key])
    status = 0
    for model_id, engine in engines.items():
        found = probes[engine.detector_spec.id]
        vectors, fiqa = _embedded(engine, found, data / f"embeddings.{model_id}.npz", args.refit)
        faces = cal.faces(found, vectors, found.weights(fiqa))
        gallery = Gallery(data / "gallery", engine)
        gallery.refresh(force=True)
        rows = cal.GalleryRows.of(gallery)
        print(f"[calibrate] {model_id}: {int(faces.single.sum())} faces and {int((~faces.single).sum())} averages of "
              f"{len(set(faces.people.tolist()))} avatars ({found.found.mean():.0%} of the windows' faces found); "
              f"{len(rows.rows)} gallery photos and templates")
        kinds = cal.KINDS if args.normalisation == "auto" else ("none", args.normalisation)
        results = cal.check(model_id, faces, rows, kinds=tuple(dict.fromkeys(kinds)), splits=args.splits,
                            seed=args.seed)
        kind = cal.choose(results, kinds=tuple(dict.fromkeys(kinds))) if args.normalisation == "auto" \
            else args.normalisation
        for k, by_target in results.items():
            cells = ", ".join(f"{float(t):.1%}: {e['fpir']:.2%} ({e['fpir_single']:.2%} single), named "
                              f"{e['tpir']:.1%}" for t, e in by_target.items())
            print(f"  held out, {k:>6}: target {cells}{'   <- kept' if k == kind else ''}")
        made = {"by": "bifold-faces calibrate", "date": datetime.date.today().isoformat(),
                "avatars": sorted(set(faces.people.tolist())), "detector": engine.detector_spec.id,
                "sha256": engine.embedder_spec.file.sha256, "eye_px": list(cal.EYE_PX), "yaws": list(cal.YAWS),
                "presets": list(cal.PRESETS), "gallery": {"yaws": list(cal.GALLERY_YAWS),
                                                          "eye_px": cal.GALLERY_EYE_PX},
                "faces": int(faces.single.sum()), "averages": int((~faces.single).sum()),
                "check": {k: {t: {m: round(v, 5) if isinstance(v, float) else v for m, v in e.items()}
                              for t, e in by_target.items()} for k, by_target in results.items()}}
        cohort_set = cal.cohort_faces(faces, seed=args.seed) if kind != "none" else None
        calibration = cal.fit(model_id, faces, rows, kind, cohort_set, made)
        cohort = cal.cohort_of(model_id, cohort_set) if cohort_set is not None else None
        for path in cal.write(model_id, calibration, cohort, args.models):
            print(f"[calibrate] wrote {path}")
        for line in cal.bins_table(calibration):
            print(line)
        cal.save_report(data / f"report.{model_id}.json", {"model": model_id, "normalisation": kind,
                                                            "check": results})
    return status


def _embedded(engine, probes, path: Path, reuse: bool):
    """``(unit embeddings, eDifFIQA-T scores or None)`` of ``probes``' faces (kept in ``path``)."""
    import numpy as np

    if reuse and path.is_file():
        with np.load(path, allow_pickle=False) as data:
            fiqa = data["fiqa"] if "fiqa" in data.files else None
            return data["vectors"].astype(np.float64), fiqa
    vectors = np.zeros((len(probes.crops), engine.dim), np.float32)
    fiqa = np.zeros(len(probes.crops), np.float32) if engine.fiqa is not None else None
    rows = np.flatnonzero(probes.found)
    for start in range(0, len(rows), 32):
        part = rows[start:start + 32]
        vectors[part] = engine.embedder.embed(probes.crops[part]).vectors
        if fiqa is not None:
            fiqa[part] = engine.fiqa.score(probes.crops[part])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, vectors=vectors, **({"fiqa": fiqa} if fiqa is not None else {}))
    return vectors.astype(np.float64), fiqa


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
    calibrate = sub.add_parser("calibrate", help="calibrate the decisions from renders of avatars")
    calibrate.add_argument("--profiles", nargs="+", default=None, choices=tuple(registry.PROFILES), metavar="P",
                           help="calibrate these profiles' recognisers (default: all three)")
    calibrate.add_argument("--characters", default=None, metavar="DIR",
                           help="the avatars (default: $BIFOLD_CHARACTERS); avatars not in the fair's library "
                                "keep the fair a fair check")
    calibrate.add_argument("--avatars", nargs="+", default=None, metavar="NAME", help="only these avatars")
    calibrate.add_argument("--data", default=None, metavar="DIR",
                           help="where the renders, faces and embeddings are kept (default: "
                                "models/calibration/data)")
    calibrate.add_argument("--refit", action="store_true",
                           help="use the faces and embeddings made before (no rendering)")
    calibrate.add_argument("--normalisation", choices=("auto", "none", "tnorm", "asnorm"), default="auto",
                           help="score normalisation (default auto: kept only when the held-out check finds it "
                                "names more at the same false-alarm rate)")
    calibrate.add_argument("--splits", type=int, default=10, help="random halvings of the avatars for the check")
    calibrate.add_argument("--seed", type=int, default=0, help="the seed of the places and the halvings")
    calibrate.add_argument("--offline", action="store_true", help="do not download missing models")
    calibrate.set_defaults(run=cmd_calibrate)
    from ..characters.headtex.cli import add_commands

    add_commands(sub)
    args = parser.parse_args(argv)
    registry.use_profile(args.face_profile)  # every engine of this run (the head textures' check too) uses it
    try:
        return args.run(args)
    except (OSError, ValueError, ImportError) as exc:  # ImportError: MediaPipe is missing (the message says how)
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
