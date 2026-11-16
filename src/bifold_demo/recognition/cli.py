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
    bifold-faces eval MANIFEST ... [--profiles P ...] [--out DIR]
                                               measure the recognition on manifests of photos, renders and test
                                               packs (metrics as JSON, curves as images: see .evaluate)
    bifold-faces eval --build p2|ladder|p3|lfw|photos [--pack lfw.bin] [--photos DIR]
                                               write a protocol's manifest (.protocols)
    bifold-faces eval P3.json --profiles best [--embed-rate R] --acceptance
                                               scene runs scored per track, judged by the acceptance test (exit 1
                                               when it fails; .evalscene)
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
    import dataclasses
    import datetime
    import time

    import numpy as np

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
    probes = {key: cal.Probes.load(path) for key, path in paths.items() if path.is_file()}
    library = None
    if args.refit:  # the faces rendered before, nothing new
        if len(probes) < len(detectors):
            raise ValueError(f"{data} holds no renders for every profile's detector: run without --refit first")
        names = args.avatars or sorted(set().union(*(set(p.avatar.tolist()) for p in probes.values())))
        print(f"[calibrate] the faces rendered before, from {data}")
    else:  # render what is missing (the windows of new avatars or sizes); what was rendered before is kept
        library = CharacterLibrary(args.characters) if args.characters else CharacterLibrary()
        names = args.avatars or sorted(library.available())
        if len(names) < 6:
            raise ValueError(f"{len(names)} avatars are too few to calibrate (prepare more: bifold-characters fetch)")
        wanted = cal.window_keys(names)
        have = {key: set(p.keys().tolist()) for key, p in probes.items()}
        common = set.intersection(*(have.get(key, set()) for key in detectors))
        missing = [k for k in wanted if k not in common]
        print(f"[calibrate] {len(names)} avatars from {library.folder}: {len(wanted)} windows, {len(missing)} to "
              "render")
        fresh = cal.render(library, sorted({k.split("|")[0] for k in missing}) if missing else [], detectors,
                           data / "gallery", seed=args.seed, skip=common)
        for key, new in fresh.items():
            old = probes.get(key)
            if old is not None:
                new = old.concat(new.subset(~np.isin(new.keys(), sorted(have.get(key, set())))))
            probes[key] = new
            new.save(paths[key])
        cal.gallery_photos(library, names, data / "gallery")  # (any still missing)
    excluded_gallery = None
    if args.exclude_faces_of != "none":  # avatars wearing a face of another library (the fair's) are left out
        other = CharacterLibrary(args.exclude_faces_of) if args.exclude_faces_of else CharacterLibrary()
        if library is None or Path(other.folder).resolve() != Path(library.folder).resolve():
            if library is not None and other.available():
                cal.gallery_photos(other, sorted(other.available()), data / "exclude")
            if (data / "exclude").is_dir():
                excluded_gallery = data / "exclude"
    status = 0
    found_shared: dict[str, str] = {}  # what the models before found (the first, the best, tells faces apart best)
    found_alias: dict[str, str] = {}
    decided_by = None
    for model_id, engine in engines.items():
        found = probes[engine.detector_spec.id]
        vectors, fiqa = _embedded(engine, found, data / f"embeddings.{model_id}.npz")
        gallery = Gallery(data / "gallery", engine)
        gallery.refresh(force=True)
        others = None
        if excluded_gallery is not None:
            other_gallery = Gallery(excluded_gallery, engine)
            other_gallery.refresh(force=True)
            others = cal.GalleryRows.of(other_gallery)
        if args.data_only:
            print(f"[calibrate] {model_id}: {int(found.found.sum())} faces embedded")
            continue
        chosen = np.isin(found.avatar, names)
        found, vectors = found.subset(chosen), vectors[chosen]
        fiqa = None if fiqa is None else fiqa[chosen]
        passed = found.passed(fiqa)  # (the faces the live gate lets through, as live)
        faces = cal.faces(found, vectors, found.weights(fiqa), passed)
        rows = cal.GalleryRows.of(gallery)
        rows = rows.subset(sorted(set(faces.people.tolist())))
        avatars = sorted(set(faces.people.tolist()))
        print(f"[calibrate] {model_id}: {int(faces.single.sum())} faces and {int((~faces.single).sum())} averages of "
              f"{len(avatars)} avatars ({found.found.mean():.0%} of the windows' faces found, "
              f"{passed.sum() / max(1, found.found.sum()):.0%} of those pass the live gate); "
              f"{len(rows.rows)} gallery photos and templates")
        shared = cal.shared_faces(faces, rows, others, known=found_shared) if others is not None else {}
        found_shared.update(shared)
        if shared:
            print("[calibrate] left out, wearing a face of the other library: "
                  + ", ".join(f"{a} ({b})" for a, b in sorted(shared.items())))
            faces = cal._subset(faces, [a for a in avatars if a not in shared])
            rows = rows.subset([a for a in avatars if a not in shared])
        present = set(faces.people.tolist())
        if decided_by is None:  # who wears whose face: the first model's say (list the best profile first)
            alias, decided_by = cal.same_faces(faces, rows), model_id
            found_alias.update(alias)
        else:  # (another model's lookalikes stay strangers to each other: its calibration must count them)
            alias = {a: b for a, b in found_alias.items() if a in present and b in present}
            print(f"[calibrate] who wears whose face, as {decided_by} found it")
        if alias:
            print("[calibrate] the same face, counted as one identity: "
                  + ", ".join(f"{a} as {b}" for a, b in sorted(alias.items())))
            faces, rows = cal.as_one(faces, rows, alias)
        kinds = cal.KINDS if args.normalisation == "auto" else ("none", args.normalisation)
        results = cal.check(model_id, faces, rows, kinds=tuple(dict.fromkeys(kinds)), splits=args.splits,
                            seed=args.seed)
        kind = cal.choose(results, kinds=tuple(dict.fromkeys(kinds))) if args.normalisation == "auto" \
            else args.normalisation
        for k, by_target in results.items():
            cells = ", ".join(f"{float(t):.1%}: {e['fpir']:.2%} ({e['fpir_single']:.2%} single; halvings "
                              f"{min(e['fpir_by_split']):.2%}-{max(e['fpir_by_split']):.2%}), named "
                              f"{e['tpir']:.1%}" for t, e in by_target.items())
            print(f"  held out, {k:>6}: target {cells}{'   <- kept' if k == kind else ''}")
        made = {"by": "bifold-faces calibrate", "date": datetime.date.today().isoformat(),
                "avatars": [a for a in avatars if a not in shared], "same_faces": alias,
                "left_out": shared, "detector": engine.detector_spec.id,
                "sha256": engine.embedder_spec.file.sha256, "eye_px": list(cal.EYE_PX), "yaws": list(cal.YAWS),
                "presets": list(cal.PRESETS), "gallery": {"yaws": list(cal.GALLERY_YAWS),
                                                          "eye_px": cal.GALLERY_EYE_PX},
                "rendered": found.rendering(),
                "faces": int(faces.single.sum()), "averages": int((~faces.single).sum()),
                "check": {k: {t: {m: round(v, 5) if isinstance(v, float) else v for m, v in e.items()}
                              for t, e in by_target.items()} for k, by_target in results.items()}}
        cohort_set = cal.cohort_faces(faces, seed=args.seed) if kind != "none" else None
        cohort = cal.cohort_of(model_id, cohort_set) if cohort_set is not None else None
        if cohort is not None:
            made["cohort"] = {"faces": len(cohort), "digest": cohort.digest}
        report = {"model": model_id, "normalisation": kind, "check": results}
        track = {}
        if args.track_splits > 0:  # the target per stranger track: offsets from some halvings, checked on others
            started = time.perf_counter()
            tracks = cal.track_check(model_id, faces, rows, kind, args.track_splits, args.seed, progress=print)
            track = cal.track_offsets(tracks, seed=args.seed)
            again = cal.track_check(model_id, faces, rows, kind, args.track_splits, args.seed + 1000, progress=print)
            report["per_track"] = measured = cal.track_rates(again, track)
            track["checked"] = {t: measured[t] for t in map(str, track["targets"])}
            print(f"[calibrate] {model_id}: per stranger track ({len(tracks)} tracks, {track['pairs']} identity "
                  f"pairs; offsets from {args.track_splits} halvings, checked on {args.track_splits} others, "
                  f"{time.perf_counter() - started:.0f} s):")
            for target, offset in zip(track["targets"], track["offsets"]):
                w, wo = measured[str(target)]["with"], measured[str(target)]["without"]
                print(f"  target {target:.1%}: offset {offset:.2f} decades; per track {w['per_track']:.2%} "
                      f"({w['interval'][0]:.2%}-{w['interval'][1]:.2%}, {w['named']} tracks from {w['pairs']} pairs), "
                      f"per decision {w['per_decision']:.3%}; without the offset {wo['per_track']:.2%} "
                      f"({wo['interval'][0]:.2%}-{wo['interval'][1]:.2%})")
        made["track"] = {"splits": args.track_splits, "seed": args.seed} if track else None
        calibration = dataclasses.replace(cal.fit(model_id, faces, rows, kind, cohort_set, made), track=track)
        for path in cal.write(model_id, calibration, cohort, args.models):
            print(f"[calibrate] wrote {path}")
        for line in cal.bins_table(calibration):
            print(line)
        cal.save_report(data / f"report.{model_id}.json", report)
    return status


def _embedded(engine, probes, path: Path, progress=print):
    """``(unit embeddings, eDifFIQA-T scores or None)`` of ``probes``' faces, kept in ``path`` by window key: only the
    windows this model file has not embedded before are embedded now, and the file is saved as it goes (a long run
    that stops keeps what it did)."""
    import time

    import numpy as np

    keys = probes.keys()
    vectors = np.zeros((len(keys), engine.dim), np.float32)
    fiqa = np.zeros(len(keys), np.float32) if engine.fiqa is not None else None
    done = np.zeros(len(keys), bool)
    sha = engine.embedder_spec.file.sha256
    if path.is_file():
        with np.load(path, allow_pickle=False) as data:
            old = data["vectors"]
            old_keys = data["keys"].tolist() if "keys" in data.files else keys[: len(old)].tolist()
            old_done = data["embedded"] if "embedded" in data.files else np.ones(len(old), bool)
            old_fiqa = data["fiqa"] if "fiqa" in data.files else None
            old_sha = str(data["sha256"]) if "sha256" in data.files else sha
        if old_sha == sha and old.shape[1:] == (engine.dim,):
            index = {k: i for i, k in enumerate(old_keys) if old_done[i]}
            for r, k in enumerate(keys.tolist()):
                i = index.get(k)
                if i is not None:
                    vectors[r], done[r] = old[i], True
                    if fiqa is not None and old_fiqa is not None:
                        fiqa[r] = old_fiqa[i]

    def save():
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, vectors=vectors, keys=keys, embedded=done, sha256=np.array(sha),
                            **({"fiqa": fiqa} if fiqa is not None else {}))

    rows = np.flatnonzero(probes.found & ~done)
    started = time.perf_counter()
    for start in range(0, len(rows), 32):
        part = rows[start:start + 32]
        vectors[part] = engine.embedder.embed(probes.crops[part]).vectors
        if fiqa is not None:
            fiqa[part] = engine.fiqa.score(probes.crops[part])
        done[part] = True
        if start // 32 % 10 == 9:
            save()
            if progress:
                progress(f"[calibrate] {engine.model_id}: {start + len(part)} of {len(rows)} faces embedded "
                         f"({time.perf_counter() - started:.0f} s)")
    save()
    return vectors.astype(np.float64), fiqa


def cmd_eval(args) -> int:
    """Measure the profiles on manifests (see :mod:`.evaluate`), or write a protocol's manifest (``--build``)."""
    import json

    from . import evaluate, protocols
    from .engine import FaceEngine

    out = Path(args.out)
    if args.record:
        from . import evalvideo

        width, _, height = args.capture.partition("x")
        evalvideo.record(args.record, args.camera, (int(width), int(height)), args.seconds)
        return 0
    if args.review:
        import os

        video = Path(args.review)
        raw = {"format": 1, "name": video.stem, "protocol": "P5", "items": [{
            "id": video.stem, "kind": "video", "path": video.name,
            "faces": os.path.relpath(Path(args.faces).resolve(), video.parent.resolve()),
            "labels": Path(args.labels).name if args.labels else f"{video.stem}.labels.csv"}]}
        args.manifests = [video.with_suffix(".manifest.json")]
        args.manifests[0].write_text(json.dumps(raw, indent=1))
    if args.build:
        if args.build in ("p2", "ladder"):
            names = args.identities or protocols.fair_identities(args.characters)
            raw = protocols.p2(names, args.size) if args.build == "p2" else protocols.ladder(names)
        elif args.build == "lfw":
            if not args.pack:
                raise ValueError("--build lfw needs --pack lfw.bin")
            raw = protocols.lfw(args.pack, None if args.all_pairs else (75, 75))
        elif args.build == "p3":
            names = args.identities or protocols.fair_identities(args.characters)
            raw = protocols.p3(names, runs=args.runs, seconds=args.seconds,
                               detector="truth" if args.truth_people else "profile")
        else:
            if not args.photos:
                raise ValueError("--build photos needs --photos DIR (laid out as the README says)")
            raw = protocols.photos(args.photos)
        folder = Path(args.photos) if args.build == "photos" else Path(".")
        manifest = evaluate.parse(raw, folder, source=f"the {args.build} protocol")
        target = Path(args.manifest_out or (Path(args.photos) / "manifest.json" if args.build == "photos" else
                                            out / "manifests" / f"{manifest.name}.json"))
        if args.build == "photos":
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(raw, indent=1))
        else:
            evaluate.dump(manifest, target)
        kinds = {k: len(manifest.of_kind(k)) for k in evaluate.KINDS if manifest.of_kind(k)}
        print(f"wrote {target}: {', '.join(f'{v} {k}' for k, v in kinds.items())}, "
              f"{len(manifest.identities())} identities")
        if not args.manifests:
            return 0
    if not args.manifests:
        raise ValueError("give manifests to measure (or --build a protocol's)")
    manifests = [evaluate.load(path) for path in args.manifests]
    engines = {}
    for name in args.profiles or list(registry.PROFILES):
        engine = FaceEngine(args.models, download=not args.offline, profile=name, progress=None)
        for note in engine.notes:
            print(f"[faces] {note}", file=sys.stderr)
        engines[name] = engine
    margins = {}
    if args.margins:  # (scripts' margins: per profile the offsets of each margin at each target per stranger track)
        for entry in json.loads(Path(args.margins).read_text()).values():
            k = entry["targets"].index(0.01)
            margins[entry["model"]] = {f"margin {c}": offs[k] for c, offs in entry["offsets"].items()}
    run = evaluate.Run(engines, out, library=args.characters, draws=args.draws, seed=args.seed,
                       models_folder=args.models, margins=margins)
    status = 0
    try:
        for manifest in manifests:
            results = {"manifest": manifest.name, "protocol": manifest.protocol, "description": manifest.description,
                       "sides": manifest.sides, "environment": evaluate.environment(),
                       "settings": {"draws": args.draws, "seed": args.seed, "splits": args.splits,
                                    "level": evaluate.metrics.LEVEL}}
            if manifest.stills("probe"):
                def done(so_far, results=results, name=manifest.name):  # (each profile's results as they come)
                    evaluate.write_json(out / name / "results.json", {**results, "stills": so_far})

                results["stills"] = run.stills(manifest, args.splits, conditions=not args.no_conditions, done=done)
            if manifest.of_kind("pack"):
                results["packs"] = run.packs(manifest)
            if manifest.of_kind("scene"):
                from . import evalscene

                name = manifest.name if args.embed_rate is None else f"{manifest.name}@{args.embed_rate:g}"

                def ran(so_far, results=results, name=name):  # (each run's results as they come)
                    evaluate.write_json(out / name / "results.json", {**results, "scenes": so_far})

                results["scenes"] = evalscene.run_scenes(run, manifest, args.embed_rate, args.yolo, done=ran)
                if args.acceptance:
                    for scored in results["scenes"].values():
                        verdict = scored["score"]["acceptance"]
                        status = max(status, 0 if verdict["verdict"] == "pass" else 1)
                path = evaluate.write_json(out / name / "results.json", results)
            else:
                if manifest.of_kind("video"):
                    from . import evalvideo

                    results["videos"] = {item.id: evalvideo.run_video(run, item, manifest.folder, args.yolo)
                                         for item in manifest.of_kind("video")}
                    if args.acceptance:
                        for scored in results["videos"].values():
                            for profile in scored.values():
                                status = max(status, 0 if profile["acceptance"]["verdict"] == "pass" else 1)
                path = evaluate.write_json(out / manifest.name / "results.json", results)
            for line in evaluate.summary_lines(results):
                print(line)
            if not args.no_plots:
                from . import evalplots

                for image in evalplots.plot_results(results, out / manifest.name / "curves"):
                    print(f"[eval] {image}")
            print(f"[eval] {manifest.name}: {path}")
    finally:
        run.close()
    return status


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
                           help="use only the faces rendered before (no rendering; embeddings are always reused)")
    calibrate.add_argument("--data-only", action="store_true",
                           help="render and embed what is missing, then stop (no calibration)")
    calibrate.add_argument("--exclude-faces-of", default="", metavar="DIR",
                           help="leave out avatars wearing a face of this character library's avatars (default: the "
                                "default library, the fair's; 'none': no such check)")
    calibrate.add_argument("--normalisation", choices=("auto", "none", "tnorm", "asnorm"), default="auto",
                           help="score normalisation (default auto: kept only when the held-out check finds it "
                                "names more at the same false-alarm rate)")
    calibrate.add_argument("--splits", type=int, default=10, help="random halvings of the avatars for the check")
    calibrate.add_argument("--track-splits", type=int, default=60, metavar="N",
                           help="halvings for the offsets that make the target a rate per stranger track, and as "
                                "many others to check them (default 60; 0: per decision, no offsets)")
    calibrate.add_argument("--seed", type=int, default=0, help="the seed of the places and the halvings")
    calibrate.add_argument("--offline", action="store_true", help="do not download missing models")
    calibrate.set_defaults(run=cmd_calibrate)
    evaluation = sub.add_parser("eval", help="measure the recognition on manifests (stills, renders, packs, scenes)",
                                description="Measure the face recognition on manifests (see the README, "
                                            "'Evaluating the face recognition'), or write a protocol's manifest.")
    evaluation.add_argument("manifests", nargs="*", help="manifests (JSON, or CSV of stills) to measure")
    evaluation.add_argument("--profiles", nargs="+", default=None, choices=tuple(registry.PROFILES), metavar="P",
                            help="the profiles to measure (default: all three)")
    evaluation.add_argument("--out", default="eval", metavar="DIR",
                            help="where results, curves and caches go (default: eval, git-ignored)")
    evaluation.add_argument("--build", choices=("p2", "ladder", "p3", "lfw", "photos"), default=None,
                            help="write a protocol's manifest first (into DIR/manifests, or --manifest-out)")
    evaluation.add_argument("--manifest-out", default=None, metavar="FILE", help="where --build writes")
    evaluation.add_argument("--size", choices=("full", "small", "tiny"), default="full",
                            help="P2's size (default full)")
    evaluation.add_argument("--identities", nargs="+", default=None, metavar="NAME",
                            help="the avatars measured (default: the fair's library's)")
    evaluation.add_argument("--characters", default=None, metavar="DIR",
                            help="the avatars' folder (default: $BIFOLD_CHARACTERS)")
    evaluation.add_argument("--pack", default=None, metavar="FILE", help="lfw.bin for --build lfw")
    evaluation.add_argument("--all-pairs", action="store_true", help="--build lfw: every pair, not F1's subset")
    evaluation.add_argument("--photos", default=None, metavar="DIR", help="a folder of photos for --build photos")
    evaluation.add_argument("--runs", type=int, default=4, help="--build p3: scene runs (default 4)")
    evaluation.add_argument("--seconds", type=float, default=60.0, help="--build p3: seconds per run (default 60)")
    evaluation.add_argument("--truth-people", action="store_true",
                            help="--build p3: people found from the scene's truth, not the profile's person detector")
    evaluation.add_argument("--embed-rate", type=float, default=None, metavar="R",
                            help="scene runs: embeddings a second (default: the profile's; a faster machine's, say)")
    evaluation.add_argument("--yolo", default=None, metavar="FILE",
                            help="scene runs: YOLO26n's weights for the best and licensed profiles")
    evaluation.add_argument("--acceptance", action="store_true",
                            help="judge the scene runs (or recordings) by the acceptance test (exit 1 when a profile "
                                 "fails it)")
    evaluation.add_argument("--record", default=None, metavar="FILE",
                            help="record a session from the camera into FILE (Motion-JPEG .avi; P5) and stop")
    evaluation.add_argument("--camera", type=int, default=0, help="--record: the camera's index (default 0)")
    evaluation.add_argument("--capture", default="1920x1080", metavar="WxH",
                            help="--record: the resolution asked of the camera (default 1920x1080)")
    evaluation.add_argument("--review", default=None, metavar="VIDEO",
                            help="run the pipeline over a recording: the first time, write its review sheets "
                                 "(<video>.review.jpg, <video>.labels.csv) to label; once labelled, score it")
    evaluation.add_argument("--labels", default=None, metavar="CSV", help="--review: the labels sheet's name")
    evaluation.add_argument("--margins", default=None, metavar="FILE",
                            help="the offsets of other margins to compare (JSON: per profile its model, targets and "
                                 "offsets per margin; recognition/margins.py)")
    evaluation.add_argument("--splits", type=int, default=None,
                            help="splits of the identities into enrolled and strangers (default: the manifest's)")
    evaluation.add_argument("--draws", type=int, default=1000, help="bootstrap draws (default 1000)")
    evaluation.add_argument("--seed", type=int, default=0, help="the seed of the splits and the bootstrap")
    evaluation.add_argument("--no-conditions", action="store_true", help="no per-condition reports (quicker)")
    evaluation.add_argument("--no-plots", action="store_true", help="write no curves")
    evaluation.add_argument("--offline", action="store_true", help="do not download missing models")
    evaluation.set_defaults(run=cmd_eval)
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
