"""``bifold-faces landmarks``, ``register`` and ``headtex``: the head-texture stages from the command line.

    bifold-faces landmarks [AVATAR ...]              the 478 landmarks on avatars' heads (cached); which way found
    bifold-faces register PHOTO[:VIEW] ... --avatar NAME [--out DIR] [--segment]
                                                     register photos to an avatar's head; draw what was found
    bifold-faces headtex PHOTO[:VIEW] ... --avatar NAME -o head.png [--views render.png] [--debug DIR]
                                                     a head texture from the photos (stages 0-4)
    bifold-faces headtex PHOTO[:VIEW] ... --avatar NAME -o head.png --full [--body body.png] [--generate]
                 [--check] [--character NAME [--face PERSON] [--created DIR]]
                                                     all stages (0-7): head, body and hair textures, checked;
                                                     optionally saved as a created character

A ``:left``, ``:right`` or ``:back`` after a photo names what it shows (needed for photos without a face).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .register import VIEW_YAWS


def add_commands(sub) -> None:
    marks = sub.add_parser("landmarks", help="put MediaPipe's face landmarks on avatars' heads (stage 0)")
    marks.add_argument("avatars", nargs="*", help="avatar names (default: every prepared avatar)")
    marks.add_argument("--characters", default=None, help="the prepared characters folder (default: as the demo)")
    marks.add_argument("--refresh", action="store_true", help="find them again instead of using the cache")
    marks.add_argument("--fallback", action="store_true", help="the canonical-face fit only (for comparison)")
    marks.set_defaults(run=cmd_landmarks)
    reg = sub.add_parser("register", help="register photos of a head to an avatar's head (stage 1)")
    reg.add_argument("photos", nargs="+", help="photos, each optionally followed by :left, :right, :back or :front")
    reg.add_argument("--avatar", required=True, help="the avatar whose head the photos are registered to")
    reg.add_argument("--characters", default=None, help="the prepared characters folder (default: as the demo)")
    reg.add_argument("--out", default=None, help="write each photo with the registration drawn on it into this folder")
    reg.add_argument("--segment", action="store_true", help="also segment hair, skin and clothes (needed for photos "
                     "without a face)")
    reg.set_defaults(run=cmd_register)
    tex = sub.add_parser("headtex", help="a head texture for an avatar from photos of a person (stages 0-4)")
    tex.add_argument("photos", nargs="+", help="photos, each optionally followed by :left, :right, :back or :front")
    tex.add_argument("--avatar", required=True, help="the avatar whose head the texture is for")
    tex.add_argument("-o", "--output", required=True, help="the texture's file (PNG)")
    tex.add_argument("--characters", default=None, help="the prepared characters folder (default: as the demo)")
    tex.add_argument("--views", default=None, help="also render the avatar wearing it (front, three-quarter, side) "
                     "into this image")
    tex.add_argument("--debug", default=None, help="write what each stage did into this folder")
    tex.add_argument("--size", type=int, default=None, help="the texture's size (default: the avatar's own)")
    tex.add_argument("--full", action="store_true", help="all stages: complete what no photo shows, match the body's "
                     "skin, the neck and the hair, check (stages 5-7)")
    tex.add_argument("--body", default=None, help="the body texture's file (default: next to -o; all stages, as "
                     "--full)")
    tex.add_argument("--generate", nargs="?", const="flux2-klein", default=None, metavar="MODEL",
                     help="a generative layer for what no photo shows (default model: flux2-klein; "
                     "needs diffusers and the model's weights)")
    tex.add_argument("--check", action="store_true", help="the identity check (renders of the avatar "
                     "against the photos; needs OpenGL and the face recognition's models)")
    tex.add_argument("--character", default=None, help="save the textures as a created character of "
                     "this name")
    tex.add_argument("--face", default=None, help="with --character: the known person whose face it is")
    tex.add_argument("--created", default="created", help="with --character: the created characters' folder")
    tex.set_defaults(run=cmd_headtex)


def _photo_and_view(text: str) -> tuple[str, str | None]:
    head, sep, tail = text.rpartition(":")
    if sep and tail in VIEW_YAWS:
        return head, tail
    return text, None


def _library(args, names=()):
    """The prepared characters, checked to have the avatars ``names`` (a mistyped name: a message, not a crash)."""
    from ..library import CharacterLibrary

    library = CharacterLibrary(args.characters)
    available = library.available()
    if not available:
        raise ValueError(f"no prepared avatars in {library.folder} (run: bifold-characters fetch --set small)")
    for name in names:
        if name not in available:
            raise ValueError(library.explain(name))
    return library


def cmd_landmarks(args) -> int:
    from .detect import Landmarker
    from .landmarks import VIEWS, head_landmarks

    library = _library(args, args.avatars)
    names = args.avatars or sorted(library.available())
    with Landmarker(args.models) as landmarker:
        for name in names:
            model = library.load(name)
            marks = head_landmarks(model, landmarker, cache_dir=args.characters, refresh=args.refresh,
                                   method="canonical" if args.fallback else "auto")
            unsure = int((marks.reliability < 0.5).sum())
            if marks.method == "mediapipe":
                how = f"MediaPipe on {marks.views} of {len(VIEWS)} renders"
            else:
                how = "fitted canonical face (" + ("--fallback" if args.fallback else "MediaPipe found no face") + ")"
            print(f"{name}: {how}; reliability {marks.reliability.mean():.2f} on average, {unsure} of 478 below 0.5")
    return 0


def cmd_register(args) -> int:
    import cv2

    from .detect import Landmarker, Segmenter
    from .landmarks import head_landmarks
    from .register import draw_registration, head_regions, register_photos

    library = _library(args, [args.avatar])
    model = library.load(args.avatar)
    photos, views = zip(*(_photo_and_view(p) for p in args.photos))
    out = Path(args.out) if args.out else None
    with Landmarker(args.models, max_faces=2) as landmarker:
        head = head_landmarks(model, landmarker, cache_dir=args.characters)
        segmenter = Segmenter(args.models) if args.segment else None
        try:
            if segmenter is not None:  # where the avatar's face skin and hair are, for the outline fit (cached)
                head_regions(model, head.primitive, segmenter, cache_dir=args.characters)
            results = register_photos(list(photos), head, model, views=views, landmarker=landmarker,
                                      segmenter=segmenter)
        finally:
            if segmenter is not None:
                segmenter.close()
    status = 0
    for photo, view in zip(photos, results):
        print(f"{photo}: {view.summary()}")
        status = status or (0 if view.usable else 1)
        if out is not None and view.camera is not None:
            out.mkdir(parents=True, exist_ok=True)
            target = out / f"{Path(photo).stem}.registered.jpg"
            cv2.imwrite(str(target), draw_registration(view, model, head))
            print(f"  drawn: {target}")
    return status


def cmd_headtex(args) -> int:
    import cv2

    from .detect import Landmarker, Segmenter
    from .texture import debug_images, head_texture, preview

    library = _library(args, [args.avatar])
    model = library.load(args.avatar)
    photos, views = zip(*(_photo_and_view(p) for p in args.photos))
    for photo in photos:
        if not Path(photo).is_file():
            raise FileNotFoundError(f"no photo {photo}")
    with Landmarker(args.models, max_faces=2) as landmarker, Segmenter(args.models) as segmenter:
        result = head_texture(list(photos), model, views=list(views), landmarker=landmarker, segmenter=segmenter,
                              size=args.size, cache_dir=args.characters, debug=args.debug is not None)
    if args.full or args.body or args.generate or args.check or args.character:  # (each says all stages)
        return _full(args, model, result, list(photos))
    print(result.summary())
    print(f"texture: {result.save(args.output)}")
    if args.views:
        target = Path(args.views)
        target.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(target), np.vstack([preview(model, result), preview(model, None)]))
        print(f"renders (top: with the texture, bottom: the avatar's own): {target}")
    _write_debug(args, result, debug_images)
    return 0


def _write_debug(args, result, debug_images) -> None:
    import cv2

    if args.debug:
        folder = Path(args.debug)
        folder.mkdir(parents=True, exist_ok=True)
        for name, image in debug_images(result).items():
            cv2.imwrite(str(folder / name), image)
        print(f"debug images: {folder}")


def _full(args, model, result, photos) -> int:
    """Stages 5 to 7 on ``result`` and the files of ``--full``."""
    import cv2

    from .completion import FluxInpainter
    from .person import finish
    from .texture import debug_images, preview

    engine = None
    if args.check:
        from ...recognition.engine import FaceEngine

        engine = FaceEngine(args.models, profile=getattr(args, "face_profile", None))
    inpainter = FluxInpainter(args.generate) if args.generate else None
    person = finish(result, model, inpainter=inpainter, check=args.check, photos=photos, engine=engine)
    print(person.summary())
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    files = {person.head_primitive: output}
    if person.body is not None:
        files[person.body_primitive] = Path(args.body) if args.body else output.with_name(f"{output.stem}_body.png")
    for name in person.textures:
        files.setdefault(name, output.with_name(f"{output.stem}_{name}.png"))
    for name, path in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), cv2.cvtColor(person.textures[name], cv2.COLOR_RGBA2BGRA)):
            raise OSError(f"cannot write {path}")
        print(f"texture of {name}: {path}")
    if args.views:
        target = Path(args.views)
        target.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(target), np.vstack([preview(model, person.textures), preview(model, None)]))
        print(f"renders (top: with the textures, bottom: the avatar's own): {target}")
    _write_debug(args, result, debug_images)
    if args.character:
        from ..created import CharacterBook

        saved = CharacterBook(args.created).save(person.created(args.character, face=args.face))
        print(f"saved as the created character {args.character!r}: {saved}")
    return 0
