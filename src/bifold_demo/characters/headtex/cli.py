"""``bifold-faces landmarks`` and ``bifold-faces register``: stages 0 and 1 from the command line.

    bifold-faces landmarks [AVATAR ...]              the 478 landmarks on avatars' heads (cached); which way found
    bifold-faces register PHOTO[:VIEW] ... --avatar NAME [--out DIR] [--segment]
                                                     register photos to an avatar's head; draw what was found

A ``:left``, ``:right`` or ``:back`` after a photo names what it shows (needed for photos without a face).
"""

from __future__ import annotations

from pathlib import Path

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


def _photo_and_view(text: str) -> tuple[str, str | None]:
    head, sep, tail = text.rpartition(":")
    if sep and tail in VIEW_YAWS:
        return head, tail
    return text, None


def _library(args):
    from ..library import CharacterLibrary

    library = CharacterLibrary(args.characters)
    if not library.available():
        raise ValueError(f"no prepared avatars in {library.folder} (run: bifold-characters fetch --set small)")
    return library


def cmd_landmarks(args) -> int:
    from .detect import Landmarker
    from .landmarks import head_landmarks

    library = _library(args)
    names = args.avatars or sorted(library.available())
    with Landmarker(args.models) as landmarker:
        for name in names:
            model = library.load(name)
            marks = head_landmarks(model, landmarker, cache_dir=args.characters, refresh=args.refresh,
                                   method="canonical" if args.fallback else "auto")
            unsure = int((marks.reliability < 0.5).sum())
            how = (f"MediaPipe on {marks.views} of 5 renders" if marks.method == "mediapipe"
                   else "fitted canonical face (MediaPipe found no face)")
            print(f"{name}: {how}; reliability {marks.reliability.mean():.2f} on average, {unsure} of 478 below 0.5")
    return 0


def cmd_register(args) -> int:
    import cv2

    from .detect import Landmarker, Segmenter
    from .landmarks import head_landmarks
    from .register import draw_registration, register_photos

    library = _library(args)
    model = library.load(args.avatar)
    photos, views = zip(*(_photo_and_view(p) for p in args.photos))
    out = Path(args.out) if args.out else None
    with Landmarker(args.models, max_faces=2) as landmarker:
        head = head_landmarks(model, landmarker, cache_dir=args.characters)
        segmenter = Segmenter(args.models) if args.segment else None
        try:
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
