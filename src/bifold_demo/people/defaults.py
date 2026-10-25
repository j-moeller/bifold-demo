"""Default people for scenes, cut out of the sample photos that ship with Ultralytics.

Used until you generate your own crowd with ``bifold-people generate``. The cut-outs
are computed once and cached in ``~/.cache/bifold-demo/default-people``.
"""

from __future__ import annotations

from pathlib import Path

import cv2

from ..scene.textures import sample_image
from .library import PeopleLibrary, Person

CACHE = Path("~/.cache/bifold-demo/default-people").expanduser()
FULL_BODY_HEIGHT = 1.78  # metres, head to feet
BUST_HEIGHT = 0.95  # metres visible for people cut off at the waist (they stand behind booth counters)


def default_people(cache=CACHE, models_dir="models") -> list[Person]:
    """Full-body people and waist-up "busts" cut from the Ultralytics sample photos (empty if unavailable)."""
    library = PeopleLibrary(cache)
    people = library.people()
    if people:
        return people
    try:
        from .cutout import PersonSegmenter

        segmenter = PersonSegmenter(models_dir=models_dir)
    except Exception as exc:  # noqa: BLE001 - no ultralytics / no network: scenes run without people
        print(f"[people] cannot create default people: {exc}")
        return []
    print("[people] cutting default people out of the Ultralytics sample photos (once) ...")
    for name in ("bus.jpg", "zidane.jpg"):
        path = sample_image(name)
        image = cv2.imread(str(path)) if path else None
        if image is None:
            continue
        for i, cutout in enumerate(segmenter.cutouts(image)):
            if {"left", "right"} & cutout.touches or cutout.confidence < 0.6:
                continue  # half outside the photo
            library.add(cutout, name=f"{Path(name).stem}_{i}",
                        height=FULL_BODY_HEIGHT if cutout.full_body else BUST_HEIGHT,
                        pose="standing", facing="front", source=f"sample:{name}")
    return library.people()
