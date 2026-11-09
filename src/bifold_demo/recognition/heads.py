"""Where a person's face is looked for, and whose face a face is.

The recognition looks for faces only in the **head regions** of the person tracks (:mod:`bifold_demo.detection.
tracker`), never over the whole frame: small faces become findable once a head is cut out and scaled up, and faces on
posters, screens and textures away from a person are never mistaken for someone. A head region is S1's
:class:`~bifold_demo.sensor.requests.HeadRegion` (a square around the top of the box) on the track's smoothed box.

A face is looked for in one of two places (:func:`find_faces`):

* in a **window** of the virtual high-resolution sensor (:mod:`bifold_demo.sensor`), delivered with the frame
  (``frame.windows``), where the track asked for one: its faces are found in the window's own pixels and mapped back
  to the frame's (:meth:`~bifold_demo.sensor.geometry.Window.to_view`); they are aligned from the window's pixels, not
  the frame's. A webcam's windows are cut from its full-resolution picture, and work the same way;
* else in a **crop of the frame** around the head region (:meth:`~.engine.FaceEngine.detect` with ``region``).

**Whose face it is** (:func:`assign_faces`). A face belongs to a track only when the point between its eyes lies
inside that track's head region, and its place and size fit where the track's head should be (:class:`HeadModel`,
measured on YOLO's boxes of the fair: the eyes 0.079 box heights below the box's top and 0.037 box heights apart, the
face box 2.95 eye distances high at every yaw): ``cost = (dx / sx)^2 + (dy / sy)^2 + (log(size / expected) /
ss)^2`` in eye distances. In a crowd two tracks' regions overlap (someone behind someone): every face is assigned to
at most one track and every track gets at most one face, at the least total cost (:func:`~bifold_demo.detection.
assignment.match`), and a face that fits no track well enough (cost above :attr:`HeadModel.limit`) goes to nobody. A
face found twice (in a window and a crop, or in two overlapping crops) counts once, the window's first.

The face's size is measured by the height of its box (:func:`face_size`, in eye distances of the face turned to the
camera): it does not shrink when the head turns, as the distance between the eye landmarks does (half at 60
degrees); the calibration's size bins use it too.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..detection.assignment import match
from ..sensor.requests import HeadRegion
from .detect import Face

BOX_PER_EYES = 2.95  # a face box's height in eye distances of the face turned to the camera (SCRFD 2.9, YuNet 3.0)
WINDOW_CANVAS = 256  # pixels: the canvas a window is scaled to for the detector (its face then ~24 px between eyes)

__all__ = ["BOX_PER_EYES", "FaceSighting", "HeadModel", "assign_faces", "face_size", "find_faces"]


def face_size(face: Face) -> float:
    """The eye distance (pixels) of ``face`` as if turned to the camera, from its box's height (see the module
    docs)."""
    return float(face.box[3]) / BOX_PER_EYES


@dataclass(frozen=True)
class HeadModel:
    """Where the head is in a person's box ``(x0, y0, x1, y1)`` and how well a face fits it (see the module docs).

    ``region``: the square faces are looked for in (S1's); ``below``, ``across``: the point between the eyes, in box
    heights below the top and from the middle; ``eyes``: the eye distance in box heights; ``sx``, ``sy``, ``ss``: the
    spreads the fit allows (eye distances across and down, and the log of the size ratio); ``limit``: the most a face
    may cost (about three spreads)."""

    region: HeadRegion = field(default_factory=HeadRegion)
    below: float = 0.079
    across: float = -0.004
    eyes: float = 0.037
    sx: float = 1.2
    sy: float = 0.6
    ss: float = 0.4
    limit: float = 12.0

    def square(self, box) -> tuple[float, float, float, float]:
        """The head region ``(x0, y0, x1, y1)``."""
        return self.region.region(box)

    def expected(self, box) -> tuple[tuple[float, float], float]:
        """``((x, y), eye distance)``: where the point between the eyes should be, and how far apart the eyes."""
        x0, y0, x1, y1 = (float(v) for v in box)
        height = max(y1 - y0, 1e-6)
        return ((x0 + x1) / 2 + self.across * height, y0 + self.below * height), self.eyes * height

    def cost(self, face: Face, box) -> float:
        """How badly ``face`` (frame pixels) fits the head of a person in ``box`` (``inf``: not in its head region)."""
        middle = face.points[:2].mean(axis=0)
        x0, y0, x1, y1 = self.square(box)
        if not (x0 <= middle[0] <= x1 and y0 <= middle[1] <= y1):
            return math.inf
        (ex, ey), eyes = self.expected(box)
        eyes = max(eyes, 1e-6)
        dx, dy = (middle[0] - ex) / eyes, (middle[1] - ey) / eyes
        size = math.log(max(face_size(face), 1e-6) / eyes)
        return float((dx / self.sx) ** 2 + (dy / self.sy) ** 2 + (size / self.ss) ** 2)


@dataclass(frozen=True)
class FaceSighting:
    """A face found in a frame, and the track it was given to.

    ``face``: in the frame's pixels; ``local``: in the pixels of ``image``, what it was found in (the frame, or a
    window's image: the face is aligned from there); ``source``: ``"window"`` or ``"frame"``; ``scale``: pixels of
    ``image`` per frame pixel; ``window``: the :class:`~bifold_demo.sensor.WindowResult` it came from; ``track``: the
    track's id (``None``: nobody's); ``cost``: how badly it fits that track's head (:meth:`HeadModel.cost`)."""

    face: Face
    local: Face
    image: np.ndarray = field(repr=False)
    source: str
    scale: float = 1.0
    window: object = field(default=None, repr=False)
    track: int | None = None
    cost: float = math.inf

    @property
    def size(self) -> float:
        """The face's eye distance as if turned to the camera, in the pixels it is aligned from."""
        return face_size(self.local)

    def assigned(self, track: int | None, cost: float) -> FaceSighting:
        return FaceSighting(self.face, self.local, self.image, self.source, self.scale, self.window, track, cost)


def _to_frame(face: Face, window) -> Face:
    """``face`` found in ``window``'s image, in the frame's pixels."""
    corners = window.to_view([[face.box[0], face.box[1]], [face.box[0] + face.box[2], face.box[1] + face.box[3]]])
    (x0, y0), (x1, y1) = corners
    marks = window.to_view(np.asarray(face.landmarks, dtype=np.float64))
    return Face((float(x0), float(y0), float(x1 - x0), float(y1 - y0)), tuple(map(tuple, marks.tolist())), face.score)


def find_faces(engine, image: np.ndarray, regions: dict, windows=(), canvas: int = WINDOW_CANVAS,
               min_size: float = 0.0) -> list[FaceSighting]:
    """The faces in the head regions of a frame: in each of ``windows`` (the frame's
    :class:`~bifold_demo.sensor.WindowResult`, scaled to ``canvas`` pixels for the detector) and in a crop of
    ``image`` (the frame) around each of ``regions`` (``{key: (x0, y0, x1, y1)}`` frame pixels; ``engine.detect`` with
    ``region``). Not yet given to tracks (:func:`assign_faces`). ``min_size``: the smallest face box to keep (frame
    pixels)."""
    found: list[FaceSighting] = []
    for window in windows:
        picture = window.image
        side = max(picture.shape[:2])
        local_canvas = canvas if side > canvas else None  # (a small window: its own size)
        for face in engine.detect(picture, min_size=0.0, canvas=local_canvas):
            in_frame = _to_frame(face, window)
            if in_frame.size >= min_size:
                found.append(FaceSighting(in_frame, face, picture, "window", float(window.window.scale), window))
    h, w = image.shape[:2]
    for region in regions.values():
        x0, y0, x1, y1 = (float(v) for v in region)
        x0, y0, x1, y1 = max(0.0, x0), max(0.0, y0), min(float(w), x1), min(float(h), y1)
        if x1 - x0 < 8 or y1 - y0 < 8:
            continue
        for face in engine.detect(image, min_size=min_size, region=(x0, y0, x1 - x0, y1 - y0)):
            found.append(FaceSighting(face, face, image, "frame"))
    return _once(found)


def _once(found: list[FaceSighting]) -> list[FaceSighting]:
    """Each face once: of faces whose eyes lie within a third of an eye distance of each other, the one from a window
    (more pixels), else the more confident."""
    order = sorted(range(len(found)), key=lambda i: (found[i].source != "window", -found[i].face.score, i))
    kept: list[FaceSighting] = []
    for i in order:
        sighting = found[i]
        eyes = sighting.face.points[:2]
        reach = max(face_size(sighting.face), 1.0) / 3.0
        if all(np.linalg.norm(eyes.mean(axis=0) - k.face.points[:2].mean(axis=0)) > reach for k in kept):
            kept.append(sighting)
    return kept


def assign_faces(sightings: list[FaceSighting], boxes: dict, head: HeadModel | None = None) -> list[FaceSighting]:
    """``sightings`` given to tracks: ``boxes`` maps a track's id to its box ``(x0, y0, x1, y1)``; each face to at most
    one track, each track at most one face, at the least total cost (see the module docs). Returns every sighting,
    with ``track`` set (``None`` for faces that fit nobody)."""
    head = head or HeadModel()
    ids = list(boxes)
    if not sightings or not ids:
        return [s.assigned(None, math.inf) for s in sightings]
    cost = np.array([[head.cost(s.face, boxes[i]) for i in ids] for s in sightings])
    pairs, _, _ = match(cost, head.limit)
    given = {f: (ids[t], float(cost[f, t])) for f, t in pairs}
    return [s.assigned(*given.get(k, (None, math.inf))) for k, s in enumerate(sightings)]
