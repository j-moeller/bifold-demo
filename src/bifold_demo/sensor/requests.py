"""What windows are asked for: requests, the head region of a person's box, and the choice of a frame's windows.

A request names a region of the main view in one of three ways: a person's box (the person detector's, in view
pixels), from which :class:`HeadRegion` takes a square around the top of the box; a region given as it is (a face
found in an earlier window, with a margin); or a head point in the scene (for the evaluation, which knows where the
heads are). It also says how many pixels the face should get between the eyes (``face_px``) and how many pixels the
window may have at most; :meth:`.geometry.Sensor.plan` turns that into a level of the sensor.

At most a few windows are rendered a frame. :class:`Scheduler` picks them: the highest priority first, and among
equal priorities the tag (a person track) that has waited longest, so that tracks take turns. Its ``priority`` hook is
where the person tracker puts its own order (undecided, frontal and large faces first).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Hashable, Iterable
from dataclasses import dataclass

import numpy as np

from .geometry import Sensor

DEFAULT_FACE_PX = 112.0  # wanted pixels between the eyes: the low end of 112-160 (see docs/research/high_res_faces.md)
DEFAULT_MAX_PIXELS = 1024 * 1024  # most pixels a window may have
DEFAULT_WINDOWS = 4  # windows a frame
EYE_DISTANCE_M = 0.063  # metres between the eye centres of an adult (about 50-75 mm; Dodgson 2004)


@dataclass(frozen=True)
class HeadRegion:
    """Where a person's head is in their box ``(x0, y0, x1, y1)`` (main-view pixels).

    A square ``side`` box heights wide, centred across the box and ``centre`` box heights below its top, enlarged by
    ``margin`` (a share of the side) so that the face detector finds the face also when the box is a little off. The
    defaults are the starting values of ``docs/research/face_recognition_sota.md`` (section 3.2, stage 3), to be tuned
    by the person tracker. ``eyes``: the distance between the eyes (as if facing the camera) as a share of the box
    height, to choose the window's level when nothing better is known.

    Measured on YOLO's boxes of the fair (91 people in 10 frames, ``docs/research/virtual_sensor_results.md``): the
    eyes lie 0.079 box heights below the top (0.060-0.105 for 90 %) and the eye distance is 0.034 box heights
    (0.033-0.036). These defaults hold the whole face in every case, the face a quarter of the square's side;
    ``HeadRegion(centre=0.10, side=0.20)`` holds it in 96 % with 2.2 times fewer pixels.
    """

    centre: float = 0.12
    side: float = 0.30
    margin: float = 0.30
    eyes: float = 0.034

    def region(self, box) -> tuple[float, float, float, float]:
        """The square around the head, in view pixels."""
        x0, y0, x1, y1 = (float(v) for v in box)
        height = y1 - y0
        cx, cy = (x0 + x1) / 2, y0 + self.centre * height
        half = self.side * (1 + self.margin) * height / 2
        return cx - half, cy - half, cx + half, cy + half

    def eye_distance(self, box) -> float:
        """About how many view pixels lie between the person's eyes."""
        return self.eyes * (float(box[3]) - float(box[1]))


@dataclass(frozen=True, eq=False)
class WindowRequest:
    """A window wanted for ``tag`` (a person track, or any label the caller chooses).

    The region, one of:
        box: the person's box ``(x0, y0, x1, y1)`` in view pixels; the window covers its :class:`HeadRegion`.
        region: exactly this rectangle in view pixels (a face found in an earlier window, with a margin).
        head: a point ``(x, y, z)`` of the scene (the evaluation's heads); the window covers a square ``head_size``
            metres wide around it, as the camera sees it.

    The scale:
        face_px: pixels wanted between the eyes: the window is at the coarsest level of the sensor that still gives
            them (the sensor itself when it cannot); ``None`` for the sensor's own scale.
        eye_distance: view pixels between the eyes, when known (else estimated from the box, or from ``eye_m``
            metres at the head point).
        max_pixels: the most pixels the window may have (coarser levels until it fits); ``None`` for no limit.

    And:
        priority: higher first (see :class:`Scheduler`).
        frame: the index of the frame the box or region comes from (the window is rendered with a later frame).
        sensor: a sensor for this request alone (else the preset's, else the service's).
        preset: a camera preset for this request alone (a name or a :class:`~.camera_model.CameraModel`).
        subject: the scene's person this is about, when the caller knows it (the evaluation): the truth and the
            camera model's motion blur use it; otherwise the person whose eyes are nearest the window's middle.
    """

    tag: Hashable
    box: tuple | None = None
    region: tuple | None = None
    head: tuple | None = None
    face_px: float | None = DEFAULT_FACE_PX
    eye_distance: float | None = None
    max_pixels: int | None = DEFAULT_MAX_PIXELS
    priority: float = 0.0
    frame: int | None = None
    sensor: Sensor | None = None
    preset: object = None
    subject: object = None
    head_size: float = 0.5
    eye_m: float = EYE_DISTANCE_M

    def __post_init__(self):
        given = [name for name in ("box", "region", "head") if getattr(self, name) is not None]
        if len(given) != 1:
            raise ValueError(f"a window request needs exactly one of box, region and head (got {given or 'none'})")
        for name in ("box", "region"):
            rect = getattr(self, name)
            if rect is not None:
                rect = tuple(float(v) for v in rect)
                if len(rect) != 4 or not (rect[2] > rect[0] and rect[3] > rect[1]):
                    raise ValueError(f"{name} must be (x0, y0, x1, y1) with x1 > x0 and y1 > y0, got {rect}")
                object.__setattr__(self, name, rect)
        if self.head is not None:
            object.__setattr__(self, "head", tuple(float(v) for v in self.head))


def head_point_region(camera, view, point, size: float = 0.5, eye_m: float = EYE_DISTANCE_M):
    """``(region, eye_distance)`` of a head at ``point`` (scene coordinates) for ``camera``'s view of ``view`` =
    (width, height) pixels: the square of ``size`` metres around it, across the line of sight, in view pixels, and the
    view pixels between two eyes ``eye_m`` apart there (frontal). ``None`` when the point is behind the camera."""
    w, h = view
    clip_from_world = camera.projection_matrix(w / h) @ camera.view_matrix()
    point = np.asarray(point, dtype=np.float64)
    view_matrix = camera.view_matrix()
    right, up = view_matrix[0, :3], view_matrix[1, :3]  # the camera's axes in the scene
    ray = point - np.asarray(camera.current_position, dtype=np.float64)
    ray /= max(np.linalg.norm(ray), 1e-12)
    across = right - (right @ ray) * ray  # across the line of sight, as level as the camera
    upward = up - (up @ ray) * ray
    across /= max(np.linalg.norm(across), 1e-12)
    upward /= max(np.linalg.norm(upward), 1e-12)

    def pixel(p):
        clip = clip_from_world @ np.append(p, 1.0)
        if clip[3] <= 1e-9:
            return None
        return np.array([(clip[0] / clip[3] + 1) / 2 * w, (1 - clip[1] / clip[3]) / 2 * h])

    points = [pixel(point + d) for d in (np.zeros(3), -across * size / 2, across * size / 2, -upward * size / 2,
                                         upward * size / 2, -across * eye_m / 2, across * eye_m / 2)]
    if any(p is None for p in points):
        return None
    centre, left, right_, low, high, eye_a, eye_b = points
    half_w, half_h = np.linalg.norm(right_ - left) / 2, np.linalg.norm(high - low) / 2
    region = (centre[0] - half_w, centre[1] - half_h, centre[0] + half_w, centre[1] + half_h)
    return region, float(np.linalg.norm(eye_b - eye_a))


class Scheduler:
    """Chooses a frame's windows among the requests: at most ``max_windows``, the highest priority first, and among
    equal priorities in turn: tags that never had a window first, then the one served longest ago (a queue: a tag
    served goes to its back), so that ``n`` tags share ``k`` windows a frame evenly.

    ``priority``: the hook the person tracker fills, ``fn(request, waited) -> float``, with ``waited`` the frames since
    the tag last had a window (since it was first asked for, when it never had one); by default the request's own
    ``priority``. A high priority can keep lower ones waiting for good; a hook that adds ``waited`` to its score makes
    them take turns.
    """

    def __init__(self, max_windows: int = DEFAULT_WINDOWS,
                 priority: Callable[[WindowRequest, int], float] | None = None):
        self.max_windows = int(max_windows)
        self.priority = priority
        self._served: dict = {}  # tag -> (index of the frame it last had a window in, place in the queue)
        self._asked: dict = {}  # tag -> index of the frame it was first asked for in
        self._count = 0  # windows handed out so far: the place in the queue of the latest

    def waited(self, tag, frame: int) -> int | None:
        """Frames since ``tag`` last had a window, or since it was first asked for (``None``: never asked for)."""
        last = self._served.get(tag)
        if last is not None:
            return frame - last[0]
        first = self._asked.get(tag)
        return None if first is None else frame - first

    def choose(self, requests: Iterable[WindowRequest], frame: int) -> list[WindowRequest]:
        """The requests that get a window in frame ``frame`` (best first); they count as served."""
        requests = list(requests)
        asked = {request.tag for request in requests}
        self._served = {tag: last for tag, last in self._served.items() if tag in asked}  # forget gone tags
        self._asked = {tag: first for tag, first in self._asked.items() if tag in asked}
        for request in requests:
            self._asked.setdefault(request.tag, frame)
        ranked = []
        for order, request in enumerate(requests):
            score = request.priority if self.priority is None else float(self.priority(request,
                                                                                       self.waited(request.tag, frame)))
            last = self._served.get(request.tag)
            ranked.append((-score, -math.inf if last is None else last[1], order))
        ranked.sort()
        chosen = [requests[order] for *_, order in ranked[:max(0, self.max_windows)]]
        for request in chosen:
            self._count += 1
            self._served[request.tag] = (frame, self._count)
        return chosen
