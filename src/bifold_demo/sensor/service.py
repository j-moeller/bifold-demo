"""Windows with the frames: requests in, windows out, rendered at the frame's moment or cut from a real camera.

:class:`WindowService` belongs to a frame source. Anyone (the person tracker, on its own thread) gives it standing
requests (:meth:`WindowService.request`); with every frame the source makes, the service's scheduler picks at most a
few of them and the service delivers their windows on the frame (``frame.windows``):

- a scene source renders them right after the frame, from the same state of the scene (:meth:`WindowService.for_scene`,
  :func:`render_windows`), so the frame and its windows show the same moment;
- a real camera cuts them from its full-resolution picture while the detector sees a smaller copy
  (:meth:`WindowService.for_image`, :func:`crop_windows`): "capture high, detect on a scaled copy".

Each :class:`WindowResult` holds the image, its :class:`~.geometry.Window` (where it is in sensor and view pixels and
its scale, to map faces back to the frame), the camera preset, the frame's time and index and what the request
asked. For a scene it also holds :class:`Truth`: who the scene knows is in the window and where their eyes are, for
the evaluation only (recognition must not use it).

The camera preset (:mod:`.camera_model`) is applied to each window at its own pixel scale. A preset that brings its
own sensor size (``lines``) places the window on that sensor instead of the declared one; a request's own ``sensor``
comes before both. For its motion blur the preset gets the person's speed from :class:`Motion` and, when they move,
their coverage of the window (:meth:`~bifold_demo.scene.renderer.Renderer.window_mask`).
"""

from __future__ import annotations

import threading
import time
import weakref
from collections.abc import Iterable
from dataclasses import dataclass, replace

import cv2
import numpy as np

from .camera_model import CameraModel, camera_preset
from .geometry import DEFAULT_SCALE, Sensor, Window
from .requests import DEFAULT_WINDOWS, HeadRegion, Scheduler, WindowRequest, head_point_region

MULTIPLE = 16  # window sides are grown to a multiple of this, so that sizes repeat and their framebuffers are reused
EYE_BONES = ("Bip01 REye", "Bip01 LEye")  # the eyes of the Rocketbox skeleton (the person's right, then left)
MOTION_GAP = 0.5  # seconds: a person seen longer ago than this has no known speed


@dataclass(frozen=True)
class Truth:
    """What the scene knows about the person in a window: for evaluation (face sizes, misses), never for recognition.

    Attributes:
        name: the scene object's name; ``avatar``: its model's name.
        distance: metres from the camera to the point between the eyes.
        eyes: ``((x, y), (x, y))``: the person's right and left eye (the eye bones) in window pixels.
        eye_distance: pixels between them in the window (fewer when the head is turned).
        frontal_eye_distance: the pixels between them if the face were turned to the camera: the face's size.
        inside: both eyes lie in the window.
        velocity: ``(x, y)`` window pixels per second of the point between the eyes (0 when unknown).
        visible: how much the person is the nearest surface where their eyes are (0-1: 0 hidden behind someone or
            something; ``None``: not measured, see ``render_windows(visibility=True)``).
    """

    name: str
    avatar: str
    distance: float
    eyes: tuple
    eye_distance: float
    inside: bool
    velocity: tuple = (0.0, 0.0)
    frontal_eye_distance: float = 0.0
    visible: float | None = None


@dataclass(eq=False)
class WindowResult:
    """A window delivered with a frame.

    Attributes:
        request: what was asked (``tag``, the box or region, the wanted face size ...).
        window: where it is: ``window.sensor_rect``, ``window.view_rect``, ``window.level``, ``window.scale`` and the
            maps between window and view pixels (``to_view``, ``box_to_view``).
        image: HxWx3 BGR pixels, after the camera model.
        time: the scene time of the frame (all windows of a frame have the frame's time); ``None`` for a real camera.
        frame: the index of the frame it was delivered with.
        preset: the name of the camera preset applied.
        eye_distance: the pixels between the eyes the request's estimate gives in the window (``None``: no estimate).
        truth: who the scene knows is in it (``None`` for a real camera, or when not asked for).
        render_ms, camera_ms: the time rendering (or cutting) and the camera model took.
    """

    request: WindowRequest
    window: Window
    image: np.ndarray
    time: float | None
    frame: int
    preset: str = "ideal"
    eye_distance: float | None = None
    truth: Truth | None = None
    render_ms: float = 0.0
    camera_ms: float = 0.0

    @property
    def tag(self):
        return self.request.tag

    @property
    def view_rect(self) -> tuple[float, float, float, float]:
        return self.window.view_rect

    def to_view(self, points) -> np.ndarray:
        """Window pixels ``(..., 2)`` to view pixels."""
        return self.window.to_view(points)

    def box_to_view(self, box) -> tuple[float, float, float, float]:
        """A box ``(x0, y0, x1, y1)`` in window pixels (a face found in it), in view pixels."""
        return self.window.box_to_view(box)


def plan_request(request: WindowRequest, sensor: Sensor, camera=None, head: HeadRegion | None = None,
                 multiple: int = 1, max_side: int | None = None) -> tuple[Window, float | None] | None:
    """``(window, eye distance in window pixels or None)`` for ``request`` on ``sensor``, or ``None`` when nothing of
    its region is in the view (or its head point is behind the camera). ``camera``: the scene's, for head points;
    ``max_side``: the longest side a window may have (a renderer's limit)."""
    head = head or HeadRegion()
    view = sensor.view
    if request.head is not None:
        if camera is None:
            raise ValueError("a window around a head point needs the scene's camera")
        found = head_point_region(camera, view, request.head, request.head_size, request.eye_m)
        if found is None:
            return None
        region, eyes = found
    elif request.box is not None:
        region, eyes = head.region(request.box), head.eye_distance(request.box)
    else:
        region, eyes = request.region, None
    if request.eye_distance is not None:
        eyes = float(request.eye_distance)
    x0, y0, x1, y1 = region
    if x1 <= 0 or y1 <= 0 or x0 >= view[0] or y0 >= view[1]:
        return None
    window = sensor.plan(region, eyes, request.face_px, request.max_pixels, multiple, max_side)
    return window, (eyes * window.scale if eyes else None)


# -- what the scene knows: eyes and their motion -----------------------------------------------------------------------

_eye_bones: dict[int, tuple] = {}  # id(model) -> (weakref, (right, left) bone indices or None)


def eye_points(person) -> np.ndarray | None:
    """``(2, 3)``: where ``person``'s right and left eye are in the scene (the eye bones, posed), or ``None`` for a
    model without them."""
    model = getattr(person, "skinned_model", None)
    if model is None:
        return None
    entry = _eye_bones.get(id(model))
    if entry is None or entry[0]() is not model:
        found = [model.skeleton.find(name) for name in EYE_BONES]
        entry = (weakref.ref(model), None if None in found else tuple(found))
        _eye_bones[id(model)] = entry
    if entry[1] is None:
        return None
    placed = person.matrices()[0]
    local = np.asarray(person.pose, dtype=np.float64)[list(entry[1]), :3, 3]
    return local @ placed[:3, :3].T + placed[:3, 3]


class Motion:
    """Where people's eyes were in the frames seen so far: their speed in the scene (for motion blur)."""

    def __init__(self):
        self._last: dict[int, tuple] = {}  # id(person) -> (weakref, time, point between the eyes)
        self._velocity: dict[int, np.ndarray] = {}

    def observe(self, scene) -> None:
        """Note where everyone visible is at the scene's time (once a frame)."""
        t = float(scene.time)
        seen = set()
        for obj in scene:
            if not obj.current_visible:
                continue
            eyes = eye_points(obj)
            if eyes is None:
                continue
            key, point = id(obj), eyes.mean(axis=0)
            seen.add(key)
            last = self._last.get(key)
            if last is not None and last[0]() is obj and 0.0 < t - last[1] <= MOTION_GAP:
                self._velocity[key] = (point - last[2]) / (t - last[1])
            elif last is None or last[0]() is not obj or t != last[1]:
                self._velocity[key] = np.zeros(3)  # first seen, or seen too long ago
            self._last[key] = (weakref.ref(obj), t, point)
        for key in list(self._last):
            if key not in seen:
                del self._last[key]
                self._velocity.pop(key, None)

    def velocity(self, person) -> np.ndarray:
        """Metres per second of the point between ``person``'s eyes (zero when unknown)."""
        last = self._last.get(id(person))
        if last is None or last[0]() is not person:
            return np.zeros(3)
        return self._velocity.get(id(person), np.zeros(3))


def find_truth(scene, window: Window, motion: Motion | None = None, subject=None) -> Truth | None:
    """Who the scene knows is in ``window``: ``subject`` when given, else, of the visible people whose eyes are in
    the window, the one nearest its middle counted in their own (frontal) eye distances, so that the person the window
    was placed on wins over a small face behind them (``None`` when nobody's eyes are in it)."""
    found = _find_person(scene, window, motion, subject)
    return None if found is None else found[1]


def _find_person(scene, window: Window, motion: Motion | None, subject) -> tuple[object, Truth] | None:
    """``(person, truth)`` for :func:`find_truth`."""
    camera = scene.camera
    clip_from_world = window.crop_matrix() @ camera.projection_matrix(window.sensor.view_aspect) @ camera.view_matrix()
    view_matrix = camera.view_matrix()
    focal = window.sensor.height / window.level / 2 / np.tan(np.radians(camera.current_fov) / 2)  # window pixels
    size = np.array(window.size, dtype=np.float64)

    def pixels(points):
        clip = np.c_[points, np.ones(len(points))] @ clip_from_world.T
        if (clip[:, 3] <= 1e-9).any():
            return None
        ndc = clip[:, :2] / clip[:, 3:]
        return np.c_[(ndc[:, 0] + 1) / 2 * size[0], (1 - ndc[:, 1]) / 2 * size[1]]

    best = None
    for person in [subject] if subject is not None else list(scene):
        if not person.current_visible:
            continue
        eyes = eye_points(person)
        if eyes is None:
            continue
        px = pixels(eyes)
        if px is None:
            continue
        middle = px.mean(axis=0)
        inside = bool(((px >= 0) & (px <= size)).all())
        if subject is None and not (0 <= middle[0] <= size[0] and 0 <= middle[1] <= size[1]):
            continue
        depth = -(view_matrix[2, :3] @ eyes.mean(axis=0) + view_matrix[2, 3])  # along the line of sight
        frontal = focal * float(np.linalg.norm(eyes[0] - eyes[1])) / max(depth, 1e-6)  # turned heads too
        rank = float(np.linalg.norm(middle - size / 2)) / max(frontal, 1e-6)
        if best is None or rank < best[0]:
            best = (rank, person, eyes, px, inside, frontal)
    if best is None:
        return None
    _rank, person, eyes, px, inside, frontal = best
    middle = eyes.mean(axis=0)
    velocity = (0.0, 0.0)
    if motion is not None:
        speed = motion.velocity(person)
        if speed.any():
            step = 1.0 / 240.0  # seconds: a small move, projected
            moved = pixels(np.array([middle, middle + speed * step]))
            if moved is not None:
                velocity = tuple(float(v) for v in (moved[1] - moved[0]) / step)
    model = getattr(person, "skinned_model", None)
    return person, Truth(name=str(person.name), avatar=str(getattr(model, "name", "")),
                         distance=float(np.linalg.norm(middle - np.asarray(camera.current_position, dtype=np.float64))),
                         eyes=tuple(tuple(float(v) for v in p) for p in px),
                         eye_distance=float(np.linalg.norm(px[0] - px[1])), inside=inside, velocity=velocity,
                         frontal_eye_distance=float(frontal))


def eyes_visible(mask: np.ndarray, eyes) -> float:
    """How much a person's coverage ``mask`` (of :meth:`~bifold_demo.scene.renderer.Renderer.window_mask`) holds at
    their ``eyes`` (window pixels): the mean over the 3x3 pixels around each eye; 0 for eyes outside the window."""
    h, w = mask.shape
    values = []
    for x, y in eyes:
        cx, cy = int(np.floor(x)), int(np.floor(y))
        if 0 <= cx < w and 0 <= cy < h:
            values.append(float(mask[max(0, cy - 1):cy + 2, max(0, cx - 1):cx + 2].mean()))
        else:
            values.append(0.0)
    return float(np.mean(values))


# -- windows ----------------------------------------------------------------------------------------------------------


def _sensor_for(request: WindowRequest, model: CameraModel, declared: Sensor, view) -> Sensor:
    """The sensor a request's window lies on: its own, else its camera's size, else the declared one."""
    if request.sensor is not None:
        return request.sensor
    if model.lines is not None:
        return Sensor.with_lines(model.lines, view)
    return declared


def render_windows(scene, renderer, requests: Iterable[WindowRequest], sensor: Sensor | None = None, frame: int = 0,
                   head: HeadRegion | None = None, multiple: int = MULTIPLE, truth: bool = True,
                   motion: Motion | None = None, preset="ideal", visibility: bool = False) -> list[WindowResult]:
    """Render the windows of ``requests`` for ``scene`` as ``renderer`` drew it last (call it right after
    ``renderer.render(scene)``: the windows then show the frame's moment and reuse its uploads), through the camera
    ``preset`` (a name or a :class:`~.camera_model.CameraModel`; a request's own comes first).

    ``sensor``: the declared sensor (default: :data:`~.geometry.DEFAULT_SCALE` times the renderer's view); requests
    whose region is not in the view are left out. ``truth``: find who is in each window (see :class:`Truth`);
    ``visibility``: also measure whether their eyes are hidden (``Truth.visible``: one more draw of the person);
    ``motion``: people's speeds, for the truth and the motion blur. ``frame``: the frame's index (results and the
    noise's seed)."""
    view = (renderer.width, renderer.height)
    declared = sensor or Sensor.scaled(DEFAULT_SCALE, view)
    results = []
    for request in requests:
        model = camera_preset(request.preset if request.preset is not None else preset)
        chosen = _sensor_for(request, model, declared, view)
        if chosen.view != view:
            raise ValueError(f"a sensor for a {chosen.view[0]}x{chosen.view[1]} view cannot make windows of a "
                             f"{view[0]}x{view[1]} renderer's view")
        planned = plan_request(request, chosen, scene.camera, head, multiple, renderer.max_window)
        if planned is None:
            continue
        window, eyes = planned
        start = time.perf_counter()
        image = renderer.render_window(scene, window.ndc, window.size, aspect=chosen.view_aspect)
        render_ms = (time.perf_counter() - start) * 1000.0
        found = _find_person(scene, window, motion, request.subject) if truth or model.needs_motion else None
        mask = None
        if truth and visibility and found is not None:
            mask = renderer.window_mask(found[0])
            found = (found[0], replace(found[1], visible=eyes_visible(mask, found[1].eyes)))
        start = time.perf_counter()
        velocity = (0.0, 0.0)
        if model.needs_motion and found is not None and np.hypot(*found[1].velocity) * model.exposure >= 0.5:
            velocity = found[1].velocity
            if mask is None:
                mask = renderer.window_mask(found[0])  # only the moving person is smeared
        else:
            mask = None
        image = model.apply(image, window.level, velocity, mask, model.window_seed(frame, request.tag,
                                                                                 window.sensor_rect))
        camera_ms = (time.perf_counter() - start) * 1000.0  # the camera model, with the mask it needs
        results.append(WindowResult(request, window, image, float(scene.time), frame, model.name, eyes,
                                    found[1] if truth and found is not None else None, render_ms, camera_ms))
    return results


def crop_windows(image: np.ndarray, view, requests: Iterable[WindowRequest], sensor: Sensor | None = None,
                 frame: int = 0, head: HeadRegion | None = None, multiple: int = MULTIPLE,
                 preset="ideal") -> list[WindowResult]:
    """Windows cut from a real camera's full-resolution ``image`` (BGR), of which the detector sees a copy of
    ``view`` = (width, height) pixels: the image is the sensor (``sensor`` must match it; a request's own sensor is
    not used), and a window at level ``k`` is its crop averaged over ``k x k`` pixels. Requests with head points are
    left out (a real camera knows no scene). ``preset`` adds a camera's faults on top of the real camera's own (its
    sensor size is not used)."""
    h, w = image.shape[:2]
    sensor = sensor or Sensor(w, h, view)
    if sensor.size != (w, h):
        raise ValueError(f"a {sensor} sensor cannot cut windows from a {w}x{h} picture")
    results = []
    for request in requests:
        if request.head is not None:
            continue
        model = camera_preset(request.preset if request.preset is not None else preset)
        planned = plan_request(request, sensor, None, head, multiple)
        if planned is None:
            continue
        window, eyes = planned
        start = time.perf_counter()
        x0, y0, x1, y1 = window.sensor_rect
        crop = image[y0:y1, x0:x1]
        crop = cv2.resize(crop, window.size, interpolation=cv2.INTER_AREA) if window.level > 1 else crop.copy()
        cut_ms = (time.perf_counter() - start) * 1000.0
        start = time.perf_counter()
        crop = model.apply(crop, window.level, seed=model.window_seed(frame, request.tag, window.sensor_rect))
        results.append(WindowResult(request, window, crop, None, frame, model.name, eyes, None, cut_ms,
                                    (time.perf_counter() - start) * 1000.0))
    return results


class WindowService:
    """Windows with the frames of one source (see the module docs).

    Args:
        sensor: the declared sensor; ``None``: :data:`~.geometry.DEFAULT_SCALE` times the view for a scene, the full
            picture for a real camera.
        max_windows: the most windows a frame (the scheduler's).
        priority: the scheduler's priority hook (see :class:`~.requests.Scheduler`).
        head: where heads are in person boxes (:class:`~.requests.HeadRegion`).
        multiple: window sides are grown to a multiple of it.
        truth: find who the scene knows is in each window; ``visibility``: and whether their eyes are hidden.
        preset: the camera preset (a name of :data:`~.camera_model.PRESETS` or a
            :class:`~.camera_model.CameraModel`); a preset with its own sensor size uses it instead of ``sensor``.

    :meth:`request` and :meth:`request_once` may be called from any thread; the windows are made on the source's.
    ``last``: the windows of the latest frame.
    """

    def __init__(self, sensor: Sensor | None = None, max_windows: int = DEFAULT_WINDOWS, priority=None,
                 head: HeadRegion | None = None, multiple: int = MULTIPLE, truth: bool = True, preset="ideal",
                 visibility: bool = False):
        self.sensor = sensor
        self.preset = camera_preset(preset)
        self.scheduler = Scheduler(max_windows, priority)
        self.head = head or HeadRegion()
        self.multiple = multiple
        self.truth, self.visibility = truth, visibility
        self.motion = Motion()
        self.last: list[WindowResult] = []
        self._lock = threading.Lock()
        self._standing: list[WindowRequest] = []
        self._once: list[WindowRequest] = []

    def request(self, requests: Iterable[WindowRequest]) -> None:
        """Replace the standing requests: they are served, a few a frame, until replaced."""
        requests = list(requests)
        with self._lock:
            self._standing = requests

    def request_once(self, requests: Iterable[WindowRequest]) -> None:
        """Requests for the next frame only (beside the standing ones)."""
        requests = list(requests)
        with self._lock:
            self._once += requests

    def clear(self) -> None:
        with self._lock:
            self._standing, self._once = [], []

    @property
    def requests(self) -> list[WindowRequest]:
        """The standing requests."""
        with self._lock:
            return list(self._standing)

    def _choose(self, frame: int) -> list[WindowRequest]:
        with self._lock:
            pending, self._once = self._standing + self._once, []
        return self.scheduler.choose(pending, frame) if pending else []

    def for_scene(self, scene, renderer, frame) -> list[WindowResult]:
        """The windows of ``frame`` (a :class:`~bifold_demo.sources.base.Frame` that ``renderer`` just drew of
        ``scene``)."""
        self.motion.observe(scene)
        chosen = self._choose(frame.index)
        self.last = render_windows(scene, renderer, chosen, self.sensor, frame.index, self.head, self.multiple,
                                   self.truth, self.motion, self.preset, self.visibility) if chosen else []
        return self.last

    def for_image(self, image: np.ndarray, frame) -> list[WindowResult]:
        """The windows of ``frame`` cut from the full-resolution picture ``image`` it was scaled from."""
        chosen = self._choose(frame.index)
        sensor = self.sensor
        if sensor is not None and sensor.size != (image.shape[1], image.shape[0]):
            sensor = None  # the camera delivers another size than declared: use what it delivers
        self.last = crop_windows(image, frame.size, chosen, sensor, frame.index, self.head, self.multiple,
                                 self.preset) if chosen else []
        return self.last
