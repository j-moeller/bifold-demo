"""A virtual high-resolution sensor: high-resolution windows on the heads of the people in the picture.

The main view stays at the size the detector and the screen use (1280 x 720); faces get their pixels from *windows*.
A window is a rectangle of a declared high-resolution sensor whose full frame is exactly the scene camera's view
(by default six times the view in each direction: 7680 x 4320). It is drawn through an off-axis part of the camera's
frustum (``crop_matrix @ P``, :meth:`bifold_demo.scene.renderer.Renderer.render_window`) at the sensor's pixel scale,
with the view's renderer and MSAA, so it is exactly that crop of a picture of the whole frame at the sensor's
resolution, for the cost of the window's pixels. Its rectangle is snapped to the sensor's pixel grid, at a binning
level of the sensor when a face would get more pixels than asked for (see :mod:`.geometry`).

Only what may show in a window is drawn (:mod:`bifold_demo.scene.frustum`).

- :class:`Sensor`: the declared sensor, its levels and the maps between view and sensor pixels;
  :meth:`Sensor.plan` places a window on a region of the view.
- :class:`Window`: a snapped rectangle: its place in sensor and view pixels, its scale, its normalised device
  coordinates and crop matrix, and the maps between its pixels and the view's.
- :class:`WindowRequest` (:mod:`.requests`): a window wanted for a tag (a person track): a person's box, a region or a
  3D head point; the face size wanted; a pixel limit; a priority. :class:`HeadRegion`: the head's square in a box.
  :class:`Scheduler`: at most a few windows a frame, by priority, tracks taking turns.
- :class:`WindowService` (:mod:`.service`): requests in, windows out with every frame of a source
  (``frame.windows``: :class:`WindowResult`, with the window's image, place, scale, time and :class:`Truth`);
  :func:`render_windows` and :func:`crop_windows` for direct use.

The design and the measurements behind it are in ``docs/research/high_res_faces.md`` (sections 2.1 and 3.4).
"""

from .geometry import DEFAULT_SCALE, Sensor, Window
from .requests import DEFAULT_FACE_PX, DEFAULT_MAX_PIXELS, HeadRegion, Scheduler, WindowRequest
from .service import Truth, WindowResult, WindowService, crop_windows, render_windows

__all__ = [
    "DEFAULT_FACE_PX",
    "DEFAULT_MAX_PIXELS",
    "DEFAULT_SCALE",
    "HeadRegion",
    "Scheduler",
    "Sensor",
    "Truth",
    "Window",
    "WindowRequest",
    "WindowResult",
    "WindowService",
    "crop_windows",
    "render_windows",
]
