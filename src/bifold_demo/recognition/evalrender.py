"""Rendered stills for the evaluation (``bifold-faces eval``, :mod:`.evaluate`): one person in the fair's empty hall,
seen by its fixed camera through the virtual sensor's window or in the 1280x720 view. Needs OpenGL.

A still's conditions (:data:`~.evaluate.RENDER`) say everything that is drawn:

* **Who.** The ``body`` avatar, with its own face (``face: own``: the avatar is the identity), or with face textures
  made from renders of the ``identity``'s avatar (``face: renders``, the head-texture stages from photos,
  ``characters.headtex``), on the body's own head or on the head fitted to those renders (``head: fitted``,
  ``characters.headshape``, S3). The renders standing in for the identity's photos are **set A**
  (``shots``, default ``"three-quarter"``: S3's six, a frontal one, turned 30 and 60 degrees each way, nodded 15,
  1024x768, 0.9 m, a plain studio light). The gallery is made of other renders (**set B**, built by
  :mod:`.protocols`: windows in the hall, its light, another expression, other turns), so no image is in both.
  The textures are kept on disk by the renders' contents (``characters.headtex``'s cache), so a re-run or another
  manifest reuses them.
* **Where and how.** At ``distance`` metres from the camera, ``side`` metres off its axis, facing the camera and
  turned ``yaw`` degrees (the body turns: up to 90), nodded ``pitch`` (the head), with an ``expression``; the hall's
  ``light`` (:data:`LIGHTS`); an ``occluder`` (:func:`occluders`: a flat board over the lower face, or a pillar edge
  over one side of the face up to the nose); the renderer's ``quality``, ``linear`` light and ``detail`` (2048 head
  textures where the eyes are 64 pixels apart, or 1024 only).
* **Through what.** ``view: window``: a window of the 8K sensor at exactly ``ied`` pixels between the eyes of the face
  turned to the camera (:class:`~bifold_demo.sensor.WindowRequest` ``eye_px``), through the camera ``preset``, its
  noise seeded by ``seed``; ``view: frame``: the 1280x720 view itself (today's system: no preset), the person placed
  so that the face has ``ied`` pixels between the eyes there, and the face looked for in the head region of the
  person's box, as the live pipeline looks in a frame.

Each still comes with what the scene knows of it (the head's yaw, pitch and eye distance as measured): the
evaluation records it, and the tests check that the conditions asked are the ones drawn.
"""

from __future__ import annotations

import math
from collections import OrderedDict

import numpy as np

VIEW = (1280, 720)
# the hall's light per condition: (direction it travels, ambient, colour, hemisphere); "fair" is the hall's own
LIGHTS = {
    "side": ((-1.0, -0.3, -0.15), 0.30, (255, 250, 240), 0.1),  # a harsh light from one side, deep shadows
    "warm": ((-0.25, -1.0, -0.35), 0.55, (255, 190, 120), 0.3),  # tungsten-coloured lamps
    "top": ((0.0, -1.0, -0.05), 0.40, (255, 250, 245), 0.5),  # a downlight: the eyes in the brows' shadow
}
FACES_KEPT = 2  # people's textures kept in memory (each 50-70 MB at 2048, and as much again on llvmpipe's GPU)
MODELS_KEPT = 3  # avatars kept loaded (each about 100 MB of textures, and as much on the GPU)
EYE_M = 0.063


def occluders(kind: str, eyes: np.ndarray, camera_position) -> list:
    """The scene objects of an ``occluder`` (``"low"``, ``"side"``) in front of a face whose eyes are ``eyes`` (``(2,
    3)``), as seen from ``camera_position``: ``low``, a board 10 cm in front of the face from the nose tip to below
    the chin (a held tablet, a mask); ``side``, a pillar 40 cm in front of the face whose edge lies on the line from
    the camera to the point between the eyes, hiding one half of the face."""
    from ..scene import Box

    if kind == "none":
        return []
    middle = np.asarray(eyes, np.float64).mean(axis=0)
    to_camera = np.asarray(camera_position, np.float64) - middle
    to_camera /= np.linalg.norm(to_camera)
    up = np.array([0.0, 1.0, 0.0])
    right = np.cross(up, to_camera)
    right /= np.linalg.norm(right)
    bearing = math.degrees(math.atan2(to_camera[0], to_camera[2]))
    if kind == "low":
        centre = middle + 0.10 * to_camera - 0.0825 * up
        return [Box(size=(0.20, 0.095, 0.012), position=tuple(centre), rotation=(0.0, bearing, 0.0),
                    color=(196, 202, 214), name="occluder low")]
    if kind == "side":
        width = 0.30
        centre = middle + 0.40 * to_camera + right * (width / 2.0) - 0.05 * up
        return [Box(size=(width, 0.60, 0.03), position=tuple(centre), rotation=(0.0, bearing, 0.0),
                    color=(70, 72, 82), name="occluder side")]
    raise ValueError(f"unknown occluder {kind!r}")


class FaceMaker:
    """The people of the evaluation (see the module docs): avatars loaded (the last :data:`MODELS_KEPT`), and faces
    made from set A, on the carrier's own or the fitted head (the last :data:`FACES_KEPT` in memory, all of them on
    disk in ``cache``). ``library``: the avatars' folder (default: ``$BIFOLD_CHARACTERS``). Needs OpenGL for set A."""

    def __init__(self, library=None, cache=None, progress=print):
        from ..characters.library import CharacterLibrary

        self.library = CharacterLibrary(library) if library else CharacterLibrary()
        self.cache = cache
        self.progress = progress or (lambda *_: None)
        self._models: OrderedDict = OrderedDict()
        self._faces: OrderedDict = OrderedDict()
        self._photo_tools = None

    def close(self) -> None:
        for tool in self._photo_tools or ():
            tool.close()
        self._photo_tools = None

    def model(self, name: str):
        """The avatar ``name``, loaded (the last :data:`MODELS_KEPT` stay loaded: the stills come ordered by body)."""
        if name in self._models:
            self._models.move_to_end(name)
            return self._models[name]
        self._models[name] = self.library.load(name)
        if len(self._models) > MODELS_KEPT:
            self._models.popitem(last=False)
            _collect()
        return self._models[name]

    def _tools(self):
        if self._photo_tools is None:
            from ..characters.headtex.detect import Landmarker, Segmenter

            self._photo_tools = (Landmarker(download=False), Segmenter(download=False))
        return self._photo_tools

    def photos(self, identity: str, shots: str) -> list:
        """Set A: renders of ``identity``'s avatar standing in for photos of the person (see the module docs)."""
        from ..characters.headshape import evaluate as ev
        from ..characters.headtex.landmarks import head_landmarks
        from ..scene.renderer import Renderer

        landmarker, segmenter = self._tools()
        model = self.model(identity)
        studio = Renderer(1024, 768)
        try:
            person = ev.Person.of(model, head_landmarks(model, landmarker)).photograph(ev.views(shots), studio,
                                                                                         landmarker, segmenter)
        finally:
            studio.release()
        return list(person.images)

    def person(self, identity: str, body: str, face: str, head: str, shots: str) -> tuple:
        """``(model, textures)``: the avatar to draw (fitted or not) and the textures it wears (``{}``: its own)."""
        if face == "own":
            return self.model(body), {}
        key = (identity, body, head, shots)
        if key in self._faces:
            self._faces.move_to_end(key)
            return self._faces[key]
        from ..characters.headtex.person import person_textures
        from ..scene.textures import Texture

        landmarker, segmenter = self._tools()
        photos = self.photos(identity, shots)
        cache = self.cache or _default_cache()
        self.progress(f"[eval] the face of {identity} on {body} ({head} head), from {len(photos)} renders")
        if head == "fitted":
            from ..characters.headshape import fitted_person

            made = fitted_person(photos, self.model(body), landmarker=landmarker, segmenter=segmenter, cache=cache)
            model, rgba = made.model, made.textures
        else:
            model = self.model(body)
            rgba = person_textures(photos, model, landmarker=landmarker, segmenter=segmenter, cache=cache).textures
        textures = {name: value if isinstance(value, Texture) else Texture(value, name=f"{name} {identity}")
                    for name, value in rgba.items()}
        self._faces[key] = (model, textures)
        if len(self._faces) > FACES_KEPT:
            self._faces.popitem(last=False)
            _collect()
        return model, textures


class StillRenderer(FaceMaker):
    """Draws the evaluation's rendered stills (see the module docs), with its people (:class:`FaceMaker`). Not thread
    safe; one OpenGL context per renderer."""

    def __init__(self, library=None, cache=None, progress=print):
        from ..scene.renderer import Renderer
        from ..scenes import fair
        from ..sensor.geometry import Sensor

        super().__init__(library, cache, progress)
        self.scene = fair.build_scene(people=[])  # the hall, its light and haze, nobody in it
        s = self.scene
        self._fair_light = (tuple(s.light_direction), float(s.ambient), tuple(s.light_color), float(s.hemisphere))
        self.renderer = Renderer(*VIEW)
        self.renderer.detail_wait = True  # (the 2048 heads read before the frame: deterministic)
        self.sensor = Sensor.scaled(6, VIEW)

    def close(self) -> None:
        super().close()
        self.renderer.release()

    # -- a still ---------------------------------------------------------------------------------------------------

    def _light(self, name: str) -> None:
        direction, ambient, colour, hemisphere = self._fair_light if name == "fair" else LIGHTS[name]
        s = self.scene
        s.light_direction, s.ambient, s.light_color, s.hemisphere = direction, ambient, colour, hemisphere

    def still(self, item) -> dict:
        """The pixels of a rendered still ``item`` (an :class:`~.evaluate.Item`) and where its face is to be looked
        for: ``{image, where, region, near, truth}`` (``truth``: the head's measured ``yaw``, ``pitch`` and ``ied``,
        the face's eye distance turned to the camera, in the image's pixels)."""
        from ..characters.character import Character
        from ..characters.faces import apply_face, face_rig
        from ..characters.model import CharacterState
        from ..sensor.requests import WindowRequest
        from ..sensor.service import eye_points, render_windows
        from .heads import HeadModel
        from .truth import frame_truth

        c = item.conditions
        model, textures = self.person(item.identity, c["body"], c["face"], c["head"], c["shots"])
        scene, renderer = self.scene, self.renderer
        self._light(c["light"])
        renderer.quality = renderer.window_quality = c["quality"]
        renderer.linear = bool(c["linear"])
        renderer.detail_from = 64.0 if int(c["detail"]) == 2048 else math.inf
        camera = scene.camera
        position = np.asarray(camera.current_position, np.float64)
        forward = np.asarray(camera.current_target, np.float64) - position
        forward[1] = 0.0
        forward /= np.linalg.norm(forward)
        across = np.array([-forward[2], 0.0, forward[0]])
        person = Character(model, name=f"subject {item.id}", shadow=False)
        for name, texture in textures.items():
            person.texture_overrides[name] = texture
        if c["expression"] != "none":
            rig = face_rig(model)
            if rig is not None:
                weights = {c["expression"]: float(c["expression_weight"])}
                person.pose_adjusters.append(lambda local, _ch, _t, w=weights, r=rig: apply_face(local, r, w))
        state = {"spot": None}

        def place(distance: float) -> None:
            spot = position + forward * distance + across * float(c["side"])
            spot[1] = 0.0
            heading = math.degrees(math.atan2(position[0] - spot[0], position[2] - spot[2])) + float(c["yaw"])
            state["spot"] = spot
            person.behaviour = (lambda _t, s=tuple(spot), h=heading: CharacterState(s, h, [], head_pitch=float(
                c["pitch"])))
            scene.update(0.0, 0.0)

        scene.add(person)
        added = []
        try:
            distance = float(c["distance"])
            if c["view"] == "frame":  # the distance that gives the face ``ied`` pixels between the eyes in the view
                f = VIEW[1] / 2.0 / math.tan(math.radians(camera.fov) / 2.0)
                distance = float(np.clip(f * EYE_M / float(c["ied"]), 0.8, 60.0))
                for _ in range(3):
                    place(distance)
                    mine = [t for t in frame_truth(scene, VIEW) if t.name == person.name]
                    if not mine or mine[0].eye_distance <= 0:
                        break
                    distance = float(np.clip(distance * mine[0].eye_distance / float(c["ied"]), 0.8, 60.0))
            place(distance)
            eyes = eye_points(person)
            for obj in occluders(c["occluder"], eyes, position):
                scene.add(obj)
                added.append(obj)
            if added:
                scene.update(0.0, 0.0)
            frame = renderer.render(scene)
            if c["view"] == "frame":
                mine = [t for t in frame_truth(scene, VIEW) if t.name == person.name]
                if not mine:
                    raise ValueError(f"{item.id}: the person is not in the view")
                truth = mine[0]
                return {"image": frame, "where": "frame", "region": HeadModel().square(truth.box),
                        "near": tuple(np.mean(np.asarray(truth.eyes, np.float64), axis=0)),
                        "truth": {"yaw": truth.yaw, "pitch": truth.pitch, "ied": truth.eye_distance,
                                  "distance": truth.distance}}
            head = tuple(eyes.mean(axis=0))
            request = WindowRequest(item.id, head=head, head_size=0.5, subject=person, eye_px=float(c["ied"]),
                                    preset=c["preset"])
            windows = render_windows(scene, renderer, [request], self.sensor, frame=int(c["seed"]), truth=True)
            if not windows:
                raise ValueError(f"{item.id}: no window (the head is not in the view)")
            window = windows[0]
            t = window.truth
            return {"image": np.ascontiguousarray(window.image), "where": "window", "region": None, "near": None,
                    "truth": {"yaw": t.yaw if t else math.nan, "pitch": t.pitch if t else math.nan,
                              "ied": t.frontal_eye_distance if t else math.nan,
                              "distance": t.distance if t else math.nan}}
        finally:
            for obj in added:
                scene.remove(obj)
            scene.remove(person)


def _collect() -> None:
    """Free what a dropped avatar or face held (its textures leave the GPU with their last reference)."""
    import gc

    gc.collect()


def _default_cache():
    import os
    from pathlib import Path

    folder = os.environ.get("BIFOLD_HEADTEX_CACHE")
    return Path(folder).expanduser() / "eval-faces" if folder else Path.home() / ".cache" / "bifold" / "eval-faces"
