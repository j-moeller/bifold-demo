"""The shape fit's evaluation with avatars as people (``docs/research/high_res_faces.md``, section 4.2).

**D1, avatars as people.** Renders of avatar A stand in for a person's photos (:class:`Shot`: the head's turn and
nod, the camera's distance and lens, the light, an expression of :mod:`~bifold_demo.characters.faces`); they are
registered to avatar B's head and B is fitted to them, never A's own head; the fitted head is compared with A's
(:mod:`.compare`) and so is B's own head, the starting point. **D2, identities from the space**: avatar B's head
changed by random identity coefficients, wearing B's texture, is the person, fitted back onto B: the fit at its best.

:class:`Person` renders and looks at a person's photos once (MediaPipe's landmarks and the segmenter's masks), so
that they can be fitted onto many heads; :func:`run` fits them onto one head and measures. The truth of a person is
its head primitive's points, its triangles and its stage-0 landmarks.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace

import numpy as np

from ..headtex.camera import PhotoCamera
from . import compare
from . import fit as shapefit
from .head import SKIN, HeadSpace, head_parts

LIGHT = (-0.4, -0.6, -0.7)  # the fair's light direction (from the camera's upper left)


@dataclass(frozen=True)
class Shot:
    """One photo of a person: the head turned ``yaw`` (to its left) and nodded ``pitch`` (down) in front of the
    camera, ``distance`` metres from the middle of the head, a vertical field of view ``fov`` (degrees) on a picture
    of ``size``; the light's direction and the ambient level; an expression of :data:`~bifold_demo.characters.faces.
    EXPRESSIONS` at ``weight``; ``view``: the name registration is given (``"left"``, ``"right"``, ``"back"``, or
    ``None`` to find it from the face)."""

    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    distance: float = 0.9
    fov: float = 30.0
    size: tuple = (1024, 768)
    light: tuple = LIGHT
    ambient: float = 0.55
    expression: str | None = None
    weight: float = 1.0
    view: str | None = None


def views(kind: str, **options) -> list[Shot]:
    """Sets of shots: ``"front"`` (one), ``"fronts"`` (three frontal, under three lights), ``"three-quarter"`` (front
    and both sides at 30 and 60 degrees, one nodded), ``"profiles"`` (that and both profiles), each with ``options``
    for every shot."""
    sets = {
        "front": [(0, 0, None)],
        "fronts": [(0, 0, None), (0, 8, None), (0, -6, None)],
        "three-quarter": [(0, 0, None), (30, 0, None), (-30, 0, None), (60, 0, None), (-60, 0, None), (0, 15, None)],
        "profiles": [(0, 0, None), (30, 0, None), (-30, 0, None), (60, 0, None), (-60, 0, None), (0, 15, None),
                     (88, 0, "right"), (-88, 0, "left")],
    }
    lights = [LIGHT, (0.5, -0.5, -0.7), (-0.2, -0.9, -0.3)]
    out = []
    for k, (yaw, pitch, view) in enumerate(sets[kind]):
        shot = Shot(yaw=yaw, pitch=pitch, view=view, **options)
        if kind == "fronts" and "light" not in options:
            shot = replace(shot, light=lights[k % len(lights)])
        out.append(shot)
    return out


@dataclass
class Person:
    """A person for the evaluation: a model (an avatar, or one changed by a shape), the truth of its head, and its
    photos once rendered and looked at."""

    model: object
    points: np.ndarray  # (N, 3) the head primitive's points (bind pose)
    triangles: np.ndarray  # (T, 3)
    marks: np.ndarray  # (478, 3) its stage-0 landmarks
    skin: np.ndarray  # (N,) bool
    shots: list = field(default_factory=list)
    images: list = field(default_factory=list)
    cameras: list = field(default_factory=list)
    faces: list = field(default_factory=list)
    masks: list = field(default_factory=list)

    @classmethod
    def of(cls, model, head) -> Person:
        """The avatar ``model`` with its stage-0 landmarks ``head`` as a person."""
        prim = model.primitives[head.primitive]
        points = np.asarray(prim.positions, np.float64)
        triangles = np.asarray(prim.indices, np.int64).reshape(-1, 3)
        marks = np.einsum("lc,lcd->ld", head.barycentric, points[triangles[head.triangle]])
        return cls(model, points, triangles, marks, head_parts(model, head.primitive) == SKIN)

    def photograph(self, shots, renderer, landmarker, segmenter=None) -> Person:
        """Render the person in ``shots`` and look at the renders (landmarks, masks); returns ``self``."""
        from ...scene import Scene
        from ..character import Character
        from ..faces import apply_face, face_rig
        from ..headtex.landmarks import face_frame

        frame = face_frame(self.model)
        rig = face_rig(self.model)
        for shot in shots:
            scene = Scene(background=(170, 180, 190), ambient=shot.ambient, light_direction=shot.light)
            person = Character(self.model, name="person", shadow=False)
            if shot.expression and rig is not None:
                face = {shot.expression: shot.weight}
                person.pose_adjusters.append(lambda local, character, t, face=face: apply_face(local, rig, face))
            scene.add(person)
            scene.update(0.0, 0.0)
            camera = PhotoCamera.around(frame.pivot, shot.distance, shot.yaw, shot.pitch, shot.roll, fov=shot.fov,
                                        size=shot.size)
            scene.camera = camera.scene_camera()
            if (renderer.width, renderer.height) != tuple(shot.size):
                raise ValueError(f"the renderer is {renderer.width}x{renderer.height}, the shot {shot.size}")
            image = renderer.render(scene)
            self.shots.append(shot)
            self.images.append(image)
            self.cameras.append(camera)
            self.faces.append(landmarker.detect(image))
            self.masks.append(segmenter.segment(image) if segmenter is not None else None)
        return self


def register(person: Person, model, head, picks=None, focal_known: bool = False, segmenter=None):
    """The person's photos (``picks``: their indices, default all) registered to ``model``'s head (``head``: its
    stage-0 landmarks), with the true focal lengths when ``focal_known``."""
    from ..headtex.register import register_photos

    picks = list(range(len(person.images))) if picks is None else list(picks)
    focals = [person.cameras[k].focal for k in picks] if focal_known else None
    return register_photos([person.images[k] for k in picks], head, model,
                           views=[person.shots[k].view for k in picks], focals=focals,
                           faces=[person.faces[k] for k in picks],
                           masks=[person.masks[k] for k in picks] if person.masks[picks[0]] is not None else None,
                           segmenter=segmenter)


def measure(person: Person, points, triangles, marks, skin) -> dict:
    """:func:`~.compare.compare` of a head (``points`` ...) against the person's."""
    return compare.compare(points, triangles, marks, person.points, person.triangles, person.marks, skin=skin,
                           target_skin=person.skin)


def run(person: Person, model, head, hs: HeadSpace, space, options=None, picks=None, focal_known: bool = False,
        registered=None) -> dict:
    """Fit ``model``'s head (``head``, ``hs``, ``space``) to the person's photos and measure it and the unfitted head
    against the person: ``{"before": ..., "after": ..., "fit": ShapeFit, "seconds": ...}`` (``after`` is ``None``
    when the fit was not usable)."""
    prim = model.primitives[hs.primitive]
    points = np.asarray(prim.positions, np.float64)
    triangles = np.asarray(prim.indices, np.int64).reshape(-1, 3)
    skin = hs.parts == SKIN

    def marks(p):
        return np.einsum("lc,lcd->ld", head.barycentric, p[triangles[head.triangle]])

    views_ = registered if registered is not None else register(person, model, head, picks, focal_known)
    photos = shapefit.photos_from_views(views_)
    start = time.perf_counter()
    result = shapefit.fit_person(photos, model, head, hs, space, options)
    seconds = time.perf_counter() - start
    before = measure(person, points, triangles, marks(points), skin)
    after = None
    if result.offsets is not None:
        fitted = points + result.offsets
        after = measure(person, fitted, triangles, marks(fitted), skin)
    return {"before": before, "after": after, "fit": result, "seconds": seconds, "photos": len(photos)}


def summary(results, kind: str = "similarity") -> dict:
    """Medians over ``results`` (of :func:`run`) of the landmark and surface medians and the regions, before and
    after (``nan`` where a fit was not usable)."""
    out = {}
    for when in ("before", "after"):
        rows = [r[when][kind] if r[when] is not None else None for r in results]
        keys = ["median", "mean", *compare.REGIONS]
        out[when] = {f"landmarks {k}": float(np.nanmedian([row["landmarks"][k] if row else math.nan for row in rows]))
                     for k in keys}
        out[when].update({f"surface {k}": float(np.nanmedian([row["surface"][k] if row else math.nan
                                                               for row in rows])) for k in ("median", "mean")})
    out["usable"] = sum(r["after"] is not None and r["fit"].usable for r in results)
    out["count"] = len(results)
    return out
