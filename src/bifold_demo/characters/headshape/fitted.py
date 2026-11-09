"""A person's head, end to end: the shape fitted to their photos, put on the avatar's rig, and textured on it (S6).

::

    from bifold_demo.characters.headshape import fitted_person

    person = fitted_person(["front.jpg", "left.jpg", "right.jpg"], model)
    print(person.summary())       # the fit, the checks, the textures' stages
    person.wear(character)        # a scene Character of that model: the fitted head and the textures

:func:`fit_shape` registers the photos to the avatar's own head (stages 0 and 1 of the head textures), fits the shape
(:func:`~.fit.fit_person`; without the identity space, the free-form fallback), makes it a :class:`~.rig.Shape` and
puts it on the rig (:func:`~.rig.apply_shape`) with its checks (:func:`~.rig.rig_checks`); a shape that fails them
is tried at three quarters and at half its size, then dropped, and the avatar's own head is kept, with the reason.

:func:`fitted_person` then runs the head-texture stages on the fitted model: the stage-0 landmarks carry over (the same
triangles and barycentric coordinates on the moved head), so do the segmenter's labels of the head's outline (the
same triangles), the registration works with the fit's focal lengths on a head whose landmarks now match the
person's (the face warp has less to do), and the UV bake, the atlas and the inner parts are made again for the new
geometry (cached like the avatar's). With ``check`` the identity check reports the renders of the fitted head and of
the avatar's own head with its own textures from the same photos: the shape's measured value. The result is kept by
the photos' contents like :func:`~bifold_demo.characters.headtex.person.person_textures`' (``shape.npz`` beside the
textures). A fitted shape is biometric data like the photos: keep it in git-ignored folders only.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from . import rig
from .fit import FitOptions, ShapeFit, fit_person, photos_from_views

SHRINK = (1.0, 0.75, 0.5)  # a shape that fails the rig's checks is tried at these sizes


@dataclass
class ShapeResult:
    """What :func:`fit_shape` made: ``shape`` (``None``: the avatar's own head is kept, ``reason`` says why), the
    fit, the checks, the model wearing the shape (or the avatar's own) and the registered views."""

    shape: rig.Shape | None
    model: object
    fit: ShapeFit | None = None
    checks: rig.RigCheck | None = None
    views: list = field(default_factory=list)
    reason: str = ""
    scale: float = 1.0  # the share of the fitted offsets kept (SHRINK)
    timings: dict = field(default_factory=dict)

    def summary(self) -> str:
        if self.shape is None:
            return f"the head keeps {getattr(self.model, 'name', 'the avatar')}'s shape: {self.reason}"
        lines = [self.fit.summary() if self.fit is not None else "head shape"]
        if self.scale < 1.0:
            lines.append(f"  the shape at {100 * self.scale:.0f} % (the full one failed the rig's checks)")
        lines.append("  " + self.shape.describe())
        if self.timings:
            lines.append("  timings: " + ", ".join(f"{k} {v:.2f} s" for k, v in self.timings.items()))
        return "\n".join(lines)


def fit_shape(photos, model, *, views=None, focals=None, landmarker=None, segmenter=None, head=None, space="load",
              options: FitOptions | None = None, cache_dir=None, registered=None) -> ShapeResult:
    """The head shape of the person in ``photos`` (paths or BGR images) on ``model`` (see the module docs).
    ``views``, ``focals``, ``landmarker``, ``segmenter``, ``head`` (stage 0's landmarks) and ``cache_dir`` as for
    :func:`~bifold_demo.characters.headtex.texture.head_texture`; ``space``: the identity space (``"load"``: the
    fetched one, else the free-form fallback; ``None``: the fallback); ``registered``: the photos registered to the
    avatar's head already. Never raises for photos it cannot use: the avatar's head is kept, with the reason."""
    from . import space as identity
    from .head import head_space

    timings = {}
    start = time.perf_counter()
    notes = []
    if isinstance(space, str) and space == "load":
        try:
            space = identity.load()
        except FileNotFoundError as exc:
            space = None
            notes.append(f"without the identity space, the free-form fallback ({exc})")
    if head is None:
        from ..headtex.landmarks import head_landmarks

        head = head_landmarks(model, landmarker, cache_dir=cache_dir)
    hs = head_space(model, head, space, cache_dir=cache_dir)
    timings["head space (once per avatar)"] = time.perf_counter() - start
    start = time.perf_counter()
    if registered is None:
        from ..headtex.register import register_photos

        registered = register_photos(list(photos), head, model, views=views, focals=focals, landmarker=landmarker,
                                     segmenter=segmenter)
    timings["registration"] = time.perf_counter() - start
    start = time.perf_counter()
    fit = fit_person(photos_from_views(registered), model, head, hs, space, options)
    timings["fit"] = time.perf_counter() - start
    fit.warnings = notes + fit.warnings
    if not fit.usable:
        return ShapeResult(None, model, fit, views=registered, reason=fit.reason, timings=timings)
    start = time.perf_counter()
    checks = None
    for scale in SHRINK:
        shape = rig.make_shape(model, hs, fit.offsets * scale, fit)
        fitted = rig.apply_shape(model, shape)
        checks = rig.rig_checks(model, fitted, hs)
        if checks.ok:
            timings["rig and checks"] = time.perf_counter() - start
            return ShapeResult(shape, fitted, fit, checks, registered, scale=scale, timings=timings)
    timings["rig and checks"] = time.perf_counter() - start
    return ShapeResult(None, model, fit, checks, registered, reason="the fitted head fails the rig's checks: "
                       + "; ".join(checks.problems[:3]), timings=timings)


def carried_landmarks(head, fitted):
    """The stage-0 landmarks ``head`` of the avatar on its ``fitted`` model: the same triangles and barycentric
    coordinates, their points on the moved head."""
    from dataclasses import replace

    prim = fitted.primitives[head.primitive]
    tri = np.asarray(prim.indices, np.int64).reshape(-1, 3)[head.triangle]
    points = np.einsum("lc,lcd->ld", head.barycentric, np.asarray(prim.positions, np.float64)[tri])
    return replace(head, points=points, key=f"{head.key}-fitted")


@dataclass
class FittedPerson:
    """A person's fitted head and textures (see the module docs): ``person`` the textures
    (a :class:`~bifold_demo.characters.headtex.person.PersonTextures`), ``shape`` (``None``: the avatar's own head),
    ``model`` the model wearing it, ``result`` the :class:`ShapeResult` (``None`` when read from a cache)."""

    person: object
    shape: rig.Shape | None
    model: object
    result: ShapeResult | None = None
    cached: bool = False

    @property
    def textures(self) -> dict:
        return self.person.textures

    def summary(self) -> str:
        lines = [self.result.summary() if self.result is not None else
                 ("head shape: " + self.shape.describe() if self.shape is not None else "the avatar's own head")]
        return "\n".join(lines + [self.person.summary()])

    def wear(self, character) -> dict:
        """Put the fitted head (:meth:`~bifold_demo.characters.character.Character.use_model`) and the textures on
        ``character`` (of the avatar); returns the textures worn (see
        :func:`~bifold_demo.characters.headtex.person.wear_textures`)."""
        if self.shape is not None:
            character.use_model(self.model)
        return self.person.wear(character)


def _unfitted_identity(photos, model, cache, engine, renderer, **options) -> dict:
    from ..headtex.person import identity, person_textures

    own = person_textures(photos, model, cache=cache, **options)
    return identity(model, own.textures, photos, engine=engine, renderer=renderer)


def fitted_person(photos, model, *, views=None, focals=None, cache=None, check: bool = False, engine=None,
                  renderer=None, inpainter=None, landmarker=None, segmenter=None, size=None, cache_dir=None,
                  space="load", options: FitOptions | None = None) -> FittedPerson:
    """The fitted head and the textures of the person in ``photos`` for ``model`` (see the module docs); the
    arguments as for :func:`fit_shape` and :func:`~bifold_demo.characters.headtex.person.person_textures`
    (``cache``: a folder that keeps the result by the photos' contents). Raises ``ImportError`` without MediaPipe and
    ``ValueError`` when no photo can be used for the textures."""
    from ..headtex.person import PersonCache, finish
    from ..headtex.register import Outline, head_regions
    from ..headtex.texture import head_texture
    from . import space as identity

    photos = list(photos)
    store = PersonCache(cache) if cache is not None and not isinstance(cache, PersonCache) else cache
    key = None
    if store is not None:
        commit = ""
        if not (isinstance(space, str) and space == "load"):
            commit = getattr(space, "commit", "")
        elif identity.available():
            commit = identity.files()["commit"]
        key = store.key(photos, model, views=views, inpainter=getattr(inpainter, "name", None), size=size,
                        shape=f"headshape-{rig.FORMAT}-{commit}")
        found = store.get(key)
        if found is not None:
            shape_file = store.folder / key / rig.FILE
            shape = None
            if shape_file.is_file():
                try:
                    shape = rig.Shape.load(shape_file)
                    fitted = rig.apply_shape(model, shape)
                except (OSError, ValueError):
                    shape, fitted = None, model
            else:
                fitted = model
            if shape is not None or not shape_file.exists():
                return FittedPerson(found, shape, fitted, cached=True)
    own_tasks = []
    try:
        if landmarker is None:
            from ..headtex.detect import Landmarker

            landmarker = Landmarker()
            own_tasks.append(landmarker)
        if segmenter is None:
            from ..headtex.detect import Segmenter

            segmenter = Segmenter()
            own_tasks.append(segmenter)
        from ..headtex.landmarks import head_landmarks

        head = head_landmarks(model, landmarker, cache_dir=cache_dir)
        labels = head_regions(model, head.primitive, segmenter, cache_dir=cache_dir)
        result = fit_shape(photos, model, views=views, focals=focals, landmarker=landmarker, segmenter=segmenter,
                           head=head, space=space, options=options, cache_dir=cache_dir)
        fitted, fitted_head = model, head
        fit_focals = focals
        if result.shape is not None:
            fitted = result.model
            fitted_head = carried_landmarks(head, fitted)
            if labels is not None:
                Outline.of(fitted, head.primitive).labels = labels
            if focals is None and result.fit is not None:
                fit_focals = _per_photo(result, photos)
        texture = head_texture(photos, fitted, views=views, focals=fit_focals, landmarker=landmarker,
                               segmenter=segmenter, head=fitted_head, size=size, cache_dir=cache_dir)
    finally:
        for task in own_tasks:
            task.close()
    texture.warnings[:0] = [f"head shape: {w}" for w in (result.fit.warnings if result.fit else [])]
    if result.shape is None:
        texture.warnings.insert(0, result.summary())
    person = finish(texture, fitted, inpainter=inpainter, check=check, engine=engine, renderer=renderer)
    if check and result.shape is not None:
        try:
            person.checks["identity of the unfitted head"] = _unfitted_identity(
                photos, model, store, engine, renderer, views=views, focals=focals, inpainter=inpainter,
                landmarker=landmarker, segmenter=segmenter)
        except (ValueError, ImportError) as exc:
            person.warnings.append(f"no identity check of the unfitted head: {exc}")
    if result.shape is not None:
        person.checks["head shape"] = result.shape.describe()
    out = FittedPerson(person, result.shape, fitted, result)
    if store is not None:
        store.put(key, person)
        if result.shape is not None and (store.folder / key).is_dir():
            try:
                result.shape.save(store.folder / key / rig.FILE)
            except OSError:
                pass
    return out


def _per_photo(result: ShapeResult, photos) -> list:
    """The fit's focal length for each of ``photos`` (``None`` for a photo the fit did not use)."""
    from .fit import fitted_view

    focal = iter(result.fit.focal)
    out = [next(focal, None) if fitted_view(view) else None for view in result.views]
    return out if len(out) == len(photos) else [None] * len(photos)


__all__ = ["FittedPerson", "ShapeResult", "carried_landmarks", "fit_shape", "fitted_person"]
