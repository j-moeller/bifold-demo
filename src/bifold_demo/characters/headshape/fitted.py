"""A person's head, end to end: the shape fitted to their photos, put on the avatar's rig, and textured on it (S6).

::

    from bifold_demo.characters.headshape import fitted_person

    person = fitted_person(["front.jpg", "left.jpg", "right.jpg"], model)
    print(person.summary())       # the fit, the checks, the textures' stages
    person.wear(character)        # a scene Character of that model: the fitted head and the textures

:func:`fit_shape` registers the photos to the avatar's own head (stages 0 and 1 of the head textures), fits the shape
(:func:`~.fit.fit_person`; without the identity space, the free-form fallback), makes it a :class:`~.rig.Shape` and
puts it on the rig (:func:`~.rig.apply_shape`) with its checks (:func:`~.rig.rig_checks`). A shape that fails them is
kept smaller near the part that fails (an eye, the mouth, a fold: :data:`LOCAL_SHRINK`, fading over
:data:`LOCAL_REACH`) and checked again, step by step, down to the avatar's own shape there; what no part explains (or
does not mend so) is tried with the whole shape at three quarters and at half its size, then dropped, and the
avatar's own head is kept, with the reason. What was kept smaller is stored with the shape (:attr:`~.rig.Shape.
shrink`).

:func:`fitted_person` then runs the head-texture stages on the fitted model: the stage-0 landmarks carry over (the same
triangles and barycentric coordinates on the moved head), so do the segmenter's labels of the head's outline (the
same triangles), the registration works on a head whose landmarks now match the person's (the face warp has less to
do; it finds the focal lengths itself, as on the avatar's head: only EXIF or given ones are held, not the fit's, which
are a means of the fit and often far from the truth), and the UV bake, the atlas and the inner parts are made again
for the new geometry. With ``check`` the identity check reports the renders of the fitted head and of
the avatar's own head with its own textures from the same photos: the shape's measured value. The result is kept by
the photos' contents like :func:`~bifold_demo.characters.headtex.person.person_textures`' (``shape.npz`` beside the
textures, both written at once), with the fit's code version (:data:`~.fit.FIT_VERSION`), the head space's
(:data:`~.head.VERSION`), the fit's options and the identity space in the key (:func:`cache_key`): a changed fit makes
the head again. A fitted shape is biometric data like the photos: keep it in git-ignored folders only.
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass, field

import numpy as np

from . import rig
from .fit import FIT_VERSION, FitOptions, ShapeFit, fit_person, photos_from_views

SHRINK = (1.0, 0.75, 0.5)  # a shape whose failure no part explains is tried at these sizes as a whole
LOCAL_SHRINK = (0.75, 0.5, 0.25, 0.0)  # a part of the head that fails the rig's checks: the offsets kept near it
LOCAL_REACH = {"eye": 0.015, "mouth": 0.02, "fold": 0.01}  # metres: how far round it (a Gaussian's spread)
LOCAL = ("right eye", "left eye", "mouth", "fold")


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
    scale: float = 1.0  # the share of the fitted offsets kept on the whole head (SHRINK)
    timings: dict = field(default_factory=dict)
    shrink: dict = field(default_factory=dict)  # part of the head (or "all") -> the share of the offsets kept there

    def summary(self) -> str:
        if self.shape is None:
            return f"the head keeps {getattr(self.model, 'name', 'the avatar')}'s shape: {self.reason}"
        lines = [self.fit.summary() if self.fit is not None else "head shape"]
        if self.shrink:
            lines.append("  the full shape failed the rig's checks: kept at " + ", ".join(
                f"{100 * v:.0f} % {'as a whole' if k == 'all' else 'near the ' + k}" for k, v in self.shrink.items()))
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
    shape, fitted, checks, shrink = put_on(model, hs, fit.offsets, fit)
    timings["rig and checks"] = time.perf_counter() - start
    if shape is None:
        return ShapeResult(None, model, fit, checks, registered, reason="the fitted head fails the rig's checks: "
                           + "; ".join(checks.problems[:3]), timings=timings)
    return ShapeResult(shape, fitted, fit, checks, registered, scale=shrink.get("all", 1.0), shrink=shrink,
                       timings=timings)


def put_on(model, hs, offsets, fit=None) -> tuple:
    """Skin ``offsets`` (a fit's) on ``model``'s rig with the checks (see the module docs): ``(shape, the model
    wearing it, the checks, the shrink)``, or ``(None, model, the last checks, {})`` when no smaller shape passes
    either. ``hs``: the head space; ``fit``: the :class:`~.fit.ShapeFit` the shape keeps."""
    base = np.asarray(offsets, np.float64)
    levels, near = {}, {}
    for _ in range(1 + len(LOCAL) * len(LOCAL_SHRINK)):  # kept smaller near the parts that fail, step by step
        keep = np.ones(len(base))
        for part, k in levels.items():
            keep *= 1.0 - (1.0 - LOCAL_SHRINK[k]) * near[part]
        shape = rig.make_shape(model, hs, base * keep[:, None], fit)
        fitted = rig.apply_shape(model, shape)
        checks = rig.rig_checks(model, fitted, hs)
        if checks.ok:
            shape.shrink = {part: LOCAL_SHRINK[k] for part, k in levels.items()}
            return shape, fitted, checks, dict(shape.shrink)
        failing = set(checks.where)
        if failing - set(LOCAL) or any(levels.get(part) == len(LOCAL_SHRINK) - 1 for part in failing):
            break  # what no part explains, or what the avatar's own shape there does not mend: the whole head
        for part in failing:
            levels[part] = levels.get(part, -1) + 1
            near[part] = _near(model, hs, part, checks)
    for scale in SHRINK[1:]:
        shape = rig.make_shape(model, hs, base * scale, fit)
        fitted = rig.apply_shape(model, shape)
        checks = rig.rig_checks(model, fitted, hs)
        if checks.ok:
            shape.shrink = {"all": scale}
            return shape, fitted, checks, {"all": scale}
    return None, model, checks, {}


def _near(model, hs, part: str, checks) -> np.ndarray:
    """``(N,)`` 0-1: how near each vertex of the head is to ``part`` (an eye, the mouth, the folds that
    ``checks`` found), with a Gaussian fall-off (:data:`LOCAL_REACH`)."""
    points = np.asarray(model.primitives[hs.primitive].positions, np.float64)
    if part in ("right eye", "left eye"):
        d = np.linalg.norm(points - hs.eye_centre[0 if part == "right eye" else 1], axis=1)
        sigma = LOCAL_REACH["eye"]
    elif part == "mouth":
        skeleton = model.skeleton
        corners = [skeleton.find(n) for n in ("Bip01 RMouthCorner", "Bip01 LMouthCorner")]
        if all(c is not None for c in corners):
            centre = np.mean([skeleton.rest_global[c, :3, 3] for c in corners], axis=0)
        else:
            centre = points[hs.inner > 0].mean(axis=0) if (hs.inner > 0).any() else points.mean(axis=0)
        d = np.linalg.norm(points - centre, axis=1)
        sigma = LOCAL_REACH["mouth"]
    else:
        folds = np.asarray(checks.folds, np.int64)
        if not len(folds):
            return np.ones(len(points))
        d = np.min([np.linalg.norm(points - points[v], axis=1) for v in folds[:200]], axis=0)
        sigma = LOCAL_REACH["fold"]
    return np.exp(-0.5 * (d / sigma) ** 2)


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

    photos = list(photos)
    store = PersonCache(cache) if cache is not None and not isinstance(cache, PersonCache) else cache
    key = None
    if store is not None:
        key = cache_key(store, photos, model, views=views, inpainter=inpainter, size=size, space=space,
                        options=options)
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
        if result.shape is not None:
            fitted = result.model
            fitted_head = carried_landmarks(head, fitted)
            if labels is not None:
                Outline.of(fitted, head.primitive).labels = labels
        texture = head_texture(photos, fitted, views=views, focals=focals, landmarker=landmarker,
                               segmenter=segmenter, head=fitted_head, size=size,
                               cache_dir=False if fitted is not model else cache_dir)  # (a fitted head's atlas in
        # memory only: the person's result is kept whole; one per person in the shared cache would only pile up)
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
    if store is not None:  # (the shape inside the entry before it appears: never textures without their head)
        store.put(key, person, files={rig.FILE: result.shape.save} if result.shape is not None else None)
    return out


def cache_key(store, photos, model, *, views=None, inpainter=None, size=None, space="load",
              options: FitOptions | None = None) -> str:
    """The key of a fitted person in ``store`` (a :class:`~bifold_demo.characters.headtex.person.PersonCache`): the
    photos' contents, the avatar and what each photo shows, as for the textures, and all that changes the shape: the
    shape file's format, the fit's and the head space's code versions, the fit's options and the identity space's
    commit."""
    from . import head as headspace
    from . import space as identity

    commit = ""
    if space is None:
        commit = "none"
    elif not (isinstance(space, str) and space == "load"):
        commit = getattr(space, "commit", "")
    elif identity.available():
        commit = identity.files()["commit"]
    return store.key(photos, model, views=views, inpainter=getattr(inpainter, "name", None), size=size,
                     shape=f"headshape-{rig.FORMAT}-{FIT_VERSION}-{headspace.VERSION}-{commit}-"
                           f"{identity.shipped_digest()}",
                     fit=dataclasses.asdict(options or FitOptions()))


__all__ = ["FittedPerson", "ShapeResult", "cache_key", "carried_landmarks", "fit_shape", "fitted_person",
           "put_on"]
