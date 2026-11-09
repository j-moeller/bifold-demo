"""The head's shape fitted to a person's photos: the avatar's head given the person's proportions, profile and outline
as far as the photos tell, on the avatar's own mesh and rig (step S3 of ``docs/research/face_recognition_focus.md``;
the approach: ``docs/research/high_res_faces.md``, section 3.2; the measurements:
``docs/research/head_shape_results.md``).

Until now a known person's face was a texture on the avatar's own head: the eyes, brows, nose and mouth warped onto
the avatar's, the head keeping the avatar's proportions, profile and outline. The pieces:

* :mod:`.space`: **the identity space**, ICT-FaceKit Light (MIT): a neutral head with 100 identity modes and 53
  expressions, fetched once (``bifold-characters identity-space``), pinned by SHA-256 and kept as one float32 file,
  with MediaPipe's landmarks on it shipped with the code.
* :mod:`.head` (S0): **the space on each avatar's head**, once per head model and cached: its parts (skin, eyeballs,
  the mouth's inside, the neck's edge, the hair and other parts that move with it), ICT's head laid onto it (a
  similarity from the landmarks, then a non-rigid registration), the modes and expressions as offsets of its vertices,
  the avatar's own coefficients (the rest is the artist's detail, which a fit keeps) and the regressors that move the
  eyeballs, the teeth and tongue, the face bones and the hair with the skin.
* :mod:`.fit` (S1-S4): **the fit**, all of a person's photos at once: the coefficients, per photo the camera and the
  expression (ICT's ARKit-named shapes, started from MediaPipe's blendshape scores, only where MediaPipe sees one),
  the focal lengths, the scale from the irises; MediaPipe's landmarks weighted by their reliability, by how squarely
  they are seen and by how faithfully they follow a head's shape (measured), the jaw line as the head's outline in
  each photo, the face's outline against the segmenter's masks (:mod:`.outline`), priors; a small free-form
  correction (:func:`~.fit.free_form`), which without the identity space is the whole fit (the fallback).
* :mod:`.rig` (S5, S7): **the fitted head on the rig** (a per-character copy of the model sharing all but the head,
  the parts that move with it and the skeleton; eyeballs rigid, teeth with the jaw and lips, face bones and inverse
  binds re-solved, normals turned, tangents made again), its **checks** in the 12 expressions of the face rig, and
  the ``shape.npz`` it is kept in (about 25 KB).
* :mod:`.fitted` (S6): **end to end**: the shape, then the head-texture stages on the fitted head
  (:func:`~.fitted.fitted_person`); failures keep the avatar's own head, with the reason.
* :mod:`.compare` and :mod:`.evaluate`: the geometry measures and the evaluation with avatars as people.

A fitted shape of a real person is biometric data like the photos: it is kept only in git-ignored folders (a created
character's, ``faces/.headtex/``) and deleted with the person or character.
"""

from .fit import FitOptions, ShapeFit, fit_person, photos_from_views
from .fitted import FittedPerson, ShapeResult, fit_shape, fitted_person
from .head import HeadSpace, head_space
from .rig import RigCheck, Shape, apply_shape, make_shape, rig_checks
from .space import IdentitySpace

__all__ = ["FitOptions", "FittedPerson", "HeadSpace", "IdentitySpace", "RigCheck", "Shape", "ShapeFit",
           "ShapeResult", "apply_shape", "fit_person", "fit_shape", "fitted_person", "head_space", "make_shape",
           "photos_from_views", "rig_checks"]
