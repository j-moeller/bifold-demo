"""Head textures from photos: a person's photos turned into a texture for an avatar's own head mesh.

Component 1 of the character work (the approach: ``docs/research/face_head_texture.md``, section 3). Casual photos of
a person's head from several angles (front, three-quarter, the sides, maybe the back; phone photos, unknown camera,
mixed light) become a texture for the avatar's *existing* head mesh (no new geometry) that blends into the whole
body. Eight stages:

0. **Landmarks on the head mesh** (:mod:`.landmarks`), once per head model: MediaPipe's 478 face landmarks found on
   renders of the avatar and lifted onto its mesh, so every photo's landmarks have 3D partners; cached on disk. With
   them, which parts of the head the segmenter calls face skin and hair (:func:`~.register.head_regions`, for photos
   placed by the head's outline).
1. **Register each photo** (:mod:`.register`): the photo's camera relative to the head in its bind pose (PnP on the
   landmarks, or the head's outline for photos without a face), a thin-plate warp that puts the person's eyes, brows,
   nose and mouth on the avatar's, segmentation masks, a quality score and warnings: a :class:`~.register.View`.
2. Normalise the photos (optional, generative): even light, neutral expression, no glasses.
3. **Project and fuse** (:mod:`.fuse`): every texel of the head texture (:mod:`..uvbake` knows its 3D point) is
   depth-tested against the head as each photo's camera sees it and read from the photo through the face warp,
   weighted by the viewing angle, the segmenter's classes, the distance from the masks' edges, the view's quality
   and the photo pixels per texel; the photos' gains are evened out on their overlaps and the photos are blended in
   five bands, island by island. A confidence per texel and a keep / refine / generate trimap come with it.
4. **Light and skin tone** (:mod:`.light`): each photo's light (spherical harmonics of the normal, fitted to its face
   skin) divided out, the avatar's own painted soft light put back, the person's skin tone (a trimmed Lab median of
   the cheeks and forehead) for stage 6, and the avatar's own texture tone-matched to it where no photo reaches.
5. **Complete what no photo shows** (:mod:`.completion`): texels the photos show poorly take their mirror image across
   the face's plane where the photos show that well, blended in by the multi-band blend; the rest keeps the avatar's
   own texture, tone-matched. An optional generative layer (FLUX.2 [klein] through ``diffusers``, or any pipeline
   passed in) fills what is left, second-class: tone-matched, feathered, at most 40 %.
6. **Match the body** (:mod:`.body`): the body texture's skin (hands, neck, bare arms and legs) given the face's
   skin tone by a gain per channel in linear light (by the neck: the head's neck colour), the neck seam levelled by
   least-squares offsets round the neck, the hair cards and unseen scalp recoloured towards the photos' hair.
7. **Check and save** (:mod:`.person`): delta E across the neck seam and from the face to the hands, before and after
   stage 6; the identity (SFace on renders of the textured avatar against the photos, relative to the avatar's own
   head; a generated layer that lowers it is dropped); the textures of head, body and hair as a
   :class:`~..created.CreatedCharacter`'s.

:func:`~.texture.head_texture` runs stages 0 to 4 (a :class:`~.texture.HeadTexture`); :func:`~.person.person_textures`
runs them all (a :class:`~.person.PersonTextures`: the textures by primitive name, cached by the photos when asked,
:meth:`~.person.PersonTextures.wear` and :meth:`~.person.PersonTextures.created`). The known faces in the fair
(``recognition.cast``) and the character creator's faces (``creator.face_texture``) use them.

The pieces: :mod:`.camera` (a pinhole camera in character space; the head's yaw, pitch and roll), :mod:`.detect`
(MediaPipe's Face Landmarker and selfie segmenter, an optional dependency), :mod:`.canonical` (which landmark is which;
MediaPipe's canonical face), :mod:`.raycast` (rays and nearest points on a mesh), :mod:`.raster` (a z-buffer of a
mesh in numpy), :mod:`.warp` (thin-plate splines), :mod:`.texture` (stages 0 to 4 end to end, previews, wearing the
result), :mod:`.person` (stages 5 to 7 end to end) and :mod:`.cli` (``bifold-faces landmarks``, ``register`` and
``headtex``). Photos and textures made from them are personal data: keep them in git-ignored folders.
"""

from .body import BodyMatch, match_body
from .camera import PhotoCamera
from .completion import Completion, FluxInpainter, Inpainter, complete, mirror_map
from .fuse import HeadAtlas, Projection, head_atlas
from .landmarks import HeadLandmarks, head_landmarks
from .light import Shading, Tone
from .person import PersonCache, PersonTextures, finish, identity, person_textures
from .register import View, draw_registration, head_regions, register_photo, register_photos
from .texture import (
    HeadTexture,
    fuse_views,
    head_texture,
    head_texture_async,
    neck_seam,
    preview,
    skin_gap,
    take_off,
    wear,
)

__all__ = ["BodyMatch", "Completion", "FluxInpainter", "HeadAtlas", "HeadLandmarks", "HeadTexture", "Inpainter",
           "PersonCache", "PersonTextures", "PhotoCamera", "Projection", "Shading", "Tone", "View", "complete",
           "draw_registration", "finish", "fuse_views", "head_atlas", "head_landmarks", "head_regions",
           "head_texture", "head_texture_async", "identity", "match_body", "mirror_map", "neck_seam",
           "person_textures", "preview", "register_photo", "register_photos", "skin_gap", "take_off", "wear"]
