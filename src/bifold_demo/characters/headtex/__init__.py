"""Head textures from photos: a person's photos turned into a texture for an avatar's own head mesh.

Component 1 of the character work (the approach: ``docs/research/face_head_texture.md``, section 3). Casual photos of
a person's head from several angles (front, three-quarter, the sides, maybe the back; phone photos, unknown camera,
mixed light) become a texture for the avatar's *existing* head mesh (no new geometry) that blends into the whole
body. Eight stages:

0. **Landmarks on the head mesh** (:mod:`.landmarks`), once per head model: MediaPipe's 478 face landmarks found on
   renders of the avatar and lifted onto its mesh, so every photo's landmarks have 3D partners; cached on disk.
1. **Register each photo** (:mod:`.register`): the photo's camera relative to the head in its bind pose (PnP on the
   landmarks, or the head's outline for photos without a face), a thin-plate warp that puts the person's eyes, brows,
   nose and mouth on the avatar's, segmentation masks, a quality score and warnings: a :class:`~.register.View`.
2. Normalise the photos (optional, generative): even light, neutral expression, no glasses.
3. Project and fuse the views in the head texture's UV layout (:mod:`..uvbake` knows each texel's 3D point).
4. Take the photos' lighting out and set the skin tone.
5. Complete what no photo shows (generative, with a deterministic fallback).
6. Match the body's skin, the neck seam and the hair colour.
7. Check the identity (face recognition on renders) and save.

This package has stages 0 and 1 so far; the later stages build on their output (``View.photo_pixels`` maps any point
of the head to the photo pixel showing it).

The pieces: :mod:`.camera` (a pinhole camera in character space; the head's yaw, pitch and roll), :mod:`.detect`
(MediaPipe's Face Landmarker and selfie segmenter, an optional dependency), :mod:`.canonical` (which landmark is which;
MediaPipe's canonical face), :mod:`.raycast` (rays and nearest points on a mesh), :mod:`.warp` (thin-plate splines)
and :mod:`.cli` (``bifold-faces landmarks`` and ``bifold-faces register``). Photos and textures made from them are
personal data: keep them in git-ignored folders.
"""

from .camera import PhotoCamera
from .landmarks import HeadLandmarks, head_landmarks
from .register import View, draw_registration, register_photo, register_photos

__all__ = ["HeadLandmarks", "PhotoCamera", "View", "draw_registration", "head_landmarks", "register_photo",
           "register_photos"]
