"""Face recognition: who the people in the picture are.

Faces are found by YuNet and turned into a 128-number embedding (a face's "fingerprint") by SFace, both small
OpenCV Zoo models that OpenCV runs itself (MIT / Apache 2.0, see :mod:`.models`). The embeddings of a face seen in
different photos lie close together: their cosine similarity is high (about 0.6-1.0 for the same person, below
about 0.3 for others). A :class:`~.gallery.Gallery` keeps the embeddings of the photos of known people (a folder per
person) and names a face by its nearest match above :data:`~.gallery.THRESHOLD`.
"""

from .engine import Face, FaceEngine
from .gallery import THRESHOLD, Gallery, Match

__all__ = ["THRESHOLD", "Face", "FaceEngine", "Gallery", "Match"]
