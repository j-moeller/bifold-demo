"""MediaPipe's 478 face landmarks: which is which, and the canonical face they come from.

The Face Landmarker puts the same 478 points on every face: 468 of a fixed face mesh (MediaPipe's *canonical face
model*, an average adult face) and 10 on the irises. The groups below name the ones the pipeline needs (the person's
left and right, as MediaPipe counts them: ``LEFT_EYE`` is the eye on the right of a frontal photo).

The canonical face model ships inside the Face Landmarker's ``.task`` bundle (a zip file) as
``geometry_pipeline_metadata_landmarks.binarypb``, a small protocol-buffer message: :func:`read_canonical_face` reads
its 468 vertices (centimetres, in the same orientation as an avatar's character space: +X the face's left, +Y up, +Z
out of the face), its texture coordinates and its 898 triangles, without needing the protobuf package.
"""

from __future__ import annotations

import struct
import zipfile
from dataclasses import dataclass

import numpy as np

COUNT = 478  # landmarks per face (468 mesh points + 10 iris points)
MESH_COUNT = 468

NOSE_TIP = 1
FOREHEAD = 10  # top of the face oval, on the midline
CHIN = 152
RIGHT_EYE_OUTER, RIGHT_EYE_INNER, LEFT_EYE_INNER, LEFT_EYE_OUTER = 33, 133, 362, 263
MOUTH_RIGHT, MOUTH_LEFT = 61, 291
UPPER_LIP, LOWER_LIP = 13, 14  # the middle of the lips' inner edges
RIGHT_CHEEK_SIDE, LEFT_CHEEK_SIDE = 234, 454  # the face oval at the ears' height
RIGHT_IRIS = tuple(range(468, 473))  # centre first, then four points on its rim
LEFT_IRIS = tuple(range(473, 478))

FACE_OVAL = (10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176,
             149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109)  # in order around the face
RIGHT_EYE = (33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246)
LEFT_EYE = (263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466)
RIGHT_BROW = (46, 53, 52, 65, 55, 70, 63, 105, 66, 107)
LEFT_BROW = (276, 283, 282, 295, 285, 300, 293, 334, 296, 336)
LIPS_OUTER = (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185)
LIPS_INNER = (78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191)
NOSE = (168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 98, 97, 326, 327, 294, 278, 344, 440, 275, 45, 220, 115, 48, 64)
FEATURES = tuple(sorted(set(RIGHT_EYE + LEFT_EYE + RIGHT_BROW + LEFT_BROW + LIPS_OUTER + LIPS_INNER + NOSE)))

METADATA = "geometry_pipeline_metadata_landmarks.binarypb"


@dataclass
class CanonicalFace:
    """MediaPipe's canonical face model."""

    points: np.ndarray  # (468, 3) centimetres: +X the face's left, +Y up, +Z out of the face
    uvs: np.ndarray  # (468, 2)
    triangles: np.ndarray  # (898, 3)


def _varint(data: bytes, i: int) -> tuple[int, int]:
    value = shift = 0
    while True:
        byte = data[i]
        i += 1
        value |= (byte & 0x7F) << shift
        shift += 7
        if byte < 0x80:
            return value, i


def _fields(data: bytes):
    """``(field number, wire type, value)`` of a protocol-buffer message (values: int or bytes)."""
    i = 0
    while i < len(data):
        key, i = _varint(data, i)
        number, wire = key >> 3, key & 7
        if wire == 0:
            value, i = _varint(data, i)
        elif wire == 1:
            value, i = data[i:i + 8], i + 8
        elif wire == 2:
            size, i = _varint(data, i)
            value, i = data[i:i + size], i + size
        elif wire == 5:
            value, i = data[i:i + 4], i + 4
        else:
            raise ValueError(f"unsupported protocol-buffer wire type {wire}")
        yield number, wire, value


def _floats(wire, value) -> list[float]:
    if wire == 5:
        return [struct.unpack("<f", value)[0]]
    return list(np.frombuffer(value, "<f4"))  # packed


def _ints(wire, value) -> list[int]:
    if wire == 0:
        return [value]
    out, i = [], 0
    while i < len(value):  # packed varints
        v, i = _varint(value, i)
        out.append(v)
    return out


def parse_canonical_face(data: bytes) -> CanonicalFace:
    """The canonical face in a ``GeometryPipelineMetadata`` message (its mesh is the one field holding a
    ``Mesh3d``: vertex buffer of x, y, z, u, v per vertex, then the triangle indices)."""
    for _number, wire, value in _fields(data):
        if wire != 2:
            continue
        floats: list[float] = []
        indices: list[int] = []
        try:
            for number, inner_wire, inner in _fields(value):
                if number == 3 and inner_wire in (2, 5):
                    floats += _floats(inner_wire, inner)
                elif number == 4 and inner_wire in (0, 2):
                    indices += _ints(inner_wire, inner)
        except (ValueError, IndexError, struct.error):
            continue
        if len(floats) == 5 * MESH_COUNT and indices and len(indices) % 3 == 0:
            vertices = np.asarray(floats, np.float64).reshape(-1, 5)
            return CanonicalFace(vertices[:, :3].copy(), vertices[:, 3:].copy(),
                                 np.asarray(indices, np.int64).reshape(-1, 3))
    raise ValueError("no canonical face mesh in the face landmarker's metadata")


def read_canonical_face(task_path) -> CanonicalFace:
    """The canonical face model inside the Face Landmarker bundle ``task_path`` (``face_landmarker.task``)."""
    with zipfile.ZipFile(task_path) as bundle:
        return parse_canonical_face(bundle.read(METADATA))
