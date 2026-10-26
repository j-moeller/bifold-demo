"""Minimal reader for binary FBX files (versions 7.x) in pure Python.

The file is a tree of named records. Each record has a list of typed properties and child
records. Array properties (optionally zlib-compressed) become numpy arrays; ``S`` (string)
properties become ``str`` and ``R`` (raw) properties ``bytes``.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

MAGIC = b"Kaydara FBX Binary  \x00\x1a\x00"

_SCALARS = {b"Y": ("<h", 2), b"C": ("<?", 1), b"I": ("<i", 4), b"F": ("<f", 4), b"D": ("<d", 8), b"L": ("<q", 8)}
_ARRAYS = {b"f": np.dtype("<f4"), b"d": np.dtype("<f8"), b"l": np.dtype("<i8"), b"i": np.dtype("<i4"),
           b"b": np.dtype("u1")}


class FBXNode:
    """One FBX record: ``name``, ``props`` (scalars, strings, bytes or numpy arrays) and ``children``."""

    __slots__ = ("name", "props", "children")

    def __init__(self, name: str, props: list | None = None, children: list[FBXNode] | None = None):
        self.name = name
        self.props = props if props is not None else []
        self.children = children if children is not None else []

    def find(self, name: str) -> FBXNode | None:
        """First direct child called ``name`` (or ``None``)."""
        for child in self.children:
            if child.name == name:
                return child
        return None

    def find_all(self, name: str) -> list[FBXNode]:
        """All direct children called ``name``."""
        return [child for child in self.children if child.name == name]

    def __repr__(self) -> str:
        return f"FBXNode({self.name!r}, props={len(self.props)}, children={len(self.children)})"


def read_fbx(path) -> FBXNode:
    """Parse a binary FBX file into a root :class:`FBXNode` (named ``""``) holding the top-level records."""
    data = Path(path).read_bytes()
    return parse_fbx(data)


def parse_fbx(data: bytes) -> FBXNode:
    """Parse the bytes of a binary FBX file (see :func:`read_fbx`)."""
    if not data.startswith(MAGIC):
        raise ValueError("not a binary FBX file")
    (version,) = struct.unpack_from("<I", data, len(MAGIC))
    wide = version >= 7500
    root = FBXNode("", [version])
    offset = len(MAGIC) + 4
    while offset < len(data):
        node, offset = _read_node(data, offset, wide)
        if node is None:
            break
        root.children.append(node)
    return root


def _read_node(data: bytes, offset: int, wide: bool) -> tuple[FBXNode | None, int]:
    if wide:
        end, count, _length = struct.unpack_from("<QQQ", data, offset)
        offset += 24
    else:
        end, count, _length = struct.unpack_from("<III", data, offset)
        offset += 12
    name_len = data[offset]
    offset += 1
    if end == 0:
        return None, offset + name_len
    name = data[offset:offset + name_len].decode("utf-8", "replace")
    offset += name_len
    props = []
    for _ in range(count):
        value, offset = _read_property(data, offset)
        props.append(value)
    node = FBXNode(name, props)
    while offset < end:
        child, offset = _read_node(data, offset, wide)
        if child is None:
            break
        node.children.append(child)
    return node, end


def _read_property(data: bytes, offset: int):
    code = data[offset:offset + 1]
    offset += 1
    scalar = _SCALARS.get(code)
    if scalar is not None:
        return struct.unpack_from(scalar[0], data, offset)[0], offset + scalar[1]
    dtype = _ARRAYS.get(code)
    if dtype is not None:
        length, encoding, size = struct.unpack_from("<III", data, offset)
        offset += 12
        raw = data[offset:offset + size]
        if encoding == 1:
            raw = zlib.decompress(raw)
        elif encoding != 0:
            raise ValueError(f"unknown FBX array encoding {encoding}")
        array = np.frombuffer(raw, dtype=dtype, count=length)
        return (array.astype(bool) if code == b"b" else array), offset + size
    if code in (b"S", b"R"):
        (size,) = struct.unpack_from("<I", data, offset)
        offset += 4
        raw = bytes(data[offset:offset + size])
        return (raw.decode("utf-8", "replace") if code == b"S" else raw), offset + size
    raise ValueError(f"unknown FBX property type {code!r} at byte {offset - 1}")
