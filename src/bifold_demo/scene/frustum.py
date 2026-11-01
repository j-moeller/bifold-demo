"""Frustum culling: which objects can show in a picture, tested by bounds against the picture's frustum.

A window of the virtual high-resolution sensor (:meth:`.renderer.Renderer.render_window`) shows a small part of the
view, so most of the scene lies outside its frustum; drawing only what may show is what makes a window cost about its
own pixels. The tests are conservative: an object is left out only when its bounds lie wholly behind one of the
frustum's six planes, so culling never changes a pixel.

- **Meshes** (boxes, planes, pictures, props): the axis-aligned box of the mesh's vertices, carried into the scene by
  the object's model matrix (centre moved, half sizes through the absolute matrix: a box that holds the moved box).
- **Characters**: one sphere per bone, around the bone's posed joint, whose radius is the largest distance of a vertex
  that the bone moves (any weight) from its joint in the bind pose (:func:`bone_radii`), plus :data:`BONE_MARGIN`. A
  rigid part keeps its distance from its joint, and a blended vertex lies between its bones' rigid places. Measured on
  the 20 prepared Rocketbox avatars in 84 poses of walk, idle, talk, turn and listen clips each, no skinned vertex lay
  outside the union of these spheres. What a character wears (``worn``) is counted with it.
- **Blob shadows**: the rectangle on the floor that the renderer draws.
"""

from __future__ import annotations

import weakref

import numpy as np

BONE_MARGIN = 0.03  # metres added to every bone's radius


def frustum_planes(clip_from_world: np.ndarray) -> np.ndarray:
    """The six planes ``(a, b, c, d)`` of the frustum of ``clip_from_world`` (a projection times a view matrix), with
    ``a x + b y + c z + d >= 0`` inside: left, right, bottom, top, near, far (Gribb and Hartmann's extraction for
    OpenGL clip space)."""
    m = np.asarray(clip_from_world, dtype=np.float64)
    return np.array([m[3] + m[0], m[3] - m[0], m[3] + m[1], m[3] - m[1], m[3] + m[2], m[3] - m[2]])


def boxes_outside(planes: np.ndarray, centres, half_sizes) -> np.ndarray:
    """Which axis-aligned boxes (``(N, 3)`` centres and half sizes) lie wholly behind a plane of the frustum."""
    centres, half_sizes = np.asarray(centres, dtype=np.float64), np.asarray(half_sizes, dtype=np.float64)
    reach = centres @ planes[:, :3].T + half_sizes @ np.abs(planes[:, :3]).T + planes[:, 3]  # (N, 6)
    return (reach < 0).any(axis=1)


def spheres_outside(planes: np.ndarray, centres, radii) -> np.ndarray:
    """Which spheres (``(N, 3)`` centres, ``(N,)`` radii) lie wholly behind a plane of the frustum."""
    centres = np.asarray(centres, dtype=np.float64)
    norms = np.linalg.norm(planes[:, :3], axis=1)
    reach = centres @ planes[:, :3].T + planes[:, 3] + np.asarray(radii, dtype=np.float64)[:, None] * norms
    return (reach < 0).any(axis=1)


def world_boxes(models, centres, half_sizes) -> tuple[np.ndarray, np.ndarray]:
    """``(centres, half sizes)`` (``(N, 3)`` each) of boxes that hold the boxes ``centres``, ``half_sizes`` moved by
    the ``(N, 4, 4)`` ``models`` matrices."""
    models = np.asarray(models, dtype=np.float64)
    linear = models[:, :3, :3]
    return (np.einsum("nij,nj->ni", linear, centres) + models[:, :3, 3],
            np.einsum("nij,nj->ni", np.abs(linear), half_sizes))


class Bounds:
    """Bounds of meshes and of characters' bones, computed once per mesh, mesh part or outfit and kept while it
    exists; :meth:`boxes` and :meth:`spheres` place them in the scene for many objects at once."""

    def __init__(self):
        self._boxes: dict[int, tuple] = {}  # id(mesh) -> (weakref, (centre, half sizes))
        self._radii: dict[int, tuple] = {}  # id(skinned primitive) -> (weakref, radii (B,))
        self._outfits: dict[int, tuple] = {}  # id(character) -> (weakref, its parts' ids, radii (B,))

    def mesh_box(self, mesh) -> tuple[np.ndarray, np.ndarray]:
        """``(centre, half sizes)`` of the box of ``mesh``'s vertices (its own coordinates)."""
        entry = self._boxes.get(id(mesh))
        if entry is None or entry[0]() is not mesh:
            positions = np.asarray(mesh.positions, dtype=np.float64)
            if len(positions):
                lo, hi = positions.min(axis=0), positions.max(axis=0)
            else:
                lo = hi = np.zeros(3)
            entry = (weakref.ref(mesh), ((lo + hi) / 2, (hi - lo) / 2))
            self._boxes[id(mesh)] = entry
        return entry[1]

    def boxes(self, objects, camera_position) -> tuple[np.ndarray, np.ndarray]:
        """``(centres, half sizes)``, ``(N, 3)`` each: the scene boxes of scene objects (their meshes, where their
        model matrices put them)."""
        n = len(objects)
        models, centres, halves = np.empty((n, 4, 4)), np.empty((n, 3)), np.empty((n, 3))
        for i, obj in enumerate(objects):
            models[i] = obj.matrices(camera_position)[0]
            centres[i], halves[i] = self.mesh_box(obj.mesh)
        return world_boxes(models, centres, halves)

    def bone_radii(self, character) -> np.ndarray:
        """``(B,)`` radii of ``character``'s bones: its model's mesh parts and what it wears (see the module docs),
        :data:`BONE_MARGIN` included."""
        model = character.skinned_model
        parts = (*model.primitives, *getattr(character, "worn", ()))
        ids = tuple(id(part) for part in parts)
        entry = self._outfits.get(id(character))
        if entry is not None and entry[0]() is character and entry[1] == ids:
            return entry[2]
        bones = len(model.inverse_bind)
        radii, bind = np.zeros(bones), None
        for part in parts:
            known = self._radii.get(id(part))
            if known is None or known[0]() is not part or len(known[1]) != bones:
                if bind is None:  # the joints where the mesh was bound to the skeleton
                    bind = np.linalg.inv(np.asarray(model.inverse_bind, dtype=np.float64))[:, :3, 3]
                known = (weakref.ref(part), bone_radii(part, bind))
                self._radii[id(part)] = known
            np.maximum(radii, known[1], out=radii)
        radii += BONE_MARGIN
        self._outfits[id(character)] = (weakref.ref(character), ids, radii)
        return radii

    def spheres(self, characters, camera_position) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(centres (M, 3), radii (M,), owners (M,))``: the posed bone spheres of ``characters`` in the scene,
        ``owners`` the index of each sphere's character."""
        models = np.array([c.matrices(camera_position)[0] for c in characters])
        joints = [np.asarray(c.pose)[:, :3, 3] for c in characters]
        owners = np.repeat(np.arange(len(characters)), [len(j) for j in joints])
        placed = models[owners]
        centres = np.einsum("mij,mj->mi", placed[:, :3, :3], np.concatenate(joints)) + placed[:, :3, 3]
        stretch = np.linalg.norm(models[:, :3, :3], axis=1).max(axis=1)  # the largest scale of each model matrix
        radii = np.concatenate([self.bone_radii(c) for c in characters]) * stretch[owners]
        return centres, radii, owners

    def forget_unused(self) -> None:
        """Drop the bounds of meshes, mesh parts and characters that no longer exist."""
        for table in (self._boxes, self._radii, self._outfits):
            for key, (ref, *_value) in list(table.items()):
                if ref() is None:
                    del table[key]


def bone_radii(part, bind_joints: np.ndarray) -> np.ndarray:
    """``(B,)``: for each bone, the largest distance from its joint (``bind_joints``: where the joints were when the
    mesh was bound, in the mesh's coordinates) of a vertex of skinned mesh ``part`` that the bone moves (with any
    weight); 0 for bones that move none of its vertices."""
    positions = np.asarray(part.positions, dtype=np.float64)
    joints = np.asarray(part.joints).astype(np.int64)
    weights = np.asarray(part.weights)
    radii = np.zeros(len(bind_joints))
    for k in range(joints.shape[1]):
        moved = weights[:, k] > 0
        bones = joints[moved, k]
        np.maximum.at(radii, bones, np.linalg.norm(positions[moved] - bind_joints[bones], axis=1))
    return radii
