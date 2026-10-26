"""Cloth coordinates: where each texel of a garment lies on the fabric, in metres, for patterns that fit the body.

A pattern drawn in the texture atlas would be cut at every UV seam, stretched where the atlas is, and of a different
size on each island. Instead every clothes texel gets coordinates on the body (:func:`surface`), worked out from its
bind-pose point (:mod:`..uvbake`), so a pattern has one size everywhere, runs on across UV cuts and matches front and
back:

* the body is cut into parts the way clothes are sewn: the torso, each arm (a sleeve), each leg (a trouser leg) and
  each foot. An upper-body texel belongs to an arm when it is nearer to the arm than to the torso (each measured
  against its own girth), which puts the seam round the shoulder; the trouser legs meet in the middle, front and back;
* a part is unrolled like a cylinder around its axis: ``u`` is the distance around it and ``v`` the height along it.
  The torso is an elliptic cylinder (wide and flat) unrolled by arc length from the middle of the front, so its
  seam is at the middle of the back; a limb is a round one around the line from its first joint to its last (the
  shoulder to the wrist, the hip to the ankle, the ankle to the toes), with the seam on its inner side (under the arm,
  the inside leg), where real sleeves and trousers have theirs. The left and right limbs are mirror images.

``u`` grows to the person's left (the viewer's right, seen from the front) on the torso, ``v`` upwards (towards the
shoulder or the hip on a limb), so a pattern's vertical stripes run down the body and along the sleeves and legs.
:attr:`Surface.footprint` is the size of each texel on the body (metres), for filtering patterns finer than a texel.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from .regions import BOTTOMS, GROUPS, OTHER, SHIRT, SHOES, SKIN, TOP, bone_group

PARTS = ("torso", "left arm", "right arm", "left leg", "right leg", "left foot", "right foot")


@dataclass
class Surface:
    """Cloth coordinates of the clothes texels of one body texture (``size`` x ``size``)."""

    part: np.ndarray  # int8: index into PARTS, -1 for skin and outside
    uv: np.ndarray  # float32 (size, size, 2): metres around and along the part (NaN where part is -1)
    footprint: np.ndarray  # float32: metres per texel on the body (NaN outside)


BEND = 0.04  # metres either side of an elbow or a knee over which the two segments' coordinates are blended


def _unit(v):
    v = np.asarray(v, np.float64)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _across(v, axis):
    """``v`` without its part along ``axis``, as a unit vector."""
    return _unit(v - np.dot(v, axis) * axis)


@dataclass
class _Segment:
    origin: np.ndarray  # its first joint
    axis: np.ndarray  # unit, towards its last joint
    length: float
    start: float  # distance along the limb to its first joint
    outer: np.ndarray  # unit, across the axis: the side opposite the seam (angle 0)
    front: np.ndarray  # unit, across the axis and the outer side (positive angles towards it)


class _Limb:
    """A limb as a chain of joints (shoulder, elbow, wrist): cylinder coordinates around each segment, blended at
    the bends so they run on smoothly."""

    def __init__(self, joints, side: float, kind: str):
        self.segments = []
        start = 0.0
        for first, last in itertools.pairwise(joints):
            axis = _unit(last - first)
            if kind == "arm":
                seam = _across(np.array([-side, -1.0, 0.0]), axis)  # under the arm, towards the body
                front = np.array([0.0, 0.0, 1.0])
            elif kind == "leg":
                seam = _across(np.array([-side, 0.0, 0.0]), axis)  # the inside leg
                front = np.array([0.0, 0.0, 1.0])
            else:  # a foot: the seam on the inner side, positive angles over the top
                seam = _across(np.array([-side, 0.0, 0.0]), axis)
                front = np.array([0.0, 1.0, 0.0])
            outer = -seam
            front = _across(front - np.dot(front, outer) * outer, axis)
            length = float(np.linalg.norm(last - first))
            self.segments.append(_Segment(np.asarray(first, np.float64), axis, length, start, outer, front))
            start += length

    def distance(self, points: np.ndarray) -> np.ndarray:
        """Distance of ``points`` to the chain of joints."""
        best = np.full(len(points), np.inf)
        for seg in self.segments:
            d = points - seg.origin
            t = np.clip(d @ seg.axis, 0.0, seg.length)
            best = np.minimum(best, np.linalg.norm(d - t[:, None] * seg.axis, axis=1))
        return best

    def coordinates(self, points: np.ndarray, radius: float) -> np.ndarray:
        """``(N, 2)``: ``radius`` x the angle round the limb (0 on the outer side, +-pi at the seam), and minus the
        distance along it (so ``v`` grows towards the body)."""
        out = None
        for k, seg in enumerate(self.segments):
            d = points - seg.origin
            t = d @ seg.axis
            radial = d - t[:, None] * seg.axis
            angle = np.arctan2(radial @ seg.front, radial @ seg.outer)
            here = np.c_[radius * angle, -(seg.start + t)]
            if out is None:
                out = here
                continue
            previous = self.segments[k - 1]
            bisector = _unit(previous.axis + seg.axis)
            w = np.clip(0.5 + ((points - seg.origin) @ bisector) / (2.0 * BEND), 0.0, 1.0)
            w = w * w * (3.0 - 2.0 * w)
            turn = 2.0 * np.pi * radius
            here[:, 0] += turn * np.round((out[:, 0] - here[:, 0]) / turn)  # the same side of the seam
            out = out * (1.0 - w)[:, None] + here * w[:, None]
        return out


def _joints(skeleton):
    """The joints the parts hang on: ``{(kind, side): [joints]}`` (bind-pose points: shoulder, elbow, wrist; hip,
    knee, ankle; ankle, toes) and the depth of the torso's axis."""
    rest = skeleton.rest_global[:, :3, 3]
    groups = [bone_group(n) for n in skeleton.names]
    found = {}
    def bones(group, side):
        return [i for i, g in enumerate(groups) if g == group and rest[i, 0] * side > 0.02]

    for side in (1.0, -1.0):
        arms, hands, legs, feet = (bones(g, side) for g in ("arms", "hands", "legs", "feet"))
        if arms:
            chain = [rest[i] for i in sorted(arms, key=lambda i: abs(rest[i, 0]))]
            if hands:
                chain.append(rest[min(hands, key=lambda i: abs(rest[i, 0]))])
            if len(chain) >= 2:
                found[("arm", side)] = chain
        if legs:
            chain = [rest[i] for i in sorted(legs, key=lambda i: -rest[i, 1])]
            ankle = rest[max(feet, key=lambda i: rest[i, 1])] if feet else None
            if ankle is not None:
                chain.append(ankle)
            if len(chain) >= 2:
                found[("leg", side)] = chain
            if feet:
                toe = rest[max(feet, key=lambda i: rest[i, 2])]
                if np.linalg.norm(toe - ankle) < 0.03:
                    toe = ankle + np.array([0.0, -0.05, 0.15])
                found[("foot", side)] = [ankle, toe]
    spine = [i for i, n in enumerate(skeleton.names) if "spine" in n.lower() or "chest" in n.lower()]
    centre_z = float(np.median(rest[spine, 2])) if spine else 0.0
    return found, centre_z


def _ellipse_arc(a: float, b: float, n: int = 2048):
    """Arc length along the ellipse ``(a sin t, b cos t)`` from t = 0, sampled on ``[0, pi]``."""
    t = np.linspace(0.0, np.pi, n)
    speed = np.sqrt((a * np.cos(t)) ** 2 + (b * np.sin(t)) ** 2)
    arc = np.concatenate([[0.0], np.cumsum(0.5 * (speed[1:] + speed[:-1]) * np.diff(t))])
    return t, arc


def footprint(prim, bake) -> np.ndarray:
    """Metres per texel on the body (from each triangle's area on the body and in the texture)."""
    tri = np.asarray(prim.indices, np.int64).reshape(-1, 3)
    p = np.asarray(prim.positions, np.float64)
    uv = np.asarray(prim.uvs, np.float64) * bake.size
    area3 = 0.5 * np.linalg.norm(np.cross(p[tri[:, 1]] - p[tri[:, 0]], p[tri[:, 2]] - p[tri[:, 0]]), axis=1)
    e1, e2 = uv[tri[:, 1]] - uv[tri[:, 0]], uv[tri[:, 2]] - uv[tri[:, 0]]
    area2 = 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    per = np.sqrt(area3 / np.maximum(area2, 1e-9)).astype(np.float32)
    good = area2 > 0.5
    if good.any():
        per = np.where(good, per, np.median(per[good]))
    out = np.full((bake.size, bake.size), np.nan, np.float32)
    out[bake.covered] = per[bake.triangle[bake.covered]]
    return out


def surface(model, regions) -> Surface:
    """The cloth coordinates of the clothes texels of ``regions`` (a :class:`.regions.BodyRegions` of ``model``)."""
    bake, labels = regions.bake, regions.labels
    size = bake.size
    joints, centre_z = _joints(model.skeleton)
    limbs = {key: _Limb(chain, key[1], key[0]) for key, chain in joints.items()}
    part = np.full((size, size), -1, np.int8)
    uv = np.full((size, size, 2), np.nan, np.float32)
    position = bake.position
    clothes = bake.covered & (labels != SKIN) & (labels != 255)
    upper = clothes & np.isin(labels, [TOP, SHIRT, OTHER])
    lower = clothes & (labels == BOTTOMS)
    shoes = clothes & (labels == SHOES)
    groups = regions.groups

    # the torso's girth (half width and half depth at the chest), for the elliptic cylinder and the shoulder seam
    torso_like = upper & np.isin(groups, [GROUPS.index("torso"), GROUPS.index("neck")])
    pts = position[torso_like] if torso_like.any() else position[clothes]
    if len(pts) == 0:
        return Surface(part, uv, footprint(model.primitives[regions.primitive], bake))
    half_width = max(float(np.percentile(np.abs(pts[:, 0]), 95)), 0.05)
    half_depth = max(float(np.percentile(np.abs(pts[:, 2] - centre_z), 95)), 0.04)

    # upper body: torso or an arm, whichever is nearer for its girth
    ys, xs = np.nonzero(upper)
    points = position[ys, xs].astype(np.float64)
    torso_distance = np.hypot(points[:, 0] / half_width, (points[:, 2] - centre_z) / half_depth)
    which = np.zeros(len(points), np.int8)
    best = torso_distance.copy()
    for k, side in ((1, 1.0), (2, -1.0)):
        limb = limbs.get(("arm", side))
        if limb is None:
            continue
        on_side = points[:, 0] * side > 0
        arm_points = np.isin(groups[ys, xs], [GROUPS.index("arms"), GROUPS.index("hands")]) & on_side
        radius = float(np.median(limb.distance(points[arm_points]))) if arm_points.sum() > 20 else 0.05
        d = limb.distance(points) / max(radius, 0.02)
        nearer = on_side & (d < best)
        which[nearer], best[nearer] = k, d[nearer]
    torso = which == 0
    theta = np.arctan2(points[torso, 0] / half_width, (points[torso, 2] - centre_z) / half_depth)
    t, arc = _ellipse_arc(half_width, half_depth)
    u = np.sign(theta) * np.interp(np.abs(theta), t, arc)
    uv[ys[torso], xs[torso]] = np.c_[u, points[torso, 1]]
    part[ys[torso], xs[torso]] = 0
    for k, side in ((1, 1.0), (2, -1.0)):
        sel = which == k
        if sel.any():
            limb = limbs[("arm", side)]
            radius = float(np.median(limb.distance(points[sel])))
            uv[ys[sel], xs[sel]] = limb.coordinates(points[sel], radius)
            part[ys[sel], xs[sel]] = k

    # trouser legs and shoes, left (+X) and right
    for mask, kind, base in ((lower, "leg", 3), (shoes, "foot", 5)):
        ys, xs = np.nonzero(mask)
        points = position[ys, xs].astype(np.float64)
        for k, side in ((base, 1.0), (base + 1, -1.0)):
            limb = limbs.get((kind, side))
            sel = points[:, 0] * side >= 0
            if limb is None or not sel.any():
                continue
            radius = float(np.median(limb.distance(points[sel])))
            uv[ys[sel], xs[sel]] = limb.coordinates(points[sel], radius)
            part[ys[sel], xs[sel]] = k
        missing = mask & (part < 0)  # no such limb in the skeleton: lay the pattern flat on the front view
        if missing.any():
            uv[missing] = position[missing][:, :2]
            part[missing] = 0
    return Surface(part, uv, footprint(model.primitives[regions.primitive], bake))
