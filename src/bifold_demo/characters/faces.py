"""Funny faces for the Rocketbox avatars, made with the small face bones in their heads.

Every Rocketbox head is skinned to 28 face bones below ``Bip01 Head``: the jaw (with the lower lip, the
sides of the lower lip and the tongue on it), the upper lip and its sides, the mouth corners, the cheeks, the muscles
beside the nose (``Caninus``) and in front of the ears (``Masseter``), the upper and lower eyelids, five eyebrow
bones, the nose and the eyeballs. It is a rig of *sliding* bones: the blinks and lip movements authored in the mocap
clips move these bones (a blink pulls the upper eyelid bone 1.4 cm down) rather than turning them; only the jaw, the
tongue and the eyeballs turn. The :class:`~.model.Poser` takes only rotations from the clips (and keeps the model's own
bone offsets), so a face starts out neutral and the clips only turn the eyes and, a little, the jaw and the tongue.

An expression is a set of offsets for some of these bones: how far each slides (to the person's left, up, forward;
centimetres on a grown-up head) and how far it turns (degrees about the same axes). Both are along the character's
own axes in the rest pose, not the head bone's (which leans some 9 degrees forward there); a bone on the jaw slides
along them as they are with the jaw shut and goes along when the jaw opens. :func:`face_rig` converts the offsets
once per skeleton into the frames of the bones' parents and scales the slides by the head size (a child's face is
about 0.8 of a grown-up's), and :func:`apply_face` adds them, each times its weight, to the local bone transforms of
a pose before forward kinematics, so a face goes along with whatever the clips and the head turn do::

    from bifold_demo.characters.faces import apply_face, face_rig

    rig = face_rig(model)                                  # None if the skeleton has no face bones
    person.pose_adjusters.append(lambda local, character, t: apply_face(local, rig, {"tongue": 1.0}))

Weights run from 0 (no change at all) to 1 (the full face) and add up: two expressions at 0.5 each are the halfway
mix of both. The faces are made to read from a few metres away, where a whole face is only some 30 to 50 pixels
tall: the jaw drops far, the tongue hangs out over the chin, the eyebrows go high. :data:`FAR_FACES` are the ones that
still read at the fair's distance; the others (the puffed cheeks, the pursed fish lips, a wink, crossed eyes) need a
closer look. Bones are written once for the person's left side and mirrored for the right (``"MouthCorner"`` is both
corners), or named with their side. Raised eyebrows also slide back a little (:data:`FOREHEAD`): the skin goes up
the forehead, which leans back, rather than off it (and through a hat).

Most offsets grow in proportion to the weight. The tongue's come in late (with the square of the weight, once the
jaw is open), and it lifts over the lower teeth on the way while the mouth opens a little wider for it (most at
weight 0.5, gone at 1), so it never pokes through the teeth while it comes out or goes back in. Out, it tips down
and hangs over the lower lip towards the chin, the one way a tongue shows from the front; it also gets longer and
narrower, as a stuck-out tongue does (the mesh has only its front 2.5 cm). Where it goes is fitted to each mouth
(:func:`tongue_reach`): the heads' tongue bones sit up to 2 cm further forward or back from their lips.

A few heads skin bits of their gums to the tongue bone, which would fly out with the tongue: the jaw gets them when
the avatar is read (:func:`.rocketbox.mend_tongue`), so every model is drawn the same from the start and
:func:`face_rig` only reads the model. It looks at one more quirk of the meshes: a headscarf lying against the cheeks
(:func:`cheek_room`) keeps them from bulging out sideways through it.
"""

from __future__ import annotations

import math
import weakref

import numpy as np

from .model import CharacterModel

ADULT_EYES = 0.062  # metres between the eyeballs of a grown-up Rocketbox head (women 6.0 cm, men 6.4 cm)
COVERED = 0.001  # metres: a headscarf this close beside the cheeks keeps them from bulging out sideways
COVERED_SHARE = 0.25  # how much of their sideways bulge the cheeks keep then
TONGUE_OUT = 0.026  # metres of a grown-up's tongue that hang out beyond the lower lip when it is stuck out
TONGUE_GAP = 0.0015  # metres the stuck-out tongue keeps above the lower lip, the teeth and the chin
FOREHEAD = -0.3  # how far raised brows also slide forward per unit up (so back): the forehead leans back

EXPRESSIONS: dict[str, str] = {
    "smile": "a broad closed-mouth smile: the mouth corners pulled up and out, the cheeks raised",
    "grin": "a wide toothy grin: the lips drawn apart over both rows of teeth, the corners far up and out",
    "tongue": "sticking the tongue out: the mouth open, the tongue long out and hanging down over the lower lip",
    "puff": "blown-up cheeks: the cheeks puffed out round beside the mouth, the lips pressed together and forward",
    "surprise": "big surprise: the jaw dropped into a round O, the eyebrows high, the eyes wide open",
    "scrunch": "a scrunched-up face: the nose wrinkled, the eyes squeezed nearly shut, brows down, upper lip up",
    "fish": "a fish face: the lips pursed far forward, the cheeks sucked in, the eyebrows up",
    "wink": "a wink: the left eye shut, the left mouth corner and cheek up, the other eyebrow raised",
    "sideways": "the mouth pulled over to the left side, the jaw shifted along with it",
    "cross_eyed": "crossed eyes: both eyeballs turned in to the nose, the eyebrows up, the mouth hanging open",
    "roar": "a monster roar: the jaw wide open, the upper lip pulled up over the teeth, the brows down and in",
    "pout": "a sulky pout: the lower lip pushed out and up, the corners down, the inner eyebrows raised",
}
# the faces that still read from the fair's camera, some 3.5 m away (a face some 30 to 50 pixels tall at 1280 x 720)
FAR_FACES = ("smile", "grin", "tongue", "surprise", "scrunch", "roar")

# The bones, as named on the Rocketbox skeletons without the "Bip01 " in front; L / R are the person's sides.
_SIDED = ("MouthCorner", "MouthBottom", "Upperlip", "Caninus", "Cheek", "Masseter", "EyeBlinkTop", "EyeBlinkBottom",
          "InnerEyebrow", "OuterEyebrow", "Eye")
_MIDDLE = ("MJaw", "MBottomLip", "MUpperLip", "MTongue", "MMiddleEyebrow", "MNose")
FACE_BONES = (*_MIDDLE, *(side + name for name in _SIDED for side in "LR"))
_TURNING = ("MJaw", "MTongue", "LEye", "REye")  # the bones whose offsets include turns (the others only slide)
_BROWS = {side + name for name in ("InnerEyebrow", "OuterEyebrow") for side in "LR"} | {"MMiddleEyebrow"}


def _move(x=0.0, y=0.0, z=0.0, *, turn=(0.0, 0.0, 0.0)):
    """A bone's offset: a slide (cm to the left / up / forward) and a turn (degrees about the same axes)."""
    return (float(x), float(y), float(z)), tuple(float(a) for a in turn)


def _turn(pitch=0.0, yaw=0.0, roll=0.0):
    """Only a turn: ``pitch`` about the left axis (positive opens the jaw, tips the tongue down), ``yaw`` about up
    (positive to the left), ``roll`` about forward."""
    return _move(turn=(pitch, yaw, roll))


# expression -> {bone: offset at weight 1}; a bone without a side is both sides, mirrored
_SHAPES: dict[str, dict] = {
    "smile": {
        "MouthCorner": _move(1.8, 1.8, -0.6),
        "MouthBottom": _move(0.9, 0.5, -0.3),
        "Upperlip": _move(0.8, 0.4, -0.2),
        "MUpperLip": _move(0.0, 0.1, 0.0),
        "MBottomLip": _move(0.0, 0.15, 0.0),
        "Cheek": _move(0.3, 1.2, 0.3),
        "Caninus": _move(0.35, 1.1, 0.15),
        "Masseter": _move(0.3, 0.6, 0.3),
        "EyeBlinkBottom": _move(0.0, 0.35, 0.0),
        "EyeBlinkTop": _move(0.0, -0.15, 0.0),
    },
    "grin": {
        "MJaw": _turn(7.0),
        "MUpperLip": _move(0.0, 0.45, 0.1),
        "MBottomLip": _move(0.0, -0.5, 0.1),
        "MouthCorner": _move(1.1, 1.0, -0.6),
        "MouthBottom": _move(0.5, -0.4, 0.0),
        "Upperlip": _move(0.4, 0.5, 0.0),
        "Cheek": _move(0.2, 0.7, 0.15),
        "Caninus": _move(0.25, 0.6, 0.0),
        "EyeBlinkBottom": _move(0.0, 0.3, 0.0),
    },
    "tongue": {
        "MJaw": _turn(12.0),
        "MBottomLip": _move(0.0, -0.3, 0.3),
        "MUpperLip": _move(0.0, 0.2, 0.0),
        "MouthCorner": _move(-0.2, 0.0, 0.0),
        "InnerEyebrow": _move(0.0, 0.4, 0.0),
        "OuterEyebrow": _move(0.0, 0.3, 0.0),
    },
    "puff": {  # the bulge spread over the cheek bones beside the mouth (few vertices each: big slides make facets)
        "Masseter": _move(3.2, 0.0, 1.6),
        "Cheek": _move(1.8, -0.5, 0.8),
        "Caninus": _move(1.2, -0.3, 0.9),
        "MouthCorner": _move(1.2, 0.1, 1.4),
        "MouthBottom": _move(1.3, 0.15, 1.1),
        "MUpperLip": _move(0.0, -0.15, 0.8),  # the lips pressed together and pushed forward
        "MBottomLip": _move(0.0, 0.3, 0.8),
        "Upperlip": _move(0.3, -0.15, 0.8),
        "EyeBlinkTop": _move(0.0, 0.2, 0.0),
        "InnerEyebrow": _move(0.0, 0.4, 0.0),
        "MMiddleEyebrow": _move(0.0, 0.3, 0.0),
    },
    "surprise": {
        "MJaw": _turn(26.0),
        "MouthCorner": _move(-0.5, 0.0, 0.3),
        "MUpperLip": _move(0.0, 0.35, 0.2),
        "MBottomLip": _move(0.0, -0.2, 0.2),
        "Upperlip": _move(-0.2, 0.3, 0.1),
        "MouthBottom": _move(-0.2, -0.2, 0.2),
        "Masseter": _move(-0.3, 0.0, 0.0),
        "InnerEyebrow": _move(0.0, 1.0, 0.1),
        "MMiddleEyebrow": _move(0.0, 0.9, 0.1),
        "OuterEyebrow": _move(0.0, 0.8, 0.0),
        "EyeBlinkTop": _move(0.0, 0.35, 0.05),
        "EyeBlinkBottom": _move(0.0, -0.25, 0.0),
    },
    "scrunch": {
        "MJaw": _turn(3.0),
        "MNose": _move(0.0, 1.2, -0.1),
        "EyeBlinkTop": _move(0.0, -1.15, 0.1),
        "EyeBlinkBottom": _move(0.0, 0.7, 0.05),
        "InnerEyebrow": _move(-0.5, -1.3, 0.15),
        "MMiddleEyebrow": _move(0.0, -1.2, 0.15),
        "OuterEyebrow": _move(0.0, -0.8, 0.0),
        "MUpperLip": _move(0.0, 0.8, 0.05),
        "Upperlip": _move(0.1, 0.9, 0.0),
        "Caninus": _move(-0.1, 1.3, 0.15),
        "Cheek": _move(0.0, 1.3, 0.15),
        "MouthCorner": _move(0.2, 0.4, -0.2),
    },
    "fish": {
        "MJaw": _turn(5.0),
        "MUpperLip": _move(0.0, 0.15, 1.6),
        "MBottomLip": _move(0.0, -0.1, 1.6),
        "Upperlip": _move(-0.5, 0.05, 1.3),
        "MouthBottom": _move(-0.5, 0.0, 1.3),
        "MouthCorner": _move(-1.4, 0.0, 0.8),
        "Masseter": _move(-1.5, 0.0, -0.4),
        "Caninus": _move(-0.9, 0.0, -0.2),
        "Cheek": _move(-0.5, -0.3, -0.2),
        "EyeBlinkTop": _move(0.0, 0.3, 0.0),
        "InnerEyebrow": _move(0.0, 0.5, 0.0),
        "OuterEyebrow": _move(0.0, 0.4, 0.0),
    },
    "wink": {
        "LEyeBlinkTop": _move(0.0, -1.4, 0.12),
        "LEyeBlinkBottom": _move(0.0, 0.6, 0.05),
        "LCheek": _move(0.3, 1.2, 0.2),
        "LCaninus": _move(0.3, 0.9, 0.1),
        "LMouthCorner": _move(1.4, 1.6, -0.5),
        "LMouthBottom": _move(0.6, 0.5, -0.3),
        "LUpperlip": _move(0.5, 0.4, -0.1),
        "LOuterEyebrow": _move(0.0, -0.5, 0.0),
        "LInnerEyebrow": _move(0.0, -0.3, 0.0),
        "RInnerEyebrow": _move(0.0, 0.6, 0.0),
        "ROuterEyebrow": _move(0.0, 0.6, 0.0),
        "MMiddleEyebrow": _move(0.0, 0.1, 0.0),
        "RMouthCorner": _move(-0.4, 0.5, -0.2),
        "RMouthBottom": _move(-0.2, 0.15, 0.0),
    },
    "sideways": {
        "MJaw": _turn(3.0, 9.0),
        "LMouthCorner": _move(1.3, 0.5, -0.3),
        "RMouthCorner": _move(1.2, 0.0, 0.3),
        "MUpperLip": _move(1.1, 0.1, 0.0),
        "MBottomLip": _move(0.3, 0.0, 0.0),
        "LUpperlip": _move(1.1, 0.3, -0.1),
        "RUpperlip": _move(1.1, 0.0, 0.1),
        "LMouthBottom": _move(0.4, 0.35, -0.1),
        "RMouthBottom": _move(0.4, 0.0, 0.1),
        "LCheek": _move(0.2, 0.6, 0.1),
        "LCaninus": _move(0.3, 0.5, 0.0),
        "RCaninus": _move(0.5, 0.0, 0.0),
        "LEyeBlinkBottom": _move(0.0, 0.3, 0.0),
    },
    "cross_eyed": {
        "Eye": _turn(0.0, -28.0),
        "MJaw": _turn(8.0),
        "MBottomLip": _move(0.0, -0.2, 0.0),
        "InnerEyebrow": _move(0.0, 0.6, 0.0),
        "MMiddleEyebrow": _move(0.0, 0.5, 0.0),
    },
    "roar": {
        "MJaw": _turn(30.0),
        "MUpperLip": _move(0.0, 0.6, 0.1),
        "Upperlip": _move(0.2, 0.7, 0.0),
        "Caninus": _move(0.2, 0.8, 0.1),
        "MNose": _move(0.0, 0.4, 0.0),
        "MouthCorner": _move(0.6, -0.3, -0.2),
        "MouthBottom": _move(0.3, -0.4, 0.0),
        "MBottomLip": _move(0.0, -0.5, 0.0),
        "InnerEyebrow": _move(-0.3, -0.7, 0.1),
        "MMiddleEyebrow": _move(0.0, -0.6, 0.1),
        "OuterEyebrow": _move(0.0, 0.3, 0.0),
        "EyeBlinkTop": _move(0.0, 0.25, 0.0),
    },
    "pout": {
        "MBottomLip": _move(0.0, 0.35, 0.7),
        "MouthBottom": _move(-0.1, 0.25, 0.5),
        "MouthCorner": _move(-0.2, -0.7, 0.1),
        "MUpperLip": _move(0.0, 0.0, 0.2),
        "InnerEyebrow": _move(0.1, 0.8, 0.1),
        "MMiddleEyebrow": _move(0.0, 0.5, 0.0),
        "OuterEyebrow": _move(0.0, -0.2, 0.0),
        "EyeBlinkTop": _move(0.0, -0.2, 0.0),
    },
}
_LATE: dict[str, dict] = {  # offsets that grow with the square of the weight (the tongue comes out once the jaw is open)
    # out and tipped down to hang over the lower lip; the slide is fitted to each mouth (tongue_reach), else this one
    "tongue": {"MTongue": _move(0.0, 0.3, 3.3, turn=(50.0, 0.0, 0.0))},
}
_MIDWAY: dict[str, dict] = {  # offsets times 4 w (1 - w): most at weight 0.5, none at 0 and 1
    "tongue": {"MTongue": _move(0.0, 1.3, 0.0), "MJaw": _turn(8.0)},  # over the lower teeth, the mouth opened for it
}
_STRETCH: dict[str, dict] = {  # bones that also stretch: factors to the left / up / forward at weight 1 (come in late)
    "tongue": {"MTongue": (0.8, 1.0, 1.65)},  # a tongue stuck out is longer and narrower (the mesh has its front only)
}
_ORDER = {name: i for i, name in enumerate(EXPRESSIONS)}
# one row of offsets per expression (in the order of EXPRESSIONS), then the late rows, then the midway rows
_PARTS = [(i, 0, _SHAPES.get(name, {})) for i, name in enumerate(EXPRESSIONS)]
_PARTS += [(_ORDER[name], curve, shape) for curve, table in ((1, _LATE), (2, _MIDWAY)) for name, shape in table.items()]
_TONGUE = _ORDER["tongue"]
_TONGUE_OUT = (_TONGUE, 1, _LATE["tongue"])  # the part whose tongue slide is fitted to each mouth


def _sides(shape: dict) -> dict[str, tuple]:
    """``{bone: (slide, turn)}`` with the side-less bones of ``shape`` written out for both sides (mirrored)."""
    out = {}
    for bone, (move, turn) in shape.items():
        if bone in _SIDED:
            x, y, z = move
            rx, ry, rz = turn
            out["L" + bone] = (move, turn)
            out["R" + bone] = ((-x, y, z), (rx, -ry, -rz))  # mirrored in the x = 0 plane (turns are axial vectors)
        else:
            out[bone] = (move, turn)
    return out


class FaceRig:
    """The face bones of one skeleton and its head size, with the expressions worked out for them.

    Attributes:
        bones: skeleton indices of the :data:`FACE_BONES`, in skeleton order.
        turning: skeleton indices of the bones that also turn (the jaw, the tongue and the eyeballs).
        size: the head size relative to a grown-up Rocketbox head (the distance between the eyes over
            :data:`ADULT_EYES`): about 1 for grown-ups, 0.8 for children. The slides are scaled by it.
        covered: a headscarf lies against the cheeks (see :func:`cheek_room`): they bulge out sideways only a
            little (:data:`COVERED_SHARE`), so they do not push through it.
        moves: ``(parts, 3 * len(bones))`` slides in metres in each bone's parent frame, one row per expression part.
        turns: ``(parts, 3 * len(turning))`` turns as rotation vectors (radians) in the parent frames.
        stretches: ``(expression, bone, rest frame, factors - 1)`` of the bones that stretch (see ``_STRETCH``).
        tongue: the slide of the stuck-out tongue fitted to the mouth (see :func:`tongue_reach`; metres, the rest
            pose's character space), or None for the one in the table.
    """

    def __init__(self, skeleton, bones: dict[str, int], size: float, covered: bool = False,
                 tongue: np.ndarray | None = None):
        names = sorted(bones, key=bones.get)
        self.bones = np.array([bones[name] for name in names], dtype=np.int64)
        self.turning = np.array([bones[name] for name in _TURNING], dtype=np.int64)
        self.size = float(size)
        self.covered = bool(covered)
        self.tongue = None if tongue is None else np.asarray(tongue, float)
        row = {name: k for k, name in enumerate(names)}
        parents = skeleton.rest_global[skeleton.parents[self.bones], :3, :3]
        parents = parents / np.linalg.norm(parents, axis=1, keepdims=True)
        moves = np.zeros((len(_PARTS), len(names), 3))
        turns = np.zeros((len(_PARTS), len(names), 3))
        for i, (_, _, shape) in enumerate(_PARTS):
            for bone, (move, turn) in _sides(shape).items():
                x, y, z = move
                if bone in _BROWS and y > 0.0:  # the skin slides up the forehead, not off it (and through a hat)
                    z += FOREHEAD * y
                moves[i, row[bone]] = np.array([x, y, z]) * (0.01 * self.size)
                turns[i, row[bone]] = np.radians(turn)
        if self.covered:
            moves[:, [row[side + name] for name in ("Masseter", "Cheek") for side in "LR"], 0] *= COVERED_SHARE
        if self.tongue is not None:
            moves[_PARTS.index(_TONGUE_OUT), row["MTongue"]] = self.tongue
        # directions in the rest pose's character space -> each bone's parent frame (P^T v)
        turning = [row[name] for name in _TURNING]
        self.moves = np.einsum("kji,pkj->pki", parents, moves).reshape(len(_PARTS), -1)
        self.turns = np.einsum("kji,pkj->pki", parents[turning], turns[:, turning]).reshape(len(_PARTS), -1)
        self._late = np.array([e for e, curve, _ in _PARTS if curve == 1], dtype=np.int64)
        self._midway = np.array([e for e, curve, _ in _PARTS if curve == 2], dtype=np.int64)
        self._curved = {int(e) for e in (*self._late, *self._midway)}
        self.stretches = []
        for name, shape in _STRETCH.items():
            for bone, factors in shape.items():
                frame = skeleton.rest_global[bones[bone], :3, :3]
                frame = frame / np.linalg.norm(frame, axis=0, keepdims=True)  # the bone's axes in character space
                self.stretches.append((_ORDER[name], bones[bone], frame, np.asarray(factors, float) - 1.0))
        # the columns of the lower lip's slides: the stuck-out tongue holds the lip where its own face puts it
        self._lower_lip = np.array([3 * row[name] + k for name in ("MBottomLip", "LMouthBottom", "RMouthBottom")
                                    for k in range(3)], dtype=np.int64)
        first = int(self.bones[0])  # the face bones usually follow each other: a slice is faster than an index
        contiguous = np.array_equal(self.bones, np.arange(first, first + len(self.bones)))
        self._slide = slice(first, first + len(self.bones)) if contiguous else self.bones

    def offsets(self, weights) -> tuple[np.ndarray, np.ndarray, list] | None:
        """``(slides (len(bones), 3), turns (len(turning), 3), [(bone, 3x3 stretch in its own frame), ...])`` for
        ``weights``; None when they change nothing."""
        w = np.zeros(len(EXPRESSIONS))
        active = curved = False
        for name, value in weights.items():
            i = _ORDER.get(name)
            if i is not None and value > 0:
                w[i] = min(float(value), 1.0)
                active = True
                curved = curved or i in self._curved
        if not active:
            return None
        stretches = []
        for e, bone, frame, factors in self.stretches:
            if w[e] > 0.0:  # the scale along the character's rest axes, seen in the bone's own frame
                stretches.append((bone, (frame.T * (1.0 + factors * w[e] * w[e])) @ frame))
        out = w[_TONGUE] * w[_TONGUE]  # how far the tongue is out (as its late parts)
        others = w.copy()
        others[_TONGUE] = 0.0
        if curved:
            late, midway = w[self._late], w[self._midway]
            w = np.concatenate((w, late * late, 4.0 * midway * (1.0 - midway)))
            moves, turns = self.moves, self.turns
        else:
            moves, turns = self.moves[:len(w)], self.turns[:len(w)]
        slides = w @ moves
        if out > 0.0 and others.any():  # the other faces' lower lip gives way to the tongue hanging over it
            slides[self._lower_lip] -= out * (others @ self.moves[:len(others), self._lower_lip])
        return slides.reshape(-1, 3), (w @ turns).reshape(-1, 3), stretches


_RIGS: dict[int, tuple[weakref.ref, FaceRig | None]] = {}  # id(model) -> (the model, its rig), while the model lives


def face_rig(model: CharacterModel) -> FaceRig | None:
    """The :class:`FaceRig` of ``model`` (worked out once per model); None when its skeleton lacks the face bones."""
    known = _RIGS.get(id(model))
    if known is not None and known[0]() is model:
        return known[1]
    skeleton = model.skeleton
    bones = {name: skeleton.find("Bip01 " + name) for name in FACE_BONES}
    rig = None
    if None not in bones.values() and skeleton.find("Bip01 Head") is not None:
        eyes = skeleton.rest_global[bones["LEye"], :3, 3] - skeleton.rest_global[bones["REye"], :3, 3]
        size = float(np.linalg.norm(eyes)) / ADULT_EYES
        if size > 0.1:
            rig = FaceRig(skeleton, bones, size, covered=cheek_room(model, bones) < COVERED,
                          tongue=tongue_reach(model, bones, size))
    _RIGS[id(model)] = (weakref.ref(model), rig)
    weakref.finalize(model, _RIGS.pop, id(model), None)
    return rig


def cheek_room(model: CharacterModel, bones: dict[str, int]) -> float:
    """Metres of room beside the cheeks (the less of both sides; ``inf`` when nothing is there).

    That is how far out sideways from the cheek skin (the vertices the ``Masseter`` bones move) the nearest vertex
    lies that no face bone moves, among those as high and as far forward as that skin. Mostly that is the skin
    around the face, a few millimetres away or more; a headscarf framing the face lies right against it.
    """
    points = np.concatenate([p.positions for p in model.primitives]).astype(np.float64)
    joints = np.concatenate([p.joints for p in model.primitives])
    weights = np.concatenate([p.weights for p in model.primitives])
    still = points[np.where(np.isin(joints, list(bones.values())), weights, 0.0).sum(axis=1) == 0]
    room = np.inf
    for side, sign in (("L", 1.0), ("R", -1.0)):
        cheek = points[np.where(joints == bones[side + "Masseter"], weights, 0.0).sum(axis=1) > 0.15]
        if not len(cheek):
            continue
        (x0, y0, z0), (x1, y1, z1) = cheek.min(axis=0), cheek.max(axis=0)
        beside = still[(still[:, 1] > y0) & (still[:, 1] < y1) & (still[:, 2] > z0) & (still[:, 2] < z1)]
        out = sign * beside[:, 0] - (x1 if sign > 0 else -x0)
        out = out[out > -0.002]  # outside the cheek (or just inside its outermost point)
        if len(out):
            room = min(room, float(out.min()))
    return room


def tongue_reach(model: CharacterModel, bones: dict[str, int], size: float) -> np.ndarray | None:
    """The slide (metres, the rest pose's character space) that lays the stuck-out tongue of ``model`` over its
    lower lip; None when the mesh has no tongue.

    The heads differ too much for one slide: their tongue bones sit up to 2 cm further forward or back from the
    lips, and their lips are thin or thick. So the tongue is turned and stretched as at the full ``"tongue"``
    weight, and slid (in the plane across the face's middle) until its underside lies :data:`TONGUE_GAP` clear of
    the lower lip, the teeth and the chin in front of it (within its width; the lip where its own slides in this
    face take it), and :data:`TONGUE_OUT` of it hangs out beyond the point it lies on.
    """
    points = np.concatenate([p.positions for p in model.primitives]).astype(np.float64)
    joints = np.concatenate([p.joints for p in model.primitives])
    weights = np.concatenate([p.weights for p in model.primitives])

    def on(name):
        return np.where(joints == bones[name], weights, 0.0).sum(axis=1)

    tongue = on("MTongue") > 0.5
    if tongue.sum() < 4:
        return None
    origin = model.skeleton.rest_global[bones["MTongue"], :3, 3]
    pitch = math.radians(_LATE["tongue"]["MTongue"][1][0])  # how far it tips down
    c, s = math.cos(pitch), math.sin(pitch)
    along, up = np.array([0.0, -s, c]), np.array([0.0, c, s])  # the turned tongue's forward and up (turn about x)
    turn = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    shape = ((points[tongue] - origin) * _STRETCH["tongue"]["MTongue"]) @ turn.T  # turned and stretched, at the joint
    jaw_side = ("MJaw", "MBottomLip", "LMouthBottom", "RMouthBottom")
    lip = (sum(on(name) for name in jaw_side) > 0.9) & ~tongue  # the lower lip, teeth and chin, all on the jaw
    lip &= (np.abs(points[:, 0] - origin[0]) < np.abs(shape[:, 0]).max()) & (points[:, 2] > points[tongue, 2].min())
    lip &= points[:, 1] > origin[1] - 0.06 * size
    if not lip.any():
        return None
    slides = _sides(_SHAPES["tongue"])
    lip_points = points[lip] + sum(on(name)[lip, None] * (np.asarray(slides[name][0]) * 0.01 * size)
                                   for name in jaw_side if name in slides)
    contact = lip_points[np.argmax(lip_points @ up)]
    rise = contact @ up + TONGUE_GAP * size - origin @ up - (shape @ up).min()
    reach = contact @ along + TONGUE_OUT * size - origin @ along - (shape @ along).max()
    return rise * up + reach * along


def apply_face(local: np.ndarray, rig: FaceRig, weights) -> None:
    """Make the faces in ``weights`` (``{expression: 0..1}``) on the ``(B, 4, 4)`` local bone transforms, in place.

    Each expression's bone offsets, times its weight, are added to the pose: bones slide in their parent's frame
    and turn about their own joint (the tongue also stretches). While the tongue is out, the lower lip stays where
    the ``"tongue"`` face puts it (the other faces' slides of it fade out as the tongue comes out), so pursed or
    pressed lips do not push into the tongue. Unknown names are ignored; with all weights 0 ``local`` is left
    untouched.
    """
    offsets = rig.offsets(weights)
    if offsets is None:
        return
    moves, turns, stretches = offsets
    for bone, stretch in stretches:
        local[bone, :3, :3] = local[bone, :3, :3] @ stretch
    local[rig._slide, :3, 3] += moves
    for bone, (x, y, z) in zip(rig.turning, turns.tolist()):  # a few bones: plain floats beat array calls here
        angle = math.sqrt(x * x + y * y + z * z)
        if angle > 0.0:
            x, y, z = x / angle, y / angle, z / angle
            c, s = math.cos(angle), math.sin(angle)
            k = 1.0 - c
            turn = np.array([[c + x * x * k, x * y * k - z * s, x * z * k + y * s],
                             [y * x * k + z * s, c + y * y * k, y * z * k - x * s],
                             [z * x * k - y * s, z * y * k + x * s, c + z * z * k]])
            local[bone, :3, :3] = turn @ local[bone, :3, :3]
