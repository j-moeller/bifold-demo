"""Person, face, identity (step F2) with stand-ins, runnable anywhere: faces in head regions and whose they are, the
evidence per track, calibration, open-set decisions and the pipeline. (The real models and avatars:
``test_identity_real.py``.)"""

import math

import numpy as np
import pytest

from bifold_demo.recognition.detect import Face
from bifold_demo.recognition.heads import BOX_PER_EYES, FaceSighting, HeadModel, assign_faces, face_size, find_faces


def _face(middle, eyes, score=0.9):
    """A frontal face with its eyes ``eyes`` pixels apart around ``middle``."""
    x, y = middle
    height = BOX_PER_EYES * eyes
    box = (x - 0.45 * height, y - 0.41 * height, 0.9 * height, height)
    marks = ((x - eyes / 2, y), (x + eyes / 2, y), (x, y + 0.5 * eyes), (x - 0.4 * eyes, y + eyes),
             (x + 0.4 * eyes, y + eyes))
    return Face(box, marks, score)


def _head_face(box, head=HeadModel(), dx=0.0, dy=0.0, scale=1.0):
    """The face of a person in ``box``, where the head model expects it (moved by ``dx``, ``dy`` eye distances)."""
    (x, y), eyes = head.expected(box)
    return _face((x + dx * eyes, y + dy * eyes), eyes * scale)


def _sighting(face, source="frame"):
    return FaceSighting(face, face, np.zeros((4, 4, 3), np.uint8), source)


# -- head regions and whose face a face is ----------------------------------------------------------------------------


def test_a_face_fits_the_head_it_belongs_to():
    head = HeadModel()
    box = (100.0, 100.0, 180.0, 400.0)
    face = _head_face(box)
    assert head.cost(face, box) == pytest.approx(0.0, abs=1e-6)
    assert face_size(face) == pytest.approx(head.expected(box)[1])
    assert 0 < head.cost(_head_face(box, dx=1.0), box) < head.cost(_head_face(box, dx=2.0), box)
    assert head.cost(_head_face(box, dy=1.0), box) > head.cost(_head_face(box, dx=1.0), box)  # rows are tighter
    assert head.cost(_head_face(box, scale=0.5), box) > 2.0
    assert head.cost(_head_face(box, dy=6.0), box) > head.limit  # (on the neck: in the region, far too low)
    assert head.cost(_head_face(box, dy=8.0), box) == math.inf  # (on the chest: outside the head region)
    assert head.cost(_face((500, 500), 11.0), box) == math.inf


def test_overlapping_head_regions_give_each_face_to_its_own_person():
    """B stands behind A, a little to the side: B's face lies in A's head region too."""
    head = HeadModel()
    a = (100.0, 120.0, 200.0, 420.0)  # nearer, bigger
    b = (150.0, 100.0, 220.0, 310.0)
    face_a, face_b = _head_face(a), _head_face(b)
    x0, y0, x1, y1 = head.square(a)
    middle = face_b.points[:2].mean(axis=0)
    assert x0 <= middle[0] <= x1 and y0 <= middle[1] <= y1  # in A's region as well as B's
    given = assign_faces([_sighting(face_b), _sighting(face_a)], {7: a, 9: b}, head)
    assert [s.track for s in given] == [9, 7]
    # with A's own face turned away (not found), B's face still does not become A's
    given = assign_faces([_sighting(face_b)], {7: a, 9: b}, head)
    assert [s.track for s in given] == [9]
    # and without B's track (B not detected), B's face fits A too badly to be given to them
    given = assign_faces([_sighting(face_b)], {7: a}, head)
    assert [s.track for s in given] == [None]


def test_each_track_gets_at_most_one_face_and_each_face_one_track():
    head = HeadModel()
    box = (100.0, 100.0, 180.0, 400.0)
    near, off = _head_face(box), _head_face(box, dx=1.5)
    given = assign_faces([_sighting(off), _sighting(near)], {3: box}, head)
    assert [s.track for s in given] == [None, 3]  # the better fit wins
    assert given[1].cost == pytest.approx(head.cost(near, box))
    assert assign_faces([], {3: box}) == [] and [s.track for s in assign_faces([_sighting(near)], {})] == [None]


class _Window:
    """A stand-in :class:`~bifold_demo.sensor.WindowResult`: frame pixels ``p`` are ``(p - offset) * scale`` here."""

    def __init__(self, image, offset, scale, tag=1):
        self.image, self.offset, self.tag = image, np.asarray(offset, float), scale
        self.window = type("W", (), {"scale": scale})()
        self.tag = tag

    def to_view(self, points):
        return np.asarray(points, float) / self.window.scale + self.offset


class _Engine:
    """Finds the faces it was given (frame pixels) in the frame, a region of it, or a window's image."""

    def __init__(self, faces, frame):
        self.faces, self.frame, self.windows, self.calls = faces, frame, {}, []

    def detect(self, image, min_size=0.0, canvas=None, region=None):
        self.calls.append((id(image), region, canvas))
        if image is self.frame:
            offset, scale = np.zeros(2), 1.0
        else:
            offset, scale = self.windows[id(image)]
        found = []
        for face in self.faces:
            if region is not None:
                x, y, w, h = region
                middle = face.points[:2].mean(axis=0)
                if not (x <= middle[0] <= x + w and y <= middle[1] <= y + h):
                    continue
            box = ((face.box[0] - offset[0]) * scale, (face.box[1] - offset[1]) * scale, face.box[2] * scale,
                   face.box[3] * scale)
            marks = tuple(tuple((np.asarray(p) - offset) * scale) for p in face.landmarks)
            local = Face(box, marks, face.score)
            inside = image is self.frame or (0 <= box[0] and box[0] + box[2] <= image.shape[1] and 0 <= box[1]
                                             and box[1] + box[3] <= image.shape[0])
            if inside and local.size >= min_size:
                found.append(local)
        return found


def test_faces_are_found_in_windows_and_crops_and_mapped_to_the_frame():
    frame = np.zeros((720, 1280, 3), np.uint8)
    head = HeadModel()
    a, b = (100.0, 100.0, 180.0, 400.0), (600.0, 200.0, 650.0, 380.0)
    face_a, face_b = _head_face(a), _head_face(b)
    engine = _Engine([face_a, face_b], frame)
    x0, y0, x1, y1 = head.square(b)
    picture = np.zeros((round((y1 - y0) * 6), round((x1 - x0) * 6), 3), np.uint8)
    window = _Window(picture, (x0, y0), 6.0)
    engine.windows[id(picture)] = (window.offset, 6.0)
    found = find_faces(engine, frame, {1: head.square(a), 2: head.square(b)}, [window])
    assert [s.source for s in found] == ["window", "frame"]  # b's face found twice: the window's is kept
    window_face, frame_face = found
    assert np.allclose(window_face.face.points, face_b.points) and np.allclose(frame_face.face.points, face_a.points)
    assert window_face.image is picture and window_face.scale == 6.0
    assert np.allclose(window_face.local.points, (face_b.points - window.offset) * 6.0)  # aligned from the window
    assert window_face.size == pytest.approx(6.0 * face_size(face_b))
    given = assign_faces(found, {1: a, 2: b}, head)
    assert [s.track for s in given] == [2, 1]
    # a region reaching out of the frame is cut to it; one with no pixels left is not looked at
    engine.calls.clear()
    find_faces(engine, frame, {1: (-50.0, -50.0, 40.0, 40.0), 2: (1300.0, 0.0, 1400.0, 50.0)})
    assert [call[1] for call in engine.calls] == [(0.0, 0.0, 40.0, 40.0)]
