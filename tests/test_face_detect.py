"""Face detection and alignment: SCRFD's decoding against hand-made outputs, the alignment against the template, and
the real detectors (skipped without their models: ``BIFOLD_FACE_MODELS``; the photo test also needs
``BIFOLD_FACE_PHOTOS_TEST_DIR`` and writes nothing)."""

import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from bifold_demo.recognition import align, detect, registry
from bifold_demo.recognition.detect import Face

PHOTOS = Path(os.environ.get("BIFOLD_FACE_PHOTOS_TEST_DIR", "/nonexistent")).expanduser()


def _rotation(angle_degrees, scale, shift):
    a = np.radians(angle_degrees)
    return np.array([[scale * np.cos(a), -scale * np.sin(a), shift[0]], [scale * np.sin(a), scale * np.cos(a),
                                                                         shift[1]]])


# -- alignment ------------------------------------------------------------------------------------------------------


def test_the_similarity_transform_is_recovered_exactly_and_never_mirrors():
    truth = _rotation(17.0, 2.5, (30.0, -12.0))
    src = align.TEMPLATE
    dst = src @ truth[:, :2].T + truth[:, 2]
    assert np.allclose(align.similarity(src, dst), truth)
    assert np.allclose(align.similarity(align.TEMPLATE, align.TEMPLATE), np.c_[np.eye(2), np.zeros(2)])
    mirrored = src * [-1.0, 1.0]
    matrix = align.similarity(mirrored, src)
    assert np.linalg.det(matrix[:, :2]) > 0  # the best rotation, not a reflection


def _dots(shape, points, radius=2):
    image = np.zeros((*shape, 3), np.uint8)
    for x, y in points:
        cv2.circle(image, (round(x * 16), round(y * 16)), radius * 16, (255, 255, 255), -1, cv2.LINE_AA, shift=4)
    return image


def _centroids(crop, around):
    grey = crop.astype(float).sum(axis=2)
    out = []
    for x, y in around:
        x0, y0 = int(x) - 6, int(y) - 6
        patch = grey[y0:y0 + 13, x0:x0 + 13]
        ys, xs = np.mgrid[y0:y0 + 13, x0:x0 + 13]
        out.append(((patch * xs).sum() / patch.sum(), (patch * ys).sum() / patch.sum()))
    return np.array(out)


def _face_at(centre, angle_degrees, scale):
    """Landmarks in an image where the crop's centre is at ``centre``, turned by ``angle_degrees``, with ``scale``
    crop pixels per image pixel."""
    a = np.radians(angle_degrees)
    rotation = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return (align.TEMPLATE - 56.0) @ rotation.T / scale + centre


@pytest.mark.parametrize("scale", [1.3, 0.25])  # a small face (a plain warp) and a big one (shrunk first)
def test_alignment_puts_the_landmarks_on_the_template(scale):
    marks = _face_at((450.0, 430.0), -12.0, scale)
    image = _dots((900, 900), marks, radius=max(2, round(2 / scale)))
    aligned = align.align(image, marks)
    assert np.allclose(aligned.matrix[:, :2] @ marks.T + aligned.matrix[:, 2:], align.TEMPLATE.T, atol=1e-9)
    assert abs(aligned.scale - scale) < 1e-9 and aligned.crop.shape == (112, 112, 3) and aligned.outside == 0.0
    found = _centroids(aligned.crop, align.TEMPLATE)
    assert np.abs(found - align.TEMPLATE).max() < 0.35, found - align.TEMPLATE


def test_the_crop_beyond_the_image_is_black_and_counted():
    image = np.full((200, 200, 3), 200, np.uint8)
    marks = align.TEMPLATE * 1.5 - [30.0, 40.0]  # the face reaches over the top-left corner
    aligned = align.align(image, marks)
    assert 0.1 < aligned.outside < 0.6
    assert (aligned.crop[:5, :5] == 0).all() and (aligned.crop[-5:, -5:] == 200).all()


def test_big_faces_are_shrunk_before_the_warp_so_fine_detail_does_not_alias():
    stripes = np.zeros((1200, 1200, 3), np.uint8)
    stripes[:, ::2] = 255  # one-pixel stripes: grey once averaged
    marks = _face_at((600.0, 600.0), 7.0, 1 / 6)
    smooth = align.align(stripes, marks).crop.astype(float)
    aliased = align.align(stripes, marks, antialias=False).crop.astype(float)
    assert abs(smooth.mean() - 127.5) < 3 and smooth.std() < 6 and aliased.std() > 40


# -- SCRFD's decoding ---------------------------------------------------------------------------------------------


def test_anchor_centres_follow_the_reference_order():
    centres = detect.anchor_centres(2, 3, 8)
    assert centres.tolist() == [[0, 0], [0, 0], [8, 0], [8, 0], [16, 0], [16, 0],
                                [0, 8], [0, 8], [8, 8], [8, 8], [16, 8], [16, 8]]
    assert detect.anchor_centres(1, 2, 16, anchors=1).tolist() == [[0, 0], [16, 0]]


def _outputs(canvas=(64, 64), faces=()):
    """SCRFD's nine outputs for ``canvas``, zero except for ``faces``: (stride, cell x, cell y, anchor, score,
    distances, landmark offsets) in stride units."""
    width, height = canvas
    scores, boxes, marks = [], [], []
    for stride in detect.STRIDES:
        n = (height // stride) * (width // stride) * 2
        scores.append(np.zeros((n, 1), np.float32))
        boxes.append(np.zeros((n, 4), np.float32))
        marks.append(np.zeros((n, 10), np.float32))
    for stride, cx, cy, anchor, score, distance, offsets in faces:
        k = detect.STRIDES.index(stride)
        i = (cy * (width // stride) + cx) * 2 + anchor
        scores[k][i], boxes[k][i], marks[k][i] = score, distance, offsets
    return scores + boxes + marks


def test_scrfd_outputs_decode_into_boxes_and_landmarks():
    offsets = [-1, -1, 1, -1, 0, 0, -1, 1, 1, 1]
    outputs = _outputs(faces=[(16, 1, 2, 1, 0.9, [1.0, 1.5, 2.0, 2.5], offsets),
                              (8, 7, 7, 0, 0.6, [0.5, 0.5, 0.5, 0.5], [0] * 10),
                              (32, 0, 0, 0, 0.3, [1, 1, 1, 1], [0] * 10)])  # below the threshold
    boxes, scores, marks = detect.decode_scrfd(outputs, (64, 64))
    assert np.allclose(scores, [0.9, 0.6])
    assert np.allclose(boxes[0], [16 - 16, 32 - 24, 16 + 32, 32 + 40])  # centre (16, 32), distances x 16
    assert np.allclose(marks[0], np.array(offsets).reshape(5, 2) * 16 + [16, 32])
    assert np.allclose(boxes[1], [52, 52, 60, 60])
    with pytest.raises(ValueError, match="anchors"):
        detect.decode_scrfd(_outputs((96, 64)), (64, 64))


def test_overlapping_boxes_are_suppressed_at_insightfaces_overlap_by_default():
    """Review L3: SCRFD's suppression and score keep InsightFace's defaults (overlap 0.4, score 0.5)."""
    assert (detect.SCRFD_SCORE, detect.SCRFD_NMS) == (0.5, 0.4)
    boxes = np.array([[0, 0, 19, 19], [6, 0, 25, 19]], float)  # overlap 280 / 520 = 0.54 (with the +1 areas)
    scores = np.array([0.9, 0.8])
    assert detect.nms(boxes, scores) == [0] and detect.nms(boxes, scores, 0.6) == [0, 1]


def test_nms_keeps_the_best_of_overlapping_boxes():
    boxes = np.array([[0, 0, 9, 9], [1, 1, 10, 10], [5, 0, 14, 9], [50, 50, 60, 60]], float)
    scores = np.array([0.8, 0.9, 0.7, 0.5])
    # 1 and 0 overlap by 81/119: 0 goes; 2 overlaps 1 by 54/146 (with the +1 areas), under 0.4: kept
    assert detect.nms(boxes, scores, 0.4) == [1, 2, 3]


class _FakeScrfd:
    """Stands in for the network: records the blob, answers with one face at canvas pixels (0, 8)-(40, 56)."""

    def __init__(self):
        self.blobs = []

    def run(self, blob):
        self.blobs.append(blob)
        height, width = blob.shape[2:]
        return _outputs((width, height), [(8, 3, 4, 0, 0.95, [3.0, 3.0, 2.0, 3.0], [0, -2, 2, -2, 1, 0, 0, 2, 2, 2])])


def test_the_scrfd_detector_letterboxes_and_maps_back():
    detector = object.__new__(detect.ScrfdDetector)
    detector.spec, detector.canvas, detector.score, detector.overlap = registry.SCRFD, 64, 0.5, 0.4
    detector.model = _FakeScrfd()
    image = np.zeros((64, 128, 3), np.uint8)
    image[..., 0] = 255  # blue
    (face,) = detector.detect(image)
    blob = detector.model.blobs[0]
    assert blob.shape == (1, 3, 64, 64)  # 128 x 64 fits as 64 x 32 at the top, the rest black
    assert np.allclose(blob[0, :, 0, 0], [-127.5 / 128, -127.5 / 128, 127.5 / 128])  # red, green, blue
    assert np.allclose(blob[0, :, 40, 0], -127.5 / 128)
    assert np.allclose(face.box, (0, 16, 80, 96)) and face.score == pytest.approx(0.95)  # canvas x 2
    assert np.allclose(face.points[0], ((24 + 0) * 2, (32 - 16) * 2))
    assert detector.detect(image, min_size=100) == []
    for canvas, shape in ((200, (224, 224)), ((300, 200), (224, 320)), (180, (192, 192))):  # (review L1: crashed)
        detector.detect(image, canvas=canvas)
        assert detector.model.blobs[-1].shape[2:] == shape


def test_canvas_sizes_and_letterboxing():
    assert detect.canvas_size((720, 1280), "image") == (1280, 736)
    assert detect.canvas_size((3000, 4000), "image") == (1600, 1216)
    assert detect.canvas_size((10, 10), 160) == (160, 160) and detect.canvas_size((10, 10), (96, 64)) == (96, 64)
    # review L1: any canvas is rounded up to a multiple of 32, SCRFD's coarsest stride (else its outputs do not tile)
    assert detect.canvas_size((10, 10), 200) == (224, 224) and detect.canvas_size((10, 10), (300, 200)) == (320, 224)
    assert detect.canvas_size((10, 10), 250) == (256, 256) and detect.canvas_size((10, 10), 20) == (32, 32)
    board, scale = detect.letterbox(np.full((50, 100, 3), 9, np.uint8), (64, 64))
    assert scale == 0.64 and (board[:32] == 9).all() and (board[32:] == 0).all()


def test_faces_map_from_a_crop_back_to_the_frame():
    face = Face((10.0, 20.0, 30.0, 40.0), ((1.0, 2.0),) * 2 + ((3.0, 4.0),) * 3, 0.9)
    moved = face.mapped(0.5, (100.0, 200.0))
    assert moved.box == (105.0, 210.0, 15.0, 20.0) and moved.landmarks[0] == (100.5, 201.0)
    assert Face((0, 0, 1, 1), ((0.0, 0.0), (3.0, 4.0), (0, 0), (0, 0), (0, 0)), 1.0).eye_distance == 5.0


# -- the real detectors ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def detectors():
    try:
        scrfd = detect.ScrfdDetector(registry.path(registry.SCRFD, download=False))
        yunet = detect.YuNetDetector(registry.path(registry.YUNET, download=False))
    except (OSError, ImportError) as exc:
        pytest.skip(f"face models unavailable (bifold-faces fetch --profile best): {exc}")
    return scrfd, yunet


@pytest.fixture(scope="module")
def zidane():
    from bifold_demo.scene.textures import sample_image

    path = sample_image("zidane.jpg")
    if not path:
        pytest.skip("the Ultralytics sample photo is not installed")
    return cv2.imread(str(path))


def _plausible(face: Face) -> bool:
    """The landmarks in the detectors' order: eyes left to right above the nose, mouth corners below it."""
    (le, re, nose, lm, rm) = face.points
    return le[0] < re[0] and lm[0] < rm[0] and max(le[1], re[1]) < nose[1] < min(lm[1], rm[1])


def test_scrfd_and_yunet_find_the_same_faces_with_landmarks_in_one_order(detectors, zidane):
    scrfd, yunet = detectors
    for canvas in (640, "image"):
        found = scrfd.detect(zidane, canvas=canvas)
        assert len(found) == 2 and all(f.score > 0.6 for f in found)
    found, other = scrfd.detect(zidane, canvas="image"), yunet.detect(zidane)
    assert len(other) == 2
    frontal = max(found, key=lambda f: f.eye_distance)  # the man facing the camera
    twin = min(other, key=lambda f: np.linalg.norm(f.points - frontal.points))
    assert _plausible(frontal) and _plausible(twin)
    assert np.linalg.norm(twin.points - frontal.points, axis=1).max() < 0.3 * frontal.eye_distance
    again = scrfd.detect(zidane, canvas="image")
    assert [f.box for f in again] == [f.box for f in found]  # deterministic


def test_the_alignment_matches_opencvs_aligncrop(detectors, zidane):
    sface = cv2.FaceRecognizerSF.create(str(registry.path(registry.SFACE, download=False)), "")
    for face in detectors[0].detect(zidane, canvas="image"):
        ours = align.align(zidane, face.landmarks, antialias=False).crop
        assert np.abs(ours.astype(int) - sface.alignCrop(zidane, face.row()).astype(int)).mean() < 0.5


def test_the_detectors_find_the_faces_in_the_test_photos(detectors):
    if not PHOTOS.is_dir():
        pytest.skip("set BIFOLD_FACE_PHOTOS_TEST_DIR to a folder of photos of people")
    scrfd, yunet = detectors
    paths = sorted(p for p in PHOTOS.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    assert paths
    for path in paths:
        image = cv2.imread(str(path))
        found, other = scrfd.detect(image, canvas=640), yunet.detect(image)
        assert found and other, path.name
        best = found[0]
        assert _plausible(best), path.name
        twin = min(other, key=lambda f: np.linalg.norm(f.points - best.points))
        assert np.median(np.linalg.norm(twin.points - best.points, axis=1)) < 0.2 * best.eye_distance, path.name
