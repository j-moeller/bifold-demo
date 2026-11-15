"""YOLOX, the commercial profile's person detector: decoding its output, and the real model (its official ONNX in the
face models' folder, ``BIFOLD_FACE_MODELS``; nothing is downloaded) on Ultralytics' sample photo."""

import os

import numpy as np
import pytest

from bifold_demo.detection.yolox import YoloxDetector, decode, letterbox


def test_the_output_is_decoded_on_the_strides_grids():
    size = (64, 64)  # strides 8, 16, 32: 8x8 + 4x4 + 2x2 anchors
    raw = np.zeros((84, 85), np.float32)
    k = 2 * 8 + 3  # stride 8, column 3, row 2
    raw[k, :4] = (0.5, 0.5, np.log(2.0), np.log(3.0))  # the centre half a cell on, 2 by 3 strides big
    raw[k, 4], raw[k, 5] = 0.9, 0.8  # objectness, "person"
    j = 64 + 1 * 4 + 2  # stride 16, column 2, row 1
    raw[j, :4] = (0.0, 0.0, 0.0, 0.0)
    out = decode(raw, size)
    assert np.allclose(out[k, :4], [28 - 8, 20 - 12, 28 + 8, 20 + 12])
    assert np.allclose(out[j, :4], [32 - 8, 16 - 8, 32 + 8, 16 + 8])
    assert out[k, 4] == pytest.approx(0.9) and out.shape == (84, 85)


def test_the_image_is_letterboxed_at_the_top_left_in_grey():
    image = np.full((100, 200, 3), 7, np.uint8)
    board, scale = letterbox(image, (64, 64))
    assert scale == pytest.approx(0.32) and board.shape == (64, 64, 3)
    assert (board[:32] == 7).all() and (board[33:] == 114).all()


@pytest.fixture(scope="module")
def yolox():
    folder = os.environ.get("BIFOLD_FACE_MODELS")
    if not folder or not (os.path.isfile(os.path.join(folder, "yolox_s.onnx"))):
        pytest.skip("YOLOX-s is not in the face models' folder (bifold-faces fetch --model yolox_s)")
    return YoloxDetector("yolox_s", models_folder=folder, download=False, progress=None)


def test_yolox_finds_the_people_and_the_bus_of_the_sample_photo(yolox):
    import cv2

    from bifold_demo.scene.textures import sample_image

    path = sample_image("bus.jpg")
    if not path:
        pytest.skip("the Ultralytics sample photo is not installed")
    image = cv2.imread(str(path))
    found = yolox.detect(image)
    people = [d for d in found if d.label == "person"]
    assert len(people) >= 3 and any(d.label == "bus" for d in found)
    # where they are (YOLOX-s's own demo on this photo: the bus, three people in front of it and one cut off at the
    # left edge), each once (no box left over that the non-maximum suppression should have taken)
    expected = [("bus", (17.8, 224.9, 805.9, 736.3), 0.942), ("person", (669.7, 386.8, 810.0, 878.9), 0.907),
                ("person", (222.5, 408.7, 346.8, 854.0), 0.892), ("person", (47.1, 399.0, 241.0, 893.5), 0.892),
                ("person", (0.1, 553.8, 63.7, 880.7), 0.538)]
    assert len(found) == len(expected)
    for (label, box, score), d in zip(expected, sorted(found, key=lambda d: -d.confidence)):
        assert d.label == label and np.allclose(d.box, box, atol=2.0), (d, box)
        assert d.confidence == pytest.approx(score, abs=0.01)
    assert all(0 <= d.box[0] < d.box[2] <= image.shape[1] and 0 <= d.box[1] < d.box[3] <= image.shape[0]
               for d in found)
    assert [d.box for d in yolox.detect(image)] == [d.box for d in found]  # the same twice
    weak = yolox.detect(image, conf=0.1)
    assert len(weak) >= len(found)  # the tracker's weak boxes: a lower score this time
    only = YoloxDetector("yolox_s", models_folder=os.environ["BIFOLD_FACE_MODELS"], download=False,
                         classes=["person"], progress=None)
    assert {d.label for d in only.detect(image)} == {"person"}
