import time

import cv2
import numpy as np
import pytest

from bifold_demo.app import DemoApp, DemoConfig, parse_args
from bifold_demo.detection import AsyncDetector, Detection, DetectionResult
from bifold_demo.display import draw_detections, draw_panel
from bifold_demo.scene import Camera, Quad, Scene, sample_image
from bifold_demo.sources import Frame, SourceSwitcher

BASIC = "bifold_demo.scenes.basic"  # light scene for app tests


class FakeDetector:
    """Reports one 'thing' box in the middle of every frame."""

    model_name, device, conf = "fake.pt", "cpu", 0.35

    def __init__(self, *args, delay=0.0, **kwargs):
        self.delay = delay
        self.seen = []

    def detect_frame(self, frame):
        time.sleep(self.delay)
        self.seen.append(frame.index)
        w, h = frame.size
        det = Detection((w * 0.25, h * 0.25, w * 0.75, h * 0.75), 0.9, 0, "thing")
        return DetectionResult([det], frame.index, frame.source, frame.size, self.delay * 1000, time.perf_counter())


def _frame(i, source="scene"):
    return Frame(np.zeros((90, 160, 3), np.uint8), i, time.perf_counter(), source)


def test_async_detector_keeps_only_the_newest_frame():
    fake = FakeDetector(delay=0.05)
    worker = AsyncDetector(fake).start()
    try:
        for i in range(1, 11):
            worker.submit(_frame(i))
        result = worker.wait_for(_frame(10), timeout=2.0)
        assert result.frame_index == 10
        assert len(fake.seen) < 10  # intermediate frames were skipped, not queued
    finally:
        worker.stop()


def test_async_detector_reports_errors():
    class Broken:
        def detect_frame(self, frame):
            raise RuntimeError("boom")

    worker = AsyncDetector(Broken()).start()
    try:
        worker.submit(_frame(1))
        deadline = time.time() + 2
        while worker.error is None and time.time() < deadline:
            time.sleep(0.01)
        assert "boom" in worker.error
    finally:
        worker.stop()


def test_overlay_draws_inside_the_image():
    image = np.zeros((720, 1280, 3), np.uint8)
    draw_detections(image, [Detection((-50, -50, 100, 100), 0.5, 3, "car"), Detection((10, 10, 640, 360), 1, 0, "x")],
                    from_size=(640, 360))
    draw_panel(image, ["hello", "world"], corner="bottom-right")
    assert image.any()


def test_switcher_falls_back_when_a_source_fails():
    class Good:
        def start(self):
            pass

        def stop(self):
            pass

    def broken():
        raise RuntimeError("no camera")

    switcher = SourceSwitcher({"scene": Good, "webcam": broken}, initial="scene")
    assert switcher.switch("scene")
    assert not switcher.toggle()
    assert switcher.active_name == "scene" and "no camera" in switcher.last_error


def test_cli_parsing():
    cfg = parse_args(["--source", "webcam", "--size", "640x480", "--classes", "person", "car", "--uploads", ""])
    assert (cfg.source, cfg.width, cfg.height, cfg.classes, cfg.uploads) == ("webcam", 640, 480, ["person", "car"], None)
    with pytest.raises(SystemExit):
        parse_args(["--size", "big"])


def test_headless_run_records_annotated_video(gl, tmp_path, monkeypatch):
    import bifold_demo.detection as detection

    monkeypatch.setattr(detection, "YoloDetector", FakeDetector)
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "out.mp4"
    app = DemoApp(DemoConfig(scene=BASIC, headless=True, frames=12, width=320, height=180, record=str(out)))
    app.run()
    cap = cv2.VideoCapture(str(out))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, image = cap.read()
    cap.release()
    assert ok and frames == 12 and image.shape == (180, 320, 3)
    assert app.detector.seen == list(range(1, 13))  # headless = synchronous: every frame analysed


def test_keys_control_the_demo(gl, tmp_path, monkeypatch):
    import bifold_demo.detection as detection

    monkeypatch.setattr(detection, "YoloDetector", FakeDetector)
    monkeypatch.chdir(tmp_path)
    app = DemoApp(DemoConfig(scene=BASIC, headless=True, frames=1, width=160, height=90, detect=False, camera=99))
    app.run()
    app.switcher = SourceSwitcher({"scene": lambda: app.scene_source, "webcam": _no_camera}, "scene")
    window = type("W", (), {"toggle_fullscreen": lambda self: None})()
    app._handle_key(ord("d"), window)
    assert app.detector is not None and app.detect_enabled
    app._handle_key(ord(" "), window)
    assert app.scene_source.clock.paused
    app._handle_key(ord("]"), window)
    assert app.detector.conf == pytest.approx(0.40)
    app._handle_key(ord("s"), window)
    assert app.switcher.active_name == "scene" and "camera" in app.switcher.last_error
    app._handle_key(ord("q"), window)
    assert not app._running


def _no_camera():
    raise RuntimeError("cannot open camera 99")


@pytest.mark.yolo
def test_yolo_finds_objects_in_the_rendered_scene(gl, tmp_path):
    pytest.importorskip("ultralytics")
    bus = sample_image("bus.jpg")
    if bus is None:
        pytest.skip("ultralytics sample images not installed")
    from bifold_demo.detection import YoloDetector
    from bifold_demo.sources import FixedStepClock, SceneSource

    try:
        detector = YoloDetector("yolo26n.pt", device="cpu", models_dir=tmp_path)
    except Exception as exc:  # noqa: BLE001 - offline machine
        pytest.skip(f"cannot load YOLO weights: {exc}")
    scene = Scene(background=(40, 40, 40))
    scene.camera = Camera(position=(0, 0, 3.2), target=(0, 0, 0), fov=50)
    scene.add(Quad(texture=bus, height=2.6, rotation=(0, 20, 0), lit=False), name="poster")
    source = SceneSource(scene, 1280, 720, clock=FixedStepClock())
    try:
        result = detector.detect_frame(source.read())
    finally:
        source.stop()
    labels = {d.label for d in result.detections}
    assert {"bus", "person"} <= labels
    for det in result.detections:
        x1, y1, x2, y2 = det.box
        assert 0 <= x1 < x2 <= 1280 and 0 <= y1 < y2 <= 720


def test_detector_failure_keeps_the_video_running(gl, tmp_path, monkeypatch):
    import bifold_demo.detection as detection

    class Offline:
        def __init__(self, *args, **kwargs):
            raise ConnectionError("no network")

    monkeypatch.setattr(detection, "YoloDetector", Offline)
    monkeypatch.chdir(tmp_path)
    app = DemoApp(DemoConfig(scene=BASIC, headless=True, frames=3, width=160, height=90))
    app.run()
    assert app.detector is None and "no network" in app._detector_error
    assert app._last_canvas is not None
