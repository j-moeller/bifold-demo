"""The demo application: video source -> YOLO detector -> on-screen overlay."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2

from .detection import DEFAULT_MODEL, AsyncDetector, DetectionResult
from .display import Window, draw_banner, draw_detections, draw_panel, draw_selection
from .display.window import TAB
from .loader import SceneReloader, load_scene
from .scene import Scene
from .selection import Selection, person_name
from .sources import FixedStepClock, Frame, SceneSource, SimulationClock, SourceSwitcher, WebcamSource

HELP = ("S source | D detect | Space pause | R reset | [ ] confidence | Y people | F fullscreen | P screenshot | "
        "H hide | Q quit")
STALE_AFTER = 1.0  # seconds after which old detections are no longer drawn


@dataclass
class DemoConfig:
    """Options for :func:`run` (the command-line flags map onto these)."""

    source: str = "scene"  # "scene" or "webcam"
    scene: str | None = None  # scene file/module; None = built-in demo
    camera: int = 0
    width: int = 1280
    height: int = 720
    fps: float = 30.0
    mirror: bool = True
    model: str = DEFAULT_MODEL
    device: str = "auto"
    conf: float = 0.35
    iou: float = 0.5
    imgsz: int = 640
    classes: list[str] | None = None
    detect: bool = True
    sync: bool = False
    fullscreen: bool = False
    record: str | None = None
    headless: bool = False
    frames: int | None = None
    reload: bool = True
    uploads: str | None = "uploads"
    title: str = "bifold demo"


class _Rate:
    """Smoothed events-per-second counter."""

    def __init__(self):
        self.value, self._last = 0.0, None

    def tick(self, now: float) -> None:
        if self._last is not None:
            dt = max(now - self._last, 1e-6)
            self.value = 1 / dt if self.value == 0 else 0.9 * self.value + 0.1 / dt
        self._last = now


class DemoApp:
    def __init__(self, config: DemoConfig, scene: Scene | None = None):
        self.config = config
        self.reloader = None
        if scene is None:
            scene, path = load_scene(config.scene)
            if config.reload and path is not None:
                self.reloader = SceneReloader(config.scene, path)
        self._prepare_scene(scene)
        clock = FixedStepClock(config.fps) if config.headless else SimulationClock()
        self.scene_source = SceneSource(scene, config.width, config.height, config.fps, clock)
        self.switcher = SourceSwitcher(
            {
                "scene": lambda: self.scene_source,
                "webcam": lambda: WebcamSource(config.camera, config.width, config.height, config.fps, config.mirror),
            },
            initial=config.source,
        )
        self.detector = None
        self.worker: AsyncDetector | None = None
        self.detect_enabled = config.detect
        self.sync = config.sync or config.headless  # headless renders faster than real time: keep boxes in step
        self.show_hud = True
        self.display_rate = _Rate()
        self._message: tuple[str, float, bool] | None = None  # (text, expiry time, is_error)
        self._detector_error: str | None = None
        self._last_submitted: tuple[str, int] | None = None
        self._writer = None
        self._record_size: tuple[int, int] | None = None
        self._last_canvas = None
        self._last_reload_check = 0.0
        self._running = True
        self.selection = Selection()  # Y: pick a person, number keys: yoga poses

    def _prepare_scene(self, scene: Scene) -> None:
        if self.config.uploads:
            scene.watch_folder(self.config.uploads)

    # -- setup ------------------------------------------------------------------------

    def _load_detector(self) -> bool:
        """Load YOLO; on failure keep the video running and show the error (``D`` retries)."""
        from .detection import YoloDetector

        cfg = self.config
        print(f"[detector] loading {cfg.model} ...")
        try:
            self.detector = YoloDetector(cfg.model, cfg.device, cfg.conf, cfg.iou, cfg.imgsz, cfg.classes)
        except ValueError:
            raise  # bad options such as an unknown class name: report them and stop
        except Exception as exc:  # noqa: BLE001 - e.g. weights cannot be downloaded
            self._detector_error = f"detector unavailable: {type(exc).__name__}: {exc}"
            print(f"[detector] {self._detector_error}")
            return False
        self._detector_error = None
        print(f"[detector] ready on {self.detector.device}")
        if not self.sync:
            self.worker = AsyncDetector(self.detector).start()
        return True

    def notify(self, message: str, seconds: float = 2.5, error: bool = False) -> None:
        print(f"[demo] {message}")
        self._message = (message, time.perf_counter() + seconds, error)

    # -- main loop --------------------------------------------------------------------

    def run(self) -> None:
        cfg = self.config
        if not self.switcher.switch(cfg.source):
            print(f"[demo] {self.switcher.last_error}")
            if cfg.source == "scene" or not self.switcher.switch("scene"):
                raise RuntimeError(self.switcher.last_error)
            self.notify(f"{cfg.source} unavailable, showing the scene instead", 5, error=True)
        if cfg.detect:
            self._load_detector()
        window = None if cfg.headless else Window(cfg.title, cfg.fullscreen)
        if window:
            print(f"[demo] controls: {HELP}")
        count = 0
        try:
            while self._running:
                frame = self.switcher.active.read(timeout=0.5)
                if frame is not None:
                    if frame.source == "scene":
                        message = self.selection.update(self.scene_source.scene, frame.size)
                        if message:
                            self.notify(message)
                    result = self._detections_for(frame)
                    self._last_canvas = self._compose(frame, result)
                    self.display_rate.tick(time.perf_counter())
                    if cfg.record:
                        self._record(self._last_canvas)
                    count += 1
                if window and self._last_canvas is not None:
                    window.show(self._last_canvas)
                    key = window.poll_key(1)
                    if key is not None:
                        self._handle_key(key, window)
                    if not window.is_open():
                        break
                if cfg.frames and count >= cfg.frames:
                    break
                self._maybe_reload()
        except KeyboardInterrupt:
            pass
        finally:
            if self.worker:
                self.worker.stop()
            self.selection.close()
            self.switcher.stop()
            if self._writer is not None:
                self._writer.release()
                print(f"[demo] saved recording to {cfg.record}")
            if window:
                window.close()

    def _detections_for(self, frame: Frame) -> DetectionResult | None:
        if not self.detect_enabled or self.detector is None:
            return None
        if self.worker is None:
            return self.detector.detect_frame(frame)
        key = (frame.source, frame.index)
        if key != self._last_submitted:
            self.worker.submit(frame)
            self._last_submitted = key
        result = self.worker.latest()
        if result is None or result.source != frame.source or time.perf_counter() - result.timestamp > STALE_AFTER:
            return None
        return result

    def _compose(self, frame: Frame, result: DetectionResult | None):
        canvas = frame.image.copy()  # the detector may still be reading the original
        if result is not None:
            draw_detections(canvas, result.detections, None if result.frame_size == frame.size else result.frame_size)
        selecting = self.selection.active and frame.source == "scene"
        if selecting and self.selection.box is not None:
            t = self.scene_source.scene.time
            draw_selection(canvas, self.selection.box, person_name(self.selection.selected),
                           self.selection.describe(self.selection.selected, t))
        if self.show_hud:
            draw_panel(canvas, self._status_lines(frame, result))
            if not self.config.headless:
                draw_panel(canvas, self.selection.help_lines() if selecting else [HELP], corner="bottom-left")
        if self._message and time.perf_counter() < self._message[1]:
            text, _expiry, is_error = self._message
            draw_banner(canvas, text, color=(40, 40, 200) if is_error else (60, 140, 40))
        else:
            error = (getattr(self.switcher.active, "error", None) or self._detector_error
                     or (self.worker.error if self.worker else None))
            if error:
                draw_banner(canvas, error)
        return canvas

    def _status_lines(self, frame: Frame, result: DetectionResult | None) -> list[str]:
        w, h = frame.size
        lines = [f"source: {frame.source}  {w}x{h}  {self.display_rate.value:4.1f} fps"]
        if frame.source == "scene":
            clock = self.scene_source.clock
            lines[0] += f"  t={clock.now():5.1f}s" + ("  PAUSED" if clock.paused else "")
        if self.detector is None or not self.detect_enabled:
            lines.append("detection: off")
        else:
            rate = self.worker.fps if self.worker else self.display_rate.value
            latency = f"{result.inference_ms:3.0f} ms" if result else "--"
            lines.append(f"{self.detector.model_name} on {self.detector.device}: {rate:4.1f} fps, {latency}, "
                         f"conf>={self.detector.conf:.2f}")
            if result is not None:
                counts: dict[str, int] = {}
                for det in result.detections:
                    counts[det.label] = counts.get(det.label, 0) + 1
                summary = ", ".join(f"{n} {label}" for label, n in sorted(counts.items(), key=lambda kv: -kv[1]))
                lines.append(f"objects: {summary or 'none'}")
        if frame.source == "scene":
            lines += self.selection.status_lines(self.scene_source.scene.time)
        return lines

    def _handle_key(self, key: int, window: Window) -> None:
        clock = self.scene_source.clock
        if self.switcher.active_name == "scene":
            handled, message = self.selection.handle_key(key, self.scene_source.scene.time)
            if handled:
                if message:
                    self.notify(message, 3.0)
                return
        if key in (ord("q"), 27):
            self._running = False
        elif key == ord("s"):
            target = "webcam" if self.switcher.active_name == "scene" else "scene"
            if self.switcher.switch(target):
                self.selection.leave()
                self.notify(f"source: {target}")
            else:
                self.notify(self.switcher.last_error, 5, error=True)
        elif key == ord("d"):
            if self.detector is None:
                if not self._load_detector():
                    return
                self.detect_enabled = False
            self.detect_enabled = not self.detect_enabled
            self.notify(f"detection {'on' if self.detect_enabled else 'off'}")
        elif key == ord(" "):
            self.notify("paused" if clock.toggle_pause() else "running", 1.0)
        elif key == ord("r"):
            self.scene_source.reset()
            self.selection.reset()
            self.notify("scene reset", 1.0)
        elif key in (ord("y"), TAB):
            if self.switcher.active_name != "scene":
                self.notify("people can only be selected in the 3D scene (S switches to it)", 3.0, error=True)
            else:
                source = self.scene_source
                self.notify(self.selection.enter(source.scene, (source.width, source.height)), 4.0)
        elif key in (ord("["), ord("]")) and self.detector is not None:
            step = 0.05 if key == ord("]") else -0.05
            self.detector.conf = round(min(0.95, max(0.05, self.detector.conf + step)), 2)
            self.notify(f"confidence threshold {self.detector.conf:.2f}", 1.0)
        elif key == ord("f"):
            window.toggle_fullscreen()
        elif key == ord("h"):
            self.show_hud = not self.show_hud
        elif key == ord("m") and self.switcher.active_name == "webcam":
            cam = self.switcher.active
            cam.mirror = not cam.mirror
        elif key == ord("p"):
            self._screenshot()

    def _screenshot(self) -> None:
        if self._last_canvas is None:
            return
        folder = Path("screenshots")
        folder.mkdir(exist_ok=True)
        path = folder / time.strftime("bifold_%Y%m%d_%H%M%S.png")
        cv2.imwrite(str(path), self._last_canvas)
        self.notify(f"saved {path}")

    def _record(self, canvas) -> None:
        h, w = canvas.shape[:2]
        if self._writer is None:
            Path(self.config.record).parent.mkdir(parents=True, exist_ok=True)
            self._writer = cv2.VideoWriter(self.config.record, cv2.VideoWriter_fourcc(*"mp4v"), self.config.fps, (w, h))
            self._record_size = (w, h)
        if (w, h) != self._record_size:
            canvas = cv2.resize(canvas, self._record_size)
        self._writer.write(canvas)

    def _maybe_reload(self) -> None:
        if self.reloader is None:
            return
        now = time.perf_counter()
        if now - self._last_reload_check < 0.5:
            return
        self._last_reload_check = now
        scene = self.reloader.check()
        if scene is not None:
            self._prepare_scene(scene)
            self.scene_source.set_scene(scene)
            self.notify("scene reloaded", 1.5)


def run(scene: Scene | str | None = None, **options) -> None:
    """Start the demo.

    ``scene`` is a :class:`Scene`, a scene file/module spec, or ``None`` for the
    built-in demo. ``options`` are the fields of :class:`DemoConfig`, e.g.
    ``run(scene, source="webcam", model="yolo26s.pt", conf=0.5)``.
    """
    if isinstance(scene, str):
        options["scene"] = scene
        scene = None
    DemoApp(DemoConfig(**options), scene=scene).run()


def _size(text: str) -> tuple[int, int]:
    try:
        w, h = text.lower().split("x")
        return int(w), int(h)
    except ValueError:
        raise argparse.ArgumentTypeError("size must look like 1280x720") from None


def parse_args(argv=None) -> DemoConfig:
    p = argparse.ArgumentParser(
        prog="bifold-demo",
        description="Real-time object detection on a webcam or a rendered 3D scene.",
        epilog=f"Keys: {HELP}",
    )
    g = p.add_argument_group("video")
    g.add_argument("--source", choices=["scene", "webcam"], default="scene", help="video source to start with")
    g.add_argument("--scene", help="scene file (path/to/scene.py[:function]) or module; default: built-in demo")
    g.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    g.add_argument("--size", type=_size, default=(1280, 720), help="frame size WxH (default 1280x720)")
    g.add_argument("--fps", type=float, default=30.0, help="target frame rate (default 30)")
    g.add_argument("--no-mirror", dest="mirror", action="store_false", help="do not mirror the webcam image")
    g.add_argument("--uploads", default="uploads",
                   help="folder whose files replace textures of same-named objects ('' to disable)")
    g.add_argument("--no-reload", dest="reload", action="store_false", help="do not reload the scene file on save")
    g = p.add_argument_group("detection")
    g.add_argument("--model", default=DEFAULT_MODEL, help=f"YOLO weights (default {DEFAULT_MODEL})")
    g.add_argument("--device", default="auto", help="auto, mps, cpu or cuda (default auto)")
    g.add_argument("--conf", type=float, default=0.35, help="confidence threshold (default 0.35)")
    g.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold (default 0.5)")
    g.add_argument("--imgsz", type=int, default=640, help="inference size in pixels (default 640)")
    g.add_argument("--classes", nargs="+", help="only detect these classes, e.g. --classes person car")
    g.add_argument("--no-detect", dest="detect", action="store_false", help="start with detection off")
    g.add_argument("--sync", action="store_true",
                   help="wait for detections on every frame (boxes never lag, frame rate follows the model)")
    g = p.add_argument_group("output")
    g.add_argument("--fullscreen", action="store_true", help="start in fullscreen")
    g.add_argument("--record", metavar="FILE.mp4", help="also write the annotated video to a file")
    g.add_argument("--headless", action="store_true", help="no window (use with --record/--frames)")
    g.add_argument("--frames", type=int, help="stop after this many frames")
    a = p.parse_args(argv)
    return DemoConfig(
        source=a.source, scene=a.scene, camera=a.camera, width=a.size[0], height=a.size[1], fps=a.fps,
        mirror=a.mirror, model=a.model, device=a.device, conf=a.conf, iou=a.iou, imgsz=a.imgsz, classes=a.classes,
        detect=a.detect, sync=a.sync, fullscreen=a.fullscreen, record=a.record, headless=a.headless,
        frames=a.frames, reload=a.reload, uploads=a.uploads or None,
    )


def main(argv=None) -> int:
    config = parse_args(argv)
    try:
        DemoApp(config).run()
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0
