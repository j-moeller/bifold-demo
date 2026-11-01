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
from .display.overlay import draw_faces
from .display.window import TAB
from .loader import SceneReloader, load_scene
from .scene import Scene
from .selection import Selection, person_name
from .sources import FixedStepClock, Frame, SceneSource, SimulationClock, SourceSwitcher, WebcamSource

HELP = ("S source | D detect | N names | W windows | Space pause | R reset | [ ] confidence | Y / click people | "
        "C creator | F fullscreen | P screenshot | H hide | Q quit")
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
    items: str | None = "items"  # looks for the hats, glasses, signs and t-shirts (see characters.accessories)
    faces: str = "faces"  # photos of the people the face recognition knows (see recognition.gallery)
    face_profile: str | None = None  # the face models: best, licensed, commercial (None: $BIFOLD_FACE_PROFILE / best)
    recognize: bool = False  # start with the face recognition on (N toggles it)
    cast: bool = True  # put the known people's faces on 3D people (when the faces folder has photos)
    created: str | None = "created"  # characters made in the character creator, shown in the fair (None: not)
    windows: bool = False  # start with the virtual sensor's windows shown (W toggles them; see bifold_demo.sensor)
    max_windows: int = 4  # windows a frame
    sensor: tuple[int, int] | None = None  # the declared high-resolution sensor (None: six times the frame)
    camera_preset: str = "ideal"  # the camera model of the windows (sensor.camera_model.PRESETS)
    webcam_capture: tuple[int, int] | None = None  # capture the webcam at this size, windows cut from it
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
        self.window_service = None  # the virtual sensor's windows (W), made when first shown
        self.windows_on = False
        self._window_tags = None
        self.switcher = SourceSwitcher(
            {
                "scene": lambda: self.scene_source,
                "webcam": lambda: WebcamSource(config.camera, config.width, config.height, config.fps, config.mirror,
                                               capture=config.webcam_capture,
                                               windows=self.window_service if self.windows_on else None),
            },
            initial=config.source,
        )
        self.detector = None
        self.worker: AsyncDetector | None = None
        self.detect_enabled = config.detect
        self.gallery = None  # the known people (recognition.gallery), loaded when first needed
        self.cast = None  # their faces on the 3D people (recognition.cast)
        self.recognizer = None  # the face recognition (N), loaded when first switched on
        self.face_worker: AsyncDetector | None = None
        self.recognize_enabled = False
        self._face_error: str | None = None
        self._faces_submitted: tuple[str, int] | None = None
        self._last_faces = None
        self.sync = config.sync or config.headless  # headless renders faster than real time: keep boxes in step
        self.show_hud = True
        self.display_rate = _Rate()
        self._message: tuple[str, float, bool] | None = None  # (text, expiry time, is_error)
        self._created_notes: tuple[str, list[str]] | None = None  # the created characters' notice shown last
        self._detector_error: str | None = None
        self._last_submitted: tuple[str, int] | None = None
        self._writer = None
        self._record_size: tuple[int, int] | None = None
        self._last_canvas = None
        self._last_reload_check = 0.0
        self._running = True
        from .characters.accessories import Wardrobe

        self.selection = Selection(wardrobe=Wardrobe(config.items))  # Y: pick a person, then yoga poses and items
        self.created = None  # the created characters in the scene (scenes.created_cast), when there is a folder
        if config.created:
            from .characters.created import CharacterBook
            from .scenes.created_cast import CreatedCast

            self.created = CreatedCast(CharacterBook(config.created), wardrobe=self.selection.wardrobe)
        self.creator = None  # the character creator (C), made when first opened
        self.creator_view = None

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

    def _load_gallery(self):
        """The known people (loading the face models, downloading them the first time), or ``None`` on failure (the
        video goes on; the error is shown)."""
        if self.gallery is not None:
            return self.gallery
        from .recognition.gallery import Gallery

        try:
            gallery = Gallery(self.config.faces, profile=self.config.face_profile)
            engine = gallery.engine  # load the models now, not on the first frame
            print(f"[faces] models: {engine.describe()}")
            for note in engine.notes:
                print(f"[faces] {note}")
            gallery.refresh(force=True)
        except Exception as exc:  # noqa: BLE001 - e.g. the models cannot be downloaded
            self._face_error = f"face recognition unavailable: {type(exc).__name__}: {exc}"
            print(f"[faces] {self._face_error}")
            return None
        self._face_error = None
        people = gallery.people()
        print(f"[faces] {len(people)} known: {', '.join(people) or f'nobody yet (photos go in {gallery.folder}/)'}")
        self.gallery = gallery
        return gallery

    def _load_recognizer(self) -> bool:
        """The face recognition (``N``); on failure keep the video running and show the error (``N`` retries)."""
        from .recognition.recognizer import FaceRecognizer

        gallery = self._load_gallery()
        if gallery is None:
            return False
        self.recognizer = FaceRecognizer(gallery)
        if not self.sync:
            self.face_worker = AsyncDetector(self.recognizer).start()
        return True

    def _load_cast(self) -> None:
        """Put the known people's faces on the 3D people, when the faces folder has photos (see recognition.cast)."""
        from .recognition.gallery import has_photos

        if not self.config.cast or not has_photos(self.config.faces):
            return
        gallery = self._load_gallery()
        if gallery is not None:
            from .recognition.cast import FaceCast

            self.cast = FaceCast(gallery)
            self.selection.cast = self.cast

    def _update_created(self, wait: bool = False) -> None:
        """Keep the created characters in the scene (new, changed and deleted ones; see scenes.created_cast), and say
        what changed in one notice (several often join at once, or a frame or two apart: a notice still on the
        screen takes the new ones in)."""
        if self.created is not None:
            from .scenes.created_cast import notice

            notes = self.created.update(self.scene_source.scene, wait=wait, view=self.scene_source.renderer)
            for note in notes:  # every one in the console; on the screen together
                print(f"[demo] {note}")
            if notes:
                shown = self._created_notes
                if shown is not None and self._message is not None and self._message[0] == shown[0] \
                        and time.perf_counter() < self._message[1]:
                    notes = shown[1] + notes
                text = notice(notes)
                self.notify(text, 3.0 + 0.5 * min(len(notes) - 1, 4),
                            error=any(note.startswith("cannot") for note in notes), echo=False)
                self._created_notes = (text, notes)

    def _open_creator(self) -> None:
        """Show the character creator (``C``); in selection mode for the selected person ("edit as a new
        character")."""
        if self.creator is None:
            from .creator import Creator, KnownFaces
            from .display.creator_view import CreatorView

            book = self.created.book if self.created is not None else None
            library = self.created.library if self.created is not None else None
            self.creator = Creator(book, library=library, wardrobe=self.selection.wardrobe,
                                   faces=KnownFaces(self.config.faces, lambda: self.gallery))
            self.creator_view = CreatorView(self.creator)
        person = self.selection.selected if self.selection.active else None
        face = self.cast.wears(person) if person is not None and self.cast is not None else None
        self.selection.leave()
        self.notify(self.creator.open(person, face), 4.0)

    def _creating(self) -> bool:
        return self.creator is not None and self.creator.active

    # -- the virtual sensor's windows (W) -------------------------------------------------------------------------

    def _show_windows(self, on: bool) -> str:
        """Show the windows of the virtual high-resolution sensor (or stop): high-resolution views of the heads of
        the people the detector finds, delivered with the frames (see bifold_demo.sensor). Until the person tracker
        asks for them, they follow the person boxes of the detections."""
        if on and self.window_service is None:
            from .display.sensor_view import BoxTags
            from .sensor import Sensor, WindowService

            cfg = self.config
            try:  # (the command line checks these; a DemoConfig made in code may not)
                sensor = Sensor(*cfg.sensor, (cfg.width, cfg.height)) if cfg.sensor else None
                self.window_service = WindowService(sensor, cfg.max_windows, preset=cfg.camera_preset)
            except ValueError as exc:
                return f"windows unavailable: {exc}"
            self._window_tags = BoxTags()
        self.windows_on = on
        for source in {id(s): s for s in (self.scene_source, *self.switcher.started())}.values():
            if hasattr(source, "windows"):
                source.windows = self.window_service if on else None
        if not on:
            if self.window_service is not None:
                self.window_service.clear()
            return "windows off"
        from .sensor import DEFAULT_SCALE, Sensor

        service, cfg = self.window_service, self.config
        preset = service.preset
        if preset.lines is not None:
            sensor = Sensor.with_lines(preset.lines, (cfg.width, cfg.height))
        else:
            sensor = service.sensor or Sensor.scaled(DEFAULT_SCALE, (cfg.width, cfg.height))
        need = "" if self.detect_enabled and self.detector is not None else " (they follow the people found: D)"
        return f"windows on: up to {cfg.max_windows} a frame, {sensor} sensor, {preset.name} camera{need}"

    def _request_windows(self, frame: Frame, result: DetectionResult | None) -> None:
        """Ask for windows on the heads of the people in ``result`` (the latest detections of ``frame``'s source); none
        without current detections (detection off, none yet for this source, or too old)."""
        if result is None or result.source != frame.source:
            self.window_service.request([])
            return
        from .sensor import WindowRequest

        sx, sy = frame.size[0] / result.frame_size[0], frame.size[1] / result.frame_size[1]
        boxes = [(d.box[0] * sx, d.box[1] * sy, d.box[2] * sx, d.box[3] * sy) for d in result.detections
                 if d.label == "person" and d.box[2] > d.box[0] and d.box[3] > d.box[1]]
        tags = self._window_tags.assign(boxes)
        self.window_service.request([WindowRequest(tag, box=box, frame=result.frame_index)
                                     for tag, box in zip(tags, boxes)])

    def _faces_for(self, frame: Frame):
        """The recognised faces of ``frame`` (or of a frame just before it), or ``None``; kept for the status panel."""
        self._last_faces = self._recognized(frame)
        return self._last_faces

    def _recognized(self, frame: Frame):
        if not self.recognize_enabled or self.recognizer is None:
            return None
        if self.face_worker is None:
            return self.recognizer.detect_frame(frame)
        key = (frame.source, frame.index)
        if key != self._faces_submitted:
            self.face_worker.submit(frame)
            self._faces_submitted = key
        result = self.face_worker.latest()
        if result is None or result.source != frame.source or time.perf_counter() - result.timestamp > STALE_AFTER:
            return None
        return result

    def notify(self, message: str, seconds: float = 2.5, error: bool = False, echo: bool = True) -> None:
        """Show ``message`` on the screen for ``seconds`` (and, with ``echo``, in the console)."""
        if echo:
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
        if cfg.recognize:
            self.recognize_enabled = self._load_recognizer()
        if cfg.windows:
            print(f"[demo] {self._show_windows(True)}")
        self._load_cast()
        self._update_created(wait=True)  # the first frame has them
        window = None if cfg.headless else Window(cfg.title, cfg.fullscreen)
        if window:
            print(f"[demo] controls: {HELP}")
        count = 0
        try:
            while self._running:
                frame = self.switcher.active.read(timeout=0.5)
                if frame is not None:
                    if frame.source == "scene":
                        self._update_created()
                        message = self.selection.update(self.scene_source.scene, frame.size)
                        if self.cast is not None:
                            for note in self.cast.update(self.scene_source.scene):
                                self.notify(note, 3.0)
                        if message:
                            self.notify(message)
                    if self.creator is not None:  # (also closed: a save asked for just before goes on)
                        for note in self.creator.update():
                            self.notify(note, 4.0, error=note.startswith("cannot"))
                    result = self._detections_for(frame)
                    if self.windows_on:  # for the frames to come: windows show their own moment
                        self._request_windows(frame, result)
                    self._last_canvas = self._compose(frame, result, self._faces_for(frame))
                    self.display_rate.tick(time.perf_counter())
                    if cfg.record:
                        self._record(self._last_canvas)
                    count += 1
                if window and self._last_canvas is not None:
                    window.show(self._last_canvas)
                    key = window.poll_key(1)
                    if key is not None:
                        self._handle_key(key, window)
                    for click in window.poll_clicks():
                        self._handle_click(click)
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
            if self.face_worker:
                self.face_worker.stop()
            if self.cast is not None:
                self.cast.close()
            if self.created is not None:
                self.created.close()
            if self.creator is not None:
                self.creator.close()
                self.creator_view.release()
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

    def _compose(self, frame: Frame, result: DetectionResult | None, faces=None):
        if self._creating():
            canvas = self.creator_view.draw(frame.image)
            self._draw_message(canvas)
            return canvas
        canvas = frame.image.copy()  # the detector may still be reading the original
        if result is not None:
            draw_detections(canvas, result.detections, None if result.frame_size == frame.size else result.frame_size)
        if faces is not None:
            draw_faces(canvas, faces.faces, None if faces.frame_size == frame.size else faces.frame_size)
        selecting = self.selection.active and frame.source == "scene"
        if selecting and self.selection.box is not None:
            t = self.scene_source.scene.time
            draw_selection(canvas, self.selection.box, person_name(self.selection.selected),
                           self.selection.detail(self.selection.selected, t))
        if self.windows_on and frame.windows:
            from .display.sensor_view import draw_windows

            draw_windows(canvas, frame.windows)
        if self.show_hud:
            draw_panel(canvas, self._status_lines(frame, result))
            if not self.config.headless:
                draw_panel(canvas, self.selection.help_lines() if selecting else [HELP], corner="bottom-left")
        self._draw_message(canvas)
        return canvas

    def _draw_message(self, canvas) -> None:
        """The message of the moment as a banner (else the error of a part that does not work, if any)."""
        if self._message and time.perf_counter() < self._message[1]:
            text, _expiry, is_error = self._message
            draw_banner(canvas, text, color=(40, 40, 200) if is_error else (60, 140, 40))
        else:
            error = (getattr(self.switcher.active, "error", None) or self._detector_error
                     or (self.worker.error if self.worker else None) or self._face_error
                     or (self.face_worker.error if self.face_worker else None))
            if error:
                draw_banner(canvas, error)

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
        if self.recognize_enabled and self.recognizer is not None:
            known = self.recognizer.gallery.people()
            faces = self._last_faces
            line = f"faces: {len(known)} known"
            if faces is not None and faces.source == frame.source:
                names = sorted(f.name for f in faces.faces if f.name)
                strangers = sum(f.name is None for f in faces.faces)
                line += f", in the picture: {', '.join(names) or 'nobody known'}"
                line += f" (+{strangers} unknown)" if strangers else ""
            lines.append(line)
        if self.windows_on and self.window_service is not None:
            asked = len(self.window_service.requests)
            spent = sum(w.render_ms + w.camera_ms for w in frame.windows)
            lines.append(f"windows: {len(frame.windows)} of {asked} asked ({self.window_service.preset.name}), "
                         f"{spent:.0f} ms")
        if frame.source == "scene":
            created = self.created.status() if self.created is not None else None
            lines += ([created] if created else []) + self.selection.status_lines(self.scene_source.scene.time)
        return lines

    def _handle_key(self, key: int, window: Window) -> None:
        clock = self.scene_source.clock
        if self._creating():  # it takes the keys while it is shown (Esc closes it)
            handled, message = self.creator.handle_key(key, getattr(window, "typed", None))
            if message:
                self.notify(message, 3.0, error=message.startswith("cannot"))
            if handled:
                return
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
        elif key == ord("n"):
            if self.recognizer is None and not self._load_recognizer():
                self.notify(self._face_error, 5, error=True)
                return
            self.recognize_enabled = not self.recognize_enabled
            self.notify(f"face recognition {'on' if self.recognize_enabled else 'off'}")
        elif key == ord("w"):
            message = self._show_windows(not self.windows_on)
            self.notify(message, 3.0, error=message.startswith("windows unavailable"))
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
        elif key == ord("c"):
            self._open_creator()
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

    def _handle_click(self, point) -> None:
        """A left click at ``point`` (pixels of the picture): a choice on the creator's screen when it is shown, else
        select the 3D person there (entering selection mode)."""
        if self._creating():
            message = self.creator_view.click(point)
            if message:
                self.notify(message, 3.0, error=message.startswith("cannot"))
            return
        if self.switcher.active_name != "scene":
            return
        source = self.scene_source
        if not self.selection.active:
            message = self.selection.enter(source.scene, (source.width, source.height))
            if not self.selection.active:
                self.notify(message, 3.0, error=True)
                return
        self.notify(self.selection.select_at(point), 3.0)

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
            self._update_created(wait=True)  # the created characters again, from its first frame
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


def _preset(text: str) -> str:
    from .sensor import camera_preset

    try:
        return camera_preset(text).name
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


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
    g.add_argument("--faces", default="faces",
                   help="folder of photos of the people to recognise, a folder per person (default: faces)")
    g.add_argument("--face-profile", default=None, choices=("best", "licensed", "commercial"),
                   help="the face models: best (research-only weights, the default), licensed (InsightFace's, "
                        "commercial with a licence from InsightFace) or commercial (default: $BIFOLD_FACE_PROFILE)")
    g.add_argument("--recognize", action="store_true", help="start with the face recognition on (N toggles it)")
    g.add_argument("--no-cast", dest="cast", action="store_false",
                   help="do not put the faces of the known people on the 3D people")
    g.add_argument("--created", default="created",
                   help="folder of the characters made in the character creator, shown in the fair (default: created; "
                        "'' for none)")
    g.add_argument("--items", default="items",
                   help="folder of looks for the hats, glasses, signs and t-shirts of selection mode, e.g. "
                        "items/sign_logo.png ('' for the built-in ones only)")
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
    g = p.add_argument_group("high-resolution windows (the virtual sensor; W shows them)")
    g.add_argument("--windows", action="store_true",
                   help="start with the windows on the heads of the people found shown")
    g.add_argument("--max-windows", type=int, default=4, metavar="N", help="windows a frame (default 4)")
    g.add_argument("--sensor", type=_size, metavar="WxH",
                   help="the declared high-resolution sensor, the frame's shape (default: six times the frame, "
                        "7680x4320 for 1280x720)")
    g.add_argument("--camera-preset", type=_preset, default="ideal", metavar="NAME",
                   help="the windows' camera model: ideal, good_8k, good_4k or budget_1080p (default ideal)")
    g.add_argument("--webcam-capture", type=_size, metavar="WxH",
                   help="capture the webcam at this size (e.g. 3840x2160), detect on a copy of the frame size and "
                        "cut the windows from the full picture")
    g = p.add_argument_group("output")
    g.add_argument("--fullscreen", action="store_true", help="start in fullscreen")
    g.add_argument("--record", metavar="FILE.mp4", help="also write the annotated video to a file")
    g.add_argument("--headless", action="store_true", help="no window (use with --record/--frames)")
    g.add_argument("--frames", type=int, help="stop after this many frames")
    a = p.parse_args(argv)
    if a.sensor:  # the sensor's whole frame is the picture's: it must have the frame's shape
        from .sensor import Sensor

        try:
            Sensor(*a.sensor, a.size)
        except ValueError:
            fit = Sensor.with_lines(a.sensor[1], a.size)
            p.error(f"--sensor {a.sensor[0]}x{a.sensor[1]} does not have the frame's shape ({a.size[0]}x{a.size[1]}); "
                    f"for example --sensor {fit}")
    if a.max_windows < 1:
        p.error("--max-windows must be at least 1")
    return DemoConfig(
        source=a.source, scene=a.scene, camera=a.camera, width=a.size[0], height=a.size[1], fps=a.fps,
        mirror=a.mirror, model=a.model, device=a.device, conf=a.conf, iou=a.iou, imgsz=a.imgsz, classes=a.classes,
        detect=a.detect, sync=a.sync, fullscreen=a.fullscreen, record=a.record, headless=a.headless,
        frames=a.frames, reload=a.reload, uploads=a.uploads or None, items=a.items or None, faces=a.faces,
        face_profile=a.face_profile, recognize=a.recognize, cast=a.cast, created=a.created or None,
        windows=a.windows, max_windows=a.max_windows, sensor=a.sensor, camera_preset=a.camera_preset,
        webcam_capture=a.webcam_capture,
    )


def main(argv=None) -> int:
    config = parse_args(argv)
    try:
        DemoApp(config).run()
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0
