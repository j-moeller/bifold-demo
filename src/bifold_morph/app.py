"""The live demo: webcam in, faces edited as the sliders say, video out."""

from __future__ import annotations

import copy
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import models
from .sources import Source, open_source
from .state import AGE_END, FrameResult, Settings
from .ui import Panel, Slider, Window, describe_label, draw_face

TITLE = "bifold morph"
KEYS_HELP = "drag the sliders | Space original | E read-out | R reset | H help | Q quit"


def pick_device(name: str = "auto"):
    import torch

    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_editor(device, max_faces: int = 2, readout: bool = True, half: bool = False, student=None,
                folder=None, download: bool = True, progress=print):
    """The :class:`~.editor.FaceEditor`: the student network (``student``, default: the one that comes with the
    package), YuNet and, for the read-out, the two judges, from the models folder (fetched when missing)."""
    from . import student as student_mod
    from .editor import FaceEditor, Judges
    from .faces import FaceDetector
    from .nets import FairFace, GenderAge

    path = student or models.STUDENT
    if not path.is_file():
        raise FileNotFoundError(f"no student network at {path} (train one: bifold-morph train)")
    detector = FaceDetector(models.ensure(models.YUNET, folder, download, progress))
    judges = None
    if readout:
        if progress:
            progress("[morph] loading the judges (FairFace, InsightFace) for the read-out ...")
        judges = Judges(FairFace(models.ensure(models.FAIRFACE, folder, download, progress), device),
                        GenderAge(models.ensure(models.GENDERAGE, folder, download, progress), device))
    return FaceEditor(student_mod.load(path), detector, device, max_faces=max_faces, judges=judges, half=half)


@dataclass
class _Job:
    frame: np.ndarray
    timestamp: float
    settings: Settings
    reset: bool


class Worker:
    """Runs the editor on a thread of its own: :meth:`submit` hands it the newest frame (replacing one it has not
    started yet), :meth:`take` returns the newest result once."""

    def __init__(self, editor):
        self.editor = editor
        self.error: BaseException | None = None
        self.busy_seconds = 0.0  # how long the newest result took
        self._cond = threading.Condition()
        self._job: _Job | None = None
        self._result: FrameResult | None = None
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="editor", daemon=True)
        self._thread.start()

    def submit(self, frame: np.ndarray, timestamp: float, settings: Settings, reset: bool = False) -> None:
        with self._cond:
            reset = reset or (self._job is not None and self._job.reset)
            self._job = _Job(frame, timestamp, copy.copy(settings), reset)
            self._cond.notify_all()

    def take(self) -> FrameResult | None:
        with self._cond:
            result, self._result = self._result, None
        return result

    def _loop(self) -> None:
        while True:
            with self._cond:
                self._cond.wait_for(lambda: self._job is not None or not self._running)
                if not self._running:
                    return
                job, self._job = self._job, None
            start = time.perf_counter()
            try:
                if job.reset:
                    self.editor.tracker.reset()
                result = self.editor.process(job.frame, job.settings, job.timestamp)
            except Exception as error:  # shown in the window; the next frame tries again
                self.error = error
                continue
            self.error = None
            self.busy_seconds = time.perf_counter() - start
            with self._cond:
                self._result = result

    def stop(self) -> None:
        with self._cond:
            self._running = False
            self._cond.notify_all()
        self._thread.join(timeout=5.0)


def make_sliders() -> tuple[Slider, Slider]:
    def years(v: float) -> str:
        return "as is" if v == 0 else f"{v:+.0f} years"

    def gender(v: float) -> str:
        return "as is" if v == 0 else f"{abs(v):.0%} towards {'female' if v < 0 else 'male'}"

    return (Slider("Age", -AGE_END, AGE_END, "younger", "older", years, step=5.0),
            Slider("Gender", -1.0, 1.0, "female", "male", gender, step=0.1))


class App:
    def __init__(self, editor, source: Source, settings: Settings, fullscreen: bool = False,
                 screenshots: Path = Path("screenshots")):
        self.editor, self.source, self.settings = editor, source, settings
        self.age, self.gender = make_sliders()
        self.age.value, self.gender.value = settings.age, settings.gender
        self.panel = Panel([self.age, self.gender])
        self.window = Window(TITLE, on_mouse=self._on_mouse, fullscreen=fullscreen)
        self.worker = Worker(editor)
        self.screenshots = screenshots
        self.show_panel = True
        self.show_boxes = True
        self.message, self._message_until = "", 0.0
        self._video_height = 0
        self._shown: FrameResult | None = None
        self._fps = 0.0
        self._last_result_time: float | None = None
        self._reset_tracks = False

    def _on_mouse(self, event: int, x: int, y: int, flags: int) -> None:
        if self.show_panel and y >= self._video_height:
            self.panel.mouse(event, x, y - self._video_height, flags)
        elif event == cv2.EVENT_LBUTTONUP:
            self.panel.mouse(event, x, y, flags)  # a drag that ended over the video

    def say(self, text: str, seconds: float = 2.5) -> None:
        self.message, self._message_until = text, time.perf_counter() + seconds

    def _status(self) -> str:
        s = self.settings
        parts = [f"{self._fps:4.1f} fps", str(self.editor.device)]
        n = len(self._shown.faces) if self._shown else 0
        parts.append(f"{n} face{'s' if n != 1 else ''}")
        if not s.enabled:
            parts.append("ORIGINAL")
        if self.editor.judges is not None:
            parts.append(f"read-out {'on' if s.readout else 'off'}")
        if self.worker.error is not None:
            parts.append(f"error: {self.worker.error}")
        elif self.source.error:
            parts.append(self.source.error)
        if time.perf_counter() < self._message_until:
            parts.append(self.message)
        return "  |  ".join(parts)

    def _compose(self, result: FrameResult) -> np.ndarray:
        image = result.image.copy()
        edited = self.settings.edits
        for face in result.faces:
            label = describe_label(face.before, face.after, edited) if self.settings.readout else ""
            draw_face(image, face.box, label, self.show_boxes)
        self._video_height = image.shape[0]
        if not self.show_panel:
            return image
        strip = self.panel.draw(image.shape[1], self._status(), KEYS_HELP)
        return np.vstack([image, strip])

    def _key(self, key: int) -> bool:
        """Handle a key; ``False`` to quit."""
        s = self.settings
        if key in (ord("q"), 27):
            return False
        if self.panel.key(key):
            pass
        elif key == ord(" "):
            s.enabled = not s.enabled
            self.say("showing the original" if not s.enabled else "editing again")
        elif key == ord("r"):
            self.age.set(0.0)
            self.gender.set(0.0)
        elif key in (ord("1"), ord("2")):
            self.age.nudge(-1 if key == ord("1") else 1)
        elif key in (ord("3"), ord("4")):
            self.gender.nudge(-1 if key == ord("3") else 1)
        elif key == ord("e"):
            s.readout = not s.readout
        elif key == ord("b"):
            self.show_boxes = not self.show_boxes
        elif key == ord("h"):
            self.show_panel = not self.show_panel
        elif key == ord("f"):
            self.window.toggle_fullscreen()
        elif key == ord("m"):
            self.source.mirror = not self.source.mirror
            self._reset_tracks = True
        elif key == ord("p") and self._shown is not None:
            self.screenshots.mkdir(parents=True, exist_ok=True)
            path = self.screenshots / time.strftime("morph_%Y%m%d_%H%M%S.png")
            cv2.imwrite(str(path), self._compose(self._shown))
            self.say(f"saved {path}")
        return True

    def run(self) -> None:
        try:
            while self.window.is_open():
                self.settings.age, self.settings.gender = self.age.value, self.gender.value
                frame = self.source.read(timeout=0.03)
                if frame is not None:
                    self.worker.submit(frame.image, frame.timestamp, self.settings, reset=self._reset_tracks)
                    self._reset_tracks = False
                result = self.worker.take()
                if result is not None:
                    now = time.perf_counter()
                    if self._last_result_time is not None:
                        rate = 1.0 / max(now - self._last_result_time, 1e-6)
                        self._fps = rate if self._fps == 0 else 0.9 * self._fps + 0.1 * rate
                    self._last_result_time = now
                    self._shown = result
                if self._shown is not None:
                    self.window.show(self._compose(self._shown))
                key = self.window.poll_key(1)
                if key is not None and not self._key(key):
                    break
        finally:
            self.worker.stop()
            self.source.stop()
            self.window.close()


def run(source: str = "0", device: str = "auto", max_faces: int = 2, readout: bool = True, half: bool = False,
        width: int = 1280, height: int = 720, mirror: bool | None = None, fullscreen: bool = False,
        age: float = 0.0, gender: float = 0.0, student=None) -> None:
    editor = load_editor(pick_device(device), max_faces=max_faces, readout=readout, half=half, student=student)
    src = open_source(source, width, height, mirror)
    settings = Settings(age=age, gender=gender, readout=readout)
    print(f"[morph] running on {editor.device}; {KEYS_HELP}")
    App(editor, src, settings, fullscreen=fullscreen).run()


def edit_file(path, output, age: float, gender: float, device: str = "auto", readout: bool = True,
              student=None) -> FrameResult:
    """Change the faces in one picture and save it to ``output``."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"cannot read image {path}")
    editor = load_editor(pick_device(device), max_faces=8, readout=readout, student=student)
    result = editor.process(image, Settings(age=age, gender=gender, readout=readout))
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), result.image):
        raise OSError(f"cannot write {output}")
    return result
