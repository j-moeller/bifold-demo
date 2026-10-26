"""Running the models in a Docker container: the window side keeps the webcam and the screen, the container does
everything that touches model files.

The two talk over the container's stdin and stdout, nothing else: no port, and the container runs with
``--network none``, a read-only file system, the models folder mounted read-only, no capabilities and as an
unprivileged user. A checkpoint that carried hostile pickled code would find itself in a box that cannot reach the
network, cannot write to your files and cannot see the camera.

Each message is a 4-byte big-endian length, a JSON header of that length and, if the header says so, the raw
bytes of a BGR uint8 image of ``shape``:

* to the container: ``{"shape": [h, w, 3], "t": seconds, "reset": bool, "settings": {...}}`` + the frame;
* back: ``{"ok": true, "shape": ..., "faces": [...], "timings": {...}, "device": "cpu", "mivolo": bool}`` + the
  edited frame, or ``{"ok": false, "error": "..."}``.

The container answers ``{"ready": true, ...}`` once its models are loaded.
"""

from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import sys
from dataclasses import asdict
from typing import BinaryIO

import numpy as np

from . import models
from .state import AgeGender, FaceResult, FrameResult, Settings

IMAGE = "bifold-morph-worker"  # docker build -t bifold-morph-worker .
MAX_HEADER = 1 << 20
MAX_SIDE = 4096  # frames bigger than this are refused (a broken or hostile peer)


def send(out: BinaryIO, header: dict, image: np.ndarray | None = None) -> None:
    if image is not None:
        header = {**header, "shape": list(image.shape)}
    data = json.dumps(header).encode()
    out.write(struct.pack(">I", len(data)) + data)
    if image is not None:
        out.write(np.ascontiguousarray(image, np.uint8).tobytes())
    out.flush()


def _read_exact(inp: BinaryIO, n: int) -> bytes:
    chunks, left = [], n
    while left:
        chunk = inp.read(left)
        if not chunk:
            raise EOFError("the other side closed the pipe")
        chunks.append(chunk)
        left -= len(chunk)
    return b"".join(chunks)


def receive(inp: BinaryIO) -> tuple[dict, np.ndarray | None]:
    (n,) = struct.unpack(">I", _read_exact(inp, 4))
    if n > MAX_HEADER:
        raise ValueError(f"message header of {n} bytes")
    header = json.loads(_read_exact(inp, n))
    shape = header.get("shape")
    if shape is None:
        return header, None
    h, w, c = (int(v) for v in shape)
    if not (0 < h <= MAX_SIDE and 0 < w <= MAX_SIDE and c == 3):
        raise ValueError(f"refusing an image of shape {shape}")
    return header, np.frombuffer(_read_exact(inp, h * w * 3), np.uint8).reshape(h, w, 3)


def _face_json(face: FaceResult) -> dict:
    return {"track_id": face.track_id, "box": list(map(float, face.box)), "quad": np.asarray(face.quad).tolist(),
            "before": asdict(face.before) if face.before else None,
            "after": asdict(face.after) if face.after else None}


def _face_from(d: dict) -> FaceResult:
    def ag(v):
        return AgeGender(float(v["age"]), float(v["male"])) if v else None
    return FaceResult(int(d["track_id"]), tuple(float(v) for v in d["box"]), np.asarray(d["quad"], np.float64),
                      ag(d["before"]), ag(d["after"]))


# --- inside the container ---------------------------------------------------------------------------------------

def serve(device: str = "cpu", size: int = 256, max_faces: int = 2, readout: bool = True) -> None:
    """Load the editor and answer frames on stdin/stdout until stdin closes. Everything printed goes to stderr."""
    out, inp = sys.stdout.buffer, sys.stdin.buffer
    sys.stdout = sys.stderr  # stdout is the protocol from here on
    from .app import load_editor, pick_device

    dev = pick_device(device)
    editor = load_editor(dev, size=size, max_faces=max_faces, readout=readout, download=False,
                         progress=lambda m: print(m, file=sys.stderr))
    send(out, {"ready": True, "device": str(dev), "mivolo": editor.mivolo is not None})
    while True:
        try:
            header, frame = receive(inp)
        except EOFError:
            return
        if frame is None:
            continue
        try:
            if header.get("reset"):
                editor.tracker.reset()
            known = Settings.__dataclass_fields__
            settings = Settings(**{k: v for k, v in header.get("settings", {}).items() if k in known})
            result = editor.process(frame, settings, float(header.get("t", 0.0)))
            send(out, {"ok": True, "faces": [_face_json(f) for f in result.faces], "timings": result.timings},
                 result.image)
        except Exception as error:  # report and go on with the next frame
            send(out, {"ok": False, "error": f"{type(error).__name__}: {error}"})


# --- on the window side -----------------------------------------------------------------------------------------

def docker_command(image: str = IMAGE, folder=None, device: str = "cpu", size: int = 256, max_faces: int = 2,
                   readout: bool = True) -> list[str]:
    """The ``docker run`` line of the locked-down container that serves frames."""
    docker = shutil.which("docker") or "docker"
    cmd = [docker, "run", "--rm", "-i", "--network", "none", "--read-only", "--tmpfs", "/tmp:size=256m",
           "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--pids-limit", "256",
           "-v", f"{models.models_dir(folder).resolve()}:/models:ro", "-e", "BIFOLD_MORPH_MODELS=/models"]
    if hasattr(os, "getuid"):
        cmd += ["--user", f"{os.getuid()}:{os.getgid()}"]  # can read your models folder, owns nothing else
    if device.startswith("cuda"):
        cmd += ["--gpus", "all"]
    cmd += [image, "serve", "--device", device, "--size", str(size), "--max-faces", str(max_faces)]
    return cmd + ([] if readout else ["--no-mivolo"])


class _Tracker:
    def __init__(self):
        self.pending = False

    def reset(self) -> None:
        self.pending = True


class IsolatedEditor:
    """Stands in for :class:`~.editor.FaceEditor` in the app: :meth:`process` sends the frame into the
    container and returns its answer. Not thread-safe; the app's worker thread is its only user."""

    def __init__(self, command: list[str]):
        self.proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        try:
            ready, _ = receive(self.proc.stdout)
        except (EOFError, ValueError) as error:
            self.close()
            raise RuntimeError(f"the model container did not start ({error}); see its messages above. "
                               "Built it? docker build -t bifold-morph-worker .") from error
        if not ready.get("ready"):
            self.close()
            raise RuntimeError(f"unexpected first message from the container: {ready}")
        self.device = f"{ready.get('device', '?')} (docker)"
        self.mivolo = True if ready.get("mivolo") else None
        self.tracker = _Tracker()

    def process(self, frame: np.ndarray, settings: Settings, t: float | None = None) -> FrameResult:
        send(self.proc.stdin, {"t": t or 0.0, "reset": self.tracker.pending, "settings": asdict(settings)}, frame)
        self.tracker.pending = False
        header, image = receive(self.proc.stdout)
        if not header.get("ok"):
            raise RuntimeError(header.get("error", "the container failed"))
        if image is None or image.shape != frame.shape:
            raise RuntimeError("the container sent back no frame or one of another size")
        return FrameResult(image.copy(), [_face_from(f) for f in header.get("faces", [])],
                           {k: float(v) for k, v in header.get("timings", {}).items()})

    def close(self) -> None:
        if self.proc.stdin:
            try:
                self.proc.stdin.close()
            except OSError:
                pass
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
