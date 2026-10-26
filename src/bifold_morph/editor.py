"""Changing how old and which gender the faces in a frame look.

For every face (up to ``max_faces``, the biggest):

1. YuNet finds it and its five landmarks; the tracker steadies them from frame to frame;
2. the FFHQ square round the face is cut out at 256 x 256 (:mod:`.align`);
3. e4e turns it into a StyleGAN2 code, ``w+`` (18 x 512), smoothed over the last frames;
4. the code moves along the age and gender directions (:mod:`.directions`) as far as the sliders say;
5. StyleGAN2 draws the moved code, and the result goes back into the frame along the same square:
   * by default only the *change* goes back: the generator draws the face as encoded and as edited, and the
     difference between the two is added to the real pixels. What e4e cannot capture (the exact identity, skin,
     glasses, background) stays as the camera saw it, and with the sliders at 0 the picture is untouched;
   * in "replace" mode the generated face itself replaces the real one inside a soft oval.

MiVOLO, if loaded, reads every face before and after the edit.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from . import align
from .directions import Directions
from .e4e import E4E, INPUT_SIZE, to_input
from .faces import FaceDetector, FaceTracker
from .mivolo import MiVOLO, crop_box
from .state import FaceResult, FrameResult, Settings

CODE_SMOOTHING = 0.5  # weight of the newest frame's code in the smoothed code of a face
CODE_KEEP = 0.3  # seconds: an older smoothed code starts afresh


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


class FaceEditor:
    def __init__(self, e4e: E4E, directions: Directions, detector: FaceDetector, device="cpu", size: int = 256,
                 max_faces: int = 2, mivolo: MiVOLO | None = None, half: bool = False):
        self.device = torch.device(device)
        self.dtype = torch.float16 if half else torch.float32
        self.e4e = e4e.to(self.device, self.dtype).eval()
        self.directions, self.detector, self.mivolo = directions, detector, mivolo
        self.size, self.max_faces = size, max_faces
        self.tracker = FaceTracker()
        self._masks = {False: align.border_mask(size), True: align.face_mask(size)}
        self._offset_key = None
        self._offset = None

    def offset(self, settings: Settings) -> torch.Tensor:
        key = (settings.age, settings.gender)
        if key != self._offset_key:
            delta = self.directions.offset(settings.age, settings.gender)
            self._offset = torch.as_tensor(delta, dtype=self.dtype, device=self.device)
            self._offset_key = key
        return self._offset

    @torch.inference_mode()
    def process(self, frame: np.ndarray, settings: Settings, t: float | None = None) -> FrameResult:
        """``frame`` (BGR uint8) with its faces edited as ``settings`` say; ``t`` (seconds) is when it was taken."""
        t = time.perf_counter() if t is None else t
        timings = {}
        start = time.perf_counter()
        tracks = self.tracker.update(self.detector.detect(frame)[:self.max_faces], t)
        timings["detect"] = time.perf_counter() - start
        out = frame.copy()
        faces = [FaceResult(tr.id, tr.face.box, align.ffhq_quad(tr.landmarks)) for tr in tracks]
        if faces and settings.edits:
            self._edit(frame, out, tracks, faces, settings, t, timings)
        if faces and settings.readout and self.mivolo is not None:
            start = time.perf_counter()
            self._read(frame, out if settings.edits else None, faces)
            timings["mivolo"] = time.perf_counter() - start
        return FrameResult(out, faces, timings)

    def _edit(self, frame, out, tracks, faces, settings: Settings, t: float, timings: dict) -> None:
        start = time.perf_counter()
        crops = [align.crop(frame, f.quad, INPUT_SIZE)[..., ::-1] for f in faces]
        codes = self.e4e.encode(to_input(crops, self.device, self.dtype))
        for i, track in enumerate(tracks):
            prev, when = track.state.get("codes"), track.state.get("t", -1e9)
            if prev is not None and t - when <= CODE_KEEP:
                codes[i] = prev + CODE_SMOOTHING * (codes[i] - prev)
            track.state.update(codes=codes[i].clone(), t=t)
        _sync(self.device)
        timings["encode"] = time.perf_counter() - start

        start = time.perf_counter()
        edited = codes + self.offset(settings)
        g = self.e4e.generator
        if settings.replace:
            layers = (g.synthesis(edited, self.size).float().clamp(-1, 1) + 1) * 127.5
        else:
            both = g.synthesis(torch.cat([codes, edited]), self.size).float().clamp(-1, 1)
            layers = (both[len(faces):] - both[:len(faces)]) * 127.5
        layers = layers.permute(0, 2, 3, 1).flip(-1).cpu().numpy()  # RGB -> BGR
        timings["generate"] = time.perf_counter() - start

        start = time.perf_counter()
        mask = self._masks[settings.replace]
        for face, layer in zip(faces, layers):
            align.paste(out, layer, mask, face.quad, settings.replace)
        timings["paste"] = time.perf_counter() - start

    def _read(self, frame, edited, faces) -> None:
        crops, where = [], []
        for i, face in enumerate(faces):
            for which, image in (("before", frame), ("after", edited)):
                c = crop_box(image, face.box) if image is not None else None
                if c is not None:
                    crops.append(c)
                    where.append((i, which))
        for (i, which), result in zip(where, self.mivolo.predict(crops) if crops else []):
            setattr(faces[i], which, result)
        if edited is None:
            for face in faces:
                face.after = face.before
