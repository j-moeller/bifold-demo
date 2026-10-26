"""Changing how old and which gender the faces in a frame look.

For every face (up to ``max_faces``, the biggest):

1. YuNet finds it and its five landmarks; the tracker steadies them from frame to frame;
2. the head is cut out on the FFHQ template at 256 x 256 (:mod:`.warp`);
3. the student network (:mod:`.student`) draws it changed by the sliders' amounts, in one pass;
4. the result goes back into the frame along the same transform, through a soft-edged box.

The judges (FairFace, InsightFace), if loaded, read every face before and after.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from . import warp
from .distill import GENDER_RANGE, SIZE
from .faces import FaceDetector, FaceTracker
from .state import AgeGender, FaceResult, FrameResult, Settings


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


class FaceEditor:
    def __init__(self, student, detector: FaceDetector, device="cpu", max_faces: int = 2, judges=None,
                 half: bool = False):
        self.device = torch.device(device)
        self.dtype = torch.float16 if half and self.device.type != "cpu" else torch.float32
        self.student = student.to(self.device, self.dtype).eval()
        self.detector, self.judges, self.max_faces = detector, judges, max_faces
        self.tracker = FaceTracker()
        self._mask = warp.box_mask(SIZE, blur=0.35)

    @staticmethod
    def amounts(settings: Settings) -> tuple[float, float]:
        """The student's gender amount (gaps, + towards female) and years for the sliders."""
        return -settings.gender * GENDER_RANGE, settings.age

    @torch.inference_mode()
    def edit_crops(self, crops: np.ndarray, gender: float, years: float) -> np.ndarray:
        """Aligned BGR uint8 faces (B, 256, 256, 3) changed by ``gender`` gaps and ``years``: BGR float32."""
        x = torch.from_numpy(np.ascontiguousarray(crops[..., ::-1])).to(self.device)
        x = (x.permute(0, 3, 1, 2).to(self.dtype) / 127.5 - 1).contiguous()
        n = len(crops)
        out = self.student(x, torch.full((n,), gender, device=self.device), torch.full((n,), years, device=self.device))
        out = ((out.float() + 1) * 127.5).permute(0, 2, 3, 1).cpu().numpy()
        return np.ascontiguousarray(out[..., ::-1])

    @torch.inference_mode()
    def process(self, frame: np.ndarray, settings: Settings, t: float | None = None) -> FrameResult:
        """``frame`` (BGR uint8) with its faces changed as ``settings`` say; ``t`` (seconds) is when it was taken."""
        t = time.perf_counter() if t is None else t
        timings = {}
        start = time.perf_counter()
        tracks = self.tracker.update(self.detector.detect(frame)[:self.max_faces], t)
        timings["detect"] = time.perf_counter() - start
        out = frame.copy()
        faces = [FaceResult(tr.id, tr.face.box, warp.matrix(tr.landmarks, "ffhq_512", SIZE)) for tr in tracks]
        if faces and settings.edits:
            start = time.perf_counter()
            crops = np.stack([warp.warp(frame, f.quad, SIZE) for f in faces])
            edited = self.edit_crops(crops, *self.amounts(settings))
            _sync(self.device)
            timings["student"] = time.perf_counter() - start
            start = time.perf_counter()
            for face, crop in zip(faces, edited):
                warp.paste(out, crop, self._mask, face.quad)
            timings["paste"] = time.perf_counter() - start
        if faces and settings.readout and self.judges is not None:
            start = time.perf_counter()
            for face, tr in zip(faces, tracks):
                face.before = self.judges.read(frame, tr)
                face.after = self.judges.read(out, tr) if settings.edits else face.before
            timings["judges"] = time.perf_counter() - start
        return FrameResult(out, faces, timings)


class Judges:
    """Both classifiers' verdict, averaged, as the window shows it ("male 97%, 31 y")."""

    def __init__(self, fairface, genderage):
        self.fairface, self.genderage = fairface, genderage

    def read(self, image: np.ndarray, track) -> AgeGender:
        a = self.fairface.read(image, [track.landmarks])[0]
        b = self.genderage.read(image, [track.face.box])[0]
        female = (a.female + b.female) / 2
        return AgeGender(age=(a.age + b.age) / 2, male=1 - female)
