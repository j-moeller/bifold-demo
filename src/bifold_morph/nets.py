"""The networks, each behind a small wrapper that takes a BGR frame and a face's five landmarks.

* :class:`ArcFace` (InsightFace ``w600k_r50``): a face's identity as a unit 512-vector;
* :class:`HyperSwap` (FaceFusion, 2025): draws the face an identity vector describes, with the pose, expression,
  lighting and hair of the face in the frame, and a mask of where it drew;
* :class:`StyleGANEX` (Yang et al., ICCV 2023): StyleGAN2 for faces with their surroundings: encodes the face
  into StyleGAN's latent codes, moves them by an offset (age, gender, ...) and draws the result;
* :class:`FRAN` (Disney Research's "Production-Ready Face Re-Aging", 2022, re-implemented by ry-lu): makes a face
  look a given age;
* :class:`FairFace` and :class:`GenderAge` (InsightFace): two independent judges of gender and age.

All of them run in PyTorch (see :mod:`.onnx_torch`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from . import warp as align
from .onnx_torch import constant, load, rewire_input

FAIRFACE_AGES = [(0, 2), (3, 9), (10, 19), (20, 29), (30, 39), (40, 49), (50, 59), (60, 69), (70, 100)]


def _to_tensor(crop_bgr: np.ndarray, device, mean=0.0, std=1.0, scale=1 / 255.0) -> torch.Tensor:
    """BGR uint8 (H, W, 3) or (B, H, W, 3) -> RGB float (B, 3, H, W), ``(x * scale - mean) / std``."""
    x = np.asarray(crop_bgr)
    if x.ndim == 3:
        x = x[None]
    x = x[..., ::-1].astype(np.float32) * scale
    x = (x - np.asarray(mean, np.float32)) / np.asarray(std, np.float32)
    return torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2))).to(device)


def _to_bgr(images: torch.Tensor) -> np.ndarray:
    """RGB float (B, 3, H, W) in [0, 1] -> BGR float32 (B, H, W, 3) in [0, 255]."""
    x = images.detach().float().clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
    return np.ascontiguousarray(x[..., ::-1] * 255.0)


class ArcFace:
    template, size = "arcface_112_v2", 112

    def __init__(self, path, device="cpu"):
        self.device = torch.device(device)
        self.net = load(path, self.device)

    @torch.inference_mode()
    def embed(self, frame: np.ndarray, landmarks_list) -> np.ndarray:
        """Unit identity vectors (n, 512) for the faces with these landmarks."""
        crops = [align.warp(frame, align.matrix(lm, self.template, self.size), self.size) for lm in landmarks_list]
        if not crops:
            return np.zeros((0, 512), np.float32)
        x = _to_tensor(np.stack(crops), self.device, mean=1.0, std=1.0, scale=1 / 127.5)  # [-1, 1]
        e = self.net(x).float().cpu().numpy()
        return e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)


class HyperSwap:
    """``swap(frame, landmarks, identity)``: the face redrawn with ``identity`` (a unit ArcFace vector)."""

    template, size = "arcface_128", 256

    def __init__(self, path, device="cpu"):
        self.device = torch.device(device)
        self.net = load(path, self.device)

    @torch.inference_mode()
    def swap(self, frame: np.ndarray, landmarks, identity: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The redrawn crop (S, S, 3) BGR float, its mask (S, S) in [0, 1] and the crop's matrix."""
        m = align.matrix(landmarks, self.template, self.size)
        crop = align.warp(frame, m, self.size)
        x = _to_tensor(crop, self.device, mean=0.5, std=0.5)
        src = torch.as_tensor(np.asarray(identity, np.float32).reshape(1, 512), device=self.device)
        image, mask = self.net(src, x)
        image = _to_bgr(image.float() * 0.5 + 0.5)[0]
        return image, mask.float()[0, 0].clamp(0, 1).cpu().numpy(), m


def scale_landmarks(landmarks, scale: float) -> np.ndarray:
    """The five landmarks scaled by ``scale`` round the nose tip."""
    lm = np.asarray(landmarks, np.float64).reshape(5, 2)
    return (lm - lm[2]) * scale + lm[2]


def match_color(reference: np.ndarray, image: np.ndarray) -> np.ndarray:
    """``image`` with the low-frequency colour of ``reference`` (same size, uint8), at three coarse scales and once
    at full size (FaceFusion's ``match_frame_color``): undoes the colour drift of a generator."""
    import cv2

    def equalize(src, dst, size):
        a = cv2.resize(src, size, interpolation=cv2.INTER_AREA).astype(np.float32)
        b = cv2.resize(dst, size, interpolation=cv2.INTER_AREA).astype(np.float32)
        diff = cv2.resize(a - b, dst.shape[:2][::-1], interpolation=cv2.INTER_CUBIC)
        return np.clip(dst.astype(np.float32) + diff, 0, 255).astype(np.uint8)

    for side in np.linspace(16, image.shape[0], 3, endpoint=False):
        n = max(1, int(side))
        reference = equalize(reference, image, (n, n))
    return equalize(reference, image, image.shape[:2][::-1])


class StyleGANEX:
    """StyleGANEX as FaceFusion exported it for ageing, rewired: the graph's built-in edit, ``codes - direction *
    AGE`` with ``AGE`` a (1, 18, 512) latent vector, becomes ``codes + offset`` with any ``offset``, and the encoded
    ``codes`` (1, 18, 512) come out as well. :attr:`age_vector` is the built-in direction (towards younger)."""

    templates = {"face": ("ffhq_512", 256), "with_background": ("styleganex_384", 384)}
    out_size = 1024  # the 1536 px output is shrunk to this for pasting
    CODES, EDIT, AGE = "/PSPEX/Add_output_0", "/Mul_output_0", "/Constant_output_0"

    def __init__(self, path, device="cpu"):
        self.device = torch.device(device)
        self.age_vector = None

        def edit(model):
            import onnx

            self.age_vector = constant(model, self.AGE).astype(np.float32).reshape(18, 512)
            rewire_input(model, self.EDIT, "offset", [1, 18, 512])
            model.graph.output.append(onnx.helper.make_tensor_value_info(self.CODES, onnx.TensorProto.FLOAT, None))

        self.net = load(path, self.device, edit=edit)

    @torch.inference_mode()
    def edit(self, frame: np.ndarray, landmarks, offset: np.ndarray):
        """The edited head with its surroundings (out_size x out_size, BGR uint8, colours matched), its matrix
        from the frame, and the face's codes (18, 512)."""
        import cv2

        m_face = align.matrix(landmarks, "ffhq_512", 256)
        face = align.warp(frame, m_face, 256)
        m_bg = align.matrix(scale_landmarks(landmarks, 0.875), "styleganex_384", 384)
        bg = align.warp(frame, m_bg, 384)
        off = torch.as_tensor(np.asarray(offset, np.float32).reshape(1, 18, 512), device=self.device)
        out, codes = self.net(_to_tensor(bg, self.device, 0.5, 0.5), _to_tensor(face, self.device, 0.5, 0.5), off)
        image = _to_bgr(out.float().clamp(-1, 1) * 0.5 + 0.5)[0].astype(np.uint8)
        image = cv2.resize(image, (self.out_size, self.out_size), interpolation=cv2.INTER_AREA)
        image = match_color(cv2.resize(bg, (self.out_size, self.out_size), interpolation=cv2.INTER_CUBIC), image)
        m_out = m_bg * (self.out_size / 384)
        return image, m_out, codes.float().cpu().numpy().reshape(18, 512)


class FRAN:
    """``reage(frame, landmarks, from_age, to_age)``: the face made to look ``to_age`` instead of ``from_age``."""

    template, size = "ffhq_512", 1024  # the graph is built for 1024 x 1024

    def __init__(self, path, device="cpu"):
        self.device = torch.device(device)
        self.net = load(path, self.device)

    @torch.inference_mode()
    def reage(self, frame: np.ndarray, landmarks, from_age: float, to_age: float):
        m = align.matrix(landmarks, self.template, self.size)
        crop = align.warp(frame, m, self.size)
        x = _to_tensor(crop, self.device)
        d = torch.tensor(np.clip([from_age / 100, to_age / 100], 0, 1), dtype=torch.float32, device=self.device)
        return _to_bgr(self.net(x, d))[0], m


@dataclass(frozen=True)
class Reading:
    """A judge's verdict on a face: how likely it looks female, and its age (years, a band's middle for FairFace)."""

    female: float
    age: float

    @property
    def gender(self) -> str:
        return "female" if self.female >= 0.5 else "male"


class FairFace:
    """The FairFace classifier (Kärkkäinen & Joo 2021, ResNet-34; trained on a dataset balanced across race,
    gender and age): gender, race and an age band."""

    template, size = "arcface_112_v2", 224

    def __init__(self, path, device="cpu"):
        self.device = torch.device(device)
        self.net = load(path, self.device, outputs=["/model/model/fc/Gemm_output_0"])  # the logits, not the ArgMax

    @torch.inference_mode()
    def read(self, frame: np.ndarray, landmarks_list) -> list[Reading]:
        crops = [align.warp(frame, align.matrix(lm, self.template, self.size), self.size) for lm in landmarks_list]
        if not crops:
            return []
        x = _to_tensor(np.stack(crops), self.device, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        logits = self.net(x).float().cpu()
        female = logits[:, 7:9].softmax(-1)[:, 1]
        ages = logits[:, 9:18].softmax(-1).numpy() @ np.array([np.mean(b) for b in FAIRFACE_AGES])
        return [Reading(float(f), float(a)) for f, a in zip(female, ages)]


class GenderAge:
    """InsightFace's gender + age model (``buffalo_l/genderage.onnx``, MobileNet on CelebA): reads a 96 x 96 crop
    round the face box."""

    size = 96

    def __init__(self, path, device="cpu"):
        self.device = torch.device(device)
        self.net = load(path, self.device)

    @torch.inference_mode()
    def read(self, frame: np.ndarray, boxes) -> list[Reading]:
        crops = []
        for x, y, w, h in boxes:
            crop, _ = align.center_crop(frame, (x + w / 2, y + h / 2), max(w, h) * 1.5, self.size)
            crops.append(crop)
        if not crops:
            return []
        x = _to_tensor(np.stack(crops), self.device, scale=1.0)  # the graph normalises itself
        out = self.net(x).float().cpu()
        female = out[:, :2].softmax(-1)[:, 0]  # 0: female, 1: male
        return [Reading(float(f), float(a) * 100) for f, a in zip(female, out[:, 2])]
