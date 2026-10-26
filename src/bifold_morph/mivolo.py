"""MiVOLO (Kuprashevich & Tolstykh, "MiVOLO: Multi-input Transformer for Age and Gender Estimation", 2023): how old
a face looks and whether it looks male or female.

The face-only checkpoint (``volo_d1``, "face_only, age, gender", trained on IMDB-clean: age MAE 4.22 years, gender
99.4 %) is a VOLO-D1 vision transformer with three outputs, so it loads into timm's own ``volo_d1_224``: no code
from the MiVOLO repository is needed. Outputs 0 and 1 are the male and female logits, output 2 the age, scaled by
the ``min_age``, ``max_age`` and ``avg_age`` the checkpoint stores.

Here it labels faces drawn by StyleGAN2, which is how :mod:`.directions` finds which way in StyleGAN's latent space
makes a face older or more female, and it shows how old and which gender the edited faces look.
"""

from __future__ import annotations

import argparse
import pickle
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch

from .state import AgeGender

INPUT_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], np.float32)  # ImageNet's, as timm uses
STD = np.array([0.229, 0.224, 0.225], np.float32)


def letterbox(image: np.ndarray, size: int = INPUT_SIZE) -> np.ndarray:
    """``image`` scaled to fit a ``size`` square, centred on black (MiVOLO's ``class_letterbox``)."""
    h, w = image.shape[:2]
    if (h, w) == (size, size):
        return image
    r = min(size / h, size / w)
    nw, nh = round(w * r), round(h * r)
    if (nw, nh) != (w, h):
        image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dw, dh = (size - nw) / 2, (size - nh) / 2
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))


def crop_box(image: np.ndarray, box) -> np.ndarray | None:
    """The pixels of ``box`` ``(x, y, w, h)`` in ``image``, or ``None`` if it is (nearly) outside."""
    h, w = image.shape[:2]
    x, y, bw, bh = box
    x0, y0 = max(int(round(x)), 0), max(int(round(y)), 0)
    x1, y1 = min(int(round(x + bw)), w), min(int(round(y + bh)), h)
    return image[y0:y1, x0:x1] if x1 - x0 >= 8 and y1 - y0 >= 8 else None


def _load(path: Path) -> dict:
    safe = [argparse.Namespace]  # timm's training checkpoints keep their arguments
    try:
        with torch.serialization.safe_globals(safe):
            return torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError:
        warnings.warn(f"{path.name}: loading with weights_only=False (it holds pickled training state)")
        return torch.load(path, map_location="cpu", weights_only=False)


def state_dict_of(ckpt: dict) -> dict:
    """The model weights of a timm training checkpoint (the EMA ones if it has them), without MiVOLO's ``fds.``
    training-only layers and DataParallel's ``module.`` prefix."""
    for key in ("state_dict_ema", "model_ema", "state_dict", "model"):
        if isinstance(ckpt.get(key), dict):
            state = ckpt[key]
            break
    else:
        state = ckpt
    return {k.removeprefix("module."): v for k, v in state.items() if not k.startswith(("fds.", "module.fds."))}


class MiVOLO:
    """Age and gender of face crops (BGR, any size: they are letterboxed to 224 x 224)."""

    def __init__(self, model: torch.nn.Module, min_age: float, max_age: float, avg_age: float, device="cpu"):
        self.model = model.to(device).eval()
        self.min_age, self.max_age, self.avg_age = float(min_age), float(max_age), float(avg_age)
        self.device = torch.device(device)

    @staticmethod
    def build(num_classes: int = 3) -> torch.nn.Module:
        import timm  # imported here: only this model needs it

        return timm.create_model("volo_d1_224", pretrained=False, num_classes=num_classes)

    @classmethod
    def load(cls, path, device="cpu") -> "MiVOLO":
        path = Path(path)
        ckpt = _load(path)
        state = state_dict_of(ckpt)
        if "patch_embed.conv1.0.weight" in state:
            raise ValueError(f"{path.name} is MiVOLO's face + body model; use the face-only age + gender one")
        if ckpt.get("no_gender"):
            raise ValueError(f"{path.name} estimates age only; use the face-only age + gender checkpoint")
        if state["pos_embed"].shape[1] * 16 != INPUT_SIZE:
            raise ValueError(f"{path.name}: a {state['pos_embed'].shape[1] * 16} px model, expected 224 px")
        model = cls.build(num_classes=3)
        result = model.load_state_dict(state, strict=False)
        if result.missing_keys:
            raise ValueError(f"{path.name} does not fit timm's volo_d1_224, missing: {result.missing_keys[:5]} ...")
        if result.unexpected_keys:
            warnings.warn(f"{path.name}: ignoring {len(result.unexpected_keys)} weights timm's VOLO has no use for")
        return cls(model, ckpt["min_age"], ckpt["max_age"], ckpt["avg_age"], device)

    def to_input(self, crops: list[np.ndarray]) -> torch.Tensor:
        batch = []
        for img in crops:
            img = cv2.cvtColor(letterbox(img), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            batch.append(((img - MEAN) / STD).transpose(2, 0, 1))
        return torch.from_numpy(np.stack(batch)).to(self.device)

    def decode(self, output: torch.Tensor) -> list[AgeGender]:
        output = output.float().cpu()
        ages = output[:, 2] * (self.max_age - self.min_age) + self.avg_age
        male = output[:, :2].softmax(-1)[:, 0]
        return [AgeGender(float(a), float(m)) for a, m in zip(ages, male)]

    @torch.inference_mode()
    def predict(self, crops: list[np.ndarray], batch_size: int = 32) -> list[AgeGender]:
        """One :class:`AgeGender` per face crop."""
        results = []
        for i in range(0, len(crops), batch_size):
            results += self.decode(self.model(self.to_input(crops[i:i + batch_size])))
        return results
