"""e4e, "Designing an Encoder for StyleGAN Image Manipulation" (Tov et al., SIGGRAPH 2021): a network that finds the
StyleGAN2 code of a photo of a face in one pass, chosen so that the code stays easy to edit.

The encoder is an IR-SE-50 (the ArcFace ResNet) with a feature pyramid. It reads an FFHQ-aligned face (256 x 256, RGB
in [-1, 1]) and predicts one ``w`` for the first layer plus a change of it for each of the other 17 layers; the
average ``w`` of the generator is added on top. The names follow the official code
(`omertov/encoder4editing <https://github.com/omertov/encoder4editing>`_, MIT licence), so ``e4e_ffhq_encode.pt``
loads as it is: it holds the encoder, the StyleGAN2 FFHQ generator and the average code.
"""

from __future__ import annotations

import math
import pickle
import warnings
from collections import namedtuple
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import BatchNorm2d, Conv2d, MaxPool2d, PReLU, Sequential

from .stylegan2 import EqualLinear, Generator

INPUT_SIZE = 256  # the encoder reads 256 x 256 aligned faces

Bottleneck = namedtuple("Bottleneck", ["in_channel", "depth", "stride"])


def _blocks_ir50() -> list[list[Bottleneck]]:
    def block(in_channel, depth, units):
        return [Bottleneck(in_channel, depth, 2)] + [Bottleneck(depth, depth, 1) for _ in range(units - 1)]
    return [block(64, 64, 3), block(64, 128, 4), block(128, 256, 14), block(256, 512, 3)]


class SEModule(nn.Module):
    def __init__(self, channels: int, reduction: int):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.sigmoid(self.fc2(self.relu(self.fc1(self.avg_pool(x)))))


class BottleneckIRSE(nn.Module):
    def __init__(self, in_channel: int, depth: int, stride: int):
        super().__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                                             BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BatchNorm2d(depth),
            SEModule(depth, 16),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res_layer(x) + self.shortcut_layer(x)


class GradualStyleBlock(nn.Module):
    """Strided convolutions down to 1 x 1, then a linear layer: one 512-vector from a feature map."""

    def __init__(self, in_c: int, out_c: int, spatial: int):
        super().__init__()
        self.out_c = out_c
        modules = [Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1), nn.LeakyReLU()]
        for _ in range(int(math.log2(spatial)) - 1):
            modules += [Conv2d(out_c, out_c, kernel_size=3, stride=2, padding=1), nn.LeakyReLU()]
        self.convs = Sequential(*modules)
        self.linear = EqualLinear(out_c, out_c, lr_mul=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.convs(x).view(-1, self.out_c))


def _upsample_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.interpolate(x, size=y.shape[2:], mode="bilinear", align_corners=True) + y


class Encoder4Editing(nn.Module):
    """Aligned faces (B, 3, 256, 256) -> codes (B, style_count, 512), relative to the generator's average code."""

    def __init__(self, stylegan_size: int = 1024):
        super().__init__()
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False), BatchNorm2d(64), PReLU(64))
        self.body = Sequential(*(BottleneckIRSE(b.in_channel, b.depth, b.stride)
                                 for block in _blocks_ir50() for b in block))
        self.style_count = 2 * int(math.log2(stylegan_size)) - 2
        self.coarse_ind, self.middle_ind = 3, 7
        self.styles = nn.ModuleList(
            GradualStyleBlock(512, 512, 16 if i < self.coarse_ind else 32 if i < self.middle_ind else 64)
            for i in range(self.style_count))
        self.latlayer1 = Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = Conv2d(128, 512, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        for i, layer in enumerate(self.body):
            x = layer(x)
            if i == 6:
                c1 = x  # 128 channels, 64 x 64
            elif i == 20:
                c2 = x  # 256 channels, 32 x 32
            elif i == 23:
                c3 = x  # 512 channels, 16 x 16
        w = self.styles[0](c3)[:, None].repeat(1, self.style_count, 1)
        features = c3
        for i in range(1, self.style_count):
            if i == self.coarse_ind:
                p2 = _upsample_add(c3, self.latlayer1(c2))
                features = p2
            elif i == self.middle_ind:
                features = _upsample_add(p2, self.latlayer2(c1))
            w[:, i] += self.styles[i](features)
        return w


def _load_checkpoint(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError:
        # the official checkpoint also pickles its training options; still only tensors and plain values
        warnings.warn(f"{path.name}: loading with weights_only=False (it holds pickled training options)")
        return torch.load(path, map_location="cpu", weights_only=False)


def _sub_state(state: dict, prefix: str) -> dict:
    return {k[len(prefix) + 1:]: v for k, v in state.items() if k.startswith(prefix + ".")}


class E4E(nn.Module):
    """The e4e encoder with its StyleGAN2 generator.

    ``encode(faces)`` gives ``w+`` codes (B, 18, 512) for aligned faces (B, 3, 256, 256) in [-1, 1];
    ``generator.synthesis(codes, size)`` turns codes back into faces.
    """

    def __init__(self, stylegan_size: int = 1024):
        super().__init__()
        self.encoder = Encoder4Editing(stylegan_size)
        self.generator = Generator(stylegan_size)
        self.register_buffer("latent_avg", torch.zeros(self.generator.n_latent, 512))

    @classmethod
    def load(cls, path, device="cpu") -> "E4E":
        """The model in ``e4e_ffhq_encode.pt`` (or a checkpoint of the same layout), in eval mode on ``device``."""
        path = Path(path)
        ckpt = _load_checkpoint(path)
        state = ckpt.get("state_dict", ckpt)
        opts = ckpt.get("opts") or {}
        size = int(opts.get("stylegan_size", 1024))
        if opts.get("encoder_type", "Encoder4Editing") != "Encoder4Editing":
            raise ValueError(f"{path.name}: an {opts['encoder_type']} checkpoint, not e4e")
        model = cls(size)
        model.encoder.load_state_dict(_sub_state(state, "encoder"), strict=True)
        model.generator.load_state_dict(_sub_state(state, "decoder"), strict=True)
        avg = ckpt.get("latent_avg")
        if avg is None:
            avg = model.generator.mean_latent()
        avg = torch.as_tensor(avg, dtype=torch.float32).reshape(-1, 512)
        model.latent_avg.copy_(avg.expand(model.generator.n_latent, 512) if avg.shape[0] == 1 else avg)
        if not opts.get("start_from_latent_avg", True):
            model.latent_avg.zero_()
        return model.to(device).eval()

    @property
    def w_avg(self) -> torch.Tensor:
        """The generator's average ``w`` (512,)."""
        return self.latent_avg[0]

    def encode(self, faces: torch.Tensor) -> torch.Tensor:
        return self.encoder(faces) + self.latent_avg


def to_input(faces_rgb: np.ndarray | list[np.ndarray], device, dtype=torch.float32) -> torch.Tensor:
    """Aligned RGB uint8 faces (B, 256, 256, 3) -> the encoder's input (B, 3, 256, 256) in [-1, 1]."""
    x = torch.from_numpy(np.ascontiguousarray(np.stack(faces_rgb))).to(device)
    return (x.permute(0, 3, 1, 2).to(dtype) / 127.5 - 1.0).contiguous()


def to_images(images: torch.Tensor) -> np.ndarray:
    """Generator output (B, 3, S, S) in [-1, 1] -> RGB uint8 (B, S, S, 3)."""
    x = ((images.detach().float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
    return x.permute(0, 2, 3, 1).cpu().numpy()
