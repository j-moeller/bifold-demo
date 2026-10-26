"""GFPGAN v1's encoder, and the StyleGAN2 it was trained against: a way into StyleGAN's latent space for real faces.

GFPGAN ([Wang et al., CVPR 2021](https://arxiv.org/abs/2101.04061)) restores blurry or damaged faces. Its first
version froze a StyleGAN2 trained on FFHQ (512 px, BasicSR's ``StyleGAN2_512_Cmul1``) and learned a U-Net encoder
for it, which reads an FFHQ-aligned face and gives

* ``w+`` codes (16 x 512) in that StyleGAN2's latent space, and
* *conditions*: a per-pixel scale and shift for half the generator's channels at each resolution (spatial feature
  transform, SFT), which pull the result towards the input's exact details.

The checkpoint ``GFPGANv1.pth`` (GitHub release ``TencentARC/GFPGAN`` v0.1.0) holds both parts, including the
generator's mapping network, so the same file also draws random faces (see :mod:`.directions`). Names follow
GFPGAN's code (Apache 2.0), so its ``params_ema`` load as they are.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .stylegan2 import EqualLinear, FusedLeakyReLU, Generator, from_basicsr, make_kernel, upfirdn2d

SIZE = 512  # GFPGAN's aligned face (the "ffhq_512" template)


class EqualConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0,
                 bias: bool = True, bias_init_val: float = 0.0):
        super().__init__()
        self.stride, self.padding = stride, padding
        self.scale = 1 / math.sqrt(in_channels * kernel_size ** 2)
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.full((out_channels,), float(bias_init_val))) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight * self.scale, self.bias, self.stride, self.padding)


class ScaledLeakyReLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(x, 0.2) * math.sqrt(2)


class Smooth(nn.Module):
    """The FIR blur before a strided convolution (BasicSR's ``UpFirDnSmooth`` for downsampling)."""

    def __init__(self, kernel_size: int):
        super().__init__()
        self.register_buffer("kernel", make_kernel([1, 3, 3, 1]), persistent=False)
        p = (4 - 2) + (kernel_size - 1)
        self.pad = ((p + 1) // 2, p // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return upfirdn2d(x, self.kernel, pad=self.pad)


class ConvLayer(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, downsample: bool = False,
                 bias: bool = True, activate: bool = True):
        layers: list[nn.Module] = []
        if downsample:
            layers.append(Smooth(kernel_size))
        layers.append(EqualConv2d(in_channels, out_channels, kernel_size, stride=2 if downsample else 1,
                                  padding=0 if downsample else kernel_size // 2, bias=bias and not activate))
        if activate:
            layers.append(FusedLeakyReLU(out_channels) if bias else ScaledLeakyReLU())
        super().__init__(*layers)


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = ConvLayer(in_channels, in_channels, 3)
        self.conv2 = ConvLayer(in_channels, out_channels, 3, downsample=True)
        self.skip = ConvLayer(in_channels, out_channels, 1, downsample=True, bias=False, activate=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.conv2(self.conv1(x)) + self.skip(x)) / math.sqrt(2)


class ConvUpLayer(nn.Module):
    """Bilinear 2x upsampling, then a convolution."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int = 0,
                 bias: bool = True, activate: bool = True):
        super().__init__()
        self.padding = padding
        self.scale = 1 / math.sqrt(in_channels * kernel_size ** 2)
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias and not activate else None
        self.activation = (FusedLeakyReLU(out_channels) if bias else ScaledLeakyReLU()) if activate else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        out = F.conv2d(out, self.weight * self.scale, self.bias, padding=self.padding)
        return self.activation(out) if self.activation is not None else out


class ResUpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = ConvLayer(in_channels, in_channels, 3)
        self.conv2 = ConvUpLayer(in_channels, out_channels, 3, padding=1)
        self.skip = ConvUpLayer(in_channels, out_channels, 1, bias=False, activate=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.conv2(self.conv1(x)) + self.skip(x)) / math.sqrt(2)


class GFPGANv1(nn.Module):
    """Aligned faces (B, 3, 512, 512, RGB in [-1, 1]) -> ``w+`` codes (B, 16, 512) and SFT conditions; the frozen
    StyleGAN2 is :attr:`stylegan_decoder`, a :class:`~.stylegan2.Generator`."""

    def __init__(self, out_size: int = SIZE, channel_multiplier: int = 1):
        super().__init__()
        half = 0.5  # GFPGAN's U-Net uses half the generator's channels
        ch = {r: int(c * half) for r, c in {4: 512, 8: 512, 16: 512, 32: 512, 64: 256 * channel_multiplier,
                                            128: 128 * channel_multiplier, 256: 64 * channel_multiplier,
                                            512: 32 * channel_multiplier, 1024: 16 * channel_multiplier}.items()}
        self.log_size = int(math.log2(out_size))
        self.conv_body_first = ConvLayer(3, ch[out_size], 1)
        self.conv_body_down = nn.ModuleList()
        c = ch[out_size]
        for i in range(self.log_size, 2, -1):
            self.conv_body_down.append(ResBlock(c, ch[2 ** (i - 1)]))
            c = ch[2 ** (i - 1)]
        self.final_conv = ConvLayer(c, ch[4], 3)
        self.conv_body_up = nn.ModuleList()
        self.toRGB = nn.ModuleList()
        self.condition_scale = nn.ModuleList()
        self.condition_shift = nn.ModuleList()
        c = ch[4]
        for i in range(3, self.log_size + 1):
            out = ch[2 ** i]
            self.conv_body_up.append(ResUpBlock(c, out))
            self.toRGB.append(EqualConv2d(out, 3, 1))  # GFPGAN's training-only intermediate outputs
            self.condition_scale.append(nn.Sequential(EqualConv2d(out, out, 3, padding=1), ScaledLeakyReLU(),
                                                      EqualConv2d(out, out, 3, padding=1, bias_init_val=1)))
            self.condition_shift.append(nn.Sequential(EqualConv2d(out, out, 3, padding=1), ScaledLeakyReLU(),
                                                      EqualConv2d(out, out, 3, padding=1)))
            c = out
        self.n_latent = self.log_size * 2 - 2
        self.final_linear = EqualLinear(ch[4] * 16, self.n_latent * 512)
        self.stylegan_decoder = Generator(out_size, channel_multiplier=channel_multiplier)

    @classmethod
    def load(cls, path, device="cpu") -> "GFPGANv1":
        ckpt = torch.load(Path(path), map_location="cpu", weights_only=True)
        state = ckpt.get("params_ema", ckpt.get("params", ckpt))
        model = cls()
        own = {k: v for k, v in state.items() if not k.startswith("stylegan_decoder.")}
        dec = from_basicsr({k[len("stylegan_decoder."):]: v for k, v in state.items()
                            if k.startswith("stylegan_decoder.")})
        missing, unexpected = model.load_state_dict({**own, **{f"stylegan_decoder.{k}": v for k, v in dec.items()}},
                                                    strict=False)
        missing = [k for k in missing if not k.endswith(".kernel")]  # blur kernels are built, not stored
        if missing or unexpected:
            raise ValueError(f"{Path(path).name} does not fit GFPGANv1: missing {missing[:4]}, unexpected "
                             f"{unexpected[:4]}")
        return model.to(device).eval()

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """``w+`` codes (B, 16, 512) and the SFT conditions for aligned faces ``x``."""
        skips = []
        feat = self.conv_body_first(x)
        for down in self.conv_body_down:
            feat = down(feat)
            skips.insert(0, feat)
        feat = self.final_conv(feat)
        codes = self.final_linear(feat.flatten(1)).view(feat.shape[0], self.n_latent, 512)
        conditions = []
        for i, up in enumerate(self.conv_body_up):
            feat = up(feat + skips[i])
            conditions += [self.condition_scale[i](feat), self.condition_shift[i](feat)]
        return codes, conditions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """GFPGAN's own restoration of ``x``."""
        codes, conditions = self.encode(x)
        return self.stylegan_decoder.synthesis(codes, conditions=conditions)


def to_input(faces_bgr: np.ndarray, device) -> torch.Tensor:
    """Aligned BGR uint8 faces (B, 512, 512, 3) -> RGB in [-1, 1] (B, 3, 512, 512)."""
    x = torch.from_numpy(np.ascontiguousarray(np.asarray(faces_bgr)[..., ::-1])).to(device)
    return (x.permute(0, 3, 1, 2).float() / 127.5 - 1.0).contiguous()


def to_bgr(images: torch.Tensor) -> np.ndarray:
    """Generator output (B, 3, S, S) in [-1, 1] -> BGR float32 (B, S, S, 3) in [0, 255]."""
    x = ((images.detach().float().clamp(-1, 1) + 1) * 127.5).permute(0, 2, 3, 1).cpu().numpy()
    return np.ascontiguousarray(x[..., ::-1])
