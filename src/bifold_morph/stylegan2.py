"""The StyleGAN2 generator (Karras et al., "Analyzing and Improving the Image Quality of StyleGAN", 2020) in plain
PyTorch.

Modules, parameters and buffers carry the names of rosinality's `stylegan2-pytorch
<https://github.com/rosinality/stylegan2-pytorch>`_, which the e4e checkpoint stores its decoder in, so its weights
load with ``strict=True``. The two compiled CUDA ops of that code (the fused bias + leaky ReLU and ``upfirdn2d``) are
written here with ordinary PyTorch calls, so the generator runs on the CPU, on CUDA and on Apple's MPS alike.

:meth:`Generator.synthesis` can stop early (``size=256``): StyleGAN2's generator adds up the RGB output of every
resolution ("skip" architecture), so the sum up to 256 is a smaller version of the same face, at a fraction of the
work of the 1024 px one.
"""

from __future__ import annotations

import math
import re

import torch
import torch.nn.functional as F
from torch import nn

SQRT2 = math.sqrt(2.0)


def fused_leaky_relu(x: torch.Tensor, bias: torch.Tensor, negative_slope: float = 0.2,
                     scale: float = SQRT2) -> torch.Tensor:
    """``leaky_relu(x + bias) * scale``, the bias added along dimension 1."""
    return F.leaky_relu(x + bias.view((1, -1) + (1,) * (x.ndim - 2)), negative_slope) * scale


class FusedLeakyReLU(nn.Module):
    def __init__(self, channel: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(channel))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fused_leaky_relu(x, self.bias)


def upfirdn2d(x: torch.Tensor, kernel: torch.Tensor, up: int = 1, down: int = 1, pad=(0, 0)) -> torch.Tensor:
    """Upsample ``x`` (B, C, H, W) by inserting zeros, pad it, filter it with the 2-D FIR ``kernel`` and keep every
    ``down``-th pixel: rosinality's ``upfirdn2d`` with the padding ``(before, after)`` on both axes."""
    b, c, h, w = x.shape
    if up > 1:
        spread = x.new_zeros(b, c, h * up, w * up)
        spread[:, :, ::up, ::up] = x
        x = spread
    p0, p1 = pad
    x = F.pad(x, [max(p0, 0), max(p1, 0), max(p0, 0), max(p1, 0)])
    if p0 < 0 or p1 < 0:
        x = x[:, :, max(-p0, 0):x.shape[2] - max(-p1, 0), max(-p0, 0):x.shape[3] - max(-p1, 0)]
    weight = torch.flip(kernel, [0, 1]).to(x.dtype)[None, None].repeat(c, 1, 1, 1)
    x = F.conv2d(x, weight, groups=c)
    return x[:, :, ::down, ::down] if down > 1 else x


class PixelNorm(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)


def make_kernel(k) -> torch.Tensor:
    k = torch.tensor(k, dtype=torch.float32)
    if k.ndim == 1:
        k = k[None, :] * k[:, None]
    return k / k.sum()


class Upsample(nn.Module):
    def __init__(self, kernel, factor: int = 2):
        super().__init__()
        self.factor = factor
        self.register_buffer("kernel", make_kernel(kernel) * factor ** 2)
        p = self.kernel.shape[0] - factor
        self.pad = ((p + 1) // 2 + factor - 1, p // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return upfirdn2d(x, self.kernel, up=self.factor, pad=self.pad)


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor: int = 1):
        super().__init__()
        self.register_buffer("kernel", make_kernel(kernel) * upsample_factor ** 2)
        self.pad = pad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return upfirdn2d(x, self.kernel, pad=self.pad)


class EqualLinear(nn.Module):
    """A linear layer with the equalized learning rate (weights scaled at run time), optionally with the fused
    leaky ReLU."""

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True, bias_init: float = 0.0, lr_mul: float = 1.0,
                 activation: str | None = None):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))
        self.bias = nn.Parameter(torch.full((out_dim,), float(bias_init))) if bias else None
        self.activation = activation
        self.scale = lr_mul / math.sqrt(in_dim)
        self.lr_mul = lr_mul

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation:
            return fused_leaky_relu(F.linear(x, self.weight * self.scale), self.bias * self.lr_mul)
        bias = self.bias * self.lr_mul if self.bias is not None else None
        return F.linear(x, self.weight * self.scale, bias=bias)


class ModulatedConv2d(nn.Module):
    """A convolution whose weights are scaled per input channel by the style (and normalized again: demodulation),
    computed for the whole batch at once as a grouped convolution."""

    def __init__(self, in_channel: int, out_channel: int, kernel_size: int, style_dim: int, demodulate: bool = True,
                 upsample: bool = False, blur_kernel=(1, 3, 3, 1)):
        super().__init__()
        self.kernel_size, self.in_channel, self.out_channel = kernel_size, in_channel, out_channel
        self.upsample = upsample
        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            self.blur = Blur(blur_kernel, pad=((p + 1) // 2 + factor - 1, p // 2 + 1), upsample_factor=factor)
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)
        self.padding = kernel_size // 2
        self.weight = nn.Parameter(torch.randn(1, out_channel, in_channel, kernel_size, kernel_size))
        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)
        self.demodulate = demodulate

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        batch, in_channel, height, width = x.shape
        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.weight * style
        if self.demodulate:
            weight = weight * torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8).view(batch, self.out_channel, 1, 1, 1)
        k = self.kernel_size
        x = x.reshape(1, batch * in_channel, height, width)
        if self.upsample:
            weight = weight.transpose(1, 2).reshape(batch * in_channel, self.out_channel, k, k)
            out = F.conv_transpose2d(x, weight, padding=0, stride=2, groups=batch)
            out = self.blur(out.view(batch, self.out_channel, *out.shape[2:]))
        else:
            out = F.conv2d(x, weight.view(batch * self.out_channel, in_channel, k, k), padding=self.padding,
                           groups=batch)
            out = out.view(batch, self.out_channel, *out.shape[2:])
        return out


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()
        return image + self.weight * noise


class ConstantInput(nn.Module):
    def __init__(self, channel: int, size: int = 4):
        super().__init__()
        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.input.repeat(latent.shape[0], 1, 1, 1)


class StyledConv(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, kernel_size: int, style_dim: int, upsample: bool = False,
                 blur_kernel=(1, 3, 3, 1)):
        super().__init__()
        self.conv = ModulatedConv2d(in_channel, out_channel, kernel_size, style_dim, upsample=upsample,
                                    blur_kernel=blur_kernel)
        self.noise = NoiseInjection()
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, x: torch.Tensor, style: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        return self.activate(self.noise(self.conv(x, style), noise=noise))


class ToRGB(nn.Module):
    def __init__(self, in_channel: int, style_dim: int, upsample: bool = True, blur_kernel=(1, 3, 3, 1)):
        super().__init__()
        if upsample:
            self.upsample = Upsample(blur_kernel)
        self.conv = ModulatedConv2d(in_channel, 3, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, x: torch.Tensor, style: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        out = self.conv(x, style) + self.bias
        return out + self.upsample(skip) if skip is not None else out


class Generator(nn.Module):
    """The StyleGAN2 generator ("config-f" for FFHQ: ``size=1024, channel_multiplier=2``).

    :meth:`mapping` turns a normal ``z`` (B, 512) into a ``w`` (B, 512); :meth:`synthesis` turns ``w+`` codes
    (B, :attr:`n_latent`, 512), one ``w`` per layer, into images (B, 3, size, size) in about [-1, 1], RGB.
    """

    def __init__(self, size: int = 1024, style_dim: int = 512, n_mlp: int = 8, channel_multiplier: int = 2,
                 blur_kernel=(1, 3, 3, 1), lr_mlp: float = 0.01):
        super().__init__()
        self.size, self.style_dim = size, style_dim
        self.style = nn.Sequential(PixelNorm(), *(
            EqualLinear(style_dim, style_dim, lr_mul=lr_mlp, activation="fused_lrelu") for _ in range(n_mlp)))
        self.channels = {4: 512, 8: 512, 16: 512, 32: 512, 64: 256 * channel_multiplier,
                         128: 128 * channel_multiplier, 256: 64 * channel_multiplier, 512: 32 * channel_multiplier,
                         1024: 16 * channel_multiplier}
        self.input = ConstantInput(self.channels[4])
        self.conv1 = StyledConv(self.channels[4], self.channels[4], 3, style_dim, blur_kernel=blur_kernel)
        self.to_rgb1 = ToRGB(self.channels[4], style_dim, upsample=False)
        self.log_size = int(math.log2(size))
        self.num_layers = (self.log_size - 2) * 2 + 1
        self.convs, self.to_rgbs = nn.ModuleList(), nn.ModuleList()
        self.noises = nn.Module()
        for layer in range(self.num_layers):
            res = (layer + 5) // 2
            self.noises.register_buffer(f"noise_{layer}", torch.randn(1, 1, 2 ** res, 2 ** res))
        in_channel = self.channels[4]
        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2 ** i]
            self.convs.append(StyledConv(in_channel, out_channel, 3, style_dim, upsample=True,
                                         blur_kernel=blur_kernel))
            self.convs.append(StyledConv(out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel))
            self.to_rgbs.append(ToRGB(out_channel, style_dim))
            in_channel = out_channel
        self.n_latent = self.log_size * 2 - 2

    def mapping(self, z: torch.Tensor) -> torch.Tensor:
        return self.style(z)

    @torch.no_grad()
    def mean_latent(self, n: int = 10_000) -> torch.Tensor:
        """The average ``w`` (512,) of ``n`` random ``z``."""
        z = torch.randn(n, self.style_dim, device=self.input.input.device)
        return self.style(z).mean(0)

    def synthesis(self, latent: torch.Tensor, size: int | None = None, randomize_noise: bool = False,
                  conditions: list[torch.Tensor] | None = None) -> torch.Tensor:
        """Images (B, 3, size, size) for ``w+`` codes ``latent`` (B, n_latent, 512), or for ``w`` (B, 512) used
        for every layer. ``size`` (a power of two up to :attr:`size`) stops the generator at that resolution. The
        noise is the generator's own fixed noise unless ``randomize_noise``: the same code gives the same image.

        ``conditions`` (GFPGAN's spatial feature transform): per resolution a scale and a shift applied to the
        second half of the channels after the first convolution, ``[scale_8, shift_8, scale_16, shift_16, ...]``."""
        if latent.ndim == 2:
            latent = latent[:, None].expand(-1, self.n_latent, -1)
        stop = int(math.log2(size or self.size))
        if not 2 <= stop <= self.log_size or 2 ** stop != (size or self.size):
            raise ValueError(f"size must be a power of two from 4 to {self.size}, not {size}")
        noise = [None if randomize_noise else getattr(self.noises, f"noise_{i}") for i in range(self.num_layers)]
        out = self.conv1(self.input(latent), latent[:, 0], noise=noise[0])
        skip = self.to_rgb1(out, latent[:, 1])
        i = 1
        for res, (conv1, conv2, to_rgb) in enumerate(zip(self.convs[::2], self.convs[1::2], self.to_rgbs), 3):
            if res > stop:
                break
            out = conv1(out, latent[:, i], noise=noise[i])
            if conditions is not None and i < len(conditions):
                same, sft = torch.split(out, out.shape[1] // 2, dim=1)
                out = torch.cat([same, sft * conditions[i - 1] + conditions[i]], dim=1)
            out = conv2(out, latent[:, i + 1], noise=noise[i + 1])
            skip = to_rgb(out, latent[:, i + 2], skip)
            i += 2
        return skip

    def forward(self, latent: torch.Tensor, size: int | None = None) -> torch.Tensor:
        return self.synthesis(latent, size)


def from_basicsr(state: dict) -> dict:
    """A BasicSR ``StyleGAN2Generator`` state dict (as in GFPGAN's checkpoints) with this module's names."""
    out = {}
    for k, v in state.items():
        k = (k.replace("style_mlp.", "style.").replace("constant_input.weight", "input.input")
             .replace("style_conv1.", "conv1.").replace("style_convs.", "convs.").replace("modulated_conv.", "conv."))
        if k.startswith("noises.noise"):
            k = "noises.noise_" + k[len("noises.noise"):]
        if re.fullmatch(r"conv1\.weight|convs\.\d+\.weight", k):
            k = k[:-len("weight")] + "noise.weight"  # BasicSR keeps the noise strength on the style conv itself
        out[k] = v
    return out
