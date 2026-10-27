"""The student: a small U-Net that changes a face's gender and age in one pass (see :mod:`.distill`).

Input: an aligned face (256 x 256, the "ffhq_512" template: the head with hair and some background, RGB in
[-1, 1]) and two amounts: the gender change in female-to-male gaps (+ towards female) and the age change in
years. Output: the same face changed by that much. The network predicts the *change* (a residual), and the two
amounts steer every level through FiLM (feature-wise scale and shift). With both at 0 the change is learned to be
nothing.

It works on 2 x 2 pixel blocks folded into channels (pixel unshuffle), so its first level is at 128 px; the change
it predicts is unfolded back to 256 px. About 6 million parameters: a few milliseconds per face on a laptop GPU.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .distill import AGE_RANGE, GENDER_RANGE, SIZE


class Block(nn.Module):
    """Two 3x3 convolutions with GroupNorm, FiLM from the amounts, SiLU, and a residual connection."""

    def __init__(self, c_in: int, c_out: int, n_cond: int = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, c_out)
        self.film = nn.Linear(n_cond, 2 * c_out)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(self.conv1(x)))
        scale, shift = self.film(cond)[:, :, None, None].chunk(2, dim=1)
        h = self.norm2(self.conv2(h)) * (1 + scale) + shift
        return F.silu(h) + self.skip(x)


class Student(nn.Module):
    def __init__(self, widths=(32, 64, 128, 192, 256)):
        super().__init__()
        # 2 x 2 pixel blocks folded into channels: the network works from 128 px, its output is unfolded to 256
        self.stem = nn.Conv2d(4 * 3 + 2, widths[0], 3, padding=1)  # the amounts also as constant maps
        self.down = nn.ModuleList()
        self.enc = nn.ModuleList()
        c = widths[0]
        self.enc.append(Block(c, c))
        for w in widths[1:]:
            self.down.append(nn.Conv2d(c, w, 3, stride=2, padding=1))
            self.enc.append(Block(w, w))
            c = w
        self.mid = Block(c, c)
        self.dec = nn.ModuleList()
        for w in reversed(widths[:-1]):
            self.dec.append(Block(c + w, w))
            c = w
        self.head = nn.Conv2d(c, 4 * 3, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    @staticmethod
    def cond(gender: torch.Tensor, years: torch.Tensor) -> torch.Tensor:
        """The amounts scaled to about [-1, 1]: (B, 2)."""
        return torch.stack([gender / GENDER_RANGE, years / AGE_RANGE], dim=1).float()

    def forward(self, x: torch.Tensor, gender: torch.Tensor, years: torch.Tensor) -> torch.Tensor:
        cond = self.cond(gender, years).to(x.dtype)
        folded = F.pixel_unshuffle(x, 2)
        maps = cond[:, :, None, None].expand(-1, -1, *folded.shape[2:])
        h = self.stem(torch.cat([folded, maps], dim=1))
        skips = []
        for i, block in enumerate(self.enc):
            if i > 0:
                h = self.down[i - 1](h)
            h = block(h, cond)
            skips.append(h)
        h = self.mid(h, cond)
        for block, skip in zip(self.dec, reversed(skips[:-1])):
            h = F.interpolate(h, size=skip.shape[2:], mode="bilinear", align_corners=False)
            h = block(torch.cat([h, skip], dim=1), cond)
        return (x + F.pixel_shuffle(self.head(h), 2)).clamp(-1, 1)


def load(path, device="cpu") -> Student:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = Student(tuple(ckpt.get("widths", (32, 64, 128, 192, 256))))
    model.load_state_dict({k: v.float() for k, v in ckpt["model"].items()})
    return model.to(device).eval()


def save(model: Student, path, **info) -> None:
    widths = [b.conv1.out_channels for b in model.enc]
    torch.save({"model": {k: v.half() for k, v in model.state_dict().items()}, "widths": widths, **info}, path)

