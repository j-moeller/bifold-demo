"""Seamless tiling inside a diffusers pipeline: latent rolling for the transformers, circular padding for convolutions.

A text-to-image model paints a picture, not a repeat: nothing makes its right edge continue its left one. Two
patches make it paint a repeat, and the fabric generator (:mod:`.fabric`) applies them for the length of one call,
so the same loaded pipeline can still make other pictures:

* **Latent rolling** (FLUX.2 [klein], Z-Image: diffusion transformers, whose attention sees the image as a sequence of
  patches with fixed positions, so there is no padding to change). After every denoising step, the
  ``callback_on_step_end`` of the pipeline rolls its latents round the image (what leaves the right edge comes back
  in at the left), so the next step sees the old edge in the middle of the picture and paints across it. The roll
  before step ``i`` is random (seeded), up to half the image at first and shrinking to nothing for the last step
  (:func:`roll_offsets`); each callback rolls from the last offset to the next, and the last one rolls the latents
  back to where they started. The pipelines' Euler step is the same at every place, so this is the same as rolling
  before each step and back after it. Whatever seam the last step leaves is on the image's true edge, where the
  circular decoder and :func:`.seamless.seam_score` look and :func:`.seamless.make_seamless` repairs. The latents in
  the callback, as diffusers 0.42 has them:

  - ``Flux2KleinPipeline``: packed tokens ``(batch, h * w, 4 * channels)``, one token per 2 x 2 latent pixels (16 x
    16 image pixels), row by row (token ``r * w + c`` is at row ``r``, column ``c``; ``_prepare_latent_ids`` gives
    the same order and ``_unpack_latents_with_ids`` puts them back by it). They are unpacked to ``(batch, h, w,
    4 * channels)``, rolled by whole tokens and packed again. A swatch is square, so ``h = w`` is the square root of
    the sequence length.
  - ``ZImagePipeline``: the latent grid itself, ``(batch, 16, H / 8, W / 8)``; the transformer cuts it into 2 x 2
    patches inside, so it is rolled by whole patches (2 latent pixels), to keep the patches whole.

* **Circular padding** (:func:`circular_padding`): every ``Conv2d`` of a module pads by wrapping round instead of with
  zeros, by switching its ``padding_mode`` for the length of the call and back afterwards (also when the call fails
  or is cancelled). A convolutional UNet (SD-Turbo) then denoises a torus and needs no rolling; the VAE decoder of
  every pipeline is made circular, so the decoded edges match. VAE tiling, if on, is switched off for the call.

Neither patch copies the pipeline (FLUX.2 [klein] is about 16 GB). The fabric generator holds the pipeline's lock while
they are on, so no other picture is made with the pipeline patched; diffusers pipelines cannot run two calls at once
anyway. :class:`StepCallback` also stops a call when its job is cancelled (it raises :class:`Cancelled` at the end of
the next step) and reports each step's progress.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass

import numpy as np


class Cancelled(Exception):
    """Raised from a pipeline's step callback to stop the call: its job was cancelled."""


def roll_offsets(steps: int, grid: tuple[int, int], seed: int) -> list[tuple[int, int]]:
    """The roll (rows, columns, in grid units) the latents carry into each of ``steps`` denoising steps: none into the
    first (pure noise looks the same rolled), then random offsets of up to ``(steps - 1 - i) / (steps - 1)`` of half
    the grid, so none into the last. ``seed`` makes them reproducible."""
    rng = np.random.default_rng([int(seed) & 0xFFFFFFFF, 0x7011])
    out = [(0, 0)]
    for i in range(1, steps):
        taper = (steps - 1 - i) / max(1, steps - 1)
        out.append(tuple(round(rng.uniform(-0.5, 0.5) * size * taper) for size in grid))
    return out


class LatentRoller:
    """A ``callback_on_step_end`` that rolls a pipeline's latents (see the module docs). ``steps``: the denoising
    steps it expects (the pipeline's own count wins when it has one, ``pipe.num_timesteps``); ``seed``: for the
    offsets; ``unit``: latent pixels per roll step for a latent grid (the transformer's patch size; packed tokens
    always roll by whole tokens). :attr:`rolls` lists ``(step, (rows, columns))`` as applied, for inspection."""

    def __init__(self, steps: int, seed: int, unit: int = 1):
        self.steps, self.seed, self.unit = int(steps), int(seed), max(1, int(unit))
        self.offset = (0, 0)  # the roll the latents carry now
        self.offsets: list[tuple[int, int]] | None = None
        self.rolls: list[tuple[int, tuple[int, int]]] = []

    def grid(self, latents) -> tuple[int, int]:
        """The latents' grid in roll units: ``(rows, columns)``."""
        if latents.ndim == 3:  # packed tokens (batch, h * w, channels) of a square image
            side = math.isqrt(int(latents.shape[1]))
            if side * side != int(latents.shape[1]):
                raise ValueError(f"{latents.shape[1]} tokens are not a square grid")
            return side, side
        if latents.ndim == 4:  # (batch, channels, height, width)
            return int(latents.shape[2]) // self.unit, int(latents.shape[3]) // self.unit
        raise ValueError(f"cannot roll latents of shape {tuple(latents.shape)}")

    def roll(self, latents, shift: tuple[int, int]):
        """``latents`` rolled by ``shift`` (rows, columns) grid units, down and to the right."""
        import torch

        if latents.ndim == 3:
            batch, length, channels = latents.shape
            side = math.isqrt(int(length))
            grid = latents.reshape(batch, side, side, channels)  # unpacked: token r * side + c is at (r, c)
            return torch.roll(grid, shifts=shift, dims=(1, 2)).reshape(batch, length, channels)
        return torch.roll(latents, shifts=(shift[0] * self.unit, shift[1] * self.unit), dims=(2, 3))

    def __call__(self, pipe, step: int, timestep, callback_kwargs: dict) -> dict:
        latents = callback_kwargs["latents"]
        total = int(getattr(pipe, "num_timesteps", 0) or self.steps)
        if self.offsets is None or len(self.offsets) != total:
            self.offsets = roll_offsets(total, self.grid(latents), self.seed)
        target = self.offsets[step + 1] if step + 1 < total else (0, 0)  # after the last step: back where they were
        shift = (target[0] - self.offset[0], target[1] - self.offset[1])
        self.offset = target
        if shift == (0, 0):
            return {}
        self.rolls.append((int(step), shift))
        return {"latents": self.roll(latents, shift)}


class StepCallback:
    """The ``callback_on_step_end`` the fabric generator passes: it stops the call when ``cancelled()`` is true
    (raising :class:`Cancelled`), rolls the latents with ``roller`` (a :class:`LatentRoller`, or ``None``), reports
    ``progress(step, steps)`` and gives ``pace()`` a chance to slow the worker down for the main thread."""

    def __init__(self, roller: LatentRoller | None = None, cancelled=None, progress=None, pace=None):
        self.roller, self.cancelled, self.progress, self.pace = roller, cancelled, progress, pace

    def __call__(self, pipe, step: int, timestep, callback_kwargs: dict) -> dict:
        if self.cancelled is not None and self.cancelled():
            raise Cancelled()
        out = self.roller(pipe, step, timestep, callback_kwargs) if self.roller is not None else {}
        if self.progress is not None:
            total = int(getattr(pipe, "num_timesteps", 0) or (self.roller.steps if self.roller else 0) or step + 1)
            self.progress(step + 1, total)
        if self.pace is not None:
            self.pace()
        return out


@contextlib.contextmanager
def circular_padding(*modules):
    """While inside, every ``torch.nn.Conv2d`` of ``modules`` (``None`` is skipped) that pads pads circularly, and VAE
    tiling (``use_tiling``) is off; afterwards all is as before, also after an exception. Yields how many
    convolutions it changed."""
    import torch

    changed, tiled = [], []
    try:
        for module in modules:
            if module is None:
                continue
            if getattr(module, "use_tiling", False):
                tiled.append(module)
                module.use_tiling = False
            for layer in module.modules():
                if isinstance(layer, torch.nn.Conv2d) and layer.padding_mode != "circular" \
                        and any(int(p) for p in layer._reversed_padding_repeated_twice):
                    changed.append((layer, layer.padding_mode))
                    layer.padding_mode = "circular"
        yield len(changed)
    finally:
        for layer, mode in changed:
            layer.padding_mode = mode
        for module in tiled:
            module.use_tiling = True


@dataclass(frozen=True)
class TilingPlan:
    """How to make a pipeline paint a repeat: ``roll`` (latent rolling, by this many latent pixels for a latent grid)
    or ``None``, and the modules whose convolutions pad circularly."""

    roll: int | None
    circular: tuple
    kind: str  # "packed transformer", "transformer", "unet" or "unknown"


def plan_for(pipe) -> TilingPlan:
    """The :class:`TilingPlan` of a diffusers pipeline: a UNet (SD-Turbo) pads circularly, UNet and VAE; a transformer
    (FLUX.2, Z-Image) rolls its latents (FLUX.2 by packed tokens, others by the transformer's patch size) and its VAE
    pads circularly. Anything else (a stand-in) rolls whatever latents it shows."""
    vae = getattr(pipe, "vae", None)
    unet = getattr(pipe, "unet", None)
    if unet is not None:
        return TilingPlan(None, (unet, vae), "unet")
    transformer = getattr(pipe, "transformer", None)
    if transformer is not None:
        if type(pipe).__name__.startswith("Flux"):
            return TilingPlan(1, (vae,), "packed transformer")
        config = getattr(transformer, "config", None)
        sizes = getattr(config, "all_patch_size", None) or (getattr(config, "patch_size", None) or 1,)
        size = sizes[0] if isinstance(sizes, (tuple, list)) else sizes
        return TilingPlan(int(size), (vae,), "transformer")
    return TilingPlan(1, (vae,), "unknown")


__all__ = ["Cancelled", "LatentRoller", "StepCallback", "TilingPlan", "circular_padding", "plan_for", "roll_offsets"]
