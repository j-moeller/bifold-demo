"""Generating images of people with an open text-to-image model, and turning them into cut-outs.

The image model runs through Hugging Face ``diffusers``; install it with
``pip install -e ".[people]"``. Weights are downloaded from Hugging Face on first use.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace

import cv2
import numpy as np

from .cutout import Cutout, PersonSegmenter
from .library import PeopleLibrary, Person
from .prompts import PersonSpec, random_spec


@dataclass(frozen=True)
class ModelPreset:
    """How to run one text-to-image model."""

    name: str
    repo: str  # Hugging Face model id
    pipeline: str  # diffusers pipeline class
    steps: int
    guidance: float
    width: int
    height: int
    dtype: str  # torch dtype on GPU / Apple silicon
    memory: str  # rough memory needed, for the user
    license: str
    variant: str | None = None


PRESETS = {
    "flux2-klein": ModelPreset(
        "flux2-klein", "black-forest-labs/FLUX.2-klein-4B", "Flux2KleinPipeline", steps=4, guidance=1.0,
        width=768, height=1344, dtype="bfloat16", memory="~16 GB (24 GB+ Mac recommended)", license="Apache-2.0",
    ),
    "z-image-turbo": ModelPreset(
        "z-image-turbo", "Tongyi-MAI/Z-Image-Turbo", "ZImagePipeline", steps=9, guidance=0.0,
        width=768, height=1344, dtype="bfloat16", memory="~24 GB (32 GB+ Mac recommended)", license="Apache-2.0",
    ),
    "sd-turbo": ModelPreset(
        "sd-turbo", "stabilityai/sd-turbo", "AutoPipelineForText2Image", steps=2, guidance=0.0,
        width=512, height=768, dtype="float16", memory="~4 GB (any Apple-silicon Mac)",
        license="Stability AI Community License (check before commercial use)", variant="fp16",
    ),
}
DEFAULT_PRESET = "flux2-klein"


def get_preset(model: str) -> ModelPreset:
    """A preset by name, or a generic preset for any other Hugging Face text-to-image model id."""
    if model in PRESETS:
        return PRESETS[model]
    if "/" not in model:
        raise ValueError(f"unknown model {model!r}; use one of {', '.join(PRESETS)} or a Hugging Face id")
    return ModelPreset(model, model, "AutoPipelineForText2Image", steps=4, guidance=0.0, width=768, height=1344,
                       dtype="bfloat16", memory="unknown", license="see the model card")


class ImageGenerator:
    """Text-to-image with a diffusers pipeline; returns BGR images.

    ``pipeline`` lets you pass an already loaded pipeline (or any callable with the
    diffusers call signature) instead of loading ``preset.repo``.
    """

    def __init__(self, preset: ModelPreset | str = DEFAULT_PRESET, device: str = "auto", pipeline=None):
        self.preset = get_preset(preset) if isinstance(preset, str) else preset
        if pipeline is None:
            from ..detection.yolo import pick_device

            self.device = pick_device(device)
            pipeline = self._load()
        else:
            self.device = device
        self.pipe = pipeline

    def _load(self):
        import diffusers
        import torch

        p = self.preset
        cls = getattr(diffusers, p.pipeline, None)
        if cls is None:
            raise RuntimeError(f"diffusers {diffusers.__version__} has no {p.pipeline}; run: pip install -U diffusers")
        on_cpu = self.device == "cpu"
        kwargs = {"dtype": torch.float32 if on_cpu else getattr(torch, p.dtype)}
        if p.variant and not on_cpu:
            kwargs["variant"] = p.variant
        print(f"[people] loading {p.repo} ({p.memory}) on {self.device} ...")
        pipe = cls.from_pretrained(p.repo, **kwargs)
        if self.device == "cuda" and hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(self.device)
        pipe.set_progress_bar_config(disable=True)
        return pipe

    def __call__(self, prompt: str, seed: int, width: int | None = None, height: int | None = None) -> np.ndarray:
        import torch

        p = self.preset
        result = self.pipe(
            prompt=prompt,
            width=width or p.width,
            height=height or p.height,
            num_inference_steps=p.steps,
            guidance_scale=p.guidance,
            generator=torch.Generator("cpu").manual_seed(seed),  # same image for a seed on every device
        )
        image = result.images[0]
        return cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)


def check_cutout(cutouts: list[Cutout], image_height: int) -> tuple[Cutout | None, str]:
    """Pick the person from a generated image, or explain why the image is unusable."""
    if not cutouts:
        return None, "no person found"
    main = cutouts[0]
    if main.confidence < 0.5:
        return None, f"unsure it is a person ({main.confidence:.2f})"
    if len(cutouts) > 1 and cutouts[1].area > 0.15 * main.area:
        return None, "more than one person"
    if not main.full_body:
        return None, "head or feet cut off"
    if {"left", "right"} & main.touches:
        return None, "cut off at the side"
    if main.rgba.shape[0] < 0.5 * image_height:
        return None, "person too small"
    return main, "ok"


class PeopleGenerator:
    """Generates random trade-fair visitors and stores them as cut-outs in a :class:`PeopleLibrary`."""

    def __init__(self, images: ImageGenerator, segmenter: PersonSegmenter, library: PeopleLibrary,
                 raw_folder=None):
        self.images, self.segmenter, self.library = images, segmenter, library
        self.raw_folder = raw_folder

    def generate_one(self, spec: PersonSpec, attempts: int = 3) -> tuple[Person | None, list[str]]:
        """Try up to ``attempts`` images for ``spec``; returns the stored person and the reasons for rejects."""
        reasons = []
        for attempt in range(attempts):
            seed = spec.seed + attempt * 1_000_003
            image = self.images(spec.prompt(), seed)
            if self.raw_folder is not None:
                self.raw_folder.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(self.raw_folder / f"{spec.seed}_{attempt}.jpg"), image)
            cutout, reason = check_cutout(self.segmenter.cutouts(image), image.shape[0])
            if cutout is None:
                reasons.append(reason)
                continue
            person = self.library.add(
                cutout, name=f"{spec.pose}_{spec.seed}", height=spec.height, pose=spec.pose, facing=spec.facing,
                source=f"generated:{self.images.preset.repo}", prompt=spec.prompt(), seed=seed,
            )
            return person, reasons
        return None, reasons

    def generate(self, count: int, seed: int = 0, pose: str | None = None, attempts: int = 3) -> list[Person]:
        """Generate ``count`` people (``pose``: ``"standing"``, ``"walking"`` or ``None`` for a mix)."""
        people, start = [], time.perf_counter()
        for i in range(count):
            spec = random_spec(seed + i, pose)
            person, reasons = self.generate_one(spec, attempts)
            elapsed = time.perf_counter() - start
            eta = elapsed / (i + 1) * (count - i - 1)
            status = f"saved {person.path}" if person else f"skipped ({'; '.join(reasons)})"
            if person and reasons:
                status += f" after rejecting {len(reasons)} image(s)"
            print(f"[people] {i + 1}/{count} {spec.pose} {spec.gender} {spec.age}: {status}  (eta {eta:.0f}s)")
            if person:
                people.append(person)
        return people


def preset_with(preset: ModelPreset, steps: int | None = None, size: tuple[int, int] | None = None) -> ModelPreset:
    """A copy of ``preset`` with a different step count and/or ``(width, height)``."""
    changes = {}
    if steps:
        changes["steps"] = steps
    if size:
        changes["width"], changes["height"] = (s // 16 * 16 for s in size)
    return replace(preset, **changes)
