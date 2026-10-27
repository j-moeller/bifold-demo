"""Fabrics from a description: text to a seamless, tileable fabric swatch, as a fill for the outfits.

Stage 2 of the outfits (``docs/research/outfit_textures.md``, section 3): someone types "blue paisley silk" and gets
a few swatches to choose from; the chosen one tiles over a garment like any :class:`~.fills.Swatch`, and the tailor
puts the garment's own folds and stitching on it.

**Making a swatch** (:class:`FabricGenerator`):

1. The description is tidied (:func:`clean_text`: spacing, odd characters, "please make me", the words the template
   already says) and checked against the kiosk's word lists (:class:`PromptFilter`: brand names, characters, logos
   and offensive words, in ``fabric_words.json``; a file of the same form adds to them). A refused description
   raises :class:`PromptRefused`, whose message can be shown as it is.
2. It goes into the review's template: :data:`TEMPLATE`.
3. A procedural preview comes first, at once: what the keyword parser (:func:`.prompts.parse_fill`) makes of the
   words it knows, labelled as such (method ``"procedural"``, no model).
4. The fallback ladder (:data:`LADDERS`), each rung for every seed (4 by default):

   - FLUX.2 [klein] 4B with latent rolling and a circular decoder (:mod:`.tiling`), at its preset's 4 steps;
   - the same at 12 steps, when every seed showed a seam;
   - SD-Turbo with circular padding, when the seams stay or klein fails (out of memory, an error);
   - the model-free repair (:func:`.seamless.make_seamless`) of the best rejected image, the last generative resort;
   - the keyword parser's procedural fabric, then a plain colour (the colour word nearest to one in the text, or
     grey).

   Each step down says why (:attr:`FabricJob.messages`). A model that fails (out of memory, an error) is not tried
   again at more steps. Packages that are missing (diffusers, torch), weights that are not downloaded or no network
   go straight to the procedural fabric (by way of the repair if an image was made by then), with the command that
   installs or fetches what is missing. Z-Image-Turbo takes klein's place with ``model="z-image-turbo"``.
5. Each image is scored (:func:`.seamless.seam_score`). Above ``threshold`` (:data:`.seamless.SEAM_THRESHOLD`) it is
   left out, but stays a candidate for the repair. Those kept get even light (:func:`.seamless.equalise_light`) and
   are shown as they come. Their size on the body (metres per repeat) depends on the kind of fabric the words name
   (:func:`fabric_scale`).

**For the creator screen.** :meth:`FabricGenerator.start` returns a :class:`FabricJob` at once and works on a daemon
thread. :meth:`FabricJob.poll` (it never waits) or iterating over the job gives each swatch as it is ready, the
preview first. :meth:`FabricJob.cancel` stops the job at the end of the next denoising step, and
:attr:`FabricJob.progress` says how far the current image is. The model loads on the worker the first time it is
needed, after the preview (that can take a minute; it cannot be cancelled half way). Torch releases the GIL in its
operations and the Python between them is short; ``pace``, a callable, is called after every denoising step and
between images, so a caller can slow the worker down further for the frame rate.

**The cache.** Finished swatches (kept or left out) are stored in ``fabrics/`` (git-ignored; ``$BIFOLD_FABRIC_CACHE``
moves it), keyed by the prompt, the model, the seed, the size and the method. The same description gives its
swatches back at once, and one whose swatches are all cached does not load the model at all.

**The fill.** :class:`Fabric` is a :class:`~.fills.Swatch` that knows how it was made. Its spec holds the description,
the prompt, the model, the seed, the method and the image (lossless WebP in a ``data:`` URI), so a saved outfit recipe
rebuilds it without the model; a spec without an image is the keyword fabric of its words. :func:`fabric_fill` makes
one from a description in one call.
"""

from __future__ import annotations

import codecs
import difflib
import hashlib
import inspect
import json
import os
import queue
import re
import threading
import time
import unicodedata
import warnings
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from .fills import COLORS, Pattern, Solid, Swatch, data_uri, hex_color
from .seamless import SEAM_THRESHOLD, equalise_light, make_seamless, seam_score
from .tiling import Cancelled, LatentRoller, StepCallback, circular_padding, plan_for

TEMPLATE = "seamless tileable fabric texture, flat lay, top-down, even diffuse light, no folds, {text}"
DEFAULT_MODEL = "flux2-klein"
MODEL_ENV = "BIFOLD_FABRIC_MODEL"  # the model of the default generator (fabric_fill without one); "none": no model
CACHE_ENV = "BIFOLD_FABRIC_CACHE"
WORDS_ENV = "BIFOLD_FABRIC_WORDS"
DEFAULT_CACHE = "fabrics"
WORDS_FILE = Path(__file__).with_name("fabric_words.json")
SEEDS = 4
SIZE = 512
MAX_WORDS = 24
CACHE_LIMIT = 400  # swatches the cache keeps; the oldest go first
VERSION = 1  # part of every cache key: raise it when what is cached changes
MARKS = "™®©"  # trademark, registered, copyright


# -- the description ----------------------------------------------------------------------------------------------

_REQUEST = {"please", "pls", "hi", "hello", "hey", "can", "could", "would", "will", "you", "make", "give", "show",
            "create", "generate", "design", "get", "me", "us", "i", "i'd", "want", "need", "like", "a", "an", "the",
            "some", "of", ","}  # words in front of the description itself ("could you make me a texture of ...")
_DROP = {"please", "pls", "thanks", "thank", "seamless", "tileable", "tiling", "texture", "textures"}


def clean_text(text: str) -> str:
    """``text`` tidied for :data:`TEMPLATE`: odd characters become spaces (letters of any script, digits and
    ``- ' & # ,`` stay), lower case, single spaces, no request words in front ("please make me a ..."), none of the
    words the template already says, at most :data:`MAX_WORDS` words. Raises ``ValueError`` when no word is left."""
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = "".join(ch if ch.isalnum() or ch in "-'&#," else " " for ch in text)
    words = [w.strip("-'") for w in text.replace(",", " , ").split()]
    words = [w for w in words if w and w not in _DROP]
    while words and words[0] in _REQUEST:
        words.pop(0)
    out = " ".join(words[:MAX_WORDS]).replace(" , ", ", ").strip(" ,")
    out = re.sub(r"(, )+", ", ", out)
    if not any(ch.isalpha() for ch in out):
        raise ValueError(f"describe a fabric (its colours, pattern and material, e.g. \"blue paisley silk\"), "
                         f"not {str(text).strip()!r}")
    return out[:160].strip(" ,")


def fabric_prompt(text: str) -> str:
    """The prompt for the model: :data:`TEMPLATE` with the cleaned ``text``."""
    return TEMPLATE.format(text=clean_text(text))


class PromptRefused(ValueError):
    """A description the kiosk does not take: :attr:`message` says why, to show as it is; :attr:`word` is the word,
    :attr:`kind` its list."""

    def __init__(self, message: str, word: str, kind: str):
        super().__init__(message)
        self.message, self.word, self.kind = message, word, kind


_LEET = str.maketrans("013457@$!|", "oieastasii")


def _tokens(text: str) -> list[str]:
    """Words of ``text`` as plain letters: no accents, lower case, digits read as the letters they look like."""
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).lower().translate(_LEET)
    return re.findall(r"[a-z]+", text)


class PromptFilter:
    """Refuses descriptions that name a brand, a character, a logo or an offensive word: for a public kiosk.

    ``lists``: ``{kind: {"message": ..., "words": [...]}}`` (the form of ``fabric_words.json``; a kind ending in
    ``_rot13`` has ROT13-encoded words); ``allow``: words never refused. A word matches whole words of a description
    whatever their case, accents, spacing ("louis vuitton", "louisvuitton", "g u c c i") or digits standing in for
    letters ("gucc1"), and its plural; "cocktail" or "Sussex" do not match anything."""

    def __init__(self, lists: dict | None = None, allow=()):
        self.lists: dict[str, tuple[str, set[str]]] = {}
        self.allowed: set[str] = set()
        self.update(lists or {})
        self.allow(allow)

    @classmethod
    def load(cls, path=None) -> PromptFilter:
        """The shipped lists (``fabric_words.json``), plus the file ``path`` or ``$BIFOLD_FABRIC_WORDS`` if given (same
        form: its words are added to the lists of the same name, new lists are added, ``"allow"`` lets words through).
        A user file that cannot be read is left out with a warning."""
        out = cls(json.loads(WORDS_FILE.read_text(encoding="utf-8")))
        extra = path if path is not None else os.environ.get(WORDS_ENV)
        if extra:
            try:
                out.update(json.loads(Path(extra).expanduser().read_text(encoding="utf-8")))
            except (OSError, ValueError, AttributeError, TypeError) as exc:
                warnings.warn(f"the fabric word list {extra} is not used: {exc}", stacklevel=2)
        return out

    def update(self, data: dict) -> None:
        """Add the lists of ``data`` (the form of ``fabric_words.json``)."""
        for kind, entry in data.items():
            if kind == "allow":
                self.allow(entry)
            elif isinstance(entry, dict):
                self.add(kind, entry.get("words", []), entry.get("message"))

    def add(self, kind: str, words, message: str | None = None) -> None:
        """Refuse ``words`` too, as ``kind`` (with ``message``, or the kind's message if it has one)."""
        old_message, known = self.lists.get(kind, (None, set()))
        decode = kind.endswith("_rot13")
        for word in words:
            word = codecs.decode(str(word), "rot13") if decode else str(word)
            joined = "".join(_tokens(word))
            if joined:
                known.add(joined)
        message = message or old_message or "That description cannot be made into a fabric here ({word})."
        self.lists[kind] = (message, known)

    def allow(self, words) -> None:
        """Never refuse ``words``."""
        self.allowed.update("".join(_tokens(w)) for w in words)

    def problem(self, text: str) -> PromptRefused | None:
        """Why ``text`` is refused, or ``None``."""
        if any(mark in str(text) for mark in MARKS):
            message = self.lists.get("marks", ("Logos and trademarks can't be made into a fabric here ({word}).",))[0]
            return PromptRefused(message.format(word="™"), "™", "marks")
        tokens = _tokens(text)
        candidates: dict[str, str] = {}  # joined form -> as written
        for i in range(len(tokens)):
            for j in range(i + 1, min(len(tokens), i + 4) + 1):
                candidates.setdefault("".join(tokens[i:j]), " ".join(tokens[i:j]))
        letters: list[str] = []
        for token in [*tokens, ""]:  # "g u c c i"
            if len(token) == 1:
                letters.append(token)
                continue
            if len(letters) >= 2:
                candidates.setdefault("".join(letters), " ".join(letters))
            letters = []
        for joined, shown in list(candidates.items()):
            if len(joined) > 3 and joined.endswith("s"):
                candidates.setdefault(joined[:-1], shown)
        for kind, (message, words) in self.lists.items():
            for joined, shown in candidates.items():
                if joined in words and joined not in self.allowed:
                    return PromptRefused(message.format(word=f'"{shown}"'), shown, kind)
        return None

    def check(self, text: str) -> None:
        """Raise :class:`PromptRefused` if ``text`` is refused."""
        refused = self.problem(text)
        if refused is not None:
            raise refused


# -- the size of a fabric on the body ---------------------------------------------------------------------------------

FABRIC_SCALES = {
    # kind: (metres per repeat of a swatch, the words that name it). The largest print named wins; a material
    # alone is a close-up of its weave.
    "material": (0.08, ("silk", "satin", "chiffon", "linen", "cotton", "denim", "jeans", "twill", "jersey", "poplin",
                        "chambray", "velvet", "velour", "corduroy", "suede", "leather", "wool", "woollen", "flannel",
                        "tweed", "cashmere", "felt", "canvas", "knit", "knitted", "boucle", "lace", "mesh", "organza",
                        "taffeta", "crepe", "fleece", "nylon", "polyester", "sequins", "sequin", "glitter")),
    "small print": (0.1, ("pinstripe", "pinstripes", "pinstriped", "polka", "dot", "dots", "dotted", "spots",
                          "spotted", "ditsy", "gingham", "houndstooth", "dogtooth", "seersucker", "micro", "stars",
                          "hearts", "confetti")),
    "print": (0.2, ("paisley", "floral", "flowers", "flower", "roses", "daisies", "damask", "ikat", "batik",
                    "geometric", "check", "checks", "checked", "checkered", "plaid", "tartan", "argyle", "herringbone",
                    "chevron", "toile", "stripes", "striped", "stripe", "zigzag", "brocade", "jacquard", "aztec",
                    "moroccan", "tile", "tiles", "hexagons", "triangles", "squares", "circles", "waves", "scales")),
    "large print": (0.35, ("camo", "camouflage", "tropical", "hawaiian", "leopard", "cheetah", "zebra", "tiger",
                           "giraffe", "snakeskin", "animal", "tie", "dye", "marble", "marbled", "abstract", "botanical",
                           "palm", "palms", "leaves", "jungle", "watercolor", "watercolour", "landscape", "galaxy",
                           "graffiti")),
}
DEFAULT_SCALE = 0.18  # metres per repeat when no word says


def fabric_scale(text: str) -> float:
    """Metres per repeat for a swatch of ``text``, from the kind of fabric it names (:data:`FABRIC_SCALES`), times
    the size words of the keyword parser (``tiny``, ``small``, ``large``, ...)."""
    from .prompts import SCALES

    words = re.findall(r"[a-z]+", str(text).lower())
    named = [scale for scale, names in FABRIC_SCALES.values() if any(w in names for w in words)]
    prints = [s for s in named if s > FABRIC_SCALES["material"][0]]
    scale = max(prints) if prints else (named[0] if named else DEFAULT_SCALE)
    for word in words:
        scale *= SCALES.get(word, 1.0)
    return round(float(scale), 4)


# -- the ladder ---------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Rung:
    """One rung of the fallback ladder: a model (a preset name of :data:`~bifold_demo.people.generator.PRESETS`, or a
    Hugging Face id) at ``steps`` denoising steps (``None``: the preset's)."""

    model: str
    steps: int | None = None

    @property
    def preset(self):
        from ...people.generator import get_preset

        return get_preset(self.model)

    @property
    def tiling(self) -> str:
        """``"rolled"`` (latent rolling and a circular decoder: the diffusion transformers), ``"circular"`` (circular
        padding: SD-Turbo's UNet) or ``"auto"`` (another model: decided when it is loaded)."""
        if self.preset.pipeline in ("Flux2KleinPipeline", "Flux2Pipeline", "ZImagePipeline"):
            return "rolled"
        return "circular" if self.model == "sd-turbo" else "auto"

    @property
    def step_count(self) -> int:
        return int(self.steps or self.preset.steps)

    @property
    def method(self) -> str:
        return f"{self.tiling}, {self.step_count} steps"

    @property
    def label(self) -> str:
        return f"{self.preset.repo.rsplit('/', 1)[-1]} ({self.method})"


LADDERS = {
    "flux2-klein": (Rung("flux2-klein"), Rung("flux2-klein", 12), Rung("sd-turbo")),
    "z-image-turbo": (Rung("z-image-turbo"), Rung("z-image-turbo", 16), Rung("sd-turbo")),
    "sd-turbo": (Rung("sd-turbo"), Rung("sd-turbo", 4)),
}


def ladder_for(model: str | None) -> tuple[Rung, ...]:
    """The rungs tried for ``model`` (a preset name or a Hugging Face id; ``None`` or ``"none"``: none, the procedural
    fabric only). Raises ``ValueError`` for an unknown preset name."""
    if model in (None, "", "none"):
        return ()
    if model in LADDERS:
        return LADDERS[model]
    rung = Rung(model)
    _ = rung.preset  # (checks the name)
    return (rung,)


@dataclass
class Failure:
    """Why a model could not run: ``kind`` is ``"missing"`` (a package), ``"weights"`` (not downloaded, no network,
    not allowed), ``"memory"`` or ``"other"``; ``message`` says it with what to do."""

    kind: str
    message: str


def explain_failure(exc: BaseException, preset) -> Failure:
    """The :class:`Failure` of loading or running the model of ``preset`` that raised ``exc``."""
    text = " ".join(str(exc).split())[:240]
    name = preset.repo
    if isinstance(exc, ImportError) or "has no " in text and "diffusers" in text:
        hint = "pip install -U diffusers" if "has no " in text else 'pip install -e ".[people]"'
        return Failure("missing", f"{name} needs packages that are not installed ({text}): {hint}")
    lowered = text.lower()
    if isinstance(exc, MemoryError) or "out of memory" in lowered or "not enough memory" in lowered:
        return Failure("memory", f"{name} does not fit in memory (it needs {preset.memory}): "
                                 f"{type(exc).__name__}: {text}")
    chain = type(exc).__mro__
    network = any(c.__module__.split(".")[0] in ("httpx", "httpx2", "requests", "urllib3", "huggingface_hub")
                  for c in chain) or any(c.__name__.endswith(("HTTPError", "ConnectionError", "ConnectError",
                                                                "ProxyError", "Timeout", "TimeoutError"))
                                         for c in chain)
    if network or isinstance(exc, OSError):
        return Failure("weights", f"cannot load {name} ({type(exc).__name__}: {text}). Its weights come from "
                                  f"huggingface.co: check the network; if the model page says it is gated, accept "
                                  f"its terms there and run: hf auth login; to fetch the weights ahead of time: "
                                  f"hf download {name} (the model needs {preset.memory})")
    return Failure("other", f"{name} failed: {type(exc).__name__}: {text}")


def load_generator(model: str, device: str = "auto", download: bool = True):
    """The :class:`~bifold_demo.people.generator.ImageGenerator` of ``model``, shared with other users of the same
    model in this process; ``download=False`` uses only weights already downloaded."""
    from ...people.generator import shared_generator

    return shared_generator(model, device=device, local_files_only=not download)


# -- swatches -----------------------------------------------------------------------------------------------------


@dataclass
class FabricSwatch:
    """One swatch of a description: the image (RGB ``uint8``, one seamless repeat) and how it was made."""

    image: np.ndarray
    text: str  # the cleaned description
    method: str  # "rolled, 4 steps", "circular, 2 steps", "..., repaired", "procedural" or "plain colour"
    model: str | None = None  # the Hugging Face id of the model that made it; None when no model did
    seed: int | None = None
    prompt: str | None = None  # what the model was given
    seam: float = 0.0  # its seam_score
    scale: float = DEFAULT_SCALE  # metres per repeat
    rotation: float = 0.0
    note: str = ""  # what it is, in a few words
    preview: bool = False  # the quick procedural preview
    ok: bool = True  # its seams are within the threshold
    cached: bool = False
    seconds: float = 0.0
    path: Path | None = None  # its cache file

    @property
    def generated(self) -> bool:
        return self.model is not None

    def describe(self) -> str:
        """How it was made, in a line: ``"FLUX.2-klein-4B, seed 2, rolled, 4 steps, seam 0.91"`` or ``"procedural:
        tartan in #c02028 and #18423a"``."""
        if self.generated:
            return f"{self.model.rsplit('/', 1)[-1]}, seed {self.seed}, {self.method}, seam {self.seam:.2f}"
        return f"{self.method}: {self.note}" if self.note else self.method

    def fill(self, scale: float | None = None, rotation: float | None = None) -> Fabric:
        """This swatch as a :class:`Fabric` fill (its own scale and rotation unless given)."""
        return Fabric(self.image, self.text, self.scale if scale is None else scale,
                      self.rotation if rotation is None else rotation, self.model, self.seed, self.method, self.seam,
                      self.prompt)


class Fabric(Swatch):
    """A fabric made from a description: a :class:`~.fills.Swatch` (``image`` tiled at ``scale`` metres per repeat,
    turned ``rotation`` degrees) that keeps ``text``, the ``model``, ``seed`` and ``prompt`` that made it, the
    ``method`` and its ``seam`` score. Its spec holds all of it and the image, so it is rebuilt without the model (see
    the module docs)."""

    def __init__(self, image, text: str, scale: float | None = None, rotation: float = 0.0, model: str | None = None,
                 seed: int | None = None, method: str = "generated", seam: float | None = None,
                 prompt: str | None = None):
        super().__init__(image, fabric_scale(text) if scale is None else scale, rotation or 0.0, seamless=True)
        self.source = None  # the spec carries the image itself, not a file that may go away
        self.text, self.model, self.method, self.prompt = str(text), model, str(method), prompt
        self.seed = None if seed is None else int(seed)
        self.seam = None if seam is None else float(seam)
        self.notes: list[str] = []  # what the generator said while making it (fabric_fill)
        self._uri: str | None = None

    def spec(self) -> dict:
        out = {"fabric": self.text, "method": self.method}
        if self.model is not None:
            out["model"] = self.model
        if self.seed is not None:
            out["seed"] = self.seed
        if self.prompt is not None:
            out["prompt"] = self.prompt
        out["scale"] = round(self.scale, 5)
        if self.rotation:
            out["rotation"] = self.rotation
        if self.seam is not None:
            out["seam"] = round(self.seam, 3)
        if self._uri is None:
            self._uri = data_uri(self.image, "webp")
        out["swatch"] = self._uri
        return out

    @classmethod
    def from_spec(cls, spec: dict) -> Fabric:
        """The fabric of a spec (see :meth:`spec`); without an image (``"swatch"``), the keyword fabric of its words."""
        text = str(spec["fabric"])
        scale, rotation = spec.get("scale"), spec.get("rotation")
        if spec.get("swatch"):
            return cls(spec["swatch"], text, scale, rotation or 0.0, spec.get("model"), spec.get("seed"),
                       spec.get("method", "generated"), spec.get("seam"), spec.get("prompt"))
        return procedural(text).fill(scale, rotation)

    def __repr__(self) -> str:
        made = f"{self.model}, seed {self.seed}, " if self.model else ""
        return f"Fabric({self.text!r}, {made}{self.method}, scale={self.scale:g})"


def _uint8(rgb) -> np.ndarray:
    return np.clip(np.asarray(rgb, np.float32) + 0.5, 0, 255).astype(np.uint8)


@lru_cache(maxsize=1)
def _known_words() -> frozenset[str]:
    from . import prompts
    from .fills import SHADES

    words = {"and", "on", "with", "of", "in", "a", "an", "the", "or"}
    for table in (COLORS, SHADES, prompts.PATTERN_WORDS, prompts.SCALES, prompts.ROTATIONS, prompts.CAMO_PALETTES):
        for key in table:
            words.update(key.split())
    return frozenset(words)


def nearest_colour(text: str) -> tuple[tuple[int, int, int], str, str | None]:
    """The colour of the first word of ``text`` that is close to a colour name (``"turqoise"``: turquoise):
    ``(rgb, name, the word)``; grey and ``None`` when there is none."""
    names = [name for name in COLORS if " " not in name]
    for word in re.findall(r"[a-z]+", str(text).lower()):
        match = difflib.get_close_matches(word, names, n=1, cutoff=0.8)
        if match:
            return COLORS[match[0]], match[0], word
    return COLORS["grey"], "grey", None


def procedural(text: str, preview: bool = False) -> FabricSwatch:
    """The swatch of ``text`` without a model: the keyword parser's pattern or colour (method ``"procedural"``), or a
    plain colour (``"plain colour"``) when it knows no word. It is the quick preview, and the ladder's last rungs."""
    from .prompts import parse_fill

    unknown = [w for w in re.findall(r"[a-z]+", text.lower()) if w not in _known_words()]
    left = f"; the keyword parser does not know: {', '.join(dict.fromkeys(unknown))}" if unknown else ""
    try:
        fill = parse_fill(text)
    except ValueError:
        fill = None
    if isinstance(fill, Pattern):
        colours = " and ".join(hex_color(c) for c in (fill.primary, fill.secondary))
        image = _uint8(fill.image)
        return FabricSwatch(image, text, "procedural", seam=seam_score(image), scale=fill.scale,
                            rotation=fill.rotation, note=f"{fill.kind} in {colours}{left}", preview=preview)
    if isinstance(fill, Solid):
        image = np.full((16, 16, 3), fill.rgb, np.uint8)
        return FabricSwatch(image, text, "procedural", scale=fabric_scale(text),
                            note=f"plain {hex_color(fill.rgb)}{left}", preview=preview)
    rgb, name, word = nearest_colour(text)
    image = np.full((16, 16, 3), rgb, np.uint8)
    why = f"{word!r} looks like {name!r}" if word and word != name else \
        ("no colour or pattern word known" if word is None else f"the colour {name}")
    return FabricSwatch(image, text, "plain colour", scale=fabric_scale(text), note=f"plain {name} ({why})",
                        preview=preview)


# -- the cache ----------------------------------------------------------------------------------------------------


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")[:40] or "fabric"


class FabricCache:
    """Finished swatches on disk, in ``folder`` (``$BIFOLD_FABRIC_CACHE``, else ``fabrics/``): a PNG and a JSON file
    each, named by the description and a key of the prompt, model, seed, size and method. At most ``limit`` swatches
    are kept (the oldest go first); a damaged entry counts as missing."""

    def __init__(self, folder=None, limit: int = CACHE_LIMIT):
        self.folder = Path(folder if folder is not None else os.environ.get(CACHE_ENV) or DEFAULT_CACHE).expanduser()
        self.limit = int(limit)

    @staticmethod
    def key(prompt: str, model: str, seed: int, size: int, method: str) -> str:
        data = json.dumps([VERSION, prompt, model, int(seed), int(size), method], ensure_ascii=False)
        return hashlib.sha256(data.encode("utf-8")).hexdigest()[:20]

    def paths(self, key: str, text: str) -> tuple[Path, Path]:
        stem = self.folder / f"{_slug(text)}-{key}"
        return stem.with_suffix(".png"), stem.with_suffix(".json")

    def get(self, key: str, text: str) -> FabricSwatch | None:
        png, meta_path = self.paths(key, text)
        if not meta_path.is_file():
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("key") != key:
                return None
            bgr = cv2.imread(str(png), cv2.IMREAD_COLOR)
            if bgr is None:
                raise OSError(f"cannot read {png}")
            return FabricSwatch(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), meta["text"], meta["method"], meta.get("model"),
                                meta.get("seed"), meta.get("prompt"), float(meta["seam"]), float(meta["scale"]),
                                float(meta.get("rotation", 0.0)), meta.get("note", ""), ok=bool(meta["ok"]),
                                cached=True, seconds=float(meta.get("seconds", 0.0)), path=png)
        except (OSError, ValueError, KeyError, TypeError):
            for path in (png, meta_path):  # damaged: made again
                path.unlink(missing_ok=True)
            return None

    def put(self, key: str, swatch: FabricSwatch) -> Path:
        png, meta_path = self.paths(key, swatch.text)
        self.folder.mkdir(parents=True, exist_ok=True)
        tmp = png.with_name(f".{png.stem}.{os.getpid()}.{threading.get_ident()}.png")
        if not cv2.imwrite(str(tmp), cv2.cvtColor(swatch.image, cv2.COLOR_RGB2BGR)):
            raise OSError(f"cannot write {tmp}")
        os.replace(tmp, png)
        meta = {"key": key, "version": VERSION, "text": swatch.text, "prompt": swatch.prompt, "model": swatch.model,
                "seed": swatch.seed, "size": int(swatch.image.shape[0]), "method": swatch.method,
                "seam": round(float(swatch.seam), 4), "ok": bool(swatch.ok), "scale": swatch.scale,
                "rotation": swatch.rotation, "note": swatch.note, "seconds": round(swatch.seconds, 2),
                "created": time.time()}
        tmp = meta_path.with_name(f".{meta_path.stem}.{os.getpid()}.{threading.get_ident()}.json")
        tmp.write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, meta_path)
        self._prune()
        return png

    def _prune(self) -> None:
        def age(path: Path) -> float:
            try:
                return path.stat().st_mtime
            except OSError:  # (gone meanwhile)
                return 0.0

        entries = sorted(self.folder.glob("*.json"), key=age)
        for meta_path in entries[:max(0, len(entries) - self.limit)]:
            meta_path.with_suffix(".png").unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)


# -- jobs ---------------------------------------------------------------------------------------------------------

_END = object()


class FabricJob:
    """A description being made into swatches on a worker thread (see :meth:`FabricGenerator.start`).

    :attr:`results` grows as swatches are ready (the preview first); :meth:`poll` returns the new ones without
    waiting, iterating waits for each (use one or the other); :attr:`ranked` and :attr:`best` order the variations by
    their seam score. :attr:`messages` says what happened (each step down of
    the ladder and why), :attr:`progress` how far the current image is (``{"model", "seed", "step", "steps"}``),
    :attr:`status` is ``"running"``, ``"done"``, ``"cancelled"`` or ``"failed"`` (:attr:`error`). ``on_result`` and
    ``on_message`` are called on the worker thread."""

    def __init__(self, text: str, prompt: str, seeds: list[int], size: int, on_result=None, on_message=None):
        self.text, self.prompt, self.seeds, self.size = text, prompt, list(seeds), int(size)
        self.results: list[FabricSwatch] = []
        self.messages: list[str] = []
        self.progress: dict = {}
        self.status = "running"
        self.error: BaseException | None = None
        self.thread: threading.Thread | None = None
        self._on_result, self._on_message = on_result, on_message
        self._queue: queue.Queue = queue.Queue()
        self._cancel = threading.Event()
        self._finished = threading.Event()

    def cancel(self) -> None:
        """Stop at the end of the next denoising step (or before the next image); the swatches so far stay."""
        self._cancel.set()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    @property
    def done(self) -> bool:
        return self._finished.is_set()

    def poll(self) -> list[FabricSwatch]:
        """The swatches made since the last poll, without waiting."""
        out = []
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                return out
            if item is not _END:
                out.append(item)

    def __iter__(self):
        while True:
            item = self._queue.get()
            if item is _END:
                return
            yield item

    def wait(self, timeout: float | None = None) -> bool:
        """Wait until the job is over (at most ``timeout`` seconds); True if it is."""
        return self._finished.wait(timeout)

    @property
    def variations(self) -> list[FabricSwatch]:
        """The swatches made, without the preview."""
        return [s for s in self.results if not s.preview]

    @property
    def ranked(self) -> list[FabricSwatch]:
        """The variations best first: those a model made, by their seam score, then the rest."""
        return sorted(self.variations, key=lambda s: (not s.generated, s.seam))

    @property
    def best(self) -> FabricSwatch | None:
        """The first of :attr:`ranked`; the preview while there is none."""
        ranked = self.ranked
        if ranked:
            return ranked[0]
        return self.results[0] if self.results else None

    # the worker's side

    def _emit(self, swatch: FabricSwatch) -> None:
        self.results.append(swatch)
        self._queue.put(swatch)
        if self._on_result is not None:
            self._on_result(swatch)

    def _say(self, message: str) -> None:
        self.messages.append(message)
        if self._on_message is not None:
            self._on_message(message)

    def _check(self) -> None:
        if self._cancel.is_set():
            raise Cancelled()

    def _finish(self, status: str) -> None:
        self.status = status
        self._finished.set()
        self._queue.put(_END)


def check_size(size: int) -> int:
    """``size`` as the side of a swatch: a multiple of 16 from 64 to 1024 (512-768 is what the models do best)."""
    size = int(size)
    if size % 16 or not 64 <= size <= 1024:
        raise ValueError(f"a swatch is 64 to 1024 pixels wide, a multiple of 16 (512-768 is best), not {size}")
    return size


def _takes(pipe, name: str) -> bool:
    """Whether calling ``pipe`` takes the keyword argument ``name``."""
    try:
        parameters = inspect.signature(pipe).parameters
    except (TypeError, ValueError):
        return True
    return name in parameters or any(p.kind == p.VAR_KEYWORD for p in parameters.values())


class FabricGenerator:
    """Makes seamless fabric swatches from descriptions (see the module docs).

    ``model``: the first rung's model (``"flux2-klein"``, ``"z-image-turbo"``, ``"sd-turbo"``, a Hugging Face id, or
    ``None`` / ``"none"`` for the procedural fabric only); ``ladder``: other rungs instead of :func:`ladder_for`'s.
    ``seeds`` swatches of ``size`` pixels per description; ``threshold``: the seam score above which a swatch is left
    out. ``cache``: ``True`` (``fabrics/``), a folder, or ``False``. ``words``: the :class:`PromptFilter` (``True``:
    :meth:`PromptFilter.load`; ``None`` or ``False``: no filter). ``loader(model, device, download)`` loads a model
    (:func:`load_generator`; tests pass stand-ins); ``download=False`` never downloads weights. ``pace()``: called
    after every denoising step and between images."""

    def __init__(self, model: str | None = DEFAULT_MODEL, device: str = "auto", seeds: int = SEEDS, size: int = SIZE,
                 threshold: float = SEAM_THRESHOLD, cache=True, words=True, loader=None, download: bool = True,
                 pace=None, ladder=None):
        self.model = None if model in (None, "", "none") else str(model)
        self.ladder = tuple(ladder) if ladder is not None else ladder_for(self.model)
        self.device, self.download, self.pace = device, bool(download), pace
        self.seeds, self.size, self.threshold = int(seeds), check_size(size), float(threshold)
        self.cache = FabricCache() if cache is True else (FabricCache(cache) if cache else None)
        self.words = PromptFilter.load() if words is True else (words or None)
        self.loader = loader or load_generator
        self._loaded: dict[str, object] = {}  # model -> its ImageGenerator, or the Failure of loading it
        self._load_lock = threading.Lock()

    def prepare(self, text: str) -> tuple[str, str]:
        """``(cleaned text, prompt)`` of a description; raises :class:`PromptRefused` or ``ValueError``."""
        if self.words is not None:
            self.words.check(text)
        cleaned = clean_text(text)
        if self.words is not None:
            self.words.check(cleaned)
        return cleaned, TEMPLATE.format(text=cleaned)

    def preview(self, text: str) -> FabricSwatch:
        """The quick procedural swatch of a (cleaned) description."""
        return procedural(text, preview=True)

    def start(self, text: str, seeds: int | None = None, first_seed: int = 0, size: int | None = None,
              on_result=None, on_message=None) -> FabricJob:
        """Start making swatches of ``text`` on a worker thread: ``seeds`` of them (seeds ``first_seed`` on) of
        ``size`` pixels. Returns the :class:`FabricJob` at once; a refused or empty description raises at once
        (:class:`PromptRefused`, ``ValueError``)."""
        cleaned, prompt = self.prepare(text)
        count = self.seeds if seeds is None else int(seeds)
        if count < 1:
            raise ValueError("ask for at least one seed")
        job = FabricJob(cleaned, prompt, [int(first_seed) + k for k in range(count)],
                        check_size(size or self.size), on_result, on_message)
        job.thread = threading.Thread(target=self._work, args=(job,), name=f"fabric: {cleaned[:30]}", daemon=True)
        job.thread.start()
        return job

    def generate(self, text: str, **kwargs) -> FabricJob:
        """:meth:`start` and wait until the job is over."""
        job = self.start(text, **kwargs)
        job.wait()
        return job

    # the worker

    def _work(self, job: FabricJob) -> None:
        status = "done"
        try:
            job._emit(self.preview(job.text))
            self._ladder(job)
        except Cancelled:
            status = "cancelled"
            job._say("cancelled")
        except Exception as exc:  # noqa: BLE001 - kept on the job; the caller still gets a fabric
            status, job.error = "failed", exc
            job._say(f"failed: {type(exc).__name__}: {exc}; the procedural fabric instead")
            job._emit(self._fallback(job))
        finally:
            job._finish(status)

    def _ladder(self, job: FabricJob) -> None:
        if not self.ladder:
            job._say("no model asked for: the procedural fabric")
            job._emit(self._fallback(job))
            return
        rejected: list[FabricSwatch] = []
        failed: set[str] = set()  # models that failed (out of memory, an error): their other rungs are not tried
        for index, rung in enumerate(self.ladder):
            if rung.model in failed:
                continue
            outcome = self._rung(job, index, rejected, failed)
            if outcome == "kept":
                return
            if outcome == "stop":
                break
        job._check()
        if rejected:
            best = min(rejected, key=lambda s: s.seam)
            repaired = self._repair(best)
            if repaired.ok:
                job._say(f"no seed came out seamless: the best one ({best.describe()}, seam {best.seam:.2f}) "
                         f"repaired without a model (seam {repaired.seam:.2f})")
                job._emit(repaired)
                return
            job._say(f"the repair left a seam (score {repaired.seam:.2f}); the procedural fabric instead")
        job._emit(self._fallback(job))

    def _next(self, index: int, failed: set, rejected: list) -> str:
        """What the ladder tries after rung ``index`` (for the messages)."""
        for rung in self.ladder[index + 1:]:
            if rung.model not in failed:
                return rung.label
        return "the seam repair" if rejected else "the procedural fabric"

    def _rung(self, job: FabricJob, index: int, rejected: list, failed: set) -> str:
        """Every seed on rung ``index``: ``"kept"`` (some swatches kept), ``"seams"`` (all left out, for
        ``rejected``), ``"failed"`` (its model failed, out of memory or with an error: added to ``failed``) or
        ``"stop"`` (packages, weights or the network missing: no other model is tried)."""
        rung = self.ladder[index]
        generator, kept = None, 0
        for seed in job.seeds:
            job._check()
            key = FabricCache.key(job.prompt, rung.preset.repo, seed, job.size, rung.method)
            swatch = self.cache.get(key, job.text) if self.cache is not None else None
            if swatch is None:
                if generator is None:
                    generator = self._load(job, rung)
                    if isinstance(generator, Failure):
                        return self._failed(job, generator, index, rejected, failed)
                    job._check()
                try:
                    swatch = self._make(job, generator, rung, seed)
                except Cancelled:
                    raise
                except Exception as exc:  # noqa: BLE001 - a step down of the ladder
                    return self._failed(job, explain_failure(exc, rung.preset), index, rejected, failed, seed)
                if self.cache is not None:
                    swatch.path = self.cache.put(key, swatch)
            if swatch.ok:
                kept += 1
                job._emit(swatch)
            else:
                rejected.append(swatch)
                job._say(f"{rung.label}, seed {seed}: a seam shows (score {swatch.seam:.2f} > {self.threshold:g}); "
                         f"left out")
            if self.pace is not None:
                self.pace()
        if kept:
            return "kept"
        job._say(f"{rung.label}: a seam shows on all {len(job.seeds)} seeds; trying "
                 f"{self._next(index, failed, rejected)}")
        return "seams"

    def _failed(self, job: FabricJob, failure: Failure, index: int, rejected: list, failed: set,
                seed: int | None = None) -> str:
        """Say why rung ``index`` failed and what comes next (see :meth:`_rung`)."""
        rung = self.ladder[index]
        where = "" if seed is None else f"{rung.label}, seed {seed}: "
        if failure.kind in ("missing", "weights"):
            then = "the seam repair of the best image so far" if rejected else "the procedural fabric instead"
            job._say(f"{where}{failure.message}; {then}")
            return "stop"
        failed.add(rung.model)
        job._say(f"{where}{failure.message}; trying {self._next(index, failed, rejected)}")
        return "failed"

    def _load(self, job: FabricJob, rung: Rung):
        """The loaded model of ``rung`` (an ImageGenerator), or the :class:`Failure` of loading it (kept, so a model
        that cannot load is not tried again by this generator)."""
        with self._load_lock:
            loaded = self._loaded.get(rung.model)
            if loaded is None:
                preset = rung.preset
                job._say(f"loading {preset.repo}, which needs {preset.memory} ...")
                try:
                    loaded = self.loader(rung.model, self.device, self.download)
                except Exception as exc:  # noqa: BLE001 - classified: a step down of the ladder
                    loaded = explain_failure(exc, preset)
                self._loaded[rung.model] = loaded
            return loaded

    def _make(self, job: FabricJob, generator, rung: Rung, seed: int) -> FabricSwatch:
        """One image of the model, tiled (see :mod:`.tiling`), scored and finished."""
        start = time.perf_counter()
        steps = rung.step_count
        plan = plan_for(generator.pipe)
        roller = LatentRoller(steps, seed, plan.roll) if plan.roll else None
        job.progress = {"model": rung.label, "seed": seed, "step": 0, "steps": steps}
        callback = StepCallback(roller, job._cancel.is_set, lambda step, total: job.progress.update(step=step,
                                                                                                    steps=total),
                                self.pace)
        kwargs = {}
        if _takes(generator.pipe, "callback_on_step_end"):
            kwargs = {"callback_on_step_end": callback, "callback_on_step_end_tensor_inputs": ["latents"]}
        with generator.lock, circular_padding(*plan.circular):
            rgb = generator.rgb(job.prompt, seed, job.size, job.size, steps=steps, **kwargs)
        job._check()  # (a stand-in without the callback cannot be stopped half way)
        rgb = _uint8(np.asarray(rgb)[..., :3])
        seam = seam_score(rgb)
        ok = seam <= self.threshold
        if ok:  # kept: even light; left out: as it came, for the repair
            rgb = _uint8(equalise_light(rgb))
            seam = seam_score(rgb)
            ok = seam <= self.threshold
        return FabricSwatch(rgb, job.text, rung.method, generator.preset.repo, seed, job.prompt, seam,
                            fabric_scale(job.text), ok=ok, seconds=time.perf_counter() - start)

    def _repair(self, swatch: FabricSwatch) -> FabricSwatch:
        """``swatch`` made seamless without a model, with even light."""
        start = time.perf_counter()
        image = _uint8(equalise_light(make_seamless(swatch.image)))
        seam = seam_score(image)
        return replace(swatch, image=image, method=f"{swatch.method}, repaired", seam=seam,
                       ok=seam <= self.threshold, cached=False, path=None,
                       seconds=swatch.seconds + time.perf_counter() - start)

    def _fallback(self, job: FabricJob) -> FabricSwatch:
        swatch = procedural(job.text)
        job._say(f"{swatch.method}: {swatch.note}")
        return swatch


_default: FabricGenerator | None = None
_default_lock = threading.Lock()


def default_generator() -> FabricGenerator:
    """The generator :func:`fabric_fill` uses when given none: ``$BIFOLD_FABRIC_MODEL`` (``"none"``: no model), else
    :data:`DEFAULT_MODEL`; made once."""
    global _default
    with _default_lock:
        if _default is None:
            _default = FabricGenerator(os.environ.get(MODEL_ENV, DEFAULT_MODEL))
        return _default


def fabric_fill(text: str, generator: FabricGenerator | None = None, scale: float | None = None,
                rotation: float | None = None, seed: int = 0, seeds: int | None = None,
                size: int | None = None) -> Fabric:
    """A fill made from a description: the best swatch ``generator`` (:func:`default_generator` when ``None``) makes of
    ``text`` (``seeds`` of them from ``seed`` on), as a :class:`Fabric` at its own scale for the kind of fabric unless
    ``scale`` is given. It waits for the job. When no model made it, it says so (a warning with the reason) and the
    fabric is the ladder's fallback; :attr:`Fabric.notes` has all the job said."""
    generator = generator if generator is not None else default_generator()
    job = generator.generate(text, seeds=seeds, first_seed=seed, size=size)
    swatch = job.best
    fill = swatch.fill(scale, rotation)
    fill.notes = list(job.messages)
    if not swatch.generated:
        reason = next((m for m in reversed(job.messages[:-1]) if "procedural" in m or "model" in m),
                      job.messages[0] if job.messages else "")
        warnings.warn(f"no model made a fabric of {job.text!r} ({reason}); {swatch.describe()}", stacklevel=2)
    return fill


__all__ = ["DEFAULT_MODEL", "FABRIC_SCALES", "LADDERS", "TEMPLATE", "Fabric", "FabricCache", "FabricGenerator",
           "FabricJob", "FabricSwatch", "Failure", "PromptFilter", "PromptRefused", "Rung", "check_size",
           "clean_text", "default_generator", "explain_failure", "fabric_fill", "fabric_prompt", "fabric_scale",
           "ladder_for", "load_generator", "nearest_colour", "procedural"]
