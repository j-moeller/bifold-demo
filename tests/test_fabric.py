"""Fabrics from a description (outfits, stage 2): the seam score, the model-free repair and even light, latent rolling
and circular padding, the fallback ladder, the job API, the cache, the kiosk filter, the fill's spec and the CLI.

No text-to-image model can run here, so the generative path runs on stand-ins: fake pipelines whose loops have the
latent shapes of diffusers' Flux2KleinPipeline, ZImagePipeline and an SD UNet pipeline and call
``callback_on_step_end`` as they do, and tiny randomly initialised diffusers models built in the test (no
downloads). Synthetic textures stand in for generated fabric."""

import codecs
import itertools
import json
import os
import threading
import time
import types
import weakref
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from bifold_demo.characters.outfit import Outfit, Swatch, fabric
from bifold_demo.characters.outfit import patterns as P
from bifold_demo.characters.outfit import regions as R
from bifold_demo.characters.outfit.fabric import (
    Fabric,
    FabricCache,
    FabricGenerator,
    PromptFilter,
    PromptRefused,
    Rung,
    clean_text,
    explain_failure,
    fabric_fill,
    fabric_prompt,
    fabric_scale,
    procedural,
)
from bifold_demo.characters.outfit.fills import from_spec
from bifold_demo.characters.outfit.seamless import (
    SEAM_THRESHOLD,
    equalise_light,
    light_of,
    make_seamless,
    resize_periodic,
    seam_score,
    seam_scores,
)
from bifold_demo.characters.outfit.tiling import (
    Cancelled,
    LatentRoller,
    StepCallback,
    circular_padding,
    plan_for,
    roll_offsets,
)
from bifold_demo.people.generator import PRESETS, ImageGenerator

torch = pytest.importorskip("torch")
CHARACTERS = Path(os.environ.get("BIFOLD_CHARACTERS_TEST_DIR", "/nonexistent")).expanduser()


@pytest.fixture(autouse=True)
def no_models_or_shared_folders(tmp_path, monkeypatch):
    """No test loads a real model or writes the fabric cache or word list outside its temporary folder."""
    monkeypatch.setenv(fabric.CACHE_ENV, str(tmp_path / "fabrics"))
    monkeypatch.setenv(fabric.MODEL_ENV, "none")
    monkeypatch.delenv(fabric.WORDS_ENV, raising=False)
    monkeypatch.setattr(fabric, "_default", None)

    def no_loading(model, device="auto", download=True):
        raise AssertionError(f"a test tried to load {model}")

    monkeypatch.setattr(fabric, "load_generator", no_loading)


# -- synthetic fabric -----------------------------------------------------------------------------------------------


def periodic(size, cycles, seed, colours=((40, 60, 120), (210, 190, 150))):
    """Smooth noise that tiles (random phases, low-pass filtered), in two colours: a stand-in for a seamless swatch."""
    rng = np.random.default_rng(seed)
    f = np.fft.fftfreq(size) * size
    radius = np.hypot(f[:, None], f[None, :])
    field = np.real(np.fft.ifft2(np.fft.fft2(rng.standard_normal((size, size))) * np.exp(-(radius / cycles) ** 2)))
    field = (field - field.min()) / np.ptp(field)
    c0, c1 = (np.asarray(c, np.float32) for c in colours)
    return (c0 * (1 - field[..., None]) + c1 * field[..., None]).astype(np.float32)


def cut_out(size, cycles, seed, colours=((40, 60, 120), (210, 190, 150))):
    """The middle of a larger texture: it does not tile (what a model without tiling paints)."""
    return periodic(size * 3, cycles * 3, seed, colours)[size:2 * size, size:2 * size].copy()


def repeat_of(tile, size, offset=(0, 0)):
    """A ``size`` crop of ``tile`` repeated, starting at ``offset``: tiles only if the tile's size divides ``size``."""
    reps = size // tile.shape[0] + 2
    big = np.tile(tile, (reps, reps, 1))
    return big[offset[0]:offset[0] + size, offset[1]:offset[1] + size].copy()


def bands(kind, period, phase, size=512):
    """Wide two-colour stripes, checks or gingham of ``period`` pixels: they tile only if the period divides
    ``size``; otherwise one band at the wrap is the wrong width."""
    x = np.arange(size)
    on = (((x + phase) // (period / 2)) % 2).astype(bool)
    navy, white, red = np.float32([30, 40, 90]), np.float32([240, 238, 230]), np.float32([170, 30, 40])
    if kind == "stripes":
        return np.where(on[None, :, None], navy, white) * np.ones((size, 1, 1), np.float32)
    if kind == "check":
        return np.where((on[None, :] ^ on[:, None])[..., None], red, white)
    both = on[None, :].astype(np.float32) + on[:, None]
    return white * (1 - both[..., None] / 2) + red * (both[..., None] / 2)


def motifs(size, count, radius, ground, colour, seed, tiling=True):
    """Ellipses of ``colour`` on a plain ``ground``, placed at random: a print, evenly lit. ``tiling``: drawn so that
    they wrap round the edges (it tiles); else cut out of a larger piece (it does not)."""
    rng = np.random.default_rng(seed)
    side = size if tiling else 3 * size
    image = np.full((side, side, 3), ground, np.float32)
    shifts = (-size, 0, size) if tiling else (0,)
    for _ in range(count if tiling else 9 * count):
        cx, cy, angle = rng.uniform(0, side), rng.uniform(0, side), rng.uniform(0, 180)
        for dx in shifts:
            for dy in shifts:
                cv2.ellipse(image, (int(cx + dx), int(cy + dy)), (radius, int(radius * 0.6)), angle, 0, 360, colour,
                            -1, cv2.LINE_AA)
    return image if tiling else image[size:2 * size, size:2 * size].copy()


def lit(image, amount=0.4, seed=0):
    """``image`` under light that varies once across it (tiling), ``exp(amount)`` from its darkest to its lightest."""
    h, w = image.shape[:2]
    y, x = (np.mgrid[0:h, 0:w] + 0.5) / np.float32([h, w])[:, None, None]
    a = np.random.default_rng(seed).uniform(0, 2 * np.pi, 3)
    field = np.cos(2 * np.pi * x + a[0]) + 0.6 * np.cos(2 * np.pi * y + a[1]) + 0.4 * np.cos(2 * np.pi * (x + y) + a[2])
    return np.clip(image * np.exp(amount * field / np.ptp(field))[..., None], 0, 255).astype(np.float32)


def spread(image, where):
    """How unevenly the pixels ``where`` (the plain ground of a print) are lit: their 99th over 1st percentile."""
    values = image[where].mean(axis=1)
    return float(np.percentile(values, 99) / max(np.percentile(values, 1), 1e-3))


def ground_of(image, colour):
    """Where ``image`` is its plain ground of ``colour``."""
    return np.all(np.abs(image - np.float32(colour)) < 1, axis=2)


# -- the description and the kiosk filter ------------------------------------------------------------------------


def test_descriptions_are_tidied_and_put_in_the_template():
    assert clean_text("  Could you make me a SEAMLESS,  tileable texture of red-and-white gingham, please :) ") == \
        "red-and-white gingham"
    assert clean_text("I want: green & gold brocade!!!") == "green & gold brocade"
    assert clean_text("#1f2a44 and white stripes") == "#1f2a44 and white stripes"
    assert clean_text("paint splatter print") == "paint splatter print"  # (not a request word)
    assert clean_text("Bleu marine, à pois blancs") == "bleu marine, à pois blancs"
    assert len(clean_text(" ".join(["blue"] * 100)).split()) == fabric.MAX_WORDS
    assert fabric_prompt("blue paisley silk") == ("seamless tileable fabric texture, flat lay, top-down, even diffuse "
                                                  "light, no folds, blue paisley silk")
    for empty in ("", "  please  ", "!!! ???", "seamless texture"):
        with pytest.raises(ValueError):
            clean_text(empty)


def test_the_kiosk_refuses_brands_characters_logos_and_offensive_words(tmp_path, monkeypatch):
    words = PromptFilter.load()
    refused = {"a Gucci print": ("brands", "gucci"), "LouisVuitton monogram": ("brands", "louisvuitton"),
               "g u c c i": ("brands", "g u c c i"), "gucc1 stripes": ("brands", "gucci"),
               "gucci!": ("brands", "gucci"),
               "levi's denim": ("brands", "levi"), "Dolce and Gabbana print": ("brands", "dolce and gabbana"),
               "Saint Laurent monogram": ("brands", "saint laurent"), "Vuitton canvas": ("brands", "vuitton"),
               "Hello Kitty pink": ("characters", "hello kitty"), "pokémon pattern": ("characters", "pokemon"),
               "pikachus print": ("characters", "pikachus"), "Mickey print": ("characters", "mickey"),
               "sex pistols tartan": ("characters", "sex pistols"), "with our logo": ("marks", "logo"),
               "brand™ stripes": ("marks", "™"), "nazi uniform": ("offensive", "nazi"),
               "f*ck you": ("offensive", "f*ck"), "fuk": ("offensive", "fuk"), "sh!t": ("offensive", "shit")}
    for text, (kind, word) in refused.items():
        problem = words.problem(text)
        assert problem is not None and (problem.kind, problem.word) == (kind, word), text
        assert problem.message and "{" not in problem.message
    assert '"gucci"' in words.problem("gucci").message  # brands are named, offensive words are not
    assert "nazi" not in words.problem("nazi").message
    slur = codecs.decode("avttre", "rot13")  # (kept ROT13-encoded in the word list)
    assert words.problem(f"{slur} print").kind == "offensive_rot13"
    for fine in ("cocktail party print", "Sussex check", "off white linen", "peacock feathers", "gore tex shell",
                 "Ferrari red", "unisex jersey", "Scunthorpe tweed", "blue paisley silk", "#1f2a44 pinstripe",
                 "titmouse birds", "polka dots!!", "100% cotton", "5* hotel stripes"):
        assert words.problem(fine) is None, fine
    with pytest.raises(PromptRefused, match="Brand names"):
        FabricGenerator(None, cache=False).start("a nike swoosh")

    extra = tmp_path / "words.json"  # the lists are extensible and configurable
    extra.write_text(json.dumps({"brands": {"words": ["Acme Cloth"]}, "rivals": {"message": "no {word}",
                                                                              "words": ["zorblax"]},
                                 "allow": ["hermes"]}))
    monkeypatch.setenv(fabric.WORDS_ENV, str(extra))
    mine = PromptFilter.load()
    assert mine.problem("acmecloth tweed").kind == "brands" and mine.problem("zorblax").message == 'no "zorblax"'
    assert mine.problem("hermes orange") is None and words.problem("hermes orange") is not None
    mine.add("brands", ["Blorp"])
    mine.allow(["nike"])
    assert mine.problem("blorp") is not None and mine.problem("nike") is None
    extra.write_text("{not json")
    with pytest.warns(UserWarning, match="not used"):
        assert PromptFilter.load().problem("gucci") is not None
    assert FabricGenerator(None, cache=False, words=None).prepare("gucci")[0] == "gucci"  # no filter at all


def test_the_kiosk_takes_some_words_only_in_their_fabric_sense(tmp_path, monkeypatch):
    words = PromptFilter.load()
    for fine in ("nude satin", "nude pink chiffon", "nude and black stripes", "naked wool", "cock pheasant feathers",
                 "hen and cock print", "bird print with blue tits", "great tits and robins", "long-tailed tits",
                 "Dick Whittington", "nud3 satin"):
        assert words.problem(fine) is None, fine
    for text, word in (("nude", "nude"), ("nude woman satin", "nude"), ("nude photo silk", "nude"),
                       ("nude landscape", "nude"), ("naked girl", "naked"), ("big tits", "tits"), ("cock", "cock"),
                       ("dicks", "dicks"), ("n*de satin", "n*de"), ("c**k pheasant", "c**k")):
        problem = words.problem(text)
        assert problem is not None and (problem.kind, problem.word) == ("in context", word), text
    assert words.problem("nudes").kind == "offensive"
    nude = words.problem("nude").message
    assert "does not accept" not in nude and "beige or blush" in nude and '"nude satin"' in nude
    extra = tmp_path / "words.json"  # a user file can add words with a fabric sense too
    extra.write_text(json.dumps({"in context": {"words": {"bolt": {"near": ["@materials"]}}}}))
    monkeypatch.setenv(fabric.WORDS_ENV, str(extra))
    mine = PromptFilter.load()
    assert mine.problem("lightning bolt") is not None and mine.problem("a bolt of silk") is None


def test_fabric_scale_follows_the_kind_of_fabric():
    silk, dots, paisley, palms = (fabric_scale(t) for t in ("navy silk", "polka dots", "blue paisley silk",
                                                            "tropical palm leaves"))
    assert silk < dots < paisley < palms  # a material alone is a close-up; the largest print named wins
    assert fabric_scale("large paisley") == pytest.approx(paisley * 1.6)
    assert fabric_scale("tiny polka dots") < dots
    assert fabric_scale("something nice") == fabric.DEFAULT_SCALE


# -- the seam score, the repair, even light -----------------------------------------------------------------------


def test_the_seam_score_tells_swatches_that_tile_from_seams():
    tiling = [periodic(512, c, s) for c, s in ((3, 1), (8, 2), (20, 3), (60, 4))]
    tiling += [np.roll(periodic(512, 10, 5), (100, 37), axis=(0, 1))]
    tiling += [repeat_of(P.tile(k, (30, 40, 80)), 512, (61, 117)) for k in ("tartan", "houndstooth", "polka dots",
                                                                             "checks", "camo", "denim")]
    for image in tiling:
        assert seam_score(image) <= SEAM_THRESHOLD
    seams = [cut_out(512, c, s) for c, s in ((3, 1), (8, 2), (20, 3), (40, 4))]
    x = np.arange(512)
    band = 1 - 0.15 * np.exp(-np.minimum(x, 511 - x) / 6.0)  # the dark rim of a decoder with zero padding
    seams.append(periodic(512, 10, 6) * band[None, :, None])
    seams.append(periodic(512, 30, 7) * (0.75 + 0.5 * x / 511)[None, :, None])  # light falling off to one side
    seams.append(repeat_of(P.stripes((30, 40, 80), (220, 220, 210), size=37), 512, (5, 9)))  # a broken rhythm
    seams.append(repeat_of(P.polka_dots((240, 230, 210), (180, 40, 60), size=90), 512, (13, 31)))
    for image in seams:
        assert seam_score(image) > SEAM_THRESHOLD
    # wide bands: one at the wrap is the wrong width (a narrow window has twins everywhere in a band)
    for kind, period, phase in (("stripes", 136, 20), ("gingham", 100, 31), ("check", 190, 30), ("stripes", 250, 33)):
        assert seam_score(bands(kind, period, phase)) > SEAM_THRESHOLD, (kind, period)
    for kind, period in (("stripes", 128), ("gingham", 64), ("check", 256)):  # (they fit: they tile)
        assert seam_score(bands(kind, period, 17)) <= SEAM_THRESHOLD, (kind, period)
    base, other = periodic(512, 20, 2), periodic(512, 20, 52)  # a seam along a fifth of the edge only
    part, fade = base.copy(), np.clip(np.arange(40)[None, :, None] / 12.0, 0, 1)
    part[:100, -40:] = part[:100, -40:] * (1 - fade) + other[:100, -40:] * fade  # (feathered on the inside)
    assert seam_score(part) > SEAM_THRESHOLD >= seam_score(base)
    parts = seam_scores(seams[-1])
    assert set(parts) == {"step x", "band x", "rhythm x", "step y", "band y", "rhythm y"}
    assert max(parts.values()) == pytest.approx(seam_score(seams[-1]))
    flat = np.full((64, 64, 3), 128, np.float32)
    assert seam_score(flat) == pytest.approx(1.0, abs=0.01)  # nothing stands out on a plain colour


def test_the_repair_makes_any_image_tile_the_same_way_every_time():
    x = np.arange(512, dtype=np.float32)
    light = (0.75 + 0.5 * x / 511)[None, :, None] * (1.1 - 0.25 * (x / 511))[:, None, None]
    cases = [cut_out(512, 8, 11), cut_out(512, 30, 12) * light,
             repeat_of(P.stripes((30, 40, 80), (220, 220, 210), size=37), 512, (5, 9)),
             repeat_of(P.polka_dots((240, 230, 210), (180, 40, 60), size=90), 512, (13, 31))]
    for image in cases:
        image = np.clip(image, 0, 255)
        start = time.perf_counter()
        fixed = make_seamless(image)
        assert time.perf_counter() - start < 2.0
        assert fixed.shape == image.shape and fixed.dtype == np.float32
        assert seam_score(fixed) <= SEAM_THRESHOLD < seam_score(image)
        assert np.array_equal(fixed, make_seamless(image))  # deterministic
        assert abs(float(fixed.mean()) - float(image.mean())) < 8  # the same fabric, not a new one
    stripes = make_seamless(cases[2])  # the overlap keeps the rhythm: every stripe as wide as the others
    profile = stripes[256, :, 0] > 128
    widths = np.diff(np.flatnonzero(np.diff(profile.astype(int))))
    assert widths.max() - widths.min() <= 4
    for image in (periodic(512, 6, 1), motifs(512, 7, 40, (25, 45, 90), (220, 120, 150), 3)):
        assert seam_score(image) <= SEAM_THRESHOLD  # it tiles already: it comes back as it is
        assert np.array_equal(make_seamless(image, even_light=False), image)
        assert np.array_equal(make_seamless(image), image)  # (and its light is even)


def test_the_repair_cuts_motifs_and_never_ghosts_them():
    for seed in range(2):
        for ground, colour, count, radius in (((235, 225, 205), (190, 40, 60), 20, 30),
                                              ((14, 14, 14), (245, 245, 240), 12, 34)):
            image = motifs(512, count, radius, ground, colour, seed, tiling=False)
            fixed = make_seamless(image)
            assert seam_score(fixed) <= SEAM_THRESHOLD < seam_score(image)
            lab, palette = (cv2.cvtColor(np.float32(a) / 255, cv2.COLOR_RGB2Lab) for a in (fixed, [[ground, colour]]))
            apart = np.linalg.norm(lab.reshape(-1, 1, 3) - palette.reshape(1, -1, 3), axis=2)
            assert (apart.min(axis=1) > 12).mean() < 0.02  # each pixel the ground or a motif (edges aside): no ghosts
            assert spread(fixed, apart[:, 0].reshape(512, 512) < 3) < 1.05  # and the ground as even as it was


def test_even_light_takes_out_light_and_leaves_prints_alone():
    ground = (235, 225, 205)
    for seed in range(3):  # a print's motifs spread unevenly are not light
        image = motifs(512, 20, 30, ground, (190, 40, 60), seed)
        assert np.array_equal(equalise_light(image), image) and not light_of(image)["light"]
        uneven = lit(image, 0.4, seed)  # but real light is: the ground comes out even, the motifs keep their colour
        where = ground_of(image, ground)
        assert spread(uneven, where) > 1.3 and spread(equalise_light(uneven), where) < 1.08
    tile = np.tile(P.tartan((160, 30, 40), (20, 60, 40), size=128), (4, 4, 1)).astype(np.float32)  # 4 setts, tiles
    light = lit(tile, 0.6, 5)

    def unevenness(image):
        return float(cv2.GaussianBlur(image @ np.float32([0.2126, 0.7152, 0.0722]), (0, 0), 40).std())

    even = equalise_light(light)
    assert unevenness(even) < 0.45 * unevenness(light) and unevenness(even) < 1.5 * unevenness(tile) + 1
    hue = [cv2.cvtColor(np.clip(i, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV)[..., 0] for i in (tile, even)]
    assert np.median(np.abs(hue[0].astype(int) - hue[1].astype(int))) == 0  # the colours stay
    lum = [float((i @ np.float32([0.2126, 0.7152, 0.0722])).mean()) for i in (light, even)]
    assert lum[1] == pytest.approx(lum[0], rel=0.02)  # and the brightness
    assert seam_score(even) <= SEAM_THRESHOLD
    verdict = light_of(light)
    assert verdict["light"] and verdict["change"] > 1.5 and verdict["prominence"] > 2.5 and verdict["colour"] < 0.5
    for blotchy in (periodic(512, 3, 1), periodic(512, 6, 2)):  # slow blotches are the fabric's own: left alone
        assert np.array_equal(equalise_light(blotchy), blotchy)
    sett = P.tartan((40, 90, 170), (240, 240, 236), size=512)  # one sett across: blue to white is colour, not light
    assert light_of(sett)["colour"] > 0.5 and np.array_equal(equalise_light(sett), sett.astype(np.float32))
    gradient = periodic(256, 12, 3) * (0.6 + 0.8 * np.arange(256) / 255)[None, :, None]
    flat = equalise_light(gradient, periodic=False)  # an image that does not tile yet: a gradient goes too
    columns = flat.mean(axis=(0, 2))
    assert abs(columns[-32:].mean() - columns[:32].mean()) < 0.25 * abs(gradient.mean(axis=(0, 2))[-32:].mean()
                                                                       - gradient.mean(axis=(0, 2))[:32].mean())
    cut = motifs(512, 20, 30, ground, (190, 40, 60), 4, tiling=False)
    assert np.array_equal(equalise_light(cut, periodic=False), cut)  # (nor is a cut-out print's layout)


def test_resizing_a_repeat_wraps_round_its_edges():
    x = (np.arange(96) + 0.5) / 96
    wave = np.repeat((128 + 100 * np.sin(2 * np.pi * x))[None, :, None], 96, axis=0).astype(np.float32)
    big = resize_periodic(wave, 128, 96)
    expected = 128 + 100 * np.sin(2 * np.pi * (np.arange(128) + 0.5) / 128)
    assert np.abs(big[10, :, 0] - expected).max() < 1.0  # the edges too
    noise = np.random.default_rng(1).random((96, 96, 3)).astype(np.float32) * 255
    moved = np.roll(noise, (3, 6), (0, 1))  # 3 and 6 of 96 pixels are 4 and 8 of 128
    assert np.allclose(resize_periodic(moved, 128), np.roll(resize_periodic(noise, 128), (4, 8), (0, 1)), atol=1e-2)
    plain = cv2.resize(noise, (128, 128), interpolation=cv2.INTER_CUBIC)  # (cv2 repeats the edge pixels instead)
    assert np.abs(cv2.resize(moved, (128, 128), interpolation=cv2.INTER_CUBIC) - np.roll(plain, (4, 8), (0, 1))).max() \
        > 10
    image = np.random.default_rng(0).random((64, 64, 3)).astype(np.float32)
    assert np.allclose(resize_periodic(image, 32), image.reshape(32, 2, 32, 2, 3).mean(axis=(1, 3)))
    swatch = Swatch(periodic(768, 9, 2), scale=0.2)  # a 768 swatch becomes 1024 for the mipmaps, still tiling
    base = swatch._levels[0]
    assert base.shape == (1024, 1024, 3)
    wrap = np.abs(base[:, 0] - base[:, -1]).mean()
    assert wrap < 1.5 * np.abs(np.diff(base, axis=1)).mean()


# -- latent rolling and circular padding --------------------------------------------------------------------------


class _Output:
    def __init__(self, image):
        self.images = [Image.fromarray(image)]


class Flux2KleinStandIn:
    """Stands in for diffusers' Flux2KleinPipeline: its loop's latents are packed tokens ``(1, h * w, 128)``, row by
    row over the token grid (h = height / 16), as ``_pack_latents`` and ``_prepare_latent_ids`` make them; after each
    Euler step it calls ``callback_on_step_end(self, i, t, {"latents": ...})`` and takes the latents back; at the end
    it puts the tokens back by their ids (``_unpack_latents_with_ids``). ``paint(seed, size)`` makes the image;
    ``velocity(latents, step)`` stands in for the transformer; ``gate(step)`` may hold a step back."""

    channels, per_token = 128, 16

    def __init__(self, paint=None, velocity=None, gate=None):
        self.vae = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 3, padding=1))
        self.transformer = types.SimpleNamespace(config=types.SimpleNamespace(patch_size=1))
        self.paint = paint or (lambda seed, size: periodic(size, 6, seed))
        self.velocity = velocity or (lambda latents, step: -0.5 * latents + 0.1)
        self.gate = gate
        self.calls, self.inputs, self.modes = [], [], []
        self.final = None
        self.num_timesteps = 0

    def start(self, generator, height, width):
        h, w = height // self.per_token, width // self.per_token
        grid = torch.randn((1, self.channels, h, w), generator=generator)
        return grid.reshape(1, self.channels, h * w).permute(0, 2, 1), (h, w)

    def finish(self, latents, shape):
        h, w = shape
        ids = torch.cartesian_prod(torch.arange(1), torch.arange(h), torch.arange(w), torch.arange(1))
        flat = ids[:, 1] * w + ids[:, 2]
        return torch.zeros_like(latents[0]).index_copy(0, flat, latents[0]).reshape(h, w, -1).permute(2, 0, 1)

    def __call__(self, prompt, width, height, num_inference_steps, guidance_scale, generator,
                 callback_on_step_end=None, callback_on_step_end_tensor_inputs=("latents",)):
        seed = generator.initial_seed()
        self.calls.append({"prompt": prompt, "size": (width, height), "steps": num_inference_steps, "seed": seed})
        latents, shape = self.start(generator, height, width)
        self.num_timesteps = num_inference_steps
        sigmas = torch.linspace(1.0, 0.0, num_inference_steps + 1)
        for i in range(num_inference_steps):
            if self.gate is not None:
                self.gate(i)
            self.inputs.append(latents.clone())
            latents = latents + (sigmas[i + 1] - sigmas[i]) * self.velocity(latents, i)
            if callback_on_step_end is not None:
                latents = callback_on_step_end(self, i, sigmas[i], {"latents": latents}).pop("latents", latents)
        self.final = self.finish(latents, shape)
        self.modes.append({m.padding_mode for m in self.vae.modules() if isinstance(m, torch.nn.Conv2d)})
        return _Output(np.clip(self.paint(seed, width), 0, 255).astype(np.uint8))


class ZImageStandIn(Flux2KleinStandIn):
    """Stands in for ZImagePipeline: its latents are the grid itself, ``(1, 16, height / 8, width / 8)``; its
    transformer cuts them into 2 x 2 patches."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.transformer = types.SimpleNamespace(config=types.SimpleNamespace(all_patch_size=(2,)))

    def start(self, generator, height, width):
        return torch.randn((1, 16, height // 8, width // 8), generator=generator), None

    def finish(self, latents, shape):
        return latents[0]


class SdTurboStandIn(ZImageStandIn):
    """Stands in for SD-Turbo's pipeline: a convolutional UNet, which pads circularly instead of rolling."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        del self.transformer
        self.unet = torch.nn.Sequential(torch.nn.Conv2d(4, 4, 3, padding=1), torch.nn.Conv2d(4, 4, 1))

    def start(self, generator, height, width):
        return torch.randn((1, 4, height // 8, width // 8), generator=generator), None


def _roll_grid(latents, shift):
    """What rolling the image means for each layout: packed tokens are unpacked to the grid and rolled there."""
    if latents.ndim == 3:
        side = round(latents.shape[1] ** 0.5)
        return torch.roll(latents.reshape(1, side, side, -1), shift, dims=(1, 2)).reshape(latents.shape)
    return torch.roll(latents, (shift[0] * 2, shift[1] * 2), dims=(2, 3))


def test_roll_offsets_are_random_shrink_and_end_where_they_began():
    offsets = roll_offsets(12, (32, 32), seed=5)
    assert offsets[0] == (0, 0) and offsets[-1] == (0, 0) and offsets == roll_offsets(12, (32, 32), seed=5)
    assert offsets != roll_offsets(12, (32, 32), seed=6)
    reach = [max(abs(r), abs(c)) for r, c in offsets]
    assert max(reach[1:4]) > max(reach[-4:]) and all(m <= 16 * (11 - i) / 11 + 0.5 for i, m in enumerate(reach))


@pytest.mark.parametrize("standin", [Flux2KleinStandIn, ZImageStandIn])
def test_rolling_in_the_step_callback_rolls_each_step_and_undoes_the_total(standin):
    plain = standin()
    plain(prompt="p", width=256, height=256, num_inference_steps=6, guidance_scale=1.0,
          generator=torch.Generator().manual_seed(3))
    rolled = standin()
    roller = LatentRoller(6, seed=9, unit=plan_for(rolled).roll)
    rolled(prompt="p", width=256, height=256, num_inference_steps=6, guidance_scale=1.0,
           generator=torch.Generator().manual_seed(3), callback_on_step_end=roller)
    # a denoiser that is the same everywhere does not care where the image is: rolled and plain end the same
    assert torch.allclose(rolled.final, plain.final, atol=1e-6)
    assert roller.rolls and roller.offset == (0, 0)
    offsets = roll_offsets(6, roller.grid(plain.inputs[0]), seed=9)
    for step, (before, after) in enumerate(zip(plain.inputs, rolled.inputs)):  # each step saw its own roll
        assert torch.allclose(after, _roll_grid(before, offsets[step]), atol=1e-6), step
    assert offsets[-1] == (0, 0) and len(roller.rolls) == sum(1 for a, b in itertools.pairwise(offsets) if a != b)


def test_rolling_packed_tokens_matches_diffusers_flux2_packing():
    diffusers = pytest.importorskip("diffusers")
    pipe = diffusers.Flux2KleinPipeline
    grid = torch.randn(1, 16, 6, 6)
    packed, ids = pipe._pack_latents(grid), pipe._prepare_latent_ids(grid)
    rolled = LatentRoller(4, 0).roll(packed, (2, -1))
    assert torch.equal(pipe._unpack_latents_with_ids(rolled, ids, 6, 6), torch.roll(grid, (2, -1), dims=(2, 3)))


def test_the_step_callback_cancels_reports_progress_and_paces():
    seen, paced = [], []
    callback = StepCallback(None, cancelled=lambda: len(seen) >= 2, progress=lambda s, n: seen.append((s, n)),
                            pace=lambda: paced.append(1))
    pipe = types.SimpleNamespace(num_timesteps=4)
    assert callback(pipe, 0, 1.0, {"latents": None}) == {} and callback(pipe, 1, 0.5, {"latents": None}) == {}
    with pytest.raises(Cancelled):
        callback(pipe, 2, 0.25, {"latents": None})
    assert seen == [(1, 4), (2, 4)] and len(paced) == 2


def test_circular_padding_is_on_for_the_call_and_off_afterwards():
    vae = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3, padding=1), torch.nn.ReLU(), torch.nn.Conv2d(8, 3, 1),
                              torch.nn.Conv2d(3, 3, 5, padding=2, padding_mode="reflect"))
    vae.use_tiling = True
    x = torch.rand(1, 3, 16, 16)
    with torch.no_grad():
        before = vae(x)
        with pytest.raises(RuntimeError), circular_padding(vae, None) as changed:
            assert changed == 2 and not vae.use_tiling  # (the 1 x 1 convolution has nothing to pad)
            inside = vae(x)
            assert torch.allclose(vae(torch.roll(x, (3, 5), (2, 3))), torch.roll(inside, (3, 5), (2, 3)), atol=1e-5)
            raise RuntimeError("the call failed")
        assert [m.padding_mode for m in vae if isinstance(m, torch.nn.Conv2d)] == ["zeros", "zeros", "reflect"]
        assert vae.use_tiling and torch.equal(vae(x), before)
    assert plan_for(Flux2KleinStandIn()).kind == "packed transformer" and plan_for(ZImageStandIn()).roll == 2
    assert plan_for(SdTurboStandIn()).roll is None and plan_for(SdTurboStandIn()).kind == "unet"


# -- tiny diffusers models, built here (no downloads) -------------------------------------------------------------


def _tiny_tokenizer(max_length=512):
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    vocab = {"[PAD]": 0, "[UNK]": 1}
    for word in [*fabric.TEMPLATE.replace(",", " ").split(), "blue", "paisley", "silk", "red"]:
        vocab.setdefault(word, len(vocab))
    tok = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    fast = transformers.PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="[PAD]", unk_token="[UNK]",
                                                model_max_length=max_length)
    fast.chat_template = "{% for m in messages %}{{ m['content'] }}{% endfor %}"
    return fast


def _tiny_vae(cls, blocks=(8, 16), **kwargs):
    return cls(block_out_channels=blocks, down_block_types=("DownEncoderBlock2D",) * len(blocks),
               up_block_types=("UpDecoderBlock2D",) * len(blocks), latent_channels=4, norm_num_groups=4,
               layers_per_block=1, sample_size=64, **kwargs).eval()


def _tiny_flux2_klein():
    diffusers = pytest.importorskip("diffusers")
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    text = transformers.Qwen3ForCausalLM(transformers.Qwen3Config(
        vocab_size=64, hidden_size=8, intermediate_size=16, num_hidden_layers=28, num_attention_heads=2,
        num_key_value_heads=1, head_dim=4, max_position_embeddings=1024))  # (the pipeline reads layers 9, 18, 27)
    transformer = diffusers.Flux2Transformer2DModel(
        patch_size=1, in_channels=16, num_layers=1, num_single_layers=1, attention_head_dim=16, num_attention_heads=2,
        joint_attention_dim=24, timestep_guidance_channels=32, mlp_ratio=2.0, axes_dims_rope=(4, 4, 4, 4),
        guidance_embeds=False)
    pipe = diffusers.Flux2KleinPipeline(scheduler=diffusers.FlowMatchEulerDiscreteScheduler(),
                                        vae=_tiny_vae(diffusers.AutoencoderKLFlux2), text_encoder=text,
                                        tokenizer=_tiny_tokenizer(), transformer=transformer, is_distilled=True)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def test_a_tiny_vae_decoder_tiles_once_it_pads_circularly():
    diffusers = pytest.importorskip("diffusers")
    torch.manual_seed(0)
    vae = _tiny_vae(diffusers.AutoencoderKL, blocks=(8, 16, 16))
    f = torch.fft.fftfreq(32)
    lowpass = torch.exp(-(f[:, None] ** 2 + f[None, :] ** 2) / 0.01)

    def decode(latents):
        with torch.no_grad():
            image = vae.decode(latents).sample[0]
        image = (image - image.mean()) / (image.std() + 1e-6) * 40 + 128
        return image.permute(1, 2, 0).clamp(0, 255).numpy()

    for seed in range(3):  # smooth latents that tile, like those of a picture
        noise = torch.randn(1, 4, 32, 32, generator=torch.Generator().manual_seed(seed))
        latents = torch.fft.ifft2(torch.fft.fft2(noise) * lowpass).real * 6
        padded = decode(latents)
        with circular_padding(vae) as changed:
            assert changed > 10
            wrapped = decode(latents)
            moved = decode(torch.roll(latents, (5, 3), dims=(2, 3)))
        assert np.abs(np.roll(wrapped, (20, 12), axis=(0, 1)) - moved).max() < 0.05  # it paints a torus
        assert seam_score(wrapped) <= SEAM_THRESHOLD < seam_score(padded)
    assert {m.padding_mode for m in vae.modules() if isinstance(m, torch.nn.Conv2d)} == {"zeros"}
    assert np.array_equal(decode(latents), padded)  # undone


def test_a_tiny_unet_denoises_a_torus_once_it_pads_circularly():
    diffusers = pytest.importorskip("diffusers")
    torch.manual_seed(0)
    unet = diffusers.UNet2DConditionModel(
        block_out_channels=(8, 16), layers_per_block=1, sample_size=16, in_channels=4, out_channels=4,
        down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"), up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
        cross_attention_dim=16, attention_head_dim=2, norm_num_groups=4).eval()
    x, text = torch.randn(1, 4, 16, 16), torch.randn(1, 5, 16)

    def run(latents):
        with torch.no_grad():
            return unet(latents, 10, encoder_hidden_states=text).sample

    shifted = (4, 6)  # (even, so the downsampling grid lands on itself)
    before = run(x)
    with circular_padding(unet):
        assert torch.allclose(run(torch.roll(x, shifted, (2, 3))), torch.roll(run(x), shifted, (2, 3)), atol=1e-4)
    assert not torch.allclose(run(torch.roll(x, shifted, (2, 3))), torch.roll(before, shifted, (2, 3)), atol=1e-4)
    assert torch.equal(run(x), before)


def test_a_tiny_flux2_klein_pipeline_rolls_its_tokens_and_decodes_circularly(tmp_path):
    pipe = _tiny_flux2_klein()
    seen = []
    decoder = next(m for m in pipe.vae.decoder.modules() if isinstance(m, torch.nn.Conv2d))
    hook = decoder.register_forward_pre_hook(lambda module, args: seen.append(module.padding_mode))
    images = ImageGenerator(PRESETS["flux2-klein"], pipeline=pipe)
    generator = FabricGenerator("flux2-klein", seeds=1, size=64, threshold=100.0, cache=tmp_path / "c",
                                loader=lambda model, device, download: images)
    job = generator.generate("blue paisley silk")
    hook.remove()
    swatch = job.variations[0]
    assert job.status == "done" and swatch.generated and swatch.image.shape == (64, 64, 3)
    assert swatch.model == "black-forest-labs/FLUX.2-klein-4B" and swatch.method == "rolled, 4 steps"
    assert seen == ["circular"] and decoder.padding_mode == "zeros"  # decoded circularly, and undone after
    assert job.progress["step"] == job.progress["steps"] == 4
    # with the transformer's output zeroed the latents stay the noise: rolled or not, they must end the same
    with torch.no_grad():
        pipe.transformer.proj_out.weight.zero_()

    def latents(callback=None):
        return pipe(prompt="red", height=64, width=64, num_inference_steps=4, guidance_scale=1.0,
                    generator=torch.Generator().manual_seed(5), output_type="latent",
                    callback_on_step_end=callback).images

    roller = LatentRoller(4, seed=9)
    assert torch.equal(latents(roller), latents()) and len(roller.rolls) >= 2


@pytest.mark.filterwarnings("ignore::DeprecationWarning")  # (numpy 2 and diffusers' Euler scheduler)
def test_tiny_z_image_and_sd_pipelines_take_the_patches():
    diffusers = pytest.importorskip("diffusers")
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    text = transformers.Qwen3Model(transformers.Qwen3Config(
        vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=1, head_dim=8, max_position_embeddings=1024))
    transformer = diffusers.ZImageTransformer2DModel(
        all_patch_size=(2,), all_f_patch_size=(1,), in_channels=4, dim=32, n_layers=1, n_refiner_layers=1, n_heads=2,
        n_kv_heads=2, cap_feat_dim=16, axes_dims=[8, 4, 4], axes_lens=[256, 64, 64])
    z_image = diffusers.ZImagePipeline(scheduler=diffusers.FlowMatchEulerDiscreteScheduler(),
                                       vae=_tiny_vae(diffusers.AutoencoderKL, scaling_factor=0.5, shift_factor=0.1),
                                       text_encoder=text, tokenizer=_tiny_tokenizer(), transformer=transformer)
    unet = diffusers.UNet2DConditionModel(
        block_out_channels=(8, 16), layers_per_block=1, sample_size=16, in_channels=4, out_channels=4,
        down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"), up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
        cross_attention_dim=16, attention_head_dim=2, norm_num_groups=4)
    clip = transformers.CLIPTextModel(transformers.CLIPTextConfig(
        vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
        max_position_embeddings=77, projection_dim=16))
    sd = diffusers.StableDiffusionPipeline(
        vae=_tiny_vae(diffusers.AutoencoderKL), text_encoder=clip, tokenizer=_tiny_tokenizer(77), unet=unet,
        scheduler=diffusers.EulerDiscreteScheduler(timestep_spacing="trailing", steps_offset=1), safety_checker=None,
        feature_extractor=None, requires_safety_checker=False)
    for pipe, model, kind, method in ((z_image, "z-image-turbo", "transformer", "rolled, 9 steps"),
                                      (sd, "sd-turbo", "unet", "circular, 2 steps")):
        pipe.set_progress_bar_config(disable=True)
        assert plan_for(pipe).kind == kind
        images = ImageGenerator(PRESETS[model], pipeline=pipe)
        job = FabricGenerator(model, seeds=1, size=64, threshold=100.0, cache=False,
                              loader=lambda m, d, w, images=images: images).generate("blue paisley silk")
        assert job.status == "done" and job.variations[0].method == method
        assert job.progress["step"] == job.progress["steps"] and job.variations[0].image.shape == (64, 64, 3)
        assert {m.padding_mode for m in pipe.vae.modules() if isinstance(m, torch.nn.Conv2d)} == {"zeros"}


# -- the ladder ---------------------------------------------------------------------------------------------------

STANDINS = {"flux2-klein": Flux2KleinStandIn, "z-image-turbo": ZImageStandIn, "sd-turbo": SdTurboStandIn}


def good(seed, size):
    return periodic(size, 6, seed)


def bad(seed, size):
    return cut_out(size, 6, seed)


def _loader(paints, record, pipes=None):
    """A loader of stand-ins: ``paints[model]`` paints their images; a paint that is an exception is raised when the
    model loads; ``record`` lists the models loaded."""

    def load(model, device="auto", download=True):
        record.append(model)
        paint = paints[model]
        if isinstance(paint, BaseException):
            raise paint
        pipe = STANDINS[model](paint=paint)
        if pipes is not None:
            pipes[model] = pipe
        return ImageGenerator(PRESETS[model], pipeline=pipe)

    return load


def test_seeds_with_seams_are_left_out_and_the_good_ones_kept(tmp_path):
    pipes, record = {}, []
    paint = lambda seed, size: (bad if seed % 2 else good)(seed, size)  # seeds 1 and 3 seamy
    generator = FabricGenerator("flux2-klein", size=256, cache=tmp_path / "c",
                                loader=_loader({"flux2-klein": paint}, record, pipes))
    job = generator.generate("blue paisley silk")
    assert job.status == "done" and record == ["flux2-klein"]
    assert [s.seed for s in job.variations] == [0, 2] and all(s.ok and s.seam <= SEAM_THRESHOLD for s in job.variations)
    assert any("seed 1: a seam shows" in m for m in job.messages)
    assert [c["steps"] for c in pipes["flux2-klein"].calls] == [4] * 4  # the preset's steps, the same prompt
    assert {c["prompt"] for c in pipes["flux2-klein"].calls} == {fabric_prompt("blue paisley silk")}
    assert all(modes == {"circular"} for modes in pipes["flux2-klein"].modes)
    assert job.best.seed in (0, 2) and job.best.scale == fabric_scale("blue paisley silk")
    assert job.ranked[0] is job.best and job.ranked[0].seam <= job.ranked[1].seam  # the seam score ranks them
    assert job.results[0].preview and not job.results[0].generated  # the keyword preview came first


def test_the_ladder_steps_down_rung_by_rung_saying_why(tmp_path):
    pipes, record = {}, []
    generator = FabricGenerator("flux2-klein", seeds=2, size=256, cache=tmp_path / "c",
                                loader=_loader({"flux2-klein": bad, "sd-turbo": bad}, record, pipes))
    job = generator.generate("blue paisley silk")
    assert record == ["flux2-klein", "sd-turbo"]  # klein loaded once for both of its rungs
    assert [c["steps"] for c in pipes["flux2-klein"].calls] == [4, 4, 12, 12]
    assert [c["steps"] for c in pipes["sd-turbo"].calls] == [2, 2]
    said = "\n".join(job.messages)
    assert "FLUX.2-klein-4B (rolled, 4 steps): a seam shows on both seeds; trying FLUX.2-klein-4B (rolled, 12 " \
           "steps)" in said
    assert "FLUX.2-klein-4B (rolled, 12 steps): a seam shows on both seeds; trying sd-turbo (circular, 2 " \
           "steps)" in said
    assert "sd-turbo (circular, 2 steps): a seam shows on both seeds; trying the seam repair" in said
    assert "repaired without a model" in said
    (swatch,) = job.variations
    assert swatch.method.endswith(", repaired") and swatch.ok and swatch.generated
    assert swatch.seam <= SEAM_THRESHOLD and pipes["sd-turbo"].modes[0] == {"circular"}


@pytest.mark.parametrize("trouble, kind, hint", [
    (ImportError("No module named 'diffusers'"), "missing", 'pip install -e ".[people]"'),
    (RuntimeError("diffusers 0.30.0 has no Flux2KleinPipeline; run: pip install -U diffusers"), "missing",
     "pip install -U diffusers"),
    (OSError("black-forest-labs/FLUX.2-klein-4B does not appear to have a file named model_index.json"), "weights",
     "hf download black-forest-labs/FLUX.2-klein-4B"),
    (type("ProxyError", (Exception,), {"__module__": "httpx"})("403 Forbidden"), "weights", "hf auth login"),
])
def test_missing_packages_weights_or_network_go_straight_to_the_keyword_fabric(trouble, kind, hint, tmp_path):
    record = []
    generator = FabricGenerator("flux2-klein", seeds=2, size=256, cache=tmp_path / "c",
                                loader=_loader({"flux2-klein": trouble, "sd-turbo": good}, record))
    job = generator.generate("red tartan")
    assert explain_failure(trouble, PRESETS["flux2-klein"]).kind == kind
    assert record == ["flux2-klein"] and job.status == "done"  # SD-Turbo is not tried
    assert any(hint in m and "the procedural fabric instead" in m for m in job.messages)
    (swatch,) = job.variations
    assert not swatch.generated and swatch.method == "procedural" and swatch.note.startswith("tartan")
    assert generator.generate("red tartan").variations[0].method == "procedural" and record == ["flux2-klein"]


def test_a_model_that_runs_out_of_memory_steps_down_to_sd_turbo(tmp_path):
    record = []

    def hungry(latents, step):
        raise RuntimeError("MPS backend out of memory (MPS allocated: 17.2 GB)")

    def load(model, device="auto", download=True):
        record.append(model)
        pipe = Flux2KleinStandIn(velocity=hungry) if model == "flux2-klein" else SdTurboStandIn(paint=good)
        return ImageGenerator(PRESETS[model], pipeline=pipe)

    job = FabricGenerator("flux2-klein", seeds=2, size=256, cache=False, loader=load).generate("blue paisley silk")
    assert record == ["flux2-klein", "sd-turbo"]
    # (klein at more steps would not fit either: a model that fails is not tried again)
    assert any("does not fit in memory" in m and "trying sd-turbo (circular, 2 steps)" in m for m in job.messages)
    assert [s.model for s in job.variations] == ["stabilityai/sd-turbo"] * 2
    oom = FabricGenerator("flux2-klein", seeds=1, size=256, cache=False,
                          loader=_loader({"flux2-klein": MemoryError(), "sd-turbo": good}, record))
    assert oom.generate("red").variations[0].model == "stabilityai/sd-turbo"  # (out of memory while loading, too)


def test_the_cache_and_a_failing_model_never_cost_a_swatch(tmp_path):
    record = []
    cache = tmp_path / "c"

    def generator(threshold, cache=cache, paint=bad, seeds=1):
        return FabricGenerator("flux2-klein", seeds=seeds, size=256, cache=cache, threshold=threshold,
                               ladder=(Rung("flux2-klein"),), loader=_loader({"flux2-klein": paint}, record))

    lenient = generator(100.0).generate("blue paisley silk")  # a seam, but this generator keeps it
    assert lenient.variations[0].ok and lenient.variations[0].method == "rolled, 4 steps"
    strict = generator(SEAM_THRESHOLD).generate("blue paisley silk")  # the cached image, judged by its own threshold
    assert record == ["flux2-klein"] and strict.variations[0].method == "rolled, 4 steps, repaired"
    said = "\n".join(strict.messages)
    assert f"> {SEAM_THRESHOLD:g}); left out" in said and "on its only seed" in said and said.count(", seam ") == 1
    blocked = tmp_path / "a file"  # a cache that cannot be written costs a message, not the swatches
    blocked.write_text("x")
    job = generator(SEAM_THRESHOLD, cache=blocked, paint=good, seeds=2).generate("navy silk")
    assert job.status == "done" and [s.generated for s in job.variations] == [True, True]
    assert any("not cached" in m for m in job.messages)

    def tired(seed, size):  # a model that runs out of memory after two seeds: those two are the job's
        if seed >= 2:
            raise RuntimeError("MPS backend out of memory")
        return good(seed, size)

    record.clear()
    job = FabricGenerator("flux2-klein", seeds=4, size=256, cache=False,
                          loader=_loader({"flux2-klein": tired, "sd-turbo": good}, record)).generate("red silk")
    assert record == ["flux2-klein"] and [(s.seed, s.generated) for s in job.variations] == [(0, True), (1, True)]
    assert "the 2 swatches made so far stay" in job.messages[-1]


def test_without_a_model_the_keyword_fabric_then_a_plain_colour(tmp_path):
    generator = FabricGenerator(None, cache=tmp_path / "c")
    assert generator.ladder == ()
    tartan = generator.generate("red tartan jacket fabric")
    assert tartan.messages[0] == "no model asked for: the procedural fabric"
    swatch = tartan.best
    assert swatch.method == "procedural" and swatch.note.startswith("tartan") and swatch.scale == pytest.approx(0.15)
    assert "jacket, fabric" in swatch.note  # the words the keyword parser left out
    assert generator.generate("turqoise").best.note == "plain turquoise ('turqoise' looks like 'turquoise')"
    plain = generator.generate("something nice").best
    assert plain.method == "plain colour" and plain.note == "plain grey (no colour or pattern word known)"
    assert not (tmp_path / "c").exists()  # nothing generated, nothing cached


# -- the cache ----------------------------------------------------------------------------------------------------


def test_finished_swatches_are_cached_by_prompt_model_seed_size_and_method(tmp_path):
    record, pipes = [], {}
    cache = tmp_path / "fabrics"
    paint = lambda seed, size: (bad if seed == 1 else good)(seed, size)

    def generator(**kwargs):
        return FabricGenerator("flux2-klein", seeds=2, size=256, cache=cache,
                               loader=_loader({"flux2-klein": paint}, record, pipes), **kwargs)

    first = generator().generate("blue paisley silk")
    assert len(list(cache.glob("*.png"))) == 2 and len(pipes["flux2-klein"].calls) == 2  # kept and left out
    record.clear()
    again = generator().generate("blue paisley silk")
    assert record == []  # all there: the model is not even loaded
    assert [s.seed for s in again.variations] == [s.seed for s in first.variations] == [0]
    assert again.variations[0].cached and np.array_equal(again.variations[0].image, first.variations[0].image)
    meta = json.loads(next(cache.glob("blue-paisley-silk-*.json")).read_text())
    assert meta["model"] == "black-forest-labs/FLUX.2-klein-4B" and meta["method"] == "rolled, 4 steps"
    keys = {FabricCache.key(fabric_prompt("blue paisley silk"), "m", 0, 256, "rolled, 4 steps")}
    for change in ({"prompt": "x"}, {"model": "n"}, {"seed": 1}, {"size": 512}, {"method": "rolled, 12 steps"}):
        args = {"prompt": fabric_prompt("blue paisley silk"), "model": "m", "seed": 0, "size": 256,
                "method": "rolled, 4 steps"}
        args.update(change)
        keys.add(FabricCache.key(**args))
    assert len(keys) == 6
    for png in cache.glob("*.png"):  # damaged files are made again
        png.write_bytes(b"not a png")
    generator().generate("blue paisley silk")
    assert record == ["flux2-klein"] and all(cv2.imread(str(p)) is not None for p in cache.glob("*.png"))
    small = FabricCache(cache, limit=1)
    small.put(FabricCache.key("p", "m", 9, 256, "x"), first.variations[0])
    assert len(list(cache.glob("*.json"))) == 1  # the oldest go first


# -- the job: a worker thread, results as they come, cancelling ---------------------------------------------------


def test_results_come_one_by_one_preview_first_without_holding_up_the_caller():
    gates = [threading.Event() for _ in range(3)]

    def paint(seed, size):
        assert gates[seed].wait(10)
        return good(seed, size)

    told, made = [], []
    generator = FabricGenerator("flux2-klein", seeds=3, size=256, cache=False,
                                loader=_loader({"flux2-klein": paint}, []))
    start = time.perf_counter()
    job = generator.start("navy linen", on_result=made.append, on_message=told.append)
    assert time.perf_counter() - start < 0.5 and not job.done  # it returns at once
    deadline = time.time() + 10
    first = []
    while not first and time.time() < deadline:
        first = job.poll()
        time.sleep(0.01)
    assert len(first) == 1 and first[0].preview and job.poll() == []  # the preview, then nothing yet
    for seed in range(3):
        gates[seed].set()
        swatch = None
        while swatch is None and time.time() < deadline:
            new = job.poll()
            swatch = new[0] if new else None
            time.sleep(0.01)
        assert swatch is not None and swatch.seed == seed  # one at a time, as each is ready
    assert job.wait(10) and job.status == "done" and made == job.results and told == job.messages
    assert job.thread.daemon and job.progress["seed"] == 2


def test_a_job_stops_at_the_next_step_when_cancelled():
    started, release = threading.Event(), threading.Semaphore(0)
    pipes = {}

    def gate(step):
        started.set()
        assert release.acquire(timeout=10)

    def load(model, device="auto", download=True):
        pipes[model] = Flux2KleinStandIn(gate=gate)
        return ImageGenerator(PRESETS[model], pipeline=pipes[model])

    job = FabricGenerator("flux2-klein", ladder=(Rung("flux2-klein", 12),), size=256, cache=False,
                          loader=load).start("blue paisley silk")
    assert started.wait(10)
    job.cancel()
    release.release()  # the step in progress ends; its callback stops the call
    assert job.wait(10) and job.status == "cancelled" and job.messages[-1] == "cancelled"
    pipe = pipes["flux2-klein"]
    assert len(pipe.inputs) == 1 and pipe.final is None  # one of 12 steps, no image made
    assert [r.preview for r in job.results] == [True] and job.best.preview
    assert {m.padding_mode for m in pipe.vae.modules() if isinstance(m, torch.nn.Conv2d)} == {"zeros"}
    before = FabricGenerator(None, cache=False).start("red")
    before.cancel()
    assert before.wait(10) and before.status in ("cancelled", "done")


# -- the fill and its spec ----------------------------------------------------------------------------------------


def test_a_fabric_keeps_its_recipe_through_outfits_and_saved_characters(tmp_path):
    from test_outfit import _person

    from bifold_demo.characters.created import CreatedCharacter, load, save
    from bifold_demo.characters.outfit.tailor import Tailor

    image = np.clip(make_seamless(cut_out(128, 5, 3, ((150, 30, 40), (240, 200, 80)))), 0, 255).astype(np.uint8)
    fill = Fabric(image, "red and gold silk paisley", model="black-forest-labs/FLUX.2-klein-4B", seed=7,
                  method="rolled, 4 steps", seam=0.93, prompt=fabric_prompt("red and gold silk paisley"))
    spec = json.loads(json.dumps(fill.spec()))
    assert {k: spec[k] for k in ("fabric", "model", "seed", "method", "prompt")} == {
        "fabric": "red and gold silk paisley", "model": "black-forest-labs/FLUX.2-klein-4B", "seed": 7,
        "method": "rolled, 4 steps", "prompt": fabric_prompt("red and gold silk paisley")}
    assert spec["swatch"].startswith("data:image/") and spec["scale"] == fabric_scale("red and gold silk paisley")
    again = from_spec(spec)
    assert isinstance(again, Fabric) and np.array_equal(again.image, fill.image)  # lossless: the very image
    assert (again.text, again.model, again.seed, again.method, again.prompt) == \
        (fill.text, fill.model, fill.seed, fill.method, fill.prompt)
    person = _person()
    tailor = Tailor(person)
    outfit = Outfit({"top": fill, "bottoms": "navy"}, name="paisley")
    texture = tailor.texture(outfit)
    rebuilt = Outfit.from_spec(json.loads(json.dumps(outfit.spec())))
    assert isinstance(rebuilt.fills["top"], Fabric) and np.array_equal(tailor.texture(rebuilt), texture)
    saved = save(CreatedCharacter("Ada", person.name, outfit=outfit.spec()), tmp_path / "created")
    loaded = load(saved)
    assert np.array_equal(tailor.texture(Outfit.from_spec(loaded.outfit)), texture)  # no model needed
    keyword = from_spec({"fabric": "red tartan"})  # a spec without its image: the keyword fabric of its words
    assert isinstance(keyword, Fabric) and keyword.method == "procedural" and keyword.model is None
    assert np.array_equal(keyword.image, procedural("red tartan").image)


def test_fabric_fill_makes_a_fill_from_a_description(monkeypatch):
    with pytest.warns(UserWarning, match="no model made a fabric of 'red tartan'"):
        keyword = fabric_fill("red tartan")  # the default generator ($BIFOLD_FABRIC_MODEL is none here)
    assert isinstance(keyword, Fabric) and keyword.method == "procedural" and keyword.notes
    generator = FabricGenerator("flux2-klein", seeds=2, size=256, cache=False,
                                loader=_loader({"flux2-klein": good}, []))
    made = fabric_fill("navy silk", generator=generator, scale=0.3, seed=5)
    assert made.model == "black-forest-labs/FLUX.2-klein-4B" and made.seed in (5, 6) and made.scale == 0.3
    assert made.prompt == fabric_prompt("navy silk") and made.spec()["prompt"] == made.prompt
    assert isinstance(Outfit({"shirt": made}).fills["shirt"], Fabric)


def test_the_image_generator_takes_steps_and_pipeline_arguments_and_shares_models(monkeypatch):
    from bifold_demo.people import generator as G

    pipe = Flux2KleinStandIn()
    images = ImageGenerator(PRESETS["flux2-klein"], pipeline=pipe)
    calls = []
    rgb = images.rgb("a fabric", seed=3, width=64, height=64, steps=7,
                     callback_on_step_end=lambda p, i, t, kw: calls.append(i) or {})
    assert rgb.shape == (64, 64, 3) and pipe.calls[-1]["steps"] == 7 and calls == list(range(7))
    assert images.lock is G.pipeline_lock(pipe) and images.lock is ImageGenerator("sd-turbo", pipeline=pipe).lock
    loads = []

    def load(self):
        loads.append(self.preset.name)
        return Flux2KleinStandIn()

    monkeypatch.setattr(G.ImageGenerator, "_load", load)
    monkeypatch.setattr(G, "_shared", {})
    monkeypatch.setattr(G, "_alive", {})
    a = G.shared_generator("flux2-klein", device="cpu")
    b = G.shared_generator(G.preset_with(PRESETS["flux2-klein"], steps=8), device="cpu")
    assert loads == ["flux2-klein"] and a.pipe is b.pipe and b.preset.steps == 8
    G.release_shared()  # a and b still hold it: given out again, never loaded a second time beside it
    assert G.shared_generator("flux2-klein", device="cpu").pipe is a.pipe and loads == ["flux2-klein"]
    pipe = weakref.ref(a.pipe)
    del a, b
    G.release_shared()
    G.free_memory()
    assert pipe() is None  # nothing holds it: freed, and loaded again when asked for
    G.shared_generator("flux2-klein", device="cpu")
    assert loads == ["flux2-klein", "flux2-klein"]
    generator = FabricGenerator("flux2-klein", seeds=1, size=64, cache=False, threshold=100.0,
                                ladder=(Rung("flux2-klein"),), loader=lambda m, d, w: G.shared_generator(m, "cpu"))
    generator.generate("blue paisley silk")
    pipe = weakref.ref(generator._loaded["flux2-klein"].pipe)
    generator.release()  # the fabric generator lets go of its models: their memory is freed
    assert pipe() is None and not G._shared and not generator._loaded


# -- the CLI ------------------------------------------------------------------------------------------------------


def test_the_fabric_cli_writes_swatches_and_dresses_an_avatar(tmp_path, capsys, monkeypatch):
    from test_outfit import _Library, _person

    from bifold_demo.characters import cli

    out, sheet, body = tmp_path / "swatch.png", tmp_path / "sheet.png", tmp_path / "body.png"
    assert cli.main(["fabric", "red tartan", "--model", "none", "-o", str(out), "--sheet", str(sheet)]) == 0
    said = capsys.readouterr().out
    assert "no model asked for" in said and "procedural: tartan" in said
    assert cv2.imread(str(out)).shape == (512, 512, 3) and cv2.imread(str(sheet)) is not None
    assert cli.main(["fabric", "a Gucci print", "--model", "none"]) == 2
    assert "Brand names" in capsys.readouterr().err
    assert cli.main(["fabric", "red", "--model", "nonsense"]) == 2
    assert cli.main(["fabric", "red", "--model", "none", "--views", "v.png"]) == 2  # (needs --avatar)
    assert cli.main(["fabric", "red", "--model", "none", "--texture", "t.png"]) == 2  # (so does --texture)
    assert cli.main(["fabric", "red", "--model", "none", "--seeds", "0"]) == 2
    assert "at least one seed" in capsys.readouterr().err
    assert cli.main(["fabric", "navy silk", "--model", "none", "--size", "128", "-o", str(out)]) == 0
    assert cv2.imread(str(out)).shape == (128, 128, 3)  # a plain colour, written at --size
    assert cli.main(["fabric", "red", "--model", "none", "--sheet", str(tmp_path / "no" / "sheet.png")]) == 1

    monkeypatch.setattr(fabric, "load_generator", _loader({"flux2-klein": good}, []))
    monkeypatch.setattr(cli, "CharacterLibrary", lambda dest: _Library(_person()))
    assert cli.main(["fabric", "blue paisley silk", "--seeds", "2", "--size", "256", "-o", str(out),
                     "--sheet", str(sheet), "--avatar", "tube_person", "--region", "top", "--texture", str(body)]) == 0
    said = capsys.readouterr().out
    assert said.count("swatch: FLUX.2-klein-4B, seed") == 2 and "tube_person: top in Fabric(" in said
    assert cv2.imread(str(out)).shape == (256, 256, 3) and cv2.imread(str(body)).shape[:2] == (256, 256)
    assert cv2.imread(str(sheet)).shape[1] == 2 * 256  # the two swatches the model made, not the preview
    assert cli.main(["fabric", "blue paisley silk", "--seeds", "2", "--size", "256"]) == 0
    assert capsys.readouterr().out.count("(from the cache)") == 2
    assert cli.main(["fabric", "red", "--model", "none", "--avatar", "nobody_here"]) == 1


def test_the_fabric_cli_renders_the_avatar_front_and_back(gl, tmp_path, monkeypatch):
    from test_outfit import _Library, _person

    from bifold_demo.characters import cli

    monkeypatch.setattr(cli, "CharacterLibrary", lambda dest: _Library(_person()))
    views = tmp_path / "views.png"
    assert cli.main(["fabric", "red tartan", "--model", "none", "--avatar", "tube_person", "--views", str(views)]) == 0
    image = cv2.imread(str(views))
    assert image.shape == (640, 720, 3)
    middle = image[:, 120:240].reshape(-1, 3).astype(int)  # (the front view: some red of the tartan)
    assert (middle[:, 2] > middle[:, 1] + 60).sum() > 200


# -- the real avatars ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    if not CHARACTERS.is_dir():
        pytest.skip("set BIFOLD_CHARACTERS_TEST_DIR to a prepared character library")
    from bifold_demo.characters.library import CharacterLibrary

    with pytest.MonkeyPatch.context() as patch:  # the labels' cache goes to a temporary folder, not the library
        patch.setenv(R.CACHE_ENV, str(tmp_path_factory.mktemp("outfit_cache")))
        yield CharacterLibrary(CHARACTERS)


def test_a_real_avatar_renders_in_a_fabric_swatch(gl, library):
    from bifold_demo.characters.character import Character
    from bifold_demo.characters.model import CharacterState
    from bifold_demo.characters.outfit import dress
    from bifold_demo.scene import Camera, Scene
    from bifold_demo.scene.renderer import Renderer

    if "Business_Male_02" not in library.available():
        pytest.skip("Business_Male_02 is not prepared")
    model = library.load("Business_Male_02")
    green = np.clip(make_seamless(cut_out(256, 6, 4, ((20, 110, 50), (120, 200, 90)))), 0, 255).astype(np.uint8)
    swatch = Fabric(green, "green and lime silk blotches", method="synthetic, repaired")  # no model made it
    images = []
    for outfit in (None, {"top": swatch}):
        person = Character(model, behaviour=lambda t: CharacterState((0, 0, 0), 0.0, []), name="p")
        if outfit:
            texture = dress(person, outfit)
            assert isinstance(texture.outfit.fills["top"], Fabric)
        scene = Scene(background=(255, 255, 255), ambient=0.6)
        scene.add(person)
        scene.update(0.0, 0.0)
        scene.camera = Camera(position=(0, 1.2, 1.6), target=(0, 1.2, 0), fov=30)
        images.append(Renderer(120, 120, samples=0).render(scene).astype(int))
    before, after = (image[50:70, 52:68].reshape(-1, 3).mean(axis=0) for image in images)  # the chest, BGR
    assert after[1] > after[2] + 25 and after[1] > after[0] + 25  # green now
    assert abs(before[1] - before[2]) < 25  # dark blue-grey before
