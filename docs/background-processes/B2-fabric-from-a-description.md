# B2: a fabric from a description (text to a seamless swatch, as an outfit fill)

**State when stopped (27 October, about 06:40 UTC, on request).**

- **Commits:** two, in the worktree `.claude/worktrees/agent-a195ea3e9885912b8` (branch
  `worktree-agent-a195ea3e9885912b8`, base `c297544`):
  - `c608f1f` "Outfits: fabrics from a description, a seamless swatch as a fill";
  - `7c5a26b` "Outfits: the fabric repair's cut search holds the GIL far less".
- **Not on the branch:** neither commit.
- **Review:** an independent review found no high issues, 8 medium and 4 low (see [the review](#the-review)).
- **Fix:** the builder was fixing them, with the work uncommitted in the worktree, when it was stopped: 6 files,
  +778/-227 lines (see [the fix, part way](#the-fix-part-way)).
- **To finish:** tests for the last three findings, the full suite, lint, commits, then integration.
- **Saved:** the commits are in [patches/B2/1-commits](patches/B2/1-commits/) and the uncommitted fix in
  [patches/B2/2-uncommitted-review-fix](patches/B2/2-uncommitted-review-fix/).

The approach is Stage 2 of [the outfit literature review](../research/outfit_textures.md) (section 3), with its
fallback ladder. **No real text-to-image model can run here** (Hugging Face is not reachable). The generative path is
written against diffusers 0.42.0's own source and tested with fake pipelines and with tiny, randomly initialised
diffusers models built locally. How it behaves with the real weights on a Mac is unmeasured.

## What it builds

- **`outfit/fabric.py`, the fabric generator.**
  - The description is cleaned, checked against a kiosk word filter, and put into the review's prompt template
    ("seamless tileable fabric texture, flat lay, top-down, even diffuse light, no folds, ...").
  - A quick procedural preview comes first.
  - Then the fallback ladder:
    - FLUX.2-klein at 4 steps, then at 12 steps;
    - then SD-Turbo;
    - then a model-free repair of the best rejected image;
    - then the keyword fabric (B1's parser);
    - then a plain colour (the nearest colour word, else grey).

    Every step down is explained in the job's messages. Missing packages, weights or network go straight to the
    keyword fabric, with the install or download command.
  - Results arrive one by one, can be cancelled, run on a daemon thread, and are cached in the git-ignored `fabrics/`
    (rejected swatches too, so a description that is cached in full never loads a model).
- **`outfit/tiling.py`: making a model paint a repeat, for one call only.**
  - For FLUX.2-klein and Z-Image the latents are rolled in `callback_on_step_end`:
    - FLUX.2's packed tokens `(B, h*w, 128)` are unpacked to their grid and rolled by whole tokens;
    - Z-Image's `(B, 16, H/8, W/8)` latents are rolled by whole 2x2 patches;
    - the offsets are random, taper to 0 for the last step, and the total is undone;
    - the VAE decoder pads circularly.
  - SD-Turbo gets circular padding in its UNet and VAE.
  - Patching flips `padding_mode` in place under a per-pipeline lock and restores it on every path.
- **`outfit/seamless.py`, the model-free parts:**
  - a seam score (a step, a line or a broken rhythm at the wrap, each against the strongest such feature inside), with
    rejection above 1.25;
  - a repair (roll by half, quilt the cross along a minimum-error cut);
  - even light (divide out the light that varies once across the swatch).
- **The `Fabric` fill** (a `Swatch`): its spec keeps the description, prompt, model, seed, method, seam score and the
  image as lossless WebP. Outfits and saved characters rebuild it exactly, without the model. The repeat size follows
  the kind of fabric and can be overridden.
- **Kiosk filter** (`fabric_words.json`, extensible through `$BIFOLD_FABRIC_WORDS`): brands, characters, marks and
  offensive words. It matches whole words regardless of case, accents, spacing, look-alike digits and plurals.
- **The rest:**
  - `people/generator.py` extended (steps and extra pipeline arguments, a lock, `local_files_only`, a shared
    generator), with existing users unaffected;
  - the CLI `bifold-characters fabric "..." -o swatch.png [--model ...] [--seeds 4] [--avatar NAME --region top
    --views views.png]`;
  - a README section with notes for the Mac;
  - `docs/previews/fabric.jpg`, made only from synthetic and procedural textures and labelled as such.

## API (for B4, the creator screen)

- `FabricGenerator(model="flux2-klein"|"z-image-turbo"|"sd-turbo"|HF id|None, device="auto", seeds=4, size=512,
  threshold=1.25, cache=True|folder|False, words=True|PromptFilter|None, loader=None, download=True, pace=None,
  ladder=None)`:
  - `.start(text, seeds=None, first_seed=0, size=None, on_result=None, on_message=None) -> FabricJob` returns at once;
    a refused or empty description raises `PromptRefused` / `ValueError` at once;
  - also `.generate(...)` (waits), `.prepare(text)` and `.preview(text)`.
- `FabricJob`:
  - `.poll()`, iteration, `.results`, `.variations`, `.ranked`, `.best`;
  - `.cancel()` (at the end of the next denoising step), `.wait(timeout)`;
  - `.status`, `.messages`, `.progress`.
- `FabricSwatch`: `.image` (uint8 RGB), `.fill(scale=None, rotation=None) -> Fabric`, `.describe()`.
- `Fabric(image, text, scale=None, ...)`, with `.spec()` and `Fabric.from_spec()`; `fills.from_spec({"fabric": ...})`.
- `fabric_fill(text, generator=None, ...) -> Fabric`.
- `PromptFilter`: `.check()`, `.problem()`, `.add()`, `.allow()`.
- `seamless`: `seam_score`, `make_seamless`, `equalise_light`.
- `tiling`: `LatentRoller`, `circular_padding`, `plan_for`.
- Environment: `BIFOLD_FABRIC_CACHE`, `BIFOLD_FABRIC_MODEL` (`none` for no model), `BIFOLD_FABRIC_WORDS`.

## Checks the builder reports

- **Full suite** on `7c5a26b` with the real avatars: 528 passed, 9 skipped. No new lint findings.
- **`tests/test_fabric.py`, 34 tests:**
  - rolling with the real latent layouts ends exactly as without rolling;
  - FLUX.2 packing matches diffusers' own;
  - a zeroed transformer gives a rolled Flux2KleinPipeline run equal to the plain run;
  - a patched VAE decoder tiles, and is restored afterwards;
  - each ladder rung failing in turn;
  - the filter, the cache, cancelling;
  - spec round trips through Outfit and through saved characters;
  - the CLI, and a real avatar dressed in a fabric.
- **Calibration on synthetic textures:**
  - 2 of 150 seamless textures score over 1.25;
  - the repair brings 46 of 48 below it.
- **Timings:** a seam score 30-60 ms, even light 10-20 ms, a repair 0.1-0.15 s at 512 px.
- **Main thread:** with a worker repairing nonstop, the 99th-percentile tick lateness dropped from 36 ms to about 1 ms
  after `7c5a26b`.

## The review

An independent reviewer checked the step on copies of `7c5a26b`. Its write-up and scripts are in the scratch folder
`steps/B2_fabric_review/` (`findings.md`, `scripts/`, `out/`).

**Confirmed true:**

- the latent rolling on the real (tiny) Flux2KleinPipeline and ZImagePipeline:
  - with a denoiser that commutes with shifts, the final latents equal the plain run bit for bit at 1, 2, 4 and 12
    steps;
  - every step's input is the plain input rolled by its scheduled offset;
  - both negative controls differ;
- circular padding on every decoder convolution, restored after errors and cancels;
- the no-network path;
- the job API, the cache, specs that round-trip without loss;
- the GIL fix;
- the suite and lint;
- the honestly labelled preview.

**Findings:**

- **Medium:**
  - **F1:** a cached swatch keeps the accept/reject verdict of whatever threshold wrote it (the threshold is not part
    of the key).
  - **F2:** a cache write error fails the whole job and throws away the swatch the model just made.
  - **F3:** a model that fails after some seeds were kept still walks down the ladder (SD-Turbo is loaded on top of
    klein; the keyword fabric is added next to the kept swatches).
  - **F4:** the kiosk filter refuses "nude", a standard fashion colour ("nude satin", "nude lace"), and other innocent
    phrases ("naked wool", "cock pheasant feathers").
  - **F6:** `release_shared()` frees nothing a `FabricGenerator` holds, and the next generator loads a second copy
    (2 × about 16 GB for klein).
  - **F7:** even light reads the layout of a print's motifs as light, so evenly lit prints come out blotchy:
    - the ground's brightness spread went from 1.00 to 1.28-2.47;
    - light grounds clip;
    - white daisies turn grey.
  - **F8:** the repair leaves ghosted motifs and straight brightness steps that the seam score cannot see, and it made
    an already seamless flower print worse (score 0.40 → 0.81).
  - **F9:** the seam score passes most wide stripes, checks and gingham whose repeat does not fit the swatch, scoring
    about 1.0 (for example 20 of 34 stripe widths).
- **Low:**
  - **F5:** ordinary spellings of listed brands and characters get through ("Dolce and Gabbana", "Saint Laurent",
    "Mickey print", "Levi denim", "f*ck"), and `problem("gucci!")` returns nothing.
  - **F10:** six claimed behaviours have no test that fails without them; 13 other mutants were caught.
  - **F11:** a job cancelled while it waits for the pipeline lock still starts its call, and identical concurrent
    jobs repeat each other's work.
  - **F12:** rough edges in the CLI:
    - `--seeds 0` ends in a traceback;
    - `--texture` without `--avatar` is ignored;
    - `-o` writes 16 × 16 for a plain colour;
    - the `--sheet` write is unchecked.

**Overstated claims:** even light "keeps the colours" (not on prints); models stay loaded "until `release_shared()`";
"every cut-out caught" (2 of 25 fine twills pass); a spec of about 370 KB per swatch (a photo-like 512 px swatch makes
about 680 KB).

## The fix, part way

The builder reproduced F1-F12 with the reviewer's scripts. Its progress log is the scratch file
`steps/B2_fabric/progress.md`, and it was stopped while writing the last tests. In the worktree, **uncommitted**:

- **F1:** the cache stores the raw images (cache version 2) and judges them again on reading.
- **F2:** a cache error costs a message, never the result.
- **F3:**
  - a model that fails once seeds were kept ends the job there;
  - after running out of memory, the model is let go before the next one loads, and that is remembered until
    `release()`.
- **F6:** `FabricGenerator.release()`, a weak registry of live pipelines in `people/generator.py`,
  `release_shared(pipe)` and `free_memory()`.
- **F11:** a cancel check and a cache look under the pipeline lock.
- **F12:** seeds checked, `--texture` needs `--avatar`, `-o` at `--size`, every write checked.
- **F4/F5:**
  - words like "nude", "naked", "cock" and "tit(s)" count only in their fabric sense, and never next to words about
    people, the body or photos;
  - wildcards (`*`), a trailing "!", and more spellings of brands and characters.
- **F7:** even light now takes out a light gradient only when it is clearly one:
  - a robust fit on coarse gradients;
  - a change of at least 5 %;
  - it stands out from the next band;
  - the colour changes little against the brightness.
- **F8:**
  - an image that already tiles is returned as it is;
  - the overlap is chosen by cut cost;
  - offset ramps replace the coarse cross-fade;
  - a wide fade only for images without crisp edges;
  - a final pass of periodic light.
- **F9:**
  - the step test runs per eighth of the seam;
  - the repeat test uses windows of 32-128 columns.
- **Calibration after the changes** (the reviewer's sets):
  - seamless textures rejected as before (2/150, 2/160; 0/30 prints, 0/24 fitting bands);
  - wide bands missed: 0/102, from 57;
  - broken rhythms missed: 0/36, from 1;
  - partial seams missed: 1/12, from 4;
  - fine twill cut-outs: 2/100, as before.
- `tests/test_fabric.py` has 148 new lines. The tests for F7, F8 and F9 were being written.

**Still to do:**

- the tests for F7-F9;
- tests for the F10 mutants;
- the full suite and lint;
- correcting the overstated claims in the README;
- commits.

## Open issues the builder admits

- **Unmeasured until a real model runs:** klein's actual seams with 4-step rolling, the rejection rate, the 1.25
  threshold on real images, the per-fabric scales and the Mac timings.
- **The seam score's limits:** it cannot check a repeat longer than half the swatch. A single mirrored repeat (a tartan
  sett) can score up to about 1.5 though it tiles.
- **The repair's limit:** the very smoothest textures (about three blobs across) still fail and fall back to the
  keyword fabric.
- **Loading a model** cannot be cancelled half way.
- **A failed load is remembered** per generator.
- **For B4:** `fabric_fill()` without a generator allows downloads, so the creator should pass
  `FabricGenerator(download=False)`.
- **The kiosk:** generated images are not checked for inappropriate content (a risk the literature review lists).
