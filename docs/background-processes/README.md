# Background processes: where the work on the two new components stands

The two new components (head textures made from photos, and the character creator with its outfits) and the saved
characters in the fair are built in three tasks. Each task is split into steps. Each step runs in the background in
its own git worktree, in three stages:

1. **implement**: build the step, with tests and docs, and commit in the worktree;
2. **review**: an independent, skeptical review of the code and of the rendered results;
3. **fix**: fix what the review confirmed, in the same worktree.

A step's commits reach the branch only once its review is fixed and the suite passes on the branch.

These pages record what the steps found, measured and left open:

- when work was paused on 26 October 2026;
- when all work was **stopped on request on 27 October 2026, at about 06:40 UTC**.

Nothing here is an instruction for the code; the module docstrings and the README describe the code itself.

## Where each step stands (stopped 27 October, about 06:40 UTC)

| Task | Step | State | Page |
|---|---|---|---|
| A: face → head texture | A1 literature review | on the branch (`e06e165`) | [docs/research/face_head_texture.md](../research/face_head_texture.md) |
| A | A2 UV bake (shared with B) | on the branch (`1d962c7`) | `src/bifold_demo/characters/uvbake.py` |
| A | **A3 landmarks on the head mesh, photos registered** (stages 0-1) | reviewed, fixed, **on the branch** (`e33a8ee`, `76edcc9`, `412fc48`) | [A3-head-landmarks-and-registration.md](A3-head-landmarks-and-registration.md) |
| A | **A4 photos fused into the head texture, lighting taken out** (stages 3-4) | being built: 3 WIP commits plus uncommitted edits; not reviewed | [A4-photos-fused-into-the-head-texture.md](A4-photos-fused-into-the-head-texture.md) |
| B: character creator | B0 literature review | on the branch (`e494da2`) | [docs/research/outfit_textures.md](../research/outfit_textures.md) |
| B | **B1 garment regions, fills, patterns, presets** (stages 0-1) | reviewed, fixed, **on the branch** (`b2adef8`, `dc46e51`, `2c75ec0`) | [B1-outfit-regions-and-fills.md](B1-outfit-regions-and-fills.md) |
| B | **B2 a fabric from a description** (stage 2) | built (`c608f1f`, `7c5a26b`) and reviewed (0 high, 8 medium, 4 low); the fix is mostly done but uncommitted | [B2-fabric-from-a-description.md](B2-fabric-from-a-description.md) |
| C: saved characters in the scene | C1 the saved format and the book | on the branch (`5054840`) | `src/bifold_demo/characters/created.py` |
| C | **C2 created characters take over places in the fair** | built, reviewed, fixed (`f88536b`), checked again by four checkers, fixed again (`2d0a1c3`); the recheck was stopped part way | [C2-created-characters-in-the-fair.md](C2-created-characters-in-the-fair.md) |

[environment-and-findings.md](environment-and-findings.md) covers what all steps share: the machine, what can and
cannot be downloaded here, the safety and privacy rules, and findings that matter across components.

### On the branch, and only in local worktrees

- **On the branch (pushed):**
  - the two literature reviews;
  - the UV bake;
  - the saved-character format;
  - B1, the outfits;
  - A3, the head landmarks and photo registration;
  - these pages.
- **The full suite on the branch** with the real avatars and all face models: 553 passed, 9 skipped. The skips are
  for data that is not here: raw Rocketbox files and wave clips.
- **Only in local worktrees:** the work of B2, C2 and A4. Worktrees live under `.claude/worktrees/` in this
  container's checkout and are **not pushed**, so they are lost if the container is reclaimed.

| Step | Worktree (branch) | Commits not on the branch | Uncommitted |
|---|---|---|---|
| B2 | `.claude/worktrees/agent-a195ea3e9885912b8` (`worktree-agent-a195ea3e9885912b8`) | `c608f1f`, `7c5a26b` | the review fix: 6 files, +778/-227 |
| C2 | `.claude/worktrees/wf_17f4d8c3-caf-1` (`worktree-wf_17f4d8c3-caf-1`) | `a9ea9c2`, `58749f6`, `ba58d8c`, `f88536b`, `2d0a1c3` | none |
| A4 | `.claude/worktrees/agent-a94760ad57c24f7dc` (`worktree-agent-a94760ad57c24f7dc`) | `ba5a144`, `2306bf0`, `792d09e` (WIP) | 4 files, +110/-60 |

The scratch folder of this session (`/tmp/claude-0/-home-user-bifold-demo/44ad5b06-773f-5ffd-8306-515788fc8497/scratchpad/`)
holds each step's brief (`briefs/`), progress logs, review findings and their scripts (`steps/<step>/`). It is local
to this container too.

## What comes next

- **A:**
  - A4: finish it (real-avatar tests, docs, preview, clean commits), then review, fix and integrate;
  - A5: generative completion of what no photo shows, and photo normalisation, behind an interface with
    deterministic fallbacks (stage 5);
  - A6: match the body's skin to the face (using B1's skin mask), level the neck seam, recolour hair and eyes, and
    check the identity (stages 6-7);
  - A7: the gallery, the fair's cast, the CLI and the docs.
- **B:**
  - B2: finish the fix (tests for F7-F10, the full suite, commits), then integrate;
  - B3: prints, decals and restyling the whole outfit;
  - B4: the character creator screen in the app (a turning preview, region tabs, colour and pattern chips, a text box
    for fabrics, presets, save).
- **C:**
  - C2: finish the recheck, decide on its three small findings, integrate;
  - C3: docs, CLI and previews.

## How the work was run, and how it goes on

Steps ran as background agents: workflow runs (implement → review → fix) or single agents, each with a written brief.
Interrupted runs went on from their saved results: finished stages returned at once, and an interrupted stage started
again, finding its predecessor's edits in the worktree and its progress log in the scratch folder.

Since the first pause the work was interrupted several more times:

- the account's usage limit, three times;
- container restarts, three times;
- the stop on request.

Each time it went on without losing work, because every review, check and fix stage writes its findings or a progress
log to its scratch folder as it goes.

To pick a step up again, start from its page here and from its worktree, if the container still has it. The pages
record each step's findings, measurements and remaining work in enough detail to redo the step where the worktree is
gone.

## Status log

**26 October, about 21:30 UTC.** Work paused for this record; then resumed (see the step pages).

**27 October, 00:45 UTC.** The container restarted and stopped all running work. The repository, the step worktrees
and the scratch folders survived, and every finished stage had saved its result, so nothing was lost:

- **A3:** the review is finished: 2 high, 3 medium and 8 low findings. Its fix stage is running.
  - **High:**
    - `View.masks` point into buffers that MediaPipe frees or reuses on its next call: garbage or a segmentation
      fault, and the focal refit of a session reads them;
    - side photos where MediaPipe makes up a face are registered as usable, with absurd poses.
  - **Medium:**
    - a session loses its reference distance and focal length for such photos;
    - outline fits come back "usable" 15-30 degrees off for back-three-quarter photos and other people's hair;
    - `facing()` uses one flat triangle's normal, so visible nose and lip landmarks drop out of the warp in
      three-quarter views.
- **C2:** four independent checkers tested the fixed step (what a viewer sees, robustness, frame timing,
  regressions). Its second fix stage is running, followed by a recheck.
  - **High:**
    - swaps still happen in plain view: a person in the middle of the picture turns into someone else between two
      frames;
    - a `character.json` with a field of the wrong shape (`"items": 5`, a file that is `[]`, `"created": null`)
      ends the demo or stops it from starting.
  - **Medium:**
    - leaving the fair is a jump in view;
    - an "unseen" swap can turn the person round;
    - newcomers made at the same time do not keep 0.6 m from each other;
    - start-up waits on the paced worker;
    - exact swaps are not paced;
    - item pictures are decoded on the main thread;
    - saving a character again deletes its item pictures;
    - a half-written PNG keeps a character out for good;
    - 16-bit PNG textures are drawn white;
    - a sign drops to the hip a second after a character appears.
  - **No regressions** in existing features: the default fair, its renders, camera play, selection mode, faces, R and
    scene reload are unchanged.
- **B2:** the text-to-fabric step started on top of B1 and goes on in its worktree
  `.claude/worktrees/agent-a195ea3e9885912b8`. It has read the diffusers source, settled its design and started three
  modules.

**27 October, 05:45 UTC.** The usage limit stopped everything once more, and the container restarted twice. Each time
the runs went on from their saved results and progress logs, and no work was lost.

- **A3:** fixed and on the branch (see its page).
- **B2:** built in two commits, `c608f1f` (fabrics from a description, a seamless swatch as a fill) and `7c5a26b` (the
  repair's cut search holds the GIL far less). The builder reports 528 passed and 9 skipped. An independent review
  is running. No real text-to-image model can run here, so the generative path is tested with fakes and with tiny
  random diffusers models built locally.
- **C2:** its second fix is still running; the recheck follows.
- **A4** (project and fuse the photos in the head texture, then take out their lighting) is briefed and starts next.

**27 October, about 06:40 UTC.** All work stopped on request, unfinished:

- **B2:** reviewed (0 high, 8 medium, 4 low). Its builder had worked fixes for most findings into the code,
  uncommitted, and was writing the last tests.
- **C2:** its second fix was committed (`2d0a1c3`). The recheck had confirmed every reproduction it re-ran, and found
  three small new issues and one wording issue.
- **A4:** three WIP commits and uncommitted edits. The fusion, the light and the CLI work, with 16 tests; the
  real-avatar tests, the docs and the preview were not done.
- **The scheduled check** to resume stopped runs was deleted.
