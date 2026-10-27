# Background processes: where the work on the two new components stands

The two new components (head textures made from photos, and the character creator with its outfits) and the saved
characters in the fair are built in three tasks. Each task is split into steps. Each step runs in the background in
its own git worktree, in three stages:

1. **implement**: build the step, with tests and docs, and commit in the worktree;
2. **review**: an independent, skeptical review of the code and of the rendered results;
3. **fix**: fix what the review confirmed, in the same worktree.

A step's commits reach the branch only once its review is fixed and the suite passes on the branch.

These pages record what the steps running when work was paused (26 October 2026) found, measured and left open, so
that nothing is lost when they go on. Nothing here is an instruction for the code; the module docstrings and the README
describe the code itself.

| Task | Step | State when paused | Page |
|---|---|---|---|
| A: face → head texture | A1 literature review | on the branch (`e06e165`) | [docs/research/face_head_texture.md](../research/face_head_texture.md) |
| A | A2 UV bake (shared with B) | on the branch (`1d962c7`) | `src/bifold_demo/characters/uvbake.py` |
| A | **A3 landmarks on the head mesh, photos registered** (stages 0-1) | implemented (`04faecd`); the review was interrupted. Since then reviewed, fixed (`d4d7031`, `3f1a6d1`) and on the branch as `e33a8ee`, `76edcc9`, `412fc48` | [A3-head-landmarks-and-registration.md](A3-head-landmarks-and-registration.md) |
| B: character creator | B0 literature review | on the branch (`e494da2`) | [docs/research/outfit_textures.md](../research/outfit_textures.md) |
| B | **B1 garment regions, fills, patterns, presets** (stages 0-1) | implemented (`3b1be67`, `17a2137`) and reviewed; the fix was interrupted, partly done. Since then fixed (`917ae88`) and on the branch as `b2adef8`, `dc46e51`, `2c75ec0` | [B1-outfit-regions-and-fills.md](B1-outfit-regions-and-fills.md) |
| C: saved characters in the scene | C1 the saved format and the book | on the branch (`5054840`) | `src/bifold_demo/characters/created.py` |
| C | **C2 created characters take over places in the fair** | implemented (`a9ea9c2`, `58749f6`, `ba58d8c`) and reviewed; the fix was interrupted, partly done. Since then the fix is finished (`f88536b`) and four independent checkers are testing it | [C2-created-characters-in-the-fair.md](C2-created-characters-in-the-fair.md) |

[environment-and-findings.md](environment-and-findings.md) covers what all steps share: the machine, what can and
cannot be downloaded here, the safety and privacy rules, and findings that matter across components.

## What comes next

- **A**:
  - A3 review and fix;
  - A4: project the photos into the head's UV layout and fuse them, then remove the photos' lighting (stages 3-4);
  - A5: generative completion of what no photo shows, and photo normalisation, behind an interface with
    deterministic fallbacks (stage 5);
  - A6: match the body's skin to the face (using B1's skin mask), level the neck seam, recolour hair and eyes, and
    check the identity (stages 6-7);
  - A7: the gallery, the fair's cast, the CLI and the docs.
- **B**:
  - B1 fix;
  - B2: text to a tileable fabric swatch (a generative model with latent rolling and a fallback ladder), plugged in
    as a `Swatch` fill;
  - B3: prints, decals and restyling the whole outfit;
  - B4: the character creator screen in the app (a turning preview, region tabs, colour and pattern chips, a text box,
    presets, save).
- **C**:
  - C2 fix;
  - C3: docs, CLI and previews.

## How the paused steps go on

Every step runs the same workflow script (implement in a worktree → review → fix), started with the step's brief.
A paused run is resumed from its run ID: stages that already finished return their saved results at once, and the
interrupted stage starts again. A fix agent that starts again finds its predecessor's uncommitted edits in the
worktree and goes on from them. Worktrees live under `.claude/worktrees/` in the main checkout and are not pushed.
Only the commits integrated into the branch are.

| Step | Run | Worktree | Resumes at |
|---|---|---|---|
| A3 | `wf_9c8d2dc1-3c2` | `.claude/worktrees/wf_9c8d2dc1-3c2-1` | review |
| B1 | `wf_c2a91440-c22` | `.claude/worktrees/wf_c2a91440-c22-1` | fix (edits in `outfit/regions.py` uncommitted) |
| C2 | `wf_17f4d8c3-caf` | `.claude/worktrees/wf_17f4d8c3-caf-1` | fix (edits in 9 files and a new `pacing.py`, uncommitted) |

Since the pause, the account's usage limit cut off the A3 review, the B1 fix and C2's checkers once more. All three
resumed the same way once the limit reset. Since then, every review, check and fix stage writes its findings or a
progress log to its scratch folder as it goes, so the next interruption loses none of it.

## Status log

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
