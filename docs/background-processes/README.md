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
| A | **A3 landmarks on the head mesh, photos registered** (stages 0-1) | implemented (`04faecd`); the review was interrupted | [A3-head-landmarks-and-registration.md](A3-head-landmarks-and-registration.md) |
| B: character creator | B0 literature review | on the branch (`e494da2`) | [docs/research/outfit_textures.md](../research/outfit_textures.md) |
| B | **B1 garment regions, fills, patterns, presets** (stages 0-1) | implemented (`3b1be67`, `17a2137`) and reviewed; the fix was interrupted, partly done | [B1-outfit-regions-and-fills.md](B1-outfit-regions-and-fills.md) |
| C: saved characters in the scene | C1 the saved format and the book | on the branch (`5054840`) | `src/bifold_demo/characters/created.py` |
| C | **C2 created characters take over places in the fair** | implemented (`a9ea9c2`, `58749f6`, `ba58d8c`) and reviewed; the fix was interrupted, partly done | [C2-created-characters-in-the-fair.md](C2-created-characters-in-the-fair.md) |

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
