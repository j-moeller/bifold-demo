# A4: the photos fused into the head texture, their lighting taken out (head textures, stages 3 and 4)

**State when stopped (27 October, about 06:40 UTC, on request).**

- **Commits:** three work-in-progress commits, meant to be squashed into clean ones at the end, in the worktree
  `.claude/worktrees/agent-a94760ad57c24f7dc` (branch `worktree-agent-a94760ad57c24f7dc`, base `2eefb1d`):
  - `ba5a144` "WIP head textures stages 3-4";
  - `2306bf0` "WIP ...: tests, anchored gains, light weights";
  - `792d09e` "WIP ...: avatar tests, neck seam, exact eyes".
- **Uncommitted edits** in `fuse.py`, `light.py`, `texture.py` and `headtex/__init__.py` (+110/-60 lines).
- **Not reviewed** and not on the branch.

The step's brief is the scratch file `briefs/A4_fuse.md` and its progress log is `steps/A4_fuse/progress.md`. The
approach is stages 3 and 4 of [the literature review](../research/face_head_texture.md) (section 3). It builds on A3
(stages 0-1), which is on the branch.

## What is built so far

- **`headtex/raster.py`, `fuse.py`, `light.py` and `texture.py`.** For each usable photo, every texel of the head
  texture finds the photo pixel that shows it, with a depth test against the head.
- **Weights:**
  - the viewing angle;
  - the distance to mask edges;
  - the segmentation class;
  - the view's quality;
  - a ramp at the head's outline, so the background does not bleed in at the silhouette.
- **Gain compensation** (Brown-Lowe least squares), anchored on the most frontal photo.
- **Multi-band blending** in UV, push-pull padding, and a confidence map with a keep / refine / generate trimap for
  stage 5.
- **Light:**
  - an order-2 spherical-harmonics fit of each photo's luminance on the known normals, robust and regularised;
  - the photo's shading divided out within the fitted range, and the avatar's own shading put back;
  - the light weights are computed after the gains, so a photo of the shadowed side does not take its shadow for
    light.
- **A skin target** for stage 6.
- **`head_texture()` end to end,** and the CLI `bifold-faces headtex`.
- **`tests/test_headtex_fuse.py`:** 16 tests, passing in 17 s. Real-avatar tests (render and reconstruct, another
  avatar's head, the neck seam, face against hands) were being written.
- **Fixed on the way:**
  - `Renderer.render` now enters its own GL context;
  - the face warp got a lean float64 kernel for exact calls and float32 for its grid (10 times faster; A3's tests
    pass).

## Findings so far

- **The head atlas** is 1024 px in the prepared cache, and a bake takes 0.6-2.3 s:
  - the face is in the middle, with the sides, ears and hair beside it and the neck below;
  - the inner mouth and one eyeball, shared by both eyes, are bottom left (that is the 1 % overlap);
  - female heads include long hair islands and a large neck and shoulder area.
- **What the segmenter calls what** (4 avatar renders and 4 real photos):
  - brows, lips, light beards and ears are face skin;
  - the neck and hands are body skin;
  - a microphone is "others".

  So brows and beards need no hair allowance, and neck texels accept body skin.
- **Masks of the mesh:**
  - `head_regions` labels are smoothed over the mesh and spread into unlabelled hair strands;
  - hair islands count as scalp, and the scalp never takes face skin;
  - the inner mouth comes from 33 z-buffers with the body as occluder;
  - the eye openings are marked.
- **Harsh light,** synthetic: three photos under a fixed light, at ambient 0.5 down to 0.08. The left-right brightness
  gap of the face went from 11.5-32 L to 1.4-13.5 L. With one front photo alone it went down to 0.4-5.2 L.
- **The same avatar rebuilt from its own renders** (4 photos at 1280x960, varied light):
  - face ΔE median 4.46, 90th percentile 9.3, PSNR about 24.4;
  - L offset -1.6 to -1.9;
  - the skin target about 3.3 ΔE from the avatar's own skin.
- **Timing:** fusion about 4 s for 4 photos, plus about 1 s of registration.
- **Environment notes:**
  - no scipy;
  - `cv2.remap` maps must be under 32767 px wide;
  - `INTER_AREA` resizing takes at most 4 channels;
  - the harness refuses `source` and compound commands, so the step used a wrapper script (`steps/A4_fuse/run.sh`).

## Still to do

- **Finish the real-avatar tests:** reconstruct onto the same avatar, onto another avatar's head, ΔE across the neck
  and between face and hands. Also a visual check on the real test photos, never committed.
- **Measure the frame rate** with a head texture being made on a worker; it was being measured when stopped.
- **Docs:** the README section, `docs/previews/headtex_fuse.jpg` (from renders only) with its entry, and the package
  docstring.
- **Wrap up:** the full suite, lint, and squashing the WIP commits into clean ones.
- **Then** an independent review, a fix, and integration.
