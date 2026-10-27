# A4, 1: three work-in-progress commits

`A4-wip-commits.patch` is a `git format-patch` mailbox of three WIP commits on base `2eefb1d` ("Docs: the
head-landmarks step is on the branch; B2 built, under review"). The builder meant to squash them into clean commits
at the end.

1. `ba5a144` WIP head textures stages 3-4 (to be squashed)
2. `2306bf0` WIP head textures stages 3-4: tests, anchored gains, light weights (to be squashed)
3. `792d09e` WIP head textures stages 3-4: avatar tests, neck seam, exact eyes (to be squashed)

**State:** work in progress, not reviewed. Built so far:

- the fusion of the photos in the head texture: `fuse.py`, `raster.py`;
- the light and skin tone: `light.py`;
- `head_texture()` end to end: `texture.py`;
- the CLI `bifold-faces headtex`;
- 16 synthetic tests (`tests/test_headtex_fuse.py`), and real-avatar tests being written
  (`tests/test_headtex_fuse_avatars.py`);
- two small fixes outside the step: `Renderer.render` enters its own GL context, and the face warp is faster.

**Files:** 12 files, +3056/-18.

Apply from the repository root:

```sh
git worktree add ../a4-fuse 2eefb1d
git -C ../a4-fuse am "$PWD/docs/background-processes/patches/A4/1-wip-commits/A4-wip-commits.patch"
```

See [the A4 page](../../../A4-photos-fused-into-the-head-texture.md).
