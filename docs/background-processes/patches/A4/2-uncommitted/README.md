# A4, 2: the edits after the last WIP commit (uncommitted)

`A4-uncommitted.patch` is a `git diff --binary` of the builder's uncommitted edits. It applies on top of
[../1-wip-commits](../1-wip-commits/), that is on `792d09e`.

**State:** work in progress, not committed. These are the builder's last changes to the fusion, the light and the
end-to-end function, made while it wrote the real-avatar tests and measured the frame rate with a head texture being
made on a worker.

**Files:** 4 files, +110/-60: `headtex/__init__.py`, `headtex/fuse.py`, `headtex/light.py`, `headtex/texture.py`.

Apply from the repository root, after the WIP commits:

```sh
git -C ../a4-fuse apply --binary "$PWD/docs/background-processes/patches/A4/2-uncommitted/A4-uncommitted.patch"
```

See [the A4 page](../../../A4-photos-fused-into-the-head-texture.md), section "Still to do".
