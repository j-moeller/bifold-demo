# B2, 1: the step's two commits

`B2-commits.patch` is a `git format-patch` mailbox of two commits on base `c297544` ("Docs: the outfit step is on the
branch; how interrupted steps go on"):

1. `c608f1f` Outfits: fabrics from a description, a seamless swatch as a fill
2. `7c5a26b` Outfits: the fabric repair's cut search holds the GIL far less

**State:** built and independently reviewed: no high findings, 8 medium, 4 low. The builder's full suite run with the
real avatars gave 528 passed, 9 skipped. The review's fixes are in [../2-uncommitted-review-fix](../2-uncommitted-review-fix/).

**Files:** 14 files, +2902/-32:

- new: `outfit/fabric.py`, `outfit/seamless.py`, `outfit/tiling.py`, `outfit/fabric_words.json`, `tests/test_fabric.py`,
  `fabrics/README.md`, `docs/previews/fabric.jpg`;
- changed: `outfit/fills.py`, `outfit/__init__.py`, `characters/cli.py`, `people/generator.py`, `README.md`,
  `docs/previews/README.md`, `.gitignore`.

Apply from the repository root:

```sh
git worktree add ../b2-fabric c297544
git -C ../b2-fabric am "$PWD/docs/background-processes/patches/B2/1-commits/B2-commits.patch"
```

See [the B2 page](../../../B2-fabric-from-a-description.md).
