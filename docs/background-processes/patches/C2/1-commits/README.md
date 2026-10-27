# C2, 1: the step's five commits

`C2-commits.patch` is a `git format-patch` mailbox of five commits on base `5054840` ("Created characters: a saved
format and a book of them"):

1. `a9ea9c2` Camera play: one more player can join those who play already
2. `58749f6` Created characters: read images when used, new looks replace the old
3. `ba58d8c` Created characters in the fair: they take over places, live
4. `f88536b` Created characters in the fair: unseen swaps, a paced worker, bad looks reported
5. `2d0a1c3` Created characters in the fair: changes nobody sees, a book that never stops the demo

**State:** built, reviewed and fixed (commit 4). Then checked again by four independent checkers and fixed again
(commit 5). The fixer's full suite run with the real avatars gave 532 passed, 9 skipped. The recheck of commit 5 was
stopped part way: everything it re-ran held, and it found three small new issues. There were no uncommitted edits.

**Files:** 24 files, +3701/-120, among them:

- new: `scenes/created_cast.py`, `pacing.py`, `tests/test_created_cast.py`, `tests/test_pacing.py`;
- changed: `characters/created.py`, `behaviour.py`, `camera_play.py`, `crowd.py`, `scenes/fair.py`, `scene/renderer.py`,
  `scene/textures.py`, `selection.py`, `app.py`;
- the README and two previews.

The base `5054840` is older than B1 and A3 on the branch, so expect the usual conflicts in the README when replaying it
onto the branch.

Apply from the repository root:

```sh
git worktree add ../c2-created 5054840
git -C ../c2-created am "$PWD/docs/background-processes/patches/C2/1-commits/C2-commits.patch"
```

See [the C2 page](../../../C2-created-characters-in-the-fair.md).
