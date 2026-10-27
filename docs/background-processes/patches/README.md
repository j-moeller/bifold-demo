# Patches: the unfinished work, saved

When all work was stopped on request (27 October 2026, about 06:40 UTC), three steps existed only in local git
worktrees, which are not pushed and are lost if the container is reclaimed. Their work is saved here, one patch per
directory.

- **Commits** are saved as `git format-patch` mailboxes, applied with `git am`. They keep each commit's message,
  author and date.
- **Uncommitted edits** are saved as `git diff --binary` patches, applied with `git apply` on top of their step's
  commits.

| Directory | What | Base | State |
|---|---|---|---|
| [B2/1-commits](B2/1-commits/) | B2 (a fabric from a description): 2 commits | `c297544` | reviewed: 0 high, 8 medium, 4 low |
| [B2/2-uncommitted-review-fix](B2/2-uncommitted-review-fix/) | B2: the fix of the review, part way | B2/1-commits | uncommitted, not fully tested |
| [C2/1-commits](C2/1-commits/) | C2 (created characters in the fair): 5 commits | `5054840` | reviewed, checked, fixed twice; recheck stopped part way |
| [A4/1-wip-commits](A4/1-wip-commits/) | A4 (photos fused into the head texture): 3 WIP commits | `2eefb1d` | work in progress, not reviewed |
| [A4/2-uncommitted](A4/2-uncommitted/) | A4: the edits after the last WIP commit | A4/1-wip-commits | uncommitted, work in progress |

**Checked:** before these were committed, each step's patches were applied to its base in a fresh checkout. The
resulting trees equal the worktrees exactly, uncommitted edits included.

The step pages one level up say what each step builds, what was found and what is left:

- [B2-fabric-from-a-description.md](../B2-fabric-from-a-description.md)
- [C2-created-characters-in-the-fair.md](../C2-created-characters-in-the-fair.md)
- [A4-photos-fused-into-the-head-texture.md](../A4-photos-fused-into-the-head-texture.md)

## Applying a step

The bases are older commits of this branch, so apply a step in a separate worktree made from its base. Run these
commands from the repository root, where the patches are. For B2, for example:

```sh
git worktree add ../b2-fabric c297544
git -C ../b2-fabric am "$PWD/docs/background-processes/patches/B2/1-commits/B2-commits.patch"
git -C ../b2-fabric apply --binary "$PWD/docs/background-processes/patches/B2/2-uncommitted-review-fix/B2-uncommitted-review-fix.patch"
```

To integrate a finished step, replay its commits onto the branch, as the other steps were. The usual conflicts are in
the README, its project layout and `docs/previews/README.md`, where each step adds lines; keep both sides.

The patches hold no real photos or anything made from them. Their only images are previews made from avatar renders
and synthetic textures (`docs/previews/fabric.jpg`, `created_characters.jpg`, `created_swaps.jpg`).
