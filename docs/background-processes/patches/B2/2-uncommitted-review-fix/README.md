# B2, 2: the review fix, part way (uncommitted)

`B2-uncommitted-review-fix.patch` is a `git diff --binary` of the builder's uncommitted edits. It applies on top of
[../1-commits](../1-commits/), that is on `7c5a26b`.

**State:** **not committed and not fully tested.** The builder was stopped while writing the tests for findings F7-F9.

- **Fixed in this diff,** each reproduced first with the reviewer's scripts:
  - F1-F3: the cache and the fallback ladder;
  - F4-F5: the kiosk filter, with fashion senses of words and more spellings of brands;
  - F6: freeing models;
  - F7: even light only for clear light gradients;
  - F8: a repair that leaves tiling images alone, without ghosting;
  - F9: a seam score that sees wide stripes and checks;
  - F11: cancelling under the pipeline lock;
  - F12: the CLI.
- **Still to do:**
  - tests for F7-F9;
  - tests for the F10 mutants;
  - the full suite and lint;
  - correcting the README's overstated claims;
  - committing.

**Files:** 6 files, +778/-227: `outfit/fabric.py`, `outfit/seamless.py`, `outfit/fabric_words.json`,
`characters/cli.py`, `people/generator.py`, `tests/test_fabric.py`.

Apply from the repository root, after the commits:

```sh
git -C ../b2-fabric apply --binary "$PWD/docs/background-processes/patches/B2/2-uncommitted-review-fix/B2-uncommitted-review-fix.patch"
```

See [the B2 page](../../../B2-fabric-from-a-description.md), sections "The review" and "The fix, part way".
