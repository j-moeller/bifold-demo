# C2: created characters take over places in the fair

**State when paused.** Implemented in three commits in the worktree `.claude/worktrees/wf_17f4d8c3-caf-1`:

- `a9ea9c2` "Camera play: one more player can join those who play already";
- `58749f6` "Created characters: read images when used, new looks replace the old";
- `ba58d8c` "Created characters in the fair: they take over places, live".

It is reviewed. The fix was interrupted while partly done: uncommitted edits in 9 files plus a new module (see the end
of this page). The fix starts again from those edits. The commits are not on the branch yet.

**Update, after the pause.** The resumed fix finished from those edits in one commit, `f88536b` "Created characters
in the fair: unseen swaps, a paced worker, bad looks reported". It reports every review finding as fixed (see [the fix,
finished](#the-fix-finished)). Before it reaches the branch, four independent checkers test the fixed step: what a
viewer sees, robustness, frame timing, and regressions. A further fix and a recheck follow if they find anything.

## What it builds

Characters saved in `created/` (by `characters/created.py`, already on the branch) now appear in the fair. Each one
takes over the place of a generic grown-up and keeps that place's behaviour.

- **Which places:** approach visitors first, then walkers, then wanderers. A child's place is never taken.

### Exact swaps

- The builder measured `stride_scale` on the 20 avatars:
  - grown-ups of one sex all get exactly 1.0 with their own sex's walk clips;
  - with the other sex's clips it is 1.032 (women) or 0.969 (men), about 3 % foot sliding.
- So, within each kind, places whose person has the same sex and age group come first.
- There the new 3D person takes over the place's own behaviour object (camera play included), plus a new PlayPose for
  its model. Position and timing are identical, so `keep_apart` and the camera play hold by construction.

### Rebuilt behaviours

Otherwise the place's behaviour is made again for the new avatar:

- the same factory and seed, the avatar's own clips and stride;
- it walks at the old person's speed when its clips can keep up (within 25 % cadence), and its loop is as long as the
  old one;
- `fair.fit_in` re-runs `keep_apart`'s 0.6 m check for this one person against everyone else. Among the timings that
  keep apart, it picks the one closest to the old track;
- `fair.join_camera_play` plans the newcomer's camera play around everyone else's moves, which stay as they are;
- the swap waits until the new person stands where the old one stands (within 5 cm), at most 60 s of scene time.

On the real fair, a man put in a woman's visitor place got:

- male clips and the same 60 s loop;
- a median track difference of 0.00 m (max 0.51 m);
- at most 2.8 % clip stretch (the limit is 5 %);
- `keep_apart` still found 0 conflicts.

### What other parts see

- The person keeps the place's name (for example `approach1`) and its kind, and `person.created` is the
  CreatedCharacter.
- `recognition/cast.py` skips them; `FaceCast.put` and the face item in selection mode refuse them, because they have
  their own face.
- The selection tag reads `approach1 (Ada)`. Yoga and items work as on anyone, and the character's own accessories
  are worn.

### The app

- It owns a `CreatedCast` over a `CharacterBook`: `DemoConfig.created`, `--created DIR`, default `created`, `''` for
  none.
- It runs on every scene frame:
  - new characters take the next free place;
  - changed ones are updated in place (new looks on the same person; another avatar gives a new person in the same
    place);
  - deleted ones give the place back to the original person.
- It says "Ada joined the fair" and shows a status line `created: Ada, Ben in the fair`.
- Restart (R) and scene reload keep them.

### Main-thread cost

- Making the 3D person takes 0.03-0.1 s when exact and 0.57 s when rebuilt, so it runs on a worker thread.
- Decoding the PNGs cost about 37 ms per 1024² texture, so `created.py` now reads images lazily (`Textures`), on the
  worker.
- The builder measured the cast's work at most 1.8 ms per frame and the swap 0.4 ms. The review found this leaves out
  the worker's effect on the main thread (see below).

### Docs

- a README section "Created characters" (where they are saved, what they replace, privacy, how to remove one);
- a pointer in the fair section and the project layout;
- `created/README.md`;
- `docs/previews/created_characters.jpg` with its entry.

## API

### `scenes/created_cast.py` (new)

- `CreatedCast(book: CharacterBook, library=None, wardrobe=None, threads=True)`:
  - `.update(scene, now=None, wait=False) -> list[str]`: call it after every frame; it returns messages. With
    `wait=True` it finishes all work now and swaps at once (for a scene nobody has seen yet).
  - `.status() -> str | None`, `.close()`.
  - `.placed: dict[name, Placement(created, original, person, version, avatar)]` and `.waiting: list[str]`.
- Constants: `PLACES = ("approach", "walker", "wanderer")`, `CADENCE = 1.25`, `JUMP = 0.05`, `LINE_UP = 60.0`,
  `LEAD = 1.0`.
- `places(scene, entry=None) -> list`: the generic people whose place may be taken, best first.
- `stand_in(original, created, library, people=(), *, wardrobe=None, now=0.0) -> Character`: builds the 3D person
  for a place, exact or rebuilt. It is safe on a worker thread.
- `remade(recipe, style, like)` and `put(scene, person)`.
- `slot_behaviour(person)`: the behaviour without selection mode's Performer.
- `planned_behaviour(person)`: also without the CameraPlay.

### Changes elsewhere

- `scenes/fair.py`:
  - `add_3d_people(scene, library, rng, slots=None, names=None)` now sets `person.kind`;
  - `fit_in(behaviour, others, *, like=None, period=LOOP_PERIOD, step=0.1, batch=16) -> int`: `keep_apart` for one
    newcomer; it returns how many are still too close;
  - `join_camera_play(person, people) -> CameraPlay | None`.
- `characters/camera_play.py`:
  - `plan_camera_play(..., planned=())` takes `(Player, moves)` pairs that are already fixed;
  - `join(person, camera, *, players=(), others=(), period, rng, moves, share, place) -> CameraPlay | None`.
- `characters/crowd.py`:
  - `Recipe(make, style, seed, library)` with `.remake(style, **options)`;
  - `same_style(a, b) -> bool`;
  - `populate()` now sets `person.recipe`.
- `characters/character.py`: `created = None` and `recipe = None` by default.
- `characters/behaviour.py`: `Behaviour.walk_speed() -> float | None` and `Behaviour.walk_rates() -> list[float]`.
- `characters/created.py`:
  - `Textures`, a lazy, thread-safe MutableMapping;
  - `load(path, lazy=False)`; the CharacterBook loads lazily;
  - `apply_looks(person, created, wardrobe=None, t=None)` takes off old items and finds item files in the
    character's folder.
- `selection.py`: `person_name()` shows `created.name` instead of the avatar's name.
- `app.py`: `DemoConfig.created = "created"`, `--created`, and `DemoApp.created` (the CreatedCast).

## Checks the builder reports

- **Full suite:** 421 passed, 43 skipped (xvfb, without the real avatars). The touched modules' real-avatar tests: 72
  passed, 1 skipped.
- **`tests/test_created_cast.py`**, 14 synthetic tests and 2 on the real avatars:
  - exact swaps: same behaviour and position, PlayPose re-attached, textures and a hat worn, `keep_apart` 0;
  - updates in place, a new avatar in the same place, a delete gives the original person back, and a character saved
    by another writer joins;
  - rebuilt behaviours: the avatar's own sex, the stride check, the period, camera play only inside watch windows,
    `keep_apart` 0, a median track difference under 0.5 m;
  - children's places are never taken; extra characters wait; a missing avatar is reported once;
  - FaceCast and selection refuse to put a face on them;
  - R and reload keep them; the threaded path swaps when lined up;
  - in the app, a character saved while it runs joins within one rescan plus 1.5 s;
  - real avatars: three characters walk up to the camera, and the magenta outfit is drawn on screen.
- **`tests/test_camera_play.py`:** `join` plans around the others.
- **`tests/test_created.py`:** new looks replace the old; images are read lazily.
- **Lint:** no new ruff findings.

## The review's findings (for the fix)

Verdict: it mostly does what the builder claims. Nothing is high, but three medium issues should be fixed first.

The reviewer confirmed:

- **Tests:**
  - 423 passed, 43 skipped without the real avatars;
  - 15 of 15 in `test_created_cast.py` with them;
  - 139 passed and 1 skipped in the related modules.
- **Swaps on the real fair:**
  - each kind fills in order, never a child's place;
  - `keep_apart` 0, and no close encounters over 360 s;
  - the camera-play moves stay apart, and the PlayPose wave renders the same on the new avatar.
- **Feet:** the distance moved per clip stride stays within 5 % in rebuilt places.
- **Faces:** FaceCast moved a known person's face elsewhere when a created character took the place, and no face
  appeared twice.
- **Renders:** with hats and glasses they look right.

The findings:

- **Medium: a newcomer can pop into view mid-aisle.**
  - `_lined_up` returns True as soon as the old person is hidden. It never checks whether the new one is in view at
    that moment.
  - Teleporting walkers vanish at the wall at the end of every pass. A rebuilt newcomer with another loop length is
    then usually mid-walk, and appears from nowhere in the near aisle.
- **Medium: one malformed character file can crash the demo.**
  - In-place look updates call `apply_looks` on the main thread unguarded. An item look of an unexpected type raises a
    TypeError, and that ends `DemoApp.run`.
  - On first join the same data is caught and reported.
- **Medium: the worker thread still makes the video stutter.**
  - Rebuilding a behaviour is CPU-bound Python and numpy on a thread of the same process.
  - While it runs, the main thread's frames take about twice as long, with spikes of 150-240 ms (the GIL).
  - The 1.8 ms figure times only `CreatedCast.update`.
- **Low:**
  - after `LINE_UP` the swap is forced, even when that is a jump of metres in view;
  - "follows the old timing closely" holds only in some places: in `approach2` (a 90 s loop) the median difference is
    0.82 m and the 90th percentile 6.74 m. The real-avatar test reaches only `approach3`;
  - a place freed by a failed job is not offered to a waiting character until the book changes;
  - when several characters join at once, their notices overwrite each other;
  - `stand_in` ignores `fit_in`'s result;
  - `join_camera_play` reads the live selection Performer from the worker thread.

## The partial fix in the worktree (uncommitted)

The edits so far (+565/-106 lines in README.md, app.py, behaviour.py, camera_play.py, created.py, created_cast.py,
fair.py and both test files):

- **New `src/bifold_demo/pacing.py`, with `tests/test_pacing.py`.** A worker thread rests now and then on purpose, so
  the main thread keeps its frames:
  - the long computations call `pacing.pause()` between their steps;
  - in a paced thread a pause rests after each `BURST` (3 ms) of work, so the work takes only `SHARE` (a quarter) of
    the time;
  - elsewhere `pause()` does nothing.
- **The swap happens when nobody can see it:** when the newcomer stands where the old person stands (`JUMP`), or when
  neither of them is in the camera's picture. After `LINE_UP` it allows a jump of up to `POP` (2 m) in view, and after
  twice that whatever there is (`swap_cost`).
- **Failures are reported, not fatal:** looks that cannot be used leave the person as they were; items an avatar cannot
  wear are listed (`apply_looks` now returns the problems); a place freed by a failed job goes to a waiting character
  (`_again`).
- **`notice(messages)`** puts one update's messages on one line ("Ada, Ben and Cleo joined the fair").
- **`fit_in`'s result** is reported, and `planned_behaviour` moves to `fair`, so `join` no longer reads the live
  Performer.
- The docstrings now say honestly how closely rebuilt places follow the old timing.

## The fix, finished

Commit `f88536b` keeps the partial edits above and adds:

- **Pacing:**
  - pause points in the behaviour's entry loops and in `_plan`. The longest stretch of work without a pause drops from
    about 23 ms to 10-16 ms; the numpy-only stretches that remain release the GIL;
  - `Behaviour.positions` is vectorised per piece.
- **Bad looks:**
  - `save()` refuses item looks of the wrong kind before writing anything, as `load()` does;
  - a changed character's textures are made on the worker;
  - a failure on the main thread leaves the person as they were and is reported once.
- **The staged swap rule, simulated:** over 36 ready times for every place, with a grown-up of the other sex and a
  child:
  - places whose loop length matches always swapped unseen;
  - the largest jump taken was 2.0 m (the review had measured a median of 15.7 m in walker6);
  - in visitors' places an unseen moment came within 13-26 s.
- **Tests:**
  - a real-avatar test rebuilds every visitor's place;
  - tests for the swap cost, a loop of another length, a freed place, joined notices and the pacing.

Checks reported:

- 439 passed, 44 skipped without the real avatars;
- 220 passed, 1 skipped for the related modules with them;
- no new ruff findings.

What the fixer leaves as designed:

- a character whose `character.json` becomes unreadable leaves the fair instead of keeping its last good version;
- in places with a loop of another length (a child's avatar, walker6, wanderer2), 7-31 % of ready times still end in
  a jump of up to 2 m, or a pop, in view after 60 s.

## Open issues the builder admits

- **Rebuilt places** follow the old timing only roughly. Turn clips differ by sex by up to 0.6 s, which gives up to
  about 0.5 m along walks. The other players' planned arm moves are not checked again against the newcomer's slightly
  different track.
- **Selection mode:**
  - places whose person has done yoga are ranked last;
  - if a placed character's avatar changes while it is doing yoga, it becomes a new 3D person and the act is dropped.
- **The face item** refuses created characters. If the creator should allow putting a face on them by hand, that rule
  has to change.
- **Capacity:** at most 13 places (3 visitors, 7 walkers, 3 wanderers, fewer where there are children). Extra
  characters wait.
- **Saving:** the creator should save through `CharacterBook.save` or `created.save`, so the change is seen through
  `created.updated`.
- **Remaining main-thread costs:**
  - in-place look updates cost about 10 ms, mostly fitting accessories;
  - the library's first `available()` call costs 19 ms.
