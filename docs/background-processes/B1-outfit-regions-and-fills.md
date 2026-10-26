# B1: garment regions, shading-preserving fills, patterns and presets (outfits, stages 0 and 1)

**State when paused.** Implemented in two commits in the worktree `.claude/worktrees/wf_c2a91440-c22-1`:

- `3b1be67` "UV bake: rebuild a bake from its triangle map";
- `17a2137` "Outfits: garment regions, shading-preserving fills, patterns and presets".

It is reviewed. The fix was interrupted while partly done: uncommitted edits in `outfit/regions.py` (see the end of
this page). The fix starts again from those edits. The commits are not on the branch yet.

**Update: on the branch.** The fix finished (after one more interruption, by the usage limit) in commit `917ae88`
"Outfits: review fixes: labels, colour blocks, bounded caches, parser". All three commits are on the branch as
`b2adef8`, `dc46e51` and `2c75ec0`. See [the fix, finished](#the-fix-finished).

## What it builds

The new package `src/bifold_demo/characters/outfit/` holds stages 0 and 1 of [the literature
review](../research/outfit_textures.md) (section 3).

### Stage 0: garment regions (`regions.py`)

Every texel of the body texture is labelled skin, top, shirt, bottoms, shoes or other. Four cues decide:

- **Bone groups and UV islands.** In Rocketbox each UV island is one piece of cloth, so a trouser hem stays with the
  trousers even though the foot bone moves it.
- **Skin.**
  - The avatar's skin colour is sampled from the face: the most common colour, so a beard or fringe doesn't count.
  - A colour model of the skin is fitted on the hand texels, and texels close to it are skin.
- **Colour clusters**, a k-means of a lightly blurred atlas, robust to painted shading:
  - the colour that covers the middle of the back is the top;
  - a colour that only shows below it (an inner hem) is not the top;
  - large colours, or colours at the collar, are the shirt;
  - thin shapes (stripes, zips, printed numbers) are other.
- **Special cases:** a tie, small earrings and necklaces stacked in the atlas, and belts are other; boots inside the
  trouser legs are shoes.

What the labels give:

- Masks are feathered over about 3 texels and are always 0 on skin.
- The face component (A6) reads `body_regions(model).skin` and `.skin_color`.
- Mirrored and overlapping islands are flagged.
- Labels are cached per avatar: 0.1-0.2 MB, about 3 s the first time, under 1 s from the cache.
- A per-avatar JSON override file can relabel texels. Two avatars needed one, and both overrides ship with the
  package: Female_Adult_06 (a long dress over trousers) and Male_Adult_21 (a robe). The leg bones move their skirts.

### Stage 1: fills (`surface.py`, `fills.py`, `patterns.py`, `prompts.py`, `tailor.py`)

- **Cloth coordinates.** Patterns are placed by position on the body in metres, not in the atlas.
  - The torso is unrolled from the middle of the front.
  - Sleeves, trouser legs and shoes are unrolled around their joints, with seams under the arm and on the inside leg.
  - So patterns keep one size, run on across UV cuts and match front and back.
- **Fills.** A fill is a solid colour, a pattern or an image swatch:
  - colours: about 80 names, light, dark, pale, deep and bright shades of them, or hex;
  - 12 numpy patterns in two colours: stripes, pinstripe, checks, windowpane, gingham, tartan, houndstooth, polka
    dots, twill, herringbone, denim and camo. The woven ones are real weaves, so gingham and tartan get cloth-like
    mixed tones;
  - `Swatch(image, scale, rotation)` tiles an image the same way. This is where B2's generated fabric plugs in.
- **Keeping the look.**
  - For each region, the original texture is split into detail D and shading S, and the new texture is
    fill × D × S, so folds, seams and stitching stay.
  - Feeding a region's own base colour back in gives the original texture back.
- **Text to design.** A keyword parser handles:
  - single fabrics: "navy pinstripe", "red tartan", "olive camo", "white polka dots on black";
  - whole outfits: "red tartan jacket, white shirt and black trousers". "suit" dresses both top and bottoms.
- **Presets** (`presets.json`), 10 of them: business navy, charcoal suit, casual denim, sporty, trade-fair staff,
  festival, all black, highland, tweed, camouflage.
- **Applying.** `dress(person, outfit)` sets `texture_overrides[body primitive name]` and reuses the same Texture when
  dressing again. `undress()` takes it off.
- **CLI:** `bifold-characters outfit AVATAR "description" -o body.png --regions regions.png --views views.png`, and
  `--presets`. The README has a new "Outfits" section.

Measured for Business_Male_02 (1024 texture, on this 4-CPU machine):

| Step | Time |
|---|---|
| labels from scratch (bake and labelling) | 3.0-3.3 s |
| labels from the disk cache | 0.76 s |
| cloth coordinates | 0.5 s |
| shading layers | 0.22 s |
| one fill at 1024 | 47 ms (navy) to 121 ms (business navy, 4 garments) |
| first outfit on an avatar | about 50 ms more per garment |
| one fill on a 2048 texture | 186-455 ms |
| 2048 setup, once | bake about 10 s; coordinates and layers about 7 s |

## API (what B2-B4 and A6 build on)

`bifold_demo.characters.outfit` re-exports `REGIONS`, `BodyRegions`, `body_regions`, `Outfit`, `Tailor`, `dress`,
`undress`, `outfit_of`, `preset`, `presets`, `tailor_for`, `Fill`, `Solid`, `Pattern`, `Swatch`, `Atlas`, `Keep`,
`from_spec`, `color`, `parse_fill`, `parse_outfit` and `PATTERNS`.

### Dressing

- `dress(character, outfit) -> Texture`:
  - `outfit` is an Outfit, a {region: fill spec} dict, a preset name or a description;
  - it sets `character.texture_overrides[body primitive name]` and reuses that Texture on later calls;
  - `texture.outfit` holds the recipe.
- `undress(character)`; `outfit_of(character) -> Outfit | None`.

### Regions

- `body_regions(model, cache=True | folder | False, overrides=True) -> BodyRegions`.
  - It is kept in memory per model.
  - On disk it goes to `$BIFOLD_OUTFIT_CACHE`, else to `<characters folder>/outfit/<Avatar>.npz`.
- `BodyRegions` fields:
  - `labels`: uint8 (size, size), indexes into `REGIONS` = ("skin", "top", "shirt", "bottoms", "shoes", "other"), and
    `OUTSIDE` = 255 where no triangle covers;
  - `groups`, `bake` (the UVBake), `skin_lab`, `primitive`;
  - `automatic`: the labels before the overrides;
  - `flags`: {"mirrored": [island], "overlapping": [island], "overrides": [paths]}.
- `BodyRegions` methods:
  - `.skin` (a bool mask) and `.skin_color` (RGB), for the face component;
  - `.mask(region)`: feathered float32; the garments sum to 1 and it is 0 on skin;
  - `.hard(region)`, `.counts()`, `.image(atlas)` (a BGR overlay).
- **Overrides:**
  - `load_overrides(avatar, folder)` reads JSON `{"rules": [{"region": R, "where": {label | island | group | bone |
    side | facing | below | above | uv}}]}`;
  - the shipped files are in `outfit/overrides/`; a user file `<Avatar>.json` in the cache folder applies after them;
  - `apply_overrides(regions, rules, model)` applies rules directly.

### Cloth coordinates and fills

- `surface.surface(model, regions) -> Surface`:
  - `part`: int8, an index into PARTS = torso, left and right arm, left and right leg, left and right foot;
  - `uv`: float32 (size, size, 2) in metres;
  - `footprint`: metres per texel.
- `Fill.colors(where: Where) -> (N, 3)` float32 RGB 0-255, with `Where(uv, part, footprint, rows, cols, size)`.
- `Fill.spec()` is JSON-friendly, and `from_spec()` turns it back into a fill.
- The fills:
  - `Solid(color)`;
  - `Pattern(kind, primary=None, secondary=None, scale=None, rotation=0)`: scale in metres per repeat, rotation in
    degrees;
  - `Swatch(image path or RGB array, scale=0.25, rotation=0, seamless=True)`;
  - `Atlas(rgb in atlas layout)` and `Keep()`.
- `color(name | "#hex" | (r, g, b))`; `patterns.tile(kind, primary, secondary) -> (T, T, 3)`.
- `prompts.parse_fill(text) -> Fill` (a ValueError if it knows no word); `prompts.parse_outfit(text) -> {region: Fill}`.

### Compositor (`tailor.py`)

- `Outfit(fills: dict, name)`, with `Outfit.parse(text or preset)`, `.spec()` and `Outfit.from_spec(dict)`.
- `presets() -> {name: fills}`; `preset(name) -> Outfit`.
- `Tailor(model, regions=None)`, or `tailor_for(model)`, cached per model:
  - `Tailor.texture(outfit)`: RGBA uint8 at the original size;
  - `Tailor.layers`: ShadingLayers (detail, shading per region, reference, albedo);
  - `Tailor.surface`, `Tailor.original`, `Tailor.where(region)`.
- `shading_layers(rgb, labels)` builds the layers directly.

### UV bake additions

- `rasterise(prim, size) -> (triangle, overlap)`;
- `from_triangles(model, primitive, triangle, overlap) -> UVBake`. `bake()` is the two together.

## Checks the builder reports

`tests/test_outfit.py` has 38 tests: 31 synthetic, and 7 on the real avatars (skipped without them).

The synthetic tests use a person built from tubes. They cover:

- the labels;
- the island flags;
- the overrides;
- the cache round trip;
- feathered masks;
- colour names;
- seamless pattern tiles and woven tones;
- tiled sampling;
- spec round trips;
- both parsers;
- the presets;
- D×S giving the texture back;
- stripes running down the body;
- dress, re-dress and undress;
- the CLI.

On the real avatars:

- plausible labels on Business_Male_02, Female_Adult_01, Male_Adult_05 and Male_Adult_14;
- Male_Adult_21's override;
- the "charcoal suit" preset leaves the skin byte-identical and takes under 1 s;
- a GL render of a dressed avatar.

The full suite passed: 474 passed, 9 skipped. No new ruff findings.

Renders of all 20 avatars in their region colours, front and back, were looked at through about 8 rounds of fixes.

## The review's findings (for the fix)

Verdict: a solid step, with some overstated claims. The tests, timings, skin preservation and the look of the patterns
hold up. The claim that the labels are right on 18 of 20 avatars without help does not hold.

- **High: Female_Party_02.** The front of her blouse below the belt is labelled bottoms, while the back of the same
  blouse is top. Every outfit splits the blouse at the front.
- **Medium: colour-blocked garments keep their dark panels under any fill.** The shading S has one reference
  luminance per region (its 75th percentile). When a region holds several base colours, their contrast stays in S.
  - Male_Adult_17's hoodie asked to be white comes out white only on its grey panel.
  - This needs a reference per colour cluster.
- **Medium: Female_Child_01's t-shirt hems are "other".** About 6-texel lighter rims form their own thin colour
  cluster, so they stay magenta in every outfit, including the shipped `docs/previews/outfits.jpg`.
- **Medium: tie knots.** On all three tie-wearing avatars the knot is its own short island reaching into the collar.
  The tie rule misses it, so it takes the jacket's or the shirt's fill.
- **Medium: memory.** The region and tailor caches are module dicts keyed by `id(model)` that hold the model itself,
  so they keep about 165 MB per dressed avatar forever. They need a weak or LRU cache.
- **Medium: the prompt parser.**
  - Its own documented example ("a nice navy pinstripe suit, please") raises ValueError.
  - One clause it does not understand sinks the whole description.
  - "under" is not a separator, so "white shirt under a navy blazer" gives only a white top.
- **Low:**
  - a damaged label cache (a bad zip, a broken member) or a malformed override JSON crashes instead of being rebuilt;
  - the belts of Business_Male_07 and Male_Adult_12 are labelled bottoms;
  - `dress()` always starts from the library texture, so A6's body skin harmonisation and an outfit cannot both
    apply (whichever comes last wins);
  - `Atlas` and array `Swatch` specs do not round-trip, and `Swatch` assumes 0-255;
  - no test crosses a UV cut or covers the problem avatars.

## The partial fix in the worktree (uncommitted)

The edits in `src/bifold_demo/characters/outfit/regions.py` (+167/-29 lines) so far:

- the labelling `VERSION` goes from 3 to 4, so old caches are rebuilt;
- `BodyRegions.colours`: the colour cluster of each clothes texel, for shading per colour cluster;
- `_hems`: thin colours of the upper body that border only one garment and have its hue are its hems
  (Female_Child_01);
- `_knots`: a small island just above a tie, in the middle of the front and of the tie's colour, is the tie's knot;
- `_belts`: a narrow band of another colour round the top of the bottoms is a belt;
- the lower part of a top or shirt below the belt keeps that garment's label however large it is (Female_Party_02);
- any damaged cache file is rebuilt, and an override JSON syntax error is reported instead of crashing.

Not started yet: the shading per colour cluster in `tailor.py`, the weak or LRU caches, the parser fixes, the
round-trips, `dress()` on top of another body texture, and the new tests.

## The fix, finished

Every review finding is fixed. Each label fix has a real-avatar test that fails on the old labels.

- **Female_Party_02:**
  - at the waist, an island or colour of an upper garment keeps that garment's label however large it is, so the
    blouse front below the belt went from 0 % to 100 % top (the back was already 99 %);
  - a shipped override (`overrides/Female_Party_02.json`) labels the belt's loose end, which the hip bones move, as
    other.
- **Colour blocks:** `regions.colour_blocks()` groups each region's colour clusters by colour. The shading layers give
  each block its own reference brightness, so Male_Adult_17's hoodie comes out even in white, festival, red, navy
  pinstripe and tartan. Regions of one colour take exactly the old path.
- **Hems:** a thin colour that, texel by texel, looks like a neighbouring garment is split among those garments.
  Female_Child_01's sleeve and waist rims are now top, or shirt where the white undershirt shows below the hem, and the
  preview has no pink lines.
- **Tie knots:** a small island of the tie's colour just above the tie, at the front centre, is other. On the visible
  knot, "other" went from 18 / 0 / 78 % to 100 / 97 / 100 % on the three business men.
- **Belts:** a band of its own colour within 6 cm of the top of the trousers, going at least a quarter of the way
  round, is other. The share of belt texels labelled other is now 86-95 %. Buckles still count as trousers: a buckle
  rule took stonewashed denim on four avatars and was reverted.
- **Memory:** `regions.Recent` is a least-recently-used cache that holds models only weakly. It keeps 4 avatars' labels
  and 3 tailors; a tailor is about 140 MB at 1024. Dressing 7 avatars in a row used to grow to 1.3 GB; it now levels
  off at about 800 MB.
- **The parser:**
  - it splits at `, ; & +`, "with", "over", "under", "underneath", "beneath", "plus", and "and" between garments;
  - parts it does not understand, or that name details (buttons, pockets), are left out, and it fails only when no
    part parses;
  - the last, then the longest, garment word names the region ("white dress shirt" is a shirt);
  - the documented example works, also from the CLI.
- **Damaged caches** of any kind are rebuilt. A bad user override file is left out with a warning.
- **Fit for the later steps:**
  - `Tailor.texture(outfit, base=...)` and `dress()` build on the body texture a person already wears (for example a
    skin matched to photos by component 1), and `undress()` puts it back;
  - image swatches and atlases round-trip as PNG data URIs or paths;
  - `fills.image_rgb` takes paths, data URIs, grey, RGBA, 16-bit and float images.
- **Tests:** patterns are checked across a UV cut on a synthetic person; there are real-avatar tests for every problem
  avatar above.

Checks reported:

- the full suite with the real avatars: 490 passed, 9 skipped;
- the outfit and UV bake tests with the real avatars: 58 passed;
- no new ruff findings.

The labels of all 20 avatars were remade and every changed area was inspected.

Labels are now right by themselves on 17 of 20 avatars. Overrides ship for Female_Adult_06, Male_Adult_21 and
Female_Party_02 (the belt's end).

Found outside the package and left alone: `face_photo._layouts` keeps every model and its small head layout alive
forever. It should get a bounded or weak cache when the face work touches it.

## Open issues the builder admits

- **Small label misses:** the black top under a short jacket or vest is "other", not "shirt" (Female_Adult_04,
  Female_Adult_13).
- **Prints show through:** a strong print in the original fabric shows through the new fill, because the detail layer
  carries it (Female_Party_02's diamonds, Business_Male_05's check, Male_Adult_12's stonewash). A detail-strength
  control could come later.
- **Only the body is dressed.** Garments in other textures stay as they are: headscarves and caps in the head texture,
  Male_Adult_21's keffiyeh in its own primitive.
- **Patterns** stretch a little on shoulder tops and break at the back-centre and shoulder seams. The seams are
  deliberate.
- **The disk cache key** covers the mesh, the texture, the skin reference and `VERSION`, not the code: bump `VERSION`
  whenever the labelling changes. Cloth coordinates and shading layers are not cached on disk (about 0.7 s per avatar
  at 1024 on each start).
- **Possible old bug:** `face_photo.py` reads head textures with `as_texture`, which treats numpy arrays as BGR, while
  the renderer treats character texture arrays as RGB. It only matters for avatars loaded with `textures="array"`.
- **A slip, corrected:** an early CLI run wrote a cache file into the shared avatar library. It was deleted at once,
  and `$BIFOLD_OUTFIT_CACHE` was added for the tests.
