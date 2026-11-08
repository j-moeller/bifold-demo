# bifold-demo

A real-time object-detection demo for a MacBook. A video stream comes either
from the **webcam** or from a **customizable 3D scene** rendered on the GPU.
Every frame goes through a **YOLO** detector, and the video is shown on screen
at its normal resolution with the detections drawn on top.

The default scene is a **trade-fair hall** seen by a fixed camera, full of
**animated 3D people**. Some walk up to the camera, stop and turn around.
Others walk by, wander, chat in small groups, staff the booths or give a talk
on the stage, and nobody looks into the camera. The people are rigged
[Microsoft Rocketbox](https://github.com/microsoft/Microsoft-Rocketbox)
avatars (MIT licence), skinned on the GPU and driven by motion-captured
walk, turn, idle and gesture clips. The crowd mixes women and men, skin
tones and ages, including children.

```
 ┌──────────────── sources ────────────────┐     ┌── detection ──┐     ┌──── display ────┐
 │ WebcamSource ─┐                         │     │               │     │                 │
 │               ├─ SourceSwitcher (key S) ├────►│ YOLO (MPS GPU)├────►│ boxes + labels  │
 │ SceneSource ──┘   (Scene → Renderer)    │     │ on a worker   │     │ HUD, window     │
 └─────────────────────────────────────────┘     └───────────────┘     └─────────────────┘
```

## Quick start (macOS)

You need Python 3.10–3.13. An Apple-silicon Mac (M1 or newer) is recommended,
because YOLO then runs on the GPU through PyTorch's MPS backend.

```bash
git clone https://github.com/j-moeller/bifold-demo.git
cd bifold-demo
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
bifold-characters fetch --set conference   # once: download + convert the 3D people (~3.8 GB)

bifold-demo                      # the trade-fair hall (default scene)
bifold-demo --source webcam      # start with the webcam
bifold-demo --scene bifold_demo.scenes.basic    # a small scene with a few photo objects
```

`bifold-characters fetch` downloads 58 avatars and 74 animation clips from
the Rocketbox repository on GitHub. It converts them once into
`~/.cache/bifold-demo/characters`, about 0.6 GB with `--delete-raw`, which
also removes the downloaded originals. Without that step the hall shows 2D
cut-out people instead. The first run downloads the YOLO weights
(`yolo26n.pt`, about 5 MB) into `models/`. The first time you use the webcam, macOS asks for camera access
for your terminal app. Allow it, then run the command again.

### Controls

| Key       | Action                                                    |
|-----------|-----------------------------------------------------------|
| `S`       | switch the source: scene ↔ webcam                         |
| `D`       | turn detection on or off                                  |
| `N`       | face recognition on or off: names on the faces of known people (see [Face recognition](#face-recognition)) |
| `W`       | high-resolution windows on the heads of the people found, shown on the right (see [the virtual sensor](#the-virtual-high-resolution-sensor)) |
| `Space`   | pause or resume the scene's simulation time               |
| `R`       | restart the scene (time 0, physics reset)                 |
| `[` / `]` | lower / raise the confidence threshold                    |
| `Y`/`Tab` | select a person in the 3D scene: yoga poses, accessories (see below) |
| `C`       | the character creator: a character of your own for the fair (see [Character creator](#character-creator)) |
| click     | select the person you click on in the 3D scene (the nearest one where people overlap) |
| `F`       | toggle fullscreen                                         |
| `P`       | save a screenshot to `screenshots/`                       |
| `M`       | mirror the webcam on or off                               |
| `H`       | hide or show the status panel                             |
| `Q`/`Esc` | quit (in selection mode `Esc` ends that first)            |

**Selecting people for yoga and accessories.** `Y` (or `Tab`) selects the most
prominent of the 3D people in the picture and marks them with a tag; clicking on
someone selects them instead. In this
selection mode the keys are:

| Key                     | Action                                                  |
|-------------------------|---------------------------------------------------------|
| click a person          | select them                                             |
| `Tab` / `Shift`+`Tab`   | the next person to the right / left of the selected one, by where they are now; after the rightmost the leftmost (also `→` / `←`) |
| `↑` / `↓` (or `I`)      | the next menu: the yoga poses (eight to a menu), then the items; `I` goes to the items and back |
| `C`                     | the character creator for the selected person: a new character of their avatar, clothes, items and known face |
| `Esc` / `Y`             | end selection mode (poses already started go on)        |

In a yoga menu:

| Key                     | Action                                                  |
|-------------------------|---------------------------------------------------------|
| `1` … `8`               | the selected person does that yoga pose (see the table in [Yoga poses](#yoga-poses)) |
| the same number again   | keep holding the pose                                   |
| `0` / `Backspace`       | stop: out of the pose and back to what they were doing  |

In the items menu (see [Accessories](#accessories)):

| Key                     | Action                                                  |
|-------------------------|---------------------------------------------------------|
| `1` hat, `2` glasses, `3` sign, `4` t-shirt | the selected person puts it on; the same number again changes it to its next look, after the last look it comes off |
| `5` face                | the face of a known person (see [Face recognition](#face-recognition)); again: the next one, after the last their own face |
| `0` / `Backspace`       | take everything off                                     |

The bottom panel shows the keys of the current menu.

People drawn smaller than 8 % of the picture height (far away) cannot be
selected. When the selected person walks out of the picture, is hidden or is
removed from the scene, the person nearest to where they were is selected
instead. A person asked for a pose stops what they are doing (someone walking
or turning slows down to a stop over half a second), stands there while a spot
is found (a second or two), and walks (up to 4 m) from there to the nearest spot
where:

- the camera sees the whole pose, from the side it is best seen from;
- nobody nearer the camera and no counter hides it;
- the mat is clear of the other people (hands and feet too) and of the furniture;
- the whole pose is on one floor level (not half on a booth platform);
- the ways there and back are clear of the furniture and on that floor level.

When no spot within reach has all of that, they take the one the camera sees
best. There they do the pose, then walk back and carry on exactly where they left
off, starting again from a standstill. A new pose asked for during one follows
it without walking back in between; one asked for while they walk or turn
carries on from the way they are facing. Several people can do poses at once;
when they are asked one after another, their spots are found in that order, so
each mat keeps clear of the ones before. People who walk around are later on
their way afterwards, so the fair's timing that keeps walkers from meeting no
longer covers them (`R` restarts everything).

Each avatar's pose is solved the first time it is needed, in the background
(up to about 3 s), and then cached in `~/.cache/bifold-demo/characters/yoga`.
While the demo is in selection mode it also solves the poses of the people in
the picture ahead of time. `bifold-characters yoga` solves every avatar's poses
once in advance, so the demo always starts them at once. On Windows the
`Shift`+`Tab` key looks the same as `Tab` to OpenCV; use `←` there.

## Accessories

[`characters/accessories.py`](src/bifold_demo/characters/accessories.py) gives
the 3D people a hat, glasses, a sign to hold or a t-shirt, in the selection
mode above or from Python:

```python
from bifold_demo.characters.accessories import dress

dress(person, "hat", "#24406a")          # any texture value: a colour, an image or video file, an array
dress(person, "glasses", "sunglasses")   # or the name of a built-in look
dress(person, "sign", "my_poster.png")   # held up beside the head in the right hand
dress(person, "tshirt", "logo.png")      # worn over their clothes
dress(person, "hat", None)               # off again
```

Each item is fitted to the avatar it is given to:

- the hat sits over the head, hair and ears down to just above the eyes;
- the glasses sit in front of the eyes, with temples back to the ears;
- the t-shirt is the avatar's own upper body, a few millimetres off their
  clothes and moving with them, cut to short sleeves at the hips;
- the sign is held up on a stick in the right fist (the arm pose is blended
  over whatever the person does) and is put away while they do a yoga pose.

**Textures.** How each item uses its texture:

| Item | Texture |
|------|---------|
| hat | wraps once around the crown (its bottom rows at the brim, so a dark bottom stripe is a hat band); also covers the brim and the top |
| glasses | the left half colours the frame, the right half the lenses (transparent pixels are clear glass, so the built-in "sunglasses" have dark ones) |
| sign | the whole image on the board, on both sides (the back reads the right way round too) |
| t-shirt | wraps around the body: the middle of the image is the centre of the chest, its left and right edges meet in the middle of the back, its bottom row is the hem and its top row the neckline |

The looks the keys step through come from a `Wardrobe`: your files first, then
the built-in ones (hats: navy, straw, red, grey; glasses: black, tortoise,
sunglasses, gold; signs: welcome, ask me, yoga; t-shirts: white, black, red,
logo). The demo looks for your files in `items/` (`--items DIR` to use another
folder). A file is a look for an item when its name, or the name of its
subfolder, starts with the item's name: `items/hat.png`, `items/hat_red.jpg`,
`items/sign/poster.png`, `items/tshirt-logo.png` (`shirt` and `t-shirt` work
too). The folder is checked every second, and a file you replace changes
everybody wearing it at once. In Python, `Wardrobe(folder, textures={"sign":
["poster.png", "#ffffff"]})` puts your own textures first.

Items fit the Rocketbox avatars (their Biped skeleton); a person can wear all
four at once.

## Outfits

[`characters/outfit/`](src/bifold_demo/characters/outfit) gives the 3D people
new clothes on their own body mesh and texture layout: preset outfits, a colour
or a pattern per garment, and short descriptions such as "navy pinstripe" or
"white polka dots on black". The folds, seams, buttons and stitching of the
avatar's own clothes stay. This is the first step of the character creator
(component 2 in
[docs/research/outfit_textures.md](docs/research/outfit_textures.md)).
Fabrics made from a description by a text-to-image model are in the next
section; the creator screen is in [Character creator](#character-creator).

```python
from bifold_demo.characters.outfit import dress, undress

dress(person, "business navy")                                   # a preset
dress(person, "red tartan jacket, white shirt, black trousers")  # a description
dress(person, {"top": "olive camo", "shoes": "#3b2a1e"})         # garment by garment
undress(person)                                                  # their own clothes again
```

```bash
bifold-characters outfit --presets                     # the presets
bifold-characters outfit Business_Male_02 "white polka dots on black" -o body.png --views views.png
bifold-characters outfit Female_Adult_06 --regions regions.png   # the garment regions it finds
```

**Garment regions.** Each texel of an avatar's body texture is labelled
`skin`, `top` (the jacket or other outer layer; a shirt worn on its own),
`shirt` (an inner layer seen under the top: a shirt front and collar, a hood,
long sleeves under a t-shirt), `bottoms`, `shoes` or `other` (belts, ties,
trims, printed numbers: an outfit leaves them alone). The labels come from:

- the bones that move each texel and its UV island: in the Rocketbox atlases
  an island is one piece of cloth, so a trouser hem stays with the trousers
  though the foot bone moves it;
- the avatar's own skin colour, sampled from its face and learnt on its hands;
- colour clusters within the garments: the colour that covers the back is the
  top, other large colours or colours at the collar are the shirt, a different
  colour below the knee is boots;
- a few shapes: a tie (a narrow island hanging in front of the chest) and its
  knot (a small island of its colour just above it) are `other`; at the waist,
  an island or a colour of an upper garment is that garment, however large (a
  blouse worn over a skirt, whose lower part the hip bones move), and a band
  of a colour of its own round the top of the trousers is a belt (`other`);
- thin colours where two garments meet in the atlas (the blurred edge of a
  t-shirt's hem against the long sleeve under it) go to the garment each texel
  looks like, instead of making lines of `other`.

On the 20 prepared avatars the labels are right on 17 by themselves
([docs/previews/outfit_regions.jpg](docs/previews/outfit_regions.jpg)). Two
wear a long dress over trousers (`Female_Adult_06`) and a robe
(`Male_Adult_21`): the leg bones move their skirts, so they would count as
trousers. The loose end of `Female_Party_02`'s belt is an island of its own
that the hip bones move, so it would count as her skirt. The package fixes
these three with small JSON override files
(`characters/outfit/overrides/<avatar>.json`). Remaining small misses:

- the black top showing under a short jacket or vest is `other` rather than
  `shirt` (`Female_Adult_04`, `Female_Adult_13`);
- belt buckles count as the trousers, and the belt across `Female_Party_02`'s
  blouse is painted on the blouse, so it takes the blouse's fill as a dark
  line;
- stripes or zips on a jacket are `other`, so they keep their colour.

An override is a list of rules that relabel texels by label, island, bone
group, side, height or a box in the atlas, for example:

```json
{"rules": [{"region": "top", "where": {"label": "bottoms", "above": 0.36}}]}
```

Put your own as `<avatar>.json` in the outfit cache folder:
`<characters folder>/outfit/`, or `$BIFOLD_OUTFIT_CACHE` if it is set. A file
that is no valid JSON or has a bad rule is left out with a warning. The labels
are kept there too, as about 0.15 MB per avatar; a damaged cache file is made
again. The skin mask
(`body_regions(model).skin`) is meant for matching the skin to a person's
photos: outfits never write skin texels. Islands that share their texels with
another part of the body are flagged (`flags["overlapping"]`); on these
avatars they are earrings and a necklace stacked in the atlas, labelled
`other`. Mirrored islands would be flagged too; there are none.

**Keeping the look.** For each region the texture is split into a detail
layer and a shading layer. The detail layer `D = L / blur(L)` holds the weave,
stitching, seams and small wrinkles. The shading layer `S = blur(L) /
reference` holds the folds and the ambient occlusion. `L` is the luminance,
and the blur (12 texels) stays inside the region. The new texture is
`fill × D × S`. A fill of the texture's own base colour gives the texture
back. Garments cross-fade over 2-3 texels where they meet, and the new colours
reach 8 texels past the islands so that mipmapping does not bleed the old ones
in. A garment sewn from blocks of different colours, such as
`Male_Adult_17`'s hoodie with its grey body, black yoke and navy sleeves,
would keep its dark blocks dark under any fill. So each colour block has its
own reference: the luminance is first divided by its block's, and a fill
covers the whole hoodie evenly, with a thin dark seam where the blocks meet.
A strong print in the original fabric still shows through a new fill, because
the detail layer carries it (`Female_Party_02`'s diamonds, `Business_Male_05`'s
check).

**Fills.** A fill can be a colour: a name such as `navy`, `light grey` or
`burgundy`, or `#1f2a44`. It can be a procedural pattern in two colours:
stripes, pinstripe, checks, windowpane, gingham, tartan, houndstooth, polka
dots, twill, herringbone, denim or camo. Or it can be an image swatch. The
woven ones are woven: a thread grid with a weave, so gingham and tartan get
their mixed tones like real cloth. Patterns and swatches are laid out on the
body in metres, not in the texture atlas. The torso is unrolled like an
elliptic cylinder from the middle of the front, each sleeve, trouser leg and
shoe around its limb's joints, with the seams where real clothes have them:
the middle of the back, under the arm, the inside leg. So a pattern keeps one
size, runs on across UV cuts and matches front and back. Stripes run down the
body and along the sleeves. `Pattern("tartan", "red", "navy", scale=0.15,
rotation=45)` sets the repeat in metres and the angle. A `Swatch("fabric.png",
scale=0.2)` tiles any seamless image the same way, and so do the fabrics made
from a description (next section).

**Descriptions.** Keywords make the fills, and the same words always give the
same result:

- `"navy"` is a colour, and `"navy pinstripe"` a pattern with its usual
  second colour;
- `"red and black checks"` takes the first colour as the ground;
- in `"white polka dots on black"`, the colour after `on` is the ground;
- `light`, `dark`, `pale` and `deep` shade a colour;
- `thin` or `large` scale a pattern, and `horizontal` or `diagonal` turn it;
- garment words pick the region: `"red tartan jacket, white shirt and black
  trousers"`, `"blue jeans with white trainers"`, `"a white shirt under a navy
  blazer"`. The last garment word counts (a "white dress shirt" is a shirt). A
  suit dresses the top and the bottoms, and a fabric without a garment word
  dresses the top, unless another part names it. After `with` a fabric tells
  more about the garment before it (`"a white shirt with navy stripes"`);
- parts it understands nothing of (`"please"`), or that are about details it
  cannot dress (`"with gold buttons"`), are left out. It only gives up when it
  understands no part at all.

**Presets** (in `characters/outfit/presets.json`): business navy, charcoal
suit, casual denim, sporty, trade-fair staff, festival, all black, highland,
tweed and camouflage. Regions an outfit leaves out keep the avatar's own
clothes.

**Speed** (on this project's 4-CPU Linux sandbox; the textures are 1024×1024
unless prepared larger): labelling an avatar takes about 3 s the first time,
and about 0.5 s from the cache. The cloth coordinates and layers take another
0.6-0.9 s once per avatar. After that an outfit takes 60-200 ms at 1024 and
0.3-0.7 s at 2048. The first outfit on an avatar also sorts out the texels of
each garment it fills: about 50 ms a garment at 1024, 1 s in all at 2048.
Dressing a person again reuses their texture, and the renderer uploads the new
pixels.
The prepared layers of the last three avatars dressed stay in memory (about
140 MB each at 1024), so going back and forth between a few avatars costs
only the fills.

`dress()` puts the outfit on the body texture the person wears: the avatar's
own, or one that something else gave them, such as a skin matched to a
person's photos. The texels the outfit leaves alone keep that texture, and
`undress()` gives it back. `dress(person, outfit, base=texture)` names another
texture to start from.
[docs/previews/outfits.jpg](docs/previews/outfits.jpg) shows six avatars in
four presets and two descriptions.

## Fabrics from a description

[`characters/outfit/fabric.py`](src/bifold_demo/characters/outfit/fabric.py)
turns a description such as "blue paisley silk" into a seamless fabric swatch
with a text-to-image model, and puts it on a garment like any other fill: the
avatar's folds and stitching stay. This is stage 2 of the outfits in
[docs/research/outfit_textures.md](docs/research/outfit_textures.md). Without
a model it falls back to the keyword fabric of the section above.

```python
from bifold_demo.characters.outfit import FabricGenerator, dress, fabric_fill

dress(person, {"top": fabric_fill("blue paisley silk")})  # the best of 4 swatches

generator = FabricGenerator()               # FLUX.2 [klein] 4B, 4 seeds of 512 px
job = generator.start("blue paisley silk")  # returns at once; the work is on a thread
new = job.poll()                            # the swatches so far, the keyword preview first
job.cancel()                                # stops at the end of the next denoising step
dress(person, {"top": job.best.fill()})
```

```bash
bifold-characters fabric "blue paisley silk" -o swatch.png --sheet all.png
bifold-characters fabric "green linen" --avatar Business_Male_02 --region top --views views.png
bifold-characters fabric "red tartan" --model none   # no model: the keyword fabric
```

**How a swatch is made.** The description is tidied (spacing, "please make me
a", the words the prompt already says) and checked against the kiosk's word
lists. It goes into the prompt `seamless tileable fabric texture, flat lay,
top-down, even diffuse light, no folds, <description>`. A preview comes first,
at once: what the keyword parser makes of the words it knows. Then the seeds
(four by default) go down the fallback ladder until some come out seamless:

1. FLUX.2 [klein] 4B with latent rolling, at its 4 steps;
2. the same at 12 steps;
3. SD-Turbo with circular padding;
4. the model-free repair of the best image the models made;
5. the keyword fabric (`parse_fill`);
6. a plain colour: the colour word nearest to one in the text, or grey.

Each step down says why: a seam on every seed, or a model that failed (out of
memory, an error; it is then not tried at more steps either). Missing
packages, weights that are not downloaded or no network go straight to the
keyword fabric, with the command that installs or fetches what is missing.
`--model z-image-turbo` puts Z-Image-Turbo in klein's place; `--model
sd-turbo` starts at SD-Turbo.

**Making a model paint a repeat.** FLUX.2 [klein] and Z-Image are diffusion
transformers: they see the picture as patches at fixed positions, with no
padding to change. So after every denoising step, the pipeline's
`callback_on_step_end` rolls the latents round the image (what leaves the
right edge comes back in at the left), and the next step paints across the old
edge. The roll is random, up to half the image at first and shrinking to none
for the last step, and the last callback rolls the latents back to where they
began. FLUX.2 keeps its latents as a packed sequence of tokens, one per 16 x 16
pixels, row by row: they are unpacked to the grid, rolled by whole tokens and
packed again. Z-Image keeps the latent grid, rolled by whole 2 x 2 patches.
The VAE decoder pads circularly, so the decoded edges match. SD-Turbo is a
convolutional UNet: circular padding of its UNet and VAE makes it paint on a
torus. Both patches last for one call only and are undone afterwards, also
after an error or a cancel, so the same loaded pipeline still makes other
pictures; the pipeline's lock is held meanwhile.

**Checking and repairing seams.** Every image gets a seam score: how much its
wrap (right edge against left, bottom against top) stands out from the inside
of the image, as a step, as a line along the edge, or as a broken rhythm of a
regular pattern, also of wide stripes and checks whose repeat does not fit
the swatch. About 1 means it tiles; above 1.25 a seam shows and the seed is
left out (the measurements are in `characters/outfit/seamless.py`; a faint jog
of a fine twill can pass). When no seed of any model tiles, the best image is
repaired without a model: it is rolled by half so its seams cross in the
middle, and the strips either side of each seam are overlaid and cut where
they agree best (image quilting's minimum-error cut). Each side keeps its own
pixels up to the cut, so motifs are cut, not ghosted, and an offset evens a
slow difference across it. On synthetic textures that do not tile, the repair
brings 45 of 48 smooth ones and 23 of 24 prints below 1.25; an image that
tiles already comes back as it is. `make_seamless(image)` repairs any image
and `seam_score(image)` scores one. Kept swatches get even light, since the
tailor adds the garment's own shading, but only light: a slow change of
brightness that is clearly stronger than the fabric's own and leaves the
colours alone. An evenly lit print stays exactly as it is; a pattern whose own
repeat is as large as the swatch keeps whatever light it has. On this
project's 4-CPU sandbox the score takes 30-60 ms, the light 15-30 ms and the
repair 0.2-0.3 s at 512 px (about 0.5 s at 768).
[docs/previews/fabric.jpg](docs/previews/fabric.jpg) shows the repair, the
ladder's keyword rungs and three avatars in repaired synthetic swatches; no
model could run where it was made.

**Size on the body.** A swatch's repeat in metres depends on the kind of
fabric its words name: a material alone (silk, denim, wool) is a close-up of
its weave, 8 cm; small prints (dots, gingham, pinstripes) 10 cm; prints
(paisley, florals, checks, plaids) 20 cm; large prints (camouflage, tropical,
animal prints) 35 cm. Size words (`tiny`, `large`, ...) scale them, and
`scale=` (or `--scale`) sets it.

**The kiosk filter.** A public fair gets every kind of request. Brand names,
characters and logos ("Gucci", "Hello Kitty", "with our logo") and offensive
words are refused with a message to show, whatever their case, accents,
spacing ("g u c c i"), digits or signs standing in for letters ("gucc1",
"sh!t") or stars for letters left out ("f*ck"). Only whole words count, so
"cocktail" and "Sussex" are fine. Words with a fabric sense as well are let
through only in it: "nude satin" and "naked wool" are colours and cloth, "blue
tits" birds, but not next to words for people, bodies or photos. The lists are
small and live in `characters/outfit/fabric_words.json`. A file of the same
form named by `$BIFOLD_FABRIC_WORDS` adds words or lists, and its `"allow"`
list lets words through.

**For the creator screen.** `FabricGenerator.start()` returns a `FabricJob` at
once. The model loads on the job's worker thread the first time, after the
preview (that can take a minute). `job.poll()` gives the new swatches without
waiting, `job.progress` the step of the image being made and `job.messages`
what happened; `on_result` and `on_message` callbacks run on the worker. Torch
lets go of the GIL in its operations and the Python work between them is
short (the repair's search for its cut, the longest, runs on pairs of rows).
Measured here with a main thread that ticks every 5 ms: while a job ran, 99
ticks of 100 were at most 0.8 ms late; while a worker repaired swatches
nonstop, at most about 1 ms, the worst 3-9 ms (5-7 ms with the worker idle).
With other processes keeping all four CPUs busy, those were 3.4-4.1 ms and
8-20 ms, about what the load gives alone. `pace=` takes a callable that is
called after every denoising step and between images, to slow the worker down
further.

**Cache and saved outfits.** The images the models made, kept or left out, are
cached as they came in `fabrics/` (git-ignored; `$BIFOLD_FABRIC_CACHE` moves
it) by prompt, model, seed, size and method, and scored and finished again
when read, so the same description comes back at once without loading the
model, judged by the generator that reads it. A cache that cannot be written
costs a message, never a swatch. A swatch becomes a `Fabric` fill
(`swatch.fill()`). Its spec holds the description, the prompt, the model, the
seed, the method and the image itself (lossless WebP: in a saved character's
JSON a 512 px swatch takes a few KB when flat, 100-400 KB when smooth or
grainy, up to about 850 KB when photo-like), so an outfit
recipe, also one saved with a created character, rebuilds it without the
model. A spec without an image, `{"fabric": "red tartan"}`, is the keyword
fabric of its words.

**On the Mac.** Install the image packages and download the weights before
the fair, so that the kiosk never starts a download:

```bash
pip install -e ".[people]"
hf download black-forest-labs/FLUX.2-klein-4B   # about 15 GB of weights, Apache-2.0
hf download stabilityai/sd-turbo                # the fallback rung, a few GB
bifold-characters fabric "blue paisley silk" --offline --sheet all.png   # only what is downloaded
```

If a model's page says it is gated, accept its terms there and run `hf auth
login` first. `--device auto` picks Apple's GPU (MPS), and FLUX.2 [klein] runs
in bfloat16. It needs about 16 GB, so a 24 GB+ Mac is recommended; SD-Turbo,
loaded only if the ladder reaches it, needs about 4 GB more (a model that runs
out of memory is let go of before it loads). Models stay loaded until
`FabricGenerator.release()`. On a 16 GB Mac start at `--model sd-turbo`. The review's estimates for
an M-series Pro, not measured (no model ran in this project's sandbox): 5-15 s
for a 512-768 px swatch at 4 steps, 20-60 s for four variations shown as each
finishes, and about three times as long per swatch on the 12-step rung. The
default 512 px is the quickest. `--offline` (or `FabricGenerator(download=False)`)
never downloads; a model that is not there goes straight to the keyword
fabric.

## Funny faces

[`characters/faces.py`](src/bifold_demo/characters/faces.py) makes the 3D people
pull faces at the camera with the 28 face bones in every Rocketbox head (jaw,
lips, mouth corners, cheeks, eyelids, eyebrows, nose, tongue, eyeballs):

| Name | Face |
|------|------|
| `smile` | a broad closed-mouth smile, the corners up and out, the cheeks raised |
| `grin` | a wide toothy grin, the lips drawn apart over both rows of teeth |
| `tongue` | the jaw open, the tongue far out and down over the lower lip |
| `puff` | blown-up cheeks, the lips pressed together |
| `surprise` | the jaw dropped into a round O, the eyebrows high, the eyes wide |
| `scrunch` | the nose wrinkled, the eyes squeezed shut, the brows down, the upper lip up |
| `fish` | the lips pursed far forward, the cheeks sucked in |
| `wink` | the left eye shut, that mouth corner and cheek up |
| `sideways` | the mouth pulled over to the left, the jaw with it |
| `cross_eyed` | both eyes turned in to the nose, the mouth hanging open |
| `roar` | a monster roar: the jaw wide open, the upper lip up, the brows down and in |
| `pout` | a sulky pout: the lower lip out and up, the corners down |

```python
from bifold_demo.characters.faces import apply_face, face_rig

rig = face_rig(model)   # once per model, before it is first drawn; None without face bones
person.pose_adjusters.append(lambda local, character, t: apply_face(local, rig, {"tongue": 1.0, "smile": 0.3}))
```

Weights run from 0 (no change) to 1 and mix: two faces at 0.5 are halfway
between them. The faces are deliberately big, because from the fair camera a
face is only about 50 pixels tall. They are slides and turns of the face bones
added to whatever the clips and the head turn do, scaled to the size of the head
(children make smaller faces). The tongue only comes out once the jaw is open
and goes over the lower teeth on the way. Two quirks of the meshes are handled:
gum vertices that two heads skin to the tongue follow the jaw instead (mended
when the avatar is loaded), and cheeks under a headscarf puff forward rather
than through it.

## The trade-fair scene

[`scenes/fair.py`](src/bifold_demo/scenes/fair.py) builds a 44 × 72 m exhibition
hall, seen by a fixed camera at eye height (`CAMERA_POSITION`, `CAMERA_TARGET`).
It contains:

- **The hall**: floor tiles, carpeted aisles, walls with high windows,
  pillars, roof trusses, ceiling lights and a light haze (fog) for depth.
- **Booths**: 8 booths for fictional exhibitors. Each has a printed back wall,
  a counter, a screen cycling through slides, a slowly spinning hanging sign
  and a plant.
- **Stage**: a keynote stage at the back, with an LED wall, trusses and a
  welcome banner.
- **36 animated people** (see [3D people](#3d-people)):

| Who | What they do | Configured by |
|-----|--------------|---------------|
| 3 in the audience | stand in a half circle 3.9 m in front of the camera (our exhibit), facing it and watching it, always there; now and then they [play with the camera](#playing-with-the-camera) | `AUDIENCE` |
| 3 visitors | walk down the aisle towards the camera, join the audience, watch (and play) for 15–30 s, turn around, walk back | `AUDIENCE`, `VISITORS_FROM` |
| 7 walkers | walk across the hall along the cross aisles, both directions; on the near aisle they come and go at the side walls, out of view, on the far aisles they turn round near the walls | `CROSSINGS` |
| 3 wanderers | stroll between random points in an area, pausing | `WANDER_AREAS` |
| 3 groups (7 people) | stand and chat, alternating talking and listening, with small shifts | `CHAT_GROUPS` |
| 8 exhibitors | stand on their booth platforms behind the counters | `booth_layout()` |
| 4 visitors | talk to exhibitors at the left-hand booths | |
| 1 speaker | talks on the stage | |

Characters made in the character creator take over some of the visitors',
walkers' and wanderers' places (see [Created characters](#created-characters)).

The camera is our own exhibit, so at any moment at least three people watch it:
the audience always, and the visitors while they stand with them. Everyone
else never looks at the camera: the behaviours guarantee that their faces
point at least 35° away from it. While someone walks towards the camera or
stands close by, the head stays turned aside or lowered. In
[selection mode](#controls) the walks to a yoga pose and back keep to this
too once under way, but the rest of a pose does not: those seen best from the
front (Yogi Squat, Warrior 2, Goddess, Tree, Triangle) face the camera on
purpose, and the walk back after them turns the face away over its first
third of a second; while someone turns into a pose, or back into what they
were doing, the head turns with the body.
The crowd is sampled from the prepared avatars with conference demographics:
about 5% children, 80% adults and 15% older people (as far as the avatars
allow, see below), half women and half men, and every skin-tone band
represented. Children join the chat groups first, otherwise they wander, and
never walk alone in the aisles. Exhibitors and the speaker get business
clothes when possible, and nobody talks to their own double.

Nobody walks through anyone else, however long the demo runs. Every moving
person's loop is stretched to a whole fraction of `LOOP_PERIOD` (6 minutes),
so the whole hall repeats every 6 minutes. `keep_apart` then shifts their
timings until, over that period, nobody comes within `MIN_DISTANCE` (0.6 m)
of anyone else.

`CROWD_SEED` gives a different crowd and different timings, and `PEOPLE_3D =
False` brings back the 2D cut-outs. If setting up the 3D people fails, for
example because of a damaged file in the character cache, the hall falls back
to the 2D cut-outs and the terminal says why. Save the file while the demo runs and the
scene reloads. The terminal prints the crowd's make-up, for example
`36 animated people: 18 female, 18 male; 1 child, 33 adult, 2 older; skin band 21 1-3, 11 4-6, 4 7-10`.

**Without prepared 3D people**, the hall shows 2D cut-outs instead: from the
`people/` folder (see [Generating 2D people](#generating-2d-people)) or, until
then, people cut out of the Ultralytics sample photos.

**Your own graphics.** Every part has a name, so you can put your own images
on it through the uploads folder, e.g. `uploads/booth1_wall.png` for a booth's
back wall. The names are `booth1_wall` … `booth8_wall`, `booth1_counter`,
`booth1_sign`, `booth1_screen`, `stage_screen`, `banner` and `floor`.

## Playing with the camera

The demo screen shows what the fixed camera sees, so the people in front of it
see themselves like in a mirror, and now and then they play with their picture
([`characters/camera_play.py`](src/bifold_demo/characters/camera_play.py)):

| Move | What they do |
|------|--------------|
| `wave` | wave at the camera with one hand (mostly the right, the left when the right one holds a sign) or both, with a light smile |
| `faces` | make one to three faces in a row, tilting and wobbling the head, sometimes with their hands: thumbs at the temples wiggling the fingers, index fingers pulling the mouth wide, palms squashing the cheeks |
| `side_steps` | step to one side, check the picture, step over to the other side and back to the middle |
| `lean_sideways` | lean to one side and the other, the hips shifting the other way, the feet staying put |
| `lean_forward` | bend forward towards the camera, the hips going back, peering up at it |
| `closer` | walk up two or three steps, peer at the camera (perhaps making a face), walk backwards to the spot |

Only the audience and the visitors play, only while they watch the camera,
and a move only starts when all of it fits in before they turn away
(visitors do one to three moves per visit). Most of the time they just watch:
about a third of the time on average, children and some adults more often
and with bigger moves, shy people seldom and small. The moves are planned for
everyone together over the hall's 6-minute loop, so nobody comes within
0.55 m of anyone else because of a move (people walking by included), and no
waving arm, elbow or leaning head reaches into anyone: such a move goes the
other way, less far, waves with one hand, leaves the hands out of a face, or
becomes another move. Going closer stops where the feet would leave the
picture. Like everything
else they are a pure function of time, so pausing, rewinding and `R` just
work. A foot only moves while it steps, one at a time, and the face keeps on
the camera through every move but the deliberate head tilts of making faces.
The expressions themselves come from
[`characters/faces.py`](src/bifold_demo/characters/faces.py) (the Rocketbox
face bones). While someone does a yoga pose in selection mode, they stop
playing; the tag of the selected person says what they do, for example
"waves at the camera".

The waves use the Rocketbox `wave_01` and `wave_02` clips (for men and women),
which `bifold-characters fetch` downloads with the other animations (also
with `--clips basic`); a cache prepared before they were part of it gets them
with the same command (it only downloads what is missing), and until then
people wave procedurally. `CAMERA_PLAY = False` in `scenes/fair.py` switches
the playing off, and `CAMERA_PLAY_MOVES` lists the moves people choose from.

```python
from bifold_demo.characters.camera_play import make_move, play_along

# people (Characters) whose behaviours watch the camera; the others' positions are kept clear of
plays = play_along(audience, camera=(0, 1.6, 6), others=everyone_else, period=360.0, rng=5)
plays[0].moves                           # Move("wave", 12.3-17.1 s), ...: kind, start, channel table
move = make_move("closer", model, rng=1, playful=1.3)   # one move, e.g. to look at its channels
```

`play_along` plans the moves (`plan_camera_play`), wraps each behaviour in a
`CameraPlay` (it moves the person and points the head at the camera from where
the eyes are) and gives the character a `PlayPose` pose adjuster (hips, torso,
leg IK, the arms, the face). It adds well under a millisecond per person and
frame while someone plays, and next to nothing otherwise.

## 3D people

The [`characters`](src/bifold_demo/characters) package turns Rocketbox avatars
into animated people. It has five parts:

1. **Assets** (`rocketbox.py`, `library.py`, `bifold-characters`).
   `bifold-characters fetch` downloads exactly the files it needs from GitHub:
   per avatar its FBX file, its colour/opacity textures and its head's normal
   and specular maps, plus the animation clips. It then converts them with a
   pure-Python FBX reader (`fbx.py`, `fbx_convert.py`), so no Blender or FBX
   SDK is needed. The result is a fast cache: an avatar loads in a few
   milliseconds. Heads are kept at 2048 pixels as well (see "High-resolution
   heads").
2. **Model and posing** (`model.py`). A skeleton (81 Biped bones), skinned
   meshes and animation clips. Clips drive the bone rotations and the avatar
   keeps its own proportions, so one clip fits adults and children.
3. **Rendering** (`character.py`). A `Character` is a scene object. It poses
   its skeleton every frame and is skinned on the GPU (4 bone weights per
   vertex). It adds head turns on top of the clip and a soft contact shadow.
4. **Behaviours** (`behaviour.py`). `approach`, `walk_by`, `wander`, `chat`,
   `stand` and `watch` build looping timelines of walking, turning (with turn
   clips) and idling (idle, look-around, talk, listen, phone clips), with
   crossfades.
   - Walk clips play at the speed people move, so feet never slide, and
     walks start and stop in the stride that best matches the standing pose.
   - The head plan keeps the face at least 35° away from the camera, except
     for people who watch something (`watch`, or `approach(..., watch=...)`):
     their head turns to the point and follows it.
   - `period=` stretches the pauses so that the loop lasts a whole fraction
     of it, so people built with the same period repeat together.
   - A behaviour is a pure function of time, so pausing (`Space`) and
     restarting (`R`) just work.
5. **Crowds** (`crowd.py`). `sample_crowd` picks avatars to match target
   shares of age group, sex and skin-tone band, using the labels in
   `rocketbox_catalog.json`. `populate` turns avatar names and behaviours into
   `Character` objects.

```bash
bifold-characters fetch --set small         # 12 diverse avatars, a quick start (~0.8 GB)
bifold-characters fetch --set conference    # 58 avatars that suit a fair, incl. children (~3.8 GB)
bifold-characters fetch --set all           # all 115, incl. uniforms (~8 GB)
bifold-characters fetch --avatars Business_Female_01 Male_Adult_04 --clips basic
bifold-characters list                      # what is prepared: sex, age group, skin tone, height
bifold-characters info Business_Female_01
bifold-characters yoga                      # solve every avatar's yoga poses ahead of time (a few minutes)
```

Re-running `fetch` skips what is already there. `--dry-run` shows the
download size first, and `--delete-raw` removes the originals after
converting. To keep the cache elsewhere, `export BIFOLD_CHARACTERS=DIR` before
running both `bifold-characters` and `bifold-demo`: the demo reads only that
variable (default `~/.cache/bifold-demo/characters`), and `--dest DIR` changes
only the folder `bifold-characters` works in. `--raw DIR` uses a Rocketbox
clone you already have, or its `Assets` folder: `prepare --raw DIR` converts
it, and `fetch --raw DIR` downloads only the files it lacks.

Using 3D people in your own scene:

```python
from functools import partial
from bifold_demo.characters.behaviour import approach, walk_by
from bifold_demo.characters.crowd import populate, sample_crowd
from bifold_demo.characters.library import CharacterLibrary

camera = (0, 1.6, 6)
library = CharacterLibrary.shared()                                       # one library, kept across scene reloads
names = sample_crowd(library.available(), 2, rng=1)
people = populate(library, names, [
    partial(approach, (0, 0, -15), camera=camera),                        # comes closer, turns around
    partial(walk_by, [(-8, 0, -5), (8, 0, -5)], camera=camera),           # walks by and back
])
for person in people:
    scene.add(person)
```

**Limits of the avatar library.** Rocketbox has 115 avatars but few clearly
elderly people: the oldest look about 58, and those count as older. It also
has few conference-suitable people in the darkest skin-tone band, since most
of those wear uniforms. The crowd therefore reaches its share of older people
only partly. Change the targets with `sample_crowd(..., weights={...},
older_from_age=...)`. Children are rated unusual at a trade fair in the
catalog, but the conference set includes them anyway.

## Yoga poses

For a later demo that recognises yoga poses, [`characters/yoga.py`](src/bifold_demo/characters/yoga.py)
animates the 3D people through yoga poses. Each pose is a looping clip: from a
starting pose into the pose, holding it for two breaths, and back. In the demo,
select a person with `Y` and press the pose's key (see [Controls](#controls)).
The yoga menus hold eight poses each: `↓ 1` means `↓` once (to the second yoga
menu), then `1`.

| Key | Pose | `yoga_clip(model, ...)` | Starts from |
|-----|------|-------------------------|-------------|
| `1` | Yogi Squat (Malasana) | `"yogi_squat"` | standing |
| `2` | Cow Pose (Bitilasana) | `"cow"` | standing, down to hands and knees (crouch, kneel with toes tucked, tabletop) |
| `3` | Warrior 1 (Virabhadrasana I) | `"warrior_1"` | standing, the right foot steps back, the arms sweep up |
| `4` | Downward Facing Dog (Adho Mukha Svanasana) | `"downward_dog"` | standing, down to hands and knees, toes tucked, the hips lift |
| `5` | Warrior 2 (Virabhadrasana II) | `"warrior_2"` | standing, the right foot steps out wide, the arms open |
| `6` | Upward Facing Dog (Urdhva Mukha Svanasana) | `"upward_dog"` | standing, down to hands and knees, a step back to plank, the hips lower |
| `7` | Chair Pose (Utkatasana) | `"chair"` | standing, the arms sweep out and up overhead, the knees bend and the hips sit back |
| `8` | Goddess Pose (Utkata Konasana) | `"goddess"` | standing, the feet step out wide one at a time, the arms open, the knees bend into cactus arms |
| `↓ 1` | Tree Pose (Vrksasana) | `"tree"` | standing, the weight shifts onto the left foot, the right foot comes to the inner calf and slides up to the thigh, the hands rise from prayer at the chest to overhead |
| `↓ 2` | Warrior 3 (Virabhadrasana III) | `"warrior_3"` | standing, the weight shifts onto the left foot, the body tips forward as the right leg lifts back and the arms swing forward |
| `↓ 3` | Triangle Pose (Utthita Trikonasana) | `"triangle"` | standing, the feet step wide and the arms open, the front foot turns out, the torso reaches over the front leg and tilts down, the lower hand to the shin, the top arm up |
| `↓ 4` | Standing Forward Bend (Uttanasana) | `"forward_bend"` | standing, the arms open wide, a flat-back hinge with the arms out (swan dive), the fold down over straight legs to the hands beside the feet |
| `↓ 5` | Plank Pose (Phalakasana) | `"plank"` | standing, down to hands and knees, toes tucked, the right foot then the left steps back, the hips lower into the plank |
| `↓ 6` | Cat Pose (Marjaryasana) | `"cat"` | standing, down to hands and knees (crouch, kneel with toes tucked, tabletop), the back rounds up and the head drops |
| `↓ 7` | Child's Pose (Balasana) | `"child"` | standing, down to hands and knees, the hips sink back towards the heels, the torso folds over the thighs, the forehead to the floor, the arms stretched forward |
| `↓ 8` | Cobra Pose (Bhujangasana) | `"cobra"` | standing, down to hands and knees, a step back to plank, down to the floor through a low plank, the chest lifts on bent arms with the hips and legs down |

```python
from bifold_demo.characters.character import Character
from bifold_demo.characters.library import CharacterLibrary
from bifold_demo.characters.model import CharacterState, Layer
from bifold_demo.characters.yoga import yoga_clip

model = CharacterLibrary.shared().load("Female_Adult_01")
clip = yoga_clip(model, "yogi_squat")      # ~15 s; clip.metadata["hold"] = when the pose is held
person = Character(model, behaviour=lambda t: CharacterState((0, 0, 0), 0.0, [Layer(clip, t)]))
```

The poses are not motion capture. They are described in body terms: where the
hips are, how the pelvis, spine and head turn, where each hand and foot goes and
which way the elbows and knees point. A solver turns that into bone rotations
for each avatar's own skeleton, so the same pose fits a child and a tall man,
and the hands, feet and knees that rest on the floor touch it exactly (checked
on the skinned mesh). Every clip starts and ends standing, so they can follow
one another. Long skirts, dresses and robes are skinned to the legs: they pass
through the floor when the person kneels.

## Face recognition

[`recognition/`](src/bifold_demo/recognition) tells who the people in the
picture are, in five steps, with the models of a *profile* (below; `best` by
default):

1. **Find the faces.** A face detector (SCRFD-10G; YuNet in the commercial
   profile) finds each face and five landmarks on it (the eyes, the nose tip
   and the mouth corners).
2. **Line them up.** Each face is turned upright and scaled by its landmarks
   onto a 112 x 112 template.
3. **Fingerprint them.** A recogniser (AdaFace IR-101 in the best profile)
   turns the face into an *embedding*: 512 numbers (128 for SFace), trained so
   that photos of one person give nearly the same numbers and other people's
   faces different ones. Two faces are compared by the cosine similarity of
   their embeddings.
4. **Judge the quality.** Each face gets a quality record: how many pixels lie
   between the eyes, which way the head is turned, how sharp and well lit the
   face is, and a learnt quality score.
5. **Compare with the known people.** A face is the known person with the
   highest score, if it reaches the model's threshold (for now one per model,
   at about one false match in a thousand pairs of strangers on the LFW
   benchmark; calibrated decisions per face size come next), else unknown.

In the demo `N` (or `--recognize`) turns it on: every face in the picture gets
a box with the name of the person and the similarity under it, or `?` for a
stranger, and the status panel lists who is in the picture. It runs on its own
thread beside the object detection. A face only 30 pixels tall gives a noisy
fingerprint, so faces are followed from frame to frame and named from the
average of their last eight fingerprints, and a named face keeps its name
while that average stays near the threshold (no flickering names).
`--face-profile` (or `BIFOLD_FACE_PROFILE`) chooses the models, for everything
in the run that looks at faces: the recognition, the character creator's face
painting and the head textures' identity check.

The known people are the photos in [`faces/`](faces/README.md) (`--faces DIR`
for another folder), a folder per person with as many photos as you like
(several photos taken on different days and from several sides make it surer).
Photos added while the demo runs are known a second later. Each photo is
checked as it comes in (one clearly biggest face, at least 40 pixels between
the eyes, sharp; a warning for small, soft or badly lit faces and for people
the model finds alike); nearly identical photos count once:

```bash
bifold-faces add "Ada Lovelace" portrait.jpg holiday.png   # copies them to faces/Ada Lovelace/
bifold-faces list      # the models, then who is known: photos per pose, rejected photos and why, look-alikes
bifold-faces who group_photo.jpg                           # who is in a photo
```

The models are downloaded into `models/` the first time (or with
`bifold-faces fetch`) and run by ONNX Runtime (SFace and YuNet by OpenCV). The
embeddings of the photos are kept per model in
`faces/.embeddings.<model>.npz`, so only new photos are looked at again. On
four CPU cores AdaFace takes about half a second per face, the licensed
profile's ResNet-50 a tenth and SFace 20-30 ms; on a Mac, CoreML should be much
faster (not measured yet). Where the CPU is all there is, the licensed or
commercial profile keeps the live view responsive.

```python
from bifold_demo.recognition import FaceEngine, Gallery

engine = FaceEngine(profile="best")       # finds, lines up, fingerprints and judges faces
gallery = Gallery("faces", engine)        # the known people
gallery.refresh()
for record in engine.analyse(image):      # every face: box, landmarks, aligned crop, embedding, quality
    match = gallery.match(record.vector)  # match.name (None: unknown), match.score, match.runner_up
    print(match.name, record.quality.ied, record.quality.pose)
```

### The recognition core: models, profiles, licences

Face recognition is becoming the core of the demo
([plan](docs/research/face_recognition_focus.md),
[review](docs/research/face_recognition_sota.md)), with stronger models
behind one switch. [`recognition/registry.py`](src/bifold_demo/recognition/registry.py)
lists every face model with its file (download address, size and SHA-256),
its input (size, colour order, normalisation), its output, its licence, what
it was trained on and where the file comes from. Three **profiles** put them
together (`--face-profile`, or `BIFOLD_FACE_PROFILE`; default `best`):

| Profile | Face detector | Recogniser (fallback) | Quality | Licences of the weights |
|---|---|---|---|---|
| `best` | SCRFD-10G | AdaFace IR-101, trained on WebFace12M (`buffalo_l` ResNet-50) | eDifFIQA-T and geometry | research use only |
| `licensed` | SCRFD-10G | InsightFace `buffalo_l` ResNet-50 | eDifFIQA-T and geometry | research use only, unless a commercial licence is bought from InsightFace |
| `commercial` | YuNet | AuraFace-v1 (SFace) | geometry | MIT and Apache-2.0 |

AdaFace IR-101 is the most accurate recogniser on small and poor faces whose
weights can be run here (TinyFace rank-1 72 %, IJB-C 97.7 % at 1e-4); its ONNX
file is a community export, so it is checked before it is trusted (its
SHA-256 is pinned, its graph has only standard operators, and its accuracy is
measured against the published one). AuraFace is on Hugging Face only and is
fetched where Hugging Face can be reached; elsewhere the commercial profile
uses SFace. "Commercial" is about the weights' licences only: the detectors
were trained on WIDER FACE and most recognisers on web-scraped faces whose
terms are for research, so anything that ships needs a legal review first.

```bash
bifold-faces fetch --profile best      # SCRFD-10G, AdaFace, buffalo_l (fallback), eDifFIQA-T: 560 MB, 460 MB kept
bifold-faces fetch --model sface_2021dec yunet_2023mar
bifold-faces list                      # the models (licences, profiles, fetched or not), then the known people
```

InsightFace publishes its models as zip packs (`buffalo_l.zip`,
`antelopev2.zip`); the download keeps only the members a profile needs and
deletes the pack. Every file is checked against its pinned SHA-256 before it
replaces anything, and a file that cannot be had from one source is tried from
the next (InsightFace's files are also mirrored, byte for byte, by the AdaFace
export and UniFace; SCRFD comes from its 17 MB copy first, so finding faces
does not cost the 289 MB pack). The models run with ONNX Runtime: on a Mac
with CoreML first (falling back to the CPU), elsewhere on the CPU
(`BIFOLD_FACE_PROVIDER` and `BIFOLD_FACE_THREADS` override). ONNX Runtime 1.29 and later send usage
telemetry to Microsoft by default; the recognition switches it off
(`ORT_DISABLE_TELEMETRY=1`) before ONNX Runtime loads.

**Detection and alignment**
([`detect.py`](src/bifold_demo/recognition/detect.py),
[`align.py`](src/bifold_demo/recognition/align.py)). Every detector gives a
box, five landmarks in one order (the eye on the image's left first, then the
other eye, the nose tip and the mouth corners left to right) and a score.
SCRFD's anchor decoding and suppression of overlapping boxes are written from
InsightFace's reference code and tested against hand-made network outputs;
SCRFD and YuNet put the landmarks within a fifth of the eye distance of each
other on every test photo. SCRFD looks at a *canvas*: 640 x 640 for photos,
the frame's own size for small faces in a camera picture, 160-192 pixels for a
head cut out of a frame (measured here on four shared cores: 12 ms at 160, 35
ms at 320, 120 ms at 640, 300-390 ms at 1280 x 736; YuNet 44 ms for a
1280 x 720 frame). Faces are lined up on the ArcFace five-point template at
112 x 112 by a least-squares similarity transform, exactly as InsightFace's
`norm_crop` and OpenCV's `alignCrop` do (the same pixels as `alignCrop`); the
crop is black where it reaches beyond the picture (as in training), the share
of it that does is a quality signal, and faces much bigger than the crop are
shrunk with area averaging first so that they do not alias.

**Embedders and quality**
([`embed.py`](src/bifold_demo/recognition/embed.py),
[`quality.py`](src/bifold_demo/recognition/quality.py)). Every embedder takes
a batch of aligned crops and gives unit embeddings and their raw lengths, in
each model's own colour order and normalisation, in batches of 8 and then
single crops. The AdaFace export normalises its embedding inside its graph,
which would lose AdaFace's quality signal (the embedding's length, which its
loss ties to how recognisable a face is); the length is read out of the graph
as an extra output instead (on the avatar renders it did not follow pose or
face size, so it is recorded but not yet used as a weight). Each face gets one
quality record: the eye distance; yaw, pitch and roll from a 3D fit of the
five landmarks (calibrated on 1620 avatar renders at known angles: with
SCRFD's landmarks the yaw comes back within 2° with a spread of 2-4°, the pose
bin — frontal, left, right, profile — in 98.6 % of the renders; YuNet's
landmarks give 84 %); sharpness, brightness, contrast and clipping on the
aligned face; the share of the crop beyond the picture; eDifFIQA-T's score
(which ranks renders by how recognisable they are with a Spearman correlation
of 0.87 and falls steadily as faces get smaller); and AdaFace's norm.

**The engine and the gallery**
([`engine.py`](src/bifold_demo/recognition/engine.py),
[`gallery.py`](src/bifold_demo/recognition/gallery.py)). `FaceEngine` puts a
profile's models together. Only the detector is loaded at once; the recogniser
and the quality model are found when first needed (downloaded if allowed, their
SHA-256 checked once per process, loaded), so code that only finds faces needs
neither, offline too. When AdaFace is missing, damaged or does not load, the
best profile takes the ResNet-50 and says why; when no model of a profile can
be had, the error points to the commercial profile (YuNet and SFace).
`analyse` embeds all the faces of an image in one batch and gives a record per
face. The gallery embeds each photo with
exactly the engine's detector, alignment and model, checks it (above), and
keeps, per person, every photo and a template per pose bin (the
quality-weighted mean of the bin's embeddings); a person's score is the best of
the two. The decision whom a face shows is a replaceable step
(`gallery.decide(scores, quality)`), so that calibrated open-set decisions per
model and face size can take its place. Embeddings of different models cannot
be compared, so the caches are per model, and a cache made by another ONNX
Runtime provider (CoreML computes in 16-bit floats) is not used.

**Checking the models** ([`checks.py`](src/bifold_demo/recognition/checks.py)).
No model is trusted before it is checked, the community exports above all:

```bash
bifold-faces check --speed                     # the profile's models: files, graphs, a test run, speed
bifold-faces check --all --photos faces        # every fetched model; the colour order on photos of known people
bifold-faces check --model adaface_ir101_wf12m --pack lfw.bin cfp_fp.bin agedb_30.bin
```

The file must have its registered SHA-256; the graph its registered inputs,
outputs and opset and only standard ONNX operators; a run on random crops must
give finite values of the right shape, the same twice, and the same for a crop
alone as in a batch. With photos of people (a folder per person) the
registered colour order must separate them better than the other order; with
InsightFace's test packs (`lfw.bin` and others: aligned faces and pair labels,
read as plain data, never as code) the 10-fold verification accuracy must come
within 0.2 points of the published one. The results here are in
[docs/research/face_recognition_f1_results.md](docs/research/face_recognition_f1_results.md):
every file and graph passes; on LFW (a balanced 1,500-pair subset here: a
ResNet-100 takes 0.4-0.6 s per face on this machine's four cores, so the full
protocol and the CFP-FP and AgeDB packs, which are on Hugging Face, run on the
Mac) SFace scores 99.60 % on all pairs, the `buffalo_l` ResNet-50 99.87 % and
AdaFace IR-101 99.87 % (published with mirrored crops: 99.82 %), so the
community export works; on 540 renders of 20 avatars a frontal photo names
every render turned up to 60 degrees with each model (d' 4.6 for SFace, 5.3 for
the ResNet-50, 5.4 for AdaFace); avatars look more alike to the models than
strangers do on LFW, so thresholds are not set on renders.

**Privacy.** A face embedding used to identify someone is biometric data, and
embeddings can be turned back into recognisable faces. The gallery's caches
live in the faces folder beside the photos and lose a person's entries in
every model's cache when their photos leave it; nothing is kept of people who
are not in the gallery; ONNX Runtime's telemetry is off. Identifying people in
a publicly accessible space is regulated (the EU AI Act, GDPR article 9):
recognise only enrolled people who agreed, say so on a sign, and get legal
advice before any real use.

### Known faces on the 3D people

When `faces/` has photos, the demo puts every known person's face on one of the
3D people, those standing in front of the camera first, so the recognition
finds them in the fair (`--no-cast` to leave the 3D people as they are). People
added to the folder while it runs get a 3D person a second later; a person
removed from it gives the face back. In selection mode `5` (in the items menu)
puts a known face on the selected person by hand.
[`recognition/cast.py`](src/bifold_demo/recognition/cast.py) does the casting,
[`characters/face_photo.py`](src/bifold_demo/characters/face_photo.py) the
painting:

```python
from bifold_demo.characters.face_photo import put_face, take_face_off

put_face(person, "faces/Ada Lovelace/portrait.jpg")   # her biggest face in the photo
take_face_off(person)
```

The photo's five landmarks are lined up with the same five points on the
avatar (its eye and mouth-corner bones and the tip of its nose, seen from the
front). Every part of the head that can be seen from the front is then painted
with the photo pixels in front of it, in the head texture's own layout (a depth
buffer of the front view leaves out the inside of the mouth and the teeth). The
painted area fades into the avatar's skin at the edges and where the head
turns sideways, only the photo's face (not its hair or background) is used, and
the photo's skin tone moves half way to the avatar's. Since the face is a
texture on the avatar's own mesh, it turns and talks with the head and makes
the funny faces. The texture is that person's own (`texture_overrides`); others
with the same avatar keep their face. Painting takes about 0.4 s.

With MediaPipe installed (see [below](#head-textures-from-photos)) the demo
does better: all the person's photos go through the
[head textures from photos](#stages-5-to-7-the-whole-person), and the 3D
person gets a new head texture (the photos projected onto the avatar's own
head, the rest completed), a body texture with the person's skin and hair
cards in the person's hair colour. That takes a few seconds per person on the
worker thread (the first time for an avatar some more), and the result is
kept in `faces/.headtex/<name>/` by the photos' contents, so the next start
reuses it; it is deleted when the person leaves the faces folder. Without
MediaPipe, or when the photos cannot be used, the face is painted from one
photo as described here, and a message says why.

Tested with photos of three real people painted on four avatars (women, men, a
child), the recognition (YuNet and SFace, before the core above) named the
right person in 24 of 24 close-ups (front and turned 40°; similarity
0.37-0.58) and, from the fair camera with the faces about 30 pixels tall, in 98
of 120 analysed frames.
[docs/previews/face_photo.jpg](docs/previews/face_photo.jpg) shows it with
renders of other avatars as the "photos".

## The virtual high-resolution sensor

Faces in the 1280x720 view have only 10-22 pixels between the eyes, while
face recognisers want 40 and more (112x112 crops with the eyes 35 pixels
apart, more for landmarks and quality checks). Rendering the whole view at 4K
would cost about 5 to 10 times as much and at 8K 13 to 30 times (on this
machine the fair's view takes 0.2-0.5 s, its 4K frame 2-3 s and its 8K frame
6-9 s), and still leave far people small. So
the view stays at its size for the detector and the screen, and the faces get
their pixels from **windows**: a rectangle around a person's head, rendered
on its own at a much higher pixel scale, as a real high-resolution camera's
crop ("digital PTZ") would deliver it.

[`sensor/`](src/bifold_demo/sensor) declares a **sensor** whose whole frame is
exactly the scene camera's view, by default six times the view in each
direction (7680x4320 for 1280x720). A window is a rectangle of that sensor,
snapped to its pixel grid, and drawn through an off-axis part of the camera's
frustum: the camera's projection `P` with a crop matrix `S` in front of it
(`S @ P`, the matrix of `gluPickMatrix` and of tiled rendering). The geometry
is exactly the camera's, so a window is exactly that crop of a render of the
whole 7680x4320 frame, with the same perspective, depth, fog, MSAA and
texture filtering, for the cost of the window's pixels. It is drawn by the
view's own renderer, with the meshes and textures already on the GPU.

When a face would get more pixels than asked for, the window is a crop of a
*binned* read-out of the sensor instead: at level `k` one window pixel is
`k x k` sensor pixels, and only the binnings that divide both sides of the
sensor are levels, so the whole frame at that level is again a sensor
(7680x4320 has 24 of them: 1, 2, 3, 4, 5, 6, 8, ... 480). `Sensor.plan`
picks the coarsest level at which the face still has the wanted eye
distance (level 1 when no level gets there), then a coarser one while the
window has more pixels than allowed. `Sensor.with_lines` and `Sensor.scaled`
choose sizes with many levels: for a view of an odd shape the width moves by
a few pixels (a 1366x768 view's 1080-line sensor is 1920x1080, not 1921x1080
with level 1 alone). On a declared sensor whose sides still share few factors
the plan goes on to whole binnings beyond the levels, whose frame leaves out
the last few pixels at the right and bottom edges, so a window never has more
pixels than allowed. Pixel coordinates map between the view,
the sensor and a window by a scale and a shift, exactly, both ways, so what is
found in a window maps back to the main frame:

```python
from bifold_demo.sensor import Sensor

sensor = Sensor(7680, 4320, view=(1280, 720))      # or Sensor.scaled(6, (1280, 720))
window = sensor.plan((600, 100, 720, 220), ied=11, face_px=112)  # region in view pixels
image = renderer.render_window(scene, window.ndc, window.size)   # after renderer.render(scene)
window.box_to_view((40, 60, 200, 240))   # a face found in the window, in view pixels
window.sensor_rect, window.view_rect, window.level, window.scale
```

`window.camera(scene.camera)` is the same window as a camera
(`scene.SubCamera`), for a renderer of the window's own size.

**Culling.** A window draws only what may show in it
([`scene/frustum.py`](src/bifold_demo/scene/frustum.py)): meshes by their
bounding boxes, blob shadows by their rectangles, and people by one sphere per
bone around its posed joint, as large as the farthest vertex that bone moves
(no skinned vertex left these spheres in 84 poses of each of the 20 prepared
avatars). The tests are conservative, so culling changes no pixel, and they
take about 1 ms per window for the 36 people and 124 objects of the fair. A
window also reuses the joint palettes its frame uploaded. On the fair
(llvmpipe, a shared 4-core machine), head windows drew 1-13 of the 36 people
and 2-38 of the 124 objects; windows up to 200 pixels took 27 ms (median)
instead of 59 ms, about 2.2 times faster, while the large windows of people
close to the camera are bound by their pixels there (1.2-1.4 times). The
whole 1280x720 view takes 220-470 ms, the whole 8K frame 6.3 s. The whole
view is not culled (`renderer.cull_view`): most of the fair is in it, and
culling gained nothing there.

**Asking for windows.** A `WindowRequest` names a region of the main view in
one of three ways: a person's **box** from the detector (the window then
covers the head's square: centred 12 % of the box height below its top, 30 %
of the height wide, plus 30 % margin for the face detector; these starting
values are `HeadRegion`'s and can be tuned), a **region** as it is (a face
found in an earlier window, with a margin), or a **head point** in the scene
(for the evaluation). It says how many pixels the face should get between the
eyes (`face_px`, 112 by default: the window is at the coarsest level that
still gives them, or at the sensor's own scale when none does), the most
pixels the window may have (1024x1024 by default), a priority and a tag (the
person track). A `WindowService` on the frame source takes standing requests
from any thread; with every frame its scheduler picks at most `max_windows`
(4) of them, the highest priority first and otherwise in turn, and the
source delivers their windows on the frame, rendered right after it from the
same state of the scene:

```python
from bifold_demo.sensor import WindowRequest, WindowService
from bifold_demo.sources import SceneSource

service = WindowService(max_windows=4)               # sensor: 6x the view unless given
source = SceneSource(scene, 1280, 720, windows=service)
service.request([WindowRequest(track.id, box=track.box, priority=track.urgency) for track in tracks])
frame = source.read()
for window in frame.windows:                         # WindowResult
    for face in engine.detect(window.image):         # recognition.FaceEngine, on the window's own pixels
        x, y, w, h = face.box
        in_frame = window.box_to_view((x, y, x + w, y + h))  # back in the main frame's pixels
    window.tag, window.time, window.window.level, window.window.scale, window.eye_distance
```

`service.scheduler.priority = fn(request, waited)` is the hook for the person
tracker's own order (`waited`: frames since the tag's last window). Only
requests whose window is in the view take part, so tracks that just left the
picture take no window from the others. A request is checked where it is
made: an unknown preset fails when the request is made, a sensor for another
view in `service.request()` (which then changes nothing); the source's frames
never fail for a request, and one that cannot be served after all is left
out, said once and kept in `service.problems`.
`render_windows(scene, renderer, requests, sensor)` renders windows directly,
for the evaluation; there a request can name its `subject` and its own
`sensor`.

**Exact eye distances (evaluation).** `face_px` asks for at least that many
pixels and is quantised by the levels (48 asked for an audience face at 4 m
gives 75 with the 8K sensor). An evaluation that sweeps face sizes asks for
`eye_px` instead: the window is drawn at a scale of its own (level 1 of a
sensor of exactly that scale), so that the face turned to the camera gets
that many pixels between the subject's own eyes, whatever the sensor's
levels, `max_pixels` or the preset's sensor size; the preset's faults still
apply, at the window's pixels. On the fair's people, 8 to 160 pixels asked
came out within 0.1 % (`truth.frontal_eye_distance`):

```python
from bifold_demo.sensor import WindowRequest, render_windows
from bifold_demo.sensor.service import eye_points

head = tuple(eye_points(person).mean(axis=0))   # or any point of the head
requests = [WindowRequest((person.name, px), head=head, subject=person, eye_px=px, preset="good 4K")
            for px in (12, 24, 48, 96)]
for result in render_windows(scene, renderer, requests, visibility=True):   # after renderer.render(scene)
    result.truth.frontal_eye_distance, result.truth.facing, result.truth.visible
```

Without a `subject` the face's size is estimated as for `face_px` (`eye_m`
metres at a head point, a box's head region, or `eye_distance`). These
windows are for scene renders; a real camera's picture leaves them out.

With a scene, each window also carries the `truth`: who the scene
knows is in it (the person the window was placed on, not a small face behind
them), where their eye bones are in the window, the face's frontal eye
distance (its size if it were turned to the camera where it is, the same
measure the planning uses), how the head is turned (`facing`: 0 looking into
the camera, 90 in profile, 180 the back of the head; and `yaw`, `pitch`,
`roll`), the distance, their speed in window pixels a second (from their
eyes' move since the frame before when that is less than 0.1 s ago, else
from their behaviour's track over the floor, so that a slow demo's frames
still blur) and, with `visibility=True`, whether their eyes are hidden behind
someone. It is for evaluation only; recognition must not use it.

**A real camera** has no virtual sensor, but the same requests work on it:
`WebcamSource(capture=(3840, 2160), windows=service)` captures at the higher
resolution, gives the detector and the screen a copy scaled to the frame
size, and cuts the windows from the full picture (averaged over `k x k`
pixels at level `k`).

**The camera model.** A render is noiseless, perfectly focused and frozen in
time; a camera's picture is not. A camera preset
([`sensor/camera_model.py`](src/bifold_demo/sensor/camera_model.py)) is
applied to every window in the order light meets the faults: in linear
light, lens blur (a Gaussian, in sensor pixels), motion blur (the person's
projected speed times the exposure, a line kernel;
only the person is smeared, over the still background, using their coverage
of the window from `Renderer.window_mask`), optionally a Bayer mosaic,
Poisson-Gaussian sensor noise in electrons (a binned pixel collects `k x k`
times the light), OpenCV demosaicing, then the sRGB and a contrast tone
curve, sharpening and JPEG. The noise is seeded by the preset's seed, the
frame, the tag and the window's place, so runs reproduce. A window binned for
a near face meets the faults at the camera's own pixels: it is drawn at the
sensor's pixels (up to a million of them, `native_pixels`), goes through the
camera there and is then averaged down, as a camera that sends its full
picture delivers a face cut out and scaled down where it is used (the
resampling comes last). `CameraModel(native=False)` models a binned read-out
instead: cheaper, and harsher, since the mosaic, the sharpening and the JPEG
blocks then work at the binned pixels.

| Preset | Sensor | Lens blur | Exposure | Electrons at white / read noise | Mosaic | JPEG |
|---|---|---|---|---|---|---|
| `ideal` | the declared one (7680x4320) | - | - | - | - | - |
| `good 8K` | 7680x4320 | 0.7 px | 1/250 s | 2500 / 1.5 | edge-aware | 90 |
| `good 4K` | 3840x2160 | 0.6 px | 1/250 s | 6000 / 2.0 | edge-aware | 90 |
| `budget 1080p` | 1920x1080 | 1.0 px | 1/60 s | 1500 / 4.0 | bilinear | 70 |

A preset with its own sensor size places its windows on that sensor, so
`budget 1080p` shows what such a camera gives the recogniser: a frontal face
at 4 m has about 75 pixels between the eyes with the 8K sensor, 37 with 4K
and 19 with 1080p (the fair's audience gets 66-81 with the 8K sensor). The numbers are plausible values for such cameras, not
measurements of one; calibrate them against the intended camera's footage.
`WindowService(preset="good 4K")` sets the preset (a request may bring its
own); `CameraModel(...)` makes another, and `preset.with_exposure(1 / 1000)`
trades motion blur for noise. On this machine's CPU a preset takes about
8 ms for a 256x256 window, 30 ms for 512x512 (40 ms with motion blur) and
60-85 ms for 768x768 of the pixels it works at, so a window binned by `k`
costs as one `k` times larger (a 128x128 window at level 4, drawn and through
the camera, 84 ms instead of 15 read out binned); `ideal` takes nothing.

**In the demo**, `W` (or `--windows`) shows the windows: up to four a frame
on the heads of the people the detector finds (so detection must be on),
taking turns, in a column on the right with each window's place marked in
the picture, its level, the eye distance in it and its time. Until the person
tracker asks for windows itself, the demo follows the detector's person boxes
and keeps their numbers from frame to frame by their overlap. The options:

```bash
bifold-demo --windows                          # start with them shown
bifold-demo --windows --camera-preset good_4k  # through a camera preset
bifold-demo --windows --sensor 15360x8640 --max-windows 2   # a larger sensor, two windows a frame
bifold-demo --source webcam --webcam-capture 3840x2160 --windows   # cut from a 4K webcam picture
```

**From the command line**, `bifold-scene windows` renders a scene at a moment
(as the headless demo would show it) and the windows on everyone in the
picture, through one or more presets, and writes the view with the windows'
places, every window, a contact sheet and `windows.json` (each window's place,
level, scale, time, and who is in it: distance, eye distance in the view and
in the window, whether a face is hidden behind someone). With `--faces` it
also runs the face detector on the view and on each window:

```bash
bifold-scene windows --time 6 --faces -o windows/               # the fair, ground-truth person boxes
bifold-scene windows --time 6 --people detect --model yolo26n.pt  # YOLO's person boxes
bifold-scene windows --time 6 --people heads --preset ideal --preset budget_1080p
```

**Measured** ([docs/research/virtual_sensor_results.md](docs/research/virtual_sensor_results.md)):
with the 8K sensor a face facing the camera has six times the view's eye
distance, 115 pixels at 3 m, 75 at 4 m, 46 at 6 m, 26 at 10 m and 13 at 20 m
(19, 12.5, 7.7, 4.3 and 2.2 in the view). The face detector (YuNet) found 36 of
36 such faces at 2-30 m in the windows, against 20 of 36 in the view (none
beyond 12 m). The windows equal the crops of whole 8K and 4K renders of the
fair (98.7-99.9 % of the pixels identical, 53-82 dB). The head square of the
review's starting values held the whole face in all 91 YOLO boxes measured,
the face a quarter of its side; `HeadRegion(centre=0.10, side=0.20)` held it
in 91 % with 2.2 times fewer pixels.
[docs/previews/virtual_sensor.jpg](docs/previews/virtual_sensor.jpg) shows
people at 2-30 m, their windows and one face through the presets.

**Agreement.** Windows equal the same pixels cut from a render of the whole
sensor: on a test scene with textured boxes, a sphere, skinned people with
their blob shadows and fog, eleven windows of 16 to 420 pixels at levels 1 to
5 of 4x and 6x sensors had 99.9 % or more of their pixels identical to the
crop (four windows all of them) and a PSNR of 64.7 dB or more; the pixels
that differ lie on silhouettes, where the MSAA samples fall a little
differently. A window's bottom-left corner lies on an even pixel of the frame
(as OpenGL counts rows, from the bottom), so that its 2x2 pixel quads, and
with them the texture filtering, line up with the whole frame's: in the
fair's head windows that keeps 99.8 % of the pixels identical, against
99.2-99.5 % at odd offsets. `tests/test_sensor.py` checks this, also with a
background image and on real avatars' heads. Windows share framebuffers of a
few size buckets, kept within `Renderer.window_bytes` (192 MB).

## High-resolution heads

The windows of the virtual sensor give a face 75 pixels between the eyes at
4 m and up to about 230 close by. A 1024-pixel head texture has about 112
texels between the eyes, so beyond about 110 pixels it is magnified and the
face goes soft; the Rocketbox sources are 2048 pixels, with a normal and a
specular map for every head ([docs/research/high_res_faces.md](docs/research/high_res_faces.md),
sections 2.2 and 3.1). Step S2 of the plan
([docs/research/face_recognition_focus.md](docs/research/face_recognition_focus.md))
brings them in.

**The assets** (cache format 2). `bifold-characters fetch` also downloads each
head's `<x>_head_normal.tga` and `<x>_head_specular.tga` (25 MB more per
avatar; no wrinkle maps), and preparing an avatar keeps, for its head:

- the colour texture at 2048 pixels as the head's `detail` texture, and a 1024
  copy as its ordinary `texture`. Textures are made smaller by averaging
  light, not sRGB-encoded values (`scene.textures.downsample`; every texture
  of a prepared avatar is now), so a fine dark-and-light pattern keeps its
  brightness;
- a *packed map* at both sizes, in one RGBA image: the normal's x and y in red
  and green (z is rebuilt when drawing), the specular level in blue, and room
  for ambient occlusion in alpha (255 for now);
- a tangent per vertex (`tangents.py`): the direction the texture's `u` grows
  in, laid into the plane of the vertex normal, angle-weighted and summed over
  the vertices that share a position and a UV (split only by their normals),
  with the handedness (+1, or -1 where the UV layout is mirrored: only a few
  triangles of the mouth on the Rocketbox heads).

`SkinnedPrimitive` gained `tangents`, `maps` and `detail` (a `Detail` with the
larger `texture` and `maps`); `primitive.full_texture` is the largest colour
texture, which work on the texture itself (head textures from photos) starts
from. Bodies, hair cards and props stay at 1024 without maps. A prepared
avatar takes about 20 MB on disk instead of 5.

**The green channel.** A tangent-space normal map can store its y component
pointing either way along the texture's `v` (OpenGL and Unity: green up;
DirectX: green down), and the Rocketbox maps were baked in 3ds Max. A normal
map baked from a height field holds its gradient, and a gradient has no curl,
unless one of its components is read backwards: `tangents.normal_green` checks
every map while preparing, at the scale of the folds (a blur of 4 texels at
2048). On the 20 prepared heads the residual curl is 0.33-0.56 of the terms
read green-up and 0.85-0.96 read green-down: the Rocketbox heads are
green-up (`+v`), and the check would flip a map stored the other way (the
result is in each avatar's metadata, `head_maps`). One head, `Male_Adult_04`,
has a fine skin texture that reads the other way below about 2 texels, while
its folds read like every other head's; the folds decide.

**Linear light.** Light adds up, and pixels average, in linear light, while
images hold sRGB-encoded colours; until now the renderer lit, filtered and
blended the encoded values. Now textures are sRGB textures (the GPU decodes
a colour when it samples it), their mipmaps are averaged in linear light on
the CPU (on the thread that decodes a character's textures, and by
`prepare_upload` for textures made on workers; textures that change, like
videos, keep the driver's), shading, fog and the blob shadows' blending are
in linear light, and the picture is an sRGB framebuffer: the GPU encodes what
the shaders write and averages the MSAA samples of a pixel as light. A fine
black-and-white pattern seen small is now the grey of half the light (level
188) instead of 128, and edges are antialiased as a camera's pixels would
average them; mipmaps of cut-outs (hair cards) are averaged with their
coverage, so transparent texels no longer darken the edges of hair. Colours
given to a scene stay sRGB colours, and `ambient` and the blob shadows'
darkening are given as they look, so surfaces in full light and in shadow
look as before and only the fall-off between them changes (lighter, as light
on a curved surface really falls off). Fog veils dark distant surfaces sooner
when it mixes light: the fair's fog strength went from 0.55 to 0.2, which
keeps its far end within a level of how it looked (3.2 levels mean absolute
difference over the whole 1280x720 view at 6 s, the faces practically
unchanged). `renderer.linear = False` draws on the stored sRGB values, as
before. The head-texture pipeline draws its own renders of an avatar that
way, whatever the renderer's setting (`landmarks.render_views`: stage 0's
landmarks and the head's regions are found on them): registration magnifies
small changes in them. With linear light they moved the landmarks by a tenth
of a millimetre (the median), which shifted the camera fitted to a small
photo enough to lower its quality by a tenth and the share of its face texels
worth refining from 0.13 to 0.08; drawn on the stored values they are the same
as before to the bit.

**Shading.** At the renderer's high quality (the default) a head is drawn
with its maps: the tangent frame, skinned like the normal, turns the map's
normal into the surface's, so the folds of lips, nostrils, eyelids and the
nasolabial folds and the pores react to the light; a specular lobe (GGX with
Schlick's Fresnel and skin's reflectance of 0.028 head-on, roughness 0.45)
is scaled by the specular map, which the Rocketbox artists painted higher on
the forehead, nose and lips, and sharper where the level is high (lips and
the wet eyes, roughness down to 0.2); and the ambient light comes more from
above than from below, as much as the scene's `hemisphere` says (the fair:
0.3, a floor-facing surface gets 0.7 of the ambient light, an upward one 1.3).
`renderer.quality = "low"` draws everything as before (Lambert and a constant
ambient light, in linear light), and `renderer.window_quality` sets the
windows' own, so a software renderer can keep the whole view cheap while the
windows get everything (`bifold-demo --quality low`). `renderer.specular` and
`renderer.normal_strength` scale the lobe and the maps.

**Detail levels.** The 2048 textures cost four times the memory of the 1024
ones (21 MiB per texture with its mipmaps, 43 MiB per head with its maps)
and add nothing to a face smaller than about 110 pixels between the eyes. So
the renderer draws a person's head with them only in a picture where that
person's eyes are at least `renderer.detail_from` (64) pixels apart: a
window of the virtual sensor, or a camera close to someone; everywhere else
the 1024 ones ([scene/detail.py](src/bifold_demo/scene/detail.py)). The
detail textures of the last `renderer.detail_heads` (6) heads drawn that way
stay on the GPU, the others are freed at once. A picture that needs them
reads them first (`renderer.detail_wait`, for tests and measurements); the
demo reads them in the background and draws the 1024 ones until they are
there, so a frame never waits for a 2048 PNG. A texture larger than 1024
that a person wears instead of the model's (a face made from photos) is drawn
from its 1024 mipmap level the same way. `renderer.memory()` says what the
textures take.

**Faces from photos at 2048.** The head-texture stages work at the size of
the head's largest texture (`fuse.texture_size`: the detail texture's), so
on a head prepared at 2048 a face from photos is made at 2048, and a created
character keeps it at that size. The fusion still marks a photo's resolution
per texel of the head's ordinary 1024 texture (`fuse.marks_size`): a photo
with about 112 pixels between the eyes has full marks at either size, so how
much the photos count against the avatar's own texture, how sure the result
is and what stage 5 fills in do not change with the size (counted per 2048
texel, a 480-pixel portrait had lost nearly all of its face to stage 5). To
fill the 2048 texture at one photo pixel per texel a photo needs about 225
pixels between the eyes (from about 0.75 m with a phone's main camera, 1.5 m
with a 2x lens). For four photos on this 4-CPU machine, shared with two test
runs, stages 3 and 4 took 19 s instead of 5 s, stages 5 to 7 16 s instead of
8, and the head's atlas 11 s instead of 4 (once per avatar; it is cached on
disk); an atlas holds 264 MiB instead of 66 (two are kept in memory,
`fuse.ATLASES`), and the process grew by 1.3 GB instead of 0.2 GB. The
texture comes back as well as at 1024: fake photos of an avatar fused onto
its own head match its texture within a median delta E of 3.3 (90 % within
8.1), 3.0 and 7.5 compared at 1024, as at 1024.

**Measured** ([docs/research/high_res_heads_results.md](docs/research/high_res_heads_results.md),
preview [docs/previews/high_res_heads.jpg](docs/previews/high_res_heads.jpg)).
The maps change faces the most: in the sensor's windows a face's embedding
moves to a cosine of 0.86 from the same window drawn flat, and the 20 avatars'
identities separate a little better (d' up by 0.3-0.4 for AdaFace IR-101,
ArcFace R50 and SFace). The 2048 textures hardly reach today's embedders,
whose crops have the eyes 35 pixels apart: from 1024 to 2048 the embeddings
move to a cosine of 0.998 and the separation does not change; they show
where a picture shows a face larger than about 110 pixels between the eyes
(close portraits, S4's close-ups). YuNet and SCRFD find every face of the
measurement (2-5 m) in every configuration. On the software renderer linear
light and the maps make the view about 1.25 times and the head windows 1.4
times slower; the maps take 107 MiB for the 20 heads, the 2048 textures 43
MiB per head drawn with them.

**Older caches** (format 1, without maps) still load and draw as before;
`bifold-characters list` names them, and `fetch` makes them again with the
maps (it downloads their files again). To upgrade a library in place:

```bash
bifold-characters fetch --avatars NAME ... --clips basic --delete-raw   # the avatars it names
```

## Head textures from photos

[`characters/headtex/`](src/bifold_demo/characters/headtex) turns photos of a
person's head, taken from several angles (front, three-quarter, the sides,
maybe the back; casual phone photos), into a texture for an avatar's own head
mesh that blends into the whole body. The approach is the one of the
literature review
([docs/research/face_head_texture.md](docs/research/face_head_texture.md),
section 3), in eight stages; all but the optional stage 2 (normalising the
photos with a generative model) are built: 0 and 1 here, 3 and 4
[below](#stages-3-and-4-the-texture), 5 to 7
[further below](#stages-5-to-7-the-whole-person):

0. **Landmarks on the head mesh**, once per head model. The avatar's head is
   rendered with its own texture in the bind pose (flat light) from five
   views: straight in front, turned 25° to each side, seen from 12° above and
   below. MediaPipe's Face Landmarker puts its 478 landmarks on each render,
   and each landmark is lifted onto the head mesh by casting the ray through
   its pixel against the head's triangles. The views are combined landmark by
   landmark (the median, then the mean of the views that agree, back onto the
   surface). Each landmark keeps its triangle and barycentric coordinates, so
   it follows the skinning, and a reliability (how many views agree, how
   closely). If MediaPipe finds no face on the renders, MediaPipe's canonical
   face model (read from the Face Landmarker's model file) is fitted to the
   avatar's eye and mouth-corner bones and to its nose tip, chin and ears
   found on the mesh. The result is cached next to the prepared characters
   (`headtex/`, keyed by the head's geometry; `BIFOLD_HEADTEX_CACHE` puts it
   elsewhere). It takes about 2 s per avatar.
1. **Register each photo** (a `View`). The photo's landmarks and the avatar's
   give the photo's camera: `cv2.solvePnPRansac` drops the landmarks that
   disagree, then the camera is fitted to the rest, again without those on
   the far side of the face. The camera is relative to the head in its bind
   pose, with the head's yaw, pitch and roll. The focal length comes from the
   caller or the photo's EXIF data. Without either, a 50° view is in effect
   assumed: MediaPipe's landmarks tell the perspective so poorly that a search
   ends near 50° whatever the lens, so the camera of a wide-angle selfie comes
   out too far away (its pose hardly suffers), and the view says the focal
   length was assumed. Photos of one session share one camera: a focal length
   known for one photo holds for the others of the same picture shape, and
   the most frontal photo's for those still without one. A thin-plate warp
   then moves the photo so the person's eyes, brows, nose and mouth land on
   the avatar's; it fades out around the face, so hair and background stay
   put. Only landmarks the camera sees steer it (the surface faces the camera,
   by the mesh's smooth normals, and no other part of the head, such as the
   nose, is in front). Projected onto the mesh, the person's features then
   move with the avatar's face bones.

   MediaPipe makes up faces in profiles and on the backs of heads. A face is
   not used when its pose cannot be right (leaning more than 45°, looking up
   or down more than 35°, more than 25° from MediaPipe's own estimate of the
   pose), when the segmenter finds hardly any face skin in it or the avatar's
   head lands where the photo has none, or, without masks, when a nearly
   frontal face is no face to YuNet (the face recognition's detector). In a
   photo named as a side or the back, the face is used only when the head's
   outline (below) agrees with it.

   A photo without a usable face (the side or the back of the head) needs its
   view named (`left`, `right`, `back`) and the segmenter's masks. The camera
   is then placed there, at the distance of the session's photo with a face,
   and refined by fitting the avatar's head to the photo's: its outline (head
   and hair) to the segmenter's head mask, and its face skin to the photo's
   (where the segmenter sees face skin on the avatar is found once per head
   model from renders, about 6 s, and cached). If they do not fit, the photo
   is reported unusable with the reason. Every view has the segmentation masks
   (hair, face and body skin, clothes; small copies, `view.mask(name)` gives
   one at the photo's size), a quality score and warnings: an open mouth,
   closed eyes, low resolution, blur, a large reprojection error, an assumed
   focal length, a pose MediaPipe may get wrong, a view placed by its outline.
   With YuNet's model (`bifold-faces fetch`), faces down to about 60 pixels
   wide are found in a 12-megapixel photo.

How well it works (`tests/test_headtex_avatars.py` and measurements: renders
of the 20 avatars from known cameras stand in for photos, registered to
another avatar's head as a person's photos are):

- Stage 0: on all 20 avatars of the test library MediaPipe found the face in
  all five renders, including glasses, a headscarf and a keffiyeh; mean
  reliabilities were 0.72-0.83. The fallback, forced on all 20, lands 6.9 mm
  from MediaPipe's landmarks on average: about 5 mm at the eyes, mouth and
  nose, 8 mm at the brows, 17 mm on the face's outline.
- Stage 1 with landmarks, focal length known: renders of the 20 avatars on
  Male_Adult_05's head turned 30° and 45° came back within 2° of their turn
  (the median; all within 6.2°); of those turned 60°, MediaPipe found 35 of
  40, within 4° (90 % within 8°, the worst 19° short). The faces differ from
  the head's, so the nod is a few degrees off (the median 2.4°, 90 % within
  6.6°; a child's face on an adult's head looks 5-7° further down), but
  through the warp each of the avatar's landmarks reads the photo within 2 %
  of the face's size of the person's (the median). An avatar on its own head:
  within 2.5° and 1.5 pixels up to 30°, 7° and 5 pixels at 60°. With the
  focal length assumed, about a degree more, and the camera's distance is off
  by as much as the lens differs from a 50° one. The test photos of real
  people (16 portraits from the web, not in the repository) registered with
  plausible poses and reprojection errors of 1-3 % of the face's size. A
  photo takes about 0.1 s.
- Made-up faces: at its default confidence MediaPipe found a "face" in 16 of
  220 renders turned 75° to 180°; none of those on the backs of heads was
  used, with masks or without. An unnamed profile turned 75° was registered
  by its half face 8 times, 12-25° short, with a low quality (0.05-0.12) and,
  7 times, a warning: name side photos and segment them.
- Stage 1 by the outline (photos named as a side or the back, with masks and
  a front photo of the session): of the 20 avatars' renders turned 75° to
  180° on Male_Adult_05's head, 111 of 200 came out usable, turned within 4°
  of the truth (the median; 90 % within 17°), but 6 were 25° to 50° off,
  mostly photos turned 135° (half way between a side and the back) pulled to
  the back; the rest were reported unusable, mostly where the hair differs
  too much from the head's (long hair, a headscarf; the segmenter also calls
  a bald scalp face skin, which misleads the fit). The nod and the lean came
  out within 5° (the median; 90 % within 13°, the worst 20° and 30°).
  Of the photos turned 60° and named as a side, 28 of 40 were usable: by the
  face when the outline agreed with it, else by the outline, within 6° (the
  median; the worst 15°). On a long-haired head (Business_Female_02) it is
  worse: half of the photos usable, within 10° (the median; the worst 41°).
  Nothing in a fit tells a bad one from a good one, so a view placed by its
  outline gets a quality of at most 0.6 and a warning. A fit takes 1-3 s.

[docs/previews/headtex_register.jpg](docs/previews/headtex_register.jpg) shows
both stages.

**MediaPipe is optional.** The rest of the demo works without it; the head
texture commands say how to install it when it is missing. Its package asks
for `opencv-contrib-python`, which must not be installed beside the project's
`opencv-python` (both provide `cv2`), so it goes in without its dependencies:

```bash
pip install -e ".[headtex]"                 # what MediaPipe needs besides numpy and OpenCV
pip install --no-deps "mediapipe>=1.0"      # MediaPipe itself
bifold-faces fetch --headtex                # face recognition's models, MediaPipe's face landmarker (3.8 MB)
                                            # and segmenter (16.4 MB), into models/
```

On Linux MediaPipe also needs libEGL and libGLESv2
(`sudo apt install libegl1 libgles2`). The models come from Google's MediaPipe
model bucket (Apache 2.0) and are checked by size and SHA-256.

```bash
bifold-faces landmarks                       # stage 0 on every prepared avatar (cached), and which way it went
bifold-faces register front.jpg side.jpg:left back.jpg:back --avatar Male_Adult_05 --segment --out registered/
```

`register` prints each photo's pose, error, quality and warnings, and with
`--out` draws the registration over each photo: the avatar's head mesh as the
photo's camera sees it, through the warp, and the landmarks.

```python
from bifold_demo.characters.headtex import head_landmarks, register_photos
from bifold_demo.characters.headtex.detect import Landmarker, Segmenter

landmarker, segmenter = Landmarker(), Segmenter()
head = head_landmarks(model, landmarker)              # stage 0 (rendered once, then cached)
views = register_photos(["front.jpg", "left.jpg"], head, model, views=[None, "left"],
                        landmarker=landmarker, segmenter=segmenter)
for view in views:
    print(view.summary())                             # yaw, pitch, roll, error, quality, warnings
    pixels = view.photo_pixels(head.points)           # where each 3D point of the head is in the photo
```

### Stages 3 and 4: the texture

3. **Project and fuse the photos** (`headtex/fuse.py`). Every texel of the
   head texture knows its point on the head (the UV bake). For each photo it
   is depth-tested against the head as the photo's camera sees it (a z-buffer
   in numpy) and read from the photo through the face warp, at the mipmap
   level that matches its size in the photo. Its weight is the product of
   the viewing angle (cos³), the segmenter's class at the pixel (face skin on
   the face; hair only on the avatar's scalp and at its hairline; face and
   body skin on the neck; never background, clothes or other people), the
   distance from the masks' edges, the photo's border and the head's
   outlines, the view's quality and the photo pixels per texel. A gain per
   photo and channel evens out exposure and white balance on the overlaps
   (least squares, anchored on the most frontal photo). A five-band blend in
   UV, island by island (nothing blurs across islands), takes the fine detail
   from the best photo at each texel and mixes the tone over wide areas. The
   avatar's own texture, tone-matched to the person's skin, is one more
   source with a small weight: what no photo shows keeps it, and the photos
   feather into it. Push-pull padding fills the gutters. The eyeballs and the
   inside of the mouth stay the avatar's own (a photo's eyes look wherever
   the person looked; no camera sees inside the mouth), the brows come from
   the photos (the segmenter calls them face skin, on renders and photos
   alike). Every texel gets a confidence and a keep / refine / generate
   trimap for stage 5.
4. **Take the light out, set the skin tone** (`headtex/light.py`). Each
   photo's light is fitted to its face skin as order-2 spherical harmonics of
   the normal (luminance only, robust) and divided out; the avatar's own
   painted soft light, fitted the same way to its texture, is multiplied back
   in, so the new face sits on the body as the old one did. The person's skin
   tone, a trimmed Lab median of the cheeks and forehead over all photos, is
   kept for stage 6 (matching the body to it).

```bash
bifold-faces headtex front.jpg left.jpg:left right.jpg:right --avatar Male_Adult_05 -o faces/head.png \
    --views faces/head_views.jpg --debug faces/head_debug/
```

`headtex` prints each photo's pose and share of the texture, how much of the
face, scalp and neck the photos show, the skin tone, each photo's light and
gains, warnings and timings. `--views` renders the avatar wearing the texture
(front, three-quarter, side) above the avatar as it is; `--debug` writes the
texture's parts, confidence, trimap and each photo's colours and weights in
the texture's layout.

```python
from bifold_demo.characters.headtex import head_texture, head_texture_async, wear

result = head_texture(["front.jpg", "left.jpg"], model, views=[None, "left"])  # a HeadTexture
print(result.summary())
wear(person, result)                  # a scene Character of that model wears it (texture_overrides)
result.skin                           # the person's skin tone (a Tone: Lab, sRGB), for stage 6
result.confidence, result.trimap      # per texel: 0-1, and keep / refine / generate for stage 5
future = head_texture_async(photos, model)   # the same on a worker thread (a Future)
```

How well it works (`tests/test_headtex_fuse.py`, synthetic;
`tests/test_headtex_fuse_avatars.py` and measurements: four 1280×960 renders
of Male_Adult_05 stand in for a person's photos: front, turned 40° each way
and from below, each under a light of its own direction and colour):

- Fused back onto his own head, the face's texels the photos show surely
  (62 % of the face; 73 % at all) come back within ΔE 4.5 of the avatar's own
  texture (the median; 90 % within 9.3; PSNR 24 dB; about 2 L darker), the
  skin tone within ΔE 2-3.5. The eyes come back unchanged, the inside of the
  mouth within ΔE 3.
- On five other avatars, the same photos show 54-61 % of the face surely,
  23-50 % of the neck and 4-5 % of the scalp (they see little of it); the
  skin tone comes out nearly the same on all (Lab 63-66, +20, +22-23: the
  lightness follows each avatar's painted light). Where an open neckline shows the
  neck seam (Female_Adult_04, Female_Adult_01) the head and the body are ΔE
  3.5 and 6.0 apart across it (1.9 and 4.0 with their own heads); the face
  and the hands 2.8-12.9 apart (4.4-12.3 before): matching the body is stage
  6's.
- Harsh light, one light from the side: the difference between the two
  halves of the face (Lab L) with the light left in and taken out:

  | renderer's ambient light (fitted contrast) | 4 photos | the front photo alone |
  |---|---|---|
  | 0.55 (5×) | 20.7 → 2.8 | 14.8 → 3.8 |
  | 0.35 (18×) | 30.2 → 8.8 | 21.7 → 6.6 |
  | 0.2 (51×) | 36.8 → 10.5 | 26.8 → 7.1 |
  | 0.1 (51×) | 40.7 → 12.0 | 30.1 → 4.8 |

  Up to a contrast of about 5 (2.3 stops) the light comes out; beyond, 5-12 L
  stay, the lit half comes out 5-10 L darker than it should, and photos of
  the shadowed side add little (they see it only in shadow). Photos with a
  contrast over 8 get a warning. Shadows the face casts on itself (a nose's
  under a spotlight) and highlights stay.
- Real photos (portraits from the web, not in the repository; checks only)
  gave plausible textures, with flaws: a hand in front of the neck (body
  skin to the segmenter) leaves a smudge on the neck, a microphone over the
  chin leaves the avatar's goatee there, and a single blurred photo with a
  colour cast gives a greyish face.
- Time: four 1280×960 photos take 4-8 s on four cores (registration 1 s,
  projection 1.3-2.4 s, light and gains 0.7 s, blend 0.5-1.5 s); the first
  time for an avatar about 10 s more (its landmarks, regions and texture
  atlas, then cached on disk). `head_texture_async` runs it on a worker
  thread. Beside a render loop paced to 30 frames a second the frames were
  not slowed (16 ms against 14) and the texture took 9 s instead of 5; a
  render loop that never waits (the software renderer at 50 ms a frame)
  starved the worker (55 s): the renderer keeps Python's interpreter lock
  while it draws.

The hair the photos show goes onto the avatar's own scalp texels only; the
avatar's hair cards (buns, long hair, a fringe: a mesh with a texture of its
own) take the person's hair colour in stage 6, and what no photo shows (the
back of the head, under the chin) is completed in stage 5.

### Stages 5 to 7: the whole person

5. **Complete what no photo shows** (`headtex/completion.py`). A texel the
   photos show poorly takes the colour of its mirror image across the face's
   plane where the photos show that well (the Rocketbox heads are symmetric:
   a mirrored texel lands within a millimetre of another, on the median), so
   a photo of one side of the face completes the other; it is copied in
   albedo and blended in by the same multi-band blend. Where neither side was
   seen (the back of the head, under the chin) the avatar's own texture,
   tone-matched, stays. A generative layer can fill what is left
   (`FluxInpainter`: FLUX.2 [klein] through `diffusers`, or any pipeline
   passed in): one call on the texture with those texels painted grey, then
   only those texels take its result, tone-matched to the rest (its detail,
   not its tone), feathered and at 40 % at most. A model that cannot load or
   fails leaves the deterministic result, with a warning. It was tested here
   with a stand-in pipeline only (Hugging Face is out of reach from the
   sandbox); what FLUX.2 [klein] makes of a texture atlas is untested.
6. **Match the body** (`headtex/body.py`). The body texture's skin (the
   outfits' skin mask: hands, neck, bare arms and legs) takes the person's
   skin tone: one gain per channel in linear light takes the hands' robust
   median to the face's skin target (by the neck, the head's neck colour), so
   the body's painted creases and soft light stay. Where the head mesh's open
   edge meets the body and both textures are skin, a smooth offset round the
   neck (least squares over 24 sectors) is spread half into each side. The
   hair cards and the scalp no photo showed are recoloured towards the hair
   the photos show (its lightness and chroma; the strands' detail kept, skin
   and eyelashes left alone). Outfits go on over the new body texture.
7. **Check and save** (`headtex/person.py`). The colour difference across the
   neck seam and from the face to the hands, before and after stage 6, and,
   with `--check` (OpenGL and the face recognition's models), the identity:
   SFace similarity between renders of the textured avatar (front, turned
   30°) and the photos, next to that of the avatar's own head. The geometry is
   the avatar's, so the similarity is only meaningful relative to that: it is
   reported, never a reason to reject; a generative layer that lowers it is
   dropped. The head, body and hair textures can be saved as a
   [created character](#created-characters).

```bash
bifold-faces headtex front.jpg left.jpg:left right.jpg:right --avatar Female_Adult_04 -o out/head.png --full \
    --check --views out/views.jpg --character "Ada at the fair" --face "Ada Lovelace"
```

`--full` writes the body texture (`out/head_body.png`) and the hair cards'
(`out/head_<primitive>.png`) next to the head's, prints what each stage did
and the checks, and with `--character` saves a created character into
`created/` (`--created DIR`). `--generate` adds the generative layer (it
needs `diffusers` and the model's weights, about 16 GB of memory).

```python
from bifold_demo.characters.headtex import person_textures

person = person_textures(["front.jpg", "left.jpg"], model, views=[None, "left"], cache="textures/")
print(person.summary())                # what each stage did, the neck and the hands, warnings
person.wear(character)                 # head, body and hair textures on a scene Character of that model
book.save(person.created("Ada at the fair", face="Ada Lovelace"))   # a CreatedCharacter
```

The [character creator](#character-creator)'s faces come from it too.

How well it works (`tests/test_headtex_finish.py`, synthetic;
`tests/test_headtex_finish_avatars.py`: Male_Adult_05's four fake photos
finished onto Female_Adult_04, whose open neckline shows the seam): across
the seam ΔE 3.4 before stage 6 and 1.2 after, the neck against the body's
skin next to it 8.7 and 1.8, the face against the hands 7.0 and 0.2. SFace
says the renders look like him: 0.74 from the front (her own head 0.25),
0.65 turned 30° (0.29). On Female_Adult_01 (three photos) 6.0 → 2.1, 4.7 →
0.8 and 7.5 → 0.3; there two of the six seam pairs are a dark line painted at
the neckline, which the levelling leaves out. Time on top of stages 0-4: the
mirror's lookup about 2 s once per avatar, then stage 5 under 1 s, stage 6
about 1.5 s, the checks about 1 s (the identity check 0.5-1.5 s more).

Photos of people and textures made from them are personal data: keep them in
git-ignored folders (like `faces/`) and never commit them.
## Character creator

`C` opens the character creator (in selection mode: for the selected person,
as a new character of their avatar, clothes, items and known face; a created
character is opened to edit it). The avatar turns on a turntable on the left,
and on the right is a row per choice:

| Row                   | Choices                                                        |
|-----------------------|----------------------------------------------------------------|
| Avatar                | the prepared avatars                                           |
| Name                  | typed: `Enter`, the name, `Enter`                              |
| Face                  | none, or a known person of `faces/` (see [Face recognition](#face-recognition)) |
| Outfit                | the presets of [Outfits](#outfits)                             |
| Garment               | top, shirt, bottoms or shoes (those the avatar has): the next rows change it |
| Pattern               | the avatar's own, a plain colour, one of the twelve patterns, or the fabric |
| Colour, 2nd colour    | the garment's colour, and the second colour of a pattern       |
| Fabric                | a description such as "blue paisley silk": `Enter`, the text, `Enter` (or *make*) |
| Variations            | the fabric's swatches as they come; one goes on at once, the best when all are made, unless you pick one |
| Scale                 | the pattern's or fabric's repeat, smaller or larger            |
| Hat, Glasses, Sign    | none, or a look (see [Accessories](#accessories))              |
| Character             | save; delete (press it twice); new                             |
| Saved                 | the saved characters: `Enter` opens one to edit it             |

| Key                     | Action                                                  |
|-------------------------|---------------------------------------------------------|
| `↑` / `↓` (`Tab`)       | the row above / below                                   |
| `←` / `→`               | the choice before / after, at once (in the last two rows `Enter` does it) |
| `Enter`                 | type in the name or fabric row (`Enter` again ends it, `Esc` undoes it); do the action marked |
| `,` / `.`               | turn the avatar by 45°; `F` / `B` its front / back; `Space` turning on or off |
| `Esc` / `C`             | close the creator (it keeps what you made)              |

A click on a chip, a box or a button under the avatar does the same. Nothing
holds up the window: preparing an avatar (labelling its garments: a few
seconds the first time), dressing it (0.1-0.5 s), making a face (seconds) and
saving run on threads of their own, the fabric on the generator's thread, and
the status line says what is going on. The fabric generator runs as in
[Fabrics from a description](#fabrics-from-a-description), never downloading:
without a downloaded model you get the keyword fabric, and the fabric row says
why. The face comes from one function, `face_texture(photos, model)` in
[`creator.py`](src/bifold_demo/creator.py): the
[head textures from photos](#stages-5-to-7-the-whole-person), all stages,
when MediaPipe is installed (the head, a body texture with the person's skin,
which the outfit then goes on, and the hair cards in the person's hair
colour), else the person's best photo painted on (`face_photo.paint_face`).

**Saving** writes the character through the demo's character book into
`created/<name>/`: its name, avatar, whose face it wears, the outfit recipe
(fabrics with their images), its items and the textures (the body texture of
the outfit, and the head texture when it has a face; a body with the face's
skin is kept beside them as `<body>.base.png`, so that a loaded character's
outfit can be changed and keep that skin). The demo picks it up
within a second and it joins the fair ("Ada joined the fair", see below).
Saving under a name that is saved already asks you to press save again.
Opening a saved character restores all of it; its saved textures are used
until you change something. [docs/previews/creator.jpg](docs/previews/creator.jpg)
shows the screen.

## Created characters

Characters made in the character creator are saved in [`created/`](created/README.md)
(`--created DIR` for another folder, `--created ''` to show none), a folder
per character: `character.json` (its name, avatar, whose face it wears, the
outfit recipe and its items) and the textures painted for it. The demo puts
them into the fair. Each takes over the place of one of the fair's own people
who walk about and keeps that place's behaviour, so what the fair planned
(nobody walks through anybody, the camera play) stays valid. The places go in
this order:

1. the visitors who walk up the aisle to the camera (they are seen best,
   watch the camera and play with it),
2. the people walking by along the cross aisles,
3. the people wandering about.

A child's place is never taken (up to 13 places in all). Within each kind, a
place whose person is like the character's avatar (a woman's place for a
woman) comes first, and someone doing a yoga pose in selection mode (or who
did one) gives up their place last. In a place like the avatar, the place's
own behaviour fits the avatar as it is: Rocketbox grown-ups of one sex have
the same legs, so the feet do not slide, and the character does exactly what
that person did, where and when they did it. Elsewhere the behaviour is made
again for the new avatar: the same path and pauses, but the avatar's own walk,
turn and idle clips. It walks as fast as the person before when its clips can
keep up (else at its own pace), its timing is fitted in so that it keeps
0.6 m from everyone (the characters made just before it count too, so several
that join at once keep apart as well), and its camera play is planned around
the others'. It follows the old timing only roughly: in some places it is in
step with the person before for most of the loop, in others (turns and pauses
that fall differently, or a loop of another length, as for a child's avatar)
metres apart for long stretches.

**Nobody sees a change.** A viewer is never to see someone turn into someone
else, jump or pop up, so every change of a place (a character coming in,
another avatar, new looks, a character leaving) is made between two frames
where nothing of it shows. The person there is taken out at a frame where no
pixel of them is seen: out of the picture, hidden while walkers come back to
the start of their aisle, or behind a booth, a plant or someone else (the
renderer counts their pixels against the frame it has just drawn). The place
stays empty until the person coming in would show no pixel either, and then
they are there. Over the fair's six-minute loop the walkers' places have such
moments every few seconds (the near aisles leave the picture, the far ones
pass behind booths), the wanderers' and the visitors' places now and then,
when someone passes in front of them, and the visitor walking up the middle of
the aisle never. So after 20 s of scene time without such a moment, a frame
where the change shows at most 0.05 % of the picture (460 of 1280 x 720
pixels: someone far down the hall, or nearly hidden) will do, never more.
Counted for the fair's own people from every second of the loop, a change in a
walker's place then comes after 0.2 s (median; at most 17 s), in a wanderer's
after 12 s (at most 46 s) and in a visitor's after 25 s (at most a minute and
a half, for the one in the middle of the aisle). Restarting the scene (`R`)
and reloading the scene file make the changes at once: the whole picture
changes anyway. A character with a sign holds it up from the start.

The character wears its own textures and items. The known faces of
[face recognition](#known-faces-on-the-3d-people) are never put on it: it
has a face of its own. Selection mode works on it as on anyone (yoga poses,
items), and its tag shows the character's name, e.g. `approach1 (Ada)`.

While the demo runs it looks at the folder once a second:

- a character saved by the creator (or copied in) is picked up and joins the
  fair in the next free place, and the demo says so ("Ada joined the fair");
- a character saved again changes in place: new looks on the same 3D person,
  or, with another avatar, a new 3D person in the same place;
- a deleted character leaves the same way, and the place's own person comes
  back; or a character waiting for a place takes it (the place stays empty in
  between, its own person does not come back for a moment).

The status panel lists who is in the fair (`created: Ada, Ben in the fair`),
how many are on the way (made, or waiting for a moment nobody sees), who is
leaving and how many wait for a free place. When several join at once, or a
frame or two apart, there is one notice for them all ("Ada and Ben joined the
fair"). Making a character's 3D person (reading its images, about 30 ms each,
loading the avatar, putting its looks on) takes about 0.1 s of work, and 0.4
to 0.75 s when its behaviour is made again. That runs on a worker thread,
paced ([`pacing`](src/bifold_demo/pacing.py)): Python runs one thread at a
time, and a busy worker would take the main thread's turn for up to 5 ms
every time it lets go, so the worker rests three quarters of the time, and
more while the frames keep the main thread busy. Its work then takes 0.4 to
0.8 s for a character in a place like its avatar, and 2 to 4.5 s when the
behaviour is made again. Measured in the demo's own loop (the real fair at
1280 x 720 and 30 frames a second, with the draw calls left out: this machine
has only a software renderer, which spends 70 to 300 ms a frame on them; four
CPUs shared with other work), the main thread's work per frame took 14 to
17 ms (median) with nothing in the making, and up to 6 frames in a hundred
came late (over 33 ms); while one character was made, 15 to 20 ms, with up to
2 in a hundred late; while four copied in at once were made, 16 to 18 ms,
with 6 to 13 in a hundred late (at most 71 ms). Not paced, 30 to 43 in a
hundred came late. What drawing a newcomer needs goes to the GPU over the
frames before they come in, one texture (or its mipmaps) a frame, so the frame
they come in has none of it left to do: 5 to 10 ms a frame with this software
renderer, up to 30 ms when the machine is busy. While a change waits for its
moment, counting the pixels of the two people takes about 7 ms a frame here.
Before the first frame and after a scene reload nobody sees the fair yet: the
demo waits for its characters then, and the worker does not rest: 0.5 s for
four characters (two made again), 1.4 s for thirteen (eight made again), on
top of building the fair. Quitting stops the work at once.

A character the demo cannot show is skipped and the others go on:

- its avatar not prepared ("cannot put Zed in the fair: ..."): it is tried
  again every 10 s, so an avatar prepared meanwhile brings it in;
- its `character.json` not readable or not one (not JSON, a list, a field of
  the wrong kind, a look of the wrong kind): said once when that turns up (and
  again if it happens again later), and counted in the status panel. A
  character that could be read before stays as it was meanwhile: an editor or
  a sync tool writing the file in place leaves it half written for a moment;
- an image it cannot read (half copied, say): it is read again as soon as the
  file changes;
- an item its avatar cannot wear, or whose picture is not a file of the
  character's folder, and textures painted for parts its avatar does not have
  (left out; the console says which).

New looks that cannot be used leave the character as it was. A folder copied
by hand (`Ada copy/`) is reported, and the character's own folder counts.
Saving a character whose name goes to another one's folder (`Ada?` and `Ada`
both go to `Ada/`) is refused, and so is deleting it for the other; saving
again keeps the other files of the folder (an item's picture). Images of any
PNG kind are read (grey, a palette, 16 bits a channel), and larger than 1024
pixels they are scaled down, like the avatars' own textures; a head's texture
keeps up to 2048 (`created.max_texture`), the size of a high-resolution head.

**Privacy.** A created character can wear a real person's face, painted from
their photos: that is personal data. `created/` stays on this computer, git
ignores it, and the demo keeps no copy of it anywhere else. To remove a
character, delete it in the creator or delete its folder (`created/Ada/`): its
files are gone at once, and its 3D person leaves the fair at the next moment
nobody sees (press `R` to make that now).

```python
from bifold_demo.characters.created import CharacterBook, CreatedCharacter

book = CharacterBook("created")
book.save(CreatedCharacter("Ada", "Female_Adult_01", items={"hat": "navy"},
                           textures={"f001_body": outfit_rgba}))  # by primitive name
book.delete("Ada")
```

[`scenes/created_cast.py`](src/bifold_demo/scenes/created_cast.py) does the
casting (`CreatedCast`), [`characters/created.py`](src/bifold_demo/characters/created.py)
the saving and loading.
[docs/previews/created_characters.jpg](docs/previews/created_characters.jpg)
shows three created characters (plain colour outfits, a hat, sunglasses) in
the visitors' places, and [docs/previews/created_swaps.jpg](docs/previews/created_swaps.jpg)
how characters come in where nobody sees it.

## Generating 2D people

The 2D alternative, used when no 3D people are prepared: `bifold-people`
creates made-up people with an open text-to-image model and cuts them out as
flat pictures. For each person it:

1. writes a random description (age, gender, appearance, trade-fair outfit,
   lanyard or coffee cup, standing or walking),
2. generates a full-body photo on a plain studio background with a
   text-to-image model,
3. cuts the person out (YOLO segmentation, edges refined with GrabCut),
4. rejects bad images (no person, several people, feet or head cut off) and
   retries with another seed,
5. saves a transparent PNG and a JSON file (height, pose, prompt, seed) into
   `people/`.

```bash
pip install -e ".[people]"          # diffusers + transformers
bifold-people models                # the available models
bifold-people generate -n 30        # 30 people with the default model (mix of standing and walking)
bifold-people sheet                 # contact sheet of everyone: people/_contact_sheet.jpg
bifold-demo                         # the hall now uses them
```

The first `generate` downloads the model from Hugging Face. Generation happens
once, before the demo. Each image takes a few seconds to a minute depending on
the Mac, and the demo itself only loads the finished cut-outs.

| Preset (`--model`)      | Model                                                                   | Memory  | Licence |
|-------------------------|-------------------------------------------------------------------------|---------|---------|
| `flux2-klein` (default) | [FLUX.2 [klein] 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B), 4 steps | ~16 GB; a 24 GB+ Mac recommended | Apache-2.0 |
| `z-image-turbo`         | [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) 6B, 8 steps; very photorealistic | ~24 GB; a 32 GB+ Mac recommended | Apache-2.0 |
| `sd-turbo`              | [SD-Turbo](https://huggingface.co/stabilityai/sd-turbo), 2 steps; lower quality, for small Macs | ~4 GB | Stability AI Community License |

Any other Hugging Face text-to-image model works with `--model owner/name`
(plus `--steps`/`--size`). Other useful options are `--pose standing|walking`,
`--seed N` for reproducible people, and `--keep-raw` to keep the uncut images.

Why one model for the whole person, rather than a face GAN plus a body? Current
diffusion models generate the face, body, clothes and lighting together, so
everything matches. Stitching a GAN face onto a separately made body leaves
seams and mismatched light. At the distances in the hall, faces are small
anyway.

**Your own photos.** `bifold-people import photo1.jpg photo2.jpg` cuts the
largest person out of each photo (`--all` for everyone). People cut off at the
waist are used as booth staff, where the counter hides the missing legs. You
can also drop a transparent PNG straight into `people/`.

## Building your own scene

A scene is a Python file with a `build_scene()` function:

```python
# my_scene.py
from bifold_demo import Box, Plane, Quad, Scene, checkerboard, spin

def build_scene():
    scene = Scene()
    scene.add(Plane(size=20, uv_repeat=8, texture=checkerboard()), name="floor")
    scene.add(Quad(texture="photos/dog.jpg", height=2, position=(-1.5, 1, 0)), name="poster")
    scene.add(Box(size=1, texture="photos/cat.jpg", position=(1.5, 0.5, 0), rotation=spin(45)), name="cube")
    return scene
```

```bash
bifold-demo --scene my_scene.py
```

When you save the file, the running demo reloads it. Relative texture paths
are resolved against the folder you started `bifold-demo` from. You can also
start the demo from Python with `bifold_demo.run(scene, source="webcam", ...)`.
The [`examples/`](examples) folder has runnable scenes for every feature:

| Example                                            | Shows                                                          |
|----------------------------------------------------|----------------------------------------------------------------|
| [`minimal.py`](examples/minimal.py)                | the smallest scene (one picture)                               |
| [`animation.py`](examples/animation.py)            | keyframes, functions of time, keyframed textures, blinking     |
| [`physics.py`](examples/physics.py)                | gravity, bouncing off the floor and walls, spin                |
| [`textures.py`](examples/textures.py)              | files, colours, arrays, a live-drawn texture, image background |
| [`custom_geometry.py`](examples/custom_geometry.py)| your own meshes from vertex arrays, animated scale             |

### Coordinates

The Y axis points up, and the default camera looks towards −Z. Units are
arbitrary; "metres" works well with gravity `-9.81`. Rotations are Euler
angles `(x, y, z)` in degrees, applied in X, then Y, then Z order.

### Objects

| Class                                  | Geometry                                                                  |
|----------------------------------------|---------------------------------------------------------------------------|
| `Quad(height=1, size=None)`            | upright picture facing the camera; width follows the image's aspect ratio |
| `Box(size=1 or (w, h, d))`             | box; every face shows the whole texture                                   |
| `Plane(size=(w, d), uv_repeat=1)`      | horizontal floor or table; `uv_repeat` tiles the texture                  |
| `Sphere(radius=0.5)`                   | sphere; the texture wraps around it like a world map                      |
| `Cylinder(radius=0.5, height=1)`       | upright cylinder; the texture wraps around the side                       |
| `SceneObject(Mesh(...))`               | any triangle mesh you build yourself                                      |

All objects take the same keyword arguments:

| Argument      | Meaning                                                                           |
|---------------|-----------------------------------------------------------------------------------|
| `texture`     | what to paint the object with (see below)                                         |
| `position`    | `(x, y, z)` of the centre                                                         |
| `rotation`    | `(x, y, z)` in degrees                                                            |
| `scale`       | a number or `(sx, sy, sz)`                                                        |
| `color`       | `(r, g, b)` tint multiplied with the texture                                      |
| `visible`     | `True` / `False`                                                                  |
| `lit`         | `False` shows the texture exactly as it is, with no shading                       |
| `face_camera` | keep turning towards the camera (a billboard)                                     |
| `body`        | a `Body` for simple physics                                                       |
| `on_update`   | `fn(obj, t, dt)` called every frame                                               |

### Animation: properties are values or functions of time

Each of `position`, `rotation`, `scale`, `color`, `texture` and `visible` (and
the camera's `position`, `target` and `fov`) accepts either a constant or a
function of the simulation time `t` in seconds:

```python
from bifold_demo import Keyframes, oscillate, orbit, spin, linear

obj.position = lambda t: (math.sin(t), 1, 0)                    # any function
obj.position = Keyframes({0: (0, 0, 0), 2: (3, 0, 0), 4: (0, 0, 0)}, loop=True, easing="smooth")
obj.rotation = spin(90)                                          # 90°/s around Y
obj.position = orbit(center=(0, 1, 0), radius=3, period=10)
obj.position = oscillate(center=(0, 1, 0), amplitude=(2, 0, 0), period=4)
obj.texture  = Keyframes({0: "cat.jpg", 3: "dog.jpg", 6: "cat.jpg"}, loop=True)  # switches
scene.camera.position = orbit(radius=6, height=2, period=20)     # a moving camera
```

For logic that doesn't fit a single property, register a callback:

```python
@scene.on_update
def every_frame(scene, t, dt):
    if t > 5:
        scene["cube"].visible = False
```

### Physics (deliberately simple)

```python
from bifold_demo import Body

ball = Sphere(radius=0.3, position=(0, 3, 0),
              body=Body(velocity=(1, 0, 0), angular_velocity=(0, 0, -90),
                        gravity=-9.81, bounce=0.8, floor=0.0,
                        bounds=((-3, 0, -2), (3, 5, 2))))   # optional walls
```

A body integrates velocity, spin and gravity. It bounces off a floor height
and/or an axis-aligned box of walls, using the object's rotated bounding box.
Objects do not collide with each other. While an object has a body, the body
owns its position and rotation. They start from the object's `position` and
`rotation`, and `R` (or `scene.reset()`) puts them back there.

### Textures: the simple upload API

Anything below works wherever a texture is expected: in the constructor, with
`obj.texture = ...`, or with `scene.set_texture(name, ...)`, at any time,
including while the demo runs.

| Value                          | Example                                                      |
|--------------------------------|--------------------------------------------------------------|
| image file                     | `"cat.jpg"`, `Path("~/Pictures/dog.png")` (PNG alpha = cut-out) |
| video file                     | `"clip.mp4"` (plays in simulation time, loops)               |
| NumPy array (OpenCV BGR/BGRA)  | `cv2.imread("cat.jpg")`, or any generated image              |
| PIL image                      | `Image.open("cat.jpg")`                                      |
| colour                         | `(255, 128, 0)` or `"#ff8000"`                               |
| `Texture`                      | `Texture.from_array(rgb, order="rgb")`; call `.update(image)` to redraw it |
| a function of time             | `lambda t: "a.jpg" if t < 5 else "b.jpg"`                     |

```python
scene["poster"].texture = "photos/dog.jpg"      # by object name
scene.set_texture("cube", cv2.imread("cat.jpg"))
```

Image files are watched: if you overwrite a file on disk, every object that
uses it updates.

**Uploads folder.** While the demo runs, drop a file into `uploads/` that is
named like an object, and it becomes that object's texture. For example,
`uploads/poster.jpg` goes onto the object named `poster`. Replace the file to
change the texture again; delete it to restore the original. The trade-fair
scene's object names are listed [above](#the-trade-fair-scene). Use
`--uploads OTHER_DIR` to pick another folder, or `--uploads ''` to turn this
off.

### Scene settings

```python
Scene(background=(28, 30, 36),          # a colour, or an image/video path (fills the frame)
      ambient=0.45,                     # brightness in shadow, 0..1
      light_direction=(-0.4, -1, -0.5), # where the light shines towards
      light_color=(255, 255, 255),
      camera=Camera(position=(0, 1.6, 6), target=(0, 1, 0), fov=55),
      fog=Fog(start=20, end=100, color=(200, 200, 200), strength=0.5))  # distance haze, or None
```

The renderer shades in linear light (see "High-resolution heads"): colours
are sRGB colours as usual, `ambient` is the brightness in shadow as it looks
on screen, and fog mixes light, so a given strength veils dark distant
surfaces more than it did when sRGB values were mixed.

## Detection options

```bash
bifold-demo --model yolo26s.pt         # bigger, more accurate model (n < s < m < l < x)
bifold-demo --model yolo11n.pt         # previous YOLO generation
bifold-demo --classes person car dog   # only these COCO classes
bifold-demo --conf 0.5 --imgsz 960     # stricter threshold, higher input resolution
bifold-demo --device cpu               # force the CPU (default: auto = mps > cuda > cpu)
bifold-demo --sync                     # wait for detections on every frame (see below)
```

YOLO is a good fit here: the nano models run in real time on Apple silicon
and recognise the 80 everyday COCO classes (people, vehicles, animals,
furniture, ...). Detection runs on a background thread that always takes the
newest frame, so the video keeps its full frame rate. If the model is slower
than the video, the boxes trail fast-moving objects slightly. With `--sync`,
each frame waits for its own detections instead: the boxes never lag, and the
frame rate follows the model. Any Ultralytics-compatible weights work with
`--model`, including your own fine-tuned `.pt` files.

## Recording and headless runs

```bash
bifold-demo --record recordings/demo.mp4                   # record what you see
bifold-demo --headless --frames 300 --record out.mp4       # no window; 10 s at 30 fps
```

Headless mode advances the scene by exactly `1/fps` per frame and analyses
every frame, so a recording plays back at the right speed.

## Project layout

```
src/bifold_demo/
  sources/     webcam.py, scene_source.py, clock.py, SourceSwitcher   ← module 1: image stream
  scene/       scene.py (objects, camera), geometry.py, textures.py,
               animation.py, physics.py, renderer.py (ModernGL),
               frustum.py (culling for windows)                         ← the 3D scene
  detection/   yolo.py (YoloDetector), worker.py (AsyncDetector)      ← module 2: recognition
  display/     overlay.py (boxes, HUD), window.py, sensor_view.py (the windows, W)  ← module 3: on screen
  characters/  model.py (skeleton, clips, posing), fbx.py + fbx_convert.py (FBX reader),
               character.py (skinned scene object), behaviour.py, crowd.py,
               yoga.py (yoga pose clips), yoga_clips.py (built in a worker process, cached),
               performer.py (a pose in the middle of a behaviour),
               accessories.py (hats, glasses, signs, t-shirts),
               faces.py (funny faces with the face bones),
               face_photo.py (a photo's face painted on a 3D person),
               uvbake.py (what each texel of a texture is on the body),
               outfit/ (garment regions, cloth coordinates, fabrics, prompts,
                        presets.json, the compositor: new clothes on an avatar;
                        fabric.py: fabrics from a description, with seamless.py
                        (seam score, repair, light) and tiling.py (rolling, padding)),
               headtex/ (head textures from photos: landmarks on the mesh, photos registered,
                         projected, fused, de-lit; the skin tone; completed, the body's
                         skin and the hair matched, checked),
               created.py (characters from the character creator, saved),
               rocketbox.py + library.py + cli.py (bifold-characters),
               rocketbox_catalog.json (avatar labels)                   ← 3D people
  recognition/ engine.py (a profile's models together), gallery.py (known people),
               recognizer.py (faces over frames), cast.py (known faces on 3D people),
               registry.py (face models and profiles), runtime.py (ONNX Runtime),
               detect.py (SCRFD, YuNet), align.py (the five-point template),
               embed.py (embedders), quality.py (a quality record per face),
               checks.py (model checks), models.py, cli.py (bifold-faces) ← face recognition
  sensor/      geometry.py (a declared high-resolution sensor, its windows and
               pixel maps), requests.py (requests, head regions, the scheduler),
               service.py (windows with the frames, the truth),
               camera_model.py (blur, noise, mosaic, tone, JPEG; presets),
               cli.py (bifold-scene windows)                            ← the virtual high-resolution sensor
  people/      prompts.py, generator.py (text-to-image), cutout.py,
               library.py, defaults.py, cli.py (bifold-people)          ← 2D people generator
  app.py       main loop and command line; loader.py loads and reloads scene files;
               selection.py (selection mode: people, their yoga poses and accessories);
               creator.py (the character creator; display/creator_view.py draws it);
               pacing.py (background work that leaves the main thread its frames)
  scenes/      fair.py (default trade-fair hall), created_cast.py (created characters in it),
               fair3d.py (static 3D poses), basic.py
examples/      example scenes
uploads/       drop textures here while the demo runs
people/        2D cut-outs (used by the fair scene when no 3D people are prepared)
faces/         photos of the people the face recognition knows (a folder per person)
created/       characters made in the character creator (a folder per character; kept out of git)
fabrics/       the cache of fabric swatches made from descriptions
tests/         pytest suite
```

## Development

```bash
pip install -e ".[dev]"
pytest
```

The tests that need an OpenGL context are skipped when none is available. On
a headless Linux machine, run them with `xvfb-run -a pytest`. The tests marked
`yolo` download the nano weights on first use. The tests never use your
prepared characters (`BIFOLD_CHARACTERS` is redirected to an empty folder).
Some tests also check real data when you point them at it, and skip otherwise:

- `BIFOLD_ROCKETBOX_TEST_DIR`: a folder with the Rocketbox `Avatars/` and
  `Animations/` folders (any name), for the FBX and asset tests.
- `BIFOLD_CHARACTERS_TEST_DIR`: a prepared character library, for the tests
  of real walk clips, yoga poses, faces and outfits (the outfit tests keep
  their labels in a temporary folder, not in the library).
- `BIFOLD_ROCKETBOX_FILES`: a fresh full file listing of the Rocketbox
  repository, instead of the committed `tests/rocketbox_files.txt`.
- `BIFOLD_FACE_PHOTOS_TEST_DIR`: a folder of portrait photos, for the test that
  registers real photos to an avatar's head (with `BIFOLD_CHARACTERS_TEST_DIR`
  and MediaPipe installed). The MediaPipe tests download the models into the
  face models folder unless `BIFOLD_NO_DOWNLOADS` is set.

The fabric tests never load a text-to-image model: they run stand-in
pipelines with the latent layouts of the real ones, and tiny randomly
initialised diffusers models that they build themselves (no downloads; skipped
without diffusers). Their swatch cache is a temporary folder.

## Troubleshooting

- **"cannot open camera"**: grant camera access to your terminal (or IDE) under
  System Settings → Privacy & Security → Camera, then restart the terminal. If
  you have more than one camera, try `--camera 1`.
- **The window is very small or very large**: frames are shown 1:1. Use
  `--size 1920x1080`, or press `F` for fullscreen.
- **The detector falls back to the CPU**: the MPS check failed. Make sure you
  use an arm64 Python (`python -c "import platform; print(platform.machine())"`
  should print `arm64`), not one running under Rosetta.
- **OpenGL errors on startup**: the scene renderer needs OpenGL 3.3 or newer.
  Current Macs provide OpenGL 4.1. On Linux without a display, use `xvfb-run`.
- **The hall shows flat cut-out people**: no 3D people are prepared, or the
  demo looks in another folder. Run `bifold-characters fetch --set conference`,
  then check `bifold-characters list`. If you fetched with `--dest DIR`, also
  `export BIFOLD_CHARACTERS=DIR` before `bifold-demo`. If a cache file is
  damaged, the terminal names it and the command that repairs it.
- **Too slow with many people**: fewer people (shorten the lists at the top of
  `scenes/fair.py`), a smaller window (`--size 960x540`), or `--imgsz 480`
  for YOLO. Every person is ~7–9k triangles and costs ~0.3 ms of Python per
  frame.
- **`bifold-people generate` runs out of memory or is very slow**: use a
  smaller model (`--model sd-turbo`) or a smaller image (`--size 576x1024`).
  Close other apps; the model needs most of the Mac's memory.
- **"cannot load … gated"**: open the model's Hugging Face page, accept its
  terms, then run `hf auth login`.

## Licensing note

The 3D people are Microsoft Rocketbox avatars and animations (MIT licence).
They are downloaded by `bifold-characters` and are not part of this
repository. Ultralytics YOLO is licensed under AGPL-3.0. That is fine for running demos.
Embedding it in a closed-source product requires an Ultralytics Enterprise
licence. The sample photos used for the default people (`bus.jpg`, `zidane.jpg`)
ship with the `ultralytics` package; this repository does not contain them. The
image models have their own licences (table above). FLUX.2 [klein] 4B and
Z-Image-Turbo are Apache-2.0; SD-Turbo's licence restricts commercial use.
