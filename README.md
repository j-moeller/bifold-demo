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
bifold-characters fetch --set conference   # once: download + convert the 3D people (~2.3 GB)

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
| `Space`   | pause or resume the scene's simulation time               |
| `R`       | restart the scene (time 0, physics reset)                 |
| `[` / `]` | lower / raise the confidence threshold                    |
| `Y`/`Tab` | select a person in the 3D scene: yoga poses, accessories (see below) |
| `F`       | toggle fullscreen                                         |
| `P`       | save a screenshot to `screenshots/`                       |
| `M`       | mirror the webcam on or off                               |
| `H`       | hide or show the status panel                             |
| `Q`/`Esc` | quit (in selection mode `Esc` ends that first)            |

**Selecting people for yoga and accessories.** `Y` (or `Tab`) selects the most
prominent of the 3D people in the picture and marks them with a tag. In this
selection mode the keys are:

| Key                     | Action                                                  |
|-------------------------|---------------------------------------------------------|
| `Tab` / `Shift`+`Tab`   | next person to the right / left (also `→` / `←`)        |
| `↑` / `↓` (or `I`)      | the next menu: the yoga poses (eight to a menu), then the items; `I` goes to the items and back |
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
| `0` / `Backspace`       | take everything off                                     |

The bottom panel shows the keys of the current menu.

People drawn smaller than 8 % of the picture height (far away) cannot be
selected. When the selected person walks out of the picture, is hidden or is
removed from the scene, the person nearest to where they were is selected
instead. A person asked for a pose stops what they are doing and walks (up to
4 m) to the nearest spot where:

- the camera sees the whole pose, from the side it is best seen from;
- nobody nearer the camera and no counter hides it;
- the mat is clear of the other people and of the furniture;
- the whole pose is on one floor level (not half on a booth platform).

When no spot within reach has all of that, they take the one the camera sees
best. There they do the pose, then walk back and carry on exactly where they left
off. A new pose asked for during one follows it without walking back in
between. Several people can do poses at once. People who walk around are
later on their way afterwards, so the fair's timing that keeps walkers from
meeting no longer covers them (`R` restarts everything).

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
| 3 in the audience | stand in a half circle 3.9 m in front of the camera (our exhibit), facing it and watching it, always there | `AUDIENCE` |
| 3 visitors | walk down the aisle towards the camera, join the audience, watch for 15–30 s, turn around, walk back | `AUDIENCE`, `VISITORS_FROM` |
| 7 walkers | walk across the hall along the cross aisles, both directions; on the near aisle they come and go at the side walls, out of view, on the far aisles they turn round near the walls | `CROSSINGS` |
| 3 wanderers | stroll between random points in an area, pausing | `WANDER_AREAS` |
| 3 groups (7 people) | stand and chat, alternating talking and listening, with small shifts | `CHAT_GROUPS` |
| 8 exhibitors | stand on their booth platforms behind the counters | `booth_layout()` |
| 4 visitors | talk to exhibitors at the left-hand booths | |
| 1 speaker | talks on the stage | |

The camera is our own exhibit, so at any moment at least three people watch it:
the audience always, and the visitors while they stand with them. Everyone
else never looks at the camera: the behaviours guarantee that their faces
point at least 35° away from it. While someone walks towards the camera or
stands close by, the head stays turned aside or lowered.
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

## 3D people

The [`characters`](src/bifold_demo/characters) package turns Rocketbox avatars
into animated people. It has five parts:

1. **Assets** (`rocketbox.py`, `library.py`, `bifold-characters`).
   `bifold-characters fetch` downloads exactly the files it needs from GitHub:
   per avatar its FBX file and colour/opacity textures, plus the animation
   clips. It then converts them with a pure-Python FBX reader (`fbx.py`,
   `fbx_convert.py`), so no Blender or FBX SDK is needed. The result is a fast
   cache: an avatar loads in a few milliseconds.
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
bifold-characters fetch --set small         # 12 diverse avatars, a quick start (~0.5 GB)
bifold-characters fetch --set conference    # 58 avatars that suit a fair, incl. children (~2.3 GB)
bifold-characters fetch --set all           # all 115, incl. uniforms (~5 GB)
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
               animation.py, physics.py, renderer.py (ModernGL)        ← the 3D scene
  detection/   yolo.py (YoloDetector), worker.py (AsyncDetector)      ← module 2: recognition
  display/     overlay.py (boxes, HUD), window.py                      ← module 3: on screen
  characters/  model.py (skeleton, clips, posing), fbx.py + fbx_convert.py (FBX reader),
               character.py (skinned scene object), behaviour.py, crowd.py,
               yoga.py (yoga pose clips), yoga_clips.py (built in a worker process, cached),
               performer.py (a pose in the middle of a behaviour),
               accessories.py (hats, glasses, signs, t-shirts),
               rocketbox.py + library.py + cli.py (bifold-characters),
               rocketbox_catalog.json (avatar labels)                   ← 3D people
  people/      prompts.py, generator.py (text-to-image), cutout.py,
               library.py, defaults.py, cli.py (bifold-people)          ← 2D people generator
  app.py       main loop and command line; loader.py loads and reloads scene files;
               selection.py (selection mode: people, their yoga poses and accessories)
  scenes/      fair.py (default trade-fair hall), fair3d.py (static 3D poses), basic.py
examples/      example scenes
uploads/       drop textures here while the demo runs
people/        2D cut-outs (used by the fair scene when no 3D people are prepared)
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
  of real walk clips.
- `BIFOLD_ROCKETBOX_FILES`: a fresh full file listing of the Rocketbox
  repository, instead of the committed `tests/rocketbox_files.txt`.

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
