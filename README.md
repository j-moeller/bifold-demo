# bifold-demo

A real-time object-detection demo for a MacBook. A video stream comes either
from the **webcam** or from a **customizable 3D scene** rendered on the GPU.
Every frame goes through a **YOLO** detector, and the video is shown on screen
at its normal resolution with the detections drawn on top.

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

bifold-demo                      # the built-in 3D demo scene
bifold-demo --source webcam      # start with the webcam
```

The first run downloads the YOLO weights (`yolo26n.pt`, about 5 MB) into
`models/`. The first time you use the webcam, macOS asks for camera access
for your terminal app. Allow it, then run the command again.

### Controls

| Key       | Action                                                    |
|-----------|-----------------------------------------------------------|
| `S`       | switch the source: scene ↔ webcam                         |
| `D`       | turn detection on or off                                  |
| `Space`   | pause or resume the scene's simulation time               |
| `R`       | restart the scene (time 0, physics reset)                 |
| `[` / `]` | lower / raise the confidence threshold                    |
| `F`       | toggle fullscreen                                         |
| `P`       | save a screenshot to `screenshots/`                       |
| `M`       | mirror the webcam on or off                               |
| `H`       | hide or show the status panel                             |
| `Q`/`Esc` | quit                                                      |

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
change the texture again; delete it to restore the original. The built-in
scene's objects are `poster`, `cube`, `ball` and `floor`. Use
`--uploads OTHER_DIR` to pick another folder, or `--uploads ''` to turn this
off.

### Scene settings

```python
Scene(background=(28, 30, 36),          # a colour, or an image/video path (fills the frame)
      ambient=0.45,                     # brightness in shadow, 0..1
      light_direction=(-0.4, -1, -0.5), # where the light shines towards
      light_color=(255, 255, 255),
      camera=Camera(position=(0, 1.6, 6), target=(0, 1, 0), fov=55))
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
  app.py       main loop and command line; loader.py loads and reloads scene files
  scenes/demo.py   the built-in demo scene
examples/      example scenes
uploads/       drop textures here while the demo runs
tests/         pytest suite
```

## Development

```bash
pip install -e ".[dev]"
pytest
```

The tests that need an OpenGL context are skipped when none is available. On
a headless Linux machine, run them with `xvfb-run -a pytest`. The tests marked
`yolo` download the nano weights on first use.

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

## Licensing note

Ultralytics YOLO is licensed under AGPL-3.0. That is fine for running demos.
Embedding it in a closed-source product requires an Ultralytics Enterprise
licence. The sample photos used by the default scene (`bus.jpg`, `zidane.jpg`)
ship with the `ultralytics` package; this repository does not contain them.
