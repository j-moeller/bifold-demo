# A3: landmarks on the head mesh, photos registered (head textures, stages 0 and 1)

**State when paused.** Implemented in one commit, `04faecd` "Head textures from photos: landmarks on the head mesh,
photos registered", in the worktree `.claude/worktrees/wf_9c8d2dc1-3c2-1`. The review had started and was
interrupted; it starts again. The commit is not on the branch yet.

## What it builds

The new package `src/bifold_demo/characters/headtex/` holds the first two of the eight stages agreed in
[the literature review](../research/face_head_texture.md) (section 3). Its docstring explains all eight stages and
where this step fits.

### Stage 0: the 478 face landmarks on each head model (`landmarks.py`)

This runs once per head model:

- The avatar is rendered unlit in its bind pose from five views: front, yaw ±25 degrees, pitch ±12 degrees.
- MediaPipe's Face Landmarker runs on each render. A sanity check rejects faces that are not the avatar's.
- Each landmark is lifted onto the head mesh by casting a ray through its pixel. Views that see the surface at a
  grazing angle don't count.
- The views are combined per landmark: the median first, then the mean of the views within 4 mm, snapped back onto
  the surface.
- The result per head:
  - points (478, 3);
  - their triangle and barycentric coordinates;
  - a reliability per landmark;
  - the method used and the number of views.
- **Fallback:** if MediaPipe finds no face, MediaPipe's canonical face model is fitted instead.
  - It ships inside the `.task` bundle, and a small parser reads it without the protobuf package.
  - The fit uses the eye and mouth-corner bones, plus the nose tip, chin and ears found on the mesh.
- **Cache:** results are kept under `<characters folder>/headtex/`, keyed by the head's geometry, or in
  `$BIFOLD_HEADTEX_CACHE`. If the folder is read-only they are kept in memory.

Measured on all 20 prepared avatars:

- every one takes the MediaPipe path, with the face found in 5 of 5 renders (glasses, headscarf and keffiyeh
  included);
- mean reliability is 0.72-0.83;
- the forced fallback lands 6.9 mm from MediaPipe's landmarks on average (about 5 mm at the eyes, mouth and nose,
  17 mm on the face outline);
- it takes about 2 s per avatar.

### Stage 1: a `View` per photo (`register.py`)

- **Camera:** RANSAC PnP against the stage-0 points. It is then refitted without the landmarks on the side of the face
  the photo does not show.
- **Focal length:**
  - It comes from EXIF when the photo has it. Otherwise it is searched, with a preference for a 50-degree view.
  - The preference is needed because MediaPipe's landmarks give poor, biased perspective cues. On fake photos it cut
    the 90th-percentile angle error from 11 to 4 degrees.
  - Photos of one session share the focal length of the most frontal photo.
- **Warp:** a thin-plate-spline warp moves the person's features onto the avatar's projected landmarks. It is exact
  at its control points and fades to no movement outside the face.
- **Photos without a face** need a view name (`left`, `right`, `back`):
  - The camera starts there, at the frontal photo's distance from the head's centre.
  - It is then refined by fitting the outline of the avatar's head and hair to the segmenter's head mask.
  - Below an overlap (IoU) of 0.8 the photo is reported unusable, with the reason. Angles the outline cannot pin down
    get a warning.
- **Also in each View:** the segmentation masks, a quality score and warnings. Small faces in big photos are found by
  retrying on crops.

Measured on renders from known cameras:

- within 3 degrees and 3 px (median) up to 30 degrees of yaw;
- about 6 degrees and 6 px at 60;
- outline fits: profiles within 6 degrees of turn; back views within 8 degrees of turn but up to 13 of nod.

On the 16 real test photos (kept in scratch, never committed):

- plausible poses, with reprojection errors of 1-3 % of the face size;
- the face takes about 0.1 s and an outline fit 1-3 s.

### Also in the commit

- `src/bifold_demo/downloads.py`, a checksum-checked download helper, which `recognition/models.py` now uses too.
- New CLI commands: `bifold-faces landmarks`, `bifold-faces register PHOTO[:VIEW] ... --avatar NAME [--segment]
  [--out DIR]` and `bifold-faces fetch --headtex`.
- MediaPipe is optional. Install it with `pip install -e ".[headtex]"`, then `pip install --no-deps "mediapipe>=1.0"`;
  Linux also needs libEGL and libGLESv2.
  - `--no-deps` is needed because MediaPipe pulls in `opencv-contrib-python`, which clashes with `opencv-python`.
  - Without MediaPipe the demo runs as before, and these commands print the install command.
- A README section, and the preview `docs/previews/headtex_register.jpg` with its entry (renders and the Ultralytics
  sample photo only).

## API (what A4-A7 build on)

`bifold_demo.characters.headtex` exports `HeadLandmarks`, `head_landmarks`, `PhotoCamera`, `View`, `register_photo`,
`register_photos` and `draw_registration`.

### Stage 0

```python
head_landmarks(model, landmarker=None, *, cache_dir=None | False, render=None, canonical=None,
               method="auto" | "mediapipe" | "canonical", refresh=False) -> HeadLandmarks
```

- `landmarker` is anything with `.detect(bgr) -> list[FaceMarks]`.
- `render(model, cameras) -> images`; by default the offscreen Renderer.

`HeadLandmarks`:

- fields:
  - `points (478, 3)` in bind-pose character space;
  - `triangle (478,)` and `barycentric (478, 3)` on the head primitive `primitive`;
  - `reliability (478,)`, from 0 to 1;
  - `method`, `views`, `key`.
- `.posed(model, palette) -> (478, 3)`: the landmarks moved by the skinning matrices.

Helpers:

- `face_frame(model)`: eyes, mouth, nose, chin, ears, `.centre`, `.pivot` (between the ears) and `.eye_distance`;
- `views(frame)`, `render_views(model, cameras, renderer)`;
- `lift(...)`, `combine(...)`, `fit_canonical(...)`, `cache_folder()`.

### Stage 1

```python
register_photo(photo, head, model, *, view=None | "front" | "left" | "right" | "back", landmarker=None,
               segmenter=None, masks=None, focal=None, reference=None, faces=None) -> View
register_photos(photos, head, model, *, views=None, focals=None, landmarker=None, segmenter=None) -> list[View]
```

`register_photos` shares the focal length and distance of the most frontal photo.

`View` fields:

- `image`, `name`, `usable`, `reason`, `method` (`landmarks` or `silhouette`);
- `camera`: a `PhotoCamera` (K, R, t, size), with `p_cam = R p + t`, OpenCV axes, character space in bind pose;
- `marks` (FaceMarks), `projected (478, 2)`, `inliers`, `error` (px);
- `warp`: a ThinPlateWarp, the backward map from avatar-projected to photo pixels;
- `masks`: `background`, `hair`, `body`, `face`, `clothes` and `others`, each (H, W) float32;
- `quality` (0-1), `warnings`, `focal_known`, `iou`, `source`.

`View` methods and properties:

- `.angles` (yaw, pitch, roll in degrees);
- `.photo_pixels(points3d) -> (N, 2)`: the photo pixel to read for any head point;
- `.warped()`, `.summary()`.

Angle conventions:

- yaw > 0: the head is turned to its own left (90 shows its right side);
- pitch > 0: looking down;
- roll > 0: leaning towards the left shoulder.

Building blocks:

- `solve_camera(img_pts, obj_pts, size, focal=None, weights=None) -> CameraFit | None`;
- `face_warp(...)`;
- `facing(head, model, camera)`: a visibility cosine per landmark;
- `fit_silhouette(...) -> OutlineFit`;
- `Outline.of(model, primitive).mask(camera, shape)`;
- `exif_focal(path, size)`.

### Other modules

- `camera.PhotoCamera`:
  - constructors `around(target, distance, yaw, pitch, roll, fov | focal, size)` and `looking_at(...)`;
  - methods `project`, `rays`, `depth`, `angles`, `scaled(f)`, `head_rotation()`, `head_angles()`;
  - `scene_camera()`: a renderer Camera that shows the same view.
- `warp.ThinPlateWarp(source, target, smoothing=0, falloff=(centre, inner, outer))`: callable; also `maps(size)`,
  `apply(img)`, `jacobian_determinant` and `bending_energy`.
- `detect`:
  - `Landmarker(models=None, download=True, max_faces=1)`, with `.detect(bgr) -> [FaceMarks]` (points (478, 2) px,
    depth, a 4x4 matrix, blendshapes);
  - `Segmenter().segment(bgr) -> dict`;
  - `mediapipe()`, which raises `MediaPipeMissing` (an ImportError) with the install command;
  - the `FACE_LANDMARKER` and `SELFIE_MULTICLASS` ModelFiles (sha256 and size), and `model_path()`.
- `canonical`: landmark index groups (`FEATURES`, `FACE_OVAL`, `RIGHT_IRIS`, ...) and `read_canonical_face(task_path)`.
- `raycast`: `intersect`, `closest`, `point_on`.
- `bifold_demo.downloads`: `ModelFile(name, url, sha256, size)`, `ensure(model, folder, download, progress,
  fetch_hint, tag)`, `sha256(path)`.

## Checks the builder reports (to be verified by the review)

- **Full suite:** 475 passed, 9 skipped in 5:19, under xvfb with `OPENBLAS_NUM_THREADS=1`, the real avatars, a private
  face-model folder and the test photos. After a last tidy-up, only the related files were run again (54 passed).
- **`tests/test_headtex.py`** (23 synthetic tests, about 13 s; no MediaPipe, GL or avatars; downloads blocked):
  - camera angle round trips and sign conventions;
  - PnP recovers known cameras within 0.5 degrees, with outliers; with the focal length searched, within 15 % and 1.5
    degrees;
  - the warp is exact at its control points, keeps a positive Jacobian, is the identity outside its ring, and
    reproduces affine maps;
  - stage 0 on a synthetic sphere head with a fake renderer and landmarker: median error under 0.5 mm;
  - registration with a fake landmarker, the outline fit on a synthetic profile, and the shared session focal;
  - parsing the canonical face (packed and unpacked protobuf, and the zip bundle);
  - checksum-checked downloads via `file://`;
  - the error when MediaPipe is missing.
- **`tests/test_headtex_avatars.py`** (17 tests, about 20 s; skipped without avatars, MediaPipe or GL):
  - stage 0 on four avatars: the iris within 3 mm of the eye bones, eye corners within 6 mm, mouth corners within
    8 mm, the nose tip within 10 mm;
  - the fallback's mean error under 12 mm;
  - fake photos at yaw 0, ±30 and ±60 and pitch ±15: within 3 degrees (7 at 60), one more when the focal length is
    searched;
  - side and back photos placed by their outline;
  - the real test photos: errors under 4 % of the face size.
- **Lint:** the same ruff findings as before.

## Open issues the builder admits

- **Beyond about 65 degrees of yaw**, MediaPipe still returns a face, but the pose can be 10-15 degrees off, and the
  view only gets a warning.
  - A face that contradicts the named view (more than 70 degrees off, or a roll over 45) is ignored in favour of the
    outline fit.
  - Combining PnP with the outline fit for named side photos is not done.
- **Without an EXIF focal length**, turned or off-centre faces can be 5-10 degrees off when the real lens is far from
  the 50-degree preference.
- **The outline fit** is checked only on renders with the avatar's own hair, the best case.
  - It fixes the nod of back views only weakly (up to 13 degrees).
  - The segmenter took one child avatar's ponytail for clothes from behind, so those photos were rejected.
  - No real side or back photos were available.
- **One segmentation fault**, in the first run of the MediaPipe avatar tests under xvfb, where MediaPipe and moderngl
  share the process. It did not recur in about 8 later runs or in the full suite.
- **MediaPipe varies slightly between runs**, so the side-view test tolerances leave room.
- **Not run on a Mac.** The install recommendation is untested on macOS.
- **For stage 3 (A4):**
  - `View.photo_pixels` gives the photo pixel for any bind-pose head point, but visibility (depth testing) is left to
    stage 3;
  - `facing()` gives a per-landmark visibility cosine.
- **On this machine**, OpenBLAS thread thrashing made small solves about 1000 times slower; everything ran with
  `OPENBLAS_NUM_THREADS=1`. No code change is needed for it.
- **A slip, corrected:** a `headtex/` folder written by an early CLI run into the shared avatar cache was removed at
  once. Since then the tests use `cache_dir=False` and the CLI uses `BIFOLD_HEADTEX_CACHE`.
