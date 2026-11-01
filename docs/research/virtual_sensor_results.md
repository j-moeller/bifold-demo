# The virtual high-resolution sensor: measured results

Step S1 of [face_recognition_focus.md](face_recognition_focus.md): windows on the heads of the people in the
picture, rendered through an off-axis part of the camera's frustum as exact crops of a declared high-resolution
sensor (7680x4320 for the 1280x720 view), culled, delivered with the frames, and passed through a camera model. The
design follows [high_res_faces.md](high_res_faces.md) (sections 2.1 and 3.4); the code is `src/bifold_demo/sensor/`,
`scene/frustum.py` and `Renderer.render_window` (README: "The virtual high-resolution sensor").

Measured on 1 November 2026 in the development sandbox: a 4-core CPU shared with other jobs, Mesa 25.2.8 llvmpipe
(OpenGL 4.5), the 20 prepared Rocketbox avatars (1024-pixel textures). Timings on this machine vary by a factor of
two between runs (the 1280x720 view took 220-470 ms in different runs); the ratios are steadier than the absolute
numbers. A GPU (the user's Mac) is much faster at the large windows, which are fill-rate bound here. The scratch
scripts are not in the repository; `bifold-scene windows --faces` reproduces the face measurements on any scene and
moment.

## In short

- **Exact crops.** Windows equal the same pixels cut from a render of the whole sensor: 98.7-99.9 % of the pixels
  identical, 52.9-82.0 dB PSNR, at most 0.13 % of the pixels more than 8 levels off (silhouettes, where MSAA samples
  fall a little differently). The whole 8K frame takes 6.3 s here; a head window 10-400 ms.
- **The faces get the pixels.** With the 8K sensor a frontal face has 6 times the view's eye distance: 115 px at 3 m,
  75 at 4 m, 46 at 6 m, 26 at 10 m, 13 at 20 m (19, 12.5, 7.7, 4.3 and 2.2 px in the view). YuNet found **36 of 36**
  frontal faces at 2-30 m in the 8K windows against **20 of 36** in the view (none beyond 12 m there).
- **Culling** leaves out all but 1-13 of the fair's 36 people and 2-38 of its 124 objects, changes no pixel, costs
  about 1 ms, and makes windows up to 200 px about 2.2 times faster (1.2-1.4 times for larger, fill-bound windows).
- **The head square** of the review's starting values (0.12, 0.30, 0.30) holds the whole face in every one of 91
  YOLO boxes of the fair, the face a quarter of its side; (0.10, 0.20, 0.30) holds it in 96 % with 2.2 times fewer
  pixels.

## 1. Agreement with full-resolution crops

A window at level `k` must equal the same rectangle cut from a render of the whole frame at `1/k` of the sensor's
resolution: the same camera, MSAA (4 samples), fog and texture filtering.

| Scene | Full renders | Windows | Identical pixels | PSNR | Pixels off by more than 8 |
|---|---|---|---|---|---|
| Test scene (textured boxes, sphere, two skinned people with blob shadows, fog; `tests/test_sensor.py`) | 4x and 6x the 320x180 view | 11, 16-420 px, levels 1-5 | 99.9-100 % (4 windows all) | 64.7 dB to identical | 0-0.004 % |
| The fair at 6 s, 36 people | 7680x4320 (level 1) | 25 head windows, 64-1056 px | 99.20-99.93 % | 53.4-75.8 dB (median 61.4) | at most 0.098 % |
| The fair at 6 s | 3840x2160 (level 2) | the same 25, 33-528 px | 98.72-99.90 % | 52.9-82.0 dB (median 63.8) | at most 0.125 % |

The pixels that differ lie on silhouettes and on alpha-tested hair: the window's matrix is a different float32
product than the full frame's, so a vertex can land a fraction of a sample away and flip a sample's coverage there. The
tests also check a background image (its part of the "cover" picture) and real avatars' heads.

## 2. Render time: windows, culling, whole frames

Whole frames of the fair (36 people, 284k character triangles, 124 other objects): 1280x720 in 220-470 ms, 3840x2160
in 2.1 s, 7680x4320 in 6.3 s (the first frame at each size several seconds more: uploads and mipmaps).

Head windows of the fair at 6 s, placed by `HeadRegion` on each visible person's skeleton box, at the 8K sensor's
pixel scale (medians of three; 25 windows):

| Window side | Windows | Culled | Not culled | Speed-up (median) | People drawn | Objects drawn |
|---|---|---|---|---|---|---|
| 64-192 px (people 15-53 m away) | 17 | 27 ms (10-61) | 59 ms (44-107) | 2.2x | 1-6 of 36 | 2-10 of 124 |
| 256-416 px (10-15 m) | 4 | 72 ms (52-95) | 119 ms (73-133) | 1.4x | 3-4 | 2-13 |
| 528-1056 px (the audience at 3.8 m) | 4 | 272 ms (121-394) | 293 ms (144-502) | 1.2x | 3-13 | 7-38 |

In an earlier run on a quieter machine the same kinds of windows took 8-80 ms culled against 42-100 ms (1.2-5x). Every
culled window was identical to the unculled one. The culling tests take about 1 ms per window for the whole fair
(vectorised over boxes and over every bone sphere); windows also reuse the joint palettes their frame uploaded.
Culling the whole 1280x720 view gained nothing (most of the fair is in it), so it is off (`Renderer.cull_view`).

What the cost is made of: on llvmpipe a window costs about its pixels (the large audience windows are fill-rate
bound; a 1056 px window has more pixels than the whole view), plus a few milliseconds of fixed cost (framebuffer
resolve and read-back; a window size not seen before also makes its framebuffers, 2-7 ms, which is why sides are
rounded up to multiples of 16). In the demo with `W` on, four windows of the audience (768-928 px) took 1.9 s in
one frame and four windows of people further away 0.36 s in the next. On a GPU these costs are a few milliseconds.

## 3. The eye distance reached, and the face detector

People of six avatars (women, men, a child) stand in the fair's empty hall in an idle pose, facing the camera, at
2-30 m, spread so that nobody hides anybody; three rounds put other avatars at each distance (36 faces). For each:
the eye distance (eye bones) in the 1280x720 view and in their window, and whether YuNet (OpenCV's face detector,
`min_size=0`) finds the face, matched by its eye landmarks; in brackets YuNet's own eye distance. Windows from a
0.5 m square around the head, `face_px` 112, at most 1024x1024 pixels. The 4K and 1080p columns are the "good 4K" and
"budget 1080p" camera presets: their own sensors, with blur, noise, mosaic, tone and JPEG.

| Distance | View: eyes (px) | View: found | 8K ideal: eyes | found | Good 4K: eyes | found | Budget 1080p: eyes | found |
|---|---|---|---|---|---|---|---|---|
| 2 m | 27.2 | 3/3 (24) | 82 (level 2) | 3/3 (77) | 82 | 3/3 (78) | 41 | 3/3 (36) |
| 3 m | 19.2 | 3/3 (19) | 115 | 3/3 (115) | 58 | 3/3 (61) | 29 | 3/3 (28) |
| 4 m | 12.5 | 3/3 (11) | 75 | 3/3 (67) | 37 | 3/3 (33) | 19 | 3/3 (17) |
| 5 m | 10.6 | 3/3 (11) | 64 | 3/3 (65) | 32 | 3/3 (32) | 16 | 3/3 (16) |
| 6 m | 7.7 | 3/3 (8) | 46 | 3/3 (46) | 23 | 3/3 (20) | 12 | 3/3 (9) |
| 8 m | 6.2 | 1/3 (7) | 37 | 3/3 (37) | 19 | 3/3 (20) | 9 | 2/3 (9) |
| 10 m | 4.3 | 2/3 (4) | 26 | 3/3 (26) | 13 | 3/3 (12) | 6 | 3/3 (6) |
| 12 m | 3.7 | 2/3 (6) | 22 | 3/3 (25) | 11 | 3/3 (13) | 6 | 3/3 (6) |
| 15 m | 2.8 | 0/3 | 17 | 3/3 (15) | 8 | 3/3 (9) | 4 | 0/3 |
| 20 m | 2.2 | 0/3 | 13 | 3/3 (14) | 7 | 2/3 (7) | 3 | 0/3 |
| 25 m | 1.7 | 0/3 | 10 | 3/3 (10) | 5 | 1/3 (5) | 3 | 0/3 |
| 30 m | 1.5 | 0/3 | 9 | 3/3 (10) | 5 | 2/3 (5) | 2 | 0/3 |
| All | | **20/36** | | **36/36** | | 32/36 | | 23/36 |

- The windows give 6 times the view's eye distance (level 1 of the 8K sensor) until the face would exceed what is
  asked: at 2 m the 0.5 m square would have had 1037x1037 pixels, over the limit, so the window is binned by 2.
- 40 px between the eyes, where a recogniser's 35-pixel template is no longer upsampled, is reached out to about
  6.5 m with the 8K sensor, 3.5 m with a 4K camera and 2 m with a 1080p camera; in the 1280x720 view only within
  about 1.1 m.
- YuNet finds faces down to 5-10 px between the eyes on these clean renders, which is far smaller than a face can be
  recognised from; finding is not naming. F1/F2/E1 will measure recognition.
- YuNet's eye distances agree with the eye bones within a few pixels: the windows' faces are where the scene says.

In the fair itself (6 s, `bifold-scene windows --time 6 --people truth --faces`), 34 people were in the picture:
10 hidden behind others where their eyes are (the truth's `visible`), most others turned away or in profile (the fair's
people do not look at the camera, except the audience). YuNet found 5 faces in the view and 9 in the windows: the four
people watching the camera at 50-81 px between the eyes instead of 11-17 (one window binned by 2: its head square
was over the pixel limit), and five more at 6-22 px.

[docs/previews/virtual_sensor.jpg](../previews/virtual_sensor.jpg) shows the lineup, its windows and one face
through the presets.

## 4. Where the head is in a person box

YOLO (`yolo26n.pt`) ran on 10 frames of the fair (1-19 s); its person boxes were matched to people by their skeleton
boxes (IoU at least 0.5): 91 matches. In box heights from the top and from the middle:

| | 5 % | median | 95 % |
|---|---|---|---|
| Eyes below the box's top | 0.060 | 0.079 | 0.105 |
| Eyes across from the box's middle | -0.077 | -0.004 | 0.044 |
| Frontal eye distance | 0.0327 | 0.0342 | 0.0361 |

How often the whole face (1.4 eye distances to each side of the eyes, 1.2 above, 2.1 below) lies in the head square:

| `HeadRegion` (centre, side, margin) | Face inside | Face width in the side (median) | Pixels |
|---|---|---|---|
| 0.12, 0.30, 0.30 (the review's starting values, the default) | 91/91 | 25 % | 1 |
| 0.10, 0.20, 0.30 | 87/91 (96 %) | 37 % | 0.44 |
| 0.09, 0.17, 0.30 | 74/91 (81 %) | 43 % | 0.32 |
| 0.09, 0.15, 0.25 | 65/91 (71 %) | 51 % | 0.23 |

The default eye-distance share of `HeadRegion` was set to the measured 0.034 (63 mm in 1.75 m, 0.036, overestimated
it by 5 %). The person tracker can tighten the square once it knows a track's face (a `region` request around the face
found in an earlier window).

## 5. The camera model's cost

Per window, in numpy and OpenCV on this machine's CPU, best of three: 256x256 6-12 ms, 512x512 28-31 ms (37-45 ms with
motion blur, which adds the person's coverage and the masked smear), 768x768 59-71 ms (71-86 ms with motion blur).
`ideal` costs nothing. The noise, the sRGB tables, the masked motion blur and JPEG are the largest parts.

## 6. Open questions

- The presets' numbers are plausible values for such cameras, not measurements of one; calibrate them against
  footage of the intended camera.
- The motion blur smears only the person, over the window itself standing in for the background behind them; within a
  blur length of the silhouette a little of the sharp person shows through.
- Everything was rendered with the 1024-pixel head textures; step S2 brings the 2048 maps, which matter above about
  110 px between the eyes (high_res_faces.md 2.2).
- The large windows of people close to the camera dominate the cost on llvmpipe; the tighter head square or a
  lower `max_pixels` halves it, at some risk of cutting a face.
