# Close-ups: measured results

Step S4 of [face_recognition_focus.md](face_recognition_focus.md): faces seen up close get a finer head mesh and
skin, eye and lash shading. The design is sections 2.3, 2.5, 3.1 (A4) and 3.3 (R3-R6) of
[high_res_faces.md](high_res_faces.md); the code is `characters/subdivide.py`, `Character.detailed` and the renderer's
"close" quality (README: "Close-ups: subdivided heads, skin, eyes and lashes"). The preview is
[docs/previews/closeups.jpg](../previews/closeups.jpg).

Measured on 16 November 2026 in the development sandbox: a 4-core CPU shared with another builder's measurements,
Mesa llvmpipe (OpenGL 4.5), the 20 Rocketbox avatars prepared in format 2 (2048 heads with their maps), the face
models of F1 (AdaFace IR-101, `buffalo_l`'s ResNet-50, SFace, with SCRFD and YuNet). Timings vary by up to two times
between runs here; the scratch scripts are not in the repository. Photos of real people were used for checks only, in
memory; only aggregate numbers are reported.

<!-- IN SHORT: filled in last -->

*Results so far: sections 5, 7 and 8 (recognition up close, faces from real photos, the windows' render times) are
being measured, and this note is completed when they are.*

## 1. The subdivided heads

**The scheme.** Every triangle of the head becomes a cubic Bezier triangle (a PN triangle, Vlachos et al. 2001)
through its corners and tangent there to the mesh's own tangent planes (the angle-weighted normals of the triangles
round each position, welded across UV seams, split at creases); the new vertices are points of these patches: the
middle of every edge for x4, three points per edge and three inside every triangle for x16. An edge's curve depends
on its two ends only, so the two triangles of an edge share it and no crack can open; the original vertices stay,
bit for bit. The plan named the modified Butterfly scheme; the curved triangles were chosen because a new point
depends on one triangle and its normals rather than on two rings of neighbours with fixed weights, which assume
even edges, while the Rocketbox heads' face edges run from 2 to 50 mm side by side (10 mm on the median), and because
creases, open edges and the parts of the head need no special stencils; the measurement below (section 2) has the
Butterfly scheme on the same topology and creases for comparison.

**What keeps its shape** (the edges' rules; on Male_Adult_05's 4,651 edges: 3,531 curved, 148 creases, 396 on the
eyeballs, 576 straight):

- *creases*: skin that touches without an edge (the closed lips' line and the lids' corners: the S3 head space's
  ties), the lid margins (where skin facing out meets skin facing the eyeball, within 6 mm of it), the edge of the
  mouth's inside, folds sharper than 100 degrees. A crease's new points lie on a curve along it (tangent at a vertex
  the mean direction of its two crease edges; straight at corners), and each side keeps its own tangent plane: the
  folds stay folds. The two lips' new points take the same offset from their edges, so closed lips stay as closed as
  they were (the largest gap of tied points: 1.108 mm before and after; the new tied points 0.19 mm at most);
- *straight*: the neck's open edge and the skin the body's bones move (the S3 fade 0), so the seam to the collar
  stays where it was; the mouth's inside (teeth, gums, tongue), whose shape stays exact;
- *the eyeballs*: on their spheres (the radius interpolated, so a cornea keeps its bulge).

A few long slivers in the eye sockets (1 by 20 mm) twisted when curved and turned new triangles over; such a patch
is flattened (its new points halved towards the flat triangle, round by round, its seam copies and tied points
alike) until no new triangle turns over or changes its area by more than five times.

**The rest of a vertex** comes from its triangle by the same barycentric weights (a test checks every new vertex):
UVs per triangle, so seams stay exact (a new vertex on a seam is split like its ends); the skin weights, of which the
four strongest are kept and renormalised; the artist's normals, interpolated, so the shading inside the face stays as
it was and the outline is what changes; tangents made again from the new mesh (at the original vertices within
0.1-0.2 degrees of the old ones on the median, 1.3-1.6 degrees at the 95th percentile; two vertices of a mirrored
seam on Male_Adult_05 take the other side's). The subdivided model shares the skeleton, the inverse binds, the other
parts and the face rig of the model it was made from, so it poses exactly as that model does: in the face rig's
"surprise" the child's subdivided head's own vertices are where the coarse head's are, to the bit.

**Checked like a fitted head.** S3's rig checks, in the 12 faces of the face rig seen from the front and the side,
compare the head subdivided on its curved surface with the same head split along its flat triangles (the coarse
head's own surface, vertex for vertex): no eyeball through a lid, no teeth through the lips, no shut mouth opened,
the neck's edge where it was, no triangle folded. All pass at both levels on Male_Adult_05, Female_Adult_04,
Business_Female_02 and Male_Child_02, and on a head given a shape (S3's `apply_shape`) and then subdivided. The pixels
of eyeball shown grow by up to 4 % (the eyeballs are rounder: the coarse ones' facets were chords inside the
sphere); the mouth's inside shown changes by 0-8 pixels of 256 x 256.

**What it costs** (the 20 avatars, medians; this CPU, one thread): the labels 1.1 s the first time for an avatar
(0.5-1.5 s; the parts found by visibility), 0.09 s from the cache; the subdivision 0.06 s once (0.04-0.07 s) and
0.20 s twice (0.11-0.27 s); the head 1,780 vertices and 3,140 triangles coarse, 6,720 and 12,570 at x4, 26,150 and
50,290 at x16; its arrays (and the buffers the renderer uploads: 20 floats a vertex with the tangents, the indices)
0.18 MB coarse, 0.69 MB at x4, 2.7 MB at x16.

## 2. The outline

Each head's triangles were drawn 8 times finer than a window's pixels at 256 px between the eyes, its outline found
and reduced to the polygon it is (Douglas-Peucker at 1/8 px), and the angle between neighbouring outline segments
measured at each corner: the face's sides seen from the front (cheeks and jaw, from the eyes down to the jaw's angle)
and the profile seen from the side (brow, nose, lips, chin). Medians over the ten test avatars of each avatar's
median, 90th percentile and largest corner; the segments' median length; and the corner's sagitta, a segment's
length times the 90th-percentile angle over 8, the pixels by which such a corner leaves a smooth curve, with the
eye distance at which it reaches half a pixel (it grows in proportion to the face's size):

| View | Head | Corners | Angle, median | 90 % | Largest | Segment | Sagitta at 256 px | Half a pixel at |
|---|---|---|---|---|---|---|---|---|
| front (jaw) | coarse | 29 | 9.6° | 32.4° | 61° | 26.8 px | 1.90 px | 68 px |
| | x4 | 56 | 6.4° | 23.1° | 47° | 13.8 px | 0.69 px | 184 px |
| | x16 | 80 | 4.4° | 14.7° | 50° | 8.8 px | 0.28 px | 455 px |
| | Butterfly x4 | 53 | 6.6° | 23.1° | 68° | 13.6 px | 0.68 px | 187 px |
| profile (nose) | coarse | 44 | 15.4° | 35.6° | 144° | 16.8 px | 1.31 px | 98 px |
| | x4 | 78 | 8.7° | 22.5° | 148° | 9.9 px | 0.48 px | 265 px |
| | x16 | 123 | 6.2° | 14.4° | 142° | 6.6 px | 0.21 px | 615 px |
| | Butterfly x4 | 83 | 9.5° | 25.2° | 147° | 9.3 px | 0.51 px | 252 px |

(The largest profile corners, 140-150 degrees, are real: where the lips meet and under the nose.)

- **The facets go.** Subdividing once halves the corners' angles and the segments; twice quarters the segments again.
  The coarse jaw's corners leave a smooth curve by half a pixel from about 68 px between the eyes, x4's from about
  184 px, x16's not below 450 px. **So the windows subdivide once from 64 px and twice from 180 px** (the brief's
  "about 150" measured: the front view's jaw needs x16 from 184 px, the profile from 265 px).
- **Curved triangles against Butterfly.** On the same topology, creases and rules, the curved triangles' x4 outline is
  as smooth as the Butterfly's from the front (the same medians; the largest corner 47 against 68 degrees) and
  smoother in profile (8.7 against 9.5 degrees on the median, 22.5 against 25.2 at the 90th percentile), as expected
  where uneven edges meet the Butterfly's even-spacing weights. Seen in windows the two x4 heads look alike; both
  round the coarse head's jaw, chin and nose tip.

## 3. In windows

The renderer's third quality, `"close"`, is the high quality with the finer heads and the shading of section 4 for
faces seen up close. The demo's live windows are drawn at it (`SceneSource.window_quality`; `bifold-scene windows`
shows them so); a renderer made for anything else keeps `window_quality = "high"`, as the recognition's
calibrations and the avatars' gallery photos were drawn (section 6 says why). The 720p view stays `"high"`, and the
`"high"` and `"low"` qualities draw as before, bit for bit (checked against the step's first commit). In a picture of
the close quality, a person whose eyes are at least `subdivide_from[0]` (64) pixels apart in it is drawn with their
head subdivided once, from `subdivide_from[1]` (180) twice; `Character.detailed(level)` is the model the person wears
at that moment (the avatar's, or after `use_model` a fitted head) with its head subdivided, sharing the skeleton, the
textures and maps, the detail textures and the face rig, so the pose, the expression, a face from photos (its
texture overrides and photo mask, which stay keyed by the coarse head) apply to it as they are. The subdivided
geometry is cached per head geometry (64 heads), so people wearing the same avatar share it. Measurements, tests and
the calibration's renders make it at once; the demo's live renderer makes it on a worker thread and draws the coarse
head until it is there (`detail_wait`, as for the 2048 textures).

## 4. Shading up close

For faces at least `close_from` (64) pixels between the eyes in a picture of the close quality (smaller faces there
are drawn exactly as at the high quality, bit for bit):

- **Skin.** The key light wraps round the skin of the head, `(n.l + w) / (1 + w)` per colour with `w` = 0.35, 0.18
  and 0.12 for red, green and blue: light that goes into skin comes out further on, so the skin is lit a little past
  where the light grazes it, reddest there; where the light falls straight on, nothing changes. This is the look of
  pre-integrated skin (Penner 2011) along the terminator without its table; the curvature term (thin noses and ears
  glowing more) is left out. `renderer.skin_wrap` scales it (0: Lambert).
- **Eyes.** The eyeballs (the vertices their eye bones move, found in the vertex shader) get a cornea's highlight: a
  GGX lobe (roughness 0.12, Fresnel from F0 0.025, a cornea's) on a normal turned as if the cornea were a sphere half
  the eyeball's radius round the eye's axis, so a light from above puts its highlight on the iris rather than under
  the upper lid, where the eyeball's own normal would. The wrap is not applied to the eyeballs. `renderer.cornea`
  scales it.
- **Lashes and hair cards** (mesh parts named "opacity" or "hair") are drawn with alpha to coverage when the picture
  has MSAA: the cut-out where the texture's alpha crosses 0.5, as before, but with its edge one pixel wide, so the
  lashes' edges are antialiased instead of stair-stepped. `renderer.coverage` turns it off.
- **A shadow map for the key light** was not made: the 2048 maps' ambient occlusion already darkens the eye sockets,
  the sides of the nose and under the lips, and in the fair's hall the key light is high and soft. What a shadow map
  would add (the nose's and the brim's cast shadows in a side light) and what it costs (a depth pass per window) is
  left measured by nobody: an open issue (section 9).

Faces from photos keep S2's rule: no avatar maps where the photo face is (its relief and shine are the person's
own), no hemisphere variation there; the wrap and the cornea are applied to them, as they are not the avatar's
features (a lit-past-the-terminator skin and a cornea's highlight are everyone's). Section 7 checks what that does to
their identity scores.

## 5. What the recognisers see up close

**How.** Windows of S1's virtual sensor (the ideal camera) on the face of a person standing alone 3 m in front of the
camera, drawn with 64, 96, 128, 192 and 256 pixels between the eyes, the head turned 0, 25 and 45 degrees, in two
lights: one like the fair's (the key light from above, in front and to the camera's right; ambient 0.55, the
hemisphere's variation 0.3) and a harsh side light (from the camera's left, nearly side-on; ambient 0.35). Ten
avatars' own faces (Business_Female_02, Business_Male_05, Female_Adult_01, Female_Adult_06, Female_Adult_13,
Female_Party_02, Male_Adult_05, Male_Adult_12, Male_Adult_17, Male_Child_02), and faces from photos as S3 measured
them (its protocol D1: avatar B's head fitted to six renders of avatar A, and textures made from them; 8 pairs among
the ten, A the identity shown). Galleries: each
identity's own renders at 112 px, turned 0 and 20 degrees each way, in the fair's light; either drawn in the same
configuration as the probes ("matched") or always with the coarse head ("fixed": people enrolled before the change,
or enrolled from photos). Configurations: the coarse head (the high quality, S2's shading); x4 and x16 (the head
subdivided, the high shading); "shading" (the coarse head with the close-up shading); x16 with the shading; and the
close quality as shipped (x4 below 180 px, x16 from 180, the shading). Each window's face found and aligned as live
(SCRFD for AdaFace IR-101 and the ResNet-50, YuNet for SFace) and embedded by the three recognisers. Measures, with
95 % intervals from a bootstrap over the identities (`recognition/metrics.py`): the same-person and stranger scores
(each probe against each identity's best gallery photo), d', rank-1 among the ten; and the change of each probe's
same-person score from the coarse head's in the same window.

<!-- RECOGNITION TABLES -->

## 6. The shipped calibrations, made at "high"

F2's calibrations (`recognition/calibrations/`) were made from windows drawn at the high quality: faces of 8 to 90
pixels between the eyes asked (8, 11, 14, 18, 24, 32, 44, 64 and 90), in size bins up to 56-80 and 80 and more. In
windows of the close quality faces from 64 pixels are drawn differently (the head x4, the close-up shading; from 180
pixels x16): the 64 and 90 px faces of the two largest bins, and the larger faces those bins also judge (window
close-ups; the webcam's near faces are not drawn and do not change). Does the change move the scores the
calibrations were made for?

**How.** The calibration's own renders (`calibrate.render`: each avatar at a place in the fair's hall drawn from its
own random numbers, turned -40 to 40 degrees, through the ideal and the budget 1080p cameras; gallery photos at 100
px) of the fair's 20 avatars, which are not among the 38 the calibrations were made from, drawn once with the
windows at "high" and once at "close", and once more for each with 128 px between the eyes (in the 80+ bin, beyond
the calibrated sizes); found, checked by the live quality gate and embedded as `bifold-faces calibrate` does
(SCRFD with AdaFace and the ResNet-50, YuNet with SFace); then judged with the shipped calibrations as the live
decisions use them (their normalisation and cohort, the offsets per stranger track). Three worlds: "high" (faces and
gallery at high, as the calibrations were made), "close" (both at close: the known people enrolled from renders
drawn at close too) and "close faces, high gallery" (the gallery photos drawn at high, as before; or people
enrolled from photos).

**What changes in the windows.** At 44 px and below the windows are the same to the bit at both qualities. At 64 px
half of them are (the renderer measures a face's eye distance at the head bone, a little behind the eyes, so faces
asked at 64 px fall either side of `close_from`); at 90 and 128 px none. Where a window changes, its embeddings at
close and at high are about 0.96 alike on the mean (cosine; 0.88-0.93 at least) for the three recognisers.

**Scores and decisions** (90 % intervals from a bootstrap over the avatars, the changes paired: the same windows;
the decisions over 100 halvings of the 20 avatars into 10 known people and 10 strangers, at 1 % asked, as live; the
per-track rates over 60 halvings with the shipped offsets of 2.20, 2.54 and 2.60 decades):

| Recogniser | eye px | same-person score at high | close faces, high gallery: change | best stranger's rarity: change (decades) | both at close: same-person change | best stranger's rarity: change |
|---|---|---|---|---|---|---|
| AdaFace IR-101 | 44 | 0.906 | 0 (the same windows) | 0 | -0.029 (-0.033 to -0.026) | -0.08 (-0.12 to -0.05) |
| | 64 | 0.929 | -0.012 (-0.017 to -0.007) | -0.06 (-0.09 to -0.03) | -0.015 (-0.021 to -0.009) | -0.11 (-0.15 to -0.06) |
| | 90 | 0.938 | -0.024 (-0.027 to -0.021) | -0.11 (-0.14 to -0.07) | -0.0005 (-0.002 to +0.001) | -0.12 (-0.17 to -0.07) |
| | 128 | 0.944 | -0.026 (-0.029 to -0.023) | -0.11 (-0.14 to -0.08) | -0.001 (-0.003 to +0.001) | -0.12 (-0.17 to -0.07) |
| ResNet-50 | 44 | 0.911 | 0 | 0 | -0.029 (-0.033 to -0.026) | -0.08 (-0.11 to -0.04) |
| | 64 | 0.932 | -0.013 (-0.018 to -0.008) | -0.07 (-0.12 to -0.04) | -0.014 (-0.020 to -0.008) | -0.12 (-0.16 to -0.08) |
| | 90 | 0.940 | -0.024 (-0.027 to -0.021) | -0.13 (-0.17 to -0.10) | +0.0005 (-0.001 to +0.002) | -0.13 (-0.16 to -0.09) |
| | 128 | 0.944 | -0.025 (-0.028 to -0.022) | -0.14 (-0.17 to -0.10) | -0.000 (-0.001 to +0.001) | -0.13 (-0.17 to -0.10) |
| SFace | 44 | 0.914 | 0 | 0 | -0.016 (-0.019 to -0.014) | -0.03 (-0.08 to +0.01) |
| | 64 | 0.931 | -0.011 (-0.017 to -0.006) | +0.01 (-0.03 to +0.04) | -0.009 (-0.014 to -0.005) | -0.00 (-0.05 to +0.05) |
| | 90 | 0.934 | -0.020 (-0.024 to -0.016) | +0.05 (+0.00 to +0.09) | +0.001 (-0.001 to +0.003) | +0.07 (+0.01 to +0.13) |
| | 128 | 0.938 | -0.021 (-0.024 to -0.018) | +0.04 (-0.01 to +0.08) | -0.002 (-0.003 to -0.000) | +0.06 (-0.00 to +0.12) |

(The same-person score: a single face against its own avatar's best gallery photo; the best stranger's rarity: the
calibration's rarity, in its bin and normalisation, of a face's highest-scoring other avatar, 1.0-1.3 decades on the
mean at high.)

- **Faces drawn at close, judged by the calibrations: within their noise.** Against a gallery drawn at high, no
  decision changes: the known people's faces of 64, 90 and 128 px are named per decision at 1 % in 100 % of
  halvings at both qualities, those of 8-44 px are the same windows; no stranger's track is named at 1 % asked, with
  the offsets or without them, at either quality, in any class of sizes (all of them; 44-64; 90; 90-128; 128 px:
  4,800-4,925 tracks from 100-118 identity pairs). The same-person scores fall by 0.01-0.03 and the strangers'
  best rarity moves by -0.14 to +0.05 decades, against offsets of 2.2-2.6 decades and F2's own intervals for the
  rate per stranger track at 1 % (0.33-1.15 %, 0.18-0.90 % and 0.08-1.26 %: their ends 3.5 to 16 times apart). The
  calibration's promise per comparison says the same: stranger comparisons reach a rarity of 2 decades (promised:
  1 %) in 0-0.03 % (AdaFace, the ResNet-50) and 0.17-0.20 % (SFace) of cases at high, in 0-0.03 % and 0.21-0.37 %
  with the faces drawn at close: on these avatars the calibrations are 5 to 50 times (or more) on the safe side at
  both qualities. So the shipped calibrations stand for the demo's close windows, and nothing needs remaking. (The
  fair's 20 avatars give no false alarm at 1 % even without the offsets: unlike the calibrating avatars, whose false
  alarms came from 5-11 pairs that look alike to the models, they hold no such pair. At 10 % asked, SFace names
  2.2 % of stranger tracks at high and 2.3 % with the faces drawn at close.)
- **Gallery photos drawn at close are not.** Their 100 px faces get the finer head and the shading, while the faces
  of 8-44 px (most of a crowd's) are drawn as at high and become less like them: known people's small faces are
  named 0.8-1.5 points less often at 1 % (AdaFace 95.6 % to 94.1 %, the change -2.1 to -0.9 points; the ResNet-50
  95.9 % to 94.6 %, -1.8 to -0.8; SFace 93.7 % to 92.9 %, -1.2 to -0.4), and SFace's stranger tracks at 10 % asked
  rise from 2.2 % to 5.3 % (the 90 px class 0.5 % to 6.5 %; still within the 10 % asked; none at 1 %).
  The cause is the close-up shading, not the finer head: with gallery photos drawn with the head subdivided and the
  high shading the small faces are named as often as with the coarse head (AdaFace 95.7 %, the ResNet-50 95.9 %,
  SFace 93.8 %; the changes' intervals hold 0), with the close-up shading on the coarse head as rarely as with both
  (93.9 %, 94.7 % and 92.7 %); their same-person scores fall by 0.016-0.026 with the shading and change by less
  than 0.001 with the finer head. **So a renderer keeps `window_quality = "high"` unless asked**: the calibration's
  renders and the avatars' gallery photos (`bifold-faces calibrate`, `calibrate.gallery_photos`) are drawn as they
  were, and the live world is the third one above, which decides as the calibrations' own; the demo's live windows
  ask for "close" (`SceneSource.window_quality`), and `bifold-scene windows` shows them so.
- **Faces of 128 px** (in the 80+ bin, beyond the calibrated sizes) behave as those of 90 px.

What remaking the two bins would take, had it been needed: `bifold-faces calibrate` keeps renders and embeddings
per window key, so in a copy of F2's data the 64 and 90 px windows of the 38 calibrating avatars (760 per detector)
and their embeddings would be removed and rendered again with the close quality (a few minutes), embedded again
(AdaFace about 15 minutes here), and the curves, the cohort and the offsets per stranger track (120 halvings) made
again: about an hour, and every shipped file and check number would change.

## 7. Faces from real photos

**How** (S2's review protocol, numbers only: no image of a real person was written anywhere). Four people of the
local test photos (three men, one woman); one photo of each painted on three avatars of their sex as the face cast
does (`face_photo.put_face`, the photo mask: no avatar maps there), the person's other photo as the gallery; S1's
windows (8K, the ideal camera) in the fair's hall at 2 m, turned 0 and 25 degrees, with 80, 128 and 192 pixels
between the eyes; the three recognisers; the same windows drawn at the high quality, at the close quality, at the
close quality with the coarse head, and at the close quality without the skin's wrap.

<!-- PHOTO NUMBERS -->

## 8. Costs

The subdivision's (section 1): 0.06 s once and 0.20 s twice per head, 0.7 and 2.7 MB per subdivided head (in memory
and on the GPU; cached for 64 head geometries), the labels 1.1 s the first time for an avatar.

<!-- WINDOW COSTS -->

## 9. Open issues

- **No shadow map for the key light.** Not made (section 4); what it would add to a face up close in a side light, and
  what a depth pass per window costs on this renderer, are not measured.
- **The skin's wrap has no curvature term**: thin parts (the nostrils' wings, the ears) do not glow more than the
  cheeks, as pre-integrated skin would make them.
- **The mouth's inside stays coarse** (teeth, gums, tongue split on straight edges): their facets can show in an open
  mouth at 200 px and more.
- **A mirrored seam's tangents.** On Male_Adult_05 two vertices of the head's mirrored UV seam take the other side's
  tangent when the tangents are made again; the normal map's relief may turn the wrong way on the few triangles round
  them (not seen in the windows looked at).
- **The Butterfly comparison is at one level** (x4) and on the outline only.

<!-- MORE OPEN ISSUES -->
