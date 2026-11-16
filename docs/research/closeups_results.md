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

**In short.**

- **A finer head where a face is seen up close.** Every triangle of the head becomes a curved one (a PN triangle),
  split into 4 or 16: the closed lips stay closed, the lids' rims and sharp folds stay creases, the neck's edge, the
  mouth's inside and the skin the body moves stay where they were, the eyeballs stay on their spheres, and the skin
  weights, UVs and tangents follow. S3's rig checks pass at both levels in the 12 faces of the face rig, on four
  avatars and on a fitted head. A head takes 0.06 s (x4) or 0.20 s (x16) and 0.7 or 2.7 MB.
- **The outline's facets go.** The coarse jaw leaves a smooth curve by half a pixel from about 68 px between the eyes,
  the x4 head's from about 184 px: windows subdivide once from 64 px and twice from 180 px. On the same topology the
  curved triangles are as smooth as the modified Butterfly scheme from the front and smoother in profile.
- **Shading up close:** the key light wraps round the skin (red furthest), the eyeballs get a cornea's highlight,
  and lashes and hair cards are antialiased by alpha to coverage. Smaller faces and the 720p view are drawn as before,
  bit for bit. No shadow map was made.
- **What the recognisers see changes little.** The finer head moves the same-person scores by less than 0.002 and
  d' by less than 0.1 (AdaFace, the ResNet-50, SFace; 64-256 px). The shading raises the same-person scores by
  0.006-0.012 and d' by about 0.4, in a hard side light and on turned faces (in the fair's soft frontal light, by
  little either way). Rank-1 among ten identities is 1.000 either way: up close the gain is mostly what the faces
  look like. Faces from photos keep their identity: S3's fitted faces move by -0.012 to +0.005, and faces painted
  from real photos gain 0.015-0.021 from the wrap.
- **The shipped calibrations stand.** Faces drawn at close (64-128 px) change no decision on the fair's 20 avatars.
  Gallery photos drawn at close would cost known people's small faces 0.8-1.5 points of names at 1 % (through the
  shading), so only the demo's live windows draw close-ups; any other renderer, the calibration's and the gallery
  photos' among them, keeps the high quality.
- **Cost:** on llvmpipe a window takes about 40 ms more with the x4 head and 140 ms more with x16; the shading costs
  nothing measurable.

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
- **In the expressions.** Contact sheets of three avatars (Male_Adult_05, Female_Adult_04, Male_Child_02) in the 13
  faces of the face rig, front and side at 160 px, coarse against x16 (not in the repository): the expressions move
  the finer head as the coarse one, with no lips parting, no gap at a lid and no eyeball through one, and the
  profile's jaw, chin and nose are rounder. The preview shows the close quality at 256 px.

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

**The avatars' own faces** (300 windows per configuration and recogniser; galleries drawn in the same configuration;
95 % intervals over the ten identities):

| Recogniser | configuration | same-person | stranger | d' | rank-1 | same-person change from the coarse head |
|---|---|---|---|---|---|---|
| AdaFace IR-101 | coarse | 0.842 | 0.138 | 7.01 (6.16-8.08) | 1.000 | |
| | x4 | 0.841 | 0.134 | 6.99 (6.10-8.08) | 1.000 | -0.001 (-0.004 to +0.002) |
| | x16 | 0.840 | 0.134 | 7.00 (6.12-8.10) | 1.000 | -0.002 (-0.005 to +0.002) |
| | shading | 0.847 | 0.136 | 7.39 (6.36-8.76) | 1.000 | +0.006 (+0.000 to +0.011) |
| | x16 + shading | 0.845 | 0.129 | 7.33 (6.30-8.70) | 1.000 | +0.004 (-0.002 to +0.009) |
| | close | 0.845 | 0.131 | 7.31 (6.26-8.70) | 1.000 | +0.003 (-0.002 to +0.008) |
| ResNet-50 | coarse | 0.837 | 0.139 | 6.75 (5.89-7.91) | 1.000 | |
| | x4 | 0.837 | 0.134 | 6.81 (5.90-8.00) | 1.000 | +0.000 (-0.001 to +0.001) |
| | x16 | 0.836 | 0.133 | 6.84 (5.96-7.98) | 1.000 | -0.000 (-0.002 to +0.001) |
| | shading | 0.846 | 0.139 | 7.17 (6.16-8.56) | 1.000 | +0.009 (+0.005 to +0.013) |
| | x16 + shading | 0.844 | 0.132 | 7.15 (6.11-8.56) | 1.000 | +0.007 (+0.002 to +0.011) |
| | close | 0.843 | 0.133 | 7.15 (6.10-8.53) | 1.000 | +0.006 (+0.002 to +0.011) |
| SFace | coarse | 0.845 | 0.265 | 5.68 (5.00-6.78) | 1.000 | |
| | x4 | 0.842 | 0.261 | 5.42 (4.79-6.54) | 0.997 | -0.001 (-0.002 to +0.002) |
| | x16 | 0.843 | 0.260 | 5.43 (4.80-6.53) | 0.997 | +0.000 (-0.002 to +0.003) |
| | shading | 0.855 | 0.267 | 6.07 (5.37-7.30) | 1.000 | +0.012 (+0.006 to +0.017) |
| | x16 + shading | 0.855 | 0.263 | 6.09 (5.34-7.41) | 1.000 | +0.011 (+0.006 to +0.017) |
| | close | 0.854 | 0.264 | 6.08 (5.35-7.41) | 1.000 | +0.010 (+0.005 to +0.016) |

d' by eye distance (64, 96, 128, 192, 256 px): AdaFace coarse 6.97, 7.00, 7.01, 7.03, 7.03; x16 7.00, 6.97, 6.99,
7.05, 7.02; close 7.21, 7.35, 7.31, 7.32, 7.37. The ResNet-50 coarse 6.70-6.79, x16 6.80-6.87, close 7.08-7.22. SFace
coarse 5.47-5.89, x16 4.54-5.93, close 5.93-6.22.

- **The finer head changes next to nothing the recognisers see.** Against galleries drawn the same way the
  same-person scores move by less than 0.002 (the intervals hold 0) and d' by less than 0.1 for AdaFace and the
  ResNet-50, at every size from 64 to 256 px; against the coarse head's gallery (people enrolled before), by -0.002
  to -0.004. SFace's d' falls from 5.68 to 5.42 with the finer heads because of one window: Male_Adult_05 turned 45
  degrees in the side light at 256 px, where YuNet's landmarks miss the face (same-person 0.03); without it, the finer
  heads score as the coarse one.
- **The close-up shading helps a little, where the light is hard.** The same-person scores rise by 0.006 (AdaFace),
  0.009 (the ResNet-50) and 0.012 (SFace), the strangers' stay, and d' rises from 7.0 to 7.4, 6.8 to 7.2 and 5.7 to
  6.1 (the intervals over ten identities overlap). The gain is in the side light (AdaFace's d' 7.8 to 8.4, the
  ResNet-50's 7.4 to 7.8, SFace's 5.8 to 6.2) and on turned faces (45 degrees: 7.3 to 8.1, 7.2 to 7.7, 5.8 to 6.3),
  where the wrapped light fills the shadow side; in the fair's light and from the front it is small either way
  (frontal faces: AdaFace 9.9 to 9.6, the ResNet-50 9.5 to 9.7, SFace 7.5 to 7.8). Against a gallery drawn at high
  the same-person scores keep their level (-0.004 to +0.005, the intervals hold 0) and d' still rises (AdaFace 7.0
  to 7.6): the scores spread less.
- **Rank-1 among the ten is 1.000** in every configuration but SFace's finer heads (0.997: the window above): faces
  of 64 px and more are all named among ten people; the differences show in the scores, not in the names.

**Faces from photos** (S3's D1: avatar B's fitted head wearing the textures made from six renders of avatar A, the
identity shown A; 8 pairs, 240 windows per configuration; galleries: A's own renders, drawn in the same
configuration, or with the coarse head as for people enrolled before):

| Recogniser | configuration | same-person | stranger | d' | rank-1 | same-person change, matched gallery | same-person change, coarse gallery |
|---|---|---|---|---|---|---|---|
| AdaFace IR-101 | coarse | 0.514 | 0.141 | 3.36 (2.71-4.41) | 0.933 | | |
| | x4 | 0.507 | 0.137 | 3.33 (2.68-4.41) | 0.929 | -0.007 (-0.009 to -0.004) | -0.004 (-0.006 to -0.001) |
| | x16 | 0.510 | 0.139 | 3.48 (2.74-4.59) | 0.938 | -0.004 (-0.010 to +0.002) | -0.001 (-0.006 to +0.004) |
| | shading | 0.505 | 0.136 | 3.35 (2.59-4.51) | 0.929 | -0.009 (-0.020 to -0.000) | -0.007 (-0.013 to -0.002) |
| | close | 0.502 | 0.132 | 3.43 (2.61-4.66) | 0.938 | -0.012 (-0.023 to -0.002) | -0.007 (-0.015 to +0.001) |
| ResNet-50 | coarse | 0.523 | 0.145 | 3.42 (2.79-4.39) | 0.925 | | |
| | x4 | 0.518 | 0.139 | 3.45 (2.81-4.41) | 0.929 | -0.005 (-0.008 to -0.001) | -0.002 (-0.006 to +0.001) |
| | x16 | 0.519 | 0.139 | 3.62 (2.88-4.55) | 0.933 | -0.003 (-0.011 to +0.005) | +0.000 (-0.007 to +0.009) |
| | shading | 0.524 | 0.143 | 3.63 (2.86-4.68) | 0.933 | +0.001 (-0.006 to +0.008) | -0.001 (-0.004 to +0.004) |
| | close | 0.522 | 0.138 | 3.79 (2.96-4.96) | 0.938 | -0.000 (-0.010 to +0.012) | +0.000 (-0.008 to +0.012) |
| SFace | coarse | 0.612 | 0.266 | 3.18 (2.77-3.81) | 0.924 | | |
| | x4 | 0.607 | 0.263 | 3.22 (2.79-3.87) | 0.924 | -0.005 (-0.008 to -0.002) | -0.001 (-0.004 to +0.003) |
| | x16 | 0.606 | 0.262 | 3.17 (2.76-3.79) | 0.920 | -0.006 (-0.010 to -0.002) | -0.001 (-0.006 to +0.003) |
| | shading | 0.619 | 0.263 | 3.51 (2.97-4.40) | 0.933 | +0.007 (-0.000 to +0.017) | +0.007 (+0.001 to +0.012) |
| | close | 0.614 | 0.260 | 3.49 (2.95-4.35) | 0.933 | +0.004 (-0.005 to +0.014) | +0.005 (-0.003 to +0.012) |

- **Faces from photos keep their identity up close.** With the close quality their same-person scores move by
  -0.012 to +0.005 (AdaFace a little lower: 2 % of its score, the strangers' as much; the ResNet-50 and SFace as
  before), d' rises from 3.2-3.4 to 3.4-3.8 and rank-1 among the ten from 0.92-0.93 to 0.93-0.94; against galleries
  drawn with the coarse head (people enrolled before) the same. The finer head alone lowers their same-person scores
  by 0.003-0.007 against matched galleries, the shading alone moves them by -0.009 to +0.007.
- They stay what S3 measured them to be: far less like the person than the avatars' own faces (same-person 0.50-0.61
  against 0.84-0.85, d' 3.2-3.8 against 5.4-7.4), whatever the mesh and the shading.

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

72 windows per configuration and recogniser (4 people, 3 avatars, 2 turns, 3 sizes); the same-person score of a
window against the person's other photo, its change from the same window at high (the mean; in how many windows it
fell; the largest fall), rank-1 among the four:

| Recogniser | high: same-person (lowest) | close | change | close, coarse head | close, no wrap | rank-1, high / close |
|---|---|---|---|---|---|---|
| AdaFace IR-101 | 0.358 (0.243) | 0.379 (0.265) | +0.021 (fell in 19 %, at most -0.049) | 0.383 (+0.025) | 0.356 (-0.002) | 1.000 / 1.000 |
| ResNet-50 | 0.344 (0.242) | 0.358 (0.244) | +0.015 (29 %, -0.056) | 0.360 (+0.016) | 0.343 (-0.001) | 1.000 / 1.000 |
| SFace | 0.375 (0.246) | 0.392 (0.273) | +0.017 (26 %, -0.035) | 0.395 (+0.020) | 0.374 (-0.001) | 0.972 / 1.000 |

**Faces from real photos keep their identity up close, and gain a little.** Their same-person scores rise by 0.015-
0.021 on the mean at the close quality (the same at 80, 128 and 192 px), the strangers' stay (-0.026 to -0.024,
-0.016 to -0.014, 0.111 to 0.113 on the mean; the highest 0.096 to 0.091, 0.109 to 0.087, 0.339 to 0.301), and
SFace names every window rightly among the four (from 97 %). The gain is the skin's wrapped light: without it the
close quality scores as the high one (-0.001 to -0.002), and with the coarse head as with the finer one. The wrap
puts none of the avatar's features on the photo's face (no maps there: S2's rule is kept), so the faces from photos
keep it.

## 8. Costs

The subdivision's (section 1): 0.06 s once and 0.20 s twice per head, 0.7 and 2.7 MB per subdivided head (in memory
and on the GPU; cached for 64 head geometries), the labels 1.1 s the first time for an avatar.

**A window's render time** on llvmpipe (the person alone, turned 20 degrees, the fair's light; the median of 7
renders after a first one that uploads and subdivides; three avatars, Male_Adult_05, Female_Adult_04 and
Male_Child_02; this CPU shared with another builder's measurements, so the times scatter by 20-40 %):

| Eye distance (window) | coarse (high) | x4 | x16 | x16 + shading | close (as shipped) |
|---|---|---|---|---|---|
| 128 px (576-720 px square) | 76-108 ms | 129-148 ms | 233-253 ms | 192-242 ms | 126-140 ms (x4) |
| 256 px (1136-1440 px square) | 221-323 ms | 305-372 ms | 409-513 ms | 429-491 ms | 389-547 ms (x16) |

The finer head is what costs: x4 adds about 40 ms to a window (40 % at 128 px, 15 % at 256 px), x16 about 140 ms (more
than doubling a 128 px window, which is why x16 waits until 180 px, where a window has four times the pixels); the
close-up shading adds no measurable time. The 720p view does not change (its quality stays high), nor windows of
faces under 64 px. On a GPU (the Mac's) meshes of 13,000 and 50,000 triangles a head should cost little; that was
not measured here.

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
- **Gallery photos must stay at the high quality.** A gallery of avatars rendered through the demo's live renderer
  (whose windows are close) would lose small faces' names (section 6); `calibrate.gallery_photos` and the calibration
  make renderers of their own and are safe. Remaking the calibrations at close would need their galleries at high,
  or the whole world at close measured again.
- **The fair's avatars cannot show a change in false alarms at 1 %**: they give none at either quality. The shift was
  judged from the rarities and the rates per comparison, which move by little; a library with lookalike pairs (as the
  calibrating avatars have) would test the 56-80 and 80+ bins harder.
- **`close_from` sits on a calibrated size.** Faces asked at 64 px fall either side of it (the eye distance is taken at
  the head bone); harmless as measured, but a threshold between the calibrated sizes (say 72 px) would make the
  calibration's 64 px faces all drawn as at high.
- **Rank-1 is at its ceiling** among ten identities at 64 px and more; a larger gallery would show whether the
  shading's gain in d' turns into names.
- **x16 is expensive on llvmpipe** (about 140 ms more per window); on the Mac it was not measured.
