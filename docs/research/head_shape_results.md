# Head shapes from photos: measured results

Step S3 of [face_recognition_focus.md](face_recognition_focus.md): the head's shape fitted to a person's photos.
The design is section 3.2 of [high_res_faces.md](high_res_faces.md); the code is `characters/headshape/` (README:
"Head shapes from photos"). Measured in November 2026 in the development sandbox (a 4-core CPU shared with other
jobs, Mesa llvmpipe), on the 20 prepared Rocketbox avatars of the test library, with ICT-FaceKit Light at commit
`da5f95a` and MediaPipe's Face Landmarker and selfie segmenter. The scratch scripts are not in the repository; the
evaluation's own code is (`characters/headshape/evaluate.py`, `compare.py`). Photos of real people were used for
checks only, in memory; only aggregate numbers are reported.

**In short.** On avatars as people (the test half, 20 pairs, six photos), the fitted head's landmarks are closer to
the person's than the avatar's own head's (median 2.65 → 2.38 mm after a similarity alignment; 3.83 → 3.19 rigid),
most in the mouth, the forehead, the nose and the profile, but the cheeks get worse and with them the surface over
the face (1.25 → 1.35 mm). The free-form correction alone, the fallback without the space, did as well on the
landmarks (2.36) and better on the surface (1.18), and made no head worse by more than 0.1 mm; three photos (front
and 30 degrees each way) did better than six. On random identities of the space laid on an avatar (D2), the fit
helps where the person is far from the avatar and stops at about 2-3 mm: MediaPipe's systematic errors, not the
solver, are the limit. The size had to be taken out of the fit (it made the heads 5-15 % too big). As a probe for
the face recognition, the fitted head wearing the person's face textures scores higher for every engine than the
avatar's own head with textures made the same way (AdaFace 0.50 → 0.57, rank 1 88 → 100 %): the textures land on a
head of the person's proportions. A fitted head costs about 0.2 MB of GPU buffers per character and 0.03 s to put on
the rig.

## How it is measured

- **D1, avatars as people.** Renders of avatar A (1024x768, a 30-degree lens, 0.9 m from the head, the fair's
  light) stand in for photos; they are registered to avatar B's head and B is fitted to them; the fitted head is
  compared with A's, and so is B's own head (the starting point). Six renders unless said otherwise: the front,
  turned 30 and 60 degrees each way, nodded 15 degrees; the focal length is not given (the registration assumes a
  50-degree view).
- **D2, identities of the space.** Avatar B's head changed by random identity coefficients (one standard deviation
  per mode, all 100 modes), wearing B's texture, photographed the same way and fitted back onto B.
- **Measures** (`compare.py`, as the NoW benchmark): both heads aligned by their stage-0 landmarks inside the face
  (not the outline or the irises), rigidly and with a scale (a similarity); the distance between corresponding
  landmarks (MediaPipe's 468 of the face mesh) and the distance between the surfaces over the face (inside each face
  oval, both ways), in millimetres, median and mean, also per region (eyes, nose, mouth, cheeks, forehead, outline:
  the face oval; profile: the midline). The similarity measures the shape; the rigid one adds the size.
- **The avatars are split in two halves** (alphabetically, every other one). Everything was tuned on the first
  half (Business_Female_01, Business_Male_02, Business_Male_07, Female_Adult_04, Female_Adult_09, Female_Child_01,
  Male_Adult_04, Male_Adult_10, Male_Adult_14, Male_Adult_21); the test half is the other ten. The size problem
  below was first seen on the test half; its fix was chosen on the tuning half and the test half run again.

## What MediaPipe's landmarks can tell

The fit sees a head through MediaPipe's landmarks. How faithfully do they follow a known change of a head's shape?
Six tuning avatars, each changed by five random sets of identity coefficients around its own (a standard deviation
each), rendered with its own texture from five views (front, 30 and 55 degrees each way); per landmark, the share of
its movement between the avatar's renders and the changed one's that the true change explains (R² through 0), and
the slope:

| Region | Slope | R² | True movement (pixels, RMS) |
|---|---|---|---|
| Eyes | 0.75 | 0.54 | 3.0 |
| Brows | 0.81 | 0.61 | 3.8 |
| Nose | 0.65 | 0.52 | 3.6 |
| Mouth | 0.87 | 0.82 | 5.7 |
| Cheeks | 0.75 | 0.49 | 4.2 |
| Forehead | 0.82 | 0.59 | 4.2 |
| Outline | 0.86 | 0.46 | 6.6 |
| Profile (midline) | 0.70 | 0.54 | 5.1 |

MediaPipe moves its landmarks about three quarters as far as the head changes, and adds errors of about the same
size that the change does not explain; on one avatar alone (Male_Adult_05) the nose followed with a slope of 0.36 and
an R² of 0.16. These errors are systematic, not independent noise: with exact landmarks of a head made from known
coefficients the fit recovers them (correlation 0.96 over the first 20; 0.998 with a weak prior), with 2 pixels of
independent noise still 0.91, but with MediaPipe's landmarks on renders of that head about 0 (D2 below: 0.1-0.2).
The per-landmark R² (over 0.6, between 0.1 and 1) is the fit's trust in each landmark (`landmark_trust.json`).

## Tuning (the first half)

Eight pairs of tuning avatars (same-sex adults, cross-sex, a child onto an adult), D1 as above; medians over the
pairs of the landmark median and the surface median after a similarity alignment, in millimetres:

| Setting | Landmarks | Surface |
|---|---|---|
| The avatar's own head (unfitted) | 2.82 | 1.29 |
| The space; all expressions from MediaPipe's scores | 2.36 | 1.19 |
| ... only the expressions MediaPipe scores 0.3 or more | 2.32 | 1.23 |
| ... the prior centred on the avatar's coefficients | 2.30 | 1.21 |
| ... and three times as firm (**the default**) | 2.29 | 1.13 |
| ... and the free-form correction after it | 2.31 | 1.24 |
| The free-form correction alone (the fallback) | 2.26 | 1.05 |

- Scoring expressions MediaPipe barely sees makes the shape worse: on renders of neutral faces MediaPipe reports
  squints and pressed lips of 0.3-0.6, and fitted they bend the shape.
- The trust weights changed the medians by about 1 %.
- The outline term (the face's outline against the segmenter's masks) changed little: on four tuning pairs with
  profiles, 2.72 mm without it and 2.70 with it (2.81 for the free-form correction alone; 3.36 unfitted).
- Transferring avatar A's own coefficients onto B's head (an "oracle" of the space) is worse than B's own head (3.88
  against 3.12 mm on one pair): between two artist-made heads the artists' detail, not the space's coefficients,
  makes most of the difference, and the space alone cannot reach A. The fit from photos does better than that oracle
  because it fits B's landmarks to A's photos directly.

## The size

Photos cannot tell a head from a larger one farther away; the irises are the ruler (11.7 mm across people). The
first version fitted the irises' size in pixels against their depth as a few more residuals. On the test half the
rigid errors then grew although the shape improved: the head grew with the turned photos. The similarity scale of
the fitted head against the person, relative to the unfitted head's (below 1: grown), and the rigid landmark error
(12 test pairs, medians; the unfitted head's rigid error 2.69 mm):

| Photos | Front only (3) | Front, 30° (3) | ... and 60° (5) | Six | Six, true focal length |
|---|---|---|---|---|---|
| Scale | 1.00 | 0.95 | 0.87 | 0.89 | 0.87 |
| Rigid landmark error (mm) | 3.14 | 3.54 | 7.06 | 6.10 | 6.81 |

MediaPipe's landmarks of turned faces are explained best by a larger head (the ICT modes that make a head bigger
also change its proportions), and two frontal photos' irises did not hold it. Holding the eye distance instead
pins a proportion, not only the size (on the tuning pairs below: 2.35 mm similarity, 3.16 rigid, against 2.29 and
2.98 for the overall size held). The fit now holds the face's overall size (the RMS spread of its landmarks) at a
target, with a uniform scale of the head as an extra unknown, and the coefficients shape it; the target is set every
round from the irises (their radius against the distance between them in the frontal photos: the person's eye
distance, which with the fitted proportions is a size), weighed against the avatar's own size (people's spread 5 %).
Inside the fit the identity modes still make the head about 20 % bigger and the uniform scale takes it back. On six
tuning pairs:

| | Landmarks (similarity) | Landmarks (rigid) | Surface (rigid) | Scale |
|---|---|---|---|---|
| Unfitted | 3.18 | 3.71 | 1.82 | 0.965 |
| Size from the irises (**the default**) | 2.29 | 2.98 | 1.35 | 1.006 |
| The avatar's size | 2.29 | 3.04 | 1.68 | 0.964 |

**The avatars' irises are larger than people's.** Seen by MediaPipe in a frontal render with a known camera, the
adult avatars' irises are 12.0-13.5 mm across (men 13.0-13.5, women 12.0-12.3; the child 10.3), against 11.7 for
people. With avatars as people the irises therefore make the head a few per cent smaller than it is: for the
same-sex test pairs the irises alone said 83-97 % of the avatar's size and the target after the prior was 93-99 %
(for the cross-sex and child pairs 85-108 % and 93-103 %). The rigid numbers below carry that bias; the similarity
numbers do not.

## D1 on the test half

Twenty pairs of the test half: ten of same-sex adults, six cross-sex, four of the child and an adult (either way).
Unfitted -> fitted, millimetres; "median" is the median over the pairs of each pair's median, "mean" the mean of the
means (the mean carries the outline's large errors); the last column counts the pairs whose landmark median got
better, and worse by more than 0.1 mm.

**Six photos, the default** (front, 30 and 60 degrees each way, nodded 15 degrees; focal length unknown):

| Similarity | Landmarks, median | Landmarks, mean | Surface, median | Surface, mean | Better / worse |
|---|---|---|---|---|---|
| Same-sex adults (10) | 2.56 → 2.34 | 4.70 → 4.40 | 1.20 → 1.44 | 1.57 → 1.70 | 6 / 4 |
| Cross-sex (6) | 2.72 → 2.36 | 4.93 → 4.38 | 1.30 → 1.31 | 1.72 → 1.69 | 4 / 2 |
| Child and adult (4) | 2.98 → 2.43 | 4.23 → 3.63 | 1.35 → 1.01 | 1.67 → 1.52 | 3 / 1 |
| **All (20)** | **2.65 → 2.38** | 4.67 → 4.24 | 1.25 → 1.35 | 1.63 → 1.66 | 13 / 7 |

| Rigid | Landmarks, median | Landmarks, mean | Surface, median | Surface, mean | Better / worse |
|---|---|---|---|---|---|
| Same-sex adults (10) | 2.59 → 3.06 | 4.92 → 5.00 | 1.19 → 1.55 | 1.63 → 1.88 | 3 / 7 |
| Cross-sex (6) | 4.82 → 3.45 | 6.94 → 5.70 | 1.89 → 1.66 | 2.38 → 2.14 | 4 / 2 |
| Child and adult (4) | 6.54 → 4.91 | 7.85 → 6.19 | 2.50 → 2.05 | 2.81 → 2.35 | 4 / 0 |
| **All (20)** | **3.83 → 3.19** | 6.11 → 5.45 | 1.48 → 1.61 | 2.09 → 2.05 | 11 / 9 |

Per region (all 20, similarity, medians over the pairs), landmarks and surface:

| | Eyes | Nose | Mouth | Cheeks | Forehead | Outline | Profile |
|---|---|---|---|---|---|---|---|
| Landmarks | 2.1 → 2.1 | 2.8 → 2.3 | 3.0 → 1.8 | 2.3 → 2.7 | 3.9 → 2.6 | 10.0 → 8.4 | 3.0 → 2.5 |
| Surface | 0.90 → 1.03 | 1.40 → 1.27 | 1.40 → 1.07 | 1.10 → 1.71 | 1.81 → 1.32 | 2.06 → 1.76 | 1.34 → 1.13 |

The unfitted heads reproduce the baseline of the plan (2.7 mm for the landmarks; the outline 10.0 mm here, 10.3
there); after a rigid alignment the outline's mean went from 13.3 to 12.4 mm and the profile's from 4.1 to 3.3.

**Sweeps** (all 20 pairs, similarity; landmarks median, surface median, better / worse):

| Photos | Landmarks | Surface | Better / worse |
|---|---|---|---|
| 1 (the front): not fitted, the avatar's own head (two photos are needed) | 2.65 | 1.25 | - |
| 3: the front, 30 degrees each way | **2.21** | **1.13** | 18 / 2 |
| 5: and 60 degrees each way | 2.43 | 1.35 | 13 / 7 |
| 8: and 45 degrees each way | 2.35 | 1.29 | 13 / 7 |
| 3 frontal ones, three lights | 2.37 | 1.28 | 15 / 4 |
| 6, the default | 2.38 | 1.35 | 13 / 7 |
| 6 and both profiles (8) | 2.39 | 1.35 | 13 / 7 |
| 6, the focal length known | 2.57 | 1.46 | 12 / 7 |
| 6 at half the resolution (512x384) | 2.37 | 1.31 | 13 / 6 |
| 6, three of them smiling or surprised | 2.46 | 1.33 | 14 / 5 |

| Method (six photos) | Landmarks | Surface | Rigid landmarks | Better / worse |
|---|---|---|---|---|
| The space (default) | 2.38 | 1.35 | 3.19 | 13 / 7 |
| ... the avatar's size instead of the irises' | 2.38 | 1.35 | - | 13 / 7 |
| ... without the outline term | 2.39 | 1.38 | - | 13 / 7 |
| ... and the free-form correction | 2.38 | 1.30 | - | 12 / 7 |
| The free-form correction alone | 2.36 | 1.18 | 3.23 | 17 / 0 |

- **The gain is smaller than on the tuning half** (there 2.82 → 2.29 and the surface 1.29 → 1.13; here 2.65 →
  2.38 and 1.25 → 1.35): the defaults were chosen there, and the halves differ.
- **Where it helps**: the mouth, the forehead, the nose, the outline and the profile, most for the cross-sex and
  child pairs (where the heads differ most); **where it hurts**: the cheeks (surface 1.10 → 1.71) and, for the
  same-sex adults, the eyes' surroundings. MediaPipe follows the cheeks least faithfully (R² 0.49), and the modes
  tie them to the rest of the face.
- **Photos turned 60 degrees hurt**: three photos (front, 30 degrees each way) gave the best result (2.21, surface
  1.13, 18 of 20 pairs better); adding the 60-degree ones made it worse (2.43). MediaPipe's landmarks of faces turned
  that far are guesses for the far half (the head textures' registration rates such a photo at about 0.25 of a
  frontal one's quality). Weighing them down, or leaving them out of the shape, is the first thing to try (on the
  tuning half).
- **The focal length known made it worse** (2.57 against 2.38): left free, it comes out at about 0.55-0.60 of the
  true one, a stronger perspective that absorbs some of MediaPipe's errors.
- **The free-form correction alone is the safer fit** on the test half: about as good on the landmarks (2.36), better
  on the surface (1.18) and never worse than the avatar's head by more than 0.1 mm (0 of 20), where the space's fit
  was worse for 7. For the size it keeps the avatar's (rigid errors 2.38 for the same-sex adults, 4.69 and 6.49 for
  the cross-sex and child pairs, where the irises help the space's fit: 3.45 and 4.91).
- **The size**: the fitted heads are 93-104 % of the avatar's size; with the irises the rigid errors improve for the
  cross-sex and child pairs and get worse for the same-sex adults (the avatars' large irises, see "The size").
- **Profiles** add nothing measurable here (2.39 against 2.38); the outline term changes the medians by 0.01-0.03 mm
  and costs about 4 s of the fit's 8 s for six photos.

## D2: identities of the space

Random identities (one standard deviation per mode, all 100 modes) laid on an avatar's head, wearing its texture,
six photos, fitted back onto that avatar: the fit at its best, a person the space can express exactly. Medians over
the identities, unfitted -> fitted, millimetres; the correlation of the first 20 fitted coefficients with the true
ones:

| Test half (8: Male_Adult_05, Female_Adult_01, Male_Child_02, Male_Adult_17, two each) | Landmarks (similarity) | Landmarks (rigid) | Surface (similarity) | Surface (rigid) | Correlation |
|---|---|---|---|---|---|
| The space (default) | 2.69 → 2.52 | 3.11 → 4.26 | 1.24 → 1.30 | 1.46 → 1.60 | 0.18 |
| ... the prior centred on ICT's mean head | 2.69 → 2.39 | 3.11 → 4.27 | 1.24 → 1.30 | 1.46 → 1.62 | 0.21 |
| ... without the outline term | 2.69 → 2.53 | 3.11 → 4.25 | 1.24 → 1.31 | 1.46 → 1.59 | 0.17 |
| ... with the two profiles (8 photos) | 2.69 → 2.46 | 3.11 → 4.34 | 1.24 → 1.24 | 1.46 → 1.60 | 0.18 |
| The free-form correction alone | 2.69 → 2.15 | 3.11 → 2.71 | 1.24 → 1.01 | 1.46 → 1.20 | - |

| Tuning half (4: Male_Adult_04, Female_Adult_04, two each) | Landmarks (similarity) | Landmarks (rigid) | Surface (similarity) | Surface (rigid) | Correlation |
|---|---|---|---|---|---|
| The space (default) | 3.94 → 3.37 | 4.93 → 5.33 | 1.84 → 1.67 | 1.96 → 2.22 | 0.14 |
| ... the prior centred on ICT's mean head | 3.94 → 3.06 | 4.93 → 4.95 | 1.84 → 1.62 | 1.96 → 2.15 | 0.13 |
| The free-form correction alone | 3.94 → 3.58 | 4.93 → 4.78 | 1.84 → 1.70 | 1.96 → 2.04 | - |

- The fitted coefficients hardly follow the true ones (correlation 0.1-0.2, against 0.96 with exact landmarks):
  MediaPipe's systematic errors, not the solver, limit the fit. Identities far from the avatar (4-5 mm) come to about
  3 mm; near ones (2-3 mm) stay there or get worse (Male_Adult_05: 2.80 → 3.34 and 2.66 → 3.19; Male_Adult_17:
  1.65 → 2.25), so the fit's floor with these landmarks is about 2-3 mm.
- The two halves disagree on the free-form correction: on the tuning half the space was better (3.37 against 3.58),
  on the test half the free-form correction alone (2.15 against 2.52; surface 1.01 against 1.30). The default stays
  as tuned; the prior centred on ICT's mean head was better on both halves of D2 and slightly worse on D1's tuning.
- The rigid errors grow because these heads wear the avatar's own eyeballs: their irises (12.0-13.5 mm, see "The
  size") make the fitted heads 3-6 % small; the free-form correction keeps the avatar's size.

## As the face recognition sees it

Eight pairs of the test half (A's six renders as the photos; same-sex, cross-sex and child pairs). The gallery:
A's own renders through S1's windows (the "ideal" camera; 112 pixels between the eyes; turned 0 and 20 degrees each
way), for all ten test identities. The probes: avatar B wearing the face textures made from A's renders, once on
B's own head and once on the head fitted to them (the same texture method, `person_textures` against
`fitted_person`), alone in front of the camera, turned 0, 25 and 45 degrees, at 40, 70 and 112 pixels between the
eyes; each probe's similarity to each identity is that of the best of its gallery images. Means over the 72 probes
of each head:

| Engine | Head | Genuine | Impostor | d' | Rank 1 |
|---|---|---|---|---|---|
| AdaFace (IR-101) | B's own | 0.502 | 0.151 | 3.60 | 88 % |
| | fitted | **0.574** | 0.143 | **4.52** | **100 %** |
| ResNet-50 (buffalo_l) | B's own | 0.506 | 0.161 | 3.49 | 99 % |
| | fitted | **0.591** | 0.154 | **5.07** | 100 % |
| SFace | B's own | 0.655 | 0.281 | 4.43 | 100 % |
| | fitted | **0.692** | 0.284 | **4.92** | 100 % |

The genuine score with AdaFace, B's own head → fitted, per turn and eye distance (the other engines alike):

| Turn | 40 pixels | 70 pixels | 112 pixels |
|---|---|---|---|
| 0° | 0.56 → 0.61 | 0.55 → 0.62 | 0.55 → 0.62 |
| 25° | 0.51 → 0.60 | 0.51 → 0.60 | 0.51 → 0.60 |
| 45° | 0.43 → 0.51 | 0.44 → 0.51 | 0.44 → 0.51 |

- The fitted head helps every engine at every size and turn, by 0.07-0.09 for AdaFace and the ResNet-50 and about
  0.04 for SFace, without raising the impostor scores: the photos' textures land on a head of the person's
  proportions instead of being warped onto the avatar's. This is where the shape pays, more than its millimetres
  suggest.
- The rig's checks shrank four of the eight shapes to 75 % (none was dropped); the scores above are with those.
- The sizes hardly matter here (the windows are rendered sharp at every size); the turn does.

## Photos of real people

Four people of the local test folder of photos (two photos each; not in the repository; nothing made from them was
written anywhere), each on a test avatar of their sex: the textures made on the avatar's own head and on the head
fitted to the photos, then renders of both (front, turned 30 degrees) compared by the face recognition with the
person's photos (genuine) and with the other three people's (impostor). Means over the four people:

| Engine | Render | Genuine, avatar's head → fitted | Impostor, avatar's head → fitted |
|---|---|---|---|
| AdaFace (IR-101) | front | 0.415 → 0.447 | -0.011 → -0.006 |
| | turned 30° | 0.323 → 0.347 | -0.005 → 0.000 |
| ResNet-50 (buffalo_l) | front | 0.412 → 0.443 | 0.002 → -0.008 |
| | turned 30° | 0.323 → 0.362 | -0.013 → -0.005 |
| SFace | front | 0.511 → 0.516 | 0.116 → 0.097 |
| | turned 30° | 0.452 → 0.483 | 0.110 → 0.116 |

All four were fitted (the size from the irises in one or two frontal photos: 97-99 % of the avatar's); the fit
lowered the robust landmark error from 1.07 to 0.72 landmark noises on average, and took 7.5 s more than the
textures alone (35 s against 27 s). The genuine scores rise by 0.02-0.04 for every engine, as with the avatars, but
these are the photos the textures were made from (no others were available), so the scores say how much more the
renders look like those photos, not how a new photo of the person would score.

## Costs

On the sandbox's shared 4-core CPU (other jobs running; Mesa llvmpipe for the renders):

| | |
|---|---|
| The space on an avatar's head (S0, once per head model, cached: 73-88 KB) | 14-18 s (the registration 12-16 s; 9-10 s on a quiet machine) |
| The fit, six photos (after registration) | 7.8 s median, 14 s at most (three photos: 2 s; the outline term about 4 s of it) |
| A person's fitted head and its textures, end to end (six photos, 1024 textures) | 55 s median, against 45 s for the textures on the avatar's own head (8 s more; 3-25 s) |
| Making the shape and putting it on the rig (`make_shape`, `apply_shape`) | 0.02 s and 0.03-0.04 s (about 2 MB of memory at the peak) |
| The rig's checks (12 expressions, front and side, 256 pixels) | 2.0-2.4 s |
| `shape.npz` | 21-29 KB |
| A fitted head in memory | 70-100 KB of arrays; 178-266 KB of GPU buffers (the head's 1,700-1,900 vertices at 64 bytes, tangents at 16, indices; and the hair cards or glasses that move with it) |

**The renderer** keys its buffers by mesh part, so a character wearing a fitted head gets its own buffers for the
moved parts only (two more for Male_Adult_05: the head and its hair cards) and shares the body, the clothes and every
texture with the other characters on the same avatar. Two characters on one avatar, one fitted: the unfitted one's
pixels did not change (0 pixels), the fitted one's did (9,267 pixels at 640x480); the first frame with the new head
took 0.155 s against 0.138 s for the next ones (the upload).

## Open issues

- **MediaPipe's landmarks limit the fit.** Their errors are systematic (they follow a change of shape by three
  quarters and add as much again that the change does not explain); the fitted coefficients barely correlate with
  the truth (0.1-0.2) and the fit stops at about 2-3 mm. The gain is in the coarse proportions (mouth, nose, brows,
  profile, jaw); the cheeks get worse and the eyes do not gain. A denser or more faithful face alignment (fitted to the
  photos' pixels, not to a landmark network's guesses) would be the next lever; profiles and silhouettes add little
  so far (the outline term changes the medians by 0.01-0.03 mm and costs about 4 s of the fit's 8 s).
- **The free-form correction** is off by default (on the tuning half it did not help after the space's fit); alone
  it was as good as the space on D1's landmarks on both halves, better on its surface, and better on D2's test half
  (worse on D2's tuning half). Which to trust for real people is open; both are kept (`FitOptions(free=True)`,
  `FitOptions(space=False)`); the space's fit is what made the textures recognisable in the test above, the
  free-form correction's was not measured there.
- **The size from the irises** is right for people only as far as their irises are 11.7 mm; on real photos its
  accuracy is not measured (that needs photos of people whose head size is known). On the avatars the ruler is
  biased (their irises are 12.0-13.5 mm), which the rigid numbers carry.
- **The ears** are not followed: ICT's ears and the avatars' differ too much for the registration, so the ears move
  with their surroundings and keep the avatar's shape; the profiles show the avatar's ears.
- **Large changes fail the rig's checks**: a random identity of 0.7 standard deviations per mode failed them on
  three of four avatars (the lids no longer cover the eyeballs when scrunching or winking; teeth seen from the side
  when cross-eyed); a fitted shape that fails is tried at 75 % and 50 % (four of the eight fitted shapes of the
  recognition test went in at 75 %; D1's numbers are of the full shapes). Corrective shapes per expression would let
  larger changes through.
- **For S4 (close-ups, subdivided heads):** subdivide the fitted head after `apply_shape` (or subdivide
  `shape.offsets` with the same stencils), then turn the normals (`rig.rotate_like`) and make the tangents again
  (`vertex_tangents`) on the subdivided mesh, and run the rig's checks on it (subdivision pulls lids and lips
  inwards); the eyeballs' and bones' moves do not depend on the subdivision. The normal and specular maps are in UV
  space and stay valid; a fitted head made from photos keeps the avatar's detail maps (pores, wrinkles), so detail of
  the person's own would be S4's. A fitted head is a per-character copy (about 0.2 MB of GPU buffers): a subdivided
  one multiplies that by the subdivision's factor (about 4 per level).
- Only four people's photos (two each, no held-out photos) were available for the check on real photos.
