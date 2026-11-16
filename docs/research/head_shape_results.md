# Head shapes from photos: measured results

Step S3 of [face_recognition_focus.md](face_recognition_focus.md): the head's shape fitted to a person's photos.
The design is section 3.2 of [high_res_faces.md](high_res_faces.md); the code is `characters/headshape/` (README:
"Head shapes from photos"). Measured in November 2026 in the development sandbox (a 4-core CPU shared with other
jobs, Mesa llvmpipe), on the 20 prepared Rocketbox avatars of the test library, with ICT-FaceKit Light at commit
`da5f95a` and MediaPipe's Face Landmarker and selfie segmenter. The scratch scripts are not in the repository; the
evaluation's own code is (`characters/headshape/evaluate.py`, `compare.py`). Photos of real people were used for
checks only, in memory; only aggregate numbers are reported.

**In short.** On avatars as people (the test half, 20 pairs, six photos), the fitted head's landmarks are closer to
the person's than the avatar's own head's: median 2.65 → 2.17 mm after a similarity alignment (3.83 → 3.00 rigid),
and the surface over the face 1.25 → 1.10 mm; 18 of the 20 pairs are better, one worse. Most of the gain is in the
mouth, the forehead, the outline, the profile and the nose. These are the numbers after an independent review's
fixes (see "After the review"): before them, the landmarks of photos turned 60 degrees pulled the cheeks out of shape
(the surface got worse, 1.35 mm) and the fit overshot. On random identities of the space laid on an avatar (D2,
before the fixes) the fit helps where the person is far from the avatar and stops at about 2-3 mm: MediaPipe's
systematic errors, not the solver, are the limit. The size had to be taken out of the fit (it made the heads 5-15 %
too big). As a probe for the face recognition, the fitted head wearing the person's face textures scores higher than
the avatar's own head with textures made the same way: over eight pairs, AdaFace 0.502 → 0.585 (rank 1 88 → 100 %),
the ResNet-50 0.506 → 0.597, SFace 0.655 → 0.698, the impostor scores no higher; pair by pair in 23 of 24
(AdaFace loses a little on the child as the person). A fitted head costs about 0.2 MB of GPU buffers per character
and 0.03 s to put on the rig.

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

## After the review

An independent review (November 2026) confirmed the measurements pair by pair and the Jacobians numerically, and
found five problems of some weight. What changed, and the numbers before and after:

**Photos turned 60 degrees made the fit worse, and caused the cheek loss.** MediaPipe guesses the far half of a face
turned that far. Now the landmarks of photos turned more than 45 degrees (`FitOptions.turned`) place their camera but
do not shape the head; their outline still does. On the tuning half (eight pairs, six photos; medians, mm):

| Landmarks of the turned photos | Landmarks | Surface | Cheeks (landmarks) | Cheeks (surface) | Rigid |
|---|---|---|---|---|---|
| All counted (before) | 2.29 | 1.17 | 2.37 | 1.56 | 2.98 |
| Weighted by cos² | 2.20 | 1.04 | 2.09 | 1.17 | 2.93 |
| Dropped beyond 40° | 2.01 | 1.00 | 1.90 | 1.08 | 2.73 |
| **Dropped beyond 45°** | **2.00** | **1.00** | **1.97** | **1.13** | **2.71** |
| Dropped beyond 50° | 1.97 | 0.97 | 1.94 | 1.09 | 2.75 |
| Dropped beyond 45°, the fitted change kept at 85 % (**the default**) | **1.99** | **0.97** | **1.90** | **1.08** | **2.57** |
| Dropped beyond 45°, the fitted change kept at 70 % | 2.06 | 0.97 | 1.92 | 1.05 | 2.58 |
| Three photos only (front, 30° each way) | 2.04 | 0.95 | 1.92 | 1.07 | 2.51 |

The unfitted heads: 3.18, 1.48, 2.90, 1.20, 3.71. **The fit is damped deliberately** (`FitOptions.damping`, 0.85):
the review found the shapes that went on the rig shrunk by its checks nearer the person than the full fits, and so
they are on the tuning half; the size is not damped (the uniform scale is solved again for the size found).

**The rig's checks were blunt.** They missed a lower lip pulled 3 mm behind the teeth (the teeth were compared only
with the frontmost lip point, the upper lip's) and judged an open mouth seen from the side as shut (a false alarm),
and any failure shrank the whole head. Now the lips' front (where the avatar's lip is 1 mm or more in front of the
tooth behind it, paired in each face as the jaw moves) may not come behind that tooth (the lips' inside, against the
teeth already, is hidden whatever it does: counted too, it kept 4 of 12 D1 shapes smaller at the mouth for nothing
to see), the mouth is judged shut from the front, the eyes are counted apiece, each problem is put down to a part of
the head, and a failing shape is kept smaller near that part only (75, 50, 25 % down to the avatar's own shape
there; the whole head only for what no part explains); what was kept smaller is stored in `shape.npz`. Heads broken
on purpose (after the review's script; four avatars: Male_Adult_05, Male_Child_02, Business_Female_02,
Male_Adult_10):

| Defect | Caught before (the review's three) | Caught now (part) |
|---|---|---|
| An eyeball 3 mm forward; the lids sunk 3 mm | 3 of 3 | 4 of 4 (that eye) |
| The teeth 4 mm forward | 3 of 3 | 4 of 4 (the mouth) |
| The teeth 2 mm forward | 2 of 3 (not the child) | 3 of 4 (the mouth; on Male_Adult_10 they stay behind the lips' front) |
| The lower lip 3 mm back | 0 of 3 | 4 of 4 (the mouth) |
| The lower lip 4 mm down (the teeth shown) | not tried | 4 of 4 (the mouth) |
| The neck's edge 0.5 mm up; a fold | 3 of 3 | 4 of 4 (the neck; the fold) |
| Nothing changed | passes | passes |

**Folds.** `make_shape` untangled the skin before moving the whole head, which set the tied lips and the skin with no
counterpart in the space from their partners again; and a near-degenerate triangle at a mouth corner stayed folded
however long its one-ring was smoothed. Now the skin is untangled again after the transfer (and kept so), and a fold
that stays widens its smoothed region ring by ring: random identities of one standard deviation per mode (eight on
each of the four avatars) left a folded triangle in 3 of 32 shapes with the second untangling alone, now in none.

**The focal lengths.** The fit's focal lengths (0.34-0.66 of the truth: a means of the fit, absorbing MediaPipe's
errors) were handed to the texture registration as known; now only EXIF or given ones are, and a session's focal
length is relative to the picture's size (photos of one camera at two resolutions share it).

**The neck.** The fade towards the neck's edge went straight through space, so a short neck held the jaw line. By
the review's measure (the lowest fade over each landmark's triangle) six of the 20 avatars had a landmark below 0.9:
Business_Female_02 0, Business_Male_02 0.21, Female_Child_01 0.57, Female_Adult_06 0.77, Female_Adult_09 0.78,
Male_Adult_10 0.88. The fade now goes along the skin and is done by the jaw line (over 2 cm at least): 0.97, 0.94,
0.91 and 0.92 for the last four. Two stay below 0.9: Business_Female_02's head mesh ends at its jaw line (the seam
to the body; 0 at four landmarks of the face oval), and on Business_Male_02 landmark 365 lies on a long triangle
reaching down the neck (one corner at 0.28; the landmark itself now moves at 0.92 of the change, 0.80 before). (A
first version shortened Business_Female_02's fade to 1 cm, and smoothing the squeezed skin then moved its seam: its
shapes failed the checks. Now the neck's edge never moves, whatever is smoothed: on four avatars, 64 random
identities move it by 0.0000 mm.)

**The caches.** A fitted person is kept under a key that holds the fit's and the head space's code versions, the
fit's options, the identity space and the shipped landmarks and trust, so a changed fit makes the heads again; the
textures and their shape are written as one; a fitted head's atlas stays in memory. A created character whose shape
file cannot be read keeps its avatar's head, with a note, instead of being unreadable.

## D1 on the test half

Twenty pairs of the test half: ten of same-sex adults, six cross-sex, four of the child and an adult (either way).
Unfitted → fitted, millimetres; "median" is the median over the pairs of each pair's median, "mean" the mean of the
means (the mean carries the outline's large errors); "better / worse" counts the pairs whose landmark median got
better, and worse by more than 0.1 mm; "before the review" is the fitted landmark median with the defaults before
the review's fixes (the turned photos' landmarks counted, no damping).

**Six photos, the default** (front, 30 and 60 degrees each way, nodded 15 degrees; focal length unknown):

| Similarity | Landmarks, median | Landmarks, mean | Surface, median | Surface, mean | Better / worse | Before the review |
|---|---|---|---|---|---|---|
| Same-sex adults (10) | 2.56 → 2.10 | 4.70 → 4.17 | 1.20 → 1.16 | 1.57 → 1.46 | 9 / 1 | 2.34 |
| Cross-sex (6) | 2.72 → 2.11 | 4.93 → 4.22 | 1.30 → 1.10 | 1.72 → 1.51 | 6 / 0 | 2.36 |
| Child and adult (4) | 2.98 → 2.36 | 4.23 → 3.65 | 1.35 → 1.01 | 1.67 → 1.46 | 3 / 0 | 2.43 |
| **All (20)** | **2.65 → 2.17** | 4.67 → 4.08 | **1.25 → 1.10** | 1.63 → 1.47 | **18 / 1** | 2.38 |

| Rigid | Landmarks, median | Landmarks, mean | Surface, median | Surface, mean | Better / worse | Before the review |
|---|---|---|---|---|---|---|
| Same-sex adults (10) | 2.59 → 2.91 | 4.92 → 4.81 | 1.19 → 1.40 | 1.63 → 1.67 | 3 / 6 | 3.06 |
| Cross-sex (6) | 4.82 → 3.14 | 6.94 → 5.58 | 1.89 → 1.44 | 2.38 → 1.96 | 4 / 2 | 3.45 |
| Child and adult (4) | 6.54 → 5.18 | 7.85 → 6.48 | 2.50 → 2.19 | 2.81 → 2.36 | 4 / 0 | 4.91 |
| **All (20)** | **3.83 → 3.00** | 6.11 → 5.37 | 1.48 → 1.46 | 2.09 → 1.89 | 11 / 8 | 3.19 |

Per region (all 20, similarity, medians over the pairs; in brackets before the review):

| | Eyes | Nose | Mouth | Cheeks | Forehead | Outline | Profile |
|---|---|---|---|---|---|---|---|
| Landmarks | 2.1 → 1.9 (2.1) | 2.8 → 2.5 (2.3) | 3.0 → 1.6 (1.8) | 2.3 → 2.0 (2.7) | 3.9 → 2.8 (2.6) | 10.0 → 8.3 (8.4) | 3.0 → 2.4 (2.5) |
| Surface | 0.90 → 0.93 (1.03) | 1.40 → 1.28 (1.27) | 1.40 → 1.10 (1.07) | 1.10 → 1.09 (1.71) | 1.81 → 1.19 (1.32) | 2.06 → 1.99 (1.76) | 1.34 → 1.14 (1.13) |

The unfitted heads reproduce the baseline of the plan (2.7 mm for the landmarks; the outline 10.0 mm here, 10.3
there). **As the shapes go on the rig** (untangled, moved onto the whole head, checked, and kept smaller near
a part that fails): landmarks 2.18, surface 1.08, rigid 2.98 mm, the same as the fits themselves; 3 of the 20 were
kept smaller, each at 75 % near one part (the mouth twice, both eyes once), none as a whole (before the review four of
the eight recognition shapes went on whole at 75 %).

**Sweeps and methods** (all 20 pairs, similarity; landmarks median, surface median, better / worse):

| Photos | Landmarks | Surface | Better / worse | Before the review |
|---|---|---|---|---|
| 1 (the front): not fitted, the avatar's own head (two photos are needed) | 2.65 | 1.25 | - | 2.65 |
| 3: the front, 30 degrees each way | 2.19 | 1.07 | 18 / 1 | 2.21 |
| 5: and 60 degrees each way | 2.17 | 1.07 | 18 / 1 | 2.43 |
| 8: and 45 degrees each way | 2.17 | 1.14 | 19 / 1 | 2.35 |
| 3 frontal ones, three lights | 2.36 | 1.19 | 17 / 1 | 2.37 |
| 6, the default | **2.17** | **1.10** | 18 / 1 | 2.38 |
| 6 and both profiles (8) | 2.17 | 1.07 | 19 / 1 | 2.39 |
| 6, the focal length known | 2.27 | 1.24 | 17 / 1 | 2.57 |
| 6 at half the resolution (512x384) | 2.16 | 1.07 | 18 / 1 | 2.37 |
| 6, three of them smiling or surprised | 2.23 | 1.14 | 17 / 2 | 2.46 |

| Method (six photos) | Landmarks | Surface | Better / worse |
|---|---|---|---|
| The space (default) | 2.17 | 1.10 | 18 / 1 |
| ... as before the review (the turned photos' landmarks counted, no damping) | 2.37 | 1.35 | 13 / 7 |
| ... the avatar's size instead of the irises' | 2.17 | 1.09 | 19 / 1 |
| ... without the outline term | 2.17 | 1.11 | 19 / 1 |
| ... and the free-form correction | 2.24 | 1.20 | 17 / 2 |
| The free-form correction alone | 2.37 | 1.18 | 17 / 0 |

- **The review's fixes** took the six photos' landmark median from 2.38 to 2.17 mm and the surface from 1.35 (worse
  than the avatar's own head) to 1.10 (better); 18 of 20 pairs are now better and one worse (13 and 7). The cheeks
  no longer lose (surface 1.10 → 1.09, against 1.71 before); the turned photos no longer hurt (three, five, six and
  eight photos all give 2.17-2.19); the same run with the old defaults gave 2.37, as before.
- **Where it helps**: the mouth (3.0 → 1.6), the forehead, the outline, the profile, the nose; the eyes' and the
  cheeks' landmarks now too. **Where not**: the eyes' surface (0.90 → 0.93); the outline's surface gains less than
  before (2.06 → 1.99; 1.76 when the 60-degree photos' landmarks pulled the jaw line); the forehead's landmarks
  gained a little more before (2.6 against 2.8).
- **The size** (rigid): better for the cross-sex and child pairs (4.82 → 3.14, 6.54 → 5.18), worse for the same-sex
  adults (2.59 → 2.91), whose size the avatars' large irises bias (see "The size").
- **The focal length known** is still a little worse than left free (2.27 against 2.17).
- **The free-form correction alone** (2.37, surface 1.18) is now behind the space's fit on the landmarks and the
  surface, on the landmarks of every region (the review's table, before its fixes, had it ahead on the eyes and the
  cheeks) and on the surface but for the eyes and the nose; added to the space's fit, it makes it worse. Medians over
  the 20 pairs, similarity, mm:

  | | Eyes | Nose | Mouth | Cheeks | Forehead | Outline | Profile |
  |---|---|---|---|---|---|---|---|
  | Landmarks, the space | **1.88** | **2.51** | **1.64** | **1.98** | **2.77** | **8.33** | **2.38** |
  | Landmarks, free-form alone | 1.99 | 2.74 | 2.28 | 2.08 | 3.35 | 9.37 | 2.85 |
  | Surface, the space | 0.93 | 1.28 | **1.10** | **1.09** | **1.19** | **1.99** | **1.14** |
  | Surface, free-form alone | **0.85** | **1.25** | 1.24 | 1.30 | 1.34 | 2.21 | 1.19 |
- **Profiles** add nothing measurable (2.17 with and without); the outline term changes nothing measurable either
  and costs about 2.4 s of the fit's 5.6 s (median; 7.9 s at most).

## D2: identities of the space

Random identities (one standard deviation per mode, all 100 modes) laid on an avatar's head, wearing its texture,
six photos, fitted back onto that avatar: the fit at its best, a person the space can express exactly. (Measured
before the review's fixes, with the defaults of then; not repeated.) Medians over
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
of each head (in brackets the fitted head before the review's fixes):

| Engine | Head | Genuine | Impostor | d' | Rank 1 |
|---|---|---|---|---|---|
| AdaFace (IR-101) | B's own | 0.502 | 0.151 | 3.60 | 88 % |
| | fitted | **0.585** (0.574) | 0.142 (0.143) | **4.58** (4.52) | **100 %** (100 %) |
| ResNet-50 (buffalo_l) | B's own | 0.506 | 0.161 | 3.49 | 99 % |
| | fitted | **0.597** (0.591) | 0.152 (0.154) | **4.98** (5.07) | 100 % (100 %) |
| SFace | B's own | 0.655 | 0.281 | 4.43 | 100 % |
| | fitted | **0.698** (0.692) | 0.281 (0.284) | **5.00** (4.92) | 100 % (100 %) |

The genuine score with AdaFace, B's own head → fitted, per turn and eye distance (the other engines alike):

| Turn | 40 pixels | 70 pixels | 112 pixels |
|---|---|---|---|
| 0° | 0.56 → 0.62 | 0.55 → 0.63 | 0.55 → 0.63 |
| 25° | 0.51 → 0.61 | 0.51 → 0.61 | 0.51 → 0.61 |
| 45° | 0.43 → 0.51 | 0.44 → 0.52 | 0.44 → 0.52 |

- The fitted head helps every engine at every size and turn on average, by 0.08-0.09 for AdaFace and the ResNet-50
  and about 0.04 for SFace, without raising the impostor scores: the photos' textures land on a head of the person's
  proportions instead of being warped onto the avatar's. This is where the shape pays, more than its millimetres
  suggest. The review's fixes added a little (AdaFace 0.574 → 0.585, the ResNet-50 0.591 → 0.597, SFace 0.692 →
  0.698; the ResNet-50's d' 5.07 → 4.98, its genuine scores a little more spread), and the rig's checks now keep
  none of the eight shapes smaller (before, four went on whole at 75 %).
- **Not in every pair**: pair by pair (and engine) it helps in 23 of 24. AdaFace still loses on the child as the
  person (Male_Child_02's renders on Male_Adult_05: 0.417 → 0.405, −0.033 before the review), where the ResNet-50
  and SFace gain (0.379 → 0.458, 0.611 → 0.626). The smallest gains: Female_Adult_01's renders on Female_Adult_06
  (0.01-0.03) and SFace on Male_Adult_12's renders on Male_Adult_17 (+0.008, −0.017 before); the largest, 0.13-0.15
  for AdaFace and the ResNet-50, on Business_Male_05's renders on Male_Adult_05, Female_Adult_06's on Male_Child_02
  and Female_Party_02's on Business_Female_02.
- **Against the free-form correction alone** (the review's measurement, the same protocol and renders, before its
  fixes; the space's column is the fit of then):

  | Engine | Unfitted | The space (genuine, d') | Free-form alone (genuine, d') |
  |---|---|---|---|
  | AdaFace (IR-101) | 0.502, 3.60 | **0.574, 4.52** | 0.538, 3.92 |
  | ResNet-50 (buffalo_l) | 0.506, 3.49 | **0.591, 5.07** | 0.549, 3.94 |
  | SFace | 0.655, 4.43 | 0.692, 4.92 | 0.681, 4.91 |

  The free-form correction gains about half of what the space gains for AdaFace and the ResNet-50 (the space is
  ahead in 7 of 8 pairs for both, and in all nine turn and size cells for AdaFace), about as much for SFace; its
  3 mm cap moved heads 2.7 mm at most, the space 8-31 mm; and it failed the rig's checks on one of the eight. On the
  child as the person it lost too (AdaFace −0.012). The space stays the default.
- The sizes hardly matter here (the windows are rendered sharp at every size); the turn does.

## Photos of real people

Four people of the local test folder of photos (two photos each; not in the repository; nothing made from them was
written anywhere), each on a test avatar of their sex: the textures made on the avatar's own head and on the head
fitted to the photos, then renders of both (front, turned 30 degrees) compared by the face recognition with the
person's photos (genuine) and with the other three people's (impostor). Means over the four people (in brackets
the fitted head before the review's fixes):

| Engine | Render | Genuine, avatar's head → fitted | Impostor, avatar's head → fitted |
|---|---|---|---|
| AdaFace (IR-101) | front | 0.415 → 0.449 (0.447) | -0.011 → -0.008 (-0.006) |
| | turned 30° | 0.323 → 0.355 (0.347) | -0.005 → -0.003 (0.000) |
| ResNet-50 (buffalo_l) | front | 0.412 → 0.446 (0.443) | 0.002 → -0.007 (-0.008) |
| | turned 30° | 0.323 → 0.354 (0.362) | -0.013 → -0.005 (-0.005) |
| SFace | front | 0.511 → 0.525 (0.516) | 0.116 → 0.100 (0.097) |
| | turned 30° | 0.452 → 0.489 (0.483) | 0.110 → 0.112 (0.116) |

All four were fitted (the size from the irises in one or two frontal photos: 97-99 % of the avatar's); the fit
lowered the robust landmark error from 1.07 to 0.72 landmark noises on average, and took 13 s more than the textures
alone (43 s against 29 s; 7.5 s more before the review, on the shared machine). The genuine scores rise by 0.01-0.04
for every engine, as with the avatars (the review's fixes changed them by -0.008 to +0.009), but these are the photos
the textures were made from (no others were available), so the scores say how much more the renders look like those
photos, not how a new photo of the person would score.

## Costs

On the sandbox's shared 4-core CPU (other jobs running; Mesa llvmpipe for the renders):

| | |
|---|---|
| The space on an avatar's head (S0, once per head model, cached: 73-88 KB) | 14-18 s (the registration 12-16 s; 9-10 s on a quiet machine) |
| The fit, six photos (after registration) | 5.6 s median, 7.9 s at most (three photos: 1.6 s; with both profiles 7.3 s; the outline term about 2.4 s of it; before the review 7.8 s) |
| A person's fitted head and its textures, end to end (six photos, 1024 textures) | 52 s median, against 41 s for the textures on the avatar's own head (per pair 9-23 s more, 15 s median; before the review 55 s against 45 s, 3-25 s more); in one pair the shape's stages took 8.7 s (the registration 1.8 s, the fit 5.7 s, the rig and its checks 1.3 s) and the fitted head's texture atlas 9.4 s |
| Making the shape and putting it on the rig (`make_shape`, `apply_shape`) | 0.02 s and 0.01-0.04 s (about 2 MB of memory at the peak) |
| The rig's checks (12 expressions, front and side, 256 pixels) | 1.0-1.6 s on three avatars (2.0-2.4 s measured before the review); a shape kept smaller is checked again at each step |
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
  the truth (0.1-0.2) and the fit stops at about 2 mm. A denser or more faithful face alignment (fitted to the
  photos' pixels, not to a landmark network's guesses) would be the next lever; profiles and silhouettes add nothing
  measurable so far (the outline term costs about 2.4 s of the fit's 5.6 s).
- **The outline's surface** gains less than before the review (2.06 → 1.99 mm, 1.76 when the 60-degree photos'
  landmarks shaped the jaw line): the outline term does not take their place. Their jaw-line landmarks alone (the
  contour, not the far half) might, on the tuning half first.
- **The size from the irises** is right for people only as far as their irises are 11.7 mm; on real photos its
  accuracy is not measured (that needs photos of people whose head size is known). On the avatars the ruler is
  biased (their irises are 12.0-13.5 mm): the same-sex pairs' rigid errors grow (2.59 → 2.91 mm). The review's
  suggestion, a clamp of the scale to about ±8 % besides the 5 % prior, is not done (the heads came out 93-104 % of
  the avatar's).
- **The ears** are not followed: ICT's ears and the avatars' differ too much for the registration, so the ears move
  with their surroundings and keep the avatar's shape (the windows up to 45 degrees show little of them).
- **Business_Female_02's jaw line** stays where it is: its head mesh ends there (the seam to the body).
- **Large changes**: a random identity of 0.7 standard deviations per mode failed the checks on three of four
  avatars before the review (the lids, scrunching or winking); now such a head is kept smaller near the eyes only.
  Corrective shapes per expression (or the lids' closure scaled with a changed eyeball) would let more through.
- **For S4 (close-ups, subdivided heads)**, after the review's notes: subdivide the fitted head after `apply_shape`,
  or subdivide `shape.offsets` with the same stencils (subdivision is affine-invariant, so the eyeballs' and teeth's
  rigid moves carry over); carry the per-vertex labels the checks need (`parts`, `eye`, `inner`, `fade`, the lids)
  to the subdivided mesh and run the checks on it; the closed lips are joined by ties, not edges, so smoothing opens
  them unless the lip line is welded or creased first; keep the neck's edge exactly fixed (boundary rules); turn the
  normals (`rig.rotate_like`) and make the tangents again (`vertex_tangents`); the shrink is stored with the shape.
  The normal and specular maps are in UV space and stay valid; a fitted head made from photos keeps the avatar's
  detail maps (pores, wrinkles): detail of the person's own would be S4's. A fitted head is a per-character copy
  (about 0.2 MB of GPU buffers); a subdivided one multiplies that by about 4 per level.
- Only four people's photos (two each, no held-out photos) were available for the check on real photos.
