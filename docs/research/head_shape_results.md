# Head shapes from photos: measured results

Step S3 of [face_recognition_focus.md](face_recognition_focus.md): the head's shape fitted to a person's photos.
The design is section 3.2 of [high_res_faces.md](high_res_faces.md); the code is `characters/headshape/` (README:
"Head shapes from photos"). Measured in November 2026 in the development sandbox (a 4-core CPU shared with other
jobs, Mesa llvmpipe), on the 20 prepared Rocketbox avatars of the test library, with ICT-FaceKit Light at commit
`da5f95a` and MediaPipe's Face Landmarker and selfie segmenter. The scratch scripts are not in the repository; the
evaluation's own code is (`characters/headshape/evaluate.py`, `compare.py`).

This note is filled in as the step goes on: so far the diagnostics and the tuning; the test half follows.

## How it is measured

- **D1, avatars as people.** Renders of avatar A (1024x768, a 30-degree lens, 0.9 m from the head, the fair's
  light) stand in for photos; they are registered to avatar B's head and B is fitted to them; the fitted head is
  compared with A's, and so is B's own head (the starting point). Six renders: the front, turned 30 and 60 degrees
  each way, nodded 15 degrees; the focal length is not given (the registration assumes a 50-degree view).
- **Measures** (`compare.py`, as the NoW benchmark): both heads aligned by their stage-0 landmarks inside the face
  (not the outline or the irises), rigidly and with a scale; the median distance between corresponding landmarks
  (MediaPipe's 468 of the face mesh) and the median distance between the surfaces over the face (inside each face
  oval, both ways), in millimetres, also per region.
- **The avatars are split in two halves** (alphabetically, every other one). Everything was tuned on the first
  half (Business_Female_01, Business_Male_02, Business_Male_07, Female_Adult_04, Female_Adult_09, Female_Child_01,
  Male_Adult_04, Male_Adult_10, Male_Adult_14, Male_Adult_21); the results below that are not marked "tuning" are on
  the other half.

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
independent noise still 0.91, but with MediaPipe's landmarks on renders of that head about 0. The per-landmark R²
(over 0.6, between 0.1 and 1) is the fit's trust in each landmark (`landmark_trust.json`).

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
- The outline term (the face's outline against the segmenter's masks) changed nothing measurable with these views
  (no profiles; see below) and costs about 15 s for six photos; it stays an option.
- Transferring avatar A's own coefficients onto B's head (an "oracle" of the space) is worse than B's own head (3.88
  against 3.12 mm on one pair): between two artist-made heads the artists' detail, not the space's coefficients,
  makes most of the difference, and the space alone cannot reach A. The fit from photos does better than that oracle
  because it fits B's landmarks to A's photos directly.
- On random identities of the space laid on an avatar's head (D2, three of them on Male_Adult_05, the default of
  the time), the space's fit brought the landmark median from 4.2-5.0 mm to 2.7-3.2 and the surface median from
  1.9-2.4 to 1.2-1.4, and the free-form correction alone only to 3.4-3.9 and 1.6-2.0: where the person is a head the
  space can express, the space is what helps. Real people are what the space was built from; the avatars are
  stylised, partly outside it, which favours the conservative free-form correction in D1.
