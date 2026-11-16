# Face recognition: results so far

Where the face-recognition work stands, as of 16 November 2026, with the numbers measured so far.

- **The plan and the full log:** [face_recognition_focus.md](face_recognition_focus.md).
- **Each step's details** are in its results note, listed at the end.
- **Where it was measured:** a shared 4-core CPU with software OpenGL, so timings are slow and noisy. The Mac's speeds
  are not measured yet.
- **No real people:** every image in the repository shows avatars only. Results on photos of real people are aggregate
  numbers only.

## In short

| Step | What it is | State |
|---|---|---|
| F1 | Recognition core: models, profiles, detection, alignment, quality, gallery | Done: built, reviewed, fixed |
| S1 | Virtual high-resolution sensor: close-up windows on heads | Done |
| S2 | High-resolution heads: 2048 textures, normal and specular maps, linear light | Done |
| F2 | From a person to an identity: tracks, faces, evidence, calibrated decisions, demo | Done |
| S3 | Head shape fitted to a person's photos | Built and reviewed. Review fixes 1-4 are on the branch, with the held-out numbers; the final checks are under way |
| E1 | Evaluation: metrics, protocols, baselines, acceptance test | Started, paused before its first commit |
| S4 | Close-ups: finer head meshes, skin and eye shading | Not started |

**The system today.**

1. People are detected and followed:
   - YOLO, or YOLOX in the commercial profile;
   - a tracker of our own.
2. Faces are looked for only in each person's head region:
   - in a close-up window of a virtual 8K sensor while the person is undecided;
   - else in the frame.
3. Faces good enough are embedded within a budget the machine can afford.
4. Each person gets one of three states: a name, "unknown", or "not decided yet".
   - The decision is calibrated, per model, face size and head turn, against one false-alarm target per stranger
     followed.
   - A wrong name is treated as the worst outcome.

## Profiles

| Profile | Person detector | Face detector | Recogniser | Licence of the weights |
|---|---|---|---|---|
| best | YOLO26n | SCRFD-10G | AdaFace IR-101 (with eDifFIQA-T quality) | research only |
| licensed | YOLO26n | SCRFD-10G | `buffalo_l` ResNet-50 | commercial licence from InsightFace |
| commercial | YOLOX-s | YuNet | AuraFace once calibrated, else SFace | permissive; no AGPL code |

The recognisers' speeds here, per face:
- AdaFace: 0.5-1 s;
- the ResNet-50: 0.1-0.2 s;
- SFace: about 15 ms.

## Results by step

### F1, the recognition core

- **The models reproduce their published accuracy** on an LFW subset of 1,500 pairs:

  | Model | Here | Published |
  |---|---|---|
  | AdaFace | 99.87 % | 99.82 |
  | ResNet-50 | 99.87 % | 99.83 |
  | SFace | 99.47 % (99.60 on all of LFW) | 99.60 |

- **On renders of the 20 avatars** (540 renders), separation between people:

  | Model | d' | Equal error rate |
  |---|---|---|
  | AdaFace | 5.42 | 0.20 % |
  | ResNet-50 | 5.31 | 0.12 % |
  | SFace | 4.62 | 1.07 % |

- **Thresholds** are counted on all 3,000 LFW stranger pairs, not estimated.
- **Review:** 6 medium and 11 low findings, all fixed.

### S1, the virtual high-resolution sensor

- **Exact crops.** Windows are exact crops of a 7680x4320 sensor behind the 1280x720 view: 98.7-99.9 % of the pixels
  identical.
- **Pixels between the eyes,** for a frontal face:

  | Distance | In a window | In the view |
  |---|---|---|
  | 3 m | 115 px | 19 px |
  | 4 m | 75 px | 12.5 px |
  | 6 m | 46 px | 7.7 px |
  | 10 m | 26 px | 4.3 px |

- **Face detection:** YuNet found 36 of 36 frontal faces at 2-30 m in windows, against 20 of 36 in the view.

### S2, high-resolution heads

- **The maps matter more than the 2048 textures.** Normal and specular maps raise AdaFace's separation of the avatars
  by d' 0.5-0.65 with realistic cameras and turned or lowered heads. 2048 textures add nothing at fair distances; they
  are for close-ups.
- **Faces from photos keep their identity.** Painted on an avatar, they are drawn without the avatar's own skin maps.
  AdaFace's same-person score is 0.359, against 0.275 before the review's fix and 0.363 with flat shading.
- **The 2048 textures stream as needed.** With 8 people near the camera: 6 texture reads instead of 96, and a frame
  with its windows takes 341 ms instead of 2.6 s.
- **Review:** 1 high, 2 medium and 7 low findings, fixed. The exception is painted hair looking glossy from behind,
  which is documented.

### F2, from a person to an identity

- **False alarms.** The calibrations were remade after the review from 87 avatars that are not in the fair, about 72
  identities. At the default target of 1 % per stranger followed, measured on held-out avatars:

  | Recogniser | Measured | 90 % interval |
  |---|---|---|
  | AdaFace | 0.70 % | 0.33-1.15 % |
  | ResNet-50 | 0.52 % | 0.18-0.90 % |
  | SFace | 0.61 % | 0.08-1.26 % |

  The strictest target, 0.1 %, is not met for AdaFace (0.25 %) and SFace (0.34 %), because of a few look-alike avatar
  pairs.
- **In the fair,** after the review, one run per profile with 5 known people among strangers and windows on:
  - no wrong name, and no stranger named;
  - the best profile names people after a median of 3.0 s (5.0 s before the review);
  - the commercial profile named 2 of 5 known people within 2 s. With the first calibrations it named 5 of 5, but
    those let through about twice the false alarms asked. **Naming speed against false alarms is the main open
    trade-off.**
- **Real faces** (LFW, numbers only). The avatar-made thresholds are cautious on real people: at 1 %, no real stranger
  pair passes for SFace or the ResNet-50, but 34 % and 49 % of same-person pairs fall below the threshold. A person
  followed for a while gives several faces, so per-person naming should be better; E1 is to measure it.
- **People detection:**
  - YOLOX-s finds 66 % of people 40-80 px tall, against 43 % for YOLO26n, at 264 ms a frame here against 110 ms;
  - the tracker follows 60-83 % of person-frames over 30 s of the fair, depending on the detector.
- **Review:** 4 high, 9 medium and 6 low findings, all fixed. The reviewer's reproduction scripts all pass on the
  branch. The full suite: 1034 passed, 8 skipped.

### S3, the head shape fitted to the person

- **What it does.** An openly licensed space of face shapes (ICT-FaceKit, MIT) is carried onto every avatar head. It
  is fitted to all of a person's photos at once, the fitted head goes onto the avatar's rig (eyes, teeth and face bones
  follow), and the face textures are made on the fitted head. Known faces, created characters and the creator use it.
- **It helps recognition clearly.** In 8 tests, one avatar stood in for a person and another avatar wore their face;
  each test had 72 probes through the windows. The fitted head against the avatar's own:

  | Model | d', own head | d', fitted head |
  |---|---|---|
  | AdaFace | 3.60 | 4.52 |
  | ResNet-50 | 3.49 | 5.07 |
  | SFace | 4.43 | 4.92 |

  AdaFace's rank-1 went from 88 % to 100 %. The face-shape space beats a simpler free-form fit alone (AdaFace d' 4.52
  against 3.92), so it stays the default. The one exception is a child standing in for a person on an adult avatar:
  its same-person score drops a little (AdaFace -0.03).
- **The geometry improves, and more since the review's fixes.** The review found that photos turned 60 degrees made
  the fit worse, notably at the cheeks. Their landmarks are now set aside (their outline still counts), the fit is
  damped, and the rig's checks shrink a shape only near a problem. On the held-out avatars (20 pairs, six photos):

  | Measure (mm) | Avatar's own head | Fitted, first version | Fitted, after the review |
  |---|---|---|---|
  | Landmarks | 2.65 | 2.38 | 2.17 |
  | Surface | 1.25 | 1.35 | 1.10 |
  | Cheek surface | 1.10 | 1.71 | 1.09 |

  18 of 20 pairs are better, against 13 before. AdaFace's same-person score rose from 0.502 with the avatar's own head
  to 0.585, against 0.574 with the first version. Of the 24 pair-and-model cases, 23 gain; the child is the
  exception.
- **Real photos,** checked locally with numbers only: same-person scores rose by 0.02-0.04. These were the same photos
  the textures were made from.
- **Review:** no high findings, 5 medium and 11 low. Fixes 1-4 are on the branch: turned photos, sharper and local rig
  checks, focal lengths, caches, tests, the neck seam, the numbers above.

## Open issues

1. **Naming speed against false alarms.**
   - The honest calibration margin slows naming.
   - E1 should measure naming rate against measured false alarms and recommend the default.
2. **Real faces.** The shipped calibrations come from avatars. For real cameras, a local calibration from consenting
   people is needed, on the Mac.
3. **The best profile on this CPU.**
   - About one AdaFace embedding a second, so a crowd waits.
   - The Mac's speed (CoreML) is not measured.
4. **AuraFace,** the commercial profile's intended recogniser, is not calibrated: it needs Hugging Face, which is
   unreachable here. Until then the commercial profile uses SFace.
5. **Tracking** has no re-identification: a person who leaves the picture for over a second comes back as a new
   track.
6. **Head shapes:**
   - the ears keep the avatar's shape;
   - the iris-based size is not validated on real heads;
   - a child standing in for a person on an adult avatar loses a little, the one exception measured.

## Resuming

1. **S3: the review fixes' final checks** (the full suite). Its builder is on them.
2. **E1, the evaluation.** Its builder works from its brief and commits results as they come.
   - metrics with intervals over people;
   - `bifold-faces eval` with manifests;
   - rendered stills and video;
   - the ladder from the 720p view to fitted heads;
   - the acceptance test;
   - the margin;
   - a guide for real-person checks on the Mac.
3. **S4, close-ups.**

## Details

- **F1:** [face_recognition_f1_results.md](face_recognition_f1_results.md).
- **S1:** [virtual_sensor_results.md](virtual_sensor_results.md).
- **S2:** [high_res_heads_results.md](high_res_heads_results.md).
- **F2:** [face_recognition_f2_results.md](face_recognition_f2_results.md).
- **S3:** [head_shape_results.md](head_shape_results.md).
- **Research:** [face_recognition_sota.md](face_recognition_sota.md) and [high_res_faces.md](high_res_faces.md).
- **Previews** (avatars only): `docs/previews/high_res_heads.jpg`, `people_identity.jpg`, `head_shape.jpg`.
