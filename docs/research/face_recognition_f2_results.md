# Person, face, identity (F2): measured results

Step F2 of [face_recognition_focus.md](face_recognition_focus.md): person tracks, faces in their head regions,
evidence per track, open-set decisions calibrated against one target false-alarm rate, the demo's `N`. The design is
sections 2.6, 2.7 and 3 of [face_recognition_sota.md](face_recognition_sota.md); the code is
`detection/tracker.py`, `detection/assignment.py`, `detection/yolox.py` and `recognition/` (`heads.py`,
`evidence.py`, `calibration.py`, `calibrate.py`, `decision.py`, `people.py`, `truth.py`); README: "Face recognition",
"From person to identity". The preview is [docs/previews/people_identity.jpg](../previews/people_identity.jpg).

Measured on 9 November 2026 in the development sandbox: four x86 cores shared with another builder (the load was
5-20 throughout, so every time here is slower than on a quiet machine of this kind), ONNX Runtime 1.31.1 on the CPU,
Mesa llvmpipe, the fair's library of 20 Rocketbox avatars at 2048 and 38 more avatars prepared for the calibration.
Only renders of avatars appear here; LFW's pairs and the local photos of real people were used for checks whose
numbers alone are reported.

## Summary

- **The calibrations ship** (`recognition/calibrations/`), made from renders of 38 avatars that are not in the
  fair, each with T-norm and its cohort: the held-out check chose T-norm for all three recognisers, which named the
  most known faces and kept the measured false-alarm rate nearest the asked one (at 1 % asked: 0.66 % for AdaFace,
  1.7 % for the ResNet-50, 0.32 % for SFace; without normalisation 2.3 %, 3.5 % and 0.9 %). Two avatars that wear
  one face had to count as one identity.
- **On real faces** (LFW's pairs; numbers only) the avatar-made T-norm calibrations roughly hold 1 % per person
  (1.0-1.6 % measured for SFace and the ResNet-50) but let 3-6 times more strangers through at 1 in 1,000, while raw
  thresholds made from avatars are far stricter than asked (no LFW stranger pair reaches AdaFace's): avatars are
  more alike than real people. On the local photos of 6 real people, AdaFace and the ResNet-50 named nobody
  wrongly and every known person rightly; SFace named 2 of 73 strangers' faces (11 pixels between the eyes).
- **The fair** (30 s per run, two crowds, half its 20 avatars known, the profile's budget, the sensor's windows):
  no stranger was named and nobody was named as somebody else in any run, and every stranger shown for a second
  was called unknown. Known people named within 2 s of their face being shown: SFace (`commercial`) 8 of 8 (median
  0.8 s), the ResNet-50 (`licensed`) 7 of 8 (1.0 s), AdaFace (`best`) 1 of 8 (5.0 s): at the one AdaFace embedding
  a second this CPU allows, a crowd waits (a name needs two faces, and every new track shares the budget); with
  ten a second the best profile named the same people in 0.8 s. Without windows the faces are 11-19 pixels apart:
  both profiles measured so named later and fewer, and strangers often waited instead of being called unknown.
- **The target** (95 % within 2 s at a false-alarm rate per track of at most 1 %) is reached here by the commercial
  profile, nearly by the licensed one, not by the best one on this CPU; the runs hold too few people (8 known and
  6 strangers per profile) to show 95 % or 1 % with confidence.
- **Tracks** (ByteTrack in numpy): with a person detector simulated from the truth, 4-10 swaps and 26-31 returns in
  30 s of the fair, 85-92 % of person-frames followed; with YOLO26n or YOLOX-s, 2-7 swaps, 7-19 returns and 60-83 %.
- **This CPU** analyses about 3-4 frames a second besides drawing the scene; the Mac was not measured.

## 1. Person tracks

ByteTrack in numpy (`PersonTracker`): a Kalman filter per box, confident boxes matched first, weak ones second, a
person out of sight kept a second. Identity switches are counted against the scene's truth (who is where, from the
posed skeletons; never shown to the tracker) as CLEAR-MOT counts them, in two kinds: **swaps**, a person taken over by
another track within a second, and **returns**, a person back after more than a second out of sight under a new
number (by design for a tracker that follows boxes alone).

With a person detector simulated from the truth (boxes jittered by 2 % of their height, weak when half hidden by
nearer people, missing when three quarters hidden), 30 s of the fair with its 36 people:

| Frames a second | Swaps | Returns | Person-frames followed |
|---|---|---|---|
| 10 | 5 | 29 | 90 % |
| 5 | 4 | 26 | 85 % |
| 30 | 10 | 31 | 92 % |

The returns are people who walk behind a booth or a group for several seconds; the swaps happen when two walkers of
about the same size cross, which a tracker of boxes alone cannot tell apart (appearance would: section 8). At 30
frames a second the boxes' own jitter is larger against each frame's move, hence more swaps. A swap matters to the
recognition only if the track had a name: the next good face of the other person no longer fits the track's evidence,
and the name goes at once (`IdentityPolicy.consistent`; the stand-in tests make a track jump to someone else).

With the real person detectors in the fair runs of section 4 (5 frames a second, 30 s; people at least 40 pixels
tall and at most three quarters hidden):

| Person detector | Crowd | Swaps | Returns | Person-frames followed |
|---|---|---|---|---|
| YOLO26n | 5 | 5 | 7 | 71 % |
| YOLO26n | 7 | 2 | 15 | 60 % |
| YOLOX-s | 5 | 7 | 15 | 83 % |
| YOLOX-s | 7 | 2 | 19 | 69 % |

A real detector misses people the simulated one finds (small, distant or half-hidden people: section 6), so fewer
person-frames are followed; YOLOX-s, which finds more of the small people, follows more of them and also loses and
finds them again more often (returns).

## 2. Faces in the head regions

A face is looked for only in a track's head region (S1's `HeadRegion` on the smoothed box), in the window the track
asked for or in a crop of the frame, and belongs to the track whose head it fits. The fit's numbers come from
measurements: the eyes 0.079 box heights below the top of YOLO's boxes and 0.037 box heights apart (S1, 91 boxes of the
fair), and the face box 2.9 (SCRFD) to 3.0 (YuNet) eye distances high at every yaw, where the distance between the eye
landmarks falls to 0.52-0.57 of it at 60 degrees (14 avatars at 0, 20, 40 and 60 degrees, 24 and 48 pixels between
the eyes). So a face's size is measured from its box: on the calibration's 3,040 windows the size measured this way
was 0.95-1.01 of the eye distance asked for, the median at every size from 8 to 64 pixels and through the
`budget 1080p` camera; SCRFD found 99.9 % of those faces and YuNet 99.6 % (96-98 % at 8 pixels through the
`budget 1080p` camera), the head turned up to 40 degrees (the landmarks read 31-34 degrees for it).

In the fair the faces of the people facing the camera are 11-19 pixels apart in the 1280x720 view (the audience
about 12-17), so without windows every face the recogniser sees is small; a window asks for 64 pixels between the
eyes, and the four windows a frame go to the tracks not decided yet, turned to the camera, with small faces, in
turns (section 4 compares the two).

## 3. The calibration

**How it was made** (`bifold-faces calibrate`, the package's calibrations): 38 avatars of Rocketbox's conference
set that are *not* in the fair's library, so that every person of the fair is a stranger to the calibration. Per
avatar, a gallery of three windows of the 8K sensor on its head at 2 m (100 pixels between the eyes; turned -25, 0
and 25 degrees), and 80 windows of its face as the pipeline sees faces: 8 sizes (8, 11, 14, 18, 24, 32, 44 and 64
pixels between the eyes) by 5 yaws (-40 to 40 degrees) by 2 cameras (ideal, and `budget 1080p` with its blur, noise
and JPEG), at 3.5, 5 or 6.5 m and up to 0.8 m to the side in the hall, drawn at the renderer's high quality with the
hall's hemisphere light (0.3) and the heads' 2048 textures and maps (recorded in each calibration's `made`). The
faces were found by the profile's detector, aligned and embedded as live, and averaged in twos and threes as a
track's evidence is (five combinations per size and camera, neighbouring yaws).

**Faces that are the same face.** Rocketbox reuses a face: Security_Male_01 wears Male_Adult_03's (AdaFace scores
their faces 0.75-0.80 against each other's photos, where each avatar scores 0.94-0.96 against its own and the next
most alike pair 0.48-0.51). Counted as two identities, the pair raised the held-out check's false alarms (AdaFace
without normalisation: 2.4 % measured at 0.1 % asked, 0.9 % with the pair as one identity; 2.9 % and 2.3 % at 1 %
asked). The builder counts such avatars as one identity: those whose faces reach 80 % of their own median score
against another's photos (AdaFace and the ResNet-50 find this pair, at 80-85 %, and no other; the next pair reaches
52-55 %), and this pair always (`KNOWN_SAME_FACES`): SFace tells faces apart less well, scoring it 75 % and two
different men 69-72 %.

**The check.** Ten random halvings of the 38 avatars (37 identities): one half calibrates; of the other, half are
known and half strangers. Every face and average of a held-out avatar is one decision: about 14,000 decisions about
strangers and 13,000 about known people per recogniser, normalisation and target, pooled over the halvings.
"Measured" is the share of strangers' decisions that named somebody (in brackets: the lowest and highest of the ten
halvings); "named" the share of known people's decisions that named them rightly (nobody was named wrongly in any
check). A name needs the runner-up a factor of ten rarer, as live.

| Recogniser (profile) | Normalisation | 0.1 % asked: measured (halvings) | named | 1 % asked: measured (halvings) | named | 10 % asked: measured | named |
|---|---|---|---|---|---|---|---|
| AdaFace IR-101 (`best`) | none | 0.94 % (0-4.6 %) | 88.1 % | 2.29 % (0-7.0 %) | 92.9 % | 6.41 % | 97.2 % |
| | **T-norm** (shipped) | 0.13 % (0-0.4 %) | 91.8 % | 0.66 % (0-2.3 %) | 95.8 % | 4.09 % | 98.5 % |
| | AS-norm | 0.60 % (0-5.0 %) | 84.3 % | 1.49 % (0-7.5 %) | 91.5 % | 6.60 % | 95.9 % |
| ResNet-50 (`licensed`) | none | 1.97 % (0-8.4 %) | 89.9 % | 3.48 % (0.1-10.6 %) | 94.0 % | 6.89 % | 97.3 % |
| | **T-norm** (shipped) | 0.70 % (0-2.9 %) | 92.7 % | 1.72 % (0-7.6 %) | 96.1 % | 6.03 % | 98.2 % |
| | AS-norm | 1.17 % (0-5.2 %) | 84.3 % | 3.09 % (0-14.8 %) | 91.3 % | 7.22 % | 96.5 % |
| SFace (`commercial`) | none | 0.47 % (0-4.1 %) | 82.2 % | 0.88 % (0-6.3 %) | 86.7 % | 3.54 % | 91.3 % |
| | **T-norm** (shipped) | 0.03 % (0-0.3 %) | 88.2 % | 0.32 % (0-1.2 %) | 91.6 % | 3.45 % | 95.2 % |
| | AS-norm | 0.21 % (0-1.7 %) | 77.5 % | 0.76 % (0-3.8 %) | 86.8 % | 3.93 % | 92.6 % |

**What it shows.** T-norm (a face's scores measured against how it scores against a cohort of other faces: here
16 renders of each calibrating avatar) names the most known faces and keeps the measured rate nearest the asked one:
it takes out how alike a face is to faces in general, which raw scores keep (a face like everyone scores high against
everyone). AS-norm, which also normalises by each gallery photo's own cohort scores, does worse with these galleries
of three renders. Without normalisation the measured rate at 1 % asked is 2.3 times the asked one for AdaFace and
3.5 times for the ResNet-50, and more at 0.1 %; with T-norm 0.7, 1.7 and 0.3 times (AdaFace, ResNet-50, SFace): the
ResNet-50's is above the target, within the factor of two the choice allows. The halvings differ much: in some, one
stranger alike to a known person makes most of the false alarms (up to 8-11 % of the decisions about strangers
without normalisation). With 18 calibrating identities a threshold far in the tail is uncertain;
the shipped calibrations have all 37, and the fair (section 4), with 20 other avatars, checks them independently.
Nobody was named as somebody else in any check. **T-norm is the default** where the check chose it, and its cohort
ships beside the calibration.

**Checked on real faces** (numbers only; LFW and photos may check a calibration, never make one). Avatars are more
alike to each other than real people are: AdaFace's calibration without normalisation puts the false-match rate of
1 % for frontal faces of 28-56 pixels at a score of 0.43-0.44 and 1 in 1,000 at 0.50-0.51, where none of LFW's 3,000
pairs of real strangers reaches 0.28 (LFW's own 1-in-1,000 cut is 0.21; F1's scores). So on real strangers raw
thresholds made from avatars are far stricter than asked.

With T-norm the shift turns round: a real face scores lower against a cohort of avatars than an avatar does, so
its normalised scores come out higher. SFace's shipped calibration, on LFW's pairs scored and normalised as live
(3,000 pairs of strangers, 1,000 of one person; aligned faces of about 35 pixels between the eyes), against its
thresholds for frontal faces of 28-56 pixels: a false-match rate of 1 % asked gave 1.4-1.6 % measured, 1 in 1,000
asked gave 3.3 in 1,000 (10 pairs), and 0.9 % of the pairs of one person fell below that threshold.
The ResNet-50's (500 pairs of strangers and 200 of one person, as many as its time here allowed): 1.0-1.2 % at 1 %
asked, 4-6 in 1,000 at 1 in 1,000 (2 and 3 pairs), and no pair of one person below. AdaFace's normalised check on
LFW did not fit in the time (about 2 s of CPU per face here). So on real faces the shipped T-norm calibrations hold
the target at 1 % per person roughly and let several times more strangers through far in the tail: the cohort of
avatars is not what real faces are measured against best.

The local photos of real people (6 people, 15 photos, each shrunk to faces of 11-49 pixels between the eyes: 73-75
searches as a stranger against the others' photos, 63-65 as a known person against their other photo): AdaFace and
the ResNet-50 named nobody wrongly and every known person rightly at 0.1 %, 1 % and 10 % asked; SFace named 2
strangers at 0.1 % and 1 % asked (both faces of 11 pixels between the eyes) and 5 at 10 %, and every known person.
Too few people to measure a rate, but the same direction as LFW.

## 4. The fair end to end

**How.** The fair's 36 people (its library's 20 avatars; some avatars appear two or three times, mostly as walkers who
never face the camera), 30 s of scene time at 5 frames a second (about what this CPU analyses live; every embedding
budget is per second of scene time, so the frame rate changes how often a face is looked at, not how many are
embedded), fixed steps (deterministic; the 2048 heads read before a frame is drawn). Half the 20 avatars are known,
enrolled from renders of themselves (the calibration's gallery renders: 100 pixels between the eyes, turned -25, 0
and 25 degrees); the others are strangers. Two crowds (`CROWD_SEED` 5 and 7) with two different known halves: in
each, six people face the camera (three in the audience, three visitors who come and go), three known and three
strangers in the first, four and two in the second. The person detector is the profile's (YOLO26n; YOLOX-s for the
commercial profile), at its real cost; the embedding budget is the profile's (AdaFace 1, ResNet-50 4, SFace 20 a
second of scene time: what this CPU allows live); with windows, the tracks ask the 8K sensor (ideal camera) for up
to four windows a frame. Each segment (one person followed by one track; `truth.Outcomes`) is scored against the
scene's truth: a face counts as *shown* while turned less than 45 degrees from the camera, its eyes not hidden by a
nearer person and at least 3 pixels apart in the view; "named within 2 s" counts from the moment the face was first
shown. Rates over segments whose face was shown for at least a second; strangers' false-alarm rate also over all
their segments. Two threads per library (ONNX Runtime, torch, llvmpipe): the machine was shared.

**Results**, pooled over the two crowds per profile (n: segments counted; "named within" counts only faces first
shown at least that long before the run ended):

| Profile (recogniser) | Windows | Named within 2 s | within 5 s | named at all | time to name, median / 90 % (s) | named wrongly | "unknown" said to a known person | strangers named (FPIR per track) | strangers called unknown | embeddings a second |
|---|---|---|---|---|---|---|---|---|---|---|
| best (AdaFace, 1 a second) | yes | 12 % (1/8) | 50 % (3/6) | 75 % (6/8) | 5.0 / 5.6 | 0 of 8 | 0 of 8 | 0 of 6 (0 of 26 followed) | 6 of 6 | 1.0 |
| best | no (crowd 5) | 0 % (0/5) | 0 % (0/3) | 40 % (2/5) | 9.0 / 10.6 | 0 of 5 | 0 of 5 | 0 of 3 (0 of 11) | 2 of 3 | 1.0 |
| licensed (ResNet-50, 4 a second) | yes | 88 % (7/8) | 100 % (6/6) | 88 % (7/8) | 1.0 / 1.5 | 0 of 8 | 0 of 8 | 0 of 6 (0 of 26) | 6 of 6 | 2.2 |
| commercial (SFace, 20 a second) | yes | 100 % (8/8) | 100 % (6/6) | 100 % (8/8) | 0.8 / 0.9 | 0 of 8 | 0 of 8 | 0 of 6 (0 of 48) | 6 of 6 | 2.9 |
| commercial | no | 25 % (2/8) | 83 % (5/6) | 62 % (5/8) | 2.6 / 2.7 | 0 of 8 | 0 of 8 | 0 of 6 (0 of 48) | 0 of 6 | 3.1 |

**Reading it.** Nobody was named as somebody else and no stranger was named in any run; with windows every
stranger whose face was shown for a second was called unknown, 0.4-3 s after it was first shown with SFace and the
ResNet-50 (2-8 s with AdaFace's budget). What differs is how soon the known people are named, and that is the
embedding budget. A name needs two faces of the person, and the budget is shared by every track not decided yet
(8-9 at the start of a run, each getting a face before any gets a second): at one AdaFace embedding a second, at
most one new person can be named in any two seconds, so the best profile named the audience after 3-6 s where SFace
(20 a second, of which the fair used about 3) and the ResNet-50 (4 a second) took about one. The best profile with a budget of ten a second, as a faster machine would allow (crowd 5, 15 s):
the three known people shown were named 0.8 s after their faces were first shown, and both strangers were called
unknown within 2 s; it used 2.1 embeddings a second of the ten. So AdaFace itself is quick to decide; this CPU's
budget is what holds it back.

**The target** (at least 95 % of known people named within 2 s at a false-alarm rate per track of at most 1 %):
reached in these runs by the commercial profile (8 of 8, no stranger named), nearly by the licensed one (7 of 8;
the one missed was a face shown for 1.4 s at the end of a run), not by the best one on this CPU (1 of 8). These
runs hold few people (8 known and 6 strangers per profile shown long enough): they show which way each profile
goes, not 95 % or 1 % with any confidence; E1's longer runs with more people must.

**With and without windows.** Without them every face is the frame's, 11-19 pixels between the eyes for the people
facing the camera. The commercial profile then named the known people later (2 of 8 within 2 s, median 2.6 s
against 0.8 s) and fewer of them (5 of 8), and called no stranger unknown: faces that small never tell known and
unknown apart decisively, so they wait. The best profile without windows named 2 of 5 known people at all, after 7
and 11 s (crowd 5). No false name either way.

**With and without normalisation** in the fair: the best profile with a calibration without normalisation (made
from the same avatars) took exactly the same decisions at the same moments as with T-norm (crowd 5, with windows):
here the budget decides when, and both calibrations agree on whom. The held-out check (section 3) is where the
normalisations differ.

## 5. What this CPU allows, and the Mac

Milliseconds per frame of the fair (1280x720, 8-14 people followed at once), the mean over the runs with windows,
under a load of 12-20 on the four cores (two threads per library; alone on this machine F1 measured 0.5-0.6 s per
AdaFace crop, 0.1 s per ResNet-50 crop and 30 ms per SFace crop, so a quiet machine is faster):

| Profile | Windows | person detector | tracker | faces | quality | embeddings | decisions | drawing the frame | of which windows |
|---|---|---|---|---|---|---|---|---|---|
| best | yes | 2,855 (YOLO26n) | 5 | 435 | 170 | 403 | 9 | 1,552 | 857 |
| licensed | yes | 2,943 (YOLO26n) | 6 | 393 | 160 | 200 | 9 | 866 | 139 |
| commercial | yes | 559 (YOLOX-s) | 9 | 83 | 16 | 26 | 9 | 1,402 | 278 |
| commercial | no | 299 (YOLOX-s) | 5 | 28 | 8 | 16 | 3 | 565 | 0 |

These times say more about the shared machine than about the CPU: YOLO26n (torch) took 2.9 s a frame on average
here, where it took 0.11 s a frame earlier the same day (section 6), and the scene's own drawing (llvmpipe) took
0.6-1.6 s. On a quiet machine of this kind the chain costs about what its parts cost alone: the person detector 0.1 s
(YOLO26n) or 0.26 s (YOLOX-s), faces in six head regions about 0.07 s with SCRFD (0.02 s with YuNet), and the
embeddings at the budget: AdaFace's one a second is about 0.5-0.6 s of every second, the ResNet-50's four about
0.4 s, SFace's twenty about 0.6 s, but the fair asked SFace for only about 3 a second. So this CPU analyses about
3-4 frames a second besides drawing the scene, and the budget, not the frame rate, sets how soon people are named:
the best profile's one AdaFace embedding a second is what it allows, and that is too few for a crowd (section 4).

**The Mac** was not measured (no Mac here): how much of each graph CoreML runs, and how fast, F1 left to be measured
there (`bifold-faces check --speed`). Section 4's run of the best profile with ten embeddings a second shows what a
machine that embeds that many would give; the budget is a setting (`--embed-rate`).

## 6. The commercial profile's person detector: YOLOX against YOLO

17 frames of the fair (every 2.5 s over 40 s), people whose box is at least 40 pixels high and at most half covered
by nearer people, matched to the detectors' person boxes at an overlap (IoU) of at least 0.5, confidence at least 0.35
(the demo's):

| Detector | Licence | People 40-80 px tall found | 80-160 px | over 160 px | Boxes matching nobody | Overlap (median) | ms a frame (median) |
|---|---|---|---|---|---|---|---|
| YOLO26n (Ultralytics) | AGPL-3.0 | 43 % (of 67) | 89 % (of 27) | 100 % (of 90) | 1 | 0.74 | 110 |
| YOLOX-s (official ONNX) | Apache-2.0 | 66 % | 96 % | 100 % | 2 | 0.73 | 264 |
| YOLOX-tiny (official ONNX) | Apache-2.0 | 12 % | 78 % | 100 % | 2 | 0.74 | 99 |

YOLOX-s finds more of the small, distant people than YOLO26n and is as precise, at 2.4 times the time on this CPU
(both under load; YOLO26n's ninetieth percentile was 397 ms in the same run, YOLOX-s's 319 ms). It is the commercial
profile's person detector; YOLOX-tiny is registered too but misses most people under 80 pixels.

## 7. For the evaluation (E1)

- **Deterministic runs**: a scene source with a fixed-step clock (which also makes the renderer wait for the 2048
  heads), the gallery refreshed (`refresh(force=True)`) before the run; the tracker, the embedding budget and the
  decisions read only the frames' times.
- **Every decision** (`DecisionLog(path)`, JSON lines): the state, name, reason and progress; the best person's and the
  runner-up's raw, normalised and calibrated scores; the thresholds of the face's bin; the model and normalisation;
  the evidence (each face's time, quality weight, size, yaw, source, eDifFIQA-T, and the resets).
- **Track-level results**: `PeopleResult.people` per frame (track id, box, decision, face), and `recognition/truth.py`
  to score them against the scene: `frame_truth`, `match_tracks`, `IdSwitches`, `Outcomes` (the shares of this note).
- **The calibration and the cohort as inputs**: `PeopleRecognizer(calibration=..., cohort=...)`,
  `Calibration.load/save`, `Cohort.load/save` (a calibration records its cohort's digest; the decisions refuse
  another); `recognition/calibrate.py` makes them (`render`, `gallery_photos`, `faces`, `same_faces`, `as_one`,
  `scores`, `fit`, `check`, the held-out check with each halving's own rate). A calibration's `made` says how it was
  made (avatars, same faces, detector, model SHA-256, sizes, yaws, cameras, the renderer's quality and hemisphere
  light, the check).

## 8. Open

- **Calibrations from avatars on real faces.** Avatars are more alike to each other than real people are (section
  3): raw thresholds made from them are far stricter on real strangers than asked, and T-norm against a cohort of
  avatars, which measures a real face against faces unlike it, lets 3-6 times more real strangers through than asked
  at 1 in 1,000 per person (SFace, the ResNet-50; AdaFace's was not measured). A calibration and cohort made from
  real faces that may be shipped (licensed photos, consented recordings) would fix both; E1 should measure the rate
  asked against the one measured on real people.
- **The best profile's budget on this CPU.** One AdaFace embedding a second cannot name a crowd within 2 s (a name
  needs two faces); a faster machine (the Mac) or a scheduler that finishes one track's evidence before starting the
  next (the first people sooner, the last no sooner) would help.
- **Few identities.** 37 identities make the far tail uncertain: in the held-out check the ResNet-50's measured rate
  with T-norm was 1.7 times the asked one at 1 % and the halvings ranged from 0 to 7.6 %. More avatars (Rocketbox has
  115) would narrow it.
- **AuraFace** (the commercial profile where Hugging Face can be reached) has no calibration yet: it gets provisional
  thresholds until `bifold-faces calibrate --profiles commercial` runs there.
- **Returns.** A person out of sight for more than a second comes back under a new track number and is identified
  again from scratch; re-identification by appearance (clothes, the face's evidence) would join the two.
- **Swaps** when two walkers of the same size cross: the name goes at once (the switch check), and comes back only
  with new evidence.
- **Windows render on the main thread** (S1): with `W` on, the view slows down on a software renderer.
- **The Mac** (CoreML) is not measured: the budgets per profile are this CPU's.
- **Painted faces** (the face cast: real people's photos on avatars) score lower against their photos than renders
  against renders (S2's measurements); a face unlike everyone known and unlike strangers too waits as "not decided
  yet" instead of being called unknown, but how long such faces take to be named was not measured here.
