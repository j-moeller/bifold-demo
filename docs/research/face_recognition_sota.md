# Face recognition for a camera watching a crowd: literature review

The core component of the demo from now on: after a person is detected, say
who they are. This review covers the whole chain for the demo's camera (person
detection, face detection in the person's region, alignment, quality,
embedding, tracking and aggregation over time, open-set identification against
the `faces/` gallery, enrolment and calibration), how to measure it, and two
stacks behind one switch: the most accurate one (research-only weights allowed)
and a commercially usable one. Written November 2026.

How the claims were checked: from the development sandbox, GitHub (raw files
and release downloads), PyPI and Google Cloud Storage could be opened; arXiv,
CVF, NIST, ISO, BSI, Hugging Face, the USPTO and vendor sites could not.
READMEs, licence files and source code were read on GitHub. Release assets were
checked with HTTP HEAD requests only (they exist, and their size); nothing was
downloaded and run. Claims that rest only on search-result excerpts are marked
*(excerpt)*. No model was run, so every timing is either quoted from the cited
source or marked (est.), an estimate from published FLOPs.

## 1. Problem and constraints

**What exists.** [`recognition/`](../../src/bifold_demo/recognition) runs
OpenCV's YuNet detector over the whole frame (scaled down only above 1600 px)
and SFace on every face, aligned to 112x112 by `FaceRecognizerSF.alignCrop`
(the ArcFace five-point template). SFace gives a 128-number embedding. Faces
are followed from frame to frame by box overlap (IoU at least 0.25); each
followed face averages its last 8 embeddings and is named after the gallery
person with the most similar photo if the cosine similarity reaches 0.363
(OpenCV's cut for SFace on LFW), keeping the name down to 0.06 below the cut. A
face unseen for one second is forgotten. The YOLO person detector runs in its
own worker and is not used by the face path. The README reports, for photos of
three real people painted on four avatars, 24 of 24 close-ups named correctly
(front and turned 40°, similarity 0.37-0.58) and 98 of 120 analysed frames from
the fair camera with faces about 30 px tall.

**What the user wants.** "After detecting a person we want a state-of-the-art
face recognition system to identify their face", extremely well made. Research
weights are acceptable, but best is a solution that allows both research and
commercial use: the most accurate stack and a commercially usable one,
switchable behind one interface.

**Operating conditions.**

- Frames are 1280x720. In the fair, faces are 30-60 px tall, so the distance
  between the eye centres (inter-eye distance, IED) is about 10-22 px (est.:
  IED is roughly 0.35 of a detector box's height). Passport standards ask for
  at least 90 px. This is surveillance territory: single frames are weak
  evidence, and waiting for a good frame matters as much as the model.
- A crowd: faces turned away, people occluding each other, motion.
- Rendered faces (Rocketbox avatars wearing textures made from the known
  people's photos) in the simulation; real faces from a webcam. A separate
  review covers high-resolution faces in the simulation and a way of seeing a
  detected person's head at high resolution; the recommendation below uses
  that view when it exists.
- Runs on the user's Apple-silicon Mac (CPU, Metal/MPS, CoreML) and in a
  CPU-only sandbox with 4 shared cores; Python 3.13.
- Network here: GitHub (including release assets), Google Cloud Storage and
  PyPI work; Hugging Face does not. The user's Mac reaches Hugging Face.
- Safety rule here: no third-party code, pickles or TorchScript. ONNX (run with
  onnxruntime or OpenCV DNN), safetensors through official libraries and
  TFLite/MediaPipe bundles are acceptable.
- Packages (checked on PyPI, 1 November 2026): onnxruntime 1.31.1 (26 October
  2026, needs Python 3.11 or newer) has cp313 wheels for macOS 14+ arm64 and
  manylinux_2_28 x86_64, and the official macOS wheel includes the CoreML
  execution provider. opencv-python 5.0.0.93 ships abi3 wheels for both
  platforms; in OpenCV 5 the default DNN engine is OpenCV's own (ONNX Runtime
  only in builds made with it). mediapipe 1.2.0 has wheels for both;
  insightface 2.1 is a pure-Python wheel.
- **ONNX Runtime telemetry.** Since 1.29.0 the official Linux and macOS builds
  send usage telemetry to Microsoft by default (Windows did before). It is off
  only if `ORT_DISABLE_TELEMETRY=1` is set before ONNX Runtime initialises. The
  sandbox has 1.29.0. A face recogniser should set it.

**What "commercially usable" has to mean here.** Four layers, only the first
of which can be read off a licence file:

1. the licence of the weights;
2. the terms of the training data: nearly every strong recogniser was trained
   on MS-Celeb-1M derivatives (withdrawn by Microsoft in 2019), Glint360K or
   WebFace260M, whose terms are non-commercial research; most detectors on
   WIDER FACE, tagged CC BY-NC-ND 4.0 *(excerpt)*;
3. patents: a granted US patent (12,525,057) describes a face recognition
   system using AdaFace's quality-adaptive margins *(excerpt; assignee and
   claims not read)*;
4. intended use in model cards: MediaPipe's BlazeFace card says "any form of
   surveillance or identity recognition is explicitly out of scope".

The commercial profile below is clean on layer 1 and flags 2-4. A legal review
is needed before anything ships.

## 2. State of the art

### 2.1 The pipeline for a camera system

Systems that identify people in video share one shape (NIST's Face in Video
Evaluation, the IJB-S surveillance protocols, InsightFace's PersonAnalysis and
PrivateFrame of September-October 2026):

1. **Person detection and tracking.** A person track survives when the face
   turns away, so identity evidence accumulates on the person, not on a face
   box that disappears for a few frames. InsightFace's PersonAnalysis (2.1)
   matches faces and bodies to registered people and learns a body reference
   after a clear face match; it does not track.
2. **Face detection in the person's head region.** Crop the top of each person
   box (from the full-resolution frame, or from the high-resolution head view),
   scale it to the detector's preferred size, and find the face and five
   landmarks. Small faces become detectable, and faces on posters or textures
   outside a person are ignored. A full-frame pass at a low rate catches faces
   without a person box (a webcam close-up).
3. **Alignment** to the 112x112 template the recogniser was trained with.
4. **Quality**: is this frame worth embedding, and how much should it count?
5. **Embedding**: a 512-number vector (128 for SFace), normalised to length 1.
6. **Aggregation over the track**, weighted by quality.
7. **Open-set identification**: the best gallery match, a calibrated
   threshold, a margin over the runner-up, and three answers (a name,
   "unknown", "not decided yet").
8. **Enrolment**: several photos per person, embedded with the same detector,
   aligner and model, checked for quality and duplicates, cached per model.
9. **Operating point**: thresholds derived from a target false-alarm rate,
   calibrated per model and per image quality.

The current code has steps 2-7 in a simple form, on the whole frame, with face
tracks only.

### 2.2 Face detection and landmarks

| Detector | WIDER FACE val, easy / medium / hard (protocol) | Landmarks | CPU speed (source) | Small, turned, occluded faces | Licence: weights; training data | Weights, form, here? |
|---|---|---|---|---|---|---|
| YuNet 2023mar / 2026may (OpenCV Zoo) | 88.4 / 86.6 / 75.0 (OpenCV eval tool); paper 83.4 / 82.4 / 70.8 | 5 | 0.69 ms at 160x120 on an i7-12700K (OpenCV Zoo) | Faces of about 10-300 px; weakest of these on hard | MIT; WIDER FACE | ONNX, 0.23 MB, GitHub LFS; works here |
| SCRFD-500M / 2.5G / 10G / 34G (InsightFace) | 90.6 / 88.1 / 68.5; 93.8 / 92.2 / 77.9; 95.2 / 93.9 / 83.1; 96.1 / 94.9 / 85.3 (VGA, single scale) | 5 (`_KPS` variants) | 500M: 28.3 ms at 640x480, 11.4 ms at 320x240, one thread (Ryzen 3950X, PyTorch) | Strong on small faces for its cost | Non-commercial (InsightFace); WIDER FACE | 10G_KPS as `det_10g.onnx` in `buffalo_l`, 500M in `buffalo_s`; ONNX, GitHub releases; works here |
| RetinaFace R50 (InsightFace) | 96.5 / 95.6 / 90.4 (multi-scale); 94.9 / 91.9 / 64.2 at VGA single scale | 5 | 21.7 ms at VGA on a GPU (SCRFD table) | Best tiny-face recall only with multi-scale testing | Non-commercial; WIDER FACE | MXNet on Google Drive; not here |
| RetinaFace MobileNetV2 / ResNet34 (yakhyo, 2024 retrain) | 94.0 / 92.3 / 83.6; 95.1 / 93.5 / 84.4 (original size) | 5 | n/a | Good landmarks; R34 heavy | MIT; WIDER FACE | ONNX, 12.5 MB / 88.5 MB, GitHub releases (`uniface`); works here |
| YOLOv5-Face n / s / l | 93.6 / 91.5 / 80.5; 94.3 / 92.6 / 83.2; 95.8 / 94.3 / 86.1 | 5 | n/a on CPU | Good small faces | GPL-3.0; WIDER FACE | `.pt` on Google Drive/Baidu; ONNX only as community exports |
| YOLOv8-Face n / s / m; YOLO11/12-Face | 94.6 / 92.3 / 79.6; 96.1 / 94.2 / 83.1; 96.6 / 95.0 / 84.7 | 5 | n/a | Good small faces | GPL-3.0 / AGPL-3.0; WIDER FACE | `.pt` (pickled); no |
| YOLO-FaceV2 s / l (Pattern Recognition 2024) | 98.3 / 97.0 / 89.3; 98.6 / 97.9 / 91.9 (protocol not stated) | none | n/a | Receptive-field module for small faces, attention module for occlusion | No licence file (all rights reserved); YOLOv5 base | `.pt` on GitHub releases; no |
| MogFace (CVPR 2022) | 94.6 / 93.6 / 87.3 (Ali-AMS, single scale) | none | heavy | Strong on hard | Not checked | Google Drive; no |
| MediaPipe BlazeFace short / full range | Not reported on WIDER; the cards report recall about 98-99 % and average precision 89-100 % on in-house smartphone photos (lowest on the multi-face set) | 6 (eyes, nose, mouth centre, ear tragions) | Short: ~275 FPS on one Pixel 2 core; full: ~50 FPS on one Pixel 3 core | Short: faces at least 20 % of the image, within 2 m; full: at least 5 %, within 5 m, yaw up to 90°, roll and pitch up to 45°; returns at most about 10 faces | Apache-2.0; trained on consented images; card excludes surveillance and identity recognition | TFLite, 0.23 / 1.08 MB, GCS; works here |
| MediaPipe Face Landmarker | Detector plus mesh | 478 3D points, 52 blendshapes, head pose | ~16 ms per image on CPU (earlier review, measured) | Selfie distance; fails when turned more than 80° or less than half visible | Apache-2.0; card: "do not provide facial recognition or identification" | `.task`, 3.8 MB, GCS; works here |

The WIDER FACE numbers are not comparable across rows: VGA single-scale,
original-size and multi-scale testing differ by up to 26 points on hard (the
same RetinaFace R50 scores 90.4 or 64.2). The hard subset consists largely of
tiny faces, many too small to recognise. For identification what
matters is recall at 15-60 px and how stable the five landmarks are there,
because alignment errors cost more accuracy as quality drops (ARoFace, below).

Two practical points from InsightFace 2.x: its `FaceAnalysis` now runs SCRFD
at 128x128 and 640x640 and merges the results (small and large faces in one
pass), and on CoreML it keeps one fixed-shape session per input size, because
CoreML works best with fixed shapes. OpenCV Zoo added
`face_detection_yunet_2026may.onnx` with dynamic height and width for OpenCV
5's ONNX Runtime engine; the repo still pins the fixed-shape 2023mar file,
which OpenCV's default engine accepts (`FaceDetectorYN` sets the input shape).

**Dense landmarks** are useful for head pose (quality gating, pose-aware
gallery), not for the recogniser's alignment: InsightFace's `2d106det` and
`1k3d68` (in every pack; non-commercial), MediaPipe's 478-point mesh
(Apache-2.0), PIPNet (MIT code; ONNX in `uniface`), 3DDFA_V2 (MIT code with
non-commercial BFM assets). Five detector landmarks give yaw well enough for
gating: the nose tip's offset from the eye midpoint relative to the eye
distance.

### 2.3 Alignment

**The five-point template.** Almost every public recogniser was trained on
112x112 crops made by one similarity transform (rotation, uniform scale,
translation) that moves the detector's five landmarks onto fixed points: the
eyes at (38.29, 51.70) and (73.53, 51.50), the nose tip at (56.03, 71.74), the
mouth corners at (41.55, 92.37) and (70.73, 92.20). OpenCV's
`FaceRecognizerSF` and InsightFace's `norm_crop` use exactly these numbers. A
similarity transform keeps the face's proportions; the recogniser has learnt
what a turned face looks like in this frame. Changing the template, or the
landmark detector, moves the crop away from what the model saw in training:
AdaFace was trained on MTCNN landmarks, InsightFace's models on
RetinaFace/SCRFD landmarks. Both are close enough to swap, but each model
should be checked with the landmarks it will get.

**Why alignment matters more at low quality.** ARoFace (ECCV 2024) shows that
recognisers are fragile under even mild misalignment of low-quality faces, and
training with random alignment perturbations gains 4.3 points of rank-1 on
IJB-S and 2.6 on TinyFace *(excerpt)*. KP-RPE (CVPR 2024) feeds the landmarks
into a ViT's position encoding, so the model adapts to where the landmarks
actually are. With WebFace4M and the AdaFace loss, TinyFace rank-1 is 72.1 %
for an IR-101, 74.8 % for a plain ViT and 75.8 % for the ViT with KP-RPE
(CVLface's table): the transformer gives most of the gain, the landmarks one
more point. For us: stable landmarks (a good detector run on an upscaled head
crop) are cheap accuracy, before any change of recogniser.

**Dense and 3D alignment, frontalisation.** 3D fitting (3DDFA_V2, MediaPipe's
mesh) estimates head pose reliably and can render a frontal view. Studies of
frontalisation find small gains on near-frontal data, some gain at extreme
poses, and visible distortions and identity drift; reviewers of one NeurIPS
2018 frontaliser noted recognition gains mostly below 1 %, and the method
outperformed by up to 7 % at 75-90° *(excerpt)*, and a 2026 CVPR workshop paper reports that GAN frontalisers
"frequently exhibit distortions or identity inconsistencies" *(excerpt)*. A
frontaliser fills the unseen half of the face from a prior, which pulls
different people towards the same average face: exactly the wrong direction
for an open-set system that must say "unknown". Modern recognisers already
handle moderate pose: 99.3 % on CFP-FP (frontal against profile) for an
IR-101 trained on WebFace12M.

**What helps turned faces instead:**

- a pose-aware gallery: enrol frontal, three-quarter and profile photos and
  compare a probe with the photos of similar pose (the idea of Masi et al.'s
  pose-aware models, CVPR 2016, which rendered 40° and 75° views);
- weighting frames by yaw when aggregating a track, and waiting for a more
  frontal frame;
- choosing the recogniser by its cross-pose results (CFP-FP, CPLFW), not LFW.

### 2.4 Recognition models

**Losses.** Since 2018 every strong recogniser is a classifier over the
training identities whose softmax adds a margin on the hypersphere, then drops
the classifier and keeps the embedding: SphereFace (2017), CosFace (2018) and
ArcFace (2019) are the base. The later losses change how hard each sample is
pushed: CurricularFace (2020) emphasises hard samples as training progresses;
MagFace (2021) ties the embedding's length to its quality; SFace (2021,
OpenCV's model) uses sigmoid-limited forces against label noise; ElasticFace
(2022) draws random margins; **AdaFace** (2022) makes the margin depend on
image quality, measured by the feature norm, which is why it leads on
low-quality benchmarks; UniFace (2023) adds a unified cross-entropy so that one
threshold separates all pairs (its main IJB-C numbers were not verified here).

**Backbones and scale.** iResNet-50/100/200 and MobileFaceNet dominate; Partial
FC (2022) made training on millions of identities feasible (WebFace42M has 2
million). Vision transformers now share the top: TransFace (ICCV 2023) with
patch-level augmentation and hard-sample mining, KP-RPE (CVPR 2024) with
landmark-aware position encoding, TopoFR (NeurIPS 2024) with topology
alignment, and LVFace (ICCV 2025), a large ViT with progressive cluster
optimisation, first on the MFR-Ongoing academic track in December 2024.
EdgeFace (T-BIOM 2024) won the compact track of the EFaR 2023 competition.

**Low-quality work since 2023**: recognisability embeddings (Chai et al., CVPR
2023, TinyFace rank-1 73.1 % *(excerpt)*), ARoFace (ECCV 2024), PETALface
(WACV 2025, low-rank adapters for low-resolution data), FaceMoE (June 2026
preprint, 76.2 % on TinyFace *(excerpt)*) and centre-aligned representations
(IJCB 2026 preprint). None of these has weights we can use here.

**Results.** TAR is the share of genuine pairs accepted at the given false
accept rate (FAR); TinyFace and IJB-S give rank-1 identification.

| Model (loss, backbone) | Training data | IJB-B @1e-4 | IJB-C @1e-4 | IJB-C @1e-5 | TinyFace R1 | IJB-S surv.-to-single R1 | CFP-FP | AgeDB-30 |
|---|---|---|---|---|---|---|---|---|
| ArcFace R100 | MS1MV2 | 94.25 | 96.03 (96.37 retrained) | 94.27 (retrained) | - | 57.35 | 98.27 | 98.28 |
| CosFace R100 | MS1MV2 | 94.80 | 96.37 | - | - | - | - | - |
| CurricularFace R100 | MS1MV2 | 94.80 | 96.10 | - | 63.68 | 62.43 | 98.36 | 98.37 |
| MagFace R100 | MS1MV2 | 94.51 | 95.97 | - | - | - | - | - |
| ElasticFace-Cos+ R100 | MS1MV2 | 95.43 | 96.65 | - | - | - | - | - |
| AdaFace R100 | MS1MV2 | 95.67 | 96.89 | - | 68.21 | 65.26 | 98.49 | 98.05 |
| **AdaFace IR-101** | WebFace12M | **96.41** | **97.66** | - | **72.29** | **71.35** | 99.26 | 98.00 |
| AdaFace ViT-B with KP-RPE | WebFace12M | 96.55 | 97.82 | - | 76.10 | - | 99.30 | 98.10 |
| ArcFace R100 | Glint360K | - | 97.55 | 96.38 | - | - | - | - |
| ArcFace R100 / ViT-L | WebFace42M | - | 97.85 / 98.00 | 96.63 / 97.23 | - | - | - | - |
| TransFace-L | Glint360K | - | 97.61 | 96.29 | - | - | - | - |
| TopoFR R200 | Glint360K | - | 97.84 | 96.71 | - | - | - | - |
| LVFace-B | Glint360K | 96.51 | 97.70 | 97.00 | - | - | - | - |
| InsightFace `buffalo_l` (ArcFace R50) | WebFace600K | - | 97.25 | - | - | - | 99.33 | 98.23 |
| InsightFace `buffalo_s` (MobileFaceNet) | WebFace600K | - | 95.02 | - | - | - | 98.00 | 96.58 |

Sources: the AdaFace, CVLface, InsightFace `arcface_torch` and model-zoo,
TransFace, TopoFR, LVFace and CurricularFace READMEs; ElasticFace from its
paper *(excerpt)*. Numbers in one row may come from different evaluations.
IJB-S numbers in particular depend on the protocol variant (PETALface reports
44.4 % surveillance-to-single for its own setting *(excerpt)*), so compare
them only within one paper.

What the table says: at IJB-C TAR@1e-4 every ResNet-50 or larger model
trained on Glint360K or WebFace sits between 97.2 and 98.0 %; the benchmark is
saturated, and the differences show at 1e-5 and below, across pose, and on
low-quality sets. On
the low-quality sets AdaFace and KP-RPE are the documented leaders, and the
training data matters as much as the loss (AdaFace R100: 68.2 % on TinyFace
with MS1MV2, 72.3 % with WebFace12M).

**NIST FRTE.** None of these downloadable weights appear on NIST's FRTE
leaderboards, which list vendor submissions. InsightFace's team reports 4th
place on FRVT 1:1 (April 2021) and first place on its VISA track (October
2021, Partial FC); those submissions are not the public packs.

**Training sets and their terms.**

| Set | Identities / images | Origin | Terms | Used by |
|---|---|---|---|---|
| CASIA-WebFace | 10K / 0.5M | Web, celebrities | Research use, no clear licence | Small baselines |
| VGGFace2 | 9K / 3.3M | Google image search | Not verified here | AdaFace R18, eDifFIQA labels |
| MS1MV2 / MS1MV3 | 85K / 5.8M (V2) | Cleaned MS-Celeb-1M | MS-Celeb-1M was withdrawn by Microsoft in June 2019; copies still circulate | ArcFace, MagFace, ElasticFace, CurricularFace |
| Glint360K | 360K / 17.1M | Merged and cleaned | "Non-commercial research purposes only", models trained on it included | `antelopev2`, TransFace, TopoFR, LVFace |
| WebFace260M (WebFace4M / 12M / 42M subsets) | 4M / 260M; 2M / 42.5M for WebFace42M | Web (Zhu et al., CVPR 2021) | Research-only access, no redistribution *(excerpt)* | `buffalo_l` (WebFace600K), AdaFace, KP-RPE, EdgeFace |
| DigiFace-1M | 110K / 1.2M renders | Microsoft, synthetic | Research Use of Data Agreement: models trained on it may not be used "in any commercial offering" | Research |
| Vec2Face, DCFace, IDiff-Face, Arc2Face sets | Up to 300K | Generated | Generators trained on WebFace or CASIA; Arc2Face CC BY-NC-SA 4.0 *(excerpt)* | Research |

Synthetic-only training still trails real data of the same size: Vec2Face's
HSFace10K averages 92.0 % on five benchmarks against 94.8 % for CASIA-WebFace,
and DigiFace-1M 86.4 % *(excerpt)*. There is no strong recogniser trained only
on data that clearly permits commercial use.

**Downloadable models that run here** (ONNX from GitHub or GCS):

| Model | Training data | Host and form | Licence of the weights | File size | GFLOPs | Embedding | Key numbers |
|---|---|---|---|---|---|---|---|
| SFace 2021dec (OpenCV Zoo; in use) | Not stated (the paper used CASIA, VGGFace2, MS1M) | ONNX on GitHub LFS (int8 variant 10.7 MB) | Apache-2.0 | 38.7 MB | n/a | 128 | LFW 99.40 % (Zoo eval); 5.1 ms on an i7-12700K in OpenCV |
| InsightFace `buffalo_l`: `w600k_r50.onnx` | WebFace600K | Zip with `det_10g`, `2d106det`, `1k3d68`, `genderage`; GitHub releases `v0.7` and `model-zoo` (zip 288.6 MB) | Non-commercial; commercial licence offered (`recognition-oss-pack@insightface.ai`) | 174 MB | 6.3 | 512 | IJB-C 97.25, CFP-FP 99.33 |
| InsightFace `buffalo_s` / `buffalo_sc`: `w600k_mbf.onnx` | WebFace600K | GitHub releases (zips 127.6 / 15.0 MB) | Non-commercial | 13 MB | 0.45 | 512 | IJB-C 95.02 |
| InsightFace `antelopev2`: `glintr100.onnx` | Glint360K | Zip with `scrfd_10g_bnkps`, `2d106det`, `1k3d68`, `genderage`; GitHub releases (zip 360.7 MB) | Non-commercial | 261 MB | 12.1 | 512 | IJB-C about 97.5 (the R100 Glint360K row) |
| InsightFace 2.x `raccoon_s/l`, `cheetah_s/l` | As `buffalo_s/l` | Zips with a manifest and a signed `MODEL.LICENSE` (non-commercial grant); `model-zoo` release | Non-commercial | 27-255 MB zips | - | 512 | Same recognition weights as `buffalo_s` / `buffalo_l`; Cheetah adds a body detector and a body ReID model |
| **AdaFace IR-101** (community export, `yakhyo/adaface-onnx`) | WebFace12M | ONNX on GitHub releases; BGR input, (x - 127.5) / 127.5, one output | MIT (the upstream repository's licence; no separate weight terms); data research-only; see the patent | 260.7 MB | 12.1 | 512 | IJB-C 97.66, TinyFace 72.3, IJB-S 71.4 |
| AdaFace IR-50 / IR-18 (same exporter) | WebFace4M | ONNX, GitHub | As above | 174.4 / 96.1 MB | 6.3 / 2.6 | 512 | IJB-C 96.98 / 94.99; TinyFace 70.2 / 66.1 |
| EdgeFace XS / S / Base (community export, `yakhyo/edgeface-onnx`) | WebFace260M subsets *(excerpt)* | ONNX, GitHub (original `.pt` state dicts also on GitHub) | Conflicting: the GitHub repo says BSD-3-Clause, Idiap's model cards say CC BY-NC-SA 4.0 *(excerpt)*; treat as non-commercial | 7.3 / 14.8 / 73.1 MB | 0.15 / 0.31 / 1.40 | 512 | S: LFW 99.78, CFP-FP 95.74, CPLFW 92.48 |

The official AdaFace checkpoints are pickled `.ckpt` files on Google Drive;
no official ONNX export exists, so the community export above is the only
acceptable form here. It must be checked against published accuracy before it
is trusted (section 4). Its wrapper expects a single output, so whether the
feature norm, AdaFace's quality signal, survives the export is unknown.

**Mac-only or not usable here:**

| Model | Training data | Host and form | Licence | Embedding | Key numbers |
|---|---|---|---|---|---|
| KP-RPE ViT-B (CVLface) | WebFace12M | Hugging Face: safetensors plus custom model code and a separate landmark model (`cvlface_DFA_mobilenet`) | Card: follow the training data's licence *(excerpt)* | 512 | IJB-C 97.82, TinyFace 76.1 |
| LVFace-T/S/B/L | Glint360K | Hugging Face: ONNX and `.pt` | Code MIT; models non-commercial | 512 | LVFace-B IJB-C 97.70 / 97.00 (1e-4 / 1e-5) |
| **AuraFace-v1** (fal) | "A commercial dataset comprising face images from various sources" *(excerpt)* | Hugging Face: ONNX (`glintr100.onnx`, 261 MB, SHA-256 a7933ea5...; differs from `antelopev2`'s 4ab1d643...) | **Apache-2.0** | 512 | LFW 99.65, CFP-FP 95.19, AgeDB 96.10, CALFW 94.70, CPLFW 90.93 (self-reported *(excerpt)*); no IJB or TinyFace numbers |
| TransFace, TopoFR, MagFace, ElasticFace, CurricularFace | MS1MV2 or Glint360K | `.pt` / `.pth` (pickled) on Google Drive, Baidu, ModelScope | Data non-commercial; ElasticFace CC BY-NC-SA 4.0 | 512 | See the results table |
| Intel `face-reidentification-retail-0095` | Not stated | OpenVINO IR on storage.openvinotoolkit.org (blocked here) | Apache-2.0 | 256 | LFW 99.47; 0.59 GFLOPs |
| dlib ResNet v1 | FaceScrub, VGG Face, scraped (7,485 identities) | dlib `.dat` on GitHub (not an allowed form) | Released to the public domain by its author | 128 | LFW 99.38 |
| GhostFaceNets | MS1MV2/V3 | Keras `.h5` on GitHub releases (not an allowed form) | MIT code | 512 | Best IJB-C 94.94 |

AuraFace's Hugging Face repository also contains InsightFace's detection and
landmark models, which stay non-commercial; only its `glintr100.onnx` is the
Apache-2.0 recogniser *(excerpt)*. Whether it was initialised from a
Glint360K-trained network is not stated.

**Embeddings.** All of these output a raw vector; normalise it to length 1 and
compare by cosine similarity (a dot product). Embeddings of different models
are not comparable, even at the same dimension: the gallery cache must be kept
per model.

**Official ONNX or not.** Published as ONNX by their authors: YuNet, SFace,
all InsightFace packs (SCRFD, ArcFace R50/R100, MobileFaceNet, landmarks,
attributes), AuraFace, LVFace, the RetinaFace retrain (by its trainer) and
YOLOX. Only as community ONNX exports: AdaFace, EdgeFace, eDifFIQA,
MiniFASNet and the YOLO-face family. No ONNX at all: KP-RPE (safetensors and
model code), TransFace, TopoFR, MagFace, ElasticFace, CurricularFace, CR-FIQA.
Inside `antelopev2` the five files are `glintr100.onnx` (261 MB),
`1k3d68.onnx` (144 MB), `scrfd_10g_bnkps.onnx` (16.9 MB), `2d106det.onnx`
(5.0 MB) and `genderage.onnx` (1.3 MB) (sizes from mirror listings
*(excerpt)*); `buffalo_l` has the same layout with `det_10g.onnx` and
`w600k_r50.onnx`.

**The most accurate we can run here:** AdaFace IR-101 (WebFace12M) for faces
at our sizes, with `buffalo_l`'s ResNet-50 and `antelopev2`'s ResNet-100 as
official ONNX alternatives or a second opinion for fusion; on the Mac, KP-RPE
and LVFace are one step further. All of these are research-only.

**Commercially usable (by the weights' licence):** AuraFace-v1 (Apache-2.0,
Hugging Face, so fetched on the Mac) and SFace (Apache-2.0, here). Both trail
the research models clearly on pose: CFP-FP 95.2 % for AuraFace against 99.3 %
for AdaFace IR-101 means about 6.5 times as many errors on frontal-to-profile
pairs. The way to have both state-of-the-art accuracy and commercial rights is
to buy them: InsightFace now offers commercial licences for its open packs
(`buffalo_l` and others) and for its InspireFace SDK.

### 2.5 Low resolution and image quality

**How many pixels a face needs.**

| Source | What it asks for | Our faces (IED 10-22 px) |
|---|---|---|
| ISO/IEC 19794-5:2011 (frontal document images) | IED of 120 px or more as best practice *(excerpt)* | 5-12 times too few |
| ISO/IEC 39794-5:2019 (its successor) | IED at least 90 px; best practice 240 px for live capture, 120 px for chip storage *(excerpt, via Merkle et al.'s 2022 survey)* | 4-9 times too few |
| ICAO (Doc 9303 refers to the ISO standard) | Keep the 90 px minimum (TAG-MRTD/17 WP 9) *(excerpt)* | as above |
| IEC 62676-4:2014 (CCTV, for human operators) | Identification 250 px/m, about 40 px across a 16 cm face; recognition 125 px/m, about 20 px *(excerpt, Axis white paper)* | Around the identification level (IED about 16 px at 250 px/m, est.) |
| NIST Face in Video Evaluation (NISTIR 8173, 2017) | 36 algorithms from 16 suppliers on 109 hours of video: small, unevenly lit and turned faces cause most errors; galleries were capped at 48,000 people *(excerpt)* | No pixel threshold found in what could be read |
| Low-resolution research | Faces under 32x32 px "begin to present significant challenges" (2018 survey); people match faces at chance near 10 px IPD (Peepers & Pixels, FG 2025); recognition degrades significantly as IPD falls from 30 to 5 px, and interpolation does not fully bring it back ("Reading Between the Eyes", 2025) *(all excerpt)* | |

A face 30 px tall in our fair (IED about 11 px) is at the level of TinyFace,
where the best public models identify 72-76 % at rank 1; a 60 px face (IED
about 21 px) is much easier. Passport-quality IED (90 px) is what the
high-resolution head view should aim at: there every modern model is
near-perfect on real faces.

**Quality assessment.** ISO/IEC 29794-5:2025 defines measurable quality
components for face images (inter-eye distance, head pose, sharpness,
exposure, occlusion, expression and others) and a unified score. OFIQ is
BSI's reference implementation (version 1.2.1): a C++ library under MIT whose
models (a Caffe face detector, ADNet landmarks, 3DDFA_V2 pose, BiSeNet face
parsing, an occlusion segmenter and, for the unified score, a MagFace
iResNet-50) are downloaded from the ISO portal or eu-LISA with their own
licences; both hosts returned 403 here. OFIQ was built for passport and
border photos. A DHS study with the Maryland Test Facility (January 2026)
found its unified score "provides extremely limited utility" in DHS use cases
across 16 commercial recognisers: faces it rated very poor still matched well,
and its quality filter did not substantially reduce errors, though it may help
pick the best of several images *(excerpt)*. The lesson: quality is relative
to a recogniser, so measure it with signals tied to the recogniser in use.

| Method | Year | Signal | Weights and form | Licence | Here? |
|---|---|---|---|---|---|
| ISO/IEC 29794-5 components, own code | 2025 | IED, yaw, pitch, roll, blur, exposure, contrast | None | Own | Yes |
| OFIQ 1.2.1 | 2024-26 | ISO components plus unified score | C++; models from the ISO portal or eu-LISA | MIT code; per-model licences | No |
| AdaFace feature norm | 2022 | Length of the raw feature, learnt as quality | Inside the model | As the model | If the export keeps it |
| MagFace | 2021 | Embedding magnitude | `.pth` on Google Drive | Apache-2.0 code; MS1MV2 | No |
| SER-FIQ | 2020 | Stability of the embedding under dropout, training-free | MXNet on Google Drive | CC BY-NC-SA 4.0 | No |
| SDD-FIQA | 2021 | Regression on similarity-distribution labels | IR-50 (MS1M) on Google Drive | Not checked | No |
| CR-FIQA | 2023 | Relative classifiability | `.pth` on Google Drive | CC BY-NC 4.0 | No |
| eDifFIQA T / S / M / L | 2024 | Diffusion-derived labels distilled into a regressor | `.pth` on OneDrive; ONNX exports on GitHub (6.9-262.8 MB) | MIT code; labels from VGGFace2 and a CosFace R100 | Yes (ONNX) |
| CLIB-FIQA | 2024 | CLIP with quality-factor labels | `.pth` on Google Drive | MIT | No |
| GraFIQs | 2024 | Gradient magnitudes of a pretrained recogniser, training-free | Own file share | CC BY-NC 4.0 | No |
| Apple Vision face capture quality | macOS 10.15+ | 0-1 capture quality | Operating-system API | Apple SDK terms | Mac only (not checked here) |

In NIST's FATE-Quality track, CR-FIQA's authors report first and second
places, and eDifFIQA-L's authors report first place in the kiosk-to-entry
scenario (authors' claims; the NIST page could not be opened).

**Super-resolution.** Upscaling cannot add identity information; a generative
face super-resolution model invents detail from its prior, which can make different
people look alike or one person unlike themselves. Grm et al. (CVPRW 2019)
found that face hallucination models trained on one degradation produce
"considerable artifacts" when blur and noise change slightly, so results on
artificially downsampled faces do not carry over to real low-resolution
images. Chai et al. (CVPR 2023) improve very low-resolution recognition in the
embedding space rather than in image quality. Luevano et al. (Idiap, IJCB
2026) found that a learned identity-aware super-resolution front-end does not
beat feeding the aligned low-resolution face directly into a strong backbone,
and that synthetic low-resolution benchmarks disagree with real TinyFace
results *(excerpt)*. Some setups gain a little by concatenating the features
of the upsampled and the super-resolved face *(excerpt)*. So: no
super-resolution in the identity path; a plain bicubic upscale of the head
crop for the detector is fine; super-resolved faces only for display.

### 2.6 Video: tracking, aggregation, decisions

**Tracking people.**

| Tracker | Year | Idea | Licence | Python |
|---|---|---|---|---|
| SORT | 2016 | Kalman filter, Hungarian matching on box overlap | Not checked | Roboflow `trackers` |
| ByteTrack | 2021 (ECCV 2022) | A second matching pass over low-score boxes keeps people who are partly occluded | MIT | `trackers` (Apache-2.0), `supervision` (MIT), Ultralytics (AGPL-3.0) |
| OC-SORT | 2022 (CVPR 2023) | Re-estimates the motion from observations after a gap; robust to occlusion and non-linear motion | MIT | `trackers`, Ultralytics (`ocsort.yaml`) |
| BoT-SORT | 2022 | ByteTrack plus camera-motion compensation and optional appearance re-identification | MIT | Ultralytics' default (re-identification off by default) |
| Deep OC-SORT and the BoxMOT collection | 2023 | OC-SORT with appearance features | BoxMOT is AGPL-3.0 | `boxmot` |

On MOT17, ByteTrack reports 63.1 HOTA and 77.3 IDF1, OC-SORT 63.2 and 77.5.
Roboflow's reimplementations score 58.4 (SORT), 60.1 (ByteTrack) and 61.9
(OC-SORT) HOTA on MOT17 and 45.0, 50.2, 51.8 on DanceTrack, where people look
alike and cross often, as in a fair crowd. Our camera does not move, so
BoT-SORT's motion compensation buys nothing. The project already uses
Ultralytics, whose `model.track(..., persist=True)` gives BoT-SORT, ByteTrack
or OC-SORT in one call, but it is AGPL-3.0 like the YOLO weights; the
Apache-2.0 `trackers` package works with any detector.

**Aggregating a track.** The IJB benchmarks build a template by averaging
the features of its images, and plain averaging of normalised embeddings is a
strong baseline. Better: weight each frame by its quality (feature norm, a
quality model, IED, yaw), so the many poor frames of a crowd do not drown the
few good ones. Learned aggregators go further. NAN (CVPR 2017) learns
attention weights; Probabilistic Face Embeddings (ICCV 2019) give every
embedding an uncertainty and fuse by it; CAFace (NeurIPS 2022) clusters the
frames into a few learnt centres and aggregates those, order-invariantly. With
an AdaFace backbone on IJB-S, CAFace raises rank-1 over plain averaging from
70.4 to 72.9 % (surveillance to single photo), 70.9 to 73.4 % (to the booking
set) and 35.1 to 39.3 % (surveillance to surveillance), with larger gains for
longer tracks (supplementary tables *(excerpt)*). Its weights are pickles on
Google Drive. InsightFace's PrivateFrame matches on the best 1, 3 or 5
eligible frames of a track (fast, balanced, accurate).

**When to decide and when to wait.** In a demo a late name costs little and a
wrong name costs a lot, so a track should carry three states: named, unknown,
and not decided yet.

- **Name** a track when its aggregated, calibrated score passes an accept
  threshold, beats the runner-up by a margin, and enough good frames have
  contributed.
- **Say "unknown"** only when enough good frames have been seen and the best
  score stays below a reject threshold. A turned or tiny face is "not decided
  yet", not "unknown".
- **Hysteresis**: a named track keeps its name down to a lower keep threshold
  (the current `KEEP_MARGIN`); switching to another name needs the accept
  threshold plus the margin over a sustained window.
- **Frames are not independent.** Consecutive frames of one face are nearly
  the same picture; adding up per-frame evidence (sums of log-likelihood
  ratios) makes the system overconfident. Score the aggregated embedding once,
  and calibrate thresholds on aggregates of similar size and quality.
- **One identity per track, one track per identity**: a person cannot stand in
  two places. Solve the assignment between active tracks and gallery people
  each update (Hungarian algorithm on calibrated scores) and demote the weaker
  claim to "not decided yet".
- **Re-identification after occlusion**: ByteTrack and OC-SORT keep a lost
  track for a while (Ultralytics' default buffer is 30 frames). After that,
  re-link a new track to an old identity by fresh face evidence, helped by
  body appearance and position. InsightFace's PersonAnalysis learns a body
  reference only after a clear face match and keeps face and body evidence
  separate; its body model is non-commercial. In the simulation, clothes
  colour histograms are a fair stand-in for a body model.

### 2.7 Open-set identification and calibration

**Deciding "unknown".** Most systems threshold the best cosine similarity, as
the demo does. Günther et al. ("Toward Open-Set Face Recognition", CVPRW 2017)
showed that thresholding verification-style scores is not enough on its own:
probabilities from an Extreme Value Machine fitted to the gallery rejected
unknowns better, and they called open-set recognition unsolved. They also
separate "known unknowns" (people seen but not enrolled, which can be used for
training and calibration) from "unknown unknowns". The avatars wearing their
own faces are exactly our known unknowns.

**The NIST measure.** FRTE 1:N reports the false negative identification rate
(FNIR: mated searches whose mate is not returned above threshold) at a fixed
false positive identification rate (FPIR: non-mated searches that return
anyone above threshold), for example FNIR at FPIR = 0.003, with the threshold
set separately for each algorithm *(excerpt)*. FPIR, not the per-comparison
false match rate (FMR), is what visitors experience: a stranger searched
against N gallery photos is falsely named with probability
FPIR ≈ 1 - (1 - FMR)^N ≈ N x FMR for small rates. With 20 people and 5 photos
each, an FPIR of 1e-3 per decision needs an FMR near 1e-5 per comparison. A
threshold read off LFW (SFace's 0.363) does not say which FPIR it gives.

**Score normalisation.**

- *Z-norm* (per gallery person): subtract the mean and divide by the spread of
  that person's scores against a cohort of other faces. Look-alike-prone
  gallery people (Doddington's "lambs") get a stricter effective threshold.
- *T-norm* (per probe): the same with the probe's scores against the cohort.
  Low-quality probes drift towards a cluster of "unrecognisable" embeddings
  (Chai et al., CVPR 2023) and score high against everybody; T-norm takes that
  inflation out. *AS-norm* uses only the top-K cohort scores.
- Cohort selection matters (Merati, Poh and Kittler, IEEE TIFS 2012): the most
  similar and the most dissimilar cohort members carry the information
  *(excerpt)*.

**Calibration across models and qualities.** Map raw scores to a common
currency, a log-likelihood ratio or a probability, per model: logistic
regression (Platt scaling) or beta calibration on mated and non-mated score
sets. FairCal (Salvador et al., ICLR 2022) calibrates per unsupervised cluster
of embeddings with beta calibration and equalises false positive rates across
groups without knowing anyone's demographic attributes. For us the important
condition is quality: low-quality mated scores slide towards impostor scores,
so one threshold for all face sizes is either too strict for small faces or
too loose for large ones. Calibrate per quality bin (IED, yaw) or with quality
as an input of the calibration model. Then the user sets one thing, a target
FPIR, and each backend translates it into its own score threshold.

**Per-person thresholds and near-twins.** Z-norm does most of the per-person
work. In addition, check the gallery itself: when two enrolled people score
above the accept threshold against each other, warn at enrolment and require a
larger margin between those two. The demo already computes the runner-up score
(`Match.runner_up`) but does not use it.

**The gallery: several photos per person.**

| Option | Strength | Weakness |
|---|---|---|
| Keep all, take the best match (the demo today) | Uses every pose and lighting; one good photo of the right pose wins | The noisiest photo sets the impostor rate; more photos per person means more chances of a false match (FPIR grows with N) |
| Mean template (normalised average) | Stable, one comparison per person, the IJB template convention | Blurs pose-specific detail; a profile probe matches a frontal-heavy mean poorly |
| Mixture: pose bins, a quality-weighted mean per bin | Pose-aware (Masi et al.'s idea) with few comparisons | Needs pose estimates and enough photos per bin |

Two hygiene steps whatever the option: drop near-duplicate photos (cosine
above 0.98, InsightFace's default), and embed the gallery with exactly the same
detector, landmarks, alignment and model as the probes, caching per model.

### 2.8 Measuring a face recognition system

- **Verification**: ROC or DET curves over mated and non-mated pairs; TAR
  (genuine accept rate) at FAR 1e-2 to 1e-5; equal error rate only as a
  summary.
- **Closed-set identification**: CMC curve, rank-1 (TinyFace, IJB-S report
  this). Rank-1 says nothing about strangers.
- **Open-set identification**: FNIR at FPIR (NIST FRTE 1:N), or the detection
  and identification rate (DIR) against the false alarm rate. This is the
  protocol that matches the demo.
- **Per-condition breakdowns**: by IED (face size), yaw, pitch, lighting,
  occlusion, blur, expression, and by demographic group where labels exist.
  NIST, IJB-S and TinyFace all show that averages hide the conditions that
  fail.
- **Statistics**: with zero errors in n independent trials the 95 % upper
  bound on the error rate is about 3/n (the rule of three). Claiming FPIR
  below 1e-3 needs about 3,000 independent non-mated searches, and frames of
  one track are not independent: count tracks or people, and bootstrap
  confidence intervals over people.

**Benchmarks and where they live.**

| Set | What it tests | Access | Reachable here? |
|---|---|---|---|
| LFW, CFP-FP, AgeDB-30, CALFW, CPLFW | High-quality verification; pose (CFP-FP, CPLFW); age | InsightFace's `.bin` packs (pickled lists of JPEG bytes) on Google Drive; `lfw.bin` (64.5 MB) on a GitHub release | Only `lfw.bin`, and it is a pickle |
| XQLFW | Cross-quality LFW pairs | Project page | No |
| TinyFace | Native low-resolution 1:N with distractors | QMUL project page | No |
| IJB-B, IJB-C | Mixed-quality templates, 1:1 and 1:N | NIST, on request | No |
| IJB-S | Surveillance video, open-set 1:N | On request | No |
| SCface, QMUL-SurvFace | Surveillance cameras at distance | Licence / project page | No |
| DigiFace-1M | Synthetic renders | Azure blob | No (403) |

On the Mac most of these can be fetched (the IJB sets after NIST approval).
Their terms are research terms; using them to evaluate a commercial product
needs checking per set.

**Tools.** The metrics are short enough to write in numpy and test against
scikit-learn's `roc_curve` (BSD-3, cp313 wheels). `bob.measure` (BSD-3) has
DET and EPC curves. InsightFace and CVLface ship IJB evaluation scripts
(research code).

### 2.9 Synthetic faces and recognisers

Known results point one way: rendered faces are a different domain for
recognisers trained on photos, and the gap is large enough to move operating
points.

- **Training on renders**: DigiFace-1M (Microsoft, WACV 2023) reaches 96.2 %
  on LFW and 86.4 % averaged over five benchmarks when trained only on renders
  *(excerpt)*, against 94.7 % for an IR-50 trained on CASIA-WebFace and 97.2 %
  for an IR-101 trained on MS1MV2 (the same five-benchmark average, AdaFace's
  README). Aggressive augmentation narrows the gap, and fine-tuning on 120K
  real images reaches 99.3 % on LFW *(excerpt)*.
- **Evaluating on renders**: Kortylewski et al. (CVPRW 2019) rendered 3D
  morphable model faces under controlled pose and found that networks trained
  on photos "cannot disentangle face pose and identity" *(excerpt)*. Colbois, de Freitas
  Pereira and Marcel (IJCB 2021) found that benchmarks built from synthetic
  identities can be a good proxy for relative comparisons *(excerpt)*.
  Luevano et al. (IJCB 2026) found synthetic low-resolution benchmarks
  disagreeing with real low-resolution data *(excerpt)*.
- **Our own number**: SFace scores 0.37-0.58 for the right person on avatar
  close-ups, against 0.6-1.0 for two photos of one person and below about 0.3
  for strangers (README). Mated scores on renders sit near the decision
  threshold.

What it means for testing in the rendered fair:

1. **Leakage.** The avatar's face texture is made from the same photos as the
   gallery, so a frontal render partly *is* the gallery photo: mated scores
   are inflated where the texture is good and deflated where the generic head
   shape and lighting differ. Texture and gallery must come from disjoint
   photos.
2. **Narrow impostor distribution.** A few dozen Rocketbox head models with
   painted faces are far fewer and more alike than real strangers; FPIR
   measured on them is not the real FPIR.
3. **Rendering is not optics.** Aliasing at 30 px, perfect focus, no sensor
   noise, no compression and no motion blur, unless simulated.
4. **Thresholds do not transfer.** Calibrate on real data (consented photos
   and webcam recordings); use renders for relative comparisons, regression
   tests and stress curves over size, pose and light, where their perfect
   labels are unbeatable.

### 2.10 Security and privacy, in brief

**Presentation attacks.** Cheap options exist for the webcam path:
MiniFASNet (Minivision's Silent-Face-Anti-Spoofing, Apache-2.0; ONNX exports of
1.7 MB on GitHub) and InsightFace 2.0's optional RGB liveness add-on
(non-commercial). A single-frame RGB check stops printed photos and screens
held up to the camera, not good masks. In the simulation every face is by
definition a presentation, so liveness must be off there.

**Privacy.**

- A face embedding used to identify someone is biometric data, a special
  category under GDPR Article 9.
- Embeddings can be inverted: Shahreza and Marcel reconstruct recognisable
  faces from ArcFace templates (NeurIPS 2023), reach an 87 % attack success
  rate on an ArcFace system from partly leaked templates, with only 20 % of
  the template known (ICASSP 2024),
  and still reconstruct from BioHashing- or encryption-protected templates
  (FG 2024) *(excerpt)*. Treat the cache (`faces/.embeddings.npz`) like the
  photos: local, deleted with the person.
- Do not keep embeddings of unknown passers-by beyond their track.
- Set `ORT_DISABLE_TELEMETRY=1` before ONNX Runtime loads (section 1).
- The EU AI Act prohibits real-time remote biometric identification in
  publicly accessible spaces for law enforcement (Article 5(1)(h)) and
  classifies remote biometric identification systems as high-risk
  *(excerpt; the Official Journal text was not read)*. A trade fair is a
  publicly accessible space: identify only enrolled, consenting people, say so
  on a sign, and get legal advice before any commercial deployment.
- MediaPipe's model cards exclude surveillance and identity recognition from
  the intended uses of BlazeFace and Face Mesh.

## 3. Recommended system for us

### 3.1 The shape

Person first, face second, decision per person track. The YOLO worker's
person boxes and track numbers feed the face stage; the face stage looks only
inside the head region of each tracked person, at the best resolution
available, embeds only frames worth embedding, and keeps the identity evidence
on the person track. Decisions have three states (a name, "unknown", "not
decided yet"), are calibrated per model and per face size, and are expressed
to the user as a target false-alarm rate (FPIR) that means the same with
either profile.

### 3.2 Stage by stage

| Stage | Best profile (research-only weights) | Commercial profile | Form and source; here? |
|---|---|---|---|
| 1. Person detection | Ultralytics YOLO26 as now (AGPL-3.0) | YOLOX-tiny or YOLOX-s (Apache-2.0, COCO) | YOLOX: official ONNX on GitHub releases, 20.2 / 35.9 MB; yes |
| 2. Person tracking | Ultralytics `track()` with ByteTrack or OC-SORT (AGPL-3.0) | ByteTrack or OC-SORT from Roboflow's `trackers` (Apache-2.0, pure Python) | PyPI; yes |
| 3. Head region | Square around the top of the person box (centre about 12 % of the box height below its top, side about 30 % of the height, plus 30 % margin; starting values to tune), cut from the full-resolution frame, or from the high-resolution head view when the simulation offers it; scaled to 160-192 px | Same | Own code |
| 4. Face and five landmarks | SCRFD-10G-KPS (`det_10g.onnx` from `buffalo_l`, 16.9 MB) on each head crop; a full-frame pass every 10th frame for faces without a person | YuNet 2026may (MIT, 0.2 MB); RetinaFace-MobileNetV2 (MIT, 12.5 MB) as the stronger option; BlazeFace full range (Apache-2.0, consented training data) if data provenance outweighs its model card | GitHub releases, GitHub LFS, GCS; yes |
| 5. Alignment | ArcFace five-point similarity transform to 112x112 (`cv2.estimateAffinePartial2D`, `cv2.warpAffine`); colour order and normalisation per model | Same | Own code |
| 6. Pose and quality | Yaw and pitch from the five landmarks; IED; sharpness (variance of the Laplacian); exposure; eDifFIQA-T (ONNX, 6.9 MB); AdaFace's feature norm if the export keeps it; MediaPipe Face Landmarker on high-resolution head views | The same geometric measures; the recogniser's feature norm if it proves informative | GitHub, GCS, own code; yes |
| 7. Embedding | **AdaFace IR-101, WebFace12M** (community ONNX, 260.7 MB, 512-d); optional second opinion `buffalo_l` ResNet-50 (official ONNX) for score fusion; on the Mac, KP-RPE ViT-B or LVFace-B to try | **AuraFace-v1** (Apache-2.0, ONNX, 261 MB, 512-d), fetched on the Mac; **SFace** (Apache-2.0, 128-d, as now) in the sandbox and as the fast fallback | GitHub (best: yes); Hugging Face (AuraFace: Mac only) |
| 8. Track aggregation | Quality-weighted mean of unit embeddings over the person track: keep the best 16 frames with a time decay; embed a track at most twice a second, and only frames better than its median so far | Same | Own code |
| 9. Matching | Per person: the best of pose-bin templates (frontal, left, right, profile; quality-weighted means) and the best single photo; T-norm against a cohort of about 1,000 non-gallery embeddings per model; calibration per model and IED bin; runner-up margin; three states; one identity per track (Hungarian); hysteresis | Same; cohort and calibration data must be licence-clean (consented photos, own renders) | Own code |
| 10. Enrolment | Same detector, aligner and model as live; reject photos with IED below 40 px (warn below 90), with two faces of similar size, or blurred; drop duplicates above 0.98; pose bins from landmarks; near-twin check; one cache per model | Same | Own code |
| 11. Liveness (webcam only, optional) | MiniFASNet (Apache-2.0, ONNX 1.7 MB) | Same | GitHub; yes |
| 12. Privacy | `ORT_DISABLE_TELEMETRY=1` at import; caches local and deleted with the person; no storage of unknown people's embeddings | Same | Own code |

A third setting fits the user's "best would be a solution that allows both":
**licensed**, the best profile with `buffalo_l` (or `antelopev2`) in place of
AdaFace under a commercial licence bought from InsightFace. It keeps
state-of-the-art accuracy on high-quality faces (IJB-C 97.25 against 97.66)
and gives up only AdaFace's low-quality edge. No commercial licence offer for
AdaFace was found, and the patent in section 1 concerns it.

### 3.3 Why these choices

- **AdaFace IR-101 (WebFace12M) for the best profile**: the best documented
  low-quality results among weights we can fetch here (TinyFace 72.3 %, IJB-S
  71.4 %, IJB-C 97.66), at the cost of any ResNet-100. Its ONNX is a
  community export, so the first task is to check it against its published
  accuracy (section 4), with `buffalo_l` (official ONNX, IJB-C 97.25) as the
  ready fallback.
- **SCRFD on upscaled head crops**: small faces become ordinary-sized for the
  detector, landmarks steadier, and the InsightFace models were trained on
  RetinaFace/SCRFD landmarks. YuNet in the commercial profile is MIT, 0.2 MB
  and already integrated; it is weaker on hard faces, which matters less on
  upscaled crops.
- **AuraFace for the commercial profile, provisionally**: the only Apache-2.0
  ResNet-100 recogniser found, but with no IJB or TinyFace numbers, about 6.5
  times the frontal-to-profile errors of AdaFace (CFP-FP), and an unexplained
  training set. Adopt it only if it beats SFace on our evaluation; otherwise
  keep SFace.
- **The high-resolution head view is the biggest lever.** Moving a face from
  IED 10-22 px to 60-90 px does more than any model change. Request it for
  tracks that are still undecided and roughly frontal (yaw under about 30°).
  With a real webcam, the same idea is cropping heads from the sensor's full
  resolution while detecting people on a smaller frame.
- **No super-resolution and no frontalisation** in the identity path
  (sections 2.3 and 2.5).
- **Score fusion** of AdaFace and `buffalo_l` usually helps a little and
  doubles the embedding cost; worth one evaluation run, not a default.

### 3.4 Expected accuracy

Benchmark anchors (higher is better; "-" means not published):

| | Today: SFace | Commercial: AuraFace-v1 | Licensed: `buffalo_l` R50 | Best: AdaFace IR-101 | Mac option: KP-RPE ViT-B |
|---|---|---|---|---|---|
| LFW | 99.40 | 99.65 | 99.83 | 99.82 | 99.82 |
| CFP-FP (frontal against profile) | - | 95.19 | 99.33 | 99.26 | 99.30 |
| CPLFW (cross-pose) | - | 90.93 | - | 94.57 | 95.65 |
| AgeDB-30 | - | 96.10 | 98.23 | 98.00 | 98.10 |
| IJB-C TAR @ FAR 1e-4 | - | - | 97.25 | 97.66 | 97.82 |
| TinyFace rank-1 (native low resolution) | - | - | - | 72.3 | 76.1 |
| IJB-S surveillance to single photo, rank-1 | - | - | - | 71.4 | - |

In the fair nothing has been measured yet beyond today's baseline (SFace: 98
of 120 frames right at about 30 px). Targets for the evaluation to confirm or
refute, not predictions: per frame at 30 px, the best profile above 90 %;
per track, at least 95 % of enrolled people's tracks with a second of yaw
under 45° named correctly within 2 seconds, at an FPIR of at most 1 % per
track (the acceptance test in section 4.4); the
commercial profile within a few points of the best on frontal faces and
clearly behind on turned and small ones.

### 3.5 Cost

| Model | Size | GFLOPs | Here, 4 shared cores (est.) | Mac CPU (est.) | Mac CoreML (est.) |
|---|---|---|---|---|---|
| YuNet on a 160 px head crop | 0.2 MB | tiny | about 1 ms (0.69 ms at 160x120 measured by OpenCV Zoo on an i7-12700K) | about 1 ms | not needed |
| SCRFD-10G on a 160 px crop | 16.9 MB | about 0.8 | 3-8 ms | 2-5 ms | 1-3 ms |
| SFace | 38.7 MB | n/a | 5-10 ms (5.1 ms measured by OpenCV Zoo on an i7-12700K; the README says about 10 ms) | 3-6 ms | - |
| eDifFIQA-T | 6.9 MB | about 0.5 | 3-6 ms | 2-4 ms | about 1 ms |
| `buffalo_l` ResNet-50 | 174 MB | 6.3 | 40-80 ms | 20-40 ms | 3-8 ms |
| AdaFace IR-101 or AuraFace (ResNet-100) | 261 MB | 12.1 | 80-160 ms | 40-80 ms | 5-15 ms |
| KP-RPE or LVFace ViT-B | n/a | about 11 | 80-160 ms | 40-80 ms | 5-15 ms |

The estimates assume ONNX Runtime FP32 reaching 150-300 GFLOP/s on four x86
cores and two to three times that on an M-series CPU (GFLOPs as published,
counted as multiply-adds), and assume that CoreML runs most of the graph on
the Neural Engine or GPU; the CoreML provider supports the operators these
ResNets and ViTs use (Conv, PRelu, Gemm, LayerNormalization, Gelu), but
`PRelu` slope shapes and dynamic input sizes can force parts back to the CPU.
Measuring this is step one of the evaluation.

**Budget.** Twenty visible people embedded twice a second is 40 embeddings a
second: comfortable for a ResNet-100 under CoreML (batched), far too much for
four CPU cores. In the sandbox, test at one embedding per track every two
seconds, or with the ResNet-50 or the MobileFaceNet.

### 3.6 The switch

One option chooses the stack: `--face-profile best|commercial|licensed`
(environment variable `BIFOLD_FACE_PROFILE`; default `best`, as the user
accepts research weights). Behind it, a small registry describes each model:

```python
@dataclass(frozen=True)
class Recogniser:
    id: str                       # "adaface_ir101_wf12m", "auraface_v1", "sface_2021dec", ...
    files: tuple[ModelFile, ...]  # URL, SHA-256 and size, as in downloads.py
    bgr: bool                     # AdaFace: BGR; InsightFace and AuraFace: RGB
    mean: float; std: float       # 127.5 / 127.5 for all of these
    dim: int                      # 512, or 128 for SFace
    licence: str                  # shown in `bifold-faces list`

PROFILES = {
    "best":       Stack("scrfd_10g_kps", "adaface_ir101_wf12m", quality="ediffiqa_t"),
    "commercial": Stack("yunet_2026may", "auraface_v1", quality="geometry", fallback="sface_2021dec"),
    "licensed":   Stack("scrfd_10g_kps", "buffalo_l_r50", quality="ediffiqa_t"),
}
```

- Every stage keeps one interface: the detector returns boxes, five landmarks
  and a score; the embedder takes a batch of aligned crops and returns unit
  vectors and raw norms; the quality model returns a score per crop.
- Caches are per model (`faces/.embeddings.<model>.npz`), as are the cohort
  and the calibration (`models/calibration/<model>.json`), because embeddings
  of different models cannot be compared.
- The user's threshold is a target FPIR per decision; each model's
  calibration turns it into a score threshold per face-size bin. Switching
  profiles then keeps the same false-alarm behaviour, not the same raw number.
- Downloads go through `downloads.ModelFile` with SHA-256 checks. The
  InsightFace packs are zips, so the downloader needs to extract single
  members (`det_10g.onnx`, `w600k_r50.onnx`).
- ONNX Runtime sessions: CoreML (`MLProgram`, all compute units) then CPU on
  the Mac, CPU here; fixed input shapes; batches of crops.
- Tests run with fake models; in the sandbox every model of the best and
  licensed profiles can run, and the commercial profile runs with SFace
  because AuraFace is on Hugging Face only.

## 4. The evaluation plan

### 4.1 Step zero: check every model before trusting it

- **Files**: SHA-256, input and output names and shapes, opset; record them
  in the registry.
- **Accuracy against published numbers**, on the Mac (the `.bin` test packs
  are pickles: load them with an unpickler that allows only numpy's array
  reconstruction, or rebuild them as image folders once). LFW, CFP-FP and
  AgeDB-30 must come within about 0.2 points of the published values:
  AdaFace IR-101 99.82 / 99.26 / 98.00, `buffalo_l` 99.83 / 99.33 / 98.23,
  AuraFace 99.65 / 95.19 / 96.10, SFace 99.40 on LFW. Run each model with both
  colour orders: the right one is the one that reproduces the paper.
- **Speed**: milliseconds per crop at batch 1, 8 and 32, here (CPU) and on the
  Mac (CPU and CoreML), plus how much of the graph CoreML actually takes.

### 4.2 Metrics

- Pairs: ROC and DET; TAR at FAR 1e-2, 1e-3, 1e-4 (1e-5 where there are
  enough pairs).
- Open-set 1:N, the demo's case: FNIR at FPIR 1e-1, 1e-2 and 1e-3, and the
  detection and identification rate against FPIR.
- Closed-set rank-1 only as a sanity check.
- **Track level, what a visitor sees**: share of enrolled people's tracks named
  correctly; share of strangers' tracks left unnamed; wrong names per track and
  per hour; time to the first correct name (median and 90th percentile); name
  switches within a track; one name on two tracks at once.
- Calibration: achieved FPIR against the target, per profile and per face-size
  bin (a factor of 2 either way is acceptable at these sample sizes).
- Cost: milliseconds per stage, embeddings per second, CPU load, memory.

All per condition: IED bins (under 10, 10-15, 15-22, 22-32, 32-45, 45-64,
64-90, over 90 px), yaw bins of 15°, pitch, lighting, occlusion, blur.

### 4.3 Protocols and data

| Protocol | Data | Ground truth | Runs | Purpose |
|---|---|---|---|---|
| P1 Model check | Public test packs | Published numbers | Mac | Are the exports right? (4.1) |
| P2 Rendered stills | Known people on avatars, textures from photo set A, gallery from a disjoint photo set B; rendered at controlled IED (8-90 px), yaw (0-90° in 15° steps), pitch (±30°), 3-4 lightings, occlusion by hands, hair and other people, blur, JPEG/H.264 compression, sensor noise. Strangers: avatars with their own faces, and avatars wearing faces of consenting people who are not enrolled | Renderer: identity, head pose, IED | Here and Mac | Relative comparison of profiles; stress curves; regression tests |
| P3 Rendered video | Scripted fair scenes with known people walking, turning, occluded; fixed seeds | Renderer | Here (slower than real time) and Mac | Track-level metrics, decision timing, exclusivity |
| P4 Real photos | At least 20 consenting people, at least 10 photos each over several days, lightings and poses; enrolment and probe photos disjoint; non-enrolled consenting people as strangers; downsampled copies for a size curve | Labels at capture | Mac | Real mated and non-mated scores; calibration |
| P5 Webcam | Recorded sessions at 1-6 m with enrolled and non-enrolled consenting people | Per-track labels, pipeline output reviewed by hand | Mac | Native low resolution; thresholds are set here |
| P6 Public benchmarks | LFW, CFP-FP, AgeDB, CPLFW, XQLFW, TinyFace; IJB-B/C if NIST grants access | Benchmark protocols | Mac | Comparison with the literature (research terms) |

Downsampled photos (P4) are not native low resolution (Grm et al.; Luevano et
al.), so P5 is the arbiter for small faces. Cohorts for T-norm and
calibration data must follow the profile's licence: public research sets are
fine for the best profile; the commercial profile uses consented photos and
own renders only.

### 4.4 Statistics and acceptance

- Count people and tracks, not frames; bootstrap confidence intervals over
  people; hold out people (not just photos) between tuning and testing.
- To show FPIR at most 1e-3 with no false names observed takes about 3,000
  independent stranger tracks: only renders (P2, P3) reach that; real data
  (P4, P5) checks the 1e-2 level.
- Proposed acceptance for the best profile, on P5 at an FPIR of 1 % per
  track: at least 95 % of enrolled people's tracks with at least one second of
  yaw under 45° named correctly within 2 seconds; no identity on two tracks at
  once; achieved FPIR within a factor of 2 of the target; better than today's
  SFace at equal FPIR on P2 to P5.
- AuraFace replaces SFace in the commercial profile only if it wins on P4 and
  P5.

### 4.5 Pitfalls of synthetic-only evaluation

Leakage between texture photos and gallery photos; an impostor distribution
narrowed by a few dozen head models; rendering without optics (no blur,
noise or compression unless added); the demographic make-up of the Rocketbox
library; thresholds tuned on renders; frames counted as independent; tuning
and testing on the same people; embeddings of one model compared with another
model's cache.

### 4.6 Tooling

A `bifold-faces eval` command reading a manifest (image or clip, identity,
conditions), writing metrics as JSON and curves as images; a small numpy
metrics module (ROC, DET, CMC, FNIR at FPIR, calibration) tested against
scikit-learn's `roc_curve`; renders and fake models for deterministic tests in
the sandbox.

## 5. Risks and open questions

1. **Licences and provenance.**
   - AdaFace's weights carry the repository's MIT licence, but were trained on
     WebFace (research-only terms), and a granted US patent (12,525,057)
     describes a recognition system with AdaFace's quality-adaptive margins.
     Its assignee and claims were not read. Research profile only.
   - EdgeFace: BSD-3-Clause on GitHub, CC BY-NC-SA 4.0 on Idiap's model cards
     *(excerpt)*. Treated as non-commercial and not recommended.
   - AuraFace: Apache-2.0, but "a commercial dataset" is not described, and
     whether training started from Glint360K weights is not stated. Take only
     its `glintr100.onnx`; the detection and landmark files beside it are
     InsightFace's non-commercial ones.
   - YuNet and the RetinaFace port are MIT, but WIDER FACE is tagged
     CC BY-NC-ND 4.0 *(excerpt)*; BlazeFace's training data is clean but its
     card excludes identity recognition. Both are legal questions, not
     technical ones.
   - The person detector (Ultralytics YOLO) and its trackers are AGPL-3.0; the
     commercial profile swaps them for YOLOX and `trackers`.
   - InsightFace's commercial licence for `buffalo_l`: terms and price unknown
     until asked.
   - Public benchmarks' terms for evaluating a commercial product, per set.
2. **Community ONNX exports** (AdaFace, EdgeFace, eDifFIQA, MiniFASNet): their
   fidelity is unknown until step zero passes (colour order, normalisation,
   opset, whether AdaFace's norm output survives). Pin their SHA-256 in the
   registry; a re-uploaded release asset would otherwise change silently.
3. **Compute.** A ResNet-100 per face is too slow for live use on four CPU
   cores; CoreML coverage and FP16 effects on the Mac are untested. If CoreML
   changes embeddings measurably, the gallery cache and the calibration must
   be made with the same execution provider as the live stream.
4. **Simulation gap.** Mated scores on renders sit near the threshold today
   (SFace 0.37-0.58); leakage between texture and gallery photos; the generic
   head shape (the separate review's head fitting changes this); few head
   models. Renders rank stacks; real data sets thresholds.
5. **Face size.** At IED 10-15 px even the best public models are weak
   (TinyFace-like rank-1 around 72-76 % against large distractor sets). The
   design leans on waiting for good frames and on the high-resolution head
   view; how often that view is available, and at what cost in rendering
   time, is an open question for the other review.
6. **Calibration data.** Enough real mated and non-mated tracks per face-size
   bin to set thresholds at 1 % FPIR; licence-clean cohorts for the
   commercial profile.
7. **Look-alikes and relatives** in a small gallery: the near-twin check warns
   but cannot fix them.
8. **Demographic differences** in error rates are documented for most
   recognisers (NIST FRVT Part 3, NISTIR 8280, 2019; not opened here), and the
   Rocketbox library's make-up limits what renders can show. Report results
   per group where labels exist.
9. **OpenCV 5.** The repo pins the fixed-shape YuNet 2023mar; the 2026may
   file has dynamic shapes. `FaceRecognizerSF` and `FaceDetectorYN` are still
   in OpenCV 5's source; the ONNX Runtime engine needs a custom build.
10. **Privacy and law.** GDPR Article 9 consent for every enrolled person; the
    AI Act's classification of a live identification demo in a publicly
    accessible space; signage; deletion. ONNX Runtime telemetry must be off.
11. **Not verified first-hand**: ISO, ICAO and IEC figures, NIST pages, the
    patent, the WebFace terms, the AuraFace and EdgeFace model cards,
    CAFace's IJB-S numbers, FaceMoE's numbers, the DHS OFIQ study, the EU AI
    Act text (all from search excerpts); SDD-FIQA's licence; UniFace's main
    results.

## 6. References

Systems, evaluations and runtime

- InsightFace (licence statement, news, commercial licensing contacts): https://github.com/deepinsight/insightface
- InsightFace model zoo: https://github.com/deepinsight/insightface/blob/master/model_zoo/README.md; Python library 2.x: https://github.com/deepinsight/insightface/blob/master/python-package/README.md, https://github.com/deepinsight/insightface/blob/master/python-package/docs/model_zoo.md, https://github.com/deepinsight/insightface/blob/master/python-package/docs/person_analysis.md, https://github.com/deepinsight/insightface/blob/master/python-package/docs/runtime.md, https://github.com/deepinsight/insightface/blob/master/python-package/CHANGELOG.md
- InsightFace releases (packs checked by HEAD request): https://github.com/deepinsight/insightface/releases/tag/v0.7, https://github.com/deepinsight/insightface/releases/tag/model-zoo
- NIST Face in Video Evaluation, NISTIR 8173 *(excerpt)*: https://nvlpubs.nist.gov/nistpubs/ir/2017/NIST.IR.8173.pdf
- NIST FRTE 1:N *(excerpt)*: https://pages.nist.gov/frvt/html/frvt1N.html; FRTE 1:1: https://pages.nist.gov/frvt/html/frvt11.html; FATE-Quality: https://pages.nist.gov/frvt/html/frvt_quality.html
- NIST FRVT Part 3, demographic effects, NISTIR 8280 (not opened here): https://nvlpubs.nist.gov/nistpubs/ir/2019/NIST.IR.8280.pdf
- ONNX Runtime on PyPI: https://pypi.org/project/onnxruntime/; CoreML execution provider: https://github.com/microsoft/onnxruntime/blob/gh-pages/docs/execution-providers/CoreML-ExecutionProvider.md; telemetry: https://github.com/microsoft/onnxruntime/blob/v1.29.0/docs/Privacy.md
- OpenCV 5 DNN engines: https://github.com/opencv/opencv/blob/5.x/modules/dnn/include/opencv2/dnn/dnn.hpp; face API: https://github.com/opencv/opencv/blob/5.x/modules/objdetect/include/opencv2/objdetect/face.hpp

Face detection and landmarks

- YuNet (OpenCV Zoo, including the 2026may file): https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet; paper: https://link.springer.com/article/10.1007/s11633-023-1423-y; training code: https://github.com/ShiqiYu/libfacedetection.train
- OpenCV Zoo benchmark: https://github.com/opencv/opencv_zoo/blob/main/benchmark/README.md
- SCRFD: https://github.com/deepinsight/insightface/tree/master/detection/scrfd, https://arxiv.org/abs/2105.04714
- RetinaFace: https://arxiv.org/abs/1905.00641, https://github.com/deepinsight/insightface/tree/master/detection/retinaface; MIT retrain with ONNX: https://github.com/yakhyo/retinaface-pytorch
- YOLO5Face: https://github.com/deepcam-cn/yolov5-face; YOLOv8-face: https://github.com/derronqi/yolov8-face; yolo-face: https://github.com/YapaLab/yolo-face; YOLO-FaceV2: https://github.com/Krasjet-Yu/YOLO-FaceV2, https://arxiv.org/abs/2208.02019; MogFace: https://github.com/damo-cv/MogFace
- MediaPipe model cards: https://storage.googleapis.com/mediapipe-assets/MediaPipe%20BlazeFace%20Model%20Card%20(Short%20Range).pdf, https://storage.googleapis.com/mediapipe-assets/MediaPipe%20BlazeFace%20Model%20Card%20(Full%20Range).pdf, https://storage.googleapis.com/mediapipe-assets/Model%20Card%20MediaPipe%20Face%20Mesh%20V2.pdf
- MediaPipe models: https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_full_range/float16/latest/blaze_face_full_range.tflite, https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
- `uniface` (ONNX exports, model table, licence table): https://github.com/yakhyo/uniface, https://github.com/yakhyo/uniface/blob/main/docs/models.md, https://github.com/yakhyo/uniface/blob/main/docs/license-attribution.md
- WIDER FACE: http://shuoyang1213.me/WIDERFACE/ (licence tag from the Hugging Face loader *(excerpt)*)
- 3DDFA_V2: https://github.com/cleardusk/3DDFA_V2

Alignment and pose

- Templates: https://github.com/deepinsight/insightface/blob/master/python-package/insightface/utils/face_align.py, https://github.com/opencv/opencv/blob/5.x/modules/objdetect/src/face_recognize.cpp
- ARoFace *(excerpt)*: https://arxiv.org/abs/2407.14972, https://github.com/msed-Ebrahimi/ARoFace
- KP-RPE: https://arxiv.org/abs/2403.14852; CVLface: https://github.com/mk-minchul/CVLface
- Masi et al., Pose-Aware Face Recognition in the Wild: https://openaccess.thecvf.com/content_cvpr_2016/html/Masi_Pose-Aware_Face_Recognition_CVPR_2016_paper.html
- Frontalisation *(excerpt)*: Cao et al., IJCV 2020: https://link.springer.com/article/10.1007/s11263-019-01229-6; NeurIPS 2018 reviews: https://papers.nips.cc/paper_files/paper/2018/file/1415db70fe9ddb119e23e9b2808cde38-Reviews.html; CVPRW 2026: https://openaccess.thecvf.com/content/CVPR2026W/FoundGen-Bio/supplemental/Cevikalp_Frontal_Face_Synthesis_CVPRW_2026_supplemental.pdf

Recognition

- SphereFace: https://arxiv.org/abs/1704.08063; CosFace: https://arxiv.org/abs/1801.09414; ArcFace: https://arxiv.org/abs/1801.07698
- CurricularFace: https://github.com/HuangYG123/CurricularFace, https://arxiv.org/abs/2004.00288; MagFace: https://github.com/IrvingMeng/MagFace; ElasticFace: https://github.com/fdbtrs/ElasticFace, https://arxiv.org/abs/2109.09416
- AdaFace: https://github.com/mk-minchul/AdaFace, https://arxiv.org/abs/2204.00964; ONNX export: https://github.com/yakhyo/adaface-onnx
- UniFace: https://openaccess.thecvf.com/content/ICCV2023/html/Zhou_UniFace_Unified_Cross-Entropy_Loss_for_Deep_Face_Recognition_ICCV_2023_paper.html, https://github.com/Jason-Zhou-JC/UniFace
- SFace: https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface, https://github.com/zhongyy/SFace, https://arxiv.org/abs/2205.12010
- `arcface_torch` model table and Partial FC (Glint360K terms): https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch, https://github.com/deepinsight/insightface/tree/master/recognition/partial_fc, https://arxiv.org/abs/2203.15565
- TransFace: https://github.com/DanJun6737/TransFace, https://arxiv.org/abs/2308.10133
- TopoFR: https://github.com/modelscope/facechain/tree/main/face_module/TopoFR, https://arxiv.org/abs/2410.10587
- LVFace: https://github.com/bytedance/LVFace, https://arxiv.org/abs/2501.13420
- EdgeFace: https://github.com/otroshi/edgeface, https://arxiv.org/abs/2307.01838; model card *(excerpt)*: https://huggingface.co/Idiap/EdgeFace-Base; ONNX export: https://github.com/yakhyo/edgeface-onnx
- AuraFace *(excerpt)*: https://huggingface.co/fal/AuraFace-v1, https://huggingface.co/blog/isidentical/auraface
- GhostFaceNets: https://github.com/HamadYA/GhostFaceNets; dlib models: https://github.com/davisking/dlib-models; Intel face-reidentification-retail-0095: https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/face-reidentification-retail-0095
- Low quality: Chai et al., CVPR 2023 *(excerpt)*: https://openaccess.thecvf.com/content/CVPR2023/html/Chai_Recognizability_Embedding_Enhancement_for_Very_Low-Resolution_Face_Recognition_and_Quality_CVPR_2023_paper.html; PETALface: https://arxiv.org/abs/2412.07771; FaceMoE *(excerpt)*: https://arxiv.org/abs/2606.32040; centre-aligned representations: https://arxiv.org/abs/2609.01014; TinyFace: https://arxiv.org/abs/1811.08965
- AdaFace-related US patent 12,525,057 *(excerpt)*: https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12525057

Training data

- InsightFace dataset list: https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_
- WebFace260M: https://arxiv.org/abs/2103.04098
- DigiFace-1M and its Research Use of Data Agreement: https://github.com/microsoft/DigiFace1M, https://arxiv.org/abs/2210.02579
- Vec2Face *(excerpt)*: https://arxiv.org/abs/2409.02979
- MS-Celeb-1M withdrawal *(excerpt)*: https://betanews.com/2019/06/07/microsoft-deletes-ms-celeb-1m-database

Face size, quality and super-resolution

- ISO/IEC 19794-5, 39794-5:2019 and 29794-5:2025 (catalogue pages): https://standards.iteh.ai/catalog/standards/iso/db31d52e-3fc7-4986-9b55-ee70ea34d8b4/iso-iec-19794-5-2005, https://standards.iteh.ai/catalog/standards/iso/1b0fc469-1e6c-454d-a7cb-df3cae12b28f/iso-iec-39794-5-2019, https://standards.iteh.ai/catalog/standards/iso/7b8c92cc-f70a-4273-832e-a2a8a95cfdc7/iso-iec-29794-5-2025
- Merkle et al., State of the Art of Quality Assessment of Facial Images *(excerpt)*: https://arxiv.org/abs/2211.08030
- ICAO TAG-MRTD/17 WP 9 *(excerpt)*: https://www2023.icao.int/Meetings/TAG-MRTD/Documents/Tag-Mrtd-17/TagMrtd17_WP009.pdf
- IEC 62676-4 pixel densities (Axis white paper) *(excerpt)*: https://whitepapers.axis.com/en-us/pixel-density-based-on-iec-62676-4-2014
- OFIQ: https://github.com/BSI-OFIQ/OFIQ-Project, https://bsi.bund.de/dok/OFIQ-e; DHS study on OFIQ *(excerpt)*: https://www.biometricupdate.com/202601/dhs-study-suggests-ofiq-value-extremely-limited-for-its-biometrics-use-cases
- SER-FIQ: https://github.com/pterhoer/FaceImageQuality; SDD-FIQA: https://github.com/Tencent/TFace/tree/master/quality; CR-FIQA: https://github.com/fdbtrs/CR-FIQA; eDifFIQA: https://github.com/LSIbabnikz/eDifFIQA, ONNX: https://github.com/yakhyo/face-image-quality-assessment; CLIB-FIQA: https://github.com/oufuzhao/CLIB-FIQA; GraFIQs: https://github.com/jankolf/GraFIQs
- Peepers & Pixels *(excerpt)*: https://arxiv.org/abs/2503.20108; Reading Between the Eyes *(excerpt)*: https://arxiv.org/abs/2505.24247; low-quality face recognition survey *(excerpt)*: https://arxiv.org/abs/1805.11519
- Grm et al., Face Hallucination Revisited: https://openaccess.thecvf.com/content_CVPRW_2019/html/Biometrics/Grm_Face_Hallucination_Revisited_An_Exploratory_Study_on_Dataset_Bias_CVPRW_2019_paper.html; Luevano et al., IJCB 2026 *(excerpt)*: https://www.idiap.ch/paper/synth-lrfr/, https://arxiv.org/abs/2608.06580; identity-preserving SR on SCface *(excerpt)*: https://arxiv.org/abs/2010.12249

Video

- ByteTrack: https://github.com/ifzhang/ByteTrack; OC-SORT: https://github.com/noahcao/OC_SORT; BoT-SORT: https://github.com/NirAharon/BoT-SORT
- Roboflow `trackers`: https://github.com/roboflow/trackers; `supervision`: https://github.com/roboflow/supervision; Ultralytics (AGPL-3.0, tracker configs): https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/trackers; BoxMOT: https://github.com/mikel-brostrom/boxmot
- CAFace: https://github.com/mk-minchul/caface, https://arxiv.org/abs/2210.10864 (IJB-S tables *(excerpt)*); Probabilistic Face Embeddings: https://arxiv.org/abs/1904.09658; NAN: https://openaccess.thecvf.com/content_cvpr_2017/html/Yang_Neural_Aggregation_Network_CVPR_2017_paper.html
- Person detectors with permissive licences: YOLOX (official ONNX): https://github.com/Megvii-BaseDetection/YOLOX/tree/main/demo/ONNXRuntime; RF-DETR: https://github.com/roboflow/rf-detr; RT-DETR: https://github.com/lyuwenyu/RT-DETR; D-FINE: https://github.com/Peterande/D-FINE

Open-set identification and calibration

- Günther et al., Toward Open-Set Face Recognition: https://arxiv.org/abs/1705.01567
- FairCal: https://arxiv.org/abs/2106.03761
- Merati, Poh and Kittler, user-specific cohort selection and score normalisation, IEEE TIFS 7(4), 2012, doi:10.1109/TIFS.2012.2198469

Synthetic faces

- Kortylewski et al., CVPRW 2019 *(excerpt)*: https://shapemodelling.cs.unibas.ch/gravis-literature/publications/2019/2019-Kortylewski_Analyzing_and_Reducing_CVPRW_2019_paper.pdf
- Colbois, de Freitas Pereira and Marcel, IJCB 2021 *(excerpt)*: https://arxiv.org/abs/2106.04215

Security and privacy

- MiniFASNet: https://github.com/minivision-ai/Silent-Face-Anti-Spoofing; ONNX: https://github.com/yakhyo/face-anti-spoofing; InsightFace liveness add-on: https://github.com/deepinsight/insightface/blob/master/python-package/docs/liveness.md
- Template inversion *(excerpt)*: https://papers.neurips.cc/paper_files/paper/2023/file/29e4b51d45dc8f534260adc45b587363-Paper-Conference.pdf, https://publications.idiap.ch/attachments/papers/2024/OtroshiShahreza_ICASSP-2_2024.pdf, https://publications.idiap.ch/attachments/papers/2024/OtroshiShahreza_FG-2_2024.pdf
- EU AI Act Article 5 *(excerpt)*: https://www.euaiact.com/article/5; overview: https://fpf.org/blog/red-lines-under-the-eu-ai-act-restricting-real-time-remote-biometric-identification-systems-for-law-enforcement-purposes/; GDPR: https://eur-lex.europa.eu/eli/reg/2016/679/oj
