# Face recognition core (F1): model checks and first measurements

Measured on 1 November 2026 in the development sandbox (4 shared x86 cores, ONNX Runtime 1.31.1 on the CPU,
OpenCV 5.0), with the models of [`recognition/registry.py`](../../src/bifold_demo/recognition/registry.py). This is
"step zero" of the [review](face_recognition_sota.md) (section 4.1: check every model before trusting it) and the
first look at how the models separate people. Thresholds, open-set decisions and track-level results are the next
steps' (F2, E1).

**In short.**

- Every model file has its pinned SHA-256 (the InsightFace, AdaFace and eDifFIQA hashes also match UniFace's
  independent pins), every graph has only standard ONNX operators, and every embedder is deterministic and
  batch-independent.
- The AdaFace IR-101 community export works: 99.87 % on a balanced 1,500-pair LFW subset (published 99.82 % on the
  full protocol), the best separation of all (d' 8.2), and BGR input as its exporter says. It stays the default.
- The registered colour orders are right for every model, but the wrong order costs less than expected (d' about
  0.2-0.3 on renders, 1.3-1.5 on photos).
- On avatar renders every model names every turned render (up to 60 degrees, down to 12 pixels between the eyes)
  from one frontal render among 20 avatars. Avatars look more alike to the models than LFW's strangers do, so
  thresholds must come from real faces.
- **Disappointing**: AdaFace's norm, its built-in quality signal, does not follow pose or size on the renders, and
  this machine is too slow for the full benchmark protocol with ResNet-100s (0.4-0.6 s per face).

**Data.** No image of a real person is in this repository; only numbers made from them are.

- **LFW** (6,000 pairs, 3,000 of one person): InsightFace's `lfw.bin` test pack (112 x 112 aligned faces), from the
  GitHub release `leondgarse/Keras_insightface` `v1.0.0` (64.5 MB, SHA-256 `00f9f2a6...`), read as plain data (a
  pickle loader that refuses every class and function). CFP-FP and AgeDB-30 packs are only on Hugging Face, which
  cannot be reached from the sandbox: `bifold-faces check --pack cfp_fp.bin agedb_30.bin` runs them on the Mac.
- **Avatar renders**: 20 Rocketbox avatars (different heads and faces: different identities), each rendered 81
  times (yaw -60 to 60 in steps of 15, pitch -15, 0, 15, three lights), 640 x 480, about 79 pixels between the eyes
  when frontal. Detection and pose use all 1,620; the recognisers, at 0.1-0.6 s per face here, the 540 under one
  light (section 4).
- **Photos of people**: the local test photos: 14 faces of 6 people after removing identical files (one photo of
  one person is there at three resolutions and counts three times), 15 mated pairs (3 of them that photo against
  itself at another resolution) and 76 non-mated pairs. Far too few for rates: they show whether a model works at
  all. The colour-order check of `bifold-faces check` used the 8 photos of 4 people in the photos' `train` folder.

## 1. The model files and graphs

`bifold-faces check --all` on every model fetched here: all 17 checks pass.

| Model | File (SHA-256, MB) | Graph | Notes |
|---|---|---|---|
| SCRFD-10G (`det_10g.onnx`) | `5838f7fe...`, 16.9 | opset 11, 158 standard operators, `input.1` 1x3xHxW, 9 outputs | Taken out of `buffalo_l.zip` (`80ffe37d...`); byte-identical to `antelopev2`'s `scrfd_10g_bnkps.onnx` and to the copies UniFace and the AdaFace export host |
| YuNet 2023mar | `8f2383e4...`, 0.2 | opset 11, fixed 640 x 640 (OpenCV reshapes it) | As before |
| AdaFace IR-101 WebFace12M | `f2eb07d0...`, 260.7 | opset 16, `input` Bx3x112x112, `output` Bx512, 311 standard operators | Community export (no export script published); the same SHA-256 is pinned by its author's UniFace toolkit. **The graph divides the embedding by its length** (`ReduceL2`, `Div`): AdaFace's quality signal is read from `/model/ReduceL2_output_0` as an extra output |
| `buffalo_l` ResNet-50 (`w600k_r50.onnx`) | `4c06341c...`, 174.4 | opset 11, `input.1` Nx3x112x112, raw 512 output | As UniFace pins it |
| `antelopev2` ResNet-100 (`glintr100.onnx`) | `4ab1d643...`, 260.7 | opset 11, raw 512 output | Optional (no profile) |
| SFace 2021dec | `0ba9fbfa...`, 38.7 | opset 11, normalises its input itself | As before |
| eDifFIQA-T | `7a83be63...`, 6.9 | opset 20 (`Gelu`), `input` Bx3x112x112, `quality` Bx1 | Community export; pinned by UniFace too |
| AuraFace-v1 | `a7933ea5...` (from Hugging Face's listing, via a search excerpt), size unknown | not seen | Hugging Face only: fetched and checked on the Mac |

Every graph has only standard ONNX operators; every embedder gives the same answer twice and the same answer for a
crop alone as in a batch of 9 (to 1e-4).

## 2. Detection, landmarks and pose

- **Both detectors found the face in all 1,620 renders** (yaw up to 60 degrees) and in all 16 test photos; SCRFD
  and YuNet put the landmarks within a fifth of the eye distance of each other on every photo.
- **Landmarks against the avatars' face bones** (frontal renders; median and 95th percentile of the error, in eye
  distances): SCRFD eyes 0.07 / 0.10, nose 0.03 / 0.10, mouth corners 0.08 / 0.14; YuNet eyes 0.08 / 0.17, nose 0.12 /
  0.22, mouth 0.16 / 0.24. (Turned, the eye bones, the eyeballs' centres, lie behind the pupils the detectors mark.)
- **Pose from the five landmarks** (a 3D mean face fitted by scaled orthography; its two depths fitted to the renders):
  with SCRFD's landmarks the yaw comes back within 2 degrees (median) with a spread of 1.6-3.6 degrees at every yaw
  from -60 to 60, and the pose bin (frontal under 20 degrees, left, right, profile from 55) is right in 98.6 % of the
  renders. The pitch reads about 5 degrees further down than the avatars look (a constant offset). YuNet's landmarks
  follow the pose less well: 84 % of the bins right, the pitch hardly at all.
- **Shrinking big faces before the alignment warp** (area averaging when the crop has fewer than 0.8 pixels per
  image pixel) makes no measurable difference on the renders (crop scale 0.48; d' 5.31 with it against 5.38 without
  for the ResNet-50, 4.62 against 4.64 for SFace) and a mixed one on the 8 training photos (scales 0.18-0.69; the
  ResNet-50 d' 15.0 against 12.6, SFace 8.8 against 9.0): within noise. It stays on, as the textbook way to shrink
  an image four or five times (enrolment photos), but it is not shown to help.
- **Speed** (SCRFD-10G, after warm-up): 12 ms on a 160 px canvas (a head cut out of a frame), 35 ms at 320, 120 ms at
  640, 300-390 ms at a whole 1280 x 736 frame; YuNet 44 ms for a whole 1280 x 720 frame, 3 ms at 160. Sessions
  with every input dimension fixed are no faster on the CPU; they are there for CoreML.
- **Tight close-ups** (found in the review): with the face box filling 83 % of the photo's height, SCRFD on a 640
  canvas found 0 of the 15 test faces (77 %: 5, 71 %: 14), on a 320 canvas all 15. The engine now tries 640 and then
  320 when the image's own canvas finds nothing: 15 of 15 at every fill from 65 to 83 % (YuNet: 2-7 of 15).
  Canvases that are not multiples of 32 are rounded up (they made SCRFD's outputs fail to tile the canvas).

## 3. Verification on LFW

**This machine is slow for ResNets**: on its four shared cores a crop takes about 100 ms with `buffalo_l`'s
ResNet-50 and 360-600 ms with a ResNet-100 (InsightFace's `glintr100` 360 ms, the AdaFace export 490 ms at batch 8;
section 6), so the full protocol with mirrored crops (24,000 crops per model) would take hours per ResNet-100. Here
the recognisers ran on a balanced subset: the first 75 mated and 75 non-mated pairs of each of LFW's ten folds (1,500
pairs, 3,000 crops), without mirrored crops; SFace also ran the full protocol, to show what the subset costs. The
full protocol, and CFP-FP and AgeDB-30, run on the Mac: `bifold-faces check --pack lfw.bin cfp_fp.bin
agedb_30.bin`.

10-fold protocol (the threshold chosen on nine folds, used on the tenth), cosine similarity. The thresholds at a
false-match rate (FMR) of 1e-3 are the mean of the non-mated scores plus 3.09 standard deviations (a Gaussian
tail); on SFace's full 3,000 non-mated pairs that gives 0.370 against 0.347 counted, so it errs on the strict side.

| Model | Accuracy (subset) | Published (with flip) | d' | EER | Mated / non-mated mean | Non-mated sd | FMR 1e-3 threshold |
|---|---|---|---|---|---|---|---|
| SFace | 99.47 % +- 0.65 (all pairs: 99.60 %; with flip 99.65 %) | 99.60 (OpenCV Zoo) | 5.7 | 0.67 % | 0.67 / 0.10 | 0.088 | 0.374 (all pairs: 0.370; counted 0.347) |
| `buffalo_l` ResNet-50 | 99.87 % +- 0.27 | 99.83 | 7.8 | 0.27 % | 0.66 / 0.01 | 0.057 | 0.184 |
| AdaFace IR-101 (WebFace12M) | 99.87 % +- 0.27 | 99.82 | 8.2 | 0.27 % | 0.68 / 0.00 | 0.055 | 0.175 |

**The AdaFace export reproduces the published accuracy** within what 1,500 pairs can tell (2 errors in 1,500, the
same as the ResNet-50; the published 99.82 % is about 1 in 550), and it separates mated from non-mated pairs best
(d' 8.2). It stays the best profile's recogniser; the ResNet-50 stays its fallback. The registry's provisional
thresholds come from this table (rounded up: AdaFace 0.18, the ResNet-50 0.19; SFace keeps OpenCV's 0.363). The
ResNet-100 of `antelopev2` was not measured: at 0.4-1.5 s per crop on this machine (more while other work ran
beside) it was left for the Mac, like AuraFace.

## 4. Avatar renders

The 540 renders under one of the three lights (20 avatars, yaw -60 to 60, pitch -15 to 15; 7,020 mated and 138,780
non-mated pairs), SCRFD's landmarks unless said otherwise. "Frontal gallery": one render per avatar (yaw 0, pitch 0)
as the known person, every other render named by its best match.

| Model | d' | EER | Mated mean (5 % quantile) | Non-mated mean (99.9 % quantile) | Frontal gallery rank-1 |
|---|---|---|---|---|---|
| SFace | 4.62 | 1.1 % | 0.75 (0.55) | 0.25 (0.57) | 100 % |
| SFace, YuNet's landmarks (the commercial profile) | 4.51 | 1.2 % | 0.73 (0.53) | 0.24 (0.56) | 100 % |
| `buffalo_l` ResNet-50 | 5.31 | 0.1 % | 0.71 (0.50) | 0.13 (0.41) | 100 % |
| AdaFace IR-101 | 5.42 | 0.2 % | 0.71 (0.51) | 0.13 (0.42) | 100 % |

**Avatars look alike to every model**: the non-mated scores between two avatars are far higher than between two
strangers on LFW (SFace 0.25 against 0.10, the ResNet-50 0.13 against 0.01), as the review expected of a few dozen
rendered heads with a common style. Thresholds must not be set on renders.

**Turning costs score, evenly** (the median score of a render against its avatar's frontal render): SFace 0.93
frontal, 0.90 at 15 degrees, 0.82 at 30, 0.70 at 45, 0.59 at 60; the ResNet-50 0.91, 0.88, 0.79, 0.67, 0.55; AdaFace
0.91, 0.89, 0.79, 0.67, 0.55. Yet with 20 avatars the frontal gallery names every turned render.

**Smaller faces** (the pitch-0 renders shrunk to 12, 16, 24 and 32 pixels between the eyes, found again by SCRFD,
against the full-size frontal gallery): SCRFD finds every face even at 12 pixels; d' falls from 4.97 (32 px) to 4.85,
4.51 and 4.30 (12 px) for the ResNet-50, from 4.96 to 4.90, 4.67 and 4.48 for AdaFace (the smallest loss, as its
low-quality results lead one to expect) and from 4.31 to 3.73 for SFace; rank-1 stays 100 %. Shrunk renders are
sharp, noiseless and uncompressed, so this is the easy end of small faces.

**The raw norm**: the ResNet-50's does not follow pose or size (22.0-22.8 everywhere): not a quality signal. SFace's
falls with the yaw (13.5 frontal, 10.8 at 60 degrees) and a little with size (12.4 at 32 px, 11.7 at 12 px).
**AdaFace's norm, which its loss is meant to tie to quality, does not follow pose or size either** (22.8-23.3 at
every yaw; 22.9 at 32 px, 23.4 at 12 px: slightly higher for the smallest faces). It is the tensor the official
code returns (the length of the final batch-normalised feature, read out of the export's graph). Perhaps sharp,
clean renders are not the low-quality images its training saw; whether it ranks real poor faces (TinyFace, webcam
recordings) is open. It is recorded in each face's quality record but used as no weight.

## 5. Photos of people

14 faces of 6 people (15 mated pairs, 3 of them one photo at two resolutions, and 76 non-mated pairs): whether a
model works at all, not rates.

| Model | d' | Lowest mated score | Highest non-mated score |
|---|---|---|---|
| SFace | 6.8 | 0.65 | 0.26 |
| `buffalo_l` ResNet-50 | 8.4 | 0.60 | 0.13 |
| AdaFace IR-101 | 8.7 | 0.59 | 0.12 |

## 6. The colour order

The registered colour order separates people better than the other one for every ONNX recogniser, on the photos and
on the renders, but by less than the review expected ("far better"): the wrong order still works, only worse.

| Model (registered order) | Renders d', registered / other | Photos d', registered / other | 8 training photos (`bifold-faces check`) |
|---|---|---|---|
| AdaFace IR-101 (BGR) | 5.42 / 5.12 | 8.65 / 7.14 | 11.26 / 9.95 |
| `buffalo_l` ResNet-50 (RGB) | 5.31 / 5.11 | 8.40 / 7.05 | 14.98 / 12.40 |
| `antelopev2` ResNet-100 (RGB) | - | - | 14.40 / 13.70 |

**eDifFIQA-T** (the quality model) was checked the same way. On the 180 pitch-0 renders under one light, full size
and shrunk to 12-32 pixels between the eyes (900 faces), its score ranks the faces by how well the ResNet-50 matches
them to their avatar's frontal render with a Spearman correlation of 0.874 in its registered colour order (RGB)
against 0.839 in the other: RGB is right, by a modest margin. Its median falls steadily with the face's size: 0.64 at
32 pixels between the eyes, 0.62 at 24, 0.56 at 16, 0.50 at 12. The ResNet-50's own norm does not rank them at all
(-0.11).

## 7. Speed per crop

Here, after warm-up, ONNX Runtime 1.31.1 on the CPU with four threads, alone on the machine:

| Model | Batch 1 | Batch 8, per crop |
|---|---|---|
| `buffalo_l` ResNet-50 | 104 ms | 101 ms |
| AdaFace IR-101 | 606 ms | 494 ms |
| `antelopev2` ResNet-100 | 459 ms | 360 ms |
| eDifFIQA-T | 9 ms | 6 ms |
| SFace (OpenCV, one crop at a time; measured beside other work) | 28 ms | 30 ms |

On a busy machine one run of several crops gains more: five crops at a load of about 4 took 887 ms with the
ResNet-50 in one run against 1,308 ms in five (AdaFace 3.9 s against 5.0 s; the review's measurement). So on the CPU
the crops of a call go in one run; with CoreML, which wants fixed shapes, in batches of 8 and then 1. Five heads
analysed in one call with the licensed profile (SCRFD on each head region, then alignment, embedding and eDifFIQA):
2.6 s before, 1.6 s after, at a load of 5-8.

## 8. Open

- **The full benchmark protocol** (all 6,000 LFW pairs with mirrored crops, CFP-FP and AgeDB-30) for AdaFace, the
  ResNet-50 and the ResNet-100s, and **AuraFace** altogether: on the Mac (`bifold-faces fetch --all`, then
  `bifold-faces check --all --speed --pack lfw.bin cfp_fp.bin agedb_30.bin`), with CoreML's speed and how much of each
  graph CoreML takes.
- **AdaFace's norm** as a quality signal: on real low-quality faces (TinyFace, webcam recordings at a distance).
- **Thresholds** are provisional cuts at one false match in a thousand LFW stranger pairs; the open-set decisions
  calibrated per model and face size against a target FPIR are the next step's (F2), on real data.
- **Renders are easy small faces**: shrunk renders are sharp and noiseless; camera faults (blur, noise, compression)
  come with the virtual sensor (S1) and the evaluation (E1).
- **YuNet's landmarks** follow the pose less well than SCRFD's (pose bins right in 84 % of the renders against
  98.6 %), which matters for the commercial profile's pose-bin templates.
