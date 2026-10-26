# Experiments: from "the sliders do nothing" to a working gender and age change

A log of what was tried, in order, with the numbers. Photos and model weights are not in the repository: the
photos are of real people, the weights are large. The scripts that produced every table are in
[`validation/`](../validation).

## Setup

**Judges.** A change of gender only counts if face classifiers *other than the one that made it* see it. Two
independent classifiers read every result:

| Judge | Network | Trained on | Reads |
|---|---|---|---|
| FairFace | ResNet-34 ([Kärkkäinen & Joo 2021](https://arxiv.org/abs/1908.04913)) | FairFace (balanced race, gender, age) | P(female), age band |
| InsightFace | MobileNet `genderage` from `buffalo_l` | CelebA | P(female), age |

On top of that, **identity** is the cosine similarity of ArcFace (`w600k_r50`) embeddings of the face before and
after the change: about 1.0 for the same picture, above roughly 0.4 for the same person, 0.0–0.2 for strangers.

**Photos.**
- *Direction set*: the 61 portraits in DeepFace's unit-test folder (31 women, 30 men, labelled by both judges, who
  agree on all of them).
- *Validation set*: none of those people. A group photo with three women and three men in different poses
  (InsightFace's sample `t1.jpg`) and OpenCV's `lena.jpg`.

**Faces** are found by YuNet with five landmarks and aligned to each network's own template (`align.py`).

## Step 0: the published models, converted to PyTorch

All models here are published as ONNX files (GitHub releases), and they run as PyTorch modules converted with
`onnx2torch`. Each conversion was checked against ONNX Runtime on the same input (`validation/check_conversion.py`):

| Model | Largest difference | Note |
|---|---|---|
| ArcFace w600k_r50 | 1e-6 | |
| FairFace | 0 | logits exposed instead of the ArgMax outputs |
| InsightFace genderage | 2e-6 | |
| HyperSwap 1a | 5e-4 (image-like input) | `Reshape allowzero=1` cleared (no zeros in its shapes) |
| FRAN | 0 | opset 18 → 17 (no feature of 18 used); fixed 1024 px input |
| StyleGANEX (age export) | 0 | computed-weight convolutions need a converter of our own (`onnx_torch.DynamicConv`) |

## Experiment 1: face swapping with a "more female" identity: does not work

Idea: HyperSwap redraws a face from an ArcFace identity vector, keeping pose, expression and lighting. So:
1. Take the person's own identity.
2. Move it along the direction from the average male to the average female identity of the direction set.
3. Let HyperSwap redraw the face with it.

The gap between the two averages is 0.53, and the projections of women and men don't overlap (0.31 ± 0.06 vs
−0.22 ± 0.07). Strength *s* moves a face by *s* gaps towards the other gender.

| Face | Was | s = 0 | s = 1 | s = 2 | s = 3 | Identity at s = 3 |
|---|---|---|---|---|---|---|
| t1 #0 | M | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.59 |
| t1 #2 | M | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.61 |
| t1 #4 | M | 0.01 / 0.01 | 0.01 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.87 |
| t1 #1 | F | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 0.99 | 0.98 |
| t1 #3 | F | 0.97 / 0.96 | 0.97 / 0.97 | 1.00 / 1.00 | 1.00 / 1.00 | 0.63 |
| t1 #5 | F | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 0.64 |
| lena | F | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 0.61 |

(P(female) by FairFace / InsightFace.) The faces turn into *other people*: identity similarity drops to about 0.6.
Neither judge sees any change of gender, not even at three times the gap. A face swapper is trained to keep
everything but identity, so the hair, the jaw line and the stubble stay. Those are the cues gender is read from.
Face swapping is the wrong tool, and that matches the literature (see [literature.md](literature.md)).
