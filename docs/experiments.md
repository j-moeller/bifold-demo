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

## Experiment 2: StyleGANEX latent editing: the built-in age edit is weak, a fitted gender vector breaks the image

FaceFusion's StyleGANEX export computes `codes − direction × AGE`, a stored (18, 512) age vector added to the
encoder's codes. The graph was rewired (`nets.StyleGANEX`): the whole offset became an input and the codes an
output. After that change it still matches ONNX Runtime exactly, and it takes 4.6 s per face on a 4-core CPU.

- **Age at FaceFusion's maximum (±2.5 × AGE)**: a slight smoothing or wrinkling only. The judges' ages move by
  0–3 years (Lena 25 → 24/25, a man in t1 26 → 26).
- **Gender**, as the female-minus-male difference of the 61 encoded direction faces with age projected out:
  no change at 1–2 gaps, a smeared face at 8 gaps, and an unrecognisable one at 16 gaps before either judge
  reads the other gender. 61 real faces are far too few to fit a direction in an 18 × 512 space. Noise dominates
  and pushes the codes off the space of faces.

## Experiment 3: InterFaceGAN directions in StyleGAN2: right for generated faces, out of reach for photos

GFPGAN v1 (Wang et al., CVPR 2021; GitHub release) froze a StyleGAN2 trained on FFHQ at 512 px and learned an
encoder into it. Its checkpoint holds the generator *with* its mapping network, so faces can be sampled:

1. **The directions**: 2000 random faces (truncation 0.7), labelled by FairFace: 55 % female, age 31 ± 13. Ridge
   regression of the gender logit and of the age on `w`, each direction made orthogonal to the other one's.
   The average female and male codes are 2.41 units apart; one unit of age is 10.7 years.
2. **On generated faces** they work. Moving 16 new random faces towards the other gender, the *independent*
   judge (InsightFace) reads the other gender in 38 % at 0.5 gaps, 50 % at 1 gap and **75 % at 1.5 gaps**
   (FairFace: 19 / 50 / 81 %). They look right: beards grow, hair gets longer, faces get softer
   (contact sheet `exp3_synthetic.jpg`).
3. **On photos, through GFPGAN's encoder**: no effect at all, even at 3 gaps (P(female) stays at 0.00/0.00 for the
   men, 1.00/1.00 for the women; identity 0.9). The encoder's codes are not ordinary StyleGAN codes. Decoded on
   their own they give a smear. What makes the face is GFPGAN's spatial feature transform (per-pixel scales and
   shifts from the photo), and it overrides any change of the codes.

## Step 4: distillation, the apps' recipe

The generator can change gender and age convincingly, but only for faces it drew itself. So we teach a
feed-forward network to do the same to any face (`distill.py`, `student.py`). This is the recipe of
"StyleGAN2 Distillation for Feed-forward Image Manipulation" (ECCV 2020), and it fits how the Snapchat lens
behaves:

- **4000 pairs**: a random face (truncation 0.5–1) and the same face moved by a random gender amount (±1.5 gaps)
  and age (±40 years); a tenth change only the gender, a tenth only the age. Saved at 256 px.
- **The student**: a U-Net (6.1 M parameters) that takes the face and the two amounts (as input channels and
  through FiLM in every block) and predicts the change. Its last layer starts at zero, so at first it changes
  nothing.
- **The loss**: L1 on pixels, plus a perceptual loss on four layers of FairFace's ResNet-34, plus FairFace's
  gender and age logits. FairFace therefore steers the training, and **InsightFace stays the independent judge**
  in the validation. ImageNet's VGG, the usual perceptual network, cannot be downloaded here.
- **Augmentation**: the same flip, ±4° turn, ±5 % scale, ±3 % shift and colour change for both images of a pair,
  so the student copes with the landmark jitter of a live camera.
