# Changing a face's gender and age: a brief literature review

What the gender and age sliders need: a network that changes how *female or male* and how *old* a face looks, on a
live webcam image, on a MacBook, with weights we can actually download, and with the person still recognisable.

## How the apps do it (Snapchat, FaceApp, TikTok, Instagram)

None of them has published its model. What is known:

- **Snapchat's gender swap lens** (May 2019) runs in real time, entirely on the phone, without a network
  connection. It only switches on when the face points in an allowed direction, and things in front of the face
  get transformed too. That points to a feed-forward image-to-image network on an aligned face crop, not a
  per-frame optimisation. Source: [Eric Jang's experiments with the lens](https://blog.evjang.com/2019/05/fun-with-snapchats-gender-swapping.html)
  ([summary](https://gigazine.net/gsc_news/en/20190514-snapchats-gender-swapping)); Jang tested its behaviour,
  he did not see its code.
- **The recipe that fits these observations** was published by Yandex: [StyleGAN2 Distillation for Feed-forward
  Image Manipulation](https://arxiv.org/abs/2003.03581) (Viazovetskyi, Ivashkin & Kashin, ECCV 2020). It works in
  three steps:
  1. Draw random faces with StyleGAN2 and move each along a gender (or age) direction in its latent space. That
     gives *pairs*: the same synthetic person as a man and as a woman.
  2. Train a pix2pixHD image-to-image network on those pairs.
  3. Run only that network. It does one pass per frame and needs no encoder or optimisation.

  Snap Research's own papers are about making such image-to-image networks small enough for phones, e.g.
  [Teachers Do More Than Teach: Compressing Image-to-Image Models](https://arxiv.org/abs/2103.03467) (Jin et al.,
  CVPR 2021). That the lens is built exactly this way is an informed guess, not something Snap has confirmed.
- **Newer "AI" effects** (e.g. TikTok's AI filters, FaceApp's newer styles) take a few seconds and are generally
  run on servers; they are most likely diffusion models (below).

The weights of all these commercial models are proprietary.

## The research families

| Family | Examples | Speed | Gender | Age | Identity | Notes |
|---|---|---|---|---|---|---|
| Image-to-image GANs | CycleGAN (2017), StarGAN v2 (CVPR 2020), StyleGAN2 distillation (ECCV 2020) | real time | yes (fixed domains) | per task | good | what apps ship; trained per edit |
| GAN inversion + latent editing | InterFaceGAN (CVPR 2020), e4e (SIGGRAPH 2021), ReStyle, HyperStyle (CVPR 2022), StyleRes (2023), StyleFeatureEditor (CVPR 2024), **StyleGANEX** (ICCV 2023) | 0.1–1 s | yes, continuous | yes, continuous | fair to good | sliders for free; aligned crops (StyleGANEX: any crop with background) |
| Dedicated re-ageing | SAM (2021), **FRAN** (Disney, 2022), TimeMachine (2025), Idiap's identity-preserving ageing (IJCB 2025) | 0.1–1 s | – | yes | very good | FRAN is used in film production |
| Diffusion editing | SDEdit (2022), InstructPix2Pix (CVPR 2023), Concept Sliders (ECCV 2024), DiffAE (2022), IP-FaceDiff (2025), DiffFERV (IJCAI 2025); FLUX.1 Kontext, Qwen-Image-Edit, Gemini 2.5 Flash Image (2025) | seconds | yes | yes | varies | best quality and most flexible; not live on a laptop |
| Real-time diffusion | Adversarial Diffusion Distillation / SD-Turbo (2023), LCM (2023), StreamDiffusion (2023) | 5–20 fps on big GPUs | via prompt | via prompt | weak | flickers and drifts from the person without extra conditioning |
| Face swapping | SimSwap (2020), inswapper (InsightFace), **HyperSwap** (FaceFusion, 2025) | real time | **no** | **no** | – | swaps *who* someone is, keeps hair, jaw shape and stubble (see experiment 1) |

Short notes on the ones that matter here:

- **InterFaceGAN** ([Shen et al. 2020](https://arxiv.org/abs/2005.09635)) found that attributes like gender and
  age are close to linear directions in StyleGAN's latent space. A direction is the normal of a linear classifier
  on labelled latent codes. Moving a code along it changes the attribute, and projecting one direction off another
  keeps the second attribute fixed ("conditional" editing).
- **StyleGANEX** ([Yang et al. 2023](https://arxiv.org/abs/2303.06146)) widens StyleGAN2's first layer so it can
  encode and redraw a face *with its surroundings* (hair, neck, background), not only a tight FFHQ crop. Its
  authors note that the editing directions found for StyleGAN2 work on it unchanged. That matters for gender: the
  hair is part of how gender is read, and a tight crop cannot change it.
- **FRAN** ([Zoss et al. 2022](https://studios.disneyresearch.com/2022/11/30/production-ready-face-re-aging-for-visual-effects/))
  is a U-Net. It takes the face and two age maps (the age now, the age wanted) and predicts the *change* to add.
  It was trained on synthetic before/after pairs, the same distillation idea as above, and it keeps identity and
  skin texture very well.
- **Face swappers** take an identity vector (ArcFace) and redraw the target face with that identity, keeping
  everything else. Experiment 1 in [experiments.md](experiments.md) shows that moving the identity vector towards
  "female" changes *who* the face looks like, but no classifier reads a different gender. Face swapping is the
  wrong tool for this.

## What we can use

The live demo has hard constraints:

- **Speed.** It has to run live on a MacBook (Apple's MPS GPU).
- **Weights we can download.** This development environment reaches GitHub releases but not Google Drive,
  Dropbox or Hugging Face, and results have to be validated here, on real photos.
- **Licences** must allow a demo.

| Model | Weights | Licence | Used for |
|---|---|---|---|
| StyleGANEX (FaceFusion's export for ageing) | GitHub release `facefusion-assets/models-3.1.0` | S-Lab License 1.0 (non-commercial) | gender and age, by latent editing |
| FRAN (ry-lu's re-implementation) | `facefusion-assets/models-3.6.0` | MIT | age, alternative |
| HyperSwap 1a/1b/1c | `facefusion-assets/models-3.3.0` | ResearchRAIL | experiment 1 only |
| ArcFace w600k_r50 | `facefusion-assets/models-3.0.0` | non-commercial (InsightFace) | identity check |
| FairFace | `facefusion-assets/models-3.0.0` | CC BY 4.0 | judge 1: gender, age |
| InsightFace genderage (`buffalo_l`) | `deepinsight/insightface` release v0.7 | non-commercial research | judge 2: gender, age |

The e4e/StyleGAN2 checkpoints of the first version, StarGAN v2, MiVOLO and every diffusion model are hosted on
Google Drive, Dropbox or Hugging Face, so they can only be used on a machine that reaches those sites, and they
could not be validated here. All of these models are licensed for research and non-commercial use at most: fine
for a fun gimmick, not for a product.

PyTorch: the newest release is 2.14.1 (October 2026). There is no 2.15 yet, so the project requires
`torch>=2.14`. The ONNX releases above are converted to PyTorch modules when they are loaded (`onnx2torch`), and
every conversion was checked against ONNX Runtime (see [experiments.md](experiments.md)).
