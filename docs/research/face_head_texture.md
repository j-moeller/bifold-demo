# Photos to head texture: literature review

Component 1 of the character work: turn casual photos of a person's head, taken
from several angles, into a texture for an existing Rocketbox head mesh, so that
the face blends into the head and the whole body. Written October 2026.

How the claims were checked: GitHub (raw files), PyPI and the MediaPipe model
bucket on Google Cloud Storage could be opened from the development sandbox;
arXiv, CVF, Hugging Face and vendor sites could not. Claims marked *(excerpt)*
rest on search-result excerpts only. Timings marked *(measured)* were run in the
sandbox (4-core CPU); the others are quoted from the cited source or marked as
estimates.

## 1. The problem and our constraints

**What exists.** [`characters/face_photo.py`](../../src/bifold_demo/characters/face_photo.py)
aligns one front photo by five landmarks, paints the front-facing head
triangles in the head texture's UV layout, feathers the edge and moves the
skin tone half way to the avatar's (about 0.4 s). What it cannot do is this
component's job: more than one view, no visible edge, no baked-in photo
lighting, and neck, hands, ears and hair that match the person.

**What the component must do.** Take photos from the front, the sides and
possibly the back (phone photos, unknown camera, mixed lighting) and write a
complete texture for the avatar's *existing* head mesh: about 2k vertices, its
own 2048x2048 UV atlas, face bones for expressions. No new geometry. Visible
skin elsewhere (neck, hands, arms; the body atlas is a separate 2048x2048
texture) must match the face, the head-to-body join at the neck must not show,
the lighting must look like the rest of the avatar, and the hair texture must
take the person's hair colour.

**Constraints.**

- Runs on the user's Apple-silicon Mac (PyTorch MPS, 16-32 GB unified memory)
  with Hugging Face access. The project already uses diffusers with
  FLUX.2-klein-4B, Z-Image-Turbo (both Apache-2.0) and SD-Turbo.
- The sandbox is CPU-only and cannot reach Hugging Face. It can reach GitHub
  releases, PyPI and the MediaPipe models on Google Cloud Storage. Everything
  except the generative step must be testable there.
- Minutes at most per character, ideally seconds.
- Permissive licences preferred; non-commercial ones are marked.

**Why our case differs from the literature's.** Most papers estimate an
unknown face shape in a research topology (FLAME, BFM, HiFi3D++) and output
that topology's UV layout. We know the mesh exactly: each photo's camera can be
solved against the Rocketbox head, and every texel knows its 3D point and
normal. The hard parts are regions no photo shows, lighting, and the colour
match between two atlases.

## 2. State of the art

### 2.1 Classical projective texturing and fusion

For every texel, look up its 3D point (rasterise the mesh in UV space),
project it into each calibrated photo, test visibility against a depth map,
and weight the sample by view angle, distance to the mask edge and image
resolution. Then fuse the views. Either pick one view per region (an MRF) and
remove the colour steps at the seams with a smooth additive correction (seam
levelling: Lempitsky and Ivanov 2007; Waechter et al. 2014), or blend all
views band by band with Laplacian pyramids (Burt and Adelson 1983), so
exposure differences mix over wide areas while pores and freckles do not ghost.
Poisson cloning (Pérez et al. 2003) inserts a region so its gradients are kept
and its border matches the surroundings. Seam Erasure (Liu et al. 2017) makes
both sides of every UV cut agree.

Registration of casual photos is now cheap. MediaPipe Face Landmarker gives
478 landmarks, 52 blendshapes and a 4x4 head pose *(measured: 15.8 ms per
image on CPU)*. Its multiclass segmenter labels background, hair, body skin,
face skin, clothes and others *(measured: about 120 ms)*. MediaPipe's canonical
face mesh has UVs for 468 of the 478 points; the 10 iris points have none.
MVMP places the same 478 landmarks on any textured mesh: it renders the mesh
from five views, runs MediaPipe on each render and ray-casts the points back
to the surface. With those points on the Rocketbox head, `cv2.solvePnP` gives
each photo's camera. MediaPipe is expected to fail beyond about 60 degrees
of yaw (not measured); 3DDFA_V2 and 6DRepNet360 cover larger head rotations.

| Method | Year | Needs | Runs on Mac? | Licence |
|---|---|---|---|---|
| MediaPipe Face Landmarker, multiclass and hair segmenters | 2023 | RGB photo; 3.8, 16.4, 0.8 MB models from GCS | Yes, CPU | Apache-2.0 (model cards) |
| MVMP; cvjena mvlm (Deep-MVLM idea) | 2026 / 2018 | Textured mesh, OpenGL | Yes, CPU | MIT |
| trimesh NRICP (Amberg 2007, Sumner 2004) | 2004-07 | Two meshes plus landmark pairs | Yes, CPU | MIT |
| 3DDFA_V2 | 2020 | Face box; ONNX | Yes, CPU (tested on macOS) | MIT code; BFM assets non-commercial |
| 6DRepNet360 | 2024 | Head crop | Yes | MIT; weights on cloud.ovgu.de only |
| Multi-band blending, Poisson cloning (OpenCV) | 1983 / 2003 | Images, masks, weights | Yes, CPU | Apache-2.0 |
| Seam levelling (mvs-texturing) | 2007 / 2014 | Mesh, calibrated photos | Reimplement in numpy | BSD-3 (C++) |
| Seam Erasure | 2017 | OBJ plus texture | Yes, with patches (see 2.4) | MIT |

The upstream mvs-texturing README has a heading that mentions patents but no
patent text; a fork's patent note could not be verified.

### 2.2 3DMM/FLAME fitting and UV completion

The most reusable design is FFHQ-UV (CVPR 2023). It normalises each photo
with a generative editor (flat lighting, no glasses, hair moved away, frontal,
neutral expression), unwraps several generated views, blends them with
visibility masks, then corrects and completes the result from a template
texture with pyramid blending. Its code is MIT, but its stack (TF 1.15, CUDA
10) is obsolete, and its data derives from FFHQ and StyleGAN2, so treat the
textures as non-commercial.

Learned UV completion has moved to diffusion. UV-IDM (CVPR 2024) generates BFM
UV textures conditioned on an identity embedding; inference code and weights
are out, without a licence file. FreeUV (CVPR 2025) completes a partly
projected FLAME UV map with two Stable Diffusion 1.5 networks; it is released
under CC BY-NC-SA 4.0 and needs the FLAME2020 model and CUDA-only
preprocessing. The Imperial reflectance line (AvatarMe++, FitMe, Relightify,
FitDiff) gives the best relightable results but releases no models. Pixel3DMM,
DECA, EMOCA, SMIRK, TRUST and AlbedoMM fit FLAME shape or albedo; they are
non-commercial or bound to the FLAME2020 licence, and none is needed when the
mesh is fixed. FLAME itself may be
less of a blocker than it was: the FLAME site reportedly released "FLAME 2023
Open" under CC-BY-4.0 in November 2025, while FLAME2020 and the FLAME albedo
files stay restricted *(excerpt)*.

The one strong new entry is TOPOS-Texture (May 2026): a LoRA on
Qwen-Image-Edit-2511 that turns one portrait into a full head UV map on the
MetaHuman topology. Its LICENSE grants MIT for the code, the checkpoints and
the bundled MetaHuman topology assets; whether Epic's terms allow the last
point is not clear. It was tested on CUDA, takes one portrait (not several
views), and its roughly 20B base model needs heavy quantisation on a Mac. Using
it would require a one-time MetaHuman-to-Rocketbox surface correspondence.
Three 2026 papers (OMGTex, PromptAvatar, a cascaded-LoRA avatar paper) report
de-lit UV textures in seconds but have only promised code; two of them and
UV-IDM come from one Beihang group, so they are not independent confirmation
of a trend.

| Method | Year | Needs | Runs on Mac? | Licence |
|---|---|---|---|---|
| FFHQ-UV | 2023 | TF 1.15, PyTorch 1.7, CUDA 10, nvdiffrast | No | Code MIT; data non-commercial in effect |
| UV-GAN; OSTeC | 2018; 2021 | BFM fit; StyleGAN2, CUDA 10 | No | No code; CC BY-NC-SA 4.0 |
| HRN / MV-HRN; Deep3DFaceRecon | 2023; 2019-21 | NVIDIA, nvdiffrast, BFM09 | No | Apache-2.0 / MIT code; BFM non-commercial |
| DECA, EMOCA, SMIRK, AlbedoMM, TRUST, Pixel3DMM | 2020-25 | FLAME fit, PyTorch3D, CUDA | No | Non-commercial or FLAME2020-bound |
| AvatarMe++, FitMe, Relightify, FitDiff | 2022-25 | Not released | No | Not available |
| UV-IDM | 2024 | BFM fit, nvdiffrast | No | No licence file |
| FreeUV | 2025 | FLAME2020 fit, nvdiffrast, SD1.5 | Diffusion step portable, preprocessing CUDA | CC BY-NC-SA 4.0 plus FLAME terms |
| TOPOS-Texture | 2026 | One portrait, Qwen-Image-Edit-2511 (~20B), MetaHuman topology | Not as shipped | MIT (code, weights, assets) |
| OMGTex, PromptAvatar, cascaded-LoRA | 2026 | Unknown | Unknown | Code promised |

### 2.3 Diffusion texturing and identity conditioning

Generative texturing of a given mesh renders geometry conditions (depth,
normals) of the mesh from several views, generates images from text or a
reference image, and bakes them into UV. TEXTure and Text2Tex paint view by
view with a keep/refine/generate trimap; SyncMVD and TexFusion share
information between views while denoising; Paint3D and TEXGen inpaint in UV
space; MV-Adapter and Hunyuan3D-2.1 Paint fine-tune multi-view generators;
SeqTex (2025) and UltraTex (2026, 2K views on FLUX) are the current high end.
UltraTex's code also accepts a FLUX.2-klein-4B backbone, but only LoRAs for
non-commercial backbones are released. None of these targets a person's
identity, and nearly all need CUDA. Two details matter for
us. Hunyuan3D-2.1 Paint remeshes by default and always re-unwraps with xatlas,
so it does not write into an existing atlas, and its licence excludes the EU,
UK and South Korea. MV-Adapter's pipeline keeps the mesh's UVs by default
(`uv_unwarp=False`), but it needs CV-CUDA and nvdiffrast.

The useful development for us is open multi-reference image editors.
FLUX.2-klein-4B is the 4-step distilled model; diffusers' `Flux2KleinPipeline`
takes a list of reference images but no mask, defaults to 50 steps (so the
caller must pass 4) and shrinks references above 1024x1024. On an M4 Pro, mflux
measures about 27 s per 1024x1024 image with 14.9 GB of bf16 weights (8.0 GB
at 8 bits, 4.3 GB at 4 bits); BFL's "about 8 GB" figure is for NVIDIA. Each
1024x1024 reference adds 4096 image tokens (16 pixels per token), so a call
with three references processes four times the tokens of a plain generation.
FLUX 3 Image Klein (October 2026, 5.6B, up to 10 references) makes a
1024x1024 image in about 7 s on the same Mac once loaded, but its licence is non-commercial except
for companies under US$5M revenue.

Identity adapters (IP-Adapter-FaceID, InstantID, PhotoMaker V2, PuLID,
Arc2Face, InfiniteYou) generate new pictures of a person rather than
reproducing their skin and lighting. Most load InsightFace models, which are
non-commercial, and PuLID-FLUX is built on FLUX.1-dev (non-commercial).
Novel-view head generators (FaceLift, DiffPortrait360, PanoHead) produce side
and back views from one portrait, but all are CUDA-bound.

| Method | Year | Needs | Runs on Mac? | Licence |
|---|---|---|---|---|
| TEXTure; Text2Tex | 2023 | Depth-to-image SD, CUDA renderer | No | MIT; CC BY-NC-SA 3.0 |
| SyncMVD | 2024 | OBJ with UVs, SD1.5 plus ControlNets, PyTorch3D | Port needed | MIT |
| Paint3D | 2024 | Kaolin, CUDA 11.6 | No | Apache-2.0 code |
| TEXGen | 2024 | ~24 GB GPU | No | No licence file |
| MV-Adapter (texture) | 2024-25 | SDXL or SD2.1, CV-CUDA, nvdiffrast | No | Apache-2.0 |
| Hunyuan3D-2.1 Paint | 2025 | 21 GB; replaces UVs | Community MPS port: 8.5 min at 256 px | Tencent licence, excludes EU/UK/KR |
| SeqTex | 2025 | A100 80 GB | No | Not stated |
| UltraTex | 2026 | LoRAs on FLUX.2-klein-base-9B or FLUX.1-dev | No | Code MIT; backbones non-commercial |
| FLUX.2-klein-4B | 2026 | Prompt plus reference images | Yes (diffusers MPS, mflux) | Apache-2.0 |
| FLUX 3 Image Klein | 2026 | Up to 10 references | Yes (mflux), 18.5 GB peak at 4 bits | FLUX Kommunity (restricted) |
| Qwen-Image-Edit-2511 | 2025 | 20B | Only quantised | Apache-2.0 |
| ID adapters (PuLID, InstantID, PhotoMaker, ...) | 2023-25 | InsightFace embeddings | Mostly CUDA | Code permissive; models non-commercial |
| FaceLift; DiffPortrait360; PanoHead | 2023-25 | One portrait, CUDA | No | Adobe Research weights; not stated; CC-BY-4.0 |

### 2.4 Blending into the body: skin, seams, lighting, hair

No open model writes casual multi-view photos into an arbitrary game-head
atlas. Body-UV generators (SMPLitex, TexDreamer, UVMap-ID) generate skin,
neck and hands together, which shows the goal, but only in the SMPL layout.
Commercial head-from-photo tools reportedly do the same thing as a classical
pipeline: measure one skin colour from the selfie and recolour the body and
the unseen template skin to it, with a dedicated neck blend mask
*(excerpt: Reallusion Headshot, Avatar SDK)*. Their reported failures (a pale
patch under the chin, a neck that does not blend, mismatched saturation) are
neck-mask and colour-picking problems.

The working parts:

- **Colour transfer.** Reinhard (Lab mean and standard deviation) or MKL
  (a 3x3 linear map) on skin masks, applied to a low-pass base layer so the
  template's pores and painted ambient occlusion survive. About 20 lines of
  numpy; color-matcher is GPL-3.0, colortrans and POT are MIT.
- **De-lighting.** With known normals, fit 9 spherical-harmonics lighting
  coefficients per channel to the skin texels and divide the shading out
  (the lighting model of DECA and Deep3DFaceRecon). It removes directional and
  coloured light, not cast shadows. Marigold-IID (diffusers
  `MarigoldIntrinsicsPipeline`, MPS supported) is the learned option with an
  acceptable licence. Careaga and Aksoy's Intrinsic, RGB↔X and LBM are
  research-only; OpenDelight (2026) is only promised.
- **Seams.** The neck is where two atlases meet: sample both at co-located
  head and body vertices, solve smooth additive offsets along the ring, and
  spread them into a band on both sides (seam levelling). Seam Erasure fixes
  UV cuts inside the head atlas but, tested in the sandbox, it does not install
  from PyPI without workarounds, crashes on SciPy 1.14 or later, and had not
  finished one colour channel at 2048x2048 after 9 minutes.
- **Skin-tone fairness.** MediaPipe's own model card shows segmentation
  quality dropping for the darkest skin tones (Monk 9-10: mean IoU 68 vs 77
  overall for the 256 model), and TRUST showed albedo estimators lightening dark skin. Skin
  statistics need white balance and several photos, not one face crop.
- **Hair.** Segment the hair, take a robust Lab colour, recolour the existing
  hair texels by chroma and mean lightness, keep their luminance detail.
  HairFastGAN works on photos, not textures.

| Method | Year | Needs | Runs on Mac? | Licence |
|---|---|---|---|---|
| Reinhard, MKL, optimal-transport colour transfer | 2001-14 | Masked pixel sets | Yes, CPU | Own code; colortrans, POT MIT; color-matcher GPL-3.0 |
| SH-quotient de-lighting | 2001 | Normals, skin mask | Yes, CPU | Textbook |
| Marigold-IID | 2025 | Photo; SD-class model | Yes, MPS | Apache-2.0 code, RAIL++-M weights |
| IC-Light | 2024 | Photo; SD1.5 | Port likely | Apache-2.0; default background remover non-commercial |
| Intrinsic; RGB↔X; LBM | 2023-25 | Photo | Partly | Academic or non-commercial only |
| PCT-Net harmonisation | 2023 | Composite plus mask | Yes, CPU | MPL-2.0 |
| BiSeNet face parsing (neck, ears classes) | 2019 | Photo | Yes | MIT code; weights trained on non-commercial CelebAMask-HQ |
| SMPLitex; TexDreamer; UVMap-ID | 2023-24 | SMPL UV layout, CUDA | No | None stated; MIT; none stated |
| HairFastGAN | 2024 | Photos, CUDA | No | MIT |

## 3. What works best for us

A hybrid: an exact-geometry classical projector as the core (deterministic,
seconds, testable on CPU), with FLUX.2-klein-4B as an optional, masked,
identity-checked generative layer on the Mac that normalises photos and fills
what no photo shows. This is FFHQ-UV's structure with the Rocketbox mesh in
place of the 3DMM and a modern editor in place of StyleGAN.

| Stage | Model | Fallback | Runtime |
|---|---|---|---|
| 0. Once per head model: 478 landmarks on the mesh; UV position, normal and triangle maps; masks; neck seam pairs | MediaPipe on moderngl renders, ray-cast (the MVMP idea, reimplemented) | ~20 hand-placed landmarks, densified with trimesh `nricp_amberg` from the canonical face mesh | Seconds, offline |
| 1. Register each photo | Face Landmarker, multiclass and hair segmenters, `solvePnP` (RANSAC, focal-length search), thin-plate warp of photo landmarks onto the avatar's | Side and back photos: the view named by the capture UI, refined by silhouette fit; 3DDFA_V2 for large yaw | ~0.15 s per photo |
| 2. Normalise photos (optional) | FLUX.2-klein-4B edit: soft even light, neutral expression, no glasses, hair off the face | Original photo | ~30 s per photo (estimate) |
| 3. Project and fuse | UV rasterisation; weights from view angle, mask edge, segmentation class; per-view gain compensation; 5-band multi-band blend; push-pull padding; confidence map | None needed | ~1-2 s (estimate) |
| 4. De-light and set the tone | SH quotient on known normals, then the template's own low-frequency shading reapplied; skin target = trimmed Lab median of cheeks and forehead over all photos | Marigold-IID for harsh light | <1 s; Marigold seconds |
| 5. Complete unseen regions | FLUX.2-klein-4B: a 2x2 grid of renders (back, ears, under the chin) plus 1-2 photos at 512x512 as references; composite with a keep/refine/generate trimap; 2 seeds | Mirror the other half across the face's symmetry plane, tone-shifted template, big-lama or MI-GAN (GitHub releases) for holes | 1-2 min (estimate); fallback seconds |
| 6. Body and seams | MKL on the body-atlas skin mask (from neck and hand bone weights); hair and eye recolour; neck seam levelling and UV-seam pass with our own sparse least-squares solve | Feathered blend without levelling | A few seconds |
| 7. Check and save | SFace on front and three-quarter renders against the gallery; ΔE across the neck and face-to-hand | Reject the generated layer, keep stages 1-4 and 6 | <1 s |

Notes on the choices:

- **The photo is warped, not the mesh.** The thin-plate warp in stage 1 puts
  the person's eyes, mouth and brows on the avatar's, so they move with its
  face bones (HeadBox builds FACS, viseme and ARKit blendshapes from these
  bones on Rocketbox heads).
- **Generated pixels are second-class.** Real photo texels with high
  confidence are never overwritten; generated ones get weight 0.3-0.5 and
  only fill "generate" texels, as in TEXTure's trimap. Stages 4 and 6 run again
  after stage 5 so generated regions are harmonised too.
- **Identity checks are relative.** SFace similarity between a render of the
  avatar and a photo is low anyway (0.37-0.58 in the current demo's close-ups),
  because the geometry is not the person's. Accept an edit or a seed when its
  similarity does not drop against the deterministic result, and when
  MediaPipe landmarks on the edited render move less than a few pixels.
- **Memory.** On a 16 GB Mac FLUX.2-klein-4B needs 8-bit or 4-bit weights.
  Quantising it under diffusers on MPS is untested; mflux (MIT, MLX)
  quantises at load time and has an edit command with several references.
  The model should be unloaded after generation.
- **Sandbox testing.** Stages 0-4, 6 and 7 run on CPU with MediaPipe from GCS
  and big-lama (205.7 MB) or MI-GAN (27.0 MB) from GitHub releases. Test data
  can be made by rendering textured Rocketbox heads from random angles as fake
  photos and reconstructing them onto another avatar (UV-space PSNR, neck ΔE).
  The generative stage sits behind an interface with a deterministic stub.

Totals: under 10 s without generation; 1-3 min with it on a 24-32 GB Mac
(unmeasured; three full-size references or three seeds would exceed it).

**Why not the alternatives.** End-to-end mesh texturers need CUDA, have no
identity input, and several replace the UVs or carry restrictive licences.
3DMM UV models (FreeUV, UV-IDM, FFHQ-UV) target the wrong topology, need CUDA
preprocessing and are mostly non-commercial. TOPOS-Texture, the closest
permissive model, is single-portrait, CUDA-tested and about 20B parameters:
worth one offline experiment through a MetaHuman-to-Rocketbox correspondence,
not a dependency. ID adapters bring InsightFace's non-commercial models and
produce a likeness, not the person's skin. FaceLift and DiffPortrait360 need
CUDA and have research-only or unlicensed weights. FLUX 3 Image Klein is the
faster upgrade where its licence allows.

**Component 2.** The same render, generate, back-project loop fits the clothes,
restricted to garment UV islands: a tileable swatch from Z-Image-Turbo or
FLUX.2-klein-4B, multiplied by the template's normalised luminance so the
folds and seams stay. Circular padding enforces tiling in convolutional models
(SD-Turbo) but not in transformers (FLUX.2, Z-Image), where roll-and-inpaint
or latent wrapping is needed (untested).

## 4. Risks and open questions

1. FLUX.2-klein-4B on MPS through diffusers: speed, memory with 2-3
   references, stability, and how far it moves pixels when repainting a render.
   Unmeasured; a short benchmark on the user's Mac comes first.
2. Whether MediaPipe finds landmarks on renders of every Rocketbox head. If not,
   hand-placed landmarks per head model.
3. Rocketbox atlas details still to inspect: overlapping or mirrored UVs,
   separate eye, teeth and hair primitives, how much lighting and ambient
   occlusion is baked in, whether head and body vertices coincide at the neck,
   where skin sits in each body atlas.
4. Back-of-head and profile photos have no reliable landmarks; silhouette or
   hairline registration needs a prototype. Phone photos often lack EXIF focal
   length.
5. Hair geometry is fixed: long or very different hairstyles cannot be shown,
   only the colour.
6. Skin tone under mixed lighting and for dark skin (segmentation and albedo
   bias). A user-confirmed skin swatch is the fallback.
7. Licences not verified first-hand: MediaPipe weights beyond the model cards,
   FLAME 2023 Open, the MetaHuman assets in TOPOS, the mvs-texturing patent
   note.
8. Privacy: face photos and textures derived from them are personal data, and
   the recognition gallery processes biometric data (GDPR Article 9). Saved
   characters need consent, local storage and deletion together with the
   gallery entry.

## 5. References

Registration, projection and seams

- MediaPipe Face Mesh / Face Landmarker: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/face_mesh.md
- MediaPipe models: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task, https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite
- MediaPipe multiclass segmentation model card: https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Multiclass%20Segmentation.pdf
- MVMP: https://pypi.org/pypi/mvmp/json; mvlm: https://github.com/cvjena/mvlm
- trimesh registration (NRICP): https://github.com/mikedh/trimesh/blob/main/trimesh/registration.py
- 3DDFA_V2: https://github.com/cleardusk/3DDFA_V2
- 6DRepNet360: https://github.com/thohemp/6DRepNet360
- Waechter et al., Let There Be Color! (mvs-texturing): https://github.com/nmoehrle/mvs-texturing
- Lempitsky and Ivanov, Seamless Mosaicing of Image-Based Texture Maps: https://mlanthology.org/cvpr/2007/lempitsky2007cvpr-seamless
- Seam Erasure: https://github.com/zfergus/seam-erasure
- OpenCV blenders and photo module: https://github.com/opencv/opencv/blob/4.x/modules/stitching/include/opencv2/stitching/detail/blenders.hpp, https://github.com/opencv/opencv/blob/4.x/modules/photo/include/opencv2/photo.hpp

3DMM fitting and face UV models

- FFHQ-UV: https://github.com/csbhr/FFHQ-UV
- UV-GAN: https://arxiv.org/pdf/1712.04695; OSTeC: https://github.com/barisgecer/OSTeC
- HRN: https://github.com/youngLBW/HRN; Deep3DFaceRecon: https://github.com/sicxu/Deep3DFaceRecon_pytorch
- DECA: https://github.com/yfeng95/DECA; AlbedoMM: https://github.com/waps101/AlbedoMM; TRUST: https://github.com/HavenFeng/TRUST
- Pixel3DMM: https://github.com/SimonGiebenhain/pixel3dmm
- FitMe: https://github.com/lattas/fitme; AvatarMe: https://github.com/lattas/AvatarMe
- UV-IDM: https://openaccess.thecvf.com/content/CVPR2024/html/Li_UV-IDM_Identity-Conditioned_Latent_Diffusion_Model_for_Face_UV-Texture_Generation_CVPR_2024_paper.html, code https://github.com/Luh1124/UV-IDM
- FreeUV: https://github.com/YangXingchao/FreeUV
- TOPOS: https://github.com/ymxbj/TOPOS, https://arxiv.org/pdf/2605.14594
- OMGTex: https://arxiv.org/pdf/2605.25778; PromptAvatar: https://arxiv.org/abs/2603.04307; cascaded-LoRA avatar: https://arxiv.org/pdf/2606.28144
- FLAME: https://flame.is.tue.mpg.de/

Diffusion texturing, editors and identity

- TEXTure: https://github.com/TEXTurePaper/TEXTurePaper; Text2Tex: https://github.com/daveredrum/Text2Tex
- SyncMVD: https://github.com/LIU-Yuxin/SyncMVD
- Paint3D: https://github.com/OpenTexture/Paint3D
- TEXGen: https://github.com/cvmi-lab/TEXGen
- MV-Adapter: https://github.com/huanngzh/MV-Adapter
- Hunyuan3D-2.1: https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1; Mac port: https://github.com/VladimirTalyzin/hunyuan3d-2.1-mac
- SeqTex: https://github.com/VAST-AI-Research/SeqTex
- UltraTex: https://github.com/yiboz2001/UltraTex
- FLUX.2: https://github.com/black-forest-labs/flux2; diffusers pipeline: https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux2/pipeline_flux2_klein.py
- mflux (FLUX.2 and FLUX 3 on MLX): https://github.com/filipstrand/mflux
- Qwen-Image: https://github.com/QwenLM/Qwen-Image; Z-Image: https://github.com/Tongyi-MAI/Z-Image
- IP-Adapter: https://github.com/tencent-ailab/IP-Adapter; InstantID: https://github.com/InstantID/InstantID; PhotoMaker: https://github.com/TencentARC/PhotoMaker; PuLID: https://github.com/ToTheBeginning/PuLID; Arc2Face: https://github.com/foivospar/Arc2Face; InfiniteYou: https://github.com/bytedance/InfiniteYou; InsightFace: https://github.com/deepinsight/insightface
- FaceLift: https://github.com/weijielyu/FaceLift; DiffPortrait360: https://github.com/FreedomGu/DiffPortrait360; PanoHead: https://github.com/SizheAn/PanoHead
- LaMa: https://github.com/advimman/lama; MI-GAN: https://github.com/Picsart-AI-Research/MI-GAN; TorchScript files: https://github.com/Sanster/models/releases

Lighting, colour, hair and bodies

- Marigold: https://github.com/prs-eth/Marigold
- IC-Light: https://github.com/lllyasviel/IC-Light
- Intrinsic: https://github.com/compphoto/Intrinsic; RGB↔X: https://github.com/zheng95z/rgbx
- PCT-Net: https://github.com/rakutentech/PCT-Net-Image-Harmonization; LBM: https://github.com/gojasper/LBM
- colortrans: https://github.com/dstein64/colortrans; color-matcher: https://github.com/hahnec/color-matcher; POT colour transfer: https://github.com/PythonOT/POT/blob/master/examples/domain-adaptation/plot_otda_color_images.py
- Face parsing (BiSeNet): https://github.com/zllrunning/face-parsing.PyTorch
- HairFastGAN: https://github.com/AIRI-Institute/HairFastGAN
- SMPLitex: https://github.com/dancasas/SMPLitex; TexDreamer: https://github.com/ggxxii/texdreamer; UVMap-ID: https://github.com/twowwj/UVMap-ID
- Reallusion Headshot neck-blend report *(excerpt)*: https://reallusion.com/FeedBackTracker/Issue/NECK-area-on-headshot-not-blending-colors-with-head
- Avatar SDK parameters *(excerpt)*: https://docs.avatarsdk.com/standalone-app/3.1.2/parameters_description.html

Rocketbox

- Microsoft Rocketbox avatar library (MIT): https://github.com/microsoft/Microsoft-Rocketbox
- HeadBox, blendshapes from Rocketbox face bones: https://github.com/openVRlab/Headbox
