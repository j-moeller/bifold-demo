# Outfit textures for the character creator: literature review

This review covers component 2, the character-creation screen: the user dresses a Rocketbox avatar from preset outfits, recolours and patterns each garment, and types a description that a generative model turns into a new clothes texture. It reflects the state of the art in October 2026.

Markers: **(unverified)** means the claim comes from search summaries only (arxiv.org, thecvf.com, huggingface.co and the game sites could not be reached from the research sandbox). **(est.)** marks our own estimate. Mac timings come from the mflux README unless stated otherwise.

## 1. Problem and constraints

The avatar keeps its mesh, skeleton and UV layout. Only the body texture changes, and the result is saved with the character so that it reappears in the fair. Four things make this hard:

- **The target is a packed atlas, not a picture.** One 2048x2048 `*_body_color*.tga` holds the jacket, shirt, trousers, shoes, hands and other skin, in islands cut at arbitrary seams that may be mirrored or overlap. Image models are not trained on such images and cannot reliably edit one directly.
- **The shading is painted in.** Folds, stitching, buttons and ambient occlusion are part of the colour map; a new fabric that drops them looks flat.
- **Patterns must be consistent:** one world-space scale, continuous across UV cuts, front and back agreeing.
- **Skin belongs to component 1,** which sets the neck and hand colour from the person's photos. The outfit step must not touch those texels.

Rocketbox facts (checked in the Microsoft repository): every body colour map has a `*_body_normal.tga` and an `*_opacity_color.tga`; 19 colour maps carry a suffix (`_blue`, `_acu`), so glob `*_body_color*.tga`; military avatars have extra equipment and helmet textures; no avatar ships colour variants, so we must author the presets.

Constraints: the user's Apple-silicon Mac (PyTorch MPS or MLX, 16-32 GB) with Hugging Face access; a CPU-only sandbox without Hugging Face (GitHub releases, GCS and PyPI only); the project's existing `ImageGenerator(preset, pipeline=...)` with FLUX.2 [klein] 4B, Z-Image-Turbo and SD-Turbo; Apache-2.0 or MIT preferred; seconds per fabric swatch and under about a minute for a full restyle.

## 2. State of the art

None of the published work textures an existing game character's packed body atlas from text out of the box. The useful work falls into five threads.

### 2.1 Tileable texture generation

| Technique | Works on | Tiling quality | Code, licence | Fit |
|---|---|---|---|---|
| `padding_mode="circular"` on UNet and VAE Conv2d (asymmetric-tiling extension) | Convolutional UNets (SD1.5, SDXL, SD-Turbo); no effect on DiTs | Canvas wraps, but "won't guarantee seamless tiling 100% of the time" | CC0-1.0; SD-Turbo under the Stability Community licence (free below US$1M revenue, registration) | Fast preview, fallback |
| Latent rolling every step (random offset shrinking to 0) plus a circular VAE decoder (Universal Seamless Tiles; DiT_Diffusion_Seamless_Tiler) | Flux, SDXL, Wan named; DiT tiler scripts for FLUX.2-klein-4B and Z-Image | "Not conv-perfect"; few-step models "may need 15-20 steps or come out softer" | MIT; DiT tiler unlicensed, so reimplement (~30 lines) | Main path |
| Tiled Diffusion (Madar and Fried, CVPR 2025) | Self- and many-to-many tiling in latent space | Paper | No code found | Idea only |
| Roll by 50 % and inpaint the cross seam (LaMa, MI-GAN) | Any image | Good on irregular fabrics, weaker on strict repeats (est.) | Apache-2.0 / MIT, CPU | Model-agnostic repair |

In diffusers the FLUX.2 latents are a packed token sequence: unpack to the 2D grid, roll and repack in `callback_on_step_end`. The FLUX.2 VAE decoder is convolutional, so circular padding applies there. A learned tileability metric exists (TexTile, CVPR 2024; no code found); a wrap-edge gradient check (gradient energy across the wrap against the interior) suffices to auto-reject bad seeds.

### 2.2 Fabric and material generation

| Method | What it does | Compute | Licence | Fit |
|---|---|---|---|---|
| FabricDiffusion (SIGGRAPH Asia 2024) | Fabric or print crop from a clothing photo → flat tileable texture | SD-based, no figures | X11 (MIT-like); HF weights (README links `-logo`, usage says `-print`) | Optional "copy the fabric of my shirt" |
| ControlMat (TOG 2024), StableMaterials (Adobe, CVPR 2026) | Tileable PBR maps from a photo (ControlMat, noise rolling) or text/image (StableMaterials) | StableMaterials LCM 4 steps: ~2 s on an RTX 4090 | OpenRAIL weights, MIT wrapper; no ControlMat code | Optional "material mode"; we only render colour |
| MatFuse (CVPR 2024) | SVBRDF from text, image, sketch and a 5-colour palette | ≥12 GB, custom VQ-VAE | MIT code | Low priority |
| DressCode (SIGGRAPH 2024) | Sewing-pattern GPT (new geometry) plus SD-2.1 tile-based PBR textures | Windows, Blender, Maya | No licence (all rights reserved) | Design reference |

A flat, evenly lit, tileable swatch is an easy target for diffusion, which is why DressCode generates swatches and maps them afterwards.

### 2.3 Garment texturing on meshes

| Line | Examples | Status | Fit |
|---|---|---|---|
| Garment DiTs in UV space guided by a UV position map (per-texel XYZ) | TexGarment (CVPR 2025), GarmentPainter (arXiv 2603.08228), OmniFabric (SIGGRAPH Asia 2026) | No code for the first two (details unverified); OmniFabric code "being prepared" | Right idea; we approximate it training-free with a per-avatar position map |
| Text to texture in one fixed body layout | SMPLitex (BMVC 2023, SD1.4 DreamBooth on SMPL UVs, no licence), TexDreamer (ECCV 2024, SD2.1 LoRA, MIT code) | Code public | Only if many Rocketbox avatars share a body layout; then a LoRA on it is possible |
| Text-to-garment with new geometry | WordRobe (ECCV 2024, no code), Garment3DGen and GarmentDreamer (both CC BY-NC 4.0, CUDA) | Slow | No: they change geometry |
| General mesh texturers | TEXTure, SyncMVD (MIT), Paint3D (Apache-2.0, kaolin), MV-Adapter (Apache-2.0, CV-CUDA), TEXGen (no licence, ~24 GB), Hunyuan3D-2.1 Paint (21 GB; excludes EU, UK, South Korea), UltraTex (MIT code with a Klein-4B backbone; only non-commercial LoRAs released) | CUDA-bound, object-trained, minutes per mesh | Borrow algorithms, not code |
| Practitioner tool | StableGen (Blender, GPL-3.0, runs on Apple silicon) | Works | Blueprint for the restyle path: colour-match views (Reinhard, MKL, histogram, MVGD), weight by view angle, bake, UV inpaint |

The recipe we reuse from TEXTure, SyncMVD and StableGen: render views, generate or edit them, back-project into the existing UVs weighted by visibility and view angle, keep confident texels (TEXTure's keep / refine / generate trimap), and fill what no view saw.

### 2.4 Instruction-based editing

| Editor | Size and steps | References and mask | Mac evidence | Licence |
|---|---|---|---|---|
| FLUX.2 [klein] 4B | 4B, 4 steps (distilled) | List of references; no mask | ~27 s per 1024² on an M4 Pro (bf16); weights 14.9 GB bf16, 8.0 GB q8, 4.3 GB q4; each reference up to 1 MP adds ~4k tokens | Apache-2.0 |
| FLUX.2 [klein] 4B Base | 50 steps with CFG (2 passes per step) | Same | ~10-25x the distilled cost at 20-50 steps | Apache-2.0 |
| FLUX 3 Image Klein (15 Oct 2026) | 5.6B, 16 steps | Up to 10 references | ~7 s per 1024² on an M4 Pro; peak 18.5 GB at q4 | FLUX Kommunity 1.0: non-commercial, commercial below US$5M revenue |
| Qwen-Image-Edit-2511 | 20B MMDiT | Multi-image; "material replacement" | ~40 GB bf16 weights; mflux support for 2511 unconfirmed | Apache-2.0 |
| Qwen-Image-2.1 (Sep 2026) | 7B plus a VL encoder | Up to 10 references, masks, RGBA (unverified) | In mflux | Reportedly research-only |
| FLUX.1 Kontext [dev] | 12B | Single image | In mflux (legacy) | Non-commercial |
| InstructPix2Pix | SD1.5 | Single image | Runs on MPS | MIT code; OpenRAIL-M weights |

BFL's "about 8 GB" for klein 4B is a GPU figure; the Qwen3 text encoder comes on top, which is why the project README recommends a 24 GB+ Mac. A UV atlas is out of distribution for every editor here, so edit renders or flat swatches. Klein has no mask input: mask its output afterwards, or, to stop face and skin drift during the edit, blend latents RePaint- or Differential-Diffusion-style in the step callback (start img2img from the render, overwrite the latents outside the garment at every step; untested with klein).

### 2.5 Applying textures to UV regions of a game atlas

This is established, deterministic practice, fully testable on CPU:

- **Region masks:** a label image, or RGB "tint masks" packed into one texture, one zone per garment.
- **Shading-preserving composite:** from the original luminance L, a detail layer D = L / blur(L, σ≈8-16 px), clipped to about 0.5-1.5, and a shading layer S = blur(L) / mean_region(L). The new texel is fabric × D × S, so folds, seams and stitching survive; a flat colour is a 1-pixel fabric.
- **3D-space sampling:** look the fabric up by bind-pose position from a UV position map (triplanar for isotropic fabrics, cylindrical per limb for stripes and grain), so the scale is uniform and the pattern continues across UV cuts.
- **Seam dilation** of each island by 6-8 px against mipmap bleeding, and **prints as decals** projected along the normal at a chest, back or sleeve anchor, multiplied by D.
- **Delighting:** principled options are Careaga and Aksoy's intrinsic decomposition (academic-only, patent pending) and Marigold IID (licence to check), which StableGen uses with StableDelight. D×S is the licence-free default.

## 3. Recommended approach

One shared compositor serves four tiers (Stages 1-4). Only FLUX.2-klein-4B is required, behind the existing `ImageGenerator` and its `pipeline=` injection, so sandbox tests run with fake pipelines.

**Stage 0. Per-avatar bake (CPU, cached, shared with component 1).** Rasterise the body triangles into UV space with moderngl: bind-pose position map, normals, dominant skinning bone and island id. Label the regions (skin, top, shirt, bottoms, shoes, other/equipment) from bone groups, split jacket from shirt by Lab k-means of the original atlas within each group, feather by 2-3 px, and allow a per-avatar JSON override. Compute D and S. Flag mirrored islands (they force symmetric patterns) and overlapping ones (they make back-projection ambiguous). Skin labels go to component 1 and are never written here.

**Stage 1. Presets and dyes (instant, no ML).** Per region: colour chips and numpy procedural patterns (stripes, checks, tartan, houndstooth, twill, camo) in a primary and a secondary dye, sampled in 3D. Outfit presets are JSON recipes (region → fill). A keyword parser maps simple prompts ("navy pinstripe", "red tartan") to procedural parameters, which gives exact, deterministic, instant "text to design" without a model.

**Stage 2. Text to fabric swatch (default generative tier).** Prompt "seamless tileable fabric texture, flat lay, top-down, even diffuse light, no folds, {user text}" into FLUX.2-klein-4B at 512-768 px with our latent-rolling patch. Generate 4 seeds and auto-reject bad seams; if seams persist, run 8-16 distilled steps before trying Base. Equalise the swatch's luminance and composite as in Stage 1. Z-Image-Turbo takes the same patch, and FLUX 3 Image Klein is the faster engine where its licence fits. While klein runs, show SD-Turbo (circular padding) or Stage 1 thumbnails.

**Stage 3. Prints and logos:** generate with klein or upload, then place as a projected decal (FabricDiffusion's print model optional).

**Stage 4. Whole-outfit restyle.** Render the avatar unlit in A-pose, front and back side by side on one ~1 MP canvas, so one generation keeps both views consistent. Edit it with klein ("make the jacket ..., keep face, skin, hair, pose and background unchanged"). Back-project with depth-tested visibility and cos(angle)^p weights, only inside the garment masks. Fill unseen texels (sides, underarms) per island with LaMa or the region mean × D × S. For plain fabrics, multiply the edit's low-pass colour by the original D to restore crisp stitching; for prints keep the edit's detail. Qwen-Image-Edit-2511, quantised on 32 GB+ Macs, is an opt-in quality backend.

**Stage 5. Save.** Store the body atlas PNG (with component 1's head atlas) and a `recipe.json` (region fills, prompts, seeds, scales, model id, avatar id). The fair loads it through the existing per-person `texture_overrides` without regenerating. The t-shirt accessory stays a separate shell over the clothes.

**Fallback ladder:** klein-4B rolled at 4 steps → klein-4B at 8-16 steps → klein-4B Base → SD-Turbo with circular padding → procedural pattern or colour.

**Expected runtimes (M-series Pro):**

| Step | Time |
|---|---|
| Stage 0 bake | 1-3 s once per avatar (est.) |
| Colour or procedural fill at 2048² | under 0.3 s (est.) |
| One 512-768 px swatch | 5-15 s (est., scaled from 27 s at 1024²) |
| 4 swatch variations, shown as each finishes | 20-60 s (est.) |
| Restyle edit (1 MP canvas as the reference) | 45-90 s (est.; the reference roughly doubles the tokens) |
| Back-projection, fill and composite | 1-3 s (est.) |
| 16 GB Mac | use mflux at q8 or q4; quality loss unmeasured |

## 4. Character-creator UX notes

- **Roblox Texture Generator** (checked in Roblox's creator-docs): a prompt, or a reference image as a style guide, gives three previews in about 10 s, then Fast (1K diffuse, under 30 s) or Quality (4K PBR, a couple of minutes); the viewport angle sets the preview orientation.
- **inZOI** (unverified): a per-item "My Texture" menu generates fabric patterns from text locally in seconds and keeps them for reuse.
- **FFXIV Dawntrail** (unverified): two dye channels per gear piece plus favourite and recent dyes; players complain when a channel recolours only the stitching. **The Sims** (unverified): fixed swatches per item.

So users expect seconds per variation, a few variations to pick from, and region-level control. Suggested screen: region tabs (Jacket/Top, Shirt, Bottoms, Shoes), each with colour chips, pattern chips with primary and secondary dye, a "Describe a fabric..." box returning 4 thumbnails, scale and rotation sliders, a logo slot (chest, back) and "Restyle whole outfit"; plus outfit presets, undo, reset, randomise, a before/after toggle, a turntable preview, and "Save character", which puts the character into the fair at once.

## 5. Risks

- **UV layouts:** jacket and shirt in one island, mirrored or overlapping islands, skirts and long coats can defeat the automatic masks. Mitigation: the overlap check and per-avatar overrides for the avatars we use.
- **Baked lighting:** D×S may not suffice on heavily shaded atlases; next steps are the normal map or an intrinsic model (licence permitting).
- **Tiling:** 4-step rolling may give soft or seamy swatches, and klein Base costs 10-25x more.
- **Edit drift:** edits can leak into skin, face or background, and front and back colours can diverge. Mitigations: post-masking, the two-view canvas, latent blending.
- **Mac budget:** memory on 16 GB machines (text encoder plus references) and mostly estimated timings. The restyle may exceed a minute; then shrink the canvas to ~0.6 MP or use FLUX 3 Klein where licensed. Measure diffusers-MPS against mflux before promising seconds.
- **Licences:** avoid FLUX.1 [dev]/Kontext, FLUX.2 klein 9B, Garment3DGen, GarmentDreamer, Hunyuan3D (territorial limits; Section 5(b) bans training other models on outputs) and GPL code (StableGen). Use FLUX 3 Klein, SD-Turbo and Qwen-Image-2.1 only within their terms. LaMa's weights are Apache-2.0 but were trained on the research-only Places2.
- **Kiosk safety and IP:** at a public fair, free text can ask for brand logos or offensive prints; filter prompts, refuse trademarks, check outputs.
- **Evidence:** TexGarment, GarmentPainter, WordRobe and Tiled Diffusion details and the inZOI, FFXIV and Sims notes are second-hand. No model runs in the sandbox, so generative tiers are tested there with fake pipelines and for real only on the Mac.

## 6. References

Models and editors

- FLUX.2 (BFL): https://github.com/black-forest-labs/flux2
- diffusers FLUX.2 pipelines: https://github.com/huggingface/diffusers/blob/main/docs/source/en/api/pipelines/flux2.md
- mflux (MLX; FLUX.2 and FLUX 3 Klein timings): https://github.com/filipstrand/mflux
- FLUX.1 / Kontext: https://github.com/black-forest-labs/flux
- Qwen-Image / Qwen-Image-Edit: https://github.com/QwenLM/Qwen-Image
- Z-Image: https://github.com/Tongyi-MAI/Z-Image
- SD-Turbo: https://huggingface.co/stabilityai/sd-turbo
- InstructPix2Pix: https://github.com/timothybrooks/instruct-pix2pix

Tiling and materials

- ComfyUI-Universal-Seamless-Tiles: https://github.com/OliverCrosby/ComfyUI-Universal-Seamless-Tiles
- DiT_Diffusion_Seamless_Tiler: https://github.com/PlanetSMH/DiT_Diffusion_Seamless_Tiler
- asymmetric-tiling-sd-webui: https://github.com/tjm35/asymmetric-tiling-sd-webui
- Tiled Diffusion (CVPR 2025): https://openaccess.thecvf.com/content/CVPR2025/html/Madar_Tiled_Diffusion_CVPR_2025_paper.html
- ControlMat: https://arxiv.org/abs/2309.01700
- StableMaterials: https://arxiv.org/abs/2406.09293 and https://github.com/P5ina/comfyui-stable-materials
- MatFuse: https://github.com/giuvecchio/matfuse
- FabricDiffusion: https://github.com/humansensinglab/fabric-diffusion and https://arxiv.org/abs/2410.01801
- DressCode: https://github.com/IHe-KaiI/DressCode

Garments and mesh texturing

- TexGarment (CVPR 2025): https://cvpr.thecvf.com/virtual/2025/poster/34341
- GarmentPainter: https://arxiv.org/abs/2603.08228
- OmniFabric: https://github.com/humansensinglab/OmniFabric
- SMPLitex: https://github.com/dancasas/SMPLitex
- TexDreamer: https://github.com/ggxxii/texdreamer
- Garment3DGen: https://github.com/nsarafianos/Garment3DGen
- GarmentDreamer: https://github.com/boqian-li/GarmentDreamer
- TEXTure: https://github.com/TEXTurePaper/TEXTurePaper
- SyncMVD: https://github.com/LIU-Yuxin/SyncMVD
- Paint3D: https://github.com/OpenTexture/Paint3D
- MV-Adapter: https://github.com/huanngzh/MV-Adapter
- TEXGen: https://github.com/CVMI-Lab/TEXGen
- Hunyuan3D-2.1: https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1
- UltraTex: https://github.com/yiboz2001/UltraTex
- StableGen: https://github.com/sakalond/StableGen

Editing, inpainting and delighting

- RePaint: https://github.com/andreas128/RePaint
- Differential Diffusion: https://github.com/exx8/differential-diffusion
- LaMa: https://github.com/advimman/lama
- MI-GAN: https://github.com/Picsart-AI-Research/MI-GAN
- Intrinsic image decomposition (Careaga and Aksoy): https://github.com/compphoto/Intrinsic
- Marigold (incl. IID): https://github.com/prs-eth/Marigold

Assets and products

- Microsoft Rocketbox: https://github.com/microsoft/Microsoft-Rocketbox
- Roblox Texture Generator docs: https://github.com/Roblox/creator-docs/blob/main/content/en-us/studio/texture-generator.md
- inZOI guide (unverified): https://www.playnews.gg/en/guides/inzoi-2026-mastering-create-a-zoi-and-generative-ai
- FFXIV dyes guide (unverified): https://www.icy-veins.com/ffxiv/use-dyes-to-change-the-colour-of-your-gear
