# Environment and findings that every step shares

## The machine the steps run on

- A CPU-only Linux container with 4 CPUs, shared by the running steps. Steps keep their runs lean: targeted tests,
  small renders, and the full suite (about 5 minutes) once at the end.
- Renders need `xvfb-run -a`.
- OpenBLAS thread thrashing made small linear solves about 1000 times slower here. Runs set
  `OPENBLAS_NUM_THREADS=1`, which is a property of this machine, not of the code.
- The 20 prepared Rocketbox avatars are shared and read only. Steps that cache per avatar write to their own folders
  through environment variables: `BIFOLD_HEADTEX_CACHE` (head landmarks) and `BIFOLD_OUTFIT_CACHE` (garment labels).
  Two early CLI runs wrote into the shared folder; both files were removed at once.

## What can be downloaded here

- **Works:** GitHub (including release assets), Google Cloud Storage (MediaPipe's `.task` bundles) and PyPI.
- **Does not work:** Hugging Face, ModelScope and download.pytorch.org.
  - So the generative models (FLUX.2-klein for completing the head texture and for fabric swatches) cannot be
    downloaded or run here.
  - They sit behind an interface with an injectable pipeline, are tested with fakes or tiny randomly initialised
    diffusers models built locally, and every generative stage has a deterministic fallback.
  - The generative path runs on the user's own machine (a Mac with Apple silicon).
- **MediaPipe 1.1.1** is installed with `--no-deps` (plus absl-py and flatbuffers), because it pulls in
  `opencv-contrib-python`, which clashes with `opencv-python`. On Linux it also needs `libegl1` and `libgles2`.

## Safety and privacy rules the steps follow

- **No code or pickled or TorchScript models from third parties.** Downloading and running the TorchScript big-LaMa
  inpainting model was refused by this environment's safety check. The design was changed instead of working around
  it: weights are used only as safetensors through diffusers or transformers, or as Google's MediaPipe `.task`
  bundles.
- **Face photos and anything made from them are personal data.**
  - The real test photos stay in a scratch folder.
  - Overlays and textures made from them are never committed.
  - The previews use renders of the avatars and a public sample photo only.
  - The user's own data lives in git-ignored folders: `faces/` and `created/`.

## Findings that matter across components

- **One head UV layout.** Most Rocketbox avatars share one head UV layout, with the nose at about UV (0.5, 0.66). So
  stage-0 landmarks are made once per head model, and a face texture can move between avatars with the same layout.
- **Overlap in the UV bake** (`characters/uvbake.py`):
  - body textures have none;
  - head textures about 1 % (the eyeballs);
  - the hair cards (`*_opacity`) 37-52 %, since they are stacked by design.

  A first count of 9.7 % on the head came from triangles sharing an edge. Counting only texel centres strictly inside
  each triangle fixed it.
- **The shared skin mask.** B1's `body_regions(model).skin` and `.skin_color` are what A6 uses to match the body's
  skin to the face. Two things must be resolved between the steps:
  - B1's `dress()` starts from the library texture, so A6's skin harmonisation and an outfit must be made to compose;
  - `face_photo.py` reads numpy head textures as BGR while the renderer reads them as RGB, which matters only for
    avatars loaded with `textures="array"`.
- **Created characters own their face.** C2 makes FaceCast and selection's face item refuse created characters,
  because their face is part of their saved textures (from component 1). If the creator should allow putting a face
  on them by hand, that rule changes in B4 or C3.
- **Background work and the GIL.** CPU-bound Python on a worker thread slows the main thread's frames two- to
  threefold, even when the measured per-frame work is small. C2's fix adds `pacing.py` for this. A4-A6 (texture
  fusion) and B2-B4 (fills in the creator) should run their heavy work paced, or in a separate process.
- **Mixing MediaPipe and moderngl in one process** crashed once (a segmentation fault) in about 10 runs. Keep an eye on
  it when the creator screen runs both live.
