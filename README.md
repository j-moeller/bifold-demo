# bifold-morph

Live webcam face editing with two sliders: **age** (younger ↔ older, in years) and **gender** (female ↔ male).

MiVOLO only *estimates* age and gender, so the generative part is a different network:

- **e4e** (encoder4editing) turns each face into a StyleGAN2 code;
- the code moves along an **age** and a **gender** direction in StyleGAN2's latent space;
- **StyleGAN2 (FFHQ)** draws the edited face, and the change is blended back into the video.

**MiVOLO** finds the directions: it labels a few thousand generated faces, and a linear model of those labels gives
the directions, calibrated in years. It also reads each face before and after the edit, live.

```bash
pip install -e ".[models]"
bifold-morph setup      # once: download the models (~1.3 GB), find the directions with MiVOLO
bifold-morph            # the live demo on webcam 0
```

## Isolated: the models in a Docker container

Model checkpoints (`.pt`, `.pth.tar`) are Python pickles, and loading one can run code. To keep them away from your
machine, `--isolate` splits the demo in two:

- **your side** (`pip install -e .`, no PyTorch): the webcam, the window and the sliders;
- **the container** (`bifold-morph-worker`): PyTorch, the model files and all the computing. It runs with
  `--network none`, a read-only file system, the models folder mounted read-only, no capabilities and as an
  unprivileged user. It cannot reach the network, your files or the camera.

The two talk only through the container's stdin and stdout: each frame goes in, the edited frame and the face
labels come back (see `src/bifold_morph/isolated.py`).

```bash
docker build -t bifold-morph-worker .
M=~/.cache/bifold-morph; mkdir -p $M
# once: downloading needs the network; finding the directions does not
docker run --rm -v $M:/models bifold-morph-worker fetch
docker run --rm --network none -v $M:/models bifold-morph-worker directions --device cpu
pip install -e .                 # the window side only
bifold-morph run --isolate
```

The price: Docker on a Mac cannot use Apple's GPU, so the isolated models run on the CPU (expect about 1 frame per
second or less at `--size 256`). On Linux with an NVIDIA GPU, `--device cuda` passes the GPU into the container,
but that needs a CUDA build of PyTorch in the image (the Dockerfile installs the CPU one).

Work in progress: full documentation follows.
