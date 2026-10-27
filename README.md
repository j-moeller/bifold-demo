# bifold-morph

A live webcam gimmick with two sliders: **gender** (female ↔ male) and **age** (younger ↔ older, in years).

```
 webcam ─► YuNet: faces + 5 landmarks ─► head cut out (FFHQ template, 256 px)
                                               │
                         sliders ─► student U-Net (one pass, ~6 M parameters)
                                               │
 window ◄── pasted back through a soft box ◄───┘     judges: "female 91%, 34 y"
```

## How it works (and how the apps do it)

Snapchat's gender swap, FaceApp and similar filters run a small feed-forward network on an aligned face crop,
on the phone, in real time. The published recipe that fits them is *StyleGAN2 distillation* (Viazovetskyi et al.,
ECCV 2020), and this project follows it:

1. **A generator that knows how gender and age look.** StyleGAN2 (FFHQ, 512 px, from GFPGAN v1's checkpoint) draws
   random faces. A face classifier (FairFace) labels 2000 of them, and linear models in StyleGAN's latent space
   give an InterFaceGAN *gender direction* and *age direction*. Moving a code along them turns the face it draws
   into the other gender, or older or younger.
2. **Pairs.** 4000 random faces, each also drawn moved by a random gender amount and age. That gives before/after
   pairs of synthetic people, with the amounts.
3. **The student.** A U-Net learns to turn *before* into *after*, given the amounts. The app runs only the student:
   one pass per face and frame, a few milliseconds on a laptop GPU.

Why not something simpler? See [docs/literature.md](docs/literature.md) for the review and
[docs/experiments.md](docs/experiments.md) for what was tried and measured:
- face swapping with a "more female" identity changes *who* someone looks like, not their gender;
- StyleGANEX's age edit is weak;
- GAN encoders for photos could not carry the edit.

## Quick start (macOS)

Python 3.10–3.13. An Apple-silicon Mac runs the student on its GPU (MPS).

```bash
git clone -b claude/stoic-hopper-9fdrhr https://github.com/j-moeller/bifold-demo.git bifold-morph
cd bifold-morph
make                    # once: venv, install, models, tests; then the webcam demo. Run it again any time.
```

`make` skips what is already done. Without it: `python3 -m venv .venv && source .venv/bin/activate &&
pip install -e . && bifold-morph`. The first run downloads YuNet and the two judges (about 90 MB).

The student network comes with the package (`src/bifold_morph/weights/student.pt`). Drag the sliders, or:

| Key | Action |
|---|---|
| `←` `→` | move the selected slider (`↑` `↓` select) |
| `1` / `2` | 5 years younger / older |
| `3` / `4` | towards female / male |
| `R` | both sliders back to 0 |
| `Space` | show the original |
| `E` | read-out on or off (what the judges think: gender and age) |
| `B` | face boxes on or off |
| `M` | mirror on or off |
| `F` | fullscreen |
| `P` | screenshot to `screenshots/` |
| `H` | hide the sliders |
| `Q` / `Esc` | quit |

Double-click or right-click a slider to set it back to 0.

```bash
bifold-morph run --source clip.mp4          # a video file or a photo instead of the webcam
bifold-morph edit me.jpg --gender -1        # one photo, towards female
bifold-morph edit me.jpg --age 30 -o old.jpg
bifold-morph train --work ~/morph           # make the student again (see below)
```

## Results

See [docs/experiments.md](docs/experiments.md), "Step 5". Measured on photos of real people the student never
saw, read by InsightFace's gender classifier, which had no part in making the change.

## Making the student again

`bifold-morph train --work DIR` repeats steps 1–3:
- downloads GFPGAN v1 (615 MB) and FairFace;
- labels 2000 random faces and fits the directions;
- draws 4000 pairs (about 1.5 GB at 256 px);
- trains for 4000 steps.

On a 4-core CPU that took about 1.5 hours for the pairs plus the training. A GPU is much faster (`--device mps`
or `cuda`). `validation/validate.py` measures a student on your own photos.

## The networks and their licences

| Network | Used for | Weights from | Licence |
|---|---|---|---|
| Student U-Net | the change | this repository, distilled from StyleGAN2 | follows StyleGAN2/FFHQ: non-commercial |
| StyleGAN2 FFHQ-512 + GFPGAN v1 encoder | making the pairs | `TencentARC/GFPGAN` release v0.1.0 | code Apache 2.0; FFHQ CC BY-NC-SA 4.0 |
| YuNet | finding faces | OpenCV Zoo | MIT |
| FairFace | training loss, read-out | `facefusion-assets` | CC BY 4.0 |
| InsightFace genderage | read-out, validation | `deepinsight/insightface` release v0.7 | non-commercial research |
| ArcFace w600k_r50 | validation (identity) | `facefusion-assets` | non-commercial research |

It is a gimmick for demos and fun, not a product. The weights' licences exclude commercial use. Gender is shown
as the binary label the classifiers were trained on; the sliders change looks, nothing else.

The ONNX models run as PyTorch modules (`onnx2torch`, with the fixes in `onnx_torch.py`), so everything runs on
PyTorch's devices. It needs PyTorch ≥ 2.14; 2.15 has not been released yet.

## Tests

```bash
pip install -e ".[dev]" && pytest
```
