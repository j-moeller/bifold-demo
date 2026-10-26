# bifold-morph

Live webcam face editing with two sliders: **age** (younger ↔ older, in years) and **gender** (female ↔ male).

MiVOLO only *estimates* age and gender, so the generative part is a different network:

- **e4e** (encoder4editing) turns each face into a StyleGAN2 code;
- the code moves along an **age** and a **gender** direction in StyleGAN2's latent space;
- **StyleGAN2 (FFHQ)** draws the edited face, and the change is blended back into the video.

**MiVOLO** finds the directions: it labels a few thousand generated faces, and a linear model of those labels gives
the directions, calibrated in years. It also reads each face before and after the edit, live.

```bash
pip install -e .
bifold-morph setup      # once: download the models (~1.3 GB), find the directions with MiVOLO
bifold-morph            # the live demo on webcam 0
```

Work in progress: full documentation follows.
