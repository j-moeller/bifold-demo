# Research script as run for docs/experiments.md: python -I exp3a_sample_labels.py REPO/src MODELS IMAGES OUT [...]
"""InterFaceGAN directions in StyleGAN2-512's w space, labelled by FairFace."""
import sys, time, numpy as np, torch
sys.path.insert(0, sys.argv[1])
from bifold_morph.warp import TEMPLATES
from bifold_morph.gfpgan import GFPGANv1, to_bgr
from bifold_morph.nets import FairFace
M, OUT, N = sys.argv[2], sys.argv[3], int(sys.argv[4])
torch.set_num_threads(4); torch.manual_seed(0)
g = GFPGANv1.load(f"{M}/GFPGANv1.pth").stylegan_decoder
ff = FairFace(f"{M}/fairface.onnx")
lm = TEMPLATES["ffhq_512"] * 512
with torch.inference_mode():
    w_avg = g.mean_latent(10000)
    W, F, A = [], [], []
    t0 = time.time()
    for i in range(0, N, 8):
        w = w_avg + 0.7 * (g.mapping(torch.randn(8, 512)) - w_avg)
        imgs = to_bgr(g.synthesis(w)).astype(np.uint8)
        for r in ff.read_many(imgs, lm):
            F.append(r.female); A.append(r.age)
        W.append(w.numpy())
        if (i // 8) % 25 == 0: print(f"{i + 8}/{N} {time.time() - t0:.0f}s", flush=True)
W = np.concatenate(W); F = np.array(F); A = np.array(A)
np.savez(f"{OUT}/samples.npz", w=W, female=F, age=A, w_avg=w_avg.numpy())
print(f"{N} faces: {np.mean(F >= .5):.0%} female, age {A.mean():.0f} ± {A.std():.0f}")
