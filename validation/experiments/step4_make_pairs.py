# Research script as run for docs/experiments.md: python -I step4_make_pairs.py REPO/src MODELS IMAGES OUT [...]
import sys, numpy as np, torch
sys.path.insert(0, sys.argv[1])
from bifold_morph.gfpgan import GFPGANv1
from bifold_morph.distill import Directions, fit_directions, make_pairs
M, OUT, N = sys.argv[2], sys.argv[3], int(sys.argv[4])
torch.set_num_threads(4)
G = GFPGANv1.load(f"{M}/GFPGANv1.pth").stylegan_decoder
s = np.load(f"{OUT}/samples.npz")
d = fit_directions(s["w"].astype(np.float64), s["female"], s["age"])
d.save(f"{OUT}/directions.npz")
print(f"gap {d.gap:.2f}, {d.years:.1f} years per unit", flush=True)
make_pairs(G, d, N, f"{OUT}/pairs", progress=lambda m: print(m, flush=True))
print("done", flush=True)
