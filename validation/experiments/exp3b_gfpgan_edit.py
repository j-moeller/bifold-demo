# Research script as run for docs/experiments.md: python -I exp3b_gfpgan_edit.py REPO/src MODELS IMAGES OUT [...]
"""Experiment 3: InterFaceGAN gender direction in StyleGAN2-512, applied to real faces through GFPGAN's encoder + SFT."""
import sys, numpy as np, cv2, torch
sys.path.insert(0, sys.argv[1])
from bifold_morph import warp
from bifold_morph.warp import TEMPLATES
from bifold_morph.faces import FaceDetector
from bifold_morph.gfpgan import GFPGANv1, to_input, to_bgr
from bifold_morph.nets import ArcFace, FairFace, GenderAge
from bifold_morph.directions import ridge, unit, without, logit
M, I, OUT = sys.argv[2], sys.argv[3], sys.argv[4]
torch.set_num_threads(4)
d = np.load(f"{OUT}/samples.npz"); W, F, A = d["w"].astype(np.float64), d["female"], d["age"]
bg, ba = ridge(W, logit(F)), ridge(W, A)
dg, da = without(unit(bg), ba), without(unit(ba), bg)        # towards female / older, conditional
p = (W - W.mean(0)) @ dg; gap = p[F >= .5].mean() - p[F < .5].mean()
print(f"gender: female-male gap {gap:.2f} units along the direction; age: {ba @ da:.1f} years per unit")
np.savez(f"{OUT}/directions_gfpgan.npz", gender=dg, age=da, gap=gap, years=ba @ da)
gf = GFPGANv1.load(f"{M}/GFPGANv1.pth"); G = gf.stylegan_decoder
ff, ga, arc = FairFace(f"{M}/fairface.onnx"), GenderAge(f"{M}/buffalo_l/genderage.onnx"), ArcFace(f"{M}/arcface_w600k_r50.onnx")
lm = TEMPLATES["ffhq_512"] * 512
box = (lm[:, 0].min() - 60, lm[:, 1].min() - 90, np.ptp(lm[:, 0]) + 120, np.ptp(lm[:, 1]) + 150)
# 1. synthetic: random faces moved along the direction, read by the independent judge (InsightFace)
torch.manual_seed(123)
with torch.inference_mode():
    w0 = G.mean_latent(10000); w = (w0 + 0.7 * (G.mapping(torch.randn(16, 512)) - w0)).numpy()
    base = to_bgr(G.synthesis(torch.from_numpy(w))).astype(np.uint8)
    rb = ga.read_many(base, box)
    sign = np.array([-1 if r.female >= .5 else 1 for r in rb])[:, None]
    for s in (0.5, 1.0, 1.5):
        e = to_bgr(G.synthesis(torch.from_numpy((w + sign * s * gap * dg).astype(np.float32)))).astype(np.uint8)
        flips = [(r.female >= .5) != (b.female >= .5) for r, b in zip(ga.read_many(e, box), rb)]
        ffl = [(r.female >= .5) != (b.female >= .5) for r, b in zip(ff.read_many(e, lm), ff.read_many(base, lm))]
        print(f"synthetic, {s:.1f} gaps: InsightFace sees the other gender in {np.mean(flips):.0%}, FairFace in {np.mean(ffl):.0%}")
        if s == 1.0: cv2.imwrite(f"{OUT}/exp3_synthetic.jpg", np.vstack([np.hstack([cv2.resize(x, (128, 128)) for x in base[:8]]), np.hstack([cv2.resize(x, (128, 128)) for x in e[:8]])]))
# 2. real faces through GFPGAN
det = FaceDetector(f"{M}/face_detection_yunet_2023mar.onnx")
rows = []
for name in ("t1.jpg", "lena.jpg"):
    img = cv2.imread(f"{I}/{name}")
    for k, f in enumerate(det.detect(img)):
        m = warp.matrix(f.landmarks, "ffhq_512", 512); crop = warp.warp(img, m, 512)
        r0 = (ff.read(img, [f.landmarks])[0].female + ga.read(img, [f.box])[0].female) / 2
        sign = -1 if r0 >= .5 else 1
        e0 = arc.embed(img, [f.landmarks])[0]
        tiles = []
        with torch.inference_mode():
            codes, cond = gf.encode(to_input(crop[None], "cpu"))
            for s in (0.0, 1.0, 2.0, 3.0):
                off = torch.from_numpy((sign * s * gap * dg).astype(np.float32))
                head = to_bgr(G.synthesis(codes + off, conditions=cond))[0]
                out = img.copy(); warp.paste(out, head, warp.box_mask(512, 0.3), m)
                a, b = ff.read(out, [f.landmarks])[0], ga.read(out, [f.box])[0]
                rows.append((f"{name}#{k}", "F" if r0 >= .5 else "M", s, a.female, b.female, float(arc.embed(out, [f.landmarks])[0] @ e0)))
                tiles.append(cv2.resize(head.astype(np.uint8), (192, 192)))
        cv2.imwrite(f"{OUT}/exp3_{name}_{k}.jpg", np.hstack(tiles))
print("face        was  gaps  FairFace P(female)  InsightFace P(female)  identity")
for r in rows: print(f"{r[0]:11s} {r[1]:4s} {r[2]:4.1f}  {r[3]:18.2f}  {r[4]:21.2f}  {r[5]:8.2f}")
