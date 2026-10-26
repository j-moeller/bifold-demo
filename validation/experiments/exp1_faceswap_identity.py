# Research script as run for docs/experiments.md: python -I exp1_faceswap_identity.py REPO/src MODELS IMAGES OUT [...]
"""Experiment 1: gender swap = HyperSwap redrawing a face with its own ArcFace identity moved along a gender direction."""
import sys, glob, time, numpy as np, cv2, torch
sys.path.insert(0, sys.argv[1])  # repo src
from bifold_morph import warp as align
from bifold_morph.faces import FaceDetector
from bifold_morph.nets import ArcFace, HyperSwap, FairFace, GenderAge
M, I, OUT = sys.argv[2], sys.argv[3], sys.argv[4]
torch.set_num_threads(4)
det = FaceDetector(f"{M}/face_detection_yunet_2023mar.onnx")
arc, ff, ga = ArcFace(f"{M}/arcface_w600k_r50.onnx"), FairFace(f"{M}/fairface.onnx"), GenderAge(f"{M}/buffalo_l/genderage.onnx")
swap = HyperSwap(f"{M}/hyperswap_1a_256.onnx")

# 1. the direction, from the DeepFace sample photos (labels: both judges must agree)
E, Y = [], []
for p in sorted(glob.glob(f"{I}/deepface/*.jpg")):
    img = cv2.imread(p); faces = det.detect(img)
    if not faces: continue
    f = faces[0]
    a, b = ff.read(img, [f.landmarks])[0], ga.read(img, [f.box])[0]
    if (a.female >= .5) != (b.female >= .5): continue
    E.append(arc.embed(img, [f.landmarks])[0]); Y.append(a.female >= .5)
E, Y = np.array(E), np.array(Y)
d = E[Y].mean(0) - E[~Y].mean(0)
gap = np.linalg.norm(d); d /= gap
proj = E @ d
print(f"direction from {len(E)} faces ({Y.sum()} female): |mean_f - mean_m| = {gap:.3f}; projections female {proj[Y].mean():.3f}±{proj[Y].std():.3f}, male {proj[~Y].mean():.3f}±{proj[~Y].std():.3f}")
np.save(f"{OUT}/gender_direction.npy", d)

# 2. validation faces: swap toward the other gender with several strengths
val = [("t1", cv2.imread(f"{I}/t1.jpg")), ("lena", cv2.imread(f"{I}/lena.jpg")), ("messi", cv2.imread(f"{I}/messi5.jpg"))]
rows = []
strengths = [0.0, 1.0, 2.0, 3.0]
for name, img in val:
    for k, f in enumerate(det.detect(img)):
        e = arc.embed(img, [f.landmarks])[0]
        r0f, r0g = ff.read(img, [f.landmarks])[0], ga.read(img, [f.box])[0]
        female = (r0f.female + r0g.female) / 2 >= .5
        sign = -1 if female else 1  # towards the other gender
        tiles = []
        for s in strengths:
            e2 = e + sign * s * gap * d; e2 /= np.linalg.norm(e2)
            out = img.copy()
            t = time.time()
            crop, mask, m = swap.swap(img, f.landmarks, e2)
            dt = time.time() - t
            align.paste(out, crop, np.minimum(mask, align.box_mask(256)), m)
            lm = f.landmarks
            a, b = ff.read(out, [lm])[0], ga.read(out, [f.box])[0]
            ident = float(arc.embed(out, [lm])[0] @ e)
            rows.append((f"{name}#{k}", "F" if female else "M", s, a.female, b.female, ident, dt))
            x, y, w, h = map(int, f.box); c = max(w, h) * 2
            cx, cy = x + w // 2, y + h // 2
            tile = cv2.resize(cv2.copyMakeBorder(out, c, c, c, c, cv2.BORDER_CONSTANT)[cy + c - c // 2:cy + c + c // 2, cx + c - c // 2:cx + c + c // 2], (192, 192))
            cv2.putText(tile, f"s={s:g} FF {a.female:.2f} IF {b.female:.2f}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            tiles.append(tile)
        cv2.imwrite(f"{OUT}/exp1_{name}_{k}.jpg", np.hstack(tiles))
print("face      was  strength  FairFace P(female)  InsightFace P(female)  identity cos  swap s")
for r in rows:
    print(f"{r[0]:9s} {r[1]:4s} {r[2]:8.1f}  {r[3]:18.2f}  {r[4]:21.2f}  {r[5]:12.2f}  {r[6]:.2f}")
