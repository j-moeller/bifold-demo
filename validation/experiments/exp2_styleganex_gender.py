# Research script as run for docs/experiments.md: python -I exp2_styleganex_gender.py REPO/src MODELS IMAGES OUT [...]
"""Experiment 2: StyleGANEX latent editing with a gender vector fitted on encoded faces."""
import sys, glob, time, numpy as np, cv2, torch
sys.path.insert(0, sys.argv[1])
from bifold_morph import warp as align
from bifold_morph.faces import FaceDetector
from bifold_morph.nets import ArcFace, StyleGANEX, FairFace, GenderAge
M, I, OUT = sys.argv[2], sys.argv[3], sys.argv[4]
torch.set_num_threads(4)
det = FaceDetector(f"{M}/face_detection_yunet_2023mar.onnx")
arc, ff, ga = ArcFace(f"{M}/arcface_w600k_r50.onnx"), FairFace(f"{M}/fairface.onnx"), GenderAge(f"{M}/buffalo_l/genderage.onnx")
t = time.time(); sgx = StyleGANEX(f"{M}/styleganex_age.onnx"); print(f"StyleGANEX loaded in {time.time()-t:.0f}s; age vector per-layer norms {np.linalg.norm(sgx.age_vector, axis=1).round(2)}")
zero = np.zeros((18, 512), np.float32)
C, Y = [], []
for p in sorted(glob.glob(f"{I}/deepface/*.jpg")):
    img = cv2.imread(p); faces = det.detect(img)
    if not faces: continue
    f = faces[0]
    a, b = ff.read(img, [f.landmarks])[0], ga.read(img, [f.box])[0]
    if (a.female >= .5) != (b.female >= .5): continue
    t = time.time(); _, _, codes = sgx.edit(img, f.landmarks, zero); dt = time.time() - t
    C.append(codes); Y.append(a.female >= .5)
C, Y = np.array(C), np.array(Y)
print(f"encoded {len(C)} faces ({Y.sum()} female), {dt:.1f}s per face on this CPU")
g = C[Y].mean(0) - C[~Y].mean(0)   # towards female, (18, 512)
a = sgx.age_vector.ravel() / np.linalg.norm(sgx.age_vector)
g = (g.ravel() - (g.ravel() @ a) * a).reshape(18, 512)  # leave age alone
print("gender vector per-layer norms", np.linalg.norm(g, axis=1).round(2))
np.save(f"{OUT}/sgx_gender.npy", g); np.save(f"{OUT}/sgx_codes.npy", C); np.save(f"{OUT}/sgx_labels.npy", Y)
val = [("t1", cv2.imread(f"{I}/t1.jpg")), ("lena", cv2.imread(f"{I}/lena.jpg"))]
rows = []
for name, img in val:
    for k, f in enumerate(det.detect(img)):
        e = arc.embed(img, [f.landmarks])[0]
        r0 = (ff.read(img, [f.landmarks])[0].female + ga.read(img, [f.box])[0].female) / 2
        female = r0 >= .5
        sign = -1 if female else 1
        tiles = []
        for s in [0.0, 1.0, 1.5, 2.0]:
            out = img.copy()
            t = time.time()
            head, m, _ = sgx.edit(img, f.landmarks, sign * s * g)
            dt = time.time() - t
            align.paste(out, head.astype(np.float32), align.box_mask(head.shape[0], 0.3), m)
            ra, rb = ff.read(out, [f.landmarks])[0], ga.read(out, [f.box])[0]
            rows.append((f"{name}#{k}", "F" if female else "M", s, ra.female, rb.female, float(arc.embed(out, [f.landmarks])[0] @ e), dt))
            x, y, w, h = map(int, f.box); c = max(w, h) * 2; cx, cy = x + w // 2, y + h // 2
            tile = cv2.resize(cv2.copyMakeBorder(out, c, c, c, c, cv2.BORDER_CONSTANT)[cy + c - c // 2:cy + c + c // 2, cx + c - c // 2:cx + c + c // 2], (192, 192))
            cv2.putText(tile, f"s={s:g} FF {ra.female:.2f} IF {rb.female:.2f}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            tiles.append(tile)
        cv2.imwrite(f"{OUT}/exp2_{name}_{k}.jpg", np.hstack(tiles))
print("face      was  strength  FairFace P(female)  InsightFace P(female)  identity cos  edit s")
for r in rows:
    print(f"{r[0]:9s} {r[1]:4s} {r[2]:8.1f}  {r[3]:18.2f}  {r[4]:21.2f}  {r[5]:12.2f}  {r[6]:.1f}")
