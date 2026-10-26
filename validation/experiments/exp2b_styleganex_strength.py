# Research script as run for docs/experiments.md: python -I exp2b_styleganex_strength.py REPO/src MODELS IMAGES OUT [...]
import sys, numpy as np, cv2, torch
sys.path.insert(0, sys.argv[1])
from bifold_morph import warp as align
from bifold_morph.faces import FaceDetector
from bifold_morph.nets import StyleGANEX, FairFace, GenderAge
M, I, OUT = sys.argv[2], sys.argv[3], sys.argv[4]
torch.set_num_threads(4)
det = FaceDetector(f"{M}/face_detection_yunet_2023mar.onnx")
ff, ga = FairFace(f"{M}/fairface.onnx"), GenderAge(f"{M}/buffalo_l/genderage.onnx")
sgx = StyleGANEX(f"{M}/styleganex_age.onnx")
g = np.load(f"{OUT}/sgx_gender.npy"); g3 = g.copy(); g3[:3] = 0
import os
img = cv2.imread(f"{I}/{os.environ["IMG"]}"); f = det.detect(img)[int(os.environ["K"])]
tests = [("none", 0 * g), ("age older 2.5", 2.5 * sgx.age_vector), ("age younger 2.5", -2.5 * sgx.age_vector),
         ("g[3:] x8", float(os.environ["SIGN"]) * 8 * g3), ("g[3:] x16", float(os.environ["SIGN"]) * 16 * g3)]
tiles = []
for name, off in tests:
    head, m, codes = sgx.edit(img, f.landmarks, off)
    out = img.copy(); align.paste(out, head.astype(np.float32), align.box_mask(head.shape[0], 0.3), m)
    a, b = ff.read(out, [f.landmarks])[0], ga.read(out, [f.box])[0]
    print(f"{name:16s} FF female {a.female:.2f} age {a.age:4.1f} | IF female {b.female:.2f} age {b.age:4.1f} | head diff {np.abs(head.astype(float) - sgx.edit(img, f.landmarks, 0*g)[0]).mean() if name!='none' else 0:.1f}")
    x, y, w, h = map(int, f.box); c = max(w, h) * 2; cx, cy = x + w // 2, y + h // 2
    tile = cv2.resize(cv2.copyMakeBorder(out, c, c, c, c, cv2.BORDER_CONSTANT)[cy + c - c // 2:cy + c + c // 2, cx + c - c // 2:cx + c + c // 2], (192, 192))
    cv2.putText(tile, name, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1); tiles.append(tile)
cv2.imwrite(f"{OUT}/exp2b_{os.environ["IMG"]}_{os.environ["K"]}.jpg", np.hstack(tiles))
