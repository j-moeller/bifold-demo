# Research script as run for docs/experiments.md: python -I check_hyperswap.py REPO/src MODELS IMAGES OUT [...]
import sys, numpy as np, onnx, onnxruntime as ort, torch
from onnx2torch import convert
path = sys.argv[1]
g = onnx.load(path)
for n in g.graph.node:
    if n.op_type == "Reshape":
        for a in n.attribute:
            if a.name == "allowzero": a.i = 0
print("Resize nodes:", [(n.name, {a.name: (a.s if a.type == 3 else a.i if a.type == 2 else a.f) for a in n.attribute}) for n in g.graph.node if n.op_type == "Resize"][:3])
print("InstanceNorm eps:", {a.f for n in g.graph.node if n.op_type == "InstanceNormalization" for a in n.attribute})
print("initializer dtypes:", {i.data_type for i in g.graph.initializer})
model = convert(g).eval()
sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
rng = np.random.default_rng(0)
src = rng.normal(size=(1, 512)).astype(np.float32); src /= np.linalg.norm(src)
img = (rng.random((1, 3, 256, 256)).astype(np.float32) * 2 - 1)
import cv2
img = cv2.GaussianBlur(img[0].transpose(1, 2, 0), (0, 0), 6).transpose(2, 0, 1)[None]  # smoother, face-like statistics
ref = sess.run(None, {"source": src, "target": img})
with torch.inference_mode():
    out = model(torch.from_numpy(src), torch.from_numpy(img))
d = np.abs(out[0].numpy() - ref[0])
print(f"image diff: max {d.max():.4f} mean {d.mean():.5f} p99 {np.percentile(d, 99):.4f}; mask diff max {np.abs(out[1].numpy() - ref[1]).max():.4f}")
