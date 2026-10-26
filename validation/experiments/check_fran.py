# Research script as run for docs/experiments.md: python -I check_fran.py REPO/src MODELS IMAGES OUT [...]
import sys, time, numpy as np, onnx, onnxruntime as ort, torch, cv2
from onnx2torch import convert
path = sys.argv[1]
g = onnx.load(path)
pads = [n for n in g.graph.node if n.op_type == "Pad"]
print("Pad inputs per node:", {len([i for i in n.input if i]) for n in pads}, "modes:", {a.s for n in pads for a in n.attribute if a.name == "mode"})
print("ScatterND reductions:", {a.s for n in g.graph.node if n.op_type == "ScatterND" for a in n.attribute})
for o in g.opset_import:
    if o.domain in ("", "ai.onnx"):
        o.version = 17
for n in g.graph.node:
    if n.op_type == "Reshape":
        for a in n.attribute:
            if a.name == "allowzero": a.i = 0
model = convert(g).eval()
sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
rng = np.random.default_rng(0)
for size in (1024,):
    img = cv2.GaussianBlur(rng.random((size, size, 3)).astype(np.float32), (0, 0), 6).transpose(2, 0, 1)[None].copy()
    d = np.array([0.3, 0.7], np.float32)
    if size == 1024:
        t = time.time(); ref = sess.run(None, {"target": img, "direction": d})[0]; to = time.time() - t
    else:
        ref, to = None, 0.0
    t = time.time()
    with torch.inference_mode():
        out = model(torch.from_numpy(img), torch.from_numpy(d)).numpy()
    tt = time.time() - t
    diff = np.abs(out - ref).max() if ref is not None else float("nan")
    print(f"{size}: ort {to:.2f}s torch {tt:.2f}s out {out.shape} max diff {diff:.5f} | output-input mean |{np.abs(out - img).mean():.4f}|")
