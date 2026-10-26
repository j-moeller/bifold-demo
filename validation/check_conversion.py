"""Check every ONNX -> PyTorch conversion against ONNX Runtime on the same input.

    python validation/check_conversion.py MODELS_DIR
"""
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bifold_morph.onnx_torch import load  # noqa: E402

M = sys.argv[1]
cases = {
  "hyperswap_1a_256.onnx": {"source": np.random.randn(1, 512).astype(np.float32), "target": np.random.rand(1, 3, 256, 256).astype(np.float32) * 2 - 1},
  "arcface_w600k_r50.onnx": {"input": np.random.rand(1, 3, 112, 112).astype(np.float32) * 2 - 1},
  "genderage.onnx": {"data": np.random.rand(1, 3, 96, 96).astype(np.float32) * 255},
  "fran.onnx": {"target": np.random.rand(1, 3, 512, 512).astype(np.float32), "direction": np.array([0.3, 0.7], np.float32)},
  "fairface.onnx": {"input": np.random.randn(1, 3, 224, 224).astype(np.float32)},
}
cases["hyperswap_1a_256.onnx"]["source"] /= np.linalg.norm(cases["hyperswap_1a_256.onnx"]["source"])
for name, feeds in cases.items():
    path = f"{M}/{name}"
    t = time.time()
    model = load(path)
    tc = time.time() - t
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    ref = sess.run(None, feeds)
    t = time.time()
    with torch.inference_mode():
        out = model(*[torch.from_numpy(feeds[i.name]) for i in sess.get_inputs()])
    tt = time.time() - t
    out = out if isinstance(out, (list, tuple)) else [out]
    diffs = [float(np.abs(np.asarray(o) - r).max()) for o, r in zip(out, ref)]
    print(f"{name}: convert {tc:.1f}s, torch forward {tt:.2f}s, max diff {diffs}, out shapes {[r.shape for r in ref]}")
