# Research script as run for docs/experiments.md: python -I check_styleganex.py REPO/src MODELS IMAGES OUT [...]
import sys, time, numpy as np, cv2
mode, path, out = sys.argv[2], sys.argv[3], sys.argv[4]
rng = np.random.default_rng(0)
bg = (cv2.GaussianBlur(rng.random((384, 384, 3)).astype(np.float32), (0, 0), 6) * 2 - 1).transpose(2, 0, 1)[None].copy()
face = (cv2.GaussianBlur(rng.random((256, 256, 3)).astype(np.float32), (0, 0), 6) * 2 - 1).transpose(2, 0, 1)[None].copy()
if mode == "ort":
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    t = time.time(); ref = sess.run(None, {"target_with_background": bg, "target": face, "direction": np.array(1.0, np.float32)})[0]
    print(f"ort {time.time() - t:.1f}s"); np.save(out, ref)
else:
    import torch
    sys.path.insert(0, sys.argv[1])
    from bifold_morph.nets import StyleGANEX
    torch.set_num_threads(4)
    t = time.time(); sgx = StyleGANEX(path); print(f"converted in {time.time() - t:.0f}s")
    t = time.time()
    with torch.inference_mode():
        o, codes = sgx.net(torch.from_numpy(bg), torch.from_numpy(face), torch.from_numpy(-1.0 * sgx.age_vector[None]))
    print(f"torch {time.time() - t:.1f}s, output {tuple(o.shape)}, codes {tuple(codes.shape)}")
    ref = np.load(out); d = np.abs(o.numpy() - ref)
    print(f"max diff {d.max():.4f} mean {d.mean():.5f}")
