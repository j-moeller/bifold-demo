import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper

from bifold_morph.onnx_torch import load


def test_patched_graph_converts_and_computes_the_same(tmp_path):
    """A graph with the two things onnx2torch cannot take as they are: a Reshape with allowzero=1 and a Conv whose
    weights are computed (here: a stored weight times an input scale, like StyleGAN's modulated convolution)."""
    rng = np.random.default_rng(0)
    w = rng.normal(size=(4, 3, 3, 3)).astype(np.float32)
    nodes = [
        helper.make_node("Mul", ["w", "scale"], ["w2"]),
        helper.make_node("Conv", ["x", "w2"], ["y"], pads=[1, 1, 1, 1]),
        helper.make_node("Reshape", ["y", "shape"], ["z"], allowzero=1),
    ]
    graph = helper.make_graph(
        nodes, "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 8, 8]),
         helper.make_tensor_value_info("scale", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 256])],
        [numpy_helper.from_array(w, "w"), numpy_helper.from_array(np.array([1, 256], np.int64), "shape")])
    path = tmp_path / "g.onnx"
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), path)
    module = load(path)
    x = rng.normal(size=(1, 3, 8, 8)).astype(np.float32)
    with torch.no_grad():
        out = module(torch.from_numpy(x), torch.tensor([2.0]))
    ref = torch.nn.functional.conv2d(torch.from_numpy(x), torch.from_numpy(w) * 2, padding=1).reshape(1, 256)
    assert torch.allclose(out, ref, atol=1e-5)
