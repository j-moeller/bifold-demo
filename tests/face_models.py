"""Tiny ONNX models for the face recognition tests, built with the ``onnx`` package (``pytest.importorskip("onnx")``
first): stand-ins with the inputs and outputs of the real ones, small enough to make in a test."""

from __future__ import annotations

import numpy as np


def save(model, path):
    import onnx

    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return path


def embedder(path, dim: int = 8, normalise: bool = False, seed: int = 0, batch="batch_size"):
    """An "embedder": ``input`` (batch, 3, 112, 112) -> ``output`` (batch, dim): the mean of each colour channel of
    each 28x28 cell, mixed by a fixed random matrix. With ``normalise`` the output is divided by its length inside the
    graph (as the AdaFace export does), and that length is the tensor ``norm``."""
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(seed)
    weights = rng.normal(size=(3 * 16, dim)).astype(np.float32)
    nodes = [helper.make_node("AveragePool", ["input"], ["cells"], kernel_shape=[28, 28], strides=[28, 28]),
             helper.make_node("Flatten", ["cells"], ["flat"], axis=1),
             helper.make_node("MatMul", ["flat", "weights"], ["raw" if normalise else "output"])]
    if normalise:
        nodes += [helper.make_node("ReduceL2", ["raw"], ["norm"], axes=[1], keepdims=1),
                  helper.make_node("Div", ["raw", "norm"], ["output"])]
    graph = helper.make_graph(
        nodes, "embedder",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [batch, 3, 112, 112])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [batch, dim])],
        [numpy_helper.from_array(weights, "weights")])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    return save(model, path), weights


def doubler(path):
    """``x`` (free shape) -> ``y = 2 x + 1`` with ``twice = 2 x`` inside the graph."""
    from onnx import TensorProto, helper, numpy_helper

    nodes = [helper.make_node("Mul", ["x", "two"], ["twice"]), helper.make_node("Add", ["twice", "one"], ["y"])]
    graph = helper.make_graph(
        nodes, "doubler", [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["n", 3])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["n", 3])],
        [numpy_helper.from_array(np.float32(2.0), "two"), numpy_helper.from_array(np.float32(1.0), "one")])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    return save(model, path)
