"""Loading the published ONNX models as PyTorch modules (with onnx2torch), so they run on PyTorch's devices: the
CPU, CUDA and Apple's MPS.

Two things in these graphs onnx2torch does not take as they are, both changed without changing what they compute:

* ``Reshape`` with ``allowzero=1`` (HyperSwap, FRAN): the flag only matters for a 0 in the target shape, and none
  of these shapes have one;
* opset 18 (FRAN): of what 18 changed, these graphs use only ``Pad`` without its new ``axes`` input and
  ``ScatterND`` without its new min/max reduction, which are the same as in opset 17.

And one onnx2torch cannot convert at all: convolutions whose weights are computed rather than stored (StyleGAN's
modulated convolutions, in StyleGANEX). :func:`_dynamic_conv` adds a converter for those.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import torch
import torch.nn.functional as F

_installed = False


class DynamicConv(torch.nn.Module):
    """An ONNX ``Conv`` / ``ConvTranspose`` whose weights (and bias) arrive as inputs."""

    def __init__(self, transpose: bool, pads, strides, dilations, groups: int, output_padding):
        super().__init__()
        self.transpose, self.groups = transpose, groups
        self.strides, self.dilations, self.output_padding = tuple(strides), tuple(dilations), tuple(output_padding)
        n = len(pads) // 2
        begin, end = list(pads[:n]), list(pads[n:])
        self.symmetric = begin == end
        self.padding = tuple(begin)
        # F.pad wants the last axis first: (left, right, top, bottom)
        self.pad = [p for b, e in zip(reversed(begin), reversed(end)) for p in (b, e)]

    def forward(self, x, weight, bias=None):
        if self.transpose:
            out = F.conv_transpose2d(x, weight, bias, self.strides, 0, self.output_padding, self.groups,
                                     self.dilations)
            if any(self.pad):  # ConvTranspose's pads crop the output
                l, r, t, b = self.pad
                out = out[..., t:out.shape[-2] - b, l:out.shape[-1] - r]
            return out
        if self.symmetric:
            return F.conv2d(x, weight, bias, self.strides, self.padding, self.dilations, self.groups)
        return F.conv2d(F.pad(x, self.pad), weight, bias, self.strides, 0, self.dilations, self.groups)


def _dynamic_conv() -> None:
    """Make onnx2torch convert convolutions with computed weights (2-D) to :class:`DynamicConv`."""
    global _installed
    if _installed:
        return
    from onnx2torch.node_converters.registry import _CONVERTER_REGISTRY, OperationDescription
    from onnx2torch.utils.common import OnnxMapping, OperationConverterResult

    for op in ("Conv", "ConvTranspose"):
        for version in (1, 11):
            key = OperationDescription("", op, version)
            original = _CONVERTER_REGISTRY[key]

            def converter(node, graph, original=original, op=op):
                if all(v in graph.initializers for v in node.input_values[1:]):
                    return original(node, graph)
                a = node.attributes
                kernel = a.get("kernel_shape", [3, 3])
                n = len(kernel)
                if n != 2 or a.get("auto_pad", "NOTSET") not in ("NOTSET", "VALID"):
                    raise NotImplementedError(f"computed-weight {op} with {n} spatial axes or auto_pad")
                module = DynamicConv(op == "ConvTranspose", a.get("pads", [0] * 2 * n), a.get("strides", [1] * n),
                                     a.get("dilations", [1] * n), a.get("group", 1),
                                     a.get("output_padding", [0] * n))
                return OperationConverterResult(module, OnnxMapping(node.input_values, node.output_values))

            _CONVERTER_REGISTRY[key] = converter
    _installed = True


def _patch(model) -> None:
    for node in model.graph.node:
        if node.op_type == "Reshape":
            for attr in node.attribute:
                if attr.name == "allowzero":
                    attr.i = 0
    for opset in model.opset_import:
        if opset.domain in ("", "ai.onnx") and opset.version == 18:
            pads_axes = any(n.op_type == "Pad" and len([i for i in n.input if i]) > 3 for n in model.graph.node)
            reduce = any(n.op_type == "ScatterND" and any(a.name == "reduction" and a.s != b"none"
                                                          for a in n.attribute) for n in model.graph.node)
            if pads_axes or reduce:
                raise ValueError("an opset 18 graph that uses what opset 17 lacks")
            opset.version = 17


def _expose(model, outputs: list[str]) -> None:
    """Make the tensors ``outputs`` the graph's outputs (e.g. logits before an ArgMax)."""
    import onnx

    del model.graph.output[:]
    for name in outputs:
        model.graph.output.append(onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None))


def rewire_input(model, tensor: str, name: str, shape: list[int]) -> None:
    """Make the graph read ``tensor`` from a new input ``name`` instead of computing it, and drop what then goes
    unused (e.g. to feed StyleGANEX a whole latent offset instead of one scalar times its built-in age vector)."""
    import onnx

    for node in model.graph.node:
        for i, inp in enumerate(node.input):
            if inp == tensor:
                node.input[i] = name
    model.graph.input.append(onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape))
    _prune(model)


def _prune(model) -> None:
    """Remove nodes, inputs and initializers nothing needed for the outputs uses."""
    needed = {o.name for o in model.graph.output}
    keep = []
    for node in reversed(model.graph.node):
        if any(o in needed for o in node.output):
            keep.append(node)
            needed.update(i for i in node.input if i)
    del model.graph.node[:]
    model.graph.node.extend(reversed(keep))
    inputs = [i for i in model.graph.input if i.name in needed]
    del model.graph.input[:]
    model.graph.input.extend(inputs)
    inits = [i for i in model.graph.initializer if i.name in needed]
    del model.graph.initializer[:]
    model.graph.initializer.extend(inits)


def constant(model, tensor: str):
    """The value of the constant ``tensor`` in the (loaded) ONNX ``model``, as a numpy array."""
    from onnx import numpy_helper

    for node in model.graph.node:
        if node.op_type == "Constant" and node.output[0] == tensor:
            return numpy_helper.to_array(node.attribute[0].t)
    for init in model.graph.initializer:
        if init.name == tensor:
            return numpy_helper.to_array(init)
    raise KeyError(tensor)


def load(path, device="cpu", outputs: list[str] | None = None, edit=None) -> torch.nn.Module:
    """The ONNX model at ``path`` as an eval-mode PyTorch module on ``device``; ``outputs`` replaces the graph's
    outputs by these tensor names, ``edit(model)`` may change the graph before it is converted. The weights keep the precision they were published in (HyperSwap's are 16-bit;
    its graph casts its inputs to match)."""
    import onnx
    from onnx2torch import convert

    _dynamic_conv()
    model = onnx.load(str(Path(path)))
    _patch(model)
    if outputs:
        _expose(model, outputs)
    if edit is not None:
        edit(model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # onnx2torch warns about every op it cannot trace exactly
        module = convert(model)
    return module.to(device).eval()
