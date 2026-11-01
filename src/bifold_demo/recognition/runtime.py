"""Running the face models' ONNX files with ONNX Runtime.

* **Telemetry off.** Since 1.29 the official Linux and macOS builds of ONNX Runtime send usage telemetry to Microsoft
  unless ``ORT_DISABLE_TELEMETRY=1`` is set before ONNX Runtime starts. Importing this package sets it (unless it is
  set already), and :func:`ort` also switches telemetry events off through the API.
* **Execution providers.** On a Mac with the CoreML provider (the official macOS wheel has it): CoreML first (ML
  Program format, all compute units), then the CPU for what CoreML does not take; elsewhere the CPU.
  ``BIFOLD_FACE_PROVIDER`` (``auto``, ``cpu`` or ``coreml``) overrides the choice and ``BIFOLD_FACE_THREADS`` sets
  the CPU threads. A session CoreML cannot build falls back to the CPU. Embeddings made with different providers
  may differ slightly (CoreML computes in 16-bit floats on the Neural Engine), so the gallery's caches remember the
  provider (see :mod:`.gallery`).
* **Fixed shapes.** :class:`OnnxModel` keeps one session per input shape. On the CPU one session takes every batch
  size (the graph's batch dimension stays free); with CoreML, which works best with fixed shapes, each input shape
  gets its own session with every dimension fixed in the graph. Callers send batches of a few fixed sizes
  (:func:`chunks`) so that both behave the same.
* **Extra outputs.** A tensor inside the graph can be made an output as well (the AdaFace export normalises its
  embedding inside the graph; its length, AdaFace's quality signal, is the tensor before that division). This edits
  the graph in memory with the ``onnx`` package; the weights are unchanged.
"""

from __future__ import annotations

import os
import threading
import warnings
from pathlib import Path

import numpy as np

os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")  # before ONNX Runtime is first imported (see the module docs)

PROVIDER_ENV = "BIFOLD_FACE_PROVIDER"
THREADS_ENV = "BIFOLD_FACE_THREADS"
COREML = "CoreMLExecutionProvider"
CPU = "CPUExecutionProvider"
BATCHES = (8, 1)  # the batch sizes crops are sent in (biggest first)

_lock = threading.Lock()
_ort = None


def ort():
    """The ``onnxruntime`` module, imported with its telemetry off (``ImportError`` says how to install it)."""
    global _ort
    with _lock:
        if _ort is None:
            os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
            try:
                import onnxruntime
            except ImportError as exc:
                raise ImportError(f"the face models need ONNX Runtime ({exc}); install it with: "
                                  "pip install onnxruntime") from exc
            switch_off = getattr(onnxruntime, "disable_telemetry_events", None)
            if switch_off is not None:  # (a no-op in builds without telemetry)
                switch_off()
            onnxruntime.set_default_logger_severity(3)  # errors only
            _ort = onnxruntime
    return _ort


def provider_choice(choice: str | None = None) -> str:
    """``"coreml"`` or ``"cpu"``: ``choice``, else ``$BIFOLD_FACE_PROVIDER``, else CoreML where it is available."""
    choice = (choice or os.environ.get(PROVIDER_ENV) or "auto").strip().lower()
    if choice not in ("auto", "cpu", "coreml"):
        raise ValueError(f"{PROVIDER_ENV} must be auto, cpu or coreml, not {choice!r}")
    if choice == "auto":
        return "coreml" if COREML in ort().get_available_providers() else "cpu"
    return choice


def providers(choice: str | None = None) -> list:
    """The execution providers for :func:`provider_choice` (with their options)."""
    if provider_choice(choice) == "coreml":
        return [(COREML, {"ModelFormat": "MLProgram", "MLComputeUnits": "ALL", "RequireStaticInputShapes": "1"}),
                CPU]
    return [CPU]


def threads() -> int:
    """CPU threads per session: ``$BIFOLD_FACE_THREADS``, else 0 (ONNX Runtime's choice: one per core)."""
    try:
        return max(0, int(os.environ.get(THREADS_ENV, "0")))
    except ValueError:
        return 0


def chunks(count: int, sizes=BATCHES) -> list[slice]:
    """``count`` items cut into batches of the fixed ``sizes`` (the biggest first): 11 -> 8, 1, 1, 1."""
    out, start = [], 0
    for size in sorted(sizes, reverse=True):
        while count - start >= size:
            out.append(slice(start, start + size))
            start += size
    if start < count:  # (only when 1 is not among the sizes)
        out.append(slice(start, count))
    return out


class OnnxModel:
    """An ONNX file run by ONNX Runtime: :meth:`run` takes one input array and gives the outputs (the graph's, then
    ``extra_outputs``, tensors inside the graph made outputs too). One session per input shape where the provider
    needs fixed shapes (``fixed``; see the module docs), the first made for ``shape`` right away; thread safe."""

    def __init__(self, path, shape, extra_outputs=(), provider: str | None = None, fixed: bool | None = None):
        self.path = Path(path)
        self.extra_outputs = tuple(extra_outputs)
        self.provider = provider_choice(provider)
        self.fixed = self.provider == "coreml" if fixed is None else bool(fixed)
        self._sessions: dict[tuple, object] = {}
        self._lock = threading.Lock()
        first = self._session(tuple(shape) if self.fixed else None)  # ``shape``: the first input shape expected
        self.input_name = first.get_inputs()[0].name
        self.input_shape = tuple(first.get_inputs()[0].shape)
        self.output_names = [o.name for o in first.get_outputs()]
        self.providers_used = first.get_providers()

    def run(self, x: np.ndarray) -> list[np.ndarray]:
        session = self._session(tuple(x.shape) if self.fixed else None)
        return session.run(self.output_names, {self.input_name: np.ascontiguousarray(x, np.float32)})

    def _session(self, shape):
        with self._lock:
            session = self._sessions.get(shape)
            if session is None:
                session = self._make(shape)
                self._sessions[shape] = session
            return session

    def _source(self, shape):
        """What a session loads: the file itself, or the graph edited in memory (extra outputs, fixed shape)."""
        if not self.extra_outputs and shape is None:
            return str(self.path)
        import onnx

        model = onnx.load(str(self.path))
        if self.extra_outputs:
            known = {o.name for o in model.graph.output}
            made = {name for node in model.graph.node for name in node.output}
            for name in self.extra_outputs:
                if name not in made:
                    raise ValueError(f"{self.path.name} has no tensor {name!r} to make an output")
                if name not in known:
                    model.graph.output.append(onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None))
        if shape is not None:  # every input dimension fixed; the outputs' shapes are left to ONNX Runtime
            initializers = {i.name for i in model.graph.initializer}
            graph_input = next(i for i in model.graph.input if i.name not in initializers)
            dims = graph_input.type.tensor_type.shape.dim
            if len(dims) != len(shape):
                raise ValueError(f"{self.path.name} takes {len(dims)}-d inputs, not {len(shape)}-d")
            for dim, value in zip(dims, shape):
                dim.ClearField("dim_param")
                dim.dim_value = int(value)
            for output in model.graph.output:
                output.type.tensor_type.ClearField("shape")
        return model.SerializeToString()

    def _make(self, shape):
        rt = ort()
        options = rt.SessionOptions()
        options.log_severity_level = 3
        count = threads()
        if count:
            options.intra_op_num_threads = count
        source = self._source(shape)
        chosen = providers(self.provider)
        try:
            return rt.InferenceSession(source, sess_options=options, providers=chosen)
        except Exception as exc:  # CoreML cannot take this graph: the CPU can
            if chosen == [CPU]:
                raise
            warnings.warn(f"{self.path.name}: CoreML failed ({exc}); running on the CPU", RuntimeWarning,
                          stacklevel=2)
            return rt.InferenceSession(source, sess_options=options, providers=[CPU])
