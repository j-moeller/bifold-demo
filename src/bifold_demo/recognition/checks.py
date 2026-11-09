"""Checking every face model before trusting it ("step zero" of docs/research/face_recognition_sota.md, section 4.1).

``bifold-faces check`` runs these on the models in the models folder:

* **The file** (:func:`check_file`): its SHA-256 and size against the registry.
* **The graph** (:func:`check_graph`): the inputs and outputs (names and shapes) and the opset the registry expects,
  only standard ONNX operators (no custom domains), the AdaFace norm tensor where one is declared; then a run on
  random crops: the outputs' shapes, finite values, the same answer twice, the same answer alone and in a batch.
* **Speed** (:func:`speed`): milliseconds per crop at batch 1 and 8 on this machine's execution provider.
* **The colour order** (:func:`colour_order`): mated and non-mated pairs embedded with the input in either colour
  order; the right order separates them far better (d-prime and the equal error rate, :func:`separation`).
* **Published accuracy** (:func:`verification`): the 10-fold verification protocol on InsightFace's test packs
  (``lfw.bin``, ``cfp_fp.bin``, ``agedb_30.bin``: 112 x 112 aligned faces as JPEG bytes and the pair labels), with
  flipped crops added as the papers do. The accuracy must come within about 0.2 points of the published one. The
  packs are Python pickles; :func:`read_pack` reads them with an unpickler that refuses every class and function, so
  only plain data (lists, bytes, booleans) can come out and no code can run (InsightFace's packs are Python 2
  pickles of plain strings; a pack re-saved by Python 3 with pickle protocol 2 stores bytes through a function call
  and is refused: re-save it with protocol 3 or later). Only ``lfw.bin`` can be had from GitHub
  (``leondgarse/Keras_insightface`` release ``v1.0.0``); the others are on Hugging Face, so those checks run where
  it can be reached (``bifold-faces check --pack DIR/cfp_fp.bin``).

**Records.** ``bifold-faces check`` keeps what it found per model in the models folder (``checks/<model id>.json``,
with the file's SHA-256; a later check adds to it, e.g. another pack): :func:`record`, :func:`recorded`. A model with
a ``gate`` in the registry (AuraFace, whose colour order and normalisation are assumed) is used by an engine only
once its record has passed every check and its accuracy on the test packs beats the gate model's on the same packs
(:func:`gate`), as the research asks ("adopt only if it beats SFace").
"""

from __future__ import annotations

import io
import json
import os
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

STANDARD_DOMAINS = ("", "ai.onnx", "ai.onnx.ml")
RECORDS = "checks"  # the folder in the models folder that keeps what bifold-faces check found, a file per model


@dataclass
class Report:
    """What a check found: ``ok`` and the ``lines`` saying why."""

    name: str
    ok: bool = True
    lines: list[str] = field(default_factory=list)
    values: dict = field(default_factory=dict)

    def fail(self, line: str) -> None:
        self.ok = False
        self.lines.append(f"FAIL {line}")

    def note(self, line: str) -> None:
        self.lines.append(line)

    def __str__(self) -> str:
        return "\n".join([f"{self.name}: {'ok' if self.ok else 'FAILED'}", *(f"  {line}" for line in self.lines)])


# -- the test packs -------------------------------------------------------------------------------------------------


class _DataOnly(pickle.Unpickler):
    """Unpickles plain data only: any class or function (the way a pickle runs code) is refused."""

    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"the pack asks for {module}.{name}: refused (only plain data is read)")


def read_pack(path) -> tuple[list[bytes], np.ndarray]:
    """``(images, same)`` of an InsightFace verification pack: the encoded images (two per pair, in order) and
    whether each pair shows one person. Read with :class:`_DataOnly` (see the module docs)."""
    data = Path(path).read_bytes()
    try:
        loaded = _DataOnly(io.BytesIO(data), encoding="bytes").load()
    except pickle.UnpicklingError as exc:
        raise ValueError(f"{path}: {exc}") from exc
    if not (isinstance(loaded, (tuple, list)) and len(loaded) == 2):
        raise ValueError(f"{path}: not a verification pack (expected the images and the pair labels)")
    images, same = loaded
    if not all(isinstance(b, (bytes, bytearray)) for b in images):
        raise ValueError(f"{path}: the images are not encoded bytes")
    same = np.asarray(same, bool)
    if len(images) != 2 * len(same):
        raise ValueError(f"{path}: {len(images)} images for {len(same)} pairs")
    return list(images), same


def decode(images, size: int = 112) -> np.ndarray:
    """The encoded ``images`` as ``(N, size, size, 3)`` BGR crops (resized if a pack holds another size)."""
    out = np.zeros((len(images), size, size, 3), np.uint8)
    for i, encoded in enumerate(images):
        image = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"image {i} of the pack cannot be decoded")
        out[i] = image if image.shape[:2] == (size, size) else cv2.resize(image, (size, size),
                                                                          interpolation=cv2.INTER_AREA)
    return out


# -- scores ---------------------------------------------------------------------------------------------------------


def separation(mated, nonmated) -> dict:
    """How well mated scores (pairs of one person) stand apart from non-mated ones: ``d_prime`` (the difference of
    the means over the root mean variance), the equal error rate ``eer`` and its threshold, and both sets' means."""
    mated, nonmated = np.asarray(mated, float), np.asarray(nonmated, float)
    if not len(mated) or not len(nonmated):
        raise ValueError("separation needs mated and non-mated scores")
    spread = np.sqrt((mated.var() + nonmated.var()) / 2.0)
    d_prime = float((mated.mean() - nonmated.mean()) / spread) if spread > 0 else float("inf")
    thresholds = np.unique(np.r_[mated, nonmated])
    false_reject = np.searchsorted(np.sort(mated), thresholds, side="left") / len(mated)
    false_accept = 1.0 - np.searchsorted(np.sort(nonmated), thresholds, side="left") / len(nonmated)
    k = int(np.argmin(np.abs(false_reject - false_accept)))
    eer = float((false_reject[k] + false_accept[k]) / 2.0)
    return {"d_prime": d_prime, "eer": eer, "eer_threshold": float(thresholds[k]), "mated_mean": float(mated.mean()),
            "nonmated_mean": float(nonmated.mean()), "mated": len(mated), "nonmated": len(nonmated)}


def threshold_at(nonmated, false_match_rate: float) -> float:
    """The smallest score threshold at which at most ``false_match_rate`` of the non-mated scores reach it."""
    scores = np.sort(np.asarray(nonmated, float))[::-1]
    allowed = int(np.floor(false_match_rate * len(scores)))
    return float(np.nextafter(scores[allowed], np.inf)) if allowed < len(scores) else float(scores[-1])


def verification(scores, same, folds: int = 10) -> dict:
    """The verification benchmarks' protocol: the pairs cut into ``folds`` in order; for each, the threshold that is
    most accurate on the other folds is used on it. ``accuracy`` (mean, per cent) and ``std`` over the folds, and the
    threshold most accurate on all pairs."""
    scores, same = np.asarray(scores, float), np.asarray(same, bool)

    def best(idx):
        """The threshold (half way between two sorted scores) that classifies the pairs ``idx`` best."""
        order = np.argsort(scores[idx], kind="stable")
        s, y = scores[idx][order], same[idx][order]
        correct = np.r_[0, np.cumsum(~y)] + np.r_[np.cumsum(y[::-1])[::-1], 0]  # cut before position i
        i = int(np.argmax(correct))
        if i == 0:
            return s[0] - 1e-6
        return s[-1] + 1e-6 if i == len(s) else (s[i - 1] + s[i]) / 2.0

    parts = np.array_split(np.arange(len(scores)), max(1, min(folds, len(scores) // 2)))
    accuracies = []
    for k, test in enumerate(parts):
        train = np.concatenate([p for j, p in enumerate(parts) if j != k])
        t = best(train)
        accuracies.append(np.mean((scores[test] >= t) == same[test]))
    return {"accuracy": float(np.mean(accuracies) * 100), "std": float(np.std(accuracies) * 100),
            "threshold": float(best(np.arange(len(scores)))), "pairs": len(scores)}


def pair_scores(vectors: np.ndarray) -> np.ndarray:
    """Cosine similarities of consecutive embeddings (pairs 0-1, 2-3, ...)."""
    return np.sum(vectors[0::2] * vectors[1::2], axis=1)


def timed(function, *args, repeat: int = 5) -> float:
    """The median seconds of ``function(*args)`` over ``repeat`` runs, after one run to warm up."""
    function(*args)
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        function(*args)
        times.append(time.perf_counter() - start)
    return float(np.median(times))


# -- the checks -----------------------------------------------------------------------------------------------------


def check_file(spec, folder) -> Report:
    """The file of ``spec`` in ``folder``: there, of the registry's size and SHA-256."""
    from .. import downloads

    report = Report(f"{spec.id}: file")
    path = Path(folder).expanduser() / spec.file.name
    if not path.is_file():
        report.fail(f"{path} is missing (bifold-faces fetch --model {spec.id})")
        return report
    size = path.stat().st_size
    if spec.file.size is not None and size != spec.file.size:
        report.fail(f"{size} bytes, not {spec.file.size}")
    digest = downloads.sha256(path)
    report.values.update(size=size, sha256=digest)
    if digest == spec.file.sha256:
        report.note(f"SHA-256 {digest[:16]}... as registered ({size / 1e6:.1f} MB)")
    else:
        report.fail(f"SHA-256 {digest} is not the registered {spec.file.sha256}")
    return report


def _dims(value) -> tuple:
    return tuple(d.dim_param or d.dim_value for d in value.type.tensor_type.shape.dim)


def _same_shape(found: tuple, expected: tuple) -> bool:
    """Shapes agree: equal fixed sizes, and a free dimension (a name) wherever one is expected."""
    if len(found) != len(expected):
        return False
    return all((isinstance(e, str) and isinstance(f, str)) or f == e for f, e in zip(found, expected))


def check_graph(spec, path) -> Report:
    """The graph of ``spec`` at ``path`` (see the module docs): inputs, outputs, opset and operators, read without
    running anything; then a run on random input."""
    import onnx

    report = Report(f"{spec.id}: graph")
    model = onnx.load(str(path))
    graph = model.graph
    opset = next((o.version for o in model.opset_import if o.domain in ("", "ai.onnx")), 0)
    report.values["opset"] = opset
    if spec.opset and opset != spec.opset:
        report.fail(f"opset {opset}, not {spec.opset}")
    nodes = list(graph.node) + [n for f in model.functions for n in f.node]
    domains = sorted({n.domain for n in nodes} - set(STANDARD_DOMAINS))
    if domains or model.functions:
        report.fail(f"custom operators: {domains or [f.name for f in model.functions]}")
    initializers = {i.name for i in graph.initializer}
    inputs = [(v.name, _dims(v)) for v in graph.input if v.name not in initializers]
    outputs = [(v.name, _dims(v)) for v in graph.output]
    report.values.update(inputs=inputs, outputs=outputs, nodes=len(graph.node))
    for kind, found, expected in (("input", inputs, spec.inputs), ("output", outputs, spec.outputs)):
        if not expected:
            continue
        if [n for n, _ in found] != [t.name for t in expected] or not all(
                _same_shape(f, t.shape) for (_, f), t in zip(found, expected)):
            report.fail(f"{kind}s {found}, not {[(t.name, t.shape) for t in expected]}")
    if spec.norm_output and spec.norm_output not in {o for n in graph.node for o in n.output}:
        report.fail(f"no tensor {spec.norm_output} (the embedding's length) in the graph")
    if report.ok:
        report.note(f"opset {opset}, {len(graph.node)} standard operators; inputs {inputs}; outputs {len(outputs)}: "
                    f"{', '.join(n for n, _ in outputs[:3])}{'...' if len(outputs) > 3 else ''}")
    _check_run(spec, path, report)
    return report


def _check_run(spec, path, report: Report) -> None:
    """Run the model on random input: output shapes, finite values, the same twice, alone as in a batch."""
    rng = np.random.default_rng(0)
    try:
        if spec.task in ("detector", "people"):
            if spec.task == "people":
                from ..detection.yolox import YoloxDetector

                detector = YoloxDetector(spec.id, models_folder=Path(path).parent, download=False, progress=None)
            else:
                from .detect import ScrfdDetector, YuNetDetector

                detector = YuNetDetector(path, spec) if spec.form == "opencv" else ScrfdDetector(path, spec)
            noise = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
            first, second = detector.detect(noise), detector.detect(noise)
            if [f.box for f in first] != [f.box for f in second]:
                report.fail("two runs on the same image differ")
            else:
                report.note(f"runs: {len(first)} {'faces' if spec.task == 'detector' else 'objects'} in noise, the "
                            f"same twice")
            return
        crops = rng.integers(0, 256, (9, 112, 112, 3), dtype=np.uint8)
        if spec.task == "quality":
            from .quality import FiqaModel

            model = FiqaModel(path, spec)
            out = model.score(crops)
            again, alone = model.score(crops), model.score(crops[4:5])
            shape = out.shape == (9,)
        else:
            from .embed import embedder

            model = embedder(spec, path)
            result = model.embed(crops)
            out = result.vectors
            again, alone = model.embed(crops).vectors, model.embed(crops[4:5]).vectors
            shape = out.shape == (9, spec.dim) and np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-4)
            report.values["norm_range"] = [float(result.norms.min()), float(result.norms.max())]
        if not shape or not np.isfinite(out).all():
            report.fail(f"outputs of shape {out.shape} (finite: {bool(np.isfinite(out).all())})")
        elif not np.allclose(out, again, atol=1e-6):
            report.fail("two runs on the same crops differ")
        elif not np.allclose(out[4:5], alone, atol=1e-4):
            report.fail(f"a crop alone gives another answer than in a batch (by {np.abs(out[4:5] - alone).max():.2g})")
        else:
            report.note("runs: outputs of the right shape, finite, the same twice, the same alone as in a batch")
    except Exception as exc:  # noqa: BLE001 - any failure to run is a failed check
        report.fail(f"cannot run: {type(exc).__name__}: {exc}")


def speed(spec, path) -> dict:
    """Milliseconds per crop at batch 1 and 8 (embedders, quality) or per image at canvases of 160 and 640 pixels
    (detectors), on this machine's execution provider."""
    rng = np.random.default_rng(0)
    if spec.task == "people":
        from ..detection.yolox import YoloxDetector

        detector = YoloxDetector(spec.id, models_folder=Path(path).parent, download=False, progress=None)
        image = rng.integers(0, 256, (720, 1280, 3), dtype=np.uint8)
        return {"1280x720 frame": timed(detector.detect, image) * 1000}
    if spec.task == "detector":
        from .detect import ScrfdDetector, YuNetDetector

        detector = YuNetDetector(path, spec) if spec.form == "opencv" else ScrfdDetector(path, spec)
        image = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
        return {f"canvas {c}": timed(detector.detect, image, 0.0, c) * 1000 for c in (160, 640)}
    if spec.task == "quality":
        from .quality import FiqaModel

        run = FiqaModel(path, spec).score
    else:
        from .embed import embedder

        run = embedder(spec, path).embed
    crops = rng.integers(0, 256, (8, 112, 112, 3), dtype=np.uint8)
    return {"batch 1": timed(run, crops[:1], repeat=10) * 1000, "batch 8 (per crop)": timed(run, crops) * 1000 / 8}


def labelled_faces(folder, detector) -> tuple[np.ndarray, np.ndarray]:
    """Aligned crops of the biggest face in each photo of a faces folder (a folder per person, as ``faces/``) and
    the person each shows; identical files counted once."""
    import hashlib

    from .align import align
    from .gallery import _person, _photos

    folder = Path(folder).expanduser()
    crops, labels, seen = [], [], set()
    for path in _photos(folder):
        name = _person(path, folder)
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if name is None or digest in seen:
            continue
        seen.add(digest)
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        faces = detector.detect(image) if image is not None else []
        if faces:
            crops.append(align(image, faces[0].landmarks).crop)
            labels.append(name)
    if not crops:
        return np.zeros((0, 112, 112, 3), np.uint8), np.array([], str)
    return np.array(crops), np.array(labels)


def colour_order(spec, path, crops: np.ndarray, labels: np.ndarray) -> Report:
    """Mated and non-mated scores of ``crops`` (``labels``: who each shows) with the input in the registry's colour
    order and in the other one: the registry's must separate them better."""
    from .embed import OnnxEmbedder

    report = Report(f"{spec.id}: colour order")
    same = labels[:, None] == labels[None, :]
    upper = np.triu(np.ones_like(same), 1).astype(bool)
    if not (same & upper).any() or not (~same & upper).any():
        report.note("skipped: needs at least two photos of one person and two people")
        return report
    found = {}
    for bgr in (spec.bgr, not spec.bgr):
        vectors = OnnxEmbedder(path, spec, bgr=bgr).embed(crops).vectors
        scores = vectors @ vectors.T
        found["BGR" if bgr else "RGB"] = separation(scores[same & upper], scores[~same & upper])
    mine, other = ("BGR", "RGB") if spec.bgr else ("RGB", "BGR")
    report.values.update(found)
    line = ", ".join(f"{k}: d' {v['d_prime']:.2f}, EER {v['eer']:.1%}" for k, v in found.items())
    if found[mine]["d_prime"] > found[other]["d_prime"]:
        report.note(f"{mine} (registered) separates better: {line}")
    else:
        report.fail(f"{other} separates better than the registered {mine}: {line}")
    return report


def published_accuracy(spec, path, pack) -> Report:
    """The 10-fold verification accuracy of the embedder ``spec`` on a test ``pack`` (with flipped crops) against
    the published one (the pack's name, e.g. ``lfw.bin``, says which)."""
    from .embed import embedder

    names = {"lfw": "LFW", "cfp_fp": "CFP-FP", "agedb_30": "AgeDB-30", "calfw": "CALFW", "cplfw": "CPLFW"}
    benchmark = names.get(Path(pack).stem, Path(pack).stem)
    report = Report(f"{spec.id}: {benchmark}")
    images, same = read_pack(pack)
    vectors = embedder(spec, path).embed(decode(images), flip=True).vectors
    result = verification(pair_scores(vectors), same)
    report.values.update(result)
    published = spec.published.get(benchmark)
    line = f"{result['accuracy']:.2f} % +- {result['std']:.2f} over {result['pairs']} pairs"
    if published is None:
        report.note(f"{line} (nothing published to compare)")
    elif result["accuracy"] >= published - 0.2:
        report.note(f"{line}, published {published:.2f} %: as published")
    elif result["accuracy"] >= published - 0.5:
        report.note(f"{line}, published {published:.2f} %: a little below (check the colour order and alignment)")
    else:
        report.fail(f"{line}, published {published:.2f} %: well below")
    return report


# -- records and gates ----------------------------------------------------------------------------------------------


def record(spec, folder, reports) -> Path:
    """Keep ``reports``, what ``bifold-faces check`` found for ``spec``, in the models ``folder`` (see the module
    docs), merged with what earlier checks of the same file found; the record's path."""
    path = Path(folder) / RECORDS / f"{spec.id}.json"
    found = recorded(spec, folder) or {"model": spec.id, "sha256": spec.file.sha256, "reports": {}}
    found["reports"].update({r.name: {"ok": r.ok, "lines": r.lines, "values": r.values} for r in reports})
    found["ok"] = all(r["ok"] for r in found["reports"].values())
    found["when"] = time.strftime("%Y-%m-%d %H:%M:%S")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(found, indent=1, default=str))
    os.replace(tmp, path)
    return path


def recorded(spec, folder) -> dict | None:
    """The record of ``spec``'s checks in the models ``folder``; ``None`` when there is none, or it is of another
    file (another SHA-256)."""
    try:
        found = json.loads((Path(folder) / RECORDS / f"{spec.id}.json").read_text())
    except (OSError, ValueError):
        return None
    return found if isinstance(found, dict) and found.get("sha256") == spec.file.sha256 else None


def _accuracies(found: dict | None) -> dict[str, float]:
    """``{benchmark: accuracy}`` of the test packs in a record."""
    out = {}
    for name, report in (found or {}).get("reports", {}).items():
        values = report.get("values") or {}
        if "accuracy" in values:
            out[name.split(": ", 1)[-1]] = float(values["accuracy"])
    return out


def gate(spec, folder) -> str | None:
    """Why an engine may not use ``spec`` yet (``None``: it may; see the module docs): a model with a ``gate`` needs
    a record in the models ``folder`` that passed every check, with a mean accuracy on the test packs it shares with
    the gate model's record above that model's."""
    if not spec.gate:
        return None
    from . import registry

    other = registry.model(spec.gate)
    how = f"bifold-faces check --model {spec.id} {other.id} --pack lfw.bin"
    mine = recorded(spec, folder)
    if mine is None:
        return f"not used until a recorded check shows it beats {other.title} ({how})"
    if not mine.get("ok"):
        failed = [name for name, r in mine["reports"].items() if not r["ok"]]
        return f"not used: its recorded check failed ({', '.join(failed)})"
    ours, theirs = _accuracies(mine), _accuracies(recorded(other, folder))
    shared = sorted(set(ours) & set(theirs))
    if not shared:
        return f"not used until a recorded check on a test pack shows it beats {other.title} ({how})"
    a, b = (float(np.mean([found[k] for k in shared])) for found in (ours, theirs))
    if a <= b:
        return f"not used: {a:.2f} % against {other.title}'s {b:.2f} % on {', '.join(shared)} (recorded checks)"
    return None
