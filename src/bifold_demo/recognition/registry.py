"""The face models the recognition can use, and the profiles that put them together.

Every model the recognition runs is described here (:class:`ModelSpec`): its file (where it comes from, its size and
SHA-256; InsightFace's files are members of zip packs), the form it is run in, what its input must look like (size,
colour order, mean and spread), what comes out (the embedding's length), its licence, where the file comes from and
how it was made, the inputs and outputs its graph must have (checked by ``bifold-faces check``), and the accuracy its
authors published. Nothing is downloaded or run that is not listed here, with its checksum.

A **profile** chooses one model per stage (:class:`Profile`): a face detector, an embedder (with fallbacks, tried in
order) and a quality model. Three profiles stand behind one switch (``--face-profile`` or ``BIFOLD_FACE_PROFILE``;
:func:`profile_name`):

* ``best`` (the default): SCRFD-10G and AdaFace IR-101 trained on WebFace12M, the most accurate recogniser on
  low-quality faces whose weights we can run (a community ONNX export, checked against its published accuracy by
  ``bifold-faces check``); InsightFace's ``buffalo_l`` ResNet-50 is the fallback. Research-only weights.
* ``licensed``: SCRFD-10G and ``buffalo_l`` ResNet-50, InsightFace's own files, for which InsightFace sells
  commercial licences. Without such a licence they are research-only too.
* ``commercial``: YuNet with AuraFace-v1 (Apache-2.0; on Hugging Face only, so it is fetched where Hugging Face can
  be reached), else SFace (Apache-2.0), with the geometric quality measures only.

"Commercially usable" here means the weights' licence allows it. The training data (WIDER FACE for the detectors,
web-scraped faces for most recognisers) and patents are further questions that need a legal review before anything
ships (see docs/research/face_recognition_sota.md, section 1).

Embeddings of different models cannot be compared, so everything made from embeddings is kept per model (the
gallery's caches are ``faces/.embeddings.<model id>.npz``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from ..downloads import ModelFile, Source, member_of
from . import models as files

PROFILE_ENV = "BIFOLD_FACE_PROFILE"
DEFAULT_PROFILE = "best"

GITHUB = "https://github.com/"
INSIGHTFACE = GITHUB + "deepinsight/insightface/releases/download/v0.7/"
BUFFALO_L = Source(INSIGHTFACE + "buffalo_l.zip",
                   sha256="80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f", size=288_621_354)
ANTELOPEV2 = Source(INSIGHTFACE + "antelopev2.zip",
                    sha256="8e182f14fc6e80b3bfa375b33eb6cff7ee05d8ef7633e738d1c89021dcf0c5c5", size=360_662_982)
ADAFACE = GITHUB + "yakhyo/adaface-onnx/releases/download/weights/"
UNIFACE = GITHUB + "yakhyo/uniface/releases/download/weights/"
FIQA = GITHUB + "yakhyo/face-image-quality-assessment/releases/download/weights/"
HUGGING_FACE = "https://huggingface.co/"

INSIGHTFACE_LICENCE = ("InsightFace's pretrained models: non-commercial research use only; InsightFace sells "
                       "commercial licences (recognition-oss-pack@insightface.ai)")


@dataclass(frozen=True)
class Tensor:
    """An input or output of a graph: its name and shape (a string for a free dimension)."""

    name: str
    shape: tuple


@dataclass(frozen=True)
class ModelSpec:
    id: str  # short and stable: the gallery's caches are named after it
    title: str  # for people
    task: str  # "detector", "embedder" or "quality"
    file: ModelFile
    form: str  # "onnxruntime" (run by ONNX Runtime) or "opencv" (run by OpenCV's own face classes)
    size: tuple[int, int]  # the input image the model sees, (width, height) (a detector's default canvas)
    bgr: bool  # the colour order of the input: True for blue-green-red, False for red-green-blue
    mean: float  # the input is (pixel - mean) / std
    std: float
    dim: int  # what comes out: the embedding's length (1 for a quality score, 0 for a detector)
    licence: str  # the weights' licence, as stated by their publisher
    commercial: bool  # whether the weights' licence allows commercial use (training data and patents aside)
    data: str  # what it was trained on
    origin: str  # who made the file and how; how it was checked here
    inputs: tuple[Tensor, ...] = ()  # what the graph must take and give (checked by ``bifold-faces check``)
    outputs: tuple[Tensor, ...] = ()
    opset: int = 0
    norm_output: str | None = None  # a tensor inside the graph holding the embedding's length before it was
    # normalised in the graph (made an extra output)
    norm_quality: bool = False  # whether that length says how good the face is (AdaFace learns it so)
    published: dict = field(default_factory=dict)  # accuracy published by the authors (per cent)
    needs: str = ""  # a host the file can only be fetched from, said when the download fails
    optional: bool = False  # in no profile: fetched only on request

    @property
    def profiles(self) -> tuple[str, ...]:
        """The profiles using this model."""
        return tuple(name for name, p in PROFILES.items() if self.id in p.models())


@dataclass(frozen=True)
class Profile:
    name: str
    detector: str  # model ids
    embedders: tuple[str, ...]  # in order of preference: the first one available is used
    quality: str | None  # a quality model, or None: the geometric measures only
    summary: str

    def models(self) -> tuple[str, ...]:
        return (self.detector, *self.embedders, *((self.quality,) if self.quality else ()))


SCRFD = ModelSpec(
    "scrfd_10g_kps", "SCRFD-10G with five landmarks (InsightFace, det_10g)", "detector",
    member_of(BUFFALO_L, "det_10g.onnx", "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91",
              16_923_827, mirrors=(ADAFACE + "det_10g.onnx",)),
    "onnxruntime", (640, 640), bgr=False, mean=127.5, std=128.0, dim=0,
    licence=INSIGHTFACE_LICENCE, commercial=False, data="WIDER FACE",
    origin="InsightFace's buffalo_l pack (GitHub release v0.7); the same file (same SHA-256) is mirrored by UniFace "
           "and the AdaFace export",
    inputs=(Tensor("input.1", (1, 3, "?", "?")),),
    outputs=tuple(Tensor(name, (n, c)) for name, n, c in (
        ("448", 12800, 1), ("471", 3200, 1), ("494", 800, 1), ("451", 12800, 4), ("474", 3200, 4),
        ("497", 800, 4), ("454", 12800, 10), ("477", 3200, 10), ("500", 800, 10))),
    opset=11, published={"WIDER FACE easy": 95.16, "WIDER FACE medium": 93.87, "WIDER FACE hard": 83.05})

YUNET = ModelSpec(
    "yunet_2023mar", "YuNet 2023mar (OpenCV Zoo)", "detector", files.DETECTOR,
    "opencv", (640, 640), bgr=True, mean=0.0, std=1.0, dim=0,
    licence="MIT", commercial=True, data="WIDER FACE",
    origin="OpenCV Zoo (GitHub LFS), run by OpenCV's FaceDetectorYN",
    inputs=(Tensor("input", (1, 3, 640, 640)),),
    outputs=tuple(Tensor(f"{kind}_{stride}", (1, (640 // stride) ** 2, width))
                  for kind, width in (("cls", 1), ("obj", 1), ("bbox", 4), ("kps", 10)) for stride in (8, 16, 32)),
    opset=11, published={"WIDER FACE easy": 88.4, "WIDER FACE medium": 86.6, "WIDER FACE hard": 75.0})

ADAFACE_IR101 = ModelSpec(
    "adaface_ir101_wf12m", "AdaFace IR-101 (WebFace12M)", "embedder",
    ModelFile("adaface_ir_101.onnx", ADAFACE + "adaface_ir_101.onnx",
              "f2eb07d03de0af560a82e1214df799fec5e09375d43521e2868f9dc387e5a43e", 260_704_652),
    "onnxruntime", (112, 112), bgr=True, mean=127.5, std=127.5, dim=512,
    licence="MIT (the AdaFace repository and the export's); no separate terms for the weights",
    commercial=False, data="WebFace12M (research-only terms); a US patent (12,525,057) describes AdaFace's "
                           "quality-adaptive margin",
    origin="a community ONNX export of the official weights by Yakhyokhuja Valikhujaev (yakhyo/adaface-onnx, "
           "GitHub release 'weights'; the export script is not published); the same SHA-256 is pinned by his UniFace "
           "toolkit. Checked here: standard operators only, accuracy on LFW, colour order (bifold-faces check)",
    inputs=(Tensor("input", ("batch_size", 3, 112, 112)),),
    outputs=(Tensor("output", ("batch_size", 512)),),
    opset=16, norm_output="/model/ReduceL2_output_0", norm_quality=True,
    published={"LFW": 99.82, "CFP-FP": 99.26, "AgeDB-30": 98.00, "IJB-C TAR@1e-4": 97.66,
               "TinyFace rank-1": 72.29})

BUFFALO_L_R50 = ModelSpec(
    "buffalo_l_r50", "ArcFace ResNet-50 (WebFace600K; InsightFace buffalo_l, w600k_r50)", "embedder",
    member_of(BUFFALO_L, "w600k_r50.onnx", "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43",
              174_383_860, mirrors=(UNIFACE + "w600k_r50.onnx",)),
    "onnxruntime", (112, 112), bgr=False, mean=127.5, std=127.5, dim=512,
    licence=INSIGHTFACE_LICENCE, commercial=False, data="WebFace600K (a WebFace260M subset; research-only terms)",
    origin="InsightFace's buffalo_l pack (GitHub release v0.7); the same file (same SHA-256) is mirrored by UniFace",
    inputs=(Tensor("input.1", ("None", 3, 112, 112)),), outputs=(Tensor("683", (1, 512)),),
    opset=11, published={"LFW": 99.83, "CFP-FP": 99.33, "AgeDB-30": 98.23, "IJB-C TAR@1e-4": 97.25})

ANTELOPEV2_R100 = ModelSpec(
    "antelopev2_r100", "ArcFace ResNet-100 (Glint360K; InsightFace antelopev2, glintr100)", "embedder",
    member_of(ANTELOPEV2, "antelopev2_glintr100.onnx",
              "4ab1d6435d639628a6f3e5008dd4f929edf4c4124b1a7169e1048f9fef534cdf", 260_665_334,
              member="antelopev2/glintr100.onnx"),
    "onnxruntime", (112, 112), bgr=False, mean=127.5, std=127.5, dim=512,
    licence=INSIGHTFACE_LICENCE, commercial=False, data="Glint360K (non-commercial research terms)",
    origin="InsightFace's antelopev2 pack (GitHub release v0.7)",
    inputs=(Tensor("input.1", ("None", 3, 112, 112)),), outputs=(Tensor("1333", (1, 512)),),
    opset=11, published={"IJB-C TAR@1e-4": 97.5}, optional=True)

SFACE = ModelSpec(
    "sface_2021dec", "SFace 2021dec (OpenCV Zoo)", "embedder", files.RECOGNIZER,
    "opencv", (112, 112), bgr=True, mean=0.0, std=1.0, dim=128,
    licence="Apache-2.0", commercial=True, data="not stated (the paper used CASIA-WebFace, VGGFace2 and MS1M)",
    origin="OpenCV Zoo (GitHub LFS), run by OpenCV's FaceRecognizerSF (its graph normalises its input itself)",
    inputs=(Tensor("data", (1, 3, 112, 112)),), outputs=(Tensor("fc1", (1, 128)),),
    opset=11, published={"LFW": 99.40})

AURAFACE = ModelSpec(
    "auraface_v1", "AuraFace-v1 (fal; ResNet-100)", "embedder",
    ModelFile("auraface_v1_glintr100.onnx", HUGGING_FACE + "fal/AuraFace-v1/resolve/main/glintr100.onnx",
              "a7933ea5330113b01c9b60351d8f4c33003f145d8470ac5f0e52ee2effe25c60", None),
    "onnxruntime", (112, 112), bgr=False, mean=127.5, std=127.5, dim=512,
    licence="Apache-2.0", commercial=True,
    data="'a commercial dataset comprising face images from various sources' (not described further)",
    origin="fal's Hugging Face repository (glintr100.onnx; not reachable from every network). Its SHA-256 is the "
           "one Hugging Face lists; its size is not known in advance. Colour order and normalisation are assumed "
           "to be InsightFace's (it is run by InsightFace's library) until bifold-faces check confirms them",
    published={"LFW": 99.65, "CFP-FP": 95.19, "AgeDB-30": 96.10, "CALFW": 94.70, "CPLFW": 90.93},
    needs="huggingface.co")

EDIFFIQA_T = ModelSpec(
    "ediffiqa_t", "eDifFIQA-T (face image quality, MobileFaceNet)", "quality",
    ModelFile("ediffiqa_t.onnx", FIQA + "ediffiqa_t.onnx",
              "7a83be63e6583ec5800fa0762e219b65b4b9f1721c4d210bfe2f16c0478832bb", 6_901_979),
    "onnxruntime", (112, 112), bgr=False, mean=127.5, std=127.5, dim=1,
    licence="MIT (the eDifFIQA code and the export); no separate terms for the weights", commercial=False,
    data="quality labels from VGGFace2 and a CosFace ResNet-100 (research data)",
    origin="a community ONNX export of the official weights (yakhyo/face-image-quality-assessment, release "
           "'weights'); the same SHA-256 is pinned by UniFace",
    inputs=(Tensor("input", ("batch_size", 3, 112, 112)),), outputs=(Tensor("quality", ("batch_size", 1)),),
    opset=20)

MODELS: dict[str, ModelSpec] = {m.id: m for m in (SCRFD, YUNET, ADAFACE_IR101, BUFFALO_L_R50, ANTELOPEV2_R100,
                                                  SFACE, AURAFACE, EDIFFIQA_T)}

PROFILES: dict[str, Profile] = {p.name: p for p in (
    Profile("best", SCRFD.id, (ADAFACE_IR101.id, BUFFALO_L_R50.id), EDIFFIQA_T.id,
            "the most accurate: SCRFD-10G and AdaFace IR-101 (research-only weights)"),
    Profile("licensed", SCRFD.id, (BUFFALO_L_R50.id,), EDIFFIQA_T.id,
            "InsightFace's own SCRFD-10G and ResNet-50, commercially usable with a licence bought from InsightFace"),
    Profile("commercial", YUNET.id, (AURAFACE.id, SFACE.id), None,
            "weights under commercial licences: YuNet (MIT) with AuraFace-v1 (Apache-2.0) where Hugging Face can be "
            "reached, else SFace (Apache-2.0)"),
)}


def model(model_id: str) -> ModelSpec:
    try:
        return MODELS[model_id]
    except KeyError:
        raise ValueError(f"unknown face model {model_id!r} (known: {', '.join(MODELS)})") from None


def profile_name(name: str | None = None) -> str:
    """``name``, else ``$BIFOLD_FACE_PROFILE``, else ``best``; checked."""
    name = (name or os.environ.get(PROFILE_ENV) or DEFAULT_PROFILE).strip().lower()
    if name not in PROFILES:
        raise ValueError(f"unknown face profile {name!r} (choose from: {', '.join(PROFILES)})")
    return name


def profile(name: str | None = None) -> Profile:
    """The profile ``name`` (see :func:`profile_name`)."""
    return PROFILES[profile_name(name)]


def path(spec: ModelSpec, folder=None, download: bool = True, progress=print):
    """The file of ``spec`` in the models folder, downloaded first if it is missing."""
    return files.ensure(spec.file, folder, download, progress)


def fetch(name: str | None = None, folder=None, progress=print, ids=None) -> dict[str, object]:
    """Download the models of the profile ``name`` (or the models ``ids``) into the models folder: ``{model id: its
    path, or the error}`` (a zip pack is downloaded once for all its files). A model with a fallback after it may
    fail (AuraFace where Hugging Face cannot be reached); :class:`OSError` when a stage has no model at all."""
    from .. import downloads

    if ids is None:
        chosen = profile(name)
        stages = [(chosen.detector,), chosen.embedders] + ([(chosen.quality,)] if chosen.quality else [])
    else:
        stages = [(model(i).id,) for i in ids]
    wanted = list({i: model(i) for stage in stages for i in stage}.values())
    folder = files.models_dir(folder)
    errors: dict[str, list[str]] = {}
    try:
        downloads.ensure_many([s.file for s in wanted], folder, True, progress, tag="faces")
    except downloads.DownloadError as exc:
        errors = exc.errors
    out: dict[str, object] = {}
    for spec in wanted:
        found = downloads.present(spec.file, folder)
        if found is not None:
            out[spec.id] = found
        else:
            why = " / ".join(errors.get(spec.file.name, [])) or "not downloaded"
            out[spec.id] = OSError(f"{spec.id}: {why}" + (f" ({spec.needs} must be reachable)" if spec.needs else ""))
    for stage in stages:
        if all(isinstance(out[i], Exception) for i in stage):
            raise OSError("; ".join(str(out[i]) for i in stage))
    return out
