"""The face models folder, and the two OpenCV Zoo files the demo has always used.

Models are downloaded once into the models folder (``--models``, else ``$BIFOLD_FACE_MODELS``, else ``models/`` in
the working directory, next to the YOLO weights) and checked by size and SHA-256 (:mod:`bifold_demo.downloads`).
:mod:`.registry` lists every face model and the profiles; this module keeps the folder and the files of the OpenCV
Zoo, which other parts of the demo use directly (the head textures use YuNet to find small faces):

* ``face_detection_yunet_2023mar.onnx`` (YuNet, MIT licence): finds faces and five landmarks (eyes, nose tip, mouth
  corners); OpenCV runs it;
* ``face_recognition_sface_2021dec.onnx`` (SFace, Apache 2.0): a 128-number embedding of an aligned face, 99.6 % on
  LFW; the commercial profile's recogniser where AuraFace cannot be fetched.
"""

from __future__ import annotations

import os
from pathlib import Path

from .. import downloads
from ..downloads import ModelFile

BASE_URL = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
FETCH_HINT = "bifold-faces fetch"

DETECTOR = ModelFile("face_detection_yunet_2023mar.onnx",
                     BASE_URL + "face_detection_yunet/face_detection_yunet_2023mar.onnx",
                     "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4", 232_589)
RECOGNIZER = ModelFile("face_recognition_sface_2021dec.onnx",
                       BASE_URL + "face_recognition_sface/face_recognition_sface_2021dec.onnx",
                       "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79", 38_696_353)
FILES = (DETECTOR, RECOGNIZER)


def models_dir(folder=None) -> Path:
    """``folder``, else ``$BIFOLD_FACE_MODELS``, else ``models/`` in the working directory (as for YOLO)."""
    return Path(folder or os.environ.get("BIFOLD_FACE_MODELS") or "models").expanduser()


def ensure(model: ModelFile, folder=None, download: bool = True, progress=print) -> Path:
    """The path of ``model`` in the models folder, downloading it first if it is missing (or damaged)."""
    return downloads.ensure(model, models_dir(folder), download, progress, fetch_hint=FETCH_HINT, tag="faces")


def ensure_all(folder=None, download: bool = True, progress=print) -> dict[str, Path]:
    """YuNet and SFace (:data:`FILES`)."""
    return downloads.ensure_many(FILES, models_dir(folder), download, progress, fetch_hint=FETCH_HINT, tag="faces")
