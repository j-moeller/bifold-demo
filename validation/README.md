# Validation

- `check_conversion.py MODELS`: every ONNX → PyTorch conversion against ONNX Runtime (step 0).
- `validate.py`: the gender and age change on photos of real people, read by judges that did not make it.
- `experiments/`: the research scripts behind each table in [docs/experiments.md](../docs/experiments.md), as
  they were run (arguments: the repository's `src`, the models folder, the photos folder, an output folder).

Photos and weights are not in the repository. The photos used are InsightFace's sample `t1.jpg`, OpenCV's
`lena.jpg`, and the 61 portraits in DeepFace's `tests/unit/dataset`, all from those projects' GitHub repositories.
