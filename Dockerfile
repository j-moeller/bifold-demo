# The model side of `bifold-morph run --isolate`: PyTorch, the networks and everything that opens model files.
#   docker build -t bifold-morph-worker .
# It is started by the app with no network, a read-only file system and the models folder mounted read-only
# (see src/bifold_morph/isolated.py). Docker on macOS cannot use Apple's GPU, so in there it runs on the CPU.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
# the CPU build of PyTorch: no CUDA libraries (several GB) in the image
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md /src/
COPY src /src/src
# opencv-python (a dependency) and opencv-python-headless share the cv2 folder: removing one deletes it, so the
# headless one (no GUI libraries needed) goes in afresh after the other is gone
RUN pip install --no-cache-dir "/src[models]" \
    && pip uninstall -y opencv-python \
    && pip install --no-cache-dir --force-reinstall --no-deps opencv-python-headless \
    && python -c "import cv2, torch, timm, gdown; cv2.FaceDetectorYN; print('ok', cv2.__version__, torch.__version__)"

RUN useradd --create-home --uid 10001 morph
USER morph
ENV BIFOLD_MORPH_MODELS=/models PYTHONDONTWRITEBYTECODE=1
ENTRYPOINT ["bifold-morph"]
CMD ["serve"]
