# Face recognition as the core: plan and progress

**The change (1 November 2026).** The project's focus moves from people detection to face recognition. In the user's
words: "After detecting a person we want a state-of-the-art face recognition system to identify their face. This is
the core part and needs to be extremely well made." For that the simulation needs "high resolution faces(!) in both
texture and mesh".

**Decisions by the user:**

- **Licences.** Research-only (non-commercial) model weights are acceptable, but "best would be a solution that allows
  both". So there are two switchable profiles behind one interface: the most accurate models, with research-only
  weights allowed, and commercially usable ones.
- **Head mesh.** The head's geometry may be **fitted to the person** from their photos, not only refined. This extends
  the photo-to-head component, which until now made textures only.

## Where we start (measured on the branch)

| Part | Today |
|---|---|
| Recognition | OpenCV's YuNet detector and SFace recogniser (128-d embeddings), run over the whole frame; a gallery of known people in `faces/` |
| Person detection | Ultralytics YOLO (`yolo26n.pt`); not connected to the face recognition |
| Camera | 1280x720 for the whole hall; faces in the fair are a few dozen pixels across |
| Head textures | Prepared at 1024 px; the Rocketbox sources are 2048x2048, with normal maps |
| Head mesh | Rocketbox "hipoly": about 1,700 vertices and 3,100 triangles per head (7,800 triangles per avatar) |
| Renderer | Forward shading with MSAA; no normal maps |
| Faces from photos | `characters/headtex/`: textures only, at 1024; the avatar's head shape is kept |

## Research (running)

Two literature reviews, each with a concrete recommendation for this code base, written in the style of the earlier
ones. They will be committed here as they grow.

1. **[face_recognition_sota.md](face_recognition_sota.md): state-of-the-art face recognition for a camera system.**
   - Covers face detection, alignment, quality assessment, the recognition models (losses, backbones, training sets,
     benchmarks), low resolution, tracking and aggregation over video, open-set identification and calibration, and
     evaluation.
   - Recommends both profiles, with where each model can be fetched (GitHub, GCS or PyPI; no Hugging Face in the
     build environment) and in which form (ONNX, safetensors, MediaPipe bundles; no pickled or TorchScript files).
2. **[high_res_faces.md](high_res_faces.md): high-resolution faces in the rendered crowd.**
   - How many pixels recognition needs, and a "virtual high-resolution sensor" that renders a detected person's head
     at high resolution through a sub-frustum of the same camera.
   - Textures at 2048 and above, with normal maps and detail.
   - Subdividing the skinned head mesh.
   - Fitting the head's shape to a person's photos: multi-view landmarks, 3D morphable models and their licences, or
     deforming the avatar's own head, and transferring the shape while keeping UVs, skinning and bones.
   - Shading that matters to recognisers, and how to evaluate it all.

## Planned steps (a draft; settled once the reviews are in)

1. **Recognition core:**
   - pluggable backends with the two profiles;
   - face detection in each detected person's region, alignment, quality, embeddings;
   - tracks with quality-weighted evidence over frames;
   - open-set decisions with calibrated thresholds;
   - the gallery and enrolment.
2. **Evaluation harness:** a benchmark through the simulation (known identities on avatars, rendered at many sizes,
   poses and lights) and on photo pairs: verification, identification and open-set metrics, broken down by condition.
3. **High-resolution heads:** full-resolution head textures with normal maps, a refined head mesh, and the renderer
   changes they need.
4. **Virtual high-resolution sensor:** after a person is detected, their head region is rendered at high resolution
   for the recogniser.
5. **Head shape fitted to the person,** combined with the existing photo-to-texture pipeline: geometry first, then the
   texture on the fitted shape; saved with created characters and used for known people in the fair.
6. **Integration in the demo:** person, then face, then identity, with the overlay and the status.

Each step is built in its own worktree, reviewed independently, fixed, tested on the branch, then pushed.

## Progress log

- **1 November:** focus changed; licence and head-mesh decisions taken; both reviews started; onnxruntime 1.31.1
  installed in the build environment.
