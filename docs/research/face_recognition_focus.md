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

## Research (done)

1. **[face_recognition_sota.md](face_recognition_sota.md): state-of-the-art face recognition for a camera system.**
   - **The shape:** person first, face second, and a decision per person track. That decision has three states: a
     name, "unknown" or "not decided yet". It is calibrated per model and per face size against one target
     false-alarm rate (FPIR) that means the same in every profile.
   - **Profiles behind one switch** (`--face-profile`, `BIFOLD_FACE_PROFILE`):

     | | Face detector | Recogniser | Quality |
     |---|---|---|---|
     | **best** (research-only weights) | SCRFD-10G | AdaFace IR-101 (WebFace12M; a community ONNX, to be checked first) | eDifFIQA-T plus geometry |
     | **licensed** | SCRFD-10G | InsightFace `buffalo_l` ResNet-50, under a licence bought from InsightFace | eDifFIQA-T plus geometry |
     | **commercial** | YuNet | AuraFace-v1 (Apache-2.0; on Hugging Face, so fetched on the Mac), else SFace as now | geometry |

   - **The gap:** faces in today's 720p view have 10-22 px between the eyes, against the 90 px the standards ask of
     reference photos. "The high-resolution head view is the biggest lever"; no model change does as much.
   - **What stays out of the identity path:** super-resolution and frontalisation.
   - **Evaluation:**
     - every model is checked first: files, inputs and outputs, colour order, published accuracy;
     - the metrics: ROC and TAR at FAR, open-set FNIR at FPIR, and track-level measures;
     - protocols on rendered stills and video, real photos and webcam recordings;
     - an acceptance test for the best profile: at least 95 % of enrolled people's tracks named correctly within 2 s,
       at 1 % FPIR per track.
2. **[high_res_faces.md](high_res_faces.md): high-resolution faces in the rendered crowd.**
   - **A virtual high-resolution sensor.** For each detected person, a window around the head is rendered through an
     off-axis sub-frustum of the same camera, exactly a crop of a declared high-resolution sensor. Then camera faults
     are added: blur, noise, demosaicing and JPEG.
     - Measured on llvmpipe: 11-23 ms per window, against 180-195 ms for the whole 720p view.
     - It matches a crop of a 5120x2880 render at 54-65 dB PSNR.
   - **Heads at 2048** with their Rocketbox normal and specular maps (every avatar has them), in linear light.
   - **The shape fit:**
     - the identity space of ICT-FaceKit (MIT, plain OBJ files on GitHub) is moved onto each Rocketbox head;
     - each person is fitted across all their photos at once, from MediaPipe landmarks, the irises for scale and the
       head outlines, with a small free-form correction;
     - the fitted shape goes onto the avatar keeping its topology, UVs, skin weights and rig (eyes and teeth moved,
       bones recomputed);
     - it is stored as a `shape.npz` of about 30 KB.

     None of the learned 3D face regressors can run here: they need pickled weights, a FLAME or Basel account, or
     CUDA.
   - **Then the texture** is made on the fitted head at 2048.
   - **Later:** subdivided heads and skin, eye and light shading for close-ups.

## Steps

| Step | What | Needs |
|---|---|---|
| **F1** Recognition core | A model registry with checked downloads (InsightFace packs from GitHub releases, zip members), onnxruntime sessions with telemetry off, SCRFD and YuNet detectors, ArcFace alignment, the AdaFace, `buffalo_l`, SFace and AuraFace embedders, quality, the three profiles, per-model gallery caches, enrolment checks, and model checks | - |
| **S1** Virtual high-resolution sensor | Sub-frustum windows snapped to a declared sensor, culling, camera presets (blur, noise, demosaicing, JPEG); windows delivered with the frames | - |
| **F2** Person, face, identity | Person tracks (YOLO with a tracker), head regions (high-resolution windows for undecided tracks), quality-weighted evidence per track, open-set decisions (cohort normalisation, calibration per model and face size, target FPIR, runner-up margin, hysteresis, one identity per track); the overlay and status in the demo | F1, S1 |
| **E1** Evaluation and first results | Metrics (ROC, DET, TAR at FAR, CMC, FNIR at FPIR, calibration), `bifold-faces eval` with manifests, rendered stills and video with disjoint photo sets for textures and gallery, baseline numbers for today's SFace against the new profiles; the results committed | F1, F2, S1 |
| **S2** High-resolution heads | 2048 head colour, normal and specular maps; linear light; normal and specular shading; the fair's light re-tuned | S1 |
| **S3** Shape fitted to the person | The ICT-FaceKit identity space on each head, multi-view fitting, transfer to the rig, `shape.npz`; textures made on the fitted head at 2048; used by created characters and known faces | S2 |
| **S4** Close-ups | Subdivided heads in windows; skin, eye and light shading | S2 |

F1 and S1 start in parallel. Each step is built in its own worktree, reviewed independently, fixed, tested on the
branch and pushed; its results go into the progress log.

## Progress log

- **1 November:** focus changed; licence and head-mesh decisions taken; both reviews started; onnxruntime 1.31.1
  installed in the build environment.
- **1 November, about 15:00 UTC:** both reviews finished and committed; the steps above settled; F1 and S1 start.
