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
- **1 November, 15:15 UTC:** first pieces on the branch.
  - F1: the model registry, the profiles, checked zip-pack downloads, ONNX Runtime; the SCRFD and YuNet
    detectors; ArcFace five-point alignment.
  - S1: windows that are exact crops of a high-resolution sensor; culling; window requests and results with
    the frames.
  - The tests of the new and touched modules: 169 passed, 3 skipped.
- **1 November, 15:57 UTC:** more pieces on the branch.
  - F1: the embedders and the quality record per face.
  - S1:
    - the camera model (blur, noise, mosaic, tone, JPEG);
    - the windows in the demo (`W`) and `bifold-scene windows`;
    - the measured results and the preview;
    - window limits per renderer.
  - The tests of the new and touched modules: 158 passed, 3 skipped.
- **1 November, 16:10 UTC:** S1 finished: all seven commits are on the branch.
  - The builder's full suite: 779 passed, 9 skipped.
  - With the default 7680x4320 sensor a frontal face has six times the eye distance of the 720p view (75 px
    against 12.5 px at 4 m).
  - YuNet finds 36 of 36 frontal faces at 2-30 m in the windows, against 20 of 36 in the view.
  - Details in [virtual_sensor_results.md](virtual_sensor_results.md).
  - Its independent review is running.
- **1 November, 17:05 UTC:** S1 reviewed independently: no high-severity findings, 5 medium, 9 low.
  - **Confirmed:**
    - near-exact crops and exact pixel maps;
    - culling is conservative: no triangle wrongly left out in about 330,000 random windows;
    - the main view renders pixel-identical to before;
    - the measurements reproduce;
    - 816 passed, 12 skipped.
  - **To fix:**
    - the evaluation truth's 'frontal' eye distance is biased off-axis, up to 1.3x;
    - the truth has no head orientation;
    - requests off the view use up window slots;
    - invalid requests fail on the render thread instead of at the caller;
    - `--sensor` of another shape ends the demo.
  - **Lows:** camera faults applied at binned scale, motion blur off at long frame gaps, windows at odd pixel
    offsets, framebuffer memory, doc numbers, test gaps, E1's control of the eye distance.
  - The S1 builder is fixing them.
- **1 November, 17:25 UTC:** on the branch:
  - S1 review fixes 1-5: the evaluation truth's face size uses the turned-to-the-camera definition and carries
    the head pose; requests are checked where they are made; requests off the view take no windows;
  - S2, part 1: head normal and specular maps, 2048 detail textures, tangents, cache format 2;
  - S2, part 2: linear light.
  - The full suite is running on the branch.
- **1 November, 17:50 UTC:** the full suite on the branch found a regression from S2's linear light.
  - The head-texture test on real portraits (it runs only when the test-photo folder is set) passes after S2
    part 1 and fails after part 2: for a 480p portrait, 7.8 % of face texels reach the keep or refine level,
    against the 10 % required.
  - The pipeline's own renders of the avatar (landmarks, regions, shading) change under linear light.
  - The S2 builder is fixing it; the rest of the suite is running.
- **1 November, 21:00 UTC** (after the usage limit, 18:00-20:30, stopped all builders):
  - the rest of the suite on the branch before these commits: 841 passed, 12 skipped, beside the known
    regression;
  - S1 review fixes 6-14 on the branch: the camera model at the camera's own pixels, speeds from frames far
    apart, even window corners, SubCamera backgrounds, framebuffers bounded by bytes, sensors with useful
    binning levels, exact eye distances for E1, the doc numbers, tests for the surviving mutants;
  - F1 part 4: the engine built from a profile, the new gallery, model checks;
  - one conflict in `renderer.py` (S1's framebuffer budget against S2's sRGB colour buffers), combined;
  - the builders are resumed and the full suite is running.
- **1 November, 21:20 UTC:** **S1 done.** All 14 review findings are fixed, none rejected. The builder's full
  suite: 799 passed, 9 skipped. The builder confirmed the `renderer.py` merge with S2. Key results:
  - one frontal eye-distance definition everywhere, and the head pose in the truth;
  - requests checked at the caller, and off-view requests take no slots;
  - the camera model at the camera's own pixels: good 4K at level 4 gives 29.4 dB, was 26.7;
  - framebuffers kept within 192 MB (42 MB in the reviewer's case, was 205);
  - `WindowRequest(eye_px=...)` for E1: 8-160 px within 0.1 %.
