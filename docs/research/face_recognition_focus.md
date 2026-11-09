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
- **2 November, 15:50 UTC: stopped on request ("stop all retries").** The account's weekly usage limit had
  stopped the builders, and the scheduled check-ins kept re-firing. All check-ins are deleted and nothing
  runs.
  - **The branch:** the full suite with S1's fixes and F1 part 4 gave 870 passed, 25 skipped; the known
    linear-light regression was left out.
  - **F1:** parts 1-4 are on the branch. The builder had finished its work and was waiting for its full suite
    and final report; its worktree is `.claude/worktrees/agent-ae1a230ca5fea359e` (last commit `ae2c3c0`). Its
    independent review has not started.
  - **S2:** parts 1-2 are on the branch. The fix for the linear-light regression (the real-photo head-texture
    test) and the remaining pieces are uncommitted in `.claude/worktrees/agent-acf4f9b29edc8aa9e`.
  - **To resume:** message the two builders (their work is saved), then review F1 and S2, and start F2 and
    E1.
- **8 November, 19:20 UTC: resumed on request ("Resume the builders and finish F1 and S2").** The container had
  restarted with a full disk; old October scratch files were deleted (19 GB free). The builders had committed
  more before the stop than the last entry says. Now on the branch:
  - F1 part 5: the README's note on CPU speed per profile (on four cores AdaFace takes about half a second per
    face, the ResNet-50 a tenth, SFace 20-30 ms);
  - F1 part 6: the results note gives the threshold estimates unrounded, as the registry uses them;
  - S2's fix for the linear-light regression: the head-texture pipeline's own renders of the avatar draw on the
    stored sRGB values whatever the light model (`Renderer.linear`), so its landmarks and regions are the same
    as before to the bit. One conflict in `renderer.py` with S1's windows, combined.
  - The tests of the touched modules, with real avatars and photos: 133 passed, including the real-photo
    head-texture test that had failed.
  - Running: F1's final suite and report, then its independent review; S2's shading and detail streaming,
    photo textures at 2048 and measurements.
- **8 November, 21:05 UTC: S2 built, F1 reviewed.**
  - **S2:** parts 3-5 are on the branch:
    - normal and specular shading, light from above, and 2048 heads only where a picture shows the eyes at least
      64 px apart;
    - faces from photos at 2048;
    - the results note [high_res_heads_results.md](high_res_heads_results.md) and the preview.
    - Three conflicts with S1's review fixes and F1's profile flag (`app.py`, `renderer.py`, `scene_source.py`),
      combined.
    - The builder's full suite: 855 passed, 8 skipped. The related tests on the branch, with the 2048 library and
      the photos: 231 passed.
    - **What changes the faces is the maps, not 2048.** On S1's windows at 2-5 m the maps move AdaFace's embedding of
      the same face to a cosine of 0.86 (0.70 at worst) and separate identities a little better (d' +0.3-0.4);
      1024 against 2048 gives 0.998, because the embedders see a 112-pixel crop. The 2048 textures matter for
      close-ups (S4).
    - Cost on llvmpipe: the 720p view 1.2-1.3 times, the head windows 1.4 times slower; 107 MiB of maps for the 20
      heads and 43 MiB per head drawn at 2048. A face from photos at 2048 takes 2-3.5 times as long.
    - Its independent review is running.
  - **F1 review:** no high findings, 6 medium, 11 low. Confirmed: every model file and hash, the preprocessing and
    colour orders, SCRFD decoding (identical to the reference), alignment, unit embeddings and bitwise determinism,
    the measurements. To fix:
    - "best" does not fall back when AdaFace is present but damaged;
    - code that only detects faces now needs, and downloads, the recogniser;
    - `--face-profile` is ignored in the creator and the head-texture check;
    - the thresholds from a Gaussian fit are not strict for the ResNets (counted on all 3,000 LFW non-mated
      pairs, the ResNet-50's cut at a false-match rate of 1e-3 is 0.216, not 0.19);
    - adding photos while the demo runs freezes the view for 0.7-3 s per photo;
    - batches of 2-7 crops run one at a time (29-47 % slower).
    - **F2's budget on this CPU:** AdaFace takes 0.7-1.0 s per face here, 5-10 times the research's estimate. F2
      should plan for about one AdaFace embedding a second, or use the ResNet-50.
    - The F1 builder is fixing them forward.
- **8 November, 22:00 UTC: F1's review fixed; F1 is done.** The builder fixed all 6 medium and 11 low findings forward
  in nine commits, now on the branch (no conflicts):
  - **"best" falls back when AdaFace is unusable.** A recogniser is used only if its file is there, its SHA-256
    matches (checked once per process) and it loads; otherwise the profile's next model is used, with a note.
  - **Detection no longer needs the recogniser.** The engine loads the detector only; the recogniser and eDifFIQA-T
    load on first use. SCRFD comes from its 17 MB copy before the 289 MB pack. Offline errors point to
    `--face-profile commercial`.
  - **`--face-profile` reaches every face model of a run:** the creator, the faces painted on the 3D people, the
    head-texture check and its command line.
  - **Counted thresholds.** On all 3,000 LFW non-mated pairs the cut at a false-match rate of 1e-3 is 0.210 for
    AdaFace and 0.216 for the ResNet-50. Both now use **0.22**; the old 0.18 and 0.19 let 5 and 6 pairs through
    instead of 3. On LFW's mated pairs this costs nothing: 2 of 750 fall below the old and the new cut alike. A test
    pins the thresholds.
  - **Adding photos no longer freezes the view.** Enrolment runs outside the gallery's lock, and in the live demo on a
    worker thread (`refresh(wait=False)`). Reading the gallery during an enrolment waited 0.4-0.6 s before; now it
    does not wait.
  - **One run per call on the CPU.** Embedding five heads through the engine took 1.17 s before and 0.83 s after.
  - **The lows:**
    - odd canvas sizes;
    - tight close-ups: 15 of 15 faces found at every fill, where at 83 % none were found before;
    - the three surviving mutants killed;
    - a replaced photo leaves every model's cache;
    - the blur check now rejects a 3-pixel blur;
    - telemetry off on every import path;
    - fetch errors;
    - pitch kept out of decisions;
    - AuraFace used only after it passes its checks and beats SFace;
    - doc errors.
  - **Four API additions for F2 and E1:**
    - model ids on every result;
    - raw scores per photo and template;
    - several images in one batch;
    - caches that remember the detector and the model's hash.
  - **Tests.** The builder's full suite: 821 passed, 9 skipped. On the branch, F1's tests and their callers (the
    head-texture pipeline, the creator, the cast, the pipeline) with every model, the photos and the 2048 library:
    333 passed, none skipped. Lint unchanged.
  - S2's independent review is still running.
- **8 November, 22:10 UTC: stopped on request, S2's review unfinished.** F1 is done; S2 is on the branch, its review
  stopped part way. The reviewer's checks so far:
  - **Found sound:**
    - `Renderer.linear = False` draws exactly as the old renderer (bit for bit, with multisampling, the view and
      three windows).
    - Tangents on all 20 heads: no degenerate triangles, skinned and turned with the head.
    - The packed maps are kept as data at upload and in their mipmaps.
    - No seams under raking light.
    - The memory figures are exact.
    - The full suite on the branch with the 2048 library: 897 passed, 14 skipped. Lint unchanged.
    - On avatars the maps help recognition: AdaFace d' +0.5-0.65 with the `good 4K` and `budget 1080p` cameras,
      on turned and lowered heads too. 2048 adds nothing at these distances.
  - **Problems found** (not yet sent to the builder):
    - **Faces from photos lose identity.** A face from photos is drawn with the avatar's own normal and specular
      maps. On 48 windows AdaFace's same-person score fell from 0.363 to 0.275 on every window (d' 6.02 to 4.72);
      the ResNet-50 lost about as much, SFace less.
    - **Detail streaming thrashes** with more than six people near the camera:
      - with waiting, every window reads its head again (a frame and four windows take 2.6 s instead of about
        0.5 s);
      - in the live demo, no window of 480 got a 2048 head.
    - **Weak tests:** 16 of 42 mutations survive the targeted tests, among them tangents not skinned or not
      turned, evicted textures not freed, and `render_views`' pinned settings ignored.
    - **Smaller:**
      - a face from photos never gets the 2048 maps;
      - the photo-face cache key ignores the texture size;
      - the stated reason for the 64 px detail threshold is wrong (the threshold itself is fine);
      - windows are 1.6-1.7 times slower, not 1.4;
      - a one-off 70-160 ms hitch copying a 2048 photo face;
      - some numbers in the docs were measured under other settings.
  - **To resume:**
    - finish the review (its notes: `scratchpad/steps/S2_review/notes.txt`) or send these findings to the S2
      builder;
    - then switch the shared avatar library to the 2048 one, run the full suite and report.
- **9 November, 02:00 UTC: resumed on request ("Continue"); S2's review finished.**
  - **Verdict:** accept with required fixes. 1 high, 2 medium and 7 low findings.
  - **High:** faces from photos are drawn with the avatar's maps (the entry above).
  - **Medium:** the detail-cache thrash; 16 of 42 mutations surviving the tests.
  - **Low:**
    - documentation numbers measured under other settings;
    - photo faces always drawn with 1024 maps;
    - a one-off hitch while a 2048 photo face is prepared;
    - `memory()` leaves out 32 MiB of CPU copies per near head;
    - no hint for libraries still in the old format, and stale photo-face caches after an upgrade;
    - painted hair turning glossy;
    - small latent issues.
  - **Checked and found true** beyond the entry above:
    - the touched and real-avatar tests on the old library: 697 passed, 15 skipped;
    - the results note's recognition table, within 0.1 d';
    - harder cases (the `good 4K` and `budget 1080p` cameras, heads turned 40 degrees or lowered 20): every face is
      found, and the maps raise AdaFace's d' by 0.5-0.65;
    - the detail streaming does no GL work off the render thread.
  - **For later steps:**
    - F2 should record the quality and the hemisphere with every calibration;
    - E1 should keep `detail_wait` and set `detail_heads` to the near avatars per frame;
    - S3 needs tangents recomputed after fitting and its own maps for fitted photo heads;
    - S4 needs the 2048 maps for photo faces.
  - **The S2 builder** is fixing all findings forward on top of the branch.
- **9 November, 03:35 UTC: S2's review fixed; F1 and S2 are done.** The S2 builder fixed every finding forward except
  painted hair's sheen (documented). Seven commits are on the branch:
  - **Faces from photos keep their identity.** Where a face from photos covers the head, it is drawn without the
    avatar's normal and specular maps and without the hemisphere light's variation; the face already carries the
    person's own relief and soft light from above. This covers the cast, the head-texture stages and created
    characters.
    - Re-measured on the branch with the review's unchanged script (painted faces, 48 windows), at the default
      quality:
      - AdaFace's same-person score 0.359, against 0.275 before and 0.363 at the low quality;
      - the ResNet-50 0.346 (0.265 before), SFace 0.397 (0.356 before);
      - no window below AdaFace's or the ResNet-50's threshold (12 and 14 of 48 below 0.22 before).
    - The stages' identity check, by the builder: 0.234 → 0.274.
    - The avatars' own faces are drawn exactly as before, so the maps' gain for them stands (AdaFace d' +0.5-0.65 in
      the harder cases).
    - `Character.maps_overrides` gives a person their own maps (S3's fitted heads).
  - **Detail textures by need.** The heads whose eyes are furthest apart keep their 2048 textures: claims fade when a
    head is not drawn, and a newcomer needs eyes 1.25 times further apart to take a place. Heads outside the picture do
    not count, and pixels are freed once the textures are on the GPU. Re-measured on the branch with 8 people near the
    camera:
    - with `detail_wait`: 6 reads instead of 96, a frame with its windows 341 ms instead of 2572;
    - in the live demo: 303 of 480 windows at 2048 instead of none.
    - (The review's script needed a one-line change: fix 5 added an argument to the cache's read function.)
  - **Tests:** all 42 of the review's mutations are now caught (26 before). One new test found NaN tangents on
    vertices without a normal (fixed).
  - **The lows:**
    - documentation numbers corrected;
    - a near 2048 photo face gets the head's 2048 maps;
    - a 2048 face's 1024 copy is made on the worker thread;
    - packed maps are no longer averaged weighted by their alpha;
    - colour images are resized in linear light;
    - the fair names avatars of the old format at start, with the command that updates them;
    - photo-face caches are made again after a library upgrade.
  - **Checks:**
    - the builder's tests on the old library: 713 passed, 9 skipped;
    - the full suite on the branch with the 2048 library, the photos and every face model: 933 passed, 8 skipped
      (they need Rocketbox's raw files);
    - lint unchanged.
  - **The shared avatar library for later steps** is `charcache2048` (format 2, 20 avatars, 367 MB). The old one
    stays for the format-1 tests.
    - A library prepared before S2 still loads and draws, but without maps or 2048 heads.
    - To update it, run the command the fair prints at start: `bifold-characters fetch --avatars <names> --clips
      basic`, with `--dest <library>` when it is not the default folder.
  - **Next, when asked:**
    - F2: person tracks → face → identity, with open-set decisions calibrated per model and face size. On a CPU like
      this one, budget about one AdaFace embedding a second, or use the ResNet-50.
    - E1: evaluation.
    - S3: fitting the head's shape to a person (ICT-FaceKit).
    - S4: close-ups.
- **9 November, 09:45 UTC: F2 and S3 started** (on request: "Continue"). Both builders work in their own worktrees
  from the branch's head.
  - **F2, person, face, identity:**
    - person tracks from YOLO's boxes, with a tracker of our own (no AGPL code in the commercial profile);
    - faces looked for only in each track's head region, in S1's windows when the track is undecided;
    - quality-weighted evidence per track within an embedding budget per profile;
    - open-set decisions in three states, calibrated per model and face size against one target FPIR, from avatar
      renders that may be shipped;
    - one identity per track;
    - the demo's overlay;
    - if time allows, YOLOX as the commercial profile's person detector.
  - **S3, the head's shape fitted to the person:**
    - ICT-FaceKit's identity space (MIT) carried onto every avatar head;
    - a multi-view fit to a person's photos (landmarks, the irises for scale, outlines, a small free-form
      correction);
    - the fitted head on the rig (eyes, teeth and face bones follow);
    - textures made on the fitted head at 2048;
    - a `shape.npz` per created character and known face;
    - measured on avatars as people and on random faces from the space, and for recognition.
  - **Then:** both reviewed independently. E1 (evaluation) follows F2, and S4 (close-ups) follows S3.
- **9 November, 12:15 UTC: F2 and S3 well along.** Each piece passed its tests on the branch before it was pushed.
  - **F2, on the branch (parts 1-8):**
    - person tracks: ByteTrack in numpy, with its own assignment solver;
    - faces looked for only in the tracks' head regions (S1's windows, else crops), each face given to the right
      track when regions overlap;
    - quality-weighted evidence per track within an embedding budget (here about 1 AdaFace embedding a second, 4 of the
      ResNet-50, 20 of SFace);
    - the decisions:
      - a name, "unknown" or "not decided yet", calibrated per model, face size and yaw against one target
        false-alarm rate;
      - hysteresis, and one identity per track;
      - a log of every decision for E1;
    - the demo's N with the new overlay (the old per-frame recogniser is gone);
    - YOLOX as the commercial profile's person detector, so that profile runs no AGPL code. On the fair it finds 66 %
      of the people 40-80 px tall, against YOLO's 43 %, but takes 260 ms a frame here against 110.
    - **Calibrations shipped for AdaFace, the ResNet-50 and SFace,** made from renders of 38 Rocketbox avatars that
      are not in the fair, with T-norm and its cohort. The choice was made by a held-out check. Avatars that wear the
      same face count as one identity.
  - **S3, on the branch (parts 1-3):**
    - ICT-FaceKit's identity space (fetched from a pinned commit, every file checked);
    - the space carried onto each avatar head;
    - a Levenberg-Marquardt fit over all photos: landmarks weighted by how faithfully MediaPipe follows a shape,
      outlines from profiles, the size from the irises held apart from the shape;
    - the fitted head on the rig (eyes, teeth and face bones follow; checks in all 12 expressions);
    - known faces, created characters and the creator wear fitted heads, kept in `shape.npz`.
    - **First numbers on avatars standing in for people (tuning pairs):** landmark error after a similarity alignment
      3.18 → 2.29 mm. Fitted with the space alone, without the outlines and the free-form correction, a head barely
      improves: the landmarks carry little of the nose, cheeks and profile.
  - Both builders are measuring and finishing. Their independent reviews follow.
- **9 November, 13:10 UTC: F2 built and measured; its independent review started.**
  - **All ten parts are on the branch;** the last is the results note
    [face_recognition_f2_results.md](face_recognition_f2_results.md) and a preview of the overlay.
  - **The builder's full suite:** 983 passed, 8 skipped.
  - **No wrong names.** In the fair runs (30 s, two crowds, half the avatars known, windows on) no stranger was named
    in any profile, and nobody was named as someone else.

    | Profile | Known people named within 2 s | Median time to a name |
    |---|---|---|
    | commercial (SFace) | 8 of 8 | 0.8 s |
    | licensed (ResNet-50) | 7 of 8 | 1.0 s |
    | best (AdaFace) | 1 of 8 | 5.0 s |

    The best profile is held back by this CPU's budget of one AdaFace embedding a second; with 10 a second it named
    3 of 3 in 0.8 s.
  - **The windows matter:** without them the faces are 11-19 px, and the commercial profile named 2 of 8 within 2 s.
  - **The calibration's held-out check** (avatars not used to make it), at a 1 % false-alarm rate asked:
    - measured with T-norm: AdaFace 0.66 %, the ResNet-50 1.72 %, SFace 0.32 %;
    - without normalisation: 2.29 %, 3.48 % and 0.88 %.
  - **Main open issue, on real faces:** the calibrations are made from avatars. Checked on LFW, T-norm against the
    avatar cohort lets 3-6 times more real strangers through than asked at 1 in 1,000, and the raw avatar thresholds
    are far too strict. E1 must measure the asked rate against the measured one on real people. For now the README
    advises a lower `--fpir` for real faces.
