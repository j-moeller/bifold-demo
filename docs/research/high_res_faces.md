# High-resolution faces in the rendered crowd: literature review

What the simulated fair needs so that a face recogniser can be tested on it properly: enough pixels on a detected
person's face, textures and meshes that hold up at that resolution, the head's shape fitted to the person's photos,
and shading that does not give the recogniser a "rendered face" handicap. Written November 2026. The recognition
models themselves are covered by the companion review (`face_recognition_sota.md`).

How the claims were checked: GitHub (raw files, and blobless clones for file listings), PyPI and Google Cloud Storage
(the MediaPipe model cards) could be opened from the development sandbox; arXiv, CVF, the MPI sites (FLAME, NoW),
NIST, ISO/IEC, Khronos' website, vendor sites and Hugging Face could not. Claims marked *(excerpt)* rest on
search-result excerpts only. Numbers marked *(measured)* were measured in the sandbox (4-core CPU, Mesa 25.2.8
llvmpipe, OpenGL 4.5) with scratch scripts that are not in the repository; the others are quoted from the cited
source or marked as estimates.

## 1. The problem and our constraints

**What exists.** Measured on the 20 prepared avatars of the test library unless a source is given.

- **The heads.** The avatars are the Rocketbox "hipoly" level (7,836 triangles for Male_Adult_05). The head primitive
  has 1,156-1,925 vertices and 1,984-3,434 triangles (median about 1,800 and 3,150) and contains the eyeballs (about
  73 vertices each, skinned to the eye bones), the teeth, gums and tongue. Eyelashes and hair cards are a separate,
  alpha-tested primitive (`*_opacity`, skinned to the head and the upper eyelids). Across the visible face an edge is
  8.5 mm long on the median (90 % are shorter than 16-21 mm) *(measured)*. 28 face bones below `Bip01 Head` slide
  rather than turn (`characters/faces.py`); the Poser takes only rotations from clips and keeps the model's own bone
  offsets (`model.py`), so moved bones stay moved under animation.
- **The skeletons are templates.** The eye bones of every adult woman are 60.0 mm apart, of every man 63.8 mm, of
  the children 48-50 mm *(measured)*: the heads differ by their meshes, not their rigs. Real adults average about
  63 mm between the pupils, most between 50 and 75 mm (Dodgson 2004, *excerpt*).
- **The textures.** `prepare()` scales every texture to 1024 pixels; the sources are 2048. On the front of the face
  the head atlas has 3.6 texels per millimetre at 2048 (median of 20 heads; 3.0-4.4), so the distance between the
  eye centres spans about **112 texels at 1024 and 225 at 2048** *(measured)*. Only the colour maps are downloaded.
- **The renderer** (`scene/renderer.py`): forward shading with one directional light, Lambert diffuse plus a
  constant ambient term, fog, 4x MSAA, mipmaps and 8x anisotropic filtering. Textures are uploaded as plain RGBA8,
  so lighting, mipmaps and the MSAA resolve all work on gamma-encoded values. No normal or specular maps, and no
  tangents in the vertex layout (`3f 3f 2f 4f 4f`: position, normal, UV, joints, weights).
- **The camera.** 1280x720 with a 55-degree vertical field of view: a focal length of 692 pixels. A frontal face shows
  **11-16 pixels between the eyes** at the audience's 3.9-4 m (more towards the picture's sides, where the
  perspective stretches it) and 4-5 pixels at 9 m *(measured)*; 1.6 pixels at the 26 m aisle (computed).
- **Photos to head texture** (`characters/headtex/`): MediaPipe's 478 landmarks lifted onto the head mesh, each
  photo's camera by PnP (without EXIF data a 50-degree lens is in effect assumed), a thin-plate warp that moves the
  person's features onto the avatar's, fusion in the head texture's own UV layout at its own size (1024 now), light
  taken out, unseen parts completed, body skin and hair matched, an identity check. With the avatar's own geometry,
  renders of a textured head reach an SFace similarity of 0.74 from the front and 0.65 at 30 degrees in a test where
  renders of another avatar stood in for the photos (the target avatar's own face: 0.25 and 0.29); three real people
  were named right in 24 of 24 close-ups and in 98 of 120 fair frames with faces about 30 pixels tall (README).

**What is wanted.** "After detecting a person we want a state-of-the-art face recognition system to identify their
face", which needs high-resolution faces "in both texture and mesh". In detail: a way to see a detected person's head
at high resolution, as a real high-resolution or zoom camera would; textures and meshes that hold up there; the head's
geometry fitted to the person's photos (until now the photos changed only the texture); shading that does not hurt
recognition; and measurements that show each step helps.

**Constraints.**

- The demo runs on an Apple-silicon Mac (OpenGL 4.1 through moderngl; Apple deprecated OpenGL but it works). The
  sandbox is CPU-only Linux with 4 CPUs and renders with llvmpipe. Everything except optional generative steps must
  be testable here.
- Network here: GitHub, Google Cloud Storage and PyPI only; no Hugging Face.
- No third-party code or pickled / TorchScript models are downloaded and run. ONNX, safetensors through official
  libraries and MediaPipe bundles are acceptable; so are plain data files (OBJ, PNG, TGA, npz without pickles).
- Research-only licences are acceptable if stated. FLAME and the Basel Face Model need a registration we cannot do.
- Photos of people and anything derived from them, including a fitted face shape, are personal (biometric) data:
  never committed, kept in git-ignored folders, deleted with the person.

**Why our case differs from the literature.**

1. **The recogniser does not need pores.** ArcFace-style recognisers look at a 112x112 crop in which the eye centres
   are 35 pixels apart (insightface's alignment template: eyes at x = 38.3 and 73.5). What matters is that the camera
   delivers at least that many clean pixels, plus margin for detection, landmarks and quality assessment. Texture and
   mesh detail far beyond what that crop can show only matters for how "real" the face looks.
2. **We know the mesh and keep it.** Most face-reconstruction papers estimate a shape in a research topology (FLAME,
   BFM) and output that topology. We must keep the Rocketbox head's topology, UV layout, skin weights, bones, eyes and
   mouth interior, so that animation, outfits and the texture pipeline keep working.
3. **Cameras are free.** A "virtual sensor" that renders only a person's head at high resolution costs milliseconds;
   a real one costs money. The question is how to make its pictures as imperfect as a real camera's.

## 2. State of the art

### 2.1 Pixels for recognition, and cameras that deliver them

**How many pixels.** The standards ask for many pixels in *reference* photos (passports, enrolment); recognisers in
video work with fewer, but degrade steeply at a few tens of pixels between the eyes. The companion review gives the
accuracy curves of current recognisers; the figures that set our targets are these.

| Source | What it asks for | Notes |
|---|---|---|
| ISO/IEC 39794-5:2019 (face image data) | Eye distance at least 90 px; 120 px for storage on travel-document chips; 240 px best practice for most capture | Reference photos. *(excerpt, via the 2022 survey "State of the Art of Quality Assessment of Facial Images")* |
| ISO/IEC 19794-5:2011 | 120 px or more as best practice | *(excerpt)* |
| IEC 62676-4:2014 (video surveillance) | "Identification" 250 px/m: 40 px across a 16 cm wide face, about 16 px between the eyes; "recognition" 125 px/m | For a human operator, not an algorithm. *(excerpt: Axis white paper)* |
| ArcFace-family recognisers | 112x112 aligned crop, eyes 35 px apart | insightface `face_align.py` (verified) |
| People looking at faces | Near chance at 10 px and 5 px between the pupils (50.7 % and 35.9 %) | Merino et al. 2025 *(excerpt)* |

Working targets for the simulation: **at least 40 px between the eyes** so that the recogniser's 35-px template is
not upsampled, **60-120 px** for reliable landmarks, pose and quality scores (and for the evaluation of fitted heads),
and up to about 200 px for close-ups. Our 720p view gives 11-16 px at 4 m.

**What that means for a camera over a hall.** At the eye distance a camera delivers, the width of scene it covers is
its width in pixels times 63 mm divided by that eye distance, whatever the distance to the people (the lens only
decides where that band lies):

| Sensor width | Band covered at 40 px between the eyes | ... at 90 px |
|---|---|---|
| 1280 px (our view) | 2.0 m | 0.9 m |
| 1920 px (1080p) | 3.0 m | 1.3 m |
| 3840 px (4K) | 6.0 m | 2.7 m |
| 7680 px (8K) | 12.1 m | 5.4 m |

The fair's hall is 44 m wide. With our camera's lens, 40 px between the eyes needs a person within 1.1 m of the
camera at 1280x720, 3.3 m at 4K and 6.5 m at 8K. A 30x PTZ camera spans about 60 degrees to 2.2-2.3 degrees across
1920 pixels and pans at 100-500 degrees a second (datasheets, *excerpt*); at its tele end (a focal length of about
47,800 px) it delivers 90 px between the eyes at 33 m, but for one person at a time.

Real systems therefore combine three ideas:

- **High-resolution fixed cameras with digital crops** ("digital PTZ"): 4K-8K and multi-sensor panoramic cameras,
  the recogniser working on crops of the full sensor.
- **Master-slave PTZ**: IBM's Face Cataloger (Hampapur et al. 2003) tracked people in 3D with fixed wide-angle
  cameras and steered pan-tilt-zoom cameras to capture their heads while they moved *(excerpt)*.
- **Gigapixel arrays**: the PANDA dataset (CVPR 2020) was recorded with a gigapixel camera covering about a square
  kilometre with up to 4,000 people and more than 100x variation in scale *(excerpt)*.

**How to give the recogniser enough pixels in the simulation.**

| Option | Eye distance at 4 m | Cost on llvmpipe *(measured, 19 standing people, no hall)* | What it models |
|---|---|---|---|
| Main view at 1280x720 (now) | 11-16 px | 180-195 ms a frame | A 0.9-megapixel camera |
| Main view at 3840x2160 | 33-48 px | 495-509 ms a frame (2.7x) | A 4K camera, every face at once |
| Main view at 5120x2880 | 44-64 px | 2.2 s | A 15-megapixel camera |
| **Virtual sensor:** 384x384 window at 4x the 720p scale | 47 px | 11-16 ms with only that person drawn, 58-67 ms with all 19 | A crop of a 5120x2880 sensor |
| **Virtual sensor:** 512x512 window at 10x | 118 px | 19-23 ms with only that person, 51-65 ms with all | A crop of a 12,800x7,200 sensor |
| PTZ camera model | Any, within its zoom range | One more render per frame | Another viewpoint, slewing, zoom delays, one target at a time |

The **virtual high-resolution sensor** renders only a detected person's head region with an off-axis sub-frustum of
the same camera. If the window covers the rectangle `[x0, x1] x [y0, y1]` of the camera's normalised device
coordinates, its projection is `S @ P`, where `P` is the camera's projection and `S` scales and shifts that rectangle
onto `[-1, 1]` (the "crop matrix" of `gluPickMatrix` and of tiled renderers such as Brian Paul's TR library; Kooima's
generalised perspective projection is the same construction for an arbitrary screen). The geometry is exactly the
camera's: mipmap levels, MSAA and fog behave as on a real sensor of that resolution. Measured here, a 384x384 window
at 4x matched the same pixels cropped from a full 5120x2880 render at 53.7-65.3 dB PSNR; no pixel differed by more
than 8 levels except a few on silhouette edges, where the MSAA samples fall differently *(measured)*. The window costs
little when only the objects inside its frustum are drawn; without culling, every person's vertices are still
processed (the 58-67 ms above). Six 256x256 windows without culling took 206 ms *(measured)*.

**Realism.** A crop of a virtual sensor is noiseless, perfectly focused and frozen in time; a real camera's is not:

- **Sensor noise** is signal-dependent: a Poisson part from photon counting plus a Gaussian part (Foi et al. 2008,
  who also fit both from one image).
- **The camera pipeline** (demosaicing, white balance, tone curve, sharpening, compression) is what "unprocessing"
  inverts to synthesise raw training data (Brooks et al. 2019); run forwards, it makes a render look camera-made.
- **Blur**: lens and focus blur, and motion blur. A person walking at 1.4 m/s moves 14 mm during a 1/100 s exposure,
  about 28 pixels at the 2 pixels per millimetre of a face with 120 px between the eyes (computed). Face capture
  systems use short exposures and pay with noise.
- **Compression**: surveillance video is H.264/H.265 at a few megabits per second, which smears exactly the fine
  detail that higher resolution adds.

Degradation recipes from blind super-resolution (Real-ESRGAN's second-order blur, resize, noise and JPEG steps;
Wang et al. 2021) are a ready-made starting point. The order matters: render at the sensor's own pixel scale, then
optical and motion blur, noise, demosaicing and tone, compression, and only then the recogniser's own resampling.

### 2.2 High-resolution face textures

**The Rocketbox sources** (checked in the Microsoft-Rocketbox repository: file listing, file sizes, TGA headers, and
one head's maps and one `_facial` FBX downloaded):

- **Every head has colour, normal and specular maps at 2048x2048.** All 346 head maps (115 colour, 117 normal, 115
  specular) are 12,582,956 bytes: uncompressed 24-bit 2048x2048 TGA. Two heads have an extra wrinkle normal map
  (`f001_head_normal_wrinkle.tga` with a Unity shader that blends two normal maps; `m003_head_normal - wr.tga`). The
  `Source/` folder holds the same 2048 maps and 3ds Max scenes. There are no displacement, ambient-occlusion,
  roughness or sculpt files for any avatar: the high-resolution sculpts the normal maps were baked from are not
  released.
- **The normal maps are tangent-space** (Male_Adult_05: mean colour 128, 128, 255) and carry the face's folds (lips,
  nostrils, nasolabial folds, eye bags) and a pore-scale micro-texture across the skin. The specular maps are grey and
  low on skin (mean 23 of 255 over the 41 % of texels that are not black). The official Unity importer
  (`Assets/Editor/FixRocketboxMaxImport.cs`) marks every texture with "normal" in its name as a normal map and changes
  nothing else; Unity's convention is green-up with MikkTSpace tangents. The maps were baked in 3ds Max, whose
  tangent basis may differ, so the convention should be checked on one head (a raking light makes a flipped green
  channel obvious).
- **The `_facial` variants.** 117 files: one per avatar plus two facial-only files. The README dates them 2/2022
  (15 visemes, 48 FACS, 30 for the Vive tracker) and adds ARKit-compatible shapes in 6/2022. Read with the project's
  FBX reader, `Male_Adult_05_facial.fbx` (1.98 MB) is one mesh for the whole avatar (4,180 control points) with 175
  blendshape channels: 15 visemes, **52 with Apple's ARKit names** (`AK_25_JawOpen` ...), 48 FACS action units, 42 for
  the Vive tracker, 17 HeadBox extras (jaw, tongue, nostrils) and a neutral *(measured)*. A shape stores offsets for
  about 2,850 vertices. HeadBox made them from the face bones in Maya and transferred them between avatars (HeadBox
  README; Volonte et al. 2022), so they add nothing the bones cannot do, but they give each avatar ARKit-named
  expressions in its own mesh.
- The fetcher downloads only the `*_color.tga` files. The head's normal and specular maps would add 25 MB per avatar
  (50 MB with the body's).

**2048 or more.** At 3.6 texels per millimetre, a render with up to about 225 pixels between the eyes samples at most
one texel per pixel, so 2048 holds up to about 200 px. 1024 is at its limit near 110 px: the 10x window in 2.1 has
118 px and the face looks soft there. 4096 would matter only beyond about 250 px between the eyes, more than any
recogniser or landmarker uses. Memory for an RGBA8 texture with mipmaps: 5.3 MiB at 1024, 21 MiB at 2048, 85 MiB at
4096. A head with colour and normal/specular maps at 2048 needs 43 MiB, so 40 different avatars in view need 1.7 GB
against 0.4 GB at 1024 (computed).

**Detail and pore maps.** The Rocketbox normal maps already carry pore-scale detail, together with the avatar's own
folds, which are part of *that avatar's* identity and would be wrong on someone else's face fitted onto the head.
Measured skin microgeometry (Graham et al. 2013) synthesises 10-micrometre detail from scanned skin patches, mainly
for the look of specular highlights in close-ups; game engines tile a "micro normal" map. For us the useful operation
is a band split: keep the high-frequency band of the avatar's normal map (pores) and attenuate the mid band (folds)
when the head is fitted to a person. All of this is far below what a 112x112 recogniser crop resolves; it changes
how real the face looks, not who it is.

**Texture super-resolution.**

| Model | Form and size | Licence | Note |
|---|---|---|---|
| OpenCV `dnn_superres`: EDSR (x2-x4), ESPCN, FSRCNN, LapSRN | TensorFlow `.pb` graphs, 38.5 MB (EDSR) down to tens of kB | Apache-2.0 (all four model repositories) | The wrapper is in opencv-contrib, which cannot be installed beside the project's opencv-python; the graphs load with `cv2.dnn.readNetFromTensorflow` (not tried with OpenCV 5). EDSR x4: 29.2 dB vs 26.5 dB for ESPCN on OpenCV's test image, at 3.3 s vs 0.01 s |
| ONNX Model Zoo super-resolution (sub-pixel CNN) | ONNX, 240 kB (Git LFS) | Apache-2.0 | x3, luminance only, trained on BSD300 |
| Real-ESRGAN | `.pth` (pickled) | BSD-3-Clause | No official ONNX |
| GFPGAN | `.pth` | Apache-2.0 except listed third-party parts | Generates a face from a prior |
| CodeFormer | `.pth` | S-Lab License 1.0 (non-commercial) | Generates a face from a prior |

Generic super-resolution adds plausible fine detail but no information about the person. Face-prior restorers
(GFPGAN, CodeFormer) generate a plausible face and move identity: CodeFormer's own paper reports that the methods it
compares against fail to preserve identity *(excerpt)*. On a known person's texture they would test the recogniser on
a partly invented face, so they are out. The 2048 Rocketbox maps need no super-resolution at all.

**Photo-based head textures at 2048 or 4096.** The fusion gives full marks at one photo pixel per texel
(`GOOD_FOOTPRINT = 1.0` in `fuse.py`), so a photo must show **about 225 px between the eyes for 2048 and 450 px for
4096**. A typical 12-megapixel phone main camera (26 mm equivalent: about 2,900 px focal length across 4,032 px)
gives 245 px at 0.75 m, 183 px at 1 m and 92 px at 2 m (computed); a 2x lens doubles each distance. So 2048 needs
portraits from about 0.75 m (1x) or 1.5 m (2x); 4096 needs selfie distance or a lens of 3x or more, and at selfie
distance perspective distorts the face (at about 30 cm the base of the nose looks about 30 % wider than at 1.5 m;
Ward et al. 2018, *excerpt*). Focus, motion blur and JPEG quality matter as much as the pixel count. Cost: the head
atlas holds about 60 MB at 1024 (`fuse.py`), so about 240 MB at 2048, and stages 3-4 (4-8 s at 1024) grow with the
number of texels (estimate: 15-30 s at 2048). `created.py` scales saved textures to 1024 (`MAX_TEXTURE`).

**Mipmaps, anisotropic filtering and sRGB.** Filtering gamma-encoded texels darkens fine, high-contrast detail
(lashes, brows, hair) as it is minified. The OpenGL sRGB-texture extension resolves that mipmaps of sRGB textures are
best generated in linear space ("naively ... the result tends to be a subtle darkening") and that conversion before
filtering is preferred but not required (EXT_texture_sRGB, issues 2 and 24), so building the mipmaps ourselves in
linear light is the portable choice. sRGB textures (core since OpenGL 2.1) and sRGB framebuffers (since 3.0) are
present in llvmpipe here and part of macOS's 4.1 core; anisotropic filtering is an extension before 4.6, present in
llvmpipe here. Up close, alpha-tested lashes and hair cards (`discard` below 0.5) show hard, aliased edges; alpha-to-coverage
with MSAA softens them.

### 2.3 High-resolution face meshes

**Why.** With 118 px between the eyes (about 1.9 px per millimetre), a median face edge of 8.5 mm spans 16 pixels:
the jaw line, the nose's profile and the ears show facets. One level of subdivision halves the edges (about 4.3 mm,
8 px), two levels quarter them (about 2.1 mm, 4 px). Shading detail inside the face comes from normal maps; the
silhouette needs geometry.

| Scheme | What it does | Original vertices kept? | Availability |
|---|---|---|---|
| Loop (1987) | Splits each triangle into four, smooths with fixed weights | No: it approximates, and shrinks convex features (lid rims, lips) | trimesh `subdivide_loop` (MIT; positions only, simple boundary rules) |
| Catmull-Clark (1978) | Quad subdivision | No | OpenSubdiv (Pixar, C++; Apache-2.0 with a changed trademark clause); turns triangles into quads |
| Modified Butterfly (Zorin et al. 1996) | Interpolating triangle subdivision | Yes | Own numpy code (small; special rules at boundaries and irregular vertices) |
| PN triangles (Vlachos et al. 2001) | One cubic Bezier patch per triangle from its corners' positions and normals | Yes | Cracks where normals are split (UV seams, hard edges) unless the normals are welded |
| Phong tessellation (Boubekeur and Alexa 2008) | Positions interpolated the way Phong shading interpolates normals | Yes | The cheapest smooth option |
| GPU tessellation shaders | Any of the above, evaluated on the GPU every frame | - | Core since OpenGL 4.0: llvmpipe here offers 4.5; macOS's 4.1 should have them (old reports disagree; not verified), and OpenGL is deprecated on macOS |

**What has to keep working.**

- **Skin weights.** Subdivision stencils are affine combinations, so interpolated weights still sum to one; a new
  vertex can collect more than four bones, so keep the four strongest and renormalise.
- **UV seams.** Subdivide positions on the welded mesh (the heads have 116-318 vertices split by seams), and
  interpolate UVs linearly per face (the layout stays exact) or with face-varying rules as OpenSubdiv does.
- **Eyes, lids and lips.** The eyeballs are closed parts: subdivide them on their own and put new vertices back on the
  eyeball's sphere. Mark the lid rims and the lip line as creases, or an approximating scheme pulls the lids away
  from the eyeballs and blurs the mouth line.
- **Teeth, gums and tongue** are mostly hidden: leave them, or subdivide them rigidly with their bones.
- **The neck seam.** The head mesh's open edge meets the body's collar: keep boundary vertices and new midpoints on
  the original edges, or cracks open.
- **Landmarks and fitted shapes.** The stage-0 landmarks (triangle and barycentric coordinates) and a fitted shape
  (vertex offsets) live on the base mesh: subdivide after applying the shape. With an interpolating scheme the
  original vertices do not move at all.
- **The face rig** moves subdivided vertices through the interpolated weights; nothing else changes.

**Displacement or normal mapping.** No displacement maps exist for the avatars (2.2), and deriving one by integrating
the normal map is not worth it: normal mapping gives the fine shading detail, subdivision the silhouette. Normal
mapping needs a tangent frame per vertex, computed from positions and UVs (MikkTSpace is the common convention for
baking and rendering alike) and recomputed whenever a shape is fitted or the mesh subdivided.

**Cost** *(measured on llvmpipe, 1280x720, 19 people)*: four times the head triangles for everyone (149k to 326k
triangles in the frame) made no measurable difference (173-262 ms against 180-195 ms in other runs; the runs vary
that much); sixteen times (1.03 M triangles) took 489-604 ms, about 2.7 times as long. Subdividing a head once on the
CPU takes milliseconds (estimate). On a Mac GPU the extra triangles are negligible (estimate).

### 2.4 Fitting the head's shape to a person's photos

**What casual photos can tell.**

- **Landmarks are an ambiguous cue.** Within the space of faces there are "flexibility modes" that leave the 2D
  landmarks unchanged, and under perspective two different faces at different distances give the same 2D geometry
  (Bas and Smith 2019). Fitting landmarks alone recovers a set of plausible shapes, not one.
- **Distance and focal length.** Close photos distort: at about 30 cm the base of the nose looks about 30 % wider than
  at 1.5 m (Ward et al. 2018, a model from average face measurements, *excerpt*), and the distance can be estimated
  from a frontal portrait's landmarks (Burgos-Artizzu et al. 2014, with the Caltech Multi-Distance Portraits: 53
  people at 7 distances, *excerpt*). Our registration already found that MediaPipe's landmarks pin the focal length
  down poorly: its search ends near a 50-degree lens whatever the lens was.
- **Scale.** Photos give shape up to scale. The iris is a ruler: its horizontal diameter is about 11.7 +/- 0.5 mm
  across people, which MediaPipe Iris uses to measure distance with a 4.3 % mean error *(excerpt)*. Within one photo
  the ratio needs no focal length: eye distance in mm = 11.7 x eye distance in pixels / iris diameter in pixels, good
  to about 4 % from the population spread alone.
- **Expressions** differ from photo to photo. MediaPipe's blendshape model turns 146 of the 478 landmarks into 52
  coefficients with ARKit's names (model card, verified), the same names as the Rocketbox `AK_` blendshapes.
- **Landmark noise.** MediaPipe's Face Mesh V2 model card reports a 2D mean absolute error of 2.6 % of the eye
  distance when tracking and 3.2 % on single images, against 2.56 % between human annotators; its z coordinate comes
  from synthetic training data and is not evaluated (verified). On an adult face that is about 2 mm per landmark.
- **How different are heads?** *(measured)* On the 20 Rocketbox heads, with MediaPipe's landmarks on their meshes
  (stage 0) and every pair aligned by a similarity transform (190 pairs), the median distance between corresponding
  landmarks is **2.7 mm** (2.0-3.5 mm for 80 % of the pairs): 2.2 mm at the eyes, 2.5 at the lips, 2.9 at the brows,
  3.0 at the nose and **10.3 mm on the face's outline**; the heads' relative scales span 0.88-1.15. Aligned without
  scale, the median is 4.4 mm.

So inside the face, the differences between people are about as large as one photo's landmark noise. Several photos
average the noise down (roughly by the square root of their number), and the outline of the face and the profile,
where heads differ most, are what silhouettes measure. A shape fit from landmarks alone will mostly get proportions
right; the profile and the jaw line need outlines.

**Multi-view landmark fitting.** All photos are fitted together (bundle adjustment). The unknowns are the shape (model
coefficients, a free deformation, or both), a pose and an expression per photo, a focal length per camera session (or
EXIF data) and a global scale. The residuals are the reprojected landmarks (with a robust loss, weighted by
visibility and by each landmark's reliability on the mesh), the jaw-line landmarks of turned faces allowed to slide
along the current silhouette (Zhu et al. 2015 call this "landmark marching": on a turned face those landmarks mark
the silhouette, not fixed points), outlines against the segmentation masks, and priors. With 478 landmarks in five
photos and a few hundred unknowns, Gauss-Newton converges in about a second in numpy (estimate). Dense landmarks are
the strongest form of the idea: Wood et al. (2022) fit a morphable model to about ten times the usual number of
landmarks, covering the whole head with the eyes and teeth, and report state-of-the-art monocular reconstructions,
over 150 frames a second on one CPU thread, and a natural way to combine frames and views *(excerpt; their landmark
model is not released)*. MediaPipe's per-view z values are "relative" depths under a weak-perspective model, learned
from synthetic renders (MediaPipe documentation): useful to start from, weak as evidence.

**3DMMs and their availability.**

| Model | Identity and expression | Geometry | Licence and access | Form | Usable here? |
|---|---|---|---|---|---|
| FLAME 2020 / 2023 | Shape and expression spaces from about 3,800 head scans | Full head with jaw, neck and eyeball joints | Non-commercial research, and no training of networks with it; account on the FLAME site *(excerpt)* | Pickled `.pkl` files | No; the user would register on the FLAME site, download, and convert the arrays to `.npz` on the Mac |
| FLAME 2023 Open (11/2025) | As FLAME 2023 | As FLAME 2023 | CC-BY-4.0, commercial use allowed *(excerpt from the FLAME site; unverified)* | `flame2023_Open.pkl` according to a third-party note *(excerpt)* | No: the site cannot be reached from here; the user would download it from the FLAME site (an account is probably needed) |
| Basel Face Model 2009 / 2017 / 2019 | PCA identity; expressions in the later versions | Face; the later versions cover more of the head (not verified) | Non-commercial; real name, institution and email | `.mat` / `.h5` | No; the user would accept the licence on the Basel site and download |
| **ICT-FaceKit Light** (2020) | **100 identity PCA modes** from light-stage scans (unit variance), **53 ARKit-named expression shapes** | Full head: face (9,409 vertices), head and neck, mouth socket, eye sockets, gums and tongue, teeth, eyeballs, lashes; 26,719 vertices | **MIT** (verified), on GitHub, no registration | 160 OBJ files of 2.56 MB (about 410 MB) | **Yes** |
| FaceScape bilinear model 1.6 | 50 or 300 identity x 52 expression parameters | Face and head, in FaceScape's own topology | Non-commercial, no redistribution; NJU or Google Drive without a key | `.npz` | No (Drive) |
| MediaPipe canonical face | Mean face only (468 vertices, cm); the blendshape model outputs coefficients, not shapes | Face | Apache-2.0 | OBJ in the repository, also inside the `.task` bundle | Yes, but no identity space |
| Rocketbox `_facial` blendshapes | 52 ARKit expressions, 48 FACS, visemes | Each avatar's own mesh | MIT | FBX | Yes, expressions only |

**Learned regressors.**

| Method | Year | Output | Code licence | Weights | Needs | Runs here? |
|---|---|---|---|---|---|---|
| DECA | 2021 | FLAME shape, expression, detail | MPI non-commercial | Pickled `.tar`, Google Drive | FLAME 2020 account; JIT rasteriser or PyTorch3D | No |
| EMOCA | 2022 | FLAME with emotion-aware expression | MPI non-commercial | MPI server | FLAME; PyTorch3D, CUDA | No |
| MICA | 2022 | Metrical FLAME shape from ArcFace features | MPI non-commercial | Pickled `.tar`, Google Drive; insightface `antelopev2` / `buffalo_l` (non-commercial) | FLAME 2020 account | No |
| SMIRK | 2024 | FLAME | MIT | `.pt` (pickled), Google Drive | FLAME account; PyTorch3D built for CUDA | No |
| TEASER | 2025 | FLAME plus tokens | No licence file | `.pt` (pickled), Google Drive | FLAME account | No |
| Pixel3DMM | 2025 | Per-pixel UV coordinates and normals, then a FLAME fit (single or several images) | CC BY-NC 4.0 | Download script | FLAME account; CUDA 11.8 | No |
| SHeaP | 2025 | FLAME with 2D Gaussians, self-supervised | Code not found | - | FLAME | No |
| RealDenseFace | 2026 | Dense UV correspondence and relative depth, then Gauss-Newton FLAME fit (84 fps on an RTX 4090) | Code not found | - | FLAME | No |
| 3DDFA_V2 | 2020 | BFM-derived morphable model | MIT | `.pth` in the repository, ONNX converted from it on first run; BFM as a pickle | CPU, tested on macOS | Not without loading pickles |
| 3DDFA-V3 | 2024 | BFM, guided by face-part segmentation | MIT | `.pth` and a pickled `.npy` on Hugging Face | CPU renderer available | No |
| HRN / MV-HRN | 2023 | BFM plus deformation and displacement maps, a multi-view mode | Apache-2.0 | `.pth`, Google Drive / ModelScope | NVIDIA GPU, nvdiffrast, BFM | No |
| Deep3DFaceRecon | 2019-21 | BFM 2009 | MIT | `.pth`, Google Drive | BFM registration; nvdiffrast | No |
| DAD-3DHeads | 2022 | FLAME head | CC BY-NC-SA 4.0 | - | FLAME | No |

How accurate they are: on the NoW benchmark (scan-to-mesh distance after a landmark-based rigid alignment with scale,
Sanyal et al. 2019) the median errors are 0.90 mm for MICA, 1.09 for DECA, 1.11 for Deep3DFaceRecon and 1.23 for
3DDFA-V2; on the metrical leaderboard (no scale fitted) 0.97 for TokenFace, 1.08 for MICA and 1.35 for DECA
*(excerpts)*. REALY (Chai et al. 2022) aligns region by region and finds the ranking changes by region *(excerpt)*.
The best single-image networks are thus about 1 mm from a scan, but none of them can run here (pickled weights,
FLAME or BFM registration, CUDA), all output a research topology that would still have to be moved onto the
Rocketbox head, and most are non-commercial.

**Silhouettes and photometric terms.** Analysis by synthesis (Blanz and Vetter 1999) fits shape, albedo and light by
rendering the model and comparing it with the photo; modern versions use differentiable rasterisers (nvdiffrast,
PyTorch3D; both CUDA-first) and dense learned cues (Pixel3DMM's UV and normal maps, plus silhouettes). The part that
suits a CPU is the silhouette: a profile photo shows the nose's projection, the chin and the forehead's slope, exactly
the depth that landmarks constrain worst. `register.py` already fits the head's outline to the segmenter's masks (an
IoU pattern search); a distance-transform residual of the outline makes it a least-squares term. A photometric
refinement (shading to normals) could reuse the spherical-harmonics light fitted in `light.py`, but it is the term
most easily fooled by albedo and light: later, if at all.

**Deforming the avatar's own head, without a 3DMM.** Laplacian surface editing (Sorkine et al. 2004) and
as-rigid-as-possible deformation (Sorkine and Alexa 2007) move a few constrained points and spread the change
smoothly while keeping the surface's detail; radial basis functions (thin-plate splines in 3D) do the same without
the mesh; non-rigid ICP (Amberg et al. 2007) pulls a template onto a target surface with a stiffness that is relaxed
step by step; embedded deformation (Sumner et al. 2007) drives the mesh by a coarse graph. All keep the avatar's look
and are cheap, but none knows what a face may look like: noisy landmarks give lumps, and depth comes only from
landmarks triangulated across photos. Libraries: trimesh (MIT; `nricp_amberg`, `nricp_sumner`, `procrustes`; pure
Python) installs anywhere; libigl's Python bindings (MPL-2.0; ARAP and biharmonic solvers) have wheels only up to
Python 3.12 (2.6.3 on PyPI, none for 3.13); OpenMesh's only for 3.9; scipy (BSD-3, wheels for 3.13 on macOS arm64 and
Linux) is not yet a dependency. A head has about 1,700 vertices, so dense numpy linear algebra also does.

**Transferring a fitted shape onto the Rocketbox head.** Keeping the topology keeps the UV layout and the skin weights
exactly. The rest needs care:

- **Correspondence.** A shape fitted in another model's space (ICT-FaceKit, FLAME) reaches the Rocketbox head through a
  correspondence made once per head model: landmarks plus non-rigid ICP give each Rocketbox vertex a point on the
  model's surface. Deformation transfer (Sumner and Popovic 2004) carries per-triangle deformations through such a
  correspondence between meshes of different connectivity; for identity offsets, interpolating the displacement at
  those points is enough.
- **Bones.** The rest positions of the sliding face bones must follow the surface they sit under. SMPL (Loper et al.
  2015) computes its joints as a fixed linear combination of the mesh's vertices (a "joint regressor"); the same per
  face bone, from the vertices it moves most, gives its new position for any deformation. Orvalho et al. (2008)
  transfer a whole facial rig to new face models from landmark correspondences. The pivots matter most: the jaw's
  hinge and the eyeballs' centres.
- **Eyes and lids.** Eyeballs are rigid: move each one (and its bone) to the fitted eye socket, scale it uniformly at
  most, and keep the lid rims around it.
- **Mouth interior.** Teeth, gums and tongue follow the lips and jaw by similarity transforms, never by free
  deformation.
- **Hair cards, the neck and props.** The hair primitive lies on the scalp: the same displacement, extended smoothly
  off the skin, has to move it, or the scalp pokes through. The neck's open edge stays fixed where it meets the body.
  Glasses and hats are props on bones and may need offsets.
- **Animation.** `faces.py` scales its bone slides by the head's size (from the eye distance), so it adapts once the
  eye bones move. The `_facial` blendshapes are absolute offsets for the original head and would need deformation
  transfer, but the demo animates faces with the bones. HeadBox's procedural transfer of blendshapes between Rocketbox
  heads (Volonte et al. 2022) shows that the rig carries over between head shapes.

**How fitting meets the texture pipeline: geometry first, then texture.** The stage-0 landmarks are stored as
triangles and barycentric coordinates, so they stay valid on the deformed head. PnP against the fitted head then
explains the photo's landmarks with the shape instead of the warp: the thin-plate warp, whose job was to fake the
shape difference in 2D, shrinks to small residuals. The fit's cameras can be handed to the registration; the UV bake,
the atlas and the outline labels are keyed by the geometry and are made again; and the identity check (renders of the
textured head against the photos) starts to measure the shape as well as the texture.

### 2.5 Shading that serves recognition

**What recognisers react to.**

- **Light.** Images of one face under different light directions can differ more than images of different faces
  under the same light (Moses, Adini and Ullman 1994); in the journal version no representation they tested overcame
  a change of light direction on its own (Adini et al. 1997) *(excerpts)*. Deep recognisers are far more robust, but
  contrast and cast shadows still move their scores. Kortylewski et al. (2018, 2019) rendered Basel-model faces to
  measure recognisers against pose and light: the networks generalised surprisingly well to unseen light, less well
  across poses when the training identities did not share their pose variation *(excerpts)*.
- **Texture before shape.** Four state-of-the-art recognisers, shown synthetic face pairs whose geometry and texture
  were swapped independently, favoured texture over shape, and their texture-based decisions leaned on the region
  around the eyes (Plesko et al., IEEE FG 2026; project-page summary only, *excerpt*). Eyes, lids, lashes and brows
  deserve the most care; the shape still decides the outline and how the features move with pose.
- **The domain gap.** Recognisers trained only on rendered faces (DigiFace-1M) reach 96.17 % on LFW, against more than
  99.8 % for those trained on real photos; aggressive augmentation narrows the gap (Bae et al. 2023, *excerpt*).
  Wood et al. (2021) showed that rendered faces can replace real ones for landmarks and face parsing when the renders
  vary hair, clothing, accessories and surroundings and are path-traced *(excerpt)*. For testing in a simulation the
  gap cuts both ways: a face that looks rendered may get lower genuine scores against photos, and renders of different
  avatars share an artist, a light and a renderer, which can raise the impostor scores among renders. Both need
  measuring (section 4).

**Shading for skin and eyes.**

| Technique | What it adds | Cost | For recognition |
|---|---|---|---|
| Linear-light (sRGB-correct) shading and mipmaps | Right falloff of light on skin; minified detail keeps its contrast | Free | Matters: shading gradients are what photo-trained recognisers expect |
| Normal maps (the Rocketbox 2048 maps) | Folds, lips, nostrils and pores react to the light | One texture fetch and a tangent per vertex | Folds matter from about 40 px between the eyes; pores are cosmetic |
| Specular (the Rocketbox maps; Blinn-Phong or GGX with Schlick's Fresnel) | Sheen on forehead, nose and lips; wet eyes | A few operations per pixel | Mostly realism; highlights shift the apparent shape a little |
| Wrap lighting or pre-integrated skin (Penner and Borshukov 2011) | Softer, slightly red light-to-shadow edge | One lookup table | Cosmetic; removes the "plastic" look |
| Screen-space or separable subsurface scattering (Jimenez et al. 2009, 2015); texture-space diffusion (d'Eon and Luebke 2007) | Light bleeding through skin | Extra passes | Cosmetic, and costly on llvmpipe |
| Hemisphere or spherical-harmonics ambient light | Light from above, darker from the floor | Cheap | Matters: real halls light faces from above |
| Baked ambient occlusion, shadow maps | Darker eye sockets and nostrils; nose, brow and hair shadows | A bake per head; one depth pass per light | Matters under a strong key light |
| Eyes: corneal highlight, iris detail, lashes with alpha-to-coverage | A living eye region | Cheap | Matters (the region around the eyes) |

### 2.6 Evaluating faces for recognition

- **Controlled synthetic tests.** Kortylewski et al. built a parametric face-image generator on the Basel model to
  vary pose and light while knowing every identity, and published it so recognisers can be benchmarked on common
  ground *(excerpt)*. Our renders allow the same, with avatars or fitted heads as identities.
- **Recognition metrics.** Verification as true-accept rate at a fixed false-accept rate, closed-set identification
  as rank-1 and CMC curves, and open-set identification as NIST's false-negative identification rate at a fixed
  false-positive identification rate (FNIR at FPIR). The companion review details them.
- **Geometry metrics.** NoW aligns a reconstruction to a scan by landmarks (rotation, translation and scale), refines
  the rigid alignment by the scan-to-mesh distance, and reports that distance (median, mean, deviation) (Sanyal et
  al. 2019; the evaluation code's README, verified); a separate metrical leaderboard fits no scale (introduced with
  MICA, *excerpt*). REALY aligns region by region and reports per region (nose, mouth, forehead, cheeks), since one
  global alignment can hide or invent errors *(excerpt)*.
- **The domain gap, measured.** Compare score distributions: photo against photo of the same person (the ceiling),
  render against photo (what the demo needs), render against render of the same identity, and impostor scores within
  each domain; and compare quality scores of renders and photos at matched face size.

## 3. Recommended approach for us

In short: look at each detected person's head through a **virtual high-resolution sensor** (a sub-frustum of the same
camera) that also blurs, adds noise and compresses like a real camera; draw heads with their **2048 colour, normal
and specular maps in linear light**; **fit each known person's head shape** to their photos on the CPU, with the MIT
ICT-FaceKit identity space moved onto the Rocketbox head, by multi-view landmark and outline fitting, keeping the
avatar's topology, UVs, weights and rig; then make the **texture on the fitted head** at 2048.

| Part | Method | Libraries and models (licence, form) | Cost | Where in the code |
|---|---|---|---|---|
| Virtual sensor (3.4) | Off-axis sub-frustum `S @ P`, culling, a camera model | Own code; OpenCV (Apache-2.0) | 11-23 ms per window on llvmpipe *(measured)* | `scene/scene.py`, `scene/renderer.py`, `sources/scene_source.py`, `recognition/engine.py` |
| Textures (3.1) | 2048 head colour, normal and specular maps; mipmaps in linear light | Rocketbox maps (MIT, TGA) | +25 MB download and 43 MiB GPU memory per head at 2048 | `rocketbox.py`, `fbx_convert.py`, `model.py`, `created.py` |
| Meshes (3.1) | Interpolating subdivision of heads seen up close | Own numpy | Milliseconds per head; draw only in windows | New `characters/subdivide.py` |
| Shape fit (3.2) | ICT-FaceKit identity space on each head, multi-view bundle adjustment, outlines, a small free-form correction, transfer to the rig | ICT-FaceKit Light (MIT, OBJ); MediaPipe Face Landmarker and segmenter (Apache-2.0, `.task`/`.tflite` from GCS); trimesh (MIT) for non-rigid ICP | About 1 min once per avatar, 2-5 s per person (estimates) | New `characters/headshape/`; `headtex/register.py`, `created.py`, `recognition/cast.py` |
| Renderer (3.3) | Linear light, normal and specular maps, ambient from above, soft skin, eyes, alpha-to-coverage | Own GLSL | +20-40 % per-pixel cost on llvmpipe (estimate) | `scene/renderer.py`, `scene/textures.py` |

### 3.1 Assets: texture and mesh

**A1. Fetch the heads' normal and specular maps.** All 115 heads have `<id>_head_normal.tga` and
`<id>_head_specular.tga` beside the colour map (12.6 MB each); the fetch lists only `*_color.tga` today. Add them for
heads (bodies optional); the two wrinkle maps are not needed. Cost: +25 MB per avatar, about 1.5 GB for 60 conference
avatars. Code: `rocketbox.Avatar.files`, `_parse_avatars`, `estimate_size`.

**A2. Prepare head maps at 2048, packed.** Prepare the head's colour map at 2048 and a second 2048 RGBA8 map with the
normal's x and y in red and green, the specular level in blue and room for baked ambient occlusion in alpha; bodies
and hair stay at 1024. Store a tangent per vertex (xyz and a handedness sign) computed from positions and UVs on the
welded mesh; `SkinnedPrimitive` gains optional `tangents` and a second texture, and the cache `FORMAT` goes to 2.
Check the normal map's green-channel convention once under a raking light and record it. Memory: 43 MiB per head
at 2048 (10.6 MiB at 1024), so upload the 2048 maps only for heads that a window or a close camera needs (the
renderer's `prepare()` already spreads uploads over frames) and draw the others with a 1024 copy. Code:
`fbx_convert._material_texture` reads only the diffuse and opacity slots (find the maps by name:
`*_head_color.tga` to `*_head_normal.tga` and `*_head_specular.tga`), `rocketbox.save_character` /
`read_character`, `model.SkinnedPrimitive`, `scene/textures.py`.

**A3. Photo textures at 2048.** The head-texture stages already work at the head texture's own size
(`fuse.texture_size`), so preparing heads at 2048 makes them fuse at 2048. Ask for portraits with at least 225 px
between the eyes (from 0.75 m with the main camera, 1.5 m with a 2x lens) and report the share of face texels that
reached one photo pixel per texel. Give a photo-textured head the avatar's normal map with its mid band (the
avatar's folds) attenuated and its pores kept, using the Laplacian pyramid already in `fuse.py`. No super-resolution
on photos or textures of known people. Cost: stages 3-4 about 15-30 s instead of 4-8 s, 240 MB per atlas (estimates;
keep one or two atlases in memory instead of three). Code: `created.MAX_TEXTURE` (2048 for heads), `fuse.ATLASES`.

**A4. Subdivided heads for close-ups.** Once per head model, after any fitted shape is applied: one or two levels of
an interpolating subdivision (modified Butterfly) on the welded face and scalp; eyeballs subdivided on their spheres;
lid rims and the lip line as creases; the neck's boundary kept on its original edges; teeth and tongue left alone;
weights by the same stencils (four strongest kept, renormalised); UVs linear per face; tangents recomputed. Draw the
x4 head in windows up to about 150 px between the eyes and the x16 head beyond (x4 for everybody also costs nothing
measurable on llvmpipe; x16 for everybody costs 2.7 times the frame). Cost: milliseconds per head on the CPU.
Code: a new `characters/subdivide.py`; the `Character` keeps an optional detailed model that window renders use.

### 3.2 Shape fitting

**S0. Once per head model** (cached next to the stage-0 landmarks, keyed by the head's geometry):

1. **Parts of the head mesh**: face skin, eyeballs (vertices led by the eye bones), mouth interior (teeth, gums,
   tongue: the `inner` part of `fuse.py` at vertex level), the neck's open edge, and the hair cards of the opacity
   primitive.
2. **The identity space on this head.** Download ICT-FaceKit Light once (MIT; the neutral mesh and the 100 identity
   OBJs: about 260 MB, kept as a float32 `.npz` of about 32 MB; download it like the avatars rather than committing
   it). Align ICT's neutral head to the avatar's head: a similarity transform from landmarks (ICT lists the vertices of
   the 68 Multi-PIE landmarks; MediaPipe's 478 contain the matching points), then non-rigid ICP (trimesh
   `nricp_amberg`) of ICT's face and head onto the Rocketbox head with the landmarks as constraints. Each Rocketbox
   vertex then has a closest point on ICT's surface, and the 100 identity modes become offsets `B` on the Rocketbox
   head.
3. **The avatar in that space**: `alpha_avatar`, the least-squares coefficients (with the unit-Gaussian prior) that
   best explain the avatar's head; the residual is the artist's detail, which the fit keeps.
4. **Expressions**: the avatar's own 52 `AK_` blendshapes from its `_facial` FBX (MIT, one more 2 MB file per avatar;
   FBX control points mapped to the prepared vertices), named like MediaPipe's 52 coefficients; ICT's 53 shapes
   through the same correspondence are the alternative.
5. **Bone regressors**: each face bone's rest position as a fixed weighted sum of the head vertices it moves most
   (least squares), and each eyeball's centre from a sphere fitted to its vertices.

Cost: one 260 MB download; per avatar about a minute, mostly the non-rigid ICP (estimate); a few MB of cache.

**S1. Register the photos** as now (`register_photos`): first cameras by PnP on the avatar's head, the masks, the
landmarks with their visibility, MediaPipe's blendshape coefficients. The capture advice that helps the fit most: a
frontal photo from 1.5 m or more (a 2x lens), three-quarter views and both profiles, EXIF data kept, a neutral
expression, hair off the forehead and ears.

**S2. Bundle adjustment** (Gauss-Newton or Levenberg-Marquardt in numpy; scipy's `least_squares` if scipy becomes a
dependency):

- shape `x = x_avatar + B (beta - alpha_avatar)` with the person's coefficients `beta` (the first 40 modes first, then
  all 100 with a firmer prior);
- per photo a pose and 52 expression weights, started from MediaPipe's coefficients and pulled towards them and
  towards neutral;
- per camera session a focal length (EXIF data, else free with a prior);
- a scale from the irises (11.7 mm) over all photos where they are large enough, else the avatar's own size, and the
  summary says which;
- residuals: the 468 mesh landmarks with a robust loss, weighted by their stage-0 reliability and visibility;
  jaw-line landmarks re-assigned each iteration to the silhouette of the current head in that view; mouth landmarks
  down-weighted when `jawOpen` is above 0.35; a unit-Gaussian prior on `beta`.

Cost: about a second (estimate).

**S3. Outlines.** Distance-transform residuals between each photo's head mask (hair taken out where it covers the face)
and the outline of the fitted head as that photo's camera sees it (the z-buffer of `raster.py` gives the outline),
alternated with S2 for a few rounds. They carry the profile: the nose's projection, the chin, the forehead. Cost:
1-3 s, like the outline fit in `register.py` (estimate).

**S4. A small free-form correction.** A smooth (bi-Laplacian) deformation of the skin that removes what remains of the
landmark and outline residuals where at least two photos agree, capped at a few millimetres, the neck's edge fixed,
eyeballs and teeth excluded. Without ICT-FaceKit this step alone, with stronger smoothing and landmarks triangulated
across the photos, is the fallback: proportions, but no plausible depth where nothing is seen.

**S5. Transfer to the avatar.**

- Head primitive: per-vertex offsets in the bind pose; eyeballs moved rigidly into the new sockets (their centres
  become the eye bones' positions); teeth, gums and tongue moved by similarity transforms fitted to the displacement of
  the lips and jaw around them; normals and tangents recomputed on the welded mesh.
- Hair cards: the skin's displacement extended to them from the nearest scalp vertices.
- Bones: new rest positions from the regressors, written into `rest_local`; `inverse_bind = inv(rest_global)` for the
  moved bones and their children. The Poser keeps the model's bone offsets, so animation and the face rig follow.
- Checks, automated: the 12 expressions of `faces.py` and the tongue rendered from the front and the side, with no
  eyeball through a lid, no teeth through the lips and no gap at the neck.

Cost: under a second (estimate).

**S6. Texture on the fitted head.** Run the head-texture stages on the fitted model: the stage-0 landmarks carry over
(triangle and barycentric coordinates), registration starts from S2's cameras, the thin-plate warp has little left to
do (its limit can then be tightened), and the bake, atlas and outline labels are made again for the new geometry.
Extend the identity check to report renders of the fitted head against renders of the unfitted one: the measured
value of the shape.

**S7. Storing it.** A created character gets a `shape.npz` (no pickles) beside `character.json`: `format`, `avatar`,
the base head's geometry key (to notice a re-prepared avatar), `head` offsets (N x 3 float32, metres, bind pose),
`hair` offsets, bone names and rest-position offsets, `beta` with the name of the identity space, the scale and the
fit's residuals and focal lengths, but no photos and no photo landmarks: about 30 KB. `created.apply_shape(model,
shape)` returns a per-character copy that shares everything but the head and hair primitives and the skeleton; the
renderer gives those primitives their own buffers by itself (it keys them by object), and `faces.face_rig` works per
skeleton. The known faces of `recognition/cast.py` keep the same file beside their textures in
`faces/.headtex/<name>/`. A fitted face shape is biometric data like the photos: git-ignored folders only, deleted
with the person or character.

### 3.3 Renderer changes

- **R1. Linear light.** Colour textures as `SRGB8_ALPHA8` (moderngl's `internal_format=0x8C43`), mipmaps built on the
  CPU in linear light and uploaded level by level (or the driver's, if they prove linear on llvmpipe and on the Mac),
  shading in linear light, the result encoded to sRGB in the shader (or an sRGB framebuffer through
  `ctx.enable_direct(0x8DB9)`); colours, backgrounds and fog converted too. The fair's look changes, so `ambient`,
  light and fog need re-tuning; the head-texture stages already compute in linear light (`light.py`).
- **R2. Normal and specular maps.** A tangent attribute (`4f`) skinned like the normal; in the fragment shader the
  tangent-space normal from red and green (blue reconstructed) and a specular lobe (Blinn-Phong or GGX, roughness
  about 0.35-0.45 for skin, Schlick's Fresnel with F0 near 0.028, scaled by the specular map), only for primitives
  that have maps.
- **R3. Light for faces.** A hemisphere or spherical-harmonics ambient term (bright above, darker from the floor)
  instead of the constant one; optionally a shadow map for the key light, only in windows (nose, brow, hat and hair
  shadows).
- **R4. Skin.** Wrap lighting or a pre-integrated skin lookup on the head's diffuse term.
- **R5. Eyes and lashes.** A sharp specular lobe on the eyeballs; alpha-to-coverage for the opacity primitive in
  windows (`ctx.enable_direct(0x809E)` around its draw).
- **R6. Detail levels.** In windows: subdivided heads and 2048 maps; in the 720p view: as now.

Cost: a few dozen operations per pixel, perhaps 20-40 % more per-pixel time on llvmpipe (estimate), negligible on the
Mac GPU; all of it behind a quality setting. Code: `renderer.py` (`_SKINNED_VS`, `_OBJECT_FS`, `_gl_texture`,
`_build_mipmaps`), `textures.py` (a colour-space flag on `Texture`; `prepare_upload` builds the mipmaps),
`SkinnedPrimitive.interleaved()`.

### 3.4 The virtual high-resolution sensor

The main view stays at 1280x720 for detection and display (a 4K main view costs 2.7 times as much on llvmpipe and
still leaves people beyond 4 m under 40 px); faces get their pixels from windows.

- **V1. The window camera.** `SubCamera(camera, rect, aspect)` overrides `projection_matrix` with `S @ P` (the version
  measured in 2.1), or `Renderer.render_window(scene, rect, size)` does the same. The rectangle is snapped to the
  pixel grid of a declared sensor (say 7680x4320, six times the 720p view, or a "gigapixel" setting), so that every
  window is exactly a crop of that sensor; MSAA as now.
- **V2. Which windows.** For each person track, the head region from the detector's person box (the top of the box,
  square, with margin for the face detector), not from ground truth; the window's size chosen for about 112-160 px
  between the eyes (estimated from the person's height in pixels), limited by the sensor's resolution; at most a few
  windows per frame (say 4), shared round-robin among tracks, unidentified, frontal and large faces first; all at the
  main view's time.
- **V3. Culling.** Draw only objects whose bounding spheres meet the window's frustum (and the hall): three to five
  times faster here *(measured)*.
- **V4. The camera model**, at the window's own pixel scale: lens blur (a Gaussian of 0.5-1 sensor pixel), motion
  blur from the person's projected speed and an exposure time (a line kernel, or 3-5 renders within the exposure
  averaged), Poisson-Gaussian noise for an exposure and gain, optionally a Bayer mosaic and OpenCV's demosaicing, a
  tone curve, and JPEG at a set quality as a stand-in for video compression. Presets ("ideal", "good 4K", "budget
  1080p"), seeded for reproducible runs.
- **V5. Recognition on windows.** The face detector and recogniser run on each window; positions are mapped back to the
  main frame for tracking and the overlay; the window's scale and camera preset are logged with each result for the
  evaluation.
- **V6. Later, a PTZ camera:** its own position and pan, tilt and zoom (about 60 to 2.3 degrees), slewing at 100-500
  degrees a second, a settling time, one target at a time, rendering ordinary views from where it hangs.

Cost: 11-23 ms per 384-512 px window with culling on llvmpipe, 51-67 ms without *(measured)*; a few milliseconds on a
Mac GPU (estimate); the camera model 1-5 ms per window in OpenCV (estimate). Code: `scene/scene.py` (`Camera`),
`scene/renderer.py` (windows, culling, detail flag), `sources/scene_source.py` (windows with frames), `detection/`
and `recognition/engine.py` (use the windows), `display/overlay.py`.

### 3.5 Order of work

| Step | What | Why first | Testable here |
|---|---|---|---|
| 1 | Virtual sensor, culling, camera model (V1-V5) | The largest gain for recognition; independent of the rest | Yes |
| 2 | Evaluation harness (section 4) and baseline numbers | Every later step must show its effect | Yes |
| 3 | Linear light, 2048 head maps with normal and specular maps (R1, R2, A1, A2) | Cheap; changes what all later tests see | Yes |
| 4 | Shape fit (S0-S7), then photo textures on fitted heads at 2048 (A3) | The user's request; needs 1-3 for evaluation | Yes (renders as photos) |
| 5 | Subdivided heads in windows (A4); skin, eye and light shading (R3-R5) | Realism up close | Yes |
| 6 | PTZ camera (V6) | Only if a PTZ installation is to be modelled | Yes |

## 4. Evaluation plan

The question is not whether the faces look better but whether the recogniser behaves on them as it would on real
people seen by a real camera, and whether each step brings it closer.

### 4.1 Data

- **D1. Avatars as people** (geometry and identity known). Renders of avatar A, at chosen poses, lights, expressions,
  distances and lenses, stand in for A's photos; they are fitted (shape and texture) onto avatar B's head, never A's,
  and the result is compared with A. The 20 prepared avatars give 380 ordered pairs; report same-sex adult pairs,
  cross-sex pairs and child-adult pairs separately.
- **D2. Faces from ICT-FaceKit's identity space** (random coefficients), with a texture borrowed from an avatar through
  the correspondence. Their geometry is exact, but they come from the same space the fit uses: D2 shows the fit at its
  best, D1 where the space does not quite fit (artist-made heads), which is closer to real people.
- **D3. Real people with their consent**: 5-10 enrolment photos each (a frontal one from 1.5 m or more with a 2x lens,
  three-quarter views, profiles, different light) and separate probe photos and short videos. Kept in git-ignored
  folders and never committed; a few people are enough for checks, not for statistics.

The public 3D face sets (NoW, FaceScape, the BFM scans, Stirling) need registration or cannot be reached from here.

### 4.2 Geometry of the fitted heads (D1, D2)

- Landmark error at the 478 landmarks (both heads have them from stage 0) and surface distance over the face (inside
  the face oval, both directions), after a rigid alignment and after a similarity alignment as NoW does: median and
  mean in millimetres, and per region as REALY does (eyes, nose, mouth, cheeks, forehead, outline, the profile curve).
- The starting point is the unfitted head: 2.7 mm median at the landmarks and 10.3 mm on the outline between two
  Rocketbox heads *(measured, 2.4)*. The fit's value is how much of that it removes.
- Sweep the number of photos (1, 3, 5, 8), the views (front only; with three-quarter views; with profiles), focal
  length known or not, the lens and distance (35-85 mm equivalent; selfie distance), the photo's resolution (60-400
  px between the eyes), expressions (neutral, smiling, mouth open) and added landmark noise. Report the scale's error
  (irises against the truth) and the cameras' pose errors too.
- Targets, to be confirmed by the first runs: with five photos including profiles, a median under 1.5 mm inside the
  face and under 3 mm on the outline after a similarity alignment (estimates).

### 4.3 Recognition against face size, pose and light

- A gallery of known people: the real cast enrolled from photos, and D1/D2 identities enrolled from renders of their
  ground-truth heads; unknown people for open-set tests: avatars not enrolled.
- Probes: virtual-sensor windows of the fitted, textured people in the fair, swept over the eye distance (8, 12, 16,
  24, 32, 48, 64, 96, 128 and 192 px), the turn (0-90 degrees in 15-degree steps), the nod (+/-30 degrees), the light
  (the fair's light, a harsh side light, coloured light), expressions (neutral and the `FAR_FACES`), camera presets
  (ideal to budget) and motion (standing, walking).
- Metrics: rank-1 and CMC curves (closed set), true-accept rate at false-accept rates of 1e-2 and 1e-3, false-negative
  identification rate at a false-positive identification rate of 1e-2 (open set), and the raw genuine and impostor
  score distributions, per condition, with bootstrap confidence intervals over identities.
- Recognisers: both profiles of the companion review, plus one that the head-texture pipeline's identity check (SFace)
  does not use, so that no step is tuned to its own judge.

### 4.4 The same identity across photo and render

- Photo against photo of the same real person: the ceiling.
- Render against photo (the fitted, textured avatar seen by the virtual sensor against the enrolment photos): what the
  demo needs; and impostor scores of renders against strangers' photos.
- Render against render of the same identity and of different identities: impostor scores among renders higher than
  among photos would mean the shared rendering style makes different people look alike.
- A realism ratio per person: the median genuine render-to-photo score over the median genuine photo-to-photo score,
  at matched eye distance and pose.
- Face-quality scores (the companion review's assessment models) of renders and photos at matched eye distance:
  renders should not score systematically higher (too clean) or lower (rendering artefacts).

### 4.5 Baselines and ablations

One change at a time, always compared at matched eye distance and pose:

| Configuration | What changes |
|---|---|
| B0 | Today: the 720p view, 1024 maps, the avatar's own head, Lambert shading on gamma-encoded values |
| B1 | + virtual sensor, ideal camera |
| B2 | + camera model presets (scores should drop towards those of real cameras) |
| B3 | + 2048 head maps |
| B4 | + linear light, normal and specular maps |
| B5 | + fitted head shape, the texture fused on the fitted head |
| B6 | + subdivided heads in windows |
| B7 | + skin, eye and ambient shading |

Two controls separate shape from texture: the fitted shape with the texture made for the unfitted head, and the
texture alone at 2048. For D1 the ceiling is avatar A's own head and texture rendered the same way.

### 4.6 Practicalities

- Fixed seeds and frame times; a few frames rendered on llvmpipe and on the Mac compared pixel by pixel.
- Thresholds and weights tuned on half of the identities, results reported on the other half.
- A `bifold-faces evaluate` command that writes CSV files and plots into a git-ignored folder; GL tests marked `gl`
  like the existing ones; the cost (milliseconds per window and per frame) reported beside the accuracy.

## 5. Risks and open questions

1. **The shape may add less than hoped.** Inside the face, people differ by about as much as one photo's landmark noise
   (2.7 mm against about 2 mm, 2.4). Expect the gains on the outline and the profile, and only with several photos
   that include profiles; the evaluation (4.2, 4.5) decides.
2. **Depth and focal length stay ambiguous** without EXIF data and a distant frontal photo: the nose's depth and the
   face's depth proportions may come out wrong while every landmark fits (Bas and Smith 2019).
3. **ICT-FaceKit** was built from light-stage scans whose population the repository does not describe, has not been
   updated since 2020, covers the back of the head less well than the face, and has to be put in correspondence with
   artist-made Rocketbox heads; the download is about 260 MB (410 MB with the expressions).
4. **MediaPipe on renders and on photos.** Stage 0 finds the landmarks on renders of the avatar, registration on
   photos; their biases may differ, and MediaPipe's depth is learned from synthetic data.
5. **The rig after deformation**: lids through eyeballs, teeth through lips, hair cards or glasses off the head. The
   checks in S5 must run on every fitted head; the `_facial` blendshapes would need deformation transfer if they were
   ever used.
6. **The normal maps' convention** (green channel, 3ds Max's tangent basis against ours) may show as seams or inverted
   folds until checked.
7. **Linear light changes the look** of the whole fair and of every constant tuned by eye (ambient 0.62, fog), and the
   head-texture stages that match the avatar's painted light; everything tuned on renders must be checked again.
8. **Memory and time on llvmpipe**: 2048 maps for 40 avatars need 1.7 GB, x16 heads triple the frame time, and the
   photo pipeline at 2048 needs four times the memory and time. Detail levels are not optional here.
9. **The Mac.** OpenGL is deprecated on macOS; sRGB framebuffers, alpha-to-coverage and anisotropic filtering depend
   on the driver. The plan needs nothing beyond OpenGL 4.1 (it avoids tessellation shaders), but none of it has been
   run on the user's Mac yet.
10. **Not verified first-hand:** FLAME 2023 Open's licence terms and file format, the NoW figures, the ISO/IEC and IEC
    values, the FG 2026 texture-before-shape study (a summary only), the DigiFace-1M and "Fake it" figures, the PTZ
    datasheets, and HeadBox's paper (its README was read).
11. **Evaluating with the judge.** The head-texture pipeline already uses SFace to accept or drop generated layers;
    evaluating with SFace alone would flatter it.
12. **A real installation's camera is unknown**: noise, compression, exposure and light. The camera model should be
    calibrated against footage from the intended camera when there is some.
13. **Synthetic-only results can mislead.** Conclusions about real cameras need the real-people checks (D3).
14. **Privacy.** Fitted shapes, textures and embeddings are biometric data: consent, local storage and deletion
    together with the person.

## 6. References

Rocketbox and the repository

- Microsoft Rocketbox avatar library (MIT; README, file listing, `Assets/Editor/FixRocketboxMaxImport.cs`): https://github.com/microsoft/Microsoft-Rocketbox
- HeadBox (Volonte et al., IEEE VR 2022 workshops): https://github.com/openVRlab/Headbox, https://www.microsoft.com/en-us/research/publication/headbox-a-facial-blendshape-animation-toolkit-for-the-microsoft-rocketbox-library/
- The earlier review: `docs/research/face_head_texture.md`; the README sections "Face recognition" and "Head textures from photos"

Pixels, cameras and the virtual sensor

- ISO/IEC 39794-5 and 19794-5 figures *(excerpt)*, via "State of the Art of Quality Assessment of Facial Images" (2022): https://arxiv.org/pdf/2211.08030
- IEC 62676-4 pixel densities *(excerpt)*: Axis white paper https://whitepapers.axis.com/en-us/pixel-density-based-on-iec-62676-4-2014
- ArcFace alignment template: https://github.com/deepinsight/insightface/blob/master/python-package/insightface/utils/face_align.py
- Merino et al., Peepers & Pixels: Human Recognition Accuracy on Low Resolution Faces (2025): https://arxiv.org/abs/2503.20108
- Dodgson, Variation and extrema of human interpupillary distance (SPIE 2004): https://doi.org/10.1117/12.529999
- Hampapur et al., Face Cataloger: Multi-Scale Imaging for Relating Identity to Location (AVSS 2003): https://research.ibm.com/publications/face-cataloger-multi-scale-imaging-for-relating-identity-to-location
- Wang et al., PANDA: A Gigapixel-Level Human-Centric Video Dataset (CVPR 2020): https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_PANDA_A_Gigapixel-Level_Human-Centric_Video_Dataset_CVPR_2020_paper.html
- PTZ datasheets *(excerpts)*: https://m.cdw.com/product/ptzoptics-move-se-pt30x-se-conference-camera/7972784, https://www.cctvsecuritypros.com/content/Support/Blue%20Line%20Spec%20Sheets/PTZ/CSP-IPPTZ30X-4MP%20Datasheet.pdf
- `gluPickMatrix` manual page: https://flex.phys.tohoku.ac.jp/texi/openGL/gluPickMatrix.html; Brian Paul's TR tile-rendering library: https://www.ssec.wisc.edu/~billh/bp/TR.html; Kooima, Generalized Perspective Projection (2008; not opened): https://csc.lsu.edu/~kooima/articles/genperspective/index.html
- Foi et al., Practical Poissonian-Gaussian noise modeling and fitting for single-image raw-data (IEEE TIP 2008): https://dblp.org/rec/journals/tip/FoiTKE08
- Brooks et al., Unprocessing Images for Learned Raw Denoising (CVPR 2019): https://openaccess.thecvf.com/content_CVPR_2019/papers/Brooks_Unprocessing_Images_for_Learned_Raw_Denoising_CVPR_2019_paper.pdf
- Wang et al., Real-ESRGAN (ICCV workshops 2021): https://github.com/xinntao/Real-ESRGAN, https://arxiv.org/abs/2107.10833

Textures and sampling

- Graham et al., Measurement-Based Synthesis of Facial Microgeometry (Eurographics 2013): https://vgl.ict.usc.edu/Research/Microgeometry/
- OpenCV `dnn_superres` and its models: https://github.com/opencv/opencv_contrib/tree/4.x/modules/dnn_superres, https://github.com/Saafke/EDSR_Tensorflow, https://github.com/fannymonori/TF-ESPCN, https://github.com/Saafke/FSRCNN_Tensorflow, https://github.com/fannymonori/TF-LapSRN
- ONNX Model Zoo super-resolution: https://github.com/onnx/models/tree/main/validated/vision/super_resolution/sub_pixel_cnn_2016
- GFPGAN: https://github.com/TencentARC/GFPGAN; CodeFormer: https://github.com/sczhou/CodeFormer, https://arxiv.org/abs/2206.11253
- Ward et al., Nasal Distortion in Short-Distance Photographs: The Selfie Effect (JAMA Facial Plastic Surgery 2018) *(excerpt)*: https://doi.org/10.1001/jamafacial.2018.0009
- EXT_texture_sRGB (issues 2 and 24): https://github.com/KhronosGroup/OpenGL-Registry/blob/main/extensions/EXT/EXT_texture_sRGB.txt

Meshes

- Loop, Smooth Subdivision Surfaces Based on Triangles (MSc thesis, University of Utah, 1987); Catmull and Clark, Recursively generated B-spline surfaces on arbitrary topological meshes (CAD 1978); Zorin, Schroeder and Sweldens, Interpolating Subdivision for Meshes with Arbitrary Topology (SIGGRAPH 1996)
- Vlachos et al., Curved PN Triangles (I3D 2001): https://doi.org/10.1145/364338.364387
- Boubekeur and Alexa, Phong Tessellation (SIGGRAPH Asia 2008): https://www.telecom-paristech.fr/~boubek/papers/PhongTessellation/
- OpenSubdiv (licence: `LICENSE.txt`): https://github.com/PixarAnimationStudios/OpenSubdiv
- trimesh (`remesh.subdivide_loop`, `registration.nricp_amberg`): https://github.com/mikedh/trimesh
- MikkTSpace: https://github.com/mmikk/MikkTSpace

Shape from photos

- Bas and Smith, What Does 2D Geometric Information Really Tell Us About 3D Face Shape? (IJCV 2019): https://arxiv.org/abs/1708.06703
- Burgos-Artizzu, Ronchi and Perona, Distance Estimation of an Unknown Person from a Portrait (ECCV 2014): https://vision.caltech.edu/~mronchi/projects/FaceDistancePortrait
- MediaPipe Iris *(excerpt)*: https://research.google/blog/mediapipe-iris-real-time-iris-tracking-depth-estimation/
- MediaPipe Face Mesh documentation: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/face_mesh.md; model cards: https://storage.googleapis.com/mediapipe-assets/Model%20Card%20MediaPipe%20Face%20Mesh%20V2.pdf, https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Blendshape%20V2.pdf; canonical face: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model.obj
- Zhu et al., High-Fidelity Pose and Expression Normalization for Face Recognition in the Wild (CVPR 2015): https://openaccess.thecvf.com/content_cvpr_2015/papers/Zhu_High-Fidelity_Pose_and_2015_CVPR_paper.pdf
- Wood et al., 3D Face Reconstruction with Dense Landmarks (ECCV 2022): https://arxiv.org/abs/2204.02776
- Blanz and Vetter, A Morphable Model for the Synthesis of 3D Faces (SIGGRAPH 1999)
- FLAME *(excerpts)*: https://flame.is.tue.mpg.de/
- Basel Face Model 2019: https://faces.dmi.unibas.ch/bfm/bfm2019.html
- ICT-FaceKit (MIT): https://github.com/ICT-VGL/ICT-FaceKit; Li et al., Learning Formation of Physically-Based Face Attributes (CVPR 2020): https://arxiv.org/abs/2004.03458
- FaceScape: https://github.com/zhuhao-nju/facescape
- DECA: https://github.com/yfeng95/DECA; EMOCA: https://github.com/radekd91/emoca; MICA: https://github.com/Zielon/MICA; SMIRK: https://github.com/georgeretsi/smirk; TEASER: https://github.com/Pixel-Talk/TEASER; Pixel3DMM: https://github.com/SimonGiebenhain/pixel3dmm
- SHeaP (ICCV 2025): https://openaccess.thecvf.com/content/ICCV2025/html/Schoneveld_SHeaP_Self-Supervised_Head_Geometry_Predictor_Learned_via_2D_Gaussians_ICCV_2025_paper.html; RealDenseFace (2026) *(excerpt)*: https://arxiv.org/abs/2608.09238
- 3DDFA_V2: https://github.com/cleardusk/3DDFA_V2; 3DDFA-V3: https://github.com/wang-zidu/3DDFA-V3; HRN: https://github.com/youngLBW/HRN; Deep3DFaceRecon: https://github.com/sicxu/Deep3DFaceRecon_pytorch; DAD-3DHeads: https://github.com/PinataFarms/DAD-3DHeads
- NoW: Sanyal et al. (CVPR 2019) https://arxiv.org/abs/1905.06817; evaluation code https://github.com/soubhiksanyal/now_evaluation; leaderboards *(excerpts)* https://now.is.tue.mpg.de/
- REALY (ECCV 2022) *(excerpt)*: https://arxiv.org/abs/2203.09729
- Sorkine et al., Laplacian Surface Editing (SGP 2004); Sorkine and Alexa, As-Rigid-As-Possible Surface Modeling (SGP 2007); Sumner, Schmid and Pauly, Embedded Deformation for Shape Manipulation (SIGGRAPH 2007)
- Amberg, Romdhani and Vetter, Optimal Step Nonrigid ICP Algorithms for Surface Registration (CVPR 2007): https://shapemodelling.cs.unibas.ch/gravis-literature/publications/2007/CVPR07_Amberg.pdf
- Sumner and Popovic, Deformation Transfer for Triangle Meshes (SIGGRAPH 2004): https://people.csail.mit.edu/sumner/research/deftransfer/
- Loper et al., SMPL: A Skinned Multi-Person Linear Model (SIGGRAPH Asia 2015): https://smpl.is.tue.mpg.de/
- Orvalho, Zacur and Susin, Transferring the Rig and Animations from a Character to Different Face Models (CGF 2008): https://diglib.eg.org/handle/10.2312/CGF.v27i8pp1997-2012
- Python packages on PyPI: https://pypi.org/project/libigl/, https://pypi.org/project/scipy/, https://pypi.org/project/trimesh/, https://pypi.org/project/openmesh/

Shading and recognisers

- Adini, Moses and Ullman, Face Recognition: The Problem of Compensating for Changes in Illumination Direction (TPAMI 1997): https://faculty.runi.ac.il/moses/papers/adinimosesullman1997.pdf
- Kortylewski et al., Empirically Analyzing the Effect of Dataset Biases on Deep Face Recognition Systems (CVPR workshops 2018): https://arxiv.org/abs/1712.01619; Analyzing and Reducing the Damage of Dataset Bias to Face Recognition with Synthetic Data (CVPR workshops 2019): https://shapemodelling.cs.unibas.ch/gravis-literature/publications/2019/2019-Kortylewski_Analyzing_and_Reducing_CVPRW_2019_paper.pdf
- Plesko et al., Shape vs. Texture: Influence of Geometric Structure and Surface Patterns on Face Recognition (IEEE FG 2026) *(excerpt)*: https://strade.fit.vutbr.cz/en/2026/05/17/shape-vs-texture-influence-of-geometric-structure-and-surface-patterns-on-face-recognition/
- Bae et al., DigiFace-1M (WACV 2023) *(excerpt)*: https://github.com/microsoft/DigiFace1M
- Wood et al., Fake It Till You Make It (ICCV 2021) *(excerpt)*: https://github.com/microsoft/FaceSynthetics
- Penner and Borshukov, Pre-Integrated Skin Shading (GPU Pro 2, 2011); Jimenez et al., Screen-Space Perceptual Rendering of Human Skin (ACM TAP 2009) and Separable Subsurface Scattering (CGF 2015); d'Eon and Luebke, Advanced Techniques for Realistic Real-Time Skin Rendering (GPU Gems 3, 2007)
