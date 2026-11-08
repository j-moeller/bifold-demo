# High-resolution heads: measured results

Step S2 of [face_recognition_focus.md](face_recognition_focus.md): the heads' 2048-pixel colour textures with their
normal and specular maps, linear light, normal-mapped and specular shading, the fair's light re-tuned. The design is
sections 3.1 (A1-A3) and 3.3 (R1-R3) of [high_res_faces.md](high_res_faces.md); the code is `characters/tangents.py`,
`characters/rocketbox.py` and `fbx_convert.py` (cache format 2), `scene/textures.py`, `scene/renderer.py` and
`scene/detail.py`, `characters/headtex/fuse.py` and `characters/created.py` (README: "High-resolution heads"). The
preview is [docs/previews/high_res_heads.jpg](../previews/high_res_heads.jpg).

Measured on 8 November 2026 in the development sandbox: a 4-core CPU shared with other jobs, Mesa 25.2.8 llvmpipe
(OpenGL 4.5), the 20 Rocketbox avatars prepared in format 2. Timings here vary by a factor of up to two between runs;
the ratios are steadier. The scratch scripts are not in the repository.

## In short

- **The faces get their shape back.** With the normal and specular maps the folds of the eyelids, the nose, the
  nasolabial folds and the lips catch the light, and the skin has a sheen where the artists painted it (the preview).
  This changes faces more than anything else in S2: to the face embedders a window's face moves to a cosine of 0.86
  (AdaFace; 0.70 at worst) from the same window drawn flat, and the identities separate a little better (d'
  +0.3-0.4).
- **2048 changes little for today's embedders.** They see a 112-pixel crop with the eyes 35 pixels apart, so a
  window's face (eyes 52-132 pixels apart in these measurements) is shrunk before they see it, and the 1024 texture
  (112 texels between the eyes) already holds what reaches them: from 1024 to 2048 the embeddings move to a cosine of
  0.998, and the separation does not change. The 2048 textures show where a picture shows the face larger than about
  110 pixels between the eyes: the close portrait of the preview, S4's close-ups, crops for other models.
- **Detection and separation stay perfect** on this synthetic set (20 avatars facing the camera at 2-5 m, turned up
  to 25 degrees, ideal camera): YuNet and SCRFD find 60 of 60 faces per distance in every configuration, the equal
  error rate is 0 and rank-1 100 % for AdaFace IR-101, ArcFace R50 and SFace.
- **Cost.** Linear light and the maps make the 1280x720 view 1.2-1.3 times and the head windows 1.4 times slower on
  the software renderer; the maps add 107 MiB of textures for the 20 heads, the detail textures 43 MiB per head
  drawn with them (4 heads in the fair at 6 s). A face from photos at 2048 takes 2-3.5 times as long as at 1024.
- **Linear light** keeps the fair's look (3.2 levels from before over the whole view, with the fog re-tuned) and
  moves the embeddings to a cosine of 0.992.

## 1. The assets

The 20 avatars of the library, prepared in format 2 (`bifold-characters fetch`, which now also downloads each head's
normal and specular maps): 367 MB on disk instead of 80 MB (18 MB an avatar instead of 4). Each head keeps its colour
texture and a packed map (normal x and y, the specular level, room for ambient occlusion) at 2048 pixels, 1024 copies
of both, and a tangent per vertex; bodies, hair and props stay at 1024 without maps. Format-1 caches still load and
draw as before.

The maps' green channel: a normal map baked from a height field holds a gradient, which has no curl unless one of its
components is read backwards. At the scale of the folds (a blur of 4 texels at 2048) the residual curl of the 20
heads' maps is 0.33-0.56 read green-up and 0.85-0.96 read green-down, so all 20 are green-up (`+v`); a raking light
from above confirms it (read the other way, the folds look lit from below). One head (`Male_Adult_04`) has a fine skin
texture that reads the other way below about 2 texels; its folds read like every other head's, and the check uses the
folds.

## 2. Linear light

Textures are sRGB textures, their mipmaps are averaged in linear light (with the coverage of cut-outs), shading, fog
and blending happen in linear light, and the picture is an sRGB framebuffer. A fine black-and-white pattern seen small
is now the grey of half the light (level 188, not 128). Surfaces in full light and in shadow look as before; the
fall-off between them is lighter, as on a real curved surface. The fair's fog, which veils dark distant surfaces sooner
when it mixes light, went from 0.55 to 0.2: over the whole 1280x720 view at 6 s the picture is then within 3.2 levels
of how it looked (mean absolute difference), the faces practically unchanged.

The head-texture pipeline finds its landmarks and head regions on renders of the avatar, and registration magnifies
small changes in them: in linear light the landmarks moved by a tenth of a millimetre (the median), which shifted the
camera fitted to a small real photo enough to drop the share of its face texels worth refining from 0.13 to 0.08
(a test with real photos caught it). Those renders are now drawn on the stored values, at the low quality and with
the 1024 textures, whatever the renderer's settings, and are the same as before to the bit.

## 3. Frame and window times, memory

The fair at 6 s (36 people), its 1280x720 view and the 34 head windows S1 asks for (`requests_for`, the "heads"
preset, 112 px faces), a fresh renderer per configuration, medians of three rounds:

| Configuration | View (range) | 34 windows | One window | Textures after the view, after the windows |
|---|---|---|---|---|
| Before S2 (stored values, flat, 1024) | 258 ms (202-281) | 894 ms | 16 ms | 336 MiB, 336 MiB |
| Linear light, flat, 1024 | 319 ms (280-345) | 1138 ms | 20 ms | 336 MiB, 336 MiB |
| Linear light, maps, 1024 | 309 ms (288-332) | 1286 ms | 25 ms | 443 MiB, 443 MiB |
| Linear light, maps, 2048 where needed (the default) | 341 ms (296-375) | 1270 ms | 25 ms | 443 MiB, 614 MiB |

- The view barely uses the maps (its faces are small) and pays mostly for linear light (the sRGB framebuffer, which
  llvmpipe encodes and blends in software); the windows, mostly face, pay for both.
- The packed 1024 maps take 107 MiB for the 20 heads (5.3 MiB each with mipmaps). A head drawn with its 2048
  textures takes 43 MiB more (colour and maps); at 6 s, 4 of the 36 people have their eyes 64 pixels or more apart in
  a window (`Renderer.detail_from`), and at most `detail_heads` (6) heads keep them.
- Reading a head's two 2048 PNGs and laying out their mipmaps takes 445 ms (422-611 for 6 heads): the live demo does
  it in the background and draws the 1024 textures until it is done; tests and measurements wait for it
  (`Renderer.detail_wait`).

## 4. Faces in the windows: detection and identity

Each of the 20 avatars stood alone in the fair's hall facing the camera, at 2, 3 and 5 m (eyes 132, 87 and 52 pixels
apart in the 8K sensor's windows), its head turned -25, 0 and 25 degrees; a window of the 8K sensor ("ideal"
camera) was drawn on its head in five configurations: before S2, linear light at 1024, with the maps at 1024, linear
light at 2048 and with the maps at 2048 (2048 only where the eyes are 64 pixels or more apart: at 2 and 3 m). Per
window: YuNet and SCRFD; SCRFD's five landmarks align the face (112 pixels); AdaFace IR-101, ArcFace R50 (InsightFace's
buffalo_l) and SFace embed it. Genuine scores: each window against the same avatar's frontal 2 m window; impostor
scores: against the 19 other avatars' frontal 2 m windows.

Detection: YuNet and SCRFD found all 60 faces at each distance in every configuration, with the same scores (YuNet
0.933-0.947, SCRFD 0.868-0.871). Identity: in every configuration the equal error rate is 0 and every window's nearest
gallery face is its own avatar; the separation (d' of genuine against impostor scores, AdaFace / R50 / SFace):

| Configuration | 2 m (eyes 132 px) | 3 m (87 px) | 5 m (52 px) |
|---|---|---|---|
| Before S2 | 11.3 / 11.3 / 8.5 | 10.6 / 10.6 / 8.4 | 10.7 / 10.6 / 8.3 |
| Linear light, 1024 | 11.1 / 11.2 / 8.6 | 10.4 / 10.5 / 8.4 | 10.4 / 10.5 / 8.4 |
| Linear light, maps, 1024 | 11.5 / 11.7 / 8.6 | 10.7 / 10.7 / 8.5 | 10.8 / 10.8 / 8.6 |
| Linear light, 2048 | 11.2 / 11.2 / 8.6 | 10.4 / 10.5 / 8.4 | 10.5 / 10.5 / 8.4 |
| Linear light, maps, 2048 | 11.6 / 11.7 / 8.6 | 10.7 / 10.8 / 8.5 | 10.8 / 10.9 / 8.6 |

How far the same window's embedding moves (mean cosine, AdaFace; R50 and SFace alike): linear light against before
0.992; the maps against flat 0.864 (0.70 at worst); 2048 against 1024 0.998 at 2 m, 0.999 at 3 m and 1.000 at 5 m
(the same textures there). In the windows' pixels (two of the avatars at 2 m), 2048 against 1024 differs by 0.4-1.1
levels on the face (mean absolute), the maps against flat by 3.5-6.2.

So, for recognition as it is built now, the maps matter and the 2048 textures do not: the embedders' crops (the eyes
35 pixels apart) are smaller than the faces in these windows, and the 1024 texture already has about 112 texels
between the eyes. The synthetic set saturates the separation (no errors anywhere), so these numbers say how faces
change, not how much recognition improves; that needs harder conditions (the camera presets, turned and lowered
heads, many more identities: E1).

## 5. Faces from photos at 2048

On a head prepared at 2048 the head-texture stages make the face at 2048 (`fuse.texture_size`), and a created
character keeps it at that size. Four fake photos (1280x960, 0.7 m, under four lights) of one avatar fused onto
another's head (`Female_Adult_04`), on the shared machine:

| | 1024 | 2048 |
|---|---|---|
| The head's atlas (once per avatar, cached on disk) | 3.8 s, 66 MiB | 10.6 s, 264 MiB |
| Stages 3 and 4 (projection, light and gains, blend, finish) | 5.3 s (2.9, 1.0, 1.0, 0.4) | 18.8 s (10.1, 3.7, 3.1, 1.9) |
| Stages 5 to 7 (completion, body, checks; no generative layer) | 7.6 s | 16.3 s |
| Growth of the process's peak memory | 0.18 GB | 1.26 GB |
| The avatar's own photos fused back onto its head: face texels kept, delta E median and 90 % | 0.54, 2.9, 7.5 | 0.54, 3.3, 8.1 (3.0, 7.5 compared at 1024) |

The fusion marks a photo's resolution per texel of the head's ordinary texture (`fuse.marks_size`: 1024), whatever
size it works at. Counted per texel of the 2048 texture, photos with less than two pixels per 1024 texel lost up to
29 % of their weight against the avatar's own texture and much of their confidence: on a 480-pixel portrait the share
of face texels at "refine" or better was 0.13 at 1024 and 0.006 at 2048, and stage 5 would have filled in nearly the
whole face. With the marks of 1024 the weights, the confidence and the trimap are the same at both sizes (a synthetic test:
the confidence within 0.0001, the trimap's shares within 0.001).

## 6. Open issues

- **Back photos in the head-texture pipeline.** The outline fit of a photo without a face also lines up the face skin
  the segmenter finds (`register.SKIN_WEIGHT` 1); seen from behind that is the ears and the nape, and it moves with
  the shading: on the fake back photo of `Male_Adult_05` the fit's head overlap is 0.93 with flat shading and 0.86
  with the maps (0.97 by the outline alone). `test_side_and_back_photos_are_placed_by_the_head_outline` draws its
  photos flat, as when its bounds were measured; the fit's sensitivity belongs to the pipeline.
- **Hair painted on the head.** Short hair painted on the head texture gets the head's specular map, which the
  artists painted with bright strands; under a light from in front and above, the crown seen from behind shows a
  grazing sheen, and on the coarse head mesh (1713 vertices on `Male_Adult_05`; the vertex normals of a crown triangle
  differ from its face normal by 15 degrees, the median) it comes out in patches. There is no hair shading model (an
  anisotropic lobe along the strands); S4's subdivided heads smooth the normals.
- **Detail levels in the live demo**: the first windows of a head that newly needs its 2048 textures show the 1024
  ones for about half a second, while they are read in the background.
- **The yoga clip caches** are made again once: their key covers the character model's source, which changed.
- **Ambient occlusion**: the packed map's alpha is 255, room for a baked occlusion map (the Rocketbox heads have
  none).
