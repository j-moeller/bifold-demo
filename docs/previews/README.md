# Previews

Rendered snapshots of the work in progress on the 3D people, refreshed as the
work goes on. The people are avatars from the
[Microsoft Rocketbox library](https://github.com/microsoft/Microsoft-Rocketbox)
(MIT licence), loaded and animated by this project's code.

- `lineup.jpg`: avatars posed by animation clips (walking, a talk gesture, a
  child, a 180° turn half done), rendered with GPU skinning and contact shadows.
- `walk_side.jpg`: one walk cycle seen from the side.
- `fair3d.jpg`: the static 3D sample scene (`bifold_demo.scenes.fair3d`): 30 people
  from 20 avatars in the trade-fair hall, frozen in idle, talk, listen and walk
  poses; nobody faces the camera.
- `fair3d_det.jpg`: the same frame with YOLO's detections.
- `behaviours_frames.jpg`: first animated test with the work-in-progress
  behaviours and a fixed camera (one frame every 2 s): two people walk towards
  the camera, stop and turn around; three walk across the aisles (one a child);
  two chat.
- `approach_closeup.jpg`: close-up of the end of that test: the woman who walks
  up to the camera keeps her head lowered and turned aside, the man behind her
  turns around.
- `fair_animated.jpg`: the default trade-fair scene with 33 animated 3D people
  (at 0, 4, 8 and 12 s) and YOLO's detections (9-14 people found per frame):
  people approaching the camera and turning around, walking by, wandering,
  chatting, staffing booths.
- `fair_closeups.jpg`: close-ups from the same scene after the review fixes:
  exhibitors on their booth platforms, a visitor at a booth, people walking
  towards the camera on the red carpet with their feet in frame and on the
  floor, walkers on the blue aisle carpet with their blob shadows, a chat group
  with a child, wanderers.
- `fair_audience.jpg`: the audience of our exhibit (at 0, 20, 40 and 60 s):
  people standing in a half circle in front of the camera, facing it and
  looking at it, always at least three, with visitors joining and leaving.
- `yoga_yogi_squat.jpg`: the Yogi Squat (Malasana) clip on two avatars: the
  held pose from five sides, and the sequence from standing into the squat and
  back.
- `yoga_cow.jpg`: the Cow Pose (Bitilasana) clip: down from standing to hands
  and knees, the back hollowing with the chest and gaze lifting, and back up.
- `yoga_warrior_1.jpg`: the Warrior 1 (Virabhadrasana I) clip: a long lunge,
  the back foot turned out with the heel down, the arms overhead.
- `yoga_downward_dog.jpg`: the Downward Facing Dog clip: down to hands and
  knees, the toes tucked, the hips lifting into an inverted V with straight
  arms and legs and the heels down.
- `yoga_warrior_2.jpg`: the Warrior 2 (Virabhadrasana II) clip: a wide stance
  across the mat, the front knee over the ankle, the arms level, the gaze over
  the front hand.
- `yoga_upward_dog.jpg`: the Upward Facing Dog (Urdhva Mukha Svanasana) clip:
  down to hands and knees, a step back to plank, then the hips lower, the
  feet roll onto their tops and the chest lifts on straight arms.
- `yoga_chair.jpg`: the Chair Pose (Utkatasana) clip: the arms sweep up
  overhead, then the knees bend deep with the heels down and the hips sitting
  back, the long back leaning forward with the arms in line with it.
- `yoga_goddess.jpg`: the Goddess Pose (Utkata Konasana) clip: the feet step
  out wide and turn out, the arms open to the sides, then the knees bend out
  over the toes with the thighs close to level and the arms bend into cactus,
  palms forward.
- `yoga_tree.jpg`: the Tree Pose (Vrksasana) clip: the weight shifts onto the
  left foot, the right foot comes to the inner left calf and slides up to the
  thigh (the sole against it, toes down, knee open to the side), and the palms
  rise from prayer at the chest to together overhead.
- `yoga_warrior_3.jpg`: the Warrior 3 (Virabhadrasana III) clip: balancing on
  the straight left leg, the body tips forward like a seesaw as the right leg
  lifts straight back (the heel peeling off first) and the arms swing forward,
  into one level line from the hands to the heel.
- `yoga_triangle.jpg`: the Triangle Pose (Utthita Trikonasana) clip: a wide
  stance with both legs straight and the front foot turned out, the hips back,
  the torso tilted over the front leg, the lower hand on the shin (at the ankle
  for the longer-armed avatars), the top arm straight up in line with it, the
  gaze to the top hand.
- `yoga_forward_bend.jpg`: the Standing Forward Bend (Uttanasana) clip: a swan
  dive with the arms wide into a fold over straight legs, the hips over the
  heels, the head hanging, the hands on the floor beside the feet (on the
  fingertips for the shorter-armed women).
- `yoga_plank.jpg`: the Plank Pose (Phalakasana) clip: down to hands and knees,
  the toes tucked, the right foot and then the left one stepping back, the hips
  lowering into one straight line from the heels to the head on straight arms,
  on the balls of the feet, the gaze on the floor a little ahead of the hands.
- `yoga_cat.jpg`: the Cat Pose (Marjaryasana) clip: down from standing to hands
  and knees, the back rounding up towards the ceiling with the tailbone tucked
  under, the arms straight under the shoulders, the head dropped with the gaze
  to the navel, and back up.
- `yoga_child.jpg`: the Child's Pose (Balasana) clip: down from standing to
  hands and knees, then the hips sink back towards the heels (swinging round
  the knees) while the hands slide a little forward, the torso folds over the
  thighs and the forehead comes to the floor, the arms stretched forward, palms
  down; and back up.
- `yoga_cobra.jpg`: the Cobra Pose (Bhujangasana) clip: down to hands and
  knees, a step back to plank, down to the floor through a low plank (the
  elbows bending back, the feet rolling onto their tops), then the chest lifts
  by a backbend on bent arms close to the body while the pelvis, thighs and
  knees stay on the floor, the gaze forward; and back up the same way.
- `selection.jpg`: selection mode in the fair: three of the audience doing Cow
  Pose, Downward Facing Dog and Upward Facing Dog (keys 2, 4 and 6), each at a
  spot they walked to where the camera sees the whole pose; the selected one is
  tagged with what they are doing, and the key help is at the bottom.
- `accessories.jpg`: the Items menu of selection mode in the fair: three of
  the audience dressed by key (a straw hat and a red t-shirt; the welcome sign,
  black glasses and the logo t-shirt; a navy hat, sunglasses, the yoga sign and
  a white t-shirt); the tag of the selected one says what they wear.
- `faces.jpg`: the funny faces of `characters/faces.py` (columns: neutral,
  smile, grin, tongue, puff, surprise, scrunch, fish, wink, sideways,
  cross-eyed, roar, pout) on four avatars: a woman in a headscarf (her cheeks
  puff forward rather than through the scarf), a man, a girl, and a man seen
  from 35°.
- `face_photo.jpg`: `characters/face_photo.py`: a "photo" (here a close-up
  render of another avatar, standing in for a photo of a real person), the
  avatar as it is, with the photo's face painted on, and turned 35°.
- `camera_play.jpg`: the fair's audience playing with the camera, cropped from
  the fixed camera's picture: each pair shows someone just before a move and
  at its most visible moment: waving (a Rocketbox wave clip), making faces
  (thumbs at the temples, wiggling the fingers; the expressions come with
  `characters/faces.py`), stepping to the side, leaning sideways, leaning in
  towards the camera and stepping up closer.
- `outfit_regions.jpg`: the garment regions `characters/outfit/regions.py`
  finds on the body textures of all 20 prepared avatars, drawn over their own
  texture (front and back): skin, top, shirt (an inner layer: a shirt front and
  collar, a hood, long sleeves under a t-shirt, a shirt under a vest), bottoms,
  shoes (also tall boots) and other (belts, a tie and its knot, sleeve stripes,
  a printed number, the black top showing under a short jacket). The three
  marked `*` have an override file: a long dress over trousers and a robe,
  whose skirts the leg bones move, and the loose end of a belt.
- `outfits.jpg`: six avatars (rows) in their own clothes, in four outfit
  presets (business navy, casual denim, trade-fair staff, festival) and in two
  descriptions ("red tartan jacket, black trousers" and "white polka dots on
  black"), front and back. The folds, seams and stitching of the avatars' own
  textures stay; the patterns keep one size across the body and run on round
  it, along the sleeves and down the trouser legs. The fifth row's hoodie is
  sewn from grey, black and navy blocks, and a fill covers all of them; the
  last row's blouse goes on below her belt, over the skirt. A strong print in
  the original fabric (her diamonds) still shows through.
- `headtex_register.jpg`: head textures from photos, stages 0 and 1
  (`characters/headtex`). Top: MediaPipe's 478 face landmarks found on renders
  of three avatars and lifted onto their head meshes, coloured by how well the
  five renders agree (green: well, red: poorly), seen straight on and turned
  30°. Middle: renders of one avatar (Business_Male_02) from known cameras
  standing in for a session's photos, registered to another avatar's head
  model (Male_Adult_05's), as a person's photos are, with the true and the
  found yaw, pitch and roll: three by their landmarks (the front photo's focal
  length given, as a phone's EXIF data would, and shared with the others),
  the right side and a back view turned 150° (named, no face) by the head's
  outline and face skin; the blue wireframe is the head mesh as the found
  camera sees it. Bottom: the Ultralytics sample photo registered to an avatar
  (focal length assumed), the warp that moves the avatar's features onto his
  (arrows three times as long), the avatar seen by the photo's camera, and the
  man on the left, whose profile MediaPipe does not find.
- `created_characters.jpg`: characters from the character creator in the fair
  (`scenes/created_cast.py`), from the fixed camera: the fair's own people at
  14 s; the same moment with Ada (orange hat) in the place of the visitor in
  the middle, waving as that visitor did; at 38 s Ben (sunglasses) in a man's
  place and Carl in a woman's place, his behaviour made again for him, both
  watching the camera; at 62 s Carl walking back up the aisle. Their outfits
  are plain colours put on the avatars' own clothes (no real faces).
- `created_swaps.jpg`: created characters coming into the fair and leaving it
  where nobody sees it (`scenes/created_cast.py`, frame by frame as in the
  demo, cropped from the fixed camera's 1280x720 picture). Above: Gina (red)
  takes over the place of a walker in a long robe while the walker passes
  behind a man of the audience; at the frame of the change nothing of either
  shows, and a second later Gina comes out from behind him. Below: Ada
  (orange), in the place of a visitor who walks up the aisle (`approach2`), is
  deleted; she goes out while Gina walks in front of her, the place stays
  empty, and 5.1 s later the visitor of the place comes back while Dan (green)
  walks in front of her. Plain colour outfits on the avatars' own clothes (no
  real faces).
- `fabric.jpg`: fabrics from a description (`characters/outfit/fabric.py`),
  without a model, since none can run where it was made. Top: the model-free
  seam repair on synthetic textures that do not tile (smooth blotches, blotches
  on a twill weave, grain lit from one side, a tartan and dots whose repeat
  does not fit the swatch), each tiled 2 x 2 as it is and repaired, with its
  seam score. Middle: the fallback ladder, and what its keyword rungs give for
  six descriptions (the procedural fabric of the words the keyword parser
  knows, or a plain colour); the model rungs are grey because they did not
  run. Bottom: three avatars wearing repaired synthetic swatches (not
  generated) on a jacket, a blouse and a blazer, front and back.
- `creator.jpg`: the character creator's screen (`C`, `creator.py`) in a
  headless run of the demo, the fair dimmed behind it: an avatar on the
  turntable in the "highland" preset with houndstooth trousers, a straw hat and
  a welcome sign, and the rows of choices beside it. The shirt's fabric "blue
  paisley silk" is the keyword fabric: no model could run where it was made,
  and the fabric row says why. No face.
- `virtual_sensor.jpg`: the virtual high-resolution sensor (`sensor/`). Top:
  avatars standing in the empty hall facing the camera at 2-30 m, in the
  1280x720 view, each with the place of their window (green where the face
  detector YuNet finds the face in the view, red where it does not). Middle:
  their windows on the 8K sensor (7680x4320), with the eye distance in the
  view and in the window; YuNet finds every face there (green boxes). Bottom:
  the person at 4 m through the camera presets "ideal", "good 8K", "good 4K"
  and "budget 1080p" (the last two on their own 4K and 1080p sensors). Renders
  of avatars only (`docs/research/virtual_sensor_results.md`).
- `high_res_heads.jpg`: high-resolution heads (step S2). Top: three avatars'
  faces in windows of the 8K sensor at 2 m (eyes about 130 pixels apart), as
  the renderer drew them before S2, in linear light with the 1024 textures,
  with the heads' normal and specular maps, and with the 2048 head textures
  as well. Bottom: a close portrait (Male_Adult_05) with the 1024 texture and
  no maps, and with the 2048 texture and its maps: the folds of the eyelids,
  the nose and the forehead, the pores and the sheen of the skin. Renders of
  avatars only (`docs/research/high_res_heads_results.md`).
- `head_shape.jpg`: head shapes from photos (step S3). Avatar A (Male_Adult_12)
  stands in for a person: six renders of it (front, 30 and 60 degrees each
  way, nodded) are the photos. From the left: A itself; avatar B
  (Male_Adult_17) as it is; B's head fitted to A's renders (its own texture);
  and B fitted wearing the face textures made from A's renders on the fitted
  head. Top: the front; bottom: turned 75 degrees. Renders of avatars only
  (`docs/research/head_shape_results.md`).
