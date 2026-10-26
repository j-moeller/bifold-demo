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