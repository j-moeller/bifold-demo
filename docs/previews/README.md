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
