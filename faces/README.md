# Known people for the face recognition

Put photos of the people the demo should recognise here, a folder per person,
named as they should be called:

    faces/
      Ada Lovelace/
        portrait.jpg
        holiday.png
      Grace Hopper.jpg      (or a single photo named after the person)

Several photos per person (different days, light and angles: frontal, turned
both ways and in profile) make the recognition surer. Or add them with
`bifold-faces add "Ada Lovelace" portrait.jpg holiday.png`, and check with
`bifold-faces list` and `bifold-faces who some_photo.jpg`.

Each photo is checked: it needs one clearly biggest face, at least 40 pixels
between the eyes (90 or more is best) and a sharp face. Photos that fail are
listed by `bifold-faces list` with the reason and not used; nearly identical
photos of one person count once; a warning says when two people look alike to
the model. Each photo's pose (frontal, turned left or right, profile) is noted,
and a person is matched against a template per pose as well as every photo.

The embeddings are cached here per face model (`.embeddings.<model>.npz`).
They are biometric data like the photos: when a photo leaves the folder, it
leaves every cache. The photos and the caches stay on this computer; this
folder is ignored by git.
