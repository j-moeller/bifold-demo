# Known people for the face recognition

Put photos of the people the demo should recognise here, a folder per person,
named as they should be called:

    faces/
      Ada Lovelace/
        portrait.jpg
        holiday.png
      Grace Hopper.jpg      (or a single photo named after the person)

Several photos per person (different days, light and angles) make the
recognition surer. The biggest face in each photo is used. Or add them with
`bifold-faces add "Ada Lovelace" portrait.jpg holiday.png`, and check with
`bifold-faces list` and `bifold-faces who some_photo.jpg`.

The photos stay on this computer; this folder is ignored by git.
