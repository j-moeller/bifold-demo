# Texture uploads

Drop an image (or video) here whose file name matches an object in the running
scene and it becomes that object's texture immediately. Some names in the
default trade-fair scene:

| File                  | Object                                  |
|-----------------------|-----------------------------------------|
| `booth1_wall.png`     | back wall of booth 1 (… `booth8_wall`)  |
| `booth1_sign.png`     | spinning sign above booth 1             |
| `booth1_screen.png`   | the screen in booth 1                   |
| `stage_screen.jpg`    | the LED wall on the stage               |
| `banner.png`          | the welcome banner above the stage      |
| `floor.jpg`           | the hall floor                          |
| `visitor1.png`        | a visitor (use a transparent PNG)       |

Overwrite the file to update the texture again; delete it to restore the
original. Matching is case-insensitive and any common image or video format
works (`.jpg`, `.png`, `.webp`, `.mp4`, `.mov`, ...). The terminal tells you if a
file matches no object.

Everything in this folder except this README is ignored by git.
