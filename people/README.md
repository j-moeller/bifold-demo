# People library

Transparent PNG cut-outs of people, used by the trade-fair scene. Each PNG
has a JSON file next to it with the person's height, pose and the prompt and
seed that made them.

```bash
bifold-people generate -n 30           # generate people (needs: pip install -e ".[people]")
bifold-people import my_photo.jpg      # cut a person out of your own photo
bifold-people list                     # what is here
bifold-people sheet                    # _contact_sheet.jpg with everyone
```

Delete a PNG (and its JSON file) to drop that person. While this folder has
no full-body people, the scene uses default people cut out of the Ultralytics
sample photos instead.

Everything in this folder except this README is ignored by git.
