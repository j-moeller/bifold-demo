"""Validate the gender and age change on photos of real people, with judges that did not make it.

    python validation/validate.py --models ~/.cache/bifold-morph --student weights/student.pt \
        --out validation/results photo1.jpg photo2.jpg ...

For every face in the photos:

* the gender slider is pushed to the other end (a man towards female, a woman towards male), and to half way;
* the age slider to +30 and -20 years.

Each result is read by InsightFace's gender/age model (never used in making the change: the independent judge),
by FairFace (whose features steered the student's training: a second opinion, not an independent one), and by
ArcFace, whose cosine similarity to the original says whether it is still the same person (above ~0.4) or someone
else. Writes ``results.md`` (the tables) and one contact sheet per face.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bifold_morph import student as student_mod  # noqa: E402
from bifold_morph import warp  # noqa: E402
from bifold_morph.editor import FaceEditor  # noqa: E402
from bifold_morph.faces import FaceDetector  # noqa: E402
from bifold_morph.nets import ArcFace, FairFace, GenderAge  # noqa: E402
from bifold_morph.state import Settings  # noqa: E402

GENDER_STEPS = (0.5, 1.0)  # slider positions towards the other gender
AGE_STEPS = (30.0, -20.0)


def tile(image: np.ndarray, box, label: str, side: int = 200) -> np.ndarray:
    x, y, w, h = box
    c = int(max(w, h) * 1.9)
    cx, cy = int(x + w / 2), int(y + h / 2)
    padded = cv2.copyMakeBorder(image, c, c, c, c, cv2.BORDER_CONSTANT)
    t = cv2.resize(padded[cy + c - c // 2:cy + c + c // 2, cx + c - c // 2:cx + c + c // 2], (side, side))
    cv2.rectangle(t, (0, 0), (side, 18), (0, 0, 0), -1)
    cv2.putText(t, label, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return t


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("photos", nargs="+")
    p.add_argument("--models", required=True, help="folder with the ONNX judges and YuNet")
    p.add_argument("--student", required=True)
    p.add_argument("--out", default="validation/results")
    args = p.parse_args(argv)
    m, out = Path(args.models), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    det = FaceDetector(m / "face_detection_yunet_2023mar.onnx")
    ff, ga, arc = FairFace(m / "fairface.onnx"), GenderAge(m / "genderage.onnx"), ArcFace(m / "arcface_w600k_r50.onnx")
    editor = FaceEditor(student_mod.load(args.student), det)

    gender_rows, age_rows = [], []
    for photo in args.photos:
        image = cv2.imread(photo)
        for k, face in enumerate(det.detect(image)):
            only = FaceEditor(editor.student, _Fixed([face]))
            r_ff, r_ga = ff.read(image, [face.landmarks])[0], ga.read(image, [face.box])[0]
            female = (r_ff.female + r_ga.female) / 2 >= 0.5
            e0 = arc.embed(image, [face.landmarks])[0]
            name = f"{Path(photo).stem}#{k}"
            tiles = [tile(image, face.box, f"original: {'F' if female else 'M'}")]
            for s in GENDER_STEPS:
                # the slider: -1 is female, +1 male
                res = only.process(image, Settings(gender=s if female else -s, readout=False)).image
                a, b = ff.read(res, [face.landmarks])[0], ga.read(res, [face.box])[0]
                ident = float(arc.embed(res, [face.landmarks])[0] @ e0)
                gender_rows.append((name, "F" if female else "M", s, r_ga.female, b.female, r_ff.female, a.female, ident))
                tiles.append(tile(res, face.box, f"{'to M' if female else 'to F'} {s:g}: IF {b.female:.2f} FF {a.female:.2f}"))
            for years in AGE_STEPS:
                res = only.process(image, Settings(age=years, readout=False)).image
                a, b = ff.read(res, [face.landmarks])[0], ga.read(res, [face.box])[0]
                ident = float(arc.embed(res, [face.landmarks])[0] @ e0)
                age_rows.append((name, years, r_ga.age, b.age, r_ff.age, a.age, ident))
                tiles.append(tile(res, face.box, f"{years:+.0f} y: IF {b.age:.0f} FF {a.age:.0f}"))
            cv2.imwrite(str(out / f"{name.replace('#', '_')}.jpg"), np.hstack(tiles))

    lines = ["## Gender", "", "P(female) before → after; identity = ArcFace cosine to the original.", "",
             "| Face | Was | Slider | InsightFace | FairFace | Identity |", "|---|---|---|---|---|---|"]
    for n, was, s, gb, ga_, fb, fa, ident in gender_rows:
        lines.append(f"| {n} | {was} | {s:g} | {gb:.2f} → {ga_:.2f} | {fb:.2f} → {fa:.2f} | {ident:.2f} |")
    for s in GENDER_STEPS:
        rows = [r for r in gender_rows if r[2] == s]
        flip_if = np.mean([(r[4] >= 0.5) != (r[1] == "F") for r in rows])
        flip_ff = np.mean([(r[6] >= 0.5) != (r[1] == "F") for r in rows])
        lines.append(f"\nSlider {s:g}: InsightFace reads the other gender in **{flip_if:.0%}** of {len(rows)} faces, "
                     f"FairFace in {flip_ff:.0%}; identity {np.mean([r[7] for r in rows]):.2f} on average.")
    lines += ["", "## Age", "", "| Face | Change | InsightFace age | FairFace age | Identity |", "|---|---|---|---|---|"]
    for n, years, b0, b1, f0, f1, ident in age_rows:
        lines.append(f"| {n} | {years:+.0f} y | {b0:.0f} → {b1:.0f} | {f0:.0f} → {f1:.0f} | {ident:.2f} |")
    for years in AGE_STEPS:
        rows = [r for r in age_rows if r[1] == years]
        lines.append(f"\n{years:+.0f} years: InsightFace reads {np.mean([r[3] - r[2] for r in rows]):+.1f} years on "
                     f"average, FairFace {np.mean([r[5] - r[4] for r in rows]):+.1f}; identity "
                     f"{np.mean([r[6] for r in rows]):.2f}.")
    (out / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


class _Fixed:
    """A detector that returns the given faces (to edit one face of a photo at a time)."""

    def __init__(self, faces):
        self.faces = faces

    def detect(self, image):
        return list(self.faces)


if __name__ == "__main__":
    sys.exit(main())
