"""``bifold-scene``: render a scene at a moment, and the virtual high-resolution sensor's windows of it.

    bifold-scene windows [--scene SPEC] [--time T] [--people truth|detect|heads] [--preset NAME ...] [--faces]
                         [-o DIR]

renders the scene (the trade fair by default) at ``--time`` seconds, as the demo would show it, asks for a window on
every person in the picture and writes ``view.png`` (the view with the windows' places), one image per window and
camera preset, ``sheet.png`` (the windows side by side) and ``windows.json`` (each window's place, level, scale, time
and what the scene knows of the person in it). Where the windows go:

- ``truth``: the boxes of the people's posed skeletons, as a perfect person detector would give them;
- ``detect``: YOLO's person boxes (``--model``);
- ``heads``: the people's heads in the scene (the evaluation's way: a square of ``--head-size`` metres).

``--faces`` runs the face detector (YuNet) on the view and on every window and says, per person, how far away they
are, how many pixels lie between their eyes in the view and in the window, and whether the detector found the face.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from ..display.overlay import FONT
from .camera_model import camera_preset
from .geometry import DEFAULT_SCALE, Sensor
from .requests import DEFAULT_FACE_PX, DEFAULT_MAX_PIXELS, WindowRequest
from .service import Motion, eye_points, render_windows


def _size(text: str) -> tuple[int, int]:
    try:
        w, h = text.lower().split("x")
        return int(w), int(h)
    except ValueError:
        raise argparse.ArgumentTypeError("size must look like 1280x720") from None


def _preset(text: str) -> str:
    try:
        return camera_preset(text).name
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def advance(scene, until: float, fps: float = 30.0, motion: Motion | None = None) -> None:
    """Advance ``scene`` from its start to ``until`` seconds in steps of ``1 / fps``, as the demo's headless runs do
    (``motion`` notes everyone's eyes at each step, for their speed)."""
    steps = max(0, round(until * fps))
    for k in range(steps + 1):
        scene.update(k / fps, 1.0 / fps if k else 0.0)
        if motion is not None:
            motion.observe(scene)
    for obj in scene:  # their textures decoded, so the frame shows them (the renderer reports one it cannot use)
        for job in getattr(obj, "texture_jobs", ()):
            job.exception()  # waits for the job without raising its error


def people_in_view(scene, size) -> list:
    """The visible people (with eyes) whose eyes are in the picture, nearest first."""
    camera = scene.camera
    clip_from_world = camera.projection_matrix(size[0] / size[1]) @ camera.view_matrix()
    found = []
    for obj in scene:
        if not obj.current_visible:
            continue
        eyes = eye_points(obj)
        if eyes is None:
            continue
        clip = np.c_[eyes, np.ones(2)] @ clip_from_world.T
        if (clip[:, 3] <= 1e-6).any():
            continue
        ndc = clip[:, :2] / clip[:, 3:]
        if (np.abs(ndc) <= 1).all():
            found.append((float(np.linalg.norm(eyes.mean(axis=0) - camera.current_position)), obj))
    return [obj for _distance, obj in sorted(found, key=lambda pair: pair[0])]


def eyes_in_view(scene, person, size) -> np.ndarray:
    """``(2, 2)``: ``person``'s eyes in the picture's pixels."""
    camera = scene.camera
    clip = np.c_[eye_points(person), np.ones(2)] @ (camera.projection_matrix(size[0] / size[1])
                                                     @ camera.view_matrix()).T
    ndc = clip[:, :2] / clip[:, 3:]
    return np.c_[(ndc[:, 0] + 1) / 2 * size[0], (1 - ndc[:, 1]) / 2 * size[1]]


def requests_for(scene, size, how: str, face_px: float, max_pixels: int, head_size: float = 0.5) -> list:
    """Window requests for the people in the picture: ``how`` = ``"truth"`` (their skeleton boxes) or ``"heads"``
    (their heads in the scene); YOLO's boxes come from :func:`detected_requests`."""
    from ..selection import person_box

    people = people_in_view(scene, size)
    if how == "heads":
        return [WindowRequest(p.name, head=tuple(eye_points(p).mean(axis=0)), head_size=head_size, face_px=face_px,
                              max_pixels=max_pixels, subject=p) for p in people]
    if how == "truth":
        camera = scene.camera
        clip_from_world = camera.projection_matrix(size[0] / size[1]) @ camera.view_matrix()
        requests = []
        for person in people:
            box = person_box(person, clip_from_world, size)
            if box is not None:
                requests.append(WindowRequest(person.name, box=box, face_px=face_px, max_pixels=max_pixels,
                                              subject=person))
        return requests
    raise ValueError(f"unknown way to place windows: {how!r}")


def detected_requests(view: np.ndarray, model: str, face_px: float, max_pixels: int) -> list:
    """Window requests for YOLO's person boxes in ``view`` (tags 1, 2, ... left to right)."""
    from ..detection import YoloDetector

    detector = YoloDetector(model, device="cpu", classes=["person"])
    boxes = sorted((d.box for d in detector.detect(view)), key=lambda box: box[0])
    return [WindowRequest(k + 1, box=box, face_px=face_px, max_pixels=max_pixels) for k, box in enumerate(boxes)]


def match_face(faces, eyes, size: float | None = None, tolerance: float = 0.35):
    """The face (of ``engine.detect``) whose two eye landmarks lie on ``eyes`` (``(2, 2)``, either order) within
    ``tolerance`` times ``size`` (the face's frontal eye distance; by default the eyes' own distance), and at least 2
    pixels, or ``None``."""
    eyes = np.asarray(eyes, dtype=np.float64)
    reach = max(2.0, tolerance * (float(np.linalg.norm(eyes[0] - eyes[1])) if size is None else size))
    best = None
    for face in faces:
        found = np.asarray(face.landmarks[:2], dtype=np.float64)
        miss = min(np.linalg.norm(found - eyes, axis=1).max(), np.linalg.norm(found[::-1] - eyes, axis=1).max())
        if miss <= reach and (best is None or miss < best[0]):
            best = (miss, face)
    return None if best is None else best[1]


def _face_size(face) -> float:
    return float(np.linalg.norm(np.subtract(face.landmarks[0], face.landmarks[1])))


def cmd_windows(args) -> int:
    from ..loader import load_scene
    from ..scene.renderer import Renderer

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    scene, _path = load_scene(args.scene)
    motion = Motion()
    advance(scene, args.time, args.fps, motion)
    size = args.size
    renderer = Renderer(*size)
    try:
        view = renderer.render(scene)
        sensor = Sensor(*args.sensor, size) if args.sensor else Sensor.scaled(DEFAULT_SCALE, size)
        if args.people == "detect":
            requests = detected_requests(view, args.model, args.face_px, args.max_pixels)
        else:
            requests = requests_for(scene, size, args.people, args.face_px, args.max_pixels, args.head_size)
        results = []
        for preset in args.preset or ["ideal"]:
            results += render_windows(scene, renderer, requests, sensor, frame=round(args.time * args.fps),
                                      motion=motion, preset=preset, visibility=True)
    finally:
        renderer.release()
    print(f"[scene] {args.scene or 'the fair'} at {scene.time:.2f} s: {len(requests)} people asked for, "
          f"{len(results)} windows ({time.perf_counter() - started:.1f} s)")

    engine = None
    if args.faces:
        from ..recognition.engine import FaceEngine

        engine = FaceEngine(args.models)
    view_faces = engine.detect(view, min_size=0) if engine else []
    report, sheet = [], []
    annotated = view.copy()
    for result in results:
        x0, y0, x1, y1 = (round(v) for v in result.view_rect)
        cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 200, 255), 1, cv2.LINE_AA)
        cv2.putText(annotated, str(result.tag), (x0 + 2, y0 + 12), FONT, 0.4, (0, 200, 255), 1, cv2.LINE_AA)
        entry = {
            "tag": str(result.tag), "preset": result.preset, "time": result.time, "level": result.window.level,
            "size": list(result.window.size), "sensor": list(result.window.sensor.size),
            "sensor_rect": list(result.window.sensor_rect), "view_rect": [round(v, 2) for v in result.view_rect],
            "scale": result.window.scale, "expected_eye_distance": result.eye_distance,
            "render_ms": round(result.render_ms, 1), "camera_ms": round(result.camera_ms, 1),
        }
        truth = result.truth
        if truth is not None:
            entry["truth"] = {"name": truth.name, "avatar": truth.avatar, "distance": round(truth.distance, 3),
                              "eye_distance": round(truth.eye_distance, 2),
                              "frontal_eye_distance": round(truth.frontal_eye_distance, 2), "inside": truth.inside,
                              "visible": truth.visible, "velocity": [round(v, 1) for v in truth.velocity],
                              "facing": round(truth.facing, 1), "yaw": round(truth.yaw, 1),
                              "pitch": round(truth.pitch, 1), "roll": round(truth.roll, 1)}
            person = scene.get(truth.name)
            if person is not None:
                in_view = eyes_in_view(scene, person, size)
                entry["truth"]["view_eye_distance"] = round(float(np.linalg.norm(in_view[0] - in_view[1])), 2)
                if engine:
                    found = match_face(view_faces, in_view, truth.frontal_eye_distance / result.window.scale)
                    entry["view_face"] = None if found is None else {"eye_distance": round(_face_size(found), 2),
                                                                     "score": round(found.score, 3)}
        image = result.image.copy()
        if engine:
            faces = engine.detect(result.image, min_size=0)
            found = match_face(faces, truth.eyes, truth.frontal_eye_distance) if truth is not None \
                else (faces[0] if faces else None)
            entry["window_faces"] = len(faces)
            entry["window_face"] = None if found is None else {"eye_distance": round(_face_size(found), 2),
                                                               "score": round(found.score, 3)}
            for face in faces:
                x, y, w, h = (round(v) for v in face.box)
                cv2.rectangle(image, (x, y), (x + w, y + h), (60, 220, 60) if face is found else (160, 160, 160), 2)
        name = f"window_{result.tag}_{result.preset.replace(' ', '_')}.png"
        cv2.imwrite(str(out / name), result.image)
        entry["file"] = name
        report.append(entry)
        sheet.append(_thumb(image, f"{result.tag} {result.preset} L{result.window.level}"))
    cv2.imwrite(str(out / "view.png"), annotated)
    if sheet:
        cv2.imwrite(str(out / "sheet.png"), _grid(sheet))
    (out / "windows.json").write_text(json.dumps({"scene": args.scene or "bifold_demo.scenes.fair",
                                                  "time": scene.time, "view": list(size), "sensor": list(sensor.size),
                                                  "people": args.people, "windows": report}, indent=1))
    _print_table(report, engine is not None)
    print(f"[scene] wrote {out}/view.png, sheet.png, windows.json and {len(report)} window images")
    return 0


def _thumb(image: np.ndarray, label: str, side: int = 240) -> np.ndarray:
    scale = side / max(image.shape[:2])
    small = cv2.resize(image, (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))),
                       interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_NEAREST)
    tile = np.full((side + 18, side, 3), 32, np.uint8)
    tile[:small.shape[0], :small.shape[1]] = small
    cv2.putText(tile, label, (3, side + 13), FONT, 0.4, (230, 230, 230), 1, cv2.LINE_AA)
    return tile


def _grid(tiles: list, columns: int = 6) -> np.ndarray:
    rows = [tiles[i:i + columns] for i in range(0, len(tiles), columns)]
    blank = np.zeros_like(tiles[0])
    return np.vstack([np.hstack(row + [blank] * (columns - len(row))) for row in rows])


def _print_table(report: list, faces: bool) -> None:
    """One line a window: the person's distance, eye distance in the view and in the window (frontal: as if facing
    the camera), whether their eyes are hidden, the window's level and size, and the faces the detector found."""
    head = (f"{'tag':>14} {'preset':>13} {'dist m':>6} {'view IED':>8} {'win IED':>7} {'frontal':>7} {'seen':>6} "
            f"{'level':>5} {'size':>9}")
    if faces:
        head += f" {'view face':>9} {'win face':>8}"
    print(head)
    nan = float("nan")
    for entry in report:
        truth = entry.get("truth", {})
        visible = truth.get("visible")
        seen = "-" if visible is None else ("yes" if visible >= 0.5 else "hidden")
        line = (f"{entry['tag']:>14} {entry['preset']:>13} {truth.get('distance', nan):6.1f} "
                f"{truth.get('view_eye_distance', nan):8.1f} {truth.get('eye_distance', nan):7.1f} "
                f"{truth.get('frontal_eye_distance', nan):7.1f} {seen:>6} "
                f"{entry['level']:5d} {entry['size'][0]:>4}x{entry['size'][1]:<4}")
        if faces:
            view_face, window_face = entry.get("view_face"), entry.get("window_face")
            line += f" {view_face['eye_distance'] if view_face else '-':>9} " \
                    f"{window_face['eye_distance'] if window_face else '-':>8}"
        print(line)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="bifold-scene", description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    w = sub.add_parser("windows", help="render a scene at a moment and the windows on its people")
    w.add_argument("--scene", help="scene file or module (default: the trade fair)")
    w.add_argument("--time", type=float, default=0.0, help="seconds into the scene (default 0)")
    w.add_argument("--fps", type=float, default=30.0, help="steps a second to get there (default 30, as the demo)")
    w.add_argument("--size", type=_size, default=(1280, 720), help="the view (default 1280x720)")
    w.add_argument("--sensor", type=_size, help="the declared sensor (default: six times the view)")
    w.add_argument("--preset", type=_preset, action="append",
                   help="camera preset, repeatable: ideal, good_8k, good_4k, budget_1080p (default ideal)")
    w.add_argument("--people", choices=["truth", "detect", "heads"], default="truth",
                   help="where the windows go (default truth: the people's skeleton boxes)")
    w.add_argument("--face-px", type=float, default=DEFAULT_FACE_PX,
                   help=f"pixels wanted between the eyes (default {DEFAULT_FACE_PX:.0f})")
    w.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS, help="most pixels a window may have")
    w.add_argument("--head-size", type=float, default=0.5, help="metres around a head (--people heads)")
    w.add_argument("--model", default="yolo26n.pt", help="YOLO weights (--people detect)")
    w.add_argument("--faces", action="store_true", help="run the face detector on the view and the windows")
    w.add_argument("--models", help="the face models' folder (default: $BIFOLD_FACE_MODELS or models/)")
    w.add_argument("-o", "--out", default="windows", help="output folder (default windows/)")
    args = p.parse_args(argv)
    try:
        return cmd_windows(args)
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
