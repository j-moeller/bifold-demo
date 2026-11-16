"""Recorded sessions for the evaluation (P5, webcam recordings with consenting people; step E1): record, review the
tracks by hand, score per track. For a Mac with a camera; tested here on renders.

1. **Record** a session (:func:`record`, ``bifold-faces eval --record session.avi``): the camera's frames as they
   come, at the resolution asked for (high: the windows are cut from the full picture, as live), Motion-JPEG at a
   high quality (a recording should not add blocks to small faces). Any video file works too.
2. **Review** (:func:`review`, ``bifold-faces eval --review session.avi``): the whole pipeline of the demo's ``N``
   runs over the recording with each profile (deterministic: the frames' times come from the file), and the first
   profile's tracks are written as a contact sheet of their faces (``<session>.review.jpg``) and a labels sheet
   (``<session>.labels.csv``: ``track, first, last, frames, faces, shown_by_the_system, label``). Fill in ``label``
   with the person's name as enrolled, ``stranger`` for a consenting person who is not enrolled, or ``skip`` (two
   people on one track, nobody, a poster). The sheets show real people: keep them in git-ignored folders.
3. **Score** (a manifest item ``{"kind": "video", "path": ..., "labels": ..., "faces": "faces"}``, or
   ``bifold-faces eval --review session.avi`` again once the labels are in): each profile's tracks get the label of
   the first profile's track they overlap most (the person detectors differ between profiles, so do the tracks), and
   are scored per track as the scenes are (:func:`~.metrics.track_summary`, :func:`~.metrics.acceptance`). Without a
   scene's truth a face counts as **shown** when the pipeline found it turned less than 45 degrees.

What to report from the Mac: per profile the counts (people, tracks), known people named within 2 and 5 s with
their intervals, wrong names, strangers named, the time to a name, and the acceptance verdict; never the sheets.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

from . import metrics

LABELS = ("stranger", "skip")
SHOWN_YAW = 45.0
SHEET_FACES = 6
REVIEW_VERSION = 1

__all__ = ["VideoFileSource", "carry_labels", "read_labels", "record", "review", "score_video"]


class VideoFileSource:
    """Frames of a video file as a frame source: each frame's time is its index over the file's frame rate (so a run
    is deterministic), scaled to fit ``width`` x ``height`` for the detector, with ``windows`` (a
    :class:`~bifold_demo.sensor.WindowService`) cut from the full picture, as a webcam's are."""

    name = "video"

    def __init__(self, path, width: int = 1280, height: int = 720, windows=None):
        self.path = Path(path)
        self.width, self.height = int(width), int(height)
        self.windows = windows
        self._cap = None
        self.index = 0
        self.fps = 30.0

    def start(self) -> None:
        if self._cap is None:
            cap = cv2.VideoCapture(str(self.path))
            if not cap.isOpened():
                raise OSError(f"cannot read {self.path}")
            self._cap = cap
            self.fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0) or 30.0

    def read(self):
        from ..sources.base import Frame

        self.start()
        ok, full = self._cap.read()
        if not ok:
            return None
        h, w = full.shape[:2]
        scale = min(1.0, self.width / w, self.height / h)
        image = full if scale >= 1.0 else cv2.resize(full, (round(w * scale), round(h * scale)),
                                                     interpolation=cv2.INTER_AREA)
        self.index += 1
        frame = Frame(image=image, index=self.index, timestamp=time.perf_counter(), source=self.name,
                      time=(self.index - 1) / self.fps)
        if self.windows is not None:
            if self.index == 1:
                self.windows.bind((image.shape[1], image.shape[0]))
            frame.windows = self.windows.for_image(full, frame)
        return frame

    def stop(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


def record(path, index: int = 0, capture=(1920, 1080), seconds: float = 60.0, fps: float = 30.0,
           progress=print) -> Path:
    """Record ``seconds`` of camera ``index`` at ``capture`` (width, height) into ``path`` (Motion-JPEG, quality
    95). Returns the path."""
    import sys

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        raise OSError(f"cannot open camera {index} (on macOS: allow camera access for the terminal)")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, capture[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, capture[1])
    cap.set(cv2.CAP_PROP_FPS, fps)
    ok, image = cap.read()
    if not ok:
        raise OSError(f"camera {index} gives no frames")
    h, w = image.shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    writer.set(cv2.VIDEOWRITER_PROP_QUALITY, 95)
    started, frames = time.perf_counter(), 0
    try:
        while ok and time.perf_counter() - started < seconds:
            writer.write(image)
            frames += 1
            ok, image = cap.read()
    finally:
        writer.release()
        cap.release()
    if progress:
        progress(f"[eval] recorded {frames} frames of {w}x{h} into {path}")
    return path


def _run(video, profile: str, engine, gallery, yolo=None, models_folder=None, windows: bool = True,
         progress=None) -> dict:
    """The pipeline over ``video`` with ``engine``'s profile: per track its boxes, decisions and faces."""
    from ..sensor import WindowService
    from .decision import DecisionLog
    from .evalscene import person_detector
    from .people import PeopleRecognizer

    service = WindowService(max_windows=4) if windows else None
    source = VideoFileSource(video, windows=service)
    source.start()
    detector = person_detector(profile, None, "profile", yolo, models_folder)
    recognizer = PeopleRecognizer(gallery, detector, windows=service, log=DecisionLog(keep=10),
                                  models_folder=models_folder)
    tracks: dict = {}
    timings: dict = {}
    frames = embedded = 0
    started = time.perf_counter()
    try:
        while True:
            frame = source.read()
            if frame is None:
                break
            result = recognizer.process(frame)
            frames += 1
            for key, value in result.timings.items():
                timings.setdefault(key, []).append(value)
            for person in result.people:
                d = person.decision
                entry = tracks.setdefault(str(person.id), {"boxes": [], "rows": [], "faces": []})
                entry["boxes"].append([round(result.time, 4), *[round(float(v), 1) for v in person.box]])
                shown = False
                if person.face is not None and person.quality is not None:
                    shown = abs(float(person.quality.yaw)) < SHOWN_YAW
                    entry["faces"].append([round(result.time, 4), round(float(person.face.size), 2),
                                           round(float(person.quality.yaw), 1)])
                embedded += int(person.embedded)
                entry["rows"].append([round(result.time, 4), bool(shown), d.candidate, _r(d.calibrated),
                                      _r(d.runner_up_calibrated), _r(d.accept_rarity), round(float(d.progress), 3),
                                      d.state, d.name])
            if progress and frames % 100 == 0:
                progress(f"[eval] {Path(video).name} {profile}: {frames} frames, {len(tracks)} tracks "
                         f"({time.perf_counter() - started:.0f} s)")
    finally:
        source.stop()
    return {"video": str(video), "profile": profile, "model": recognizer.model, "frames": frames, "fps": source.fps,
            "tracks": tracks, "cost": metrics.cost_summary(timings, embedded, frames / max(source.fps, 1e-9),
                                                           frames) | {"wall_s": round(time.perf_counter() - started,
                                                                                      1)}}


def _r(value):
    return None if value is None or not np.isfinite(value) else round(float(value), 4)


def _sheet(video, tracks: dict, path: Path) -> None:
    """A contact sheet of each track's biggest frontal faces (up to :data:`SHEET_FACES`), cut from the recording."""
    wanted = {}
    for track, entry in tracks.items():
        faces = sorted(entry["faces"], key=lambda f: (abs(f[2]) >= SHOWN_YAW, -f[1]))[:SHEET_FACES]
        boxes = {b[0]: b[1:] for b in entry["boxes"]}
        for t, _size, _yaw in faces:
            wanted.setdefault(round(t, 4), []).append((track, boxes.get(round(t, 4))))
    tiles: dict = {}
    source = VideoFileSource(video)
    try:
        while True:
            frame = source.read()
            if frame is None:
                break
            for track, box in wanted.get(round(frame.time, 4), []):
                if box is None:
                    continue
                x0, y0, x1, y1 = (round(v) for v in box)
                height = max(1, y1 - y0)
                crop = frame.image[max(0, y0):max(0, y0) + height // 3, max(0, x0):max(0, x1)]
                if crop.size:
                    tiles.setdefault(track, []).append(cv2.resize(crop, (96, 96), interpolation=cv2.INTER_AREA))
    finally:
        source.stop()
    rows = []
    for track in sorted(tiles, key=int):
        row = np.full((96, 96 * (SHEET_FACES + 1), 3), 255, np.uint8)
        cv2.putText(row, f"#{track}", (6, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        for k, tile in enumerate(tiles[track][:SHEET_FACES]):
            row[:, 96 * (k + 1):96 * (k + 2)] = tile
        rows.append(row)
    if rows:
        cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])


def review(video, runs: dict, labels_path=None, sheet: bool = True) -> Path:
    """Write the review sheets of a recording from its first profile's run (``runs``: ``{profile: run}``): the
    contact sheet and the labels sheet (an existing labels sheet is never overwritten: its labels stay). Returns the
    labels sheet's path."""
    video = Path(video)
    first = next(iter(runs.values()))
    labels_path = Path(labels_path) if labels_path else video.with_suffix(".labels.csv")
    if sheet:
        _sheet(video, first["tracks"], video.with_suffix(".review.jpg"))
    if not labels_path.exists():
        with labels_path.open("w", newline="") as handle:
            out = csv.writer(handle)
            out.writerow(["track", "first", "last", "frames", "faces", "shown_by_the_system", "label"])
            for track, entry in sorted(first["tracks"].items(), key=lambda kv: int(kv[0])):
                names = [r[8] for r in entry["rows"] if r[7] == "named"]
                out.writerow([track, entry["boxes"][0][0], entry["boxes"][-1][0], len(entry["boxes"]),
                              len(entry["faces"]), names[-1] if names else "", ""])
    return labels_path


def read_labels(path) -> dict[str, str]:
    """``{track: label}`` from a labels sheet (empty labels left out)."""
    with Path(path).open(newline="") as handle:
        return {row["track"].strip(): (row.get("label") or "").strip() for row in csv.DictReader(handle)
                if (row.get("label") or "").strip()}


def _iou(a, b) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def carry_labels(reference: dict, labels: dict, other: dict, min_iou: float = 0.3) -> dict[str, str]:
    """``other`` run's tracks labelled as the ``reference`` run's track they overlap most (mean overlap over the
    frames both show, at least ``min_iou``): the profiles' person detectors and so their tracks differ."""
    boxes = {track: {b[0]: b[1:] for b in entry["boxes"]} for track, entry in reference["tracks"].items()
             if track in labels}
    out = {}
    for track, entry in other["tracks"].items():
        best, label = 0.0, None
        mine = {b[0]: b[1:] for b in entry["boxes"]}
        for ref, theirs in boxes.items():
            shared = [t for t in mine if t in theirs]
            if not shared:
                continue
            overlap = sum(_iou(mine[t], theirs[t]) for t in shared) / len(mine)
            if overlap > best:
                best, label = overlap, labels[ref]
        if label is not None and best >= min_iou:
            out[track] = label
    return out


def score_video(run: dict, labels: dict, known, shown: float = 1.0, draws: int = metrics.DRAWS,
                seed: int = 0) -> list:
    """The :class:`~.truth.Segment` items of a run's labelled tracks (``labels``: ``{track: name, "stranger" or
    "skip"}``; ``known``: the enrolled names): one segment per track (a track is one person: tracks with two people
    are labelled ``skip``)."""
    from .truth import Segment

    known = set(known)
    segments = []
    for track, entry in run["tracks"].items():
        label = labels.get(track)
        if not label or label == "skip":
            continue
        rows = entry["rows"]
        first_shown, shown_s, last, correct_at, named_at, wrong, unknown_at = None, 0.0, None, None, None, [], None
        for t, is_shown, _candidate, _best, _runner, _accept, _progress, state, name in rows:
            if is_shown:
                if first_shown is None:
                    first_shown = t
                elif last is not None:
                    shown_s += t - last
            last = t
            if state == "named":
                named_at = t if named_at is None else named_at
                if name == label:
                    correct_at = t if correct_at is None else correct_at
                elif name not in wrong:
                    wrong.append(name)
            elif state == "unknown" and unknown_at is None:
                unknown_at = t
        segments.append(Segment(f"{label} #{track}", label, int(track), label in known, rows[0][0], rows[-1][0],
                                shown_s, first_shown, correct_at, tuple(wrong), unknown_at, named_at, len(rows)))
    return segments


def run_video(run, item, folder, yolo=None) -> dict:
    """A ``video`` item of a manifest through every profile of ``run`` (each run kept beside the recording as
    ``<video>.<profile>.json``, made again only when the recording or the review's version changed), scored with its
    labels (when there are any): ``{profile: {"summary", "acceptance", "tracks", "labelled"}}``."""
    from .evaluate import file_digest
    from .gallery import Gallery

    video = Path(folder) / item.path
    faces = Path(folder) / item.extra.get("faces", "faces")
    digest = file_digest(video)[:24]
    runs = {}
    for profile, engine in run.engines.items():
        stored = video.with_name(f"{video.stem}.{profile}.json")
        if stored.is_file():
            data = json.loads(stored.read_text())
            if data.get("digest") == digest and data.get("version") == REVIEW_VERSION:
                runs[profile] = data
                continue
        gallery = Gallery(faces, engine)
        gallery.refresh(force=True)
        data = _run(video, profile, engine, gallery, yolo, run.models_folder, progress=run.progress)
        data.update(digest=digest, version=REVIEW_VERSION, known=sorted(gallery.people()))
        stored.write_text(json.dumps(data))
        runs[profile] = data
    labels_path = Path(folder) / item.extra["labels"] if item.extra.get("labels") else None
    if labels_path is None or not labels_path.is_file():
        path = review(video, runs, labels_path)
        run.progress(f"[eval] {video.name}: label the tracks in {path} (see {video.with_suffix('.review.jpg')})")
        return {}
    labels = read_labels(labels_path)
    first = next(iter(runs))
    out = {}
    for profile, data in runs.items():
        mine = labels if profile == first else carry_labels(runs[first], labels, data)
        segments = score_video(data, mine, data["known"])
        summary = metrics.track_summary(segments, within=(2.0, 5.0), draws=run.draws, seed=run.seed)
        out[profile] = {"summary": summary, "acceptance": metrics.acceptance(summary), "tracks": len(data["tracks"]),
                        "labelled": len(mine), "cost": data["cost"]}
    return out
