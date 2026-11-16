"""The margin of the per-track offsets (step E1): what the shipped 95 % bound costs in names, against the false
alarms it saves.

A live track is decided again after every face, so the calibration adds an **offset** to a single decision's accept
rarity for each target false-alarm rate per stranger track (:meth:`~.calibration.Calibration.track_offset`). The
shipped offsets (:func:`~.calibrate.track_offsets`) are the smallest whose **upper bound** keeps the share of
simulated stranger tracks named at the target: the 95th percentile of a bootstrap over identity pairs (a stranger's
false alarms come from the few people they resemble). The margins compared here, each a way of reading the same
simulated tracks (:data:`CHOICES`):

* ``bound``: the upper end of the bootstrap at 95 % (shipped);
* ``median``: the bootstrap's median;
* ``point``: the share of the tracks itself (the point estimate);
* ``none``: no offset (the single decision's threshold).

:func:`calibration_data` makes the calibration's faces again from what ``bifold-faces calibrate`` kept (its renders'
faces, their embeddings and the gallery: read only), with its avatars, its same faces and those it left out;
:func:`offsets` computes each margin's offsets from simulated stranger tracks of some halvings of those avatars (with
the same seeds as the calibration: the ``bound`` offsets must come out as shipped, :func:`offsets` checks it), and
:func:`check` measures each margin per stranger track on other halvings. The cost in names is measured on identities
the calibration never saw: the fair's, in the stills (:func:`~.evaluate.score_stills`, per decision) and in the scene
runs (:func:`~.evalscene.replayed`, per track).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from . import calibrate as cal
from .calibration import per_person

CHOICES = ("bound", "median", "point", "none")

__all__ = ["CHOICES", "calibration_data", "check", "offsets"]


def calibration_data(data, gallery_folder, engine, calibration) -> tuple:
    """``(faces, gallery rows)`` of ``calibration`` (a shipped :class:`~.calibration.Calibration`) made again from
    ``bifold-faces calibrate``'s ``data`` folder (``probes.<detector>.npz``, ``embeddings.<model>.npz``: read only) and
    a copy of its gallery (``gallery_folder``: its photos and their cached embeddings, which the gallery may write
    to), with ``engine`` (the calibration's model and detector): its avatars, those wearing one face counted as one,
    and the faces the live gate passes, as ``bifold-faces calibrate`` made them."""
    from .gallery import Gallery

    data = Path(data)
    made = calibration.made
    probes = cal.Probes.load(data / f"probes.{made['detector']}.npz")
    with np.load(data / f"embeddings.{calibration.model}.npz", allow_pickle=False) as stored:
        index = {k: i for i, k in enumerate(stored["keys"].tolist()) if stored["embedded"][i]}
        if str(stored["sha256"]) != made["sha256"]:
            raise ValueError(f"the embeddings in {data} are of another file of {calibration.model}")
        vectors = np.zeros((len(probes.found), stored["vectors"].shape[1]))
        fiqa = np.zeros(len(probes.found)) if "fiqa" in stored.files else None
        for r, key in enumerate(probes.keys().tolist()):
            i = index.get(key)
            if i is not None:
                vectors[r] = stored["vectors"][i]
                if fiqa is not None:
                    fiqa[r] = stored["fiqa"][i]
    chosen = np.isin(probes.avatar, made["avatars"])
    probes, vectors = probes.subset(chosen), vectors[chosen]
    fiqa = None if fiqa is None else fiqa[chosen]
    if engine.fiqa is None:
        fiqa = None
    faces = cal.faces(probes, vectors, probes.weights(fiqa), probes.passed(fiqa))
    gallery = Gallery(gallery_folder, engine)
    gallery.refresh(force=True)
    rows = cal.GalleryRows.of(gallery).subset(made["avatars"])
    alias = {a: b for a, b in (made.get("same_faces") or {}).items()}
    return cal.as_one(faces, rows, alias)


def offsets(tracks: cal.StrangerTracks, choice: str, targets=cal.TRACK_TARGETS, seed: int = 0,
            draws: int = 2000) -> dict:
    """The offsets of ``choice`` (:data:`CHOICES`) from simulated stranger ``tracks``: ``{"targets", "offsets"}``, each
    target's threshold with its offset at least as strict as a looser target's (as :func:`~.calibrate.track_offsets`
    keeps them)."""
    if choice == "bound":
        return cal.track_offsets(tracks, targets, 0.95, draws, seed)
    if choice == "median":
        return cal.track_offsets(tracks, targets, 0.5, draws, seed)
    if choice == "none":
        return {"targets": sorted(targets, reverse=True), "offsets": [0.0] * len(targets)}
    if choice != "point":
        raise ValueError(f"unknown margin {choice!r} (one of {', '.join(CHOICES)})")
    grid = np.arange(0.0, cal.OFFSET_MOST + cal.OFFSET_STEP / 2, cal.OFFSET_STEP)
    out = {"targets": [], "offsets": []}
    previous = None
    for target in sorted(targets, reverse=True):
        excess = tracks.excess(target)
        rate = np.array([np.mean(excess >= g - 1e-12) for g in grid])
        ok = np.flatnonzero(rate <= target)
        offset = float(grid[ok[0]]) if len(ok) else float(grid[-1])
        if previous is not None:  # (no looser than the looser target's threshold)
            loose_target, loose_offset = previous
            n = int(np.median(tracks.people))
            floor = loose_offset - (math.log10(per_person(loose_target, n)) - math.log10(per_person(target, n)))
            offset = max(offset, round(floor, 4))
        previous = (target, offset)
        out["targets"].append(target)
        out["offsets"].append(round(offset, 4))
    return out


def check(tracks: cal.StrangerTracks, chosen: dict, draws: int = 2000, seed: int = 1) -> dict:
    """Each margin's offsets (``chosen``: ``{choice: offsets}``) measured on other simulated ``tracks``: per target the
    share of stranger tracks named, with a 90 % interval over identity pairs (:func:`~.calibrate.track_rates`)."""
    return {choice: cal.track_rates(tracks, track, draws=draws, seed=seed) for choice, track in chosen.items()}
