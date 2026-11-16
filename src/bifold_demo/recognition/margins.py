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


def simulated_tracks(stills, rows, owners, calibration, cohort, group, offsets: dict, fpir: float = 0.01,
                     count: int = 20, seed: int = 0, margin: float = 1.0, min_weight: float = 0.5,
                     min_frames: int = 2, single_face: float = 40.0, single_yaw: float = 25.0, draws: int = 1000,
                     level: float = 0.95) -> dict:
    """Tracks of held-out people simulated from stills (:class:`~.evaluate.Stills` of the evaluation) as live: an
    identity's usable faces of one ``group`` (``group(item)``: a track's key, e.g. the kind of face and the camera)
    arrive one by one in a seeded random order, and after each (from the ``min_frames``-th, or the first while the
    mean face is ``single_face`` pixels between the eyes and turned less than ``single_yaw``, once the quality weights
    reach ``min_weight``) the running quality-weighted mean is decided against the enrolled half's gallery (``rows``,
    ``owners``; a worn face never against its carrier), normalised as the calibration says, in its bin: named when the
    best rarity reaches the accept rarity at ``fpir`` plus each of ``offsets`` (``{name: decades}``) and beats the
    runner-up by ``margin``. Over ``count`` seeded halvings: per offset the share of enrolled people's tracks named
    rightly (and how many faces it took, the median), named wrongly, and of strangers' tracks named, with intervals
    over people (strangers' over identity pairs: the stranger and whom they were named as, else their best
    candidate)."""
    from .calibrate import rarities
    from .evaluate import carriers, person_scores

    worn = carriers(stills)
    usable = stills.found & stills.passed
    keys = np.array([str(group(item)) for item in stills.items])
    rng = np.random.default_rng(seed)
    enrollable = sorted(set(owners.tolist()))
    acc = {name: {"known": [], "faces": [], "wrong": [], "known_people": [], "strangers": [], "pairs": []}
           for name in offsets}
    for _ in range(count):
        order = rng.permutation(len(enrollable))
        enrolled = sorted(enrollable[i] for i in order[: len(enrollable) // 2])
        mask = np.isin(owners, enrolled)
        accept = -math.log10(per_person(fpir, len(enrolled)))
        for identity in sorted(set(stills.identity.tolist())):
            for key in sorted(set(keys[stills.identity == identity].tolist())):
                idx = np.flatnonzero((stills.identity == identity) & (keys == key) & usable)
                if not len(idx):
                    continue
                idx = rng.permutation(idx)
                w = np.maximum(stills.weight[idx], 1e-6)
                total = np.cumsum(w)
                mean = np.cumsum(w[:, None] * stills.vectors[idx], axis=0)
                mean /= np.maximum(np.linalg.norm(mean, axis=1, keepdims=True), 1e-12)
                size = np.cumsum(w * stills.size[idx]) / total
                yaw = np.cumsum(w * np.abs(stills.yaw[idx])) / total
                one = (size >= single_face) & (yaw < single_yaw)
                decide = ((np.arange(1, len(idx) + 1) >= min_frames) | one) & (total >= min_weight - 1e-9)
                scores = person_scores(mean, rows[mask], owners[mask], enrolled, cohort, calibration.normalisation)
                carrier = worn[idx[0]]
                if carrier in enrolled:
                    scores[:, enrolled.index(carrier)] = -np.inf
                rarity = rarities(calibration, scores, size, yaw)
                ranked = np.argsort(-rarity, axis=1, kind="stable")
                best = rarity[np.arange(len(idx)), ranked[:, 0]]
                second = rarity[np.arange(len(idx)), ranked[:, 1]] if len(enrolled) > 1 else np.zeros(len(idx))
                is_known = identity in enrolled
                for name, offset in offsets.items():
                    named = decide & (best >= accept + offset - 1e-12) & (best - second >= margin - 1e-12)
                    first = int(np.argmax(named)) if named.any() else None
                    who = enrolled[ranked[first, 0]] if first is not None else enrolled[ranked[int(np.argmax(best)), 0]]
                    a = acc[name]
                    if is_known:
                        a["known"].append(first is not None and who == identity)
                        a["wrong"].append(first is not None and who != identity)
                        a["faces"].append(first + 1 if first is not None and who == identity else None)
                        a["known_people"].append(identity)
                    else:
                        a["strangers"].append(first is not None)
                        a["pairs"].append("|".join(sorted((identity, who))))
    from . import metrics

    out = {"splits": count, "fpir": fpir, "margin": margin, "offsets": {k: round(float(v), 4) for k, v in
                                                                        offsets.items()}}
    for name, a in acc.items():
        faces = [f for f in a["faces"] if f is not None]
        out[name] = {"known_named": metrics.bootstrap_rate(a["known"], a["known_people"], draws, seed, level),
                     "faces_to_name": float(np.median(faces)) if faces else None,
                     "known_wrong": metrics.bootstrap_rate(a["wrong"], a["known_people"], draws, seed, level),
                     "strangers_named": metrics.bootstrap_rate(a["strangers"], a["pairs"], draws, seed, level),
                     "known_tracks": len(a["known"]), "stranger_tracks": len(a["strangers"]),
                     "people": len(set(a["known_people"])), "pairs": len(set(a["pairs"]))}
    return out
