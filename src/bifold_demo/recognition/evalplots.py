"""Curves of an evaluation's results (``bifold-faces eval``, :mod:`.evaluate`) as PNG images, drawn from the results'
JSON alone (so they can be drawn again without measuring): DET curves per kind of face; the share of known people
named against the eye distance (per camera) and the turn, beside the share the scores alone would find at a measured
false-alarm rate of 1 % (what the calibration costs); the ladder's rungs; a test pack's DET; and, for scene runs, the
share of known people's tracks named against the seconds since their face was first shown.

The profiles keep one colour and one marker each in every figure (best blue circles, licensed orange squares,
commercial aqua triangles), lines are also labelled at their ends where they do not collide, and the intervals are
shaded; the numbers themselves are in the results' JSON and the results note's tables. Needs matplotlib (it comes with
Ultralytics; without it no curves are drawn, and the run says so).
"""

from __future__ import annotations

from pathlib import Path

PROFILE_COLOURS = {"best": "#2a78d6", "licensed": "#eb6834", "commercial": "#1baf7a"}  # (validated as a set)
MARKERS = {"best": "o", "licensed": "s", "commercial": "^"}
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de"
VARIANT_TITLES = {"own": "the avatar's own face", "renders": "face from set A, the avatar's head",
                  "fitted": "face from set A, the fitted head"}

__all__ = ["plot_results"]


def _style(plt):
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                         "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
                         "axes.titlesize": 9.5, "legend.frameon": False, "lines.linewidth": 1.3,
                         "lines.markersize": 4.5})


def _parse(value: str) -> dict:
    out = {}
    for part in value.split(", "):
        key, _, v = part.partition("=")
        out[key] = v
    return out


def _number(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _named(entry: dict):
    """The share of known people's probes named rightly at 1 % per decision, with its interval (or ``None``)."""
    target = (entry.get("system") or {}).get("targets", {}).get("0.01")
    if not target:
        return None
    r = target["known"]["named_right"]
    return r["rate"], r["interval"]


def _found(entry: dict):
    """The share of known people's probes found at a measured FPIR of 1 % (the detection and identification rate,
    1 - FNIR, with the threshold set on the condition's own strangers), or ``None``."""
    e = ((entry.get("open_set") or {}).get("fnir_at_fpir") or {}).get("0.01")
    return None if not e else 1.0 - e["fnir"]


def _label_ends(ax, ends, gap: float = 0.07):
    """Direct labels at the lines' ends (``ends``: ``[(x, y, text)]``), in muted ink, moved apart vertically so that
    none overlap (``gap``: the least distance between two, a share of the y axis's span)."""
    if not ends:
        return
    lo, hi = ax.get_ylim()
    step = (hi - lo) * gap
    order = sorted(range(len(ends)), key=lambda k: ends[k][1])
    placed = []
    for k in order:
        y = ends[k][1]
        placed.append(y if not placed or y >= placed[-1] + step else placed[-1] + step)
    over = placed[-1] - hi  # (pushed above the top: move the column down)
    if over > 0:
        placed = [y - over for y in placed]
    for k, y in zip(order, placed):
        x, _, text = ends[k]
        ax.annotate(text, (x, y), xytext=(4, 0), textcoords="offset points", va="center", fontsize=7.5,
                    color=MUTED, annotation_clip=False)


def plot_results(results: dict, folder) -> list[Path]:
    """Draw the curves of ``results`` (one manifest's) into ``folder``; returns the images written."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[eval] no curves: matplotlib is not installed")
        return []
    _style(plt)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    written = []
    stills = results.get("stills") or {}
    if stills:
        written += _det_by_variant(plt, stills, folder)
        written += _swept(plt, stills, folder, "ied by preset", "ied", "eye distance asked (pixels, log)",
                          "ied.png", fixed={"preset": "ideal"}, log=True)
        written += _swept(plt, stills, folder, "ied by preset", "ied", "eye distance asked (pixels, log)",
                          "ied_good_4k.png", fixed={"preset": "good 4K"}, log=True)
        written += _swept(plt, stills, folder, "ied by preset", "ied", "eye distance asked (pixels, log)",
                          "ied_budget.png", fixed={"preset": "budget 1080p"}, log=True)
        written += _swept(plt, stills, folder, "turn", "turn", "head turned (degrees)", "turn.png",
                          title="Named against the turn of the head (38 pixels between the eyes)")
        written += _ladder(plt, stills, folder)
    for pack, by_profile in (results.get("packs") or {}).items():
        written += _pack(plt, pack, by_profile, folder)
    if results.get("scenes"):
        written += _named_over_time(plt, results["scenes"], folder)
    plt.close("all")
    return written


def _named_over_time(plt, scenes: dict, folder: Path, shown: float = 1.0) -> list[Path]:
    """The scene runs: the share of known people's tracks (their face shown at least ``shown`` seconds) named
    correctly within t seconds of the face being first shown, per profile."""
    import numpy as np

    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    grid = np.linspace(0.0, 12.0, 121)
    drawn = 0
    ends = []
    for key, scored in scenes.items():
        segments = [g for r in scored.get("runs", []) for g in r["segments"]
                    if g["known"] and g["first_shown"] is not None and g["shown"] >= shown - 1e-9]
        if not segments:
            continue
        drawn += 1
        delays = np.array([g["correct_at"] - g["first_shown"] if g["correct_at"] is not None else np.inf
                           for g in segments])
        share = [(delays <= t + 1e-9).mean() for t in grid]
        profile = key.split("@")[0]
        colour = PROFILE_COLOURS.get(profile, MUTED)
        style = "--" if "@" in key else "-"
        ax.plot(grid, share, color=colour, linestyle=style, drawstyle="steps-post", label=f"{key} ({len(segments)})")
        ends.append((grid[-1], share[-1], key))
    if not drawn:
        plt.close(fig)
        return []
    ax.axvline(2.0, color=GRID, linewidth=1.0)
    ax.set_xlabel("seconds since the face was first shown")
    ax.set_ylabel("known people's tracks named rightly")
    ax.set_ylim(-0.02, 1.02)
    _label_ends(ax, ends)
    ax.legend(loc="lower right", fontsize=7)
    ax.set_title("Scene runs: named within t (tracks counted in brackets)")
    fig.tight_layout()
    path = folder / "named_over_time.png"
    fig.savefig(path, dpi=150)
    return [path]


def _det_by_variant(plt, stills: dict, folder: Path) -> list[Path]:
    """DET (false rejects against false accepts, both on normal-deviate axes) per kind of face, a line per profile."""
    from .metrics import probit

    variants = []
    for report in stills.values():
        for value in (report.get("conditions") or {}).get("variant", {}):
            v = _parse(value).get("variant")
            if v not in variants:
                variants.append(v)
    panels = [v for v in ("own", "renders", "fitted") if v in variants] or [None]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 3.2), squeeze=False, sharey=True)
    for ax, variant in zip(axes[0], panels):
        for profile, report in stills.items():
            entry = report.get("verification") if variant is None else \
                (report.get("conditions") or {}).get("variant", {}).get(f"variant={variant}", {}).get("verification")
            curve = (entry or {}).get("curve")
            if curve:
                _det_line(ax, curve, profile, probit)
        _det_axes(ax, probit)
        ax.set_title(VARIANT_TITLES.get(variant, "all probes"))
    axes[0][0].set_ylabel("false reject rate")
    axes[0][-1].legend(loc="upper right", fontsize=7.5)
    fig.suptitle("Verification on the stills (every probe against every identity's gallery)", fontsize=10)
    fig.tight_layout()
    path = folder / "det.png"
    fig.savefig(path, dpi=150)
    return [path]


def _det_line(ax, curve: dict, profile: str, probit) -> None:
    """A DET curve (from a report's ``curve``: FAR and TAR) on normal-deviate axes, with the profile's marker at
    FAR 1e-3, 1e-2 and 0.1 (its identity is not colour alone)."""
    far = [min(max(x, 1e-6), 0.999) for x in curve["x"]]
    frr = [min(max(1.0 - t, 1e-4), 0.999) for t in curve["tar"]]
    marks = [k for k, x in enumerate(curve["x"]) if any(abs(x - m) <= 1e-3 * m for m in (1e-3, 1e-2, 0.1))]
    ax.plot(probit(far), probit(frr), color=PROFILE_COLOURS.get(profile, MUTED), label=profile,
            drawstyle="steps-post", marker=MARKERS.get(profile, "o"), markevery=marks)


def _det_axes(ax, probit) -> None:
    ticks = [1e-4, 1e-3, 1e-2, 0.1, 0.5]
    ax.set_xticks(probit(ticks), ["1e-4", "1e-3", "0.01", "0.1", "0.5"])
    ax.set_xlim(*probit([6e-5, 0.6]))
    ax.set_ylim(*probit([5e-4, 0.6]))
    ax.set_yticks(probit([1e-3, 1e-2, 0.05, 0.2, 0.5]), ["0.001", "0.01", "0.05", "0.2", "0.5"])
    ax.set_xlabel("false accept rate")


def _swept(plt, stills: dict, folder: Path, breakdown: str, key: str, xlabel: str, name: str, fixed=None,
           log: bool = False, title: str | None = None) -> list[Path]:
    """The share of known people named rightly at 1 % per decision against one swept condition, per kind of face
    (panels), a line per profile with its interval shaded; dashed, the share found at a measured FPIR of 1 % (the
    threshold set on the condition's own strangers: what the scores allow, against what the calibration names)."""
    fixed = fixed or {}
    series: dict = {}
    for profile, report in stills.items():
        for value, entry in ((report.get("conditions") or {}).get(breakdown) or {}).items():
            parts = _parse(value)
            if any(parts.get(k) != v for k, v in fixed.items()):
                continue
            x, named = _number(parts.get(key)), _named(entry)
            if x is None or named is None:
                continue
            series.setdefault(parts.get("variant"), {}).setdefault(profile, []).append((x, *named, _found(entry)))
    if not series:
        return []
    panels = [v for v in ("own", "renders", "fitted") if v in series]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 3.1), squeeze=False, sharey=True)
    for ax, variant in zip(axes[0], panels):
        ax.set_ylim(-0.02, 1.04)
        ends = []
        for profile, points in series[variant].items():
            points.sort(key=lambda p: p[0])
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            lo = [p[2][0] if p[2][0] is not None else p[1] for p in points]
            hi = [p[2][1] if p[2][1] is not None else p[1] for p in points]
            colour = PROFILE_COLOURS.get(profile, MUTED)
            ax.fill_between(xs, lo, hi, color=colour, alpha=0.12, linewidth=0)
            ax.plot(xs, ys, color=colour, marker=MARKERS.get(profile, "o"), label=profile)
            found = [(p[0], p[3]) for p in points if p[3] is not None]
            if found:
                ax.plot([f[0] for f in found], [f[1] for f in found], color=colour, linestyle="--", linewidth=0.9)
            ends.append((xs[-1], ys[-1], profile))
        _label_ends(ax, ends)
        if log:
            ax.set_xscale("log")
            ax.set_xticks([8, 12, 18, 26, 38, 54, 76, 100], ["8", "12", "18", "26", "38", "54", "76", "100"])
            ax.minorticks_off()
        ax.set_xlabel(xlabel)
        ax.set_title(VARIANT_TITLES.get(variant, variant))
    axes[0][0].set_ylabel("known probes named rightly")
    handles, labels = axes[0][-1].get_legend_handles_labels()
    from matplotlib.lines import Line2D

    handles += [Line2D([], [], color=MUTED, linestyle="-"), Line2D([], [], color=MUTED, linestyle="--")]
    labels += ["named: calibrated, 1 % per decision, after the quality gate",
               "found: any face, at a measured FPIR of 1 %"]
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=7, bbox_to_anchor=(0.5, 0.0))
    extra = ", ".join(f"{k} {v}" for k, v in fixed.items())
    fig.suptitle((title or f"Named against the {xlabel.split(' (')[0]}") + (f" ({extra})" if extra else ""),
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    path = folder / name
    fig.savefig(path, dpi=150)
    return [path]


LADDER_ORDER = ("B0", "B1", "B1=", "B2 good 4K", "B2 budget 1080p", "B3", "B4", "B5")
LADDER_TICKS = {"B2 good 4K": "B2\n4K", "B2 budget 1080p": "B2\n1080p"}


def _ladder(plt, stills: dict, folder: Path) -> list[Path]:
    """The ladder's rungs per kind of face (columns), per profile: above, the share of known probes named rightly at
    1 % per decision (intervals as bars) and, dashed, the share found at a measured FPIR of 1 %; below, d' (the
    separation of the scores)."""
    series: dict = {}
    order = []
    for profile, report in stills.items():
        for value, entry in ((report.get("conditions") or {}).get("rung") or {}).items():
            parts = _parse(value)
            named = _named(entry)
            if named is None:
                continue
            variant = "own" if parts.get("variant") == "own" else "renders"  # (B5 continues the faces from set A)
            if parts["rung"] not in order:
                order.append(parts["rung"])
            d = (entry.get("verification") or {}).get("d_prime")
            series.setdefault(variant, {}).setdefault(profile, {})[parts["rung"]] = (*named, _found(entry), d)
    if not series:
        return []
    rungs = sorted(order, key=lambda r: (LADDER_ORDER.index(r) if r in LADDER_ORDER else len(LADDER_ORDER), r))
    panels = [v for v in ("own", "renders") if v in series]
    fig, axes = plt.subplots(2, len(panels), figsize=(4.8 * len(panels), 5.4), squeeze=False, sharex="col",
                             gridspec_kw={"height_ratios": (3, 2)})
    for column, variant in enumerate(panels):
        ax, low = axes[0][column], axes[1][column]
        here = [r for r in rungs if any(r in s for s in series[variant].values())]
        for k, (profile, points) in enumerate(series[variant].items()):
            mine = [r for r in here if r in points]
            xs = [here.index(r) + (k - 1) * 0.12 for r in mine]
            ys = [points[r][0] for r in mine]
            err = [[max(0.0, points[r][0] - (points[r][1][0] if points[r][1][0] is not None else points[r][0]))
                    for r in mine],
                   [max(0.0, (points[r][1][1] if points[r][1][1] is not None else points[r][0]) - points[r][0])
                    for r in mine]]
            colour = PROFILE_COLOURS.get(profile, MUTED)
            ax.errorbar(xs, ys, yerr=err, color=colour, marker=MARKERS.get(profile, "o"), linestyle="-",
                        linewidth=1.0, capsize=2, label=profile)
            found = [(x, points[r][2]) for x, r in zip(xs, mine) if points[r][2] is not None]
            if found:
                ax.plot([f[0] for f in found], [f[1] for f in found], color=colour, linestyle="--", linewidth=0.9)
            dee = [(x, points[r][3]) for x, r in zip(xs, mine) if points[r][3] is not None]
            low.plot([p[0] for p in dee], [p[1] for p in dee], color=colour, marker=MARKERS.get(profile, "o"),
                     linewidth=1.0)
        ax.set_ylim(-0.02, 1.04)
        ax.set_title("the avatars' own faces" if variant == "own" else "faces from set A (B5: the fitted head)")
        low.set_xticks(range(len(here)), [LADDER_TICKS.get(r, r) for r in here], fontsize=7.5)
        low.set_ylim(bottom=0.0)
    axes[0][0].set_ylabel("known probes named rightly")
    axes[1][0].set_ylabel("d'")
    from matplotlib.lines import Line2D

    handles, labels = axes[0][-1].get_legend_handles_labels()
    handles += [Line2D([], [], color=MUTED, linestyle="-"), Line2D([], [], color=MUTED, linestyle="--")]
    labels += ["named: calibrated, 1 % per decision", "found: at a measured FPIR of 1 %"]
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=7, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("The ladder B0-B5, one change at a time (matched distance and pose)", fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    path = folder / "ladder.png"
    fig.savefig(path, dpi=150)
    return [path]


def _pack(plt, pack: str, by_profile: dict, folder: Path) -> list[Path]:
    """A pack's DET curve per profile."""
    from .metrics import probit

    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    for profile, report in by_profile.items():
        if report.get("curve"):
            _det_line(ax, report["curve"], profile, probit)
    _det_axes(ax, probit)
    ax.set_ylabel("false reject rate")
    ax.legend(loc="upper right", fontsize=7.5)
    ax.set_title(f"{pack}: verification (DET)")
    fig.tight_layout()
    path = folder / f"{pack}_det.png"
    fig.savefig(path, dpi=150)
    return [path]
