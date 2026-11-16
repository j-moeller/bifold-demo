"""Curves of an evaluation's results (``bifold-faces eval``, :mod:`.evaluate`) as PNG images, drawn from the results'
JSON alone (so they can be drawn again without measuring): DET curves and FNIR against FPIR per kind of face, the
share of known people named against the eye distance, the turn and the ladder's rungs, per profile.

The profiles keep one colour each in every figure (best blue, licensed orange, commercial aqua), each line is also
labelled at its end, and the intervals are shaded; the numbers themselves are in the results' JSON and the results
note's tables. Needs matplotlib (it comes with Ultralytics; without it no curves are drawn, and the run says so).
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


def _label_end(ax, x, y, text, colour):
    ax.annotate(text, (x, y), xytext=(4, 0), textcoords="offset points", va="center", fontsize=7.5, color=MUTED)


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
                          "ied_budget.png", fixed={"preset": "budget 1080p"}, log=True)
        written += _swept(plt, stills, folder, "turn", "turn", "head turned (degrees)", "turn.png")
        written += _ladder(plt, stills, folder)
    for pack, by_profile in (results.get("packs") or {}).items():
        written += _pack(plt, pack, by_profile, folder)
    plt.close("all")
    return written


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
    fig, axes = plt.subplots(1, len(panels), figsize=(3.3 * len(panels), 3.2), squeeze=False, sharey=True)
    ticks = [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.5]
    for ax, variant in zip(axes[0], panels):
        for profile, report in stills.items():
            entry = report.get("verification") if variant is None else \
                (report.get("conditions") or {}).get("variant", {}).get(f"variant={variant}", {}).get("verification")
            curve = (entry or {}).get("curve")
            if not curve:
                continue
            far = [max(x, 1e-6) for x in curve["x"]]
            frr = [min(max(1.0 - t, 1e-5), 0.999) for t in curve["tar"]]
            colour = PROFILE_COLOURS.get(profile, MUTED)
            ax.plot(probit(far), probit(frr), color=colour, label=profile, drawstyle="steps-post")
            _label_end(ax, probit(far[0]), probit(frr[0]), profile, colour)
        ax.set_xticks(probit(ticks), [f"{t:g}" for t in ticks])
        ax.set_yticks(probit([1e-3, 1e-2, 0.05, 0.2, 0.5]), ["0.001", "0.01", "0.05", "0.2", "0.5"])
        ax.set_xlabel("false accept rate")
        ax.set_title(VARIANT_TITLES.get(variant, "all probes"))
    axes[0][0].set_ylabel("false reject rate")
    axes[0][-1].legend(loc="upper right", fontsize=7.5)
    fig.suptitle("Verification on the stills (every probe against every identity's gallery)", fontsize=10)
    fig.tight_layout()
    path = folder / "det.png"
    fig.savefig(path, dpi=150)
    return [path]


def _swept(plt, stills: dict, folder: Path, breakdown: str, key: str, xlabel: str, name: str, fixed=None,
           log: bool = False) -> list[Path]:
    """The share of known people named rightly at 1 % per decision against one swept condition, per kind of face
    (panels), a line per profile with its interval shaded."""
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
            series.setdefault(parts.get("variant"), {}).setdefault(profile, []).append((x, *named))
    if not series:
        return []
    panels = [v for v in ("own", "renders", "fitted") if v in series]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.3 * len(panels), 3.0), squeeze=False, sharey=True)
    for ax, variant in zip(axes[0], panels):
        for profile, points in series[variant].items():
            points.sort()
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            lo = [p[2][0] if p[2][0] is not None else p[1] for p in points]
            hi = [p[2][1] if p[2][1] is not None else p[1] for p in points]
            colour = PROFILE_COLOURS.get(profile, MUTED)
            ax.fill_between(xs, lo, hi, color=colour, alpha=0.12, linewidth=0)
            ax.plot(xs, ys, color=colour, marker=MARKERS.get(profile, "o"), label=profile)
            _label_end(ax, xs[-1], ys[-1], profile, colour)
        if log:
            ax.set_xscale("log")
            ax.set_xticks([8, 12, 18, 26, 38, 54, 76, 100], ["8", "12", "18", "26", "38", "54", "76", "100"])
            ax.minorticks_off()
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel(xlabel)
        ax.set_title(VARIANT_TITLES.get(variant, variant))
    axes[0][0].set_ylabel("known probes named rightly (1 % per decision)")
    axes[0][-1].legend(loc="lower right", fontsize=7.5)
    extra = ", ".join(f"{k} {v}" for k, v in fixed.items())
    fig.suptitle(f"Named against the {xlabel.split(' (')[0]}" + (f" ({extra})" if extra else ""), fontsize=10)
    fig.tight_layout()
    path = folder / name
    fig.savefig(path, dpi=150)
    return [path]


def _ladder(plt, stills: dict, folder: Path) -> list[Path]:
    """The ladder's rungs: the share of known probes named rightly at 1 % per decision, per profile."""
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
            series.setdefault(variant, {}).setdefault(profile, {})[parts["rung"]] = named
    if not series:
        return []
    rungs = sorted(order, key=lambda r: (r[:2], r))
    panels = [v for v in ("own", "renders") if v in series]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 3.2), squeeze=False, sharey=True)
    for ax, variant in zip(axes[0], panels):
        here = [r for r in rungs if any(r in s for s in series[variant].values())]
        for k, (profile, points) in enumerate(series[variant].items()):
            xs = [here.index(r) + (k - 1) * 0.12 for r in here if r in points]
            ys = [points[r][0] for r in here if r in points]
            err = [[max(0.0, points[r][0] - (points[r][1][0] if points[r][1][0] is not None else points[r][0]))
                    for r in here if r in points],
                   [max(0.0, (points[r][1][1] if points[r][1][1] is not None else points[r][0]) - points[r][0])
                    for r in here if r in points]]
            colour = PROFILE_COLOURS.get(profile, MUTED)
            ax.errorbar(xs, ys, yerr=err, color=colour, marker=MARKERS.get(profile, "o"), linestyle="-",
                        linewidth=1.0, capsize=2, label=profile)
        ax.set_xticks(range(len(here)), [r.replace(" ", "\n", 1) for r in here], fontsize=7.5)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title("the avatars' own faces" if variant == "own" else "faces from set A (B5: the fitted head)")
    axes[0][0].set_ylabel("known probes named rightly (1 % per decision)")
    axes[0][-1].legend(loc="lower right", fontsize=7.5)
    fig.suptitle("The ladder B0-B5, one change at a time (matched distance and pose)", fontsize=10)
    fig.tight_layout()
    path = folder / "ladder.png"
    fig.savefig(path, dpi=150)
    return [path]


def _pack(plt, pack: str, by_profile: dict, folder: Path) -> list[Path]:
    """A pack's DET curve per profile."""
    from .metrics import probit

    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    for profile, report in by_profile.items():
        curve = report.get("curve")
        if not curve:
            continue
        far = [max(x, 1e-6) for x in curve["x"]]
        frr = [min(max(1.0 - t, 1e-5), 0.999) for t in curve["tar"]]
        colour = PROFILE_COLOURS.get(profile, MUTED)
        ax.plot(probit(far), probit(frr), color=colour, label=profile, drawstyle="steps-post")
        _label_end(ax, probit(far[0]), probit(frr[0]), profile, colour)
    ticks = [1e-4, 1e-3, 1e-2, 0.1, 0.5]
    ax.set_xticks(probit(ticks), [f"{t:g}" for t in ticks])
    ax.set_yticks(probit([1e-3, 1e-2, 0.05, 0.2, 0.5]), ["0.001", "0.01", "0.05", "0.2", "0.5"])
    ax.set_xlabel("false accept rate")
    ax.set_ylabel("false reject rate")
    ax.legend(loc="upper right", fontsize=7.5)
    ax.set_title(f"{pack}: verification (DET)")
    fig.tight_layout()
    path = folder / f"{pack}_det.png"
    fig.savefig(path, dpi=150)
    return [path]
