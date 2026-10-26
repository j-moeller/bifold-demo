"""Text to design without a model: short descriptions turned into fills and outfits by keywords.

:func:`parse_fill` reads one fabric: colours (names from :data:`.fills.COLORS`, optionally ``light``, ``dark``,
``pale``, ``deep`` or ``bright``, or ``#rrggbb``), a pattern word and a few modifiers:

* ``"navy"``: a solid colour; ``"navy pinstripe"``: the pattern in that colour, with its usual second colour;
* ``"red and black checks"``: the first colour is the ground, the second the pattern's;
* ``"white polka dots on black"``: what comes before ``on`` is the pattern's colour, after it the ground;
* sizes: ``tiny``, ``small``/``fine``/``thin``/``narrow``, ``large``/``big``/``bold``/``wide``/``broad``, ``huge``;
  directions: ``vertical``, ``horizontal``, ``diagonal``.

Pattern words and their synonyms are in :data:`PATTERN_WORDS` (``plaid`` is tartan, ``dogtooth`` houndstooth,
``camouflage`` camo, ``jeans`` denim, ...). The same words always give the same fill: it is the instant,
deterministic "describe a design" (a generative model plugs in later as another :class:`~.fills.Fill`).

:func:`parse_outfit` reads a whole outfit, split into parts at commas, ``;``, ``with``, layer words (``over``,
``under``) and ``and`` between garments: ``"red tartan jacket, white shirt and black trousers"``, ``"a white shirt
under a navy blazer"``. A garment word names the region of its part (:data:`GARMENT_WORDS`; the last one, as in ``"a
white dress shirt"``), a suit dresses the top and the bottoms, and a garment word may imply a fabric (``jeans`` are
denim). A part without a garment word dresses the top, unless another part names the top; after ``with`` it tells
more about the garment before it (``"a white shirt with navy stripes"``). Words it does not know are left out, and so
are parts it understands nothing of (``"please"``) or that are about details it cannot dress (:data:`DETAIL_WORDS`:
``"with gold buttons"``): ``"a nice navy pinstripe suit, please"`` works. It raises ``ValueError`` only when it
understands no part at all.
"""

from __future__ import annotations

import re

from . import patterns
from .fills import COLORS, SHADES, Fill, Pattern, Solid, color, shade

PATTERN_WORDS = {
    "stripes": "stripes", "stripe": "stripes", "striped": "stripes", "stripy": "stripes", "breton": "stripes",
    "pinstripe": "pinstripe", "pinstripes": "pinstripe", "pinstriped": "pinstripe", "pin stripe": "pinstripe",
    "pin stripes": "pinstripe", "chalk stripe": "pinstripe", "chalkstripe": "pinstripe",
    "checks": "checks", "check": "checks", "checked": "checks", "checkered": "checks", "chequered": "checks",
    "checkerboard": "checks", "checkered pattern": "checks",
    "windowpane": "windowpane", "window pane": "windowpane", "grid": "windowpane",
    "gingham": "gingham", "vichy": "gingham",
    "tartan": "tartan", "plaid": "tartan", "buffalo check": "gingham",
    "houndstooth": "houndstooth", "hounds tooth": "houndstooth", "dogtooth": "houndstooth",
    "pied de poule": "houndstooth",
    "polka dots": "polka dots", "polka dot": "polka dots", "polka": "polka dots", "dots": "polka dots",
    "dot": "polka dots", "dotted": "polka dots", "spots": "polka dots", "spotted": "polka dots", "spotty": "polka dots",
    "twill": "twill", "gabardine": "twill",
    "herringbone": "herringbone", "chevron": "herringbone", "tweed": "herringbone",
    "denim": "denim", "jeans": "denim", "chambray": "denim",
    "camo": "camo", "camouflage": "camo", "military": "camo",
}
SCALES = {"tiny": 0.45, "small": 0.65, "fine": 0.65, "thin": 0.65, "narrow": 0.65, "skinny": 0.6, "mini": 0.5,
          "large": 1.6, "big": 1.6, "bold": 1.6, "wide": 1.6, "broad": 1.6, "chunky": 1.8, "huge": 2.5,
          "giant": 2.5, "oversized": 2.2}
ROTATIONS = {"vertical": 0.0, "horizontal": 90.0, "diagonal": 45.0, "slanted": 45.0, "bias": 45.0}
CAMO_PALETTES = {"desert": ((196, 170, 120), (150, 112, 74)), "urban": ((128, 130, 134), (70, 72, 76)),
                 "snow": ((226, 228, 230), (150, 154, 160)), "woodland": ((86, 92, 52), (92, 64, 38)),
                 "navy": ((40, 52, 86), (24, 30, 52))}
GARMENT_WORDS = {
    "suit": ("suit", "tracksuit", "jumpsuit", "overalls"),  # the top and the bottoms
    "top": ("jacket", "blazer", "coat", "top", "jumper", "sweater", "pullover", "hoodie", "cardigan", "vest",
            "waistcoat", "gilet", "parka", "anorak", "robe", "dress", "tunic", "t shirt", "tshirt", "tee"),
    "shirt": ("shirt", "blouse", "collar", "polo", "turtleneck", "undershirt"),
    "bottoms": ("trousers", "pants", "jeans", "skirt", "shorts", "slacks", "chinos", "leggings", "joggers"),
    "shoes": ("shoes", "shoe", "boots", "boot", "sneakers", "trainers", "loafers", "heels", "sandals", "footwear"),
    "other": ("belt", "tie", "trims", "trim", "details"),
}
_IMPLIED = {"jeans": "denim"}
DETAIL_WORDS = {"button", "buttons", "pocket", "pockets", "zip", "zips", "zipper", "zippers", "logo", "logos", "lapel",
                "lapels", "lining", "sleeve", "sleeves", "cuff", "cuffs", "hood", "patch", "patches", "badge", "badges",
                "embroidery", "lettering", "stitching"}  # (not "print": "a red floral print dress" is a fabric)
SEPARATORS = r"[,;&+]|\b(?:with|over|under|underneath|beneath|plus)\b"


def _tokens(text: str) -> list[str]:
    text = text.lower().replace("-", " ").replace("_", " ")
    text = re.sub(r"[^a-z0-9#\s]", " ", text)
    return text.split()


def _phrases(words, table) -> tuple[str, int] | None:
    """The longest phrase of ``table`` (2 words or 1) at the start of ``words``: ``(key, words used)``."""
    for n in (3, 2, 1):
        if len(words) >= n:
            key = " ".join(words[:n])
            if key in table:
                return key, n
    return None


def parse_fill(text: str) -> Fill:
    """A fill from a short description (see the module docs). Raises ``ValueError`` when no colour or pattern is
    recognised."""
    words = _tokens(text)
    before, after = [], []  # colours before and after "on"
    kind, scale, rotation, palette, tone = None, 1.0, None, None, None
    pending = None  # a shade word waiting for its colour
    on = False
    i = 0
    while i < len(words):
        rest = words[i:]
        word = rest[0]
        as_pattern, as_colour = _phrases(rest, PATTERN_WORDS), _phrases(rest, COLORS)
        if word == "on" and (before or kind):
            on = True
            i += 1
        elif word in CAMO_PALETTES and len(rest) > 1 and PATTERN_WORDS.get(rest[1]) == "camo":
            palette = CAMO_PALETTES[word]
            i += 1
        elif word.startswith("#") or (as_colour and (not as_pattern or as_colour[1] >= as_pattern[1])):
            rgb = color(word) if word.startswith("#") else COLORS[as_colour[0]]
            if pending:
                rgb, pending = shade(rgb, pending), None
            (after if on else before).append(rgb)
            i += 1 if word.startswith("#") else as_colour[1]
        elif as_pattern:
            kind = PATTERN_WORDS[as_pattern[0]]
            tone, pending = pending, None  # "light denim": a shade of the pattern's own colour
            i += as_pattern[1]
        else:
            if word in SHADES:
                pending = word
            elif word in SCALES:
                scale *= SCALES[word]
            elif word in ROTATIONS:
                rotation = ROTATIONS[word]
            i += 1
    if on:  # "<marks> <pattern> on <ground>"
        primary, secondary = (after[0] if after else None), (before[0] if before else None)
    else:
        primary, secondary = (before[0] if before else None), (before[1] if len(before) > 1 else None)
    if kind is None:
        if primary is None:
            raise ValueError(f"cannot make a fabric of {text!r}: no colour or pattern I know "
                             f"(patterns: {', '.join(patterns.PATTERNS)}; colours like {', '.join(list(COLORS)[:8])})")
        return Solid(primary)
    if primary is None and palette is not None:
        primary, secondary = palette
    if primary is None and tone is not None:
        primary = shade(patterns.DEFAULT_PRIMARY.get(kind, COLORS["navy"]), tone)
    if kind == "stripes" and "breton" in words:
        primary, secondary = primary or COLORS["white"], secondary or COLORS["navy"]
        rotation = 90.0 if rotation is None else rotation
    return Pattern(kind, primary, secondary, patterns.PATTERNS[kind][1] * scale, rotation or 0.0)


def _garment(words: list[str]) -> tuple[str | None, list[str]]:
    """The region a phrase names, and the phrase without that garment word (but with a fabric it implies). The last
    garment word names it, the longer one when two end together: a "dress shirt" is a shirt, a "t shirt" a top; the
    other garment words stay with the fabric ("jeans jacket": a jacket of denim)."""
    best = None  # (end, length, region, start, name)
    for region, names in GARMENT_WORDS.items():
        for name in names:
            parts = name.split()
            for k in range(len(words) - len(parts) + 1):
                if words[k:k + len(parts)] == parts and (best is None or (k + len(parts), len(parts)) > best[:2]):
                    best = (k + len(parts), len(parts), region, k, name)
    if best is None:
        return None, words
    end, _, region, start, name = best
    rest = words[:start] + words[end:]
    if name in _IMPLIED and _IMPLIED[name] not in rest:
        rest.append(_IMPLIED[name])
    return region, rest


def _pieces(text: str) -> list[tuple[list[str], str | None]]:
    """The parts of a description of an outfit: ``(words, the separator before it)``."""
    pieces = []
    split = re.split(f"({SEPARATORS})", text.lower())
    for k in range(0, len(split), 2):
        before = split[k - 1].strip() if k else None
        words = _tokens(split[k])
        # "and" between two garments splits them; "black and white stripes" stays one fabric
        current = []
        for i, word in enumerate(words):
            if word == "and" and _garment(current)[0] is not None and _garment(words[i + 1:])[0] is not None:
                pieces.append((current, before))
                current, before = [], "and"
            else:
                current.append(word)
        if current:
            pieces.append((current, before))
    return pieces


def parse_outfit(text: str) -> dict[str, Fill]:
    """``{region: fill}`` from a description of an outfit (see the module docs)."""
    out: dict[str, Fill] = {}
    loose: list[Fill] = []  # fabrics without a garment: the top's, unless a part names the top
    previous = None  # (region, fabric words) of the last part that named a garment
    for words, before in _pieces(text):
        if DETAIL_WORDS.intersection(words):
            continue  # "with gold buttons": a detail it cannot dress
        region, rest = _garment(words)
        if region is None and before == "with" and previous is not None:
            region, rest = previous[0], previous[1] + rest  # "a white shirt with navy stripes"
        try:
            fill = parse_fill(" ".join(rest))
        except ValueError:
            continue  # nothing it knows: "please", "a jacket"
        if region is None:
            loose.append(fill)
            continue
        for name in ("top", "bottoms") if region == "suit" else (region,):
            out[name] = fill
        previous = (region, rest)
    if loose and "top" not in out:
        out["top"] = loose[0]
    if not out:
        raise ValueError(f"cannot make an outfit of {text!r}: no colour or pattern I know "
                         f"(patterns: {', '.join(patterns.PATTERNS)}; colours like {', '.join(list(COLORS)[:8])})")
    return out
