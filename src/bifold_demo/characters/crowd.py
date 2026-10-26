"""Who is in the crowd: avatar names with conference-appropriate demographics, and people built from them.

:func:`sample_crowd` picks avatars from a catalog (``rocketbox_catalog.json``: ``sex``, ``age_group``, ``skin_tone``
on the Monk scale 1-10, ``outfit``, ``conference_fit`` 0-3 and optionally an estimated ``age`` in years):

* demographics: by default about 5 % children, 80 % adults and 15 % older people (``age_group == "older"`` or an
  ``age`` of ``older_from_age`` or more), half women and half men, and every Monk skin-tone band (1-3, 4-6, 7-10)
  represented when the catalog has it, rarer bands a little over-represented relative to the catalog (band shares
  follow the square root of the band sizes). The three are matched jointly (iterative proportional fitting), so that
  e.g. an all-male older group is balanced by more women among the adults;
* only avatars with ``conference_fit >= min_fit`` (2: casual to business clothes, no uniforms), except that an age
  group with too few of them for its share takes its best other avatars (fit 1, e.g. the children), and a pool of
  fewer than ``n / 2`` avatars admits lower fits, best first;
* no avatar twice until every pooled avatar is used; then the repeats follow the shares too and are spread out over
  the list (each name's copies about ``n / copies`` apart);
* groups that are empty or too small give their share to the others; the result is deterministic for ``rng``.

:func:`describe_crowd` counts what a list of names contains, :func:`avatar_style` turns catalog metadata into the
``style`` behaviours take, and :func:`populate` builds :class:`~bifold_demo.characters.character.Character` objects.
"""

from __future__ import annotations

import math
import numbers
import random
from collections import Counter

AGE_SHARES = {"child": 0.05, "adult": 0.80, "older": 0.15}
SEX_SHARES = {"female": 0.5, "male": 0.5}
SKIN_BANDS = {"1-3": (1, 3), "4-6": (4, 6), "7-10": (7, 10)}
OLDER_FROM_AGE = 58
MIN_FIT = 2
SKIN_FLATTEN = 0.5  # default band shares follow (band size) ** SKIN_FLATTEN
_ATTRIBUTES = ("age_group", "sex", "skin_band")


def _as_random(rng) -> random.Random:
    if isinstance(rng, random.Random):
        return rng
    if rng is None or isinstance(rng, numbers.Integral):
        return random.Random(0 if rng is None else int(rng))
    if hasattr(rng, "integers"):
        return random.Random(int(rng.integers(2 ** 62)))
    raise TypeError(f"rng must be an int seed, random.Random or numpy Generator, not {type(rng).__name__}")


def age_group(entry: dict, older_from_age: float = OLDER_FROM_AGE) -> str:
    """``"child"``, ``"adult"`` or ``"older"``: the label, but an estimated ``age >= older_from_age`` is older."""
    group = str(entry.get("age_group") or "adult").lower()
    if group not in AGE_SHARES:
        group = "adult"
    age = entry.get("age")
    if group == "adult" and isinstance(age, (int, float)) and age >= older_from_age:
        return "older"
    return group


def sex(entry: dict) -> str:
    return "female" if str(entry.get("sex", "")).lower().startswith("f") else "male"


def skin_band(entry: dict) -> str:
    """The Monk-scale band (``"1-3"``, ``"4-6"`` or ``"7-10"``) of the entry's ``skin_tone``; ``"?"`` if unknown."""
    tone = entry.get("skin_tone")
    if isinstance(tone, (int, float)):
        for band, (low, high) in SKIN_BANDS.items():
            if low <= tone <= high:
                return band
    return "?"


def _fit(entry: dict, min_fit: int) -> int:
    fit = entry.get("conference_fit")
    return int(fit) if isinstance(fit, (int, float)) else min_fit


def sample_crowd(catalog: dict[str, dict], n: int, rng, weights=None, *, older_from_age: float = OLDER_FROM_AGE,
                 min_fit: int = MIN_FIT) -> list[str]:
    """``n`` avatar names from ``catalog`` (``{name: entry}``) with the demographics described in the module docs.

    Args:
        rng: seed (int), ``random.Random`` or numpy ``Generator``.
        weights: shares replacing the defaults, any of ``{"age_group": {"child": .., "adult": .., "older": ..},
            "sex": {"female": .., "male": ..}, "skin_band": {"1-3": .., "4-6": .., "7-10": ..}}`` (normalised).
        older_from_age: estimated age (``entry["age"]``) from which an adult counts as older.
        min_fit: the lowest ``conference_fit`` used while there are enough avatars.
    """
    n = int(n)
    if n < 0:
        raise ValueError("n must be >= 0")
    if n == 0:
        return []
    if not catalog:
        raise ValueError("the catalog is empty")
    rng = _as_random(rng)
    keys = {name: (age_group(e, older_from_age), sex(e), skin_band(e)) for name, e in catalog.items()}
    fits = {name: _fit(e, min_fit) for name, e in catalog.items()}
    weights = dict(weights or {})
    for key in weights:
        if key not in _ATTRIBUTES:
            raise ValueError(f"unknown weights {key!r}; use {_ATTRIBUTES}")
    age_shares = _normalised(weights.get("age_group", AGE_SHARES))
    pool = _pool(sorted(catalog), keys, fits, n, age_shares, min_fit)
    band_sizes = Counter(keys[name][2] for name in pool)
    shares = (age_shares, _normalised(weights.get("sex", SEX_SHARES)),
              _normalised(weights.get("skin_band", {b: c ** SKIN_FLATTEN for b, c in band_sizes.items()})))
    cells: dict[tuple, list[str]] = {}
    for name in pool:
        cells.setdefault(keys[name], []).append(name)
    for names in cells.values():  # best fit first, random among equals
        order = {name: rng.random() for name in names}
        names.sort(key=lambda name: (-fits[name], order[name]))
    counts: Counter = Counter()
    used = Counter()  # picks per attribute value so far
    taken = Counter()  # picks per cell so far (repeats cycle through the cell's avatars)
    first = min(n, len(pool))
    rounds = [(first, {key: len(names) for key, names in cells.items()})]  # everyone once before any repeat
    if n > first:
        rounds.append((n - first, {key: n for key in cells}))  # then repeats, as many per cell as the shares ask
    for m, capacity in rounds:
        total = sum(counts.values()) + m
        targets = [{v: max(0.0, share.get(v, 0.0) * total - used[(a, v)]) for v in share}
                   for a, share in enumerate(shares)]
        picks = _allocate(m, capacity, targets, rng)
        for key, k in picks.items():
            names = cells[key]
            for j in range(k):
                counts[names[(taken[key] + j) % len(names)]] += 1
            taken[key] += k
            for a, value in enumerate(key):
                used[(a, value)] += k
    return _spread(counts, rng)


def _normalised(shares) -> dict:
    shares = {k: float(v) for k, v in dict(shares).items() if float(v) > 0.0}
    total = sum(shares.values())
    if total <= 0.0:
        raise ValueError("shares must not all be zero")
    return {k: v / total for k, v in shares.items()}


def _pool(names, keys, fits, n, age_shares, min_fit) -> list[str]:
    """The avatars to choose from: ``conference_fit >= min_fit``, relaxed for small age groups and small pools."""
    pool = {name for name in names if fits[name] >= min_fit}
    for group, share in age_shares.items():
        need = math.ceil(n * share - 1e-9)
        members = [name for name in names if keys[name][0] == group]
        for level in range(min_fit - 1, 0, -1):
            if sum(1 for name in members if name in pool) >= need:
                break
            pool.update(name for name in members if fits[name] == level)
    level = min_fit - 1
    while len(pool) < n / 2 and level >= 1:
        pool.update(name for name in names if fits[name] >= level)
        level -= 1
    return [name for name in names if name in pool] or list(names)


def _allocate(m: int, capacity: dict[tuple, int], targets: list[dict], rng: random.Random) -> dict[tuple, int]:
    """Split ``m`` picks over cells ``(age, sex, band)`` (at most ``capacity`` each) to match the marginal ``targets``.

    Expected counts come from iterative proportional fitting over the cells, capping full cells and refitting the
    rest; they are rounded by systematic sampling (each cell gets the floor or the ceiling of its expectation) and
    then every band the cells offer gets at least one pick when ``m`` allows.
    """
    cells = sorted(key for key, c in capacity.items() if c > 0)
    fixed: dict[tuple, float] = {}
    while True:
        free = [key for key in cells if key not in fixed]
        rest = m - sum(fixed.values())
        expected = _fit_marginals(free, rest, targets, fixed)
        over = [key for key in free if expected[key] > capacity[key] + 1e-9]
        if not over:
            break
        for key in over:
            fixed[key] = float(capacity[key])
    expected.update(fixed)
    start, cumulative, counts = rng.random(), 0.0, {}
    for key in cells:
        before = math.floor(cumulative + start)
        cumulative += expected[key]
        counts[key] = min(capacity[key], math.floor(cumulative + start + 1e-9) - before)
    error = sum(counts.values()) - m  # from floating-point rounding (and capping) only
    for key in sorted(cells, key=lambda k: (counts[k] - expected[k]) * (1 if error < 0 else -1)):
        if error < 0 and counts[key] < capacity[key]:
            counts[key] += 1
            error += 1
        elif error > 0 and counts[key] > 0:
            counts[key] -= 1
            error -= 1
    bands = sorted({key[2] for key in cells})
    if m >= len(bands):
        for band in bands:
            if any(counts[key] for key in cells if key[2] == band):
                continue
            give = max((key for key in cells if key[2] == band), key=lambda k: expected[k])
            donors = [key for key in cells if counts[key] > 0 and key[2] != band and
                      sum(counts[k] for k in cells if k[2] == key[2]) > 1]
            if not donors:
                continue
            take = max(donors, key=lambda k: (k[:2] == give[:2], counts[k] - expected[k], counts[k]))
            counts[take] -= 1
            counts[give] += 1
    return {key: k for key, k in counts.items() if k}


def _fit_marginals(cells: list[tuple], m: float, targets: list[dict], fixed: dict[tuple, float]) -> dict:
    """Iterative proportional fitting of ``m`` over ``cells`` to the marginal targets left after ``fixed``."""
    if not cells or m <= 0:
        return {key: 0.0 for key in cells}
    goals = []
    for a, target in enumerate(targets):
        values = sorted({key[a] for key in cells})
        goal = {v: max(0.0, target.get(v, 0.0) - sum(c for key, c in fixed.items() if key[a] == v)) for v in values}
        if sum(goal.values()) <= 1e-12:
            goal = {v: 1.0 for v in values}
        scale = m / sum(goal.values())
        goals.append({v: g * scale for v, g in goal.items()})
    expected = {key: 1.0 for key in cells}
    for _ in range(200):
        change = 0.0
        for a, goal in enumerate(goals):
            sums: dict = {}
            for key in cells:
                sums[key[a]] = sums.get(key[a], 0.0) + expected[key]
            for key in cells:
                if sums[key[a]] > 0.0:
                    new = expected[key] * goal[key[a]] / sums[key[a]]
                    change = max(change, abs(new - expected[key]))
                    expected[key] = new
        if change < 1e-9:
            break
    return expected


def _spread(counts: Counter, rng: random.Random) -> list[str]:
    """The names, each as often as counted, in random order with repeated names evenly spaced."""
    keyed = []
    for name in sorted(counts):
        k, u = counts[name], rng.random()
        keyed += [((j + u) / k, rng.random(), name) for j in range(k)]
    return [name for *_, name in sorted(keyed)]


def describe_crowd(catalog: dict[str, dict], names, older_from_age: float = OLDER_FROM_AGE) -> dict:
    """Counts for ``names``: ``people``, ``unique``, ``most_repeats``, and per ``age_group`` (older as in
    :func:`sample_crowd`), ``sex``, ``skin_band``, ``outfit`` and ``conference_fit``."""
    names = list(names)
    entries = [catalog[name] for name in names]
    copies = Counter(names)

    def count(values, order=()):
        found = Counter(values)
        return {key: found.get(key, 0) for key in [*order, *sorted(set(found) - set(order), key=str)]}

    return {
        "people": len(names),
        "unique": len(copies),
        "most_repeats": max(copies.values(), default=0),
        "age_group": count((age_group(e, older_from_age) for e in entries), AGE_SHARES),
        "sex": count((sex(e) for e in entries), SEX_SHARES),
        "skin_band": count((skin_band(e) for e in entries), SKIN_BANDS),
        "outfit": count(str(e.get("outfit", "?")) for e in entries),
        "conference_fit": count(e.get("conference_fit") for e in entries),
    }


def avatar_style(entry: dict, model=None, motions=None, older_from_age: float = OLDER_FROM_AGE) -> dict:
    """The ``style`` behaviours take, from catalog metadata: ``sex``, ``age_group`` (older from the estimated age
    too) and, given the model and a :class:`~.behaviour.MotionLibrary`, the ``stride_scale`` that keeps the feet
    from sliding."""
    style = {"sex": sex(entry), "age_group": age_group(entry, older_from_age)}
    if model is not None and motions is not None and motions.has("walk"):
        from .behaviour import stride_scale

        style["stride_scale"] = stride_scale(model, motions.role("walk", style["sex"]))
    return style


def populate(library, names, behaviours, *, rng=0, older_from_age: float = OLDER_FROM_AGE, **character_options):
    """:class:`~bifold_demo.characters.character.Character` objects for ``names`` (e.g. from :func:`sample_crowd`).

    Args:
        library: a :class:`~.library.CharacterLibrary` (anything with ``available()``, ``load(name)``, ``motions()``).
        names: avatar names; ``behaviours``: one per name, each a callable taking ``style``, ``library`` and ``rng``
            keywords, e.g. ``functools.partial(walk_by, [(-9, 0, -7), (9, 0, -7)], camera=CAMERA_POSITION)``.
        rng: seed for the behaviours' seeds. Other keywords go to ``Character`` (``name`` gets ``"<avatar>_<i>"``).
    """
    from .character import Character

    names, behaviours = list(names), list(behaviours)
    if len(names) != len(behaviours):
        raise ValueError(f"{len(names)} names but {len(behaviours)} behaviours")
    rng = _as_random(rng)
    available, motions = library.available(), library.motions()
    people = []
    for i, (name, make) in enumerate(zip(names, behaviours)):
        model = library.load(name)
        style = avatar_style(available.get(name, {}), model, motions, older_from_age)
        behaviour = make(style=style, library=motions, rng=rng.randrange(2 ** 31))
        people.append(Character(model, behaviour=behaviour, name=f"{name}_{i}", **character_options))
    return people
