import functools
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from bifold_demo.characters import behaviour as B
from bifold_demo.characters.crowd import age_group, avatar_style, describe_crowd, populate, sample_crowd, skin_band

CATALOG_FILE = Path(__file__).resolve().parents[1] / "src" / "bifold_demo" / "characters" / "rocketbox_catalog.json"


def make_catalog(children=2, older=3, older_women=True, fit_children=1, uniforms=10, ages=False):
    """A catalog shaped like the Rocketbox one: mostly light-skinned adults, few children, older and dark-skinned."""
    catalog = {}

    def add(prefix, sex, group, tone, fit, outfit="casual", **extra):
        name = f"{prefix}_{sex[0]}{len(catalog):03d}"
        catalog[name] = {"sex": sex, "age_group": group, "skin_tone": tone, "outfit": outfit, "conference_fit": fit,
                         **extra}

    for sex in ("female", "male"):
        for tone, count in ((2, 18), (5, 6), (8, 2)):
            for k in range(count):
                add("Adult", sex, "adult", tone, 3 if k % 2 else 2, **({"age": 30 + k} if ages else {}))
        for k in range(children):
            add("Child", sex, "child", (2, 5, 8)[k % 3], fit_children)
        if older_women or sex == "male":
            for k in range(older):
                if ages:  # nobody labelled older, but estimated ages say so
                    add("Grey", sex, "adult", (2, 5)[k % 2], 3, age=60 + k)
                else:
                    add("Grey", sex, "older", (2, 5)[k % 2], 3)
        for k in range(uniforms // 2):
            add("Uniform", sex, "adult", 3, k % 2, outfit="uniform")
    return catalog


def shares(catalog, names, key, **kwargs):
    counts = describe_crowd(catalog, names, **kwargs)[key]
    return {k: v / len(names) for k, v in counts.items()}


def test_default_demographics_over_many_crowds():
    catalog = make_catalog()
    totals, n = Counter(), 40
    for seed in range(150):
        names = sample_crowd(catalog, n, seed)
        assert len(names) == n and len(set(names)) == n  # no repeats while the pool lasts
        summary = describe_crowd(catalog, names)
        assert all(summary["skin_band"][band] > 0 for band in ("1-3", "4-6", "7-10"))
        assert all(catalog[name]["conference_fit"] >= 2 or catalog[name]["age_group"] == "child" for name in names)
        for key in ("age_group", "sex"):
            totals.update({(key, k): v for k, v in summary[key].items()})
    total = 150 * n
    assert totals[("age_group", "child")] / total == pytest.approx(0.05, abs=0.01)
    assert totals[("age_group", "adult")] / total == pytest.approx(0.80, abs=0.015)
    assert totals[("age_group", "older")] / total == pytest.approx(0.15, abs=0.015)
    assert totals[("sex", "female")] / total == pytest.approx(0.5, abs=0.01)


def test_rare_skin_tones_are_over_represented_a_little():
    catalog = make_catalog()
    pool = [name for name, e in catalog.items() if e["conference_fit"] >= 2]
    catalog_share = sum(skin_band(catalog[name]) == "7-10" for name in pool) / len(pool)
    crowd = np.mean([shares(catalog, sample_crowd(catalog, 30, seed), "skin_band")["7-10"] for seed in range(60)])
    assert catalog_share < crowd < 0.2
    for n in (3, 5, 12):  # every band from three people up
        assert all(describe_crowd(catalog, sample_crowd(catalog, n, s))["skin_band"][b] for s in range(20)
                   for b in ("1-3", "4-6", "7-10"))


def test_older_people_from_estimated_age():
    catalog = make_catalog(ages=True)
    assert not any(e["age_group"] == "older" for e in catalog.values())
    assert age_group(next(e for name, e in catalog.items() if name.startswith("Grey"))) == "older"
    older = np.mean([shares(catalog, sample_crowd(catalog, 40, s), "age_group")["older"] for s in range(40)])
    assert older == pytest.approx(0.15, abs=0.02)
    names = sample_crowd(catalog, 40, 1, older_from_age=100)
    assert describe_crowd(catalog, names, older_from_age=100)["age_group"]["older"] == 0
    assert describe_crowd(catalog, names)["age_group"]["older"] > 0  # counted with the default threshold


def test_empty_and_tiny_groups_give_their_share_away():
    catalog = make_catalog(children=0, older=1, older_women=False)
    for seed in range(30):
        names = sample_crowd(catalog, 40, seed)
        summary = describe_crowd(catalog, names)
        assert len(names) == 40 and len(set(names)) == 40
        assert summary["age_group"]["child"] == 0 and summary["age_group"]["older"] == 1  # the only older man, once
        assert summary["sex"]["female"] == 20  # the women make up for the all-male older group
    no_women = {k: v for k, v in make_catalog().items() if v["sex"] == "male"}
    assert len(sample_crowd(no_women, 20, 0)) == 20


def test_conference_fit_and_relaxed_pools():
    catalog = make_catalog(fit_children=1)
    picked = {name for seed in range(30) for name in sample_crowd(catalog, 40, seed)}
    assert not any(name.startswith("Uniform") for name in picked)  # fit 0 and 1 adults are never needed
    assert any(name.startswith("Child") for name in picked)  # children have fit 1 only, and are admitted
    few = {name: e for name, e in catalog.items() if name.startswith(("Uniform", "Child")) or
           (e["sex"] == "female" and e["age_group"] == "older")}  # 3 fitting avatars for 12 people: too few
    names = sample_crowd(few, 12, 0)
    assert len(names) == 12 and len(set(names)) >= 6
    assert not any(few[name]["conference_fit"] == 0 for name in names)


def test_repeats_only_after_the_pool_and_spread_out():
    catalog = {f"A{i}": {"sex": ("female", "male")[i % 2], "age_group": "adult", "skin_tone": 2 + 3 * (i % 3),
                         "conference_fit": 3} for i in range(8)}
    names = sample_crowd(catalog, 21, 4)
    counts = Counter(names)
    assert len(counts) == 8 and set(counts.values()) <= {2, 3}
    for name in counts:
        positions = [i for i, n in enumerate(names) if n == name]
        assert min(np.diff(positions)) >= 4  # copies about 21 / copies apart
    assert describe_crowd(catalog, names)["most_repeats"] == 3


def test_sampling_is_deterministic():
    catalog = make_catalog()
    assert sample_crowd(catalog, 30, 7) == sample_crowd(catalog, 30, 7)
    assert sample_crowd(catalog, 30, 7) != sample_crowd(catalog, 30, 8)
    assert sample_crowd(catalog, 30, np.random.default_rng(2)) == sample_crowd(catalog, 30, np.random.default_rng(2))
    assert sample_crowd(dict(reversed(list(catalog.items()))), 30, 7) == sample_crowd(catalog, 30, 7)


def test_weights_and_arguments():
    catalog = make_catalog()
    adults = sample_crowd(catalog, 20, 1, weights={"age_group": {"adult": 1.0}})
    assert describe_crowd(catalog, adults)["age_group"]["adult"] == 20
    women = sample_crowd(catalog, 20, 1, weights={"sex": {"female": 1.0}})
    assert describe_crowd(catalog, women)["sex"]["female"] == 20
    assert sample_crowd(catalog, 0, 1) == []
    with pytest.raises(ValueError):
        sample_crowd(catalog, 5, 1, weights={"height": {"tall": 1.0}})
    with pytest.raises(ValueError):
        sample_crowd({}, 5, 1)
    summary = describe_crowd(catalog, adults)
    assert summary["people"] == 20 and sum(summary["skin_band"].values()) == 20
    assert sum(summary["conference_fit"].values()) == 20 and sum(summary["outfit"].values()) == 20


@pytest.mark.skipif(not CATALOG_FILE.is_file(), reason="no Rocketbox catalog")
def test_real_catalog_crowds():
    catalog = json.loads(CATALOG_FILE.read_text(encoding="utf-8"))
    for n in (12, 30, 60):
        names = sample_crowd(catalog, n, 1)
        summary = describe_crowd(catalog, names)
        assert len(names) == n and summary["unique"] >= min(n, 50)  # no repeats before the ~58 fitting avatars
        assert all(summary["skin_band"][band] for band in ("1-3", "4-6", "7-10"))
        assert summary["age_group"]["older"] >= 1 and abs(summary["sex"]["female"] - summary["sex"]["male"]) <= n / 5
        assert all(catalog[name]["conference_fit"] >= 2 or catalog[name]["age_group"] == "child" for name in names)


def test_populate_builds_characters_with_matching_strides():
    from test_behaviour import CAMERA, synthetic_clips, synthetic_model

    catalog = {"Tall": {"sex": "male", "age_group": "adult", "age": 64}, "Small": {"sex": "female",
                                                                                     "age_group": "child"}}
    models = {"Tall": synthetic_model(1.05), "Small": synthetic_model(0.7)}
    motions = B.MotionLibrary(synthetic_clips())

    class Library:
        def available(self):
            return catalog

        def load(self, name):
            return models[name]

        def motions(self):
            return motions

    walk = functools.partial(B.walk_by, [(-5, 0, -4), (5, 0, -4)], camera=CAMERA)
    people = populate(Library(), ["Tall", "Small"], [walk, walk], rng=3)
    assert [p.name for p in people] == ["Tall_0", "Small_1"]
    tall, small = (p.behaviour.style for p in people)
    assert tall.age == "older" and small.age == "child" and small.sex == "female"
    assert tall.stride == pytest.approx(1.05, rel=0.01) and small.stride == pytest.approx(0.7, rel=0.01)
    people[0].update(3.0)
    assert people[0].state is not None and people[0].skin_palette.shape == (len(models["Tall"].skeleton), 4, 4)
    assert avatar_style(catalog["Small"]) == {"sex": "female", "age_group": "child"}
    with pytest.raises(ValueError):
        populate(Library(), ["Tall"], [walk, walk])
