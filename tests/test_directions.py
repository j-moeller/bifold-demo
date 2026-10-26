import numpy as np
import pytest

from bifold_morph.directions import GENDER_END, Directions, fit, ridge, unit, without


def synthetic(n=4000, seed=0):
    """Codes where age follows one direction and gender another, partly overlapping, like in StyleGAN."""
    rng = np.random.default_rng(seed)
    scales = np.linspace(0.3, 2.0, 512)  # columns that vary by different amounts
    w = rng.normal(size=(n, 512)) * scales
    u = unit(rng.normal(size=512) * scales)
    v = unit(0.5 * u + unit(rng.normal(size=512) * scales))
    age = 40 + 8 * (w @ u) + rng.normal(0, 2, n)
    male = 1 / (1 + np.exp(-3 * (w @ v)))
    return w, age, male, u, v


def test_ridge_recovers_a_linear_model():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(2000, 20))
    beta = rng.normal(size=20)
    y = x @ beta + 3.0
    assert np.allclose(ridge(x, y, alpha=1e-6), beta, atol=1e-3)


def test_directions_are_conditional():
    w, age, male, u, v = synthetic()
    d = fit(w, age, male)
    beta_age = ridge(w, age)
    p = np.clip(male, 0.02, 0.98)
    beta_gender = ridge(w, np.log(p / (1 - p)))
    assert np.linalg.norm(d.age) == pytest.approx(1.0) and np.linalg.norm(d.gender) == pytest.approx(1.0)
    # a step along "age" leaves the gender model's reading alone, and the other way round
    assert abs(d.age @ beta_gender) < 1e-9 * np.linalg.norm(beta_gender) + 1e-9
    assert abs(d.gender @ beta_age) < 1e-9 * np.linalg.norm(beta_age) + 1e-9
    # and they still point roughly the true ways
    assert d.age @ u > 0.5 and d.gender @ v > 0.5
    assert d.years_per_unit > 1.0 and d.gender_gap > 0.5
    assert d.info["gender_accuracy"] > 0.9 and d.info["age_r2"] > 0.8


def test_fit_needs_both_genders():
    w, age, male, *_ = synthetic(200)
    with pytest.raises(ValueError):
        fit(w, age, np.ones_like(male))


def test_offset_and_round_trip(tmp_path):
    d = Directions(unit(np.arange(512.0)), unit(np.ones(512)), years_per_unit=10.0, gender_gap=2.0,
                   info={"samples": 3})
    assert np.allclose(d.offset(20, 0), 2 * d.age)
    assert np.allclose(d.offset(0, -1), -GENDER_END * 2.0 * d.gender)
    path = d.save(tmp_path / "sub" / "directions.npz")
    back = Directions.load(path)
    assert np.allclose(back.age, d.age, atol=1e-6) and np.allclose(back.gender, d.gender, atol=1e-6)
    assert back.years_per_unit == 10.0 and back.gender_gap == 2.0 and back.info == {"samples": 3}


def test_without():
    a, b = np.array([1.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])
    assert np.allclose(without(a, b), [0.0, 1.0, 0.0])
