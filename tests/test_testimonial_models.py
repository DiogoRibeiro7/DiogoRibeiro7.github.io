"""Check the article's model against numerical integration and generated data."""

import importlib.util
from math import sqrt
from pathlib import Path
from statistics import NormalDist, mean, variance

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "testimonial", ROOT / "assets/viz/generate_testimonial_figures.py")
model = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(model)


def test_selected_moments_against_direct_numerical_integration():
    # Integrate the joint-model conditional mean over the baseline tail;
    # this does not use the generator's truncated-normal formula.
    normal = NormalDist(50, 10)
    dx = 0.002
    xs = [65 + (i + .5) * dx for i in range(35000)]
    weights = [normal.pdf(x) * dx for x in xs]
    mass = sum(weights)
    row = model.selected_moments()
    assert row["selected_fraction"] == pytest.approx(mass, abs=1e-9)
    assert row["baseline_mean"] == pytest.approx(sum(x*w for x, w in zip(xs, weights)) / mass, abs=1e-6)
    followup = sum((50 + .64*(x-50))*w for x, w in zip(xs, weights)) / mass
    assert row["followup_mean"] == pytest.approx(followup, abs=1e-6)


def test_latent_variable_simulation_agrees_with_exact_selected_change():
    pairs = model.simulate_pairs(n=300000)
    changes = [y-x for x, y in pairs if x >= 65]
    row = model.selected_moments()
    standard_error = sqrt(row["change_variance"] / len(changes))
    assert abs(mean(changes) - row["mean_change"]) < 5 * standard_error
    assert variance(changes) == pytest.approx(row["change_variance"], rel=.04)


def test_counterfactual_effect_is_preserved_despite_all_three_improving():
    control = model.selected_moments()
    for effect in (-3, 3):
        intervention = model.selected_moments(effect=effect)
        assert intervention["mean_change"] < 0
        assert intervention["followup_mean"] - control["followup_mean"] == pytest.approx(effect)
        assert intervention["mean_change"] - control["mean_change"] == pytest.approx(effect)


def test_perfect_repeatability_and_independent_measurement_limits():
    perfect = model.selected_moments(within_sd=0)
    independent = model.selected_moments(between_sd=0)
    assert perfect["mean_change"] == perfect["change_variance"] == 0
    assert independent["followup_mean"] == 50
    assert independent["slope"] == 0


def test_score_units_and_baseline_averaging():
    original = model.selected_moments()
    rescaled = model.selected_moments(cutoff=200, mu=155, between_sd=24, within_sd=18)
    assert rescaled["selected_fraction"] == pytest.approx(original["selected_fraction"])
    assert rescaled["mean_change"] == pytest.approx(3 * original["mean_change"])
    assert rescaled["change_variance"] == pytest.approx(9 * original["change_variance"])
    rows = [model.selected_moments(baseline_count=k) for k in (1, 4, 16)]
    assert all(left["mean_change"] < right["mean_change"] < 0 for left, right in zip(rows, rows[1:]))
    assert all(left["selected_fraction"] > right["selected_fraction"] for left, right in zip(rows, rows[1:]))


def test_invalid_model_parameters():
    for parameters in ({"within_sd": -1}, {"between_sd": 0, "within_sd": 0},
                       {"baseline_count": 0}, {"baseline_count": 1.5},
                       {"baseline_count": True}, {"effect": float("nan")}):
        with pytest.raises(ValueError):
            model.selected_moments(**parameters)
