"""Independent checks of the article's Gaussian probability calculations."""

import importlib.util
from math import sqrt
from pathlib import Path
from random import Random
from statistics import NormalDist

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "pvalue_evidence", ROOT / "assets/viz/generate_pvalue_evidence_figures.py")
model = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(model)


def test_rejection_probability_matches_direct_density_integration():
    cutoff = NormalDist().inv_cdf(.975)
    normal = NormalDist(2, 1)
    inside = sum(normal.pdf(-cutoff + (i+.5)*2*cutoff/16000)
                 for i in range(16000)) * 2*cutoff/16000
    assert model.rejection_probability() == pytest.approx(1-inside, abs=1e-8)
    assert model.two_sided_p(2) == pytest.approx(2*NormalDist().cdf(-2))


def test_null_calibration_and_sign_symmetry():
    for alpha in (.001, .01, .05, .2):
        assert model.rejection_probability(0, alpha) == pytest.approx(alpha)
        assert model.rejection_probability(2, alpha) == pytest.approx(model.rejection_probability(-2, alpha))
    assert model.two_sided_p(2) == model.two_sided_p(-2)
    assert model.two_sided_p(0) == 1


def test_posterior_agrees_with_density_weighting_and_preserves_direction():
    for prior in (.01, .1, .5):
        for z in (-2, 0, 2):
            signal = prior * NormalDist(2, 1).pdf(z)
            null = (1-prior) * NormalDist(0, 1).pdf(z)
            assert model.posterior_signal(z, prior) == pytest.approx(signal/(signal+null))
    assert model.posterior_signal(-2) < .001 < model.posterior_signal(2)
    assert model.posterior_signal(2, 0) == 0
    assert model.posterior_signal(2, 1) == 1


def test_selected_counts_conserve_population_and_match_simulation():
    row = model.selected_studies()
    counts = [row[k] for k in ("signal_rejections", "null_rejections",
                              "signal_nonrejections", "null_nonrejections")]
    assert sum(counts) == pytest.approx(10000)
    assert row["signal_rejections"] + row["signal_nonrejections"] == pytest.approx(1000)
    rng = Random(20241107)
    cutoff = NormalDist().inv_cdf(.975)
    selected, selected_signal = 0, 0
    for _ in range(250000):
        signal = rng.random() < .1
        z = rng.gauss(2 if signal else 0, 1)
        if abs(z) >= cutoff:
            selected += 1
            selected_signal += signal
    q = row["signal_given_rejection"]
    assert abs(selected_signal/selected-q) < 5*sqrt(q*(1-q)/selected)


def test_independent_replication_matches_joint_probability():
    row = model.selected_studies()
    power = row["power"]
    both = .1*power**2 + .9*.05**2
    first = .1*power + .9*.05
    assert row["repeat_rejection_given_first_rejection"] == pytest.approx(both/first)


def test_study_comparison_and_score_units():
    a, b = model.study_summary(.2, .1), model.study_summary(.19, .11)
    assert a["p"] < .05 < b["p"]
    difference = model.study_summary(a["estimate"]-b["estimate"], sqrt(a["se"]**2+b["se"]**2))
    assert difference["ci95"][0] < 0 < difference["ci95"][1]
    assert difference["p"] > .9
    scaled = model.study_summary(2, 1)
    assert scaled["p"] == a["p"]
    assert scaled["ci95"] == pytest.approx([10*x for x in a["ci95"]])


def test_invalid_inputs():
    for alpha in (0, 1, -1, float("nan")):
        with pytest.raises(ValueError):
            model.rejection_probability(alpha=alpha)
    for prior in (-.1, 1.1, float("nan")):
        with pytest.raises(ValueError):
            model.posterior_signal(2, prior)
    with pytest.raises(ValueError):
        model.study_summary(.2, 0)
