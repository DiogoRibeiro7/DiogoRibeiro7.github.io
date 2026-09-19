"""Verify polling calculations against individual records and exhaustive sampling."""

import importlib.util
from itertools import combinations
from math import sqrt
from pathlib import Path
from statistics import mean, pvariance

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "poll_selection", ROOT / "assets/viz/generate_poll_selection_figures.py")
model = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(model)


def test_finite_identity_against_individual_records():
    y = [1]*600 + [0]*400
    r = [1]*48 + [0]*552 + [1]*8 + [0]*392
    covariance = mean([(a-mean(y))*(b-mean(r)) for a, b in zip(y, r)])
    rho = covariance/sqrt(pvariance(y)*pvariance(r))
    row = model.finite_summary(600, 400, 48, 8)
    observed = mean([a for a, b in zip(y, r) if b])
    assert row["respondent_share"] == observed
    assert row["data_defect_correlation"] == pytest.approx(rho)
    assert rho*sqrt(pvariance(y))*sqrt((1-mean(r))/mean(r)) == pytest.approx(observed-mean(y))


def test_srs_variance_against_all_possible_samples():
    population = [1]*6 + [0]*4
    estimates = [mean(sample) for sample in combinations(population, 4)]
    assert mean(estimates) == pytest.approx(.6)
    assert model.srs_standard_error(.6, 10, 4)**2 == pytest.approx(pvariance(estimates))
    assert model.srs_standard_error(.6, 10, 10) == 0


def test_expanding_reach_preserves_error_while_naive_intervals_shrink():
    rows = model.examples()["sample_size_examples"]
    widths = [r["naive_interval"]["half_width"] for r in rows]
    assert all(r["error"] == pytest.approx(9/35) for r in rows)
    assert all(a > b for a, b in zip(widths, widths[1:]))
    assert widths[0]/widths[-1] == pytest.approx(sqrt(rows[-1]["sample"]/rows[0]["sample"]))
    assert all(r["naive_interval"]["lower"] > .6 for r in rows)
    for row in rows:
        f = row["recorded_fraction"]
        assert row["data_defect_correlation"]*row["population_sd"]*sqrt((1-f)/f) == pytest.approx(row["error"])


def test_weighting_by_expanding_each_observed_record():
    for dependent in (False, True):
        result = model.weighting_example(dependent)
        pairs = [(y, g["weight"]) for g in result["groups"]
                 for y, count in ((1, g["seen_yes"]), (0, g["seen_no"])) for _ in range(count)]
        direct = sum(y*w for y, w in pairs)/sum(w for _, w in pairs)
        assert result["weighted_share"] == pytest.approx(direct)
        for group in result["groups"]:
            assert group["weight"]*group["recorded"] == pytest.approx(group["total"])
    assert model.weighting_example()["weighted_share"] == pytest.approx(.6)
    assert model.weighting_example(True)["weighted_share"] == pytest.approx(.4*(36/37)+.6*(8/11))
    assert model.weighting_example(True)["weighted_share"] > .82


def test_distinct_populations_have_identical_observed_data():
    worlds = model.examples()["compatible_worlds"]
    assert [r["population_share"] for r in worlds] == [.5, .6, .75]
    for row in worlds:
        assert row["sample"] == 1120000
        assert row["respondent_share"] == pytest.approx(6/7)
        ratio = row["yes_recording_rate"]/row["no_recording_rate"]
        assert model.population_from_ratio(row["respondent_share"], ratio) == pytest.approx(row["population_share"])
    assert model.population_from_ratio(6/7, 2) == pytest.approx(.75)
    assert model.population_from_ratio(6/7, 6) == pytest.approx(.5)


def test_bounds_against_every_completion_of_unseen_people():
    # Three yes and one no observed in a population of ten; enumerate every
    # possible number of yes answers among the remaining six people.
    rows = [model.finite_summary(3+k, 7-k, 3, 1) for k in range(7)]
    shares = [r["population_share"] for r in rows]
    assert rows[0]["no_assumption_bounds"] == pytest.approx([min(shares), max(shares)])
    assert model.finite_summary(600, 400, 60, 40)["error"] == 0
    census = model.finite_summary(600, 400, 600, 400)
    assert census["error"] == 0
    assert census["data_defect_correlation"] is None
    assert census["no_assumption_bounds"] == pytest.approx([.6, .6])


def test_invalid_counts_and_sensitivity_inputs():
    for values in ((600, 400, 601, 1), (600, 400, 0, 0), (600, 400, True, 1), (0, 4, 0, 1)):
        with pytest.raises(ValueError):
            model.finite_summary(*values)
    for ratio in (0, -1, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            model.population_from_ratio(.8, ratio)
