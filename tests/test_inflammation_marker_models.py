"""Check the arithmetic and the quoted numbers of "Inflammation Is Not a Diagnosis"."""

import importlib.util
from math import exp, log, sqrt
from pathlib import Path
from random import Random

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("inflammation", ROOT / "assets/viz/generate_inflammation_marker_figures.py")
model = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(model)
POST = (ROOT / "_posts/healthcare/2026-09-02-inflammation_is_not_a_diagnosis_social_media_myths.md").read_text(encoding="utf-8")
NUMBERS = model.summary()


def test_the_standard_formula_reproduces_the_published_critical_difference():
    assert NUMBERS["critical difference from Macy's components, %"] == model.MACY["published critical difference"]


def test_lognormal_change_value_is_asymmetric_and_consistent():
    rise, fall = model.lognormal_change_value(model.EFLM_WITHIN)
    assert rise > -fall > 0
    assert (1 + rise / 100) * (1 + fall / 100) == pytest.approx(1)       # a rise and the matching fall undo each other


def test_single_result_interval_against_simulation():
    rng = Random(20260919)
    sigma = sqrt(log(1 + (model.EFLM_WITHIN / 100) ** 2))
    draws = sorted(3.0 * exp(rng.gauss(0, sigma)) for _ in range(200_000))
    low, high = model.single_result_interval(3.0, model.EFLM_WITHIN)
    assert draws[int(0.025 * len(draws))] == pytest.approx(low, rel=0.02)
    assert draws[int(0.975 * len(draws))] == pytest.approx(high, rel=0.02)
    # the simulated within-person CV is the one put in
    mean = sum(draws) / len(draws)
    assert sqrt(sum((d - mean) ** 2 for d in draws) / len(draws)) / mean == pytest.approx(model.EFLM_WITHIN / 100, rel=0.03)


def test_benefit_and_harm_in_cantos():
    assert NUMBERS["CANTOS events prevented per 1,000 person-years"] == pytest.approx(10 * (4.50 - 3.86))
    assert NUMBERS["CANTOS extra fatal infections per 1,000 person-years"] == pytest.approx(10 * (0.31 - 0.18))
    assert 4.5 < NUMBERS["CANTOS events prevented per extra fatal infection"] < 5.5      # "about one ... for every five"


def test_numbers_quoted_in_the_post_match_the_model():
    rise, fall = NUMBERS["change needed, EFLM variation, log-normal, %"]
    low, high = NUMBERS["single results from a usual level of 3 mg/L"]
    quoted = [
        f"more than {rise}% higher or {-fall}% lower", f"will range from {low} to {high} mg/L",
        f"at {model.EFLM_WITHIN}%, from six studies", "returns 118% from those components",
        f"prevented {NUMBERS['CANTOS events prevented per 1,000 person-years']} cardiovascular events per 1,000 person-years",
        f"{NUMBERS['CANTOS extra fatal infections per 1,000 person-years']} additional deaths from infection",
    ]
    for text in quoted:
        assert text in POST, text
    for label, ratio, low_ci, high_ci, _ in model.ESTIMATES:
        assert f"{label}: {ratio:.2f}, {low_ci:.2f} to {high_ci:.2f}." in POST, label
    assert 0.98 / (high - low) < 0.25          # "less than a quarter of the range of single results"
    assert 7 < 0.98 / 0.13 < 8                 # "seven or eight kilograms of weight loss"


def test_every_cited_author_has_a_reference():
    body, references = POST.split("## References")
    for author in ("Bleakley", "Bower", "Calder", "Costenbader", "CRP CHD Genetics Collaboration", "Cushman", "DeGoma", "Dehzad",
                   "Emerging Risk Factors Collaboration", "Furman", "Gleeson", "Hotamisligil", "IL6R MR Consortium", "Macy",
                   "Medzhitov", "Nidorf", "Pedersen", "Pepys", "Ridker", "Roberts", "Sahebkar", "Schwingshackl", "Selvin",
                   "Tardif", "Visser", "Wannamethee"):
        assert author in body, author
        assert author in references, author
