"""Check the arithmetic and the quoted numbers of "Leaky Gut: Real Physiology, Weak Diagnosis"."""

import importlib.util
from math import sqrt
from pathlib import Path
from random import Random

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("leaky", ROOT / "assets/viz/generate_leaky_gut_figures.py")
model = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(model)
POST = (ROOT / "_posts/healthcare/2026-06-03-leaky_gut_social_media_myths.md").read_text(encoding="utf-8")
NUMBERS = model.summary()


def test_share_truly_high_at_the_ends_of_the_scale():
    assert model.share_truly_high(0.0) == pytest.approx(model.TOP, abs=1e-4)      # an uninformative test picks at random
    assert model.share_truly_high(1.0) == 1.0
    values = [model.share_truly_high(r) for r in (0.0, 0.2, 0.4, 0.6, 0.8, 0.95)]
    assert values == sorted(values)


def test_share_truly_high_against_simulation():
    rng = Random(20260919)
    correlation, cut = 0.5, model.STANDARD.inv_cdf(1 - model.TOP)
    positives = hits = 0
    for _ in range(200_000):
        x = rng.gauss(0, 1)
        y = correlation * x + sqrt(1 - correlation ** 2) * rng.gauss(0, 1)
        if x > cut:
            positives += 1
            hits += y > cut
    assert model.share_truly_high(correlation) == pytest.approx(hits / positives, abs=0.01)


def test_fisher_interval_contains_zero_and_the_estimate():
    r, low, high = model.correlation_interval(0.004, 39)
    assert low < 0 < r < high
    assert r == pytest.approx(sqrt(0.004))
    assert (low, high) == (pytest.approx(-0.26, abs=0.005), pytest.approx(0.37, abs=0.005))


def test_risks_by_test_recover_the_overall_risk_and_the_ratio():
    overall = 50 / 1420
    for share in (0.1, 0.2, 0.3):
        normal, abnormal = model.risks_by_test(share, overall, 3.03)
        assert abnormal / normal == pytest.approx(3.03)
        assert share * abnormal + (1 - share) * normal == pytest.approx(overall)
        assert abnormal < 0.10            # "more than nine abnormal results in ten were followed by no disease"


def test_numbers_quoted_in_the_post_match_the_model():
    share = NUMBERS["truly high among test positives, %"]
    r, low, high = NUMBERS["correlation and 95% interval"]
    risks = NUMBERS["risk with a normal and an abnormal test, %"]["0.2"]
    responders = NUMBERS["glutamine trial responders, %"]
    overall = NUMBERS["relatives who developed Crohn's disease, %"]
    quoted = [
        f"explains {NUMBERS['variance explained, %']}% of the variation",
        f"At the observed correlation of {r} it is {share['at the observed correlation']}%",
        f"from {low} to {high}",
        f"would be right {share['at the top of its interval']}% of the time",
        f"50 in 1,420, or {overall}%",
        f"the risk was {risks[0]}% after a normal test and {risks[1]}% after an abnormal one",
        f"{responders[0]}%, against 3 of 52 on placebo, {responders[1]}%",
    ]
    for text in quoted:
        assert text in POST, text


def test_every_cited_author_has_a_reference():
    body, references = POST.split("## References")
    for author in ("Abbasi", "Ajamian", "Camilleri", "Chantler", "Cummins", "Hoilat", "Nascimento", "Power", "Rath",
                   "Scheffler", "Turpin", "Zheng", "Zhou"):
        assert author in body, author
        assert author in references, author
