"""Check the arithmetic and the quoted numbers of "Hormones Are Not a Single Balance"."""

import importlib.util
from math import comb
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("hormones", ROOT / "assets/viz/generate_hormone_testing_figures.py")
model = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(model)
POST = (ROOT / "_posts/healthcare/2026-02-18-hormone_balance_social_media_myth.md").read_text(encoding="utf-8")
NUMBERS = model.summary()


def test_chance_of_a_flag_against_the_binomial_distribution():
    for analytes in (1, 5, 12, 20):
        at_least_one = sum(comb(analytes, k) * 0.05 ** k * 0.95 ** (analytes - k) for k in range(1, analytes + 1))
        assert model.any_flag(analytes) == pytest.approx(at_least_one)


def test_simulation_agrees_with_the_formula_when_analytes_are_independent():
    assert model.any_flag_correlated(12, 0.0, repeats=30_000) == pytest.approx(model.any_flag(12), abs=0.012)


def test_correlation_lowers_the_chance_but_not_to_one_in_twenty():
    correlated = model.any_flag_correlated(12, 0.5, repeats=30_000)
    assert 0.05 < correlated < model.any_flag(12)
    assert correlated > 0.25


def test_reference_change_value_reproduces_a_published_one():
    computed = NUMBERS["salivary cortisol reference change value from Casals's components, %"]
    assert computed == pytest.approx(model.CASALS["published RCV"], abs=0.5)


def test_false_positive_share_follows_from_the_two_rates():
    rates = model.NAUGLER
    assert NUMBERS["share of abnormal results that are false positives, %"] == round(100 * rates["expected in the healthy"] / rates["observed abnormal"]) == 58


def test_numbers_quoted_in_the_post_match_the_model():
    flags = NUMBERS["chance of at least one flag, %"]
    change = NUMBERS["minimum reference change value, %"]
    estradiol, progesterone = NUMBERS["estradiol capsule content against label, %"], NUMBERS["progesterone capsule content against label, %"]
    casals = NUMBERS["salivary cortisol reference change value from Casals's components, %"]
    quoted = [
        f"{flags[5]}% for five analytes, {flags[12]:.1f}% for twelve and {flags[20]}% for twenty",
        f"still flags {NUMBERS['chance of at least one flag, 12 analytes, correlation 0.5, %']}% of healthy people",
        "about 58% of abnormal results",
        f"must differ by {change['TSH']}%, two serum cortisol results by {change['cortisol']}%, two testosterone results by "
        f"{change['testosterone']}% and two prolactin results by {change['prolactin']}%",
        f"more than the {change['estradiol']}% by which estradiol differs",
        f"returns {casals}% from those components",
        f"{-estradiol[0]}% below the label to {estradiol[1]}% above", f"{-progesterone[0]}% below to {progesterone[1]}% above",
        f"a {NUMBERS['progesterone, luteal over follicular']}-fold rise",
    ]
    for text in quoted:
        assert text in POST, text
    alt = "It is " + ", ".join(f"{v}% for {k}" for k, v in sorted(change.items(), key=lambda item: model.WITHIN_SUBJECT_CV[item[0]]))
    assert alt in POST


def test_every_cited_author_has_a_reference():
    body, references = POST.split("## References")
    for author in ("Akturk", "Anckaert", "Andersen", "Brambilla", "Brito", "Cadegiani", "Casals", "Danese", "Kang", "McNulty",
                   "Musazadeh", "Nagarajan", "Naugler", "Nickel", "Santoro", "Srinivasa Gopalan", "Stanczyk"):
        assert author in body, author
        assert author in references, author
