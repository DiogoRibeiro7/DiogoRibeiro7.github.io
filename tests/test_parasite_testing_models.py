"""Check the arithmetic and the quoted numbers of "Parasites Are Diagnosed by Species, Not by Symptom Lists"."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("parasites", ROOT / "assets/viz/generate_parasite_testing_figures.py")
model = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(model)
POST = (ROOT / "_posts/healthcare/2026-01-24-parasite_cleanse_social_media_myth.md").read_text(encoding="utf-8")
NUMBERS = model.summary()


def test_bayes_rule_reproduces_the_published_negative_predictive_values():
    computed = NUMBERS["negative predictive value at the published prevalences, %"]
    for prevalence, published in model.BRANDA["published NPV, %"].items():
        assert abs(computed[str(prevalence)] - published) < 0.6, prevalence     # the paper says "approximately"


def test_repeat_sampling_is_close_to_independent():
    assert abs(NUMBERS["stool examination, two specimens, if independent %"] - NUMBERS["stool examination, two specimens, observed %"]) < 3
    assert abs(NUMBERS["tape test, three mornings, if independent %"] - 100 * model.TAPE["three mornings"]) < 3


def test_negative_results_compound():
    for prior in model.PRIORS:
        one, two, three = (model.left_after_negatives(prior, n, model.BRANDA["sensitivity"]) for n in (1, 2, 3))
        assert prior > one > two > three
        # in odds, each negative result divides by 1 / (1 - sensitivity)
        odds = lambda p: p / (1 - p)
        assert odds(one) / odds(two) == pytest.approx(1 / (1 - model.BRANDA["sensitivity"]))
    assert 1 / (1 - model.BRANDA["sensitivity"]) == pytest.approx(3.6, abs=0.05)


def test_left_after_negatives_against_direct_counting():
    # 100,000 people, 5% infected, three independent examinations at 72%, no false positives
    infected, healthy = 5_000, 95_000
    still_negative = infected * (1 - 0.72) ** 3
    assert model.left_after_negatives(0.05, 3, 0.72) == pytest.approx(still_negative / (still_negative + healthy))


def test_a_common_checklist_barely_moves_a_prior_and_a_laboratory_test_does():
    assert model.after_positive(0.01, model.likelihood_ratio(0.95, 0.5)) < 0.02
    assert NUMBERS["antigen test likelihood ratio at that limit"] > 8 * NUMBERS["checklist likelihood ratio"]["0.5"]
    # zero false positives in 50: the exact one-sided limit, and the rule of three as a check on it
    assert model.specificity_lower_bound(50) == pytest.approx(0.05 ** (1 / 50))
    assert model.specificity_lower_bound(50) == pytest.approx(1 - 3 / 50, abs=0.003)


def test_numbers_quoted_in_the_post_match_the_model():
    left = NUMBERS["left after negatives, %"]
    halved = NUMBERS["left after three negatives at half the sensitivity, %"]
    checklist = NUMBERS["after a positive checklist, prior 1%, %"]
    quoted = [
        "283 of them, 75.9%", "343, or 92%", "predicts 94.2%", "87.5% over three mornings",
        "98.5%, 97.0%, 95.3% and 93.5%",
        f"is at {left['0.05'][0]:.1f}% after one negative result and {left['0.05'][2]:.2f}% after three",
        f"is at {left['0.01'][2]:.3f}% after three",
        f"take 5% to {halved['0.05']}% and 1% to {halved['0.01']}%",
        f"a chance of {checklist['0.5']}% after it", f"the 1% becomes {checklist['0.8']}%", f"take the 1% to {checklist['0.2']}%",
        f"at least {NUMBERS['antigen test specificity, lower 95% limit, %']}% with 95% confidence",
        f"is {NUMBERS['antigen test likelihood ratio at that limit']}, and it takes the same 1% to {NUMBERS['after a positive antigen test, prior 1%, %']:.0f}%",
    ]
    for text in quoted:
        assert text in POST, text


def test_every_cited_author_has_a_reference():
    body, references = POST.split("## References")
    for author in ("Branda", "Butler", "Cartwright", "Garcia", "Hoang", "Hosiian", "Janes", "Lachenmeier", "Lashaki",
                   "Moser", "Munyangi", "Temple", "Volinsky", "Wendt"):
        assert author in body, author
        assert author in references, author
