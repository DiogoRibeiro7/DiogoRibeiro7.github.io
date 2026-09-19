"""Check the arithmetic and the quoted numbers of "A Stool Sample Is Not a Diagnosis"."""

import importlib.util
from math import sqrt
from pathlib import Path
from random import Random

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("microbiome", ROOT / "assets/viz/generate_microbiome_testing_figures.py")
model = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(model)
POST = (ROOT / "_posts/healthcare/2026-03-05-consumer_microbiome_testing_limits.md").read_text(encoding="utf-8")
NUMBERS = model.summary()


def test_closure_against_a_community_counted_directly():
    counts = [300.0, 250.0, 200.0, 150.0, 100.0]                     # the first taxon holds 30%
    before = [c / sum(counts) for c in counts]
    counts[0] *= 10
    after = [c / sum(counts) for c in counts]
    for b, a in zip(before[1:], after[1:]):                          # nothing happened to these four
        assert a / b - 1 == pytest.approx(model.apparent_change(0.30, 10))
    assert model.apparent_change(0.30, 1) == 0


def test_flag_counts():
    assert NUMBERS["expected flags among 100 taxa with 95% ranges"] == 5
    assert NUMBERS["chance of at least one flag among 100 taxa, %"] == pytest.approx(100 * (1 - 0.95 ** 100), abs=0.05)


def test_flagged_again_at_the_ends_and_against_simulation():
    assert model.flagged_again(0.0) == pytest.approx(0.05, abs=1e-4)
    assert model.flagged_again(1.0) == 1.0
    rng = Random(20260919)
    icc, cut = 0.5, model.STANDARD.inv_cdf(0.95)
    flagged = again = 0
    for _ in range(400_000):
        first = rng.gauss(0, 1)
        if first > cut:
            flagged += 1
            again += icc * first + sqrt(1 - icc ** 2) * rng.gauss(0, 1) > cut
    assert model.flagged_again(icc) == pytest.approx(again / flagged, abs=0.012)


def test_numbers_quoted_in_the_post_match_the_model_and_the_sources():
    again = NUMBERS["flagged again, %"]
    study = model.SERVETAS
    quoted = [
        f"ranged from {study['fewest genera']} to {study['most genera']}", f"{study['unique taxa']:,} distinct taxa", "only three genera",
        f"{model.VANDEPUTTE['absolute abundance, %']}% of genera varied more within", f"{model.VANDEPUTTE['relative abundance, %']}% of genera varied more within",
        f"flagged again in a second sample {again['0.5']}% of the time, and at 0.3 only {again['0.3']}% of the time",
        f"appears to fall by {-NUMBERS['other shares when a 30% taxon doubles, %']}%",
        f"appear to fall by {-NUMBERS['other shares when a 30% taxon rises tenfold, %']}%",
        f"a {NUMBERS['chance of at least one flag among 100 taxa, %']}% chance of flagging at least one",
        f"explained {model.BERRY['gut microbiome']}% of the variance and the macronutrient content of the meal {model.BERRY['meal macronutrients']}%",
    ]
    for text in quoted:
        assert text in POST, text
    assert 100 - again["0.9"] > 33          # "more than a third of flags disappear on retesting"


def test_every_cited_author_has_a_reference():
    body, references = POST.split("## References")
    for author in ("Bermingham", "Berry", "Falony", "Magne", "Nishijima", "Olsson", "Porcari", "Servetas", "Sze", "Vandeputte",
                   "Wei", "Zhernakova"):
        assert author in body, author
        assert author in references, author
