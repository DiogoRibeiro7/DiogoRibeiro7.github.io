"""Check the testimonial-wall calculation and the numbers quoted in "When Results Become Rhetoric"."""

import importlib.util
from math import pi, sqrt
from pathlib import Path
from statistics import NormalDist

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("rhetoric", ROOT / "assets/viz/generate_results_rhetoric_figures.py")
model = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(model)
POST = (ROOT / "_posts/healthcare/2026-05-11-results_are_not_evidence_influencer_science.md").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def numbers():
    return model.summary()


def test_exact_wall_mean_against_known_order_statistics():
    # the expected larger of two standard normal values is 1 / sqrt(pi), and of three 3 / (2 sqrt(pi))
    assert model.wall_mean(2, wall=1) == pytest.approx(1 / sqrt(pi), abs=1e-4)
    assert model.wall_mean(3, wall=1) == pytest.approx(3 / (2 * sqrt(pi)), abs=1e-4)
    # a wall that holds everybody shows the mean effect and nothing else
    assert model.wall_mean(20, wall=20, effect=0.5) == 0.5
    assert model.wall_mean(50, wall=50) == 0.0


def test_exact_wall_mean_against_simulation():
    # 400 walls of the best 20 of 1,000: the standard error of their mean is about 0.006
    assert model.wall_mean(1_000) == pytest.approx(model.wall_mean_by_simulation(1_000), abs=0.03)
    assert model.wall_mean(200, wall=5) == pytest.approx(model.wall_mean_by_simulation(200, wall=5), abs=0.05)


def test_truncated_normal_approximation_quoted_in_the_post_is_close_when_the_wall_is_small():
    for clients in (1_000, 100_000):
        p = model.WALL / clients
        approximation = NormalDist().pdf(NormalDist().inv_cdf(1 - p)) / p
        assert approximation == pytest.approx(model.wall_mean(clients), rel=0.01)


def test_the_effect_shifts_the_wall_and_selection_grows_with_the_business(numbers):
    assert model.wall_mean(1_000, effect=1.0) == pytest.approx(1.0 + model.wall_mean(1_000), abs=1e-9)
    walls = list(numbers["wall of 20, no effect, by number of clients"].values())
    assert walls == sorted(walls) and walls[0] > 1.5


def test_a_programme_with_no_effect_matches_an_effective_one_at_the_quoted_size(numbers):
    matched = numbers["clients at which a programme with no effect matches it"]
    target = numbers["wall of 20, effect of 1 SD, 1,000 clients"]
    assert model.wall_mean(matched) == pytest.approx(target, abs=0.02)
    assert f"{matched:,} clients" in POST


def test_arithmetic_on_the_published_figures(numbers):
    finley = numbers["finley"]
    assert finley["left before a year, %"] == 93.4
    assert finley["clients still attending at a year"] == 3971
    assert finley["clients who left within four weeks"] == 16244
    assert 15 <= finley["enrolled per client still attending at a year"] < 15.5
    assert numbers["share of the change in treated patients that the treatment explains"] == 0.56
    assert numbers["before-and-after change over the treatment's own effect"] == 1.8


def test_numbers_quoted_in_the_post_match_the_model(numbers):
    walls = numbers["wall of 20, no effect, by number of clients"]
    body_weight = numbers["wall of 20, no effect, % of body weight at the illustrative SD"]
    selection = numbers["share of the wall that is selection, effect of 1 SD, 1,000 clients"]
    quoted = [
        f"{walls[200]} with 200 clients", f"{walls[1_000]} with 1,000", f"{walls[10_000]} with 10,000", f"{walls[100_000]} with 100,000",
        f"{numbers['wall of 20, effect of 1 SD, 1,000 clients']} standard deviations",
        f"{round(100 * selection)}% is selection and {round(100 * (1 - selection))}% is the programme",
        f"{body_weight[1_000]}% of body weight with 1,000 clients and {body_weight[100_000]}% with 100,000",
        f"{model.SCALE}% of body weight",
        "60,164", "6.6%", "93.4%", "15.6%", "1.1%",
        "accounts for 56% of the change", "by a factor of 1.8",
        "improved by 0.24, those given placebo by 0.44 and those given the active treatment by 1.01",
    ]
    for text in quoted:
        assert text in POST, text


def test_every_cited_author_has_a_reference():
    references = POST.split("## References")[1]
    for author in ("Barnett", "Bhasin", "Denniss", "Finley", "Guyatt", "Helou", "Hernán", "Hubal", "Krogsbøll", "Mathur", "Powell"):
        assert author in POST.split("## References")[0], author
        assert author in references, author
