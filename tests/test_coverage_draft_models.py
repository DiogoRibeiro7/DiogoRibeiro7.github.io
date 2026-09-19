"""Check the articles against enumeration, energy conservation and closed forms."""

import importlib.util
from collections import Counter
from itertools import combinations
from math import isclose
from pathlib import Path
from statistics import mean, pvariance

SCRIPT = Path(__file__).resolve().parents[1] / "assets/viz/generate_coverage_draft_figures.py"
SPEC = importlib.util.spec_from_file_location("coverage_models", SCRIPT)
models = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(models)


def test_balance_distribution_matches_all_allocations():
    # Unequal arms and a non-half prevalence exercise the support boundaries.
    for size, high, treated in ((8, 3, 5), (6, 5, 2), (6, 0, 3)):
        allocations = list(combinations(range(size), treated))
        counts = Counter(sum(i < high for i in arm) for arm in allocations)
        rows = models.balance_probabilities(size, high, treated)
        assert {r["high_treated"] for r in rows} == set(counts)
        assert isclose(sum(r["probability"] for r in rows), 1)
        for row in rows:
            assert isclose(row["probability"], counts[row["high_treated"]] / len(allocations))
        assert isclose(sum(r["difference"] * r["probability"] for r in rows), 0, abs_tol=1e-12)


def test_adjustment_variance_matches_enumerated_estimates():
    x = [0, 0, 0, 1, 1, 1, 1, 1]
    y = [3, 5, 6, 4, 9, 8, 6, 11]
    for treated in (3, 4):
        for coefficient in (-1, 0, 2, 5):
            estimates = []
            for arm in combinations(range(8), treated):
                estimates.append(mean(y[i] + 1.5 - coefficient * x[i] for i in arm)
                                 - mean(y[i] - coefficient * x[i] for i in range(8) if i not in arm))
            assert isclose(mean(estimates), 1.5)
            assert isclose(pvariance(estimates), models.adjusted_sd(y, x, treated, coefficient) ** 2)


def test_storage_conserves_energy_and_respects_bounds():
    for hours in (0.25, 1.0, 3.0):
        for capacity, power, initial in ((0, 2, 0), (6, 0, 3), (6, 2, 0), (10, 4, 7)):
            solar, demand = [0, 6, 9, 1, 0, 0], [2, 1, 3, 5, 2, 2]
            rows = models.storage_ledger(solar, demand, capacity, power, initial, hours=hours)
            for r in rows:
                assert 0 <= r["after"] <= capacity
                assert 0 <= r["charge"] <= power and 0 <= r["discharge"] <= power
                assert r["charge"] * r["discharge"] == 0
                assert r["imports"] >= 0 and r["curtailment"] >= 0 and r["loss"] >= 0
                assert isclose(r["generation"] + r["imports"] + r["discharge"],
                               r["demand"] + r["charge"] + r["curtailment"], abs_tol=1e-12)
            supplied = initial + hours * sum(r["generation"] + r["imports"] for r in rows)
            used = rows[-1]["after"] + sum(r["loss"] + hours * (r["demand"] + r["curtailment"]) for r in rows)
            assert isclose(supplied, used, abs_tol=1e-12)


def test_evening_supply_requires_both_energy_and_power():
    # Three hours of 2 kW load: the power bottleneck can leave stored energy unused.
    slow = models.storage_ledger([0] * 3, [2] * 3, 6, 1, initial=6)
    fast = models.storage_ledger([0] * 3, [2] * 3, 6, 2, initial=6)
    assert isclose(sum(r["imports"] for r in slow), 3)
    assert isclose(slow[-1]["after"], 8 / 3)
    assert isclose(sum(r["imports"] for r in fast), 0.6)
    assert isclose(fast[-1]["after"], 0, abs_tol=1e-12)


def test_adaptation_matches_geometric_step_response():
    for alpha in (0, 0.02, 0.1, 0.25, 1):
        rows = models.adaptive_baseline([7.0] * 40, alpha, initial=3)
        for t, row in enumerate(rows):
            assert isclose(row["score"], 4 * (1 - alpha) ** t, abs_tol=1e-12)
        if alpha > 0:
            assert isclose(sum(r["score"] for r in rows),
                           4 * (1 - (1 - alpha) ** 40) / alpha, abs_tol=1e-10)


def test_missing_observations_do_not_become_zero_measurements():
    sparse = models.adaptive_baseline([4, None, None, 4, None, 4], 0.1)
    dense = models.adaptive_baseline([4, 4, 4], 0.1)
    assert [r for r in sparse if r["observation"] is not None] == dense
    for row in sparse:
        if row["observation"] is None:
            assert row["score"] is None
            assert row["before"] == row["after"]
