"""Check causal bookkeeping and selection against independently counted histories."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "screening", ROOT / "assets/viz/generate_screening_survival_figures.py")
model = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(model)


def test_diagnosis_changes_preserve_every_persons_death_history():
    reference = model.cohort()
    for earlier, additional in ((True, False), (False, True), (True, True)):
        changed = model.cohort(earlier, additional)
        assert [(p["person"], p["death"], p["cause"]) for p in changed] == [
            (p["person"], p["death"], p["cause"]) for p in reference]
        assert model.summarise(changed)["cancer_deaths"] == 75
        assert model.summarise(changed)["all_deaths"] == 125
    earlier = model.cohort(earlier=True)
    for p, q in zip(reference[:120], earlier[:120]):
        assert (q["death"] - q["diagnosis"]) - (p["death"] - p["diagnosis"]) == 4


def test_survival_denominators_and_horizon_are_distinct():
    results = model.examples()["cohort_scenarios"]
    assert [r["diagnoses"] for r in results.values()] == [120, 120, 300, 300, 300]
    assert [r["five_year_survivors"] for r in results.values()] == [45, 120, 225, 300, 300]
    assert [r["five_year_survival"] for r in results.values()] == [.375, 1, .75, 1, 1]
    # An independent tiny history fixes the definition at equality and confirms
    # that undiagnosed people still contribute to the mortality denominator.
    rows = [{"diagnosis": 2, "death": 7, "cause": "cancer"},
            {"diagnosis": 2, "death": 8, "cause": "other"},
            {"diagnosis": None, "death": 4, "cause": "other"}]
    summary = model.summarise(rows, horizon=7)
    assert summary["five_year_survival"] == .5
    assert summary["cancer_death_risk"] == pytest.approx(1/3)
    assert summary["all_death_risk"] == pytest.approx(2/3)


def test_benefit_changes_exactly_18_histories_but_not_five_year_survival():
    ineffective = model.cohort(True, True)
    beneficial = model.cohort(True, True, 18)
    changed = [(a, b) for a, b in zip(ineffective, beneficial) if a != b]
    assert len(changed) == 18
    assert all(b["death"] - a["death"] == 11 for a, b in changed)
    before, after = model.summarise(ineffective), model.summarise(beneficial)
    assert before["five_year_survival"] == after["five_year_survival"] == 1
    assert after["cancer_deaths"] == 57
    assert before["all_deaths"] - after["all_deaths"] == 18
    # Postponing death does not mean the people never die.
    assert model.summarise(ineffective, horizon=20)["all_deaths"] == 1000
    assert model.summarise(beneficial, horizon=20)["all_deaths"] == 1000


def test_snapshot_weights_against_counted_intervals():
    # Forty entrants per year, evenly spaced; enumerate who is in a detectable
    # interval at time zero instead of using incidence multiplied by duration.
    entries = [-10 + (i + .5)/40 for i in range(800)]
    counts = [sum(t <= 0 < t+d for t in entries) for d in (1, 4)]
    assert counts == [40, 160]
    result = model.sampling()
    assert result["snapshot_stock"] == counts
    assert result["snapshot_weights"] == pytest.approx([n/sum(counts) for n in counts])
    durations_of_detected = [d for d, n in zip((1, 4), counts) for _ in range(n)]
    assert result["snapshot_mean_duration"] == pytest.approx(sum(durations_of_detected)/len(durations_of_detected))
    assert result["snapshot_mean_duration"] == pytest.approx(2.5 + 2.25/2.5)


def test_repeated_screen_detection_against_uniform_entry_phases():
    waits = [(i + .5) * 2/10000 for i in range(10000)]
    for duration in (.3, 1, 2, 4):
        counted = sum(wait < duration for wait in waits)/len(waits)
        assert model.sampling((1,), (duration,))["detection_probabilities"][0] == pytest.approx(counted)
    assert model.sampling()["repeated_weights"][1] == pytest.approx(2/3)


def test_equal_windows_and_time_unit_changes():
    equal = model.sampling((10, 30), (3, 3))
    assert equal["incident_weights"] == equal["snapshot_weights"] == equal["repeated_weights"]
    years = model.sampling()
    months = model.sampling((40/12, 40/12), (12, 48), interval=24)
    for key in ("snapshot_stock", "snapshot_weights", "detection_probabilities", "repeated_weights"):
        assert months[key] == pytest.approx(years[key])
    assert months["snapshot_mean_duration"] == pytest.approx(12*years["snapshot_mean_duration"])


def test_invalid_model_inputs():
    for count in (-1, 76, True, 1.5):
        with pytest.raises(ValueError):
            model.cohort(earlier=True, postponed_deaths=count)
    with pytest.raises(ValueError):
        model.cohort(postponed_deaths=1)
    for args in ({"interval": 0}, {"durations": (1,)}, {"rates": (0, 0)},
                 {"durations": (-1, 4)}, {"rates": (float("nan"), 1)}):
        with pytest.raises(ValueError):
            model.sampling(**args)
