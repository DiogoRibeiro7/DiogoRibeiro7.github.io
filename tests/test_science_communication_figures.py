"""Check the physical and probability models behind the archive illustrations."""

import importlib.util
from itertools import product
from math import cos, isclose, pi, radians, sin
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "assets/viz/generate_science_communication_figures.py"
SPEC = importlib.util.spec_from_file_location("science_figures", SCRIPT)
science = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(science)


def test_daily_solar_energy_matches_direct_integration():
    # Independently integrate the instantaneous projected flux through a day.
    samples = 20_000
    for latitude in (0, 45, -45, 70):
        for declination in (-23.44, 0, 23.44):
            phi, delta = radians(latitude), radians(declination)
            flux = [max(0, sin(phi) * sin(delta) + cos(phi) * cos(delta)
                        * cos(-pi + 2 * pi * (i + 0.5) / samples))
                    for i in range(samples)]
            _, _, analytic_mean = science.solar_geometry(latitude, declination)
            assert isclose(analytic_mean, sum(flux) / samples, abs_tol=1e-8)


def test_seasons_reverse_between_hemispheres():
    for latitude in (0, 20, 45, 70):
        north = science.solar_geometry(latitude, 23.44)
        south_opposite_season = science.solar_geometry(-latitude, -23.44)
        assert all(isclose(a, b, abs_tol=1e-12) for a, b in zip(north, south_opposite_season))
    altitude, hours, mean = science.solar_geometry(0, 0)
    assert altitude == 90
    assert hours == 12
    assert isclose(mean, 1 / pi)


def test_unknown_coin_prediction_matches_enumerated_sequences():
    sequences = list(product((0, 1), repeat=5))
    for heads in range(5):
        denominator = numerator = 0
        for sequence in sequences:
            probability = sum(0.5 * p**sum(sequence) * (1-p)**(5-sum(sequence))
                              for p in (0.25, 0.75))
            if all(sequence[i] == 1 for i in range(heads)):
                denominator += probability
                if sequence[heads] == 1:
                    numerator += probability
        assert isclose(science.next_head_probabilities(heads)[2], numerator / denominator)


def test_selection_changes_composition_without_population_growth():
    rows = science.selection_rows()
    for before, after in zip(rows, rows[1:]):
        assert after[1] + after[2] < before[1] + before[2]
        assert after[2] < before[2]
        assert after[3] > before[3]
        before_odds = before[3] / (1 - before[3])
        after_odds = after[3] / (1 - after[3])
        assert isclose(after_odds / before_odds, 80, rel_tol=1e-10)
