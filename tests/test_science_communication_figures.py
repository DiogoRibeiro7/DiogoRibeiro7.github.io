"""Check the physical and probability models behind the archive illustrations."""

import importlib.util
from itertools import product
from math import cos, isclose, log, pi, radians, sin
from pathlib import Path
from statistics import NormalDist

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


def test_normal_kl_matches_density_ratio_integration():
    # Integrate the original density expression independently of the KL formula.
    samples = 40_000
    for mean_p, sd_p, mean_q, sd_q in ((7, 5, 5, 5), (7, 7, 5, 5),
                                     (5, 5, 7, 7), (5, 5, 5, 5)):
        p, q = NormalDist(mean_p, sd_p), NormalDist(mean_q, sd_q)
        step = 18 * sd_p / samples
        result = 0.0
        for i in range(samples):
            x = mean_p - 9 * sd_p + (i + 0.5) * step
            result += p.pdf(x) * log(p.pdf(x) / q.pdf(x)) * step
        assert isclose(result, science.normal_kl(mean_p, sd_p, mean_q, sd_q), abs_tol=1e-10)
    p, q = NormalDist(7, 5), NormalDist(5, 5)
    for x in (-10, 0, 6, 15, 25):
        assert isclose(science.climate_log_ratio(x), log(p.pdf(x) / q.pdf(x)), abs_tol=1e-12)


def test_processing_loses_information_but_changing_units_does_not():
    p, q = NormalDist(7, 5), NormalDist(5, 5)
    full = science.normal_kl(7, 5, 5, 5)
    for threshold in (-10, 0, 6, 15, 25):
        binary = science.binary_kl(p.cdf(threshold), q.cdf(threshold))
        assert 0 <= binary <= full
    fahrenheit = science.normal_kl(7 * 1.8 + 32, 5 * 1.8, 5 * 1.8 + 32, 5 * 1.8)
    assert isclose(full, fahrenheit, abs_tol=1e-12)


def test_run_probability_matches_enumerated_records():
    for n in (0, 1, 4, 8):
        for run_length in (1, 3, 4):
            for p in (0.3, 0.5):
                total = 0.0
                for sequence in product((0, 1), repeat=n):
                    if '1' * run_length in ''.join(map(str, sequence)):
                        total += p**sum(sequence) * (1 - p)**(n - sum(sequence))
                assert isclose(science.run_probability(n, run_length, p), total, abs_tol=1e-12)


def test_thermal_response_matches_integrated_energy_balance():
    # Integrate tau*dT/dt + T = cos(omega*t), starting from zero, then compare
    # the final annual cycle to the analytic periodic solution. RK4 is independent
    # of the amplitude/phase derivation used by the article.
    omega, step = 2 * pi / 365, 0.5
    for tau in (10, 30, 90):
        amplitude, lag = science.thermal_response(tau)
        temperature = 0.0
        for i in range(int(10 * 365 / step)):
            t = i * step
            def derivative(time, value):
                return (cos(omega * time) - value) / tau
            k1 = derivative(t, temperature)
            k2 = derivative(t + step / 2, temperature + step * k1 / 2)
            k3 = derivative(t + step / 2, temperature + step * k2 / 2)
            k4 = derivative(t + step, temperature + step * k3)
            temperature += step * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            if t >= 9 * 365:
                expected = amplitude * cos(omega * (t + step - lag))
                assert isclose(temperature, expected, abs_tol=1e-8)
