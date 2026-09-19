"""Verify both the printed Python example and the figure's CUSUM implementation."""

import ast
import importlib.util
import re
from math import isclose
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
POST = ROOT / "_posts/data_science/2024-02-14-advanced_sequential_changepoint.md"
SCRIPT = ROOT / "assets/viz/generate_sequential_changepoint_figure.py"
SPEC = importlib.util.spec_from_file_location("sequential_example", SCRIPT)
example = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(example)

# Parse the complete block, so the former truncated np.concatenate line fails
# immediately. Exercise the printed function without requiring plot libraries.
BLOCK = re.search(r"```python\n(.*?)\n```", POST.read_text(encoding="utf-8"), re.S).group(1)
TREE = ast.parse(BLOCK)
FUNCTION_NODES = [node for node in TREE.body if isinstance(node, ast.FunctionDef)
                  or isinstance(node, ast.ImportFrom) and node.module == "math"]
NAMESPACE = {}
exec(compile(ast.Module(body=FUNCTION_NODES, type_ignores=[]), str(POST), "exec"), NAMESPACE)


@pytest.fixture(params=[example.cusum, NAMESPACE["cusum"]], ids=["generator", "article"])
def detector(request):
    return request.param


def test_scores_equal_exhaustive_suffix_maxima(detector):
    data = [-3.0, 1.0, 4.0, -2.0, 7.0, 2.0, 0.0]
    for delta in (0.5, 2.0, 4.0):
        scores, alarm = detector(data, mu_0=1, delta=delta, h=3)
        expected = [max([0.0] + [sum(x - 1 - delta / 2 for x in data[k:n])
                                  for k in range(n)]) for n in range(1, len(data) + 1)]
        assert all(isclose(a, b, abs_tol=1e-12) for a, b in zip(scores, expected))
        assert len(scores) == len(data)
        assert alarm == next((i for i, value in enumerate(expected) if value > 3), None)


def test_first_sample_strict_threshold_and_complete_trajectory(detector):
    assert detector([], 0, 2, 5) == ([], None)
    assert detector([0, -4, 1], 0, 2, 5) == ([0, 0, 0], None)
    assert detector([6], 0, 2, 5) == ([5], None)
    assert detector([8, 8, 8], 0, 2, 5) == ([7, 14, 21], 0)


def test_scores_and_alarms_respect_measurement_units(detector):
    data = [0.0, 3.0, 2.0, -1.0, 6.0, 4.0]
    scores, alarm = detector(data, 0, 2, 5)
    shifted, shifted_alarm = detector([x + 10 for x in data], 10, 2, 5)
    scaled, scaled_alarm = detector([x * 3 for x in data], 0, 6, 15)
    assert scores == shifted
    assert all(isclose(3 * a, b) for a, b in zip(scores, scaled))
    assert alarm == shifted_alarm == scaled_alarm


def test_invalid_observations_and_parameters_are_rejected(detector):
    for bad in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError):
            detector([0, bad], 0, 2, 5)
    for delta, threshold in ((0, 5), (-2, 5), (2, 0), (2, -1)):
        with pytest.raises(ValueError):
            detector([0], 0, delta, threshold)


def test_seeded_example_separates_change_and_alarm(detector):
    result = example.example()
    scores, alarm = detector(result["data"], 0, result["delta"], result["h"])
    assert result["first_changed_index"] == 60
    assert alarm == result["alarm_index"] == 64
    assert scores[63] < 5 < scores[64]
    assert len(scores) == 100
