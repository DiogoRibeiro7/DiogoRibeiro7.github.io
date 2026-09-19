"""Check the model against direct projections, optimisation, and tensor products."""

import cmath
import importlib.util
from itertools import product
from math import acos, cos, pi, sin, sqrt
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "quantum_observer", ROOT / "assets/viz/generate_quantum_observer_figures.py")
model = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(model)


def test_joint_probabilities_match_interference_formula_and_conserve_mass():
    for gamma, phi in product((0, .2, .6, .95, 1), (0, .41, pi/2, pi, 2*pi)):
        state = model.state(phi, gamma)
        assert sum(abs(a)**2 for row in state for a in row) == pytest.approx(1)
        joint = model.joint_probabilities(phi, gamma)
        assert sum(sum(row) for row in joint) == pytest.approx(1)
        assert sum(joint[0]) == pytest.approx((1+gamma*cos(phi))/2)
        assert all(p >= 0 for row in joint for p in row)


def test_reduced_matrix_is_a_density_matrix_with_the_reported_purity():
    for gamma, phi in product((0, .3, .6, 1), (.2, pi/2)):
        rho = model.reduced_path_state(phi, gamma)
        assert rho[0][1] == pytest.approx(gamma*cmath.exp(-1j*phi)/2)
        assert rho[1][0] == pytest.approx(rho[0][1].conjugate())
        assert rho[0][0]+rho[1][1] == pytest.approx(1)
        determinant = (rho[0][0]*rho[1][1]-rho[0][1]*rho[1][0]).real
        assert determinant >= -1e-15
        assert determinant == pytest.approx((1-gamma*gamma)/4)
        assert model.metrics(gamma)["path_purity"] == pytest.approx((1+gamma*gamma)/2)


def test_optimal_path_guess_against_independent_measurement_search():
    # Search real orthonormal measurement vectors directly; the two marker
    # states are real, so an optimal measurement exists in this plane.
    for gamma in (0, .2, .6, .95, 1):
        marker1 = (gamma, sqrt(1-gamma*gamma))
        successes = []
        for i in range(8192):
            angle = pi*i/8192
            e0, e1 = (cos(angle), sin(angle)), (-sin(angle), cos(angle))
            successes.append(.5*(e0[0]**2 + sum(a*b for a, b in zip(e1, marker1))**2))
        predicted = model.metrics(gamma)["optimal_path_guess_probability"]
        assert max(successes) == pytest.approx(predicted, abs=4e-8)


def test_marker_basis_does_not_change_unsorted_path_statistics():
    for gamma, phi, beta, eta in product((0, .6, 1), (.3, 1.7), (0, .23, pi/4), (0, .71)):
        table = model.joint_probabilities(phi, gamma, beta, eta)
        expected = (1+gamma*cos(phi))/2
        assert sum(table[0]) == pytest.approx(expected)
        assert sum(table[1]) == pytest.approx(1-expected)
    table = model.joint_probabilities(pi/3, 0, pi/4)
    assert table[0][0]/sum(row[0] for row in table) == pytest.approx(.75)
    assert table[0][1]/sum(row[1] for row in table) == pytest.approx(.25)
    assert sum(table[0]) == pytest.approx(.5)


def test_local_projectors_commute_on_the_full_joint_state():
    path = (1/sqrt(2), -1/sqrt(2))
    marker = (cos(.37), cmath.exp(.8j)*sin(.37))
    outer = lambda v: [[a*complex(b).conjugate() for b in v] for a in v]
    p, q = outer(path), outer(marker)
    identity = ((1, 0), (0, 1))
    tensor = lambda a, b: [[a[i//2][j//2]*b[i%2][j%2] for j in range(4)] for i in range(4)]
    apply = lambda a, v: [sum(a[i][j]*v[j] for j in range(4)) for i in range(4)]
    p_full, q_full = tensor(p, identity), tensor(identity, q)
    state = [a for row in model.state(1.3, .6) for a in row]
    first = apply(p_full, apply(q_full, state))
    second = apply(q_full, apply(p_full, state))
    assert first == pytest.approx(second)


def test_phase_averaging_and_entanglement_can_have_identical_local_states():
    for gamma in (0, .6, 1):
        phi, alpha = .71, acos(gamma)
        vectors = [(1/sqrt(2), cmath.exp(1j*(phi+shift))/sqrt(2)) for shift in (-alpha, alpha)]
        mixture = [[sum(v[i]*complex(v[j]).conjugate() for v in vectors)/2 for j in range(2)] for i in range(2)]
        reduced = model.reduced_path_state(phi, gamma)
        for actual, expected in zip(mixture, reduced):
            assert actual == pytest.approx(expected)


def test_product_environment_overlap_against_explicit_tensor_states():
    overlaps = (.8, .6, .3)
    zero = [1]
    one = [1]
    for gamma in overlaps:
        zero = [a*b for a in zero for b in (1, 0)]
        one = [a*b for a in one for b in (gamma, sqrt(1-gamma*gamma))]
    direct = sum(a*b for a, b in zip(zero, one))
    assert model.environment_overlap(overlaps) == pytest.approx(direct)
    assert model.environment_overlap([]) == 1
    assert model.environment_overlap((.8, 0, .9)) == 0


def test_invalid_model_parameters():
    for gamma in (-.1, 1.1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            model.state(0, gamma)
    with pytest.raises(ValueError):
        model.state(float("nan"), .5)
    with pytest.raises(ValueError):
        model.marker_basis(float("inf"))
