import json

import pytest

from beamfem.optimize.backends import ExactBackend, GreedyBackend
from beamfem.optimize.backends.base import design_values
from examples.nonlinear_truss_optimization import build_problem, run_example


def test_exact_and_greedy_select_feasible_elastoplastic_topology_and_sections():
    problem = build_problem()
    exact = ExactBackend().solve(problem)
    greedy = GreedyBackend(penalty=1.0e6, pairwise=True).solve(problem)

    assert exact.feasible and greedy.feasible
    assert design_values(exact.design) == (1, 1)
    assert design_values(greedy.design) == (1, 1)
    assert greedy.objective == pytest.approx(exact.objective)
    evaluation = exact.evaluation
    assert evaluation.mass == pytest.approx(7850.0 * 2.0 * 1.2e-3)
    assert evaluation.objective == evaluation.mass
    assert evaluation.first_yield_load_factor is not None
    assert evaluation.limit_load_factor == pytest.approx(1.0)
    assert evaluation.residual_displacement_norm > 0.0
    assert 0.0 < evaluation.maximum_equivalent_plastic_strain < 3.0e-3
    assert evaluation.dissipated_energy > 0.0


def test_optional_and_understrength_candidates_are_rejected_by_nonlinear_fem():
    problem = build_problem()
    all_off = problem.evaluate((0, 0))
    understrength = problem.evaluate((1, 0))
    assert not all_off.feasible
    assert all_off.nonlinear_result.limit_state.collapse_detected
    assert all_off.nonlinear_result.limit_state.progressive_collapse_sequence[0].event == (
        "candidate_invalid"
    )
    assert not understrength.feasible
    assert understrength.nonlinear_result.converged
    assert understrength.maximum_equivalent_plastic_strain > 3.0e-3
    assert any(
        record.constraint_id == "maximum_equivalent_plastic_strain" and not record.satisfied
        for record in understrength.constraints
    )


def test_example_payload_is_finite_and_exposes_nonlinear_metrics():
    payload = run_example()
    encoded = json.dumps(payload, allow_nan=False)
    assert encoded
    assert payload["exact"]["mass"] == payload["exact"]["objective"]
    assert payload["rejected_all_off"]["feasible"] is False

