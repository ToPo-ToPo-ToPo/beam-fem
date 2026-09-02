from dataclasses import dataclass

import numpy as np
import pytest

from beamfem.optimize.backends import (
    ExactBackend, GreedyBackend, MILPBackend, MILPFormulation, QAOABackend,
    SequentialQUBOOptimizer, SimulatedAnnealingBackend,
)
from beamfem.optimize.backends.base import design_values
from beamfem.optimize.qubo import (
    AdaptivePenalty, BinaryEncoding, LocalQUBOBuilder, LocalQUBOProblemAdapter,
    OneHotEncoding, QUBOModel, QUBOSolution, TrustRegion, select_candidates,
)
from scipy.optimize import Bounds, LinearConstraint


@dataclass(frozen=True)
class State:
    choices: tuple[int, ...]


@dataclass(frozen=True)
class Evaluation:
    objective: float
    feasible: bool
    constraints: tuple[float, ...]


class ToyProblem:
    initial_design = State((0, 0))
    domains = ((0, 1, 2), (0, 1, 2))

    def evaluate(self, design):
        x = design_values(design)
        feasible = x[0] + x[1] >= 2
        return Evaluation(float(3 * x[0] + x[1]), feasible,
                          (float(2 - x[0] - x[1]),))


def test_qubo_energy_normalization_and_exact_solution():
    model = QUBOModel(np.array([-2.0, -1.0]), np.array([[0.0, 4.0], [0.0, 0.0]]), 3.0)
    assert model.energy((1, 0)) == pytest.approx(1.0)
    assert model.energy((1, 1)) == pytest.approx(4.0)
    solution = model.exact_solution()
    assert solution.bits == (1, 0)
    normalized, scale = model.normalized()
    assert normalized.energy(solution.bits) == pytest.approx(solution.energy * scale)


def test_one_hot_and_binary_encodings_and_penalty():
    encoding = OneHotEncoding((2, 3))
    bits = encoding.encode((1, 2))
    assert encoding.decode(bits) == (1, 2)
    linear, quadratic, constant = encoding.constraint_penalty()
    penalty = QUBOModel(linear, quadratic, constant)
    assert penalty.energy(bits) == pytest.approx(0.0)
    assert penalty.energy((0, 0, 0, 0, 0)) == pytest.approx(2.0)
    binary = BinaryEncoding((3, 5))
    assert binary.decode(binary.encode((2, 4))) == (2, 4)


def test_adaptive_penalty_trust_region_and_candidate_selection():
    penalty = AdaptivePenalty(value=10.0, target_feasible_rate=0.5)
    assert penalty.update(0.1) == 20.0
    assert penalty.update(1.0) == 16.0
    region = TrustRegion(radius=2, minimum=1, maximum=3)
    rho, accepted = region.update(4.0, 4.0)
    assert (rho, accepted, region.radius) == (1.0, True, 3)
    rho, accepted = region.update(4.0, -1.0)
    assert not accepted and region.radius == 2
    selected = select_candidates([
        {"index": 5, "mass_saving": 0.2}, {"index": 2, "mass_saving": 0.9}
    ], 1)
    assert selected[0].index == 2


def test_exact_and_greedy_share_problem_protocol():
    problem = ToyProblem()
    exact = ExactBackend().solve(problem)
    assert exact.feasible
    assert design_values(exact.design) == (0, 2)
    assert exact.objective == 2.0
    greedy = GreedyBackend(pairwise=True, penalty=100.0).solve(problem)
    assert greedy.feasible
    assert greedy.objective == 2.0


def test_sa_solves_qubo_and_revalidates_decoded_design():
    # Minimum is x=(0, 1), decoded as the globally optimal feasible toy design.
    qubo = QUBOModel(np.array([1.0, -2.0]), np.zeros((2, 2)))
    backend = SimulatedAnnealingBackend(
        qubo=qubo, decoder=lambda bits: (0, 2) if bits == (0, 1) else (2, 0),
        sweeps=100, restarts=3, seed=7,
    )
    result = backend.solve(ToyProblem())
    assert result.feasible and result.objective == 2.0
    assert result.solver_metadata["qubo_energy"] == pytest.approx(-2.0)


def test_milp_requires_explicit_formulation_and_revalidates_with_fem_problem():
    formulation = MILPFormulation(
        objective=[3.0, 1.0], integrality=[1, 1], bounds=Bounds([0, 0], [1, 1]),
        constraints=LinearConstraint([[1.0, 1.0]], [1.0], [np.inf]),
        decoder=lambda x: (0, 2) if x[1] > 0.5 else (2, 0),
    )
    result = MILPBackend(formulation).solve(ToyProblem())
    assert result.feasible and result.objective == 2.0
    assert result.solver_metadata["linear_objective"] == pytest.approx(1.0)
    with pytest.raises(NotImplementedError, match="explicit linear formulation"):
        MILPBackend().solve(ToyProblem())


def test_qaoa_backend_is_importable_without_loading_qiskit():
    backend = QAOABackend(reps=1, maxiter=7)
    assert backend.reps == 1 and backend.maxiter == 7
    with pytest.raises(ValueError, match="maxiter"):
        QAOABackend(maxiter=0)


def test_qaoa_current_sampler_v2_api_when_optional_dependencies_are_installed():
    pytest.importorskip("qiskit_optimization")
    model = QUBOModel(np.array([1.0, -2.0]), np.zeros((2, 2)))
    solution = QAOABackend(reps=1, shots=256, seed=3).solve_qubo(model)
    assert len(solution.bits) == 2
    assert solution.energy == pytest.approx(model.energy(solution.bits))


def test_local_qubo_adapter_connects_canonical_problem_to_sa_and_qaoa_path():
    problem = ToyProblem()
    builder = LocalQUBOBuilder(problem, max_candidates=4,
                               penalty=AdaptivePenalty(value=100.0))
    adapter = LocalQUBOProblemAdapter(problem, builder)
    sa = SimulatedAnnealingBackend(sweeps=300, restarts=4, seed=4)
    result = sa.solve(adapter)
    assert result.feasible and result.objective == 2.0
    assert builder.last_metadata["pair_evaluations"] <= 6

    class ExactQAOAPath(QAOABackend):
        def solve_qubo(self, model):
            return model.exact_solution()

    quantum_path = ExactQAOAPath().solve(adapter)
    assert quantum_path.feasible and quantum_path.objective == 2.0


def test_sequential_qubo_updates_from_predicted_and_actual_fem_improvement():
    problem = ToyProblem()
    builder = LocalQUBOBuilder(problem, max_candidates=4,
                               penalty=AdaptivePenalty(value=100.0),
                               trust_region=TrustRegion(radius=2))
    solver = SimulatedAnnealingBackend(sweeps=300, restarts=4, seed=5)
    result = SequentialQUBOOptimizer(solver, builder, max_iterations=3).solve(problem)
    assert result.feasible and result.objective == 2.0
    assert isinstance(result.solver_metadata["qubo_energy"], float)
    assert result.solver_metadata["qubo_history"][0]["fem_feasible"]
    assert builder.trust_region.history
    assert "rho" in builder.trust_region.history[0]
