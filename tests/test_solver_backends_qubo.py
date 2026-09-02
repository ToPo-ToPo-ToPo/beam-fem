from dataclasses import dataclass

import numpy as np
import pytest

from beamfem.optimize.backends import (
    ExactBackend, GreedyBackend, MILPBackend, MILPFormulation, QAOABackend,
    MultiStartBackend, SequentialQUBOOptimizer, SimulatedAnnealingBackend,
)
from beamfem.optimize.backends.milp import build_truss_sizing_milp
from beamfem.optimize.topology import GroundStructure
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


def test_milp_can_auditably_repair_a_candidate_rejected_by_common_fem():
    class FeasibleAnchorProblem(ToyProblem):
        initial_design = State((2, 0))

    formulation = MILPFormulation(
        objective=[1.0], integrality=[1], bounds=Bounds([0], [1]),
        constraints=None, decoder=lambda _x: (0, 0),
        metadata={"formulation_scope": "deliberately_incomplete_test_scope"},
    )
    result = MILPBackend(
        formulation, fem_repair_backend=GreedyBackend(penalty=100.0),
    ).solve(FeasibleAnchorProblem())
    assert result.feasible
    assert result.backend == "milp_fem_repair"
    assert result.solver_metadata["milp_candidate_fem_feasible"] is False
    assert result.solver_metadata["fem_repair_performed"] is True
    assert result.solver_metadata["fem_repair_start"] == "initial_design"


def test_truss_milp_decodes_explicit_problem_state_indices():
    gs = GroundStructure(
        nodes=np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]),
        members=[(0, 1), (0, 2), (1, 2)],
        supports={0: [0, 1], 1: [1]},
        load_cases=[{(2, 1): -1.0}],
    )
    formulation = build_truss_sizing_milp(
        gs, [0.0, 0.1], density=1.0, tensile_stress=100.0,
        state_indices=[2, 4],
    )

    class StateIndexProblem:
        initial_design = State((4, 4, 4))
        domains = ((2, 4),) * 3

        def evaluate(self, design):
            return Evaluation(float(sum(design_values(design))), True, ())

    result = MILPBackend(formulation).solve(StateIndexProblem())
    assert set(result.design.choices) <= {2, 4}
    assert result.solver_metadata["decoded_state_indices"] == [[2, 4]] * 3
    with pytest.raises(ValueError, match="nonnegative integers"):
        build_truss_sizing_milp(
            gs, [0.0, 0.1], density=1.0, tensile_stress=100.0,
            state_indices=[0, 1.5],
        )


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
    assert solution.metadata["logical_circuit_depth"] >= 1
    assert solution.metadata["logical_qubits"] == 2
    assert solution.metadata["energy_gap_to_exact"] >= -1e-12


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


def test_parallel_local_evaluation_is_explicit_and_reports_phase_timings():
    problem = ToyProblem()
    with pytest.raises(ValueError, match="parallel_safe"):
        LocalQUBOBuilder(problem, parallel_workers=2)
    builder = LocalQUBOBuilder(problem, max_candidates=4, parallel_workers=2,
                               parallel_safe=True, penalty=AdaptivePenalty(value=100.0))
    builder.build(problem.initial_design)
    assert builder.last_metadata["parallel_workers"] == 2
    assert builder.last_metadata["build_seconds"] >= builder.last_metadata["screening_seconds"]


def test_process_parallel_local_evaluation_matches_sequential_qubo_bitwise():
    problem = ToyProblem()
    sequential, _ = LocalQUBOBuilder(
        problem, max_candidates=4, penalty=AdaptivePenalty(value=100.0),
    ).build(problem.initial_design)
    with LocalQUBOBuilder(
        problem, max_candidates=4, parallel_workers=2,
        parallel_backend="process", persistent_workers=True,
        penalty=AdaptivePenalty(value=100.0),
    ) as builder:
        parallel, _ = builder.build(problem.initial_design)
        assert builder.last_metadata["parallel_backend"] == "process"
        assert builder.last_metadata["persistent_workers"] is True
    assert np.array_equal(parallel.linear, sequential.linear)
    assert np.array_equal(parallel.quadratic, sequential.quadratic)
    assert parallel.constant == sequential.constant
    assert parallel.variable_names == sequential.variable_names


def test_parallel_backend_rejects_unknown_executor_kind():
    with pytest.raises(ValueError, match="parallel_backend"):
        LocalQUBOBuilder(ToyProblem(), parallel_backend="gpu")


def test_checkpoint_resume_and_optimality_gap(tmp_path):
    checkpoint = tmp_path / "optimizer.json"
    problem = ToyProblem()
    first_builder = LocalQUBOBuilder(problem, max_candidates=4,
                                     penalty=AdaptivePenalty(value=100.0))
    first = SequentialQUBOOptimizer(
        SimulatedAnnealingBackend(sweeps=200, restarts=3, seed=2), first_builder,
        max_iterations=1, checkpoint_path=checkpoint, reference_objective=2.0,
    ).solve(problem)
    assert checkpoint.exists() and first.solver_metadata["optimality_gap"] == pytest.approx(0.0)
    second_builder = LocalQUBOBuilder(problem, max_candidates=4,
                                      penalty=AdaptivePenalty(value=100.0))
    resumed = SequentialQUBOOptimizer(
        SimulatedAnnealingBackend(sweeps=200, restarts=3, seed=2), second_builder,
        max_iterations=1, checkpoint_path=checkpoint, resume=True,
    ).solve(problem)
    assert resumed.solver_metadata["resumed"]
    assert resumed.objective == first.objective


def test_multi_start_reports_all_runs_and_reference_gap():
    result = MultiStartBackend(lambda seed: GreedyBackend(penalty=100.0),
                               starts=3, seed=9, reference_objective=2.0).solve(ToyProblem())
    assert result.feasible and result.objective == 2.0
    assert len(result.solver_metadata["multi_start"]) == 3
    assert result.solver_metadata["optimality_gap"] == pytest.approx(0.0)


def test_truss_milp_builder_states_exact_scope_and_fem_revalidation():
    gs = GroundStructure(
        nodes=np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]),
        members=[(0, 1), (0, 2), (1, 2)],
        supports={0: [0, 1], 1: [1]},
        load_cases=[{(2, 1): -1.0}],
    )
    formulation = build_truss_sizing_milp(gs, [0.0, 0.1], density=1.0,
                                           tensile_stress=100.0)

    class TrussProblem:
        initial_design = State((0, 0, 0))
        domains = ((0, 1),) * 3
        def evaluate(self, design):
            choices = design_values(design)
            return Evaluation(float(sum(choices)), True, ())

    result = MILPBackend(formulation).solve(TrussProblem())
    assert result.feasible
    assert result.solver_metadata["formulation_scope"] == "truss_equilibrium_and_section_capacity"
    assert not result.solver_metadata["elastic_compatibility"]
