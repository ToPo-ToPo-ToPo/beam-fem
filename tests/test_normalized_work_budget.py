from dataclasses import dataclass

import numpy as np

from beamfem.optimize.backends import (
    MultiStartBackend, OptimizationResult, SimulatedAnnealingBackend, SolverLimits,
)
from beamfem.optimize.qubo import AdaptivePenalty, LocalQUBOBuilder, QUBOModel


@dataclass(frozen=True)
class Evaluation:
    objective: float = 1.0
    feasible: bool = True
    constraints: tuple = ()


class Problem:
    initial_design = (0,)
    domains = ((0, 1),)

    def evaluate(self, design):
        return Evaluation()


class OneEvaluationBackend:
    def solve(self, problem, start, limits):
        return OptimizationResult(
            start, 1.0, True, evaluations=1, iterations=1, backend="unit",
        )


def test_sa_reports_separate_fem_qubo_and_quantum_work_dimensions():
    qubo = QUBOModel(np.array([-1.0]), np.zeros((1, 1)))
    result = SimulatedAnnealingBackend(
        qubo=qubo, sweeps=100, restarts=4, seed=1,
    ).solve(Problem(), limits=SolverLimits(max_iterations=7))
    work = result.solver_metadata["normalized_work"]
    assert result.iterations == 7
    assert work["fem_evaluations"] == 1
    assert work["classical_objective_evaluations"] == 8
    assert work["quantum_shots"] == 0
    assert work["budget_dimensions_are_not_interchangeable"] is True


def test_multistart_uses_one_aggregate_evaluation_budget():
    result = MultiStartBackend(lambda seed: OneEvaluationBackend(), starts=5).solve(
        Problem(), limits=SolverLimits(max_evaluations=2)
    )
    assert result.evaluations == 2
    assert result.solver_metadata["starts_executed"] == 2
    assert result.solver_metadata["budget_scope"] == "aggregate_across_starts"


def test_default_local_qubo_path_records_engineering_indicator_selection():
    builder = LocalQUBOBuilder(
        Problem(), max_candidates=1, penalty=AdaptivePenalty(value=10.0),
    )
    builder.build(Problem.initial_design)
    assert builder.last_metadata["candidate_selection"] == (
        "normalized_engineering_indicators_v1"
    )
    indicators = builder.last_metadata["candidate_indicators"][0]
    assert set(indicators) == {
        "mass_saving", "utilization", "strain_energy", "buckling_margin",
        "connectivity", "recent_improvement",
    }
