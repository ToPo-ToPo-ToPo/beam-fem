"""Sequential local-QUBO optimization with FEM acceptance checks."""

from __future__ import annotations

from time import perf_counter
from typing import Any

from ..qubo.local import LocalQUBOBuilder
from .base import (
    HistoryEntry, OptimizationResult, SolverLimits, design_values, evaluate_problem,
    evaluation_constraints, evaluation_feasible, evaluation_objective,
    evaluation_violation, make_design,
)


class SequentialQUBOOptimizer:
    """Repeatedly rebuild a local QUBO, solve it, and verify with FEM."""

    def __init__(self, qubo_solver: Any, builder: LocalQUBOBuilder,
                 max_iterations: int = 20, improvement_tolerance: float = 1e-9):
        self.qubo_solver, self.builder = qubo_solver, builder
        self.max_iterations = int(max_iterations)
        self.improvement_tolerance = float(improvement_tolerance)

    def _merit(self, evaluation: Any) -> float:
        return (evaluation_objective(evaluation) +
                self.builder.penalty.value * evaluation_violation(evaluation))

    def solve(self, problem: Any, initial_design: Any | None = None,
              limits: SolverLimits | None = None) -> OptimizationResult:
        started = perf_counter()
        template = initial_design if initial_design is not None else problem.initial_design
        current = design_values(template)
        current_eval = evaluate_problem(problem, make_design(problem, current, template))
        history = [HistoryEntry(0, evaluation_objective(current_eval),
                                evaluation_feasible(current_eval), current)]
        total_fem, message = 1, "maximum iterations reached"
        qubo_history: list[dict[str, Any]] = []
        iteration_cap = min(self.max_iterations,
                            limits.max_iterations if limits and limits.max_iterations else self.max_iterations)
        for iteration in range(1, iteration_cap + 1):
            model, decoder = self.builder.build(make_design(problem, current, template))
            total_fem += int(self.builder.last_metadata["fem_evaluations"])
            solution = self.qubo_solver.solve_qubo(model)
            candidate = tuple(decoder(solution.bits))
            candidate_eval = evaluate_problem(problem, make_design(problem, candidate, template))
            total_fem += 1
            base_bits = self.builder.last_metadata["base_bits"]
            predicted = model.energy(base_bits) - solution.energy
            actual = self._merit(current_eval) - self._merit(candidate_eval)
            rho, accepted = self.builder.trust_region.update(predicted, actual)
            qubo_history.append({
                "iteration": iteration,
                "energy": solution.energy,
                "base_energy": model.energy(base_bits),
                "bits": solution.bits,
                "predicted_improvement": predicted,
                "fem_objective": evaluation_objective(candidate_eval),
                "fem_feasible": evaluation_feasible(candidate_eval),
                "accepted": accepted and actual > self.improvement_tolerance,
                "solver": dict(solution.metadata or {}),
            })
            self.builder.penalty.update(1.0 if evaluation_feasible(candidate_eval) else 0.0)
            if accepted and actual > self.improvement_tolerance:
                current, current_eval = candidate, candidate_eval
                history.append(HistoryEntry(iteration, evaluation_objective(current_eval),
                    evaluation_feasible(current_eval), current,
                    {"predicted_improvement": predicted, "actual_improvement": actual, "rho": rho}))
            else:
                message = "trust-region step rejected or no improvement"
                break
            if limits and limits.max_evaluations and total_fem >= limits.max_evaluations:
                message = "maximum evaluations reached"
                break
            if limits and limits.time_limit and perf_counter() - started >= limits.time_limit:
                message = "time limit reached"
                break
        return OptimizationResult(make_design(problem, current, template), evaluation_objective(current_eval),
            evaluation_feasible(current_eval), evaluation_constraints(current_eval),
            len(history)-1, total_fem, perf_counter()-started, "sequential_qubo", "success",
            message, {"qubo_solver": type(self.qubo_solver).__name__,
                      "qubo_energy": qubo_history[-1]["energy"] if qubo_history else None,
                      "qubo_history": tuple(qubo_history),
                      "trust_region_history": tuple(self.builder.trust_region.history),
                      "penalty_history": tuple(self.builder.penalty.history)},
            tuple(history), current_eval)
