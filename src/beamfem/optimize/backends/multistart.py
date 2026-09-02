"""Deterministic multi-start wrapper for heuristic optimizers."""

from __future__ import annotations

from dataclasses import replace
from time import perf_counter
from typing import Any, Callable, Sequence

import numpy as np

from .base import OptimizationResult, SolverLimits, design_values, make_design
from .exact import infer_domains


class MultiStartBackend:
    """Run independent solver instances and retain the best FEM result.

    ``backend_factory`` receives a deterministic integer seed.  No speed-up or
    quantum advantage is inferred from repeated runs; all aggregate wall time
    and individual outcomes are reported.
    """

    def __init__(self, backend_factory: Callable[[int], Any], starts: int = 8,
                 seed: int = 0, initial_designs: Sequence[Any] | None = None,
                 reference_objective: float | None = None):
        if starts < 1:
            raise ValueError("starts must be positive")
        self.backend_factory = backend_factory
        self.starts, self.seed = int(starts), int(seed)
        self.initial_designs = None if initial_designs is None else tuple(initial_designs)
        self.reference_objective = reference_objective

    def solve(self, problem: Any, initial_design: Any | None = None,
              limits: SolverLimits | None = None) -> OptimizationResult:
        started = perf_counter()
        template = initial_design if initial_design is not None else problem.initial_design
        rng = np.random.default_rng(self.seed)
        domains = infer_domains(problem, design_values(template))
        supplied = list(self.initial_designs or ())
        starts = []
        for index in range(self.starts):
            if index < len(supplied):
                starts.append(supplied[index])
            elif index == 0:
                starts.append(template)
            else:
                values = tuple(int(rng.choice(domain)) for domain in domains)
                starts.append(make_design(problem, values, template))
        results = []
        for index, start in enumerate(starts):
            results.append(self.backend_factory(self.seed + index).solve(problem, start, limits))
        feasible = [result for result in results if result.feasible]
        candidates = feasible or results
        best = min(candidates, key=lambda result: result.objective)
        gap = None
        if self.reference_objective is not None:
            gap = ((best.objective - self.reference_objective) /
                   max(abs(self.reference_objective), 1e-12))
        summary = tuple({"start": index, "objective": result.objective,
                         "feasible": result.feasible, "runtime": result.runtime,
                         "backend": result.backend}
                        for index, result in enumerate(results))
        metadata = dict(best.solver_metadata)
        metadata.update({"multi_start": summary, "starts": self.starts,
                         "feasible_starts": len(feasible), "optimality_gap": gap,
                         "aggregate_solver_runtime": sum(result.runtime for result in results)})
        return replace(best, runtime=perf_counter()-started, backend="multi_start",
                       evaluations=sum(result.evaluations for result in results),
                       iterations=sum(result.iterations for result in results),
                       solver_metadata=metadata)
