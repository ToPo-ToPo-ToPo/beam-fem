"""Exact Pareto-front generation for auditable small discrete problems."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import prod
from time import perf_counter
from typing import Any, Mapping, Sequence

from .backends.base import (
    EvaluationCache, SolverLimits, StopController, design_values,
    evaluation_feasible, make_design,
)
from .backends.exact import infer_domains
from .objectives import impact_components


@dataclass(frozen=True)
class ParetoPoint:
    design: Any
    objectives: Mapping[str, float]
    evaluation: Any

    def as_dict(self) -> dict[str, Any]:
        return {
            "design": list(design_values(self.design)),
            "objectives": dict(self.objectives),
            "feasible": bool(evaluation_feasible(self.evaluation)),
        }


@dataclass(frozen=True)
class ParetoResult:
    points: tuple[ParetoPoint, ...]
    evaluations: int
    runtime: float
    status: str
    message: str
    solver_metadata: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "points": [point.as_dict() for point in self.points],
            "evaluations": self.evaluations,
            "runtime": self.runtime,
            "status": self.status,
            "message": self.message,
            "solver_metadata": dict(self.solver_metadata),
        }


def _dominates(left: ParetoPoint, right: ParetoPoint, tolerance: float) -> bool:
    keys = left.objectives
    return (
        all(left.objectives[key] <= right.objectives[key] + tolerance for key in keys)
        and any(left.objectives[key] < right.objectives[key] - tolerance for key in keys)
    )


class ParetoFrontBackend:
    """Enumerate and retain nondominated feasible mass/cost/carbon designs."""

    def __init__(self, objectives: Sequence[str] = ("mass", "cost", "carbon"),
                 domains: Sequence[Sequence[int]] | None = None,
                 max_combinations: int = 200_000, tolerance: float = 1e-12):
        names = tuple(dict.fromkeys(str(name) for name in objectives))
        if not names or set(names) - {"mass", "cost", "carbon"}:
            raise ValueError("Pareto objectives must be mass, cost, and/or carbon")
        if max_combinations < 1 or tolerance < 0:
            raise ValueError("invalid Pareto enumeration settings")
        self.objectives = names
        self.domains = None if domains is None else tuple(tuple(d) for d in domains)
        self.max_combinations = int(max_combinations)
        self.tolerance = float(tolerance)

    def solve(self, problem: Any, initial_design: Any | None = None,
              limits: SolverLimits | None = None) -> ParetoResult:
        started = perf_counter()
        template = initial_design if initial_design is not None else problem.initial_design
        domains = self.domains or infer_domains(problem, design_values(template))
        combinations = prod(len(domain) for domain in domains)
        if combinations > self.max_combinations:
            raise ValueError(
                f"combination count {combinations} exceeds limit {self.max_combinations}"
            )
        cache, stop = EvaluationCache(problem, template), StopController(limits)
        front: list[ParetoPoint] = []
        status, message = "success", "global Pareto enumeration complete"
        for iteration, values in enumerate(product(*domains)):
            reason = stop.reached(cache.evaluations, iteration)
            if reason:
                status, message = "stopped", reason
                break
            evaluation = cache(values)
            if not evaluation_feasible(evaluation):
                continue
            design = make_design(problem, values, template)
            all_values = impact_components(problem, design)
            point = ParetoPoint(
                design,
                {name: all_values[name] for name in self.objectives},
                evaluation,
            )
            if any(_dominates(existing, point, self.tolerance) for existing in front):
                continue
            front = [
                existing for existing in front
                if not _dominates(point, existing, self.tolerance)
            ]
            front.append(point)
        front.sort(key=lambda point: (
            tuple(point.objectives[name] for name in self.objectives),
            design_values(point.design),
        ))
        metadata = {
            "objectives": self.objectives,
            "combinations": combinations,
            "global_for_enumerated_scope": status == "success",
            "normalized_work": {
                "fem_evaluations": cache.evaluations,
                "classical_objective_evaluations": combinations if status == "success" else cache.evaluations,
                "quantum_shots": 0,
                "budget_dimensions_are_not_interchangeable": True,
            },
        }
        return ParetoResult(
            tuple(front), cache.evaluations, perf_counter() - started,
            status if front else "infeasible", message, metadata,
        )
