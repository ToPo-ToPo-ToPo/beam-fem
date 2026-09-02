"""Exhaustive backend used to establish small-problem reference optima."""

from __future__ import annotations

from itertools import product
from math import prod
from time import perf_counter
from typing import Any, Sequence

from .base import (
    EvaluationCache, HistoryEntry, OptimizationResult, SolverLimits,
    StopController, design_values, evaluation_constraints, evaluation_feasible,
    evaluation_objective, make_design,
)


def infer_domains(problem: Any, initial: Sequence[int]) -> tuple[tuple[int, ...], ...]:
    domains = getattr(problem, "domains", None)
    if domains is None:
        domains = getattr(problem, "state_domains", None)
    if domains is None and hasattr(problem, "catalogs"):
        domains = [range(len(c)) for c in problem.catalogs]
    if domains is None and hasattr(problem, "section_catalogs"):
        domains = [range(len(c) + 1) for c in problem.section_catalogs]
    if domains is None:
        raise TypeError("problem must expose domains/state_domains, or backend domains must be supplied")
    result = tuple(tuple(int(v) for v in domain) for domain in domains)
    if len(result) != len(initial) or any(not domain for domain in result):
        raise ValueError("domains must contain one non-empty domain per design variable")
    return result


class ExactBackend:
    def __init__(self, domains: Sequence[Sequence[int]] | None = None, max_combinations: int = 200_000):
        self.domains = None if domains is None else tuple(tuple(d) for d in domains)
        self.max_combinations = int(max_combinations)

    def solve(self, problem: Any, initial_design: Any | None = None,
              limits: SolverLimits | None = None) -> OptimizationResult:
        started = perf_counter()
        template = initial_design if initial_design is not None else problem.initial_design
        initial = design_values(template)
        domains = self.domains or infer_domains(problem, initial)
        total = prod(len(d) for d in domains)
        if total > self.max_combinations:
            raise ValueError(f"combination count {total} exceeds limit {self.max_combinations}")
        cache, stop = EvaluationCache(problem, template), StopController(limits)
        best_values, best_eval, history = None, None, []
        status, message, iteration = "success", "global enumeration complete", 0
        for iteration, values in enumerate(product(*domains), 1):
            reason = stop.reached(cache.evaluations, iteration - 1)
            if reason:
                status, message = "stopped", reason
                break
            evaluation = cache(values)
            if evaluation_feasible(evaluation) and (
                best_eval is None or evaluation_objective(evaluation) < evaluation_objective(best_eval)
            ):
                best_values, best_eval = tuple(values), evaluation
                history.append(HistoryEntry(iteration, evaluation_objective(evaluation), True, tuple(values)))
        if best_eval is None:
            return OptimizationResult(make_design(problem, initial, template), float("inf"), False,
                iterations=iteration, evaluations=cache.evaluations, runtime=perf_counter()-started,
                backend="exact", status="infeasible", message="no feasible design found")
        return OptimizationResult(make_design(problem, best_values, template), evaluation_objective(best_eval), True,
            evaluation_constraints(best_eval), iteration, cache.evaluations, perf_counter()-started,
            "exact", status, message, {"combinations": total}, tuple(history), best_eval)
