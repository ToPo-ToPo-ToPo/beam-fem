"""Feasibility-first local search backend."""

from __future__ import annotations

from time import perf_counter
import math
from typing import Any, Sequence

from .base import (
    EvaluationCache, HistoryEntry, OptimizationResult, SolverLimits, StopController,
    design_values, evaluation_constraints, evaluation_feasible, evaluation_objective,
    evaluation_violation, make_design,
)
from .exact import infer_domains


def merit(evaluation: Any, penalty: float) -> float:
    return evaluation_objective(evaluation) + penalty * evaluation_violation(evaluation)


class GreedyBackend:
    def __init__(self, domains: Sequence[Sequence[int]] | None = None,
                 penalty: float = 1e6, pairwise: bool = True,
                 max_pair_candidates: int | None = 24):
        self.domains = None if domains is None else tuple(tuple(d) for d in domains)
        self.penalty, self.pairwise = float(penalty), bool(pairwise)
        if not math.isfinite(self.penalty) or self.penalty <= 0:
            raise ValueError("penalty must be finite and positive")
        if max_pair_candidates is not None and max_pair_candidates < 2:
            raise ValueError("max_pair_candidates must be at least two or None")
        self.max_pair_candidates = max_pair_candidates

    def solve(self, problem: Any, initial_design: Any | None = None,
              limits: SolverLimits | None = None) -> OptimizationResult:
        started = perf_counter()
        template = initial_design if initial_design is not None else problem.initial_design
        current = design_values(template)
        domains = self.domains or infer_domains(problem, current)
        cache, stop = EvaluationCache(problem, template), StopController(limits)
        current_eval, iteration = cache(current), 0
        history = [HistoryEntry(0, evaluation_objective(current_eval), evaluation_feasible(current_eval), current)]
        message = "local optimum reached"
        while True:
            reason = stop.reached(cache.evaluations, iteration)
            if reason:
                message = reason
                break
            single_candidates = set()
            for i, domain in enumerate(domains):
                for value in domain:
                    if value != current[i]:
                        candidate = list(current); candidate[i] = value
                        single_candidates.add(tuple(candidate))
            ranked_singles = []
            for candidate in single_candidates:
                if stop.reached(cache.evaluations, iteration):
                    break
                ev = cache(candidate)
                ranked_singles.append((merit(ev, self.penalty), candidate, ev))
            ranked_singles.sort(key=lambda item: item[0])
            candidates = set(single_candidates)
            if self.pairwise:
                one_moves = ranked_singles
                if self.max_pair_candidates is not None:
                    one_moves = one_moves[:self.max_pair_candidates]
                for _, a, _ in one_moves:
                    changed = next(i for i in range(len(current)) if a[i] != current[i])
                    for j, domain in enumerate(domains):
                        if j != changed:
                            for value in domain:
                                if value != current[j]:
                                    candidate = list(a); candidate[j] = value
                                    candidates.add(tuple(candidate))
            ranked = list(ranked_singles)
            for candidate in candidates - single_candidates:
                if stop.reached(cache.evaluations, iteration):
                    break
                ev = cache(candidate)
                ranked.append((merit(ev, self.penalty), candidate, ev))
            if not ranked:
                break
            _, candidate, candidate_eval = min(ranked, key=lambda item: item[0])
            if merit(candidate_eval, self.penalty) >= merit(current_eval, self.penalty) - 1e-12:
                break
            current, current_eval = candidate, candidate_eval
            iteration += 1
            history.append(HistoryEntry(iteration, evaluation_objective(current_eval),
                                        evaluation_feasible(current_eval), current))
        return OptimizationResult(make_design(problem, current, template), evaluation_objective(current_eval),
            evaluation_feasible(current_eval), evaluation_constraints(current_eval), iteration,
            cache.evaluations, perf_counter()-started, "greedy", "success", message,
            {"penalty": self.penalty, "pairwise": self.pairwise}, tuple(history), current_eval)
