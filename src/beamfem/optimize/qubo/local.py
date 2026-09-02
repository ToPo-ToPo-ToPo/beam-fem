"""Build sparse local QUBOs from actual structural FEM evaluations."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from math import inf
from typing import Any, Sequence

import numpy as np

from .model import QUBOModel
from .penalties import AdaptivePenalty
from .trust_region import TrustRegion


@dataclass(frozen=True)
class DesignMove:
    member: int
    state: int
    single_merit: float
    predicted_improvement: float


def _values(design: Any) -> tuple[int, ...]:
    for name in ("choices", "states", "indices", "values"):
        if hasattr(design, name):
            return tuple(int(v) for v in getattr(design, name))
    return tuple(int(v) for v in design)


def _make(template: Any, values: Sequence[int]) -> Any:
    values = tuple(int(v) for v in values)
    if isinstance(template, (tuple, list)):
        return values
    try:
        return type(template)(values)
    except (TypeError, ValueError):
        try:
            return replace(template, choices=values)
        except (TypeError, ValueError):
            return values


def _objective(evaluation: Any) -> float:
    for name in ("objective", "score", "mass"):
        if hasattr(evaluation, name):
            return float(getattr(evaluation, name))
    if isinstance(evaluation, tuple):
        return float(evaluation[0])
    return float(evaluation["objective"])


def _violation(evaluation: Any) -> float:
    constraints = getattr(evaluation, "constraints", ())
    total = 0.0
    for item in constraints:
        if hasattr(item, "satisfied") and hasattr(item, "utilization"):
            total += 0.0 if item.satisfied else max(0.0, float(item.utilization) - 1.0)
        else:
            total += max(0.0, float(item))
    feasible = getattr(evaluation, "feasible", True)
    return total if feasible or total > 0 else inf


class LocalQUBOBuilder:
    """Quadratic FEM response model in a bounded design neighbourhood.

    Every possible single-member move is screened, but pair evaluations are
    limited to ``max_candidates`` selected moves. This avoids O(n²) FEM work
    as the ground structure grows while retaining measured interaction terms.
    """

    def __init__(self, problem: Any, domains: Sequence[Sequence[int]] | None = None,
                 max_candidates: int = 8, trust_region: TrustRegion | None = None,
                 penalty: AdaptivePenalty | None = None):
        if max_candidates < 1:
            raise ValueError("max_candidates must be positive")
        self.problem, self.domains = problem, domains
        self.max_candidates = int(max_candidates)
        self.trust_region = trust_region or TrustRegion(radius=2)
        self.penalty = penalty or AdaptivePenalty(value=10.0)
        self.last_metadata: dict[str, Any] = {}

    @staticmethod
    def _merit(evaluation: Any, penalty: float) -> float:
        return _objective(evaluation) + penalty * _violation(evaluation)

    def build(self, initial_design: Any) -> tuple[QUBOModel, Any]:
        base = _values(initial_design)
        if self.domains is not None:
            domains = tuple(tuple(int(v) for v in d) for d in self.domains)
        elif hasattr(self.problem, "domains"):
            domains = tuple(tuple(int(v) for v in d) for d in self.problem.domains)
        elif hasattr(self.problem, "catalogs"):
            domains = tuple(tuple(range(len(c))) for c in self.problem.catalogs)
        else:
            raise TypeError("problem must expose domains or catalogs")
        cache_values: dict[tuple[int, ...], Any] = {}
        def evaluate(values):
            key = tuple(values)
            if key not in cache_values:
                cache_values[key] = self.problem.evaluate(_make(initial_design, key))
            return cache_values[key]
        base_eval = evaluate(base)
        base_merit = self._merit(base_eval, self.penalty.value)
        moves = []
        for member, domain in enumerate(domains):
            for state in domain:
                if state == base[member]:
                    continue
                changed = list(base); changed[member] = state
                score = self._merit(evaluate(changed), self.penalty.value)
                moves.append(DesignMove(member, state, score, base_merit - score))
        moves.sort(key=lambda move: (-move.predicted_improvement, move.member, move.state))
        moves = moves[:self.max_candidates]

        n_actions = len(moves)
        radius = min(self.trust_region.radius, max(1, len(base)))
        # Binary slack represents radius - selected_actions.
        slack_width = max(1, int(np.ceil(np.log2(radius + 1))))
        slack_weights = tuple(1 << i for i in range(slack_width))
        n = n_actions + slack_width
        linear, quadratic = np.zeros(n), np.zeros((n, n))
        linear[:n_actions] = [move.single_merit - base_merit for move in moves]

        pair_evaluations = 0
        for i, first in enumerate(moves):
            for j in range(i + 1, n_actions):
                second = moves[j]
                if first.member == second.member:
                    quadratic[i, j] += self.penalty.value
                    continue
                changed = list(base)
                changed[first.member], changed[second.member] = first.state, second.state
                pair_merit = self._merit(evaluate(changed), self.penalty.value)
                quadratic[i, j] += pair_merit - first.single_merit - second.single_merit + base_merit
                pair_evaluations += 1

        coefficients = [1.0] * n_actions + [float(v) for v in slack_weights]
        surrogate_scale = max(float(np.max(np.abs(linear), initial=0.0)),
                              float(np.max(np.abs(quadratic), initial=0.0)), 1.0)
        cardinality_penalty = max(self.penalty.value, 10.0 * surrogate_scale)
        constant = base_merit + cardinality_penalty * radius * radius
        for i, coefficient in enumerate(coefficients):
            linear[i] += cardinality_penalty * (coefficient * coefficient - 2 * radius * coefficient)
            for j in range(i + 1, n):
                quadratic[i, j] += 2 * cardinality_penalty * coefficient * coefficients[j]

        def decode(bits):
            chosen = [i for i in range(n_actions) if int(bits[i])]
            chosen.sort(key=lambda i: (linear[i], moves[i].member, moves[i].state))
            result, changed_members = list(base), set()
            for i in chosen:
                move = moves[i]
                if move.member not in changed_members and len(changed_members) < radius:
                    result[move.member] = move.state
                    changed_members.add(move.member)
            return tuple(result)

        names = tuple([f"move_m{m.member}_s{m.state}" for m in moves] +
                      [f"trust_slack_{i}" for i in range(slack_width)])
        self.last_metadata = {
            "base_merit": base_merit, "moves": tuple(moves), "radius": radius,
            "screened_moves": sum(len(d) - 1 for d in domains),
            "pair_evaluations": pair_evaluations, "fem_evaluations": len(cache_values),
            "cardinality_penalty": cardinality_penalty,
            "base_bits": tuple([0] * n_actions + [
                (radius >> i) & 1 for i in range(slack_width)
            ]),
        }
        return QUBOModel(linear, quadratic, constant, names), decode


class LocalQUBOProblemAdapter:
    """Add ``build_qubo`` to a canonical problem without modifying it."""

    def __init__(self, problem: Any, builder: LocalQUBOBuilder | None = None):
        self.problem = problem
        self.initial_design = problem.initial_design
        self.builder = builder or LocalQUBOBuilder(problem)

    def evaluate(self, design: Any):
        return self.problem.evaluate(design)

    def build_qubo(self, initial_design: Any):
        return self.builder.build(initial_design)

    def __getattr__(self, name: str):
        return getattr(self.problem, name)
