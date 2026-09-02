"""Build sparse local QUBOs from actual structural FEM evaluations."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from math import inf, isfinite
import inspect
from time import perf_counter
from typing import Any, Sequence

import numpy as np

from .model import QUBOModel
from .candidate_selection import select_candidates
from .penalties import AdaptivePenalty
from .trust_region import TrustRegion


_PROCESS_PROBLEM: Any | None = None
_PROCESS_SUPPORTS_CACHE_CONTROL = False


def _initialize_process_problem(problem: Any) -> None:
    """Install one private problem copy per worker (also works with spawn)."""
    global _PROCESS_PROBLEM, _PROCESS_SUPPORTS_CACHE_CONTROL
    _PROCESS_PROBLEM = problem
    _PROCESS_SUPPORTS_CACHE_CONTROL = "use_cache" in inspect.signature(problem.evaluate).parameters


def _evaluate_process_design(design: Any) -> Any:
    if _PROCESS_PROBLEM is None:  # pragma: no cover - defensive worker failure
        raise RuntimeError("parallel FEM worker was not initialized")
    if _PROCESS_SUPPORTS_CACHE_CONTROL:
        evaluation = _PROCESS_PROBLEM.evaluate(design, use_cache=False)
    else:
        evaluation = _PROCESS_PROBLEM.evaluate(design)
    return _EvaluationSummary(
        _objective(evaluation),
        _violation(evaluation),
        float(getattr(evaluation, "mass", _objective(evaluation))),
        _engineering_indicators(evaluation, 0.0, 0.0),
    )


@dataclass(frozen=True)
class DesignMove:
    member: int
    state: int
    single_merit: float
    predicted_improvement: float
    indicators: dict[str, float] | None = None


@dataclass(frozen=True)
class _EvaluationSummary:
    """Small process-transfer object containing all QUBO-builder inputs."""
    objective: float
    violation: float
    mass: float
    engineering_indicators: dict[str, float]


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
    if isinstance(evaluation, _EvaluationSummary):
        return evaluation.violation
    constraints = getattr(evaluation, "constraints", ())
    total = 0.0
    for item in constraints:
        if hasattr(item, "satisfied") and hasattr(item, "utilization"):
            total += 0.0 if item.satisfied else max(0.0, float(item.utilization) - 1.0)
        else:
            total += max(0.0, float(item))
    feasible = getattr(evaluation, "feasible", True)
    return total if feasible or total > 0 else inf


def _engineering_indicators(evaluation: Any, mass_saving: float,
                            recent_improvement: float) -> dict[str, float]:
    """Extract auditable, backend-neutral screening indicators."""
    violation = _violation(evaluation)
    if isinstance(evaluation, _EvaluationSummary):
        indicators = dict(evaluation.engineering_indicators)
        indicators["mass_saving"] = float(mass_saving)
        indicators["recent_improvement"] = float(recent_improvement)
        return indicators
    constraints = getattr(evaluation, "constraints", ())
    utilizations = []
    buckling = []
    for item in constraints:
        utilization = getattr(item, "utilization", None)
        if utilization is None or not isfinite(float(utilization)):
            continue
        utilizations.append(float(utilization))
        if getattr(item, "kind", "") == "euler_buckling":
            buckling.append(float(utilization))
    strain_energy = 0.0
    for analysis in getattr(evaluation, "analyses", {}).values():
        displacement = analysis.static.u
        strain_energy += max(0.0, float(0.5 * displacement @ (analysis.static.K @ displacement)))
    stable = 0.0 if not isfinite(violation) else 1.0
    maximum_utilization = max(utilizations, default=0.0)
    maximum_buckling = max(buckling, default=0.0)
    return {
        "mass_saving": float(mass_saving),
        "utilization": 1.0 / (1.0 + max(0.0, maximum_utilization - 1.0)),
        "strain_energy": strain_energy,
        "buckling_margin": 1.0 / (1.0 + max(0.0, maximum_buckling)),
        "connectivity": stable,
        "recent_improvement": float(recent_improvement),
    }


def _normalize_indicator_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = (
        "mass_saving", "utilization", "strain_energy", "buckling_margin",
        "connectivity", "recent_improvement",
    )
    output = [dict(record) for record in records]
    for key in keys:
        values = np.asarray([float(record.get(key, 0.0)) for record in output])
        finite = np.isfinite(values)
        if not np.any(finite):
            normalized = np.zeros_like(values)
        else:
            low, high = float(np.min(values[finite])), float(np.max(values[finite]))
            normalized = np.zeros_like(values)
            if high > low:
                normalized[finite] = (values[finite] - low) / (high - low)
        for record, value in zip(output, normalized):
            record[key] = float(value)
    return output


class LocalQUBOBuilder:
    """Quadratic FEM response model in a bounded design neighbourhood.

    Every possible single-member move is screened, but pair evaluations are
    limited to ``max_candidates`` selected moves. This avoids O(n²) FEM work
    as the ground structure grows while retaining measured interaction terms.
    """

    def __init__(self, problem: Any, domains: Sequence[Sequence[int]] | None = None,
                 max_candidates: int = 8, trust_region: TrustRegion | None = None,
                 penalty: AdaptivePenalty | None = None, parallel_workers: int = 1,
                 parallel_safe: bool = False, parallel_backend: str = "thread",
                 persistent_workers: bool = False):
        if max_candidates < 1:
            raise ValueError("max_candidates must be positive")
        self.problem, self.domains = problem, domains
        self.max_candidates = int(max_candidates)
        self.trust_region = trust_region or TrustRegion(radius=2)
        self.penalty = penalty or AdaptivePenalty(value=10.0)
        self.parallel_workers = max(1, int(parallel_workers))
        self.parallel_safe = bool(parallel_safe)
        if parallel_backend not in {"thread", "process"}:
            raise ValueError("parallel_backend must be 'thread' or 'process'")
        self.parallel_backend = parallel_backend
        self.persistent_workers = bool(persistent_workers)
        self._executor: ProcessPoolExecutor | ThreadPoolExecutor | None = None
        if (self.parallel_workers > 1 and self.parallel_backend == "thread"
                and not self.parallel_safe and not hasattr(problem, "evaluate_many")):
            raise ValueError("parallel evaluation requires parallel_safe=True or problem.evaluate_many")
        self.last_metadata: dict[str, Any] = {}

    def _create_executor(self):
        if self.parallel_backend == "process":
            return ProcessPoolExecutor(
                max_workers=self.parallel_workers,
                initializer=_initialize_process_problem,
                initargs=(self.problem,),
            )
        return ThreadPoolExecutor(max_workers=self.parallel_workers)

    def close(self) -> None:
        """Release persistent worker resources; safe to call repeatedly."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    @staticmethod
    def _merit(evaluation: Any, penalty: float) -> float:
        return _objective(evaluation) + penalty * _violation(evaluation)

    def build(self, initial_design: Any) -> tuple[QUBOModel, Any]:
        if self.parallel_workers <= 1 or hasattr(self.problem, "evaluate_many"):
            return self._build(initial_design, None)
        if self.persistent_workers:
            if self._executor is None:
                self._executor = self._create_executor()
            return self._build(initial_design, self._executor)
        with self._create_executor() as executor:
            return self._build(initial_design, executor)

    def _build(self, initial_design: Any, executor: Any | None) -> tuple[QUBOModel, Any]:
        build_started = perf_counter()
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
        def evaluate_many(values_list):
            missing = [tuple(values) for values in values_list if tuple(values) not in cache_values]
            if missing:
                batch = getattr(self.problem, "evaluate_many", None)
                if batch is not None:
                    results = batch([_make(initial_design, values) for values in missing])
                elif executor is not None:
                    evaluator = (_evaluate_process_design
                                 if self.parallel_backend == "process"
                                 else self.problem.evaluate)
                    results = list(executor.map(
                        evaluator,
                        [_make(initial_design, values) for values in missing],
                    ))
                else:
                    results = [self.problem.evaluate(_make(initial_design, values)) for values in missing]
                cache_values.update(zip(missing, results))
            return [cache_values[tuple(values)] for values in values_list]
        base_eval = evaluate(base)
        base_merit = self._merit(base_eval, self.penalty.value)
        screening_started = perf_counter()
        single_specs = []
        for member, domain in enumerate(domains):
            for state in domain:
                if state == base[member]:
                    continue
                changed = list(base); changed[member] = state
                single_specs.append((member, state, tuple(changed)))
        single_evaluations = evaluate_many([spec[2] for spec in single_specs])
        moves = []
        indicator_records = []
        for position, ((member, state, _), single_evaluation) in enumerate(
            zip(single_specs, single_evaluations)
        ):
            score = self._merit(single_evaluation, self.penalty.value)
            improvement = base_merit - score
            base_mass = float(getattr(base_eval, "mass", _objective(base_eval)))
            candidate_mass = float(
                getattr(single_evaluation, "mass", _objective(single_evaluation))
            )
            indicators = _engineering_indicators(
                single_evaluation, base_mass - candidate_mass, improvement
            )
            moves.append(DesignMove(member, state, score, improvement, indicators))
            indicator_records.append({"index": position, **indicators})
        ranked = select_candidates(
            _normalize_indicator_records(indicator_records), self.max_candidates
        )
        moves = [moves[candidate.index] for candidate in ranked]
        screening_seconds = perf_counter() - screening_started

        n_actions = len(moves)
        radius = min(self.trust_region.radius, max(1, len(base)))
        # Binary slack represents radius - selected_actions.
        slack_width = max(1, int(np.ceil(np.log2(radius + 1))))
        slack_weights = tuple(1 << i for i in range(slack_width))
        n = n_actions + slack_width
        linear, quadratic = np.zeros(n), np.zeros((n, n))
        linear[:n_actions] = [move.single_merit - base_merit for move in moves]

        pair_started = perf_counter()
        pair_specs = []
        for i, first in enumerate(moves):
            for j in range(i + 1, n_actions):
                second = moves[j]
                if first.member == second.member:
                    quadratic[i, j] += self.penalty.value
                    continue
                changed = list(base)
                changed[first.member], changed[second.member] = first.state, second.state
                pair_specs.append((i, j, first, second, tuple(changed)))
        pair_results = evaluate_many([spec[4] for spec in pair_specs])
        for (i, j, first, second, _), pair_evaluation in zip(pair_specs, pair_results):
            pair_merit = self._merit(pair_evaluation, self.penalty.value)
            quadratic[i, j] += pair_merit - first.single_merit - second.single_merit + base_merit
        pair_seconds = perf_counter() - pair_started

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
            "candidate_selection": "normalized_engineering_indicators_v1",
            "candidate_indicators": tuple(move.indicators for move in moves),
            "screened_moves": sum(len(d) - 1 for d in domains),
            "pair_evaluations": len(pair_specs), "fem_evaluations": len(cache_values),
            "cardinality_penalty": cardinality_penalty,
            "parallel_workers": self.parallel_workers,
            "parallel_backend": self.parallel_backend,
            "persistent_workers": self.persistent_workers,
            "screening_seconds": screening_seconds, "pair_seconds": pair_seconds,
            "build_seconds": perf_counter() - build_started,
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
