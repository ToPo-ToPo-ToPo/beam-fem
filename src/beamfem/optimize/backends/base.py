"""Common interfaces and result types for discrete optimization backends."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import inf
from time import perf_counter
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable


@runtime_checkable
class DiscreteProblemProtocol(Protocol):
    """Minimal interface consumed by the solver backends.

    Concrete problems may expose either ``evaluate`` or the legacy-friendly
    ``evaluate_design`` method.  The adapter below deliberately avoids a
    dependency on the concrete structural problem classes.
    """

    initial_design: Any

    def evaluate(self, design: Any) -> Any: ...


@dataclass(frozen=True)
class SolverLimits:
    max_evaluations: int | None = None
    max_iterations: int | None = None
    time_limit: float | None = None


@dataclass(frozen=True)
class HistoryEntry:
    iteration: int
    objective: float
    feasible: bool
    design: tuple[int, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OptimizationResult:
    design: Any
    objective: float
    feasible: bool
    constraints: Any = field(default_factory=dict)
    iterations: int = 0
    evaluations: int = 0
    runtime: float = 0.0
    backend: str = "unknown"
    status: str = "success"
    message: str = ""
    solver_metadata: Mapping[str, Any] = field(default_factory=dict)
    history: tuple[HistoryEntry, ...] = ()
    evaluation: Any = None

    def as_dict(self) -> dict[str, Any]:
        """Return a stable summary without expanding heavy FEM matrices."""
        evaluation = self.evaluation
        if evaluation is not None and hasattr(evaluation, "as_dict"):
            evaluation = evaluation.as_dict()
        constraints = []
        source = self.constraints.values() if isinstance(self.constraints, Mapping) else self.constraints
        try:
            for item in source:
                constraints.append(item.as_dict() if hasattr(item, "as_dict") else item)
        except TypeError:
            constraints = self.constraints
        return {
            "design": list(design_values(self.design)),
            "objective": self.objective,
            "feasible": self.feasible,
            "constraints": constraints,
            "iterations": self.iterations,
            "evaluations": self.evaluations,
            "runtime": self.runtime,
            "backend": self.backend,
            "status": self.status,
            "message": self.message,
            "solver_metadata": dict(self.solver_metadata),
            "history": [
                {
                    "iteration": entry.iteration,
                    "objective": entry.objective,
                    "feasible": entry.feasible,
                    "design": list(entry.design),
                    "metadata": dict(entry.metadata),
                }
                for entry in self.history
            ],
            "evaluation": evaluation,
        }


class DiscreteOptimizerBackend(Protocol):
    def solve(
        self,
        problem: DiscreteProblemProtocol,
        initial_design: Any | None = None,
        limits: SolverLimits | None = None,
    ) -> OptimizationResult: ...


def design_values(design: Any) -> tuple[int, ...]:
    """Return a stable tuple from a sequence or a DesignState-like object."""
    for name in ("choices", "states", "indices", "values"):
        if hasattr(design, name):
            return tuple(int(v) for v in getattr(design, name))
    if isinstance(design, Mapping):
        return tuple(int(v) for v in design.values())
    return tuple(int(v) for v in design)


def make_design(problem: Any, values: Sequence[int], template: Any | None = None) -> Any:
    """Rebuild a problem-specific DesignState when possible."""
    values_tuple = tuple(int(v) for v in values)
    template = template if template is not None else getattr(problem, "initial_design", None)
    if template is None or isinstance(template, (tuple, list)):
        return values_tuple
    cls = type(template)
    for kwargs in ({"choices": values_tuple}, {"states": values_tuple},
                   {"indices": values_tuple}, {"values": values_tuple}):
        try:
            return cls(**kwargs)
        except (TypeError, ValueError):
            pass
    try:
        return replace(template, states=values_tuple)
    except (TypeError, ValueError):
        return values_tuple


def evaluate_problem(problem: Any, design: Any) -> Any:
    evaluator = getattr(problem, "evaluate", None)
    if evaluator is None:
        evaluator = getattr(problem, "evaluate_design", None)
    if evaluator is None:
        raise TypeError("problem must define evaluate(design) or evaluate_design(design)")
    return evaluator(design)


def evaluation_objective(evaluation: Any) -> float:
    """Extract a scalar objective from common evaluation result shapes."""
    if isinstance(evaluation, tuple) and evaluation:
        return float(evaluation[0])
    for name in ("objective", "objective_value", "score", "mass"):
        value = getattr(evaluation, name, None)
        if value is None and isinstance(evaluation, Mapping):
            value = evaluation.get(name)
        if value is not None:
            if hasattr(value, "value"):
                value = value.value
            if isinstance(value, Mapping):
                value = value.get("value", value.get("total"))
            return float(value)
    raise TypeError("evaluation does not expose objective, score, or mass")


def evaluation_feasible(evaluation: Any) -> bool:
    if isinstance(evaluation, tuple) and len(evaluation) >= 2:
        constraints = evaluation[1]
        try:
            return all(float(v) <= 0.0 for v in constraints)
        except TypeError:
            pass
    value = getattr(evaluation, "feasible", None)
    if value is None and isinstance(evaluation, Mapping):
        value = evaluation.get("feasible")
    return bool(value) if value is not None else True


def evaluation_constraints(evaluation: Any) -> Any:
    if isinstance(evaluation, tuple) and len(evaluation) >= 2:
        return evaluation[1]
    value = getattr(evaluation, "constraints", None)
    if value is None and isinstance(evaluation, Mapping):
        value = evaluation.get("constraints", {})
    return {} if value is None else value


def evaluation_violation(evaluation: Any) -> float:
    value = getattr(evaluation, "total_violation", None)
    if value is None and isinstance(evaluation, Mapping):
        value = evaluation.get("total_violation")
    if value is not None:
        return max(0.0, float(value))
    constraints = evaluation_constraints(evaluation)
    if isinstance(constraints, Mapping):
        constraints = constraints.values()
    total = 0.0
    try:
        for item in constraints:
            if hasattr(item, "violation"):
                total += max(0.0, float(item.violation))
            elif hasattr(item, "satisfied") and hasattr(item, "utilization"):
                total += 0.0 if item.satisfied else max(0.0, float(item.utilization) - 1.0)
            elif isinstance(item, Mapping):
                total += max(0.0, float(item.get("violation", item.get("value", 0.0))))
            else:
                total += max(0.0, float(item))
    except TypeError:
        return 0.0 if evaluation_feasible(evaluation) else inf
    return total


class EvaluationCache:
    """Memoizing evaluator with design reconstruction and accounting."""

    def __init__(self, problem: Any, template: Any | None = None):
        self.problem = problem
        self.template = template if template is not None else getattr(problem, "initial_design", None)
        self.cache: dict[tuple[int, ...], Any] = {}
        self.evaluations = 0

    def __call__(self, values: Sequence[int]) -> Any:
        key = tuple(int(v) for v in values)
        if key not in self.cache:
            self.cache[key] = evaluate_problem(self.problem, make_design(self.problem, key, self.template))
            self.evaluations += 1
        return self.cache[key]


class StopController:
    def __init__(self, limits: SolverLimits | None):
        self.limits = limits or SolverLimits()
        self.started = perf_counter()

    def reached(self, evaluations: int, iterations: int) -> str | None:
        if self.limits.max_evaluations is not None and evaluations >= self.limits.max_evaluations:
            return "maximum evaluations reached"
        if self.limits.max_iterations is not None and iterations >= self.limits.max_iterations:
            return "maximum iterations reached"
        if self.limits.time_limit is not None and perf_counter() - self.started >= self.limits.time_limit:
            return "time limit reached"
        return None
