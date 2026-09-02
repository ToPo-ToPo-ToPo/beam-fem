"""Common interfaces and result types for discrete optimization backends."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import inf, isfinite
import os
import sys
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
    memory_limit_mb: float | None = None

    def __post_init__(self) -> None:
        if self.max_evaluations is not None and self.max_evaluations < 1:
            raise ValueError("max_evaluations must be positive")
        if self.max_iterations is not None and self.max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if self.time_limit is not None and (
            not isfinite(float(self.time_limit)) or self.time_limit <= 0
        ):
            raise ValueError("time_limit must be finite and positive")
        if self.memory_limit_mb is not None and (
            not isfinite(float(self.memory_limit_mb)) or self.memory_limit_mb <= 0
        ):
            raise ValueError("memory_limit_mb must be finite and positive")


def peak_resident_memory_bytes() -> int | None:
    """Return process peak RSS using only the Python standard library."""

    if os.name == "nt":  # pragma: no cover - exercised by Windows CI
        try:
            import ctypes
            from ctypes import wintypes

            class _Counters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = _Counters()
            counters.cb = ctypes.sizeof(counters)
            process = ctypes.windll.kernel32.GetCurrentProcess()
            ok = ctypes.windll.psapi.GetProcessMemoryInfo(
                process, ctypes.byref(counters), counters.cb
            )
            return int(counters.PeakWorkingSetSize) if ok else None
        except (AttributeError, OSError, TypeError):
            return None
    try:
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value if sys.platform == "darwin" else value * 1024
    except (ImportError, OSError, ValueError):  # pragma: no cover
        return None


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

    def __post_init__(self) -> None:
        metadata = dict(self.solver_metadata)
        existing = dict(metadata.get("normalized_work", {}))
        qaoa_calls = metadata.get("cost_function_evaluations")
        qubo_calls = metadata.get("qubo_energy_evaluations", qaoa_calls)
        work = {
            "fem_evaluations": int(self.evaluations),
            "optimizer_iterations": int(self.iterations),
            "classical_objective_evaluations": (
                None if qubo_calls is None else int(qubo_calls)
            ),
            "quantum_shots": int(metadata.get("shots") or 0),
            "quantum_circuit_evaluations": (
                None if qaoa_calls is None else int(qaoa_calls)
            ),
            "normalized_fem_equivalents": int(self.evaluations),
            "budget_dimensions_are_not_interchangeable": True,
        }
        work.update(existing)
        metadata["normalized_work"] = work
        object.__setattr__(self, "solver_metadata", metadata)

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
        objective = self.objective
        if isinstance(objective, float) and not isfinite(objective):
            objective = None
        return {
            "design": list(design_values(self.design)),
            "objective": objective,
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
        result = float(evaluation[0])
        if not isfinite(result):
            raise ValueError("evaluation objective must be finite")
        return result
    for name in ("objective", "objective_value", "score", "mass"):
        value = getattr(evaluation, name, None)
        if value is None and isinstance(evaluation, Mapping):
            value = evaluation.get(name)
        if value is not None:
            if hasattr(value, "value"):
                value = value.value
            if isinstance(value, Mapping):
                value = value.get("value", value.get("total"))
            result = float(value)
            if not isfinite(result):
                raise ValueError("evaluation objective must be finite")
            return result
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
    if isinstance(value, float) and not isfinite(value):
        raise ValueError("evaluation feasibility must be finite")
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
        converted = float(value)
        if not isfinite(converted):
            raise ValueError("evaluation violation must be finite")
        return max(0.0, converted)
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
                converted = float(item)
                if not isfinite(converted):
                    raise ValueError("constraint violation must be finite")
                total += max(0.0, converted)
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
        if self.limits.memory_limit_mb is not None:
            used = peak_resident_memory_bytes()
            if used is None:
                return "memory limit cannot be verified on this platform"
            if used >= float(self.limits.memory_limit_mb) * 1024.0 * 1024.0:
                return "memory limit reached"
        return None
