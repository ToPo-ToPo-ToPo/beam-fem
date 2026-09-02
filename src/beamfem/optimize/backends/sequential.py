"""Sequential local-QUBO optimization with FEM acceptance checks."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any

from ..qubo.local import LocalQUBOBuilder
from .base import (
    HistoryEntry, OptimizationResult, SolverLimits, StopController, design_values, evaluate_problem,
    evaluation_constraints, evaluation_feasible, evaluation_objective,
    evaluation_violation, make_design,
)


class SequentialQUBOOptimizer:
    """Repeatedly rebuild a local QUBO, solve it, and verify with FEM."""

    def __init__(self, qubo_solver: Any, builder: LocalQUBOBuilder,
                 max_iterations: int = 20, improvement_tolerance: float = 1e-9,
                 checkpoint_path: str | Path | None = None, resume: bool = False,
                 cycle_window: int = 20, restoration_attempts: int = 3,
                 reference_objective: float | None = None):
        self.qubo_solver, self.builder = qubo_solver, builder
        self.max_iterations = int(max_iterations)
        self.improvement_tolerance = float(improvement_tolerance)
        if self.max_iterations < 1 or not math.isfinite(self.improvement_tolerance) or self.improvement_tolerance < 0:
            raise ValueError("max_iterations and improvement_tolerance must be finite and nonnegative")
        self.checkpoint_path = None if checkpoint_path is None else Path(checkpoint_path)
        self.resume = bool(resume)
        self.cycle_window = max(2, int(cycle_window))
        self.restoration_attempts = max(0, int(restoration_attempts))
        self.reference_objective = reference_objective

    @staticmethod
    def _canonical(payload: dict[str, Any]) -> str:
        return json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            allow_nan=False,
        )

    def _context_checksum(self, problem: Any, template: Any) -> str:
        if self.builder.domains is not None:
            domains = [list(map(int, domain)) for domain in self.builder.domains]
        elif hasattr(problem, "domains"):
            domains = [list(map(int, domain)) for domain in problem.domains]
        elif hasattr(problem, "catalogs"):
            domains = [
                [
                    {
                        "name": str(getattr(option, "name", index)),
                        "active": bool(getattr(option, "active", True)),
                        "section": {
                            key: getattr(getattr(option, "section", None), key, None)
                            for key in ("A", "Iy", "Iz", "J")
                        },
                        "material": {
                            key: getattr(getattr(option, "material", None), key, None)
                            for key in ("E", "nu", "rho")
                        },
                    }
                    for index, option in enumerate(catalog)
                ]
                for catalog in problem.catalogs
            ]
        else:
            domains = []
        solver_settings = {
            key: value for key, value in vars(self.qubo_solver).items()
            if value is None or isinstance(value, (str, int, float, bool))
        }
        solver_components = {
            key: f"{type(value).__module__}.{type(value).__qualname__}"
            for key, value in vars(self.qubo_solver).items()
            if value is not None and not isinstance(value, (str, int, float, bool))
        }
        model = getattr(problem, "model", None)
        nodes = getattr(model, "nodes", None)
        explicit_fingerprint = getattr(problem, "checkpoint_fingerprint", None)
        if callable(explicit_fingerprint):
            explicit_fingerprint = explicit_fingerprint()
        problem_signature = {
            "nodes": None if nodes is None else nodes.tolist(),
            "elements": [
                {
                    "type": type(element).__qualname__,
                    "n1": int(element.n1),
                    "n2": int(element.n2),
                }
                for element in getattr(model, "elements", ())
            ],
            "load_cases": [
                {
                    "name": case.name,
                    "loads": sorted(
                        [int(node), int(dof), float(value)]
                        for (node, dof), value in case.loads.items()
                    ),
                }
                for case in getattr(problem, "load_cases", ())
            ],
            "load_combinations": [
                {"name": combo.name, "factors": sorted(combo.factors.items())}
                for combo in getattr(problem, "load_combinations", ())
            ],
            "constraints": [repr(item) for item in getattr(problem, "constraints", ())],
            "objective": repr(getattr(problem, "objective", None)),
            "explicit_fingerprint": explicit_fingerprint,
        }
        context = {
            "problem_type": f"{type(problem).__module__}.{type(problem).__qualname__}",
            "initial_design": list(design_values(template)),
            "domains": domains,
            "solver_type": f"{type(self.qubo_solver).__module__}.{type(self.qubo_solver).__qualname__}",
            "solver_settings": solver_settings,
            "solver_components": solver_components,
            "problem_signature": problem_signature,
            "builder": {
                "max_candidates": self.builder.max_candidates,
                "parallel_workers": self.builder.parallel_workers,
                "parallel_backend": self.builder.parallel_backend,
                "persistent_workers": self.builder.persistent_workers,
                "penalty": {
                    key: getattr(self.builder.penalty, key)
                    for key in (
                        "value", "minimum", "maximum", "increase", "decrease",
                        "target_feasible_rate",
                    )
                },
                "trust_initial_radius": self.builder.trust_region.radius,
                "trust_minimum": self.builder.trust_region.minimum,
                "trust_maximum": self.builder.trust_region.maximum,
            },
        }
        return hashlib.sha256(self._canonical(context).encode("utf-8")).hexdigest()

    def _write_checkpoint(self, current, iteration, seen, context_checksum) -> None:
        if self.checkpoint_path is None:
            return
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.checkpoint_path.with_suffix(self.checkpoint_path.suffix + ".tmp")
        payload = {"schema_version": 2, "current": list(current), "iteration": iteration,
                   "seen": [list(values) for values in seen[-self.cycle_window:]],
                   "penalty": self.builder.penalty.value,
                   "trust_radius": self.builder.trust_region.radius,
                   "context_checksum": context_checksum}
        payload["checksum"] = hashlib.sha256(
            self._canonical(payload).encode("utf-8")
        ).hexdigest()
        temporary.write_text(
            json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
        )
        temporary.replace(self.checkpoint_path)

    def _load_checkpoint(self, problem: Any, template: Any, context_checksum: str):
        if not self.resume or self.checkpoint_path is None or not self.checkpoint_path.exists():
            return None
        payload = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 2:
            raise ValueError("unsupported optimizer checkpoint schema")
        supplied_checksum = str(payload.pop("checksum", ""))
        actual_checksum = hashlib.sha256(
            self._canonical(payload).encode("utf-8")
        ).hexdigest()
        if not hmac.compare_digest(supplied_checksum, actual_checksum):
            raise ValueError("optimizer checkpoint checksum mismatch")
        if not hmac.compare_digest(str(payload.get("context_checksum", "")), context_checksum):
            raise ValueError("optimizer checkpoint problem/solver context mismatch")
        current = tuple(int(v) for v in payload["current"])
        seen = [tuple(int(v) for v in values) for values in payload.get("seen", [])]
        iteration = int(payload["iteration"])
        penalty = float(payload["penalty"])
        radius = int(payload["trust_radius"])
        if iteration < 0 or not math.isfinite(penalty) or penalty <= 0:
            raise ValueError("invalid optimizer checkpoint iteration or penalty")
        if not self.builder.trust_region.minimum <= radius <= self.builder.trust_region.maximum:
            raise ValueError("invalid optimizer checkpoint trust radius")
        domains = self.builder.domains
        if domains is None and hasattr(problem, "domains"):
            domains = problem.domains
        if domains is None and hasattr(problem, "catalogs"):
            domains = tuple(range(len(catalog)) for catalog in problem.catalogs)
        if domains is not None:
            normalized = tuple(tuple(int(value) for value in domain) for domain in domains)
            for values in (current, *seen):
                if len(values) != len(normalized) or any(
                    value not in normalized[index] for index, value in enumerate(values)
                ):
                    raise ValueError("optimizer checkpoint design is outside problem domains")
        # Mutate adaptive state only after every integrity check passed.
        self.builder.penalty.value = penalty
        self.builder.trust_region.radius = radius
        return current, iteration, seen

    def _merit(self, evaluation: Any) -> float:
        return (evaluation_objective(evaluation) +
                self.builder.penalty.value * evaluation_violation(evaluation))

    def solve(self, problem: Any, initial_design: Any | None = None,
              limits: SolverLimits | None = None) -> OptimizationResult:
        started = perf_counter()
        template = initial_design if initial_design is not None else problem.initial_design
        context_checksum = self._context_checksum(problem, template)
        resumed = self._load_checkpoint(problem, template, context_checksum)
        current = resumed[0] if resumed else design_values(template)
        start_iteration = resumed[1] if resumed else 0
        seen = resumed[2] if resumed else [current]
        current_eval = evaluate_problem(problem, make_design(problem, current, template))
        history = [HistoryEntry(0, evaluation_objective(current_eval),
                                evaluation_feasible(current_eval), current)]
        total_fem, message, restoration_count = 1, "maximum iterations reached", 0
        status = "success"
        stop = StopController(limits)
        qubo_history: list[dict[str, Any]] = []
        build_seconds = solve_seconds = validation_seconds = 0.0
        iteration_cap = min(self.max_iterations,
                            limits.max_iterations if limits and limits.max_iterations else self.max_iterations)
        for iteration in range(start_iteration + 1, start_iteration + iteration_cap + 1):
            reason = stop.reached(total_fem, iteration - start_iteration - 1)
            if reason:
                message, status = reason, "stopped"
                break
            model, decoder = self.builder.build(make_design(problem, current, template))
            build_seconds += float(self.builder.last_metadata.get("build_seconds", 0.0))
            total_fem += int(self.builder.last_metadata["fem_evaluations"])
            solve_started = perf_counter()
            solution = self.qubo_solver.solve_qubo(model)
            iteration_solve_seconds = perf_counter() - solve_started
            solve_seconds += iteration_solve_seconds
            candidate = tuple(decoder(solution.bits))
            validation_started = perf_counter()
            candidate_eval = evaluate_problem(problem, make_design(problem, candidate, template))
            iteration_validation_seconds = perf_counter() - validation_started
            validation_seconds += iteration_validation_seconds
            total_fem += 1
            base_bits = self.builder.last_metadata["base_bits"]
            predicted = model.energy(base_bits) - solution.energy
            actual = self._merit(current_eval) - self._merit(candidate_eval)
            rho, accepted = self.builder.trust_region.update(predicted, actual)
            if evaluation_feasible(current_eval) and not evaluation_feasible(candidate_eval):
                accepted = False
            qubo_history.append({
                "iteration": iteration,
                "energy": solution.energy,
                "base_energy": model.energy(base_bits),
                "bits": solution.bits,
                "predicted_improvement": predicted,
                "fem_objective": evaluation_objective(candidate_eval),
                "fem_feasible": evaluation_feasible(candidate_eval),
                "accepted": accepted and actual > self.improvement_tolerance,
                "build_seconds": self.builder.last_metadata.get("build_seconds"),
                "qubo_solve_seconds": iteration_solve_seconds,
                "fem_validation_seconds": iteration_validation_seconds,
                "solver": dict(solution.metadata or {}),
            })
            self.builder.penalty.update(1.0 if evaluation_feasible(candidate_eval) else 0.0)
            if candidate in seen[-self.cycle_window:] and candidate != current:
                message = "design cycle detected"
                self._write_checkpoint(current, iteration, seen, context_checksum)
                break
            if accepted and actual > self.improvement_tolerance:
                current, current_eval = candidate, candidate_eval
                seen.append(current)
                restoration_count = 0
                history.append(HistoryEntry(iteration, evaluation_objective(current_eval),
                    evaluation_feasible(current_eval), current,
                    {"predicted_improvement": predicted, "actual_improvement": actual, "rho": rho}))
                self._write_checkpoint(current, iteration, seen, context_checksum)
            elif not evaluation_feasible(candidate_eval) and restoration_count < self.restoration_attempts:
                restoration_count += 1
                message = "feasibility restoration exhausted"
                self._write_checkpoint(current, iteration, seen, context_checksum)
                continue
            else:
                message = "trust-region step rejected or no improvement"
                self._write_checkpoint(current, iteration, seen, context_checksum)
                break
            reason = stop.reached(total_fem, iteration - start_iteration)
            if reason:
                message, status = reason, "stopped"
                break
        objective = evaluation_objective(current_eval)
        gap = None
        if self.reference_objective is not None:
            gap = (objective - self.reference_objective) / max(abs(self.reference_objective), 1e-12)
        return OptimizationResult(make_design(problem, current, template), objective,
            evaluation_feasible(current_eval), evaluation_constraints(current_eval),
            len(history)-1, total_fem, perf_counter()-started, "sequential_qubo", status,
            message, {"qubo_solver": type(self.qubo_solver).__name__,
                      "qubo_energy": qubo_history[-1]["energy"] if qubo_history else None,
                      "qubo_history": tuple(qubo_history),
                      "trust_region_history": tuple(self.builder.trust_region.history),
                      "penalty_history": tuple(self.builder.penalty.history),
                      "optimality_gap": gap,
                      "timing": {"qubo_build_seconds": build_seconds,
                                 "qubo_solve_seconds": solve_seconds,
                                 "fem_validation_seconds": validation_seconds,
                                 "total_seconds": perf_counter() - started},
                      "resumed": resumed is not None,
                      "checkpoint": None if self.checkpoint_path is None else str(self.checkpoint_path)},
            tuple(history), current_eval)
