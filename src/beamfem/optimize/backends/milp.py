"""MILP backend for problems that provide an explicit linear formulation.

General FEM constraints are nonlinear in categorical section choices and are
therefore not silently approximated here.  A problem must explicitly provide
``build_milp(initial_design)`` (or a formulation may be supplied to the
backend), making any surrogate/linearization an auditable modelling decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from .base import (
    OptimizationResult, SolverLimits, evaluate_problem, evaluation_constraints,
    evaluation_feasible, evaluation_objective, make_design,
)


@dataclass(frozen=True)
class MILPFormulation:
    objective: Sequence[float]
    integrality: Sequence[int]
    bounds: Bounds
    constraints: LinearConstraint | Sequence[LinearConstraint] | None
    decoder: Callable[[np.ndarray], Any]
    options: dict[str, Any] | None = None


class MILPBackend:
    def __init__(self, formulation: MILPFormulation | None = None):
        self.formulation = formulation

    def _formulation(self, problem: Any, initial_design: Any) -> MILPFormulation:
        if self.formulation is not None:
            return self.formulation
        builder = getattr(problem, "build_milp", None)
        if builder is None:
            raise NotImplementedError(
                "MILPBackend requires an explicit linear formulation. General FEM "
                "constraints are not automatically linearized; use ExactBackend for "
                "small reference problems or implement problem.build_milp()."
            )
        return builder(initial_design)

    def solve(self, problem: Any, initial_design: Any | None = None,
              limits: SolverLimits | None = None) -> OptimizationResult:
        started = perf_counter()
        template = initial_design if initial_design is not None else problem.initial_design
        formulation = self._formulation(problem, template)
        options = dict(formulation.options or {})
        if limits and limits.time_limit is not None:
            options.setdefault("time_limit", limits.time_limit)
        result = milp(c=np.asarray(formulation.objective, dtype=float),
                      integrality=np.asarray(formulation.integrality, dtype=int),
                      bounds=formulation.bounds, constraints=formulation.constraints,
                      options=options or None)
        if result.x is None:
            return OptimizationResult(template, float("inf"), False, runtime=perf_counter()-started,
                backend="milp", status="failed", message=str(result.message),
                solver_metadata={"milp_status": int(result.status), "success": bool(result.success)})
        decoded = formulation.decoder(np.asarray(result.x))
        design = make_design(problem, decoded, template) if isinstance(decoded, (tuple, list, np.ndarray)) else decoded
        evaluation = evaluate_problem(problem, design)
        metadata = {"milp_status": int(result.status), "success": bool(result.success),
                    "mip_gap": getattr(result, "mip_gap", None),
                    "mip_node_count": getattr(result, "mip_node_count", None),
                    "linear_objective": float(result.fun)}
        return OptimizationResult(design, evaluation_objective(evaluation), evaluation_feasible(evaluation),
            evaluation_constraints(evaluation), evaluations=1, runtime=perf_counter()-started,
            backend="milp", status="success" if result.success else "stopped",
            message=str(result.message), solver_metadata=metadata, evaluation=evaluation)
