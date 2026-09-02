"""MILP backend for problems that provide an explicit linear formulation.

General FEM constraints are nonlinear in categorical section choices and are
therefore not silently approximated here.  A problem must explicitly provide
``build_milp(initial_design)`` (or a formulation may be supplied to the
backend), making any surrogate/linearization an auditable modelling decision.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import scipy.sparse as sp
from scipy.optimize import Bounds, LinearConstraint, milp

from ..topology import GroundStructure, equilibrium_matrix

from .base import (
    OptimizationResult, SolverLimits, design_values, evaluate_problem,
    evaluation_constraints, evaluation_feasible, evaluation_objective, make_design,
)


@dataclass(frozen=True)
class MILPFormulation:
    objective: Sequence[float]
    integrality: Sequence[int]
    bounds: Bounds
    constraints: LinearConstraint | Sequence[LinearConstraint] | None
    decoder: Callable[[np.ndarray], Any]
    options: dict[str, Any] | None = None
    metadata: Mapping[str, Any] | None = None


def build_truss_sizing_milp(
    ground_structure: GroundStructure,
    section_areas: Sequence[float] | Sequence[Sequence[float]],
    density: float,
    tensile_stress: float,
    compressive_stress: float | None = None,
    euler_capacities: Sequence[Sequence[float]] | None = None,
    state_indices: Sequence[int] | Sequence[Sequence[int]] | None = None,
) -> MILPFormulation:
    """Build an exact equilibrium/capacity MILP for discrete truss sizing.

    This is exact for the stated lower-bound plastic/static formulation only.
    Elastic compatibility and displacement constraints are intentionally not
    approximated.  ``MILPBackend`` always revalidates the selected design with
    the caller's FEM problem, which may therefore reject the MILP candidate.
    ``state_indices`` maps formulation columns to the caller's catalog indices;
    this permits an all-active sizing formulation without pretending that a
    zero-area OFF state is elastically stable.
    """
    scalar_inputs = (density, tensile_stress,
                     tensile_stress if compressive_stress is None else compressive_stress)
    if not all(math.isfinite(float(value)) and float(value) > 0 for value in scalar_inputs):
        raise ValueError("density and allowable stresses must be positive")
    if not ground_structure.load_cases:
        raise ValueError("ground structure requires at least one load case")
    member_count = ground_structure.n_member
    raw = list(section_areas)
    if not raw:
        raise ValueError("section_areas cannot be empty")
    if np.isscalar(raw[0]):
        areas = np.tile(np.asarray(raw, dtype=float), (member_count, 1))
    else:
        areas = np.asarray(raw, dtype=float)
    if areas.ndim != 2 or areas.shape[0] != member_count or areas.shape[1] < 1:
        raise ValueError("section_areas must be shared 1D or member-by-section 2D")
    if not np.all(np.isfinite(areas)) or np.any(areas < 0):
        raise ValueError("section areas must be finite and nonnegative")
    sections = areas.shape[1]
    cases = len(ground_structure.load_cases)
    compression = float(compressive_stress or tensile_stress)
    euler = None if euler_capacities is None else np.asarray(euler_capacities, dtype=float)
    if euler is not None and euler.shape != areas.shape:
        raise ValueError("euler_capacities must match section_areas")
    if euler is not None and (not np.all(np.isfinite(euler)) or np.any(euler < 0)):
        raise ValueError("euler_capacities must be finite and nonnegative")
    if state_indices is None:
        decoded_states = np.tile(np.arange(sections, dtype=int), (member_count, 1))
    else:
        raw_states = list(state_indices)
        if not raw_states:
            raise ValueError("state_indices cannot be empty")
        if np.isscalar(raw_states[0]):
            candidate_states = np.tile(np.asarray(raw_states, dtype=float), (member_count, 1))
        else:
            candidate_states = np.asarray(raw_states, dtype=float)
        if (candidate_states.shape != areas.shape
                or not np.all(np.isfinite(candidate_states))
                or np.any(candidate_states < 0)
                or not np.all(candidate_states == np.floor(candidate_states))):
            raise ValueError(
                "state_indices must match section_areas and contain nonnegative integers"
            )
        decoded_states = candidate_states.astype(int)

    # x = [y(member,section), force(case,member,section)]
    n_y = member_count * sections
    n_force = cases * n_y
    n_variables = n_y + n_force
    objective = np.zeros(n_variables)
    objective[:n_y] = (density * ground_structure.lengths()[:, None] * areas).ravel()
    integrality = np.zeros(n_variables, dtype=int); integrality[:n_y] = 1
    lower = np.r_[np.zeros(n_y), np.full(n_force, -np.inf)]
    upper = np.r_[np.ones(n_y), np.full(n_force, np.inf)]

    rows, lb, ub = [], [], []
    # Exactly one state, including an optional zero-area OFF state.
    one_hot = sp.lil_matrix((member_count, n_variables))
    for member in range(member_count):
        one_hot[member, member * sections:(member + 1) * sections] = 1.0
    rows.append(one_hot.tocsr()); lb.extend(np.ones(member_count)); ub.extend(np.ones(member_count))

    equilibrium, free, dof_index = equilibrium_matrix(ground_structure)
    for case_index, loads in enumerate(ground_structure.load_cases):
        matrix = sp.lil_matrix((len(free), n_variables))
        for section in range(sections):
            columns = n_y + case_index * n_y + np.arange(member_count) * sections + section
            matrix[:, columns] = equilibrium
        force_vector = np.zeros(len(free))
        for (node, dof), value in loads.items():
            global_dof = node * ground_structure.dim + dof
            if global_dof in dof_index:
                force_vector[dof_index[global_dof]] += value
        rows.append(matrix.tocsr()); lb.extend(force_vector); ub.extend(force_vector)

    capacity_rows = sp.lil_matrix((2 * cases * n_y, n_variables))
    row = 0
    for case_index in range(cases):
        for member in range(member_count):
            for section in range(sections):
                y = member * sections + section
                force = n_y + case_index * n_y + y
                tension_capacity = tensile_stress * areas[member, section]
                compression_capacity = compression * areas[member, section]
                if euler is not None:
                    compression_capacity = min(compression_capacity, euler[member, section])
                capacity_rows[row, force] = 1.0; capacity_rows[row, y] = -tension_capacity; row += 1
                capacity_rows[row, force] = -1.0; capacity_rows[row, y] = -compression_capacity; row += 1
    rows.append(capacity_rows.tocsr()); lb.extend(np.full(row, -np.inf)); ub.extend(np.zeros(row))
    constraint = LinearConstraint(sp.vstack(rows, format="csr"), np.asarray(lb), np.asarray(ub))

    def decode(x):
        selected = np.asarray(x[:n_y]).reshape(member_count, sections)
        return tuple(
            int(decoded_states[member, np.argmax(row_values)])
            for member, row_values in enumerate(selected)
        )

    return MILPFormulation(objective, integrality, Bounds(lower, upper), constraint, decode,
        metadata={"formulation_scope": "truss_equilibrium_and_section_capacity",
                  "elastic_compatibility": False, "displacement_constraints": False,
                  "global_optimum_for_scope": True, "members": member_count,
                  "load_cases": cases, "sections_per_member": sections,
                  "decoded_state_indices": decoded_states.tolist()})


class MILPBackend:
    def __init__(self, formulation: MILPFormulation | None = None,
                 fem_repair_backend: Any | None = None):
        self.formulation = formulation
        self.fem_repair_backend = fem_repair_backend

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
        if not np.all(np.isfinite(result.x)) or result.fun is None or not math.isfinite(float(result.fun)):
            return OptimizationResult(
                template, float("inf"), False, runtime=perf_counter()-started,
                backend="milp", status="failed",
                message="MILP returned non-finite candidate/objective",
                solver_metadata={"milp_status": int(result.status), "success": False},
            )
        decoded = formulation.decoder(np.asarray(result.x))
        design = make_design(problem, decoded, template) if isinstance(decoded, (tuple, list, np.ndarray)) else decoded
        evaluation = evaluate_problem(problem, design)
        def finite_or_none(value):
            if value is None:
                return None
            converted = float(value)
            return converted if math.isfinite(converted) else None
        metadata = {"milp_status": int(result.status), "success": bool(result.success),
                    "mip_gap": finite_or_none(getattr(result, "mip_gap", None)),
                    "mip_node_count": finite_or_none(getattr(result, "mip_node_count", None)),
                    "linear_objective": float(result.fun)}
        metadata.update(dict(formulation.metadata or {}))
        if not evaluation_feasible(evaluation) and self.fem_repair_backend is not None:
            anchor = template
            anchor_evaluation = evaluate_problem(problem, anchor)
            repair_start = anchor if evaluation_feasible(anchor_evaluation) else design
            repaired = self.fem_repair_backend.solve(problem, repair_start, limits)
            metadata.update({
                "milp_candidate": list(design_values(design)),
                "milp_candidate_fem_feasible": False,
                "fem_repair_performed": True,
                "fem_repair_backend": repaired.backend,
                "fem_repair_start": "initial_design" if repair_start is anchor else "milp_candidate",
                "fem_repair_feasible": repaired.feasible,
            })
            return OptimizationResult(
                repaired.design, repaired.objective, repaired.feasible, repaired.constraints,
                repaired.iterations, repaired.evaluations + 2,
                perf_counter() - started, "milp_fem_repair",
                "success" if repaired.feasible else "infeasible",
                ("MILP candidate failed common FEM; " + repaired.message), metadata,
                repaired.history, repaired.evaluation,
            )
        metadata.update({
            "milp_candidate_fem_feasible": evaluation_feasible(evaluation),
            "fem_repair_performed": False,
        })
        return OptimizationResult(design, evaluation_objective(evaluation), evaluation_feasible(evaluation),
            evaluation_constraints(evaluation), evaluations=1, runtime=perf_counter()-started,
            backend="milp", status="success" if result.success else "stopped",
            message=str(result.message), solver_metadata=metadata, evaluation=evaluation)
