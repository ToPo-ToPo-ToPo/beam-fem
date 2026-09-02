"""Incremental material/geometric nonlinear analysis of 2D/3D trusses."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Mapping, Sequence
import warnings

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .model import DOF_PER_NODE, Model, TrussElement
from .nonlinear_material import (
    UniaxialMaterialModel,
    UniaxialMaterialResponse,
    UniaxialMaterialState,
)


class NonlinearConvergenceError(RuntimeError):
    """Raised when an increment cannot converge above the minimum step size."""

    def __init__(self, message: str, *, load_factor: float, residual_norm: float):
        self.load_factor = float(load_factor)
        self.residual_norm = float(residual_norm)
        super().__init__(message)


@dataclass(frozen=True)
class NonlinearElementResult:
    element: int
    strain: float
    stress: float
    axial_force: float
    tangent_modulus: float
    yielded: bool
    current_length: float
    state: UniaxialMaterialState


@dataclass(frozen=True)
class NonlinearStepResult:
    load_factor: float
    requested_load_factor: float
    increment: float
    iterations: int
    residual_norm: float
    cutbacks: int
    displacement_norm: float
    dissipated_energy: float
    yielded_elements: tuple[int, ...]
    u: np.ndarray
    reactions: np.ndarray
    elements: tuple[NonlinearElementResult, ...]

    def node_disp(self, node: int) -> np.ndarray:
        start = int(node) * DOF_PER_NODE
        return self.u[start:start + DOF_PER_NODE]


@dataclass(frozen=True)
class CollapseEvent:
    sequence: int
    load_factor: float
    event: str
    elements: tuple[int, ...]
    detail: str


@dataclass(frozen=True)
class LimitStateReport:
    maximum_absolute_load_factor: float
    last_converged_load_factor: float
    first_yield_load_factor: float | None
    yielded_elements: tuple[int, ...]
    collapse_detected: bool
    reason: str | None
    failed_load_factor: float | None = None
    failure_residual_norm: float | None = None
    progressive_collapse_sequence: tuple[CollapseEvent, ...] = ()


@dataclass(frozen=True)
class NonlinearTrussResult:
    converged: bool
    u: np.ndarray
    reactions: np.ndarray
    element_states: tuple[UniaxialMaterialState, ...]
    history: tuple[NonlinearStepResult, ...]
    limit_state: LimitStateReport
    diagnostic: str | None = None

    @property
    def feasible(self) -> bool:
        return self.converged and not self.limit_state.collapse_detected

    @property
    def dissipated_energy(self) -> float:
        return self.history[-1].dissipated_energy if self.history else 0.0

    @property
    def residual_displacement(self) -> np.ndarray | None:
        if not self.history or abs(self.history[-1].load_factor) > 1.0e-12:
            return None
        return self.u.copy()

    def node_disp(self, node: int) -> np.ndarray:
        start = int(node) * DOF_PER_NODE
        return self.u[start:start + DOF_PER_NODE]

    def as_dict(self) -> dict[str, Any]:
        return {
            "converged": self.converged,
            "feasible": self.feasible,
            "diagnostic": self.diagnostic,
            "dissipated_energy": self.dissipated_energy,
            "displacement": self.u.tolist(),
            "reactions": self.reactions.tolist(),
            "element_states": [
                {
                    "strain": state.strain,
                    "stress": state.stress,
                    "plastic_strain": state.plastic_strain,
                    "equivalent_plastic_strain": state.equivalent_plastic_strain,
                    "dissipated_energy_density": state.dissipated_energy_density,
                }
                for state in self.element_states
            ],
            "limit_state": {
                "maximum_absolute_load_factor": self.limit_state.maximum_absolute_load_factor,
                "last_converged_load_factor": self.limit_state.last_converged_load_factor,
                "first_yield_load_factor": self.limit_state.first_yield_load_factor,
                "yielded_elements": list(self.limit_state.yielded_elements),
                "collapse_detected": self.limit_state.collapse_detected,
                "reason": self.limit_state.reason,
                "failed_load_factor": self.limit_state.failed_load_factor,
                "failure_residual_norm": self.limit_state.failure_residual_norm,
                "progressive_collapse_sequence": [
                    {
                        "sequence": event.sequence,
                        "load_factor": event.load_factor,
                        "event": event.event,
                        "elements": list(event.elements),
                        "detail": event.detail,
                    }
                    for event in self.limit_state.progressive_collapse_sequence
                ],
            },
            "steps": [
                {
                    "load_factor": step.load_factor,
                    "requested_load_factor": step.requested_load_factor,
                    "increment": step.increment,
                    "iterations": step.iterations,
                    "residual_norm": step.residual_norm,
                    "cutbacks": step.cutbacks,
                    "displacement_norm": step.displacement_norm,
                    "dissipated_energy": step.dissipated_energy,
                    "yielded_elements": list(step.yielded_elements),
                    "elements": [
                        {
                            "element": element.element,
                            "strain": element.strain,
                            "stress": element.stress,
                            "axial_force": element.axial_force,
                            "tangent_modulus": element.tangent_modulus,
                            "yielded": element.yielded,
                            "plastic_strain": element.state.plastic_strain,
                            "equivalent_plastic_strain": (
                                element.state.equivalent_plastic_strain
                            ),
                        }
                        for element in step.elements
                    ],
                }
                for step in self.history
            ],
        }


@dataclass(frozen=True)
class _AssemblyResult:
    tangent: sp.csr_matrix
    internal: np.ndarray
    states: tuple[UniaxialMaterialState, ...]
    elements: tuple[NonlinearElementResult, ...]


def _material_sequence(
    materials: UniaxialMaterialModel | Mapping[int, UniaxialMaterialModel]
    | Sequence[UniaxialMaterialModel],
    count: int,
) -> tuple[UniaxialMaterialModel, ...]:
    if isinstance(materials, Mapping):
        missing = set(range(count)) - set(materials)
        extra = set(materials) - set(range(count))
        if missing or extra:
            raise ValueError(
                f"nonlinear material mapping mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
            )
        result = tuple(materials[index] for index in range(count))
    elif isinstance(materials, Sequence):
        result = tuple(materials)
        if len(result) != count:
            raise ValueError("one nonlinear material model is required per truss element")
    else:
        result = (materials,) * count
    for material in result:
        if not isinstance(material, UniaxialMaterialModel):
            raise TypeError("materials must implement UniaxialMaterialModel")
    return result


def _translation_to_full(values: np.ndarray, node_count: int) -> np.ndarray:
    full = np.zeros(node_count * DOF_PER_NODE, dtype=float)
    for node in range(node_count):
        full[node * DOF_PER_NODE:node * DOF_PER_NODE + 3] = values[node * 3:node * 3 + 3]
    return full


def _validate_model(model: Model) -> None:
    if model.shells or model.quad_shells or not model.elements:
        raise ValueError("nonlinear truss analysis requires a nonempty truss-only model")
    if any(not isinstance(element, TrussElement) for element in model.elements):
        raise ValueError("nonlinear truss analysis does not accept frame elements")
    if model.nodes.ndim != 2 or model.nodes.shape[1] != 3 or not np.all(np.isfinite(model.nodes)):
        raise ValueError("model nodes must be finite three-dimensional coordinates")


def _reference_force(model: Model) -> np.ndarray:
    force = np.zeros(model.n_nodes * 3, dtype=float)
    for (node, dof), value in model.nodal_loads.items():
        if not math.isfinite(float(value)):
            raise ValueError("nodal loads must be finite")
        if dof >= 3 and value != 0.0:
            raise ValueError("truss nonlinear analysis does not accept applied moments")
        if dof < 3:
            force[node * 3 + dof] += float(value)
    return force


def _assemble(
    model: Model,
    displacement: np.ndarray,
    material_models: tuple[UniaxialMaterialModel, ...],
    committed_states: tuple[UniaxialMaterialState, ...],
    geometric_nonlinear: bool,
) -> _AssemblyResult:
    size = model.n_nodes * 3
    internal = np.zeros(size, dtype=float)
    rows: list[int] = []
    columns: list[int] = []
    data: list[float] = []
    states = []
    element_results = []
    identity = np.eye(3)

    for index, (element, material, committed) in enumerate(
        zip(model.elements, material_models, committed_states)
    ):
        first = np.asarray(model.nodes[element.n1], dtype=float)
        second = np.asarray(model.nodes[element.n2], dtype=float)
        initial_delta = second - first
        initial_length = float(np.linalg.norm(initial_delta))
        area = float(element.sec.A)
        if not math.isfinite(initial_length) or initial_length <= 0.0:
            raise ValueError(f"truss element {index} has zero or non-finite length")
        if not math.isfinite(area) or area <= 0.0:
            raise ValueError(f"truss element {index} has non-positive area")
        first_dofs = np.arange(element.n1 * 3, element.n1 * 3 + 3)
        second_dofs = np.arange(element.n2 * 3, element.n2 * 3 + 3)
        relative = displacement[second_dofs] - displacement[first_dofs]

        if geometric_nonlinear:
            current_delta = initial_delta + relative
            current_length = float(np.linalg.norm(current_delta))
            if not math.isfinite(current_length) or current_length <= 1.0e-14 * initial_length:
                raise ArithmeticError(f"truss element {index} collapsed to zero current length")
            direction = current_delta / current_length
            strain = (current_length - initial_length) / initial_length
        else:
            current_length = initial_length
            direction = initial_delta / initial_length
            strain = float(direction @ relative) / initial_length

        response: UniaxialMaterialResponse = material.update(strain, committed)
        if not all(math.isfinite(value) for value in (response.stress, response.tangent)):
            raise ArithmeticError(f"material {index} returned a non-finite response")
        axial_force = area * response.stress
        internal[first_dofs] -= axial_force * direction
        internal[second_dofs] += axial_force * direction

        material_stiffness = area * response.tangent / initial_length * np.outer(direction, direction)
        geometric_stiffness = np.zeros((3, 3))
        if geometric_nonlinear:
            geometric_stiffness = axial_force / current_length * (
                identity - np.outer(direction, direction)
            )
        block = material_stiffness + geometric_stiffness
        local = np.block([[block, -block], [-block, block]])
        dofs = np.concatenate((first_dofs, second_dofs))
        for local_row, global_row in enumerate(dofs):
            for local_column, global_column in enumerate(dofs):
                rows.append(int(global_row))
                columns.append(int(global_column))
                data.append(float(local[local_row, local_column]))

        states.append(response.state)
        element_results.append(NonlinearElementResult(
            element=index,
            strain=strain,
            stress=response.stress,
            axial_force=axial_force,
            tangent_modulus=response.tangent,
            yielded=response.yielded,
            current_length=current_length,
            state=response.state,
        ))

    tangent = sp.coo_matrix((data, (rows, columns)), shape=(size, size)).tocsr()
    return _AssemblyResult(tangent, internal, tuple(states), tuple(element_results))


def _solve_increment(
    model: Model,
    material_models: tuple[UniaxialMaterialModel, ...],
    committed_states: tuple[UniaxialMaterialState, ...],
    committed_displacement: np.ndarray,
    load_factor: float,
    reference_force: np.ndarray,
    constrained: np.ndarray,
    constrained_values: np.ndarray,
    free: np.ndarray,
    geometric_nonlinear: bool,
    max_iterations: int,
    relative_tolerance: float,
    absolute_tolerance: float,
    displacement_tolerance: float,
) -> tuple[bool, np.ndarray, _AssemblyResult | None, int, float, str | None]:
    displacement = committed_displacement.copy()
    displacement[constrained] = constrained_values
    target_force = load_factor * reference_force
    final_assembly = None
    residual_norm = math.inf

    for iteration in range(1, max_iterations + 1):
        try:
            assembly = _assemble(
                model, displacement, material_models, committed_states, geometric_nonlinear
            )
        except (ValueError, ArithmeticError) as exc:
            return False, displacement, None, iteration, math.inf, str(exc)
        residual = target_force - assembly.internal
        residual_free = residual[free]
        residual_norm = float(np.linalg.norm(residual_free))
        force_scale = max(float(np.linalg.norm(target_force[free])), 1.0)
        if residual_norm <= absolute_tolerance + relative_tolerance * force_scale:
            return True, displacement, assembly, iteration, residual_norm, None
        if free.size == 0:
            return False, displacement, assembly, iteration, residual_norm, "no free equilibrium DOFs"

        tangent = assembly.tangent[free][:, free].tocsc()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", spla.MatrixRankWarning)
                increment = spla.spsolve(tangent, residual_free)
        except (RuntimeError, ValueError, spla.MatrixRankWarning) as exc:
            return False, displacement, assembly, iteration, residual_norm, f"singular tangent: {exc}"
        increment = np.asarray(increment, dtype=float)
        if increment.shape != (free.size,) or not np.all(np.isfinite(increment)):
            return False, displacement, assembly, iteration, residual_norm, "non-finite Newton increment"
        if float(np.linalg.norm(increment)) <= displacement_tolerance:
            return False, displacement, assembly, iteration, residual_norm, "Newton iteration stagnated"

        accepted = False
        for scale in (1.0, 0.5, 0.25, 0.125, 0.0625):
            candidate = displacement.copy()
            candidate[free] += scale * increment
            candidate[constrained] = constrained_values
            try:
                candidate_assembly = _assemble(
                    model, candidate, material_models, committed_states, geometric_nonlinear
                )
            except (ValueError, ArithmeticError):
                continue
            candidate_norm = float(np.linalg.norm(
                (target_force - candidate_assembly.internal)[free]
            ))
            if candidate_norm < residual_norm or candidate_norm <= (
                absolute_tolerance + relative_tolerance * force_scale
            ):
                displacement = candidate
                final_assembly = candidate_assembly
                accepted = True
                break
        if not accepted:
            return False, displacement, final_assembly or assembly, iteration, residual_norm, (
                "Newton line search failed to reduce the residual"
            )

    return False, displacement, final_assembly, max_iterations, residual_norm, (
        f"Newton iteration limit ({max_iterations}) reached"
    )


def solve_nonlinear_truss(
    model: Model,
    materials: UniaxialMaterialModel | Mapping[int, UniaxialMaterialModel]
    | Sequence[UniaxialMaterialModel],
    *,
    load_factors: Sequence[float] | None = None,
    n_steps: int = 20,
    geometric_nonlinear: bool = False,
    displacement_pattern: Mapping[tuple[int, int], float] | None = None,
    initial_states: Sequence[UniaxialMaterialState] | None = None,
    max_iterations: int = 30,
    relative_tolerance: float = 1.0e-8,
    absolute_tolerance: float = 1.0e-6,
    displacement_tolerance: float = 1.0e-14,
    adaptive: bool = True,
    minimum_step: float = 1.0e-5,
    maximum_step: float | None = None,
    cutback_factor: float = 0.5,
    growth_factor: float = 1.5,
    max_accepted_steps: int = 10_000,
    raise_on_failure: bool = False,
) -> NonlinearTrussResult:
    """Solve a force/displacement path by adaptive incremental Newton-Raphson.

    ``model.nodal_loads`` is the reference force pattern and ``load_factors``
    gives the requested load history (for example ``[0, 1, 0, 0.5]``).
    ``displacement_pattern`` supplies prescribed translations per unit load
    factor and enables displacement-controlled tracing through a plastic
    plateau.  Material and geometric nonlinearity can be enabled independently.
    """

    _validate_model(model)
    if n_steps < 1 or max_iterations < 1 or max_accepted_steps < 1:
        raise ValueError("step and iteration limits must be positive")
    for name, value in (
        ("relative_tolerance", relative_tolerance),
        ("absolute_tolerance", absolute_tolerance),
        ("displacement_tolerance", displacement_tolerance),
        ("minimum_step", minimum_step),
    ):
        if not math.isfinite(float(value)) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite")
    if not 0.0 < cutback_factor < 1.0:
        raise ValueError("cutback_factor must be between zero and one")
    if not math.isfinite(growth_factor) or growth_factor < 1.0:
        raise ValueError("growth_factor must be finite and at least one")

    requested = tuple(float(value) for value in (
        load_factors if load_factors is not None else np.linspace(0.0, 1.0, n_steps + 1)
    ))
    if not requested or not all(math.isfinite(value) for value in requested):
        raise ValueError("load_factors must be a nonempty finite sequence")
    if requested[0] != 0.0:
        requested = (0.0,) + requested
    maximum_segment = max(
        (abs(second - first) for first, second in zip(requested, requested[1:])),
        default=1.0,
    )
    step_cap = float(maximum_step) if maximum_step is not None else max(
        maximum_segment / n_steps, minimum_step
    )
    if not math.isfinite(step_cap) or step_cap <= 0.0:
        raise ValueError("maximum_step must be positive and finite")
    if step_cap < minimum_step:
        raise ValueError("maximum_step cannot be smaller than minimum_step")

    material_models = _material_sequence(materials, len(model.elements))
    if initial_states is None:
        committed_states = tuple(material.initial_state() for material in material_models)
    else:
        committed_states = tuple(initial_states)
        if len(committed_states) != len(model.elements):
            raise ValueError("initial_states must match the truss element count")
        if any(not isinstance(state, UniaxialMaterialState) for state in committed_states):
            raise TypeError("initial_states must contain UniaxialMaterialState values")

    reference_force = _reference_force(model)
    pattern = dict(displacement_pattern or {})
    base_constraints: dict[int, float] = {}
    for (node, dof), value in model.constraints.items():
        if dof < 3:
            if value != 0.0:
                raise ValueError(
                    "nonzero support settlements must be supplied through displacement_pattern"
                )
            base_constraints[node * 3 + dof] = float(value)
        elif value != 0.0:
            raise ValueError("nonzero prescribed rotations are unsupported for trusses")
    prescribed_pattern: dict[int, float] = {}
    for (node, dof), value in pattern.items():
        if not 0 <= node < model.n_nodes or not 0 <= dof < 3:
            raise ValueError("displacement_pattern keys must be valid translational DOFs")
        if not math.isfinite(float(value)):
            raise ValueError("displacement_pattern values must be finite")
        prescribed_pattern[node * 3 + dof] = float(value)
    constrained = np.asarray(sorted(set(base_constraints) | set(prescribed_pattern)), dtype=int)
    all_dofs = np.arange(model.n_nodes * 3, dtype=int)
    free = np.setdiff1d(all_dofs, constrained, assume_unique=True)

    committed_displacement = np.zeros(model.n_nodes * 3, dtype=float)
    for dof, value in base_constraints.items():
        committed_displacement[dof] = value
    current_factor = 0.0
    history: list[NonlinearStepResult] = []
    first_yield: float | None = None
    ever_yielded: set[int] = set()
    diagnostic = None
    collapse = False
    failed_load_factor: float | None = None
    failure_residual_norm: float | None = None
    collapse_events: list[CollapseEvent] = []
    last_reactions = np.zeros(model.n_nodes * 3)

    def constrained_at(factor: float) -> np.ndarray:
        return np.asarray([
            base_constraints.get(int(dof), 0.0)
            + factor * prescribed_pattern.get(int(dof), 0.0)
            for dof in constrained
        ], dtype=float)

    for target in requested[1:]:
        direction = 1.0 if target >= current_factor else -1.0
        step_size = min(step_cap, abs(target - current_factor))
        cutbacks = 0
        while abs(target - current_factor) > 1.0e-14:
            if len(history) >= max_accepted_steps:
                collapse = True
                failed_load_factor = current_factor
                diagnostic = f"accepted-step limit ({max_accepted_steps}) reached"
                collapse_events.append(CollapseEvent(
                    len(collapse_events) + 1, current_factor, "analysis_limit",
                    (), diagnostic,
                ))
                break
            increment_size = direction * min(step_size, abs(target - current_factor))
            trial_factor = current_factor + increment_size
            converged, displacement, assembly, iterations, residual_norm, reason = _solve_increment(
                model,
                material_models,
                committed_states,
                committed_displacement,
                trial_factor,
                reference_force,
                constrained,
                constrained_at(trial_factor),
                free,
                geometric_nonlinear,
                max_iterations,
                relative_tolerance,
                absolute_tolerance,
                displacement_tolerance,
            )
            if not converged or assembly is None:
                if adaptive and step_size * cutback_factor >= minimum_step:
                    step_size *= cutback_factor
                    cutbacks += 1
                    continue
                collapse = True
                failed_load_factor = trial_factor
                failure_residual_norm = residual_norm
                diagnostic = (
                    f"increment to load factor {trial_factor:.12g} did not converge: {reason}; "
                    f"residual={residual_norm:.6g}"
                )
                collapse_events.append(CollapseEvent(
                    len(collapse_events) + 1,
                    trial_factor,
                    "global_nonconvergence",
                    tuple(sorted(ever_yielded)),
                    diagnostic,
                ))
                break

            committed_displacement = displacement
            committed_states = assembly.states
            current_factor = trial_factor
            yielded = tuple(result.element for result in assembly.elements if result.yielded)
            newly_yielded = tuple(sorted(set(yielded) - ever_yielded))
            if yielded and first_yield is None:
                first_yield = current_factor
            if newly_yielded:
                collapse_events.append(CollapseEvent(
                    len(collapse_events) + 1,
                    current_factor,
                    "first_yield",
                    newly_yielded,
                    "element entered the plastic range",
                ))
            ever_yielded.update(yielded)
            internal_reaction = assembly.internal - current_factor * reference_force
            last_reactions = internal_reaction
            energy = sum(
                state.dissipated_energy_density * element.sec.A
                * float(np.linalg.norm(model.nodes[element.n2] - model.nodes[element.n1]))
                for state, element in zip(committed_states, model.elements)
            )
            full_u = _translation_to_full(committed_displacement, model.n_nodes)
            full_reactions = _translation_to_full(last_reactions, model.n_nodes)
            history.append(NonlinearStepResult(
                load_factor=current_factor,
                requested_load_factor=target,
                increment=increment_size,
                iterations=iterations,
                residual_norm=residual_norm,
                cutbacks=cutbacks,
                displacement_norm=float(np.linalg.norm(committed_displacement)),
                dissipated_energy=float(energy),
                yielded_elements=yielded,
                u=full_u,
                reactions=full_reactions,
                elements=assembly.elements,
            ))
            cutbacks = 0
            if adaptive and iterations <= max(2, max_iterations // 4):
                step_size = min(step_size * growth_factor, step_cap)
        if collapse:
            break

    full_u = _translation_to_full(committed_displacement, model.n_nodes)
    full_reactions = _translation_to_full(last_reactions, model.n_nodes)
    maximum_factor = max((abs(step.load_factor) for step in history), default=0.0)
    limit = LimitStateReport(
        maximum_absolute_load_factor=maximum_factor,
        last_converged_load_factor=current_factor,
        first_yield_load_factor=first_yield,
        yielded_elements=tuple(sorted(ever_yielded)),
        collapse_detected=collapse,
        reason=diagnostic,
        failed_load_factor=failed_load_factor,
        failure_residual_norm=failure_residual_norm,
        progressive_collapse_sequence=tuple(collapse_events),
    )
    result = NonlinearTrussResult(
        converged=not collapse,
        u=full_u,
        reactions=full_reactions,
        element_states=committed_states,
        history=tuple(history),
        limit_state=limit,
        diagnostic=diagnostic,
    )
    if collapse and raise_on_failure:
        residual = 0.0
        if diagnostic and "residual=" in diagnostic:
            try:
                residual = float(diagnostic.rsplit("residual=", 1)[1])
            except ValueError:
                residual = math.inf
        raise NonlinearConvergenceError(
            diagnostic or "nonlinear analysis failed",
            load_factor=current_factor if failed_load_factor is None else failed_load_factor,
            residual_norm=residual,
        )
    return result


@dataclass(frozen=True)
class NonlinearConstraintRecord:
    constraint_id: str
    satisfied: bool
    utilization: float
    message: str


@dataclass(frozen=True)
class NonlinearDesignEvaluation:
    design: Any
    objective: float
    mass: float
    feasible: bool
    constraints: tuple[NonlinearConstraintRecord, ...]
    nonlinear_result: NonlinearTrussResult

    @property
    def first_yield_load_factor(self) -> float | None:
        return self.nonlinear_result.limit_state.first_yield_load_factor

    @property
    def limit_load_factor(self) -> float:
        return self.nonlinear_result.limit_state.maximum_absolute_load_factor

    @property
    def residual_displacement_norm(self) -> float | None:
        residual = self.nonlinear_result.residual_displacement
        return None if residual is None else float(np.linalg.norm(residual))

    @property
    def maximum_equivalent_plastic_strain(self) -> float:
        return max(
            (state.equivalent_plastic_strain for state in self.nonlinear_result.element_states),
            default=0.0,
        )

    @property
    def dissipated_energy(self) -> float:
        return self.nonlinear_result.dissipated_energy

    def as_dict(self) -> dict[str, Any]:
        return {
            "design": list(self.design) if isinstance(self.design, (tuple, list)) else repr(self.design),
            "objective": self.objective,
            "mass": self.mass,
            "feasible": self.feasible,
            "first_yield_load_factor": self.first_yield_load_factor,
            "limit_load_factor": self.limit_load_factor,
            "residual_displacement_norm": self.residual_displacement_norm,
            "maximum_equivalent_plastic_strain": self.maximum_equivalent_plastic_strain,
            "dissipated_energy": self.dissipated_energy,
            "nonlinear_result": self.nonlinear_result.as_dict(),
        }


class NonlinearTrussSubproblem:
    """Adapter exposing nonlinear FEM through the common optimization protocol.

    The factories keep section/material selection outside the nonlinear solver,
    while every candidate receives the same convergence and collapse checks.
    Nonconvergence is returned as an infeasible evaluation rather than promoted
    to a usable design.
    """

    def __init__(
        self,
        initial_design: Any,
        domains: Sequence[Sequence[int]],
        model_factory: Callable[[Any], Model],
        material_factory: Callable[[Any], Any],
        objective: Callable[[Any, Model], float],
        mass: Callable[[Any, Model], float] | None = None,
        maximum_equivalent_plastic_strain: float | None = None,
        maximum_residual_displacement: float | None = None,
        **solver_options: Any,
    ):
        self.initial_design = initial_design
        self.domains = tuple(tuple(int(value) for value in domain) for domain in domains)
        self.model_factory = model_factory
        self.material_factory = material_factory
        self.objective_function = objective
        self.mass_function = mass or objective
        self.maximum_equivalent_plastic_strain = maximum_equivalent_plastic_strain
        self.maximum_residual_displacement = maximum_residual_displacement
        self.solver_options = dict(solver_options)

    def evaluate(self, design: Any) -> NonlinearDesignEvaluation:
        model = self.model_factory(design)
        objective = float(self.objective_function(design, model))
        mass = float(self.mass_function(design, model))
        if not math.isfinite(objective) or not math.isfinite(mass):
            raise ValueError("nonlinear design objective and mass must be finite")
        try:
            result = solve_nonlinear_truss(
                model, self.material_factory(design), **self.solver_options
            )
        except (ValueError, TypeError, ArithmeticError, NonlinearConvergenceError) as exc:
            size = model.n_nodes * DOF_PER_NODE
            reason = f"nonlinear candidate rejected: {exc}"
            result = NonlinearTrussResult(
                converged=False,
                u=np.zeros(size),
                reactions=np.zeros(size),
                element_states=(),
                history=(),
                limit_state=LimitStateReport(
                    0.0, 0.0, None, (), True, reason,
                    progressive_collapse_sequence=(CollapseEvent(
                        1, 0.0, "candidate_invalid", (), reason,
                    ),),
                ),
                diagnostic=reason,
            )
        records = [NonlinearConstraintRecord(
            constraint_id="nonlinear_convergence_and_collapse",
            satisfied=result.feasible,
            utilization=0.0 if result.feasible else 2.0,
            message=result.diagnostic or "nonlinear path converged without collapse",
        )]
        maximum_plastic = max(
            (state.equivalent_plastic_strain for state in result.element_states),
            default=0.0,
        )
        if self.maximum_equivalent_plastic_strain is not None:
            limit = float(self.maximum_equivalent_plastic_strain)
            if not math.isfinite(limit) or limit <= 0.0:
                raise ValueError("maximum_equivalent_plastic_strain must be positive")
            records.append(NonlinearConstraintRecord(
                "maximum_equivalent_plastic_strain",
                maximum_plastic <= limit,
                maximum_plastic / limit,
                f"maximum equivalent plastic strain={maximum_plastic:.6g}, limit={limit:.6g}",
            ))
        residual = result.residual_displacement
        if self.maximum_residual_displacement is not None:
            limit = float(self.maximum_residual_displacement)
            if not math.isfinite(limit) or limit <= 0.0:
                raise ValueError("maximum_residual_displacement must be positive")
            magnitude = 2.0 * limit if residual is None else float(np.linalg.norm(residual))
            records.append(NonlinearConstraintRecord(
                "maximum_residual_displacement",
                magnitude <= limit,
                magnitude / limit,
                f"residual displacement norm={magnitude:.6g}, limit={limit:.6g}",
            ))
        feasible = all(record.satisfied for record in records)
        return NonlinearDesignEvaluation(
            design, objective, mass, feasible, tuple(records), result
        )
