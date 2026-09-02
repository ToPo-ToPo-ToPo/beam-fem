"""Undamped linear modal analysis with a transparent lumped-mass model."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .assembly import assemble_stiffness, constrained_dofs
from .model import DOF_PER_NODE, Model


@dataclass(frozen=True)
class ModalResult:
    eigenvalues: np.ndarray
    circular_frequencies: np.ndarray
    frequencies_hz: np.ndarray
    periods: np.ndarray
    modes: np.ndarray
    mass_diagonal: np.ndarray
    dynamic_dofs: np.ndarray
    condensed_dofs: np.ndarray

    def node_mode(self, mode: int, node: int) -> np.ndarray:
        start = int(node) * DOF_PER_NODE
        return self.modes[start:start + DOF_PER_NODE, int(mode)]


def _triangle_area(first: np.ndarray, second: np.ndarray, third: np.ndarray) -> float:
    return 0.5 * float(np.linalg.norm(np.cross(second - first, third - first)))


def assemble_lumped_mass(model: Model) -> sp.csr_matrix:
    """Assemble nodal translational mass from members and flat shells.

    Rotational inertia is omitted. During modal solution, free massless DOFs are
    statically condensed, so frame/shell rotations still participate through
    stiffness compatibility. This formulation is intentionally documented as a
    lumped-mass approximation rather than a consistent beam/shell mass matrix.
    """

    diagonal = np.zeros(model.n_dof, dtype=float)

    def add_node_mass(node: int, mass: float) -> None:
        if not math.isfinite(mass) or mass < 0.0:
            raise ValueError("element mass must be finite and nonnegative")
        diagonal[node * DOF_PER_NODE:node * DOF_PER_NODE + 3] += mass

    for element in model.elements:
        length = model.element_length(element)
        mass = float(element.mat.rho) * float(element.sec.A) * length
        add_node_mass(element.n1, 0.5 * mass)
        add_node_mass(element.n2, 0.5 * mass)
    for shell in model.shells:
        area = _triangle_area(
            model.nodes[shell.n1], model.nodes[shell.n2], model.nodes[shell.n3]
        )
        mass = float(shell.mat.rho) * float(shell.thickness) * area
        for node in (shell.n1, shell.n2, shell.n3):
            add_node_mass(node, mass / 3.0)
    for shell in model.quad_shells:
        first, second, third, fourth = (
            model.nodes[shell.n1], model.nodes[shell.n2],
            model.nodes[shell.n3], model.nodes[shell.n4],
        )
        area = _triangle_area(first, second, third) + _triangle_area(first, third, fourth)
        mass = float(shell.mat.rho) * float(shell.thickness) * area
        for node in (shell.n1, shell.n2, shell.n3, shell.n4):
            add_node_mass(node, mass / 4.0)
    if not np.any(diagonal > 0.0):
        raise ValueError("modal analysis requires positive material density and structural mass")
    return sp.diags(diagonal, format="csr")


def solve_modes(model: Model, number: int = 6, *,
                eigenvalue_tolerance: float = 1.0e-10) -> ModalResult:
    """Solve the lowest positive undamped modes of the constrained structure."""

    if number < 1:
        raise ValueError("number of modes must be positive")
    if not math.isfinite(eigenvalue_tolerance) or eigenvalue_tolerance <= 0.0:
        raise ValueError("eigenvalue_tolerance must be positive and finite")
    stiffness = assemble_stiffness(model).tocsr()
    mass = assemble_lumped_mass(model).tocsr()
    fixed, _ = constrained_dofs(model)
    free = np.setdiff1d(np.arange(model.n_dof), fixed, assume_unique=False)
    if free.size == 0:
        raise ValueError("modal analysis has no free DOFs")
    stiffness_norm = np.asarray(abs(stiffness[free][:, free]).sum(axis=1)).ravel()
    mass_diagonal = mass.diagonal()
    zero_stiffness_massive = free[
        (stiffness_norm <= eigenvalue_tolerance)
        & (mass_diagonal[free] > 0.0)
    ]
    if zero_stiffness_massive.size:
        raise ValueError(
            "free massive DOFs have zero stiffness; supports leave a rigid mode or mechanism: "
            f"{zero_stiffness_massive.tolist()}"
        )
    active = free[stiffness_norm > 0.0]
    if active.size == 0:
        raise ValueError("modal analysis has no active stiffness DOFs")
    dynamic = active[mass_diagonal[active] > 0.0]
    algebraic = np.setdiff1d(active, dynamic, assume_unique=False)
    if dynamic.size == 0:
        raise ValueError("modal analysis has no massive active DOFs")

    kdd = stiffness[dynamic][:, dynamic].toarray()
    recovery = None
    if algebraic.size:
        kaa = stiffness[algebraic][:, algebraic].tocsc()
        kad = stiffness[algebraic][:, dynamic].toarray()
        try:
            recovery = -spla.spsolve(kaa, kad)
        except (RuntimeError, ValueError) as exc:
            raise ValueError("massless modal DOFs cannot be condensed; mechanism present") from exc
        recovery = np.asarray(recovery, dtype=float)
        if recovery.ndim == 1:
            recovery = recovery[:, None]
        if not np.all(np.isfinite(recovery)):
            raise ValueError("modal condensation produced non-finite values")
        kdd = kdd + stiffness[dynamic][:, algebraic].toarray() @ recovery
    kdd = 0.5 * (kdd + kdd.T)
    mdd = np.diag(mass_diagonal[dynamic])
    try:
        eigenvalues, vectors = la.eigh(kdd, mdd, check_finite=True)
    except la.LinAlgError as exc:
        raise ValueError("generalized eigenproblem failed; check supports and mass") from exc
    scale = max(float(np.max(np.abs(eigenvalues), initial=0.0)), 1.0)
    if np.any(eigenvalues <= eigenvalue_tolerance * scale):
        raise ValueError(
            "zero/negative vibration eigenvalue detected; supports leave a rigid mode or mechanism"
        )
    positive = np.flatnonzero(eigenvalues > eigenvalue_tolerance * scale)
    if positive.size == 0:
        raise ValueError("no positive vibration modes; structure is a mechanism")
    chosen = positive[: min(number, positive.size)]
    eigenvalues = eigenvalues[chosen]
    vectors = vectors[:, chosen]
    full_modes = np.zeros((model.n_dof, len(chosen)), dtype=float)
    full_modes[dynamic, :] = vectors
    if algebraic.size and recovery is not None:
        full_modes[algebraic, :] = recovery @ vectors
    for index in range(full_modes.shape[1]):
        maximum = float(np.max(np.abs(full_modes[:, index])))
        if maximum > 0.0:
            full_modes[:, index] /= maximum
    circular = np.sqrt(eigenvalues)
    frequencies = circular / (2.0 * math.pi)
    periods = 1.0 / frequencies
    return ModalResult(
        eigenvalues=eigenvalues,
        circular_frequencies=circular,
        frequencies_hz=frequencies,
        periods=periods,
        modes=full_modes,
        mass_diagonal=mass_diagonal,
        dynamic_dofs=dynamic,
        condensed_dofs=algebraic,
    )
