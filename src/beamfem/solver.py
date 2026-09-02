"""静的線形解析ソルバ。

境界条件は自由度の分割法（partitioning）で処理する。拘束自由度を消去した
縮約系 K_ff u_f = F_f - K_fc u_c を疎直接法で解く。

ソルバ本体は `_solve_sparse` に集約し、後から PARDISO / CHOLMOD などへ
差し替え可能にしている（業務での性能要求に対応）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .assembly import (
    assemble_load_vector,
    assemble_stiffness,
    constrained_dofs,
    element_dof_map,
)
from .element3d import rotation_matrix
from .model import DOF_PER_NODE, Element, Model, TrussElement


class StructuralMechanismError(RuntimeError):
    """剛性を持たない荷重自由度または特異な機構を検出した。"""

    def __init__(self, message: str, dofs=()):
        self.dofs = tuple(int(dof) for dof in dofs)
        suffix = f" affected_dofs={list(self.dofs)}" if self.dofs else ""
        super().__init__(message + suffix)


@runtime_checkable
class SparseSolver(Protocol):
    """Public adapter contract for sparse direct-solver integrations."""

    name: str

    def factorize(self, matrix: sp.csc_matrix) -> object: ...


@dataclass(frozen=True)
class SciPyLUSolver:
    name: str = "scipy_splu"

    def factorize(self, matrix: sp.csc_matrix) -> object:
        return spla.splu(matrix)


_SPARSE_SOLVERS: dict[str, SparseSolver] = {"scipy_splu": SciPyLUSolver()}


def register_sparse_solver(name: str, solver: SparseSolver, *, replace: bool = False) -> None:
    """Register an external sparse solver without changing FEM assembly code."""
    key = str(name).strip()
    if not key or not isinstance(solver, SparseSolver):
        raise ValueError("sparse solver requires a name and factorize(matrix) method")
    if key in _SPARSE_SOLVERS and not replace:
        raise ValueError(f"sparse solver {key!r} is already registered")
    _SPARSE_SOLVERS[key] = solver


def available_sparse_solvers() -> tuple[str, ...]:
    return tuple(sorted(_SPARSE_SOLVERS))


def get_sparse_solver(solver: str | SparseSolver | None = None) -> SparseSolver:
    if solver is None:
        return _SPARSE_SOLVERS["scipy_splu"]
    if isinstance(solver, str):
        try:
            return _SPARSE_SOLVERS[solver]
        except KeyError as exc:
            raise ValueError(
                f"unknown sparse solver {solver!r}; available={available_sparse_solvers()}"
            ) from exc
    if not isinstance(solver, SparseSolver):
        raise TypeError("sparse_solver must be a registered name or SparseSolver")
    return solver


@dataclass
class StaticResult:
    """静解析の結果。"""

    u: np.ndarray  # 全自由度の変位 (n_dof,)
    reactions: np.ndarray  # 全自由度の反力 (n_dof,)。自由自由度は0。
    K: sp.csr_matrix  # 組み立て済み全体剛性（再利用用）

    def node_disp(self, node: int) -> np.ndarray:
        """指定節点の 6 自由度変位 [ux,uy,uz,rx,ry,rz] を返す。"""
        s = node * DOF_PER_NODE
        return self.u[s : s + DOF_PER_NODE]


@dataclass
class StaticFactorization:
    """Reusable stiffness factorization for identical geometry and supports.

    The object is intentionally tied to one assembled stiffness matrix.  It
    only accepts a new load vector; callers must create a new factorization
    whenever elements, sections, materials, or supports change.
    """

    K: sp.csr_matrix
    free: np.ndarray
    constrained: np.ndarray
    constrained_values: np.ndarray
    lu: object
    inactive_free: np.ndarray
    solver_name: str = "scipy_splu"

    def solve_load(self, force: np.ndarray) -> StaticResult:
        force = np.asarray(force, dtype=float)
        if force.shape != (self.K.shape[0],):
            raise ValueError("荷重ベクトルの自由度数が剛性行列と一致しません")
        u = np.zeros(self.K.shape[0])
        u[self.constrained] = self.constrained_values
        rhs = force[self.free].copy()
        if self.inactive_free.size:
            load_scale = max(float(np.max(np.abs(force), initial=0.0)), 1.0)
            loaded = self.inactive_free[np.abs(force[self.inactive_free]) > 1e-12 * load_scale]
            if loaded.size:
                raise StructuralMechanismError("荷重自由度に剛性がありません", loaded)
        if self.constrained.size and np.any(self.constrained_values != 0.0):
            rhs -= self.K[self.free][:, self.constrained] @ self.constrained_values
        if self.free.size:
            u[self.free] = self.lu.solve(rhs)
        reactions = np.zeros_like(u)
        full_reaction = self.K @ u - force
        reactions[self.constrained] = full_reaction[self.constrained]
        return StaticResult(u=u, reactions=reactions, K=self.K)

    def solve_model(self, model: Model) -> StaticResult:
        if model.n_dof != self.K.shape[0]:
            raise ValueError("モデル自由度数が再利用剛性行列と一致しません")
        return self.solve_load(assemble_load_vector(model))


def _solve_sparse(A: sp.csr_matrix, b: np.ndarray,
                  sparse_solver: str | SparseSolver | None = None) -> np.ndarray:
    """疎線形系 A x = b を解く。ここを差し替えればソルバを交換できる。"""
    # splu は LU 分解を保持でき、複数右辺・反復解析で再利用しやすい
    try:
        lu = get_sparse_solver(sparse_solver).factorize(A.tocsc())
    except RuntimeError as exc:
        raise StructuralMechanismError(
            "縮約剛性行列が特異です。支持条件または部材接続を確認してください"
        ) from exc
    return lu.solve(b)


def factorize_static(model: Model, dof_maps=None,
                     sparse_solver: str | SparseSolver | None = None) -> StaticFactorization:
    """Assemble and factorize one static system for multiple load cases."""
    if dof_maps is None:
        dof_maps = element_dof_map(model)
    K = assemble_stiffness(model, dof_maps)
    constrained, constrained_values = constrained_dofs(model)
    all_free = np.setdiff1d(np.arange(model.n_dof), constrained, assume_unique=False)
    if all_free.size == 0:
        raise ValueError("自由自由度がありません（過剰拘束）。")
    Kff = K[all_free][:, all_free].tocsr()
    # トラスの回転自由度など、完全にゼロの無荷重自由度は因数分解から除く。
    row_norm = np.asarray(abs(Kff).sum(axis=1)).ravel()
    active_mask = row_norm != 0.0
    free = all_free[active_mask]
    inactive = all_free[~active_mask]
    # Only rotations at nodes connected exclusively to axial trusses are
    # intentionally absent from the formulation.  A zero translational row is
    # a structural mechanism even when the current load case does not excite
    # it, and must never be silently removed.
    truss_nodes: set[int] = set()
    non_truss_nodes: set[int] = set()
    for element in model.elements:
        target = truss_nodes if isinstance(element, TrussElement) else non_truss_nodes
        target.update((element.n1, element.n2))
    for shell in model.shells:
        non_truss_nodes.update((shell.n1, shell.n2, shell.n3))
    for shell in model.quad_shells:
        non_truss_nodes.update((shell.n1, shell.n2, shell.n3, shell.n4))
    truss_only_nodes = truss_nodes - non_truss_nodes
    ignorable = {
        node * DOF_PER_NODE + dof
        for node in truss_only_nodes
        for dof in (3, 4, 5)
    }
    # A released frame-end rotation with no other connected stiffness is an
    # intentional internal hinge, analogous to a truss-only nodal rotation.
    for element in model.elements:
        if not isinstance(element, Element):
            continue
        rotation = rotation_matrix(
            model.nodes[element.n1], model.nodes[element.n2], element.vref
        )
        for node, releases in (
            (element.n1, element.release_n1), (element.n2, element.release_n2)
        ):
            if set(releases) == {3, 4, 5}:
                ignorable.update(node * DOF_PER_NODE + dof for dof in (3, 4, 5))
                continue
            for local_dof in releases:
                direction = np.abs(rotation[local_dof - 3])
                aligned = np.flatnonzero(direction > 1.0 - 1e-10)
                if aligned.size == 1:
                    ignorable.add(node * DOF_PER_NODE + 3 + int(aligned[0]))
    unexpected_inactive = np.asarray([dof for dof in inactive if int(dof) not in ignorable], dtype=int)
    if unexpected_inactive.size:
        raise StructuralMechanismError(
            "並進または非トラス回転自由度に剛性がありません",
            unexpected_inactive,
        )
    solver = get_sparse_solver(sparse_solver)
    lu = None
    if free.size:
        try:
            lu = solver.factorize(K[free][:, free].tocsc())
        except RuntimeError as exc:
            raise StructuralMechanismError(
                "縮約剛性行列が特異です。支持条件または部材接続を確認してください",
                free,
            ) from exc
    return StaticFactorization(
        K, free, constrained, constrained_values, lu, inactive, solver.name
    )


def solve_static(model: Model, dof_maps=None,
                 sparse_solver: str | SparseSolver | None = None) -> StaticResult:
    """線形静解析を実行する。"""
    return factorize_static(model, dof_maps, sparse_solver).solve_model(model)
