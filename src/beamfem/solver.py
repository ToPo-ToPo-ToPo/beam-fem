"""静的線形解析ソルバ。

境界条件は自由度の分割法（partitioning）で処理する。拘束自由度を消去した
縮約系 K_ff u_f = F_f - K_fc u_c を疎直接法で解く。

ソルバ本体は `_solve_sparse` に集約し、後から PARDISO / CHOLMOD などへ
差し替え可能にしている（業務での性能要求に対応）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .assembly import (
    assemble_load_vector,
    assemble_stiffness,
    constrained_dofs,
    element_dof_map,
)
from .model import DOF_PER_NODE, Model


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


def _solve_sparse(A: sp.csr_matrix, b: np.ndarray) -> np.ndarray:
    """疎線形系 A x = b を解く。ここを差し替えればソルバを交換できる。"""
    # splu は LU 分解を保持でき、複数右辺・反復解析で再利用しやすい
    lu = spla.splu(A.tocsc())
    return lu.solve(b)


def solve_static(model: Model, dof_maps=None) -> StaticResult:
    """線形静解析を実行する。"""
    if dof_maps is None:
        dof_maps = element_dof_map(model)

    n = model.n_dof
    K = assemble_stiffness(model, dof_maps)
    F = assemble_load_vector(model)

    c_idx, c_val = constrained_dofs(model)
    all_dofs = np.arange(n)
    free = np.setdiff1d(all_dofs, c_idx, assume_unique=False)

    if free.size == 0:
        raise ValueError("自由自由度がありません（過剰拘束）。")

    # 縮約系の構築
    Kff = K[free][:, free]
    u = np.zeros(n)
    u[c_idx] = c_val  # 強制変位の代入

    # 右辺: F_f - K_fc u_c
    rhs = F[free].copy()
    if c_idx.size and np.any(c_val != 0.0):
        Kfc = K[free][:, c_idx]
        rhs -= Kfc @ c_val

    u[free] = _solve_sparse(Kff.tocsr(), rhs)

    # 反力 R = K u - F （拘束自由度のみ意味を持つ）
    reactions = np.zeros(n)
    full_R = K @ u - F
    reactions[c_idx] = full_R[c_idx]

    return StaticResult(u=u, reactions=reactions, K=K)
