"""全体剛性行列の組み立て（疎行列）。

数千要素以上を想定し、COO 形式でトリプレットを蓄積してから CSR に変換する。
最適化の反復では同じ構造を繰り返し解くため、各要素の全体自由度マップを
事前計算して再利用できるようにしている。
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .element3d import element_stiffness_global
from .model import DOF_PER_NODE, Model


def element_dof_map(model: Model) -> list[np.ndarray]:
    """各要素の 12 個の全体自由度番号を返す。"""
    maps = []
    for e in model.elements:
        d1 = e.n1 * DOF_PER_NODE
        d2 = e.n2 * DOF_PER_NODE
        dofs = np.concatenate(
            [np.arange(d1, d1 + DOF_PER_NODE), np.arange(d2, d2 + DOF_PER_NODE)]
        )
        maps.append(dofs)
    return maps


def assemble_stiffness(model: Model, dof_maps: list[np.ndarray] | None = None) -> sp.csr_matrix:
    """全体剛性行列 K (n_dof x n_dof) を CSR 疎行列で返す。"""
    if dof_maps is None:
        dof_maps = element_dof_map(model)

    n = model.n_dof
    ne = len(model.elements)
    # 各要素 12x12 = 144 エントリ
    rows = np.empty(ne * 144, dtype=np.int64)
    cols = np.empty(ne * 144, dtype=np.int64)
    data = np.empty(ne * 144, dtype=float)

    for i, e in enumerate(model.elements):
        p1 = model.nodes[e.n1]
        p2 = model.nodes[e.n2]
        ke = element_stiffness_global(p1, p2, e.mat, e.sec, e.vref)
        dofs = dof_maps[i]
        rr, cc = np.meshgrid(dofs, dofs, indexing="ij")
        sl = slice(i * 144, (i + 1) * 144)
        rows[sl] = rr.ravel()
        cols[sl] = cc.ravel()
        data[sl] = ke.ravel()

    K = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    return K


def assemble_load_vector(model: Model) -> np.ndarray:
    """全体荷重ベクトル F を返す。"""
    F = np.zeros(model.n_dof)
    for (node, dof), val in model.nodal_loads.items():
        F[node * DOF_PER_NODE + dof] += val
    return F


def constrained_dofs(model: Model) -> tuple[np.ndarray, np.ndarray]:
    """拘束自由度番号と強制変位値の配列を返す。"""
    idx = []
    vals = []
    for (node, dof), val in model.constraints.items():
        idx.append(node * DOF_PER_NODE + dof)
        vals.append(val)
    return np.array(idx, dtype=np.int64), np.array(vals, dtype=float)
