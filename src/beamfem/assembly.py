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
from .shell3d import shell_stiffness_global
from .shell_mitc4 import quad_shell_stiffness_global


def _node_dofs(node: int) -> np.ndarray:
    """節点の 6 自由度の全体番号。"""
    base = node * DOF_PER_NODE
    return np.arange(base, base + DOF_PER_NODE)


def element_dof_map(model: Model) -> list[np.ndarray]:
    """各梁要素の 12 個の全体自由度番号を返す。"""
    return [np.concatenate([_node_dofs(e.n1), _node_dofs(e.n2)]) for e in model.elements]


def shell_dof_map(model: Model) -> list[np.ndarray]:
    """各シェル要素（3節点）の 18 個の全体自由度番号を返す。"""
    return [
        np.concatenate([_node_dofs(s.n1), _node_dofs(s.n2), _node_dofs(s.n3)])
        for s in model.shells
    ]


def quad_shell_dof_map(model: Model) -> list[np.ndarray]:
    """各四角形シェル要素（4節点）の 24 個の全体自由度番号を返す。"""
    return [
        np.concatenate([_node_dofs(s.n1), _node_dofs(s.n2),
                        _node_dofs(s.n3), _node_dofs(s.n4)])
        for s in model.quad_shells
    ]


def assemble_stiffness(
    model: Model,
    dof_maps: list[np.ndarray] | None = None,
    shell_maps: list[np.ndarray] | None = None,
    quad_maps: list[np.ndarray] | None = None,
) -> sp.csr_matrix:
    """全体剛性行列 K (n_dof x n_dof) を CSR 疎行列で返す。

    梁要素（12x12）・3節点シェル（18x18）・4節点シェル（24x24）を組み立てる。
    """
    if dof_maps is None:
        dof_maps = element_dof_map(model)
    if shell_maps is None:
        shell_maps = shell_dof_map(model)
    if quad_maps is None:
        quad_maps = quad_shell_dof_map(model)

    n = model.n_dof
    ne = len(model.elements)
    ns = len(model.shells)
    nq = len(model.quad_shells)
    # 梁 12x12=144、3節点シェル 18x18=324、4節点シェル 24x24=576 エントリ
    n_entries = ne * 144 + ns * 324 + nq * 576
    rows = np.empty(n_entries, dtype=np.int64)
    cols = np.empty(n_entries, dtype=np.int64)
    data = np.empty(n_entries, dtype=float)

    for i, e in enumerate(model.elements):
        p1 = model.nodes[e.n1]
        p2 = model.nodes[e.n2]
        ke = element_stiffness_global(p1, p2, e.mat, e.sec, e.vref, e.offset)
        dofs = dof_maps[i]
        rr, cc = np.meshgrid(dofs, dofs, indexing="ij")
        sl = slice(i * 144, (i + 1) * 144)
        rows[sl] = rr.ravel()
        cols[sl] = cc.ravel()
        data[sl] = ke.ravel()

    off = ne * 144
    for i, s in enumerate(model.shells):
        ks = shell_stiffness_global(
            model.nodes[s.n1], model.nodes[s.n2], model.nodes[s.n3], s.mat, s.thickness
        )
        dofs = shell_maps[i]
        rr, cc = np.meshgrid(dofs, dofs, indexing="ij")
        sl = slice(off + i * 324, off + (i + 1) * 324)
        rows[sl] = rr.ravel()
        cols[sl] = cc.ravel()
        data[sl] = ks.ravel()

    off += ns * 324
    for i, s in enumerate(model.quad_shells):
        ks = quad_shell_stiffness_global(
            model.nodes[s.n1], model.nodes[s.n2], model.nodes[s.n3], model.nodes[s.n4],
            s.mat, s.thickness,
        )
        dofs = quad_maps[i]
        rr, cc = np.meshgrid(dofs, dofs, indexing="ij")
        sl = slice(off + i * 576, off + (i + 1) * 576)
        rows[sl] = rr.ravel()
        cols[sl] = cc.ravel()
        data[sl] = ks.ravel()

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
