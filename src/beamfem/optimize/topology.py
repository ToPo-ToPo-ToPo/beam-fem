"""トポロジー／部材配置最適化（Ground Structure 法・トラスLP）。

候補部材を密に張った「地盤構造」から、最小体積となる部材配置を線形計画で求める。
塑性設計の下界定理に基づく古典的定式化（Dorn et al. 1964）で、凸（LP）ゆえ
大域最適が保証される。剛性は不要で、平衡条件と応力制約のみを用いる::

    min   Σ_e L_e A_e                       (総体積)
    s.t.  B n^(k) = f^(k)        各荷重ケース k   (節点平衡)
          -σ_c A_e ≤ n^(k)_e ≤ σ_t A_e        (応力, 引張+/圧縮-)
          A_e ≥ A_min ≥ 0

n^(k)_e は荷重ケース k の部材軸力（引張正）。断面積 A_e は全ケースで共有し、
最悪ケースで決まる。2D・3D いずれも方向余弦で同様に扱える。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog


def grid_nodes(nx: int, ny: int, lx: float, ly: float, nz: int = 1, lz: float = 0.0):
    """矩形（または直方体）格子の節点座標を返す。

    節点番号は 2D で iy*nx+ix、3D で (iz*ny+iy)*nx+ix。
    nz=1 のとき 2D（dim=2）の節点を返す。
    """
    xs = np.linspace(0.0, lx, nx)
    ys = np.linspace(0.0, ly, ny)
    if nz <= 1:
        pts = [(x, y) for y in ys for x in xs]
        return np.array(pts, dtype=float)
    zs = np.linspace(0.0, lz, nz)
    pts = [(x, y, z) for z in zs for y in ys for x in xs]
    return np.array(pts, dtype=float)


def generate_members(nodes, max_length: float | None = None, collinear_tol: float = 1e-9):
    """全節点対から候補部材を生成し、共線で重複する長い部材を除く。

    ある部材 (i,j) の線分上に別の節点 k が乗る場合、その部材は短い区間の
    組合せで表せるため冗長として除外する（標準的な地盤構造の生成法）。
    """
    nodes = np.asarray(nodes, dtype=float)
    N = len(nodes)
    members = []
    for i in range(N):
        for j in range(i + 1, N):
            vij = nodes[j] - nodes[i]
            L = np.linalg.norm(vij)
            if L == 0:
                continue
            if max_length is not None and L > max_length * (1 + 1e-9):
                continue
            blocked = False
            for k in range(N):
                if k == i or k == j:
                    continue
                vik = nodes[k] - nodes[i]
                t = np.dot(vik, vij) / L**2
                if 0.0 < t < 1.0:
                    perp = np.linalg.norm(vik - t * vij)
                    if perp <= collinear_tol * L:
                        blocked = True
                        break
            if not blocked:
                members.append((i, j))
    return members


@dataclass
class GroundStructure:
    """地盤構造（候補部材ネットワーク）の定義。

    nodes : (N, dim) 座標（dim=2 or 3）
    members : 候補部材の (i, j) 節点インデックスのリスト
    supports : {node: [固定する並進自由度 0=x,1=y,2=z, ...]}
    load_cases : 荷重ケースのリスト。各々 {(node, dof): value}
    """

    nodes: np.ndarray
    members: list[tuple[int, int]]
    supports: dict[int, list[int]] = field(default_factory=dict)
    load_cases: list[dict] = field(default_factory=list)

    @property
    def dim(self) -> int:
        return self.nodes.shape[1]

    @property
    def n_member(self) -> int:
        return len(self.members)

    def lengths(self) -> np.ndarray:
        L = np.empty(self.n_member)
        for e, (i, j) in enumerate(self.members):
            L[e] = np.linalg.norm(self.nodes[j] - self.nodes[i])
        return L

    def free_dofs(self) -> np.ndarray:
        """自由（非拘束）並進自由度のグローバル番号配列。"""
        dim = self.dim
        fixed = set()
        for node, dofs in self.supports.items():
            for d in dofs:
                fixed.add(node * dim + d)
        all_dofs = np.arange(self.nodes.shape[0] * dim)
        return np.array([d for d in all_dofs if d not in fixed], dtype=np.int64)


@dataclass
class TopologyResult:
    """トポロジー最適化の結果。"""

    gs: GroundStructure
    areas: np.ndarray          # 各部材の断面積 (M,)
    forces: np.ndarray         # 各荷重ケースの部材軸力 (K, M) 引張+
    volume: float
    status: str

    def active(self, rel_tol: float = 1e-3) -> np.ndarray:
        """有効部材（断面積が最大の rel_tol 倍超）のインデックス。"""
        amax = self.areas.max() if self.areas.size else 0.0
        return np.where(self.areas > rel_tol * amax)[0] if amax > 0 else np.array([], dtype=int)


def equilibrium_matrix(gs: GroundStructure):
    """平衡行列 B (n_free_dof × M) を疎行列で返す。

    部材 (i,j) の単位ベクトル d=(j-i)/L に対し、列は節点 i に -d、節点 j に +d。
    軸力 n（引張正）について B n = f（外力）が節点平衡を表す。
    """
    dim = gs.dim
    nodes = gs.nodes
    free = gs.free_dofs()
    dof_index = {d: k for k, d in enumerate(free)}  # グローバル -> 縮約

    rows, cols, data = [], [], []
    for e, (i, j) in enumerate(gs.members):
        vec = nodes[j] - nodes[i]
        L = np.linalg.norm(vec)
        d = vec / L
        for a in range(dim):
            gi = i * dim + a
            gj = j * dim + a
            if gi in dof_index:
                rows.append(dof_index[gi]); cols.append(e); data.append(-d[a])
            if gj in dof_index:
                rows.append(dof_index[gj]); cols.append(e); data.append(d[a])
    B = sp.coo_matrix((data, (rows, cols)), shape=(len(free), gs.n_member)).tocsr()
    return B, free, dof_index


def _load_vector(gs: GroundStructure, case: dict, dof_index: dict, n_free: int) -> np.ndarray:
    f = np.zeros(n_free)
    dim = gs.dim
    for (node, dof), val in case.items():
        g = node * dim + dof
        if g in dof_index:
            f[dof_index[g]] += val
    return f


def solve_min_volume(
    gs: GroundStructure,
    sigma_t: float,
    sigma_c: float | None = None,
    area_min: float = 0.0,
) -> TopologyResult:
    """最小体積トポロジーを線形計画で解く。

    sigma_t : 許容引張応力
    sigma_c : 許容圧縮応力（省略時 sigma_t と同じ）
    area_min : 部材断面積の下限（0 で完全除去を許容）
    """
    if sigma_c is None:
        sigma_c = sigma_t
    if not gs.load_cases:
        raise ValueError("荷重ケースがありません")

    M = gs.n_member
    K = len(gs.load_cases)
    L = gs.lengths()
    B, free, dof_index = equilibrium_matrix(gs)
    n_free = len(free)

    # 変数順: [A_0..A_{M-1}, n^(0)_0..n^(0)_{M-1}, ..., n^(K-1)_...]
    nvar = M * (1 + K)

    # 目的: Σ L_e A_e
    c = np.zeros(nvar)
    c[:M] = L

    # 等式: 各ケースの平衡 B n^(k) = f^(k)
    eq_rows = []
    eq_b = []
    for k in range(K):
        # [0(A部) | 0...| B(該当ケース列) | ...0]
        left = sp.csr_matrix((n_free, M))  # A 部
        blocks = [left]
        for kk in range(K):
            blocks.append(B if kk == k else sp.csr_matrix((n_free, M)))
        eq_rows.append(sp.hstack(blocks, format="csr"))
        eq_b.append(_load_vector(gs, gs.load_cases[k], dof_index, n_free))
    A_eq = sp.vstack(eq_rows, format="csr")
    b_eq = np.concatenate(eq_b)

    # 不等式: n^(k)_e - σ_t A_e ≤ 0 ,  -n^(k)_e - σ_c A_e ≤ 0
    ub_rows = []
    for k in range(K):
        # 行ブロック (M 行)
        # 引張: n - σ_t A ≤ 0
        Acol = sp.diags(-sigma_t * np.ones(M))
        ncols = [sp.diags(np.ones(M)) if kk == k else sp.csr_matrix((M, M)) for kk in range(K)]
        ub_rows.append(sp.hstack([Acol] + ncols, format="csr"))
        # 圧縮: -n - σ_c A ≤ 0
        Acol2 = sp.diags(-sigma_c * np.ones(M))
        ncols2 = [sp.diags(-np.ones(M)) if kk == k else sp.csr_matrix((M, M)) for kk in range(K)]
        ub_rows.append(sp.hstack([Acol2] + ncols2, format="csr"))
    A_ub = sp.vstack(ub_rows, format="csr")
    b_ub = np.zeros(A_ub.shape[0])

    # 変数境界
    bounds = [(area_min, None)] * M + [(None, None)] * (M * K)

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"LP が解けませんでした: {res.message}")

    x = res.x
    areas = x[:M]
    forces = np.array([x[M + k * M : M + (k + 1) * M] for k in range(K)])
    volume = float(c @ x)
    return TopologyResult(gs=gs, areas=areas, forces=forces, volume=volume, status=res.message)
