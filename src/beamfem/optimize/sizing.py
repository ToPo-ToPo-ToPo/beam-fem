"""断面サイジング最適化の問題定義と解析的感度（直接法）。

定式化::

    minimize   W(x) = Σ_e ρ_e L_e A_e(x)          (総質量)
    s.t.       σ_max,e(x) / σ_allow,e - 1 ≤ 0       (要素応力)
               |u_j(x)| / u_lim,j - 1 ≤ 0            (たわみ・両側)
               x_min ≤ x ≤ x_max                     (寸法スケール)

感度は **直接法**（解析的）。Kff の LU 分解を 1 回だけ作り、各設計変数 i に
ついて K (du/dx_i) = -(dK/dx_i) u を後退代入で解く（分解を再利用）。要素剛性の
∂k/∂(A,Iy,Iz,J) は解析式（element3d.local_stiffness_derivs）を用いる。設計変数が
少なく制約が多い本問題に適した方式。

梁に加えてシェル要素・オフセット梁にも対応する:
- シェル要素は固定剛性として全体行列に加わる（板厚は設計変数ではない）。
- オフセット梁は剛体腕 G を含めた変換 B=T·G を用い、剛性・感度・応力回収が
  自動的に整合する（合成剛性 EA·e² が感度に反映される）。
これによりリブ補強板（シェル板＋オフセットリブ）のサイジングが解ける。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ..assembly import (
    assemble_load_vector,
    element_dof_map,
    quad_shell_dof_map,
    shell_dof_map,
)
from ..element3d import (
    local_stiffness,
    local_stiffness_derivs,
    rigid_offset_matrix,
    rotation_matrix,
    transformation_matrix,
)
from ..model import DOF_PER_NODE, Model
from ..shell3d import shell_stiffness_global
from ..shell_mitc4 import quad_shell_stiffness_global
from .sections import ScaledSection


@dataclass
class DesignVar:
    """1 つの設計変数（スケール係数）と、それが支配する要素群。"""

    family: ScaledSection
    elements: list[int]
    x0: float = 1.0
    xmin: float = 0.2
    xmax: float = 5.0
    name: str = ""


@dataclass
class DispLimit:
    """たわみ制約 |u(node, dof)| ≤ limit。"""

    node: int
    dof: int
    limit: float


@dataclass
class SizingProblem:
    """断面サイジング最適化問題。"""

    model: Model
    design_vars: list[DesignVar]
    sigma_allow: float | dict[int, float] | None = None  # 要素応力許容値
    disp_limits: list[DispLimit] = field(default_factory=list)

    # 内部キャッシュ
    _elem_of_var: dict = field(default_factory=dict, init=False)

    def __post_init__(self):
        # 要素 -> 設計変数 の対応（各要素は高々1変数に支配される前提）
        for i, dv in enumerate(self.design_vars):
            for e in dv.elements:
                if e in self._elem_of_var:
                    raise ValueError(f"要素 {e} が複数の設計変数に割り当てられています")
                self._elem_of_var[e] = i

    # ------------------------------------------------------------------
    @property
    def n_var(self) -> int:
        return len(self.design_vars)

    def bounds(self):
        xmin = np.array([dv.xmin for dv in self.design_vars])
        xmax = np.array([dv.xmax for dv in self.design_vars])
        return xmin, xmax

    def x0(self) -> np.ndarray:
        return np.array([dv.x0 for dv in self.design_vars])

    def _sigma_allow_of(self, e: int) -> float:
        sa = self.sigma_allow
        if sa is None:
            return np.inf
        if isinstance(sa, dict):
            return sa.get(e, np.inf)
        return sa

    # ------------------------------------------------------------------
    def _apply_x(self, x):
        """設計変数を要素断面に反映する。"""
        for i, dv in enumerate(self.design_vars):
            sec = dv.family.make(float(x[i]))
            for e in dv.elements:
                self.model.elements[e].sec = sec

    def _analyze(self, x):
        """静解析を行い、感度計算に必要な要素データを返す。"""
        self._apply_x(x)
        m = self.model
        n = m.n_dof
        dof_maps = element_dof_map(m)

        # 要素ごとの局所剛性・変換・微分を構築しつつ全体行列を組む。
        # オフセット梁は剛体腕 G を含め、B = T @ G を「変換」として保持する
        # （K_node = G^T (T^T k T) G = B^T k B）。以降の感度・応力回収は B を
        # 用いるだけでオフセットに整合する。
        rows, cols, data = [], [], []
        elem_data = []
        for e, el in enumerate(m.elements):
            p1, p2 = m.nodes[el.n1], m.nodes[el.n2]
            L = float(np.linalg.norm(p2 - p1))
            klocal = local_stiffness(el.mat.E, el.mat.G, L, el.sec)
            R = rotation_matrix(p1, p2, el.vref)
            T = transformation_matrix(R)
            G = rigid_offset_matrix(el.offset)
            B = T if G is None else T @ G
            kg = B.T @ klocal @ B
            dofs = dof_maps[e]
            rr, cc = np.meshgrid(dofs, dofs, indexing="ij")
            rows.append(rr.ravel())
            cols.append(cc.ravel())
            data.append(kg.ravel())
            elem_data.append(dict(L=L, klocal=klocal, T=B, dofs=dofs, mat=el.mat, sec=el.sec))

        # シェル要素（板）は固定剛性として全体行列に加える（設計変数ではない）
        if m.shells:
            for s, sdofs in zip(m.shells, shell_dof_map(m)):
                ks = shell_stiffness_global(
                    m.nodes[s.n1], m.nodes[s.n2], m.nodes[s.n3], s.mat, s.thickness
                )
                rr, cc = np.meshgrid(sdofs, sdofs, indexing="ij")
                rows.append(rr.ravel())
                cols.append(cc.ravel())
                data.append(ks.ravel())
        if m.quad_shells:
            for s, sdofs in zip(m.quad_shells, quad_shell_dof_map(m)):
                ks = quad_shell_stiffness_global(
                    m.nodes[s.n1], m.nodes[s.n2], m.nodes[s.n3], m.nodes[s.n4],
                    s.mat, s.thickness,
                )
                rr, cc = np.meshgrid(sdofs, sdofs, indexing="ij")
                rows.append(rr.ravel())
                cols.append(cc.ravel())
                data.append(ks.ravel())

        K = sp.coo_matrix(
            (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n, n),
        ).tocsr()
        F = assemble_load_vector(m)

        # 境界条件（同次のみ想定）
        c_idx = np.array(
            [nd * DOF_PER_NODE + d for (nd, d) in m.constraints.keys()], dtype=np.int64
        )
        free = np.setdiff1d(np.arange(n), c_idx)
        Kff = K[free][:, free].tocsc()
        lu = spla.splu(Kff)
        u = np.zeros(n)
        u[free] = lu.solve(F[free])

        return dict(K=K, u=u, lu=lu, free=free, elem=elem_data, dof_maps=dof_maps)

    def _du_dx(self, state):
        """全設計変数についての du/dx を (n_var, n_dof) で返す（直接法）。"""
        m = self.model
        n = m.n_dof
        u = state["u"]
        free = state["free"]
        lu = state["lu"]
        du = np.zeros((self.n_var, n))

        for i, dv in enumerate(self.design_vars):
            dprop = dv.family.derivs(float(self._cur_x[i]))
            rhs = np.zeros(n)  # (dK/dx_i) u
            for e in dv.elements:
                ed = state["elem"][e]
                dk = self._dk_local(ed, dprop)  # dk_local/dx_i
                dkg = ed["T"].T @ dk @ ed["T"]
                dofs = ed["dofs"]
                rhs[dofs] += dkg @ u[dofs]
            # K du = -(dK/dx) u
            du[i, free] = -lu.solve(rhs[free])
        return du

    @staticmethod
    def _dk_local(ed, dprop):
        """dk_local/dx = Σ_prop ∂k/∂prop · dprop/dx。"""
        d = local_stiffness_derivs(ed["mat"].E, ed["mat"].G, ed["L"], ed["sec"])
        return (
            d["A"] * dprop["A"]
            + d["Iy"] * dprop["Iy"]
            + d["Iz"] * dprop["Iz"]
            + d["J"] * dprop["J"]
        )

    # ------------------------------------------------------------------
    def _mass_and_grad(self, x):
        m = self.model
        W = 0.0
        dW = np.zeros(self.n_var)
        for i, dv in enumerate(self.design_vars):
            dA = dv.family.derivs(float(x[i]))["A"]
            for e in dv.elements:
                el = m.elements[e]
                L = float(np.linalg.norm(m.nodes[el.n2] - m.nodes[el.n1]))
                rho = el.mat.rho
                W += rho * L * el.sec.A
                dW[i] += rho * L * dA
        return W, dW

    @staticmethod
    def _sigma_and_grad(f_local, sec):
        """要素の最大合成応力 σ と、∂σ/∂f_local(12,)・∂σ/∂(A,Iy,Iz,cy,cz) を返す。"""
        A, Iy, Iz = sec.A, sec.Iy, sec.Iz
        cy = sec.cy if sec.cy is not None else 0.0
        cz = sec.cz if sec.cz is not None else 0.0
        f = f_local

        # 両端の σ
        def end_sigma(iN, iMy, iMz):
            return (
                abs(f[iN]) / A
                + (cy / Iz) * abs(f[iMz])
                + (cz / Iy) * abs(f[iMy])
            )

        s1 = end_sigma(0, 4, 5)
        s2 = end_sigma(6, 10, 11)
        if s1 >= s2:
            iN, iMy, iMz = 0, 4, 5
            sigma = s1
        else:
            iN, iMy, iMz = 6, 10, 11
            sigma = s2

        dsdf = np.zeros(12)
        dsdf[iN] = np.sign(f[iN]) / A
        dsdf[iMz] = np.sign(f[iMz]) * cy / Iz
        dsdf[iMy] = np.sign(f[iMy]) * cz / Iy

        dprops = dict(
            A=-abs(f[iN]) / A**2,
            Iz=-cy * abs(f[iMz]) / Iz**2,
            Iy=-cz * abs(f[iMy]) / Iy**2,
            cy=abs(f[iMz]) / Iz,
            cz=abs(f[iMy]) / Iy,
        )
        return sigma, dsdf, dprops

    # ------------------------------------------------------------------
    def evaluate(self, x):
        """MMA 用に f0, df0, f(制約,m), dfdx(m,n) を返す。"""
        x = np.asarray(x, dtype=float)
        self._cur_x = x
        state = self._analyze(x)
        u = state["u"]
        du = self._du_dx(state)  # (n_var, n_dof)

        # --- 目的（質量） ---
        f0, df0 = self._mass_and_grad(x)

        cons = []      # 制約値（≤0 で満足）
        dcons = []     # 各制約の勾配 (n_var,)

        # --- 応力制約（要素ごと） ---
        for e, el in enumerate(self.model.elements):
            sa = self._sigma_allow_of(e)
            if not np.isfinite(sa):
                continue
            ed = state["elem"][e]
            T, dofs, klocal = ed["T"], ed["dofs"], ed["klocal"]
            f_local = klocal @ (T @ u[dofs])
            sigma, dsdf, dsp = self._sigma_and_grad(f_local, el.sec)

            g = sigma / sa - 1.0
            dg = np.zeros(self.n_var)
            ivar = self._elem_of_var.get(e, None)
            for i, dv in enumerate(self.design_vars):
                dprop = dv.family.derivs(float(x[i]))
                # df_local/dx_i = (dk_local/dx_i) T u_e [eが支配下] + klocal T du_e
                df = klocal @ (T @ du[i, dofs])
                if e in dv.elements:
                    dk = self._dk_local(ed, dprop)
                    df = df + dk @ (T @ u[dofs])
                dg[i] = dsdf @ df
                # 断面変化による陽な項（e が i に支配される場合）
                if ivar == i:
                    dg[i] += (
                        dsp["A"] * dprop["A"]
                        + dsp["Iy"] * dprop["Iy"]
                        + dsp["Iz"] * dprop["Iz"]
                        + dsp["cy"] * dv.family.deriv_cy(float(x[i]))
                        + dsp["cz"] * dv.family.deriv_cz(float(x[i]))
                    )
            cons.append(g / 1.0)
            dcons.append(dg / sa)

        # --- たわみ制約（両側 |u_j| ≤ limit） ---
        for dl in self.disp_limits:
            j = dl.node * DOF_PER_NODE + dl.dof
            uj = u[j]
            duj = du[:, j]  # (n_var,)
            # u_j/limit - 1 ≤ 0  と  -u_j/limit - 1 ≤ 0
            cons.append(uj / dl.limit - 1.0)
            dcons.append(duj / dl.limit)
            cons.append(-uj / dl.limit - 1.0)
            dcons.append(-duj / dl.limit)

        f = np.array(cons)
        dfdx = np.array(dcons) if dcons else np.zeros((0, self.n_var))
        return f0, df0, f, dfdx

    def evaluate_values(self, x):
        """目的 f0 と制約 f のみを返す（感度なし・高速）。離散最適化の関数評価用。"""
        x = np.asarray(x, dtype=float)
        self._cur_x = x
        state = self._analyze(x)
        u = state["u"]
        f0, _ = self._mass_and_grad(x)

        cons = []
        for e, el in enumerate(self.model.elements):
            sa = self._sigma_allow_of(e)
            if not np.isfinite(sa):
                continue
            ed = state["elem"][e]
            f_local = ed["klocal"] @ (ed["T"] @ u[ed["dofs"]])
            sigma, _, _ = self._sigma_and_grad(f_local, el.sec)
            cons.append(sigma / sa - 1.0)
        for dl in self.disp_limits:
            j = dl.node * DOF_PER_NODE + dl.dof
            cons.append(u[j] / dl.limit - 1.0)
            cons.append(-u[j] / dl.limit - 1.0)
        return f0, np.array(cons)

    def current_sections(self, x):
        """設計変数 x に対応する各設計グループの断面を返す。"""
        return {i: dv.family.make(float(x[i])) for i, dv in enumerate(self.design_vars)}

    def element_scales(self, x) -> np.ndarray:
        """各要素のスケール係数（属する設計変数の値）。非支配要素は 1.0。"""
        sc = np.ones(len(self.model.elements))
        for i, dv in enumerate(self.design_vars):
            for e in dv.elements:
                sc[e] = float(x[i])
        return sc

    def element_values(self, x, kind: str = "area") -> np.ndarray:
        """構造形態の図示用に、各要素の代表量を返す。

        kind : "area"(断面積) / "scale"(スケール係数) / "size"(代表寸法 √A)。
        """
        if kind == "scale":
            return self.element_scales(x)
        self._apply_x(x)
        A = np.array([el.sec.A for el in self.model.elements])
        if kind == "area":
            return A
        if kind == "size":
            return np.sqrt(A)
        raise ValueError(f"未知の kind: {kind}")
