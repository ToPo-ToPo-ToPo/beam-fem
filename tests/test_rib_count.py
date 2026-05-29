"""リブ本数最適化（構成列挙＋サイジング）の基礎検証。

各構成でサイジング最適化が実行可能解を返すこと、および本数を増やすと
（純グリラージュモデルでは）最適質量が増えるという傾向を確認する。
"""

import numpy as np

from beamfem import Material, Section, Model, radial_grillage, lump_pressure, UZ
from beamfem.optimize import SizingProblem, DesignVar, DispLimit, ScaledSection, minimize_mass

STEEL = Material(E=200e9, nu=0.3, rho=7850.0)
BASE = Section.circle(d=0.03)


def _optimize_config(n_radial, n_rings, R=2.0, q=3000.0):
    m = Model()
    g = radial_grillage(m, STEEL, BASE, R, n_radial, n_rings)
    for j in range(n_radial):
        m.fix(g.ring_nodes[n_rings][j])
    lump_pressure(m, g.triangles, q)
    dvs = [DesignVar(ScaledSection(BASE), g.radial_bands[b], x0=1.0, xmin=0.15, xmax=6.0)
           for b in range(n_rings)]
    dvs += [DesignVar(ScaledSection(BASE), g.rings[k], x0=1.0, xmin=0.15, xmax=6.0)
            for k in range(1, n_rings + 1)]
    dl = [DispLimit(g.center, UZ, 0.012)]
    dl += [DispLimit(g.ring_nodes[k][0], UZ, 0.012) for k in range(1, n_rings)]
    prob = SizingProblem(m, dvs, sigma_allow=150e6, disp_limits=dl)
    res = minimize_mass(prob, maxiter=120, move=0.15, tol=1e-6)
    return res


def test_configs_feasible_and_count_trend():
    """異なるリブ本数で最適化でき、本数増で質量が増える傾向。"""
    r6 = _optimize_config(6, 2)
    r10 = _optimize_config(10, 2)
    assert r6.constraints.max() <= 1e-3
    assert r10.constraints.max() <= 1e-3
    # 放射本数が多い方が（同条件で）重い
    assert r10.mass > r6.mass


def test_more_rings_not_heavier_at_fixed_radial():
    """同じ放射本数ならリング追加で質量は概ね減る（or 同等）。"""
    r_nk2 = _optimize_config(8, 2)
    r_nk4 = _optimize_config(8, 4)
    assert r_nk2.constraints.max() <= 1e-3
    assert r_nk4.constraints.max() <= 1e-3
    assert r_nk4.mass <= r_nk2.mass * 1.02  # リング追加で重くはならない
