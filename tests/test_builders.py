"""グリラージュ生成と圧力の等価節点化の検証。"""

import numpy as np

from beamfem import Material, Section, Model, solve_static, radial_grillage, lump_pressure, UZ
from beamfem.model import DOF_PER_NODE

STEEL = Material(E=200e9, nu=0.3, rho=7850.0)


def test_radial_grillage_counts():
    m = Model()
    nr, nk = 8, 4
    g = radial_grillage(m, STEEL, Section.circle(d=0.03), R=2.0, n_radial=nr, n_rings=nk)
    # 節点: 中心1 + nr*nk
    assert m.n_nodes == 1 + nr * nk
    # 要素: 放射 nr*nk + 周方向 nr*nk
    assert len(m.elements) == nr * nk + nr * nk
    # 設計グループ被覆: 全要素がちょうど1グループに入る
    covered = [e for band in g.radial_bands for e in band] + [
        e for k in range(1, nk + 1) for e in g.rings[k]
    ]
    assert sorted(covered) == list(range(len(m.elements)))


def test_lump_pressure_total_and_equilibrium():
    m = Model()
    nr, nk = 8, 4
    q = 2500.0
    g = radial_grillage(m, STEEL, Section.circle(d=0.03), R=2.0, n_radial=nr, n_rings=nk)
    for j in range(nr):
        m.fix(g.ring_nodes[nk][j])
    total = lump_pressure(m, g.triangles, q)

    # 総載荷 = q × 内接多角形面積
    poly_area = 0.5 * nr * np.sin(2 * np.pi / nr) * 2.0**2
    assert np.isclose(total, q * poly_area, rtol=1e-9)

    # 鉛直つり合い: 外周反力合計 = 総載荷
    res = solve_static(m)
    rz = sum(res.reactions[g.ring_nodes[nk][j] * DOF_PER_NODE + UZ] for j in range(nr))
    assert np.isclose(rz, total, rtol=1e-6)


def test_lump_pressure_direction():
    """既定は下向き（-z）。"""
    m = Model()
    g = radial_grillage(m, STEEL, Section.circle(d=0.03), R=1.0, n_radial=6, n_rings=2)
    lump_pressure(m, g.triangles, 1000.0)
    # 中心節点の z 荷重は負
    assert m.nodal_loads[(g.center, UZ)] < 0
