"""片持ち梁の解析解との比較検証。

Timoshenko 梁の厳密な要素剛性を使えば、先端集中荷重の片持ち梁は
1要素でも先端たわみが解析解と一致する::

    delta = P L^3 / (3 E I) + P L / (k G A)   (曲げ + せん断)
"""

import numpy as np

from beamfem import Material, Section, Model, solve_static, UX, UY, UZ, RX, RY, RZ


STEEL = Material(E=200e9, nu=0.3, rho=7850.0, name="steel")


def analytic_tip(P, L, E, I, G, A, k):
    return P * L**3 / (3.0 * E * I) + P * L / (k * G * A)


def test_cantilever_xy_bending():
    """x-y 面内曲げ（局所z軸まわり, Iz）。"""
    L = 2.0
    sec = Section.rectangle(b=0.05, h=0.10)  # b:z方向, h:y方向
    P = 1000.0

    m = Model()
    n0 = m.add_node(0, 0, 0)
    n1 = m.add_node(L, 0, 0)
    m.add_element(n0, n1, STEEL, sec)
    m.fix(n0)  # 完全固定
    m.add_load(n1, UY, -P)  # y方向にP

    res = solve_static(m)
    uy = res.node_disp(n1)[UY]

    expected = -analytic_tip(P, L, STEEL.E, sec.Iz, STEEL.G, sec.A, sec.ky)
    assert np.isclose(uy, expected, rtol=1e-9), (uy, expected)


def test_cantilever_xz_bending():
    """x-z 面外曲げ（局所y軸まわり, Iy）。"""
    L = 2.0
    sec = Section.rectangle(b=0.05, h=0.10)
    P = 1000.0

    m = Model()
    n0 = m.add_node(0, 0, 0)
    n1 = m.add_node(L, 0, 0)
    m.add_element(n0, n1, STEEL, sec)
    m.fix(n0)
    m.add_load(n1, UZ, -P)

    res = solve_static(m)
    uz = res.node_disp(n1)[UZ]

    expected = -analytic_tip(P, L, STEEL.E, sec.Iy, STEEL.G, sec.A, sec.kz)
    assert np.isclose(uz, expected, rtol=1e-9), (uz, expected)


def test_axial():
    """軸方向伸び delta = P L / (E A)。"""
    L = 3.0
    sec = Section.circle(d=0.02)
    P = 5000.0

    m = Model()
    n0 = m.add_node(0, 0, 0)
    n1 = m.add_node(L, 0, 0)
    m.add_element(n0, n1, STEEL, sec)
    m.fix(n0)
    m.add_load(n1, UX, P)

    res = solve_static(m)
    ux = res.node_disp(n1)[UX]
    expected = P * L / (STEEL.E * sec.A)
    assert np.isclose(ux, expected, rtol=1e-12), (ux, expected)


def test_mesh_convergence_matches_single_element():
    """複数要素に分割しても先端たわみは不変（厳密要素の確認）。"""
    L = 2.0
    sec = Section.rectangle(b=0.05, h=0.10)
    P = 1000.0

    def tip_with_n(nel):
        m = Model()
        nodes = [m.add_node(L * i / nel, 0, 0) for i in range(nel + 1)]
        for i in range(nel):
            m.add_element(nodes[i], nodes[i + 1], STEEL, sec)
        m.fix(nodes[0])
        m.add_load(nodes[-1], UY, -P)
        return solve_static(m).node_disp(nodes[-1])[UY]

    one = tip_with_n(1)
    many = tip_with_n(10)
    assert np.isclose(one, many, rtol=1e-9), (one, many)


def test_reactions_equilibrium():
    """支点反力が外力と釣り合う。"""
    L = 2.0
    sec = Section.rectangle(b=0.05, h=0.10)
    P = 1000.0

    m = Model()
    n0 = m.add_node(0, 0, 0)
    n1 = m.add_node(L, 0, 0)
    m.add_element(n0, n1, STEEL, sec)
    m.fix(n0)
    m.add_load(n1, UY, -P)

    res = solve_static(m)
    # 反力のy成分 = +P、固定端モーメント(z) = P*L
    assert np.isclose(res.reactions[UY], P, rtol=1e-9)
    assert np.isclose(res.reactions[RZ], P * L, rtol=1e-9)
