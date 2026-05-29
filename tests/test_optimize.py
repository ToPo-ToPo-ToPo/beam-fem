"""断面サイジング最適化の検証。

- 解析的感度 vs 有限差分（不静定構造）
- MMA vs 解析的最適解（応力・たわみ制約）
- MMA vs SLSQP（多変数・複合制約）
"""

import numpy as np
import pytest

from beamfem import Material, Section, Model, UY, UZ, RZ
from beamfem.optimize import (
    SizingProblem,
    DesignVar,
    DispLimit,
    ScaledSection,
    minimize_mass,
)

STEEL = Material(E=200e9, nu=0.3, rho=7850.0)


def _cantilever(base, P=3000.0, L=2.0, x0=2.0):
    m = Model()
    a = m.add_node(0, 0, 0)
    b = m.add_node(L, 0, 0)
    m.add_element(a, b, STEEL, base)
    m.fix(a)
    m.add_load(b, UY, -P)
    return m, b


def test_sensitivity_matches_fd():
    """解析的勾配が有限差分と一致（不静定・応力+たわみ）。"""
    base = Section.i_section(h=0.2, bf=0.1, tf=0.01, tw=0.006)
    m = Model()
    nodes = [m.add_node(i * 1.0, 0, 0) for i in range(4)]
    for i in range(3):
        m.add_element(nodes[i], nodes[i + 1], STEEL, base)
    m.fix(nodes[0])
    m.fix(nodes[-1])
    m.add_load(nodes[1], UZ, -8000.0)
    m.add_load(nodes[2], UZ, -5000.0)
    dvs = [
        DesignVar(ScaledSection(base), [0, 1], x0=1.2),
        DesignVar(ScaledSection(base), [2], x0=0.9),
    ]
    prob = SizingProblem(
        m, dvs, sigma_allow=200e6,
        disp_limits=[DispLimit(nodes[1], UZ, 0.01), DispLimit(nodes[2], UZ, 0.01)],
    )
    x = np.array([1.2, 0.9])
    f0, df0, f, dfdx = prob.evaluate(x)

    h = 1e-6
    for i in range(prob.n_var):
        xp, xm = x.copy(), x.copy()
        xp[i] += h
        xm[i] -= h
        f0p, _, fp, _ = prob.evaluate(xp)
        f0m, _, fm, _ = prob.evaluate(xm)
        assert np.isclose(df0[i], (f0p - f0m) / (2 * h), rtol=1e-6)
        assert np.allclose(dfdx[:, i], (fp - fm) / (2 * h), rtol=1e-5, atol=1e-8)


def test_sensitivity_ribbed_shell_offset_matches_fd():
    """シェル板＋オフセットリブ（梁）の連成でも解析的勾配が有限差分と一致。"""
    t = 0.01
    rib = Section.rectangle(b=0.006, h=0.030)
    e = t / 2 + 0.030 / 2
    vref = [0.0, 0.0, 1.0]

    m = Model()
    n00 = m.add_node(0.0, 0.0, 0.0)
    n10 = m.add_node(1.0, 0.0, 0.0)
    n11 = m.add_node(1.0, 1.0, 0.0)
    n01 = m.add_node(0.0, 1.0, 0.0)
    # 板（シェル2枚）
    m.add_shell(n00, n10, n11, STEEL, t)
    m.add_shell(n00, n11, n01, STEEL, t)
    # オフセットリブ（梁）: 下辺と対角。シェルと節点共有
    r0 = m.add_element(n00, n10, STEEL, rib, vref=vref, offset=[0, 0, -e])
    r1 = m.add_element(n00, n11, STEEL, rib, vref=vref, offset=[0, 0, -e])
    # 左辺を完全固定（片持ち板）、面内ドリリングは全節点拘束
    for nd in (n00, n01, n10, n11):
        m.fix(nd, [RZ])  # ドリリングは全節点拘束
    m.fix(n00)
    m.fix(n01)
    m.add_load(n11, UZ, -2000.0)
    m.add_load(n10, UZ, -1200.0)

    dvs = [
        DesignVar(ScaledSection(rib), [r0], x0=1.1),
        DesignVar(ScaledSection(rib), [r1], x0=0.9),
    ]
    prob = SizingProblem(
        m, dvs, sigma_allow=200e6,
        disp_limits=[DispLimit(n11, UZ, 0.05)],
    )
    x = np.array([1.1, 0.9])
    f0, df0, f, dfdx = prob.evaluate(x)

    h = 1e-6
    for i in range(prob.n_var):
        xp, xm = x.copy(), x.copy()
        xp[i] += h
        xm[i] -= h
        f0p, _, fp, _ = prob.evaluate(xp)
        f0m, _, fm, _ = prob.evaluate(xm)
        assert np.isclose(df0[i], (f0p - f0m) / (2 * h), rtol=1e-6)
        assert np.allclose(dfdx[:, i], (fp - fm) / (2 * h), rtol=1e-5, atol=1e-8)


def test_mma_stress_constrained_analytic():
    """静定片持ち・応力制約の解析的最適スケールに一致。"""
    base = Section.rectangle(b=0.05, h=0.10)
    P, L, sa = 3000.0, 2.0, 200e6
    s_star = (P * L * base.cy / (base.Iz * sa)) ** (1.0 / 3.0)
    m, tip = _cantilever(base, P, L)
    prob = SizingProblem(m, [DesignVar(ScaledSection(base), [0], x0=2.0)], sigma_allow=sa)
    res = minimize_mass(prob, maxiter=80, move=0.2, tol=1e-7)
    assert res.converged
    assert np.isclose(res.x[0], s_star, rtol=1e-5)
    assert res.constraints.max() < 1e-6  # 制約満足


def test_mma_deflection_constrained_analytic():
    """静定片持ち・たわみ制約の解析的最適スケールに一致。"""
    from scipy.optimize import brentq

    base = Section.rectangle(b=0.05, h=0.10)
    P, L, dmax = 3000.0, 2.0, 0.01

    def delta(s):
        return P * L**3 / (3 * STEEL.E * base.Iz * s**4) + P * L / (
            base.ky * STEEL.G * base.A * s**2
        )

    s_star = brentq(lambda s: delta(s) - dmax, 0.1, 10)
    m, tip = _cantilever(base, P, L, x0=2.5)
    prob = SizingProblem(
        m, [DesignVar(ScaledSection(base), [0], x0=2.5, xmax=8.0)],
        disp_limits=[DispLimit(tip, UY, dmax)],
    )
    res = minimize_mass(prob, maxiter=120, move=0.15, tol=1e-8)
    assert np.isclose(res.x[0], s_star, rtol=1e-5)


def test_mma_matches_slsqp_multivar():
    """多変数・複合制約で MMA と SLSQP が一致。"""
    from scipy.optimize import minimize

    base = Section.i_section(h=0.2, bf=0.1, tf=0.01, tw=0.006)
    m = Model()
    nodes = [m.add_node(i * 1.0, 0, 0) for i in range(6)]
    for i in range(5):
        m.add_element(nodes[i], nodes[i + 1], STEEL, base)
    m.fix(nodes[0])
    m.fix(nodes[-1])
    m.add_load(nodes[2], UZ, -12000.0)
    m.add_load(nodes[3], UZ, -9000.0)
    dvs = [DesignVar(ScaledSection(base), g, x0=1.5) for g in ([0, 1], [2], [3, 4])]
    prob = SizingProblem(
        m, dvs, sigma_allow=180e6,
        disp_limits=[DispLimit(nodes[2], UZ, 0.006), DispLimit(nodes[3], UZ, 0.006)],
    )
    res = minimize_mass(prob, maxiter=150, move=0.15, tol=1e-8)

    xmin, xmax = prob.bounds()
    sol = minimize(
        lambda x: prob.evaluate(x)[0],
        prob.x0(),
        jac=lambda x: prob.evaluate(x)[1],
        bounds=list(zip(xmin, xmax)),
        constraints=[{
            "type": "ineq",
            "fun": lambda x: -prob.evaluate(x)[2],
            "jac": lambda x: -prob.evaluate(x)[3],
        }],
        method="SLSQP",
        options={"maxiter": 200, "ftol": 1e-12},
    )
    assert np.allclose(res.x, sol.x, rtol=1e-4)
    assert np.isclose(res.mass, sol.fun, rtol=1e-5)


def test_element_values_for_form_plot():
    """構造形態図示用の要素別代表量。"""
    base = Section.rectangle(b=0.05, h=0.10)
    m = Model()
    nodes = [m.add_node(i, 0, 0) for i in range(4)]
    for i in range(3):
        m.add_element(nodes[i], nodes[i + 1], STEEL, base)
    m.fix(nodes[0])
    dvs = [
        DesignVar(ScaledSection(base), [0, 1], x0=1.0),
        DesignVar(ScaledSection(base), [2], x0=1.0),
    ]
    prob = SizingProblem(m, dvs)
    x = np.array([2.0, 0.5])
    sc = prob.element_scales(x)
    assert np.allclose(sc, [2.0, 2.0, 0.5])
    area = prob.element_values(x, kind="area")
    assert np.isclose(area[0], base.A * 2.0**2)  # A ∝ s^2
    assert np.isclose(area[2], base.A * 0.5**2)
    assert np.allclose(prob.element_values(x, kind="size"), np.sqrt(area))


def test_duplicate_element_assignment_raises():
    base = Section.circle(d=0.05)
    m, tip = _cantilever(base)
    m.add_element(0, 1, STEEL, base)  # 2要素目
    with pytest.raises(ValueError):
        SizingProblem(
            m,
            [
                DesignVar(ScaledSection(base), [0]),
                DesignVar(ScaledSection(base), [0]),  # 重複
            ],
        )
