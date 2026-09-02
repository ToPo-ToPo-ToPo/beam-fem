"""離散サイジング最適化の検証。"""

import numpy as np
import pytest

from beamfem import Material, Section, Model, UX, UY, UZ, RZ
from beamfem.optimize import (
    SizingProblem, DesignVar, DispLimit, ScaledSection,
    solve_discrete_exhaustive, solve_discrete_greedy,
)

STEEL = Material(E=200e9, nu=0.3, rho=7850.0)


def _ribbed_shell_problem():
    """シェル板＋オフセット梁リブ（片持ち板）の小規模な離散サイジング問題。"""
    t = 0.01
    rib = Section.rectangle(b=0.006, h=0.030)
    e = t / 2 + 0.030 / 2
    vref = [0.0, 0.0, 1.0]
    m = Model()
    n00 = m.add_node(0.0, 0.0, 0.0)
    n10 = m.add_node(1.0, 0.0, 0.0)
    n11 = m.add_node(1.0, 1.0, 0.0)
    n01 = m.add_node(0.0, 1.0, 0.0)
    m.add_shell(n00, n10, n11, STEEL, t)
    m.add_shell(n00, n11, n01, STEEL, t)
    r0 = m.add_element(n00, n10, STEEL, rib, vref=vref, offset=[0, 0, -e])
    r1 = m.add_element(n00, n11, STEEL, rib, vref=vref, offset=[0, 0, -e])
    for nd in (n00, n10, n11, n01):
        m.fix(nd, [RZ])
    m.fix(n00)
    m.fix(n01)
    m.add_load(n11, UZ, -3000.0)
    m.add_load(n10, UZ, -1500.0)
    dvs = [
        DesignVar(ScaledSection(rib), [r0], x0=1.5, xmin=0.5, xmax=3.0),
        DesignVar(ScaledSection(rib), [r1], x0=1.5, xmin=0.5, xmax=3.0),
    ]
    return SizingProblem(m, dvs, sigma_allow=1e9,
                         disp_limits=[DispLimit(n11, UZ, 0.025)])


def _indeterminate_problem():
    base = Section.i_section(h=0.2, bf=0.1, tf=0.01, tw=0.006)
    m = Model()
    nodes = [m.add_node(i * 1.0, 0, 0) for i in range(6)]
    for i in range(5):
        m.add_element(nodes[i], nodes[i + 1], STEEL, base)
    m.fix(nodes[0])
    m.fix(nodes[-1])
    m.add_load(nodes[2], UZ, -12000.0)
    m.add_load(nodes[3], UZ, -9000.0)
    dvs = [DesignVar(ScaledSection(base), g, x0=1.5, xmin=0.3, xmax=4.0)
           for g in ([0, 1], [2], [3, 4])]
    return SizingProblem(
        m, dvs, sigma_allow=180e6,
        disp_limits=[DispLimit(nodes[2], UZ, 0.006), DispLimit(nodes[3], UZ, 0.006)],
    )


def test_evaluate_values_matches_evaluate():
    prob = _indeterminate_problem()
    x = np.array([1.2, 1.6, 1.4])
    f0a, _, fa, _ = prob.evaluate(x)
    f0b, fb = prob.evaluate_values(x)
    assert np.isclose(f0a, f0b)
    assert np.allclose(fa, fb)


def test_greedy_matches_exhaustive_global_optimum():
    """貪欲解が総当たり（大域最適）と一致し、評価回数は少ない。"""
    prob = _indeterminate_problem()
    catalog = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    ex = solve_discrete_exhaustive(prob, catalog)
    gr = solve_discrete_greedy(prob, catalog)
    assert gr.indices == ex.indices
    assert np.isclose(gr.mass, ex.mass)
    assert gr.n_eval < ex.n_eval


def test_discrete_feasible_and_above_continuous():
    """離散解は実行可能で、連続最適（下界）を下回らない。"""
    from beamfem.optimize import minimize_mass

    prob = _indeterminate_problem()
    cont = minimize_mass(prob, maxiter=80, move=0.2, tol=1e-6)
    catalog = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    res = solve_discrete_greedy(prob, catalog)
    assert res.feasible
    assert res.constraints.max() <= 1e-6
    assert res.mass >= cont.mass - 1e-9  # 離散 ≥ 連続


def test_discrete_values_are_from_catalog():
    prob = _indeterminate_problem()
    catalog = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    res = solve_discrete_greedy(prob, catalog)
    for v in res.x:
        assert np.min(np.abs(np.array(catalog) - v)) < 1e-12


def test_per_variable_catalog():
    """変数ごとに異なるカタログを与えられる。"""
    prob = _indeterminate_problem()
    catalogs = [[1.0, 2.0, 3.0], [0.5, 1.5, 2.5], [1.0, 2.0]]
    res = solve_discrete_exhaustive(prob, catalogs)
    assert res.x[0] in (1.0, 2.0, 3.0)
    assert res.x[2] in (1.0, 2.0)


def test_discrete_ribbed_shell_offset_greedy_matches_exhaustive():
    """シェル板＋オフセットリブの離散サイジングで貪欲解が大域最適と一致。"""
    prob = _ribbed_shell_problem()
    catalog = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    ex = solve_discrete_exhaustive(prob, catalog)
    gr = solve_discrete_greedy(prob, catalog)
    assert ex.feasible and gr.feasible
    assert gr.indices == ex.indices
    assert np.isclose(gr.mass, ex.mass)
    assert gr.constraints.max() <= 1e-6


def _ribbed_quad_problem():
    """四角形 MITC4 シェル板＋オフセットリブ（片持ち板）の離散サイジング問題。"""
    t = 0.01
    rib = Section.rectangle(b=0.006, h=0.030)
    e = t / 2 + 0.030 / 2
    vref = [0.0, 0.0, 1.0]
    m = Model()
    n00 = m.add_node(0.0, 0.0, 0.0)
    n10 = m.add_node(1.0, 0.0, 0.0)
    n11 = m.add_node(1.0, 1.0, 0.0)
    n01 = m.add_node(0.0, 1.0, 0.0)
    m.add_quad_shell(n00, n10, n11, n01, STEEL, t)
    r0 = m.add_element(n00, n10, STEEL, rib, vref=vref, offset=[0, 0, -e])
    r1 = m.add_element(n00, n11, STEEL, rib, vref=vref, offset=[0, 0, -e])
    for nd in (n00, n10, n11, n01):
        m.fix(nd, [RZ])
    m.fix(n00)
    m.fix(n01)
    m.add_load(n11, UZ, -3000.0)
    m.add_load(n10, UZ, -1500.0)
    dvs = [
        DesignVar(ScaledSection(rib), [r0], x0=1.5, xmin=0.5, xmax=3.0),
        DesignVar(ScaledSection(rib), [r1], x0=1.5, xmin=0.5, xmax=3.0),
    ]
    return SizingProblem(m, dvs, sigma_allow=1e9,
                         disp_limits=[DispLimit(n11, UZ, 0.020)])


def test_discrete_quad_shell_offset_greedy_matches_exhaustive():
    """四角形シェル板＋オフセットリブの離散サイジングで貪欲解が大域最適と一致。"""
    prob = _ribbed_quad_problem()
    catalog = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    ex = solve_discrete_exhaustive(prob, catalog)
    gr = solve_discrete_greedy(prob, catalog)
    assert ex.feasible and gr.feasible
    assert gr.indices == ex.indices
    assert np.isclose(gr.mass, ex.mass)
    assert gr.constraints.max() <= 1e-6


def test_exhaustive_too_many_combos_raises():
    prob = _indeterminate_problem()
    with pytest.raises(ValueError):
        solve_discrete_exhaustive(prob, [0.5, 1.0, 1.5, 2.0], max_combos=10)
