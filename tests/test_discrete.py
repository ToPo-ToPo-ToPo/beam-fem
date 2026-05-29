"""離散サイジング最適化の検証。"""

import numpy as np
import pytest

from beamfem import Material, Section, Model, UZ
from beamfem.optimize import (
    SizingProblem, DesignVar, DispLimit, ScaledSection,
    solve_discrete_exhaustive, solve_discrete_greedy,
)

STEEL = Material(E=200e9, nu=0.3, rho=7850.0)


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


def test_exhaustive_too_many_combos_raises():
    prob = _indeterminate_problem()
    with pytest.raises(ValueError):
        solve_discrete_exhaustive(prob, [0.5, 1.0, 1.5, 2.0], max_combos=10)
