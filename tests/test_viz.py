"""図示機能のスモークテスト（ヘッドレス）。

描画が例外なく実行でき、要素数に応じた線が描かれることを確認する。
matplotlib が無い環境ではスキップする。
"""

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")  # 表示なし

from beamfem import Material, Section, Model, solve_static, UX, UZ  # noqa: E402
from beamfem import viz  # noqa: E402

STEEL = Material(E=200e9, nu=0.3)


def _portal_2d():
    col = Section.rectangle(b=0.2, h=0.3)
    beam = Section.rectangle(b=0.2, h=0.4)
    m = Model()
    n0 = m.add_node(0, 0)
    n1 = m.add_node(0, 3)
    n2 = m.add_node(5, 3)
    n3 = m.add_node(5, 0)
    m.add_element(n0, n1, STEEL, col)
    m.add_element(n1, n2, STEEL, beam)
    m.add_element(n2, n3, STEEL, col)
    m.fix(n0)
    m.fix(n3)
    m.fix_to_plane_xy()
    m.add_load(n1, UX, 10000.0)
    return m


def test_plot_model_2d():
    m = _portal_2d()
    fig, ax = viz.plot_model(m)
    assert len(ax.lines) == len(m.elements)  # 各要素1本
    matplotlib.pyplot.close(fig)


def test_plot_deformed_2d_planar():
    m = _portal_2d()
    res = solve_static(m)
    assert viz._is_planar(m, res) is True
    fig, ax = viz.plot_deformed(m, res, scale="auto")
    # 各要素: 変形曲線 + 無変形破線 = 2本
    assert len(ax.lines) == 2 * len(m.elements)
    matplotlib.pyplot.close(fig)


def test_plot_deformed_3d():
    sec = Section.circle(d=0.05)
    m = Model()
    a = m.add_node(0, 0, 0)
    b = m.add_node(2, 0, 0)
    c = m.add_node(2, 1.5, 0)
    m.add_element(a, b, STEEL, sec)
    m.add_element(b, c, STEEL, sec)
    m.fix(a)
    m.add_load(c, UZ, -500.0)  # 面外荷重で 3D 判定になる
    res = solve_static(m)
    assert viz._is_planar(m, res) is False
    fig, ax = viz.plot_deformed(m, res)
    matplotlib.pyplot.close(fig)


def test_plot_diagram_2d():
    from beamfem import recover_forces

    m = _portal_2d()
    fr = recover_forces(m, solve_static(m))
    fig, ax = viz.plot_diagram(fr, "Mz", scale="auto")
    matplotlib.pyplot.close(fig)


def test_deformed_curve_endpoints_match_nodes():
    """補間曲線の両端が節点変位（scale倍）と一致すること。"""
    m = _portal_2d()
    res = solve_static(m)
    from beamfem.assembly import element_dof_map

    scale = 100.0
    e = m.elements[0]
    dofs = element_dof_map(m)[0]
    p1, p2 = m.nodes[e.n1], m.nodes[e.n2]
    curve = viz.element_deformed_curve(p1, p2, res.u[dofs], e.vref, scale=scale, n=10)
    exp1 = p1 + scale * res.node_disp(e.n1)[:3]
    exp2 = p2 + scale * res.node_disp(e.n2)[:3]
    assert np.allclose(curve[0], exp1, atol=1e-9)
    assert np.allclose(curve[-1], exp2, atol=1e-9)
