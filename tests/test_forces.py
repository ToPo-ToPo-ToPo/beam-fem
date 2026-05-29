"""要素内力・応力の回収の検証（解析解との比較）。"""

import numpy as np
import pytest

from beamfem import Material, Section, Model, solve_static, recover_forces, UX, UY, UZ

STEEL = Material(E=200e9, nu=0.3)


def test_cantilever_shear_moment():
    """片持ち梁・先端横荷重: せん断一定 P、固定端モーメント PL。"""
    L, P = 2.0, 1000.0
    sec = Section.rectangle(b=0.05, h=0.10)
    m = Model()
    n0 = m.add_node(0, 0, 0)
    n1 = m.add_node(L, 0, 0)
    m.add_element(n0, n1, STEEL, sec)
    m.fix(n0)
    m.add_load(n1, UY, -P)
    ef = recover_forces(m, solve_static(m))[0]

    assert np.isclose(ef.max_abs("Vy"), P, rtol=1e-9)
    assert np.isclose(ef.max_abs("Mz"), P * L, rtol=1e-9)
    assert np.isclose(ef.max_abs("N"), 0.0, atol=1e-6)
    # 先端モーメントは0
    assert np.isclose(ef.ends("Mz")[1], 0.0, atol=1e-6)


def test_axial_force_and_stress():
    """軸引張: N=P（引張正）、軸応力 P/A。"""
    L, P = 2.0, 1000.0
    sec = Section.rectangle(b=0.05, h=0.10)
    m = Model()
    a = m.add_node(0, 0, 0)
    b = m.add_node(L, 0, 0)
    m.add_element(a, b, STEEL, sec)
    m.fix(a)
    m.add_load(b, UX, P)
    ef = recover_forces(m, solve_static(m))[0]

    assert np.isclose(ef.ends("N")[0], P, rtol=1e-9)
    assert np.isclose(ef.stress_ends("sigma_a")[0], P / sec.A, rtol=1e-9)


def test_bending_stress():
    """片持ち固定端の曲げ応力 = Mz*cy/Iz。"""
    L, P = 2.0, 1000.0
    sec = Section.rectangle(b=0.05, h=0.10)
    m = Model()
    n0 = m.add_node(0, 0, 0)
    n1 = m.add_node(L, 0, 0)
    m.add_element(n0, n1, STEEL, sec)
    m.fix(n0)
    m.add_load(n1, UY, -P)
    ef = recover_forces(m, solve_static(m))[0]

    expected = (P * L) * sec.cy / sec.Iz
    assert np.isclose(ef.stress_ends("sigma_b")[0], expected, rtol=1e-9)


def test_simply_supported_midspan_moment():
    """単純梁・中央集中荷重: 中央曲げモーメント = PL/4。"""
    n, span, P = 10, 4.0, 1000.0
    sec = Section.rectangle(b=0.1, h=0.2)
    m = Model()
    nodes = [m.add_node(span * i / n, 0, 0) for i in range(n + 1)]
    for i in range(n):
        m.add_element(nodes[i], nodes[i + 1], STEEL, sec)
    m.pin(nodes[0])
    m.fix(nodes[-1], [1, 2])  # ローラー
    m.fix_to_plane_xy()
    m.add_load(nodes[n // 2], UY, -P)
    fr = recover_forces(m, solve_static(m))

    mmax = max(fr[i].max_abs("Mz") for i in range(len(fr)))
    vmax = max(fr[i].max_abs("Vy") for i in range(len(fr)))
    assert np.isclose(mmax, P * span / 4.0, rtol=1e-6)
    assert np.isclose(vmax, P / 2.0, rtol=1e-6)


def test_out_of_plane_bending_My():
    """先端 -z 荷重: My の固定端値 = PL、Vz 一定。"""
    L, P = 2.0, 800.0
    sec = Section.circle(d=0.05)
    m = Model()
    a = m.add_node(0, 0, 0)
    b = m.add_node(L, 0, 0)
    m.add_element(a, b, STEEL, sec)
    m.fix(a)
    m.add_load(b, UZ, -P)
    ef = recover_forces(m, solve_static(m))[0]

    assert np.isclose(ef.max_abs("My"), P * L, rtol=1e-9)
    assert np.isclose(ef.max_abs("Vz"), P, rtol=1e-9)


def test_table_selects_only_requested_items():
    """table は指定した項目のみ含む。"""
    L, P = 2.0, 1000.0
    sec = Section.rectangle(b=0.05, h=0.10)
    m = Model()
    n0 = m.add_node(0, 0, 0)
    n1 = m.add_node(L, 0, 0)
    m.add_element(n0, n1, STEEL, sec)
    m.fix(n0)
    m.add_load(n1, UY, -P)
    fr = recover_forces(m, solve_static(m))

    txt = fr.table(items=["Mz"], at="max")
    assert "Mz" in txt
    assert "Vy" not in txt and "N " not in txt
    with pytest.raises(KeyError):
        fr.table(items=["NOPE"])


def test_to_csv(tmp_path):
    L, P = 2.0, 1000.0
    sec = Section.rectangle(b=0.05, h=0.10)
    m = Model()
    n0 = m.add_node(0, 0, 0)
    n1 = m.add_node(L, 0, 0)
    m.add_element(n0, n1, STEEL, sec)
    m.fix(n0)
    m.add_load(n1, UY, -P)
    fr = recover_forces(m, solve_static(m))

    p = tmp_path / "forces.csv"
    fr.to_csv(str(p), items=["N", "Vy", "Mz"], at="ends")
    lines = p.read_text().strip().splitlines()
    assert lines[0] == "elem,location,N,Vy,Mz"
    assert len(lines) == 1 + 2  # ヘッダ + 1要素×両端
