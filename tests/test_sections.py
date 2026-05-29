"""各種断面形状の諸量と、ソルバとの統合検証。"""

import numpy as np
import pytest

from beamfem import Material, Section, Model, solve_static, UY

STEEL = Material(E=200e9, nu=0.3)


def test_i_section_properties():
    h, bf, tf, tw = 0.3, 0.15, 0.012, 0.008
    s = Section.i_section(h=h, bf=bf, tf=tf, tw=tw)
    hw = h - 2 * tf
    assert np.isclose(s.A, 2 * bf * tf + hw * tw)
    assert np.isclose(s.Iz, (bf * h**3 - (bf - tw) * hw**3) / 12.0)
    assert np.isclose(s.Iy, 2 * (tf * bf**3 / 12.0) + hw * tw**3 / 12.0)
    assert s.Iz > s.Iy  # 強軸 > 弱軸
    assert np.isclose(s.cy, h / 2) and np.isclose(s.cz, bf / 2)


def test_box_properties():
    b, h, t = 0.2, 0.3, 0.01
    s = Section.box(b=b, h=h, t=t)
    bi, hi = b - 2 * t, h - 2 * t
    assert np.isclose(s.A, b * h - bi * hi)
    assert np.isclose(s.Iz, (b * h**3 - bi * hi**3) / 12.0)
    assert np.isclose(s.Iy, (h * b**3 - hi * bi**3) / 12.0)


def test_pipe_properties():
    d, t = 0.1, 0.005
    s = Section.pipe(d=d, t=t)
    di = d - 2 * t
    assert np.isclose(s.A, np.pi * (d**2 - di**2) / 4.0)
    assert np.isclose(s.J, 2 * s.Iz)  # 円形は J=2I


def test_invalid_thickness():
    with pytest.raises(ValueError):
        Section.box(b=0.1, h=0.1, t=0.06)
    with pytest.raises(ValueError):
        Section.pipe(d=0.05, t=0.03)
    with pytest.raises(ValueError):
        Section.i_section(h=0.02, bf=0.1, tf=0.012, tw=0.006)


def test_shear_factor_override():
    s = Section.box(b=0.2, h=0.3, t=0.01, ky=0.5, kz=0.5)
    assert s.ky == 0.5 and s.kz == 0.5


@pytest.mark.parametrize(
    "sec",
    [
        Section.i_section(h=0.3, bf=0.15, tf=0.012, tw=0.008),
        Section.box(b=0.2, h=0.3, t=0.01),
        Section.pipe(d=0.1, t=0.005),
    ],
)
def test_cantilever_matches_analytic(sec):
    """各断面の片持ち梁先端たわみが Timoshenko 解析解と一致。"""
    L, P = 3.0, 5000.0
    m = Model()
    a = m.add_node(0, 0, 0)
    b = m.add_node(L, 0, 0)
    m.add_element(a, b, STEEL, sec)
    m.fix(a)
    m.add_load(b, UY, -P)
    fe = solve_static(m).node_disp(b)[UY]
    analytic = -(P * L**3 / (3 * STEEL.E * sec.Iz) + P * L / (sec.ky * STEEL.G * sec.A))
    assert np.isclose(fe, analytic, rtol=1e-9)
