"""オフセット梁（剛体腕）の検証。

- 剛体腕変換 G の基本性質（剛体運動を保つ＝オフセット要素剛性が剛体モードで
  ゼロエネルギー、対称性）
- 剛体リンク（極めて剛な短梁）で明示的に組んだモデルとの等価性
  （T 形断面の合成剛性 EA·e² を含む連成挙動の一致）
- オフセット梁の軸-曲げ連成（節点回転が軸力を生む）
"""

import numpy as np

from beamfem import Material, Section, Model, solve_static, recover_forces, UZ
from beamfem.element3d import element_stiffness_global, rigid_offset_matrix


STEEL = Material(E=200e9, nu=0.3, rho=7850.0)


def test_rigid_offset_matrix_form():
    """G は剛体腕関係 u_beam = u_node + theta x r を表す。"""
    r = np.array([0.0, 0.0, -0.02])
    G = rigid_offset_matrix(r)
    # 節点に単位回転 theta=(theta_x,theta_y,theta_z) のみ与えたとき、
    # 梁端並進は theta x r になる。
    for axis in range(3):
        th = np.zeros(3)
        th[axis] = 1.0
        q = np.concatenate([np.zeros(3), th, np.zeros(6)])
        beam = G @ q
        expected = np.cross(th, r)
        assert np.allclose(beam[0:3], expected), (axis, beam[0:3], expected)
        assert np.allclose(beam[3:6], th)  # 回転は不変


def test_offset_element_rigid_body_and_symmetry():
    """オフセット要素の全体剛性は対称で、6 剛体モードでゼロエネルギー。"""
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([1.3, 0.4, 0.0])
    sec = Section.rectangle(b=0.01, h=0.03)
    offset = np.array([0.0, 0.0, -0.02])
    K = element_stiffness_global(p1, p2, STEEL, sec, offset=offset)
    assert np.allclose(K, K.T, atol=1e-6 * np.max(np.abs(K)))

    # マスター節点(p1,p2)に与える 6 剛体運動でエネルギー≈0
    def rb_modes(p1, p2):
        modes = []
        # 並進3
        for d in range(3):
            v = np.zeros(12)
            v[d] = 1.0
            v[6 + d] = 1.0
            modes.append(v)
        # 回転3（原点まわり）: u_i = w x p_i, theta_i = w
        for a in range(3):
            w = np.zeros(3)
            w[a] = 1.0
            v = np.zeros(12)
            v[0:3] = np.cross(w, p1)
            v[3:6] = w
            v[6:9] = np.cross(w, p2)
            v[9:12] = w
            modes.append(v)
        return modes

    kmax = np.max(np.abs(K))
    for v in rb_modes(p1, p2):
        assert abs(float(v @ K @ v)) < 1e-9 * kmax


# 合成梁の検証パラメータ（細メッシュ・両端固定・偏心荷重で EA·e² を発現させる）
_SPAN = 2.0
_NEL = 16
_PLATE = Section.rectangle(b=0.10, h=0.010)
_RIB = Section.circle(d=0.025)             # 円形でリブ向きを問わない
_E = 0.010 / 2 + 0.025 / 2                 # 板下面にリブを付ける偏心
_P = -3000.0
_LOAD_NODE = _NEL // 4                     # 非対称（1/4 スパン）に載荷


def _composite_offset(e):
    """オフセット梁で組んだ細メッシュ合成梁（両端固定）の節点列とモデル。"""
    m = Model()
    nodes = [m.add_node(_SPAN * i / _NEL, 0.0, 0.0) for i in range(_NEL + 1)]
    rib_ids = []
    for i in range(_NEL):
        m.add_element(nodes[i], nodes[i + 1], STEEL, _PLATE)
        off = [0.0, 0.0, -e] if e != 0.0 else None
        rib_ids.append(m.add_element(nodes[i], nodes[i + 1], STEEL, _RIB, offset=off))
    m.fix(nodes[0])
    m.fix(nodes[-1])
    m.add_load(nodes[_LOAD_NODE], UZ, _P)
    return m, nodes, rib_ids


def _composite_rigidlink(e):
    """剛体リンク（極めて剛な短梁）で明示的に組んだ同じ合成梁。"""
    stiff = Material(E=STEEL.E * 1e6, nu=0.3)
    link = Section.rectangle(b=0.3, h=0.3)
    m = Model()
    top = [m.add_node(_SPAN * i / _NEL, 0.0, 0.0) for i in range(_NEL + 1)]
    bot = [m.add_node(_SPAN * i / _NEL, 0.0, -e) for i in range(_NEL + 1)]
    for i in range(_NEL):
        m.add_element(top[i], top[i + 1], STEEL, _PLATE)
        m.add_element(bot[i], bot[i + 1], STEEL, _RIB)
    for i in range(_NEL + 1):
        m.add_element(top[i], bot[i], stiff, link)  # 各節点で鉛直剛体リンク
    m.fix(top[0])
    m.fix(top[-1])
    m.add_load(top[_LOAD_NODE], UZ, _P)
    return m, top


def test_offset_equivalence_to_rigid_link():
    """オフセット梁が剛体リンク明示モデルと一致する（EA·e² 連成込み）。"""
    mo, nodes, _ = _composite_offset(_E)
    ro = solve_static(mo)
    w_off = ro.node_disp(nodes[_LOAD_NODE])[UZ]

    me, top = _composite_rigidlink(_E)
    re = solve_static(me)
    w_link = re.node_disp(top[_LOAD_NODE])[UZ]

    assert np.isclose(w_off, w_link, rtol=2e-3), (w_off, w_link)


def test_offset_stiffens_and_induces_axial():
    """偏心配置はたわみを減らし、リブに軸力（合成効果 EA·e²）を生む。"""
    mo, nodes, rib_ids = _composite_offset(_E)
    ro = solve_static(mo)
    w_off = ro.node_disp(nodes[_LOAD_NODE])[UZ]

    mc, ncs, _ = _composite_offset(0.0)
    rc = solve_static(mc)
    w_con = rc.node_disp(ncs[_LOAD_NODE])[UZ]

    # 偏心ありは有意に剛い（合成効果でたわみが減る）
    assert abs(w_off) < 0.95 * abs(w_con), (w_off, w_con)

    # オフセットありのリブに有意な軸力が出る（T 形合成のあかし）
    forces = recover_forces(mo, ro)
    n_rib = max(abs(forces[rid].max_abs("N")) for rid in rib_ids)
    assert n_rib > 1.0
