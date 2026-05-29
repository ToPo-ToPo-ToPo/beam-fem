"""フラットシェル要素（CST 膜 + DKT 板曲げ）の検証。

- 局所剛性の剛体モード（膜 3 + 曲げ 3 = 6 個のゼロエネルギー）
- 膜のパッチテスト（一様引張 -> 一様応力・線形変位, 厳密一致）
- 単純支持正方形板の中央たわみ（Navier 級数解への収束）
- 応力回収（膜応力 = 付与した一様応力）
"""

import numpy as np

from beamfem import (
    Material,
    Model,
    solve_static,
    recover_shell_forces,
    UX,
    UY,
    UZ,
    RX,
    RY,
    RZ,
)
from beamfem.shell3d import shell_local_stiffness, shell_local_frame


STEEL = Material(E=200e9, nu=0.3, name="steel")


def test_local_rigid_body_modes():
    """局所剛性は 6 つの物理剛体運動でエネルギーを持たない。

    並進3＋回転3。面法線まわり回転 θz は架空ドリリング剛性が一様回転に対し
    ゼロとなるよう構成されているため、これも剛体運動を再現する。
    """
    x = np.array([0.0, 1.7, 0.6])
    y = np.array([0.0, 0.0, 1.3])
    area = 0.5 * 1.7 * 1.3
    K = shell_local_stiffness(STEEL.E, STEEL.nu, 0.01, x, y, area)
    assert np.allclose(K, K.T)

    # 局所自由度順 [u,v,w,θx,θy,θz] の 6 剛体モード
    def mode(per_node):
        return np.concatenate([per_node(i) for i in range(3)])

    modes = {
        "trans_x": mode(lambda i: [1, 0, 0, 0, 0, 0]),
        "trans_y": mode(lambda i: [0, 1, 0, 0, 0, 0]),
        "trans_z": mode(lambda i: [0, 0, 1, 0, 0, 0]),
        # θx まわり: w = θx·y
        "rot_x": mode(lambda i: [0, 0, y[i], 1, 0, 0]),
        # θy まわり: w = -θy·x
        "rot_y": mode(lambda i: [0, 0, -x[i], 0, 1, 0]),
        # θz まわり: u = -θz·y, v = θz·x
        "rot_z": mode(lambda i: [-y[i], x[i], 0, 0, 0, 1]),
    }
    kmax = np.max(np.abs(K))
    for name, v in modes.items():
        v = np.asarray(v, dtype=float)
        energy = float(v @ K @ v)
        assert abs(energy) < 1e-9 * kmax, (name, energy)


def _membrane_patch_model(nx=2, ny=2, lx=2.0, ly=1.0, t=0.02):
    """矩形をシェルで分割したモデルと節点格子を返す。"""
    m = Model()
    ids = {}
    for j in range(ny + 1):
        for i in range(nx + 1):
            ids[(i, j)] = m.add_node(lx * i / nx, ly * j / ny, 0.0)
    for j in range(ny):
        for i in range(nx):
            n00 = ids[(i, j)]
            n10 = ids[(i + 1, j)]
            n11 = ids[(i + 1, j + 1)]
            n01 = ids[(i, j + 1)]
            m.add_shell(n00, n10, n11, STEEL, t)
            m.add_shell(n00, n11, n01, STEEL, t)
    return m, ids


def test_membrane_patch_uniaxial_tension():
    """一様引張で面内変位が線形・応力が一様になる（CST のパッチテスト）。"""
    lx, ly, t = 2.0, 1.0, 0.02
    nx, ny = 3, 2
    m, ids = _membrane_patch_model(nx, ny, lx, ly, t)

    # 板曲げ・面外を拘束し純粋な面内問題にする
    for (i, j), n in ids.items():
        m.fix(n, [UZ, RX, RY, RZ])
    # 左端を x 固定、左下隅を y も固定（剛体除去）
    for j in range(ny + 1):
        m.fix(ids[(0, j)], [UX])
    m.fix(ids[(0, 0)], [UY])

    # 右端に一様引張（合計 F = σ * ly * t）。各右端節点へ等価節点力
    sigma = 50e6
    F = sigma * ly * t
    right_nodes = [ids[(nx, j)] for j in range(ny + 1)]
    # 端の節点は半分の負担
    for k, n in enumerate(right_nodes):
        w = 0.5 if k in (0, ny) else 1.0
        m.add_load(n, UX, F * w / ny)

    res = solve_static(m)

    # 解析解: ux = σ/E * x, uy = -ν σ/E * y
    for (i, j), n in ids.items():
        xpos = lx * i / nx
        ypos = ly * j / ny
        d = res.node_disp(n)
        assert np.isclose(d[UX], sigma / STEEL.E * xpos, rtol=1e-9, atol=1e-14)
        assert np.isclose(d[UY], -STEEL.nu * sigma / STEEL.E * ypos, rtol=1e-9, atol=1e-14)

    # 応力回収: 各要素のローカル系応力でも主応力は (σ, 0) の一軸状態
    # （ローカル x が要素辺向きのため、対角三角形では成分は回転して現れる）
    sf = recover_shell_forces(m, res)
    for s in sf.shells:
        sx, sy, sxy = s.get("sx"), s.get("sy"), s.get("sxy")
        avg = 0.5 * (sx + sy)
        r = np.hypot(0.5 * (sx - sy), sxy)
        p1, p2 = avg + r, avg - r
        assert np.isclose(p1, sigma, rtol=1e-6), (sx, sy, sxy)
        assert abs(p2) < 1e-6 * sigma
        # 曲げモーメントはゼロ（面内のみの問題）
        assert abs(s.get("Mx")) < 1e-6 * sigma * t**2


def _ss_plate_center_deflection(n, a=1.0, t=0.01, q=1.0e4):
    """単純支持正方形板を n×n 分割で解き、中央たわみを返す。"""
    m = Model()
    ids = {}
    for j in range(n + 1):
        for i in range(n + 1):
            ids[(i, j)] = m.add_node(a * i / n, a * j / n, 0.0)
    for j in range(n):
        for i in range(n):
            n00 = ids[(i, j)]
            n10 = ids[(i + 1, j)]
            n11 = ids[(i + 1, j + 1)]
            n01 = ids[(i, j + 1)]
            m.add_shell(n00, n10, n11, STEEL, t)
            m.add_shell(n00, n11, n01, STEEL, t)

    # 面内自由度（膜・ドリリング）は全節点で拘束（純曲げ問題）
    for n_id in ids.values():
        m.fix(n_id, [UX, UY, RZ])
    # 周辺単純支持: 縁の w(=UZ) を固定（回転は自由）
    for i in range(n + 1):
        for jj in (0, n):
            m.fix(ids[(i, jj)], [UZ])
            m.fix(ids[(jj, i)], [UZ])

    # 一様圧力 q を等価節点力に集約（三角形面積/3 を各節点へ, 下向き）
    for s in m.shells:
        p1, p2, p3 = m.nodes[s.n1], m.nodes[s.n2], m.nodes[s.n3]
        _, _, _, area = shell_local_frame(p1, p2, p3)
        f = -q * area / 3.0
        for nd in (s.n1, s.n2, s.n3):
            m.add_load(nd, UZ, f)

    res = solve_static(m)
    center = ids[(n // 2, n // 2)]
    return res.node_disp(center)[UZ]


def test_ss_plate_center_deflection():
    """単純支持正方形板の中央たわみが Navier 解に収束する。"""
    a, t, q = 1.0, 0.01, 1.0e4
    nu = STEEL.nu
    D = STEEL.E * t**3 / (12.0 * (1.0 - nu**2))
    w_exact = -0.00406 * q * a**4 / D  # 下向き

    w_coarse = _ss_plate_center_deflection(8, a, t, q)
    w_fine = _ss_plate_center_deflection(16, a, t, q)

    # 細かいメッシュで 3% 以内
    assert np.isclose(w_fine, w_exact, rtol=0.03), (w_fine, w_exact)
    # 収束していること（細かい方が解析解に近い）
    assert abs(w_fine - w_exact) <= abs(w_coarse - w_exact) + 1e-12


def _circular_plate_center(n_radial, n_rings, R=1.0, t=0.01, q=2.0e4, clamped=True):
    """円板を放射分割しシェルで解き、中央たわみを返す。"""
    m = Model()
    center = m.add_node(0.0, 0.0, 0.0)
    ring_nodes = [[center] * n_radial]
    for k in range(1, n_rings + 1):
        r = R * k / n_rings
        ring_nodes.append(
            [m.add_node(r * np.cos(2 * np.pi * j / n_radial),
                        r * np.sin(2 * np.pi * j / n_radial), 0.0)
             for j in range(n_radial)]
        )
    tris = []
    for j in range(n_radial):
        jn = (j + 1) % n_radial
        tris.append((center, ring_nodes[1][j], ring_nodes[1][jn]))
    for k in range(1, n_rings):
        for j in range(n_radial):
            jn = (j + 1) % n_radial
            a, b = ring_nodes[k][j], ring_nodes[k][jn]
            c, d = ring_nodes[k + 1][jn], ring_nodes[k + 1][j]
            tris.append((a, b, c))
            tris.append((a, c, d))
    for (i, j, k) in tris:
        m.add_shell(i, j, k, STEEL, t)

    for i in range(m.n_nodes):
        m.fix(i, [UX, UY, RZ])
    for nd in ring_nodes[n_rings]:
        m.fix(nd, [UZ, RX, RY] if clamped else [UZ])

    for (a, b, c) in tris:
        p1, p2, p3 = m.nodes[a][:2], m.nodes[b][:2], m.nodes[c][:2]
        area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1])
                         - (p3[0] - p1[0]) * (p2[1] - p1[1]))
        for nd in (a, b, c):
            m.add_load(nd, UZ, -q * area / 3.0)

    res = solve_static(m)
    return res.node_disp(center)[UZ]


def test_clamped_circular_plate():
    """周辺固定の円板の中央たわみが解析解 q R^4/(64 D) に収束する。"""
    R, t, q = 1.0, 0.01, 2.0e4
    D = STEEL.E * t**3 / (12.0 * (1.0 - STEEL.nu**2))
    w_exact = -q * R**4 / (64.0 * D)

    w = _circular_plate_center(24, 8, R, t, q, clamped=True)
    assert np.isclose(w, w_exact, rtol=0.02), (w, w_exact)


def test_simply_supported_circular_plate():
    """単純支持の円板の中央たわみが解析解 (5+ν)/(1+ν)·q R^4/(64 D) に収束する。"""
    R, t, q = 1.0, 0.01, 2.0e4
    nu = STEEL.nu
    D = STEEL.E * t**3 / (12.0 * (1.0 - nu**2))
    w_exact = -(5.0 + nu) / (1.0 + nu) * q * R**4 / (64.0 * D)

    w = _circular_plate_center(24, 8, R, t, q, clamped=False)
    assert np.isclose(w, w_exact, rtol=0.02), (w, w_exact)


def test_assembled_stiffness_symmetric():
    """シェルを含む全体剛性が対称。"""
    m, _ = _membrane_patch_model(2, 2)
    from beamfem.assembly import assemble_stiffness

    K = assemble_stiffness(m).toarray()
    # E~2e11 スケールのため絶対許容値はピーク値に比例させる
    assert np.allclose(K, K.T, atol=1e-6 * np.max(np.abs(K)))


def test_shell_viz_smoke():
    """シェルを含むモデルの描画が例外なく完了する（2D 平面・3D 両方）。"""
    import pytest

    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from beamfem import viz

    # 平面（xy）板
    m, ids = _membrane_patch_model(2, 2)
    for nid in ids.values():
        m.fix(nid, [UX, UY, RZ])
    for i in range(3):
        m.fix(ids[(0, i)], [UZ])
    m.add_load(ids[(2, 1)], UZ, -1.0e3)
    res = solve_static(m)
    viz.plot_model(m)
    viz.plot_deformed(m, res, scale=1.0)


def test_orientation_invariance():
    """要素を全体空間で剛体回転しても剛性のスペクトルは不変。"""
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([2.0, 0.0, 0.0])
    p3 = np.array([0.5, 1.5, 0.0])
    from beamfem.shell3d import shell_stiffness_global

    K0 = shell_stiffness_global(p1, p2, p3, STEEL, 0.01)

    # 任意の回転行列
    th = 0.7
    Rz = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    th2 = 0.4
    Rx = np.array([[1, 0, 0], [0, np.cos(th2), -np.sin(th2)], [0, np.sin(th2), np.cos(th2)]])
    Q = Rx @ Rz
    K1 = shell_stiffness_global(Q @ p1, Q @ p2, Q @ p3, STEEL, 0.01)

    assert np.allclose(np.linalg.eigvalsh(K0), np.linalg.eigvalsh(K1), atol=1e-3)
