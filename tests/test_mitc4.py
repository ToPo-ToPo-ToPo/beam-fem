"""MITC4 四角形シェル要素の検証。

第1段階（板曲げ単体・Mindlin+MITC タイング）:
- 剛体モード（w 並進・θx/θy 回転でゼロエネルギー）
- 薄板の単純支持板が Kirchhoff(Navier) 解に収束し、**せん断ロックしない**
- 厚板はせん断変形でたわみが増える（収束する）
"""

import numpy as np

from beamfem import Material, Model, solve_static, UX, UY, UZ, RX, RY, RZ
from beamfem.shell_mitc4 import mitc4_plate_stiffness, q4_shape, q4_jacobian


E, NU = 200e9, 0.3
STEEL = Material(E=E, nu=NU, name="steel")


def test_mitc4_rigid_body_modes():
    """板曲げ要素は w 並進・θx/θy 回転の 3 剛体モードでエネルギーを持たない。"""
    x = np.array([0.0, 1.2, 1.0, 0.1])
    y = np.array([0.0, 0.2, 1.3, 1.1])
    K = mitc4_plate_stiffness(E, NU, 0.02, x, y)
    assert np.allclose(K, K.T, atol=1e-6 * np.max(np.abs(K)))

    wt = np.array([1, 0, 0] * 4, dtype=float)
    rx = np.concatenate([[y[i], 1, 0] for i in range(4)])    # θx: w=y
    ry = np.concatenate([[-x[i], 0, 1] for i in range(4)])   # θy: w=-x
    kmax = np.max(np.abs(K))
    for v in (wt, rx, ry):
        assert abs(float(v @ K @ v)) < 1e-9 * kmax


def _ss_plate_center(n, a, t, q=1.0e4):
    """n×n の MITC4 で単純支持正方形板を解き中央たわみを返す（板曲げのみ）。"""
    xs = np.linspace(0.0, a, n + 1)
    nid = lambda i, j: j * (n + 1) + i
    nnode = (n + 1) ** 2
    ndof = 3 * nnode
    K = np.zeros((ndof, ndof))
    F = np.zeros(ndof)
    gp = [(-0.5773502691896257, -0.5773502691896257),
          (0.5773502691896257, -0.5773502691896257),
          (0.5773502691896257, 0.5773502691896257),
          (-0.5773502691896257, 0.5773502691896257)]
    for ej in range(n):
        for ei in range(n):
            ns = [nid(ei, ej), nid(ei + 1, ej), nid(ei + 1, ej + 1), nid(ei, ej + 1)]
            ex = np.array([xs[ei], xs[ei + 1], xs[ei + 1], xs[ei]])
            ey = np.array([xs[ej], xs[ej], xs[ej + 1], xs[ej + 1]])
            ke = mitc4_plate_stiffness(E, NU, t, ex, ey)
            dofs = np.concatenate([[3 * nd, 3 * nd + 1, 3 * nd + 2] for nd in ns])
            K[np.ix_(dofs, dofs)] += ke
            for xi, eta in gp:  # 一様圧の整合節点荷重（w 自由度）
                N, dxi, deta = q4_shape(xi, eta)
                _, detJ, _ = q4_jacobian(ex, ey, dxi, deta)
                for k in range(4):
                    F[3 * ns[k]] += -q * N[k] * detJ
    fixed = set()
    for i in range(n + 1):
        for j in range(n + 1):
            if i in (0, n) or j in (0, n):
                fixed.add(3 * nid(i, j))  # 周辺 w=0（単純支持）
    free = np.array([d for d in range(ndof) if d not in fixed])
    u = np.zeros(ndof)
    u[free] = np.linalg.solve(K[np.ix_(free, free)], F[free])
    return u[3 * nid(n // 2, n // 2)]


def _kirchhoff_center(a, t, q=1.0e4):
    D = E * t**3 / (12.0 * (1.0 - NU**2))
    return -0.00406 * q * a**4 / D


def test_mitc4_thin_plate_no_shear_locking():
    """薄板（a/t=1000）が Kirchhoff 解に収束し、粗メッシュでもロックしない。"""
    a, t = 1.0, 1.0e-3
    wk = _kirchhoff_center(a, t)
    w4 = _ss_plate_center(4, a, t)
    w16 = _ss_plate_center(16, a, t)
    # 細メッシュで Kirchhoff に 1.5% 以内
    assert np.isclose(w16, wk, rtol=0.015), (w16, wk)
    # 粗メッシュでもロックしない（極端な過小評価が起きない）
    assert w4 / wk > 0.9


def test_mitc4_thick_plate_shear_deformation():
    """厚板（a/t=10）はせん断変形で Kirchhoff よりたわむ（収束する）。"""
    a, t = 1.0, 0.1
    wk = _kirchhoff_center(a, t)
    w8 = _ss_plate_center(8, a, t)
    w16 = _ss_plate_center(16, a, t)
    # せん断変形ぶんたわみが増える（5%以上, 過大でもない）
    assert 1.05 < w16 / wk < 1.20, w16 / wk
    # メッシュ収束（細分化で値が安定）
    assert abs(w16 - w8) / abs(w16) < 0.03


# ======================================================================
# 第2段階: フラットシェル（膜 Q4 + MITC4 + ドリリング）を Model 経由で検証
# ======================================================================

def _quad_grid(m, nx, ny, lx, ly, t):
    """nx×ny の四角形シェルで矩形板を張り、節点 id 辞書を返す。"""
    ids = {}
    for j in range(ny + 1):
        for i in range(nx + 1):
            ids[(i, j)] = m.add_node(lx * i / nx, ly * j / ny, 0.0)
    for j in range(ny):
        for i in range(nx):
            m.add_quad_shell(ids[(i, j)], ids[(i + 1, j)],
                             ids[(i + 1, j + 1)], ids[(i, j + 1)], STEEL, t)
    return ids


def test_quad_shell_membrane_patch():
    """一様引張で面内変位が線形になる（Q4 膜のパッチテスト・厳密一致）。"""
    lx, ly, t = 2.0, 1.0, 0.02
    nx, ny = 3, 2
    m = Model()
    ids = _quad_grid(m, nx, ny, lx, ly, t)
    for (i, j), n in ids.items():
        m.fix(n, [UZ, RX, RY, RZ])     # 面外を拘束し純面内に
    for j in range(ny + 1):
        m.fix(ids[(0, j)], [UX])       # 左端 x 固定
    m.fix(ids[(0, 0)], [UY])           # 剛体除去

    sigma = 50e6
    F = sigma * ly * t
    for j in range(ny + 1):
        w = 0.5 if j in (0, ny) else 1.0
        m.add_load(ids[(nx, j)], UX, F * w / ny)

    res = solve_static(m)
    for (i, j), n in ids.items():
        d = res.node_disp(n)
        assert np.isclose(d[UX], sigma / E * (lx * i / nx), rtol=1e-9, atol=1e-13)
        assert np.isclose(d[UY], -NU * sigma / E * (ly * j / ny), rtol=1e-9, atol=1e-13)


def _ss_quad_plate(n, a, t, q=1.0e4):
    """四角形フラットシェルで単純支持正方形板を解き中央たわみを返す。"""
    m = Model()
    ids = _quad_grid(m, n, n, a, a, t)
    for nid in ids.values():
        m.fix(nid, [UX, UY, RZ])       # 面内・ドリリングは拘束（純曲げ）
    for i in range(n + 1):
        for jj in (0, n):
            m.fix(ids[(i, jj)], [UZ])
            m.fix(ids[(jj, i)], [UZ])
    # 一様圧 → 各四角形の 1/4 を節点へ
    for s in m.quad_shells:
        p = [m.nodes[s.n1], m.nodes[s.n2], m.nodes[s.n3], m.nodes[s.n4]]
        area = 0.5 * abs((p[2][0] - p[0][0]) * (p[3][1] - p[1][1])
                         - (p[3][0] - p[1][0]) * (p[2][1] - p[0][1]))
        for nd in (s.n1, s.n2, s.n3, s.n4):
            m.add_load(nd, UZ, -q * area / 4.0)
    res = solve_static(m)
    return res.node_disp(ids[(n // 2, n // 2)])[UZ]


def test_quad_shell_ss_plate_convergence():
    """四角形フラットシェルの単純支持正方形板が Navier 解に収束する。"""
    a, t, q = 1.0, 0.01, 1.0e4
    D = E * t**3 / (12.0 * (1.0 - NU**2))
    w_exact = -0.00406 * q * a**4 / D
    w8 = _ss_quad_plate(8, a, t, q)
    w16 = _ss_quad_plate(16, a, t, q)
    assert np.isclose(w16, w_exact, rtol=0.02), (w16, w_exact)
    assert abs(w16 - w_exact) <= abs(w8 - w_exact) + 1e-12


def test_quad_shell_assembled_symmetric():
    """四角形シェルを含む全体剛性が対称。"""
    from beamfem.assembly import assemble_stiffness

    m = Model()
    _quad_grid(m, 2, 2, 1.0, 1.0, 0.01)
    K = assemble_stiffness(m).toarray()
    assert np.allclose(K, K.T, atol=1e-6 * np.max(np.abs(K)))


def test_quad_shell_orientation_invariance():
    """要素を空間で剛体回転してもスペクトル不変（フラットシェルの座標変換）。"""
    from beamfem.shell_mitc4 import quad_shell_stiffness_global

    P = [np.array(p, dtype=float) for p in
         ([0, 0, 0], [1.2, 0, 0], [1.1, 0.9, 0], [0.1, 1.0, 0])]
    K0 = quad_shell_stiffness_global(*P, STEEL, 0.01)
    th = 0.6
    Rz = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, np.cos(0.5), -np.sin(0.5)], [0, np.sin(0.5), np.cos(0.5)]])
    Q = Rx @ Rz
    K1 = quad_shell_stiffness_global(*[Q @ p for p in P], STEEL, 0.01)
    assert np.allclose(np.linalg.eigvalsh(K0), np.linalg.eigvalsh(K1), atol=1e-3)
