"""4節点四角形シェル要素 MITC4（Mindlin-Reissner + 仮定横せん断ひずみ）。

第1段階として **板曲げ単体**（たわみ w・回転 θx, θy の 3 自由度/節点）を実装する。
Reissner-Mindlin 板理論に基づき横せん断変形を含むため、厚板から薄板まで扱える。
素の Mindlin 四角形は薄板でせん断ロックするので、横せん断ひずみを辺中点で
タイングする **MITC4**（Bathe-Dvorkin, 1984）で回避する。

回転の規約は DKT（``shell3d``）と共通::

    薄板極限で  θx = ∂w/∂y,  θy = -∂w/∂x
    曲げ曲率    κx = ∂θy/∂x,  κy = -∂θx/∂y,  κxy = ∂θy/∂y - ∂θx/∂x
    横せん断    γxz = ∂w/∂x + θy,  γyz = ∂w/∂y - θx

自由度の並びは節点ごと [w, θx, θy]、節点 1..4（反時計まわり）。

参考: Dvorkin & Bathe (1984), "A continuum mechanics based four-node shell
element for general nonlinear analysis", Eng. Comput. 1.
"""

from __future__ import annotations

import numpy as np

from .material import Material
from .shell3d import DRILLING_FACTOR

# 2x2 ガウス積分点と重み
_g = 1.0 / np.sqrt(3.0)
_GAUSS2 = ((-_g, -_g, 1.0), (_g, -_g, 1.0), (_g, _g, 1.0), (-_g, _g, 1.0))

# せん断補正係数（Reissner）
SHEAR_K = 5.0 / 6.0


def q4_shape(xi: float, eta: float):
    """双一次 Q4 形状関数 N と自然座標微分 dN/dξ, dN/dη を返す（各 4 成分）。

    節点順: 1=(-1,-1), 2=(1,-1), 3=(1,1), 4=(-1,1)。
    """
    N = 0.25 * np.array([
        (1 - xi) * (1 - eta),
        (1 + xi) * (1 - eta),
        (1 + xi) * (1 + eta),
        (1 - xi) * (1 + eta),
    ])
    dNdxi = 0.25 * np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)])
    dNdeta = 0.25 * np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)])
    return N, dNdxi, dNdeta


def q4_jacobian(x, y, dNdxi, dNdeta):
    """ヤコビアン J=[[x_ξ,y_ξ],[x_η,y_η]] と detJ, Jinv を返す。"""
    x_xi, y_xi = dNdxi @ x, dNdxi @ y
    x_eta, y_eta = dNdeta @ x, dNdeta @ y
    J = np.array([[x_xi, y_xi], [x_eta, y_eta]])
    detJ = x_xi * y_eta - y_xi * x_eta
    if detJ <= 0:
        raise ValueError("MITC4 要素のヤコビアンが非正です（節点順・形状を確認）")
    Jinv = np.array([[y_eta, -y_xi], [-x_eta, x_xi]]) / detJ
    return J, detJ, Jinv


def _bending_B(dNdx, dNdy):
    """曲げ B 行列 (3x12)。DOF 並び [w,θx,θy]×4。

    κx = Σ dNi/dx · θy_i, κy = Σ -dNi/dy · θx_i,
    κxy = Σ dNi/dy · θy_i - dNi/dx · θx_i。
    """
    B = np.zeros((3, 12))
    for i in range(4):
        B[0, 3 * i + 2] = dNdx[i]
        B[1, 3 * i + 1] = -dNdy[i]
        B[2, 3 * i + 1] = -dNdx[i]
        B[2, 3 * i + 2] = dNdy[i]
    return B


def _covariant_shear_row(xi, eta, x, y, comp):
    """タイング点での共変横せん断ひずみ（γ_ξ または γ_η）の DOF 係数行 (12,)。

    γ_ξ = ∂w/∂ξ + x_ξ θy - y_ξ θx, γ_η = ∂w/∂η + x_η θy - y_η θx。
    """
    N, dNdxi, dNdeta = q4_shape(xi, eta)
    if comp == "xi":
        dw, xg, yg = dNdxi, dNdxi @ x, dNdxi @ y
    else:
        dw, xg, yg = dNdeta, dNdeta @ x, dNdeta @ y
    row = np.zeros(12)
    for i in range(4):
        row[3 * i + 0] = dw[i]        # ∂w/∂(ξ|η)
        row[3 * i + 2] = xg * N[i]    # +x_(ξ|η) θy
        row[3 * i + 1] = -yg * N[i]   # -y_(ξ|η) θx
    return row


def _mitc4_shear_B(xi, eta, x, y, Jinv):
    """MITC4 の仮定横せん断 B 行列 (2x12) を点 (ξ,η) で返す。

    辺中点 A(0,-1),C(0,1) で γ_ξ を、D(-1,0),B(1,0) で γ_η をサンプルし、
    γ_ξ=½(1-η)γ_ξ^A+½(1+η)γ_ξ^C, γ_η=½(1-ξ)γ_η^D+½(1+ξ)γ_η^B と内挿、
    最後に [γxz;γyz]=Jinv·[γ_ξ;γ_η] でデカルト成分へ変換する。
    """
    Bxi_A = _covariant_shear_row(0.0, -1.0, x, y, "xi")
    Bxi_C = _covariant_shear_row(0.0, 1.0, x, y, "xi")
    Beta_D = _covariant_shear_row(-1.0, 0.0, x, y, "eta")
    Beta_B = _covariant_shear_row(1.0, 0.0, x, y, "eta")

    g_xi = 0.5 * (1 - eta) * Bxi_A + 0.5 * (1 + eta) * Bxi_C
    g_eta = 0.5 * (1 - xi) * Beta_D + 0.5 * (1 + xi) * Beta_B
    return Jinv @ np.vstack([g_xi, g_eta])   # 2x12


def mitc4_plate_stiffness(E, nu, t, x, y, ks: float = SHEAR_K):
    """MITC4 板曲げ要素剛性 (12x12)。DOF 並び [w,θx,θy]×4。

    x, y は 4 節点の平面内座標。曲げ Db と横せん断 Ds を 2x2 ガウスで積分する。
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    Db = (E * t**3 / (12.0 * (1.0 - nu**2))) * np.array(
        [[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, (1.0 - nu) / 2.0]]
    )
    G = E / (2.0 * (1.0 + nu))
    Ds = ks * G * t * np.eye(2)

    K = np.zeros((12, 12))
    for xi, eta, w in _GAUSS2:
        _, dNdxi, dNdeta = q4_shape(xi, eta)
        _, detJ, Jinv = q4_jacobian(x, y, dNdxi, dNdeta)
        dN = Jinv @ np.vstack([dNdxi, dNdeta])  # [dN/dx; dN/dy] (2x4)
        Bb = _bending_B(dN[0], dN[1])
        Bs = _mitc4_shear_B(xi, eta, x, y, Jinv)
        K += (Bb.T @ Db @ Bb + Bs.T @ Ds @ Bs) * detJ * w
    return K


# ======================================================================
# 第2段階: 膜 Q4 ＋ ドリリングを足したフラットシェル要素
# ======================================================================

def q4_membrane_stiffness(E, nu, t, x, y):
    """Q4 平面応力（膜）要素剛性 (8x8)。DOF 並び [u,v]×4。面積も返す。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    D = (E / (1.0 - nu**2)) * np.array(
        [[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, (1.0 - nu) / 2.0]]
    )
    K = np.zeros((8, 8))
    area = 0.0
    for xi, eta, w in _GAUSS2:
        _, dNdxi, dNdeta = q4_shape(xi, eta)
        _, detJ, Jinv = q4_jacobian(x, y, dNdxi, dNdeta)
        dN = Jinv @ np.vstack([dNdxi, dNdeta])
        B = np.zeros((3, 8))
        for i in range(4):
            B[0, 2 * i] = dN[0, i]
            B[1, 2 * i + 1] = dN[1, i]
            B[2, 2 * i] = dN[1, i]
            B[2, 2 * i + 1] = dN[0, i]
        K += t * (B.T @ D @ B) * detJ * w
        area += detJ * w
    return K, area


def quad_drilling_stiffness(E, t, area):
    """ドリリング自由度（θz）の架空剛性 4x4（一様回転＝剛体回転でゼロ）。"""
    k = DRILLING_FACTOR * E * t * area
    return k * (np.eye(4) - np.ones((4, 4)) / 4.0)


def quad_shell_frame(p1, p2, p3, p4):
    """四角形の局所座標系 R と平面内節点座標 (x,y) を返す（平面フェセット近似）。

    対角線の外積で法線（局所 z）を、辺の平均方向で局所 x を定める。節点を平均面へ
    射影して平面内座標とする（平坦な四角形では厳密、わずかな反りは無視）。
    """
    P = [np.asarray(p, dtype=float) for p in (p1, p2, p3, p4)]
    c = sum(P) / 4.0
    normal = np.cross(P[2] - P[0], P[3] - P[1])
    nlen = np.linalg.norm(normal)
    if nlen < 1e-14:
        raise ValueError("四角形シェル要素が退化しています（対角線が平行/ゼロ）")
    e3 = normal / nlen
    v = (P[1] + P[2]) - (P[0] + P[3])      # ξ 方向の平均
    e1 = v - (v @ e3) * e3
    n1 = np.linalg.norm(e1)
    if n1 < 1e-14:
        raise ValueError("四角形シェル要素の局所 x 方向を決められません")
    e1 = e1 / n1
    e2 = np.cross(e3, e1)
    R = np.vstack([e1, e2, e3])
    x = np.array([(p - c) @ e1 for p in P])
    y = np.array([(p - c) @ e2 for p in P])
    return R, x, y


# 局所 24 自由度（節点ごと u,v,w,θx,θy,θz）における各サブ系のインデックス
_Q_MEMBRANE = np.array([0, 1, 6, 7, 12, 13, 18, 19])           # u,v ×4
_Q_BENDING = np.array([2, 3, 4, 8, 9, 10, 14, 15, 16, 20, 21, 22])  # w,θx,θy ×4
_Q_DRILL = np.array([5, 11, 17, 23])                            # θz ×4


def quad_shell_local_stiffness(E, nu, t, x, y):
    """局所座標系での 24x24 四角形シェル剛性（節点ごと [u,v,w,θx,θy,θz]）。"""
    Km, area = q4_membrane_stiffness(E, nu, t, x, y)
    Kb = mitc4_plate_stiffness(E, nu, t, x, y)
    Kd = quad_drilling_stiffness(E, t, area)
    K = np.zeros((24, 24))
    K[np.ix_(_Q_MEMBRANE, _Q_MEMBRANE)] += Km
    K[np.ix_(_Q_BENDING, _Q_BENDING)] += Kb
    K[np.ix_(_Q_DRILL, _Q_DRILL)] += Kd
    return K


def quad_shell_transformation(R):
    """3x3 の R から 24x24 の変換行列 T を組む（8 ブロック対角）。"""
    T = np.zeros((24, 24))
    for b in range(8):
        T[3 * b: 3 * b + 3, 3 * b: 3 * b + 3] = R
    return T


def quad_shell_stiffness_global(p1, p2, p3, p4, mat: Material, thickness: float):
    """全体座標系での 24x24 四角形シェル要素剛性 K = T^T k T。

    自由度順は節点ごと [u_x,u_y,u_z,θx,θy,θz]（全体系）で、節点 1,2,3,4 の順。
    """
    R, x, y = quad_shell_frame(p1, p2, p3, p4)
    k_local = quad_shell_local_stiffness(mat.E, mat.nu, thickness, x, y)
    T = quad_shell_transformation(R)
    return T.T @ k_local @ T
