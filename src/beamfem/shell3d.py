"""3節点フラットシェル要素（CST 膜 + DKT 板曲げ + ドリリング）。

各節点 6 自由度（梁と共通）の平面三角形要素。面内挙動（膜）と面外挙動
（板曲げ）を平面内で重ね合わせる「フラットシェル」方式::

    膜    : 定ひずみ三角形 CST（面内変位 u, v）             -> 6 自由度
    板曲げ: 離散 Kirchhoff 三角形 DKT（たわみ w, 回転 θx, θy） -> 9 自由度
    ドリリング: 面法線まわり回転 θz の微小架空剛性          -> 3 自由度

合計 3×6 = 18 自由度。要素は自身の平面内で定式化し、方向余弦行列で全体系へ
変換する。

自由度の並び（節点ごと, 局所・全体で共通の並び順）::

    [u_x, u_y, u_z, theta_x, theta_y, theta_z]

局所座標系::

    x : 節点1 -> 節点2
    z : (節点2-節点1) x (節点3-節点1) の向き（要素法線）
    y : z x x

膜は平面応力 D、板曲げは曲げ剛性 D_b = E t^3 / (12(1-ν^2)) を用いる。DKT は
薄板（Kirchhoff）理論に基づき、せん断変形は無視する（薄肉シェル向け）。

参考: Batoz, Bathe, Ho (1980), "A study of three-node triangular plate
bending elements", IJNME 15.
"""

from __future__ import annotations

import numpy as np

from .material import Material

# 板曲げ DKT の数値積分点（参照三角形の辺中点・重み 1/6）。
# DKT の被積分関数は (ξ,η) について 2 次なので 3 点で厳密。
_DKT_GAUSS = ((0.5, 0.0), (0.5, 0.5), (0.0, 0.5))

# ドリリング自由度の架空剛性係数（膜剛性に対する相対値）。
# 面法線まわり回転を拘束し系の特異性を除くためのもので、結果には実質影響
# しないよう十分小さく取る（フラットシェル定番の手当）。
DRILLING_FACTOR = 1.0e-3


def shell_local_frame(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray):
    """三角形の局所座標系 R と局所節点座標を返す。

    戻り値 ``(R, x, y, area)``:
      R    : 3x3 方向余弦行列（各行が局所 x, y, z 軸を全体座標で表す）。
             v_local = R @ v_global。
      x, y : 3 節点の局所平面内座標（節点1を原点, 節点2 を +x 上に取る）。
      area : 三角形面積。
    """
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    p3 = np.asarray(p3, dtype=float)

    e1 = p2 - p1
    L1 = np.linalg.norm(e1)
    if L1 == 0:
        raise ValueError("シェル要素の辺長がゼロです（節点1,2が重複）")
    e1 = e1 / L1

    normal = np.cross(p2 - p1, p3 - p1)
    nlen = np.linalg.norm(normal)
    if nlen < 1e-14:
        raise ValueError("シェル要素の3節点が同一直線上にあります")
    e3 = normal / nlen
    e2 = np.cross(e3, e1)

    R = np.vstack([e1, e2, e3])

    # 局所平面内座標（節点1を原点）。node1=(0,0), node2=(L1,0)。
    d2 = p2 - p1
    d3 = p3 - p1
    x = np.array([0.0, e1 @ d2, e1 @ d3])
    y = np.array([0.0, e2 @ d2, e2 @ d3])  # = [0, 0, +y3]
    area = 0.5 * nlen
    return R, x, y, area


def _plane_stress_D(E: float, nu: float) -> np.ndarray:
    """平面応力の構成行列（係数 E/(1-ν^2) は呼び出し側で付与）。"""
    return np.array(
        [[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, (1.0 - nu) / 2.0]]
    )


def cst_membrane_stiffness(E, nu, t, x, y, area):
    """定ひずみ三角形（CST）の膜剛性。

    自由度順 [u1, v1, u2, v2, u3, v3] の 6x6 行列を返す。
    """
    b = np.array([y[1] - y[2], y[2] - y[0], y[0] - y[1]])  # ∂N/∂x * 2A
    c = np.array([x[2] - x[1], x[0] - x[2], x[1] - x[0]])  # ∂N/∂y * 2A
    B = np.zeros((3, 6))
    for i in range(3):
        B[0, 2 * i] = b[i]
        B[1, 2 * i + 1] = c[i]
        B[2, 2 * i] = c[i]
        B[2, 2 * i + 1] = b[i]
    B /= 2.0 * area
    D = (E / (1.0 - nu**2)) * _plane_stress_D(E, nu)
    return t * area * (B.T @ D @ B), B, D


def _dkt_params(x, y):
    """DKT の辺パラメータ (P, q, r, t) を返す（添字 4,5,6 を 0,1,2 に対応）。"""
    # 辺差分（ij = i - j）。k=0:辺23, k=1:辺31, k=2:辺12
    xij = np.array([x[1] - x[2], x[2] - x[0], x[0] - x[1]])
    yij = np.array([y[1] - y[2], y[2] - y[0], y[0] - y[1]])
    l2 = xij**2 + yij**2
    P = -6.0 * xij / l2
    t = -6.0 * yij / l2
    q = 3.0 * xij * yij / l2
    r = 3.0 * yij**2 / l2
    return P, q, r, t


def _dkt_H_derivs(P, q, r, t, xi, eta):
    """DKT 形状関数 βx, βy の (ξ, η) 偏微分ベクトル（各 9 成分）を返す。

    自由度順 [w1, θx1, θy1, w2, θx2, θy2, w3, θx3, θy3]。
    P,q,r,t は添字 4,5,6 を配列インデックス 0,1,2 に対応させた辺パラメータ。
    """
    P4, P5, P6 = P
    q4, q5, q6 = q
    r4, r5, r6 = r
    t4, t5, t6 = t

    HxXi = np.array([
        P6 * (1 - 2 * xi) + (P5 - P6) * eta,
        q6 * (1 - 2 * xi) - (q5 + q6) * eta,
        -4 + 6 * (xi + eta) + r6 * (1 - 2 * xi) - eta * (r5 + r6),
        -P6 * (1 - 2 * xi) + eta * (P4 + P6),
        q6 * (1 - 2 * xi) - eta * (q6 - q4),
        -2 + 6 * xi + r6 * (1 - 2 * xi) + eta * (r4 - r6),
        -eta * (P5 + P4),
        eta * (q4 - q5),
        -eta * (r5 - r4),
    ])
    HyXi = np.array([
        t6 * (1 - 2 * xi) + (t5 - t6) * eta,
        1 + r6 * (1 - 2 * xi) - (r5 + r6) * eta,
        -q6 * (1 - 2 * xi) + eta * (q5 + q6),
        -t6 * (1 - 2 * xi) + eta * (t4 + t6),
        -1 + r6 * (1 - 2 * xi) + eta * (r4 - r6),
        -q6 * (1 - 2 * xi) - eta * (q4 - q6),
        -eta * (t4 + t5),
        eta * (r4 - r5),
        -eta * (q4 - q5),
    ])
    HxEta = np.array([
        -P5 * (1 - 2 * eta) - (P6 - P5) * xi,
        q5 * (1 - 2 * eta) - (q5 + q6) * xi,
        -4 + 6 * (xi + eta) + r5 * (1 - 2 * eta) - xi * (r5 + r6),
        xi * (P4 + P6),
        xi * (q4 - q6),
        -xi * (r6 - r4),
        P5 * (1 - 2 * eta) - xi * (P4 + P5),
        q5 * (1 - 2 * eta) + xi * (q4 - q5),
        -2 + 6 * eta + r5 * (1 - 2 * eta) + xi * (r4 - r5),
    ])
    HyEta = np.array([
        -t5 * (1 - 2 * eta) - (t6 - t5) * xi,
        1 + r5 * (1 - 2 * eta) - (r5 + r6) * xi,
        -q5 * (1 - 2 * eta) + xi * (q5 + q6),
        xi * (t4 + t6),
        xi * (r4 - r6),
        -xi * (q4 - q6),
        t5 * (1 - 2 * eta) - xi * (t4 + t5),
        -1 + r5 * (1 - 2 * eta) + xi * (r4 - r5),
        -q5 * (1 - 2 * eta) - xi * (q4 - q5),
    ])
    return HxXi, HyXi, HxEta, HyEta


def dkt_curvature_B(x, y, area, xi, eta):
    """DKT の曲率-変位行列 B (3x9) を局所点 (ξ,η) で返す。

    曲率 κ = [κx, κy, κxy] = B q（q は曲げ自由度 9 成分）。
    """
    P, q, r, t = _dkt_params(x, y)
    HxXi, HyXi, HxEta, HyEta = _dkt_H_derivs(P, q, r, t, xi, eta)

    x31 = x[2] - x[0]
    x12 = x[0] - x[1]
    y31 = y[2] - y[0]
    y12 = y[0] - y[1]

    B = np.zeros((3, 9))
    B[0] = y31 * HxXi + y12 * HxEta
    B[1] = -x31 * HyXi - x12 * HyEta
    B[2] = -x31 * HxXi - x12 * HxEta + y31 * HyXi + y12 * HyEta
    return B / (2.0 * area)


def dkt_bending_stiffness(E, nu, t, x, y, area):
    """DKT 板曲げ剛性。

    自由度順 [w1, θx1, θy1, w2, θx2, θy2, w3, θx3, θy3] の 9x9 行列を返す。
    """
    Db = (E * t**3 / (12.0 * (1.0 - nu**2))) * _plane_stress_D(E, nu)
    K = np.zeros((9, 9))
    for xi, eta in _DKT_GAUSS:
        B = dkt_curvature_B(x, y, area, xi, eta)
        K += B.T @ Db @ B
    # ∫ dA = 2A * Σ w_gp(=1/6) -> (A/3) Σ
    return (area / 3.0) * K


def drilling_stiffness(E, t, area):
    """ドリリング自由度（θz）の架空剛性 3x3。

    剛体回転（全節点等回転）はエネルギーを持たないよう [1,-.5,-.5] 型に取る。
    """
    k = DRILLING_FACTOR * E * t * area
    return k * np.array([
        [1.0, -0.5, -0.5],
        [-0.5, 1.0, -0.5],
        [-0.5, -0.5, 1.0],
    ])


# 局所 18 自由度のうち、各サブ系が占めるインデックス（節点ごと u,v,w,θx,θy,θz）
_MEMBRANE_DOF = np.array([0, 1, 6, 7, 12, 13])   # u,v of n1,n2,n3
_BENDING_DOF = np.array([2, 3, 4, 8, 9, 10, 14, 15, 16])  # w,θx,θy
_DRILL_DOF = np.array([5, 11, 17])  # θz


def shell_local_stiffness(E, nu, t, x, y, area):
    """局所座標系での 18x18 シェル剛性（節点ごと [u,v,w,θx,θy,θz] 順）。"""
    Km, _, _ = cst_membrane_stiffness(E, nu, t, x, y, area)
    Kb = dkt_bending_stiffness(E, nu, t, x, y, area)
    Kd = drilling_stiffness(E, t, area)

    K = np.zeros((18, 18))
    K[np.ix_(_MEMBRANE_DOF, _MEMBRANE_DOF)] += Km
    K[np.ix_(_BENDING_DOF, _BENDING_DOF)] += Kb
    K[np.ix_(_DRILL_DOF, _DRILL_DOF)] += Kd
    return K


def shell_transformation(R: np.ndarray) -> np.ndarray:
    """3x3 の R から 18x18 の変換行列 T を組む（6 ブロック対角）。"""
    T = np.zeros((18, 18))
    for b in range(6):
        T[3 * b : 3 * b + 3, 3 * b : 3 * b + 3] = R
    return T


def shell_stiffness_global(
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    mat: Material,
    thickness: float,
) -> np.ndarray:
    """全体座標系での 18x18 シェル要素剛性 K = T^T k T。

    自由度順は節点ごと [u_x,u_y,u_z,θx,θy,θz]（全体系）で、節点1,2,3 の順。
    """
    R, x, y, area = shell_local_frame(p1, p2, p3)
    k_local = shell_local_stiffness(mat.E, mat.nu, thickness, x, y, area)
    T = shell_transformation(R)
    return T.T @ k_local @ T
