"""3次元 Timoshenko 梁要素（2節点・各節点6自由度）。

自由度の並び（局所・全体とも共通）::

    [u_x, u_y, u_z, theta_x, theta_y, theta_z]  (節点1)
    [u_x, u_y, u_z, theta_x, theta_y, theta_z]  (節点2)

局所座標系::

    x : 節点1 -> 節点2 の材軸方向
    y, z : 断面の主軸（Iz は z 軸まわり=面内, Iy は y 軸まわり=面外）

Timoshenko 効果はせん断パラメータ Phi で取り込む::

    Phi_y = 12 E Iz / (k_y G A L^2)   (x-y 面内曲げ)
    Phi_z = 12 E Iy / (k_z G A L^2)   (x-z 面外曲げ)

Phi -> 0 で Euler-Bernoulli 解に一致する。
"""

from __future__ import annotations

import numpy as np

from .material import Material, Section


def local_stiffness(E: float, G: float, L: float, sec: Section) -> np.ndarray:
    """局所座標系での 12x12 要素剛性行列。"""
    A, Iy, Iz, J = sec.A, sec.Iy, sec.Iz, sec.J
    Asy = sec.ky * A
    Asz = sec.kz * A

    # せん断パラメータ
    Phi_y = 12.0 * E * Iz / (G * Asy * L**2)  # x-y 面 (Iz, theta_z)
    Phi_z = 12.0 * E * Iy / (G * Asz * L**2)  # x-z 面 (Iy, theta_y)

    k = np.zeros((12, 12))

    # 以降は上三角のみ記入し、最後に対称化する。

    # --- 軸方向 (u_x) ---
    ax = E * A / L
    k[0, 0] = ax
    k[0, 6] = -ax
    k[6, 6] = ax

    # --- ねじり (theta_x) ---
    tx = G * J / L
    k[3, 3] = tx
    k[3, 9] = -tx
    k[9, 9] = tx

    # --- x-y 面内曲げ: 自由度 u_y(1,7), theta_z(5,11), 断面 Iz ---
    ez = E * Iz / ((1.0 + Phi_y) * L**3)
    k[1, 1] = 12.0 * ez
    k[1, 5] = 6.0 * L * ez
    k[1, 7] = -12.0 * ez
    k[1, 11] = 6.0 * L * ez
    k[5, 5] = (4.0 + Phi_y) * L**2 * ez
    k[5, 7] = -6.0 * L * ez
    k[5, 11] = (2.0 - Phi_y) * L**2 * ez
    k[7, 7] = 12.0 * ez
    k[7, 11] = -6.0 * L * ez
    k[11, 11] = (4.0 + Phi_y) * L**2 * ez

    # --- x-z 面外曲げ: 自由度 u_z(2,8), theta_y(4,10), 断面 Iy ---
    # theta_y と u_z の連成は符号が反転する（dw/dx = -theta_y）
    ey = E * Iy / ((1.0 + Phi_z) * L**3)
    k[2, 2] = 12.0 * ey
    k[2, 4] = -6.0 * L * ey
    k[2, 8] = -12.0 * ey
    k[2, 10] = -6.0 * L * ey
    k[4, 4] = (4.0 + Phi_z) * L**2 * ey
    k[4, 8] = 6.0 * L * ey
    k[4, 10] = (2.0 - Phi_z) * L**2 * ey
    k[8, 8] = 12.0 * ey
    k[8, 10] = 6.0 * L * ey
    k[10, 10] = (4.0 + Phi_z) * L**2 * ey

    # 対称化（上三角のみ記入したため）
    k = k + k.T - np.diag(np.diag(k))
    return k


def rotation_matrix(p1: np.ndarray, p2: np.ndarray, vref: np.ndarray | None = None) -> np.ndarray:
    """局所→全体 の 3x3 方向余弦行列 R を返す。

    R の各行が局所 x, y, z 軸を全体座標で表したもの。すなわち
    v_local = R @ v_global。

    規約: vref は「局所 y 軸の希望方向」を与える参照ベクトル。材軸成分を
    取り除いて局所 y 軸を作るため、軸に沿った梁では局所軸が全体軸と一致する::

        e1 = 材軸 (節点1->2)
        e2 = vref から材軸成分を除き正規化 (局所 y)
        e3 = e1 x e2                          (局所 z)

    既定の vref は全体 Y 軸。材軸が全体 Y と平行な場合のみ全体 Z 軸を使う。
    これにより全体 X 方向の梁は局所軸=全体軸となり、x-y 面内曲げが Iz に対応する。
    """
    e1 = p2 - p1
    L = np.linalg.norm(e1)
    if L == 0:
        raise ValueError("要素長がゼロです（節点が重複）")
    e1 = e1 / L

    if vref is None:
        # 材軸が全体Yとほぼ平行なら参照軸をZに切替え
        if abs(e1[1]) > 1.0 - 1e-6:
            vref = np.array([0.0, 0.0, 1.0])
        else:
            vref = np.array([0.0, 1.0, 0.0])
    else:
        vref = np.asarray(vref, dtype=float)

    # vref から材軸成分を除いて局所 y 軸を作る
    e2 = vref - np.dot(vref, e1) * e1
    n2 = np.linalg.norm(e2)
    if n2 < 1e-12:
        raise ValueError("参照ベクトル vref が材軸と平行です。vref を見直してください。")
    e2 = e2 / n2
    e3 = np.cross(e1, e2)

    return np.vstack([e1, e2, e3])


def transformation_matrix(R: np.ndarray) -> np.ndarray:
    """3x3 の R から 12x12 の変換行列 T を組む（4ブロック対角）。"""
    T = np.zeros((12, 12))
    for b in range(4):
        T[3 * b : 3 * b + 3, 3 * b : 3 * b + 3] = R
    return T


def element_stiffness_global(
    p1: np.ndarray,
    p2: np.ndarray,
    mat: Material,
    sec: Section,
    vref: np.ndarray | None = None,
) -> np.ndarray:
    """全体座標系での 12x12 要素剛性行列 K = T^T k T。"""
    L = float(np.linalg.norm(p2 - p1))
    k_local = local_stiffness(mat.E, mat.G, L, sec)
    R = rotation_matrix(p1, p2, vref)
    T = transformation_matrix(R)
    return T.T @ k_local @ T
