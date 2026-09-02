"""2D/3D空間で任意方向を向く2節点軸力トラス要素。"""

from __future__ import annotations

import numpy as np

from .material import Material, Section


def _geometry_and_rigidity(p1, p2, material, section):
    first = np.asarray(p1, dtype=float)
    second = np.asarray(p2, dtype=float)
    if first.shape != (3,) or second.shape != (3,) or not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        raise ValueError("トラス節点座標は有限な3成分ベクトルでなければなりません")
    delta = second - first
    length = float(np.linalg.norm(delta))
    if not np.isfinite(length) or length <= 0.0:
        raise ValueError("トラス要素長は正の有限値でなければなりません")
    rigidity = float(material.E) * float(section.A)
    if not np.isfinite(rigidity) or material.E <= 0.0 or section.A <= 0.0:
        raise ValueError("トラスのヤング率と断面積は正の有限値でなければなりません")
    return delta, length, rigidity


def truss_stiffness_global(
    p1: np.ndarray,
    p2: np.ndarray,
    material: Material,
    section: Section,
) -> np.ndarray:
    """6自由度/節点モデル用の12x12全体座標剛性を返す。

    トラスは並進3自由度だけに剛性を持ち、節点回転には剛性を与えない。
    """

    delta, length, rigidity = _geometry_and_rigidity(p1, p2, material, section)
    direction = delta / length
    k3 = rigidity / length * np.outer(direction, direction)
    stiffness = np.zeros((12, 12), dtype=float)
    a = np.arange(3)
    b = np.arange(6, 9)
    stiffness[np.ix_(a, a)] = k3
    stiffness[np.ix_(a, b)] = -k3
    stiffness[np.ix_(b, a)] = -k3
    stiffness[np.ix_(b, b)] = k3
    return stiffness


def truss_axial_force(
    p1: np.ndarray,
    p2: np.ndarray,
    material: Material,
    section: Section,
    u_element: np.ndarray,
) -> float:
    """要素伸びから軸力 [N]（引張正）を回収する。"""

    delta, length, rigidity = _geometry_and_rigidity(p1, p2, material, section)
    displacement = np.asarray(u_element, dtype=float)
    if displacement.shape != (12,) or not np.all(np.isfinite(displacement)):
        raise ValueError("トラス要素変位は有限な12成分ベクトルでなければなりません")
    direction = delta / length
    extension = float(direction @ (displacement[6:9] - displacement[0:3]))
    return rigidity / length * extension
