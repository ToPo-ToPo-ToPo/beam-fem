"""材料断面の定義。

単位系は SI 一貫 (N, m, Pa, kg) を推奨。本ライブラリは単位を強制せず、
入力された値の一貫性をユーザーに委ねる。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Material:
    """線形等方弾性材料。

    Parameters
    ----------
    E : ヤング率 [Pa]
    nu : ポアソン比 [-]
    rho : 密度 [kg/m^3]（動解析・質量行列で使用）
    name : 任意の名称
    """

    E: float
    nu: float = 0.3
    rho: float = 0.0
    name: str = ""

    @property
    def G(self) -> float:
        """せん断弾性係数 G = E / (2(1+nu))。"""
        return self.E / (2.0 * (1.0 + self.nu))


@dataclass(frozen=True)
class Section:
    """梁断面のプロパティ。

    断面の主軸を局所 y, z 軸にとる。曲げは
      - 局所 z 軸まわり (面内 x-y 曲げ) -> Iz
      - 局所 y 軸まわり (面外 x-z 曲げ) -> Iy
    で表す。

    Parameters
    ----------
    A : 断面積 [m^2]
    Iy : 局所 y 軸まわりの断面二次モーメント [m^4]
    Iz : 局所 z 軸まわりの断面二次モーメント [m^4]
    J : サン・ブナンのねじり定数 [m^4]
    ky, kz : せん断補正係数 [-]。せん断有効断面積 As = k * A。
             長方形 ~5/6, 円形 ~0.9 など。Timoshenko 効果に効く。
    name : 任意の名称
    """

    A: float
    Iy: float
    Iz: float
    J: float
    ky: float = 5.0 / 6.0
    kz: float = 5.0 / 6.0
    # 中立軸から縁端までの距離（応力計算用）。cy は局所y方向, cz は局所z方向。
    # 曲げ応力: Mz 曲げは sigma = Mz*cy/Iz、My 曲げは sigma = My*cz/Iy。
    # None の場合は曲げ応力を計算できない（軸応力のみ）。
    cy: float | None = None
    cz: float | None = None
    name: str = ""

    @classmethod
    def rectangle(cls, b: float, h: float, **kw) -> "Section":
        """長方形断面 (幅 b: 局所 z 方向, 高さ h: 局所 y 方向)。"""
        A = b * h
        Iz = b * h**3 / 12.0  # 局所 z 軸まわり（高さ h が効く）
        Iy = h * b**3 / 12.0  # 局所 y 軸まわり（幅 b が効く）
        # 矩形断面のねじり定数（近似式, b>=h で a=長辺/2, ただし簡易に長短で評価）
        a = max(b, h) / 2.0
        c = min(b, h) / 2.0
        J = a * c**3 * (16.0 / 3.0 - 3.36 * (c / a) * (1.0 - (c**4) / (12.0 * a**4)))
        return cls(A=A, Iy=Iy, Iz=Iz, J=J, ky=5.0 / 6.0, kz=5.0 / 6.0, cy=h / 2.0, cz=b / 2.0, **kw)

    @classmethod
    def circle(cls, d: float, **kw) -> "Section":
        """中実円形断面（直径 d）。"""
        import math

        A = math.pi * d**2 / 4.0
        I = math.pi * d**4 / 64.0
        J = math.pi * d**4 / 32.0
        return cls(A=A, Iy=I, Iz=I, J=J, ky=0.9, kz=0.9, cy=d / 2.0, cz=d / 2.0, **kw)
