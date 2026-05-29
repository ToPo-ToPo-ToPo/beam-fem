"""サイジング最適化のための断面ファミリ（設計変数→断面 + 解析的微分）。

設計変数は断面の「スケール係数」s。基準断面の全寸法を s 倍するため、
任意の断面形状（矩形・I形・箱型・パイプ等）に適用できる::

    A  = A0  * s^2
    Iy = Iy0 * s^4 ,  Iz = Iz0 * s^4 ,  J = J0 * s^4
    cy = cy0 * s   ,  cz = cz0 * s          (縁端距離は線寸法に比例)

せん断係数 ky, kz は形状不変なので変えない。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..material import Section


@dataclass
class ScaledSection:
    """基準断面をスケール係数 s で相似拡大する断面ファミリ。"""

    base: Section

    def make(self, s: float) -> Section:
        """スケール係数 s の断面を返す。"""
        b = self.base
        cy = None if b.cy is None else b.cy * s
        cz = None if b.cz is None else b.cz * s
        return Section(
            A=b.A * s**2,
            Iy=b.Iy * s**4,
            Iz=b.Iz * s**4,
            J=b.J * s**4,
            ky=b.ky,
            kz=b.kz,
            cy=cy,
            cz=cz,
            name=b.name,
        )

    def derivs(self, s: float) -> dict:
        """s に関する諸量の微分 dA/ds, dIy/ds, dIz/ds, dJ/ds。"""
        b = self.base
        return {
            "A": 2.0 * b.A * s,
            "Iy": 4.0 * b.Iy * s**3,
            "Iz": 4.0 * b.Iz * s**3,
            "J": 4.0 * b.J * s**3,
        }

    def deriv_cy(self, s: float) -> float:
        return 0.0 if self.base.cy is None else self.base.cy

    def deriv_cz(self, s: float) -> float:
        return 0.0 if self.base.cz is None else self.base.cz
