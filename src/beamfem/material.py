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

    @classmethod
    def pipe(cls, d: float, t: float, **kw) -> "Section":
        """中空円形断面（パイプ）。外径 d, 肉厚 t。"""
        import math

        di = d - 2.0 * t
        if di <= 0:
            raise ValueError("肉厚 t が外径 d に対して大きすぎます")
        A = math.pi * (d**2 - di**2) / 4.0
        I = math.pi * (d**4 - di**4) / 64.0
        J = math.pi * (d**4 - di**4) / 32.0
        ky = kw.pop("ky", 0.5)  # 薄肉円管の近似せん断係数
        kz = kw.pop("kz", 0.5)
        return cls(A=A, Iy=I, Iz=I, J=J, ky=ky, kz=kz, cy=d / 2.0, cz=d / 2.0, **kw)

    @classmethod
    def box(cls, b: float, h: float, t: float, **kw) -> "Section":
        """矩形中空断面（箱型・角形鋼管）。

        幅 b（局所 z 方向）, 高さ h（局所 y 方向）, 肉厚 t（一様）。
        ねじりは閉断面の Bredt 式（薄肉）で評価。
        """
        bi, hi = b - 2.0 * t, h - 2.0 * t
        if bi <= 0 or hi <= 0:
            raise ValueError("肉厚 t が外形に対して大きすぎます")
        A = b * h - bi * hi
        Iz = (b * h**3 - bi * hi**3) / 12.0  # 局所 z 軸まわり
        Iy = (h * b**3 - hi * bi**3) / 12.0  # 局所 y 軸まわり
        # 閉断面ねじり定数（中心線で囲む面積 Am と周長で評価）
        bm, hm = b - t, h - t
        Am = bm * hm
        J = 2.0 * t * Am**2 / (bm + hm)
        # せん断有効断面の近似: 鉛直せん断は2枚のウェブ、水平せん断は2枚のフランジ
        ky = kw.pop("ky", (2.0 * t * h) / A)
        kz = kw.pop("kz", (2.0 * t * b) / A)
        return cls(A=A, Iy=Iy, Iz=Iz, J=J, ky=ky, kz=kz, cy=h / 2.0, cz=b / 2.0, **kw)

    @classmethod
    def i_section(cls, h: float, bf: float, tf: float, tw: float, **kw) -> "Section":
        """I 形断面（H 形鋼・広幅フランジ）。

        全せい h（局所 y 方向, ウェブ鉛直）, フランジ幅 bf（局所 z 方向）,
        フランジ厚 tf, ウェブ厚 tw。Iz が強軸（鉛直荷重に抵抗）。
        ねじりは開断面の St.Venant 近似 J≈Σ(1/3)b t^3（そりは無視）。
        """
        hw = h - 2.0 * tf  # ウェブ高さ
        if hw <= 0:
            raise ValueError("フランジ厚 tf が全せい h に対して大きすぎます")
        A = 2.0 * bf * tf + hw * tw
        # 強軸 Iz: 外形矩形から両側の空隙を引く
        Iz = (bf * h**3 - (bf - tw) * hw**3) / 12.0
        # 弱軸 Iy: フランジ2枚 + ウェブ
        Iy = 2.0 * (tf * bf**3 / 12.0) + hw * tw**3 / 12.0
        # 開断面の St.Venant ねじり定数
        J = (2.0 * bf * tf**3 + hw * tw**3) / 3.0
        # せん断: 鉛直はウェブ、水平はフランジが主に負担（近似）
        ky = kw.pop("ky", (tw * h) / A)
        kz = kw.pop("kz", (5.0 / 6.0) * (2.0 * bf * tf) / A)
        return cls(A=A, Iy=Iy, Iz=Iz, J=J, ky=ky, kz=kz, cy=h / 2.0, cz=bf / 2.0, **kw)
