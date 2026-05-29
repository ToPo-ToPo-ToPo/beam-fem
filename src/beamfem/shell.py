"""シェル要素の応力・断面力（応力合力）の回収と出力。

各シェル要素について、局所座標系での節点変位から
  - 膜応力      σx, σy, τxy        （板厚方向に一様、要素内で一定）
  - 曲げモーメント Mx, My, Mxy     （単位幅あたりの応力合力, 重心で評価）
  - 曲げ縁端応力 σbx, σby          （板上下面 z=±t/2 の曲げ応力 6M/t^2）
を回収する。膜は CST のため要素内一定、曲げは DKT のため重心 (ξ=η=1/3) で
代表値を取る。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import Model
from .shell3d import (
    _BENDING_DOF,
    _MEMBRANE_DOF,
    _plane_stress_D,
    cst_membrane_stiffness,
    dkt_curvature_B,
    shell_local_frame,
    shell_transformation,
)
from .solver import StaticResult

# 成分キー
MEMBRANE_COMPONENTS = ("sx", "sy", "sxy")
MOMENT_COMPONENTS = ("Mx", "My", "Mxy")
BENDING_STRESS_COMPONENTS = ("sbx", "sby")
ALL_SHELL_COMPONENTS = MEMBRANE_COMPONENTS + MOMENT_COMPONENTS + BENDING_STRESS_COMPONENTS

_LABELS = {
    "sx": "σx [Pa]",
    "sy": "σy [Pa]",
    "sxy": "τxy [Pa]",
    "Mx": "Mx [N·m/m]",
    "My": "My [N·m/m]",
    "Mxy": "Mxy [N·m/m]",
    "sbx": "σbx [Pa]",
    "sby": "σby [Pa]",
}


@dataclass
class ShellForces:
    """1シェル要素の応力・断面力（局所座標系）。"""

    index: int
    thickness: float
    membrane_stress: np.ndarray  # [σx, σy, τxy]
    moment: np.ndarray  # [Mx, My, Mxy] 単位幅あたり

    def get(self, comp: str) -> float:
        if comp == "sx":
            return float(self.membrane_stress[0])
        if comp == "sy":
            return float(self.membrane_stress[1])
        if comp == "sxy":
            return float(self.membrane_stress[2])
        if comp == "Mx":
            return float(self.moment[0])
        if comp == "My":
            return float(self.moment[1])
        if comp == "Mxy":
            return float(self.moment[2])
        # 曲げ縁端応力 σ = 6 M / t^2
        if comp == "sbx":
            return float(6.0 * self.moment[0] / self.thickness**2)
        if comp == "sby":
            return float(6.0 * self.moment[1] / self.thickness**2)
        raise KeyError(f"未知の成分: {comp}. 選択肢={ALL_SHELL_COMPONENTS}")


@dataclass
class ShellForceResults:
    """全シェル要素の応力・断面力コンテナ。"""

    model: Model
    shells: list[ShellForces]

    def __getitem__(self, i: int) -> ShellForces:
        return self.shells[i]

    def __len__(self) -> int:
        return len(self.shells)

    def table(self, items=("sx", "sy", "sxy"), fmt: str = "{:12.4g}") -> str:
        """指定した成分の一覧表を文字列で返す。"""
        items = list(items)
        for c in items:
            if c not in ALL_SHELL_COMPONENTS:
                raise KeyError(f"未知の成分: {c}. 選択肢={ALL_SHELL_COMPONENTS}")
        header = ["shell"] + [_LABELS[c] for c in items]
        cells = [
            [str(i)] + [fmt.format(sf.get(c)) for c in items]
            for i, sf in enumerate(self.shells)
        ]
        widths = [
            max(len(header[j]), max((len(r[j]) for r in cells), default=0))
            for j in range(len(header))
        ]

        def line(parts):
            return "  ".join(p.rjust(widths[j]) for j, p in enumerate(parts))

        out = [line(header), line(["-" * w for w in widths])]
        out += [line(c) for c in cells]
        return "\n".join(out)

    def print_table(self, items=("sx", "sy", "sxy")) -> None:
        print(self.table(items=items))

    def as_dict(self, items=ALL_SHELL_COMPONENTS) -> dict:
        return {i: {c: sf.get(c) for c in items} for i, sf in enumerate(self.shells)}


def recover_shell_forces(model: Model, result: StaticResult) -> ShellForceResults:
    """静解析結果から全シェル要素の応力・断面力を回収する。"""
    from .assembly import shell_dof_map

    maps = shell_dof_map(model)
    out: list[ShellForces] = []
    for i, (s, dofs) in enumerate(zip(model.shells, maps)):
        p1, p2, p3 = model.nodes[s.n1], model.nodes[s.n2], model.nodes[s.n3]
        R, x, y, area = shell_local_frame(p1, p2, p3)
        T = shell_transformation(R)
        d_local = T @ result.u[dofs]  # 局所 18 自由度

        E, nu, t = s.mat.E, s.mat.nu, s.thickness

        # 膜応力（CST: 要素内一定）
        _, Bm, _ = cst_membrane_stiffness(E, nu, t, x, y, area)
        Dm = (E / (1.0 - nu**2)) * _plane_stress_D(E, nu)
        eps = Bm @ d_local[_MEMBRANE_DOF]
        sig = Dm @ eps

        # 曲げモーメント（DKT: 重心 ξ=η=1/3 で評価）
        Bb = dkt_curvature_B(x, y, area, 1.0 / 3.0, 1.0 / 3.0)
        Db = (E * t**3 / (12.0 * (1.0 - nu**2))) * _plane_stress_D(E, nu)
        kappa = Bb @ d_local[_BENDING_DOF]
        mom = Db @ kappa

        out.append(
            ShellForces(index=i, thickness=t, membrane_stress=sig, moment=mom)
        )
    return ShellForceResults(model=model, shells=out)
