"""シェル要素の応力・断面力（応力合力）の回収と出力。

各シェル要素について、局所座標系での節点変位から
  - 膜応力      σx, σy, τxy        （板厚方向に一様、要素内で一定）
  - 曲げモーメント Mx, My, Mxy     （単位幅あたりの応力合力, 重心で評価）
  - 曲げ縁端応力 σbx, σby          （板上下面 z=±t/2 の曲げ応力 6M/t^2）
を回収する。膜は CST のため要素内一定、曲げは DKT のため重心 (ξ=η=1/3) で
代表値を取る。
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
from .shell_mitc4 import (
    quad_shell_frame,
    quad_shell_stress_resultants,
    quad_shell_transformation,
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
    """全シェル要素の応力・断面力コンテナ。

    shells       : 3節点シェル（CST+DKT）の結果
    quad_shells  : 4節点シェル（Q4+MITC4）の結果
    """

    model: Model
    shells: list[ShellForces]
    quad_shells: list[ShellForces] = field(default_factory=list)

    def __getitem__(self, i: int) -> ShellForces:
        return self.shells[i]

    def __len__(self) -> int:
        return len(self.shells)

    def table(self, items=("sx", "sy", "sxy"), fmt: str = "{:12.4g}",
              which: str = "all") -> str:
        """指定した成分の一覧表を文字列で返す。

        which : "tri"=3節点のみ / "quad"=4節点のみ / "all"=両方（接頭辞 T/Q）。
        """
        items = list(items)
        for c in items:
            if c not in ALL_SHELL_COMPONENTS:
                raise KeyError(f"未知の成分: {c}. 選択肢={ALL_SHELL_COMPONENTS}")
        rows = []
        if which in ("tri", "all"):
            rows += [(f"T{i}", sf) for i, sf in enumerate(self.shells)]
        if which in ("quad", "all"):
            rows += [(f"Q{i}", sf) for i, sf in enumerate(self.quad_shells)]

        header = ["shell"] + [_LABELS[c] for c in items]
        cells = [[name] + [fmt.format(sf.get(c)) for c in items] for name, sf in rows]
        widths = [
            max(len(header[j]), max((len(r[j]) for r in cells), default=0))
            for j in range(len(header))
        ]

        def line(parts):
            return "  ".join(p.rjust(widths[j]) for j, p in enumerate(parts))

        out = [line(header), line(["-" * w for w in widths])]
        out += [line(c) for c in cells]
        return "\n".join(out)

    def print_table(self, items=("sx", "sy", "sxy"), which: str = "all") -> None:
        print(self.table(items=items, which=which))

    def as_dict(self, items=ALL_SHELL_COMPONENTS) -> dict:
        """3節点・4節点の結果を {"tri": {...}, "quad": {...}} で返す。"""
        return {
            "tri": {i: {c: sf.get(c) for c in items} for i, sf in enumerate(self.shells)},
            "quad": {i: {c: sf.get(c) for c in items}
                     for i, sf in enumerate(self.quad_shells)},
        }


def recover_shell_forces(model: Model, result: StaticResult) -> ShellForceResults:
    """静解析結果から全シェル要素（3節点・4節点）の応力・断面力を回収する。"""
    from .assembly import quad_shell_dof_map, shell_dof_map

    # --- 3節点シェル（CST + DKT）---
    out: list[ShellForces] = []
    for i, (s, dofs) in enumerate(zip(model.shells, shell_dof_map(model))):
        p1, p2, p3 = model.nodes[s.n1], model.nodes[s.n2], model.nodes[s.n3]
        R, x, y, area = shell_local_frame(p1, p2, p3)
        T = shell_transformation(R)
        d_local = T @ result.u[dofs]  # 局所 18 自由度

        E, nu, t = s.mat.E, s.mat.nu, s.thickness

        # 膜応力（CST: 要素内一定）
        _, Bm, _ = cst_membrane_stiffness(E, nu, t, x, y, area)
        Dm = (E / (1.0 - nu**2)) * _plane_stress_D(E, nu)
        sig = Dm @ (Bm @ d_local[_MEMBRANE_DOF])

        # 曲げモーメント（DKT: 重心 ξ=η=1/3 で評価）
        Bb = dkt_curvature_B(x, y, area, 1.0 / 3.0, 1.0 / 3.0)
        Db = (E * t**3 / (12.0 * (1.0 - nu**2))) * _plane_stress_D(E, nu)
        mom = Db @ (Bb @ d_local[_BENDING_DOF])

        out.append(ShellForces(index=i, thickness=t, membrane_stress=sig, moment=mom))

    # --- 4節点シェル（Q4 + MITC4, 要素中心で評価）---
    quad_out: list[ShellForces] = []
    for i, (s, dofs) in enumerate(zip(model.quad_shells, quad_shell_dof_map(model))):
        p = [model.nodes[s.n1], model.nodes[s.n2], model.nodes[s.n3], model.nodes[s.n4]]
        R, x, y = quad_shell_frame(*p)
        T = quad_shell_transformation(R)
        d_local = T @ result.u[dofs]  # 局所 24 自由度
        sig, mom = quad_shell_stress_resultants(s.mat.E, s.mat.nu, s.thickness, x, y, d_local)
        quad_out.append(
            ShellForces(index=i, thickness=s.thickness, membrane_stress=sig, moment=mom)
        )

    return ShellForceResults(model=model, shells=out, quad_shells=quad_out)
