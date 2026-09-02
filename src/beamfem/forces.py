"""要素内力・応力の回収と出力。

各要素の局所端力を ``f = k_local @ (T @ u_global)`` で求め、部材内の断面力
分布を得る（節点荷重のみ＝部材内分布荷重なしの前提では、軸力・せん断・
ねじりは一定、曲げモーメントは線形）。

内力成分（局所座標, 引張・右ねじを正）::

    N   軸力（引張正）
    Vy  局所y方向せん断
    Vz  局所z方向せん断
    T   ねじりモーメント
    My  局所y軸まわり曲げ（x-z面）
    Mz  局所z軸まわり曲げ（x-y面）

応力成分::

    sigma_a    軸応力 N/A
    sigma_b    曲げによる縁端応力の最大 |Mz|cy/Iz + |My|cz/Iy
    sigma_max  合成縁端応力 |N/A| + sigma_b（保守的な評価）
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .element3d import (
    local_stiffness,
    released_local_stiffness,
    rigid_offset_matrix,
    rotation_matrix,
    transformation_matrix,
)
from .model import Model, TrussElement
from .solver import StaticResult
from .truss3d import truss_axial_force

# 内力成分と応力成分のキー
FORCE_COMPONENTS = ("N", "Vy", "Vz", "T", "My", "Mz")
STRESS_COMPONENTS = ("sigma_a", "sigma_b", "sigma_max")
ALL_COMPONENTS = FORCE_COMPONENTS + STRESS_COMPONENTS

# 表示用の単位ヒント（出力ラベルにのみ使用）
_LABELS = {
    "N": "N [N]",
    "Vy": "Vy [N]",
    "Vz": "Vz [N]",
    "T": "T [N·m]",
    "My": "My [N·m]",
    "Mz": "Mz [N·m]",
    "sigma_a": "σ_a [Pa]",
    "sigma_b": "σ_b [Pa]",
    "sigma_max": "σ_max [Pa]",
}


@dataclass
class ElementForces:
    """1要素の内力。端値（節点1, 節点2）を基本に分布を構成する。"""

    index: int
    L: float
    f_local: np.ndarray  # 12成分の局所端力
    sec: object  # Section（応力計算に使用）
    axial_only: bool = False
    axial_strain: float | None = None
    axial_extension: float | None = None

    # ---- 端での内力値 (node1, node2) ----
    def ends(self, comp: str) -> tuple[float, float]:
        """成分 comp の (節点1値, 節点2値) を返す。"""
        f = self.f_local
        if comp == "N":  # 引張正
            return (-f[0], f[6])
        if comp == "Vy":
            return (f[1], -f[7])
        if comp == "Vz":
            return (f[2], -f[8])
        if comp == "T":
            return (-f[3], f[9])
        if comp == "My":
            return (-f[4], f[10])
        if comp == "Mz":
            return (-f[5], f[11])
        raise KeyError(f"未知の内力成分: {comp}")

    def value(self, comp: str, xi: np.ndarray | float):
        """局所位置 xi∈[0,1] における内力値（線形補間）。"""
        v0, v1 = self.ends(comp)
        xi = np.asarray(xi, dtype=float)
        return (1.0 - xi) * v0 + xi * v1

    def max_abs(self, comp: str) -> float:
        """成分の絶対値最大（端値で評価、分布は線形のため十分）。"""
        v0, v1 = self.ends(comp)
        return float(max(abs(v0), abs(v1)))

    # ---- 応力 ----
    def _bending_stress_at(self, My: float, Mz: float) -> float:
        if self.axial_only:
            return 0.0
        sec = self.sec
        s = 0.0
        if sec.cy is not None:
            s += abs(Mz) * sec.cy / sec.Iz
        if sec.cz is not None:
            s += abs(My) * sec.cz / sec.Iy
        return s

    def stress_ends(self, comp: str) -> tuple[float, float]:
        """応力成分の (節点1, 節点2) 値。"""
        A = self.sec.A
        N0, N1 = self.ends("N")
        My0, My1 = self.ends("My")
        Mz0, Mz1 = self.ends("Mz")
        if comp == "sigma_a":
            return (N0 / A, N1 / A)
        sb0 = self._bending_stress_at(My0, Mz0)
        sb1 = self._bending_stress_at(My1, Mz1)
        if comp == "sigma_b":
            return (sb0, sb1)
        if comp == "sigma_max":
            return (abs(N0 / A) + sb0, abs(N1 / A) + sb1)
        raise KeyError(f"未知の応力成分: {comp}")

    def get_ends(self, comp: str) -> tuple[float, float]:
        """内力・応力どちらの成分でも端値を返す統一アクセサ。"""
        if comp in FORCE_COMPONENTS:
            return self.ends(comp)
        if comp in STRESS_COMPONENTS:
            return self.stress_ends(comp)
        raise KeyError(f"未知の成分: {comp}")

    def get_max_abs(self, comp: str) -> float:
        e0, e1 = self.get_ends(comp)
        return float(max(abs(e0), abs(e1)))


@dataclass
class ForceResults:
    """全要素の内力結果コンテナ。出力（表/CSV）を担う。"""

    model: Model
    elements: list[ElementForces]

    def __getitem__(self, i: int) -> ElementForces:
        return self.elements[i]

    def __len__(self) -> int:
        return len(self.elements)

    # ------------------------------------------------------------------
    # 出力（表示する項目を items で指定。常に全項目を出さない）
    # ------------------------------------------------------------------
    def _rows(self, items, at, element_ids):
        ids = element_ids if element_ids is not None else range(len(self.elements))
        rows = []
        for i in ids:
            ef = self.elements[i]
            if at == "max":
                rows.append((str(i), [ef.get_max_abs(c) for c in items]))
            else:  # "ends": 各端を別行に
                e0 = [ef.get_ends(c)[0] for c in items]
                e1 = [ef.get_ends(c)[1] for c in items]
                rows.append((f"{i}@n1", e0))
                rows.append((f"{i}@n2", e1))
        return rows

    def table(
        self,
        items=("N", "Vy", "Mz"),
        at: str = "max",
        element_ids=None,
        fmt: str = "{:12.4g}",
    ) -> str:
        """指定した項目のみの表を文字列で返す。

        items : 表示する成分（FORCE_COMPONENTS / STRESS_COMPONENTS から選択）
        at    : "max"=要素内絶対値最大 / "ends"=両端値を別行で
        element_ids : 表示する要素番号（省略時は全要素）
        """
        items = list(items)
        for c in items:
            if c not in ALL_COMPONENTS:
                raise KeyError(f"未知の成分: {c}. 選択肢={ALL_COMPONENTS}")
        header = ["elem"] + [_LABELS[c] for c in items]
        rows = self._rows(items, at, element_ids)

        widths = [max(len(header[0]), max((len(r[0]) for r in rows), default=0))]
        cells = []
        for name, vals in rows:
            cells.append([name] + [fmt.format(v) for v in vals])
        for j in range(1, len(header)):
            w = max(len(header[j]), max((len(c[j]) for c in cells), default=0))
            widths.append(w)

        def line(parts):
            return "  ".join(p.rjust(widths[j]) for j, p in enumerate(parts))

        out = [line(header), line(["-" * w for w in widths])]
        out += [line(c) for c in cells]
        return "\n".join(out)

    def print_table(self, items=("N", "Vy", "Mz"), at: str = "max", element_ids=None) -> None:
        """指定した項目のみを標準出力に表示する。"""
        print(self.table(items=items, at=at, element_ids=element_ids))

    def to_csv(self, path: str, items=FORCE_COMPONENTS, at: str = "ends") -> str:
        """指定した項目を CSV 出力する。相対パスは workspace フォルダに保存し、保存先を返す。"""
        import csv

        from .workspace import resolve

        path = resolve(path)
        items = list(items)
        with open(path, "w", newline="") as fp:
            w = csv.writer(fp)
            w.writerow(["elem", "location"] + list(items))
            for i, ef in enumerate(self.elements):
                if at == "max":
                    w.writerow([i, "max_abs"] + [ef.get_max_abs(c) for c in items])
                else:
                    w.writerow([i, "n1"] + [ef.get_ends(c)[0] for c in items])
                    w.writerow([i, "n2"] + [ef.get_ends(c)[1] for c in items])
        return path

    def as_dict(self, items=FORCE_COMPONENTS, at: str = "ends") -> dict:
        """プログラムからの利用向けに辞書で返す。"""
        out = {}
        for i, ef in enumerate(self.elements):
            if at == "max":
                out[i] = {c: ef.get_max_abs(c) for c in items}
            else:
                out[i] = {c: ef.get_ends(c) for c in items}
        return out


def recover_forces(model: Model, result: StaticResult) -> ForceResults:
    """静解析結果から全要素の内力を回収する。"""
    from .assembly import element_dof_map

    dof_maps = element_dof_map(model)
    efs: list[ElementForces] = []
    for i, (e, dofs) in enumerate(zip(model.elements, dof_maps)):
        p1, p2 = model.nodes[e.n1], model.nodes[e.n2]
        L = float(np.linalg.norm(p2 - p1))
        u_elem = result.u[dofs]
        if isinstance(e, TrussElement):
            axial = truss_axial_force(p1, p2, e.mat, e.sec, u_elem)
            strain = axial / (e.mat.E * e.sec.A)
            extension = strain * L
            f_local = np.zeros(12)
            f_local[0], f_local[6] = -axial, axial
        else:
            k = released_local_stiffness(
                local_stiffness(e.mat.E, e.mat.G, L, e.sec),
                e.release_n1, e.release_n2,
            )
            R = rotation_matrix(p1, p2, e.vref)
            T = transformation_matrix(R)
            G = rigid_offset_matrix(e.offset)
            if G is not None:
                u_elem = G @ u_elem  # 節点変位 → 梁図心の変位（剛体腕）
            f_local = k @ (T @ u_elem)
            strain = extension = None
        efs.append(ElementForces(index=i, L=L, f_local=f_local, sec=e.sec,
                                 axial_only=isinstance(e, TrussElement),
                                 axial_strain=strain, axial_extension=extension))
    return ForceResults(model=model, elements=efs)
