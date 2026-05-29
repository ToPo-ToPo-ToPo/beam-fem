"""構造の生成と荷重の補助（グリラージュ・面分布荷重の等価節点化）。

円形リブ構造（放射スポーク＋同心リング）の生成や、面分布荷重（圧力）を三角形分割で
等価節点荷重へ変換する処理をまとめる。本ライブラリは節点荷重を入力とするため、面荷重は
ここで節点化する。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .material import Material, Section
from .model import UZ, Model


@dataclass
class Grillage:
    """円形リブ・グリラージュの構成情報。"""

    center: int                       # 中心節点（include_center=False なら -1）
    ring_nodes: list[list[int]]       # ring_nodes[k][j]: 半径レベル k・角度 j の節点
    radial_bands: list[list[int]]     # radial_bands[b]: バンド b の放射リブ要素番号
    rings: list[list[int]]            # rings[k]: リング k の周方向リブ要素番号
    triangles: list[tuple] = field(default_factory=list)  # 載荷面の三角形分割

    def interior_nodes(self) -> list[int]:
        """支持しない内部節点（中心＋外周以外）。"""
        ns = []
        if self.center >= 0:
            ns.append(self.center)
        for k in range(1, len(self.ring_nodes) - 1):
            ns.extend(self.ring_nodes[k])
        return ns


def radial_grillage(
    model: Model,
    mat: Material,
    sec: Section,
    R: float,
    n_radial: int,
    n_rings: int,
    include_center: bool = True,
) -> Grillage:
    """円形リブ・グリラージュ（放射スポーク＋同心リング）を model に追加する。

    水平面 (x-y) に配置。半径 R を n_rings 等分し、n_radial 本のスポークを置く。
    載荷面（円板）の三角形分割も生成する（圧力の節点化に使用）。
    """
    angles = [2.0 * np.pi * j / n_radial for j in range(n_radial)]
    radii = [R * k / n_rings for k in range(n_rings + 1)]  # radii[0]=0

    center = model.add_node(0.0, 0.0, 0.0) if include_center else -1
    ring_nodes = [[center] * n_radial]  # k=0 は中心（重複参照）
    for k in range(1, n_rings + 1):
        row = [
            model.add_node(radii[k] * np.cos(a), radii[k] * np.sin(a), 0.0)
            for a in angles
        ]
        ring_nodes.append(row)

    # 放射リブ（バンド b は半径レベル b→b+1）
    radial_bands = [[] for _ in range(n_rings)]
    for j in range(n_radial):
        if include_center:
            radial_bands[0].append(model.add_element(center, ring_nodes[1][j], mat, sec))
        for k in range(1, n_rings):
            radial_bands[k].append(
                model.add_element(ring_nodes[k][j], ring_nodes[k + 1][j], mat, sec)
            )

    # 周方向リブ（リング k）
    rings = [[] for _ in range(n_rings + 1)]
    for k in range(1, n_rings + 1):
        for j in range(n_radial):
            jn = (j + 1) % n_radial
            rings[k].append(model.add_element(ring_nodes[k][j], ring_nodes[k][jn], mat, sec))

    # 三角形分割（内側ファン＋環状四角形を2分割）
    triangles = []
    if include_center:
        for j in range(n_radial):
            jn = (j + 1) % n_radial
            triangles.append((center, ring_nodes[1][j], ring_nodes[1][jn]))
    for k in range(1, n_rings):
        for j in range(n_radial):
            jn = (j + 1) % n_radial
            a, b = ring_nodes[k][j], ring_nodes[k][jn]
            c, d = ring_nodes[k + 1][jn], ring_nodes[k + 1][j]
            triangles.append((a, b, c))
            triangles.append((a, c, d))

    return Grillage(center, ring_nodes, radial_bands, rings, triangles)


def lump_pressure(
    model: Model,
    triangles: list[tuple],
    pressure: float,
    dof: int = UZ,
    sign: float = -1.0,
) -> float:
    """面分布荷重（圧力）を三角形分割の頂点へ 1/3 ずつ等価節点化する。

    各三角形の荷重 pressure×面積 を 3 頂点に等分配して節点荷重に加える。
    既定は下向き (-z)。総載荷荷重の大きさ（pressure×総面積）を返す。
    面積は x-y 平面で評価する。
    """
    nodes = model.nodes
    total_area = 0.0
    for (a, b, c) in triangles:
        p1, p2, p3 = nodes[a][:2], nodes[b][:2], nodes[c][:2]
        area = 0.5 * abs(
            (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])
        )
        total_area += area
        f = sign * pressure * area / 3.0
        for nd in (a, b, c):
            model.add_load(nd, dof, f)
    return pressure * total_area
