"""有限要素モデルの定義（節点・要素・境界条件・荷重）。

3D を基本とし、各節点 6 自由度。2D 問題も同じデータ構造で表現でき、
面外自由度 (u_z, theta_x, theta_y) を拘束すれば面内骨組として解ける
（`Model.fix_to_plane_xy` を利用）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .material import Material, Section

DOF_PER_NODE = 6
# 自由度ラベル（節点内インデックス）
UX, UY, UZ, RX, RY, RZ = range(6)


@dataclass
class Element:
    """2節点梁要素。"""

    n1: int  # 節点インデックス
    n2: int
    mat: Material
    sec: Section
    vref: np.ndarray | None = None  # 局所y軸の参照ベクトル（断面の向き）


@dataclass
class ShellElement:
    """3節点フラットシェル要素（膜 CST + 板曲げ DKT）。

    各節点 6 自由度（梁と共通の並び）。局所座標系は 3 節点の位置から決まる
    （詳細は :mod:`beamfem.shell3d`）。thickness は板厚 [m]。
    """

    n1: int
    n2: int
    n3: int
    mat: Material
    thickness: float


@dataclass
class Model:
    """梁構造モデル。

    nodes : (N, 3) の節点座標配列
    elements : Element のリスト
    """

    nodes: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    elements: list[Element] = field(default_factory=list)
    shells: list[ShellElement] = field(default_factory=list)
    # 拘束: {(node, local_dof): 強制変位値}。値0で固定支持。
    constraints: dict[tuple[int, int], float] = field(default_factory=dict)
    # 節点荷重: {(node, local_dof): 値}
    nodal_loads: dict[tuple[int, int], float] = field(default_factory=dict)

    # ---- 構築用ヘルパ ----
    def add_node(self, x: float, y: float, z: float = 0.0) -> int:
        """節点を追加しインデックスを返す。"""
        p = np.array([[x, y, z]], dtype=float)
        self.nodes = np.vstack([self.nodes, p]) if self.nodes.size else p
        return len(self.nodes) - 1

    def add_element(
        self,
        n1: int,
        n2: int,
        mat: Material,
        sec: Section,
        vref: np.ndarray | None = None,
    ) -> int:
        """要素を追加しインデックスを返す。"""
        self.elements.append(Element(n1, n2, mat, sec, vref))
        return len(self.elements) - 1

    def add_shell(
        self,
        n1: int,
        n2: int,
        n3: int,
        mat: Material,
        thickness: float,
    ) -> int:
        """3節点フラットシェル要素を追加しインデックスを返す。"""
        self.shells.append(ShellElement(n1, n2, n3, mat, thickness))
        return len(self.shells) - 1

    def fix(self, node: int, dofs: list[int] | None = None) -> None:
        """節点を固定する。dofs 省略時は全6自由度（完全固定）。"""
        if dofs is None:
            dofs = list(range(DOF_PER_NODE))
        for d in dofs:
            self.constraints[(node, d)] = 0.0

    def pin(self, node: int) -> None:
        """ピン支持（並進3自由度を固定、回転自由）。"""
        self.fix(node, [UX, UY, UZ])

    def add_load(self, node: int, dof: int, value: float) -> None:
        """節点荷重・モーメントを追加（同一自由度は加算）。"""
        key = (node, dof)
        self.nodal_loads[key] = self.nodal_loads.get(key, 0.0) + value

    def fix_to_plane_xy(self) -> None:
        """全節点の面外自由度 (u_z, theta_x, theta_y) を拘束し、
        x-y 面内の 2D 骨組として解けるようにする。"""
        for i in range(len(self.nodes)):
            for d in (UZ, RX, RY):
                self.constraints.setdefault((i, d), 0.0)

    # ---- 情報 ----
    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_dof(self) -> int:
        return self.n_nodes * DOF_PER_NODE

    def element_length(self, e: Element) -> float:
        return float(np.linalg.norm(self.nodes[e.n2] - self.nodes[e.n1]))
