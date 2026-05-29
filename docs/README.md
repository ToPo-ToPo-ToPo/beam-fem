# beam-fem ドキュメント

梁モデルの FEM 解析と構造最適化ライブラリ `beamfem` の理論・数式・実装をまとめる。

数式は GitHub の Markdown 数式（`$...$`, `$$...$$`）で記述している。

## 目次

| 章 | 内容 | 主な実装 |
|---|---|---|
| [1. FEM 理論](01_fem_theory.md) | 3D Timoshenko 梁要素の剛性行列・座標変換 | `element3d.py` |
| [2. ソルバ](02_solver.md) | 全体行列の組み立て・境界条件・静的線形解析 | `assembly.py`, `solver.py` |
| [3. 内力・応力](03_forces_stress.md) | 断面力の回収・応力評価・符号規約 | `forces.py` |
| [4. 断面諸量](04_sections.md) | 各断面形状の A, I, J, 縁端距離の式 | `material.py` |
| [5. サイジング最適化](05_sizing_optimization.md) | 質量最小化・解析的感度（直接法）・MMA | `optimize/sizing.py`, `optimize/mma.py` |
| [6. トポロジー最適化](06_topology_optimization.md) | Ground Structure 法（トラスLP） | `optimize/topology.py` |
| [7. 可視化](07_visualization.md) | 変形図・断面力図・構造形態図 | `viz.py` |
| [8. コード構成](08_code_structure.md) | モジュール構成と API 一覧 | パッケージ全体 |

## 全体像

```
入力（モデル）        解析                      最適化
─────────────       ─────────────             ─────────────
節点・要素      ─→   静的線形解析        ─→    サイジング最適化（感度+MMA）
材料・断面           K u = F                   応力・たわみ制約下の質量最小化
境界条件・荷重  ─→   内力・応力の回収    ─→    トポロジー最適化（LP）
                     可視化                     Ground Structure 法
```

## 設計の指針

- **要素は 3D Timoshenko 梁**（各節点6自由度）を基本とし、2D 面内骨組も同じデータ構造で扱う。
- **疎行列**による組み立てと直接法ソルバで数千〜数万要素規模に対応。ソルバは差し替え可能。
- 最適化の**感度は解析的**（サイジングは直接法、トポロジーは LP の双対性）。
  数値は有限差分・解析解・別ソルバ（SLSQP/linprog）で検証済み。
- 出力（図・CSV）は `workspace/` フォルダに集約。

## 単位系

単位は強制しない。SI 一貫（N, m, Pa, kg, kg/m³）を推奨し、入力の一貫性は利用者が担保する。

## 座標系と自由度の規約（全章共通）

節点自由度の並びは

$$\mathbf{u}_{\text{node}} = [\,u_x,\; u_y,\; u_z,\; \theta_x,\; \theta_y,\; \theta_z\,]$$

断面の主軸を局所 $y, z$ 軸にとる。曲げは

- 局所 $z$ 軸まわり（面内 $x$-$y$ 曲げ）… $I_z$
- 局所 $y$ 軸まわり（面外 $x$-$z$ 曲げ）… $I_y$

で表す。`UX, UY, UZ, RX, RY, RZ = 0..5` は局所自由度インデックス。
