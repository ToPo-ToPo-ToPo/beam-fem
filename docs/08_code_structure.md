# 8. コード構成と API

## 8.1 ディレクトリ構成

```
beam-fem/
├── src/beamfem/
│   ├── __init__.py        トップレベル API の再エクスポート
│   ├── material.py        Material, Section（断面コンストラクタ）
│   ├── model.py           Model, Element, ShellElement, 自由度定数, 境界条件・荷重
│   ├── element3d.py       3D Timoshenko 要素剛性・座標変換・剛体オフセット・剛性微分
│   ├── shell3d.py         三角形フラットシェル要素剛性（CST 膜 + DKT 板曲げ）・座標変換
│   ├── assembly.py        疎行列での全体剛性・荷重組み立て（梁・シェル混在）
│   ├── solver.py          静的線形解析（StaticResult）
│   ├── forces.py          梁の内力・応力の回収（ForceResults）
│   ├── shell.py           シェルの応力・断面力の回収（ShellForceResults）
│   ├── viz.py             可視化（matplotlib・任意依存）
│   ├── builders.py        グリラージュ生成・面分布荷重の等価節点化
│   ├── workspace.py       出力先 workspace フォルダの管理
│   └── optimize/
│       ├── sections.py    ScaledSection（スケール断面ファミリ）
│       ├── sizing.py      SizingProblem（解析的感度・直接法）
│       ├── mma.py         MMA（mmasub / subsolv）
│       ├── driver.py      minimize_mass（駆動ループ・OptResult）
│       ├── discrete.py    離散サイジング（総当たり・貪欲局所探索）
│       └── topology.py    Ground Structure 法（トラス LP）
├── tests/                 検証テスト（pytest, 43 件）
├── examples/              使用例
├── docs/                  本ドキュメント
└── pyproject.toml
```

## 8.2 依存関係

- 必須：`numpy`, `scipy`
- 任意：`matplotlib`（`viz`）、`pytest`（テスト）

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[viz,dev]"
.venv/bin/python -m pytest
```

## 8.3 主要 API

### モデル構築

```python
from beamfem import Material, Section, Model
from beamfem import UX, UY, UZ, RX, RY, RZ   # 局所自由度インデックス 0..5

mat = Material(E, nu=0.3, rho=0.0)
sec = Section.rectangle(b, h)   # circle / pipe / box / i_section / Section(...)

m = Model()
n0 = m.add_node(x, y, z)
e0 = m.add_element(n0, n1, mat, sec, vref=None, offset=None)  # offset で偏心配置（剛体オフセット）
s0 = m.add_shell(n0, n1, n2, mat, thickness)   # 三角形フラットシェル（3節点6自由度）
m.fix(node[, dofs]); m.pin(node); m.fix_to_plane_xy()
m.add_load(node, dof, value)
```

### シェル要素

```python
from beamfem import recover_shell_forces

s = m.add_shell(n0, n1, n2, mat, thickness)   # CST 膜 + DKT 板曲げ
res = solve_static(m)                          # 梁と同じソルバ（混在可）
sf = recover_shell_forces(m, res)              # ShellForceResults（要素ローカル系）
sf.print_table(items=["sx", "sy", "sxy"])      # 膜応力 / Mx,My,Mxy / sbx,sby
sf[e].get("Mx")                                 # 単位幅あたり曲げモーメント
```

DKT は薄板理論（せん断変形を無視）。ドリリング θz には微小架空剛性のみ与える
ため、シェルのみの平面モデルでは θz を拘束する（梁と連成時は不要）。

### 解析

```python
from beamfem import solve_static, recover_forces

res = solve_static(m)                 # StaticResult: u, reactions, K, node_disp(node)
forces = recover_forces(m, res)       # ForceResults
forces.print_table(items=[...], at="max"|"ends", element_ids=None)
forces.to_csv(path, items=[...], at=...)      # workspace/ へ
forces[e].max_abs("Mz"); forces[e].get_ends("sigma_max")
```

### 可視化

```python
from beamfem import viz
viz.plot_model(m); viz.plot_deformed(m, res, scale="auto")
viz.plot_diagram(forces, "Mz")
viz.plot_member_sizes(m, values, label=...)
viz.plot_truss(nodes, members, areas, show_all=True)
viz.savefig("out.png"); viz.show()
```

### サイジング最適化

```python
from beamfem.optimize import (
    SizingProblem, DesignVar, DispLimit, ScaledSection, minimize_mass
)
dvs = [DesignVar(ScaledSection(base), elements=[...], x0=1.5, xmin=0.3, xmax=4.0)]
prob = SizingProblem(model, dvs, sigma_allow=..., disp_limits=[DispLimit(node, dof, limit)])
res = minimize_mass(prob, maxiter=100, move=0.2, tol=1e-6)
# res: x, mass, constraints, sections, iterations, converged, history
prob.element_values(res.x, kind="area"|"scale"|"size")
```

### 離散サイジング（規格サイズ）

```python
from beamfem.optimize import solve_discrete_greedy, solve_discrete_exhaustive
catalog = [0.5, 1.0, 1.5, 2.0]                      # スケール係数のカタログ（共有）
res = solve_discrete_greedy(prob, catalog)          # 実用規模（貪欲局所探索）
res = solve_discrete_exhaustive(prob, catalog)      # 小規模で大域最適
# res: x, indices, mass, constraints, feasible, n_eval, method
```

### トポロジー最適化

```python
from beamfem.optimize import (
    GroundStructure, generate_members, grid_nodes, solve_min_volume
)
nodes = grid_nodes(nx, ny, lx, ly)
members = generate_members(nodes)
gs = GroundStructure(nodes, members, supports={node: [dofs]}, load_cases=[{(node,dof): val}])
res = solve_min_volume(gs, sigma_t=..., sigma_c=None, area_min=0.0)
# res: areas, forces, volume, active(rel_tol)
```

### 構造の生成と面荷重

```python
from beamfem import radial_grillage, lump_pressure
g = radial_grillage(model, mat, sec, R, n_radial, n_rings)  # 円形リブ・グリラージュ
# g: center, ring_nodes[k][j], radial_bands[b], rings[k], triangles, interior_nodes()
total = lump_pressure(model, g.triangles, pressure, dof=UZ, sign=-1.0)  # 圧力->節点荷重
```

### 出力先

```python
from beamfem import set_workspace, get_workspace
set_workspace("results/case1")   # 既定は ./workspace。相対パス保存はこの中へ
```

## 8.4 テストと検証の対応

| テスト | 検証内容 |
|---|---|
| `test_cantilever.py` | Timoshenko 解析解・要素分割不変性・反力釣り合い |
| `test_shell.py` | フラットシェル（剛体モード・膜パッチ・単純支持板の Navier 解収束・応力回収） |
| `test_offset.py` | 剛体オフセット梁（剛体腕の性質・剛体リンク明示モデルとの一致・軸-曲げ連成 EA·e²） |
| `test_sections.py` | 各断面諸量・片持ち解析解との一致 |
| `test_forces.py` | 内力・応力（せん断/モーメント/軸/曲げ応力）の解析解一致 |
| `test_optimize.py` | 感度 vs 有限差分、MMA vs 解析解／SLSQP |
| `test_topology.py` | LP の解析解一致・平衡・複数ケース |
| `test_viz.py` | 描画のスモークテスト（ヘッドレス） |
| `test_workspace.py` | 出力先の解決 |
| `test_builders.py` | グリラージュ生成・圧力の節点化・つり合い |
| `test_discrete.py` | 離散最適化（貪欲 vs 総当たり大域最適・実行可能性） |
| `test_rib_count.py` | リブ本数最適化（各構成の実行可能性・本数トレンド） |

## 8.5 設計上の不変条件（保守の指針）

- **剛性行列は上三角のみ記入して対称化**（[1.3 節](01_fem_theory.md)の注記）。新しい要素・項を
  足すときも同様に。軸・ねじり項を両側に書くと非対角が 2 倍になり平衡が崩れる。
- 線形ソルバは `solver._solve_sparse` に集約。性能要求時はここを差し替える。
- 最適化の感度を変更したら、必ず有限差分／解析解と再照合する。
- **シェルのドリリング自由度 θz** には実剛性が無く架空剛性のみ（`shell3d.DRILLING_FACTOR`）。
  剛体回転を保つため一様回転に対しゼロとなる形にしてあり、その代償としてシェル
  のみの平面モデルでは θz の大域スプリアスモードが残る。θz を拘束するか梁と連成
  させて使う。要素を追加・修正したら剛体モード（6 物理モードのエネルギー≈0）と
  解析解（単純支持板）への収束を再確認する。
