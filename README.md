# beam-fem

梁モデルの FEM 解析と構造最適化を行うコード。

3D Timoshenko 梁要素（2節点・各節点6自由度）を中核とし、2D 面内骨組も
同じデータ構造で扱える。**三角形フラットシェル要素（CST 膜 + DKT 板曲げ）**も
同じ節点6自由度で混在させられる。疎行列ソルバにより数千〜数万要素規模に対応する。

📖 **理論・数式・実装の詳細は [`docs/`](docs/README.md) にまとめている。**

## 特徴

- **3D Timoshenko 梁要素**（せん断変形を考慮、Euler-Bernoulli を極限に含む）
- 軸・ねじり・2方向曲げ・せん断を統合した 12×12 要素剛性
- **三角形フラットシェル要素**（3節点・各節点6自由度）：膜＝定ひずみ三角形 CST、板曲げ＝離散 Kirchhoff 三角形 DKT。梁と混在可。単純支持板の Navier 解との一致を pytest で検証済み（誤差 <1%）
- 疎行列（CSR）による全体剛性の組み立てと直接法ソルバ（後から PARDISO 等へ差替可）
- 解析解（片持ち梁の Timoshenko 厳密たわみ）との一致を pytest で検証済み
- 2D 面内骨組は `Model.fix_to_plane_xy()` で面外自由度を拘束して解く
- **変形図の描画**（matplotlib）：要素ごとに形状関数で曲げを滑らかに補間、2D/3D 自動判定、支持・荷重も表示
- **要素内力・応力の回収**：軸力 N・せん断 Vy/Vz・ねじり T・曲げ My/Mz、および軸/曲げ/合成応力。表・CSV 出力は**表示項目を指定可能**（常に全項目を出さない）。断面力図も描画
- **多様な断面形状**：矩形・円・I 形（H 形鋼）・箱型（角形鋼管）・パイプ（中空円）、および完全自由な断面（A,I,J 直接指定）
- **断面サイジング最適化**：応力・たわみ制約下の質量最小化。**解析的感度（直接法）＋ MMA**。解析解・SLSQP と一致を検証済み。最適化結果の構造形態（部材サイズ分布）も図示できる
- **離散サイジング最適化**：規格サイズのカタログから選ぶ組合せ最適化（総当たり＝大域最適／貪欲局所探索＝実用規模）
- **トポロジー／部材配置最適化**：**Ground Structure 法（トラスLP）**。応力制約下の最小体積を線形計画で**大域最適**に解く。複数荷重ケース・引張/圧縮別許容応力・2D/3D 対応。最適配置を図示
- **出力は `workspace/` フォルダへ**：図・CSV は相対パス指定で `workspace/` に自動保存（フォルダも自動生成、`set_workspace` で変更可）

## セットアップ

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[viz,dev]"
.venv/bin/python -m pytest        # 検証テスト
```

## 使い方（最小例：片持ち梁）

```python
from beamfem import Material, Section, Model, solve_static, UY

steel = Material(E=200e9, nu=0.3)
sec = Section.rectangle(b=0.05, h=0.10)

m = Model()
n0 = m.add_node(0, 0, 0)
n1 = m.add_node(2, 0, 0)
m.add_element(n0, n1, steel, sec)
m.fix(n0)                  # 完全固定
m.add_load(n1, UY, -1000)  # 先端に -1000 N

res = solve_static(m)
print(res.node_disp(n1))   # [ux, uy, uz, rx, ry, rz]
```

2D 門型ラーメンの例は [`examples/portal_frame_2d.py`](examples/portal_frame_2d.py)。

### シェル要素（三角形フラットシェル）

3節点の平面三角形要素で、面内（膜）と面外（板曲げ）を平面内で重ね合わせる。
膜＝CST（定ひずみ三角形）、板曲げ＝DKT（離散 Kirchhoff 三角形）。各節点6自由度
なので梁と同じモデルに混在できる。

```python
from beamfem import Material, Model, solve_static, recover_shell_forces, UX, UY, UZ, RZ

steel = Material(E=200e9, nu=0.3)
m = Model()
n0 = m.add_node(0, 0, 0)
n1 = m.add_node(1, 0, 0)
n2 = m.add_node(1, 1, 0)
n3 = m.add_node(0, 1, 0)
m.add_shell(n0, n1, n2, steel, thickness=0.01)   # 三角形1
m.add_shell(n0, n2, n3, steel, thickness=0.01)   # 三角形2

res = solve_static(m)
sf = recover_shell_forces(m, res)         # 膜応力・曲げモーメント（要素ローカル系）
sf.print_table(items=["sx", "sy", "sxy"])
sf[0].get("Mx")                            # 単位幅あたり曲げモーメント
```

応力・断面力は**要素ローカル座標系**で返る（ローカル x は節点1→2 の辺方向、
法線がローカル z）。成分キー: 膜応力 `sx, sy, sxy` / 曲げモーメント `Mx, My, Mxy`
/ 曲げ縁端応力 `sbx, sby`（=6M/t²）。

> **注意**：DKT は薄板（Kirchhoff）理論でせん断変形を無視するため、薄肉シェル
> 向き。面法線まわり回転（ドリリング, θz）には微小な架空剛性のみを与えている
> ので、**シェルのみの平面モデルでは θz を拘束する**（梁と連成する場合は不要）。
> 単純支持正方形板の例は [`examples/plate_shell.py`](examples/plate_shell.py)。

### 剛体オフセット（偏心したリブ・スティフナ）

梁を節点位置からずらして配置するには `add_element(..., offset=...)` を使う。板に
付くリブを中立面より下げると、板の曲げ回転がリブの軸伸縮を生む**軸-曲げ連成
（T 形断面の合成剛性 EA·e²）**が立ち上がり、補強効果が大きく増す。

```python
e = t/2 + h_rib/2                     # 板下面にリブを付ける偏心
m.add_element(n0, n1, steel, rib,
              offset=[0, 0, -e])      # リブ図心を板中立面から e 下げる
```

合成効果はリブの軸力を板の膜が分担して初めて働くため、**面内自由度 UX,UY は
内部で自由**にし、外周など最小限で面内を保持する。偏心なし（同心）だとリブ軸力が
立たず効果は出ない。リブ補強円板で「めり込み（e=0）vs 正規オフセット」を比較する
例は [`examples/ribbed_plate_shell.py`](examples/ribbed_plate_shell.py)。

### 内力・応力の出力（項目を指定）

```python
from beamfem import recover_forces

forces = recover_forces(m, res)

# 表示したい成分だけを選んで出力（常に全項目は出さない）
forces.print_table(items=["Mz", "Vy"], at="max")          # 要素内の絶対値最大
forces.print_table(items=["N", "Mz"], at="ends",          # 両端値・特定要素のみ
                   element_ids=[0, 5])
forces.to_csv("out.csv", items=["N", "Vy", "Mz", "sigma_max"])

# プログラムから値を取得
mz_max = forces[3].max_abs("Mz")     # 要素3の最大曲げ
```

成分キー: 内力 `N, Vy, Vz, T, My, Mz` / 応力 `sigma_a, sigma_b, sigma_max`。
断面力図は `viz.plot_diagram(forces, "Mz")` で描画。内力・応力の例は
[`examples/beam_forces.py`](examples/beam_forces.py)。

### 例題一覧

- [`examples/portal_frame_2d.py`](examples/portal_frame_2d.py) — 2D 門型ラーメン（水平荷重・変形図）
- [`examples/plate_shell.py`](examples/plate_shell.py) — 単純支持正方形板のシェル解析（CST+DKT・Navier 解と比較）
- [`examples/circular_plate_shell.py`](examples/circular_plate_shell.py) — 円形膜（円板）に等分布荷重（周辺固定/単純支持・Kirchhoff 円板解と比較）
- [`examples/ribbed_plate_shell.py`](examples/ribbed_plate_shell.py) — 円形膜のリブ補強（シェル板＋梁リブの連成・剛体オフセットで T 形合成効果を比較）
- [`examples/ribbed_plate_shell_sizing.py`](examples/ribbed_plate_shell_sizing.py) — リブ補強板のサイジング最適化（シェル板＋オフセットリブ・たわみ制約下の質量最小化）
- [`examples/beam_forces.py`](examples/beam_forces.py) — 単純梁の内力・応力と項目指定出力
- [`examples/spider_web_3d.py`](examples/spider_web_3d.py) — 円形「蜘蛛の巣」フレームに面分布荷重（面外グリラージュ／3D 曲げ・ねじり）
- [`examples/sizing_optimization.py`](examples/sizing_optimization.py) — 先細り片持ち梁の質量最小化（サイジング最適化）
- [`examples/topology_ground_structure.py`](examples/topology_ground_structure.py) — 片持ちトラスの部材配置最適化（Ground Structure 法）
- [`examples/ribbed_plate_optimization.py`](examples/ribbed_plate_optimization.py) — 円形膜を下から補強するリブ構造の最適化（グリラージュ＋サイジング）
- [`examples/ribbed_plate_discrete.py`](examples/ribbed_plate_discrete.py) — 上記の離散版（規格リブ径から選定）
- [`examples/ribbed_plate_count_optimization.py`](examples/ribbed_plate_count_optimization.py) — リブ本数（放射数・リング数）の最適化（2段階＋膜スパン制約）

### 断面サイジング最適化

```python
from beamfem.optimize import SizingProblem, DesignVar, DispLimit, ScaledSection, minimize_mass

# 設計変数 = 断面スケール係数（要素グループごと）。任意の基準断面を相似拡大
dvs = [DesignVar(ScaledSection(base), elements=[0, 1], x0=1.5, xmin=0.3, xmax=4.0)]
prob = SizingProblem(
    model, dvs,
    sigma_allow=160e6,                          # 要素応力の許容値
    disp_limits=[DispLimit(node=tip, dof=UY, limit=0.02)],  # たわみ制約
)
res = minimize_mass(prob, maxiter=100, move=0.2)   # 解析的感度 + MMA
print(res.x, res.mass, res.sections)

# 最適化結果の構造形態を図示（部材の線幅・色＝断面サイズ）
viz.plot_member_sizes(model, prob.element_values(res.x, kind="area"),
                      label="cross-section area")
```

### トポロジー／部材配置最適化（Ground Structure 法）

```python
from beamfem.optimize import GroundStructure, generate_members, grid_nodes, solve_min_volume
from beamfem import viz

nodes = grid_nodes(nx=6, ny=5, lx=5.0, ly=4.0)   # 格子節点
members = generate_members(nodes)                 # 候補部材（共線重複は除去）
gs = GroundStructure(
    nodes, members,
    supports={iy * 6: [0, 1] for iy in range(5)},  # 左端列を固定
    load_cases=[{(2 * 6 + 5, 1): -50e3}],          # 右端中央に下向き荷重
)
res = solve_min_volume(gs, sigma_t=200e6)          # 最小体積トラス（LP・大域最適）
viz.plot_truss(nodes, members, res.areas, show_all=True)   # 最適配置を図示
```

## 断面形状

```python
Section.rectangle(b=0.1, h=0.2)              # 矩形
Section.circle(d=0.05)                        # 中実円
Section.pipe(d=0.1, t=0.005)                  # 中空円（パイプ）
Section.box(b=0.2, h=0.3, t=0.01)             # 箱型（角形鋼管）
Section.i_section(h=0.3, bf=0.15, tf=0.012, tw=0.008)  # I 形（H 形鋼）
Section(A=..., Iy=..., Iz=..., J=..., cy=..., cz=...)   # 完全自由
```

`Iz` が局所 z 軸まわり（面内 x-y 曲げ）、`Iy` が局所 y 軸まわり（面外 x-z 曲げ）。
I 形は `Iz` が強軸。せん断係数 `ky, kz` は各断面で妥当な近似値を自動設定し、
キーワードで上書きもできる。応力計算には縁端距離 `cy, cz` を使う。

## 出力先（workspace）

図・CSV は相対パスを渡すと `workspace/` フォルダに保存される（自動生成）。

```python
from beamfem import set_workspace
set_workspace("results/case1")   # 出力先を変更（既定は ./workspace）
viz.savefig("deformed.png")       # -> results/case1/deformed.png
forces.to_csv("forces.csv")       # 絶対パスを渡せばそのまま使う
```

## 座標系・規約

- 節点自由度の並び: `[ux, uy, uz, theta_x, theta_y, theta_z]`
- 断面の主軸は局所 y, z 軸。`Iz` は局所 z 軸まわり（面内 x-y 曲げ）、`Iy` は局所 y 軸まわり（面外 x-z 曲げ）
- 要素の局所 y 軸の向きは `add_element(..., vref=...)` で指定（既定は全体 Y 軸）。
  全体 X 方向の梁では局所軸＝全体軸となる。

## 構成

```
src/beamfem/
  material.py   材料・断面（Material, Section）
  model.py      モデル定義（Node, Element, ShellElement, Model, 境界条件・荷重）
  element3d.py  3D Timoshenko 要素剛性と座標変換
  shell3d.py    三角形フラットシェル要素剛性（CST 膜 + DKT 板曲げ）と座標変換
  shell.py      シェルの応力・断面力の回収（ShellForceResults）
  assembly.py   疎行列での全体剛性・荷重組み立て（梁・シェル混在）
  solver.py     静的線形解析ソルバ
tests/          解析解との比較検証
examples/       使用例
```

## ロードマップ

- [x] 3D Timoshenko 静解析（検証済み）
- [x] 変形図の描画（2D/3D）
- [x] 要素内力・応力の回収（断面力図・項目指定出力）
- [x] 断面サイジング最適化（解析的感度 + MMA、応力・たわみ制約下の質量最小化）
- [x] トポロジー／部材配置最適化（Ground Structure 法・トラスLP）
- [x] 三角形フラットシェル要素（CST 膜 + DKT 板曲げ、単純支持板で検証）
- [ ] 固有値（モーダル）解析
- [ ] 断面サイジング最適化（解析的感度 + MMA）
- [ ] トポロジー / 部材配置最適化（Ground Structure 法）
```
