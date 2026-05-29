# beam-fem

梁モデルの FEM 解析と構造最適化を行うコード。

3D Timoshenko 梁要素（2節点・各節点6自由度）を中核とし、2D 面内骨組も
同じデータ構造で扱える。疎行列ソルバにより数千〜数万要素規模に対応する。

## 特徴

- **3D Timoshenko 梁要素**（せん断変形を考慮、Euler-Bernoulli を極限に含む）
- 軸・ねじり・2方向曲げ・せん断を統合した 12×12 要素剛性
- 疎行列（CSR）による全体剛性の組み立てと直接法ソルバ（後から PARDISO 等へ差替可）
- 解析解（片持ち梁の Timoshenko 厳密たわみ）との一致を pytest で検証済み
- 2D 面内骨組は `Model.fix_to_plane_xy()` で面外自由度を拘束して解く

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

## 座標系・規約

- 節点自由度の並び: `[ux, uy, uz, theta_x, theta_y, theta_z]`
- 断面の主軸は局所 y, z 軸。`Iz` は局所 z 軸まわり（面内 x-y 曲げ）、`Iy` は局所 y 軸まわり（面外 x-z 曲げ）
- 要素の局所 y 軸の向きは `add_element(..., vref=...)` で指定（既定は全体 Y 軸）。
  全体 X 方向の梁では局所軸＝全体軸となる。

## 構成

```
src/beamfem/
  material.py   材料・断面（Material, Section）
  model.py      モデル定義（Node, Element, Model, 境界条件・荷重）
  element3d.py  3D Timoshenko 要素剛性と座標変換
  assembly.py   疎行列での全体剛性・荷重組み立て
  solver.py     静的線形解析ソルバ
tests/          解析解との比較検証
examples/       使用例
```

## ロードマップ

- [x] 3D Timoshenko 静解析（検証済み）
- [ ] 要素内力・応力の回収（断面力図用）
- [ ] 固有値（モーダル）解析
- [ ] 断面サイジング最適化（解析的感度 + MMA）
- [ ] トポロジー / 部材配置最適化（Ground Structure 法）
```
