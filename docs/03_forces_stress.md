# 3. 内力・応力の回収

実装：[`src/beamfem/forces.py`](../src/beamfem/forces.py)

## 3.1 局所端力の回収

変位解 $\mathbf{u}$ から、各要素の局所座標における端力（12 成分）を回収する：

$$\mathbf{f}^e_{\text{local}} = \mathbf{k}\,(\mathbf{T}\,\mathbf{u}^e)$$

$\mathbf{k}$ は局所剛性、$\mathbf{T}$ は変換行列、$\mathbf{u}^e$ は当該要素の全体変位（12）。
成分の並びは自由度と同じ：

$$\mathbf{f}^e_{\text{local}} = [\,F_{x1}, F_{y1}, F_{z1}, M_{x1}, M_{y1}, M_{z1},\;
                                  F_{x2}, F_{y2}, F_{z2}, M_{x2}, M_{y2}, M_{z2}\,]$$

## 3.2 内力成分と符号規約

節点荷重のみ（部材内分布荷重なし）の前提では、軸力・せん断・ねじりは部材内で**一定**、
曲げモーメントは**線形**。内力は端値から線形補間する。引張・右ねじを正とする。

| 成分 | 意味 | 節点1の値 | 節点2の値 |
|---|---|---|---|
| `N`  | 軸力（引張 +） | $-F_{x1}$ | $F_{x2}$ |
| `Vy` | 局所 $y$ せん断 | $F_{y1}$ | $-F_{y2}$ |
| `Vz` | 局所 $z$ せん断 | $F_{z1}$ | $-F_{z2}$ |
| `T`  | ねじり | $-M_{x1}$ | $M_{x2}$ |
| `My` | $y$ 軸まわり曲げ（$x$-$z$ 面） | $-M_{y1}$ | $M_{y2}$ |
| `Mz` | $z$ 軸まわり曲げ（$x$-$y$ 面） | $-M_{z1}$ | $M_{z2}$ |

位置 $\xi\in[0,1]$ での値は端値の線形補間 $v(\xi) = (1-\xi)v_0 + \xi v_1$。

検証（[`tests/test_forces.py`](../tests/test_forces.py)）：

- 片持ち梁先端横荷重：せん断一定 $P$、固定端モーメント $PL$
- 単純梁中央集中荷重：中央モーメント $PL/4$、せん断 $P/2$
- 面外曲げ $M_y$、軸力 $N=P$ いずれも解析解と一致

## 3.3 応力評価

断面の縁端距離 $c_y, c_z$（中立軸から縁端まで）を用いて、合成縁端応力を評価する。

$$\sigma_a = \frac{N}{A}\quad(\text{軸応力})$$

$$\sigma_b = \frac{|M_z|\,c_y}{I_z} + \frac{|M_y|\,c_z}{I_y}\quad(\text{曲げ縁端応力})$$

$$\sigma_{\max} = \left|\frac{N}{A}\right| + \sigma_b
\quad(\text{合成縁端応力・保守的評価})$$

両端それぞれで評価し、要素の代表値には大きい方を用いる。$c_y, c_z$ は断面が
持つ場合のみ曲げ応力を計算する（汎用断面で未指定なら軸応力のみ）。

## 3.4 出力（表示項目の指定）

`recover_forces(model, result)` は `ForceResults` を返す。出力は**表示したい項目を指定**でき、
常に全項目を出力しない設計：

```python
forces = recover_forces(m, res)
forces.print_table(items=["Mz", "Vy"], at="max")          # 要素内の絶対値最大
forces.print_table(items=["N", "Mz"], at="ends",          # 両端値・特定要素のみ
                   element_ids=[0, 5])
forces.to_csv("out.csv", items=["N", "Vy", "Mz", "sigma_max"])  # workspace/ へ
mz = forces[3].max_abs("Mz")                               # プログラムから取得
```

- `items`：成分キー（内力 `N,Vy,Vz,T,My,Mz` / 応力 `sigma_a,sigma_b,sigma_max`）
- `at`：`"max"`（要素内の絶対値最大）/ `"ends"`（両端値を別行）
- `element_ids`：表示する要素番号（省略時は全要素）

## 3.5 断面力図

`viz.plot_diagram(forces, "Mz")` で指定成分の断面力図を描く。各部材の材軸に直交方向へ
値をオフセットして描画（`Mz/Vy` は局所 $y$、`My/Vz` は局所 $z$）。詳細は [7 章](07_visualization.md)。
