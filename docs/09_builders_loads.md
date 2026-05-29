# 9. 構造生成と面荷重

実装：[`src/beamfem/builders.py`](../src/beamfem/builders.py)

梁要素のグリラージュ構造の生成と、面分布荷重（圧力）の等価節点荷重化をまとめる。
本ライブラリは節点荷重を入力とするため、面荷重はここで節点化する。

## 9.1 円形リブ・グリラージュ `radial_grillage`

水平面 ($x$-$y$) に、放射スポーク＋同心リングからなる円形リブ構造を生成する。
半径 $R$ を $n_{\text{rings}}$ 等分し、$n_{\text{radial}}$ 本のスポークを置く。
円形膜のリブ補強や蜘蛛の巣状フレームのモデル化に使う。

```python
g = radial_grillage(model, mat, sec, R, n_radial, n_rings, include_center=True)
```

返り値 `Grillage` の構成：

- `center`：中心節点（`include_center=False` なら −1）
- `ring_nodes[k][j]`：半径レベル $k$・角度 $j$ の節点番号
- `radial_bands[b]`：バンド $b$（半径レベル $b\to b{+}1$）の放射リブ**要素番号**のリスト
- `rings[k]`：リング $k$ の周方向リブ要素番号のリスト
- `triangles`：載荷面（円板）の三角形分割
- `interior_nodes()`：支持しない内部節点（中心＋外周以外）

`radial_bands` と `rings` は最適化の**設計グループ**にそのまま使える（バンド別・リング別に
断面を変える）。

## 9.2 面分布荷重の等価節点化 `lump_pressure`

面に作用する一様圧力 $q$ [N/m²] を、領域の三角形分割を用いて各頂点へ集約する。
三角形 $T$（面積 $A_T$）の荷重 $q A_T$ を 3 頂点へ 1/3 ずつ振り分ける（一定圧の
consistent な集約）：

$$f_{\text{node}} \mathrel{+}= \text{sign}\cdot q \sum_{T \ni \text{node}} \frac{A_T}{3}$$

```python
total = lump_pressure(model, triangles, pressure, dof=UZ, sign=-1.0)
```

既定は下向き ($-z$)。返り値は総載荷の大きさ $q\sum_T A_T$。面積は $x$-$y$ 平面で評価する。

> 三角形分割の総面積は格子に内接する多角形の面積に一致する。総載荷が
> $q\times(\text{内接多角形面積})$ と一致し、静解析の鉛直反力合計が総載荷に等しくなることを
> [`tests/test_builders.py`](../tests/test_builders.py) で確認している。

## 9.3 使用例

- [`examples/spider_web_3d.py`](../examples/spider_web_3d.py)：円形「蜘蛛の巣」フレームの面外載荷
- [`examples/ribbed_plate_optimization.py`](../examples/ribbed_plate_optimization.py)：円形膜のリブ補強最適化（連続）
- [`examples/ribbed_plate_discrete.py`](../examples/ribbed_plate_discrete.py)：同・規格サイズ（離散）
- [`examples/ribbed_plate_count_optimization.py`](../examples/ribbed_plate_count_optimization.py)：リブ本数の最適化

## 9.4 モデル化の前提（重要）

`radial_grillage` はリブ（梁）のグリラージュであり、**膜そのものの剛性は含まない**。
膜は圧力をリブ節点へ伝える媒体として扱い、その局所的なたわみ・スパン挙動は表現しない。
したがって：

- 圧力は `lump_pressure` でリブ節点へ集約される（膜の面内・面外剛性は無視）。
- リブ本数を最適化すると構造制約だけでは本数最少が最軽量になるため、膜の能力に相当する
  **リブ間スパンの上限**を別途課す必要がある（[5.9 節](05_sizing_optimization.md)）。

より厳密には膜をプレート/シェル要素でモデル化すべきだが、本梁ベースライブラリでは
上記の前提と補助的な制約で実用的に扱っている。
