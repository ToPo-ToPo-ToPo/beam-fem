# 6. トポロジー／部材配置最適化（Ground Structure 法）

実装：[`src/beamfem/optimize/topology.py`](../src/beamfem/optimize/topology.py)

## 6.1 考え方

候補部材を密に張った「**地盤構造（ground structure）**」から出発し、不要な部材の断面積を
ゼロに追い込むことで最適な部材配置を得る。トラス（軸力部材）では、塑性設計の**下界定理**に
基づき**最小体積問題が線形計画（LP）**に帰着する（Dorn et al. 1964）。LP は凸なので
**大域最適**が保証され、剛性は不要（平衡条件と応力制約のみ）。

## 6.2 定式化（LP）

$$
\begin{aligned}
\min_{A_e,\, n_e^{(k)}}\quad & V = \sum_e L_e A_e \\
\text{s.t.}\quad & \mathbf{B}\,\mathbf{n}^{(k)} = \mathbf{f}^{(k)} && \text{各荷重ケース } k\ (\text{節点平衡}) \\
& -\sigma_c A_e \le n_e^{(k)} \le \sigma_t A_e && \forall e, k\ (\text{応力}) \\
& A_e \ge A_{\min} \ge 0
\end{aligned}
$$

- $n_e^{(k)}$：荷重ケース $k$ の部材 $e$ の軸力（引張 +）
- $A_e$：断面積（**全ケースで共有**、最悪ケースで決まる）
- $\sigma_t, \sigma_c$：許容引張・圧縮応力（既定では等しい）

変数順は $[\,A_0,\dots,A_{M-1},\; n^{(0)},\dots,n^{(K-1)}\,]$。`scipy.optimize.linprog`
（HiGHS）で解く。疎行列で制約を構成する。

## 6.3 平衡行列 $\mathbf{B}$

部材 $e=(i,j)$ の単位ベクトル $\mathbf{d}=(\mathbf{p}_j-\mathbf{p}_i)/L$ に対し、$\mathbf{B}$ の列 $e$ は
節点 $i$ の自由度に $-\mathbf{d}$、節点 $j$ の自由度に $+\mathbf{d}$ を持つ。軸力 $n$（引張 +）に
対して

$$\mathbf{B}\,\mathbf{n} = \mathbf{f}\quad(\text{自由自由度での節点平衡})$$

が成り立つ。拘束（支持）自由度には平衡式を課さない（反力が吸収）。2D・3D いずれも
方向余弦で同様に扱える。

## 6.4 地盤構造の生成

- `grid_nodes(nx, ny, lx, ly[, nz, lz])`：矩形（直方体）格子の節点
- `generate_members(nodes, max_length=None)`：全節点対から候補部材を生成し、**共線で重複する
  長い部材を除外**する（線分上に別の節点が乗る部材は短い区間の組合せで表せるため冗長）。

共線判定：部材 $(i,j)$ について、別の節点 $k$ の投影パラメータ $t=\dfrac{(\mathbf{p}_k-\mathbf{p}_i)\cdot\mathbf{d}}{L}$ が
$0<t<1$ かつ垂直距離 $\lVert(\mathbf{p}_k-\mathbf{p}_i)-tL\mathbf{d}\rVert$ が微小なら、その部材を除外。

## 6.5 結果

`solve_min_volume(gs, sigma_t, sigma_c=None, area_min=0)` は `TopologyResult` を返す：

- `areas`：各部材の断面積 $(M,)$
- `forces`：各荷重ケースの部材軸力 $(K, M)$
- `volume`：最適体積
- `active(rel_tol)`：有効部材（$A_e > \text{rel\_tol}\cdot A_{\max}$）のインデックス

## 6.6 使い方

```python
from beamfem.optimize import (
    GroundStructure, generate_members, grid_nodes, solve_min_volume
)
from beamfem import viz

nodes = grid_nodes(nx=6, ny=5, lx=5.0, ly=4.0)
members = generate_members(nodes)                  # 候補部材（共線重複は除去）
gs = GroundStructure(
    nodes, members,
    supports={iy * 6: [0, 1] for iy in range(5)},  # 左端列を固定（0=x,1=y）
    load_cases=[{(2 * 6 + 5, 1): -50e3}],          # 右端中央に下向き荷重
)
res = solve_min_volume(gs, sigma_t=200e6)          # 最小体積トラス（大域最適）
viz.plot_truss(nodes, members, res.areas, show_all=True)
```

## 6.7 検証

[`tests/test_topology.py`](../tests/test_topology.py)：

| 検証 | 期待 |
|---|---|
| 単一バー | 体積 $PL/\sigma$、軸力 $P$（引張） |
| 対称 2 バー（静定） | 体積 $2P/\sigma$、各軸力 $P/\sqrt2$ |
| 冗長な候補集合 | 荷重と一直線の最短バーのみ採用（大域最適・部材除去） |
| 解の平衡 | $\mathbf{B}\,\mathbf{n}=\mathbf{f}$ を満たす |
| 複数荷重ケース | 断面が最悪ケースで決まる |
| 引張/圧縮別許容応力 | 体積が許容応力に反比例 |

## 6.8 例

[`examples/topology_ground_structure.py`](../examples/topology_ground_structure.py)：
6×5 格子（289 候補部材）の片持ちトラス。左端固定・右端中央荷重で解くと、289 候補から
約 10 部材に削減され、左端支持から荷重点へ上下弦材が収束する **Michell 型の片持ちトラス**
形態が得られる。

## 6.9 限界と拡張余地

- **トラス（軸力）のみ**。曲げを含むフレームのトポロジーは非凸で本 LP では扱えない。
- 座屈は考慮しない（圧縮材は $\sigma_c$ で扱うが座屈長は無視）。
- 得られた配置は**形態**であり、剛性・変形は別途 FEM で確認する（断面を割り当てて解析）。
