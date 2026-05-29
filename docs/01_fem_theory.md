# 1. FEM 理論：3D Timoshenko 梁要素

実装：[`src/beamfem/element3d.py`](../src/beamfem/element3d.py)

## 1.1 要素の概要

2 節点・各節点 6 自由度（並進 3・回転 3）の 3 次元 Timoshenko 梁要素。
1 要素あたり 12 自由度。自由度の並び（局所・全体で共通）：

$$\mathbf{u}^e = [\,u_{x1}, u_{y1}, u_{z1}, \theta_{x1}, \theta_{y1}, \theta_{z1},\;
                    u_{x2}, u_{y2}, u_{z2}, \theta_{x2}, \theta_{y2}, \theta_{z2}\,]^\top$$

局所座標系：

- $x$：節点1 → 節点2 の材軸方向
- $y, z$：断面の主軸（$I_z$ は $z$ 軸まわり＝面内、$I_y$ は $y$ 軸まわり＝面外）

Euler–Bernoulli 梁との違いは**せん断変形**を考慮する点で、後述のせん断パラメータ
$\Phi \to 0$ の極限で Euler–Bernoulli 解に一致する。

## 1.2 せん断パラメータ

Timoshenko 効果は無次元のせん断パラメータ $\Phi$ で取り込む：

$$\Phi_y = \frac{12 E I_z}{k_y\, G\, A\, L^2}\quad(\text{$x$-$y$ 面内曲げ}),\qquad
  \Phi_z = \frac{12 E I_y}{k_z\, G\, A\, L^2}\quad(\text{$x$-$z$ 面外曲げ})$$

ここで

- $E$：ヤング率、$G = \dfrac{E}{2(1+\nu)}$：せん断弾性係数
- $A$：断面積、$L$：要素長
- $I_y, I_z$：断面二次モーメント
- $k_y, k_z$：せん断補正係数（せん断有効断面積 $A_s = kA$）

$\Phi$ が大きいほど（太短い梁ほど）せん断変形の寄与が大きい。

## 1.3 局所剛性行列

局所座標系での $12\times12$ 剛性行列 $\mathbf{k}$ は、独立な 4 つの寄与（軸・ねじり・
2 方向曲げ）の重ね合わせ。

### 軸方向（$u_x$）

$$k_{u_x} = \frac{EA}{L}\begin{bmatrix} 1 & -1 \\ -1 & 1 \end{bmatrix}
\quad\text{（自由度 } u_{x1}, u_{x2}\text{）}$$

### ねじり（$\theta_x$）

$$k_{\theta_x} = \frac{GJ}{L}\begin{bmatrix} 1 & -1 \\ -1 & 1 \end{bmatrix}
\quad\text{（自由度 } \theta_{x1}, \theta_{x2}\text{）}$$

$J$ はサン・ブナンのねじり定数（[4 章](04_sections.md)）。そり拘束は無視。

### $x$-$y$ 面内曲げ（自由度 $u_y, \theta_z$、断面 $I_z$）

$$e_z \equiv \frac{E I_z}{(1+\Phi_y)L^3}$$

$$
\mathbf{k}_{xy} = e_z
\begin{bmatrix}
12 & 6L & -12 & 6L \\
6L & (4+\Phi_y)L^2 & -6L & (2-\Phi_y)L^2 \\
-12 & -6L & 12 & -6L \\
6L & (2-\Phi_y)L^2 & -6L & (4+\Phi_y)L^2
\end{bmatrix}
\quad\text{（} u_{y1}, \theta_{z1}, u_{y2}, \theta_{z2}\text{）}
$$

### $x$-$z$ 面外曲げ（自由度 $u_z, \theta_y$、断面 $I_y$）

$$e_y \equiv \frac{E I_y}{(1+\Phi_z)L^3}$$

並進と回転の連成符号が面内曲げと反転する（$\mathrm{d}w/\mathrm{d}x = -\theta_y$ のため）：

$$
\mathbf{k}_{xz} = e_y
\begin{bmatrix}
12 & -6L & -12 & -6L \\
-6L & (4+\Phi_z)L^2 & 6L & (2-\Phi_z)L^2 \\
-12 & 6L & 12 & 6L \\
-6L & (2-\Phi_z)L^2 & 6L & (4+\Phi_z)L^2
\end{bmatrix}
\quad\text{（} u_{z1}, \theta_{y1}, u_{z2}, \theta_{y2}\text{）}
$$

> **実装上の注意（既知の不具合と対策）**：剛性行列は上三角のみ記入し、最後に
> $\mathbf{k} \leftarrow \mathbf{k} + \mathbf{k}^\top - \mathrm{diag}(\mathbf{k})$ で対称化する。
> 軸・ねじり項を上下両方に書いてから対称化すると非対角が 2 倍になり、剛体並進モードが
> 失われて平衡が崩れる。新しい寄与を足すときも上三角のみ記入する規約に従う。

検証として、組み上げた剛性に対し剛体並進ベクトルを掛けると（数値誤差を除き）ゼロになる
（$\mathbf{K}\,\mathbf{t}_{\text{rigid}} \approx \mathbf{0}$）。

## 1.4 局所→全体の座標変換

材軸方向の単位ベクトルを $\mathbf{e}_1 = (\mathbf{p}_2-\mathbf{p}_1)/L$ とする。
断面の向きは参照ベクトル $\mathbf{v}_{\text{ref}}$（**局所 $y$ 軸の希望方向**）で与える：

$$\mathbf{e}_2 = \frac{\mathbf{v}_{\text{ref}} - (\mathbf{v}_{\text{ref}}\cdot\mathbf{e}_1)\,\mathbf{e}_1}
                     {\lVert\,\cdot\,\rVert},\qquad
  \mathbf{e}_3 = \mathbf{e}_1 \times \mathbf{e}_2$$

既定では $\mathbf{v}_{\text{ref}}$ に全体 $Y$ 軸を用い、材軸が $Y$ と平行な場合のみ全体 $Z$ 軸へ
切替える。この規約により**全体 $X$ 方向の梁は局所軸＝全体軸**となり、2D（$x$-$y$ 面）への
埋め込みも自然になる（面内曲げが $I_z$ に対応）。

方向余弦行列（各行が局所軸を全体座標で表す）：

$$\mathbf{R} = \begin{bmatrix} \mathbf{e}_1^\top \\ \mathbf{e}_2^\top \\ \mathbf{e}_3^\top \end{bmatrix},
\qquad \mathbf{v}_{\text{local}} = \mathbf{R}\,\mathbf{v}_{\text{global}}$$

$12\times12$ の変換行列 $\mathbf{T}$ は $\mathbf{R}$ を 4 ブロック対角に並べたもの。全体座標の
要素剛性は

$$\boxed{\;\mathbf{K}^e = \mathbf{T}^\top \mathbf{k}\, \mathbf{T}\;}$$

## 1.5 剛体オフセット（偏心配置）

梁図心を節点位置からずらして配置したい場合（板に付くリブ・スティフナなど）、
**剛体腕（rigid offset）**で図心の自由度を節点（マスター）自由度に結ぶ。両端
共通のオフセット $\mathbf{r}$（節点→図心の全体ベクトル）に対し、微小変形では

$$\mathbf{u}_{\text{beam}} = \mathbf{u}_{\text{node}} + \boldsymbol{\theta}_{\text{node}}\times\mathbf{r},
\qquad \boldsymbol{\theta}_{\text{beam}} = \boldsymbol{\theta}_{\text{node}}$$

これは $6\times6$ ブロック

$$\mathbf{G}_{\text{node}} = \begin{bmatrix} \mathbf{I}_3 & -[\mathbf{r}]_\times \\ \mathbf{0} & \mathbf{I}_3 \end{bmatrix},
\qquad [\mathbf{r}]_\times \boldsymbol{\theta} = \mathbf{r}\times\boldsymbol{\theta}$$

を両端に並べた $12\times12$ 変換 $\mathbf{G}$ で表せる。図心で組んだ要素剛性を節点へ

$$\boxed{\;\mathbf{K}^e_{\text{node}} = \mathbf{G}^\top \big(\mathbf{T}^\top \mathbf{k}\,\mathbf{T}\big)\, \mathbf{G}\;}$$

と移す。これにより**節点回転がオフセット点の軸伸縮を生む軸-曲げ連成**が現れ、
T 形断面の合成剛性 $EA\,e^2$（$e=\lVert\mathbf{r}\rVert$）が取り込まれる。両端
共通オフセットでは長さ・向きは不変なので、図心剛性 $\mathbf{T}^\top\mathbf{k}\mathbf{T}$
自体は変わらず、連成はすべて $\mathbf{G}$ が担う。`add_element(..., offset=...)`
で指定し、内力回収も $\mathbf{G}$ を介して図心の値に整合させる。

> 合成効果は、オフセット部材の軸力を相手（板の膜や連続体）が拘束して初めて
> 働く。孤立した片持ち梁では軸力が立たず、たわみは変わらない点に注意（検証は
> [`tests/test_offset.py`](../tests/test_offset.py)：剛体リンク明示モデルとの一致）。

コード対応: 剛体腕 $\mathbf{G}$ は `element3d.rigid_offset_matrix`、要素剛性への適用は
`element3d.element_stiffness_global`（`offset` 引数）、内力回収は
`forces.recover_forces`（$\mathbf{G}$ を介して図心変位へ）。`Element.offset` に保持し
`Model.add_element(..., offset=...)` で与える。リブ補強板の例
`examples/ribbed_plate_shell.py`（めり込み $e=0$ vs 正規オフセット $e=t/2+h/2$ の比較）。

## 1.6 検証（解析解との一致）

実装は片持ち梁の Timoshenko 厳密たわみと**1 要素でも厳密一致**する：

$$\delta_{\text{tip}} = \frac{P L^3}{3 E I} + \frac{P L}{k\,G\,A}
\quad(\text{曲げ} + \text{せん断})$$

テスト [`tests/test_cantilever.py`](../tests/test_cantilever.py) で
2 方向曲げ・軸・要素分割不変性・反力の釣り合いを確認している。

## 1.7 剛性の解析的偏微分（最適化用）

サイジング最適化の感度計算のため、局所剛性の断面諸量に関する偏微分
$\partial\mathbf{k}/\partial A,\ \partial\mathbf{k}/\partial I_y,\ \partial\mathbf{k}/\partial I_z,\ \partial\mathbf{k}/\partial J$
を解析式で実装している（`local_stiffness_derivs`）。

曲げ項は $\Phi$ を介して $A, I$ の非線形関数になる。$\Phi = c\,I/A$（$c$ は定数）とおくと、
$D = 1+\Phi$、$e = EI/(DL^3)$ について

$$\frac{\partial e}{\partial I} = \frac{E}{L^3 D^2},\qquad
  \frac{\partial e}{\partial A} = \frac{E\,c\,I^2}{L^3 A^2 D^2},\qquad
  \frac{\partial \Phi}{\partial I} = \frac{c}{A},\qquad
  \frac{\partial \Phi}{\partial A} = -\frac{c\,I}{A^2}$$

を用い、$(4+\Phi)L^2 e$ などの積は積の微分で展開する。詳細は
[5 章](05_sizing_optimization.md)。中心差分との一致（相対誤差 $\sim10^{-10}$）を確認済み。
