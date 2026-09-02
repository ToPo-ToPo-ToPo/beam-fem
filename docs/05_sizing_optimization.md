# 5. 断面サイジング最適化

実装：[`src/beamfem/optimize/sizing.py`](../src/beamfem/optimize/sizing.py),
[`mma.py`](../src/beamfem/optimize/mma.py),
[`sections.py`](../src/beamfem/optimize/sections.py),
[`driver.py`](../src/beamfem/optimize/driver.py)

## 5.1 問題の定式化

応力・たわみ制約のもとで総質量を最小化する：

$$
\begin{aligned}
\min_{\mathbf{x}}\quad & W(\mathbf{x}) = \sum_e \rho_e L_e A_e(\mathbf{x}) \\
\text{s.t.}\quad & g^\sigma_e(\mathbf{x}) = \frac{\sigma_{\max,e}(\mathbf{x})}{\sigma_{\text{allow},e}} - 1 \le 0 \\
& g^u_j(\mathbf{x}) = \frac{|u_j(\mathbf{x})|}{u_{\lim,j}} - 1 \le 0 \\
& x^{\min}_i \le x_i \le x^{\max}_i
\end{aligned}
$$

設計変数 $x_i$ は設計グループ $i$ の**断面スケール係数**。各要素は高々 1 つの設計変数に
支配される。

## 5.2 設計変数：スケール断面 `ScaledSection`

基準断面を係数 $s$ で**相似拡大**する。全寸法が $s$ 倍になるため、任意の断面形状
（矩形・I 形・箱型・パイプ等）に適用できる：

$$A = A_0 s^2,\quad I_y = I_{y0} s^4,\quad I_z = I_{z0} s^4,\quad J = J_0 s^4,\quad
  c_y = c_{y0}\, s,\quad c_z = c_{z0}\, s$$

せん断係数 $k_y, k_z$ は形状不変なので変えない。微分は

$$\frac{\mathrm{d}A}{\mathrm{d}s} = 2A_0 s,\quad
  \frac{\mathrm{d}I}{\mathrm{d}s} = 4I_0 s^3,\quad
  \frac{\mathrm{d}J}{\mathrm{d}s} = 4J_0 s^3,\quad
  \frac{\mathrm{d}c}{\mathrm{d}s} = c_0$$

## 5.3 解析的感度（直接法）

平衡式 $\mathbf{K}\mathbf{u}=\mathbf{F}$（荷重は設計変数に依存しない）を $x_i$ で微分：

$$\frac{\partial \mathbf{K}}{\partial x_i}\mathbf{u} + \mathbf{K}\frac{\partial \mathbf{u}}{\partial x_i} = \mathbf{0}
\;\;\Longrightarrow\;\;
\boxed{\;\mathbf{K}\frac{\partial \mathbf{u}}{\partial x_i} = -\frac{\partial \mathbf{K}}{\partial x_i}\mathbf{u}\;}$$

**$\mathbf{K}_{ff}$ の LU 分解を 1 回だけ作り**、各設計変数についてこの系を後退代入で解く
（分解を再利用するため安価）。設計変数が少なく制約が多い本問題に適した**直接法**。

要素剛性の微分は連鎖律：

$$\frac{\partial \mathbf{k}}{\partial x_i}
= \frac{\partial \mathbf{k}}{\partial A}\frac{\mathrm{d}A}{\mathrm{d}x_i}
+ \frac{\partial \mathbf{k}}{\partial I_y}\frac{\mathrm{d}I_y}{\mathrm{d}x_i}
+ \frac{\partial \mathbf{k}}{\partial I_z}\frac{\mathrm{d}I_z}{\mathrm{d}x_i}
+ \frac{\partial \mathbf{k}}{\partial J}\frac{\mathrm{d}J}{\mathrm{d}x_i}$$

$\partial\mathbf{k}/\partial(\cdot)$ は解析式（[1.6 節](01_fem_theory.md)）、$\mathrm{d}(\cdot)/\mathrm{d}x_i$ は
スケール断面の微分。全体への寄与は $\partial\mathbf{K}^e/\partial x_i = \mathbf{T}^\top(\partial\mathbf{k}/\partial x_i)\mathbf{T}$。

### 目的関数の勾配

$$\frac{\partial W}{\partial x_i} = \sum_{e\in\text{group }i} \rho_e L_e \frac{\mathrm{d}A_e}{\mathrm{d}x_i}$$

### たわみ制約の勾配

$$\frac{\partial g^u_j}{\partial x_i} = \frac{1}{u_{\lim,j}}\frac{\partial u_j}{\partial x_i}
\quad(\text{両側制約として } \pm u_j \text{ を扱う})$$

### 応力制約の勾配

要素の合成応力 $\sigma_{\max,e}$ は端力 $\mathbf{f}^e=\mathbf{k}(\mathbf{T}\mathbf{u}^e)$ と断面諸量の
両方に依存する。連鎖律で（陰な項＋陽な項）：

$$\frac{\mathrm{d}\sigma_e}{\mathrm{d}x_i}
= \underbrace{\frac{\partial \sigma_e}{\partial \mathbf{f}^e}\cdot\frac{\mathrm{d}\mathbf{f}^e}{\mathrm{d}x_i}}_{\text{力経由}}
+ \underbrace{\sum_{p}\frac{\partial \sigma_e}{\partial p}\frac{\mathrm{d}p}{\mathrm{d}x_i}}_{\text{断面変化（}e\text{ が }i\text{ 支配下のみ）}}$$

$$\frac{\mathrm{d}\mathbf{f}^e}{\mathrm{d}x_i}
= \frac{\partial \mathbf{k}}{\partial x_i}(\mathbf{T}\mathbf{u}^e)\Big|_{e\in i}
+ \mathbf{k}\,\mathbf{T}\frac{\partial \mathbf{u}^e}{\partial x_i}$$

$\partial\sigma/\partial\mathbf{f}$ は $\sigma_{\max}=|N|/A+|M_z|c_y/I_z+|M_y|c_z/I_y$ を
端力成分で微分したもの（絶対値は符号で処理）、$\partial\sigma/\partial p$ は断面諸量
$p\in\{A,I_y,I_z,c_y,c_z\}$ に関する陽な微分。

> **検証**：不静定構造で解析的勾配（目的・応力・たわみ制約すべて）を有限差分と照合し、
> 相対誤差 $\sim10^{-10}$（[`tests/test_optimize.py`](../tests/test_optimize.py)）。不静定なので
> $\partial\mathbf{u}/\partial x$ の陰な寄与も厳密に検証される。

## 5.4 最適化アルゴリズム：MMA

Method of Moving Asymptotes（Svanberg）。各反復で**移動漸近線** $L_j < x_j < U_j$ を用い、
分離可能・凸な部分問題を構成して解く。目的・制約を

$$\tilde f_i(\mathbf{x}) = \sum_j\!\left(\frac{p_{ij}}{U_j-x_j} + \frac{q_{ij}}{x_j-L_j}\right) + r_i$$

の形に近似する（$p_{ij}, q_{ij}$ は勾配の正負成分から定める）。部分問題は人工変数 $y_i, z$ を
加えた

$$\min\; f_0(\mathbf{x}) + a_0 z + \sum_i\Bigl(c_i y_i + \tfrac12 d_i y_i^2\Bigr)
\quad\text{s.t.}\quad \tilde f_i - a_i z - y_i \le 0,\;\; \mathbf{x}\in[\boldsymbol\alpha,\boldsymbol\beta]$$

を**主双対内点法**（`subsolv`）で解く。既定 $a_0=1,\ a_i=0,\ c_i$ 大,$\ d_i=0$ で
通常の制約付き最小化になる。漸近線は反復履歴（振動の有無）に応じて拡大・縮小する。

実装は Svanberg の `mmasub`/`subsolv` の忠実な移植。駆動ループ `minimize_mass` が
反復し、設計変数の変化量が `tol` を下回ったら収束とする。

> **検証**：
> - 静定片持ち・応力制約：解析的最適スケール $s^\* = (PL\,c_{y0}/(I_{z0}\sigma_{\text{allow}}))^{1/3}$ と一致（相対誤差 $\sim10^{-9}$）
> - 静定片持ち・たわみ制約：解析解と一致
> - 多変数・複合制約：**SLSQP と一致**（相対誤差 $\sim10^{-9}$）

## 5.5 使い方

```python
from beamfem.optimize import (
    SizingProblem, DesignVar, DispLimit, ScaledSection, minimize_mass
)

dvs = [
    DesignVar(ScaledSection(base), elements=[0, 1], x0=1.5, xmin=0.3, xmax=4.0),
    DesignVar(ScaledSection(base), elements=[2, 3], x0=1.5),
]
prob = SizingProblem(
    model, dvs,
    sigma_allow=160e6,                              # 要素応力の許容値（スカラ or {elem: 値}）
    disp_limits=[DispLimit(node=tip, dof=UY, limit=0.02)],
)
res = minimize_mass(prob, maxiter=100, move=0.2, tol=1e-6)
print(res.x, res.mass, res.sections, res.converged)
```

`OptResult`：最適設計変数 `x`、`mass`、`constraints`（$\le0$ で満足）、`sections`、
`iterations`、`converged`、`history`。

要素別の代表量は `prob.element_values(res.x, kind="area"|"scale"|"size")` で取得でき、
[構造形態の図示](07_visualization.md)に使う。

## 5.6 例

[`examples/sizing_optimization.py`](../examples/sizing_optimization.py)：先細り片持ち梁。
固定端側ほど曲げが大きいため、最適解は**固定端で太く先端で細い先細り**になり、質量を
約 40% 削減する。

## 5.7 応用例：円形膜のリブ補強

[`examples/ribbed_plate_optimization.py`](../examples/ribbed_plate_optimization.py)：上から
一様圧を受ける円形膜を、下面の放射＋同心リング状リブ（グリラージュ）で補強する設計。

- 膜は曲げ剛性を持たないものとし、圧力を分担面積で等価節点荷重へ変換（`builders.lump_pressure`）
- リブは 3D Timoshenko 梁の**面外曲げ＋ねじり**で荷重を支える（`builders.radial_grillage`）
- 外周固定・中心/中間のたわみ制約・各リブ応力制約のもと、リブ総質量を最小化
- 設計変数＝リブのスケール（放射バンド別・リング別）。下限を小さくすると不要リブが細る

結果は **放射リブが支配的（外周側ほど太い）・周方向リングは下限まで縮小** という、円形板の
補強として妥当な配置になる（`viz.plot_member_sizes` で形態を図示）。荷重の節点化と
グリラージュ生成は [`builders.py`](../src/beamfem/builders.py)（[8 章](08_code_structure.md)）。

### シェル板＋オフセットリブの連成サイジング

[`examples/ribbed_plate_shell_sizing.py`](../examples/ribbed_plate_shell_sizing.py)：
膜を曲げ剛性ゼロとみなす上記の近似に対し、**板そのものをフラットシェル**
（[10 章](10_shell_element.md)）でモデル化し、リブを**剛体オフセット付き梁**
（[1.5 節](01_fem_theory.md)）として連成させた本格的なリブ補強板のサイジング。

`SizingProblem` は次を扱う:

- **シェル要素**（3 節点 CST+DKT・4 節点 MITC4 の両方）は固定剛性として全体行列に
  加わる（板厚は設計変数ではない）。四角形シェル板の例は
  [`examples/ribbed_plate_quad_sizing.py`](../examples/ribbed_plate_quad_sizing.py)。
- **オフセット梁**は剛体腕 $\mathbf{G}$ を含めて剛性・感度・応力回収を行う
  （`B=T G` を変換として扱い、$EA e^2$ の合成効果を感度に反映）。解析的感度は
  シェル・オフセットを含めても有限差分と一致する（`tests/test_optimize.py`）。

リブの偏心 $e$ は最適化中は固定（取り付け深さを保ち、断面はその図心まわりで相似
拡大する近似）。合成効果のため面内自由度 $u_x,u_y$ は内部で自由にし、外周で面内を
保持する。結果は中央たわみ制約を活かしつつリブ総質量を大きく削減できる。

### どこのリブが太くなるか —— 境界条件と制約で変わる

「拘束部（外周）のリブが最も太くなる」とは限らない。最適配置は**境界条件**と
**効かせる制約**の組合せで決まる。例
[`examples/ribbed_plate_layout_study.py`](../examples/ribbed_plate_layout_study.py)
は同じ円板を 3 条件で比較する:

| 条件 | 最も太くなる位置 | 理由 |
|---|---|---|
| 単純支持 + たわみ制約 | **中間半径のリング** | 単純支持は縁のモーメントが 0・縁は既に拘束済み。中間リングが「追加の支持リング」として実効スパンを縮め、たわみ低減の効率が最大 |
| 完全固定 + たわみ制約 | 外周側がやや太い | 固定で板自体が曲げを負担するためリブは軽量で済む |
| 完全固定 + 応力制約 | **外周（固定端）** | 固定端で曲げモーメント最大。応力制約が縁の材料を要求（＝直感どおり） |

たわみ制約は**感度（du/dx）が大きい場所**＝中間半径に、応力制約（固定端）は
**モーメントが大きい場所**＝外周に材料を集める。直感的な「支持部が最太」は固定端
＋応力設計で成り立つ。

## 5.8 離散サイジング（規格サイズ）

実装：[`src/beamfem/optimize/discrete.py`](../src/beamfem/optimize/discrete.py)

実務ではリブ断面を任意寸法ではなく**規格材のカタログ**から選ぶ。各設計変数のスケール係数を
離散カタログ $\mathcal{C}_i = \{c_{i1}, c_{i2}, \dots\}$ からのみ選ぶ組合せ最適化：

$$\min_{x_i \in \mathcal{C}_i} \sum_e \rho_e L_e A_e(\mathbf{x})
  \quad\text{s.t.}\quad g_e^\sigma\le0,\; g_j^u\le0$$

連続版と同じ `SizingProblem` を関数評価器として用いる（感度不要の軽量評価
`evaluate_values` を使用）。2 つの解法：

| 解法 | 関数 | 特徴 |
|---|---|---|
| 総当たり | `solve_discrete_exhaustive` | 全組合せを評価し**大域最適**。小規模（既定 ≤ 20 万通り）・検証用 |
| 貪欲局所探索 | `solve_discrete_greedy` | 連続最適解を丸めて開始 → 実行可能化 → 近傍探索（単変数縮小＋交換移動）。実用規模 |

```python
from beamfem.optimize import solve_discrete_greedy, solve_discrete_exhaustive

catalog = [d/1000/0.03 for d in [20,25,30,40,50,60]]  # 規格リブ径→スケール
res = solve_discrete_greedy(prob, catalog)             # 共有カタログ
# res: x, indices, mass, constraints, feasible, n_eval, method
# 変数ごとに別カタログも可: solve_discrete_greedy(prob, [cat0, cat1, ...])
```

> **検証**（[`tests/test_discrete.py`](../tests/test_discrete.py)）：
> - `evaluate_values` が `evaluate` の目的・制約と一致
> - **貪欲解が総当たり（大域最適）と一致**し、関数評価回数は大幅に少ない
> - 離散解は実行可能で、連続最適（下界）を下回らない
>
> - **シェル板（三角形・四角形）＋オフセットリブの連成**でも貪欲解が大域最適に一致（5.7 の拡張）

例 [`examples/ribbed_plate_discrete.py`](../examples/ribbed_plate_discrete.py)：規格リブ径
から選ぶリブ補強。168 万通りを貪欲探索が約 270 評価で解き、連続比 +数% に収まる。
シェル板＋オフセットリブ版は
[`examples/ribbed_plate_shell_discrete.py`](../examples/ribbed_plate_shell_discrete.py)：
12 設計変数・7 サイズ（約 138 億通り）を貪欲法が約 1000 評価で解き、規格リブ高さ
（15〜60 mm）を選定して連続比 +9% 程度に収まる。四角形 MITC4 シェル板版は
[`examples/ribbed_plate_quad_discrete.py`](../examples/ribbed_plate_quad_discrete.py)
（2 群・規格高さから選定、総当たり大域最適と貪欲解が一致）。

## 5.9 リブ本数の最適化（2 段階）

リブの太さ（サイズ）だけでなく**本数**（放射スポーク数・リング数）も最適化したい場合、
本数は離散的な構成の選択なので **2 段階最適化**で扱う：

- 外側：候補となるリブ本数 $(n_{\text{radial}}, n_{\text{rings}})$ を列挙
- 内側：各構成でサイジング最適化（応力・たわみ制約下の質量最小化）
- 実行可能な構成の中で最小質量となる本数を選ぶ

**重要な前提**：本ライブラリはリブ（梁）のグリラージュのみを扱い、膜がリブ間で局所的に
たわむ挙動はモデル化しない。そのため構造制約（応力・たわみ）だけでは「本数が少ないほど
軽い」となり、本数の最適解が**最少本数に張り付く**。本数の下限は実際には**膜自身の能力＝
リブ間スパンの許容値**で決まる。そこで膜の許容パネルスパン $s_{\max}$ を制約として加える：

$$\frac{2\pi R}{n_{\text{radial}}} \le s_{\max}\quad(\text{外周の円周方向間隔}),\qquad
  \frac{R}{n_{\text{rings}}} \le s_{\max}\quad(\text{半径方向間隔})$$

これにより本数の下限が定まり、物理的に意味のある最適本数が得られる。例
[`examples/ribbed_plate_count_optimization.py`](../examples/ribbed_plate_count_optimization.py)
では、$s_{\max}=0.9$ m で **放射 14 本 × リング 4 環** が最適（質量 ≈ 298 kg）となり、
質量 vs 本数の推移も図示する。

> より厳密にはシェル/プレート要素で膜の局所スパンを直接モデル化すべきだが、本梁ベース
> ライブラリでは許容スパン制約で代替している。

## 5.10 限界と拡張余地

- スケール係数1変数／グループ（多寸法の独立最適化は将来拡張）
- 座屈・固有振動数制約は未実装
- 大規模設計変数では随伴法（adjoint）の方が有利（現状は直接法）
