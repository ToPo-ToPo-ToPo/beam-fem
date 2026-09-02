# 11. 四角形フラットシェル要素 MITC4（Mindlin-Reissner ＋ 仮定横せん断）

3 節点 DKT（[10 章](10_shell_element.md)）は薄板（Kirchhoff）専用だが、4 節点
**MITC4** 四角形フラットシェル（`shell_mitc4.py`）は **Mindlin-Reissner 板**
（横せん断変形込み）なので**厚板から薄板まで**扱える。素の Mindlin 四角形は薄板で
せん断ロックするため、横せん断ひずみを辺中点でタイングする MITC4（Dvorkin-Bathe,
1984）でロックを回避する。

膜は双一次 Q4 平面応力、ドリリングは三角形と同じ架空剛性で、各節点 6 自由度

$$\mathbf{u}_{\text{node}} = [\,u_x,\; u_y,\; u_z,\; \theta_x,\; \theta_y,\; \theta_z\,]$$

なので梁・三角形シェルと混在できる。`Model.add_quad_shell(n1,n2,n3,n4,mat,t)` で
追加し、**節点は反時計まわり**に与える。

## 11.1 局所座標系（平面フェセット）

4 節点 $\mathbf{p}_1,\dots,\mathbf{p}_4$ の平均面に要素を載せる。重心
$\mathbf{c}=\tfrac14\sum\mathbf{p}_i$、対角線の外積で法線を取る。

$$
\mathbf{e}_3 = \frac{(\mathbf{p}_3-\mathbf{p}_1)\times(\mathbf{p}_4-\mathbf{p}_2)}
{\lVert\cdot\rVert},\quad
\mathbf{v}=(\mathbf{p}_2+\mathbf{p}_3)-(\mathbf{p}_1+\mathbf{p}_4),\quad
\mathbf{e}_1=\frac{\mathbf{v}-(\mathbf{v}\!\cdot\!\mathbf{e}_3)\mathbf{e}_3}{\lVert\cdot\rVert},\quad
\mathbf{e}_2=\mathbf{e}_3\times\mathbf{e}_1
$$

平面内座標は $x_i=(\mathbf{p}_i-\mathbf{c})\!\cdot\!\mathbf{e}_1,\;
y_i=(\mathbf{p}_i-\mathbf{c})\!\cdot\!\mathbf{e}_2$。平坦な四角形では厳密で、わずかな
反りは無視する（フラットシェル近似）。方向余弦行列 $\mathbf{R}=[\mathbf{e}_1;\mathbf{e}_2;\mathbf{e}_3]$。

## 11.2 形状関数とヤコビアン

自然座標 $(\xi,\eta)\in[-1,1]^2$、節点順 $1(-1,-1),2(1,-1),3(1,1),4(-1,1)$。

$$N_i=\tfrac14(1+\xi_i\xi)(1+\eta_i\eta)$$

ヤコビアン $\mathbf{J}=\begin{bmatrix}x_{,\xi}&y_{,\xi}\\x_{,\eta}&y_{,\eta}\end{bmatrix}$
（$x_{,\xi}=\sum N_{i,\xi}x_i$ など）。デカルト微分は
$[\,\partial_x;\partial_y\,]=\mathbf{J}^{-1}[\,\partial_\xi;\partial_\eta\,]$。

## 11.3 膜：Q4 平面応力

ひずみ $\boldsymbol\varepsilon=\mathbf{B}_m\mathbf{q}_m$（自由度順 $[u,v]\times4$）、

$$
\mathbf{B}_m=\begin{bmatrix}
N_{1,x}&0&\cdots\\ 0&N_{1,y}&\cdots\\ N_{1,y}&N_{1,x}&\cdots
\end{bmatrix},\qquad
\mathbf{D}_m=\frac{E}{1-\nu^2}\begin{bmatrix}1&\nu&0\\\nu&1&0\\0&0&\tfrac{1-\nu}{2}\end{bmatrix}
$$

膜剛性 $\mathbf{K}_m=\displaystyle\int t\,\mathbf{B}_m^\top\mathbf{D}_m\mathbf{B}_m\,\mathrm{d}A$
を 2×2 ガウスで積分（面積 $A=\int\det\mathbf{J}\,\mathrm{d}\xi\mathrm{d}\eta$ も同時に得る）。

## 11.4 板曲げ：MITC4（Mindlin + 仮定横せん断）

たわみ $w$ と回転 $\theta_x,\theta_y$ を独立に双一次補間する。回転の規約は DKT と共通。

**曲げ曲率**（$\mathbf{D}_b$ は [10 章](10_shell_element.md) と同じ曲げ剛性行列）::

$$
\kappa_x=\partial_x\theta_y,\quad \kappa_y=-\partial_y\theta_x,\quad
\kappa_{xy}=\partial_y\theta_y-\partial_x\theta_x
\;\Rightarrow\; \boldsymbol\kappa=\mathbf{B}_b\mathbf{q}_b
$$

**横せん断ひずみ**（薄板極限で 0）::

$$\gamma_{xz}=\partial_x w+\theta_y,\qquad \gamma_{yz}=\partial_y w-\theta_x$$

### 仮定横せん断（タイング）

これを素直に補間すると薄板でロックする。MITC4 は**共変**横せん断

$$
\gamma_\xi=\partial_\xi w + x_{,\xi}\theta_y - y_{,\xi}\theta_x,\qquad
\gamma_\eta=\partial_\eta w + x_{,\eta}\theta_y - y_{,\eta}\theta_x
$$

を辺中点 $A(0,\!-1),\,C(0,1)$（$\gamma_\xi$）, $D(\!-1,0),\,B(1,0)$（$\gamma_\eta$）で
サンプルし、線形に内挿する。

$$
\gamma_\xi=\tfrac12(1-\eta)\gamma_\xi^{A}+\tfrac12(1+\eta)\gamma_\xi^{C},\qquad
\gamma_\eta=\tfrac12(1-\xi)\gamma_\eta^{D}+\tfrac12(1+\xi)\gamma_\eta^{B}
$$

ガウス点でデカルト成分へ戻す（$\mathbf{J}$ はガウス点で評価）。

$$\begin{bmatrix}\gamma_{xz}\\\gamma_{yz}\end{bmatrix}=\mathbf{J}^{-1}\begin{bmatrix}\gamma_\xi\\\gamma_\eta\end{bmatrix}=\mathbf{B}_s\mathbf{q}_b$$

### 要素剛性

曲げ＋横せん断を 2×2 ガウスで積分する（横せん断剛性 $\mathbf{D}_s=k\,G\,t\,\mathbf{I}_2$,
$k=5/6$, $G=E/2(1+\nu)$）。

$$
\mathbf{K}_b^{\text{MITC4}}=\int\big(\mathbf{B}_b^\top\mathbf{D}_b\mathbf{B}_b
+\mathbf{B}_s^\top\mathbf{D}_s\mathbf{B}_s\big)\det\mathbf{J}\,\mathrm{d}\xi\mathrm{d}\eta
$$

自由度順は $[w,\theta_x,\theta_y]\times4$（12 自由度）。

## 11.5 ドリリング自由度 θz

三角形と同様、$\theta_z$ に一様回転（剛体回転）でゼロとなる微小架空剛性を与える。

$$
\mathbf{K}_d=\alpha\,E\,t\,A\Big(\mathbf{I}_4-\tfrac14\mathbf{1}\mathbf{1}^\top\Big),
\qquad \alpha=\texttt{DRILLING\_FACTOR}=10^{-3}
$$

$\mathbf{1}\mathbf{1}^\top$ は全要素 1 の $4\times4$ 行列。シェルのみの平面モデルでは
$\theta_z$ の大域スプリアスモードが残るので 1 点以上拘束するか梁と連成させる。

## 11.6 局所剛性の組み立てと座標変換

膜（$[u,v]\times4$）・板曲げ（$[w,\theta_x,\theta_y]\times4$）・ドリリング
（$[\theta_z]\times4$）を節点ごと $[u,v,w,\theta_x,\theta_y,\theta_z]$ 並びの 24×24
局所剛性 $\mathbf{k}$ に散らし込み、8 ブロック対角の変換
$\mathbf{T}=\mathrm{diag}(\mathbf{R},\dots,\mathbf{R})$ で

$$\mathbf{K}=\mathbf{T}^\top\mathbf{k}\,\mathbf{T}$$

として全体系 24×24 要素剛性を得る。組み立て（`assembly.py`）では梁（144）・3 節点
シェル（324）・4 節点シェル（576 エントリ）を同じ COO トリプレットに加える。

## 11.7 応力・断面力の回収

`shell.recover_shell_forces` は **要素中心 $\xi=\eta=0$** で膜応力
$\boldsymbol\sigma=\mathbf{D}_m\mathbf{B}_m\mathbf{q}_m$ と曲げモーメント
$\mathbf{M}=\mathbf{D}_b\mathbf{B}_b\mathbf{q}_b$（単位幅あたり）を評価し、結果を
`ShellForceResults.quad_shells` に格納する（要素ローカル系）。曲げ縁端応力は
$\sigma_b=6M/t^2$。成分キーは三角形と共通（`sx,sy,sxy / Mx,My,Mxy / sbx,sby`）。
`print_table(..., which="quad"|"tri"|"all")` で表示対象を切替える。

## 11.8 最適化との連成

四角形シェルは固定剛性として `SizingProblem`（[5 章](05_sizing_optimization.md)）に
組み込まれる（板厚は設計変数ではない）。梁リブ（剛体オフセット可）を設計変数に
すれば、四角形シェル板＋リブのサイジング最適化が解ける。解析的感度はシェル・
オフセットを含めても有限差分と一致する。連続版・離散版の例:

- `examples/ribbed_plate_quad_sizing.py`（連続サイジング・四角形シェル応力回収）
- `examples/ribbed_plate_quad_discrete.py`（規格寸法・総当たり大域最適と貪欲が一致）

## 11.9 検証（`tests/test_mitc4.py` / `examples/plate_mitc4.py`）

- **剛体モード**：$w$ 並進・$\theta_x,\theta_y$ 回転でゼロエネルギー。
- **薄板でロックしない**：$a/t=1000$ の単純支持板が Kirchhoff(Navier) 解
  $w=0.00406\,qa^4/D$ へ収束（16×16 で誤差 <1%、粗メッシュでも過小評価しない）。
- **厚板のせん断変形**：$a/t=10$ で Kirchhoff より $\sim$12% 大きいたわみへ収束。
- **膜パッチ（Q4 は線形場を厳密再現）・座標変換不変性・応力回収**。

## 11.10 コード対応

| 数式・処理 | 関数（`shell_mitc4.py`） |
|---|---|
| 形状関数・微分 | `q4_shape` |
| ヤコビアン $\mathbf{J},\det\mathbf{J},\mathbf{J}^{-1}$ | `q4_jacobian` |
| 膜 $\mathbf{B}_m$ | `_membrane_B` |
| 膜剛性 $\mathbf{K}_m$・面積 | `q4_membrane_stiffness` |
| 曲げ $\mathbf{B}_b$ | `_bending_B` |
| 共変横せん断 $\gamma_\xi,\gamma_\eta$ | `_covariant_shear_row` |
| 仮定横せん断 $\mathbf{B}_s$（タイング＋$\mathbf{J}^{-1}$） | `_mitc4_shear_B` |
| 板曲げ＋せん断剛性（12×12） | `mitc4_plate_stiffness` |
| ドリリング $\mathbf{K}_d$ | `quad_drilling_stiffness` |
| 局所座標系 $\mathbf{R},x_i,y_i$ | `quad_shell_frame` |
| 局所 24×24 剛性（散らし込み） | `quad_shell_local_stiffness`（`_Q_MEMBRANE/_Q_BENDING/_Q_DRILL`） |
| 変換 $\mathbf{T}$（8 ブロック） | `quad_shell_transformation` |
| 全体 24×24 剛性 $\mathbf{T}^\top\mathbf{k}\mathbf{T}$ | `quad_shell_stiffness_global` |
| 応力合力（中心） | `quad_shell_stress_resultants` |
| 全体組み立て・自由度マップ | `assembly.assemble_stiffness` / `quad_shell_dof_map` |
| モデル登録 | `Model.add_quad_shell` / `QuadShellElement` |

参考: E. Dvorkin, K.-J. Bathe (1984), "A continuum mechanics based four-node
shell element for general non-linear analysis", *Eng. Comput.* 1, 77–88.
K.-J. Bathe, *Finite Element Procedures*.
