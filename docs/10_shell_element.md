# 10. 三角形フラットシェル要素（CST 膜 + DKT 板曲げ）

梁要素に加え、3 節点の平面三角形「フラットシェル」要素を実装している
（`shell3d.py`）。面内挙動（膜）と面外挙動（板曲げ）をそれぞれ別の定式化で
評価し、要素自身の平面内で重ね合わせて全体系へ変換する。各節点の自由度は
梁と共通の 6 自由度

$$\mathbf{u}_{\text{node}} = [\,u_x,\; u_y,\; u_z,\; \theta_x,\; \theta_y,\; \theta_z\,]$$

なので、梁と同じモデル・同じソルバに混在できる。

## 10.1 局所座標系

3 節点の位置 $\mathbf{p}_1,\mathbf{p}_2,\mathbf{p}_3$ から要素の平面と局所軸を決める。

$$
\mathbf{e}_1 = \frac{\mathbf{p}_2-\mathbf{p}_1}{\lVert\mathbf{p}_2-\mathbf{p}_1\rVert},\quad
\mathbf{e}_3 = \frac{(\mathbf{p}_2-\mathbf{p}_1)\times(\mathbf{p}_3-\mathbf{p}_1)}
{\lVert(\mathbf{p}_2-\mathbf{p}_1)\times(\mathbf{p}_3-\mathbf{p}_1)\rVert},\quad
\mathbf{e}_2 = \mathbf{e}_3\times\mathbf{e}_1
$$

局所 $x$ は辺 1→2、局所 $z$ は要素法線。方向余弦行列 $\mathbf{R}=[\mathbf{e}_1;\mathbf{e}_2;\mathbf{e}_3]$
（各行が局所軸）で $\mathbf{v}_{\text{local}} = \mathbf{R}\,\mathbf{v}_{\text{global}}$。
節点1を原点、節点2を局所 $+x$ 上に置いた平面内座標 $(x_i,y_i)$ を用いて剛性を作る。

## 10.2 膜：定ひずみ三角形（CST）

面内変位 $(u,v)$ を線形補間する定ひずみ三角形。ひずみ
$\boldsymbol\varepsilon=[\varepsilon_x,\varepsilon_y,\gamma_{xy}]^\top$ は要素内で一定で、

$$
\mathbf{B}_m = \frac{1}{2A}
\begin{bmatrix}
b_1 & 0 & b_2 & 0 & b_3 & 0\\
0 & c_1 & 0 & c_2 & 0 & c_3\\
c_1 & b_1 & c_2 & b_2 & c_3 & b_3
\end{bmatrix},\quad
b_i = y_j-y_k,\; c_i = x_k-x_j
$$

（$i,j,k$ は巡回、$A$ は三角形面積）。平面応力の構成則

$$
\mathbf{D}_m = \frac{E}{1-\nu^2}
\begin{bmatrix} 1 & \nu & 0\\ \nu & 1 & 0\\ 0 & 0 & \tfrac{1-\nu}{2}\end{bmatrix}
$$

を用い、膜剛性は $\mathbf{K}_m = t\,A\,\mathbf{B}_m^\top \mathbf{D}_m \mathbf{B}_m$（$t$ は板厚）。
自由度順は $[u_1,v_1,u_2,v_2,u_3,v_3]$。

## 10.3 板曲げ：離散 Kirchhoff 三角形（DKT）

たわみ $w$ と回転 $\theta_x,\theta_y$ を自由度に持つ Batoz の DKT 要素。法線の回転
$\beta_x,\beta_y$ を2次の不完全多項式で補間し、辺中点で Kirchhoff 拘束
（横せん断ひずみ＝0）を離散的に課して節点自由度に縮約する。曲率

$$
\boldsymbol\kappa = [\,\partial_x\beta_x,\;\partial_y\beta_y,\;\partial_y\beta_x+\partial_x\beta_y\,]^\top
= \mathbf{B}_b(\xi,\eta)\,\mathbf{q}_b
$$

は面積座標 $(\xi,\eta)$ の1次式となる。曲げ構成則（曲げ剛性 $D=\dfrac{Et^3}{12(1-\nu^2)}$）

$$
\mathbf{D}_b = \frac{E t^3}{12(1-\nu^2)}
\begin{bmatrix} 1 & \nu & 0\\ \nu & 1 & 0\\ 0 & 0 & \tfrac{1-\nu}{2}\end{bmatrix}
$$

を用い、板曲げ剛性は

$$
\mathbf{K}_b = \iint_\Omega \mathbf{B}_b^\top \mathbf{D}_b \mathbf{B}_b\,\mathrm{d}\Omega
= \frac{A}{3}\sum_{g=1}^{3}\mathbf{B}_b(\xi_g,\eta_g)^\top \mathbf{D}_b \mathbf{B}_b(\xi_g,\eta_g)
$$

被積分関数が $(\xi,\eta)$ について2次なので、辺中点 $(\tfrac12,0),(\tfrac12,\tfrac12),(0,\tfrac12)$
の3点積分で厳密。自由度順は $[w_1,\theta_{x1},\theta_{y1},\,w_2,\theta_{x2},\theta_{y2},\,w_3,\theta_{x3},\theta_{y3}]$。
回転の規約は $\partial w/\partial y=\theta_x$, $\partial w/\partial x=-\theta_y$（剛体回転
$w=\theta_x y-\theta_y x$ がゼロエネルギーになることで検証される）。

DKT は薄板（Kirchhoff）理論に基づき横せん断変形を無視するため、**薄肉シェル向き**
（厚板では Mindlin 系要素が必要）。

## 10.4 ドリリング自由度 θz

CST と DKT はともに面法線まわり回転 $\theta_z$ に剛性を持たない。フラットシェルを
6 自由度で扱うため、$\theta_z$ に微小な架空（ドリリング）剛性

$$
\mathbf{K}_d = \alpha\,E\,t\,A
\begin{bmatrix} 1 & -\tfrac12 & -\tfrac12\\ -\tfrac12 & 1 & -\tfrac12\\ -\tfrac12 & -\tfrac12 & 1\end{bmatrix},
\quad \alpha = \text{\texttt{DRILLING\_FACTOR}}\;(=10^{-3})
$$

を与える。行和が0なので一様回転（剛体回転）に対してはゼロエネルギーとなり、
6 つの剛体運動は厳密に保たれる。その代償として、**シェルのみの平面モデルでは
$\theta_z$ の大域スプリアスモードが残る**ため、$\theta_z$ を1点以上拘束するか
梁要素と連成させて用いる。

## 10.5 局所剛性の組み立てと座標変換

膜・板曲げ・ドリリングを節点ごと $[u,v,w,\theta_x,\theta_y,\theta_z]$ の並びの
18×18 局所剛性 $\mathbf{k}$ に散らし込み、6 ブロック対角の変換行列
$\mathbf{T}=\mathrm{diag}(\mathbf{R},\dots,\mathbf{R})$ で

$$\mathbf{K} = \mathbf{T}^\top \mathbf{k}\,\mathbf{T}$$

として全体系の 18×18 要素剛性を得る。組み立て（`assembly.py`）では梁（144 エントリ）
とシェル（324 エントリ）を同じ COO トリプレットに加える。

## 10.6 梁との連成（リブ補強板）

シェルと梁は同じ節点6自由度・同じ全体行列に組み立てられるので、**同一節点を
共有**させれば板（シェル）とリブ（梁）が連成するスティフナ付き板を直接解ける。
板の曲げ・面内剛性とリブの曲げ・ねじり剛性が節点でつながり、荷重を分担する。

リブを板中立面より下げる**偏心（オフセット）**は、梁要素の ``offset`` 引数
（剛体オフセット, [1.5 節](01_fem_theory.md)）で表現できる。``add_element(..., offset=[0,0,-e])``
とすると梁図心を板下面（偏心 $e=t/2+h/2$）に下げ、板の曲げ回転がリブの軸伸縮を
生む軸-曲げ連成（T 形断面の合成剛性 $EA e^2$）が立ち上がる。

この合成効果は、リブの軸力を板の膜（シェル面内剛性）が分担して初めて働くため、
**面内自由度 UX, UY を内部で自由**にしておく必要がある（外周など最小限で面内を
保持して剛体運動を除く）。偏心を入れない（``offset`` 無し＝中立面に同心）場合は
リブ軸力が立たず合成効果が出ないので、補強効果を控えめ（安全側）に評価する。
リブ補強円板の例 `examples/ribbed_plate_shell.py` では、めり込み（$e=0$）と
正規オフセット（$e=20$ mm）でたわみが大きく変わる様子を比較している。

## 10.7 応力・断面力の回収

`shell.recover_shell_forces` は局所節点変位から
膜応力 $\boldsymbol\sigma=\mathbf{D}_m\mathbf{B}_m\mathbf{q}_m$（要素内一定）と、
重心 $(\xi=\eta=\tfrac13)$ での曲げモーメント
$\mathbf{M}=\mathbf{D}_b\mathbf{B}_b\mathbf{q}_b$（単位幅あたり）を返す。
曲げ縁端応力は $\sigma_b = 6M/t^2$。**いずれも要素ローカル座標系**の値である。

成分キー: 膜応力 `sx, sy, sxy` / 曲げモーメント `Mx, My, Mxy` / 曲げ縁端応力 `sbx, sby`。

## 10.8 検証

`tests/test_shell.py` で次を確認している。

- **剛体モード**：局所剛性が 6 つの物理剛体運動でエネルギーを持たない。
- **膜パッチテスト**：一様引張で変位が線形・応力が一様（CST は厳密）。
- **単純支持正方形板**：中央たわみが Navier 級数解 $w=0.00406\,qa^4/D$ に収束
  （16×16 分割で誤差 1% 未満、`examples/plate_shell.py`）。
- **等分布荷重を受ける円板**：中央たわみが Kirchhoff 円板の解析解に収束
  （`examples/circular_plate_shell.py`, 放射 24×8 分割で誤差 1% 未満）。

$$
w_{\text{center}}^{\text{clamped}} = \frac{q R^4}{64 D},\qquad
w_{\text{center}}^{\text{simply supported}} = \frac{5+\nu}{1+\nu}\,\frac{q R^4}{64 D}
$$

- **応力回収・座標変換不変性**。

## 10.9 四角形フラットシェル MITC4（厚板対応）

3 節点 DKT は薄板（Kirchhoff）専用だが、4 節点 **MITC4**（`shell_mitc4.py`）は
**Mindlin-Reissner 板**（横せん断変形込み）なので厚板から薄板まで扱える。膜は
Q4 平面応力、ドリリングは三角形と同じ架空剛性、各節点 6 自由度で梁・三角形シェル
と混在できる（`Model.add_quad_shell`、節点は反時計まわり）。

### 板曲げ（Mindlin + 仮定横せん断）

回転 $\theta_x,\theta_y$ とたわみ $w$ を独立に双一次補間する。曲げ曲率は DKT と同じ
規約（$\kappa_x=\partial\theta_y/\partial x$ など）。横せん断ひずみ

$$\gamma_{xz}=\partial w/\partial x+\theta_y,\qquad \gamma_{yz}=\partial w/\partial y-\theta_x$$

を素直に補間すると薄板で**せん断ロック**するため、**MITC4**（Dvorkin-Bathe）で
共変横せん断を辺中点 $A(0,\!-1),B(1,\!0),C(0,\!1),D(\!-1,\!0)$ でサンプルし

$$\gamma_\xi=\tfrac12(1-\eta)\gamma_\xi^A+\tfrac12(1+\eta)\gamma_\xi^C,\quad
  \gamma_\eta=\tfrac12(1-\xi)\gamma_\eta^D+\tfrac12(1+\xi)\gamma_\eta^B$$

と仮定し、$[\gamma_{xz},\gamma_{yz}]^\top=\mathbf{J}^{-1}[\gamma_\xi,\gamma_\eta]^\top$ で
デカルト成分へ戻す。要素剛性は曲げ $\mathbf{D}_b$ と横せん断
$\mathbf{D}_s=k\,G\,t\,\mathbf{I}_2$（$k=5/6$）を 2×2 ガウスで積分する。

### 検証（`tests/test_mitc4.py` / `examples/plate_mitc4.py`）

- **剛体モード**：$w$ 並進・$\theta_x,\theta_y$ 回転でゼロエネルギー。
- **薄板でロックしない**：$a/t=1000$ の単純支持板が Kirchhoff(Navier) 解へ収束
  （16×16 で誤差 <1%、粗メッシュでも過小評価しない）。
- **厚板のせん断変形**：$a/t=10$ で Kirchhoff より $\sim$12% 大きいたわみへ収束。
- **膜パッチ（Q4）・座標変換不変性**。

参考: J.-L. Batoz, K.-J. Bathe, L.-W. Ho (1980), "A study of three-node
triangular plate bending elements", *Int. J. Numer. Methods Eng.* 15.
E. Dvorkin, K.-J. Bathe (1984), "A continuum mechanics based four-node shell
element for general non-linear analysis", *Eng. Comput.* 1.
