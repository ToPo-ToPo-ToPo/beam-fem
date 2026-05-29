# 2. ソルバ：組み立て・境界条件・静的線形解析

実装：[`src/beamfem/assembly.py`](../src/beamfem/assembly.py), [`src/beamfem/solver.py`](../src/beamfem/solver.py)

## 2.1 全体剛性の組み立て

全体剛性行列は各要素の寄与の重ね合わせ：

$$\mathbf{K} = \sum_e \mathbf{L}_e^\top\, \mathbf{K}^e\, \mathbf{L}_e$$

$\mathbf{L}_e$ は要素自由度（12）を全体自由度（$6N$）へ写す選択行列。実装では明示的に
作らず、各要素の全体自由度番号 `dof_map`（節点1の6自由度＋節点2の6自由度）へ
$12\times12$ 行列を散布（scatter）する。

数千要素規模を想定し、**COO 形式のトリプレット**（行・列・値）を蓄積してから
**CSR 疎行列**へ変換する。各要素 $12\times12=144$ エントリ。

```
K = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
```

5000 要素・約 3 万自由度の組み立て＋求解が 1 秒未満（疎ソルバ）。

## 2.2 荷重ベクトル

節点荷重・モーメントを全体自由度に配置：

$$F_{\,6\cdot\text{node}+\text{dof}} \mathrel{+}= \text{value}$$

> 現状は**節点荷重のみ**。部材分布荷重は等価節点荷重に変換して入力する（例：
> [蜘蛛の巣フレーム例](../examples/spider_web_3d.py)では面分布荷重を三角形分割・1/3 集約で
> 節点荷重化）。

## 2.3 境界条件と縮約系

自由度を**自由 (free)** と**拘束 (constrained)** に分割する。拘束自由度には強制変位
$\mathbf{u}_c$（固定支持なら 0）を与える。剛性を分割すると

$$\begin{bmatrix} \mathbf{K}_{ff} & \mathbf{K}_{fc} \\ \mathbf{K}_{cf} & \mathbf{K}_{cc} \end{bmatrix}
  \begin{bmatrix} \mathbf{u}_f \\ \mathbf{u}_c \end{bmatrix} =
  \begin{bmatrix} \mathbf{F}_f \\ \mathbf{F}_c \end{bmatrix}$$

自由自由度について縮約系を解く：

$$\boxed{\;\mathbf{K}_{ff}\, \mathbf{u}_f = \mathbf{F}_f - \mathbf{K}_{fc}\, \mathbf{u}_c\;}$$

固定支持（$\mathbf{u}_c=\mathbf{0}$）では右辺第2項は消える。

支持の種類（`model.py`）：

- `fix(node)`：全 6 自由度を固定（完全固定）
- `fix(node, [dofs])`：指定自由度のみ固定（ローラー等）
- `pin(node)`：並進 3 自由度を固定、回転自由
- `fix_to_plane_xy()`：全節点の面外自由度 $(u_z, \theta_x, \theta_y)$ を拘束 → 2D 面内骨組

## 2.4 求解と反力

$\mathbf{K}_{ff}$ を LU 分解（`scipy.sparse.linalg.splu`）して解く。分解は保持でき、
複数右辺・最適化の反復で再利用できる。

反力は

$$\mathbf{R} = \mathbf{K}\mathbf{u} - \mathbf{F}$$

を計算し、拘束自由度成分のみが物理的な反力として意味を持つ。

### ソルバの差し替え

求解本体は `solver._solve_sparse(A, b)` に集約しており、ここを差し替えるだけで
PARDISO（`pypardiso`）や CHOLMOD（`scikit-sparse`）へ交換できる。業務での性能要求に
対応するための設計。

## 2.5 結果オブジェクト

`solve_static` は `StaticResult` を返す：

- `u`：全自由度変位 $(6N,)$
- `reactions`：全自由度反力（自由自由度は 0）
- `K`：組み立て済み全体剛性（再利用用）
- `node_disp(node)`：指定節点の 6 自由度変位 $[u_x,u_y,u_z,\theta_x,\theta_y,\theta_z]$

## 2.6 検証

- 片持ち梁先端たわみが Timoshenko 解析解と厳密一致（[1 章](01_fem_theory.md)）
- 門型ラーメンで反力の釣り合い（水平反力合計 = 外力）を確認
- 剛体並進モードに対し $\mathbf{K}\mathbf{t}_{\text{rigid}}\approx\mathbf{0}$
