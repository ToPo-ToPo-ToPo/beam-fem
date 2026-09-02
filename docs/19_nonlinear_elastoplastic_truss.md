# 非線形・弾塑性トラス解析

`beamfem.nonlinear_truss` は、既存の線形 `solve_static()` と独立した増分解析経路で
ある。線形APIと結果型は変更せず、トラスについて材料履歴、除荷・再載荷、残留変形
および任意のcorotational幾何非線形を扱う。

## 材料更新

`UniaxialMaterialModel` は次の2操作を定義する。

- `initial_state()`：塑性ひずみ、相当塑性ひずみ、応力、散逸エネルギーをゼロ初期化
- `update(strain, committed)`：直前の収束済み状態から応力、一貫接線、試行状態を返す

実装済み材料は以下である。

- `ElasticPerfectlyPlastic(E, yield_stress)`
- `BilinearIsotropicHardening(E, yield_stress, tangent_modulus)`

試行応力と降伏関数は

\[
 \sigma_{trial}=E(\varepsilon-\varepsilon^p_n),\qquad
 f=|\sigma_{trial}|-(\sigma_y+H\alpha_n)
\]

である。`f > 0` のとき、後退Euler return-mappingを

\[
 \Delta\gamma={f\over E+H},\quad
 \varepsilon^p_{n+1}=\varepsilon^p_n+
 \Delta\gamma\,\mathrm{sign}(\sigma_{trial})
\]

として実行する。二直線モデルに入力する `tangent_modulus = E_t` と内部硬化係数は

\[
 H={E E_t\over E-E_t},\qquad E_{alg}={EH\over E+H}=E_t
\]

の関係にある。Newton反復中は各反復の試行状態を次の反復へ累積せず、増分開始時の
収束済み状態から毎回更新する。状態は増分収束時だけcommitされる。

散逸エネルギー密度は `sigma_y * Δgamma` を履歴加算し、解析結果では初期体積を
乗じた全要素の合計をJで返す（SI入力時）。等方硬化に保存されるエネルギーとは区別
している。

## 増分Newton-Raphson法

`solve_nonlinear_truss()` は `model.nodal_loads` を基準荷重とし、例えば
`load_factors=[0, 1, 0, 0.5]` のような荷重履歴を追跡する。各増分で

\[
 K_t\Delta u=\lambda F_{ref}-F_{int}(u)
\]

を解き、力残差の絶対・相対許容値をともに満たすまで反復する。線形探索で残差を
減少させ、失敗した増分は `cutback_factor` で縮小する。少ない反復で収束した増分は
`growth_factor` で再拡大する。最小増分でも収束しなければ、最後に収束した荷重係数、
残差および原因を `LimitStateReport` に保存する。

```python
from beamfem import (
    BilinearIsotropicHardening, Material, Model, Section,
    UX, UY, UZ, solve_nonlinear_truss,
)

E, fy, Et = 200e9, 250e6, 10e9
model = Model()
n0 = model.add_node(0, 0, 0)
n1 = model.add_node(2, 0, 0)
section = Section(A=1e-3, Iy=1e-8, Iz=1e-8, J=2e-8)
model.add_truss(n0, n1, Material(E), section)
model.pin(n0)
model.fix(n1, [UY, UZ])
model.add_load(n1, UX, 300e3)

result = solve_nonlinear_truss(
    model,
    BilinearIsotropicHardening(E, fy, Et),
    load_factors=[0, 1, 0],
    n_steps=10,
)

assert result.converged
print(result.node_disp(n1)[UX])       # 0.0095 m residual displacement
print(result.dissipated_energy)       # 2375 J
print(result.element_states[0].plastic_strain)
```

完全塑性の水平 plateauを追跡する場合は、荷重制御では物理的に接線剛性がゼロになり、
降伏荷重が極限荷重として報告される。post-yield経路そのものを追うには
`displacement_pattern={(node, UX): target_displacement}` を指定する。

## corotational幾何非線形

`geometric_nonlinear=True` では現在座標から軸方向と長さを更新し、

\[
 \varepsilon={l-L_0\over L_0}
\]

を材料へ渡す。接線には材料剛性に加えて初期応力（幾何）剛性

\[
 {N\over l}(I-n\otimes n)
\]

を含める。大回転・小ひずみトラスを対象とし、荷重制御で平衡経路を追えなくなった
場合はcollapseとして返す。arc-length法によるlimit point後の経路追跡は未対応で
ある。

## 結果と最適化連携

`NonlinearTrussResult` は以下を保持する。

- 全accepted stepの荷重係数、反復数、残差、cutback数、変位、反力
- 要素応力、軸力、ひずみ、接線、降伏状態
- 塑性ひずみ、相当塑性ひずみ、散逸エネルギー
- 初回降伏係数、最大到達係数、最後の収束係数、collapse理由
- 部材の初回降伏から全体非収束までを順序付けた
  `progressive_collapse_sequence`
- 最終荷重係数がゼロの場合の `residual_displacement`

`NonlinearTrussSubproblem` はモデル・材料・目的関数factoryを受け取り、既存のExact、
Greedy、SA、QUBO backendが期待する `initial_design`、`domains`、`evaluate()` を公開
する。非収束またはcollapseした候補は `feasible=False` と有限の違反度で返り、最適化
結果へ昇格しない。質量、目的値、初回降伏・極限係数、最大相当塑性ひずみ、残留変位、
散逸エネルギーは `NonlinearDesignEvaluation` から直接取得できる。塑性ひずみと残留
変位の上限も候補制約として設定できる。

`examples/nonlinear_truss_optimization.py` は2本の候補平行部材について
`OFF / small / large` を選ぶトポロジー・断面最適化をExactとGreedyで解く。全部材OFF
は機構候補、1本smallは過大塑性・残留変位として棄却され、両smallが最小質量の可行
設計として選ばれる。

`progressive_collapse_sequence` は、完全塑性または硬化材料で各部材が初めて降伏した
順序と、最終的な全体接線特異・非収束を記録する。これは「逐次降伏による塑性機構
形成」の機械可読な記録である。破断、局部座屈、要素削除を模擬したものではなく、
それらを含む物理的な進行性破壊解析とは称さない。

## Verificationと適用限界

自動試験は、材料点return-mappingと数値接線、完全塑性棒の載荷・除荷・再載荷、
硬化棒の残留変位とエネルギー、二部材V型トラス、corotational応答、完全塑性極限荷重、
非収束候補の最適化上の棄却を検証する。

現時点ではトラス軸力だけを対象とする。フレーム塑性ヒンジ、局部座屈、接触、破断・
要素削除、動的collapse、arc-length法、温度依存性は含まない。また、外部非線形
ソルバーおよび実験との独立Validationを終えるまでは、結果を設計承認に使用しては
ならない。
