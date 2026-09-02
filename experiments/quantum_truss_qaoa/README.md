# 量子・古典ハイブリッドトラス最適化

離散トラス最適化を、共通の **QUBO Master + Classical FEM Subproblem** で
Simulated Annealing（SA）と Qiskit QAOA に解かせて比較する実験フォルダです。

ユーザー提供の過去コードを `legacy/` に無改変で保存し、そのうち
`hybrid_qubo_truss_sa.py` の形状、断面、荷重ケース、FEM、変位・応力・Euler座屈判定、
制約違反スコア、貪欲warm startを比較版から直接再利用しています。

## ファイル

- `hybrid_qubo_truss_compare.py` — 同一QUBOをSAとQAOAで比較する実行ファイル
- `comparison_results.json` — 検証実行の機械可読結果
- `requirements.txt` — 実験用の追加依存関係
- `legacy/basic_truss_milp.py` — 最初の小規模MILP版
- `legacy/mixed_integer_truss_milp.py` — 複数荷重・断面選択・座屈を含むMILP版
- `legacy/hybrid_qubo_truss_sa.py` — 元のQUBO Master + FEM + SA版

添付されていた小規模MILPコード2本は内容とSHA-256が一致していたため、1本だけ保存しています。

## セットアップ

リポジトリのルートで実行します。

```bash
python3 -m venv .venv-qaoa
.venv-qaoa/bin/python -m pip install -r experiments/quantum_truss_qaoa/requirements.txt
```

Python 3.10以降を推奨します。検証時のバージョンは次のとおりです。

- Python 3.13.9
- NumPy 2.5.2
- SciPy 1.18.1
- Qiskit 2.5.2
- Qiskit Optimization 0.7.0
- Qiskit Aer 0.17.2

## 実行

SAとローカルstatevector QAOAを比較します。

```bash
.venv-qaoa/bin/python \
  experiments/quantum_truss_qaoa/hybrid_qubo_truss_compare.py \
  --solver both
```

結果をJSONにも保存する場合：

```bash
.venv-qaoa/bin/python \
  experiments/quantum_truss_qaoa/hybrid_qubo_truss_compare.py \
  --solver both \
  --json experiments/quantum_truss_qaoa/comparison_results.json
```

Aer Sampler V2を使う場合：

```bash
.venv-qaoa/bin/python \
  experiments/quantum_truss_qaoa/hybrid_qubo_truss_compare.py \
  --solver qaoa \
  --qaoa-backend aer
```

SAだけなら `--solver sa`、最適トラスを描画する場合は `--plot` を指定します。

## バックエンドの差し替え

Pythonから `solve_qubo_qaoa(..., sampler=..., pass_manager=...)` を呼べます。
CLIでは、引数なしのfactory関数を次の形式で指定できます。

```bash
--sampler-factory package.module:function
```

factoryは `sampler`、または `(sampler, pass_manager)` を返します。バックエンドを使う
Sampler V2では、そのバックエンド用に生成したpass managerも渡してください。

## 実装上の比較条件

元コードは16部材×5状態の80個のone-hot変数を持ちます。これをそのままstatevectorで
シミュレーションするのは現実的でないため、現在の局所設計から選んだ6個の設計変更と、
「最大2部材変更」を表す2個のslack bitによる8量子ビットQUBOを作ります。

1部材変更と2部材同時変更の係数は元のFEM結果から作り、SAとQAOAには完全に同じ
QUBOを渡します。QAOAのHamiltonianスケールを安定させるためenergyは無次元正規化
しています。ソルバーの出力は必ず、元コードの `analyze_design()` で再解析します。

比較項目は次のとおりです。

- 正規化QUBO energy
- 元FEMによるscore
- mass [kg]
- feasibilityと制約違反量
- 実行時間と選択された部材変更

## 検証結果

seed 123、QAOA `reps=1`、1024 shots、COBYLA 60 iterationsで確認した結果です。

| 段階・ソルバー | FEM score | mass [kg] | feasible |
|---|---:|---:|:---:|
| 元の全L断面設計 | 159.044916 | 159.044916 | True |
| 元の貪欲warm start | 91.719151 | 91.719151 | True |
| SA | 88.703279 | 88.703279 | True |
| QAOA | 88.703279 | 88.703279 | True |

SAとQAOAはいずれも `M12: NONE -> S` と `M13: M -> NONE` の同時変更を選び、
元の会話で得られていた88.70 kgの設計を再現します。

## Qiskit API

現行構成では次を使用しています。

- `qiskit.primitives.StatevectorSampler`（Sampler V2）
- `qiskit_optimization.minimum_eigensolvers.QAOA`
- `qiskit_optimization.algorithms.MinimumEigenOptimizer`
- `qiskit_optimization.optimizers.COBYLA`

参考：

- [Qiskit Optimization: Minimum Eigen Optimizer](https://qiskit-community.github.io/qiskit-optimization/tutorials/03_minimum_eigen_optimizer.html)
- [Qiskit Optimization: QAOA API](https://qiskit-community.github.io/qiskit-optimization/stubs/qiskit_optimization.minimum_eigensolvers.QAOA.html)
- [IBM Quantum: primitives](https://docs.quantum.ibm.com/api/qiskit/primitives)

## 注意

このフォルダは研究・学習用の比較実験です。実務設計、設計規準への適合確認、
安全性判定を目的としたコードではありません。
