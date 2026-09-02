# 実用向け離散構造・量子最適化

本章は、`beamfem` のFEMを唯一の構造評価器として、離散断面・部材配置を複数の
古典／量子バックエンドで最適化するための設計方針を定める。

## 適用範囲

最初の実用対象は、線形弾性範囲の2D/3D鋼製トラス・骨組の基本設計支援である。
荷重ケースと組合せ、離散断面、部材有無、質量、応力、変位、部材座屈および
製作上の基本制約を扱う。法規適合の最終判定、接合部詳細、塑性、疲労、動的応答を
自動的に保証するものではない。

## 設計原則

1. FEM評価はソルバーから独立させ、全バックエンドで共有する。
2. 最適化結果は必ずFEMで再評価し、QUBO energyだけで採用しない。
3. Qiskitはoptional dependencyとし、古典機能は単独で動作させる。
4. 単位はSIで統一し、入力境界でのみ変換する。
5. seed、設定、バージョン、Git commit、評価履歴を監査情報として保存する。
6. 量子優位性を仮定せず、Exact/MILP/Greedy/SAと同じ条件で比較する。
7. 機構、特異行列、非有限値、適用範囲外を明示的な失敗として扱う。

## アーキテクチャ

```text
Model + catalog + load combinations + constraints
                         |
                         v
            DiscreteStructuralProblem
                         |
                         v
              StructuralEvaluator
       (objective, utilization, diagnostics)
                         |
          +--------------+--------------+
          |              |              |
       Exact/MILP    Greedy/SA     QUBO/QAOA
          |              |              |
          +--------------+--------------+
                         |
                         v
              exact FEM re-evaluation
                         |
                         v
              result + audit + reports
```

`DiscreteStructuralProblem` は設計状態、断面カタログ、荷重組合せ、制約、目的関数を
所有する。`evaluate(design)` はソルバーに依存しない `EvaluationResult` を返す。
バックエンドは問題を変更せず、候補設計の生成だけを担当する。

## 制約の扱い

制約は機械可読なレコードとして返し、少なくとも次を保持する。

- 制約種別と対象ID
- 荷重ケース／組合せ
- demand、capacity、利用率、余裕
- 合否と警告
- 支配組合せ

QUBOのペナルティは探索用であり、実際の合否はFEM側の制約レコードで決める。
ペナルティは係数範囲とfeasibilityを観測して調整し、採用値を監査情報へ記録する。

## 局所QUBOとtrust region

大規模なone-hot QUBOを直接statevectorで解かず、現在設計周辺の候補変更から疎な
局所QUBOを作る。候補は利用率、ひずみエネルギー、質量差、座屈余裕、接続性、
過去の改善履歴から選ぶ。全組合せFEM評価で候補を事前選別してはならない。

予測改善量とFEM実改善量から

```text
rho = actual_improvement / predicted_improvement
```

を計算する。`rho` が高ければtrust regionを拡大し、低ければ縮小する。改善しない
候補はrollbackし、同じ設計の周期、評価上限、時間上限でも停止する。

## バックエンド契約

各バックエンドは共通して、設計、目的値、feasibility、制約、評価回数、実行時間、
履歴、solver metadataを返す。近似解法は複数seedを受け取れるようにする。

小規模問題ではExactまたはMILPを基準にoptimality gapを測る。QAOAではqubit数、
shots、reps、回路深さ、ゲート数、サンプル分布、最良解確率も保存する。

静定トラスの釣合い・断面耐力MILPは、その明示された範囲でのみ大域最適である。
不静定トラスでは弾性適合・変位・剛性安定性を含まない下界候補になり得るため、
共通FEMで不合格となった候補をMILP解と偽って採用しない。必要に応じて、既知の
FEM可行anchorからfeasibility-first探索を実行し、raw MILP候補、修復backend、
修復後設計を別々の監査項目として保存する。

## 性能方針

- 設計状態のハッシュによるFEMキャッシュ
- 荷重組合せ間の剛性分解再利用
- 不可能な接続状態のFEM前診断
- 候補評価の並列化
- 評価回数、wall time、メモリの上限
- checkpointと再開可能な履歴

ベンチマークではFEM、候補生成、QUBO構築、ソルバー、最終検証の時間を分離する。

`LocalQUBOBuilder` の既定候補選別は、質量差、制約利用率、ひずみエネルギー、
Euler座屈余裕、機構・連結性、直近merit改善を候補集合内で正規化して順位付けする。
方式と指標は `candidate_selection` / `candidate_indicators` に保存する。

小規模問題では `ParetoFrontBackend` が可行設計を列挙し、mass・cost・carbonの
非支配解を `ParetoResult` として返す。全結果の `normalized_work` はFEM評価、
古典目的関数評価、optimizer反復、量子shots、量子回路評価を別軸で記録する。
計算内容が異なる軸を相互換算せず、multi-startの上限は全startの合計に適用する。

## 検証

次の順に検証する。

1. 単一部材・教科書例題と解析解
2. 小規模離散問題と総当たり
3. MILP大域解との比較
4. legacyの159.0449、91.7192、88.7033 kg回帰
5. 機構、ゼロ長、重複、過拘束、NaN/Infなどの異常系
6. small/medium/largeの性能回帰
7. 複数seedによるSA/QAOAの成功率と分布

## 実用判定

ベータリリースには、全バックエンドの共通FEM、SI単位、再現可能な入力と監査ログ、
機構の安全な拒否、支配制約の出力、小規模大域解との比較、Medium規模の所定時間内
完了、QAOA失敗時の古典フォールバック、および構造設計者によるレビューを要求する。
