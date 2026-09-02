# 開発計画の最終実施状況

最終更新: 2026-09-02
対象ブランチ: `feature/production-v1`

この文書は、合意したフェーズ0〜8と、その後に追加した非線形・弾塑性ロードマップの
最終チェックリストである。`[x]` は実装と自動検証が完了した項目、`[ ] (外部作業)` は
ソースコードだけでは成立せず、資格、実案件、外部アカウントまたは署名が必要な項目を
表す。外部作業を実行済みと偽装することは、リリースゲートで拒否される。

## フェーズ0: 要件と合格基準

- [x] AISC 360-22軸力部材を最初の限定プレビュー規準として選定
- [x] トラス、フレーム、シェル、混合構造のモデル化方針を文書化
- [x] ピン・剛接合と、フレーム端の局所 `RX/RY/RZ` 任意端部解放をAPI・Schemaへ実装
- [x] small / medium / largeの規模、時間、メモリ、評価回数基準を数値化
- [x] FEM誤差、釣合い残差、最適性gap、確率的再現性の合格基準を数値化
- [x] SI単位、符号、局所座標、荷重組合せ規約を文書化
- [x] 非対応入力をSchema v2で拒否または明示
- [x] 製品要求、適用範囲、受入基準、RC／正式版の判定を文書化

主な証跡: [製品要求](15_product_requirements.md)、
[`acceptance_v1.json`](../validation/acceptance_v1.json)、
`tests/test_frame_releases_and_truss_strain.py`

## フェーズ1: トラス・混合要素解析

- [x] 2D/3D軸力トラス要素、方向余弦変換、引張正・圧縮負を実装
- [x] 軸力、軸応力、軸ひずみ、伸縮量を公開結果として回収
- [x] トラス・Timoshenkoフレーム・三角形／四角形シェルを共通全体方程式で解析
- [x] トラス・フレーム・シェル3種混在の閉形式V&Vを追加
- [x] Schema v2の `member_type: truss | frame` とフレーム端部解放を実装
- [x] 自重の等価節点荷重を実装
- [x] 回転自由度処理、機構・特異剛性診断を実装
- [x] ゼロ長、重複部材、非有限値、不安定構造を事前診断
- [x] 単一棒、三角トラス、3D tripod、Pratt、Warrenを手計算値と照合
- [x] 剛体回転不変性と既存回帰を検証

主な証跡: `truss3d.py`、`validation/mixed_assembly.py`、
[`reference_cases/`](../validation/reference_cases/)

## フェーズ2: Verification & Validation

- [x] 変位、反力、全体釣合い、軸力、曲げ、せん断、応力を解析解と照合
- [x] Euler座屈、自重、荷重組合せを検証
- [x] 2D/3D座標変換、剛体回転不変性、混合モデルを検証
- [x] 教科書解、手計算、レガシー159.04／91.72／88.70 kg回帰を自動化
- [x] ランダム小規模モデル、対称性、異常入力を検証
- [x] OpenSeesPyと線形棒変位・反力、および二直線材料応答を独立照合
- [x] 最大相対誤差0.5%以下、釣合い残差 `1e-8` 以下を自動ゲート化

主な証跡: [`reference_evidence.json`](../validation/reference_evidence.json)、
[`opensees_crosscheck_evidence.json`](../validation/opensees_crosscheck_evidence.json)

## フェーズ3: 設計基準照査

- [x] FEM需要値と規準照査を分離するインターフェースを実装
- [x] 引張、圧縮、細長比、有効座屈長を扱うAISC 360-22限定プレビューを実装
- [x] 使用限界変位、荷重組合せ、材料・断面係数を共通評価へ統合
- [x] 条文、式、係数、版、出典、errata確認日を結果へ保存
- [x] 対応範囲の圧延I形断面分類（AISC Table B4.1b Cases 10/15）を実装
- [x] 軸力・二軸曲げ相関（AISC H1-1a/H1-1b限定プレビュー）を実装
- [x] AISC v16 Companion公開例H.1Aとの数値回帰を証跡化
- [x] 未確認の適用条件・耐力を `NOT_VERIFIED` とし、自動承認を禁止
- [ ] (外部作業) 採用規準、公開errata、適用条件、係数の独立した有資格者レビューと署名

ここでの `[x]` は明記した限定範囲の実装完了を意味し、AISC全章への適合認証ではない。
主な証跡: [`code_check_reference_evidence.json`](../validation/code_check_reference_evidence.json)、
`validation/combined_steel_rules.py`

## フェーズ4: 性能と大規模化

- [x] 複数荷重組合せの剛性分解を再利用
- [x] 設計hashによる重複FEM評価を除外
- [x] 候補FEM、QUBO構築、QUBO求解、FEM再検証の時間を分離記録
- [x] 候補FEM評価をプロセス並列化しworkerを再利用
- [x] 公開 `SparseSolver` 契約、registry、名前／adapterによる差替えを実装
- [x] 問題contextとpayload SHA-256を検証する原子的な永続FEMキャッシュを実装
- [x] checkpoint、完全性検査、再開を実装
- [x] 時間、評価回数、メモリ上限と安全停止を実装
- [x] medium（51候補）／large（201候補）の性能・制限・再開証跡を生成
- [x] CI性能ゲートと3倍speedup基準を実装
- [x] 並列数によらない決定論的順位と最終FEM再評価を維持

主な証跡: [`performance_evidence.json`](../validation/performance_evidence.json)、
`optimize/persistent_cache.py`

## フェーズ5: 最適化の信頼性

- [x] トラス平衡・離散断面耐力の明示的MILPを実装
- [x] 不要なBig-Mを使わない厳密定式化を採用し、変数境界を直接制約へ反映
- [x] small問題でExactとMILPの設計・質量・FEM再評価一致を確認
- [x] Greedy等の決定論的multi-startを実装
- [x] SAを10 seed以上で評価し、feasibility率とbest/median/worstを保存
- [x] 一部材変更・二部材相互作用から局所QUBOを構築
- [x] 利用率、ひずみエネルギー、Euler余裕、連結性、改善履歴を既定候補選別へ統合
- [x] trust-region周期検出とfeasibility restorationを実装
- [x] 参照解がある場合の最適性gapを保存
- [x] 質量・コスト・CO2の重み付き目的と厳密小規模Paretoフロントを実装
- [x] solver checkpointと再開を実装
- [x] FEM評価、古典目的評価、反復、量子shots、回路評価を別軸の正規化作業量として保存
- [x] 非実行可能解を最終採用しない共通結果形式を全backendで使用

主な証跡: [`exact_milp_micro_evidence.json`](../validation/exact_milp_micro_evidence.json)、
`optimize/pareto.py`、`tests/test_normalized_work_budget.py`

## フェーズ6: 量子・ノイズ検証

- [x] Qiskit Samplerを差替え可能なQAOA backendとして実装
- [x] StatevectorSampler、Aer shot、noise modelを切替可能
- [x] pass manager、shots、reps、seed、optimizer上限を設定・保存
- [x] qubit数、回路深さ、2量子ビットgate数を保存
- [x] QUBO energy、選択確率、exact gap、FEM score、mass、feasibilityを保存
- [x] ノイズ付きローカルQAOAとSAのsmoke比較を実装
- [x] 独立readout mitigationを実装し、raw／mitigated分布を保存
- [x] CVaR目的関数を実装
- [x] queue、execution、wall時間のmetadata契約と欠損時の明示を実装
- [x] QAOA失敗時の宣言的な古典fallbackを実装
- [ ] (外部作業) IBM等の認可済み実量子hardwareでraw provider countsと複数seed成功率を取得

主な証跡: [量子検証](16_quantum_validation.md)、
[`quantum_evidence.json`](../validation/quantum_evidence.json)

## フェーズ7: 製品運用

- [x] Schema v2とv1→v2移行を実装
- [x] versionと任意SHA-256を持つ外部材料・断面CSVカタログを実装
- [x] 入力例とsmall / medium / large生成器を用意
- [x] CLI終了コード、例外、機械可読診断を統一
- [x] Python APIと依存追加不要のWSGI REST APIを実装
- [x] manifest、checkpoint、再開を実装
- [x] JSON、CSV、HTML、依存追加不要のPDFレポートを実装
- [x] HTMLへ利用率表示と最適化履歴SVGを統合
- [x] 支配荷重組合せ、部材、制約を保存
- [x] 入出力、レポート、依存関係、release archiveのchecksumを保存・検証
- [x] seed、solver設定、環境、Git commitを保存
- [x] 過去2実行の専用比較レポートを実装

主な証跡: `tests/io/test_phase67_completion.py`、`io/catalog_loader.py`、
`io/pdf_report.py`、`api.py`

## フェーズ8: 独立検証とリリース

- [x] 基準解、Exact/MILP、レガシー回帰、OpenSees照合を自動証跡化
- [x] 外部レビュー記録の資格・独立性・commit・署名参照を検証するgateを実装
- [x] property-based相当、異常入力、回帰試験を実装
- [x] 依存、既知脆弱性、ライセンス、固定版、SBOM相当inventoryを監査
- [x] configurable endurance runner、30秒large耐久、決定性・時間・メモリgateを実装
- [x] Linux、macOS、WindowsとPython 3.10〜3.13のCIを実装
- [x] pilot入力・出力checksum、独立照合、署名、異なるIDを検証するgateを実装
- [x] 適用範囲、免責、既知制限、RC表記を文書化
- [x] 自動RCゲート13項目を通過
- [ ] (外部作業) 実装担当者以外の有資格構造技術者による署名レビュー
- [ ] (外部作業) 匿名化した異なる実案件pilotを最低2件実施し外部署名
- [ ] (外部作業) 対象運用環境で必要時間のsoak試験を実施
- [ ] (外部作業) 責任者による正式な `v1.0` 承認

外部記録をリポジトリ内で `approved` に書き換えてもgateは信頼しない。主な証跡:
[`release_gate_evidence.json`](../validation/release_gate_evidence.json)、
[`endurance_evidence.json`](../validation/endurance_evidence.json)

## 非線形・弾塑性トラス拡張

- [x] 材料model protocolと履歴stateを実装
- [x] 1軸弾完全塑性・二直線等方硬化を実装
- [x] 降伏判定、後退Euler return-mapping、一貫接線を実装
- [x] 荷重増分、Newton-Raphson、line search、収束判定、自動cutback／growthを実装
- [x] 荷重・変位制御、除荷・再載荷を実装
- [x] 塑性ひずみ、相当塑性ひずみ、残留変位、散逸energyを保存
- [x] corotationalトラスと幾何剛性を実装
- [x] 初回降伏、逐次降伏、極限荷重、全体接線特異／非収束の順序付きplastic-mechanism評価を実装
- [x] 弾塑性FEMを共通subproblemに使う部材有無・離散断面最適化を実装
- [x] 非収束、機構、過大塑性、過大残留変位を有限違反度の非実行可能候補として処理
- [x] 単一棒、除荷、硬化、二部材系、剛体回転、逐次降伏、collapseを検証
- [x] Exact／Greedy E2E例で同一最小質量設計と不適合候補棄却を確認
- [x] OpenSees Steel01と二直線材料応答を独立照合
- [ ] (外部作業) 実構造実験または校正済み高忠実度modelとのValidation

「進行性崩壊」は、この版では逐次降伏による塑性機構形成を指す。破断、局部座屈、
接触、要素削除、動的collapse、arc-length、フレームplastic hingeは対象外であり、
それらを実装済みとは扱わない。

主な実装: `nonlinear_material.py`、`nonlinear_truss.py`、
[`nonlinear_truss_optimization.py`](../examples/nonlinear_truss_optimization.py)、
[非線形解析章](19_nonlinear_elastoplastic_truss.md)

## 最終判定

- [x] フェーズ0〜8のリポジトリ内で実装・自動化可能な全項目
- [x] 線形・モーダル・弾塑性トラス解析
- [x] Exact / Greedy / MILP / SA / QUBO / QAOAの共通比較基盤
- [x] 弾塑性部材配置・離散断面最適化
- [x] 自動RC受入証跡
- [ ] (外部作業) 資格・実案件・実機・署名を伴う正式な実用承認

したがって、ソフトウェア実装計画は完了しているが、現在の正しいrelease stageは
`release-candidate` であり、外部承認なしに正式な実用版を名乗ることはできない。
