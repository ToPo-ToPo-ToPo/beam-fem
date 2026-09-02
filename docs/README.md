# beam-fem ドキュメント

梁モデルの FEM 解析と構造最適化ライブラリ `beamfem` の理論・数式・実装をまとめる。

数式は GitHub の Markdown 数式（`$...$`, `$$...$$`）で記述している。

## 目次

| 章 | 内容 | 主な実装 |
|---|---|---|
| [1. FEM 理論](01_fem_theory.md) | 3D Timoshenko 梁要素の剛性行列・座標変換 | `element3d.py` |
| [2. ソルバ](02_solver.md) | 全体行列の組み立て・境界条件・静的線形解析 | `assembly.py`, `solver.py` |
| [3. 内力・応力](03_forces_stress.md) | 断面力の回収・応力評価・符号規約 | `forces.py` |
| [4. 断面諸量](04_sections.md) | 各断面形状の A, I, J, 縁端距離の式 | `material.py` |
| [5. サイジング最適化](05_sizing_optimization.md) | 質量最小化・解析的感度（直接法）・MMA・離散・本数 | `optimize/sizing.py`, `mma.py`, `discrete.py` |
| [6. トポロジー最適化](06_topology_optimization.md) | Ground Structure 法（トラスLP） | `optimize/topology.py` |
| [7. 可視化](07_visualization.md) | 変形図・断面力図・構造形態図 | `viz.py` |
| [8. コード構成](08_code_structure.md) | モジュール構成と API 一覧 | パッケージ全体 |
| [9. 構造生成と面荷重](09_builders_loads.md) | グリラージュ生成・面分布荷重の節点化 | `builders.py` |
| [10. 三角形シェル要素](10_shell_element.md) | 三角形フラットシェル（CST 膜 + DKT 板曲げ・薄板）・ドリリング・梁連成・応力回収 | `shell3d.py`, `shell.py` |
| [11. 四角形シェル MITC4](11_quad_shell_mitc4.md) | 四角形フラットシェル（Q4 膜 + MITC4 板曲げ・厚板〜薄板）・タイング・応力回収・最適化連成 | `shell_mitc4.py`, `shell.py` |
| [12. 実用向け離散構造・量子最適化](12_discrete_quantum_optimization.md) | 共通FEM、離散バックエンド、局所QUBO、監査・検証方針 | `optimize/`, `validation/` |
| [13. 入力・監査・ベンチマーク](13_input_audit_benchmarks.md) | JSON/YAML、監査、比較ケース | `io/`, `benchmarks/` |
| [14. 製品範囲・設計照査](14_product_scope_and_code_checks.md) | Schema v2、照査trace、AISC preview、外部review gate | `validation/`, `io/` |
| [15. 製品要件・リリース条件](15_product_requirements.md) | 適用範囲、精度・性能基準、RC判定 | `validation/` |
| [16. 量子バックエンド検証](16_quantum_validation.md) | Statevector/Aer/実機差替え、証拠要件 | `optimize/backends/qaoa.py` |
| [17. リリース運用](17_release_operations.md) | 成果物保持、完全性検証、ロールバック | `io/release_archive.py` |
| [18. 開発計画の実施状況](18_development_plan_status.md) | フェーズ0〜8の完了・一部完了・未実装、弾塑性ロードマップ | 実装・検証証跡全体 |

## 例題

| 例 | 内容 |
|---|---|
| [portal_frame_2d](../examples/portal_frame_2d.py) | 2D 門型ラーメン（水平荷重・変形図） |
| [plate_shell](../examples/plate_shell.py) | 単純支持正方形板のシェル解析（三角形 CST+DKT・Navier 解と比較） |
| [plate_mitc4](../examples/plate_mitc4.py) | 単純支持板の MITC4 四角形シェル解析（薄板/厚板・せん断変形と収束） |
| [ribbed_plate_quad_sizing](../examples/ribbed_plate_quad_sizing.py) | 四角形シェル板＋オフセットリブのサイジング最適化＋四角形シェル応力回収 |
| [ribbed_plate_quad_discrete](../examples/ribbed_plate_quad_discrete.py) | 同・離散サイジング（規格リブ寸法・総当たり大域最適と貪欲法の一致） |
| [circular_plate_shell](../examples/circular_plate_shell.py) | 円形膜（円板）に等分布荷重（周辺固定/単純支持・Kirchhoff 円板解と比較） |
| [ribbed_plate_shell](../examples/ribbed_plate_shell.py) | 円形膜のリブ補強（シェル板＋梁リブの連成・剛体オフセットで T 形合成効果を比較） |
| [ribbed_plate_shell_sizing](../examples/ribbed_plate_shell_sizing.py) | リブ補強板のサイジング最適化（シェル板＋オフセットリブ・たわみ制約下の質量最小化） |
| [ribbed_plate_shell_discrete](../examples/ribbed_plate_shell_discrete.py) | リブ補強板の離散サイジング（規格リブ寸法カタログから選定・貪欲法） |
| [ribbed_plate_layout_study](../examples/ribbed_plate_layout_study.py) | 境界条件×制約で最適リブ配置がどう変わるか（単純支持/固定・たわみ/応力の比較） |
| [beam_forces](../examples/beam_forces.py) | 単純梁の内力・応力と項目指定出力 |
| [spider_web_3d](../examples/spider_web_3d.py) | 円形蜘蛛の巣フレーム・面外グリラージュ |
| [sizing_optimization](../examples/sizing_optimization.py) | 先細り片持ち梁の質量最小化（連続サイジング） |
| [topology_ground_structure](../examples/topology_ground_structure.py) | 片持ちトラスの部材配置（Ground Structure 法） |
| [ribbed_plate_optimization](../examples/ribbed_plate_optimization.py) | 円形膜のリブ補強（連続サイジング） |
| [ribbed_plate_discrete](../examples/ribbed_plate_discrete.py) | 同・規格リブ径から選定（離散サイジング） |
| [ribbed_plate_count_optimization](../examples/ribbed_plate_count_optimization.py) | リブ本数の最適化（2段階＋膜スパン制約） |

## 全体像

```
入力（モデル）         解析                    最適化
─────────────        ─────────────           ──────────────────────────────
節点・要素       ─→   静的線形解析       ─→   サイジング最適化（感度+MMA）
材料・断面            K u = F                  ├ 連続 / 離散（規格カタログ）
境界条件・荷重   ─→   内力・応力の回収   ─→   └ リブ本数（2段階）
グリラージュ生成      可視化                   トポロジー最適化（Ground Structure LP）
面荷重の節点化                                 ＊応力・たわみ制約下の質量/体積最小化
```

## 実装済み機能の総覧

- **解析**：3D Timoshenko 梁と2D/3D軸力トラスの静的線形解析、混在、内力・応力の回収
- **シェル**：三角形（CST+DKT, 薄板）・四角形 MITC4（Q4+Mindlin, 厚板〜薄板）フラットシェル（各節点6自由度・梁と混在可）
- **断面**：矩形・円・パイプ・箱型・I 形・自由断面
- **可視化**：変形図・断面力図・構造形態図・トラス配置図
- **サイジング最適化**：連続（解析的感度＋MMA）／離散（カタログ）／リブ本数（2段階）。
  応力制約・たわみ制約に対応
- **トポロジー最適化**：Ground Structure 法（トラス最小体積 LP・大域最適）
- **構造生成・荷重**：円形リブ・グリラージュ生成、面分布荷重の等価節点化
- **出力**：`workspace/` への図・CSV 保存

## 設計の指針

- **要素は 3D Timoshenko 梁**（各節点6自由度）を基本とし、2D 面内骨組も同じデータ構造で扱う。
  同じ節点6自由度の**三角形フラットシェル**（CST 膜 + DKT 板曲げ）を混在できる。
- **疎行列**による組み立てと直接法ソルバで数千〜数万要素規模に対応。ソルバは差し替え可能。
- 最適化の**感度は解析的**（サイジングは直接法、トポロジーは LP の双対性）。
  数値は有限差分・解析解・別ソルバ（SLSQP/linprog）で検証済み。
- 出力（図・CSV）は `workspace/` フォルダに集約。

## 単位系

Python APIは単位変換を行わないため一貫単位を利用者が担保する。バージョン付き
JSON/YAML最適化入力ではSI（N, m, Pa, kg, kg/m³）を必須とし、入力検証で拒否する。

## 座標系と自由度の規約（全章共通）

節点自由度の並びは

$$\mathbf{u}_{\text{node}} = [\,u_x,\; u_y,\; u_z,\; \theta_x,\; \theta_y,\; \theta_z\,]$$

断面の主軸を局所 $y, z$ 軸にとる。曲げは

- 局所 $z$ 軸まわり（面内 $x$-$y$ 曲げ）… $I_z$
- 局所 $y$ 軸まわり（面外 $x$-$z$ 曲げ）… $I_y$

で表す。`UX, UY, UZ, RX, RY, RZ = 0..5` は局所自由度インデックス。
