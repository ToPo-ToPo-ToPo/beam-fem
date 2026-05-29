# 7. 可視化

実装：[`src/beamfem/viz.py`](../src/beamfem/viz.py)（matplotlib・任意依存）

2D（全節点 $z=0$ かつ面外変位が無視できる）か 3D かを自動判定して描画する。
保存（`savefig`）は相対パスを `workspace/` フォルダに出力する（[8 章](08_code_structure.md)）。

## 7.1 モデル図 `plot_model(model)`

変形前のモデル（節点・要素・支持・荷重）を描く。支持は固定＝四角、ピン＝三角の
マーカー、荷重は矢印で表示する。

## 7.2 変形図 `plot_deformed(model, result)`

要素ごとに**エルミート 3 次形状関数**で曲げの曲率を滑らかに補間して変形後の曲線を描く。
局所たわみは端部のたわみ・回転から

$$v(\xi) = H_1 v_1 + L H_2\,\theta_1 + H_3 v_2 + L H_4\,\theta_2$$

$$H_1 = 1-3\xi^2+2\xi^3,\;\; H_2=\xi-2\xi^2+\xi^3,\;\; H_3=3\xi^2-2\xi^3,\;\; H_4=-\xi^2+\xi^3$$

として求め、局所→全体へ変換し無変形位置に加える。

- `scale="auto"`：変形がモデル寸法の約 5% に見えるよう拡大率を自動調整
- `n`：要素ごとの補間点数

## 7.3 断面力図 `plot_diagram(forces, component)`

指定した 1 成分（`N, Vy, Vz, T, My, Mz`）の断面力図を描く。各部材の材軸に直交方向へ
値をオフセット（`Mz/Vy` は局所 $y$、`My/Vz` は局所 $z$、`N/T` は局所 $y$）し、塗りつぶしと
縦ハッチで表示する。`scale="auto"` でモデル寸法の約 12% に正規化。

## 7.4 構造形態図 `plot_member_sizes(model, values)`

各部材を、与えた値に比例した**線幅・色**で描く。サイジング最適化結果の「どの部材が
太く／細くなったか」を可視化する用途。`values` は要素数と同じ長さの配列（断面積・スケール・
代表寸法）。カラーバー付き。

```python
viz.plot_member_sizes(model, prob.element_values(res.x, kind="area"),
                      label="cross-section area")
```

## 7.5 トラス配置図 `plot_truss(nodes, members, areas)`

トポロジー最適化の結果を、線幅・色を断面積に比例させて描く。既定では断面積が最大の
`rel_tol` 倍以下の部材（≈除去）を描かない。`show_all=True` で全候補部材を薄線で重畳する。

```python
viz.plot_truss(nodes, members, res.areas, show_all=True, label="area")
```

## 7.6 表示・保存

- `viz.show()`：ウィンドウ表示（`plt.show` のラッパ）
- `viz.savefig(path)`：保存。相対パスは `workspace/` 内に保存され、保存先を返す

## 7.7 物理的妥当性の確認例

- 門型ラーメン：両柱が層間変形し、固定端で勾配ゼロの二重曲率
- 単純梁の `Mz` 図：中央最大の三角形分布（$PL/4$）
- 蜘蛛の巣フレーム：軸対称のお椀形に沈下
- サイジング最適化：固定端が太い先細り形態
- トポロジー最適化：Michell 型の片持ちトラス
