# 4. 断面諸量

実装：[`src/beamfem/material.py`](../src/beamfem/material.py)

## 4.1 断面が持つ量

| 量 | 記号 | 用途 |
|---|---|---|
| 断面積 | $A$ | 軸剛性・質量・軸応力 |
| 断面二次モーメント | $I_y, I_z$ | 曲げ剛性・曲げ応力 |
| ねじり定数 | $J$ | ねじり剛性 |
| せん断補正係数 | $k_y, k_z$ | Timoshenko せん断（$A_s=kA$） |
| 縁端距離 | $c_y, c_z$ | 曲げ応力 $\sigma=Mc/I$ |

規約：局所 $y$ が高さ方向、局所 $z$ が幅方向。$I_z$ が $z$ 軸まわり（面内曲げ）、
$I_y$ が $y$ 軸まわり（面外曲げ）。$I_z$ が大きい向きを強軸にとる。

材料は `Material(E, nu, rho)` で、$G = E/[2(1+\nu)]$ を導く。

## 4.2 矩形 `Section.rectangle(b, h)`

幅 $b$（$z$ 方向）・高さ $h$（$y$ 方向）：

$$A = bh,\quad I_z = \frac{bh^3}{12},\quad I_y = \frac{hb^3}{12},\quad
  c_y = \frac{h}{2},\quad c_z = \frac{b}{2},\quad k_y=k_z=\tfrac{5}{6}$$

ねじり定数は矩形の近似式（$a=\max(b,h)/2,\ c=\min(b,h)/2$）：

$$J = a c^3\!\left(\frac{16}{3} - 3.36\frac{c}{a}\Bigl(1-\frac{c^4}{12a^4}\Bigr)\right)$$

## 4.3 中実円 `Section.circle(d)`

直径 $d$：

$$A=\frac{\pi d^2}{4},\quad I_y=I_z=\frac{\pi d^4}{64},\quad J=\frac{\pi d^4}{32}=2I,\quad
  c_y=c_z=\frac{d}{2},\quad k=0.9$$

## 4.4 中空円（パイプ）`Section.pipe(d, t)`

外径 $d$、肉厚 $t$、内径 $d_i=d-2t$：

$$A=\frac{\pi}{4}(d^2-d_i^2),\quad I=\frac{\pi}{64}(d^4-d_i^4),\quad J=2I,\quad
  c=\frac{d}{2},\quad k=0.5\ (\text{薄肉近似})$$

## 4.5 箱型（角形鋼管）`Section.box(b, h, t)`

外形 $b\times h$、肉厚 $t$（一様）、内形 $b_i=b-2t,\ h_i=h-2t$：

$$A = bh - b_i h_i,\quad
  I_z = \frac{bh^3 - b_i h_i^3}{12},\quad
  I_y = \frac{hb^3 - h_i b_i^3}{12}$$

ねじりは**閉断面の Bredt 式**（中心線で囲む面積 $A_m=(b-t)(h-t)$）：

$$J = \frac{2\,t\,A_m^2}{(b-t)+(h-t)}$$

せん断有効断面の近似（鉛直＝2 枚のウェブ、水平＝2 枚のフランジ）：

$$k_y = \frac{2th}{A},\qquad k_z = \frac{2tb}{A}$$

## 4.6 I 形（H 形鋼）`Section.i_section(h, bf, tf, tw)`

全せい $h$、フランジ幅 $b_f$、フランジ厚 $t_f$、ウェブ厚 $t_w$、ウェブ高さ $h_w=h-2t_f$：

$$A = 2 b_f t_f + h_w t_w$$

$$I_z = \frac{b_f h^3 - (b_f-t_w)h_w^3}{12}\quad(\text{強軸}),\qquad
  I_y = 2\cdot\frac{t_f b_f^3}{12} + \frac{h_w t_w^3}{12}\quad(\text{弱軸})$$

ねじりは**開断面の St.Venant 近似**（そりは無視）：

$$J = \frac{1}{3}\bigl(2 b_f t_f^3 + h_w t_w^3\bigr)$$

せん断（鉛直＝ウェブ、水平＝フランジが主に負担）：

$$k_y = \frac{t_w h}{A},\qquad k_z = \frac{5}{6}\cdot\frac{2 b_f t_f}{A}$$

縁端距離 $c_y=h/2,\ c_z=b_f/2$。

## 4.7 完全自由な断面

```python
Section(A=..., Iy=..., Iz=..., J=..., cy=..., cz=..., ky=..., kz=...)
```

任意の $A, I, J$ を直接指定できる。せん断係数 $k_y,k_z$ は各コンストラクタで妥当な
近似値を自動設定するが、キーワードで上書き可能。

## 4.8 検証

- 各断面の $A, I_y, I_z$ を矩形分解による独立計算と照合（[`tests/test_sections.py`](../tests/test_sections.py)）
- I 形で $I_z>I_y$（強軸／弱軸）を確認
- **各断面を用いた片持ち梁の先端たわみが Timoshenko 解析解と厳密一致**（ソルバ統合の確認）

> せん断補正係数は工学的近似であり、せん断が支配的な短スパンでは厳密値（Cowper 等）との
> 差が出る。必要に応じて `ky, kz` を上書きする。
