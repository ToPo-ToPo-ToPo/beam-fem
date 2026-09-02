"""円形膜（円板パネル）に上から等分布荷重が作用する例題。

半径 R・板厚 t の円板を三角形フラットシェル（CST 膜 + DKT 板曲げ）で分割し、
上面に一様圧力 q（下向き）を与える。周辺固定（クランプ）として解き、中央
たわみを Kirchhoff 円板の解析解と比較する::

    クランプ円板（周辺固定）: w_center = q R^4 / (64 D)
    単純支持円板           : w_center = (5+ν)/(1+ν) · q R^4 / (64 D)
    （D = E t^3 / 12(1-ν^2) は曲げ剛性）

メッシュは中心節点＋同心リングの放射分割（builders.radial_grillage と同じ
配置）で、各三角形にシェル要素を1枚張る。面圧は builders.lump_pressure で
等価節点荷重に変換する。

実行::

    .venv/bin/python examples/circular_plate_shell.py
"""

import numpy as np

from beamfem import (
    Material,
    Model,
    solve_static,
    recover_shell_forces,
    lump_pressure,
    UX,
    UY,
    UZ,
    RX,
    RY,
    RZ,
)

STEEL = Material(E=200e9, nu=0.3, name="steel")

R = 1.0        # 半径 [m]
t = 0.01       # 板厚 [m]
q = 20_000.0   # 等分布荷重（圧力）[Pa]
n_radial = 24  # 周方向分割数
n_rings = 8    # 半径方向分割数
clamped = True  # True: 周辺固定 / False: 単純支持


def build_disk(m: Model, mat, thickness, R, n_radial, n_rings):
    """円板を中心＋同心リングで放射分割し、各三角形にシェルを張る。

    戻り値 (center, ring_nodes, triangles)。
    """
    angles = [2.0 * np.pi * j / n_radial for j in range(n_radial)]
    center = m.add_node(0.0, 0.0, 0.0)
    ring_nodes = [[center] * n_radial]  # k=0 は中心（重複参照）
    for k in range(1, n_rings + 1):
        r = R * k / n_rings
        ring_nodes.append(
            [m.add_node(r * np.cos(a), r * np.sin(a), 0.0) for a in angles]
        )

    triangles = []
    # 内側ファン（中心まわり）
    for j in range(n_radial):
        jn = (j + 1) % n_radial
        triangles.append((center, ring_nodes[1][j], ring_nodes[1][jn]))
    # 環状の四角形を2三角形に分割
    for k in range(1, n_rings):
        for j in range(n_radial):
            jn = (j + 1) % n_radial
            a, b = ring_nodes[k][j], ring_nodes[k][jn]
            c, d = ring_nodes[k + 1][jn], ring_nodes[k + 1][j]
            triangles.append((a, b, c))
            triangles.append((a, c, d))

    for (i, j, k) in triangles:
        m.add_shell(i, j, k, mat, thickness)
    return center, ring_nodes, triangles


# --- モデル構築 ---
m = Model()
center, ring_nodes, triangles = build_disk(m, STEEL, t, R, n_radial, n_rings)

# 面内自由度（膜・ドリリング）は荷重が無いので全節点で拘束（純曲げ問題）
for i in range(m.n_nodes):
    m.fix(i, [UX, UY, RZ])

# 周辺の境界条件（外周リング）
rim = ring_nodes[n_rings]
for nd in rim:
    if clamped:
        m.fix(nd, [UZ, RX, RY])  # 固定: たわみ・回転=0
    else:
        m.fix(nd, [UZ])          # 単純支持: たわみ=0（回転自由）

# --- 等分布荷重（上から＝下向き -z）を等価節点荷重へ ---
total = lump_pressure(m, triangles, q, dof=UZ, sign=-1.0)

res = solve_static(m)

# --- 中央たわみと解析解の比較 ---
D = STEEL.E * t**3 / (12.0 * (1.0 - STEEL.nu**2))
if clamped:
    w_exact = -q * R**4 / (64.0 * D)
    bc = "周辺固定（クランプ）"
else:
    w_exact = -(5.0 + STEEL.nu) / (1.0 + STEEL.nu) * q * R**4 / (64.0 * D)
    bc = "単純支持"
w_center = res.node_disp(center)[UZ]

print("=== 円形膜（円板）に等分布荷重（フラットシェル CST+DKT）===")
print(f"境界条件 : {bc}")
print(f"メッシュ : 放射 {n_radial} × 半径 {n_rings}, "
      f"三角形 {len(triangles)} 要素, 節点 {m.n_nodes}")
print(f"圧力     : q = {q/1e3:.1f} kPa")
print(f"総荷重   : {total/1e3:.3f} kN (理論 π R² q = {np.pi*R*R*q/1e3:.3f} kN)")
print(f"中央たわみ: {w_center*1e3:8.4f} mm")
print(f"解析解   : {w_exact*1e3:8.4f} mm")
print(f"誤差     : {abs(w_center - w_exact)/abs(w_exact)*100:5.2f} %")

# --- 中央付近の曲げモーメント ---
sf = recover_shell_forces(m, res)
mr = sf.shells[0]  # 中心まわりのファン要素
print(f"\n中心付近要素の曲げモーメント Mx={mr.get('Mx'):.1f}, "
      f"My={mr.get('My'):.1f} N·m/m")

# --- 変形図 ---
try:
    from beamfem import viz

    fig, ax = viz.plot_deformed(m, res, scale="auto")
    path = viz.savefig("circular_plate_shell_deformed.png", dpi=120)
    print(f"\n変形図を {path} に保存しました。")
except ImportError:
    print('\n(matplotlib 未導入のため図はスキップ。pip install -e ".[viz]")')
