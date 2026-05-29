"""四辺単純支持の正方形板を三角形フラットシェルで解く例。

一様圧力 q を受ける単純支持正方形板（辺 a, 板厚 t）。中央たわみを
Navier 級数解 w = 0.00406 q a^4 / D（D=曲げ剛性, ν=0.3）と比較する。

実行::

    .venv/bin/python examples/plate_shell.py
"""

import numpy as np

from beamfem import Material, Model, solve_static, recover_shell_forces, UX, UY, UZ, RZ
from beamfem.shell3d import shell_local_frame

STEEL = Material(E=200e9, nu=0.3, name="steel")

a = 1.0       # 辺長 [m]
t = 0.01      # 板厚 [m]
q = 10_000.0  # 圧力 [Pa]（下向き）
n = 12        # 1 辺の分割数

# --- メッシュ生成（各四角セルを2三角形に分割）---
m = Model()
ids = {}
for j in range(n + 1):
    for i in range(n + 1):
        ids[(i, j)] = m.add_node(a * i / n, a * j / n, 0.0)

for j in range(n):
    for i in range(n):
        n00 = ids[(i, j)]
        n10 = ids[(i + 1, j)]
        n11 = ids[(i + 1, j + 1)]
        n01 = ids[(i, j + 1)]
        m.add_shell(n00, n10, n11, STEEL, t)
        m.add_shell(n00, n11, n01, STEEL, t)

# --- 境界条件 ---
# 面内自由度（膜・ドリリング）は荷重が無いので全節点で拘束（純曲げ問題）
for nid in ids.values():
    m.fix(nid, [UX, UY, RZ])
# 周辺単純支持: 縁のたわみ w(=UZ) を固定（回転は自由）
for i in range(n + 1):
    for jj in (0, n):
        m.fix(ids[(i, jj)], [UZ])
        m.fix(ids[(jj, i)], [UZ])

# --- 一様圧力を等価節点力へ（三角形面積/3 を各節点へ, 下向き）---
total = 0.0
for s in m.shells:
    _, _, _, area = shell_local_frame(m.nodes[s.n1], m.nodes[s.n2], m.nodes[s.n3])
    f = -q * area / 3.0
    for nd in (s.n1, s.n2, s.n3):
        m.add_load(nd, UZ, f)
        total += f

res = solve_static(m)

# --- 中央たわみと解析解の比較 ---
D = STEEL.E * t**3 / (12.0 * (1.0 - STEEL.nu**2))
w_exact = -0.00406 * q * a**4 / D
w_center = res.node_disp(ids[(n // 2, n // 2)])[UZ]

print("=== 単純支持正方形板（フラットシェル CST+DKT）===")
print(f"メッシュ: {n}x{n} セル, 三角形 {len(m.shells)} 要素, 節点 {m.n_nodes}")
print(f"総荷重   : {total/1e3:.3f} kN (理論 {-q*a*a/1e3:.3f} kN)")
print(f"中央たわみ: {w_center*1e3:8.4f} mm")
print(f"Navier 解 : {w_exact*1e3:8.4f} mm")
print(f"誤差      : {abs(w_center - w_exact)/abs(w_exact)*100:5.2f} %")

# --- 中央要素の曲げモーメント ---
sf = recover_shell_forces(m, res)
mid = sf.shells[len(sf.shells) // 2]
print(f"\n中央付近要素の曲げモーメント Mx={mid.get('Mx'):.1f}, "
      f"My={mid.get('My'):.1f} N·m/m")

# --- 変形図 ---
try:
    from beamfem import viz

    fig, ax = viz.plot_deformed(m, res, scale="auto")
    path = viz.savefig("plate_shell_deformed.png", dpi=120)
    print(f"\n変形図を {path} に保存しました。")
except ImportError:
    print('\n(matplotlib 未導入のため図はスキップ。pip install -e ".[viz]")')
