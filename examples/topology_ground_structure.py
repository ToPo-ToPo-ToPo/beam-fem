"""トポロジー／部材配置最適化の例：片持ちトラスの Ground Structure 法。

格子状の節点に候補部材を密に張った地盤構造から、左端固定・右端中央に下向き荷重の
条件で最小体積となる部材配置を線形計画で求める。結果は Michell 型の片持ちトラスに
近い形態になる。大域最適（LP）。
"""

import numpy as np

from beamfem.optimize import (
    GroundStructure,
    generate_members,
    grid_nodes,
    solve_min_volume,
)

# --- 格子と地盤構造 ---
nx, ny = 6, 5
lx, ly = 5.0, 4.0
nodes = grid_nodes(nx, ny, lx, ly)  # 節点番号 = iy*nx + ix
members = generate_members(nodes)   # 全結合（共線重複は除去）

# 左端列（ix=0）をピン支持
supports = {iy * nx + 0: [0, 1] for iy in range(ny)}

# 右端列の中央高さに下向き荷重
load_node = (ny // 2) * nx + (nx - 1)
P = 50_000.0
load_cases = [{(load_node, 1): -P}]

gs = GroundStructure(nodes=nodes, members=members, supports=supports, load_cases=load_cases)

sigma = 200e6
res = solve_min_volume(gs, sigma_t=sigma)

print("=== トポロジー最適化（片持ちトラス・Ground Structure 法） ===")
print(f"節点数={len(nodes)}, 候補部材数={len(members)}")
print(f"最適体積 = {res.volume:.6e} m^3")
print(f"有効部材数 = {len(res.active())} / {len(members)}  (LP 大域最適)")

# 比較: 単純な水平片持ち（参考）の体積下界感
print(f"最大部材断面積 = {res.areas.max()*1e4:.2f} cm^2")

# --- 図示（workspace/） ---
try:
    from beamfem import viz

    fig, _ = viz.plot_truss(nodes, members, res.areas, show_all=True,
                            label="cross-section area [m^2]")
    p = viz.savefig("topology_layout.png", dpi=130)
    print(f"\n最適配置を {p} に保存しました（薄線=候補, 太線=採用部材）。")
except ImportError:
    print("\n(matplotlib 未導入のため図はスキップ)")
