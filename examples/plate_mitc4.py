"""4節点 MITC4 フラットシェルによる単純支持正方形板（薄板・厚板）。

MITC4 は Mindlin-Reissner 板（横せん断変形込み）なので、薄板では Kirchhoff
（Navier）解に収束し（せん断ロックなし）、厚板ではせん断変形ぶんたわみが増える。
三角形 DKT（薄板専用, examples/plate_shell.py）に対し、四角形メッシュと厚板対応が
利点。ここでは薄板 (a/t=100) と厚板 (a/t=10) を解いて確認する。

実行::

    .venv/bin/python examples/plate_mitc4.py
"""

import numpy as np

from beamfem import Material, Model, solve_static, UX, UY, UZ, RZ

STEEL = Material(E=200e9, nu=0.3, name="steel")
a = 1.0          # 辺長 [m]
q = 1.0e4        # 圧力 [Pa]


def solve_ss_plate(n, t):
    """n×n の MITC4 四角形シェルで単純支持正方形板を解き、中央たわみを返す。"""
    m = Model()
    ids = {}
    for j in range(n + 1):
        for i in range(n + 1):
            ids[(i, j)] = m.add_node(a * i / n, a * j / n, 0.0)
    for j in range(n):
        for i in range(n):
            m.add_quad_shell(ids[(i, j)], ids[(i + 1, j)],
                             ids[(i + 1, j + 1)], ids[(i, j + 1)], STEEL, t)
    # 面内・ドリリングは拘束（純曲げ）、外周は単純支持 w=0
    for nid in ids.values():
        m.fix(nid, [UX, UY, RZ])
    for i in range(n + 1):
        for jj in (0, n):
            m.fix(ids[(i, jj)], [UZ])
            m.fix(ids[(jj, i)], [UZ])
    # 一様圧 → 各四角形の 1/4 を節点へ
    for s in m.quad_shells:
        p = [m.nodes[s.n1], m.nodes[s.n2], m.nodes[s.n3], m.nodes[s.n4]]
        area = 0.5 * abs((p[2][0] - p[0][0]) * (p[3][1] - p[1][1])
                         - (p[3][0] - p[1][0]) * (p[2][1] - p[0][1]))
        for nd in (s.n1, s.n2, s.n3, s.n4):
            m.add_load(nd, UZ, -q * area / 4.0)
    res = solve_static(m)
    return res.node_disp(ids[(n // 2, n // 2)])[UZ], m, res, ids


print("=== 単純支持正方形板（MITC4 四角形フラットシェル）===")
for label, ratio in [("薄板  a/t=100", 100), ("厚板  a/t=10 ", 10)]:
    t = a / ratio
    D = STEEL.E * t**3 / (12.0 * (1.0 - STEEL.nu**2))
    w_kirch = -0.00406 * q * a**4 / D
    print(f"\n--- {label} (t={t*1e3:.0f} mm) ---")
    print(f"Kirchhoff(薄板)解: {w_kirch*1e3:10.4f} mm")
    for n in (4, 8, 16):
        wc, *_ = solve_ss_plate(n, t)
        print(f"  {n:2d}×{n:<2d}: w_c={wc*1e3:10.4f} mm  (w/Kirchhoff={wc/w_kirch:.3f})")
    note = "薄板極限へ収束（ロックなし）" if ratio == 100 else "せん断変形でたわみ増（厚板効果）"
    print(f"  → {note}")

# --- 変形図（厚板）---
t = a / 10
_, m, res, ids = solve_ss_plate(12, t)
try:
    from beamfem import viz

    fig, ax = viz.plot_deformed(m, res, scale="auto")
    p = viz.savefig("plate_mitc4_deformed.png", dpi=120)
    print(f"\n厚板の変形図を {p} に保存しました。")
except ImportError:
    print('\n(matplotlib 未導入のため図はスキップ)')
