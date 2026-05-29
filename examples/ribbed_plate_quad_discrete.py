"""長方形リブ補強板の離散サイジング（四角形 MITC4 シェル板＋オフセット梁リブ）。

[`ribbed_plate_quad_sizing.py`](ribbed_plate_quad_sizing.py) の離散値版。四角形
MITC4 シェルで張った長方形板を格子線リブ（剛体オフセット付き梁）で補強し、リブ
断面を **規格リブ寸法のカタログ**から選んで中央たわみ制約下の質量を最小化する。

設計変数は x 方向リブ群・y 方向リブ群の 2 つなので、総当たり（大域最適）も実行
でき、貪欲局所探索と一致することを確認できる。`solve_discrete_*` は連続版と同じ
`SizingProblem` を関数評価器に使い、四角形シェルの固定剛性とオフセットの合成効果
を含めて評価する。

実行::

    .venv/bin/python examples/ribbed_plate_quad_discrete.py
"""

import numpy as np

from beamfem import (
    Material, Section, Model, solve_static, recover_shell_forces, UX, UY, UZ, RZ,
)
from beamfem.optimize import (
    SizingProblem, DesignVar, DispLimit, ScaledSection,
    minimize_mass, solve_discrete_greedy, solve_discrete_exhaustive,
)

# --- パラメータ（連続版と同じ）---
LX, LY = 1.5, 1.0
t = 0.008
q = 6_000.0
NX, NY = 8, 6
sigma_allow = 200e6
defl_limit = 0.0025

STEEL = Material(E=200e9, nu=0.3, rho=7850.0, name="steel")
RIB_B0, RIB_H0 = 0.006, 0.030
BASE_RIB = Section.rectangle(b=RIB_B0, h=RIB_H0, name="rib")
RIB_VREF = np.array([0.0, 0.0, 1.0])
E_OFFSET = t / 2 + RIB_H0 / 2

# 規格リブ高さ [mm] → スケール係数（相似拡大, = h / 基準高さ）
catalog_h_mm = [9, 12, 15, 18, 24, 30]
catalog = [h / 1000.0 / RIB_H0 for h in catalog_h_mm]
print("利用可能な規格リブ高さ [mm]:", catalog_h_mm)


def build():
    m = Model()
    ids = {}
    for j in range(NY + 1):
        for i in range(NX + 1):
            ids[(i, j)] = m.add_node(LX * i / NX, LY * j / NY, 0.0)
    for j in range(NY):
        for i in range(NX):
            m.add_quad_shell(ids[(i, j)], ids[(i + 1, j)],
                             ids[(i + 1, j + 1)], ids[(i, j + 1)], STEEL, t)
    off = [0.0, 0.0, -E_OFFSET]
    x_ribs, y_ribs = [], []
    for j in range(1, NY):
        for i in range(NX):
            x_ribs.append(m.add_element(ids[(i, j)], ids[(i + 1, j)], STEEL,
                                        BASE_RIB, vref=RIB_VREF, offset=off))
    for i in range(1, NX):
        for j in range(NY):
            y_ribs.append(m.add_element(ids[(i, j)], ids[(i, j + 1)], STEEL,
                                        BASE_RIB, vref=RIB_VREF, offset=off))
    for n in ids.values():
        m.fix(n, [RZ])
    for i in range(NX + 1):
        for jj in (0, NY):
            m.fix(ids[(i, jj)], [UX, UY, UZ])
    for j in range(NY + 1):
        for ii in (0, NX):
            m.fix(ids[(ii, j)], [UX, UY, UZ])
    for s in m.quad_shells:
        p = [m.nodes[s.n1], m.nodes[s.n2], m.nodes[s.n3], m.nodes[s.n4]]
        area = 0.5 * abs((p[2][0] - p[0][0]) * (p[3][1] - p[1][1])
                         - (p[3][0] - p[1][0]) * (p[2][1] - p[0][1]))
        for nd in (s.n1, s.n2, s.n3, s.n4):
            m.add_load(nd, UZ, -q * area / 4.0)
    return m, ids, x_ribs, y_ribs


m, ids, x_ribs, y_ribs = build()
center = ids[(NX // 2, NY // 2)]
dvs = [
    DesignVar(ScaledSection(BASE_RIB), x_ribs, x0=1.0,
              xmin=min(catalog), xmax=max(catalog), name="x-ribs"),
    DesignVar(ScaledSection(BASE_RIB), y_ribs, x0=1.0,
              xmin=min(catalog), xmax=max(catalog), name="y-ribs"),
]
prob = SizingProblem(m, dvs, sigma_allow=sigma_allow,
                     disp_limits=[DispLimit(center, UZ, defl_limit)])

print("=== 長方形リブ補強板の離散サイジング（四角形 MITC4 シェル＋オフセットリブ）===")
print(f"板 {LX}×{LY} m, 板厚 {t*1e3:.0f} mm, 四角形 {len(m.quad_shells)} 枚, リブ {len(m.elements)} 本")
print(f"設計変数={prob.n_var}, カタログ={len(catalog)}サイズ "
      f"(全組合せ={len(catalog)**prob.n_var} 通り), 中央たわみ≤{defl_limit*1e3:.1f} mm")

# --- 連続最適（参考下界）---
cont = minimize_mass(prob, maxiter=120, move=0.15, tol=1e-6)
print(f"\n連続最適（参考下界）: リブ質量={cont.mass:.2f} kg")

# --- 離散最適: 総当たり（大域最適）と貪欲を比較 ---
ex = solve_discrete_exhaustive(prob, catalog)
gr = solve_discrete_greedy(prob, catalog)
print(f"\n総当たり（大域最適）: リブ質量={ex.mass:.2f} kg  indices={ex.indices}  評価={ex.n_eval}回")
print(f"貪欲局所探索        : リブ質量={gr.mass:.2f} kg  indices={gr.indices}  評価={gr.n_eval}回")
print(f"  → 一致={gr.indices == ex.indices}（連続比 +{(ex.mass/cont.mass-1)*100:.1f}%）")

res_opt = solve_static(m)  # モデルは離散最適点に確定済み
print(f"  中央たわみ = {res_opt.node_disp(center)[UZ]*1e3:.3f} mm (許容 {defl_limit*1e3:.1f} mm)")
print(f"  最大制約   = {ex.constraints.max():+.3e}")
print("\nグループ別 選定リブ高さ [mm]:")
for k, dv in enumerate(dvs):
    print(f"  {dv.name}: h={ex.x[k]*RIB_H0*1e3:.0f} mm (×{ex.x[k]:.3f})")

# --- 四角形シェルの応力 ---
sf = recover_shell_forces(m, res_opt)
sbx = max(abs(s.get("sbx")) for s in sf.quad_shells)
print(f"\n板の最大曲げ縁端応力 σbx_max = {sbx/1e6:.1f} MPa")

# --- 図示 ---
try:
    from beamfem import viz

    fig1, _ = viz.plot_member_sizes(m, prob.element_values(ex.x, kind="area") * 1e4,
                                    label="rib area [cm^2]")
    p1 = viz.savefig("ribbed_plate_quad_discrete_form.png", dpi=130)
    print(f"\n離散リブ配置を {p1} に保存しました。")
except ImportError:
    print("\n(matplotlib 未導入のため図はスキップ)")
