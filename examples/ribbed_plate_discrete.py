"""円形膜のリブ補強・離散サイジング最適化（規格サイズ版）。

[`ribbed_plate_optimization.py`](ribbed_plate_optimization.py) の離散値版。リブ断面を
連続スケールではなく、**規格リブ径のカタログ**から選んで質量を最小化する。実用では
任意寸法ではなく規格材から選ぶため、こちらが現実的。

解法は貪欲局所探索（連続最適解を丸めて開始→実行可能化→近傍探索）。設計変数が少なければ
総当たり（solve_discrete_exhaustive）で大域最適も得られる。
"""

import numpy as np

from beamfem import (
    Material, Section, Model, solve_static, recover_forces,
    radial_grillage, lump_pressure, UZ,
)
from beamfem.optimize import (
    SizingProblem, DesignVar, DispLimit, ScaledSection,
    solve_discrete_greedy, minimize_mass,
)

# --- パラメータ（連続版と同じ） ---
R, N_RADIAL, N_RINGS = 2.0, 8, 4
q = 3000.0
sigma_allow = 150e6
defl_limit = 0.012

STEEL = Material(E=200e9, nu=0.3, rho=7850.0)
BASE_D = 0.030
base_rib = Section.circle(d=BASE_D)

# 規格リブ径のカタログ [mm] → スケール係数（= d / 基準径）
catalog_d_mm = [20, 25, 30, 40, 50, 60]
catalog = [d / 1000.0 / BASE_D for d in catalog_d_mm]
print("利用可能な規格リブ径 [mm]:", catalog_d_mm)

# --- モデル ---
m = Model()
g = radial_grillage(m, STEEL, base_rib, R, N_RADIAL, N_RINGS)
for j in range(N_RADIAL):
    m.fix(g.ring_nodes[N_RINGS][j])
total_load = lump_pressure(m, g.triangles, q)

dvs = []
for b in range(N_RINGS):
    dvs.append(DesignVar(ScaledSection(base_rib), g.radial_bands[b],
                         x0=1.5, xmin=min(catalog), xmax=max(catalog), name=f"radial{b}"))
for k in range(1, N_RINGS + 1):
    dvs.append(DesignVar(ScaledSection(base_rib), g.rings[k],
                         x0=1.5, xmin=min(catalog), xmax=max(catalog), name=f"ring{k}"))

disp_limits = [DispLimit(g.center, UZ, defl_limit)]
for k in range(1, N_RINGS):
    disp_limits.append(DispLimit(g.ring_nodes[k][0], UZ, defl_limit))

prob = SizingProblem(m, dvs, sigma_allow=sigma_allow, disp_limits=disp_limits)

print("=== 円形膜のリブ補強・離散サイジング最適化 ===")
print(f"設計変数={prob.n_var}, カタログ={len(catalog)}サイズ "
      f"(全組合せ={len(catalog)**prob.n_var:,} 通り)")

# --- 連続最適化（参考・下界） ---
cont = minimize_mass(prob, maxiter=120, move=0.15, tol=1e-6)
print(f"\n連続最適（参考下界）: 質量={cont.mass:.2f} kg")

# --- 離散最適化（貪欲局所探索） ---
res = solve_discrete_greedy(prob, catalog, warm_start="continuous")
print(f"\n離散最適: 質量={res.mass:.2f} kg  実行可能={res.feasible}  関数評価={res.n_eval} 回")
print(f"  （連続比 +{(res.mass/cont.mass-1)*100:.1f}%。離散は連続を下回れない）")

res_opt = solve_static(m)
print(f"  中心たわみ = {res_opt.node_disp(g.center)[UZ]*1e3:.2f} mm (許容 {defl_limit*1e3:.0f} mm)")
print(f"  最大制約 = {res.constraints.max():+.3e}")

print("\nグループ別 選定リブ径:")
for k, dv in enumerate(dvs):
    d_mm = res.x[k] * BASE_D * 1000.0
    print(f"  {dv.name:9s}: φ{d_mm:.0f} mm")

# --- 図示（workspace/） ---
try:
    from beamfem import viz

    fig1, _ = viz.plot_member_sizes(
        m, prob.element_values(res.x, kind="area") * 1e4, label="rib area [cm^2]")
    p1 = viz.savefig("ribbed_plate_discrete_form.png", dpi=130)
    print(f"\n離散リブ配置を {p1} に保存しました。")
except ImportError:
    print("\n(matplotlib 未導入のため図はスキップ)")

forces = recover_forces(m, res_opt)
csv = forces.to_csv("ribbed_plate_discrete_forces.csv", items=["My", "Mz", "sigma_max"], at="max")
print(f"リブ内力を {csv} に保存しました。")
