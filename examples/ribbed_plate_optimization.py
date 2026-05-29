"""円形膜を下から補強するリブ構造の最適化。

上から一様な圧力 q [N/m^2] を受ける円形膜を、下面の放射＋同心リング状のリブ
（グリラージュ）で補強する。膜自体は曲げ剛性を持たないものとし、圧力は分担面積で
リブの節点へ等価節点荷重として伝える。リブは 3D Timoshenko 梁の面外曲げ＋ねじりで
荷重を支える。

サイジング最適化で、外周固定・中心/中間のたわみ制約・各リブの応力制約のもとに
リブ総質量を最小化する。設計変数はリブのスケール係数（放射バンド別・リング別）。
設計変数の下限を小さくしておくと、不要なリブが細り、効く配置の傾向も見える。
"""

import numpy as np

from beamfem import (
    Material, Section, Model, solve_static, recover_forces,
    radial_grillage, lump_pressure, UZ,
)
from beamfem.optimize import (
    SizingProblem, DesignVar, DispLimit, ScaledSection, minimize_mass,
)

# --- パラメータ ---
R = 2.0            # 円形膜の半径 [m]
N_RADIAL = 8       # 放射リブ本数
N_RINGS = 4        # 同心リング数
q = 3000.0         # 上面圧力 [N/m^2]
sigma_allow = 150e6
defl_limit = 0.012  # たわみ許容 12 mm

STEEL = Material(E=200e9, nu=0.3, rho=7850.0)
base_rib = Section.circle(d=0.03)  # 基準リブ断面（円形: Iy=Iz で向き非依存）

# --- グリラージュとモデル ---
m = Model()
g = radial_grillage(m, STEEL, base_rib, R, N_RADIAL, N_RINGS)
for j in range(N_RADIAL):
    m.fix(g.ring_nodes[N_RINGS][j])     # 外周を固定支持
total_load = lump_pressure(m, g.triangles, q)  # 圧力 -> 等価節点荷重（-z）

print("=== 円形膜のリブ補強・サイジング最適化 ===")
print(f"半径={R} m, 放射={N_RADIAL}, リング={N_RINGS}, 節点={m.n_nodes}, リブ要素={len(m.elements)}")
print(f"圧力 q={q} N/m^2, 総載荷={total_load:.1f} N")

# --- 設計変数: 放射バンド別・リング別のスケール ---
dvs = []
for b in range(N_RINGS):
    dvs.append(DesignVar(ScaledSection(base_rib), g.radial_bands[b],
                         x0=1.5, xmin=0.2, xmax=4.0, name=f"radial{b}"))
for k in range(1, N_RINGS + 1):
    dvs.append(DesignVar(ScaledSection(base_rib), g.rings[k],
                         x0=1.5, xmin=0.2, xmax=4.0, name=f"ring{k}"))

# --- たわみ制約: 中心と各内側リングの代表点（対称性より1点ずつ） ---
disp_limits = [DispLimit(g.center, UZ, defl_limit)]
for k in range(1, N_RINGS):
    disp_limits.append(DispLimit(g.ring_nodes[k][0], UZ, defl_limit))

prob = SizingProblem(m, dvs, sigma_allow=sigma_allow, disp_limits=disp_limits)

# 初期状態
f0_init = prob.evaluate(prob.x0())[0]
res0 = solve_static(m)
print(f"\n初期: 質量={f0_init:.2f} kg, 中心たわみ={res0.node_disp(g.center)[UZ]*1e3:.2f} mm")

# --- 最適化 ---
res = minimize_mass(prob, maxiter=120, move=0.15, tol=1e-6, verbose=False)
print(f"\n最適質量 = {res.mass:.2f} kg  （初期比 {(1-res.mass/f0_init)*100:.1f}% 削減）")
print(f"収束={res.converged}, 反復={res.iterations}, 最大制約={res.constraints.max():+.3e}")

res_opt = solve_static(m)
print(f"最適時 中心たわみ = {res_opt.node_disp(g.center)[UZ]*1e3:.2f} mm (許容 {defl_limit*1e3:.0f} mm)")

print("\nグループ別 最適スケール:")
for k, dv in enumerate(dvs):
    print(f"  {dv.name:9s}: s={res.x[k]:.3f}  (d≈{30*res.x[k]:.1f} mm 相当)")

# --- 図示（workspace/） ---
try:
    from beamfem import viz

    fig1, _ = viz.plot_member_sizes(
        m, prob.element_values(res.x, kind="area") * 1e4,
        label="rib area [cm^2]")
    p1 = viz.savefig("ribbed_plate_form.png", dpi=130)
    fig2, _ = viz.plot_deformed(m, res_opt, scale="auto")
    p2 = viz.savefig("ribbed_plate_deformed.png", dpi=130)
    print(f"\nリブ配置を {p1}、変形図を {p2} に保存しました。")
except ImportError:
    print("\n(matplotlib 未導入のため図はスキップ)")

# 最適リブの内力（項目指定）
forces = recover_forces(m, res_opt)
csv = forces.to_csv("ribbed_plate_forces.csv", items=["My", "Mz", "T", "sigma_max"], at="max")
print(f"リブ内力を {csv} に保存しました。")
