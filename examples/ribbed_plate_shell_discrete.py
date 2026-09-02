"""リブ補強板の離散サイジング最適化（規格リブ寸法版・シェル板＋オフセット梁）。

[`ribbed_plate_shell_sizing.py`](ribbed_plate_shell_sizing.py) の離散値版。板を
フラットシェル、リブを剛体オフセット付き梁で連成させたモデルで、リブ断面を
連続スケールではなく **規格リブ寸法のカタログ**から選んで質量を最小化する。

解法は貪欲局所探索（連続最適解を丸めて開始→実行可能化→近傍探索）。同じ
`SizingProblem` を関数評価器に使い、シェルの固定剛性とオフセット梁の合成効果
（EA·e²）を含めて評価する。

実行::

    .venv/bin/python examples/ribbed_plate_shell_discrete.py
"""

import numpy as np

from beamfem import Material, Section, Model, solve_static, lump_pressure
from beamfem import UX, UY, UZ, RZ
from beamfem.optimize import (
    SizingProblem, DesignVar, DispLimit, ScaledSection,
    solve_discrete_greedy, minimize_mass,
)

# --- パラメータ（連続版と同じ）---
R = 1.0
t = 0.010
q = 8_000.0
N_RADIAL = 12
N_RINGS = 6
sigma_allow = 200e6
defl_limit = 0.006  # 中央たわみ許容 6 mm

STEEL = Material(E=200e9, nu=0.3, rho=7850.0, name="steel")
RIB_B0, RIB_H0 = 0.006, 0.030          # 基準リブ（スケール 1.0）
BASE_RIB = Section.rectangle(b=RIB_B0, h=RIB_H0, name="rib")
RIB_VREF = np.array([0.0, 0.0, 1.0])
E_OFFSET = t / 2 + RIB_H0 / 2          # 偏心 = 20 mm（最適化中は固定）

# 規格リブ寸法のカタログ（高さ h [mm]）→ スケール係数（相似拡大, = h / 基準高さ）
catalog_h_mm = [15, 20, 25, 30, 40, 50, 60]
catalog = [h / 1000.0 / RIB_H0 for h in catalog_h_mm]
print("利用可能な規格リブ高さ [mm]:", catalog_h_mm, "（幅は高さの 1/5）")


def build():
    """円板（シェル）＋オフセットリブ（梁）を組み、リブのグループ分けを返す。"""
    m = Model()
    angles = [2.0 * np.pi * j / N_RADIAL for j in range(N_RADIAL)]
    center = m.add_node(0.0, 0.0, 0.0)
    ring_nodes = [[center] * N_RADIAL]
    for k in range(1, N_RINGS + 1):
        r = R * k / N_RINGS
        ring_nodes.append([m.add_node(r * np.cos(a), r * np.sin(a), 0.0) for a in angles])

    triangles = []
    for j in range(N_RADIAL):
        jn = (j + 1) % N_RADIAL
        triangles.append((center, ring_nodes[1][j], ring_nodes[1][jn]))
    for k in range(1, N_RINGS):
        for j in range(N_RADIAL):
            jn = (j + 1) % N_RADIAL
            a, b = ring_nodes[k][j], ring_nodes[k][jn]
            c, d = ring_nodes[k + 1][jn], ring_nodes[k + 1][j]
            triangles += [(a, b, c), (a, c, d)]
    for (i, j, k) in triangles:
        m.add_shell(i, j, k, STEEL, t)

    off = [0.0, 0.0, -E_OFFSET]
    radial_bands = [[] for _ in range(N_RINGS)]
    for j in range(N_RADIAL):
        radial_bands[0].append(
            m.add_element(center, ring_nodes[1][j], STEEL, BASE_RIB, vref=RIB_VREF, offset=off))
        for k in range(1, N_RINGS):
            radial_bands[k].append(
                m.add_element(ring_nodes[k][j], ring_nodes[k + 1][j], STEEL, BASE_RIB,
                              vref=RIB_VREF, offset=off))
    rings = [[] for _ in range(N_RINGS + 1)]
    for k in range(1, N_RINGS + 1):
        for j in range(N_RADIAL):
            jn = (j + 1) % N_RADIAL
            rings[k].append(
                m.add_element(ring_nodes[k][j], ring_nodes[k][jn], STEEL, BASE_RIB,
                              vref=RIB_VREF, offset=off))

    for i in range(m.n_nodes):
        m.fix(i, [RZ])
    for nd in ring_nodes[N_RINGS]:
        m.fix(nd, [UX, UY, UZ])

    total = lump_pressure(m, triangles, q, dof=UZ, sign=-1.0)
    return m, center, ring_nodes, radial_bands, rings, total


m, center, ring_nodes, radial_bands, rings, total = build()

dvs = []
for b in range(N_RINGS):
    dvs.append(DesignVar(ScaledSection(BASE_RIB), radial_bands[b],
                         x0=1.2, xmin=min(catalog), xmax=max(catalog), name=f"radial{b}"))
for k in range(1, N_RINGS + 1):
    dvs.append(DesignVar(ScaledSection(BASE_RIB), rings[k],
                         x0=1.2, xmin=min(catalog), xmax=max(catalog), name=f"ring{k}"))

disp_limits = [DispLimit(center, UZ, defl_limit)]
for k in range(1, N_RINGS):
    disp_limits.append(DispLimit(ring_nodes[k][0], UZ, defl_limit))

prob = SizingProblem(m, dvs, sigma_allow=sigma_allow, disp_limits=disp_limits)


def rib_mass():
    return sum(e.mat.rho * e.sec.A * m.element_length(e) for e in m.elements)


print("=== リブ補強板の離散サイジング最適化（シェル板＋オフセット梁リブ）===")
print(f"円板: R={R} m, 板厚 t={t*1e3:.0f} mm, 放射 {N_RADIAL} × 半径 {N_RINGS}, 偏心 e={E_OFFSET*1e3:.0f} mm")
print(f"設計変数={prob.n_var}, カタログ={len(catalog)}サイズ "
      f"(全組合せ={len(catalog)**prob.n_var:,} 通り → 総当たりは非現実的、貪欲法で解く)")
print(f"制約: σ≤{sigma_allow/1e6:.0f} MPa, 中央たわみ≤{defl_limit*1e3:.0f} mm")

# --- 連続最適化（参考・下界）---
cont = minimize_mass(prob, maxiter=120, move=0.15, tol=1e-6)
print(f"\n連続最適（参考下界）: リブ質量={cont.mass:.2f} kg")

# --- 離散最適化（貪欲局所探索）---
res = solve_discrete_greedy(prob, catalog, warm_start="continuous")
print(f"\n離散最適: リブ質量={res.mass:.2f} kg  実行可能={res.feasible}  関数評価={res.n_eval} 回")
print(f"  （連続比 +{(res.mass/cont.mass - 1)*100:.1f}%。離散は連続を下回れない）")

res_opt = solve_static(m)
print(f"  中央たわみ = {res_opt.node_disp(center)[UZ]*1e3:.3f} mm (許容 {defl_limit*1e3:.0f} mm)")
print(f"  最大制約   = {res.constraints.max():+.3e}")

print("\nグループ別 選定リブ高さ [mm]:")
for k, dv in enumerate(dvs):
    h_mm = res.x[k] * RIB_H0 * 1000.0
    print(f"  {dv.name:9s}: h={h_mm:.0f} mm (×{res.x[k]:.3f})")

# --- 図示（workspace/）---
try:
    from beamfem import viz

    fig1, _ = viz.plot_member_sizes(
        m, prob.element_values(res.x, kind="area") * 1e4, label="rib area [cm^2]")
    p1 = viz.savefig("ribbed_plate_discrete_shell_form.png", dpi=130)
    fig2, _ = viz.plot_deformed(m, res_opt, scale="auto")
    p2 = viz.savefig("ribbed_plate_discrete_shell_deformed.png", dpi=130)
    print(f"\n離散リブ配置を {p1}、変形図を {p2} に保存しました。")
except ImportError:
    print("\n(matplotlib 未導入のため図はスキップ)")
