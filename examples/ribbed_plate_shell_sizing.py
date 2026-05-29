"""円形膜のリブ補強板のサイジング最適化（シェル板＋オフセット梁リブの連成）。

円板を三角形フラットシェル（CST 膜 + DKT 板曲げ）で、補強リブを剛体オフセット
付きの 3D Timoshenko 梁でモデル化し（T 形断面の合成効果 EA·e² を含む）、上面
一様圧力のもとで中央たわみ制約・リブ応力制約を満たしつつ **リブ総質量を最小化**
する。

設計変数は放射バンド別・リング別のリブ断面スケール係数。解析的感度（直接法）＋
MMA で解く。板（シェル）は固定剛性として連成し、設計変数ではない。

注意:
- リブの偏心 e=t/2+h/2 は最適化中は固定（リブを板下面に付けた取り付け深さを
  保持し、断面はその図心位置まわりで相似拡大する近似）。
- 合成効果のため面内自由度 UX,UY は内部で自由にし、外周で面内を保持する。

実行::

    .venv/bin/python examples/ribbed_plate_shell_sizing.py
"""

import numpy as np

from beamfem import Material, Section, Model, solve_static, lump_pressure
from beamfem import UX, UY, UZ, RZ
from beamfem.optimize import (
    SizingProblem, DesignVar, DispLimit, ScaledSection, minimize_mass,
)

# --- パラメータ ---
R = 1.0            # 円板半径 [m]
t = 0.010          # 板厚 [m]
q = 8_000.0        # 上面圧力 [Pa]
N_RADIAL = 12      # 放射リブ本数
N_RINGS = 6        # 半径方向分割数
sigma_allow = 200e6
defl_limit = 0.006  # 中央たわみ許容 6 mm

STEEL = Material(E=200e9, nu=0.3, rho=7850.0, name="steel")
RIB_B, RIB_H = 0.006, 0.030
BASE_RIB = Section.rectangle(b=RIB_B, h=RIB_H, name="rib")
RIB_VREF = np.array([0.0, 0.0, 1.0])   # 局所 y を鉛直に（強軸 Iz で鉛直曲げ）
E_OFFSET = t / 2 + RIB_H / 2           # 偏心 = 20 mm


def build():
    """円板（シェル）＋オフセットリブ（梁）を組み、リブのグループ分けを返す。"""
    m = Model()
    angles = [2.0 * np.pi * j / N_RADIAL for j in range(N_RADIAL)]
    center = m.add_node(0.0, 0.0, 0.0)
    ring_nodes = [[center] * N_RADIAL]
    for k in range(1, N_RINGS + 1):
        r = R * k / N_RINGS
        ring_nodes.append([m.add_node(r * np.cos(a), r * np.sin(a), 0.0) for a in angles])

    # 三角形分割 → シェル
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
    # 放射リブ（バンド b は半径レベル b→b+1）
    radial_bands = [[] for _ in range(N_RINGS)]
    for j in range(N_RADIAL):
        radial_bands[0].append(
            m.add_element(center, ring_nodes[1][j], STEEL, BASE_RIB, vref=RIB_VREF, offset=off))
        for k in range(1, N_RINGS):
            radial_bands[k].append(
                m.add_element(ring_nodes[k][j], ring_nodes[k + 1][j], STEEL, BASE_RIB,
                              vref=RIB_VREF, offset=off))
    # 同心リング
    rings = [[] for _ in range(N_RINGS + 1)]
    for k in range(1, N_RINGS + 1):
        for j in range(N_RADIAL):
            jn = (j + 1) % N_RADIAL
            rings[k].append(
                m.add_element(ring_nodes[k][j], ring_nodes[k][jn], STEEL, BASE_RIB,
                              vref=RIB_VREF, offset=off))

    # 境界条件: ドリリング全拘束、外周は単純支持＋面内保持
    for i in range(m.n_nodes):
        m.fix(i, [RZ])
    for nd in ring_nodes[N_RINGS]:
        m.fix(nd, [UX, UY, UZ])

    total = lump_pressure(m, triangles, q, dof=UZ, sign=-1.0)
    return m, center, ring_nodes, radial_bands, rings, total


m, center, ring_nodes, radial_bands, rings, total = build()

# --- 設計変数: 放射バンド別・リング別のスケール ---
dvs = []
for b in range(N_RINGS):
    dvs.append(DesignVar(ScaledSection(BASE_RIB), radial_bands[b],
                         x0=1.2, xmin=0.3, xmax=3.0, name=f"radial{b}"))
for k in range(1, N_RINGS + 1):
    dvs.append(DesignVar(ScaledSection(BASE_RIB), rings[k],
                         x0=1.2, xmin=0.3, xmax=3.0, name=f"ring{k}"))

# --- たわみ制約: 中心＋内側リング代表点（対称性より1点ずつ）---
disp_limits = [DispLimit(center, UZ, defl_limit)]
for k in range(1, N_RINGS):
    disp_limits.append(DispLimit(ring_nodes[k][0], UZ, defl_limit))

prob = SizingProblem(m, dvs, sigma_allow=sigma_allow, disp_limits=disp_limits)


def rib_mass():
    return sum(e.mat.rho * e.sec.A * m.element_length(e) for e in m.elements)


print("=== リブ補強板のサイジング最適化（シェル板＋オフセット梁リブ）===")
print(f"円板: R={R} m, 板厚 t={t*1e3:.0f} mm, 放射 {N_RADIAL} × 半径 {N_RINGS}")
print(f"節点 {m.n_nodes}, シェル {len(m.shells)} 枚, リブ梁 {len(m.elements)} 本, 偏心 e={E_OFFSET*1e3:.0f} mm")
print(f"圧力 q={q/1e3:.1f} kPa, 総荷重 {total/1e3:.3f} kN")
print(f"制約: σ≤{sigma_allow/1e6:.0f} MPa, 中央たわみ≤{defl_limit*1e3:.0f} mm")

# 初期状態
prob.evaluate(prob.x0())
mass0 = rib_mass()
res0 = solve_static(m)
print(f"\n初期: リブ質量={mass0:.2f} kg, 中央たわみ={res0.node_disp(center)[UZ]*1e3:.3f} mm")

# --- 最適化 ---
res = minimize_mass(prob, maxiter=120, move=0.15, tol=1e-6, verbose=False)
res_opt = solve_static(m)
print(f"\n最適リブ質量 = {res.mass:.2f} kg  （初期比 {(1 - res.mass / mass0)*100:.1f}% 削減）")
print(f"収束={res.converged}, 反復={res.iterations}, 最大制約={res.constraints.max():+.3e}")
print(f"最適時 中央たわみ = {res_opt.node_disp(center)[UZ]*1e3:.3f} mm (許容 {defl_limit*1e3:.0f} mm)")

print("\nグループ別 最適スケール（d≈基準6×30mmの相似倍率）:")
for k, dv in enumerate(dvs):
    print(f"  {dv.name:9s}: s={res.x[k]:.3f}")

# --- 図示（リブ断面分布・変形図）---
try:
    from beamfem import viz

    fig1, _ = viz.plot_member_sizes(
        m, prob.element_values(res.x, kind="area") * 1e4, label="rib area [cm^2]")
    p1 = viz.savefig("ribbed_plate_sizing_form.png", dpi=130)
    fig2, _ = viz.plot_deformed(m, res_opt, scale="auto")
    p2 = viz.savefig("ribbed_plate_sizing_deformed.png", dpi=130)
    print(f"\nリブ断面分布を {p1}、変形図を {p2} に保存しました。")
except ImportError:
    print('\n(matplotlib 未導入のため図はスキップ)')
