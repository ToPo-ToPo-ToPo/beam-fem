"""長方形リブ補強板のサイジング最適化（四角形 MITC4 シェル板＋オフセット梁リブ）。

構造格子に素直に張れる四角形 MITC4 シェルで長方形板をモデル化し、格子線に沿って
剛体オフセット付きのリブ（梁）を配置する。上面一様圧のもと中央たわみ制約下で
リブ質量を最小化し、最適後に **四角形シェルの応力（膜応力・曲げモーメント）** も
回収する。

要点:
- 四角形シェルが `SizingProblem` に固定剛性として組み込まれる（板厚は設計変数外）。
- リブは x 方向群・y 方向群でグループ化（長方形なので両者は異なる太さに最適化）。
- 合成効果（EA·e²）のため面内 UX,UY は内部自由、外周で面内保持。

実行::

    .venv/bin/python examples/ribbed_plate_quad_sizing.py
"""

import numpy as np

from beamfem import (
    Material, Section, Model, solve_static, recover_shell_forces, UX, UY, UZ, RZ,
)
from beamfem.optimize import (
    SizingProblem, DesignVar, DispLimit, ScaledSection, minimize_mass,
)

# --- パラメータ ---
LX, LY = 1.5, 1.0      # 板の寸法 [m]（長方形）
t = 0.008              # 板厚 [m]
q = 6_000.0            # 上面圧力 [Pa]
NX, NY = 8, 6          # 四角形分割数
sigma_allow = 200e6
defl_limit = 0.0025    # 中央たわみ許容 2.5 mm（制約がアクティブになる水準）

STEEL = Material(E=200e9, nu=0.3, rho=7850.0, name="steel")
RIB_B, RIB_H = 0.006, 0.030
BASE_RIB = Section.rectangle(b=RIB_B, h=RIB_H, name="rib")
RIB_VREF = np.array([0.0, 0.0, 1.0])
E_OFFSET = t / 2 + RIB_H / 2


def build():
    m = Model()
    ids = {}
    for j in range(NY + 1):
        for i in range(NX + 1):
            ids[(i, j)] = m.add_node(LX * i / NX, LY * j / NY, 0.0)
    # 四角形シェル（反時計まわり）
    for j in range(NY):
        for i in range(NX):
            m.add_quad_shell(ids[(i, j)], ids[(i + 1, j)],
                             ids[(i + 1, j + 1)], ids[(i, j + 1)], STEEL, t)

    off = [0.0, 0.0, -E_OFFSET]
    x_ribs, y_ribs = [], []
    # x 方向リブ（内側の水平格子線）
    for j in range(1, NY):
        for i in range(NX):
            x_ribs.append(m.add_element(ids[(i, j)], ids[(i + 1, j)], STEEL,
                                        BASE_RIB, vref=RIB_VREF, offset=off))
    # y 方向リブ（内側の鉛直格子線）
    for i in range(1, NX):
        for j in range(NY):
            y_ribs.append(m.add_element(ids[(i, j)], ids[(i, j + 1)], STEEL,
                                        BASE_RIB, vref=RIB_VREF, offset=off))

    # 境界条件: ドリリング全拘束、外周は単純支持＋面内保持
    for n in ids.values():
        m.fix(n, [RZ])
    for i in range(NX + 1):
        for jj in (0, NY):
            m.fix(ids[(i, jj)], [UX, UY, UZ])
    for j in range(NY + 1):
        for ii in (0, NX):
            m.fix(ids[(ii, j)], [UX, UY, UZ])

    # 一様圧 → 各四角形の 1/4 を節点へ
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
    DesignVar(ScaledSection(BASE_RIB), x_ribs, x0=1.2, xmin=0.3, xmax=3.0, name="x-ribs"),
    DesignVar(ScaledSection(BASE_RIB), y_ribs, x0=1.2, xmin=0.3, xmax=3.0, name="y-ribs"),
]
prob = SizingProblem(m, dvs, sigma_allow=sigma_allow,
                     disp_limits=[DispLimit(center, UZ, defl_limit)])


def rib_mass():
    return sum(e.mat.rho * e.sec.A * m.element_length(e) for e in m.elements)


print("=== 長方形リブ補強板のサイジング（四角形 MITC4 シェル＋オフセットリブ）===")
print(f"板 {LX}×{LY} m, 板厚 {t*1e3:.0f} mm, 四角形 {len(m.quad_shells)} 枚, リブ {len(m.elements)} 本")
print(f"設計変数: x方向リブ群({len(x_ribs)}本)・y方向リブ群({len(y_ribs)}本), 偏心 e={E_OFFSET*1e3:.0f} mm")

prob.evaluate(prob.x0())
mass0 = rib_mass()
res0 = solve_static(m)
print(f"\n初期: リブ質量={mass0:.2f} kg, 中央たわみ={res0.node_disp(center)[UZ]*1e3:.3f} mm")

res = minimize_mass(prob, maxiter=120, move=0.15, tol=1e-6)
res_opt = solve_static(m)
print(f"\n最適リブ質量 = {res.mass:.2f} kg （初期比 {(1-res.mass/mass0)*100:.1f}% 削減, 収束={res.converged}）")
print(f"中央たわみ = {res_opt.node_disp(center)[UZ]*1e3:.3f} mm (許容 {defl_limit*1e3:.1f} mm)")
for k, dv in enumerate(dvs):
    print(f"  {dv.name}: s={res.x[k]:.3f}  (h≈{RIB_H*1e3*res.x[k]:.1f} mm)")

# --- 四角形シェルの応力回収 ---
sf = recover_shell_forces(m, res_opt)
sbx = [abs(s.get("sbx")) for s in sf.quad_shells]
sx = [abs(s.get("sx")) for s in sf.quad_shells]
imax = int(np.argmax(sbx))
print(f"\n四角形シェル応力（最大曲げ縁端応力の要素 Q{imax}）:")
sf.print_table(items=["sx", "sy", "Mx", "My", "sbx"], which="quad")  # 全要素
print(f"\n板の最大曲げ縁端応力 σbx_max = {max(sbx)/1e6:.1f} MPa, 最大膜応力 |σx|_max = {max(sx)/1e6:.2f} MPa")

# --- 図示 ---
try:
    from beamfem import viz

    fig1, _ = viz.plot_member_sizes(m, prob.element_values(res.x, kind="area") * 1e4,
                                    label="rib area [cm^2]")
    p1 = viz.savefig("ribbed_plate_quad_sizing_form.png", dpi=130)
    fig2, _ = viz.plot_deformed(m, res_opt, scale="auto")
    p2 = viz.savefig("ribbed_plate_quad_sizing_deformed.png", dpi=130)
    print(f"\nリブ断面分布を {p1}、変形図を {p2} に保存しました。")
except ImportError:
    print("\n(matplotlib 未導入のため図はスキップ)")
