"""断面サイジング最適化の例：先細り片持ち梁の質量最小化。

片持ち梁を複数セグメントに分け、各セグメントを設計グループとして断面スケールを
独立に変える。応力（各要素）と先端たわみの制約下で総質量を最小化する。
固定端側ほど曲げが大きいので、最適解は固定端で太く先端で細い「先細り」になる。

解析的感度 + MMA。出力（収束履歴・最適断面）は workspace/ に保存。
"""

import numpy as np

from beamfem import Material, Section, Model, solve_static, recover_forces, UY
from beamfem.optimize import SizingProblem, DesignVar, DispLimit, ScaledSection, minimize_mass

STEEL = Material(E=200e9, nu=0.3, rho=7850.0)
base = Section.box(b=0.1, h=0.2, t=0.008)  # 箱型を基準断面に

# --- 片持ち梁: 長さ 4m を 8 要素、2要素ずつ4グループ ---
L_total, n_elem = 4.0, 8
m = Model()
nodes = [m.add_node(L_total * i / n_elem, 0, 0) for i in range(n_elem + 1)]
for i in range(n_elem):
    m.add_element(nodes[i], nodes[i + 1], STEEL, base)
m.fix(nodes[0])
m.fix_to_plane_xy()

# 全節点に下向き荷重（自重を模した分布荷重相当）＋先端集中荷重
for nd in nodes[1:]:
    m.add_load(nd, UY, -2000.0)
m.add_load(nodes[-1], UY, -5000.0)

groups = [[0, 1], [2, 3], [4, 5], [6, 7]]
dvs = [
    DesignVar(ScaledSection(base), g, x0=1.5, xmin=0.3, xmax=4.0, name=f"seg{k}")
    for k, g in enumerate(groups)
]
prob = SizingProblem(
    m,
    dvs,
    sigma_allow=160e6,                      # 許容応力 160 MPa
    disp_limits=[DispLimit(nodes[-1], UY, 0.02)],  # 先端たわみ 20mm 以下
)

print("=== 断面サイジング最適化（先細り片持ち梁） ===")
print(f"要素数={n_elem}, 設計変数={prob.n_var}, 制約数={len(prob.evaluate(prob.x0())[2])}")

# 初期質量
f0_init = prob.evaluate(prob.x0())[0]
res = minimize_mass(prob, maxiter=100, move=0.2, tol=1e-6, verbose=True)

print(f"\n初期質量 = {f0_init:.2f} kg")
print(f"最適質量 = {res.mass:.2f} kg  （{(1-res.mass/f0_init)*100:.1f}% 削減）")
print(f"収束={res.converged}, 反復={res.iterations}")
print(f"最大制約値 = {res.constraints.max():+.3e}  (≤0 で全制約満足)")

print("\nグループ別 最適スケール（固定端→先端）:")
for k, dv in enumerate(dvs):
    sec = res.sections[k]
    print(f"  seg{k}: s={res.x[k]:.4f}  A={sec.A*1e4:.2f} cm^2  Iz={sec.Iz*1e8:.2f} cm^4")

# --- 出力（workspace/） ---
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from beamfem import viz

    # 収束履歴
    hist = np.array([(h[0], h[1]) for h in res.history])
    fig, ax = plt.subplots()
    ax.plot(hist[:, 0], hist[:, 1], "o-")
    ax.set_xlabel("iteration")
    ax.set_ylabel("mass [kg]")
    ax.set_title("MMA convergence")
    ax.grid(True, alpha=0.3)
    p1 = viz.savefig("sizing_history.png", dpi=120)

    # 最適形状の変形図
    res_static = solve_static(m)
    fig2, _ = viz.plot_deformed(m, res_static, scale="auto")
    p2 = viz.savefig("sizing_deformed.png", dpi=120)

    # 構造形態の図示（部材の線幅・色＝最適断面積）。先細りが一目で分かる
    fig3, _ = viz.plot_member_sizes(
        m, prob.element_values(res.x, kind="area") * 1e4,
        label="cross-section area [cm^2]",
    )
    p3 = viz.savefig("sizing_form.png", dpi=120)
    print(f"\n収束履歴を {p1}、変形図を {p2}、構造形態を {p3} に保存しました。")
except ImportError:
    print("\n(matplotlib 未導入のため図はスキップ)")

# 最適断面の内力・応力（項目指定でCSV）
forces = recover_forces(m, solve_static(m))
csv = forces.to_csv("sizing_forces.csv", items=["Mz", "Vy", "sigma_max"], at="max")
print(f"最適断面の内力を {csv} に保存しました。")
