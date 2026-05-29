"""円形膜のリブ補強：リブ本数（放射数・リング数）の最適化。

2 段階最適化：
  外側 … リブ本数の構成 (n_radial, n_rings) を列挙
  内側 … 各構成でサイジング最適化（応力・たわみ制約下の質量最小化）
実行可能な構成の中で最小質量となる本数を選ぶ。

重要な前提：本ライブラリはリブ（梁）のグリラージュのみを扱い、膜がリブ間で局所的に
たわむ挙動はモデル化しない。そのため構造制約（応力・たわみ）だけだと「本数が少ない
ほど軽い」となり最適本数が定まらない。本数の下限は **膜自身の能力＝リブ間スパンの
許容値** で決まるため、ここでは膜の許容パネルスパン `MAX_PANEL_SPAN` を制約として加える。
（円周方向の最大間隔 2πR/n_radial と半径方向の間隔 R/n_rings が許容スパン以下）
"""

import numpy as np

from beamfem import (
    Material, Section, Model, solve_static,
    radial_grillage, lump_pressure, UZ,
)
from beamfem.optimize import (
    SizingProblem, DesignVar, DispLimit, ScaledSection, minimize_mass,
)

# --- パラメータ ---
R = 2.0
q = 3000.0
sigma_allow = 150e6
defl_limit = 0.012
MAX_PANEL_SPAN = 0.9    # 膜の許容パネルスパン [m]（リブ間隔の上限）

STEEL = Material(E=200e9, nu=0.3, rho=7850.0)
BASE_D = 0.03
base_rib = Section.circle(d=BASE_D)

CANDIDATES_RADIAL = [6, 8, 10, 12, 14, 16, 20]
CANDIDATES_RINGS = [2, 3, 4, 5]


def panel_span(n_radial, n_rings):
    """最大パネル寸法：外周の円周方向間隔と半径方向間隔の大きい方。"""
    circumferential = 2.0 * np.pi * R / n_radial   # 外周での隣接スポーク間隔
    radial = R / n_rings                            # リング間の半径方向間隔
    return max(circumferential, radial)


def evaluate_config(n_radial, n_rings):
    """構成 (n_radial, n_rings) のサイジング最適化を実行する。"""
    m = Model()
    g = radial_grillage(m, STEEL, base_rib, R, n_radial, n_rings)
    for j in range(n_radial):
        m.fix(g.ring_nodes[n_rings][j])
    lump_pressure(m, g.triangles, q)

    dvs = [
        DesignVar(ScaledSection(base_rib), g.radial_bands[b], x0=1.0, xmin=0.15, xmax=6.0)
        for b in range(n_rings)
    ]
    dvs += [
        DesignVar(ScaledSection(base_rib), g.rings[k], x0=1.0, xmin=0.15, xmax=6.0)
        for k in range(1, n_rings + 1)
    ]
    dl = [DispLimit(g.center, UZ, defl_limit)]
    dl += [DispLimit(g.ring_nodes[k][0], UZ, defl_limit) for k in range(1, n_rings)]

    prob = SizingProblem(m, dvs, sigma_allow=sigma_allow, disp_limits=dl)
    res = minimize_mass(prob, maxiter=150, move=0.15, tol=1e-6)
    feasible = res.constraints.max() <= 1e-3
    return dict(nr=n_radial, nk=n_rings, mass=res.mass, feasible=feasible,
                model=m, grillage=g, prob=prob, x=res.x)


# --- 列挙 ---
print("=== リブ本数の最適化（放射数 × リング数） ===")
print(f"膜の許容パネルスパン = {MAX_PANEL_SPAN} m\n")
header = f'{"放射":>4} {"リング":>5} {"パネル[m]":>9} {"質量[kg]":>9} {"構造":>5} {"スパン":>6} {"採否":>5}'
print(header)
print("-" * len(header))

configs = []
for nk in CANDIDATES_RINGS:
    for nr in CANDIDATES_RADIAL:
        span = panel_span(nr, nk)
        span_ok = span <= MAX_PANEL_SPAN + 1e-9
        # スパンを満たさない構成は内側最適化を省略（高速化）
        if span_ok:
            c = evaluate_config(nr, nk)
        else:
            c = dict(nr=nr, nk=nk, mass=np.nan, feasible=False, model=None, x=None)
        c["span"] = span
        c["span_ok"] = span_ok
        c["accept"] = span_ok and c["feasible"]
        configs.append(c)
        mass_s = f'{c["mass"]:9.1f}' if not np.isnan(c["mass"]) else f'{"-":>9}'
        struct_s = ("○" if c["feasible"] else "×") if span_ok else "-"  # スパン不可は構造未評価
        print(f'{nr:4d} {nk:5d} {span:9.3f} {mass_s} '
              f'{struct_s:>5} {"○" if span_ok else "×":>6} '
              f'{"◎" if c["accept"] else "":>5}')

# --- 最適構成の選択 ---
acceptable = [c for c in configs if c["accept"]]
best = min(acceptable, key=lambda c: c["mass"])
print(f'\n★ 最適リブ本数: 放射 {best["nr"]} 本 × リング {best["nk"]} 環  '
      f'→ 質量 {best["mass"]:.1f} kg')
print(f'   （膜スパン {best["span"]:.3f} m ≤ 許容 {MAX_PANEL_SPAN} m、構造制約も満足）')

# --- 図示（workspace/） ---
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from beamfem import viz

    m, g, prob = best["model"], best["grillage"], best["prob"]
    fig1, _ = viz.plot_member_sizes(
        m, prob.element_values(best["x"], kind="area") * 1e4, label="rib area [cm^2]")
    fig1.gca().set_title(f'Optimal ribs: {best["nr"]} radial x {best["nk"]} rings, '
                         f'{best["mass"]:.0f} kg')
    p1 = viz.savefig("rib_count_best_form.png", dpi=130)

    # 質量 vs 放射本数（リング数ごと、採用可のみ）
    fig2, ax = plt.subplots()
    for nk in CANDIDATES_RINGS:
        pts = [(c["nr"], c["mass"]) for c in configs if c["nk"] == nk and c["accept"]]
        if pts:
            pts = np.array(pts)
            ax.plot(pts[:, 0], pts[:, 1], "o-", label=f"{nk} rings")
    ax.set_xlabel("number of radial ribs")
    ax.set_ylabel("optimized mass [kg]")
    ax.set_title(f"mass vs rib count (panel span <= {MAX_PANEL_SPAN} m)")
    ax.legend(); ax.grid(True, alpha=0.3)
    p2 = viz.savefig("rib_count_mass_trend.png", dpi=130)
    print(f"\n最適リブ配置を {p1}、質量推移を {p2} に保存しました。")
except ImportError:
    print("\n(matplotlib 未導入のため図はスキップ)")
