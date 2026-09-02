"""リブ補強板：境界条件・制約でリブ最適配置がどう変わるかの比較（教材）。

同じ円形リブ補強板（フラットシェル板＋剛体オフセット梁リブ）を、境界条件と
効かせる制約を変えてサイジング最適化し、「どこのリブが太くなるか」を比べる。

直感では「拘束部（外周）のリブが最も太くなる」と思いがちだが、それが成り立つ
のは**固定端でモーメントが最大になる構造を応力で設計する**ときである。実際には
配置は《境界条件（単純支持／完全固定）×制約（たわみ／応力）》で大きく変わる::

    (A) 単純支持 + たわみ制約 : 中間半径のリングが最太（追加の中間支持として働く）
    (B) 完全固定 + たわみ制約 : 外周側がやや太い（ただし板自体が効くので軽い）
    (C) 完全固定 + 応力制約   : 外周（固定端）に向かって単調に太い ← 直感どおり

板を薄く（t=4mm）してリブが必ず効くようにしている。各ケースのリブ断面積分布を
1枚に並べて図示する。

実行::

    .venv/bin/python examples/ribbed_plate_layout_study.py
"""

import numpy as np

from beamfem import Material, Section, Model, solve_static, lump_pressure
from beamfem import UX, UY, UZ, RX, RY, RZ
from beamfem.optimize import (
    SizingProblem, DesignVar, DispLimit, ScaledSection, minimize_mass,
)

# --- パラメータ ---
R = 1.0
t = 0.004          # 薄板 4mm（リブが必ず効くように）
q = 8_000.0
N_RADIAL = 12
N_RINGS = 6

STEEL = Material(E=200e9, nu=0.3, rho=7850.0, name="steel")
RIB = Section.rectangle(b=0.006, h=0.030, name="rib")
RIB_VREF = np.array([0.0, 0.0, 1.0])
E_OFFSET = t / 2 + 0.030 / 2


def build(clamped: bool):
    """円板（シェル）＋オフセットリブ（梁）を組む。clamped で外周固定／単純支持。"""
    m = Model()
    ang = [2.0 * np.pi * j / N_RADIAL for j in range(N_RADIAL)]
    c = m.add_node(0.0, 0.0, 0.0)
    rn = [[c] * N_RADIAL]
    for k in range(1, N_RINGS + 1):
        r = R * k / N_RINGS
        rn.append([m.add_node(r * np.cos(a), r * np.sin(a), 0.0) for a in ang])

    tris = []
    for j in range(N_RADIAL):
        jn = (j + 1) % N_RADIAL
        tris.append((c, rn[1][j], rn[1][jn]))
    for k in range(1, N_RINGS):
        for j in range(N_RADIAL):
            jn = (j + 1) % N_RADIAL
            a, b = rn[k][j], rn[k][jn]
            cc, d = rn[k + 1][jn], rn[k + 1][j]
            tris += [(a, b, cc), (a, cc, d)]
    for (i, j, k) in tris:
        m.add_shell(i, j, k, STEEL, t)

    off = [0.0, 0.0, -E_OFFSET]
    rb = [[] for _ in range(N_RINGS)]
    for j in range(N_RADIAL):
        rb[0].append(m.add_element(c, rn[1][j], STEEL, RIB, vref=RIB_VREF, offset=off))
        for k in range(1, N_RINGS):
            rb[k].append(m.add_element(rn[k][j], rn[k + 1][j], STEEL, RIB,
                                       vref=RIB_VREF, offset=off))
    rg = [[] for _ in range(N_RINGS + 1)]
    for k in range(1, N_RINGS + 1):
        for j in range(N_RADIAL):
            jn = (j + 1) % N_RADIAL
            rg[k].append(m.add_element(rn[k][j], rn[k][jn], STEEL, RIB,
                                       vref=RIB_VREF, offset=off))

    for i in range(m.n_nodes):
        m.fix(i, [RZ])
    edge = [UX, UY, UZ, RX, RY] if clamped else [UX, UY, UZ]
    for nd in rn[N_RINGS]:
        m.fix(nd, edge)

    lump_pressure(m, tris, q, dof=UZ, sign=-1.0)
    return m, c, rn, rb, rg


def optimize(clamped, defl=None, sa=None):
    """1ケースを最適化し (prob, res, model, center, radial_bands, rings) を返す。"""
    m, c, rn, rb, rg = build(clamped)
    dvs = []
    for b in range(N_RINGS):
        dvs.append(DesignVar(ScaledSection(RIB), rb[b], x0=1.5, xmin=0.3, xmax=4.0,
                             name=f"radial{b}"))
    for k in range(1, N_RINGS + 1):
        dvs.append(DesignVar(ScaledSection(RIB), rg[k], x0=1.5, xmin=0.3, xmax=4.0,
                             name=f"ring{k}"))
    dl = []
    if defl is not None:
        dl = [DispLimit(c, UZ, defl)] + [DispLimit(rn[k][0], UZ, defl)
                                         for k in range(1, N_RINGS)]
    prob = SizingProblem(m, dvs, sigma_allow=sa, disp_limits=dl)
    res = minimize_mass(prob, maxiter=200, move=0.15, tol=1e-6)
    return prob, res, m, c, rb, rg


CASES = [
    ("(A) 単純支持 + たわみ制約", "(A) simply supported + deflection", dict(clamped=False, defl=0.010)),
    ("(B) 完全固定 + たわみ制約", "(B) clamped + deflection", dict(clamped=True, defl=0.010)),
    ("(C) 完全固定 + 応力制約",   "(C) clamped + stress", dict(clamped=True, sa=120e6)),
]

print("=== リブ補強板：境界条件・制約による最適リブ配置の比較 ===")
print(f"円板 R={R} m, 板厚 t={t*1e3:.0f} mm（薄板）, 放射 {N_RADIAL} × 半径 {N_RINGS}, q={q/1e3:.0f} kPa\n")

results = []
for title, title_en, kw in CASES:
    prob, res, m, c, rb, rg = optimize(**kw)
    rad = res.x[:N_RINGS]
    rng = res.x[N_RINGS:]
    results.append((title_en, prob, res, m))
    print(f"{title}:  リブ質量 = {res.mass:5.1f} kg  (収束={res.converged})")
    print(f"   radial(中心0→外周{N_RINGS-1}): " + " ".join(f"{v:.2f}" for v in rad))
    print(f"   ring  (内1→外周{N_RINGS}):   " + " ".join(f"{v:.2f}" for v in rng))
    # 最太グループ
    allnames = [dv.name for dv in prob.design_vars]
    imax = int(np.argmax(res.x))
    print(f"   → 最太グループ: {allnames[imax]} (×{res.x[imax]:.2f})\n")

print("ポイント: 単純支持(A)は支持部のモーメントが0・縁は既に拘束済みのため、")
print("  中間半径のリングが追加支持として効き最太になる。固定端(C)は固定部で")
print("  モーメント最大のため、外周に向かってリブが太くなる（直感どおり）。")

# --- 図示: 3ケースのリブ断面積分布を並べる（図中ラベルは英語）---
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from beamfem import viz

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    for ax, (title_en, prob, res, m) in zip(axes, results):
        area_cm2 = prob.element_values(res.x, kind="area") * 1e4
        viz.plot_member_sizes(m, area_cm2, label="rib area [cm^2]",
                              max_width=8.0, min_width=0.8, ax=ax)
        ax.set_aspect("equal")
        ax.set_title(f"{title_en}\nrib mass = {res.mass:.1f} kg")
    fig.suptitle("Optimal rib layout vs. boundary condition x constraint", fontsize=13)
    fig.tight_layout()
    p = viz.savefig("ribbed_plate_layout_study.png", dpi=130)
    print(f"\n比較図を {p} に保存しました。")
except ImportError:
    print("\n(matplotlib 未導入のため図はスキップ)")
