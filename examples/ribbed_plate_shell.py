"""円形膜（円板パネル）をリブで補強したシェル＋梁の連成解析。

円板そのものを三角形フラットシェル（CST 膜 + DKT 板曲げ）でモデル化し、その
下面を放射＋同心リング状のリブ（3D Timoshenko 梁）で補強する。シェルと梁は
**同じ節点**を共有するので、板の曲げとリブの曲げ・ねじりが連成して荷重を支える
「リブ補強板（stiffened plate）」になる。

上から一様圧力 q（下向き）を与え、外周単純支持で解く。リブ無し（板のみ）と
リブ有りを比較し、補強による中央たわみ・板の曲げモーメントの低減を確認する。

注意:
- リブは板の中立面に同心で配置する（中立面より下げる偏心は無視）。実際の
  スティフナはオフセットでさらに効くため、本モデルはやや安全側（補強効果を
  控えめに評価）になる。
- 面内（膜）荷重は無いので面内・ドリリング自由度は全節点で拘束する。

実行::

    .venv/bin/python examples/ribbed_plate_shell.py
"""

import numpy as np

from beamfem import (
    Material,
    Section,
    Model,
    solve_static,
    recover_forces,
    recover_shell_forces,
    lump_pressure,
    UX,
    UY,
    UZ,
    RX,
    RY,
    RZ,
)
from beamfem.model import DOF_PER_NODE

# --- パラメータ ---
R = 1.0            # 円板半径 [m]
t = 0.010          # 板厚 [m]
q = 8_000.0        # 上面圧力 [Pa]（下向き）
N_RADIAL = 12      # 放射分割数（＝放射リブ本数）
N_RINGS = 6        # 半径方向分割数

STEEL = Material(E=200e9, nu=0.3, rho=7850.0, name="steel")
# リブ断面（縦長の矩形: 鉛直方向に深い）。局所 y を鉛直にとり強軸 Iz で曲げを受ける。
RIB = Section.rectangle(b=0.006, h=0.030, name="rib")  # 幅6mm × 高30mm
RIB_VREF = np.array([0.0, 0.0, 1.0])  # 局所 y 軸を鉛直に → Iz が鉛直曲げの強軸


def build_disk_nodes(m: Model, R, n_radial, n_rings):
    """円板の節点（中心＋同心リング）と三角形分割を作る。"""
    angles = [2.0 * np.pi * j / n_radial for j in range(n_radial)]
    center = m.add_node(0.0, 0.0, 0.0)
    ring_nodes = [[center] * n_radial]
    for k in range(1, n_rings + 1):
        r = R * k / n_rings
        ring_nodes.append(
            [m.add_node(r * np.cos(a), r * np.sin(a), 0.0) for a in angles]
        )
    triangles = []
    for j in range(n_radial):
        jn = (j + 1) % n_radial
        triangles.append((center, ring_nodes[1][j], ring_nodes[1][jn]))
    for k in range(1, n_rings):
        for j in range(n_radial):
            jn = (j + 1) % n_radial
            a, b = ring_nodes[k][j], ring_nodes[k][jn]
            c, d = ring_nodes[k + 1][jn], ring_nodes[k + 1][j]
            triangles.append((a, b, c))
            triangles.append((a, c, d))
    return center, ring_nodes, triangles


def build_model(with_ribs: bool):
    """円板シェルモデルを作る。with_ribs=True で放射＋リングのリブ（梁）を足す。"""
    m = Model()
    center, ring_nodes, triangles = build_disk_nodes(m, R, N_RADIAL, N_RINGS)

    # 板（シェル）
    for (i, j, k) in triangles:
        m.add_shell(i, j, k, STEEL, t)

    # リブ（梁）: 放射スポーク＋同心リング。シェルと同じ節点を共有。
    if with_ribs:
        for j in range(N_RADIAL):
            m.add_element(center, ring_nodes[1][j], STEEL, RIB, vref=RIB_VREF)
            for k in range(1, N_RINGS):
                m.add_element(ring_nodes[k][j], ring_nodes[k + 1][j], STEEL, RIB,
                              vref=RIB_VREF)
        for k in range(1, N_RINGS + 1):
            for j in range(N_RADIAL):
                jn = (j + 1) % N_RADIAL
                m.add_element(ring_nodes[k][j], ring_nodes[k][jn], STEEL, RIB,
                              vref=RIB_VREF)

    # 面内自由度（膜・ドリリング）は全節点で拘束（純曲げ問題）
    for i in range(m.n_nodes):
        m.fix(i, [UX, UY, RZ])
    # 外周単純支持: たわみ w=0（回転は自由）
    for nd in ring_nodes[N_RINGS]:
        m.fix(nd, [UZ])

    # 上面圧力 → 等価節点荷重（下向き）
    total = lump_pressure(m, triangles, q, dof=UZ, sign=-1.0)
    return m, center, ring_nodes, total


def rib_mass(m: Model) -> float:
    """梁（リブ）要素の総質量 [kg]。"""
    mass = 0.0
    for e in m.elements:
        L = m.element_length(e)
        mass += e.mat.rho * e.sec.A * L
    return mass


# --- リブ無し（板のみ）---
m0, c0, _, total0 = build_model(with_ribs=False)
res0 = solve_static(m0)
w0 = res0.node_disp(c0)[UZ]

# --- リブ有り（リブ補強板）---
m1, c1, ring1, total1 = build_model(with_ribs=True)
res1 = solve_static(m1)
w1 = res1.node_disp(c1)[UZ]

print("=== 円形膜のリブ補強（シェル板＋梁リブの連成）===")
print(f"円板: R={R} m, 板厚 t={t*1e3:.0f} mm, 放射 {N_RADIAL} × 半径 {N_RINGS}")
print(f"節点 {m1.n_nodes}, シェル {len(m1.shells)} 枚, リブ梁 {len(m1.elements)} 本")
print(f"リブ断面: {RIB.name} (6×30 mm), リブ総質量 {rib_mass(m1):.2f} kg")
print(f"圧力 q={q/1e3:.1f} kPa, 総荷重 {total1/1e3:.3f} kN "
      f"(理論 πR²q={np.pi*R*R*q/1e3:.3f} kN)")

print("\n--- 中央たわみの比較 ---")
print(f"リブ無し（板のみ）: {w0*1e3:8.3f} mm")
print(f"リブ有り（補強板）: {w1*1e3:8.3f} mm")
print(f"低減率            : {(1 - w1/w0)*100:5.1f} %")

# --- 鉛直つり合いの確認（外周反力 = 総荷重）---
rim = ring1[N_RINGS]
rz = sum(res1.reactions[nd * DOF_PER_NODE + UZ] for nd in rim)
print(f"\n外周の鉛直反力合計 = {rz/1e3:.3f} kN (総荷重と釣り合う)")

# --- 板の曲げモーメント低減（中心付近シェル要素）---
sf0 = recover_shell_forces(m0, res0)
sf1 = recover_shell_forces(m1, res1)
print("\n--- 中心付近シェル要素の曲げモーメント |Mx| ---")
print(f"リブ無し: {abs(sf0.shells[0].get('Mx')):8.1f} N·m/m")
print(f"リブ有り: {abs(sf1.shells[0].get('Mx')):8.1f} N·m/m")

# --- リブ内力（最大応力上位）---
forces = recover_forces(m1, res1)
order = sorted(range(len(forces)), key=lambda i: forces[i].get_max_abs("sigma_max"),
               reverse=True)
print("\n--- 内力が大きいリブ 上位3（曲げ My・ねじり T・最大応力）---")
forces.print_table(items=["My", "Mz", "T", "sigma_max"], at="max",
                   element_ids=order[:3])

# --- 図示 ---
try:
    from beamfem import viz

    fig, ax = viz.plot_deformed(m1, res1, scale="auto")
    p = viz.savefig("ribbed_plate_shell_deformed.png", dpi=130)
    fig2, ax2 = viz.plot_model(m1)
    p2 = viz.savefig("ribbed_plate_shell_model.png", dpi=130)
    print(f"\n変形図を {p}、モデル図を {p2} に保存しました。")
except ImportError:
    print('\n(matplotlib 未導入のため図はスキップ。pip install -e ".[viz]")')
