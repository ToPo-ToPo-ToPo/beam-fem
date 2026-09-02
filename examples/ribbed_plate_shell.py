"""円形膜（円板パネル）をリブで補強したシェル＋梁の連成解析。

円板そのものを三角形フラットシェル（CST 膜 + DKT 板曲げ）でモデル化し、その
下面を放射＋同心リング状のリブ（3D Timoshenko 梁）で補強する。シェルと梁は
**同じ節点**を共有するので、板の曲げとリブの曲げ・ねじりが連成して荷重を支える
「リブ補強板（stiffened plate）」になる。

リブは板の中立面より下げて配置するのが正しい（T 形断面の合成効果）。本例では
剛体オフセット（offset 引数）でリブ図心を板下面に下げ、軸-曲げ連成（EA·e²）を
取り込む。比較のため次の3ケースを解く::

    1) 板のみ（リブ無し）
    2) リブ有り・めり込み（offset=0: リブ図心が板中立面と一致＝従来の近似）
    3) リブ有り・正規オフセット（offset = t/2 + h/2: 板下面にリブを付ける）

合成効果（EA·e²）はリブの軸力を板の膜（シェル面内剛性）が分担して初めて働く
ため、面内自由度 UX,UY は内部で自由にし、外周で面内も保持する。

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
RIB_B, RIB_H = 0.006, 0.030               # リブ断面 幅6mm × 高30mm
RIB = Section.rectangle(b=RIB_B, h=RIB_H, name="rib")
RIB_VREF = np.array([0.0, 0.0, 1.0])      # 局所 y 軸を鉛直に → Iz が鉛直曲げの強軸
E_OFFSET = t / 2 + RIB_H / 2              # 板下面にリブを付ける偏心 = 5+15 = 20 mm


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


def build_model(with_ribs: bool, rib_offset=None):
    """円板シェルモデルを作る。

    with_ribs=True で放射＋リングのリブ（梁）を足す。rib_offset を与えると
    リブ図心を板中立面から下げる（[0,0,-e]）。
    """
    m = Model()
    center, ring_nodes, triangles = build_disk_nodes(m, R, N_RADIAL, N_RINGS)

    for (i, j, k) in triangles:
        m.add_shell(i, j, k, STEEL, t)

    if with_ribs:
        off = None if rib_offset is None else [0.0, 0.0, -rib_offset]
        for j in range(N_RADIAL):
            m.add_element(center, ring_nodes[1][j], STEEL, RIB, vref=RIB_VREF, offset=off)
            for k in range(1, N_RINGS):
                m.add_element(ring_nodes[k][j], ring_nodes[k + 1][j], STEEL, RIB,
                              vref=RIB_VREF, offset=off)
        for k in range(1, N_RINGS + 1):
            for j in range(N_RADIAL):
                jn = (j + 1) % N_RADIAL
                m.add_element(ring_nodes[k][j], ring_nodes[k][jn], STEEL, RIB,
                              vref=RIB_VREF, offset=off)

    # ドリリング θz は実剛性が無いので全節点で拘束。面内 UX,UY は内部で自由に
    # して、リブ軸力を板の膜が分担できるようにする（合成効果に必須）。
    for i in range(m.n_nodes):
        m.fix(i, [RZ])
    # 外周: 単純支持（w=0）＋面内保持（UX,UY=0）。面内剛体運動も除去する。
    for nd in ring_nodes[N_RINGS]:
        m.fix(nd, [UX, UY, UZ])

    total = lump_pressure(m, triangles, q, dof=UZ, sign=-1.0)
    return m, center, ring_nodes, total


def rib_mass(m: Model) -> float:
    """梁（リブ）要素の総質量 [kg]。"""
    return sum(e.mat.rho * e.sec.A * m.element_length(e) for e in m.elements)


# --- 3ケースを解く ---
m0, c0, ring0, total = build_model(with_ribs=False)
res0 = solve_static(m0)
w0 = res0.node_disp(c0)[UZ]

m1, c1, ring1, _ = build_model(with_ribs=True, rib_offset=None)       # めり込み e=0
res1 = solve_static(m1)
w1 = res1.node_disp(c1)[UZ]

m2, c2, ring2, _ = build_model(with_ribs=True, rib_offset=E_OFFSET)   # 正規オフセット
res2 = solve_static(m2)
w2 = res2.node_disp(c2)[UZ]

print("=== 円形膜のリブ補強（シェル板＋梁リブの連成）===")
print(f"円板: R={R} m, 板厚 t={t*1e3:.0f} mm, 放射 {N_RADIAL} × 半径 {N_RINGS}")
print(f"節点 {m2.n_nodes}, シェル {len(m2.shells)} 枚, リブ梁 {len(m2.elements)} 本")
print(f"リブ断面: {RIB_B*1e3:.0f}×{RIB_H*1e3:.0f} mm, リブ総質量 {rib_mass(m2):.2f} kg")
print(f"偏心 e = t/2 + h/2 = {E_OFFSET*1e3:.0f} mm")
print(f"圧力 q={q/1e3:.1f} kPa, 総荷重 {total/1e3:.3f} kN "
      f"(理論 πR²q={np.pi*R*R*q/1e3:.3f} kN)")

print("\n--- 中央たわみの比較 ---")
print(f"1) 板のみ            : {w0*1e3:8.3f} mm")
print(f"2) リブ有り・めり込み : {w1*1e3:8.3f} mm  (板のみ比 {(1-w1/w0)*100:4.1f}% 低減)")
print(f"3) リブ有り・オフセット: {w2*1e3:8.3f} mm  (板のみ比 {(1-w2/w0)*100:4.1f}% 低減)")
print(f"   → オフセットの追加効果: めり込み比 {(1-w2/w1)*100:4.1f}% 低減")

# --- 鉛直つり合いの確認 ---
rz = sum(res2.reactions[nd * DOF_PER_NODE + UZ] for nd in ring2[N_RINGS])
print(f"\n外周の鉛直反力合計 = {rz/1e3:.3f} kN (総荷重と釣り合う)")

# --- リブ軸力（合成効果の指標）---
f1 = recover_forces(m1, res1)
f2 = recover_forces(m2, res2)
nmax1 = max(abs(f1[i].max_abs("N")) for i in range(len(f1)))
nmax2 = max(abs(f2[i].max_abs("N")) for i in range(len(f2)))
print("\n--- リブ最大軸力 |N|（T 形合成のあかし）---")
print(f"めり込み : {nmax1/1e3:7.2f} kN")
print(f"オフセット: {nmax2/1e3:7.2f} kN")

# --- 板の曲げモーメント低減（中心付近シェル要素）---
sf0 = recover_shell_forces(m0, res0)
sf2 = recover_shell_forces(m2, res2)
print("\n--- 中心付近シェル要素の曲げモーメント |Mx| ---")
print(f"板のみ    : {abs(sf0.shells[0].get('Mx')):8.1f} N·m/m")
print(f"オフセット: {abs(sf2.shells[0].get('Mx')):8.1f} N·m/m")

# --- 図示（オフセット補強板）---
try:
    from beamfem import viz

    fig, ax = viz.plot_deformed(m2, res2, scale="auto")
    p = viz.savefig("ribbed_plate_shell_deformed.png", dpi=130)
    fig2, ax2 = viz.plot_model(m2)
    p2 = viz.savefig("ribbed_plate_shell_model.png", dpi=130)
    print(f"\n変形図を {p}、モデル図を {p2} に保存しました。")
except ImportError:
    print('\n(matplotlib 未導入のため図はスキップ。pip install -e ".[viz]")')
