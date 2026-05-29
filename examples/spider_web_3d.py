"""円形の蜘蛛の巣型フレームに上面全体から等分布荷重が作用する例題。

水平面 (x-y) 内に放射状スポークと同心リングからなる「蜘蛛の巣」フレームを置き、
上から一様な面分布荷重 q [N/m^2] を下向き (-z) に与える。フレームは面外に
曲げ・ねじりを受ける **グリラージュ（格子）問題** となり、3D Timoshenko 梁の
曲げ + ねじり挙動を確認できる。

面分布荷重は領域を三角形分割し、各三角形の荷重を頂点に1/3ずつ振り分ける
標準的な集約法で等価節点荷重に変換する（本ライブラリは節点荷重を入力とするため）。

蜘蛛の糸は円形断面（Iy=Iz）として扱うので、部材の向きに依存せず面外曲げを扱える。
"""

import numpy as np

from beamfem import Material, Section, Model, solve_static, recover_forces, UZ
from beamfem.model import DOF_PER_NODE

# ---- パラメータ -------------------------------------------------------
R = 2.0          # 外周半径 [m]
N_RADIAL = 8     # 放射スポーク本数
N_RINGS = 4      # 同心リング数（外周含む）
q = 500.0        # 面分布荷重 [N/m^2]（下向き）

SILK = Material(E=200e9, nu=0.3, name="silk")  # 材料（ここでは鋼材物性）
sec = Section.circle(d=0.02)                    # 円形断面 d=20mm

# ---- 幾何（節点・要素）の構築 ----------------------------------------
m = Model()
radii = [R * k / N_RINGS for k in range(N_RINGS + 1)]      # radii[0]=0（中心）
angles = [2 * np.pi * j / N_RADIAL for j in range(N_RADIAL)]

center = m.add_node(0.0, 0.0, 0.0)

# ring_nodes[k][j] : 半径 radii[k]・角度 angles[j] の節点番号（k>=1）
ring_nodes = [[center] * N_RADIAL]  # k=0 はすべて中心
for k in range(1, N_RINGS + 1):
    row = []
    for j in range(N_RADIAL):
        x = radii[k] * np.cos(angles[j])
        y = radii[k] * np.sin(angles[j])
        row.append(m.add_node(x, y, 0.0))
    ring_nodes.append(row)

# 放射スポーク（中心→外周へ各リングを接続）
for j in range(N_RADIAL):
    m.add_element(center, ring_nodes[1][j], SILK, sec)
    for k in range(1, N_RINGS):
        m.add_element(ring_nodes[k][j], ring_nodes[k + 1][j], SILK, sec)

# 同心リング（各リング上で隣接スポーク間を弦で接続）
for k in range(1, N_RINGS + 1):
    for j in range(N_RADIAL):
        jn = (j + 1) % N_RADIAL
        m.add_element(ring_nodes[k][j], ring_nodes[k][jn], SILK, sec)

# ---- 外周リングを固定支持 --------------------------------------------
for j in range(N_RADIAL):
    m.fix(ring_nodes[N_RINGS][j])

# ---- 面分布荷重 → 等価節点荷重（三角形分割・1/3集約） ----------------
def node_pos(k, j):
    return m.nodes[ring_nodes[k][j]][:2]

triangles = []
for j in range(N_RADIAL):  # 内側ファン（中心とリング1）
    jn = (j + 1) % N_RADIAL
    triangles.append((center, ring_nodes[1][j], ring_nodes[1][jn]))
for k in range(1, N_RINGS):  # 環状の四角形を2三角形に分割
    for j in range(N_RADIAL):
        jn = (j + 1) % N_RADIAL
        a, b = ring_nodes[k][j], ring_nodes[k][jn]
        c, d = ring_nodes[k + 1][jn], ring_nodes[k + 1][j]
        triangles.append((a, b, c))
        triangles.append((a, c, d))

def tri_area(n1, n2, n3):
    p1, p2, p3 = m.nodes[n1][:2], m.nodes[n2][:2], m.nodes[n3][:2]
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))

total_area = 0.0
for n1, n2, n3 in triangles:
    A = tri_area(n1, n2, n3)
    total_area += A
    f = -q * A / 3.0  # 下向き、頂点に1/3ずつ
    for nd in (n1, n2, n3):
        m.add_load(nd, UZ, f)

print(f"蜘蛛の巣フレーム: スポーク{N_RADIAL}本, リング{N_RINGS}環, 節点{m.n_nodes}, 要素{len(m.elements)}")
print(f"載荷面積={total_area:.4f} m^2, 総荷重={q*total_area:.1f} N (q={q} N/m^2)")

# ---- 解析 -------------------------------------------------------------
res = solve_static(m)
forces = recover_forces(m, res)

# 中心の鉛直たわみ
print(f"\n中心の鉛直たわみ uz = {res.node_disp(center)[UZ]*1e3:.3f} mm")

# 外周反力の合計（鉛直方向のつり合い確認）
rz_total = sum(
    res.reactions[ring_nodes[N_RINGS][j] * DOF_PER_NODE + UZ] for j in range(N_RADIAL)
)
print(f"外周の鉛直反力合計 = {rz_total:.1f} N (総荷重 {q*total_area:.1f} N と釣り合う)")

# ---- 内力の出力（項目を指定） ----------------------------------------
# 円形断面は Iy=Iz なので曲げ抵抗は My,Mz の合成。代表として最大値を確認。
print("\n◆ 最も内力が大きい要素 上位5（合成最大応力で抽出）")
order = sorted(range(len(forces)), key=lambda i: forces[i].get_max_abs("sigma_max"), reverse=True)
forces.print_table(items=["My", "Mz", "T", "sigma_max"], at="max", element_ids=order[:5])

# ---- 図示（変形図と曲げモーメント図） --------------------------------
try:
    from beamfem import viz

    fig1, _ = viz.plot_deformed(m, res, scale="auto")
    p1 = viz.savefig("spider_web_deformed.png", dpi=120)
    fig2, _ = viz.plot_diagram(forces, "Mz", scale="auto")
    p2 = viz.savefig("spider_web_Mz.png", dpi=120)
    print(f"\n変形図を {p1}、Mz 図を {p2} に保存しました。")
except ImportError:
    print('\n(matplotlib 未導入のため図はスキップ)')
