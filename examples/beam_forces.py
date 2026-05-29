"""要素内力・応力の回収と「項目を指定した」出力の例。

単純梁（中央集中荷重）について、表示したい内力成分だけを選んで出力する。
常に全成分を出すのではなく、メインスクリプト側で items を指定する。
"""

from beamfem import Material, Section, Model, solve_static, recover_forces, UY

STEEL = Material(E=200e9, nu=0.3)
# I 形断面（H 形鋼）。他に rectangle / box / pipe / circle / i_section が使える。
sec = Section.i_section(h=0.3, bf=0.15, tf=0.012, tw=0.008)

# 単純梁: スパン4m を 10要素に分割、中央に下向き 10 kN
n, span, P = 10, 4.0, 10_000.0
m = Model()
nodes = [m.add_node(span * i / n, 0, 0) for i in range(n + 1)]
for i in range(n):
    m.add_element(nodes[i], nodes[i + 1], STEEL, sec)
m.pin(nodes[0])
m.fix(nodes[-1], [1, 2])  # 右端ローラー
m.fix_to_plane_xy()
m.add_load(nodes[n // 2], UY, -P)

res = solve_static(m)
forces = recover_forces(m, res)

# --- 表示したい項目だけを選んで出力 ---------------------------------
print("◆ 曲げモーメントとせん断（要素内の絶対値最大）")
forces.print_table(items=["Mz", "Vy"], at="max")

print("\n◆ 応力のチェック（最大合成応力のみ）")
forces.print_table(items=["sigma_max"], at="max")

print("\n◆ 特定要素のみ・両端値（要素0と中央要素）")
forces.print_table(items=["N", "Mz"], at="ends", element_ids=[0, n // 2])

# --- CSV 出力（項目指定）。相対パスは workspace/ に保存される ---------
csv_path = forces.to_csv("beam_forces.csv", items=["N", "Vy", "Mz", "sigma_max"], at="ends")
print(f"\nCSV を {csv_path} に保存しました（項目: N, Vy, Mz, sigma_max）。")

# --- 断面力図（指定した成分のみ描画） -------------------------------
try:
    from beamfem import viz

    fig, ax = viz.plot_diagram(forces, "Mz", scale="auto")
    png_path = viz.savefig("beam_Mz_diagram.png", dpi=120)
    print(f"曲げモーメント図を {png_path} に保存しました。")
except ImportError:
    print('(matplotlib 未導入のため図はスキップ)')
