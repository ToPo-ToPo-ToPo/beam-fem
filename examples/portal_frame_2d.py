"""2D 門型ラーメン（ポータルフレーム）の静解析の例。

         w (水平荷重)
         ->  ┌────────────┐
             │            │
             │ (柱)       │ (柱)
             │            │
            ▟▟           ▟▟   (固定支持)

x-y 面内問題として解く。面外自由度は fix_to_plane_xy() で拘束する。
"""

import numpy as np

from beamfem import Material, Section, Model, solve_static, UX, UY, RZ

STEEL = Material(E=200e9, nu=0.3, name="steel")
col = Section.rectangle(b=0.2, h=0.3, name="column")  # 柱
beam = Section.rectangle(b=0.2, h=0.4, name="beam")  # 梁

H = 3.0  # 階高
W = 5.0  # スパン

m = Model()
n0 = m.add_node(0.0, 0.0)  # 左脚
n1 = m.add_node(0.0, H)  # 左肩
n2 = m.add_node(W, H)  # 右肩
n3 = m.add_node(W, 0.0)  # 右脚

m.add_element(n0, n1, STEEL, col)  # 左柱
m.add_element(n1, n2, STEEL, beam)  # 梁
m.add_element(n2, n3, STEEL, col)  # 右柱

m.fix(n0)  # 脚部固定
m.fix(n3)
m.fix_to_plane_xy()  # 2D 面内問題に拘束

m.add_load(n1, UX, 10_000.0)  # 左肩に水平荷重 10 kN

res = solve_static(m)

print("=== 2D ポータルフレーム ===")
for name, n in [("左肩 n1", n1), ("右肩 n2", n2)]:
    d = res.node_disp(n)
    print(f"{name}: ux={d[UX]*1e3:8.4f} mm  uy={d[UY]*1e3:8.4f} mm  rz={d[RZ]*1e3:8.4f} mrad")

print("\n支点反力:")
print(f"  左脚 n0: Fx={res.reactions[n0*6+UX]/1e3:7.3f} kN  Fy={res.reactions[n0*6+UY]/1e3:7.3f} kN")
print(f"  右脚 n3: Fx={res.reactions[n3*6+UX]/1e3:7.3f} kN  Fy={res.reactions[n3*6+UY]/1e3:7.3f} kN")

# 水平方向のつり合い確認: 反力合計 = -外力
total_fx = res.reactions[n0 * 6 + UX] + res.reactions[n3 * 6 + UX]
print(f"\n水平反力合計 = {total_fx/1e3:.3f} kN (外力 -10 kN と釣り合う)")
