import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import milp, Bounds, LinearConstraint


# ============================================================
# 1. トラス形状
# ============================================================

# 節点座標 [m]
#
# 3 ----- 4 ----- 5
# | \   / | \   / |
# |  \ /  |  \ /  |
# 0 ----- 1 ----- 2
#
nodes = np.array([
    [0.0, 0.0],   # node 0
    [1.0, 0.0],   # node 1
    [2.0, 0.0],   # node 2
    [0.0, 1.0],   # node 3
    [1.0, 1.0],   # node 4
    [2.0, 1.0],   # node 5
])

# 候補部材（Ground Structure）
members = [
    (0, 1),
    (1, 2),

    (3, 4),
    (4, 5),

    (0, 3),
    (1, 4),
    (2, 5),

    (0, 4),
    (1, 3),

    (1, 5),
    (2, 4),
]

n_node = len(nodes)
n_member = len(members)

n_dof = 2 * n_node


# ============================================================
# 2. 材料・断面
# ============================================================

E = 210e9                 # Young率 [Pa]
rho = 7850.0              # 密度 [kg/m^3]

A0 = 300e-6               # 断面積 300 mm^2
sigma_allow = 250e6       # 許容応力 [Pa]

# 各部材の最大許容軸力
N_allow = sigma_allow * A0


# ============================================================
# 3. 荷重
# ============================================================

f = np.zeros(n_dof)

# node 4 に鉛直下向き100 kN
f[2 * 4 + 1] = -100e3


# ============================================================
# 4. 支持条件
# ============================================================

# node 0 : pin
# ux = 0
# uy = 0
#
# node 2 : roller
# uy = 0

fixed_dofs = [
    2 * 0,
    2 * 0 + 1,
    2 * 2 + 1,
]

free_dofs = [
    i for i in range(n_dof)
    if i not in fixed_dofs
]

n_free = len(free_dofs)


# ============================================================
# 5. トラスの幾何行列 G
# ============================================================
#
# G @ N = f
#
# N : 部材軸力
#

G = np.zeros((n_dof, n_member))

lengths = np.zeros(n_member)

for m, (i, j) in enumerate(members):

    xi, yi = nodes[i]
    xj, yj = nodes[j]

    dx = xj - xi
    dy = yj - yi

    L = np.sqrt(dx**2 + dy**2)

    c = dx / L
    s = dy / L

    lengths[m] = L

    # axial extension = g^T u
    G[2*i,     m] = -c
    G[2*i + 1, m] = -s

    G[2*j,     m] = c
    G[2*j + 1, m] = s


Gf = G[free_dofs, :]
ff = f[free_dofs]


# ============================================================
# 6. 設計変数
# ============================================================
#
# z =
#
# [ x_1 ... x_M,
#   N_1 ... N_M,
#   u_1 ... u_ndof ]
#
#
# x : 部材有無（binary）
# N : 軸力（continuous）
# u : 自由DOFの変位（continuous）
#

nx = n_member
nN = n_member
nu = n_free

n_var = nx + nN + nu

ix = slice(0, nx)
iN = slice(nx, nx + nN)
iu = slice(nx + nN, n_var)


# ============================================================
# 7. 目的関数：重量最小化
# ============================================================

c_obj = np.zeros(n_var)

for m in range(n_member):

    mass = rho * A0 * lengths[m]

    c_obj[m] = mass


# ============================================================
# 8. 変数の上下限
# ============================================================

lb_var = np.full(n_var, -np.inf)
ub_var = np.full(n_var,  np.inf)


# x : 0 <= x <= 1
lb_var[ix] = 0.0
ub_var[ix] = 1.0


# 軸力
lb_var[iN] = -N_allow
ub_var[iN] = N_allow


# 変位の一般的な上限
u_max = 20e-3       # ±20 mm

lb_var[iu] = -u_max
ub_var[iu] = u_max


# 荷重点 node 4 のY方向変位を ±5 mm に制限
load_y_global_dof = 2 * 4 + 1

load_y_local_dof = free_dofs.index(load_y_global_dof)

load_y_var = nx + nN + load_y_local_dof

u_limit = 5e-3

lb_var[load_y_var] = -u_limit
ub_var[load_y_var] = u_limit


bounds = Bounds(lb_var, ub_var)


# ============================================================
# 9. 整数変数
# ============================================================

integrality = np.zeros(n_var, dtype=int)

# xだけ整数
integrality[ix] = 1


# ============================================================
# 10. 線形制約
# ============================================================

rows = []
lower = []
upper = []


def add_constraint(row, lb, ub):

    rows.append(row)
    lower.append(lb)
    upper.append(ub)


# ------------------------------------------------------------
# 10-1. 節点力の釣り合い
#
# G N = f
# ------------------------------------------------------------

for d in range(n_free):

    row = np.zeros(n_var)

    row[iN] = Gf[d, :]

    add_constraint(
        row,
        ff[d],
        ff[d]
    )


# ------------------------------------------------------------
# 10-2. 部材が無い場合 N = 0
#
# |N_i| <= N_allow * x_i
# ------------------------------------------------------------

for m in range(n_member):

    # N - N_allow*x <= 0

    row = np.zeros(n_var)

    row[m] = -N_allow
    row[nx + m] = 1.0

    add_constraint(
        row,
        -np.inf,
        0.0
    )

    # -N - N_allow*x <= 0

    row = np.zeros(n_var)

    row[m] = -N_allow
    row[nx + m] = -1.0

    add_constraint(
        row,
        -np.inf,
        0.0
    )


# ------------------------------------------------------------
# 10-3. 弾性関係
#
# 部材が存在する場合
#
# N = EA/L * g^T u
#
# Big-MでON/OFFする
# ------------------------------------------------------------

for m in range(n_member):

    L = lengths[m]

    k = E * A0 / L

    g = Gf[:, m]

    # u の範囲から十分大きな Big-M を作る
    extension_bound = u_max * np.sum(np.abs(g))

    M = N_allow + k * extension_bound


    # N - k*g^T*u <= M*(1-x)
    #
    # N - k*g^T*u + M*x <= M

    row = np.zeros(n_var)

    row[m] = M
    row[nx + m] = 1.0
    row[iu] = -k * g

    add_constraint(
        row,
        -np.inf,
        M
    )


    # -(N - k*g^T*u) <= M*(1-x)
    #
    # -N + k*g^T*u + M*x <= M

    row = np.zeros(n_var)

    row[m] = M
    row[nx + m] = -1.0
    row[iu] = k * g

    add_constraint(
        row,
        -np.inf,
        M
    )


# 行列化
Acon = np.vstack(rows)

constraint = LinearConstraint(
    Acon,
    np.array(lower),
    np.array(upper)
)


# ============================================================
# 11. MILPを解く
# ============================================================

result = milp(
    c=c_obj,
    integrality=integrality,
    bounds=bounds,
    constraints=constraint,
    options={
        "disp": True,
        "time_limit": 30.0,
        "mip_rel_gap": 1e-6,
    }
)


# ============================================================
# 12. 結果
# ============================================================

if not result.success:

    print("Optimization failed")
    print(result.message)

    raise RuntimeError("MILP could not find an optimum.")


z = result.x

x_opt = z[ix]
N_opt = z[iN]
u_opt = z[iu]


print()
print("====================================")
print("Optimization result")
print("====================================")

print(f"Mass = {result.fun:.3f} kg")

print()
print("Selected members")
print("------------------------------------")

for m, (i, j) in enumerate(members):

    if x_opt[m] > 0.5:

        stress = N_opt[m] / A0 / 1e6

        print(
            f"member {m:2d}: "
            f"{i} -- {j}   "
            f"L = {lengths[m]:.3f} m   "
            f"N = {N_opt[m]/1e3:8.2f} kN   "
            f"sigma = {stress:8.2f} MPa"
        )


# ============================================================
# 13. 節点変位
# ============================================================

u_full = np.zeros(n_dof)

u_full[free_dofs] = u_opt

print()
print("Node displacement [mm]")
print("------------------------------------")

for i in range(n_node):

    ux = u_full[2*i] * 1000
    uy = u_full[2*i + 1] * 1000

    print(
        f"node {i}: "
        f"ux = {ux:8.4f} mm, "
        f"uy = {uy:8.4f} mm"
    )


# ============================================================
# 14. 支点反力
# ============================================================

reaction = G @ N_opt - f

print()
print("Support reactions")
print("------------------------------------")

for dof in fixed_dofs:

    node = dof // 2
    direction = "x" if dof % 2 == 0 else "y"

    print(
        f"node {node} R{direction} = "
        f"{reaction[dof]/1e3:8.2f} kN"
    )


# ============================================================
# 15. 最適トラスを描画
# ============================================================

plt.figure(figsize=(8, 5))


# 全候補部材
for i, j in members:

    p1 = nodes[i]
    p2 = nodes[j]

    plt.plot(
        [p1[0], p2[0]],
        [p1[1], p2[1]],
        "--",
        linewidth=0.7,
        alpha=0.2
    )


# 選択された部材
for m, (i, j) in enumerate(members):

    if x_opt[m] > 0.5:

        p1 = nodes[i]
        p2 = nodes[j]

        utilization = abs(N_opt[m]) / N_allow

        plt.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            "-",
            linewidth=2 + 4 * utilization
        )


# 節点
plt.scatter(
    nodes[:, 0],
    nodes[:, 1],
    zorder=5
)

for i, (x, y) in enumerate(nodes):

    plt.text(
        x + 0.03,
        y + 0.03,
        str(i)
    )


plt.axis("equal")
plt.grid(True)

plt.xlabel("X [m]")
plt.ylabel("Y [m]")

plt.title(
    f"Optimized truss\n"
    f"Mass = {result.fun:.2f} kg"
)

plt.tight_layout()

plt.show()
