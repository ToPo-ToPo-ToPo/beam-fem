import math
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import milp, Bounds, LinearConstraint


# ============================================================
# Mixed-Integer Truss Topology + Size Optimization
#
# Units:
#   Length       : mm
#   Force        : kN
#   Stress       : kN/mm^2
#   E            : kN/mm^2
#   Area         : mm^2
#   I            : mm^4
#   Mass         : kg
#
# Variables:
#
#   y[m,s] = 1 if member m uses section s
#            0 otherwise
#
#   N[l,m,s] = axial force [kN]
#              load case l
#              member m
#              section s
#
#   u[l,d] = displacement [mm]
#
#
# Objective:
#
#   minimize total mass
#
#
# Constraints:
#
#   - one section at most per candidate member
#   - equilibrium
#   - tension allowable stress
#   - compression allowable stress
#   - Euler buckling
#   - linear elastic compatibility
#   - displacement limits
#
# ============================================================


# ============================================================
# 1. Geometry
# ============================================================

#
#     4 ------- 5 ------- 6 ------- 7
#     |\       /|\       /|\       /|
#     | \     / | \     / | \     / |
#     |  \   /  |  \   /  |  \   /  |
#     |   \ /   |   \ /   |   \ /   |
#     0 ------- 1 ------- 2 ------- 3
#
# node 0: pin
# node 3: roller
#

nodes = np.array([
    [   0.0,    0.0],   # 0
    [1500.0,    0.0],   # 1
    [3000.0,    0.0],   # 2
    [4500.0,    0.0],   # 3

    [   0.0, 1200.0],   # 4
    [1500.0, 1200.0],   # 5
    [3000.0, 1200.0],   # 6
    [4500.0, 1200.0],   # 7
])


# Ground Structure
#
# あらかじめ存在し得る部材を全て用意する
#
members = [
    # bottom chord
    (0, 1),
    (1, 2),
    (2, 3),

    # top chord
    (4, 5),
    (5, 6),
    (6, 7),

    # vertical
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),

    # diagonals
    (0, 5),
    (1, 4),

    (1, 6),
    (2, 5),

    (2, 7),
    (3, 6),
]


n_node = len(nodes)
n_member = len(members)

n_dof = 2 * n_node


# ============================================================
# 2. Material
# ============================================================

# Steel
E = 210.0
# 210 kN/mm^2 = 210 GPa

rho = 7.85e-6
# kg/mm^3


# allowable stresses
sigma_t_allow = 0.250
# 250 MPa

sigma_c_allow = 0.200
# 200 MPa


# ============================================================
# 3. Discrete section catalogue
# ============================================================
#
# 実務ではここを実際の鋼管・H形鋼などの
# A, I に置き換える
#

sections = [
    {
        "name": "S",
        "A": 300.0,
        "I": 2.5e4,
    },
    {
        "name": "M",
        "A": 500.0,
        "I": 6.5e4,
    },
    {
        "name": "L",
        "A": 800.0,
        "I": 1.8e5,
    },
    {
        "name": "XL",
        "A": 1200.0,
        "I": 4.2e5,
    },
]

n_section = len(sections)

Asec = np.array([
    sec["A"]
    for sec in sections
])

Isec = np.array([
    sec["I"]
    for sec in sections
])


# ============================================================
# 4. Supports
# ============================================================

# node 0:
#
# ux = 0
# uy = 0
#
# node 3:
#
# uy = 0
#

fixed_dofs = [
    2 * 0,
    2 * 0 + 1,

    2 * 3 + 1,
]

free_dofs = [
    d
    for d in range(n_dof)
    if d not in fixed_dofs
]

n_free = len(free_dofs)


# ============================================================
# 5. Load cases
# ============================================================

load_cases = []


# ------------------------------------------------------------
# LC1
#
# Gravity-like vertical load
# ------------------------------------------------------------

f1 = np.zeros(n_dof)

f1[2 * 5 + 1] = -80.0
f1[2 * 6 + 1] = -80.0

load_cases.append({
    "name": "LC1 Vertical",
    "f": f1,
})


# ------------------------------------------------------------
# LC2
#
# Vertical + horizontal load
# ------------------------------------------------------------

f2 = np.zeros(n_dof)

f2[2 * 5 + 1] = -50.0
f2[2 * 6 + 1] = -50.0

f2[2 * 7] = 40.0

load_cases.append({
    "name": "LC2 Vertical + Wind",
    "f": f2,
})


n_load = len(load_cases)


# ============================================================
# 6. Geometry matrix
# ============================================================
#
# equilibrium:
#
# G @ N = f
#
# member extension:
#
# delta = g^T u
#

G = np.zeros(
    (n_dof, n_member)
)

lengths = np.zeros(n_member)


for m, (i, j) in enumerate(members):

    xi, yi = nodes[i]
    xj, yj = nodes[j]

    dx = xj - xi
    dy = yj - yi

    L = math.sqrt(
        dx**2 + dy**2
    )

    c = dx / L
    s = dy / L

    lengths[m] = L

    G[2 * i,     m] = -c
    G[2 * i + 1, m] = -s

    G[2 * j,     m] = c
    G[2 * j + 1, m] = s


Gf = G[free_dofs, :]


# load vectors at free DOF
ff = np.array([
    lc["f"][free_dofs]
    for lc in load_cases
])


# ============================================================
# 7. Section capacities
# ============================================================

# tension
T_capacity = (
    sigma_t_allow * Asec
)

# compression by material stress
C_capacity = (
    sigma_c_allow * Asec
)


# Euler buckling:
#
# Pcr = pi^2 E I / (K L)^2
#
# K = effective length coefficient
#
K_effective = 1.0


Pcr = np.zeros(
    (n_member, n_section)
)


for m in range(n_member):

    L = lengths[m]

    for s in range(n_section):

        I = Isec[s]

        Pcr[m, s] = (
            math.pi**2
            * E
            * I
            / (K_effective * L)**2
        )


# ============================================================
# 8. Variable indexing
# ============================================================
#
# z =
#
# [
#   y[m,s],
#
#   N[l,m,s],
#
#   u[l,d]
# ]
#

n_y = (
    n_member
    * n_section
)

n_N = (
    n_load
    * n_member
    * n_section
)

n_u = (
    n_load
    * n_free
)

n_var = (
    n_y
    + n_N
    + n_u
)


def iy(m, s):

    return (
        m * n_section
        + s
    )


def iN(l, m, s):

    return (
        n_y
        + (l * n_member + m)
        * n_section
        + s
    )


def iu(l, d):

    return (
        n_y
        + n_N
        + l * n_free
        + d
    )


# ============================================================
# 9. Objective function
# ============================================================
#
# mass =
#
# rho * A * L
#

c_obj = np.zeros(n_var)


for m in range(n_member):

    for s in range(n_section):

        mass = (
            rho
            * Asec[s]
            * lengths[m]
        )

        c_obj[iy(m, s)] = mass


# ============================================================
# 10. Variable bounds
# ============================================================

lb = np.full(
    n_var,
    -np.inf
)

ub = np.full(
    n_var,
    np.inf
)


# ------------------------------------------------------------
# y
# ------------------------------------------------------------

lb[:n_y] = 0.0
ub[:n_y] = 1.0


# ------------------------------------------------------------
# N
#
# loose global bounds;
# detailed bounds imposed later
# ------------------------------------------------------------

max_force = max(
    np.max(T_capacity),
    np.max(C_capacity)
)


lb[n_y:n_y + n_N] = -max_force
ub[n_y:n_y + n_N] = max_force


# ------------------------------------------------------------
# displacement
# ------------------------------------------------------------

global_u_limit = 20.0
# mm


for l in range(n_load):

    for d in range(n_free):

        lb[iu(l, d)] = -global_u_limit
        ub[iu(l, d)] = global_u_limit


# ============================================================
# 11. Serviceability displacement constraints
# ============================================================

# upper nodes
top_nodes = [
    4, 5, 6, 7
]


vertical_limit = 8.0
# mm

horizontal_limit = 12.0
# mm


for l in range(n_load):

    for node in top_nodes:

        # x
        global_dof = 2 * node

        if global_dof in free_dofs:

            local_dof = free_dofs.index(
                global_dof
            )

            lb[iu(l, local_dof)] = (
                -horizontal_limit
            )

            ub[iu(l, local_dof)] = (
                horizontal_limit
            )

        # y
        global_dof = 2 * node + 1

        if global_dof in free_dofs:

            local_dof = free_dofs.index(
                global_dof
            )

            lb[iu(l, local_dof)] = (
                -vertical_limit
            )

            ub[iu(l, local_dof)] = (
                vertical_limit
            )


bounds = Bounds(
    lb,
    ub
)


# ============================================================
# 12. Integer variable specification
# ============================================================

integrality = np.zeros(
    n_var,
    dtype=int
)

# only y variables are integer
integrality[:n_y] = 1


# ============================================================
# 13. Constraint construction
# ============================================================

rows = []
lower = []
upper = []


def add_constraint(
    row,
    low=-np.inf,
    high=np.inf
):

    rows.append(row)

    lower.append(low)

    upper.append(high)


# ============================================================
# 14. At most one section per member
# ============================================================
#
# sum_s y[m,s] <= 1
#
# all zero:
# member does not exist
#

for m in range(n_member):

    row = np.zeros(n_var)

    for s in range(n_section):

        row[iy(m, s)] = 1.0

    add_constraint(
        row,
        high=1.0
    )


# ============================================================
# 15. Equilibrium
# ============================================================
#
# sum_m sum_s
#
# G[d,m] N[l,m,s]
#
# = f[l,d]
#

for l in range(n_load):

    for d in range(n_free):

        row = np.zeros(n_var)

        for m in range(n_member):

            for s in range(n_section):

                row[iN(l, m, s)] = (
                    Gf[d, m]
                )

        add_constraint(
            row,
            low=ff[l, d],
            high=ff[l, d]
        )


# ============================================================
# 16. Member capacity + compatibility
# ============================================================

for l in range(n_load):

    # displacement bounds used for tighter Big-M

    u_abs_bound = np.array([
        max(
            abs(lb[iu(l, d)]),
            abs(ub[iu(l, d)])
        )
        for d in range(n_free)
    ])


    for m in range(n_member):

        g = Gf[:, m]

        L = lengths[m]


        # conservative maximum possible
        # |g^T u|
        extension_bound = np.sum(
            np.abs(g)
            * u_abs_bound
        )


        for s in range(n_section):

            y_idx = iy(m, s)

            N_idx = iN(
                l,
                m,
                s
            )


            # =================================================
            # 16-1. Tension
            #
            # N <= Nt * y
            # =================================================

            row = np.zeros(n_var)

            row[N_idx] = 1.0

            row[y_idx] = (
                -T_capacity[s]
            )

            add_constraint(
                row,
                high=0.0
            )


            # =================================================
            # 16-2. Compression stress
            #
            # -N <= Nc * y
            # =================================================

            row = np.zeros(n_var)

            row[N_idx] = -1.0

            row[y_idx] = (
                -C_capacity[s]
            )

            add_constraint(
                row,
                high=0.0
            )


            # =================================================
            # 16-3. Euler buckling
            #
            # -N <= Pcr * y
            # =================================================

            row = np.zeros(n_var)

            row[N_idx] = -1.0

            row[y_idx] = (
                -Pcr[m, s]
            )

            add_constraint(
                row,
                high=0.0
            )


            # =================================================
            # 16-4. Elastic compatibility
            #
            # if y = 1:
            #
            # N = EA/L * g^T u
            #
            # if y = 0:
            #
            # relation is relaxed
            # =================================================

            k = (
                E
                * Asec[s]
                / L
            )


            # maximum force allowed
            # in either direction
            compression_limit = min(
                C_capacity[s],
                Pcr[m, s]
            )

            force_bound = max(
                T_capacity[s],
                compression_limit
            )


            # automatic Big-M
            BigM = (
                force_bound
                + k * extension_bound
            )


            # -------------------------------------------------
            # N - k g^T u <= M(1-y)
            #
            # N - k g^T u + M y <= M
            # -------------------------------------------------

            row = np.zeros(n_var)

            row[N_idx] = 1.0
            row[y_idx] = BigM

            for d in range(n_free):

                row[iu(l, d)] += (
                    -k
                    * g[d]
                )

            add_constraint(
                row,
                high=BigM
            )


            # -------------------------------------------------
            # -(N - k g^T u) <= M(1-y)
            #
            # -N + k g^T u + M y <= M
            # -------------------------------------------------

            row = np.zeros(n_var)

            row[N_idx] = -1.0
            row[y_idx] = BigM

            for d in range(n_free):

                row[iu(l, d)] += (
                    k
                    * g[d]
                )

            add_constraint(
                row,
                high=BigM
            )


# ============================================================
# 17. Convert constraints
# ============================================================

A_constraint = np.vstack(
    rows
)

constraints = LinearConstraint(
    A_constraint,
    np.array(lower),
    np.array(upper)
)


# ============================================================
# 18. Solve MILP
# ============================================================

result = milp(
    c=c_obj,

    integrality=integrality,

    bounds=bounds,

    constraints=constraints,

    options={
        "disp": True,

        "time_limit": 60.0,

        "mip_rel_gap": 1e-6,
    }
)


if not result.success:

    print()
    print("Optimization failed")
    print(result.message)

    raise RuntimeError(
        "MILP solver could not find an optimum."
    )


z = result.x


# ============================================================
# 19. Extract selected sections
# ============================================================

selected = {}


for m in range(n_member):

    for s in range(n_section):

        if z[iy(m, s)] > 0.5:

            selected[m] = s


# ============================================================
# 20. Basic result
# ============================================================

print()
print("========================================")
print("OPTIMIZATION RESULT")
print("========================================")

print(
    f"Total mass = {result.fun:.3f} kg"
)

print(
    f"Candidate members = {n_member}"
)

print(
    f"Selected members = {len(selected)}"
)


print()
print("SELECTED MEMBERS")
print("----------------------------------------")


for m, s in selected.items():

    i, j = members[m]

    sec = sections[s]

    print(
        f"Member {m:2d}: "
        f"{i} -- {j}   "
        f"{sec['name']:>2s}   "
        f"A={sec['A']:7.1f} mm^2   "
        f"I={sec['I']:9.1f} mm^4   "
        f"L={lengths[m]:8.1f} mm"
    )


# ============================================================
# 21. Load-case results
# ============================================================

all_force = []
all_displacement = []


for l, load_case in enumerate(load_cases):

    print()
    print(
        "========================================"
    )

    print(
        load_case["name"]
    )

    print(
        "========================================"
    )


    # ----------------------------------------
    # displacement
    # ----------------------------------------

    u_full = np.zeros(n_dof)

    for d, global_dof in enumerate(free_dofs):

        u_full[global_dof] = (
            z[iu(l, d)]
        )

    all_displacement.append(
        u_full
    )


    # ----------------------------------------
    # forces
    # ----------------------------------------

    N_total = np.zeros(n_member)


    print()
    print("Member forces")
    print("----------------------------------------")


    for m, s in selected.items():

        N = z[
            iN(l, m, s)
        ]

        N_total[m] = N

        A = Asec[s]

        stress = (
            N / A
        )

        pcr = (
            Pcr[m, s]
        )


        if N >= 0:

            stress_util = (
                N
                / T_capacity[s]
            )

            buckling_util = 0.0

        else:

            stress_util = (
                abs(N)
                / C_capacity[s]
            )

            buckling_util = (
                abs(N)
                / pcr
            )


        utilization = max(
            stress_util,
            buckling_util
        )


        print(
            f"M{m:02d} "
            f"{members[m][0]}-{members[m][1]}   "
            f"{sections[s]['name']:>2s}   "
            f"N={N:8.2f} kN   "
            f"sigma={stress * 1000:8.1f} MPa   "
            f"util={utilization:6.3f}"
        )


    all_force.append(
        N_total
    )


    # ----------------------------------------
    # displacement output
    # ----------------------------------------

    print()
    print("Node displacements")
    print("----------------------------------------")


    for node in range(n_node):

        ux = u_full[
            2 * node
        ]

        uy = u_full[
            2 * node + 1
        ]

        print(
            f"Node {node}: "
            f"ux={ux:9.4f} mm   "
            f"uy={uy:9.4f} mm"
        )


    # ----------------------------------------
    # reactions
    # ----------------------------------------

    reaction = (
        G @ N_total
        - load_case["f"]
    )


    print()
    print("Support reactions")
    print("----------------------------------------")


    for dof in fixed_dofs:

        node = dof // 2

        direction = (
            "X"
            if dof % 2 == 0
            else "Y"
        )

        print(
            f"Node {node} R{direction} = "
            f"{reaction[dof]:9.3f} kN"
        )


# ============================================================
# 22. Plot optimized topology
# ============================================================

plt.figure(
    figsize=(10, 4)
)


# all candidate members
for m, (i, j) in enumerate(members):

    p1 = nodes[i]
    p2 = nodes[j]

    plt.plot(
        [p1[0], p2[0]],
        [p1[1], p2[1]],
        "--",
        linewidth=0.6,
        alpha=0.15
    )


# selected members
for m, s in selected.items():

    i, j = members[m]

    p1 = nodes[i]
    p2 = nodes[j]

    # section area controls line thickness
    width = (
        1.5
        + 3.0
        * Asec[s]
        / np.max(Asec)
    )

    plt.plot(
        [p1[0], p2[0]],
        [p1[1], p2[1]],
        linewidth=width
    )


# nodes
plt.scatter(
    nodes[:, 0],
    nodes[:, 1],
    zorder=10
)


for i, (x, y) in enumerate(nodes):

    plt.text(
        x + 40,
        y + 40,
        str(i)
    )


plt.axis("equal")

plt.xlabel("X [mm]")
plt.ylabel("Y [mm]")

plt.title(
    f"Optimized Truss  |  "
    f"Mass = {result.fun:.2f} kg"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# 23. Plot deformed shapes
# ============================================================

deformation_scale = 30.0


for l, load_case in enumerate(load_cases):

    u_full = all_displacement[l]

    deformed_nodes = (
        nodes
        + deformation_scale
        * u_full.reshape(-1, 2)
    )


    plt.figure(
        figsize=(10, 4)
    )


    # original selected topology
    for m in selected:

        i, j = members[m]

        p1 = nodes[i]
        p2 = nodes[j]

        plt.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            "--",
            linewidth=1.0,
            alpha=0.35
        )


    # deformed
    for m in selected:

        i, j = members[m]

        p1 = deformed_nodes[i]
        p2 = deformed_nodes[j]

        plt.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            linewidth=2.0
        )


    plt.axis("equal")

    plt.xlabel("X [mm]")
    plt.ylabel("Y [mm]")

    plt.title(
        f"{load_case['name']} "
        f"(deformation x{deformation_scale:g})"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.show()
