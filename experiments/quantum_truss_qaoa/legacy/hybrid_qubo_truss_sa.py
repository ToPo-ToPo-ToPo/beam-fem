import math
import random
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Hybrid QUBO Master + Classical FEM Subproblem
# Sequential local-QUBO method for discrete truss optimization
#
# Units:
#   length: mm
#   force : kN
#   E/stress: kN/mm^2
#   mass  : kg
#
# No quantum SDK is required.
# The QUBO is solved here by simulated annealing.
# Later, solve_qubo_sa() can be replaced by a QAOA/annealer backend.
# ============================================================

RNG = random.Random(7)

# ------------------------------------------------------------
# 1. Geometry
# ------------------------------------------------------------

nodes = np.array([
    [   0.0,    0.0],   # 0
    [1500.0,    0.0],   # 1
    [3000.0,    0.0],   # 2
    [4500.0,    0.0],   # 3
    [   0.0, 1200.0],   # 4
    [1500.0, 1200.0],   # 5
    [3000.0, 1200.0],   # 6
    [4500.0, 1200.0],   # 7
], dtype=float)

members = [
    (0, 1), (1, 2), (2, 3),       # bottom chord
    (4, 5), (5, 6), (6, 7),       # top chord
    (0, 4), (1, 5), (2, 6), (3, 7),  # verticals
    (0, 5), (1, 4),
    (1, 6), (2, 5),
    (2, 7), (3, 6),               # diagonals
]

n_node = len(nodes)
n_member = len(members)
n_dof = 2 * n_node

# ------------------------------------------------------------
# 2. Material and discrete section catalogue
# ------------------------------------------------------------

E = 210.0          # kN/mm^2 = 210 GPa
rho = 7.85e-6      # kg/mm^3

sigma_t_allow = 0.250   # 250 MPa
sigma_c_allow = 0.200   # 200 MPa

# state 0 means "member absent"
sections = [
    {"name": "NONE", "A": 0.0,    "I": 0.0},
    {"name": "S",    "A": 300.0,  "I": 2.5e4},
    {"name": "M",    "A": 500.0,  "I": 6.5e4},
    {"name": "L",    "A": 800.0,  "I": 1.8e5},
    {"name": "XL",   "A": 1200.0, "I": 4.2e5},
]

n_state = len(sections)

# ------------------------------------------------------------
# 3. Supports
# ------------------------------------------------------------

# node 0: pin -> ux = uy = 0
# node 3: roller -> uy = 0
fixed_dofs = [0, 1, 2 * 3 + 1]
free_dofs = [d for d in range(n_dof) if d not in fixed_dofs]

# ------------------------------------------------------------
# 4. Load cases
# ------------------------------------------------------------

load_cases = []

f1 = np.zeros(n_dof)
f1[2 * 5 + 1] = -80.0
f1[2 * 6 + 1] = -80.0
load_cases.append(("LC1 vertical", f1))

f2 = np.zeros(n_dof)
f2[2 * 5 + 1] = -50.0
f2[2 * 6 + 1] = -50.0
f2[2 * 7] = 40.0
load_cases.append(("LC2 vertical + wind", f2))

# ------------------------------------------------------------
# 5. Serviceability limits
# ------------------------------------------------------------

top_nodes = [4, 5, 6, 7]
ux_limit = 12.0   # mm
uy_limit = 8.0    # mm

# Penalty weight used only by the master objective.
# FEM feasibility is always checked explicitly.
CONSTRAINT_PENALTY = 1.0e5

# ------------------------------------------------------------
# 6. Precompute member geometry
# ------------------------------------------------------------

member_geom = []

for i, j in members:
    dx = nodes[j, 0] - nodes[i, 0]
    dy = nodes[j, 1] - nodes[i, 1]
    L = math.hypot(dx, dy)
    c = dx / L
    s = dy / L

    # axial extension = g^T u_e
    g = np.array([-c, -s, c, s], dtype=float)
    dofs = [2 * i, 2 * i + 1, 2 * j, 2 * j + 1]

    member_geom.append({
        "L": L,
        "c": c,
        "s": s,
        "g": g,
        "dofs": dofs,
    })


# ============================================================
# 7. Classical FEM subproblem
# ============================================================

def analyze_design(states, return_details=False):
    """
    Exact FEM evaluation for a discrete design.

    states[m] = 0 ... n_state-1

    Returns:
        score      : mass + large violation penalty
        feasible   : True if all checked constraints are satisfied
        mass       : kg
        violation  : normalized squared violation
        details    : optional FEM results
    """

    K = np.zeros((n_dof, n_dof))
    mass = 0.0

    active_data = [None] * n_member

    for m, state in enumerate(states):
        if state == 0:
            continue

        sec = sections[state]
        A = sec["A"]
        I = sec["I"]

        geom = member_geom[m]
        L = geom["L"]
        g = geom["g"]
        dofs = geom["dofs"]

        ke = (E * A / L) * np.outer(g, g)
        K[np.ix_(dofs, dofs)] += ke

        mass += rho * A * L

        active_data[m] = {
            "A": A,
            "I": I,
            "L": L,
            "g": g,
            "dofs": dofs,
            "state": state,
        }

    Kff = K[np.ix_(free_dofs, free_dofs)]

    # Mechanism / singular structure check
    try:
        eigvals = np.linalg.eigvalsh(Kff)
    except np.linalg.LinAlgError:
        return 1.0e12, False, mass, float("inf"), None

    if eigvals[0] < 1.0e-7:
        return 1.0e12, False, mass, float("inf"), None

    violation = 0.0
    load_results = []

    for load_name, f in load_cases:
        try:
            u = np.zeros(n_dof)
            u[free_dofs] = np.linalg.solve(Kff, f[free_dofs])
        except np.linalg.LinAlgError:
            return 1.0e12, False, mass, float("inf"), None

        # displacement constraints
        for node in top_nodes:
            rx = abs(u[2 * node]) / ux_limit
            ry = abs(u[2 * node + 1]) / uy_limit

            violation += max(rx - 1.0, 0.0) ** 2
            violation += max(ry - 1.0, 0.0) ** 2

        member_forces = np.zeros(n_member)
        member_utils = np.zeros(n_member)

        # stress and buckling constraints
        for m, data in enumerate(active_data):
            if data is None:
                continue

            A = data["A"]
            I = data["I"]
            L = data["L"]
            g = data["g"]
            dofs = data["dofs"]

            extension = g @ u[dofs]
            N = (E * A / L) * extension
            member_forces[m] = N

            if N >= 0.0:
                capacity = sigma_t_allow * A
            else:
                stress_capacity = sigma_c_allow * A
                euler_capacity = math.pi**2 * E * I / L**2
                capacity = min(stress_capacity, euler_capacity)

            util = abs(N) / capacity if capacity > 0.0 else float("inf")
            member_utils[m] = util
            violation += max(util - 1.0, 0.0) ** 2

        load_results.append({
            "name": load_name,
            "u": u,
            "N": member_forces,
            "util": member_utils,
        })

    feasible = violation <= 1.0e-12
    score = mass + CONSTRAINT_PENALTY * violation

    details = None
    if return_details:
        details = {
            "K": K,
            "load_results": load_results,
        }

    return score, feasible, mass, violation, details


# ============================================================
# 8. One-hot QUBO encoding
# ============================================================

def bit_index(member, state):
    return member * n_state + state


n_bits = n_member * n_state


def states_to_bits(states):
    x = np.zeros(n_bits, dtype=int)
    for m, state in enumerate(states):
        x[bit_index(m, state)] = 1
    return x


def bits_to_states(x):
    states = []
    for m in range(n_member):
        block = x[m * n_state:(m + 1) * n_state]
        states.append(int(np.argmax(block)))
    return states


# ============================================================
# 9. Local sequential-QUBO model
# ============================================================

def build_local_qubo(incumbent):
    """
    Build a second-order local surrogate around the incumbent.

    For each possible single member-state change:
        a_i(s) = F(i->s) - F0

    For each pair of member changes:
        b_ij(s,t) =
            F(i->s, j->t) - F0 - a_i(s) - a_j(t)

    Then:
        Q(x) = F0 + sum a*x + sum b*x*x

    This is an exact second-order interpolation of all one- and
    two-member perturbations around the incumbent. For larger
    simultaneous changes it is a surrogate, so every proposed
    design is rechecked by FEM.
    """

    base_score, *_ = analyze_design(incumbent)

    h = np.zeros(n_bits)
    J = np.zeros((n_bits, n_bits))

    single_delta = {}

    # linear terms
    for m in range(n_member):
        s0 = incumbent[m]

        for s in range(n_state):
            idx = bit_index(m, s)

            if s == s0:
                h[idx] = 0.0
                single_delta[(m, s)] = 0.0
                continue

            trial = incumbent.copy()
            trial[m] = s

            score, *_ = analyze_design(trial)
            delta = score - base_score

            h[idx] = delta
            single_delta[(m, s)] = delta

    # pairwise interaction terms
    for i in range(n_member):
        for j in range(i + 1, n_member):

            si0 = incumbent[i]
            sj0 = incumbent[j]

            for si in range(n_state):
                if si == si0:
                    continue

                for sj in range(n_state):
                    if sj == sj0:
                        continue

                    trial = incumbent.copy()
                    trial[i] = si
                    trial[j] = sj

                    score, *_ = analyze_design(trial)

                    interaction = (
                        score
                        - base_score
                        - single_delta[(i, si)]
                        - single_delta[(j, sj)]
                    )

                    ii = bit_index(i, si)
                    jj = bit_index(j, sj)

                    J[ii, jj] = interaction
                    J[jj, ii] = interaction

    return {
        "constant": base_score,
        "h": h,
        "J": J,
        "incumbent": incumbent.copy(),
    }


def qubo_energy_states(states, qubo):
    """
    QUBO energy restricted to the one-hot feasible subspace.
    Since each member always has exactly one categorical state,
    no explicit one-hot penalty is needed by this SA solver.

    A gate-model QAOA implementation can instead use:
        lambda * (sum_s x_ms - 1)^2
    for each member.
    """

    chosen = [bit_index(m, s) for m, s in enumerate(states)]

    e = qubo["constant"]

    for idx in chosen:
        e += qubo["h"][idx]

    J = qubo["J"]

    for a in range(len(chosen)):
        ia = chosen[a]
        for b in range(a + 1, len(chosen)):
            ib = chosen[b]
            e += J[ia, ib]

    return float(e)


# ============================================================
# 10. Classical QUBO solver: simulated annealing
# ============================================================

def solve_qubo_sa(
    qubo,
    start,
    restarts=24,
    steps=5000,
    t_start=200.0,
    t_end=0.05,
    max_changes=2,
):
    """
    Simulated annealing over the one-hot state space.
    It minimizes exactly the QUBO energy defined above.

    Replace this function later with a QAOA or quantum-annealer solver.
    """

    global_best = start.copy()
    global_best_e = qubo_energy_states(global_best, qubo)

    candidates = {}

    for restart in range(restarts):

        if restart == 0:
            state = start.copy()
        else:
            state = start.copy()

            # random perturbation of the initial state
            n_mut = RNG.randint(1, min(max_changes, n_member))
            for m in RNG.sample(range(n_member), n_mut):
                state[m] = RNG.randrange(n_state)

        e = qubo_energy_states(state, qubo)

        best = state.copy()
        best_e = e

        for step in range(steps):

            # geometric temperature schedule
            alpha = step / max(steps - 1, 1)
            T = t_start * (t_end / t_start) ** alpha

            m = RNG.randrange(n_member)
            old_state = state[m]

            new_state = RNG.randrange(n_state - 1)
            if new_state >= old_state:
                new_state += 1

            trial = state.copy()
            trial[m] = new_state

            # Trust region: the local QUBO is constructed from exact
            # one- and two-member FEM perturbations. Restricting the
            # master move to at most two changed members prevents
            # uncontrolled extrapolation of the quadratic surrogate.
            n_changed = sum(
                1 for mm in range(n_member)
                if trial[mm] != start[mm]
            )
            if n_changed > max_changes:
                continue

            e_new = qubo_energy_states(trial, qubo)
            de = e_new - e

            if de <= 0.0 or RNG.random() < math.exp(-de / max(T, 1e-12)):
                state = trial
                e = e_new

                if e < best_e:
                    best = state.copy()
                    best_e = e

        key = tuple(best)
        candidates[key] = best_e

        if best_e < global_best_e:
            global_best = best.copy()
            global_best_e = best_e

    # Return unique candidates ranked by QUBO energy
    ranked = sorted(candidates.items(), key=lambda kv: kv[1])

    return [
        (list(states), energy)
        for states, energy in ranked
    ]


# ============================================================
# 11. Classical local improvement
# ============================================================

def greedy_single_member_improvement(start, max_passes=100):
    """
    Cheap local improvement using exact FEM.
    This provides a strong starting incumbent before building QUBOs.
    """

    incumbent = start.copy()
    incumbent_score, *_ = analyze_design(incumbent)

    for _ in range(max_passes):

        best_score = incumbent_score
        best_design = None

        for m in range(n_member):
            old_state = incumbent[m]

            for s in range(n_state):
                if s == old_state:
                    continue

                trial = incumbent.copy()
                trial[m] = s

                score, *_ = analyze_design(trial)

                if score < best_score - 1.0e-9:
                    best_score = score
                    best_design = trial

        if best_design is None:
            break

        incumbent = best_design
        incumbent_score = best_score

    return incumbent


# ============================================================
# 12. Sequential hybrid optimization
# ============================================================

def hybrid_optimize(max_iterations=6):

    # Dense feasible initial structure
    # Start with section L everywhere.
    incumbent = [3] * n_member

    score, feasible, mass, violation, _ = analyze_design(incumbent)

    print("Initial design")
    print(
        f"  score={score:.4f}, mass={mass:.4f} kg, "
        f"feasible={feasible}, violation={violation:.4e}"
    )

    # Improve obvious single-member choices first
    incumbent = greedy_single_member_improvement(incumbent)

    score, feasible, mass, violation, _ = analyze_design(incumbent)

    print("\nAfter exact greedy warm start")
    print(
        f"  score={score:.4f}, mass={mass:.4f} kg, "
        f"feasible={feasible}, violation={violation:.4e}"
    )

    history = [mass]

    for iteration in range(max_iterations):

        print("\n" + "=" * 68)
        print(f"Sequential QUBO iteration {iteration + 1}")
        print("=" * 68)

        qubo = build_local_qubo(incumbent)

        ranked = solve_qubo_sa(
            qubo,
            start=incumbent,
            restarts=24,
            steps=4000,
            max_changes=2,
        )

        incumbent_score, *_ = analyze_design(incumbent)

        improved = False
        best_design = incumbent
        best_score = incumbent_score
        best_info = None

        # Re-check the best QUBO candidates with exact FEM
        for candidate, surrogate_energy in ranked[:20]:

            exact_score, feasible, mass, violation, _ = analyze_design(candidate)

            print(
                f"candidate: QUBO={surrogate_energy:12.3f}  "
                f"FEM={exact_score:12.3f}  "
                f"mass={mass:8.3f} kg  "
                f"feasible={feasible}"
            )

            if exact_score < best_score - 1.0e-8:
                best_score = exact_score
                best_design = candidate
                best_info = (feasible, mass, violation)
                improved = True

        if not improved:
            print("No exact FEM improvement found. Stop.")
            break

        incumbent = greedy_single_member_improvement(best_design)

        score, feasible, mass, violation, _ = analyze_design(incumbent)
        history.append(mass)

        print("\nAccepted design")
        print(
            f"  score={score:.4f}, mass={mass:.4f} kg, "
            f"feasible={feasible}, violation={violation:.4e}"
        )

    return incumbent, history


# ============================================================
# 13. Reporting
# ============================================================

def report_design(states):

    score, feasible, mass, violation, details = analyze_design(
        states,
        return_details=True,
    )

    print("\n" + "=" * 68)
    print("FINAL DESIGN")
    print("=" * 68)

    print(f"score      = {score:.6f}")
    print(f"mass       = {mass:.6f} kg")
    print(f"feasible   = {feasible}")
    print(f"violation  = {violation:.6e}")

    print("\nSelected members")
    print("-" * 68)

    for m, state in enumerate(states):
        if state == 0:
            continue

        i, j = members[m]
        sec = sections[state]

        print(
            f"M{m:02d}: {i}-{j}  "
            f"{sec['name']:>2s}  "
            f"A={sec['A']:7.1f} mm^2  "
            f"I={sec['I']:9.1f} mm^4  "
            f"L={member_geom[m]['L']:8.1f} mm"
        )

    if details is not None:
        for result in details["load_results"]:

            print("\n" + result["name"])
            print("-" * 68)

            u = result["u"]
            N = result["N"]
            util = result["util"]

            print("Top-node displacements")
            for node in top_nodes:
                print(
                    f"  node {node}: "
                    f"ux={u[2*node]:9.4f} mm, "
                    f"uy={u[2*node+1]:9.4f} mm"
                )

            print("Active member forces / utilization")
            for m, state in enumerate(states):
                if state == 0:
                    continue

                print(
                    f"  M{m:02d}: "
                    f"N={N[m]:9.3f} kN, "
                    f"util={util[m]:7.4f}"
                )

    return score, feasible, mass, violation, details


# ============================================================
# 14. Plotting
# ============================================================

def plot_design(states, details=None, deformation_scale=30.0):

    plt.figure(figsize=(10, 4))

    # candidate ground structure
    for i, j in members:
        plt.plot(
            [nodes[i, 0], nodes[j, 0]],
            [nodes[i, 1], nodes[j, 1]],
            "--",
            linewidth=0.6,
            alpha=0.15,
        )

    # selected members
    for m, state in enumerate(states):
        if state == 0:
            continue

        i, j = members[m]
        A = sections[state]["A"]

        linewidth = 1.0 + 4.0 * A / sections[-1]["A"]

        plt.plot(
            [nodes[i, 0], nodes[j, 0]],
            [nodes[i, 1], nodes[j, 1]],
            linewidth=linewidth,
        )

    plt.scatter(nodes[:, 0], nodes[:, 1], zorder=10)

    for i, (x, y) in enumerate(nodes):
        plt.text(x + 35, y + 35, str(i))

    plt.axis("equal")
    plt.grid(True)
    plt.xlabel("X [mm]")
    plt.ylabel("Y [mm]")
    plt.title("Hybrid QUBO + FEM optimized truss")
    plt.tight_layout()
    plt.show()

    if details is None:
        return

    for result in details["load_results"]:

        u = result["u"]
        deformed = nodes + deformation_scale * u.reshape(-1, 2)

        plt.figure(figsize=(10, 4))

        for m, state in enumerate(states):
            if state == 0:
                continue

            i, j = members[m]

            # undeformed
            plt.plot(
                [nodes[i, 0], nodes[j, 0]],
                [nodes[i, 1], nodes[j, 1]],
                "--",
                linewidth=0.8,
                alpha=0.35,
            )

            # deformed
            plt.plot(
                [deformed[i, 0], deformed[j, 0]],
                [deformed[i, 1], deformed[j, 1]],
                linewidth=2.0,
            )

        plt.axis("equal")
        plt.grid(True)
        plt.xlabel("X [mm]")
        plt.ylabel("Y [mm]")
        plt.title(
            f"{result['name']}  "
            f"(deformation x{deformation_scale:g})"
        )
        plt.tight_layout()
        plt.show()


# ============================================================
# 15. Run
# ============================================================

if __name__ == "__main__":

    best_states, history = hybrid_optimize(
        max_iterations=5
    )

    score, feasible, mass, violation, details = report_design(
        best_states
    )

    print("\nState vector")
    print(best_states)

    print("\nMass history [kg]")
    print(history)

    plot_design(
        best_states,
        details=details,
        deformation_scale=30.0,
    )
