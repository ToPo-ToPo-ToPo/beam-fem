#!/usr/bin/env python3
"""Compare SA and Qiskit QAOA on the original truss QUBO/FEM model.

This module imports the user's original ``legacy/hybrid_qubo_truss_sa.py`` and
therefore uses its geometry, section catalogue, two load cases, FEM analysis,
stress/buckling/displacement checks, penalty score and greedy warm start.

Both solvers receive the same compact local QUBO.  Six candidate member-state
moves plus two trust-region slack bits require only eight qubits by default.
Every returned design is re-evaluated by the original classical FEM function.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from legacy import hybrid_qubo_truss_sa as original


@dataclass(frozen=True)
class Move:
    member: int
    new_state: int
    label: str


@dataclass
class QUBO:
    """E(x) = c + linear*x + sum(i<j) quadratic[i,j]*x_i*x_j."""

    constant: float
    linear: np.ndarray
    quadratic: np.ndarray
    design_variable_count: int
    labels: list[str]

    @property
    def size(self) -> int:
        return len(self.linear)

    def energy(self, bits: Sequence[int]) -> float:
        x = np.asarray(bits, dtype=float)
        if x.shape != (self.size,):
            raise ValueError(f"expected {self.size} bits, got {x.shape}")
        return float(
            self.constant
            + self.linear @ x
            + np.sum(np.triu(self.quadratic, 1) * np.outer(x, x))
        )


@dataclass
class SolverResult:
    solver: str
    bits: list[int]
    qubo_energy: float
    runtime_s: float
    metadata: dict[str, Any]


def initial_local_design() -> list[int]:
    """Reproduce the original L-section start and exact greedy warm start."""
    return original.greedy_single_member_improvement([3] * original.n_member)


def apply_moves(
    baseline: Sequence[int], moves: Sequence[Move], bits: Sequence[int]
) -> list[int]:
    design = list(baseline)
    for move, bit in zip(moves, bits):
        if bit:
            design[move.member] = move.new_state
    return design


def choose_moves(
    baseline: Sequence[int], count: int = 6
) -> list[Move]:
    """Select a small move set while retaining the best two-member interaction."""
    if count < 2:
        raise ValueError("at least two design moves are required")
    base_score = original.analyze_design(baseline)[0]
    ranked: list[tuple[float, Move]] = []
    for member, old_state in enumerate(baseline):
        for new_state in range(original.n_state):
            if new_state == old_state:
                continue
            trial = list(baseline)
            trial[member] = new_state
            delta = original.analyze_design(trial)[0] - base_score
            old_name = original.sections[old_state]["name"]
            new_name = original.sections[new_state]["name"]
            ranked.append(
                (abs(delta), Move(member, new_state, f"M{member}:{old_name}->{new_name}"))
            )
    ranked.sort(key=lambda item: item[0])

    # The original local QUBO evaluates all two-member FEM perturbations.  The
    # compact QAOA version keeps the best interaction, then fills remaining
    # qubits with the closest single-member alternatives on distinct members.
    candidates = [move for _, move in ranked]
    best_pair: tuple[Move, Move] | None = None
    best_pair_score = base_score
    for i, first in enumerate(candidates):
        for second in candidates[i + 1 :]:
            if first.member == second.member:
                continue
            trial = list(baseline)
            trial[first.member] = first.new_state
            trial[second.member] = second.new_state
            score = original.analyze_design(trial)[0]
            if score < best_pair_score:
                best_pair_score = score
                best_pair = first, second

    chosen: list[Move] = []
    used_members: set[int] = set()
    if best_pair is not None:
        chosen.extend(best_pair)
        used_members.update(move.member for move in best_pair)
    for _, move in ranked:
        if move.member in used_members:
            continue
        chosen.append(move)
        used_members.add(move.member)
        if len(chosen) == count:
            break
    if len(chosen) < count:
        raise ValueError(f"cannot select {count} moves on distinct members")
    return chosen


def _add_square_penalty(
    qubo: QUBO, coefficients: Sequence[float], rhs: float, penalty: float
) -> None:
    """Add penalty*(sum(a_i*x_i)-rhs)^2, using x_i^2=x_i."""
    a = np.asarray(coefficients, dtype=float)
    qubo.constant += penalty * rhs * rhs
    qubo.linear += penalty * (a * a - 2.0 * rhs * a)
    for i in range(qubo.size):
        for j in range(i + 1, qubo.size):
            qubo.quadratic[i, j] += 2.0 * penalty * a[i] * a[j]


def build_compact_qubo(
    baseline: Sequence[int], moves: Sequence[Move]
) -> QUBO:
    """Fit the original FEM score for every selected one/two-move design."""
    base_score = original.analyze_design(baseline)[0]
    score_cap = base_score + 1000.0

    def merit(bits: Sequence[int]) -> float:
        # Mechanisms score 1e12 in the original model.  Clipping the regression
        # target avoids a badly scaled Hamiltonian; final FEM checks are never
        # clipped.  Valid one/two-move scores, including the optimum, are exact.
        return min(original.analyze_design(apply_moves(baseline, moves, bits))[0], score_cap)

    n = len(moves)
    single = np.empty(n)
    linear = np.zeros(n + 2)
    quadratic = np.zeros((n + 2, n + 2))
    for i in range(n):
        bits = np.zeros(n, dtype=int)
        bits[i] = 1
        single[i] = merit(bits)
        linear[i] = single[i] - base_score
    for i in range(n):
        for j in range(i + 1, n):
            bits = np.zeros(n, dtype=int)
            bits[[i, j]] = 1
            pair = merit(bits)
            quadratic[i, j] = pair - single[i] - single[j] + base_score

    labels = [move.label for move in moves] + ["trust_slack_1", "trust_slack_2"]
    qubo = QUBO(base_score, linear, quadratic, n, labels)

    # sum(move bits) <= 2, encoded as sum(x)+s1+2*s2=2.
    coefficient_scale = max(
        1.0,
        float(np.max(np.abs(linear[:n]))),
        float(np.max(np.abs(quadratic[:n, :n]))),
    )
    _add_square_penalty(
        qubo,
        [1.0] * n + [1.0, 2.0],
        rhs=2.0,
        penalty=20.0 * coefficient_scale,
    )

    # Scaling changes no minimizer and makes QAOA angles numerically practical.
    normalizer = max(
        1.0,
        float(np.max(np.abs(qubo.linear))),
        float(np.max(np.abs(qubo.quadratic))),
    )
    qubo.constant /= normalizer
    qubo.linear /= normalizer
    qubo.quadratic /= normalizer
    return qubo


def solve_qubo_sa(
    qubo: QUBO,
    *,
    seed: int = 123,
    reads: int = 64,
    sweeps: int = 1500,
) -> SolverResult:
    """Classical simulated annealing on the same binary QUBO used by QAOA."""
    rng = np.random.default_rng(seed)
    started = time.perf_counter()
    best_bits = np.zeros(qubo.size, dtype=int)
    best_energy = qubo.energy(best_bits)
    for _ in range(reads):
        bits = rng.integers(0, 2, size=qubo.size, dtype=int)
        energy = qubo.energy(bits)
        for step in range(sweeps):
            alpha = step / max(sweeps - 1, 1)
            temperature = 20.0 * (0.01 / 20.0) ** alpha
            index = int(rng.integers(qubo.size))
            trial = bits.copy()
            trial[index] ^= 1
            trial_energy = qubo.energy(trial)
            delta = trial_energy - energy
            scale = max(1.0, abs(energy), abs(trial_energy))
            if delta <= 0.0 or rng.random() < math.exp(-delta / (temperature * scale)):
                bits, energy = trial, trial_energy
            if energy < best_energy:
                best_bits, best_energy = bits.copy(), energy
    return SolverResult(
        "SA",
        best_bits.tolist(),
        float(best_energy),
        time.perf_counter() - started,
        {"seed": seed, "reads": reads, "sweeps": sweeps},
    )


def make_qaoa_execution(
    backend: str, seed: int, shots: int
) -> tuple[Any, Any | None]:
    """Create a current Qiskit Sampler V2 and any required pass manager."""
    if backend == "statevector":
        from qiskit.primitives import StatevectorSampler

        return StatevectorSampler(default_shots=shots, seed=seed), None
    if backend == "aer":
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from qiskit_aer import AerSimulator
        from qiskit_aer.primitives import SamplerV2

        simulator = AerSimulator()
        return (
            SamplerV2(seed=seed, default_shots=shots),
            generate_preset_pass_manager(optimization_level=2, backend=simulator),
        )
    raise ValueError(f"unknown backend: {backend}")


def solve_qubo_qaoa(
    qubo: QUBO,
    *,
    sampler: Any | None = None,
    pass_manager: Any | None = None,
    reps: int = 1,
    maxiter: int = 100,
    seed: int = 123,
    shots: int = 2048,
) -> SolverResult:
    """Run Qiskit Optimization QAOA with an injectable execution backend."""
    try:
        from qiskit.primitives import StatevectorSampler
        from qiskit_optimization import QuadraticProgram
        from qiskit_optimization.algorithms import MinimumEigenOptimizer
        from qiskit_optimization.minimum_eigensolvers import QAOA
        from qiskit_optimization.optimizers import COBYLA
        from qiskit_optimization.utils import algorithm_globals
    except ImportError as exc:
        raise RuntimeError(
            "Install qiskit>=2.1,<3 and qiskit-optimization>=0.7,<0.8, "
            "or run with --solver sa."
        ) from exc

    algorithm_globals.random_seed = seed
    if sampler is None:
        sampler = StatevectorSampler(default_shots=shots, seed=seed)
    problem = QuadraticProgram("original_truss_local_master")
    for index in range(qubo.size):
        problem.binary_var(f"x{index}")
    linear = {i: float(v) for i, v in enumerate(qubo.linear) if v != 0.0}
    quadratic = {
        (i, j): float(qubo.quadratic[i, j])
        for i in range(qubo.size)
        for j in range(i + 1, qubo.size)
        if qubo.quadratic[i, j] != 0.0
    }
    problem.minimize(
        constant=qubo.constant, linear=linear, quadratic=quadratic
    )

    started = time.perf_counter()
    solver = QAOA(
        sampler=sampler,
        optimizer=COBYLA(maxiter=maxiter),
        reps=reps,
        initial_point=np.full(2 * reps, 0.5),
        pass_manager=pass_manager,
    )
    result = MinimumEigenOptimizer(solver).solve(problem)
    samples = list(result.samples or [])
    if samples:
        selected = min(
            samples,
            key=lambda sample: qubo.energy(np.rint(sample.x).astype(int)),
        )
        bits = np.rint(selected.x).astype(int)
        probability = float(selected.probability)
    else:
        bits = np.rint(result.x).astype(int)
        probability = float("nan")
    return SolverResult(
        "QAOA",
        bits.tolist(),
        qubo.energy(bits),
        time.perf_counter() - started,
        {
            "seed": seed,
            "shots": shots,
            "reps": reps,
            "maxiter": maxiter,
            "selected_probability": probability,
            "sample_count": len(samples),
        },
    )


def load_sampler_factory(spec: str) -> tuple[Any, Any | None]:
    module_name, separator, function_name = spec.partition(":")
    if not separator:
        raise ValueError("factory must be package.module:function")
    factory: Callable[[], Any] = getattr(
        importlib.import_module(module_name), function_name
    )
    made = factory()
    if isinstance(made, tuple):
        if len(made) != 2:
            raise ValueError("factory tuple must be (sampler, pass_manager)")
        return made
    return made, None


def comparison_row(
    baseline: Sequence[int],
    moves: Sequence[Move],
    qubo: QUBO,
    solved: SolverResult,
) -> dict[str, Any]:
    design_bits = solved.bits[: qubo.design_variable_count]
    design = apply_moves(baseline, moves, design_bits)
    score, feasible, mass, violation, _ = original.analyze_design(design)
    return {
        "solver": solved.solver,
        "qubo_energy": solved.qubo_energy,
        "fem_score": score,
        "mass_kg": mass,
        "feasible": feasible,
        "violation": violation,
        "selected_moves": [
            moves[i].label for i, bit in enumerate(design_bits) if bit
        ],
        "bits": solved.bits,
        "design_states": design,
        "runtime_s": solved.runtime_s,
        "metadata": solved.metadata,
    }


def print_comparison(rows: Sequence[dict[str, Any]]) -> None:
    print("\nSolver comparison (same QUBO, original exact FEM check)")
    print("solver    QUBO energy (norm.)    FEM score     mass [kg]  feasible  runtime [s]")
    print("--------  -------------------  ------------  ------------  --------  -----------")
    for row in rows:
        print(
            f"{row['solver']:<8}  {row['qubo_energy']:>19.8f}  "
            f"{row['fem_score']:>12.6f}  {row['mass_kg']:>12.6f}  "
            f"{str(row['feasible']):>8}  {row['runtime_s']:>11.3f}"
        )
        print("  moves:", ", ".join(row["selected_moves"]) or "none")
        print(f"  FEM violation: {row['violation']:.6e}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Original truss model: SA versus Qiskit QAOA",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--solver", choices=("sa", "qaoa", "both"), default="both")
    parser.add_argument("--moves", type=int, default=6)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--sa-reads", type=int, default=64)
    parser.add_argument("--sa-sweeps", type=int, default=1500)
    parser.add_argument("--qaoa-reps", type=int, default=1)
    parser.add_argument("--qaoa-maxiter", type=int, default=100)
    parser.add_argument("--shots", type=int, default=2048)
    parser.add_argument("--qaoa-backend", choices=("statevector", "aer"), default="statevector")
    parser.add_argument("--sampler-factory", help="package.module:function returning sampler or (sampler, pass_manager)")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--plot", action="store_true", help="show the selected design after comparison")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    initial = [3] * original.n_member
    initial_score, initial_feasible, initial_mass, _, _ = original.analyze_design(initial)
    baseline = initial_local_design()
    base_score, base_feasible, base_mass, base_violation, _ = original.analyze_design(baseline)
    print(
        f"Original dense design: score={initial_score:.6f}, "
        f"mass={initial_mass:.6f} kg, feasible={initial_feasible}"
    )
    print(
        f"Original greedy design: score={base_score:.6f}, "
        f"mass={base_mass:.6f} kg, feasible={base_feasible}"
    )

    moves = choose_moves(baseline, args.moves)
    qubo = build_compact_qubo(baseline, moves)
    print(
        f"Master: {len(moves)} design bits + 2 trust-region slack bits "
        f"= {qubo.size} qubits"
    )

    solved: list[SolverResult] = []
    if args.solver in ("sa", "both"):
        solved.append(
            solve_qubo_sa(
                qubo,
                seed=args.seed,
                reads=args.sa_reads,
                sweeps=args.sa_sweeps,
            )
        )
    if args.solver in ("qaoa", "both"):
        try:
            if args.sampler_factory:
                sampler, pass_manager = load_sampler_factory(args.sampler_factory)
            else:
                sampler, pass_manager = make_qaoa_execution(
                    args.qaoa_backend, args.seed, args.shots
                )
            solved.append(
                solve_qubo_qaoa(
                    qubo,
                    sampler=sampler,
                    pass_manager=pass_manager,
                    reps=args.qaoa_reps,
                    maxiter=args.qaoa_maxiter,
                    seed=args.seed,
                    shots=args.shots,
                )
            )
        except (ImportError, RuntimeError) as exc:
            print(f"QAOA unavailable: {exc}", file=sys.stderr)
            if args.solver == "qaoa":
                return 2

    rows = [comparison_row(baseline, moves, qubo, result) for result in solved]
    print_comparison(rows)
    if args.json:
        payload = {
            "original_dense": {
                "score": initial_score,
                "mass_kg": initial_mass,
                "feasible": initial_feasible,
            },
            "greedy_baseline": {
                "score": base_score,
                "mass_kg": base_mass,
                "feasible": base_feasible,
                "violation": base_violation,
                "states": baseline,
            },
            "qubo": {
                "size": qubo.size,
                "design_variables": qubo.design_variable_count,
                "labels": qubo.labels,
            },
            "results": rows,
        }
        args.json.write_text(
            json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8"
        )
        print(f"\nJSON written to {args.json}")
    if args.plot and rows:
        original.plot_design(rows[0]["design_states"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
