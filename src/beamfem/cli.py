"""Command-line entry point for audited discrete structural optimization."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .io import build_discrete_problem, load_problem_spec, write_result_json
from .optimize.backends import (
    ExactBackend,
    GreedyBackend,
    QAOABackend,
    QiskitNotInstalledError,
    SequentialQUBOOptimizer,
    SimulatedAnnealingBackend,
    SolverLimits,
)
from .optimize.qubo import AdaptivePenalty, LocalQUBOBuilder, TrustRegion
from .validation import build_audit_metadata, diagnose_problem_spec


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run reproducible discrete frame optimization from JSON/YAML."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("exact", "greedy", "sa", "qaoa"), default="greedy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time-limit", type=float)
    parser.add_argument("--max-evaluations", type=int)
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--exact-max-combinations", type=int, default=200_000)
    parser.add_argument("--penalty", type=float, default=1.0e6)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--trust-radius", type=int, default=2)
    parser.add_argument("--sa-sweeps", type=int, default=1500)
    parser.add_argument("--sa-restarts", type=int, default=8)
    parser.add_argument("--qaoa-reps", type=int, default=1)
    parser.add_argument("--qaoa-maxiter", type=int, default=100)
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--fallback", choices=("none", "greedy", "sa"), default="greedy")
    return parser


def _sequential(problem, args, quantum: bool):
    radius = max(1, args.trust_radius)
    builder = LocalQUBOBuilder(
        problem,
        max_candidates=args.candidates,
        trust_region=TrustRegion(radius=radius, minimum=1, maximum=max(8, radius)),
        penalty=AdaptivePenalty(value=args.penalty),
    )
    if quantum:
        solver = QAOABackend(
            reps=args.qaoa_reps,
            shots=args.shots,
            seed=args.seed,
            maxiter=args.qaoa_maxiter,
        )
    else:
        solver = SimulatedAnnealingBackend(
            sweeps=args.sa_sweeps, restarts=args.sa_restarts, seed=args.seed
        )
    return SequentialQUBOOptimizer(
        solver, builder, max_iterations=args.max_iterations
    )


def _backend(problem, args):
    if args.backend == "exact":
        return ExactBackend(max_combinations=args.exact_max_combinations)
    if args.backend == "greedy":
        return GreedyBackend(penalty=args.penalty, pairwise=True)
    if args.backend == "sa":
        return _sequential(problem, args, False)
    return _sequential(problem, args, True)


def _fallback(problem, args):
    if args.fallback == "greedy":
        return GreedyBackend(penalty=args.penalty, pairwise=True)
    if args.fallback == "sa":
        return _sequential(problem, args, False)
    return None


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    spec = load_problem_spec(args.input)
    preflight = diagnose_problem_spec(spec.data)
    built = build_discrete_problem(spec)
    limits = SolverLimits(
        max_evaluations=args.max_evaluations,
        max_iterations=args.max_iterations,
        time_limit=args.time_limit,
    )
    warnings = list(preflight.warnings)
    selected_backend = args.backend
    try:
        result = _backend(built.problem, args).solve(built.problem, limits=limits)
    except (QiskitNotInstalledError, RuntimeError) as exc:
        fallback = _fallback(built.problem, args) if args.backend == "qaoa" else None
        if fallback is None:
            raise
        warnings.append(f"QAOA failed; used {args.fallback} fallback: {exc}")
        selected_backend = args.fallback
        result = fallback.solve(built.problem, limits=limits)

    audit = build_audit_metadata(
        solver=selected_backend,
        seed=args.seed,
        solver_settings={
            key: value for key, value in vars(args).items()
            if key not in {"input", "output"}
        },
        warnings=warnings,
        repository=Path(__file__).resolve().parents[2],
    )
    payload = {
        "problem": spec.data.get("name", args.input.stem),
        "input": str(args.input.resolve()),
        "node_ids": dict(built.node_ids),
        "member_ids": dict(built.member_ids),
        "preflight": [
            {
                "code": item.code,
                "severity": item.severity.value,
                "message": item.message,
                "location": item.location,
            }
            for item in preflight.diagnostics
        ],
        "optimization": result,
    }
    write_result_json(payload, args.output, audit=audit)
    print(
        f"{result.backend}: objective={result.objective:.6g}, "
        f"feasible={result.feasible}, evaluations={result.evaluations}, "
        f"output={args.output}"
    )
    return 0 if result.feasible else 2


if __name__ == "__main__":
    sys.exit(main())
