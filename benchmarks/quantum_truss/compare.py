"""Compare classical and QAOA backends through the same FEM evaluator."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from beamfem.io import build_discrete_problem, load_problem_spec, write_result_json
from beamfem.optimize.backends import (
    ExactBackend,
    GreedyBackend,
    QAOABackend,
    SequentialQUBOOptimizer,
    SimulatedAnnealingBackend,
    SolverLimits,
)
from beamfem.optimize.qubo import AdaptivePenalty, LocalQUBOBuilder, TrustRegion


@dataclass(frozen=True)
class ComparisonRecord:
    backend: str
    status: str
    qubo_energy: float | None
    fem_score: float | None
    mass: float | None
    feasible: bool | None
    governing_constraint: str | None
    evaluations: int
    runtime_seconds: float
    message: str


def _sequential(problem: Any, name: str, settings: dict[str, Any]):
    builder = LocalQUBOBuilder(
        problem,
        max_candidates=settings["candidates"],
        trust_region=TrustRegion(radius=settings["trust_radius"]),
        penalty=AdaptivePenalty(value=settings["penalty"]),
    )
    if name == "sa":
        solver = SimulatedAnnealingBackend(
            sweeps=settings["sa_sweeps"],
            restarts=settings["sa_restarts"],
            seed=settings["seed"],
        )
    else:
        solver = QAOABackend(
            reps=settings["qaoa_reps"],
            maxiter=settings["qaoa_maxiter"],
            shots=settings["shots"],
            seed=settings["seed"],
        )
    return SequentialQUBOOptimizer(solver, builder, max_iterations=settings["iterations"])


def run_comparison(spec, backends: Iterable[str], **overrides) -> list[ComparisonRecord]:
    """Run isolated backend evaluations and return directly comparable fields."""

    settings = {
        "seed": 42,
        "iterations": 5,
        "max_evaluations": 10_000,
        "exact_max_combinations": 200_000,
        "penalty": 1.0e6,
        "candidates": 8,
        "trust_radius": 2,
        "sa_sweeps": 1500,
        "sa_restarts": 8,
        "qaoa_reps": 1,
        "qaoa_maxiter": 100,
        "shots": 1024,
    }
    settings.update(overrides)
    records: list[ComparisonRecord] = []
    for name in backends:
        # Rebuild to keep FEM caches and evaluation counts independent.
        problem = build_discrete_problem(spec).problem
        if name == "exact":
            backend = ExactBackend(max_combinations=settings["exact_max_combinations"])
        elif name == "greedy":
            backend = GreedyBackend(penalty=settings["penalty"], pairwise=True)
        elif name in {"sa", "qaoa"}:
            backend = _sequential(problem, name, settings)
        else:
            raise ValueError(f"unsupported backend: {name}")
        try:
            result = backend.solve(
                problem,
                limits=SolverLimits(
                    max_iterations=settings["iterations"],
                    max_evaluations=settings["max_evaluations"],
                ),
            )
            evaluation = result.evaluation
            governing = getattr(evaluation, "governing_constraint", None)
            records.append(ComparisonRecord(
                backend=name,
                status=result.status,
                qubo_energy=result.solver_metadata.get("qubo_energy"),
                fem_score=result.objective,
                mass=getattr(evaluation, "mass", None),
                feasible=result.feasible,
                governing_constraint=(
                    None if governing is None else governing.constraint_id
                ),
                evaluations=result.evaluations,
                runtime_seconds=result.runtime,
                message=result.message,
            ))
        except Exception as exc:  # One unavailable backend must not erase other results.
            records.append(ComparisonRecord(
                backend=name,
                status="error",
                qubo_energy=None,
                fem_score=None,
                mass=None,
                feasible=None,
                governing_constraint=None,
                evaluations=0,
                runtime_seconds=0.0,
                message=f"{type(exc).__name__}: {exc}",
            ))
    return records


def _write_csv(records: list[ComparisonRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(record) for record in records]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--backends", nargs="+", choices=("exact", "greedy", "sa", "qaoa"),
                        default=("greedy", "sa", "qaoa"))
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--max-evaluations", type=int, default=10_000)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--sa-sweeps", type=int, default=1500)
    parser.add_argument("--sa-restarts", type=int, default=8)
    parser.add_argument("--qaoa-reps", type=int, default=1)
    parser.add_argument("--qaoa-maxiter", type=int, default=100)
    parser.add_argument("--shots", type=int, default=1024)
    args = parser.parse_args()
    spec = load_problem_spec(args.input)
    records = run_comparison(
        spec,
        args.backends,
        seed=args.seed,
        iterations=args.iterations,
        max_evaluations=args.max_evaluations,
        candidates=args.candidates,
        sa_sweeps=args.sa_sweeps,
        sa_restarts=args.sa_restarts,
        qaoa_reps=args.qaoa_reps,
        qaoa_maxiter=args.qaoa_maxiter,
        shots=args.shots,
    )
    write_result_json({"problem": spec.data.get("name"), "comparison": records}, args.output_json)
    _write_csv(records, args.output_csv)
    for record in records:
        print(
            f"{record.backend:>7}: FEM={record.fem_score!s:<12} "
            f"mass={record.mass!s:<12} feasible={record.feasible!s:<5} "
            f"QUBO={record.qubo_energy!s}"
        )


if __name__ == "__main__":
    main()
