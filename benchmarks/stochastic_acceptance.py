"""Generate repeated-seed acceptance evidence for stochastic optimizers.

Every backend is evaluated through the same structural problem adapter and
classical FEM evaluator.  The output deliberately retains every individual
run as well as aggregate best/median/worst statistics.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
import statistics
import subprocess
from typing import Any, Iterable

from beamfem.io import validate_problem_spec

from .quantum_truss.compare import ComparisonRecord, run_comparison
from .quantum_truss.generate_cases import generate_case


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def summarize_runs(
    records: Iterable[tuple[int, ComparisonRecord]],
    *,
    minimum_seeds: int,
    minimum_feasibility_rate: float,
) -> dict[str, Any]:
    """Return JSON-safe run records and aggregate acceptance statistics."""

    pairs = list(records)
    successful = [record for _, record in pairs if record.status != "error"]
    feasible = [record for record in successful if record.feasible is True]
    objectives = [float(record.fem_score) for record in feasible if record.fem_score is not None]
    masses = [float(record.mass) for record in feasible if record.mass is not None]
    rate = len(feasible) / len(pairs) if pairs else 0.0

    def spread(values: list[float]) -> dict[str, float | None]:
        return {
            "best": min(values) if values else None,
            "median": statistics.median(values) if values else None,
            "worst": max(values) if values else None,
        }

    return {
        "runs": [
            {
                "seed": seed,
                "status": record.status,
                "qubo_energy": record.qubo_energy,
                "fem_score": record.fem_score,
                "mass": record.mass,
                "feasible": record.feasible,
                "governing_constraint": record.governing_constraint,
                "evaluations": record.evaluations,
                "runtime_seconds": record.runtime_seconds,
                "message": record.message,
            }
            for seed, record in pairs
        ],
        "seed_count": len(pairs),
        "successful_run_count": len(successful),
        "feasible_run_count": len(feasible),
        "feasibility_rate": rate,
        "objective": spread(objectives),
        "mass": spread(masses),
        "evaluation_budget": {
            "best": min((r.evaluations for _, r in pairs), default=0),
            "median": statistics.median([r.evaluations for _, r in pairs]) if pairs else 0,
            "worst": max((r.evaluations for _, r in pairs), default=0),
        },
        "acceptance": {
            "minimum_seeds": minimum_seeds,
            "minimum_feasibility_rate": minimum_feasibility_rate,
            "enough_seeds": len(pairs) >= minimum_seeds,
            "feasibility_rate_met": rate >= minimum_feasibility_rate,
            "all_runs_completed": len(successful) == len(pairs),
            "passed": (
                len(pairs) >= minimum_seeds
                and rate >= minimum_feasibility_rate
                and len(successful) == len(pairs)
            ),
        },
    }


def collect_evidence(
    seeds: Iterable[int] = range(10),
    *,
    minimum_feasibility_rate: float = 0.9,
) -> dict[str, Any]:
    seed_values = tuple(int(seed) for seed in seeds)
    if len(set(seed_values)) != len(seed_values):
        raise ValueError("seeds must be unique")
    if not seed_values:
        raise ValueError("at least one seed is required")
    if not 0.0 <= minimum_feasibility_rate <= 1.0:
        raise ValueError("minimum_feasibility_rate must be between zero and one")

    spec = validate_problem_spec(generate_case("small"))
    backend_settings: dict[str, dict[str, Any]] = {
        "sa": {
            "iterations": 1,
            "candidates": 2,
            "sa_sweeps": 200,
            "sa_restarts": 4,
            "max_evaluations": 1_000,
        },
        "qaoa": {
            "iterations": 1,
            "candidates": 2,
            "qaoa_reps": 1,
            "qaoa_maxiter": 10,
            "shots": 256,
            "max_evaluations": 1_000,
        },
    }
    backends: dict[str, Any] = {}
    for backend, settings in backend_settings.items():
        runs: list[tuple[int, ComparisonRecord]] = []
        for seed in seed_values:
            record = run_comparison(spec, (backend,), seed=seed, **settings)[0]
            runs.append((seed, record))
        summary = summarize_runs(
            runs,
            minimum_seeds=10,
            minimum_feasibility_rate=minimum_feasibility_rate,
        )
        summary["settings"] = settings
        backends[backend] = summary

    passed = all(item["acceptance"]["passed"] for item in backends.values())
    return {
        "evidence_schema_version": "1.0",
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "git_commit": _git_commit(),
        },
        "problem": "small quantum-truss benchmark",
        "common_authoritative_evaluator": "classical beamfem FEM",
        "seeds": list(seed_values),
        "backends": backends,
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("validation/stochastic_evidence.json"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    args = parser.parse_args()
    evidence = collect_evidence(args.seeds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
