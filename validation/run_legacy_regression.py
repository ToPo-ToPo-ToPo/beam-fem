"""Reproduce the original 159/91/88 kg truss optimization milestones."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "quantum_truss_qaoa"
if str(EXPERIMENT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT))

from experiments.quantum_truss_qaoa.hybrid_qubo_truss_compare import (  # noqa: E402
    build_compact_qubo,
    choose_moves,
    comparison_row,
    initial_local_design,
    original,
    solve_qubo_sa,
)


EXPECTED = {
    "dense_mass_kg": 159.04491637994093,
    "greedy_mass_kg": 91.71915091622414,
    "local_qubo_mass_kg": 88.70327940039327,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def collect_evidence(tolerance_kg: float = 1e-8) -> dict:
    dense = original.analyze_design([3] * original.n_member)
    baseline = initial_local_design()
    greedy = original.analyze_design(baseline)
    moves = choose_moves(baseline)
    qubo = build_compact_qubo(baseline, moves)
    local = comparison_row(
        baseline,
        moves,
        qubo,
        solve_qubo_sa(qubo, seed=123, reads=32, sweeps=1000),
    )
    observed = {
        "dense_mass_kg": float(dense[2]),
        "greedy_mass_kg": float(greedy[2]),
        "local_qubo_mass_kg": float(local["mass_kg"]),
    }
    checks = {
        key: abs(observed[key] - expected) <= tolerance_kg
        for key, expected in EXPECTED.items()
    }
    source_files = [
        EXPERIMENT / "legacy" / "hybrid_qubo_truss_sa.py",
        EXPERIMENT / "hybrid_qubo_truss_compare.py",
    ]
    return {
        "evidence_schema_version": "1.0",
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "git_commit": _git_commit(),
        },
        "source_sha256": {str(path.relative_to(ROOT)): _sha256(path) for path in source_files},
        "expected": EXPECTED,
        "observed": observed,
        "tolerance_kg": tolerance_kg,
        "feasibility": {
            "dense": bool(dense[1]),
            "greedy": bool(greedy[1]),
            "local_qubo": bool(local["feasible"]),
        },
        "sa": {
            "seed": 123,
            "reads": 32,
            "sweeps": 1000,
            "qubo_energy": local["qubo_energy"],
            "selected_moves": local["selected_moves"],
            "fem_score": local["fem_score"],
            "mass_kg": local["mass_kg"],
            "feasible": local["feasible"],
        },
        "checks": checks,
        "passed": all(checks.values()) and dense[1] and greedy[1] and local["feasible"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "validation" / "legacy_regression_evidence.json"
    )
    args = parser.parse_args()
    evidence = collect_evidence()
    args.output.write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
