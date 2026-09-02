from dataclasses import replace

import pytest

from benchmarks.quantum_truss.compare import ComparisonRecord
from benchmarks.stochastic_acceptance import summarize_runs


def _record(*, feasible=True, objective=10.0, evaluations=20):
    return ComparisonRecord(
        backend="sa",
        status="success",
        qubo_energy=-1.0,
        fem_score=objective,
        mass=objective,
        feasible=feasible,
        governing_constraint=None,
        evaluations=evaluations,
        runtime_seconds=0.1,
        message="ok",
    )


def test_repeated_seed_summary_reports_every_run_and_spread():
    runs = [(seed, _record(objective=float(seed + 1), evaluations=seed + 10)) for seed in range(10)]
    summary = summarize_runs(runs, minimum_seeds=10, minimum_feasibility_rate=0.9)
    assert [run["seed"] for run in summary["runs"]] == list(range(10))
    assert summary["feasibility_rate"] == 1.0
    assert summary["objective"] == {"best": 1.0, "median": 5.5, "worst": 10.0}
    assert summary["evaluation_budget"] == {"best": 10, "median": 14.5, "worst": 19}
    assert summary["acceptance"]["passed"] is True


def test_repeated_seed_summary_fails_below_feasibility_gate():
    runs = [(seed, _record(feasible=seed < 8)) for seed in range(10)]
    summary = summarize_runs(runs, minimum_seeds=10, minimum_feasibility_rate=0.9)
    assert summary["feasibility_rate"] == pytest.approx(0.8)
    assert summary["acceptance"]["passed"] is False
