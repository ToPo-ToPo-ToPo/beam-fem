import json
from pathlib import Path

from benchmarks.performance_acceptance import factorization_evidence, parallel_candidate_evidence


def test_factorization_runner_reports_raw_samples_median_and_speedup():
    evidence = factorization_evidence(repeats=3, load_cases=6)
    assert len(evidence["baseline_seconds"]) == len(evidence["reuse_seconds"]) == 3
    assert evidence["baseline_median_seconds"] > 0.0
    assert evidence["reuse_median_seconds"] > 0.0
    assert evidence["speedup"] == (
        evidence["baseline_median_seconds"] / evidence["reuse_median_seconds"]
    )


def test_parallel_runner_uses_medium_process_workers_and_checks_exact_results():
    evidence = parallel_candidate_evidence(repeats=1, workers=2, target_speedup=0.0)
    assert evidence["case"] == "medium"
    assert evidence["execution_model"] == "persistent isolated worker processes"
    assert evidence["results_bitwise_equal"] is True
    assert evidence["threshold_met"] is True


def test_committed_performance_evidence_does_not_promote_failed_gates():
    path = Path(__file__).parents[2] / "validation" / "performance_evidence.json"
    evidence = json.loads(path.read_text(encoding="utf-8"))
    parallel = evidence["parallel_candidate_evaluation"]
    assert parallel["threshold_met"] == (parallel["speedup"] >= parallel["target_speedup"])
    gates = evidence["solution_quality_gates"]
    expected = (
        gates["small_milp_fem_feasible"]
        and gates["medium_candidate_feasible"]
        and gates["large_candidate_feasible"]
    )
    assert gates["all_solution_quality_gates_passed"] is expected
    assert expected is True
    required = evidence["required_performance_gates"]
    assert required["all_required_performance_gates_passed"] is all(
        value for key, value in required.items()
        if key != "all_required_performance_gates_passed"
    )
