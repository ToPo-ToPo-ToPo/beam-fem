"""Scalable benchmark generation and backend-neutral runner tests."""

from benchmarks.quantum_truss.generate_cases import CASE_SIZES, generate_case
from benchmarks.quantum_truss.runner import run_benchmark
from beamfem.io import build_discrete_problem, validate_problem_spec


def test_all_benchmark_sizes_validate_and_scale():
    counts = []
    for size in CASE_SIZES:
        case = generate_case(size)
        validate_problem_spec(case)
        counts.append((len(case["nodes"]), len(case["members"])))
    assert counts == sorted(counts)
    assert counts[0] == (8, 16)


def test_scaled_cases_hold_physical_envelope_and_have_a_feasible_anchor():
    for size in CASE_SIZES:
        case = generate_case(size)
        assert max(node["xyz"][0] for node in case["nodes"]) == 4.5
        assert sum(load["force"][1] for load in case["load_cases"]["gravity"]) == -60_000.0
        assert sum(load["force"][0] for load in case["load_cases"]["wind"]) == 12_000.0
        for member in case["members"]:
            member["member_type"] = "truss"
        problem = build_discrete_problem(case).problem
        assert problem.evaluate(problem.initial_design).feasible


def test_runner_passes_validated_mapping_and_seed():
    observed = {}

    def fake_solver(problem, settings):
        observed["name"] = problem["name"]
        observed["seed"] = settings["seed"]
        return {"mass": 12.5, "feasible": True}

    record, audit = run_benchmark(
        generate_case("small"),
        case="small",
        solver_name="fake",
        solver=fake_solver,
        seed=11,
    )
    assert record.node_count == 8
    assert record.member_count == 16
    assert record.result["feasible"]
    assert observed == {"name": "quantum-truss-small", "seed": 11}
    assert audit.solver == "fake"
    assert audit.seed == 11


def test_runner_dry_run_is_available_without_optional_backend():
    record, _ = run_benchmark(
        generate_case("small"), case="small", solver_name="dry-run"
    )
    assert record.result["status"] == "validated"
