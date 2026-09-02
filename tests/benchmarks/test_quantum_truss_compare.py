from benchmarks.quantum_truss.compare import run_comparison
from benchmarks.quantum_truss.generate_cases import generate_case
from beamfem.io import validate_problem_spec


def test_comparison_reports_common_fem_and_qubo_fields():
    spec = validate_problem_spec(generate_case("small"))
    records = run_comparison(
        spec,
        ("greedy", "sa"),
        iterations=1,
        candidates=4,
        sa_sweeps=40,
        sa_restarts=2,
        max_evaluations=500,
    )
    assert [record.backend for record in records] == ["greedy", "sa"]
    assert all(record.feasible for record in records)
    assert records[0].qubo_energy is None
    assert isinstance(records[1].qubo_energy, float)
    assert all(record.fem_score == record.mass for record in records)
