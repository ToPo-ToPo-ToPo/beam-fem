import json

from beamfem.cli import main
from benchmarks.quantum_truss.generate_cases import generate_case


def test_cli_runs_common_sa_pipeline_and_writes_audited_result(tmp_path):
    input_path = tmp_path / "small.json"
    output_path = tmp_path / "result.json"
    input_path.write_text(json.dumps(generate_case("small")), encoding="utf-8")

    exit_code = main([
        str(input_path),
        "--output", str(output_path),
        "--backend", "sa",
        "--seed", "9",
        "--max-iterations", "1",
        "--candidates", "4",
        "--sa-sweeps", "40",
        "--sa-restarts", "2",
    ])

    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert result["result"]["optimization"]["feasible"] is True
    assert result["result"]["optimization"]["backend"] == "sequential_qubo"
    assert result["audit"]["seed"] == 9
    assert result["audit"]["solver"] == "sa"
