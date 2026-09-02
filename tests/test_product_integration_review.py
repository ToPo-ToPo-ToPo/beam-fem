"""Independent integration review of optimizer, checkpoint, CLI, and audit paths."""

from dataclasses import dataclass
import json

import numpy as np
import pytest

from benchmarks.quantum_truss.generate_cases import generate_case
from beamfem import cli
from beamfem.io import (
    RunStatus, create_run_manifest, load_run_manifest, write_result_json,
    write_run_manifest,
)
from beamfem.validation import sha256_file
from beamfem.optimize.backends import (
    GreedyBackend, OptimizationResult, QAOABackend, SequentialQUBOOptimizer,
    SimulatedAnnealingBackend, SolverLimits,
)
from beamfem.optimize.backends.base import StopController
from beamfem.optimize.backends.milp import build_truss_sizing_milp
from beamfem.optimize.qubo import AdaptivePenalty, LocalQUBOBuilder, QUBOModel, TrustRegion
from beamfem.optimize.topology import GroundStructure


@dataclass(frozen=True)
class _Evaluation:
    objective: float
    feasible: bool = True
    constraints: tuple = ()


class _Problem:
    initial_design = (1, 1)
    domains = ((0, 1), (0, 1))

    def __init__(self, fingerprint="problem-a"):
        self.fingerprint = fingerprint

    def checkpoint_fingerprint(self):
        return self.fingerprint

    def evaluate(self, design):
        return _Evaluation(float(sum(design)))


def _sequential(checkpoint, *, resume=False, seed=7, problem=None):
    problem = problem or _Problem()
    builder = LocalQUBOBuilder(
        problem, max_candidates=2, penalty=AdaptivePenalty(value=100.0)
    )
    optimizer = SequentialQUBOOptimizer(
        SimulatedAnnealingBackend(sweeps=20, restarts=2, seed=seed),
        builder, max_iterations=1, checkpoint_path=checkpoint, resume=resume,
    )
    return optimizer, problem


def test_optimizer_checkpoint_detects_tampering_before_state_mutation(tmp_path):
    path = tmp_path / "optimizer.json"
    optimizer, problem = _sequential(path)
    optimizer.solve(problem)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["current"] = [99, 99]
    path.write_text(json.dumps(payload), encoding="utf-8")
    resumed, problem = _sequential(path, resume=True)
    original_penalty = resumed.builder.penalty.value
    with pytest.raises(ValueError, match="checksum mismatch"):
        resumed.solve(problem)
    assert resumed.builder.penalty.value == original_penalty


@pytest.mark.parametrize(
    "changed", ["solver", "problem"],
)
def test_optimizer_checkpoint_rejects_changed_problem_or_solver(tmp_path, changed):
    path = tmp_path / "optimizer.json"
    optimizer, problem = _sequential(path)
    optimizer.solve(problem)
    if changed == "solver":
        resumed, problem = _sequential(path, resume=True, seed=8)
    else:
        resumed, problem = _sequential(
            path, resume=True, problem=_Problem("problem-b")
        )
    with pytest.raises(ValueError, match="context mismatch"):
        resumed.solve(problem)


def test_run_manifest_integrity_prevents_completed_status_spoofing(tmp_path):
    path = tmp_path / "manifest.json"
    manifest = create_run_manifest({"model": 1}, solver="sa", seed=1)
    write_run_manifest(manifest, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["status"] = "completed"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="integrity checksum"):
        load_run_manifest(path)


def test_nonpositive_prediction_is_strict_json_serializable(tmp_path):
    trust = TrustRegion(radius=2)
    rho, accepted = trust.update(0.0, 0.0)
    assert rho is None and not accepted
    write_result_json({"trust_history": trust.history}, tmp_path / "result.json")


def test_failed_backend_result_has_serializable_nonfinite_objective(tmp_path):
    result = OptimizationResult((0,), float("inf"), False, backend="milp", status="failed")
    path = write_result_json(result, tmp_path / "failure.json")
    assert json.loads(path.read_text(encoding="utf-8"))["result"]["objective"] is None


def test_milp_builder_rejects_nonfinite_coefficients():
    structure = GroundStructure(
        nodes=np.array([[0.0, 0.0], [1.0, 0.0]]),
        members=[(0, 1)], supports={0: [0, 1], 1: [1]},
        load_cases=[{(1, 0): 1.0}],
    )
    with pytest.raises(ValueError, match="finite"):
        build_truss_sizing_milp(structure, [0.0, float("nan")], 7850.0, 250e6)
    with pytest.raises(ValueError, match="positive"):
        build_truss_sizing_milp(structure, [0.0, 1e-4], float("inf"), 250e6)


def test_nonfinite_backend_settings_and_evaluations_are_rejected_early():
    with pytest.raises(ValueError, match="penalty"):
        GreedyBackend(penalty=float("nan"))
    with pytest.raises(ValueError, match="annealing"):
        SimulatedAnnealingBackend(initial_temperature=float("inf"))
    with pytest.raises(ValueError, match="time_limit"):
        SolverLimits(time_limit=float("nan"))
    with pytest.raises(ValueError, match="memory_limit_mb"):
        SolverLimits(memory_limit_mb=0)
    with pytest.raises(ValueError, match="shots"):
        QAOABackend(shots=0)

    class _NonfiniteProblem(_Problem):
        def evaluate(self, design):
            return _Evaluation(float("nan"))

    with pytest.raises(ValueError, match="objective must be finite"):
        GreedyBackend().solve(_NonfiniteProblem())


def test_memory_limit_is_machine_readable_and_enforced():
    controller = StopController(SolverLimits(memory_limit_mb=0.001))
    assert controller.reached(0, 0) == "memory limit reached"


def test_qaoa_empty_samples_use_none_probability(monkeypatch, tmp_path):
    class _QP:
        def __init__(self, name): pass
        def binary_var(self, name): pass
        def minimize(self, **kwargs): pass

    class _Sampler:
        def __init__(self, **kwargs): pass

    class _Optimizer:
        def __init__(self, **kwargs): pass

    class _QAOA:
        def __init__(self, **kwargs): pass

    class _Result:
        samples = []
        x = np.array([0.0, 1.0])
        fval = 1.0
        status = "SUCCESS"
        min_eigen_solver_result = None

    class _Minimum:
        def __init__(self, solver): pass
        def solve(self, qp): return _Result()

    class _Globals:
        random_seed = None

    monkeypatch.setattr(
        "beamfem.optimize.backends.qaoa._qiskit_components",
        lambda: (_Sampler, _Optimizer, _QP, _Minimum, _QAOA, _Globals),
    )
    model = QUBOModel(np.array([0.0, 1.0]), np.zeros((2, 2)))
    solution = QAOABackend(shots=10).solve_qubo(model)
    assert solution.metadata["selected_probability"] is None
    write_result_json(solution.metadata, tmp_path / "result.json")


def test_cli_failure_records_failed_manifest(monkeypatch, tmp_path):
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "result.json"
    manifest_path = tmp_path / "manifest.json"
    input_path.write_text(json.dumps(generate_case("small")), encoding="utf-8")

    class _Broken:
        def solve(self, *args, **kwargs):
            raise ValueError("deliberate solver failure")

    monkeypatch.setattr(cli, "_backend", lambda problem, args: _Broken())
    with pytest.raises(ValueError, match="deliberate"):
        cli.main([
            str(input_path), "--output", str(output_path), "--manifest", str(manifest_path)
        ])
    manifest = load_run_manifest(manifest_path)
    assert manifest.status is RunStatus.FAILED
    assert manifest.checkpoint["error_type"] == "ValueError"


def test_cli_artifact_checksums_report_and_completed_resume_are_consistent(
    monkeypatch, tmp_path
):
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "result.json"
    manifest_path = tmp_path / "manifest.json"
    report_path = tmp_path / "report.html"
    dependency_path = tmp_path / "dependencies.json"
    input_path.write_text(json.dumps(generate_case("small")), encoding="utf-8")
    calls = {"count": 0}

    class _Success:
        def solve(self, problem, **kwargs):
            calls["count"] += 1
            evaluation = problem.evaluate(problem.initial_design)
            return OptimizationResult(
                problem.initial_design, evaluation.objective, evaluation.feasible,
                evaluation.constraints, evaluations=1, backend="review-double",
                evaluation=evaluation,
            )

    monkeypatch.setattr(cli, "_backend", lambda problem, args: _Success())
    arguments = [
        str(input_path), "--output", str(output_path),
        "--manifest", str(manifest_path), "--html-report", str(report_path),
        "--dependency-audit", str(dependency_path),
    ]
    assert cli.main(arguments) == 0
    manifest = load_run_manifest(manifest_path)
    assert manifest.status is RunStatus.COMPLETED
    assert manifest.artifacts == {
        "result": sha256_file(output_path).digest,
        "html_report": sha256_file(report_path).digest,
        "dependency_audit": sha256_file(dependency_path).digest,
    }
    assert "External professional review: REQUIRED" in report_path.read_text(encoding="utf-8")
    assert cli.main(arguments + ["--resume"]) == 0
    assert calls["count"] == 1
    output_path.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact checksum mismatch"):
        cli.main(arguments + ["--resume"])


def test_cli_qaoa_failure_records_fallback_in_audit(monkeypatch, tmp_path):
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "result.json"
    input_path.write_text(json.dumps(generate_case("small")), encoding="utf-8")

    class _Unavailable:
        def solve(self, *args, **kwargs):
            raise cli.QiskitNotInstalledError("not installed")

    class _Fallback:
        def solve(self, problem, **kwargs):
            evaluation = problem.evaluate(problem.initial_design)
            return OptimizationResult(
                problem.initial_design, evaluation.objective, evaluation.feasible,
                evaluation.constraints, backend="greedy", evaluation=evaluation,
            )

    monkeypatch.setattr(cli, "_backend", lambda problem, args: _Unavailable())
    monkeypatch.setattr(cli, "_fallback", lambda problem, args: _Fallback())
    assert cli.main([
        str(input_path), "--output", str(output_path), "--backend", "qaoa",
        "--fallback", "greedy",
    ]) == 0
    output = json.loads(output_path.read_text(encoding="utf-8"))
    assert output["audit"]["solver"] == "greedy"
    assert "QAOA failed" in output["audit"]["warnings"][0]
    assert output["audit"]["solver_settings"]["selected_backend"] == "greedy"
