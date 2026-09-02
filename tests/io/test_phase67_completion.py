"""Phase 6/7 completion tests for catalogs, reports, REST, and QAOA metadata."""

from __future__ import annotations

from io import BytesIO
import json

import numpy as np
import pytest

from benchmarks.quantum_truss.generate_cases import generate_case
from beamfem import cli
from beamfem.api import create_wsgi_app, optimize_document
from beamfem.io import (
    build_discrete_problem, load_problem_spec, render_comparison_report,
    render_design_report, validate_problem_spec, write_design_pdf,
)
from beamfem.optimize.backends import IndependentReadoutMitigator, QAOABackend
from beamfem.optimize.qubo import QUBOModel


def test_external_versioned_material_and_section_csv_catalogs(tmp_path):
    material = tmp_path / "materials.csv"
    material.write_text(
        "id,E,density,nu,tension_allowable,compression_allowable\n"
        "steel,205000000000,7850,0.3,150000000,120000000\n", encoding="utf-8"
    )
    section = tmp_path / "round.csv"
    section.write_text(
        "id,area,I,slenderness\nS,0.0002,3.2e-9,25\nM,0.0004,1.27e-8,20\n"
        "L,0.0007,3.9e-8,18\nXL,0.001,7.96e-8,15\n", encoding="utf-8"
    )
    document = generate_case("small")
    document["materials"] = {}
    document["section_catalogs"] = {}
    document["external_catalogs"] = {
        "materials": {"steel": {"path": "materials.csv", "version": "2026.1"}},
        "sections": {"round_bar": {"path": "round.csv", "version": "2026.1"}},
    }
    source = tmp_path / "problem.json"
    source.write_text(json.dumps(document), encoding="utf-8")
    spec = load_problem_spec(source)
    assert spec.data["catalog_sources"]["materials"]["steel"]["version"] == "2026.1"
    assert len(spec.data["catalog_sources"]["sections"]["round_bar"]["sha256"]) == 64
    assert build_discrete_problem(spec).problem.n_members == 16

    document["external_catalogs"]["materials"]["steel"]["sha256"] = "0" * 64
    source.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_problem_spec(source)
    malformed = generate_case("small")
    malformed["external_catalogs"] = {"materials": {"steel2": {"path": "x.csv"}}}
    with pytest.raises(ValueError, match="version"):
        validate_problem_spec(malformed)


def test_html_pdf_and_two_run_comparison_reports_include_visual_evidence(tmp_path):
    first = {"optimization": {
        "backend": "greedy", "objective": 10.0, "feasible": True, "runtime": 2.0,
        "evaluations": 4, "iterations": 1,
        "constraints": [{"constraint_id": "stress", "utilization": 0.8}],
        "history": [{"iteration": 0, "objective": 12.0}, {"iteration": 1, "objective": 10.0}],
    }}
    second = {"optimization": {**first["optimization"], "backend": "qaoa", "objective": 9.0}}
    html = render_design_report(first)
    assert "Constraint utilization" in html and "constraint utilization chart" in html
    assert "Optimization history" in html and "optimization objective history" in html
    comparison = render_comparison_report(first, second)
    assert "Baseline" in comparison and "Candidate" in comparison and "-1" in comparison
    path = write_design_pdf(first, tmp_path / "report.pdf")
    content = path.read_bytes()
    assert content.startswith(b"%PDF-1.4") and content.endswith(b"%%EOF\n")
    assert b"EXTERNAL PROFESSIONAL REVIEW REQUIRED" in content
    args = cli._parser().parse_args([
        "problem.json", "--output", "result.json", "--backend", "qaoa",
        "--qaoa-cvar-alpha", "0.25", "--readout-error-rate", "0.02",
        "--pdf-report", "report.pdf", "--compare-with", "old.json",
        "--comparison-report", "comparison.html",
    ])
    assert args.qaoa_cvar_alpha == 0.25 and args.pdf_report.name == "report.pdf"


def _wsgi_request(app, method, path, payload=None, token=None):
    body = b"" if payload is None else json.dumps(payload).encode()
    environ = {
        "REQUEST_METHOD": method, "PATH_INFO": path,
        "CONTENT_LENGTH": str(len(body)), "wsgi.input": BytesIO(body),
    }
    if token is not None:
        environ["HTTP_AUTHORIZATION"] = f"Bearer {token}"
    state = {}
    def start_response(status, headers):
        state["status"], state["headers"] = status, dict(headers)
    response = b"".join(app(environ, start_response))
    return int(state["status"].split()[0]), json.loads(response)


def test_dependency_free_rest_api_health_auth_validation_and_optimization():
    app = create_wsgi_app(bearer_token="secret")
    status, error = _wsgi_request(app, "GET", "/health")
    assert status == 401 and error["error"]["code"] == "unauthorized"
    status, health = _wsgi_request(app, "GET", "/health", token="secret")
    assert status == 200 and health["api_version"] == "v1"
    problem = generate_case("small")
    status, validated = _wsgi_request(
        app, "POST", "/v1/validate", {"problem": problem}, token="secret"
    )
    assert status == 200 and validated["valid"] is True
    optimized = optimize_document(problem, backend="greedy", settings={"max_iterations": 1})
    assert optimized["api_version"] == "v1"
    assert optimized["optimization"]["backend"] == "greedy"


def test_independent_readout_mitigation_and_qaoa_cvar_metadata(monkeypatch):
    mitigator = IndependentReadoutMitigator(0.1)
    corrected = mitigator({"0": 0.9, "1": 0.1})
    assert corrected["0"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match=r"\[0, 0.5\)"):
        IndependentReadoutMitigator(0.5)

    captured = {}
    class _QP:
        def __init__(self, name): pass
        def binary_var(self, name): pass
        def minimize(self, **kwargs): pass
    class _Sampler:
        def __init__(self, **kwargs): pass
    class _Optimizer:
        def __init__(self, **kwargs): pass
    class _QAOA:
        def __init__(self, **kwargs): captured.update(kwargs)
    class _Sample:
        def __init__(self, x, probability):
            self.x, self.probability = np.array(x), probability
    class _Eigenstate(dict):
        def binary_probabilities(self): return dict(self)
    class _MinimumResult:
        eigenstate = _Eigenstate({"0": 0.9, "1": 0.1})
        optimizer_time = 0.2
        cost_function_evals = 3
        optimal_circuit = None
    class _Result:
        samples = [_Sample([0], 0.9), _Sample([1], 0.1)]
        x = np.array([0]); fval = 0.0; status = "SUCCESS"
        min_eigen_solver_result = _MinimumResult()
    class _Minimum:
        def __init__(self, solver): pass
        def solve(self, qp): return _Result()
    class _Globals: random_seed = None
    monkeypatch.setattr(
        "beamfem.optimize.backends.qaoa._qiskit_components",
        lambda: (_Sampler, _Optimizer, _QP, _Minimum, _QAOA, _Globals),
    )
    model = QUBOModel(np.array([0.0]), np.zeros((1, 1)))
    solution = QAOABackend(
        shots=10, cvar_alpha=0.25, readout_mitigator=mitigator,
        execution_metadata_provider=lambda result, sampler: {
            "queue_time": 1.5, "quantum_execution_time": 0.5,
        },
    ).solve_qubo(model)
    assert captured["aggregation"] == 0.25
    assert solution.metadata["objective_aggregation"] == "cvar"
    assert solution.metadata["raw_distribution"] == {"0": 0.9, "1": 0.1}
    assert solution.metadata["raw_counts"] == {"0": 9, "1": 1}
    assert solution.metadata["mitigated_distribution"]["0"] == pytest.approx(1.0)
    assert solution.metadata["quantum_timing"]["queue_time"] == 1.5
    assert solution.metadata["quantum_timing"]["execution_time"] == 0.5
    with pytest.raises(ValueError, match="cvar_alpha"):
        QAOABackend(cvar_alpha=0.0)
