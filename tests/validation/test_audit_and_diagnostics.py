"""Audit and preflight diagnostic tests."""

from datetime import datetime, timezone

from beamfem.validation import Severity, build_audit_metadata, diagnose_problem_spec
from benchmarks.quantum_truss.generate_cases import generate_case


def test_audit_captures_reproducible_settings(tmp_path):
    instant = datetime(2026, 1, 2, 3, 4, tzinfo=timezone.utc)
    audit = build_audit_metadata(
        solver="sa",
        seed=7,
        solver_settings={"sweeps": 100},
        warnings=["example"],
        repository=tmp_path,
        created_at=instant,
    )
    assert audit.created_at_utc == "2026-01-02T03:04:00+00:00"
    assert audit.seed == 7
    assert audit.solver_settings == {"sweeps": 100}
    assert audit.git_commit is None
    assert audit.warnings == ("example",)


def test_diagnostics_reports_duplicates_isolation_and_support_warning():
    spec = generate_case("small")
    spec["nodes"].append({"id": "isolated", "xyz": [99.0, 99.0]})
    spec["members"].append(
        {
            "id": "duplicate",
            "nodes": spec["members"][0]["nodes"],
            "material": "steel",
            "catalog": "round_bar",
        }
    )
    spec["supports"] = [{"node": "b0", "dofs": ["UX"]}]
    report = diagnose_problem_spec(spec)
    codes = {item.code for item in report.diagnostics}
    assert {"duplicate-member", "isolated-node", "possibly-underconstrained"} <= codes
    assert not report.has_errors
    assert all(item.severity is not Severity.ERROR for item in report.diagnostics)


def test_diagnostics_detects_coincident_end_coordinates():
    spec = generate_case("small")
    spec["nodes"][1]["xyz"] = list(spec["nodes"][0]["xyz"])
    report = diagnose_problem_spec(spec)
    assert report.has_errors
    assert any(item.code == "zero-length-member" for item in report.diagnostics)
