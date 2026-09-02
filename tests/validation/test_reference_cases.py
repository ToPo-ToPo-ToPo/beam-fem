"""独立手計算fixtureとJSON evidence runnerの回帰試験。"""

import json
from pathlib import Path

from beamfem.validation.reference_cases import run_reference_suite, write_reference_evidence
from beamfem.io import SchemaValidationError, migrate_v1_to_v2, validate_problem_spec
from benchmarks.quantum_truss.generate_cases import generate_case
import pytest


FIXTURES = Path(__file__).parents[2] / "validation" / "reference_cases"


def test_all_hand_calculated_reference_cases_pass_and_evidence_is_json(tmp_path):
    evidence = run_reference_suite(sorted(FIXTURES.glob("*.json")))
    assert evidence["passed"]
    assert {case["case_id"] for case in evidence["cases"]} == {
        "triangle-2d-handcalc-v1",
        "tripod-3d-handcalc-v1",
        "mixed-frame-truss-series-v1",
    }
    for case in evidence["cases"]:
        assert case["metrics"]["equilibrium_residual"] <= 1e-8
    triangle = next(case for case in evidence["cases"] if case["case_id"].startswith("triangle"))
    assert triangle["metrics"]["rotation_invariance_relative_error"] <= 1e-10

    path = write_reference_evidence(evidence, tmp_path / "evidence.json")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["passed"] is True
    assert loaded["evidence_schema_version"] == "1.0"
    assert loaded["audit"]["created_at_utc"].endswith("+00:00")
    assert loaded["audit"]["python_version"]
    assert loaded["audit"]["platform"]
    assert set(loaded["fixture_sha256"]) == {
        "triangle_2d.json", "tripod_3d.json", "mixed_frame_truss.json"
    }
    assert all(len(digest) == 64 for digest in loaded["fixture_sha256"].values())


def test_schema_v2_formulation_matches_member_types():
    mixed = migrate_v1_to_v2(generate_case("small"))
    mixed["members"][0]["member_type"] = "truss"
    mixed["analysis"]["element_formulation"] = "mixed"
    validate_problem_spec(mixed)

    mixed["analysis"]["element_formulation"] = "frame"
    with pytest.raises(SchemaValidationError, match="inconsistent"):
        validate_problem_spec(mixed)

    all_truss = migrate_v1_to_v2(generate_case("small"))
    for member in all_truss["members"]:
        member["member_type"] = "truss"
    all_truss["analysis"]["element_formulation"] = "axial_truss"
    validate_problem_spec(all_truss)
