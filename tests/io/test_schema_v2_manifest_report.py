"""Schema migration, resumability, and HTML product-output tests."""

import json
import random

import pytest

from benchmarks.quantum_truss.generate_cases import generate_case
from beamfem.io import (
    RunStatus, SchemaMigrationError, SchemaValidationError, create_run_manifest,
    load_run_manifest, migrate_problem_spec, migrate_v1_to_v2,
    render_design_report, validate_problem_spec, verify_resume_compatibility,
    write_run_manifest,
)


def test_v1_migrates_to_explicit_governed_v2_without_mutation():
    original = generate_case("small")
    migrated = migrate_v1_to_v2(original)
    assert original["schema_version"] == "1.0"
    assert migrated["schema_version"] == "2.0"
    assert migrated["analysis"] == {
        "element_formulation": "frame",
        "dimension": 2,
        "linearity": "linear_elastic",
    }
    assert migrated["governance"]["external_review_required"] is True
    assert validate_problem_spec(migrated).schema_version == "2.0"


def test_no_reverse_or_unknown_migration_is_invented():
    migrated = migrate_v1_to_v2(generate_case("small"))
    with pytest.raises(SchemaMigrationError, match="no migration path"):
        migrate_problem_spec(migrated, target_version="1.0")


def test_migration_preserves_explicit_axial_truss_formulation():
    source = generate_case("small")
    for member in source["members"]:
        member["member_type"] = "truss"
    migrated = migrate_v1_to_v2(source)
    assert migrated["analysis"]["element_formulation"] == "axial_truss"


def test_v2_rejects_disabled_external_review_gate():
    migrated = migrate_v1_to_v2(generate_case("small"))
    migrated["governance"]["external_review_required"] = False
    with pytest.raises(SchemaValidationError, match="external_review_required"):
        validate_problem_spec(migrated)


def test_property_randomized_broken_references_and_nonfinite_numbers_rejected():
    rng = random.Random(90210)
    for _ in range(40):
        case = generate_case("small")
        if rng.random() < 0.5:
            case["members"][rng.randrange(16)]["nodes"][0] = f"missing-{rng.random()}"
            expected = "unknown node"
        else:
            case["materials"]["steel"]["E"] = rng.choice([float("nan"), float("inf")])
            expected = "must be positive"
        with pytest.raises(SchemaValidationError, match=expected):
            validate_problem_spec(case)


def test_manifest_roundtrip_checkpoint_and_resume_guard(tmp_path):
    problem = generate_case("small")
    settings = {"sweeps": 10}
    manifest = create_run_manifest(problem, solver="sa", solver_settings=settings, seed=4)
    manifest = manifest.advance("input_validated", checkpoint={"candidate": 3})
    path = write_run_manifest(manifest, tmp_path / "run.json")
    loaded = load_run_manifest(path)
    assert loaded.status is RunStatus.RUNNING
    assert loaded.completed_steps == ("input_validated",)
    verify_resume_compatibility(
        loaded, problem, solver="sa", solver_settings=settings, seed=4
    )
    changed = generate_case("small")
    changed["name"] = "changed"
    with pytest.raises(ValueError, match="problem checksum"):
        verify_resume_compatibility(
            loaded, changed, solver="sa", solver_settings=settings, seed=4
        )


def test_manifest_rejects_malformed_and_future_schema(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"manifest_schema_version": "999"}), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported"):
        load_run_manifest(path)


def test_html_report_escapes_content_and_keeps_review_gate():
    report = render_design_report({"name": "<script>alert(1)</script>", "feasible": True})
    assert "<script>alert(1)</script>" not in report
    assert "&lt;script&gt;" in report
    assert "外部レビューが必須" in report
    assert "External professional review: REQUIRED" in report
