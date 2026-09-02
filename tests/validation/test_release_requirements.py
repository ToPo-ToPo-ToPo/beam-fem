import json
import hashlib
from pathlib import Path

from validation.run_release_gate import _approved_pilot, _approved_review, _external_record


ROOT = Path(__file__).resolve().parents[2]


def test_release_acceptance_manifest_has_safety_and_human_review_gates():
    manifest = json.loads((ROOT / "validation/acceptance_v1.json").read_text())
    assert manifest["release_target"].endswith("rc2")
    assert manifest["analysis"]["equilibrium_residual_max"] <= 1e-8
    assert manifest["optimization"]["require_final_fem_recheck"] is True
    assert manifest["safety"]["external_engineer_review_required"] is True


def test_review_templates_cannot_look_preapproved_in_source_tree():
    for name in ("independent_review_template.json", "pilot_review_template.json"):
        template = json.loads((ROOT / "validation" / name).read_text())
        assert template["status"] == "pending"


def test_status_only_cannot_bypass_external_review_gate():
    record = json.loads((ROOT / "validation/independent_review_template.json").read_text())
    record["status"] = "approved"
    approved, errors = _approved_review(record, "a" * 40)
    assert not approved
    assert "candidate_commit does not match" in " ".join(errors)
    assert "signature_or_approval_reference is missing" in errors


def test_complete_independent_review_record_is_accepted():
    commit = "b" * 40
    record = json.loads((ROOT / "validation/independent_review_template.json").read_text())
    record.update({
        "status": "approved", "candidate_commit": commit,
        "reviewed_at": "2026-09-02T12:00:00+09:00",
        "blocking_findings": [], "signature_or_approval_reference": "signed-review-42",
    })
    record["reviewer"] = {
        "name": "Independent Reviewer", "organization": "Review Org",
        "qualification": "licensed structural engineer",
        "independent_from_implementation": True,
    }
    record["checks"] = {key: "approved" for key in record["checks"]}
    assert _approved_review(record, commit) == (True, [])


def test_pilot_gate_verifies_artifacts_checksums_and_comparisons(tmp_path):
    commit = "c" * 40
    input_path, output_path = tmp_path / "input.json", tmp_path / "output.json"
    input_path.write_text("{}\n")
    output_path.write_text("{}\n")
    record = json.loads((ROOT / "validation/pilot_review_template.json").read_text())
    record.update({
        "status": "approved", "pilot_id": "pilot-1", "candidate_commit": commit,
        "input_path": "input.json", "output_path": "output.json",
        "input_checksum": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "output_checksum": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "independent_reference": "external-solver-run-1",
        "accepted_by": "Independent Reviewer", "accepted_at": "2026-09-02T12:00:00+09:00",
        "signature_or_approval_reference": "pilot-signature-1",
    })
    record["comparisons"] = {key: "pass" for key in record["comparisons"]}
    assert _approved_pilot(record, tmp_path, commit) == (True, [])
    record["output_checksum"] = "0" * 64
    approved, errors = _approved_pilot(record, tmp_path, commit)
    assert not approved and "output checksum does not match" in errors


def test_release_gate_never_trusts_approval_json_committed_in_repository(tmp_path):
    inside = tmp_path / "approval.json"
    inside.write_text('{"status":"approved"}\n')
    record, error = _external_record(inside, tmp_path)
    assert record == {}
    assert "inside the source repository" in error

    outside_root = tmp_path / "repository"
    outside_root.mkdir()
    record, error = _external_record(inside, outside_root)
    assert record["status"] == "approved" and error is None
