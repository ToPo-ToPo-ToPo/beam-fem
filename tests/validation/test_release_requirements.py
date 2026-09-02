import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_release_acceptance_manifest_has_safety_and_human_review_gates():
    manifest = json.loads((ROOT / "validation/acceptance_v1.json").read_text())
    assert manifest["release_target"].endswith("rc1")
    assert manifest["analysis"]["equilibrium_residual_max"] <= 1e-8
    assert manifest["optimization"]["require_final_fem_recheck"] is True
    assert manifest["safety"]["external_engineer_review_required"] is True


def test_review_templates_cannot_look_preapproved_in_source_tree():
    for name in ("independent_review_template.json", "pilot_review_template.json"):
        template = json.loads((ROOT / "validation" / name).read_text())
        assert template["status"] == "pending"
