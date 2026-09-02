import json
from pathlib import Path

from benchmarks.exact_milp_micro import collect_evidence


def test_structural_micro_case_exact_and_milp_have_same_global_design():
    evidence = collect_evidence()
    assert evidence["strict_acceptance_passed"]
    assert evidence["exact"]["design"] == evidence["milp"]["design"] == [2, 2]
    assert evidence["exact"]["objective_mass_kg"] == evidence["milp"]["objective_mass_kg"]
    assert evidence["milp"]["mip_gap"] == 0.0


def test_strict_json_evidence_records_off_section_index_mapping():
    path = Path(__file__).parents[2] / "validation" / "exact_milp_micro_evidence.json"
    evidence = json.loads(path.read_text(encoding="utf-8"))
    assert evidence["problem"]["section_index_map"]["0"] == "OFF"
    assert evidence["checks"]["off_index_preserved"]
    assert evidence["strict_acceptance_passed"]
