from validation.run_code_check_reference import generate_evidence


def test_code_check_reference_evidence_passes_and_retains_review_gate():
    evidence = generate_evidence()
    assert evidence["passed"]
    assert all(evidence["checks"].values())
    assert evidence["external_review_required"] is True
    assert all("aisc.org" in source for source in evidence["sources"] if source)
    assert evidence["checks"]["aisc_example_h1a_interaction"]
    assert evidence["checks"]["aisc_example_h1a_branch"]
