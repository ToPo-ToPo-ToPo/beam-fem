from validation.run_security_audit import summarize_pip_audit


def test_pip_audit_summary_counts_dependencies_and_vulnerabilities():
    payload = {
        "dependencies": [
            {"name": "safe", "vulns": []},
            {"name": "affected", "vulns": [{"id": "X"}, {"id": "Y"}]},
        ]
    }
    assert summarize_pip_audit(payload) == (2, 2)
