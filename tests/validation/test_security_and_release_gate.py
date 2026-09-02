from validation.run_security_audit import summarize_pip_audit
from validation.run_dependency_inventory import _locked_names


def test_pip_audit_summary_counts_dependencies_and_vulnerabilities():
    payload = {
        "dependencies": [
            {"name": "safe", "vulns": []},
            {"name": "affected", "vulns": [{"id": "X"}, {"id": "Y"}]},
        ]
    }
    assert summarize_pip_audit(payload) == (2, 2)


def test_dependency_lock_parser_ignores_comments(tmp_path):
    lock = tmp_path / "release.lock"
    lock.write_text("# comment\nnumpy==2.5.2\nqiskit-aer==0.17.2\n", encoding="utf-8")
    assert _locked_names(lock) == ["numpy", "qiskit-aer"]
