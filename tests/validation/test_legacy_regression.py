import pytest

from validation.run_legacy_regression import EXPECTED, collect_evidence


@pytest.fixture(scope="module")
def evidence():
    return collect_evidence()


def test_original_optimization_milestones_are_preserved(evidence):
    assert evidence["passed"] is True
    for key, expected in EXPECTED.items():
        assert evidence["observed"][key] == pytest.approx(expected, abs=1e-8)


def test_legacy_regression_retains_common_fem_acceptance(evidence):
    assert all(evidence["feasibility"].values())
    assert evidence["sa"]["selected_moves"] == ["M12:NONE->S", "M13:M->NONE"]
