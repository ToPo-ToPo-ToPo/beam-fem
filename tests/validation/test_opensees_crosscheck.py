import importlib.util

import pytest


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("openseespy") is None,
    reason="optional OpenSeesPy external validation dependency is absent",
)


def test_opensees_linear_and_bilinear_crosscheck():
    from validation.run_opensees_crosscheck import generate_evidence

    evidence = generate_evidence()
    assert evidence["passed"]
    assert all(evidence["checks"].values())
    assert evidence["external_solver"]["name"] == "OpenSeesPy"
