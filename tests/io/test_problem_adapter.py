import json

import pytest

from benchmarks.quantum_truss.generate_cases import generate_case
from beamfem.io import build_discrete_problem, write_result_json


def test_portable_small_case_builds_and_evaluates(tmp_path):
    built = build_discrete_problem(generate_case("small"))
    problem = built.problem
    assert problem.n_members == 16
    assert len(built.node_ids) == 8
    assert all(catalog[0].name == "OFF" for catalog in problem.catalogs)
    result = problem.evaluate(problem.initial_design)
    assert result.feasible
    assert result.mass > 0.0
    assert result.governing_constraint is not None

    path = write_result_json(result, tmp_path / "result.json")
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["result"]["design"] == list(problem.initial_design.choices)
    assert "analyses" not in document["result"]


def test_adapter_rejects_constraint_without_implementation():
    document = generate_case("small")
    document["constraints"] = [{"type": "not-implemented"}]
    with pytest.raises(ValueError, match="unsupported constraint"):
        build_discrete_problem(document)


def test_adapter_passes_self_weight_to_common_problem():
    document = generate_case("small")
    document["self_weight"] = [0.0, -9.80665, 0.0]
    problem = build_discrete_problem(document).problem
    assert tuple(problem.self_weight) == (0.0, -9.80665, 0.0)
