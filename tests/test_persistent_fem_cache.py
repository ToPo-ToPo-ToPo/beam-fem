from pathlib import Path

import pytest

from benchmarks.quantum_truss.generate_cases import generate_case
from beamfem.io import build_discrete_problem


def _problem(load_scale=1.0):
    document = generate_case("small")
    for member in document["members"]:
        member["member_type"] = "truss"
    document["load_cases"]["gravity"][0]["force"][1] *= load_scale
    return build_discrete_problem(document).problem


def test_persistent_fem_cache_round_trip_and_context_guard(tmp_path: Path):
    path = tmp_path / "evaluations.cache"
    first = _problem()
    first.enable_persistent_cache(path)
    expected = first.evaluate(first.initial_design)
    assert path.is_file() and not expected.cache_hit

    second = _problem()
    second.enable_persistent_cache(path)
    cached = second.evaluate(second.initial_design)
    assert cached.cache_hit
    assert cached.objective == expected.objective
    assert second._evaluator.persistent_cache_hits == 1

    changed = _problem(load_scale=1.01)
    with pytest.raises(ValueError, match="context mismatch"):
        changed.enable_persistent_cache(path)


def test_persistent_fem_cache_rejects_tampering_before_deserialization(tmp_path: Path):
    path = tmp_path / "evaluations.cache"
    problem = _problem()
    problem.enable_persistent_cache(path)
    problem.evaluate(problem.initial_design)
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 1
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="checksum mismatch"):
        _problem().enable_persistent_cache(path)
