"""Portable input schema and result writer tests."""

from dataclasses import dataclass
import csv
import json

import numpy as np
import pytest

from beamfem.io import (
    SchemaValidationError,
    load_problem_spec,
    validate_problem_spec,
    write_result_csv,
    write_result_json,
)
from benchmarks.quantum_truss.generate_cases import generate_case


def test_generated_case_validates_and_is_copied():
    source = generate_case("small")
    spec = validate_problem_spec(source)
    assert spec.schema_version == "1.0"
    assert len(spec.data["nodes"]) == 8
    assert len(spec.data["members"]) == 16
    source["nodes"][0]["xyz"][0] = 999.0
    assert spec.data["nodes"][0]["xyz"][0] == 0.0


def test_schema_collects_reference_and_unit_errors():
    source = generate_case("small")
    source["units"] = "mm"
    source["members"][0]["nodes"][1] = "missing"
    with pytest.raises(SchemaValidationError) as caught:
        validate_problem_spec(source)
    assert any("units" in error for error in caught.value.errors)
    assert any("unknown node" in error for error in caught.value.errors)


def test_json_load_retains_source(tmp_path):
    path = tmp_path / "problem.json"
    path.write_text(json.dumps(generate_case("small")), encoding="utf-8")
    spec = load_problem_spec(path)
    assert spec.source == path.resolve()


def test_unknown_input_extension_rejected(tmp_path):
    path = tmp_path / "problem.txt"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="json"):
        load_problem_spec(path)


@dataclass
class _Result:
    mass: float
    feasible: bool
    vector: np.ndarray


def test_json_and_csv_result_writers(tmp_path):
    result = _Result(88.7, True, np.array([1, 2]))
    json_path = write_result_json(result, tmp_path / "nested" / "result.json")
    document = json.loads(json_path.read_text(encoding="utf-8"))
    assert document["result_schema_version"] == "1.0"
    assert document["result"]["vector"] == [1, 2]

    csv_path = write_result_csv(result, tmp_path / "result.csv")
    with csv_path.open(encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    assert row["mass"] == "88.7"
    assert row["feasible"] == "True"


def test_result_writer_rejects_non_finite_float(tmp_path):
    with pytest.raises(ValueError, match="NaN"):
        write_result_json({"score": float("nan")}, tmp_path / "bad.json")
