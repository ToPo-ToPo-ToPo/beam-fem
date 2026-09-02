"""Portable-schema coverage for fabrication and topology constraints."""

from __future__ import annotations

import pytest

from benchmarks.quantum_truss.generate_cases import generate_case
from beamfem.io import SchemaValidationError, build_discrete_problem, validate_problem_spec
from beamfem.optimize import (
    ActiveMemberCount,
    Connectivity,
    MaxSectionTypes,
    MemberLengthRange,
    RelativeDisplacementLimit,
    SameSectionGroup,
    SectionSlendernessLimit,
    SymmetryPairs,
)


def _extended_document():
    document = generate_case("small")
    for index, entry in enumerate(document["section_catalogs"]["round_bar"]):
        entry["slenderness"] = 20.0 + index
    document["constraints"] = [
        {"type": "same_section_group", "id": "chords", "members": ["m0", "m1"]},
        {"type": "max_section_types", "maximum": 1, "include_off": False},
        {"type": "active_member_count", "minimum": 16, "maximum": 16},
        {"type": "symmetry_pairs", "pairs": [["m0", "m2"], ["m1", "m3"]]},
        {"type": "connectivity", "nodes": ["b0", "b3"]},
        {"type": "member_length_range", "minimum": 1.0, "maximum": 2.0},
        {
            "type": "relative_displacement",
            "node_a": "b0",
            "node_b": "b1",
            "dof": "UX",
            "limit": 1.0,
            "combinations": ["ultimate_gravity"],
        },
        {"type": "section_slenderness", "maximum": 30.0, "members": ["m0", "m1"]},
    ]
    return document


def test_all_extended_constraints_round_trip_through_portable_adapter():
    document = _extended_document()
    validate_problem_spec(document)
    problem = build_discrete_problem(document).problem
    expected_types = {
        SameSectionGroup,
        MaxSectionTypes,
        ActiveMemberCount,
        SymmetryPairs,
        Connectivity,
        MemberLengthRange,
        RelativeDisplacementLimit,
        SectionSlendernessLimit,
    }
    assert {type(item) for item in problem.constraints} == expected_types
    result = problem.evaluate(problem.initial_design)
    assert result.feasible
    assert {record.kind for record in result.constraints} >= {
        "same_section_group",
        "max_section_types",
        "active_member_count",
        "symmetry_pair",
        "connectivity",
        "member_length_range",
        "relative_displacement",
        "section_slenderness",
    }


@pytest.mark.parametrize(
    ("constraint", "message"),
    [
        ({"type": "same_section_group", "members": ["m0"]}, "at least 2"),
        ({"type": "max_section_types", "maximum": 0}, "positive integer"),
        ({"type": "active_member_count"}, "minimum and/or maximum"),
        ({"type": "active_member_count", "minimum": 3, "maximum": 2}, ">= minimum"),
        ({"type": "symmetry_pairs", "pairs": [["m0", "missing"]]}, "unknown member"),
        ({"type": "connectivity", "nodes": ["b0"]}, "at least 2"),
        ({"type": "member_length_range", "minimum": 2.0, "maximum": 1.0}, ">= minimum"),
        ({"type": "relative_displacement", "node_a": "b0", "node_b": "b0", "dof": "UX", "limit": 1.0}, "must differ"),
        ({"type": "section_slenderness", "maximum": -1.0}, "must be positive"),
        ({"type": "stress", "unexpected": 1}, "unsupported fields"),
        ({"type": "unknown"}, "unsupported constraint type"),
    ],
)
def test_extended_constraint_schema_rejects_invalid_inputs(constraint, message):
    document = _extended_document()
    document["constraints"] = [constraint]
    with pytest.raises(SchemaValidationError, match=message):
        validate_problem_spec(document)


def test_section_slenderness_requires_catalog_values_for_every_selected_option():
    document = _extended_document()
    document["section_catalogs"]["round_bar"][0].pop("slenderness")
    with pytest.raises(SchemaValidationError, match="requires positive slenderness"):
        validate_problem_spec(document)


def test_section_slenderness_is_included_in_common_fem_feasibility():
    document = _extended_document()
    document["constraints"] = [
        {"type": "section_slenderness", "maximum": 22.0, "members": ["m0"]}
    ]
    problem = build_discrete_problem(document).problem
    result = problem.evaluate(problem.initial_design)
    record = next(item for item in result.constraints if item.kind == "section_slenderness")
    assert record.value == 23.0
    assert record.limit == 22.0
    assert not record.satisfied
    assert not result.feasible


def test_schema_rejects_unknown_dof_before_adapter_construction():
    document = _extended_document()
    document["supports"][0]["dofs"] = ["UX", "BAD"]
    with pytest.raises(SchemaValidationError, match="unknown DOF"):
        validate_problem_spec(document)
