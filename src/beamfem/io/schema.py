"""Versioned input schema for discrete structural optimization.

The validator deliberately uses only the standard library.  It checks the
portable interchange format and leaves construction of beamfem domain objects
to an adapter in the optimization package.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping


CURRENT_SCHEMA_VERSION = "2.0"
_SUPPORTED_VERSIONS = frozenset({"1.0", CURRENT_SCHEMA_VERSION})


class SchemaValidationError(ValueError):
    """Raised when an input document does not satisfy the portable schema."""

    def __init__(self, errors: list[str]):
        self.errors = tuple(errors)
        super().__init__("Invalid problem specification:\n- " + "\n- ".join(errors))


@dataclass(frozen=True)
class ProblemSpec:
    """Validated optimization input and its source location."""

    data: dict[str, Any]
    source: Path | None = None

    @property
    def schema_version(self) -> str:
        return str(self.data["schema_version"])


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _require_mapping(value: Any, path: str, errors: list[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        errors.append(f"{path} must be an object")
        return {}
    return value


def _require_list(value: Any, path: str, errors: list[str]) -> list[Any]:
    if not isinstance(value, list):
        errors.append(f"{path} must be an array")
        return []
    return value


def validate_problem_spec(document: Mapping[str, Any]) -> ProblemSpec:
    """Validate and copy a supported discrete-optimization input document.

    The schema is intentionally structural: it validates identifiers,
    references, dimensions, and common physical fields without prescribing a
    particular FEM or optimizer implementation.
    """

    if not isinstance(document, Mapping):
        raise SchemaValidationError(["document must be an object"])

    data = deepcopy(dict(document))
    errors: list[str] = []
    version = data.get("schema_version")
    if version not in _SUPPORTED_VERSIONS:
        errors.append(
            f"schema_version must be one of {sorted(_SUPPORTED_VERSIONS)}; got {version!r}"
        )
    if data.get("units") != "SI":
        errors.append("units must be 'SI'")
    if version == "2.0":
        metadata = _require_mapping(data.get("metadata"), "metadata", errors)
        if not isinstance(metadata.get("model_id"), str) or not metadata.get("model_id"):
            errors.append("metadata.model_id must be a non-empty string")
        analysis = _require_mapping(data.get("analysis"), "analysis", errors)
        if analysis.get("element_formulation") not in {"axial_truss", "frame", "mixed"}:
            errors.append(
                "analysis.element_formulation must be 'axial_truss', 'frame', or 'mixed'"
            )
        if analysis.get("dimension") not in {2, 3}:
            errors.append("analysis.dimension must be 2 or 3")
        if analysis.get("linearity") != "linear_elastic":
            errors.append("analysis.linearity must be 'linear_elastic'")
        governance = _require_mapping(data.get("governance"), "governance", errors)
        if governance.get("design_status") not in {
            "verification_only",
            "preliminary_design",
        }:
            errors.append(
                "governance.design_status must be verification_only or preliminary_design"
            )
        if governance.get("external_review_required") is not True:
            errors.append("governance.external_review_required must be true")
    self_weight = data.get("self_weight")
    if self_weight is not None and (
        not isinstance(self_weight, list)
        or len(self_weight) != 3
        or not all(_is_number(value) for value in self_weight)
    ):
        errors.append("self_weight must contain 3 acceleration components [m/s2]")

    nodes = _require_list(data.get("nodes"), "nodes", errors)
    node_ids: set[str] = set()
    coordinate_dimensions: set[int] = set()
    for i, node_value in enumerate(nodes):
        node = _require_mapping(node_value, f"nodes[{i}]", errors)
        node_id = node.get("id")
        if not isinstance(node_id, str) or not node_id:
            errors.append(f"nodes[{i}].id must be a non-empty string")
        elif node_id in node_ids:
            errors.append(f"nodes[{i}].id duplicates {node_id!r}")
        else:
            node_ids.add(node_id)
        xyz = node.get("xyz")
        if not isinstance(xyz, list) or len(xyz) not in (2, 3) or not all(
            _is_number(v) for v in xyz
        ):
            errors.append(f"nodes[{i}].xyz must contain 2 or 3 numbers")
        else:
            coordinate_dimensions.add(len(xyz))
    if len(coordinate_dimensions) > 1:
        errors.append("all nodes must use the same coordinate dimension")
    if version == "2.0" and coordinate_dimensions:
        declared_dimension = analysis.get("dimension")
        actual_dimension = next(iter(coordinate_dimensions))
        if declared_dimension != actual_dimension:
            errors.append("analysis.dimension does not match node coordinates")

    materials = _require_mapping(data.get("materials"), "materials", errors)
    for name, material_value in materials.items():
        material = _require_mapping(material_value, f"materials.{name}", errors)
        for field in ("E", "density"):
            value = material.get(field)
            if not _is_number(value) or value <= 0:
                errors.append(f"materials.{name}.{field} must be positive")

    catalogs = _require_mapping(
        data.get("section_catalogs"), "section_catalogs", errors
    )
    for name, entries_value in catalogs.items():
        entries = _require_list(entries_value, f"section_catalogs.{name}", errors)
        if not entries:
            errors.append(f"section_catalogs.{name} must not be empty")
        entry_ids: set[str] = set()
        for j, entry_value in enumerate(entries):
            entry = _require_mapping(
                entry_value, f"section_catalogs.{name}[{j}]", errors
            )
            entry_id = entry.get("id")
            if not isinstance(entry_id, str) or not entry_id:
                errors.append(
                    f"section_catalogs.{name}[{j}].id must be a non-empty string"
                )
            elif entry_id in entry_ids:
                errors.append(
                    f"section_catalogs.{name}[{j}].id duplicates {entry_id!r}"
                )
            else:
                entry_ids.add(entry_id)
            area = entry.get("area")
            if not _is_number(area) or area <= 0:
                errors.append(f"section_catalogs.{name}[{j}].area must be positive")
            inertia = entry.get("I")
            if inertia is not None and (not _is_number(inertia) or inertia <= 0):
                errors.append(f"section_catalogs.{name}[{j}].I must be positive when provided")

    members = _require_list(data.get("members"), "members", errors)
    member_ids: set[str] = set()
    observed_member_types: set[str] = set()
    for i, member_value in enumerate(members):
        member = _require_mapping(member_value, f"members[{i}]", errors)
        member_id = member.get("id")
        if not isinstance(member_id, str) or not member_id:
            errors.append(f"members[{i}].id must be a non-empty string")
        elif member_id in member_ids:
            errors.append(f"members[{i}].id duplicates {member_id!r}")
        else:
            member_ids.add(member_id)
        ends = member.get("nodes")
        if not isinstance(ends, list) or len(ends) != 2:
            errors.append(f"members[{i}].nodes must contain exactly 2 node ids")
        else:
            for end in ends:
                if end not in node_ids:
                    errors.append(f"members[{i}].nodes references unknown node {end!r}")
            if len(ends) == 2 and ends[0] == ends[1]:
                errors.append(f"members[{i}] has identical end nodes")
        if member.get("material") not in materials:
            errors.append(f"members[{i}].material references an unknown material")
        if member.get("catalog") not in catalogs:
            errors.append(f"members[{i}].catalog references an unknown catalog")
        member_type = member.get("member_type", "frame")
        if member_type not in {"frame", "truss"}:
            errors.append(f"members[{i}].member_type must be 'frame' or 'truss'")
        else:
            observed_member_types.add(member_type)
        if member_type == "frame" and member.get("catalog") in catalogs:
            for j, entry in enumerate(catalogs[member["catalog"]]):
                if not _is_number(entry.get("I")) or entry["I"] <= 0:
                    errors.append(
                        f"section_catalogs.{member['catalog']}[{j}].I is required for frame members"
                    )
    if version == "2.0" and observed_member_types:
        declared = analysis.get("element_formulation")
        expected_types = {
            "axial_truss": {"truss"},
            "frame": {"frame"},
            "mixed": {"frame", "truss"},
        }.get(declared)
        if expected_types is not None and observed_member_types != expected_types:
            errors.append(
                "analysis.element_formulation is inconsistent with members[].member_type: "
                f"declared={declared!r}, observed={sorted(observed_member_types)}"
            )

    supports = _require_list(data.get("supports", []), "supports", errors)
    for i, support_value in enumerate(supports):
        support = _require_mapping(support_value, f"supports[{i}]", errors)
        if support.get("node") not in node_ids:
            errors.append(f"supports[{i}].node references an unknown node")
        dofs = support.get("dofs")
        if not isinstance(dofs, list) or not dofs or not all(
            isinstance(dof, str) for dof in dofs
        ):
            errors.append(f"supports[{i}].dofs must be a non-empty string array")

    load_cases = _require_mapping(data.get("load_cases"), "load_cases", errors)
    for case_name, loads_value in load_cases.items():
        loads = _require_list(loads_value, f"load_cases.{case_name}", errors)
        for j, load_value in enumerate(loads):
            load = _require_mapping(load_value, f"load_cases.{case_name}[{j}]", errors)
            if load.get("node") not in node_ids:
                errors.append(
                    f"load_cases.{case_name}[{j}].node references an unknown node"
                )
            force = load.get("force")
            if not isinstance(force, list) or len(force) not in (2, 3) or not all(
                _is_number(v) for v in force
            ):
                errors.append(
                    f"load_cases.{case_name}[{j}].force must contain 2 or 3 numbers"
                )
            elif coordinate_dimensions and len(force) != next(iter(coordinate_dimensions)):
                errors.append(
                    f"load_cases.{case_name}[{j}].force dimension does not match nodes"
                )

    combinations = _require_mapping(
        data.get("load_combinations"), "load_combinations", errors
    )
    for combo_name, factors_value in combinations.items():
        factors = _require_mapping(
            factors_value, f"load_combinations.{combo_name}", errors
        )
        if not factors:
            errors.append(f"load_combinations.{combo_name} must not be empty")
        for case_name, factor in factors.items():
            if case_name not in load_cases:
                errors.append(
                    f"load_combinations.{combo_name} references unknown case {case_name!r}"
                )
            if not _is_number(factor):
                errors.append(
                    f"load_combinations.{combo_name}.{case_name} must be numeric"
                )

    constraints = _require_list(data.get("constraints", []), "constraints", errors)
    member_by_id = {
        member.get("id"): member for member in members if isinstance(member, Mapping)
    }
    for index, constraint_value in enumerate(constraints):
        constraint = _require_mapping(constraint_value, f"constraints[{index}]", errors)
        if constraint.get("type") != "euler_buckling":
            continue
        selected = constraint.get("members")
        if selected is None:
            selected_ids = set(member_by_id)
        elif isinstance(selected, list) and all(isinstance(value, str) for value in selected):
            selected_ids = set(selected)
            unknown = selected_ids - set(member_by_id)
            if unknown:
                errors.append(f"constraints[{index}].members references unknown members")
        else:
            errors.append(f"constraints[{index}].members must be a list of member ids")
            selected_ids = set()
        required_catalogs = {
            member_by_id[member_id].get("catalog")
            for member_id in selected_ids if member_id in member_by_id
        }
        for catalog_name in required_catalogs:
            if catalog_name not in catalogs:
                continue
            if any(not _is_number(entry.get("I")) or entry["I"] <= 0
                   for entry in catalogs[catalog_name]):
                errors.append(
                    f"section_catalogs.{catalog_name} requires positive I for euler_buckling"
                )
    objective = _require_mapping(data.get("objective"), "objective", errors)
    if objective.get("type") not in {"mass", "cost", "co2", "weighted"}:
        errors.append("objective.type must be mass, cost, co2, or weighted")

    if not nodes:
        errors.append("nodes must not be empty")
    if not members:
        errors.append("members must not be empty")
    if not materials:
        errors.append("materials must not be empty")
    if not load_cases:
        errors.append("load_cases must not be empty")
    if not combinations:
        errors.append("load_combinations must not be empty")

    if errors:
        raise SchemaValidationError(errors)
    return ProblemSpec(data=data)


def load_problem_spec(path: str | Path) -> ProblemSpec:
    """Load and validate JSON or optional YAML input.

    YAML support is activated only when PyYAML is installed. JSON remains the
    dependency-free interchange format.
    """

    source = Path(path)
    suffix = source.suffix.lower()
    with source.open("r", encoding="utf-8") as stream:
        if suffix == ".json":
            document = json.load(stream)
        elif suffix in {".yaml", ".yml"}:
            try:
                import yaml  # type: ignore[import-not-found]
            except ImportError as exc:
                raise RuntimeError(
                    "YAML input requires the optional 'PyYAML' dependency"
                ) from exc
            document = yaml.safe_load(stream)
        else:
            raise ValueError("input file must use .json, .yaml, or .yml")
    validated = validate_problem_spec(document)
    return ProblemSpec(data=validated.data, source=source.resolve())
