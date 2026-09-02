"""Explicit, deterministic migrations between portable schema versions."""

from __future__ import annotations

from copy import deepcopy
import re
from typing import Any, Mapping

from .schema import CURRENT_SCHEMA_VERSION, ProblemSpec, validate_problem_spec


class SchemaMigrationError(ValueError):
    """Raised when no loss-aware migration path is available."""


def _safe_model_id(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value or "beamfem-model"))
    return text.strip("-") or "beamfem-model"


def migrate_v1_to_v2(document: Mapping[str, Any]) -> dict[str, Any]:
    """Migrate v1 in place semantics to the explicit-governance v2 schema.

    Legacy members default to ``frame``. Explicit v1 ``member_type`` values are
    preserved as ``axial_truss``/``frame``/``mixed`` rather than silently
    reinterpreting the analysis. The external-review gate is always enabled.
    """

    validated = validate_problem_spec(document)
    if validated.schema_version != "1.0":
        raise SchemaMigrationError("migrate_v1_to_v2 requires schema_version '1.0'")
    data = deepcopy(validated.data)
    dimension = max(len(node["xyz"]) for node in data["nodes"])
    data["schema_version"] = "2.0"
    data["metadata"] = {
        "model_id": _safe_model_id(data.get("name")),
        "migrated_from": "1.0",
    }
    observed_types = {member.get("member_type", "frame") for member in data["members"]}
    formulation = (
        "axial_truss" if observed_types == {"truss"}
        else "frame" if observed_types == {"frame"}
        else "mixed"
    )
    data["analysis"] = {
        "element_formulation": formulation,
        "dimension": dimension,
        "linearity": "linear_elastic",
    }
    data["governance"] = {
        "design_status": "verification_only",
        "external_review_required": True,
        "compliance_claim": "none",
    }
    return validate_problem_spec(data).data


def migrate_problem_spec(
    document: Mapping[str, Any], *, target_version: str = CURRENT_SCHEMA_VERSION
) -> ProblemSpec:
    """Migrate a document to a requested supported version and validate it."""

    source = str(document.get("schema_version"))
    if source == target_version:
        return validate_problem_spec(document)
    if source == "1.0" and target_version == "2.0":
        return ProblemSpec(migrate_v1_to_v2(document))
    raise SchemaMigrationError(f"no migration path from {source!r} to {target_version!r}")
