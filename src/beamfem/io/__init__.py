"""Versioned, dependency-light input and result serialization helpers."""

from .schema import (
    CURRENT_SCHEMA_VERSION,
    ProblemSpec,
    SchemaValidationError,
    load_problem_spec,
    validate_problem_spec,
)
from .result_writer import to_serializable, write_result_csv, write_result_json
from .problem_adapter import BuiltProblem, DOF_NAMES, build_discrete_problem
from .migration import SchemaMigrationError, migrate_problem_spec, migrate_v1_to_v2
from .run_manifest import (
    MANIFEST_SCHEMA_VERSION, RunManifest, RunStatus, canonical_checksum,
    create_run_manifest, load_run_manifest, verify_resume_compatibility,
    write_run_manifest,
)
from .html_report import DISCLAIMER, render_design_report, write_design_report

__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "ProblemSpec",
    "SchemaValidationError",
    "load_problem_spec",
    "validate_problem_spec",
    "to_serializable",
    "write_result_csv",
    "write_result_json",
    "BuiltProblem",
    "DOF_NAMES",
    "build_discrete_problem",
    "SchemaMigrationError",
    "migrate_problem_spec",
    "migrate_v1_to_v2",
    "MANIFEST_SCHEMA_VERSION",
    "RunManifest",
    "RunStatus",
    "canonical_checksum",
    "create_run_manifest",
    "load_run_manifest",
    "verify_resume_compatibility",
    "write_run_manifest",
    "DISCLAIMER",
    "render_design_report",
    "write_design_report",
]
