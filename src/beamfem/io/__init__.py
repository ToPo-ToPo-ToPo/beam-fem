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
]
