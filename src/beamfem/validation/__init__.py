"""Diagnostics and reproducibility metadata for structural optimization."""

from .audit import AuditMetadata, build_audit_metadata
from .diagnostics import Diagnostic, DiagnosticReport, Severity, diagnose_problem_spec

__all__ = [
    "AuditMetadata",
    "build_audit_metadata",
    "Diagnostic",
    "DiagnosticReport",
    "Severity",
    "diagnose_problem_spec",
]
