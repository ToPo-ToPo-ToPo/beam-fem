"""Diagnostics and reproducibility metadata for structural optimization."""

from .audit import AuditMetadata, build_audit_metadata
from .diagnostics import Diagnostic, DiagnosticReport, Severity, diagnose_problem_spec
from .code_checks import (
    CheckStatus, CodeCheckResult, CodeCheckRun, CoefficientTrace, RuleCitation,
    RuleSet, trace_as_dict,
)
from .axial_steel_rules import (
    AxialSteelCheckInput, aisc360_22_axial_lrfd_preview_ruleset,
    verification_axial_steel_ruleset,
)
from .combined_steel_rules import (
    CombinedSteelCheckInput,
    ElementClassification,
    SectionClass,
    SectionClassificationResult,
    aisc360_22_combined_lrfd_preview_ruleset,
    aisc360_22_i_shape_flexural_classification,
)
from .dependency_audit import (
    ArtifactChecksum, DependencyAudit, DependencyRecord, build_dependency_audit,
    sha256_file, write_dependency_audit,
)
from .mixed_assembly import run_mixed_assembly_case

__all__ = [
    "AuditMetadata",
    "build_audit_metadata",
    "Diagnostic",
    "DiagnosticReport",
    "Severity",
    "diagnose_problem_spec",
    "CheckStatus",
    "CodeCheckResult",
    "CodeCheckRun",
    "CoefficientTrace",
    "RuleCitation",
    "RuleSet",
    "trace_as_dict",
    "AxialSteelCheckInput",
    "aisc360_22_axial_lrfd_preview_ruleset",
    "verification_axial_steel_ruleset",
    "CombinedSteelCheckInput",
    "ElementClassification",
    "SectionClass",
    "SectionClassificationResult",
    "aisc360_22_combined_lrfd_preview_ruleset",
    "aisc360_22_i_shape_flexural_classification",
    "ArtifactChecksum",
    "DependencyAudit",
    "DependencyRecord",
    "build_dependency_audit",
    "sha256_file",
    "write_dependency_audit",
    "run_mixed_assembly_case",
]
