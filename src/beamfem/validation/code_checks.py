"""Traceable code-check engine with an explicit external-review gate."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence


class CheckStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    NOT_APPLICABLE = "not_applicable"
    NOT_VERIFIED = "not_verified"


@dataclass(frozen=True)
class RuleCitation:
    document: str
    edition: str
    section: str
    equation: str
    source_url: str
    errata_url: str | None = None
    errata_issue_date: str | None = None
    errata_checked_on: str | None = None


@dataclass(frozen=True)
class CoefficientTrace:
    symbol: str
    value: float
    meaning: str
    source: RuleCitation


@dataclass(frozen=True)
class CodeCheckResult:
    member_id: str
    rule_id: str
    title: str
    status: CheckStatus
    demand: float | None
    capacity: float | None
    utilization: float | None
    units: str
    expression: str
    substituted_expression: str
    citation: RuleCitation
    coefficients: tuple[CoefficientTrace, ...] = ()
    assumptions: tuple[str, ...] = ()
    omissions: tuple[str, ...] = ()
    external_review_required: bool = True


class CodeCheckRule(Protocol):
    rule_id: str

    def evaluate(self, context: Any) -> CodeCheckResult: ...


@dataclass(frozen=True)
class CodeCheckRun:
    rule_set_id: str
    rule_set_version: str
    status_label: str
    results: tuple[CodeCheckResult, ...]
    external_review_required: bool = True
    approval_eligible: bool = False
    disclaimer: str = (
        "Verification/preliminary output only; not a certification or a substitute "
        "for review by the responsible licensed structural engineer."
    )

    @property
    def evaluated_pass(self) -> bool:
        evaluated = [
            item for item in self.results if item.status is not CheckStatus.NOT_APPLICABLE
        ]
        return bool(evaluated) and all(item.status is CheckStatus.PASS for item in evaluated)


@dataclass(frozen=True)
class RuleSet:
    rule_set_id: str
    version: str
    rules: tuple[CodeCheckRule, ...]
    status_label: str = "verification_only"
    external_review_required: bool = True

    def evaluate(self, contexts: Sequence[Any]) -> CodeCheckRun:
        results = tuple(rule.evaluate(context) for context in contexts for rule in self.rules)
        return CodeCheckRun(
            rule_set_id=self.rule_set_id,
            rule_set_version=self.version,
            status_label=self.status_label,
            results=results,
            external_review_required=self.external_review_required,
            # Product policy: automated checks can never approve a design.
            approval_eligible=False,
        )


def trace_as_dict(run: CodeCheckRun) -> Mapping[str, Any]:
    """Return an audit-friendly representation without hiding failed checks."""

    from ..io.result_writer import to_serializable

    return to_serializable(run)
