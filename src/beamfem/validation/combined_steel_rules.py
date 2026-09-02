"""Traceable steel section-classification and combined-force previews.

These helpers intentionally require the caller to provide already established
member strengths.  They do not infer lateral-torsional buckling, second-order
effects, connection behavior, or the applicability of AISC Chapter H.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

from .code_checks import (
    CheckStatus,
    CodeCheckResult,
    CoefficientTrace,
    RuleCitation,
    RuleSet,
)


_AISC_SPECIFICATION_URL = (
    "https://www.aisc.org/aisc/publications/current-standards/aisc-360/"
)
_AISC_ERRATA_URL = (
    "https://www.aisc.org/globalassets/aisc/publications/revisions-and-errata/"
    "errata_360-22_1st-printing_01.23.2025.pdf"
)


def _citation(section: str, equation: str) -> RuleCitation:
    return RuleCitation(
        document="ANSI/AISC 360-22 Specification for Structural Steel Buildings",
        edition="2022, first printing; preview adapter",
        section=section,
        equation=equation,
        source_url=_AISC_SPECIFICATION_URL,
        errata_url=_AISC_ERRATA_URL,
        errata_issue_date="2025-01-23",
        errata_checked_on="2026-09-02",
    )


class SectionClass(str, Enum):
    COMPACT = "compact"
    NONCOMPACT = "noncompact"
    SLENDER = "slender"


@dataclass(frozen=True)
class ElementClassification:
    name: str
    ratio: float
    compact_limit: float
    slender_limit: float
    classification: SectionClass
    citation: RuleCitation


@dataclass(frozen=True)
class SectionClassificationResult:
    member_id: str
    governing_class: SectionClass
    elements: tuple[ElementClassification, ...]
    assumptions: tuple[str, ...]
    external_review_required: bool = True


def _classify(name: str, ratio: float, compact: float, slender: float,
              citation: RuleCitation) -> ElementClassification:
    values = (ratio, compact, slender)
    if not all(math.isfinite(value) and value > 0.0 for value in values):
        raise ValueError("section slenderness ratios and limits must be positive and finite")
    if compact > slender:
        raise ValueError("compact limit cannot exceed slender limit")
    if ratio <= compact:
        result = SectionClass.COMPACT
    elif ratio <= slender:
        result = SectionClass.NONCOMPACT
    else:
        result = SectionClass.SLENDER
    return ElementClassification(name, ratio, compact, slender, result, citation)


def aisc360_22_i_shape_flexural_classification(
    member_id: str,
    *,
    elastic_modulus: float,
    yield_stress: float,
    flange_width_thickness_ratio: float,
    web_depth_thickness_ratio: float,
    ratios_confirmed: bool = False,
) -> SectionClassificationResult:
    """Classify a doubly symmetric I-shape for major-axis flexure.

    The supplied ratios must follow the Table B4.1b definitions.  This preview
    covers only the commonly used rolled-I flange/web limits and never decides
    whether Chapter F or Chapter H applies to a real member.
    """

    if not all(math.isfinite(value) and value > 0.0
               for value in (elastic_modulus, yield_stress)):
        raise ValueError("elastic_modulus and yield_stress must be positive and finite")
    root = math.sqrt(elastic_modulus / yield_stress)
    citation = _citation("Table B4.1b, Cases 10 and 15", "lambda_p and lambda_r")
    elements = (
        _classify("flange", flange_width_thickness_ratio,
                  0.38 * root, 1.00 * root, citation),
        _classify("web", web_depth_thickness_ratio,
                  3.76 * root, 5.70 * root, citation),
    )
    order = {SectionClass.COMPACT: 0, SectionClass.NONCOMPACT: 1, SectionClass.SLENDER: 2}
    governing = max((element.classification for element in elements), key=order.get)
    assumptions = (
        "User confirmed the shape and width-to-thickness definitions of Table B4.1b."
        if ratios_confirmed else
        "Ratios are unconfirmed; classification is verification-only and must not be used for approval.",
        "Major-axis flexure of a doubly symmetric rolled I-shape is assumed.",
    )
    return SectionClassificationResult(member_id, governing, elements, assumptions)


@dataclass(frozen=True)
class CombinedSteelCheckInput:
    """Available-strength inputs for a Chapter H1 interaction preview.

    All demands and available strengths are nonnegative and use one consistent
    unit system.  Required second-order moments must be supplied by the caller.
    """

    member_id: str
    axial_demand: float
    axial_capacity: float
    moment_y_demand: float = 0.0
    moment_y_capacity: float = 1.0
    moment_z_demand: float = 0.0
    moment_z_capacity: float = 1.0
    symmetric_section_confirmed: bool = False
    second_order_effects_included: bool = False
    capacities_independently_verified: bool = False

    def __post_init__(self) -> None:
        demands = (self.axial_demand, self.moment_y_demand, self.moment_z_demand)
        capacities = (self.axial_capacity, self.moment_y_capacity, self.moment_z_capacity)
        if not all(math.isfinite(value) and value >= 0.0 for value in demands):
            raise ValueError("combined-force demands must be nonnegative and finite")
        if not all(math.isfinite(value) and value > 0.0 for value in capacities):
            raise ValueError("combined-force capacities must be positive and finite")


class AISC36022CombinedAxialFlexurePreview:
    """AISC 360-22 H1-1a/H1-1b interaction using supplied capacities."""

    rule_id = "AISC360-22-PREVIEW-H1-1-LRFD"

    def evaluate(self, item: CombinedSteelCheckInput) -> CodeCheckResult:
        citation = _citation("H1.1", "H1-1a/H1-1b")
        axial = item.axial_demand / item.axial_capacity
        flexure = (
            item.moment_y_demand / item.moment_y_capacity
            + item.moment_z_demand / item.moment_z_capacity
        )
        confirmed = (
            item.symmetric_section_confirmed
            and item.second_order_effects_included
            and item.capacities_independently_verified
        )
        if not confirmed:
            return CodeCheckResult(
                item.member_id,
                self.rule_id,
                "AISC 360-22 combined axial force and flexure preview",
                CheckStatus.NOT_VERIFIED,
                None,
                1.0,
                None,
                "dimensionless",
                "interaction <= 1.0",
                "scope assertions missing",
                citation,
                assumptions=(
                    "Requires a qualifying symmetric member and independently established available strengths.",
                    "Required moments must include applicable second-order effects.",
                ),
                omissions=(
                    "Applicability, stability method, torsion, shear interaction, and capacity calculations are not inferred.",
                ),
            )
        if axial >= 0.2:
            utilization = axial + (8.0 / 9.0) * flexure
            branch = "H1-1a"
        else:
            utilization = axial / 2.0 + flexure
            branch = "H1-1b"
        return CodeCheckResult(
            item.member_id,
            self.rule_id,
            "AISC 360-22 combined axial force and flexure preview",
            CheckStatus.PASS if utilization <= 1.0 else CheckStatus.FAIL,
            utilization,
            1.0,
            utilization,
            "dimensionless",
            "Pr/Pc + interaction(Mr/Mc) <= 1.0",
            f"branch={branch}; Pr/Pc={axial:.12g}; sum(Mr/Mc)={flexure:.12g}",
            citation,
            coefficients=(
                CoefficientTrace("axial_branch", 0.2, "H1-1 branch boundary", citation),
                CoefficientTrace("high_axial_flexure_factor", 8.0 / 9.0,
                                 "H1-1a flexure coefficient", citation),
            ),
            assumptions=(
                "Caller confirmed member symmetry, Chapter H1 applicability, and second-order demands.",
                "Axial and flexural available strengths were independently verified.",
            ),
            omissions=("Capacity derivation and all limit states outside H1-1 are not evaluated.",),
        )


def aisc360_22_combined_lrfd_preview_ruleset() -> RuleSet:
    return RuleSet(
        "ansi-aisc-360-22.combined-lrfd.preview",
        "0.1.0",
        (AISC36022CombinedAxialFlexurePreview(),),
        status_label="preview_not_certified",
    )
