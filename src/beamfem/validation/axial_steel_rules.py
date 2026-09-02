"""Verification rules and a narrowly scoped ANSI/AISC 360-22 LRFD preview.

The preview is deliberately not an AISC compliance implementation. It evaluates
only gross-section tensile yielding or flexural buckling of a user-asserted
nonslender, concentrically loaded member. Connection rupture, slender elements,
torsional/flexural-torsional buckling, combined actions, fatigue, seismic rules,
and system stability are outside its scope.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from .code_checks import (
    CheckStatus,
    CodeCheckResult,
    CoefficientTrace,
    RuleCitation,
    RuleSet,
)


@dataclass(frozen=True)
class AxialSteelCheckInput:
    """Axial member data in SI units.

    ``axial_force`` is N (tension positive), ``area`` is m2,
    ``inertia_min`` is m4, ``length`` is m, and stresses/modulus are Pa.
    For the AISC compression preview, ``governing_axis`` must explicitly name
    the user-reviewed axis represented by ``inertia_min``.
    """

    member_id: str
    axial_force: float
    area: float
    inertia_min: float
    length: float
    elastic_modulus: float
    yield_stress: float
    effective_length_factor: float = 1.0
    nonslender_elements_confirmed: bool = False
    flexural_buckling_controls_confirmed: bool = False
    gross_section_yielding_only_confirmed: bool = False
    governing_axis: str | None = None

    def __post_init__(self) -> None:
        positive = {
            "area": self.area,
            "inertia_min": self.inertia_min,
            "length": self.length,
            "elastic_modulus": self.elastic_modulus,
            "yield_stress": self.yield_stress,
            "effective_length_factor": self.effective_length_factor,
        }
        invalid = [name for name, value in positive.items() if not math.isfinite(value) or value <= 0]
        if invalid or not math.isfinite(self.axial_force):
            raise ValueError(f"non-finite or non-positive axial check input: {invalid}")
        if self.governing_axis not in {None, "y", "z", "user_confirmed_minimum"}:
            raise ValueError("governing_axis must be y, z, or user_confirmed_minimum")


_VERIFY_SOURCE = RuleCitation(
    document="beamfem axial-member verification note",
    edition="1.0",
    section="V-AXIAL",
    equation="V-1/V-2",
    source_url="https://github.com/ToPo-ToPo-ToPo/beam-fem",
)


class VerificationTensionRule:
    rule_id = "BF-VERIFY-TENSION-001"

    def evaluate(self, item: AxialSteelCheckInput) -> CodeCheckResult:
        if item.axial_force <= 0:
            status = CheckStatus.NOT_APPLICABLE
            demand = capacity = utilization = None
        else:
            demand = item.axial_force
            capacity = item.yield_stress * item.area
            if math.isfinite(capacity):
                utilization = demand / capacity
                status = CheckStatus.PASS if utilization <= 1.0 else CheckStatus.FAIL
            else:
                capacity = utilization = None
                status = CheckStatus.NOT_VERIFIED
        return CodeCheckResult(
            item.member_id, self.rule_id, "Verification gross-section tension",
            status, demand, capacity, utilization, "N", "R = Fy A",
            f"R = {item.yield_stress:.12g} * {item.area:.12g}", _VERIFY_SOURCE,
            assumptions=("Validation equation; no design-standard compliance claim.",),
            omissions=("Net-section rupture and connections are not evaluated.",),
        )


class VerificationEulerCompressionRule:
    rule_id = "BF-VERIFY-EULER-001"

    def evaluate(self, item: AxialSteelCheckInput) -> CodeCheckResult:
        if item.axial_force >= 0:
            status = CheckStatus.NOT_APPLICABLE
            demand = capacity = utilization = None
        else:
            demand = abs(item.axial_force)
            inverse_length = 1.0 / item.effective_length_factor / item.length
            capacity = (
                item.elastic_modulus * item.inertia_min * inverse_length**2
                * math.pi * math.pi
            )
            utilization = None if capacity <= 0.0 or not math.isfinite(capacity) else demand / capacity
            status = (
                CheckStatus.PASS
                if utilization is not None and utilization <= 1.0
                else CheckStatus.FAIL
            )
        return CodeCheckResult(
            item.member_id, self.rule_id, "Verification Euler compression",
            status, demand, capacity, utilization, "N", "R = pi^2 E I / (K L)^2",
            "R = pi^2 * E * Imin / (K * L)^2", _VERIFY_SOURCE,
            assumptions=("Ideal elastic pin-ended-equivalent column.",),
            omissions=("Imperfections, residual stress, local and torsional buckling.",),
        )


def verification_axial_steel_ruleset() -> RuleSet:
    return RuleSet(
        "beamfem.validation.axial-steel",
        "1.0",
        (VerificationTensionRule(), VerificationEulerCompressionRule()),
        status_label="verification_only",
    )


_AISC_SPECIFICATION_URL = (
    "https://www.aisc.org/aisc/publications/current-standards/aisc-360/"
)
_AISC_ERRATA_URL = (
    "https://www.aisc.org/globalassets/aisc/publications/revisions-and-errata/"
    "errata_360-22_1st-printing_01.23.2025.pdf"
)
_AISC_ERRATA_CHECKED = "2026-09-02"


def _aisc_citation(section: str, equation: str) -> RuleCitation:
    return RuleCitation(
        document="ANSI/AISC 360-22 Specification for Structural Steel Buildings",
        edition="2022, first printing; preview adapter",
        section=section,
        equation=equation,
        source_url=_AISC_SPECIFICATION_URL,
        errata_url=_AISC_ERRATA_URL,
        errata_issue_date="2025-01-23",
        errata_checked_on=_AISC_ERRATA_CHECKED,
    )


class AISC36022GrossYieldingPreview:
    rule_id = "AISC360-22-PREVIEW-D2-1-LRFD"

    def evaluate(self, item: AxialSteelCheckInput) -> CodeCheckResult:
        citation = _aisc_citation("D2(a)", "D2-1")
        if item.axial_force < 0:
            status = CheckStatus.NOT_APPLICABLE
            demand = capacity = utilization = None
        elif item.axial_force == 0:
            status = CheckStatus.NOT_APPLICABLE
            demand = capacity = utilization = None
        elif not item.gross_section_yielding_only_confirmed:
            status = CheckStatus.NOT_VERIFIED
            demand, capacity, utilization = item.axial_force, None, None
        else:
            phi = 0.90
            demand = item.axial_force
            capacity = phi * item.yield_stress * item.area
            if math.isfinite(capacity):
                utilization = demand / capacity
                status = CheckStatus.PASS if utilization <= 1.0 else CheckStatus.FAIL
            else:
                capacity = utilization = None
                status = CheckStatus.NOT_VERIFIED
        return CodeCheckResult(
            item.member_id, self.rule_id, "AISC 360-22 gross-section tensile yielding preview",
            status, demand, capacity, utilization, "N", "phi_t Pn = phi_t Fy Ag",
            f"0.90 * {item.yield_stress:.12g} * {item.area:.12g}", citation,
            coefficients=(CoefficientTrace("phi_t", 0.90, "LRFD resistance factor", citation),),
            assumptions=("Concentric axial LRFD demand; gross-section yielding is user-confirmed.",),
            omissions=("D2-2 net-section rupture and all connection limit states.",),
        )


class AISC36022FlexuralBucklingPreview:
    rule_id = "AISC360-22-PREVIEW-E3-LRFD"

    def evaluate(self, item: AxialSteelCheckInput) -> CodeCheckResult:
        citation = _aisc_citation("E3", "E3-1 through E3-4")
        if item.axial_force > 0:
            return CodeCheckResult(
                item.member_id, self.rule_id, "AISC 360-22 flexural buckling preview",
                CheckStatus.NOT_APPLICABLE, None, None, None, "N", "phi_c Pn = phi_c Fcr Ag",
                "not applicable to tension", citation,
            )
        if item.axial_force == 0:
            return CodeCheckResult(
                item.member_id, self.rule_id, "AISC 360-22 flexural buckling preview",
                CheckStatus.NOT_APPLICABLE, None, None, None, "N",
                "phi_c Pn = phi_c Fcr Ag", "zero-force policy: no axial check",
                citation,
            )
        demand = abs(item.axial_force)
        if not (
            item.nonslender_elements_confirmed
            and item.flexural_buckling_controls_confirmed
            and item.governing_axis is not None
        ):
            return CodeCheckResult(
                item.member_id, self.rule_id, "AISC 360-22 flexural buckling preview",
                CheckStatus.NOT_VERIFIED, demand, None, None, "N",
                "phi_c Pn = phi_c Fcr Ag", "scope assertions missing", citation,
                assumptions=(
                    "Requires nonslender elements, flexural buckling control, and an "
                    "explicit user-confirmed governing axis for inertia_min.",
                ),
                omissions=("E4/E5/E6/E7 and Chapter C stability are not evaluated.",),
            )
        radius = math.sqrt(item.inertia_min / item.area)
        # Compute the inverse slenderness by sequential division. This avoids
        # overflow in (K L / r)^2 for valid but extreme finite SI inputs.
        inverse_slenderness = radius / item.effective_length_factor / item.length
        elastic_stress = (
            item.elastic_modulus * inverse_slenderness**2 * math.pi * math.pi
        )
        if not math.isfinite(elastic_stress):
            return CodeCheckResult(
                item.member_id, self.rule_id, "AISC 360-22 flexural buckling preview",
                CheckStatus.NOT_VERIFIED, demand, None, None, "N",
                "phi_c Pn = phi_c Fcr Ag", "derived Fe is outside finite range",
                citation,
                assumptions=("All input fields use the documented SI units.",),
                omissions=("Numerical range exceeded; independent calculation required.",),
            )
        ratio = math.inf if elastic_stress <= 0.0 else item.yield_stress / elastic_stress
        if math.isfinite(ratio) and ratio <= 2.25:
            critical_stress = 0.658**ratio * item.yield_stress
            branch = "E3-2"
        else:
            critical_stress = 0.877 * elastic_stress
            branch = "E3-3"
        phi = 0.90
        capacity = phi * critical_stress * item.area
        utilization = None if capacity <= 0.0 else demand / capacity
        if utilization is not None and not math.isfinite(utilization):
            utilization = None
        status = (
            CheckStatus.PASS
            if utilization is not None and utilization <= 1.0
            else CheckStatus.FAIL
        )
        return CodeCheckResult(
            item.member_id, self.rule_id, "AISC 360-22 flexural buckling preview",
            status, demand, capacity, utilization, "N", "phi_c Pn = phi_c Fcr Ag",
            f"branch={branch}; Fe={elastic_stress:.12g}; Fcr={critical_stress:.12g}", citation,
            coefficients=(
                CoefficientTrace(
                    "phi_c", phi, "LRFD resistance factor",
                    _aisc_citation("E1", "LRFD compression resistance factor"),
                ),
                CoefficientTrace("lambda_boundary", 2.25, "Fy/Fe branch boundary", citation),
                CoefficientTrace("inelastic_base", 0.658, "E3-2 coefficient", citation),
                CoefficientTrace("elastic_factor", 0.877, "E3-3 coefficient", citation),
            ),
            assumptions=(
                "Concentric axial LRFD demand.",
                "User confirmed nonslender elements and flexural buckling control.",
                f"inertia_min is user-confirmed governing axis {item.governing_axis!r}.",
            ),
            omissions=(
                "Torsional/flexural-torsional buckling, slender elements, system stability, "
                "combined actions, fatigue, seismic, and connection checks.",
            ),
        )


def aisc360_22_axial_lrfd_preview_ruleset() -> RuleSet:
    """Return the non-certifying, external-review-required preview adapter."""

    return RuleSet(
        "ansi-aisc-360-22.axial-lrfd.preview",
        "0.1.0",
        (AISC36022GrossYieldingPreview(), AISC36022FlexuralBucklingPreview()),
        status_label="preview_not_certified",
    )
