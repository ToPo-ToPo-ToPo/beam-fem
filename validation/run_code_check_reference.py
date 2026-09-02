"""Generate deterministic hand-calculation evidence for preview design rules."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import subprocess

import beamfem

from beamfem.validation import (
    AxialSteelCheckInput,
    CombinedSteelCheckInput,
    SectionClass,
    aisc360_22_axial_lrfd_preview_ruleset,
    aisc360_22_combined_lrfd_preview_ruleset,
    aisc360_22_i_shape_flexural_classification,
)


ROOT = Path(__file__).resolve().parents[1]
_AISC_V16_DESIGN_EXAMPLES_URL = (
    "https://www.aisc.org/globalassets/aisc/university-programs/teaching-aids/"
    "first-semester-design-examples---v16.0.pdf"
)


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def generate_evidence() -> dict:
    axial_input = AxialSteelCheckInput(
        member_id="T1", axial_force=100_000.0, area=1.0e-3,
        inertia_min=8.0e-8, length=2.0, elastic_modulus=200.0e9,
        yield_stress=250.0e6, gross_section_yielding_only_confirmed=True,
    )
    tension = aisc360_22_axial_lrfd_preview_ruleset().evaluate([axial_input]).results[0]
    expected_tension_capacity = 0.90 * 250.0e6 * 1.0e-3

    combined_input = CombinedSteelCheckInput(
        member_id="BM1", axial_demand=300.0, axial_capacity=1000.0,
        moment_y_demand=100.0, moment_y_capacity=500.0,
        moment_z_demand=50.0, moment_z_capacity=250.0,
        symmetric_section_confirmed=True, second_order_effects_included=True,
        capacities_independently_verified=True,
    )
    combined = aisc360_22_combined_lrfd_preview_ruleset().evaluate([combined_input]).results[0]
    expected_interaction = 0.3 + 8.0 / 9.0 * (0.2 + 0.2)

    # AISC 16th Edition companion, Example H.1A (LRFD).  The example publishes
    # Pu=400 kip, Mux=250 kip-ft, Muy=80 kip-ft and the tabulated combined
    # strength parameters p=0.887e-3/kip, bx=1.38e-3/(kip-ft), and
    # by=2.85e-3/(kip-ft).  For the H1-1a adapter, Pc=1/p and Mc=(8/9)/b.
    # This independently published example exercises the H1-1a branch.
    official_input = CombinedSteelCheckInput(
        member_id="AISC-v16-H.1A-LRFD",
        axial_demand=400.0,
        axial_capacity=1.0 / (0.887e-3),
        moment_y_demand=250.0,
        moment_y_capacity=(8.0 / 9.0) / (1.38e-3),
        moment_z_demand=80.0,
        moment_z_capacity=(8.0 / 9.0) / (2.85e-3),
        symmetric_section_confirmed=True,
        second_order_effects_included=True,
        capacities_independently_verified=True,
    )
    official = aisc360_22_combined_lrfd_preview_ruleset().evaluate([official_input]).results[0]
    official_expected = 0.887e-3 * 400.0 + 1.38e-3 * 250.0 + 2.85e-3 * 80.0

    root = math.sqrt(200.0e9 / 250.0e6)
    classification = aisc360_22_i_shape_flexural_classification(
        "W-test", elastic_modulus=200.0e9, yield_stress=250.0e6,
        flange_width_thickness_ratio=0.30 * root,
        web_depth_thickness_ratio=4.00 * root,
        ratios_confirmed=True,
    )
    checks = {
        "tension_capacity": math.isclose(
            float(tension.capacity), expected_tension_capacity, rel_tol=1e-12
        ),
        "combined_interaction": math.isclose(
            float(combined.utilization), expected_interaction, rel_tol=1e-12
        ),
        "aisc_example_h1a_interaction": math.isclose(
            float(official.utilization), official_expected, rel_tol=1e-12
        ),
        "aisc_example_h1a_branch": "branch=H1-1a" in official.substituted_expression,
        "flange_compact": classification.elements[0].classification is SectionClass.COMPACT,
        "web_noncompact": classification.elements[1].classification is SectionClass.NONCOMPACT,
        "governing_noncompact": classification.governing_class is SectionClass.NONCOMPACT,
        "external_review_retained": (
            tension.external_review_required
            and combined.external_review_required
            and classification.external_review_required
        ),
    }
    return {
        "evidence_schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "platform": platform.platform(), "python": platform.python_version(),
            "beamfem": beamfem.__version__, "git_commit": _git_commit(),
        },
        "scope": "AISC 360-22 preview hand-calculation regression; not certification",
        "sources": sorted({
            tension.citation.source_url,
            combined.citation.source_url,
            tension.citation.errata_url,
            combined.citation.errata_url,
            _AISC_V16_DESIGN_EXAMPLES_URL,
        }),
        "expected": {
            "tension_capacity": expected_tension_capacity,
            "combined_interaction": expected_interaction,
            "aisc_example_h1a_interaction": official_expected,
            "flange_class": "compact",
            "web_class": "noncompact",
        },
        "actual": {
            "tension_capacity": tension.capacity,
            "combined_interaction": combined.utilization,
            "aisc_example_h1a_interaction": official.utilization,
            "flange_class": classification.elements[0].classification.value,
            "web_class": classification.elements[1].classification.value,
        },
        "checks": checks,
        "passed": all(checks.values()),
        "external_review_required": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "validation" / "code_check_reference_evidence.json",
    )
    args = parser.parse_args()
    evidence = generate_evidence()
    args.output.write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if not evidence["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
