"""Independent hand calculations for section class and H1 interaction."""

import math

import pytest

from beamfem.validation import (
    CheckStatus,
    CombinedSteelCheckInput,
    SectionClass,
    aisc360_22_combined_lrfd_preview_ruleset,
    aisc360_22_i_shape_flexural_classification,
)


def test_i_shape_flexural_classification_reports_each_element_and_governing_class():
    root = math.sqrt(200e9 / 250e6)
    result = aisc360_22_i_shape_flexural_classification(
        "W-test",
        elastic_modulus=200e9,
        yield_stress=250e6,
        flange_width_thickness_ratio=0.30 * root,
        web_depth_thickness_ratio=4.00 * root,
        ratios_confirmed=True,
    )
    assert result.elements[0].classification is SectionClass.COMPACT
    assert result.elements[1].classification is SectionClass.NONCOMPACT
    assert result.governing_class is SectionClass.NONCOMPACT
    assert result.external_review_required
    assert result.elements[0].citation.section.startswith("Table B4.1b")


def test_i_shape_flexural_classification_rejects_invalid_ratios():
    with pytest.raises(ValueError):
        aisc360_22_i_shape_flexural_classification(
            "bad", elastic_modulus=200e9, yield_stress=250e6,
            flange_width_thickness_ratio=0.0, web_depth_thickness_ratio=20.0,
        )


def _combined(**changes):
    values = dict(
        member_id="C1",
        axial_demand=300.0,
        axial_capacity=1000.0,
        moment_y_demand=100.0,
        moment_y_capacity=500.0,
        moment_z_demand=50.0,
        moment_z_capacity=250.0,
        symmetric_section_confirmed=True,
        second_order_effects_included=True,
        capacities_independently_verified=True,
    )
    values.update(changes)
    return CombinedSteelCheckInput(**values)


def test_h1_high_axial_branch_matches_independent_hand_calculation():
    result = aisc360_22_combined_lrfd_preview_ruleset().evaluate([_combined()]).results[0]
    expected = 0.3 + (8.0 / 9.0) * (0.2 + 0.2)
    assert result.status is CheckStatus.PASS
    assert result.utilization == pytest.approx(expected)
    assert "H1-1a" in result.substituted_expression
    assert result.citation.edition.startswith("2022")


def test_h1_low_axial_branch_and_failure_are_reported():
    low = aisc360_22_combined_lrfd_preview_ruleset().evaluate([
        _combined(axial_demand=100.0)
    ]).results[0]
    assert low.utilization == pytest.approx(0.05 + 0.4)
    assert "H1-1b" in low.substituted_expression
    failed = aisc360_22_combined_lrfd_preview_ruleset().evaluate([
        _combined(moment_y_demand=400.0, moment_z_demand=200.0)
    ]).results[0]
    assert failed.status is CheckStatus.FAIL


def test_h1_requires_scope_and_second_order_assertions():
    result = aisc360_22_combined_lrfd_preview_ruleset().evaluate([
        _combined(second_order_effects_included=False)
    ]).results[0]
    assert result.status is CheckStatus.NOT_VERIFIED
    assert result.utilization is None
    assert result.external_review_required
