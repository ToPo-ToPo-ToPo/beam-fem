"""Traceable axial rules, preview scope gates, checksums, and SBOM tests."""

import hashlib

import pytest

from beamfem.validation import (
    AxialSteelCheckInput, CheckStatus, aisc360_22_axial_lrfd_preview_ruleset,
    build_dependency_audit, sha256_file, verification_axial_steel_ruleset,
)


def _member(force=100_000.0, **changes):
    values = dict(
        member_id="M1", axial_force=force, area=1.0e-3,
        inertia_min=8.0e-8, length=2.0, elastic_modulus=200.0e9,
        yield_stress=250.0e6,
    )
    values.update(changes)
    return AxialSteelCheckInput(**values)


def test_verification_rules_are_traceable_and_never_approval_eligible():
    run = verification_axial_steel_ruleset().evaluate([_member()])
    result = next(item for item in run.results if item.status is not CheckStatus.NOT_APPLICABLE)
    assert result.status is CheckStatus.PASS
    assert result.citation.document.startswith("beamfem")
    assert run.external_review_required
    assert not run.approval_eligible


def test_aisc_preview_tension_requires_scope_assertion_and_traces_errata():
    rules = aisc360_22_axial_lrfd_preview_ruleset()
    unconfirmed = rules.evaluate([_member()]).results[0]
    assert unconfirmed.status is CheckStatus.NOT_VERIFIED
    confirmed = rules.evaluate([
        _member(gross_section_yielding_only_confirmed=True)
    ]).results[0]
    assert confirmed.status is CheckStatus.PASS
    assert confirmed.capacity == pytest.approx(225_000.0)
    assert confirmed.citation.edition.startswith("2022")
    assert confirmed.citation.section == "D2(a)"
    assert confirmed.citation.equation == "D2-1"
    assert confirmed.citation.errata_issue_date == "2025-01-23"
    assert confirmed.citation.errata_checked_on == "2026-09-02"
    assert "aisc.org" in confirmed.citation.source_url


def test_aisc_preview_compression_scope_and_both_equation_branches():
    rules = aisc360_22_axial_lrfd_preview_ruleset()
    not_verified = rules.evaluate([_member(force=-50_000.0)]).results[1]
    assert not_verified.status is CheckStatus.NOT_VERIFIED
    short = rules.evaluate([_member(
        force=-50_000.0, length=0.5, nonslender_elements_confirmed=True,
        flexural_buckling_controls_confirmed=True, governing_axis="y",
    )]).results[1]
    assert "branch=E3-2" in short.substituted_expression
    slender = rules.evaluate([_member(
        force=-5_000.0, length=20.0, nonslender_elements_confirmed=True,
        flexural_buckling_controls_confirmed=True, governing_axis="z",
    )]).results[1]
    assert "branch=E3-3" in slender.substituted_expression
    assert all(item.external_review_required for item in (short, slender))


def test_aisc_preview_requires_governing_axis_and_handles_extreme_length_and_zero():
    rules = aisc360_22_axial_lrfd_preview_ruleset()
    missing_axis = rules.evaluate([_member(
        force=-1.0, nonslender_elements_confirmed=True,
        flexural_buckling_controls_confirmed=True,
    )]).results[1]
    assert missing_axis.status is CheckStatus.NOT_VERIFIED
    extreme = rules.evaluate([_member(
        force=-1.0, length=1e308, nonslender_elements_confirmed=True,
        flexural_buckling_controls_confirmed=True,
        governing_axis="user_confirmed_minimum",
    )]).results[1]
    assert extreme.status is CheckStatus.FAIL
    assert extreme.utilization is None
    zero = rules.evaluate([_member(force=0.0)]).results
    assert all(item.status is CheckStatus.NOT_APPLICABLE for item in zero)


@pytest.mark.parametrize("field", ["area", "inertia_min", "length", "elastic_modulus", "yield_stress"])
def test_axial_rule_input_rejects_nonpositive_physics(field):
    with pytest.raises(ValueError):
        _member(**{field: 0.0})


def test_checksum_and_sbom_like_inventory(tmp_path):
    artifact = tmp_path / "result.json"
    artifact.write_bytes(b"beamfem-result")
    checksum = sha256_file(artifact)
    assert checksum.digest == hashlib.sha256(b"beamfem-result").hexdigest()
    audit = build_dependency_audit(packages=("definitely-not-installed-package",), artifacts=(artifact,))
    assert audit.format == "beamfem-sbom-lite/1.0"
    assert audit.dependencies[0].version == "not-installed"
    assert audit.artifacts[0].digest == checksum.digest
