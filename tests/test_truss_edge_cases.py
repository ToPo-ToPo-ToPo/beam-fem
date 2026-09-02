"""Independent edge-case review for the axial TrussElement integration."""

import numpy as np
import pytest

from beamfem import (
    Material, Model, Section, StructuralMechanismError, UX, UY, UZ,
    recover_forces, solve_static,
)
from beamfem.truss3d import truss_axial_force, truss_stiffness_global


STEEL = Material(E=200e9, nu=0.3, rho=7850.0)
SECTION = Section(A=4e-4, Iy=0.0, Iz=0.0, J=0.0)


def test_unloaded_translational_mechanism_is_not_silently_removed():
    model = Model()
    fixed = model.add_node(0.0, 0.0, 0.0)
    tip = model.add_node(1.0, 0.0, 0.0)
    model.add_truss(fixed, tip, STEEL, SECTION)
    model.pin(fixed)
    model.add_load(tip, UX, 1000.0)
    with pytest.raises(StructuralMechanismError) as caught:
        solve_static(model)
    assert {tip * 6 + UY, tip * 6 + UZ} <= set(caught.value.dofs)


def test_truss_only_rotations_are_ignored_but_applied_moment_is_rejected():
    model = Model()
    fixed = model.add_node(0.0, 0.0, 0.0)
    tip = model.add_node(1.0, 0.0, 0.0)
    model.add_truss(fixed, tip, STEEL, SECTION)
    model.pin(fixed)
    model.fix(tip, [UY, UZ])
    model.add_load(tip, UX, 1000.0)
    assert solve_static(model).node_disp(tip)[UX] > 0.0
    model.add_load(tip, 5, 10.0)
    with pytest.raises(StructuralMechanismError, match="荷重自由度"):
        solve_static(model)


def test_truss_stiffness_is_symmetric_positive_semidefinite_and_rigid_translation_free():
    p1 = np.array([-2.0, 1.0, 0.5])
    p2 = np.array([3.0, -4.0, 2.0])
    stiffness = truss_stiffness_global(p1, p2, STEEL, SECTION)
    assert np.allclose(stiffness, stiffness.T)
    assert np.linalg.eigvalsh(stiffness).min() >= -1e-6
    translation = np.zeros(12)
    translation[:3] = translation[6:9] = [0.7, -0.2, 1.1]
    assert np.allclose(stiffness @ translation, 0.0, atol=1e-8)


def test_zero_length_and_nonfinite_element_inputs_are_rejected():
    displacement = np.zeros(12)
    for p2 in (np.zeros(3), np.array([np.nan, 0.0, 0.0]), np.array([np.inf, 0.0, 0.0])):
        with pytest.raises(ValueError, match="有限"):
            truss_stiffness_global(np.zeros(3), p2, STEEL, SECTION)
        with pytest.raises(ValueError, match="有限"):
            truss_axial_force(np.zeros(3), p2, STEEL, SECTION, displacement)
    with pytest.raises(ValueError, match="3成分"):
        truss_stiffness_global(np.zeros(2), np.ones(2), STEEL, SECTION)
    with pytest.raises(ValueError, match="12成分"):
        truss_axial_force(np.zeros(3), np.ones(3), STEEL, SECTION, np.zeros(6))
    invalid_section = Section(A=0.0, Iy=0.0, Iz=0.0, J=0.0)
    with pytest.raises(ValueError, match="ヤング率と断面積"):
        truss_stiffness_global(np.zeros(3), np.ones(3), STEEL, invalid_section)


def test_mixed_frame_truss_force_order_and_pure_axial_stress_are_stable():
    model = Model()
    fixed = model.add_node(0.0, 0.0, 0.0)
    joint = model.add_node(1.0, 0.0, 0.0)
    tip = model.add_node(2.0, 0.0, 0.0)
    model.add_element(fixed, joint, STEEL, Section.rectangle(0.05, 0.08))
    model.add_truss(joint, tip, STEEL, SECTION)
    model.fix(fixed)
    model.fix(tip, [UY, UZ])
    model.add_load(tip, UX, 2500.0)
    forces = recover_forces(model, solve_static(model))
    assert len(forces.elements) == 2
    assert np.isclose(forces[1].ends("N")[0], 2500.0)
    assert forces[1].max_abs("Vy") == forces[1].max_abs("Mz") == 0.0


def test_area_only_truss_stress_recovery_stays_finite():
    model = Model()
    fixed = model.add_node(0.0, 0.0, 0.0)
    tip = model.add_node(1.0, 0.0, 0.0)
    model.add_truss(fixed, tip, STEEL, SECTION)
    model.pin(fixed)
    model.fix(tip, [UY, UZ])
    model.add_load(tip, UX, 4000.0)
    force = recover_forces(model, solve_static(model))[0]
    assert force.stress_ends("sigma_b") == (0.0, 0.0)
    assert np.all(np.isfinite(force.stress_ends("sigma_max")))
    assert force.stress_ends("sigma_max")[0] == pytest.approx(4000.0 / SECTION.A)
