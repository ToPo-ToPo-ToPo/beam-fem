import math

import numpy as np
import pytest

from beamfem import Material, Model, Section, UX, UY, UZ, solve_modes
from beamfem.modal import assemble_lumped_mass


def _bar(*, density=7850.0):
    model = Model()
    first = model.add_node(0.0, 0.0, 0.0)
    second = model.add_node(2.0, 0.0, 0.0)
    material = Material(200e9, rho=density)
    section = Section(A=0.01, Iy=1e-6, Iz=1e-6, J=1e-6)
    model.add_truss(first, second, material, section)
    model.fix(first)
    model.fix(second, [UY, UZ])
    return model, second, material, section


def test_single_bar_axial_frequency_matches_lumped_mass_closed_form():
    model, second, material, section = _bar()
    result = solve_modes(model, number=1)
    expected_omega = math.sqrt(
        (material.E * section.A / 2.0) /
        (material.rho * section.A * 2.0 / 2.0)
    )
    assert result.circular_frequencies[0] == pytest.approx(expected_omega)
    assert abs(result.node_mode(0, second)[UX]) == pytest.approx(1.0)
    assert result.modes.shape == (model.n_dof, 1)


def test_modal_mass_is_symmetric_and_requires_density():
    model, *_ = _bar()
    mass = assemble_lumped_mass(model)
    assert np.array_equal(mass.toarray(), mass.toarray().T)
    zero, *_ = _bar(density=0.0)
    with pytest.raises(ValueError, match="positive material density"):
        solve_modes(zero)


def test_modal_rejects_mechanism_without_positive_modes():
    model, *_ = _bar()
    model.constraints.clear()
    with pytest.raises(ValueError):
        solve_modes(model)


def test_modal_rejects_zero_stiffness_transverse_massive_dof():
    model, second, *_ = _bar()
    # Remove only the tip UY restraint. The axial mode remains computable, but
    # accepting it would hide a genuine transverse rigid-body mechanism.
    model.constraints.pop((second, UY))
    with pytest.raises(ValueError, match="free massive DOFs have zero stiffness"):
        solve_modes(model)
