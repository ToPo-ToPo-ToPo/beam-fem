import numpy as np

from beamfem import Material, Model, Section, UX
from beamfem.solver import factorize_static, solve_static


def test_factorization_reuses_stiffness_for_multiple_load_vectors():
    material = Material(E=200e9, nu=0.3, rho=7850.0)
    section = Section.circle(d=0.02)
    model = Model()
    fixed = model.add_node(0.0, 0.0, 0.0)
    tip = model.add_node(2.0, 0.0, 0.0)
    model.add_element(fixed, tip, material, section)
    model.fix(fixed)
    factorization = factorize_static(model)

    model.add_load(tip, UX, 1000.0)
    first = factorization.solve_model(model)
    reference_first = solve_static(model)
    model.nodal_loads[(tip, UX)] = 2500.0
    second = factorization.solve_model(model)
    reference_second = solve_static(model)

    assert np.allclose(first.u, reference_first.u)
    assert np.allclose(second.u, reference_second.u)
    assert np.isclose(second.node_disp(tip)[UX] / first.node_disp(tip)[UX], 2.5)
