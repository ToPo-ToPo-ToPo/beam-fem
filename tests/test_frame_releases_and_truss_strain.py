import numpy as np
import pytest

from beamfem import Material, Model, Section, UX, UY, recover_forces, solve_static
from beamfem.io import build_discrete_problem, validate_problem_spec
from beamfem.io.schema import SchemaValidationError
from beamfem.model import RZ
from benchmarks.quantum_truss.generate_cases import generate_case


def test_truss_public_strain_and_extension_are_tension_positive():
    model = Model()
    n0, n1 = model.add_node(0.0, 0.0), model.add_node(2.0, 0.0)
    material = Material(E=200e9, nu=0.3, rho=7850.0)
    section = Section(A=2e-4, Iy=1e-8, Iz=1e-8, J=2e-8)
    model.add_truss(n0, n1, material, section)
    model.fix(n0)
    model.fix(n1, [UY, 2, 3, 4, 5])
    model.add_load(n1, UX, 40_000.0)
    result = recover_forces(model, solve_static(model))[0]
    assert result.axial_strain == pytest.approx(40_000.0 / (material.E * section.A))
    assert result.axial_extension == pytest.approx(result.axial_strain * 2.0)


def test_pin_pin_frame_end_releases_match_closed_form_and_zero_end_moments():
    model = Model()
    n0, nc, n1 = (model.add_node(x, 0.0) for x in (0.0, 2.0, 4.0))
    material = Material(E=200e9, nu=0.3, rho=7850.0)
    section = Section.rectangle(1.0, 0.2)
    model.add_element(n0, nc, material, section, release_n1=(RZ,))
    model.add_element(nc, n1, material, section, release_n2=(RZ,))
    model.fix_to_plane_xy()
    model.fix(n0, [UX, UY])
    model.fix(n1, [UY])
    load, span = 1000.0, 4.0
    model.add_load(nc, UY, -load)
    static = solve_static(model)
    forces = recover_forces(model, static)
    expected = -(load * span**3 / (48 * material.E * section.Iz)
                 + load * span / (4 * section.ky * material.G * section.A))
    assert static.node_disp(nc)[UY] == pytest.approx(expected, rel=1e-11)
    assert forces[0].ends("Mz")[0] == pytest.approx(0.0, abs=1e-9)
    assert forces[1].ends("Mz")[1] == pytest.approx(0.0, abs=1e-9)


def test_schema_builds_frame_end_releases_and_rejects_truss_releases():
    document = generate_case("small")
    document["members"][0]["end_releases"] = {"n1": ["RZ"], "n2": ["RY"]}
    validated = validate_problem_spec(document)
    element = build_discrete_problem(validated).problem.model.elements[0]
    assert element.release_n1 == (RZ,)
    assert element.release_n2 == (4,)

    document["members"][0]["member_type"] = "truss"
    with pytest.raises(SchemaValidationError, match="only valid for frame"):
        validate_problem_spec(document)


def test_release_stiffness_remains_symmetric():
    model = Model()
    n0, n1 = model.add_node(0.0, 0.0), model.add_node(1.0, 0.0)
    material = Material(E=200e9, nu=0.3, rho=7850.0)
    section = Section.rectangle(0.1, 0.2)
    model.add_element(n0, n1, material, section, release_n1=(RZ,), release_n2=(RZ,))
    from beamfem.assembly import assemble_stiffness

    stiffness = assemble_stiffness(model).toarray()
    assert np.allclose(stiffness, stiffness.T, rtol=0.0, atol=1e-10)
