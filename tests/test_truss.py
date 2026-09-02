"""軸力専用TrussElementの解析解・回転不変性・混在モデルV&V。"""

import math

import numpy as np
import pytest

from beamfem import (
    Material,
    Model,
    Section,
    StructuralMechanismError,
    TrussElement,
    UX,
    UY,
    UZ,
    recover_forces,
    solve_static,
)
from beamfem.io import SchemaValidationError, build_discrete_problem, validate_problem_spec
from beamfem.optimize import (
    DesignState,
    DiscreteStructuralProblem,
    LoadCase,
    LoadCombination,
    SectionCatalog,
    SectionOption,
)
from benchmarks.quantum_truss.generate_cases import generate_case


STEEL = Material(E=200e9, nu=0.3, rho=7850.0)
AREA = 4.0e-4
SECTION = Section(A=AREA, Iy=1e-8, Iz=1e-8, J=2e-8, name="bar")


@pytest.mark.parametrize("direction", [
    np.array([1.0, 0.0, 0.0]),
    np.array([2.0, -3.0, 4.0]) / math.sqrt(29.0),
])
def test_single_bar_axial_solution_is_coordinate_rotation_invariant(direction):
    length, load = 2.7, 12_000.0
    model = Model()
    fixed = model.add_node(0.0, 0.0, 0.0)
    tip = model.add_node(*(length * direction))
    model.add_truss(fixed, tip, STEEL, SECTION)
    model.pin(fixed)
    # 主材に直交する2本の無応力材で横方向剛体移動だけを拘束する。
    trial = np.array([0.0, 0.0, 1.0])
    if abs(float(trial @ direction)) > 0.9:
        trial = np.array([0.0, 1.0, 0.0])
    perpendicular_a = np.cross(direction, trial)
    perpendicular_a /= np.linalg.norm(perpendicular_a)
    perpendicular_b = np.cross(direction, perpendicular_a)
    for perpendicular in (perpendicular_a, perpendicular_b):
        anchor = model.add_node(*(length * direction + length * perpendicular))
        model.add_truss(tip, anchor, STEEL, SECTION)
        model.pin(anchor)
    for dof, value in enumerate(load * direction):
        model.add_load(tip, dof, value)

    result = solve_static(model)
    expected_extension = load * length / (STEEL.E * AREA)
    assert np.allclose(result.node_disp(tip)[:3], expected_extension * direction, rtol=1e-11)
    forces = recover_forces(model, result)
    force = forces[0]
    assert np.isclose(force.ends("N")[0], load, rtol=1e-11)
    assert force.max_abs("Vy") == force.max_abs("Mz") == 0.0
    assert np.allclose([forces[1].max_abs("N"), forces[2].max_abs("N")], 0.0, atol=1e-8)


def test_symmetric_triangle_truss_matches_closed_form():
    half_span, height, load = 1.5, 2.0, 30_000.0
    length = math.hypot(half_span, height)
    sine = height / length
    model = Model()
    left = model.add_node(-half_span, 0.0)
    right = model.add_node(half_span, 0.0)
    apex = model.add_node(0.0, height)
    model.add_truss(left, right, STEEL, SECTION)
    model.add_truss(left, apex, STEEL, SECTION)
    model.add_truss(right, apex, STEEL, SECTION)
    model.pin(left)
    model.pin(right)
    model.fix(apex, [UZ])
    model.add_load(apex, UY, -load)

    result = solve_static(model)
    expected = -load * length / (2.0 * STEEL.E * AREA * sine**2)
    assert np.isclose(result.node_disp(apex)[UY], expected, rtol=1e-11)
    forces = recover_forces(model, result)
    assert np.isclose(forces[0].max_abs("N"), 0.0, atol=1e-8)
    assert np.isclose(forces[1].ends("N")[0], -load / (2.0 * sine), rtol=1e-11)


def test_symmetric_3d_tripod_truss_matches_closed_form():
    radius, height, load = 1.2, 2.5, 24_000.0
    model = Model()
    base = [
        model.add_node(radius * math.cos(theta), radius * math.sin(theta), 0.0)
        for theta in (0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0)
    ]
    apex = model.add_node(0.0, 0.0, height)
    for node in base:
        model.add_truss(node, apex, STEEL, SECTION)
        model.pin(node)
    model.add_load(apex, UZ, -load)

    result = solve_static(model)
    length = math.hypot(radius, height)
    expected = -load * length**3 / (3.0 * STEEL.E * AREA * height**2)
    assert np.isclose(result.node_disp(apex)[UZ], expected, rtol=1e-11)
    assert np.allclose(result.node_disp(apex)[:2], 0.0, atol=1e-15)


def test_frame_and_truss_can_share_nodes_and_force_recovery():
    load = 8_000.0
    beam_length, truss_length = 1.2, 1.8
    model = Model()
    fixed = model.add_node(0.0, 0.0, 0.0)
    joint = model.add_node(beam_length, 0.0, 0.0)
    tip = model.add_node(beam_length + truss_length, 0.0, 0.0)
    beam_section = Section.rectangle(0.08, 0.10)
    model.add_element(fixed, joint, STEEL, beam_section)
    model.add_truss(joint, tip, STEEL, SECTION)
    model.fix(fixed)
    model.fix(tip, [UY, UZ])  # truss自由端の横移動をガイド拘束
    model.fix(tip, [UY, UZ])
    model.add_load(tip, UX, load)

    result = solve_static(model)
    expected = load * beam_length / (STEEL.E * beam_section.A)
    expected += load * truss_length / (STEEL.E * AREA)
    assert np.isclose(result.node_disp(tip)[UX], expected, rtol=1e-11)
    assert all(np.isclose(force.ends("N")[0], load) for force in recover_forces(model, result).elements)


def test_loaded_zero_stiffness_dof_reports_mechanism_and_dofs():
    model = Model()
    fixed = model.add_node(0.0, 0.0, 0.0)
    tip = model.add_node(1.0, 0.0, 0.0)
    model.add_truss(fixed, tip, STEEL, SECTION)
    model.pin(fixed)
    model.add_load(tip, UY, -1000.0)
    with pytest.raises(StructuralMechanismError, match="剛性") as caught:
        solve_static(model)
    assert tip * 6 + UY in caught.value.dofs


def test_truss_self_weight_and_discrete_evaluator():
    length = 2.0
    model = Model()
    fixed = model.add_node(0.0, 0.0, 0.0)
    tip = model.add_node(length, 0.0, 0.0)
    model.add_truss(fixed, tip, STEEL, SECTION)
    model.pin(fixed)
    model.fix(tip, [UY, UZ])
    catalog = SectionCatalog("bar", [SectionOption("bar", SECTION, STEEL)])
    problem = DiscreteStructuralProblem(
        model, [catalog],
        load_cases=[LoadCase("dead", {})],
        load_combinations=[LoadCombination("dead", {"dead": 1.0})],
        initial_design=DesignState([0]),
        self_weight=np.array([-9.80665, 0.0, 0.0]),
    )
    result = problem.evaluate(problem.initial_design)
    mass = STEEL.rho * AREA * length
    assert result.feasible
    assert np.isclose(result.analyses["dead"].static.reactions[UX], mass * 9.80665)
    assert isinstance(result.analyses["dead"].model.elements[0], TrussElement)


def test_schema_member_type_builds_truss_and_frame_remains_default():
    document = generate_case("small")
    document["members"][0]["member_type"] = "truss"
    built = build_discrete_problem(document)
    assert isinstance(built.problem.model.elements[0], TrussElement)
    assert not isinstance(built.problem.model.elements[1], TrussElement)

    document["members"][0]["member_type"] = "cable"
    with pytest.raises(SchemaValidationError, match="member_type"):
        validate_problem_spec(document)


def test_schema_allows_area_only_catalog_for_truss_but_not_frame():
    truss_document = generate_case("small")
    for member in truss_document["members"]:
        member["member_type"] = "truss"
    for option in truss_document["section_catalogs"]["round_bar"]:
        option.pop("I")
    truss_document["constraints"] = [
        constraint for constraint in truss_document["constraints"]
        if constraint["type"] != "euler_buckling"
    ]
    validate_problem_spec(truss_document)

    truss_document["members"][0]["member_type"] = "frame"
    with pytest.raises(SchemaValidationError, match="required for frame"):
        validate_problem_spec(truss_document)


def test_schema_requires_inertia_when_euler_buckling_is_requested():
    document = generate_case("small")
    for member in document["members"]:
        member["member_type"] = "truss"
    for option in document["section_catalogs"]["round_bar"]:
        option.pop("I")
    with pytest.raises(SchemaValidationError, match="euler_buckling"):
        validate_problem_spec(document)
