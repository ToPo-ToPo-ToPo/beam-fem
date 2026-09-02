"""共通離散構造問題・FEM評価器の検証。"""

import math
import json

import numpy as np

from beamfem import Material, Model, Section, UX, UY
from beamfem.optimize import (
    DesignState,
    DiscreteDisplacementLimit,
    DiscreteStructuralProblem,
    EulerBucklingLimit,
    ForbiddenMembers,
    LoadCase,
    LoadCombination,
    MassObjective,
    MaxSectionTypes,
    ActiveMemberCount,
    SymmetryPairs,
    Connectivity,
    MemberLengthRange,
    RelativeDisplacementLimit,
    RequiredMembers,
    SameSectionGroup,
    SectionCatalog,
    SectionOption,
    StressLimit,
)


STEEL = Material(E=200e9, nu=0.3, rho=7850.0, name="steel")
SEC_SMALL = Section.rectangle(0.05, 0.05, name="S")
SEC_LARGE = Section.rectangle(0.08, 0.08, name="L")


def catalog(with_off=True):
    options = []
    if with_off:
        options.append(SectionOption("OFF", None))
    options.extend([
        SectionOption("S", SEC_SMALL, tensile_strength=250e6, compressive_strength=250e6),
        SectionOption("L", SEC_LARGE, tensile_strength=250e6, compressive_strength=250e6),
    ])
    return SectionCatalog("square", options)


def axial_problem(*, load=10_000.0, constraints=(), combinations=None):
    model = Model()
    n0 = model.add_node(0.0, 0.0, 0.0)
    n1 = model.add_node(2.0, 0.0, 0.0)
    model.add_element(n0, n1, STEEL, SEC_SMALL)
    model.fix(n0)
    # 軸方向以外を拘束して純軸力1自由度モデルにする
    model.fix(n1, [1, 2, 3, 4, 5])
    cases = [LoadCase("P", {(n1, UX): load})]
    combos = combinations or [LoadCombination("service", {"P": 1.0})]
    return DiscreteStructuralProblem(
        model=model,
        catalogs=[catalog()],
        load_cases=cases,
        load_combinations=combos,
        constraints=constraints,
        objective=MassObjective(),
        initial_design=DesignState([1]),
    )


def test_section_catalog_and_design_names_validation():
    p = axial_problem()
    assert p.design_from_names(["L"]) == DesignState([2])
    assert p.catalogs[0].off_index == 0
    assert hash(DesignState([1])) == hash(DesignState([1]))


def test_axial_fem_mass_stress_displacement_and_machine_readable_result():
    P, L = 10_000.0, 2.0
    constraints = (
        StressLimit(tension=250e6, compression=250e6),
        DiscreteDisplacementLimit(node=1, dof=UX, maximum=1e-3),
    )
    p = axial_problem(load=P, constraints=constraints)
    r = p.evaluate(DesignState([1]))

    expected_mass = STEEL.rho * SEC_SMALL.A * L
    expected_u = P * L / (STEEL.E * SEC_SMALL.A)
    assert np.isclose(r.mass, expected_mass)
    assert np.isclose(r.objective, expected_mass)
    assert np.isclose(r.analyses["service"].static.node_disp(1)[UX], expected_u)
    stress = next(c for c in r.constraints if c.kind == "tensile_stress")
    assert np.isclose(stress.value, P / SEC_SMALL.A)
    assert r.feasible
    payload = r.as_dict()
    assert payload["governing_constraint"]["constraint_id"] in {"stress", "displacement"}
    assert payload["constraints"][0]["load_combination"] == "service"


def test_multiple_load_combinations_are_solved_and_governing_is_reported():
    p = axial_problem(
        constraints=(StressLimit(tension=250e6),),
        combinations=[LoadCombination("service", {"P": 1.0}), LoadCombination("ultimate", {"P": 1.5})],
    )
    r = p.evaluate(DesignState([1]))
    records = [x for x in r.constraints if x.kind == "tensile_stress"]
    assert {x.load_combination for x in records} == {"service", "ultimate"}
    assert max(records, key=lambda x: x.utilization).load_combination == "ultimate"


def test_euler_buckling_uses_compressive_force_and_minimum_inertia():
    P = -20_000.0
    p = axial_problem(load=P, constraints=(EulerBucklingLimit(),))
    r = p.evaluate(DesignState([1]))
    record = next(x for x in r.constraints if x.kind == "euler_buckling")
    expected = math.pi**2 * STEEL.E * min(SEC_SMALL.Iy, SEC_SMALL.Iz) / 2.0**2
    assert np.isclose(record.value, abs(P))
    assert np.isclose(record.limit, expected)


def test_topology_and_group_constraints_return_explicit_failures():
    model = Model()
    a = model.add_node(0, 0, 0)
    b = model.add_node(1, 0, 0)
    c = model.add_node(2, 0, 0)
    model.add_element(a, b, STEEL, SEC_SMALL)
    model.add_element(b, c, STEEL, SEC_SMALL)
    model.fix(a)
    model.fix(b, [1, 2, 3, 4, 5])
    model.fix(c, [1, 2, 3, 4, 5])
    problem = DiscreteStructuralProblem(
        model, [catalog(), catalog()],
        constraints=(
            RequiredMembers([0]),
            ForbiddenMembers([1]),
            SameSectionGroup([0, 1]),
            MaxSectionTypes(1),
        ),
        initial_design=DesignState([1, 2]),
    )
    result = problem.evaluate(DesignState([1, 2]))
    assert not result.feasible
    failed = {r.kind for r in result.constraints if not r.satisfied}
    assert failed == {"forbidden_member", "same_section_group", "max_section_types"}


def test_relative_displacement_and_cache():
    p = axial_problem(constraints=(RelativeDisplacementLimit(1, 0, UX, 1e-3),))
    first = p.evaluate(DesignState([1]))
    second = p.evaluate(DesignState([1]))
    assert first.feasible and second.cache_hit
    assert p._evaluator.cache_info == {"size": 1, "analyses": 1, "hits": 1}


def test_off_design_is_diagnosed_as_unstable_not_raised():
    p = axial_problem()
    result = p.evaluate(DesignState([0]))
    assert not result.feasible
    assert result.diagnostic is not None
    assert result.constraints[0].kind == "mechanism_or_singular_stiffness"
    assert result.constraints[0].utilization > 1e20
    json.dumps(result.as_dict(), allow_nan=False)


def test_member_count_symmetry_connectivity_and_length_constraints():
    model = Model()
    a = model.add_node(0, 0, 0)
    b = model.add_node(1, 0, 0)
    c = model.add_node(2, 0, 0)
    model.add_element(a, b, STEEL, SEC_SMALL)
    model.add_element(b, c, STEEL, SEC_SMALL)
    model.fix(a)
    model.fix(b, [1, 2, 3, 4, 5])
    model.fix(c, [1, 2, 3, 4, 5])
    problem = DiscreteStructuralProblem(
        model, [catalog(), catalog()],
        constraints=(
            ActiveMemberCount(minimum=2, maximum=2),
            SymmetryPairs([(0, 1)]),
            Connectivity([a, c]),
            MemberLengthRange(0.9, 1.1),
        ),
        initial_design=DesignState([1, 1]),
    )
    valid = problem.evaluate(DesignState([1, 1]))
    assert valid.feasible
    invalid = problem.evaluate(DesignState([1, 0]))
    failed = {r.kind for r in invalid.constraints if not r.satisfied}
    assert {"active_member_count", "symmetry_pair", "connectivity"} <= failed


def test_self_weight_is_added_as_equivalent_nodal_load_to_every_combination():
    length = 2.0
    model = Model()
    a = model.add_node(0, 0, 0)
    b = model.add_node(length, 0, 0)
    model.add_element(a, b, STEEL, SEC_SMALL)
    model.fix(a)
    problem = DiscreteStructuralProblem(
        model, [catalog(with_off=False)],
        load_cases=[LoadCase("empty", {})],
        load_combinations=[LoadCombination("dead", {"empty": 1.0})],
        self_weight=np.array([0.0, -9.80665, 0.0]),
        initial_design=DesignState([0]),
    )
    result = problem.evaluate(DesignState([0]))
    mass = STEEL.rho * SEC_SMALL.A * length
    assert result.feasible
    # 全重量は固定端反力と釣り合う（両端へ1/2ずつ集中荷重として作用）。
    assert np.isclose(result.analyses["dead"].static.reactions[a * 6 + UY], mass * 9.80665)
