import math
import json

import numpy as np
import pytest

from beamfem import (
    BilinearIsotropicHardening,
    ElasticPerfectlyPlastic,
    Material,
    Model,
    NonlinearConvergenceError,
    NonlinearTrussSubproblem,
    Section,
    UX,
    UY,
    UZ,
    solve_nonlinear_truss,
)


E = 200.0e9
YIELD = 250.0e6
AREA = 1.0e-3
SECTION = Section(A=AREA, Iy=1.0e-8, Iz=1.0e-8, J=2.0e-8)


def guided_bar(reference_force: float = 0.0):
    model = Model()
    fixed = model.add_node(0.0, 0.0, 0.0)
    tip = model.add_node(2.0, 0.0, 0.0)
    model.add_truss(fixed, tip, Material(E), SECTION)
    model.pin(fixed)
    model.fix(tip, [UY, UZ])
    if reference_force:
        model.add_load(tip, UX, reference_force)
    return model, tip


def test_displacement_controlled_perfect_plastic_loading_unloading_reloading():
    model, tip = guided_bar()
    result = solve_nonlinear_truss(
        model,
        ElasticPerfectlyPlastic(E, YIELD),
        load_factors=[0.0, 1.0, 0.0, 0.5, 1.0],
        n_steps=4,
        displacement_pattern={(tip, UX): 4.0e-3},
    )
    assert result.converged and not result.limit_state.collapse_detected
    peak_steps = [step for step in result.history if step.load_factor == pytest.approx(1.0)]
    assert peak_steps[-1].elements[0].stress == pytest.approx(YIELD)
    assert result.element_states[0].plastic_strain == pytest.approx(0.75e-3)
    energies = [step.dissipated_energy for step in result.history]
    assert all(second >= first for first, second in zip(energies, energies[1:]))
    assert result.limit_state.first_yield_load_factor is not None
    assert result.limit_state.progressive_collapse_sequence[0].event == "first_yield"


def test_hardening_bar_force_cycle_has_analytical_residual_displacement_and_energy():
    post_yield = 10.0e9
    model, tip = guided_bar(300.0e3)
    result = solve_nonlinear_truss(
        model,
        BilinearIsotropicHardening(E, YIELD, post_yield),
        load_factors=[0.0, 1.0, 0.0],
        n_steps=10,
    )
    peak_strain = YIELD / E + (300.0e6 - YIELD) / post_yield
    residual_strain = peak_strain - 300.0e6 / E
    expected_residual = 2.0 * residual_strain
    expected_energy = YIELD * residual_strain * AREA * 2.0
    assert result.converged and result.residual_displacement is not None
    assert result.node_disp(tip)[UX] == pytest.approx(expected_residual, rel=1.0e-10)
    assert result.element_states[0].plastic_strain == pytest.approx(residual_strain)
    assert result.dissipated_energy == pytest.approx(expected_energy)
    assert abs(result.reactions[tip * 6 + UX]) < 1.0e-7
    payload = result.as_dict()
    assert payload["element_states"][0]["plastic_strain"] == pytest.approx(residual_strain)
    assert "NaN" not in json.dumps(payload, allow_nan=False)


def test_symmetric_two_bar_hardening_solution_and_corotational_option():
    post_yield = 10.0e9
    model = Model()
    left = model.add_node(-1.0, 0.0, 0.0)
    right = model.add_node(1.0, 0.0, 0.0)
    apex = model.add_node(0.0, 1.0, 0.0)
    model.add_truss(left, apex, Material(E), SECTION)
    model.add_truss(right, apex, Material(E), SECTION)
    model.pin(left)
    model.pin(right)
    model.fix(apex, [UZ])
    model.add_load(apex, UY, -450.0e3)
    material = BilinearIsotropicHardening(E, YIELD, post_yield)

    small = solve_nonlinear_truss(model, material, n_steps=20)
    member_stress = -450.0e3 / (2.0 * AREA / math.sqrt(2.0))
    member_strain = -(
        YIELD / E + (abs(member_stress) - YIELD) / post_yield
    )
    expected_vertical = 2.0 * member_strain
    assert small.converged
    assert small.node_disp(apex)[UX] == pytest.approx(0.0, abs=1.0e-13)
    assert small.node_disp(apex)[UY] == pytest.approx(expected_vertical, rel=1.0e-9)
    assert all(state.equivalent_plastic_strain > 0.0 for state in small.element_states)

    corotational = solve_nonlinear_truss(
        model, material, n_steps=20, geometric_nonlinear=True
    )
    assert corotational.converged
    assert corotational.node_disp(apex)[UY] < small.node_disp(apex)[UY]
    assert corotational.history[-1].residual_norm < 1.0e-3


def test_corotational_truss_is_objective_for_a_finite_rigid_rotation():
    model = Model()
    fixed = model.add_node(0.0, 0.0, 0.0)
    tip = model.add_node(1.0, 0.0, 0.0)
    model.add_truss(fixed, tip, Material(E), SECTION)
    model.pin(fixed)
    model.fix(tip, [UZ])
    pattern = {(tip, UX): -1.0, (tip, UY): 1.0}
    elastic_range = ElasticPerfectlyPlastic(E, 1.0e15)
    corotational = solve_nonlinear_truss(
        model,
        elastic_range,
        load_factors=[0.0, 1.0],
        n_steps=1,
        displacement_pattern=pattern,
        geometric_nonlinear=True,
    )
    small_rotation = solve_nonlinear_truss(
        model,
        elastic_range,
        load_factors=[0.0, 1.0],
        n_steps=1,
        displacement_pattern=pattern,
        geometric_nonlinear=False,
    )
    assert corotational.history[-1].elements[0].strain == pytest.approx(0.0)
    assert corotational.history[-1].elements[0].stress == pytest.approx(0.0)
    assert small_rotation.history[-1].elements[0].strain == pytest.approx(-1.0)


def test_progressive_collapse_sequence_orders_successive_member_yielding():
    model = Model()
    fixed = model.add_node(0.0, 0.0, 0.0)
    tip = model.add_node(2.0, 0.0, 0.0)
    model.add_truss(fixed, tip, Material(E), SECTION)
    model.add_truss(fixed, tip, Material(E), SECTION)
    model.pin(fixed)
    model.fix(tip, [UY, UZ])
    result = solve_nonlinear_truss(
        model,
        (
            BilinearIsotropicHardening(E, 200.0e6, 10.0e9),
            BilinearIsotropicHardening(E, 300.0e6, 10.0e9),
        ),
        load_factors=[0.0, 1.0],
        n_steps=8,
        displacement_pattern={(tip, UX): 4.0e-3},
    )
    events = [
        event for event in result.limit_state.progressive_collapse_sequence
        if event.event == "first_yield"
    ]
    assert [event.elements for event in events] == [(0,), (1,)]
    assert events[0].load_factor < events[1].load_factor


def test_perfect_plastic_force_control_reports_limit_and_nonconvergence():
    model, _ = guided_bar(300.0e3)
    result = solve_nonlinear_truss(
        model,
        ElasticPerfectlyPlastic(E, YIELD),
        n_steps=4,
        minimum_step=2.0e-3,
    )
    expected_limit = YIELD * AREA / 300.0e3
    assert not result.converged and not result.feasible
    assert result.limit_state.collapse_detected
    assert result.limit_state.last_converged_load_factor == pytest.approx(
        expected_limit, abs=3.0e-3
    )
    assert result.limit_state.failed_load_factor > result.limit_state.last_converged_load_factor
    assert result.limit_state.failure_residual_norm > 0.0
    assert result.limit_state.progressive_collapse_sequence[-1].event == (
        "global_nonconvergence"
    )
    assert "singular tangent" in result.diagnostic
    with pytest.raises(NonlinearConvergenceError):
        solve_nonlinear_truss(
            model,
            ElasticPerfectlyPlastic(E, YIELD),
            n_steps=4,
            minimum_step=2.0e-3,
            raise_on_failure=True,
        )


def test_nonlinear_subproblem_marks_nonconvergence_infeasible_for_optimizers():
    model, _ = guided_bar(300.0e3)
    subproblem = NonlinearTrussSubproblem(
        initial_design=(0,),
        domains=((0,),),
        model_factory=lambda _design: model,
        material_factory=lambda _design: ElasticPerfectlyPlastic(E, YIELD),
        objective=lambda _design, _model: 1.0,
        n_steps=4,
        minimum_step=2.0e-3,
    )
    evaluation = subproblem.evaluate((0,))
    assert not evaluation.feasible
    assert not evaluation.constraints[0].satisfied
    assert evaluation.constraints[0].utilization > 1.0


def test_nonlinear_solver_rejects_frame_and_invalid_material_mapping():
    model = Model()
    first = model.add_node(0.0, 0.0, 0.0)
    second = model.add_node(1.0, 0.0, 0.0)
    model.add_element(first, second, Material(E), Section.rectangle(0.1, 0.1))
    with pytest.raises(ValueError, match="frame"):
        solve_nonlinear_truss(model, ElasticPerfectlyPlastic(E, YIELD))

    truss, _ = guided_bar()
    with pytest.raises(ValueError, match="mapping mismatch"):
        solve_nonlinear_truss(truss, {})
