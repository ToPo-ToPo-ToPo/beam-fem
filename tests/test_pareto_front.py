import pytest

from beamfem import Material, Model, Section
from beamfem.optimize import (
    DesignState, DiscreteStructuralProblem, ParetoFrontBackend,
    SectionCatalog, SectionOption,
)
from beamfem.optimize.backends import SolverLimits


def _tradeoff_problem():
    material = Material(E=1000.0, nu=0.3, rho=1.0)
    light = Section(A=1.0, Iy=1.0, Iz=1.0, J=1.0)
    cheap = Section(A=2.0, Iy=1.0, Iz=1.0, J=1.0)
    model = Model()
    n0, n1 = model.add_node(0, 0), model.add_node(1, 0)
    model.add_truss(n0, n1, material, light)
    model.fix(n0)
    model.fix(n1, [1, 2, 3, 4, 5])
    catalog = SectionCatalog("tradeoff", [
        SectionOption("light", light, material, cost_per_kg=10.0, carbon_per_kg=1.0),
        SectionOption("cheap", cheap, material, cost_per_kg=1.0, carbon_per_kg=1.0),
    ])
    return DiscreteStructuralProblem(
        model, [catalog], initial_design=DesignState((0,)),
    )


def test_exact_pareto_front_retains_nondominated_tradeoff_and_serializes():
    result = ParetoFrontBackend(objectives=("mass", "cost")).solve(_tradeoff_problem())
    assert result.status == "success"
    assert [point.design.choices for point in result.points] == [(0,), (1,)]
    assert result.points[0].objectives == {"mass": 1.0, "cost": 10.0}
    assert result.points[1].objectives == {"mass": 2.0, "cost": 2.0}
    assert result.as_dict()["solver_metadata"]["global_for_enumerated_scope"] is True


def test_pareto_front_respects_shared_evaluation_budget():
    result = ParetoFrontBackend(objectives=("mass",)).solve(
        _tradeoff_problem(), limits=SolverLimits(max_evaluations=1)
    )
    assert result.status == "stopped"
    assert result.evaluations == 1
    assert len(result.points) == 1
    assert result.solver_metadata["global_for_enumerated_scope"] is False


def test_pareto_rejects_unknown_objective():
    with pytest.raises(ValueError, match="mass, cost"):
        ParetoFrontBackend(objectives=("stiffness",))
