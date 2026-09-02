"""End-to-end elastoplastic topology/section optimization example."""

from __future__ import annotations

import json

from beamfem import (
    BilinearIsotropicHardening,
    Material,
    Model,
    NonlinearTrussSubproblem,
    Section,
    UX,
    UY,
    UZ,
)
from beamfem.optimize.backends import ExactBackend, GreedyBackend
from beamfem.optimize.backends.base import design_values


ELASTIC_MODULUS = 200.0e9
YIELD_STRESS = 250.0e6
POST_YIELD_TANGENT = 10.0e9
DENSITY = 7850.0
LENGTH = 2.0
REFERENCE_LOAD = 330.0e3
AREAS = {0: 0.0, 1: 0.6e-3, 2: 1.3e-3}  # OFF / small / large


def build_problem() -> NonlinearTrussSubproblem:
    def selected(design):
        return tuple(int(value) for value in design_values(design))

    def model_factory(design):
        model = Model()
        fixed = model.add_node(0.0, 0.0, 0.0)
        tip = model.add_node(LENGTH, 0.0, 0.0)
        for choice in selected(design):
            area = AREAS[choice]
            if area > 0.0:
                section = Section(A=area, Iy=1.0e-8, Iz=1.0e-8, J=2.0e-8)
                model.add_truss(fixed, tip, Material(ELASTIC_MODULUS), section)
        model.pin(fixed)
        model.fix(tip, [UY, UZ])
        model.add_load(tip, UX, REFERENCE_LOAD)
        return model

    def material_factory(design):
        return tuple(
            BilinearIsotropicHardening(
                ELASTIC_MODULUS, YIELD_STRESS, POST_YIELD_TANGENT
            )
            for choice in selected(design)
            if AREAS[choice] > 0.0
        )

    def mass(design, _model):
        return DENSITY * LENGTH * sum(AREAS[choice] for choice in selected(design))

    return NonlinearTrussSubproblem(
        initial_design=(2, 2),
        domains=((0, 1, 2), (0, 1, 2)),
        model_factory=model_factory,
        material_factory=material_factory,
        objective=mass,
        mass=mass,
        load_factors=[0.0, 1.0, 0.0],
        n_steps=8,
        maximum_equivalent_plastic_strain=3.0e-3,
        maximum_residual_displacement=6.0e-3,
    )


def run_example() -> dict[str, object]:
    problem = build_problem()
    exact = ExactBackend().solve(problem)
    greedy = GreedyBackend(penalty=1.0e6, pairwise=True).solve(problem)
    return {
        "exact": exact.evaluation.as_dict(),
        "greedy": greedy.evaluation.as_dict(),
        "rejected_all_off": problem.evaluate((0, 0)).as_dict(),
        "rejected_understrength": problem.evaluate((1, 0)).as_dict(),
    }


if __name__ == "__main__":
    print(json.dumps(run_example(), indent=2, ensure_ascii=False, allow_nan=False))

