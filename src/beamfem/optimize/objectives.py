"""離散構造最適化の目的関数。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from .problem import DiscreteStructuralProblem, DesignState


def impact_components(problem: "DiscreteStructuralProblem",
                      design: "DesignState") -> dict[str, float]:
    """Return unweighted mass, cost, and embodied-carbon totals."""
    totals = {"mass": 0.0, "cost": 0.0, "carbon": 0.0}
    for i, choice in enumerate(design.choices):
        option = problem.catalogs[i][choice]
        if not option.active:
            continue
        material = option.material or problem.model.elements[i].mat
        length = problem.model.element_length(problem.model.elements[i])
        mass = material.rho * option.section.A * length
        totals["mass"] += mass
        totals["cost"] += mass * option.cost_per_kg
        totals["carbon"] += mass * option.carbon_per_kg
    return {key: float(value) for key, value in totals.items()}


class StructuralObjective(Protocol):
    name: str

    def evaluate(self, problem: "DiscreteStructuralProblem", design: "DesignState") -> float:
        ...


@dataclass(frozen=True)
class MassObjective:
    """総部材質量 [kg] を最小化する。"""

    name: str = "mass"

    def evaluate(self, problem: "DiscreteStructuralProblem", design: "DesignState") -> float:
        return impact_components(problem, design)["mass"]


@dataclass(frozen=True)
class WeightedImpactObjective:
    """質量・コスト・embodied carbon の重み付き目的関数。"""

    mass_weight: float = 1.0
    cost_weight: float = 0.0
    carbon_weight: float = 0.0
    name: str = "weighted_impact"

    def evaluate(self, problem: "DiscreteStructuralProblem", design: "DesignState") -> float:
        values = impact_components(problem, design)
        return float(
            self.mass_weight * values["mass"]
            + self.cost_weight * values["cost"]
            + self.carbon_weight * values["carbon"]
        )
