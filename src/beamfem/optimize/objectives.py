"""離散構造最適化の目的関数。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from .problem import DiscreteStructuralProblem, DesignState


class StructuralObjective(Protocol):
    name: str

    def evaluate(self, problem: "DiscreteStructuralProblem", design: "DesignState") -> float:
        ...


@dataclass(frozen=True)
class MassObjective:
    """総部材質量 [kg] を最小化する。"""

    name: str = "mass"

    def evaluate(self, problem: "DiscreteStructuralProblem", design: "DesignState") -> float:
        mass = 0.0
        for i, choice in enumerate(design.choices):
            option = problem.catalogs[i][choice]
            if not option.active:
                continue
            material = option.material or problem.model.elements[i].mat
            length = problem.model.element_length(problem.model.elements[i])
            mass += material.rho * option.section.A * length
        return float(mass)


@dataclass(frozen=True)
class WeightedImpactObjective:
    """質量・コスト・embodied carbon の重み付き目的関数。"""

    mass_weight: float = 1.0
    cost_weight: float = 0.0
    carbon_weight: float = 0.0
    name: str = "weighted_impact"

    def evaluate(self, problem: "DiscreteStructuralProblem", design: "DesignState") -> float:
        total = 0.0
        for i, choice in enumerate(design.choices):
            option = problem.catalogs[i][choice]
            if not option.active:
                continue
            material = option.material or problem.model.elements[i].mat
            length = problem.model.element_length(problem.model.elements[i])
            mass = material.rho * option.section.A * length
            total += mass * (
                self.mass_weight
                + self.cost_weight * option.cost_per_kg
                + self.carbon_weight * option.carbon_per_kg
            )
        return float(total)
