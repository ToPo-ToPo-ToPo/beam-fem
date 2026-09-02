"""離散設計を共通FEMで解析し、目的関数と全制約を評価する。"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
import math
from typing import Mapping, TYPE_CHECKING

import numpy as np

from ..forces import ElementForces, ForceResults, recover_forces
from ..model import Element, Model
from ..solver import StaticResult, solve_static
from .constraints import ConstraintRecord, FAILED_UTILIZATION
from .objectives import MassObjective
from .problem import DesignState, DiscreteStructuralProblem

if TYPE_CHECKING:
    from .catalogs import SectionOption


@dataclass(frozen=True)
class CombinationAnalysis:
    """1荷重組合せのFEM結果と元部材index対応。"""

    name: str
    model: Model
    static: StaticResult
    forces: ForceResults
    original_to_active: Mapping[int, int]

    def member_force(self, original_member: int) -> ElementForces | None:
        active = self.original_to_active.get(original_member)
        return None if active is None else self.forces[active]


@dataclass(frozen=True)
class EvaluationResult:
    design: DesignState
    objective: float
    mass: float
    feasible: bool
    constraints: tuple[ConstraintRecord, ...]
    analyses: Mapping[str, CombinationAnalysis]
    diagnostic: str | None = None
    cache_hit: bool = False

    @property
    def governing_constraint(self) -> ConstraintRecord | None:
        valid = [r for r in self.constraints if not math.isnan(r.utilization)]
        return max(valid, key=lambda r: r.utilization, default=None)

    def as_dict(self) -> dict[str, object]:
        governing = self.governing_constraint
        return {
            "design": list(self.design.choices),
            "objective": self.objective,
            "mass": self.mass,
            "feasible": self.feasible,
            "diagnostic": self.diagnostic,
            "governing_constraint": None if governing is None else governing.as_dict(),
            "constraints": [r.as_dict() for r in self.constraints],
            "load_combinations": list(self.analyses),
        }


@dataclass(frozen=True)
class EvaluationContext:
    problem: DiscreteStructuralProblem
    design: DesignState
    analyses: Mapping[str, CombinationAnalysis]

    def option(self, member: int) -> "SectionOption":
        return self.problem.catalogs[member][self.design.choices[member]]

    def selected_analyses(self, names: tuple[str, ...] | None):
        # FEM失敗時は空。設計だけで判定できる制約の報告を継続するため、解析依存
        # 制約には空iteratorを返す。
        if not self.analyses:
            return iter(())
        if names is None:
            return self.analyses.items()
        missing = set(names) - set(self.analyses)
        if missing:
            raise ValueError(f"制約が未知の荷重組合せを参照: {sorted(missing)}")
        return ((name, self.analyses[name]) for name in names)


class StructuralEvaluator:
    """設計hashで解析結果をキャッシュするFEM評価器。"""

    def __init__(self, problem: DiscreteStructuralProblem):
        self.problem = problem
        self._cache: dict[DesignState, EvaluationResult] = {}
        self.n_analysis = 0
        self.n_cache_hits = 0

    def clear_cache(self) -> None:
        self._cache.clear()
        self.n_analysis = 0
        self.n_cache_hits = 0

    @property
    def cache_info(self) -> dict[str, int]:
        return {"size": len(self._cache), "analyses": self.n_analysis, "hits": self.n_cache_hits}

    def _model_for_design(self, design: DesignState) -> tuple[Model, dict[int, int]]:
        model = copy.deepcopy(self.problem.model)
        elements: list[Element] = []
        mapping: dict[int, int] = {}
        for original, (element, choice) in enumerate(zip(self.problem.model.elements, design.choices)):
            option = self.problem.catalogs[original][choice]
            if not option.active:
                continue
            mapping[original] = len(elements)
            elements.append(replace(
                copy.deepcopy(element),
                sec=option.section,
                mat=option.material or element.mat,
            ))
        model.elements = elements
        return model, mapping

    def _combined_loads(self, combination, design: DesignState) -> dict[tuple[int, int], float]:
        cases = {c.name: c for c in self.problem.load_cases}
        loads: dict[tuple[int, int], float] = {}
        for case_name, factor in combination.factors.items():
            for key, value in cases[case_name].loads.items():
                loads[key] = loads.get(key, 0.0) + factor * value
        if self.problem.self_weight is not None:
            for member, choice in enumerate(design.choices):
                option = self.problem.catalogs[member][choice]
                if not option.active:
                    continue
                element = self.problem.model.elements[member]
                material = option.material or element.mat
                length = self.problem.model.element_length(element)
                mass = material.rho * option.section.A * length
                for node in (element.n1, element.n2):
                    for dof, acceleration in enumerate(self.problem.self_weight):
                        key = (node, dof)
                        loads[key] = loads.get(key, 0.0) + 0.5 * mass * float(acceleration)
        return loads

    def _failure(self, design: DesignState, objective: float, mass: float, message: str) -> EvaluationResult:
        records = [ConstraintRecord(
            constraint_id="fem_stability",
            kind="mechanism_or_singular_stiffness",
            satisfied=False,
            utilization=FAILED_UTILIZATION,
            message=message,
        )]
        # FEMを必要としないトポロジー・製作制約は、機構設計でも併せて報告する。
        context = EvaluationContext(self.problem, design, {})
        for constraint in self.problem.constraints:
            records.extend(constraint.evaluate(context))
        return EvaluationResult(design, objective, mass, False, tuple(records), {}, diagnostic=message)

    def evaluate(self, design: DesignState, *, use_cache: bool = True) -> EvaluationResult:
        self.problem.validate_design(design)
        if use_cache and design in self._cache:
            self.n_cache_hits += 1
            return replace(self._cache[design], cache_hit=True)

        mass = MassObjective().evaluate(self.problem, design)
        objective = self.problem.objective.evaluate(self.problem, design)
        analyses: dict[str, CombinationAnalysis] = {}
        try:
            base_model, mapping = self._model_for_design(design)
            if not base_model.elements and not base_model.shells and not base_model.quad_shells:
                raise RuntimeError("active structural elements are empty")
            for combination in self.problem.load_combinations:
                model = copy.deepcopy(base_model)
                model.nodal_loads = self._combined_loads(combination, design)
                static = solve_static(model)
                if not np.all(np.isfinite(static.u)) or not np.all(np.isfinite(static.reactions)):
                    raise RuntimeError("FEM returned non-finite displacement or reaction")
                forces = recover_forces(model, static)
                analyses[combination.name] = CombinationAnalysis(
                    combination.name, model, static, forces, dict(mapping)
                )
                self.n_analysis += 1
        except (RuntimeError, ValueError, ArithmeticError) as exc:
            result = self._failure(design, objective, mass, f"FEM analysis failed: {exc}")
            if use_cache:
                self._cache[design] = result
            return result

        context = EvaluationContext(self.problem, design, analyses)
        records: list[ConstraintRecord] = []
        for constraint in self.problem.constraints:
            records.extend(constraint.evaluate(context))
        feasible = all(record.satisfied for record in records)
        result = EvaluationResult(design, objective, mass, feasible, tuple(records), analyses)
        if use_cache:
            self._cache[design] = result
        return result
