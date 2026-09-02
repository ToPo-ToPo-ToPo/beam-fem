"""共通の離散構造問題と荷重組合せモデル。"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Iterable, Mapping, TYPE_CHECKING

import numpy as np

from ..model import DOF_PER_NODE, Model
from .catalogs import SectionCatalog
from .objectives import MassObjective, StructuralObjective

if TYPE_CHECKING:
    from .constraints import StructuralConstraint
    from .evaluation import EvaluationResult, StructuralEvaluator


@dataclass(frozen=True)
class DesignState:
    """各候補部材のカタログindexを持つ、hash可能な設計状態。"""

    choices: tuple[int, ...]

    def __init__(self, choices: Iterable[int]):
        values = tuple(int(v) for v in choices)
        object.__setattr__(self, "choices", values)

    def __len__(self) -> int:
        return len(self.choices)

    def changed(self, member: int, choice: int) -> "DesignState":
        values = list(self.choices)
        values[member] = int(choice)
        return DesignState(values)


@dataclass(frozen=True)
class LoadCase:
    """節点荷重ケース。キーは ``(node, dof)``、値はNまたはN m。"""

    name: str
    loads: Mapping[tuple[int, int], float]

    def __init__(self, name: str, loads: Mapping[tuple[int, int], float]):
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "loads", MappingProxyType(dict(loads)))


@dataclass(frozen=True)
class LoadCombination:
    """荷重ケース名から組合せ係数への写像。"""

    name: str
    factors: Mapping[str, float]

    def __init__(self, name: str, factors: Mapping[str, float]):
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "factors", MappingProxyType(dict(factors)))


@dataclass
class DiscreteStructuralProblem:
    """FEM評価器と全最適化backendが共有する問題定義。

    ``catalogs[i]`` は必ず ``model.elements[i]`` に対応する。モデル・荷重・断面は
    SI一貫単位（m, N, Pa, kg）を前提とする。現時点の ``Model.elements`` は
    Timoshenko梁なので、本評価器も曲げ・せん断を含む骨組解析である。軸力だけの
    truss要素へ暗黙に置換はしない。
    """

    model: Model
    catalogs: tuple[SectionCatalog, ...] | list[SectionCatalog]
    load_cases: tuple[LoadCase, ...] | list[LoadCase] = field(default_factory=tuple)
    load_combinations: tuple[LoadCombination, ...] | list[LoadCombination] = field(default_factory=tuple)
    constraints: tuple["StructuralConstraint", ...] | list["StructuralConstraint"] = field(default_factory=tuple)
    objective: StructuralObjective = field(default_factory=MassObjective)
    initial_design: DesignState | None = None
    self_weight: np.ndarray | tuple[float, float, float] | None = None
    _evaluator: "StructuralEvaluator | None" = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.catalogs = tuple(self.catalogs)
        self.load_cases = tuple(self.load_cases)
        self.load_combinations = tuple(self.load_combinations)
        self.constraints = tuple(self.constraints)
        if self.self_weight is not None:
            gravity = np.asarray(self.self_weight, dtype=float).copy()
            if gravity.shape != (3,) or not np.all(np.isfinite(gravity)):
                raise ValueError("self_weight は有限な3成分加速度ベクトル [m/s2] です")
            gravity.flags.writeable = False
            self.self_weight = gravity
        if len(self.catalogs) != len(self.model.elements):
            raise ValueError("断面カタログ数は梁要素数と一致しなければなりません")
        cases = {c.name: c for c in self.load_cases}
        if len(cases) != len(self.load_cases):
            raise ValueError("荷重ケース名は一意でなければなりません")
        if not self.load_cases:
            self.load_cases = (LoadCase("base", self.model.nodal_loads),)
            cases = {"base": self.load_cases[0]}
        if not self.load_combinations:
            self.load_combinations = tuple(LoadCombination(c.name, {c.name: 1.0}) for c in self.load_cases)
        combo_names = [c.name for c in self.load_combinations]
        if len(set(combo_names)) != len(combo_names):
            raise ValueError("荷重組合せ名は一意でなければなりません")
        for case in self.load_cases:
            for node, dof in case.loads:
                if not 0 <= node < self.model.n_nodes or not 0 <= dof < DOF_PER_NODE:
                    raise ValueError(f"荷重ケース {case.name!r} に不正な自由度 {(node, dof)} があります")
        for combo in self.load_combinations:
            missing = set(combo.factors) - set(cases)
            if missing:
                raise ValueError(f"荷重組合せ {combo.name!r} が未知のケースを参照: {sorted(missing)}")
        if self.initial_design is None:
            self.initial_design = DesignState(next((i for i, o in enumerate(c.options) if o.active), 0) for c in self.catalogs)
        self.validate_design(self.initial_design)

    @property
    def n_members(self) -> int:
        return len(self.catalogs)

    def validate_design(self, design: DesignState) -> None:
        if len(design) != self.n_members:
            raise ValueError(f"設計変数数 {len(design)} が部材数 {self.n_members} と一致しません")
        for i, choice in enumerate(design.choices):
            if not 0 <= choice < len(self.catalogs[i]):
                raise ValueError(f"部材 {i} の断面index {choice} は範囲外です")

    def design_from_names(self, names: Iterable[str]) -> DesignState:
        names = tuple(names)
        if len(names) != self.n_members:
            raise ValueError("断面名の数が部材数と一致しません")
        return DesignState(self.catalogs[i].index(name) for i, name in enumerate(names))

    def evaluate(self, design: DesignState, *, use_cache: bool = True) -> "EvaluationResult":
        if self._evaluator is None:
            from .evaluation import StructuralEvaluator

            self._evaluator = StructuralEvaluator(self)
        return self._evaluator.evaluate(design, use_cache=use_cache)

    def clear_cache(self) -> None:
        if self._evaluator is not None:
            self._evaluator.clear_cache()
