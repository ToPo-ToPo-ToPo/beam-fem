"""離散構造問題の制約と機械可読な判定結果。"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Iterable, Mapping, Protocol, TYPE_CHECKING

from ..model import DOF_PER_NODE

FAILED_UTILIZATION = 1.0e30  # JSON-safe sentinel for a non-numeric/absolute failure

if TYPE_CHECKING:
    from .evaluation import EvaluationContext


@dataclass(frozen=True)
class ConstraintRecord:
    """1制約・1支配点の判定。utilization <= 1 が合格。"""

    constraint_id: str
    kind: str
    satisfied: bool
    utilization: float
    value: float | None = None
    limit: float | None = None
    margin: float | None = None
    load_combination: str | None = None
    member: int | None = None
    node: int | None = None
    dof: int | None = None
    message: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "constraint_id": self.constraint_id,
            "kind": self.kind,
            "satisfied": self.satisfied,
            "utilization": self.utilization,
            "value": self.value,
            "limit": self.limit,
            "margin": self.margin,
            "load_combination": self.load_combination,
            "member": self.member,
            "node": self.node,
            "dof": self.dof,
            "message": self.message,
            "metadata": dict(self.metadata),
        }


class StructuralConstraint(Protocol):
    constraint_id: str

    def evaluate(self, context: "EvaluationContext") -> Iterable[ConstraintRecord]:
        ...


def _record(
    constraint_id: str,
    kind: str,
    value: float,
    limit: float,
    **kwargs,
) -> ConstraintRecord:
    utilization = value / limit if limit > 0.0 else FAILED_UTILIZATION
    return ConstraintRecord(
        constraint_id=constraint_id,
        kind=kind,
        satisfied=bool(math.isfinite(utilization) and utilization <= 1.0 + 1e-12),
        utilization=float(utilization),
        value=float(value),
        limit=float(limit),
        margin=float(limit - value),
        **kwargs,
    )


@dataclass(frozen=True)
class RequiredMembers:
    members: tuple[int, ...]
    constraint_id: str = "required_members"

    def __init__(self, members: Iterable[int], constraint_id: str = "required_members"):
        object.__setattr__(self, "members", tuple(int(i) for i in members))
        object.__setattr__(self, "constraint_id", constraint_id)

    def evaluate(self, context: "EvaluationContext") -> Iterable[ConstraintRecord]:
        for i in self.members:
            active = context.option(i).active
            yield ConstraintRecord(self.constraint_id, "required_member", active, 0.0 if active else 2.0,
                                   member=i, message="required member is active" if active else "required member is OFF")


@dataclass(frozen=True)
class ForbiddenMembers:
    members: tuple[int, ...]
    constraint_id: str = "forbidden_members"

    def __init__(self, members: Iterable[int], constraint_id: str = "forbidden_members"):
        object.__setattr__(self, "members", tuple(int(i) for i in members))
        object.__setattr__(self, "constraint_id", constraint_id)

    def evaluate(self, context: "EvaluationContext") -> Iterable[ConstraintRecord]:
        for i in self.members:
            inactive = not context.option(i).active
            yield ConstraintRecord(self.constraint_id, "forbidden_member", inactive, 0.0 if inactive else 2.0,
                                   member=i, message="forbidden member is OFF" if inactive else "forbidden member is active")


@dataclass(frozen=True)
class SameSectionGroup:
    members: tuple[int, ...]
    constraint_id: str = "same_section_group"

    def __init__(self, members: Iterable[int], constraint_id: str = "same_section_group"):
        values = tuple(int(i) for i in members)
        if len(values) < 2:
            raise ValueError("同一断面グループには2部材以上必要です")
        object.__setattr__(self, "members", values)
        object.__setattr__(self, "constraint_id", constraint_id)

    def evaluate(self, context: "EvaluationContext") -> Iterable[ConstraintRecord]:
        names = [context.option(i).name for i in self.members]
        ok = len(set(names)) == 1
        yield ConstraintRecord(self.constraint_id, "same_section_group", ok, 0.0 if ok else 2.0,
                               message="same section" if ok else "section choices differ",
                               metadata={"members": self.members, "section_names": names})


@dataclass(frozen=True)
class MaxSectionTypes:
    maximum: int
    members: tuple[int, ...] | None = None
    include_off: bool = False
    constraint_id: str = "max_section_types"

    def __post_init__(self) -> None:
        if self.maximum < 1:
            raise ValueError("最大断面種類数は1以上でなければなりません")

    def evaluate(self, context: "EvaluationContext") -> Iterable[ConstraintRecord]:
        ids = self.members if self.members is not None else tuple(range(context.problem.n_members))
        names = {context.option(i).name for i in ids if self.include_off or context.option(i).active}
        yield _record(self.constraint_id, "max_section_types", len(names), self.maximum,
                      message=f"{len(names)} section types used", metadata={"section_names": sorted(names)})


@dataclass(frozen=True)
class ActiveMemberCount:
    minimum: int = 0
    maximum: int | None = None
    members: tuple[int, ...] | None = None
    constraint_id: str = "active_member_count"

    def __post_init__(self) -> None:
        if self.minimum < 0 or (self.maximum is not None and self.maximum < self.minimum):
            raise ValueError("部材本数の上下限が不正です")

    def evaluate(self, context: "EvaluationContext") -> Iterable[ConstraintRecord]:
        ids = self.members if self.members is not None else tuple(range(context.problem.n_members))
        count = sum(context.option(i).active for i in ids)
        ok = count >= self.minimum and (self.maximum is None or count <= self.maximum)
        if count < self.minimum:
            utilization = 1.0 + (self.minimum - count) / max(self.minimum, 1)
            margin = count - self.minimum
        elif self.maximum is not None:
            utilization = count / self.maximum if self.maximum else (0.0 if count == 0 else FAILED_UTILIZATION)
            margin = self.maximum - count
        else:
            utilization, margin = 0.0, None
        yield ConstraintRecord(self.constraint_id, "active_member_count", ok, float(utilization),
                               value=float(count), limit=None if self.maximum is None else float(self.maximum),
                               margin=None if margin is None else float(margin),
                               message=f"{count} active members",
                               metadata={"minimum": self.minimum, "maximum": self.maximum})


@dataclass(frozen=True)
class SymmetryPairs:
    """対称位置にある部材ペアへ同じ名称の断面候補を要求する。"""

    pairs: tuple[tuple[int, int], ...]
    constraint_id: str = "symmetry_pairs"

    def __init__(self, pairs: Iterable[tuple[int, int]], constraint_id: str = "symmetry_pairs"):
        object.__setattr__(self, "pairs", tuple((int(a), int(b)) for a, b in pairs))
        object.__setattr__(self, "constraint_id", constraint_id)

    def evaluate(self, context: "EvaluationContext") -> Iterable[ConstraintRecord]:
        for a, b in self.pairs:
            name_a, name_b = context.option(a).name, context.option(b).name
            ok = name_a == name_b
            yield ConstraintRecord(self.constraint_id, "symmetry_pair", ok, 0.0 if ok else 2.0,
                                   message="symmetric choices match" if ok else "symmetric choices differ",
                                   metadata={"member_a": a, "member_b": b,
                                             "section_a": name_a, "section_b": name_b})


@dataclass(frozen=True)
class Connectivity:
    """active部材グラフで指定節点が相互に連結していることを要求する。"""

    required_nodes: tuple[int, ...]
    constraint_id: str = "connectivity"

    def __init__(self, required_nodes: Iterable[int], constraint_id: str = "connectivity"):
        nodes = tuple(dict.fromkeys(int(i) for i in required_nodes))
        if len(nodes) < 2:
            raise ValueError("連結性制約には2節点以上必要です")
        object.__setattr__(self, "required_nodes", nodes)
        object.__setattr__(self, "constraint_id", constraint_id)

    def evaluate(self, context: "EvaluationContext") -> Iterable[ConstraintRecord]:
        graph: dict[int, set[int]] = {}
        for i, element in enumerate(context.problem.model.elements):
            if context.option(i).active:
                graph.setdefault(element.n1, set()).add(element.n2)
                graph.setdefault(element.n2, set()).add(element.n1)
        visited = {self.required_nodes[0]}
        stack = list(visited)
        while stack:
            stack.extend(graph.get(stack.pop(), set()) - visited)
            visited.update(stack)
        missing = sorted(set(self.required_nodes) - visited)
        ok = not missing
        disconnected_fraction = len(missing) / len(self.required_nodes)
        yield ConstraintRecord(self.constraint_id, "connectivity", ok,
                               0.0 if ok else 1.0 + disconnected_fraction,
                               message="required nodes connected" if ok else "required nodes disconnected",
                               metadata={"required_nodes": self.required_nodes, "disconnected_nodes": missing})


@dataclass(frozen=True)
class MemberLengthRange:
    minimum: float = 0.0
    maximum: float = math.inf
    members: tuple[int, ...] | None = None
    constraint_id: str = "member_length_range"

    def __post_init__(self) -> None:
        if self.minimum < 0.0 or self.maximum <= 0.0 or self.maximum < self.minimum:
            raise ValueError("部材長の上下限が不正です")

    def evaluate(self, context: "EvaluationContext") -> Iterable[ConstraintRecord]:
        ids = self.members if self.members is not None else tuple(range(context.problem.n_members))
        for member in ids:
            if not context.option(member).active:
                continue
            length = context.problem.model.element_length(context.problem.model.elements[member])
            ok = self.minimum - 1e-12 <= length <= self.maximum + 1e-12
            violation = max(self.minimum - length, length - self.maximum, 0.0)
            if length < self.minimum:
                utilization = self.minimum / length if length > 0.0 else FAILED_UTILIZATION
            elif length > self.maximum:
                utilization = length / self.maximum
            else:
                utilization = 0.0
            yield ConstraintRecord(self.constraint_id, "member_length_range", ok,
                                   utilization, value=length,
                                   message="active member length within range" if ok else "active member length outside range",
                                   member=member, metadata={"minimum": self.minimum,
                                                            "maximum": None if math.isinf(self.maximum) else self.maximum,
                                                            "violation": violation})


@dataclass(frozen=True)
class DisplacementLimit:
    node: int
    dof: int
    maximum: float
    combinations: tuple[str, ...] | None = None
    constraint_id: str = "displacement"

    def __post_init__(self) -> None:
        if self.maximum <= 0.0:
            raise ValueError("変位上限は正値でなければなりません")
        if not 0 <= self.dof < DOF_PER_NODE:
            raise ValueError("自由度indexが範囲外です")

    def evaluate(self, context: "EvaluationContext") -> Iterable[ConstraintRecord]:
        for name, analysis in context.selected_analyses(self.combinations):
            value = abs(float(analysis.static.node_disp(self.node)[self.dof]))
            yield _record(self.constraint_id, "displacement", value, self.maximum,
                          load_combination=name, node=self.node, dof=self.dof,
                          message="absolute nodal displacement")


@dataclass(frozen=True)
class RelativeDisplacementLimit:
    node_a: int
    node_b: int
    dof: int
    maximum: float
    combinations: tuple[str, ...] | None = None
    constraint_id: str = "relative_displacement"

    def __post_init__(self) -> None:
        if self.maximum <= 0.0:
            raise ValueError("相対変位上限は正値でなければなりません")
        if not 0 <= self.dof < DOF_PER_NODE:
            raise ValueError("自由度indexが範囲外です")

    def evaluate(self, context: "EvaluationContext") -> Iterable[ConstraintRecord]:
        for name, analysis in context.selected_analyses(self.combinations):
            ua = analysis.static.node_disp(self.node_a)[self.dof]
            ub = analysis.static.node_disp(self.node_b)[self.dof]
            value = abs(float(ua - ub))
            yield _record(self.constraint_id, "relative_displacement", value, self.maximum,
                          load_combination=name, dof=self.dof,
                          message="absolute relative nodal displacement",
                          metadata={"node_a": self.node_a, "node_b": self.node_b})


@dataclass(frozen=True)
class StressLimit:
    """引張・圧縮の合成縁端応力制約 [Pa]。

    候補断面に強度が設定されていればそれを優先し、未設定時はここで指定した
    default値を用いる。曲げは両側縁の厳しい側を保守的に評価する。
    """

    tension: float | None = None
    compression: float | None = None
    members: tuple[int, ...] | None = None
    combinations: tuple[str, ...] | None = None
    constraint_id: str = "stress"

    def __post_init__(self) -> None:
        if self.tension is not None and self.tension <= 0.0:
            raise ValueError("引張強度は正値でなければなりません")
        if self.compression is not None and self.compression <= 0.0:
            raise ValueError("圧縮強度は正値でなければなりません")
        if self.tension is None and self.compression is None:
            # 候補断面固有の強度だけを使う場合は両方Noneを許容する。
            return

    def evaluate(self, context: "EvaluationContext") -> Iterable[ConstraintRecord]:
        ids = self.members if self.members is not None else tuple(range(context.problem.n_members))
        for combo, analysis in context.selected_analyses(self.combinations):
            for member in ids:
                ef = analysis.member_force(member)
                if ef is None:
                    continue
                option = context.option(member)
                axial = ef.stress_ends("sigma_a")
                bending = ef.stress_ends("sigma_b")
                tensile = max(max(n + b, 0.0) for n, b in zip(axial, bending))
                compressive = max(max(-n + b, 0.0) for n, b in zip(axial, bending))
                tlim = option.tensile_strength or self.tension
                clim = option.compressive_strength or self.compression
                if tlim is not None:
                    yield _record(self.constraint_id, "tensile_stress", tensile, tlim,
                                  load_combination=combo, member=member, message="maximum tensile edge stress")
                if clim is not None:
                    yield _record(self.constraint_id, "compressive_stress", compressive, clim,
                                  load_combination=combo, member=member, message="maximum compressive edge stress")


@dataclass(frozen=True)
class EulerBucklingLimit:
    """部材圧縮力に対するEuler座屈制約。"""

    effective_length_factor: float = 1.0
    axis: str = "min"
    members: tuple[int, ...] | None = None
    combinations: tuple[str, ...] | None = None
    constraint_id: str = "euler_buckling"

    def __post_init__(self) -> None:
        if self.effective_length_factor <= 0.0:
            raise ValueError("有効座屈長係数は正値でなければなりません")
        if self.axis not in ("y", "z", "min"):
            raise ValueError("axis は 'y', 'z', 'min' のいずれかです")

    def evaluate(self, context: "EvaluationContext") -> Iterable[ConstraintRecord]:
        ids = self.members if self.members is not None else tuple(range(context.problem.n_members))
        for combo, analysis in context.selected_analyses(self.combinations):
            for member in ids:
                ef = analysis.member_force(member)
                if ef is None:
                    continue
                option = context.option(member)
                element = context.problem.model.elements[member]
                material = option.material or element.mat
                sec = option.section
                inertia = sec.Iy if self.axis == "y" else sec.Iz if self.axis == "z" else min(sec.Iy, sec.Iz)
                length = context.problem.model.element_length(element)
                capacity = math.pi**2 * material.E * inertia / (self.effective_length_factor * length) ** 2
                demand = max(0.0, -min(ef.ends("N")))
                yield _record(self.constraint_id, "euler_buckling", demand, capacity,
                              load_combination=combo, member=member,
                              message="compressive force / Euler critical force",
                              metadata={"effective_length_factor": self.effective_length_factor, "axis": self.axis})
