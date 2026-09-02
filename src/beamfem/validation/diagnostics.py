"""Non-fatal diagnostics for validated portable problem documents."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Mapping


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class Diagnostic:
    code: str
    severity: Severity
    message: str
    location: str | None = None


@dataclass(frozen=True)
class DiagnosticReport:
    diagnostics: tuple[Diagnostic, ...]

    @property
    def has_errors(self) -> bool:
        return any(item.severity is Severity.ERROR for item in self.diagnostics)

    @property
    def warnings(self) -> tuple[str, ...]:
        return tuple(
            item.message
            for item in self.diagnostics
            if item.severity is Severity.WARNING
        )


def diagnose_problem_spec(document: Mapping[str, Any]) -> DiagnosticReport:
    """Report suspicious geometry and under-specified supports before FEM.

    These checks are conservative and do not claim to replace a stiffness-rank
    or mechanism analysis performed by the FEM evaluator.
    """

    diagnostics: list[Diagnostic] = []
    nodes = document.get("nodes", [])
    members = document.get("members", [])
    supports = document.get("supports", [])
    coordinates = {
        node["id"]: tuple(float(x) for x in node["xyz"])
        for node in nodes
        if isinstance(node, Mapping) and "id" in node and "xyz" in node
    }

    used_nodes: set[str] = set()
    seen_pairs: dict[tuple[str, str], str] = {}
    for member in members:
        if not isinstance(member, Mapping):
            continue
        ends = member.get("nodes", [])
        if not isinstance(ends, list) or len(ends) != 2:
            continue
        a, b = str(ends[0]), str(ends[1])
        used_nodes.update((a, b))
        pair = tuple(sorted((a, b)))
        if pair in seen_pairs:
            diagnostics.append(
                Diagnostic(
                    "duplicate-member",
                    Severity.WARNING,
                    f"members {seen_pairs[pair]!r} and {member.get('id')!r} share endpoints",
                    f"members.{member.get('id')}",
                )
            )
        else:
            seen_pairs[pair] = str(member.get("id"))
        if a in coordinates and b in coordinates:
            ca, cb = coordinates[a], coordinates[b]
            dimensions = min(len(ca), len(cb))
            length = math.sqrt(sum((ca[i] - cb[i]) ** 2 for i in range(dimensions)))
            if length <= 1e-12:
                diagnostics.append(
                    Diagnostic(
                        "zero-length-member",
                        Severity.ERROR,
                        f"member {member.get('id')!r} has effectively zero length",
                        f"members.{member.get('id')}",
                    )
                )

    supported_nodes = {
        str(item.get("node")) for item in supports if isinstance(item, Mapping)
    }
    for node_id in coordinates:
        if node_id not in used_nodes and node_id not in supported_nodes:
            diagnostics.append(
                Diagnostic(
                    "isolated-node",
                    Severity.WARNING,
                    f"node {node_id!r} is neither connected nor supported",
                    f"nodes.{node_id}",
                )
            )

    constrained_dofs = sum(
        len(item.get("dofs", [])) for item in supports if isinstance(item, Mapping)
    )
    dimension = max((len(xyz) for xyz in coordinates.values()), default=2)
    rigid_body_minimum = 3 if dimension == 2 else 6
    if constrained_dofs < rigid_body_minimum:
        diagnostics.append(
            Diagnostic(
                "possibly-underconstrained",
                Severity.WARNING,
                f"only {constrained_dofs} constrained DOFs were declared; "
                "the FEM rank check must confirm stability",
                "supports",
            )
        )

    if not diagnostics:
        diagnostics.append(
            Diagnostic(
                "preflight-ok",
                Severity.INFO,
                "portable input preflight found no obvious issue",
            )
        )
    return DiagnosticReport(tuple(diagnostics))
