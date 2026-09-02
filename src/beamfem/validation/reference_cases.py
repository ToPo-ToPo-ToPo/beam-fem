"""独立した手計算fixtureを実行し、機械可読V&V evidenceを生成する。"""

from __future__ import annotations

import json
import math
from dataclasses import asdict
import hashlib
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..forces import recover_forces
from ..material import Material, Section
from ..model import Model, UX, UY, UZ, RX, RY, RZ
from ..solver import solve_static
from .audit import build_audit_metadata


DOFS = {"UX": UX, "UY": UY, "UZ": UZ, "RX": RX, "RY": RY, "RZ": RZ}
DEFAULT_TOLERANCES = {
    "reference_relative_error_max": 0.005,
    "equilibrium_residual_max": 1.0e-8,
    "rotation_invariance_relative_error_max": 1.0e-10,
}


def _section(data: dict[str, Any]) -> Section:
    if data["type"] == "rectangle":
        return Section.rectangle(float(data["b"]), float(data["h"]))
    inertia = float(data["I"])
    return Section(
        A=float(data["A"]), Iy=inertia, Iz=inertia, J=2.0 * inertia,
        cy=0.0, cz=0.0,
    )


def _build(case: dict[str, Any], rotation_degrees: float = 0.0) -> Model:
    material_data = case["material"]
    material = Material(
        E=float(material_data["E"]), nu=float(material_data["nu"]),
        rho=float(material_data["density"]),
    )
    sections = {name: _section(data) for name, data in case["sections"].items()}
    angle = math.radians(rotation_degrees)
    rotation = np.array([
        [math.cos(angle), -math.sin(angle), 0.0],
        [math.sin(angle), math.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    model = Model()
    for coordinates in case["nodes"]:
        xyz = rotation @ np.asarray(coordinates, dtype=float)
        model.add_node(*xyz)
    for member in case["members"]:
        add = model.add_truss if member["type"] == "truss" else model.add_element
        add(member["nodes"][0], member["nodes"][1], material, sections[member["section"]])
    for support in case["supports"]:
        model.fix(int(support["node"]), [DOFS[name] for name in support["dofs"]])
    for load in case["loads"]:
        force = rotation @ np.asarray(load["force"], dtype=float)
        for dof, value in enumerate(force):
            if value:
                model.add_load(int(load["node"]), dof, float(value))
    return model


def _relative_error(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1.0e-15))


def _equilibrium_residual(model: Model, reactions: np.ndarray) -> float:
    external = np.zeros(3)
    for (node, dof), value in model.nodal_loads.items():
        if dof < 3:
            external[dof] += value
    reaction = np.zeros(3)
    for (node, dof) in model.constraints:
        if dof < 3:
            reaction[dof] += reactions[node * 6 + dof]
    return float(np.linalg.norm(external + reaction) / max(np.linalg.norm(external), 1.0))


def run_reference_case(case: dict[str, Any], tolerances: dict[str, float] | None = None) -> dict[str, Any]:
    """1 fixtureを解析し、閉形式期待値との差を返す。"""

    limits = dict(DEFAULT_TOLERANCES if tolerances is None else tolerances)
    model = _build(case)
    result = solve_static(model)
    forces = recover_forces(model, result)
    expected = case["expected"]
    node = int(expected["node"])
    actual_disp = result.node_disp(node)[:3]
    expected_disp = np.asarray(expected["displacement"], dtype=float)
    actual_forces = np.asarray([member.ends("N")[0] for member in forces.elements])
    expected_forces = np.asarray(expected["axial_forces"], dtype=float)
    displacement_error = _relative_error(actual_disp, expected_disp)
    force_error = _relative_error(actual_forces, expected_forces)
    equilibrium = _equilibrium_residual(model, result.reactions)

    rotation_error = None
    if "rotation_check_degrees" in case:
        angle = float(case["rotation_check_degrees"])
        rotated_model = _build(case, angle)
        rotated_result = solve_static(rotated_model)
        radians = math.radians(angle)
        rotation = np.array([
            [math.cos(radians), -math.sin(radians), 0.0],
            [math.sin(radians), math.cos(radians), 0.0],
            [0.0, 0.0, 1.0],
        ])
        rotation_error = _relative_error(
            rotated_result.node_disp(node)[:3], rotation @ actual_disp
        )

    passed = (
        displacement_error <= limits["reference_relative_error_max"]
        and force_error <= limits["reference_relative_error_max"]
        and equilibrium <= limits["equilibrium_residual_max"]
        and (rotation_error is None or rotation_error <= limits["rotation_invariance_relative_error_max"])
    )
    return {
        "case_id": case["case_id"],
        "description": case["description"],
        "expected_formula": expected["formula"],
        "actual": {
            "displacement": actual_disp.tolist(),
            "axial_forces": actual_forces.tolist(),
        },
        "expected": {
            "displacement": expected_disp.tolist(),
            "axial_forces": expected_forces.tolist(),
        },
        "metrics": {
            "displacement_relative_error": displacement_error,
            "axial_force_relative_error": force_error,
            "equilibrium_residual": equilibrium,
            "rotation_invariance_relative_error": rotation_error,
        },
        "passed": passed,
    }


def run_reference_suite(
    fixture_paths: Iterable[str | Path],
    *,
    tolerances: dict[str, float] | None = None,
) -> dict[str, Any]:
    """複数fixtureを実行する。順序は入力順で決定的。"""

    limits = dict(DEFAULT_TOLERANCES if tolerances is None else tolerances)
    paths = [Path(path).resolve() for path in fixture_paths]
    cases = []
    fixture_checksums: dict[str, str] = {}
    for path in paths:
        raw = path.read_bytes()
        fixture_checksums[path.name] = hashlib.sha256(raw).hexdigest()
        with path.open(encoding="utf-8") as stream:
            case = json.load(stream)
        cases.append(run_reference_case(case, limits))
    repository = paths[0].parents[2] if paths and len(paths[0].parents) > 2 else None
    audit = build_audit_metadata(
        solver="closed-form-reference-vv",
        solver_settings={"fixture_count": len(paths), "units": "SI"},
        repository=repository,
    )
    return {
        "evidence_schema_version": "1.0",
        "method": "closed_form_reference_cases",
        "units": "SI",
        "tolerances": limits,
        "audit": asdict(audit),
        "fixture_sha256": fixture_checksums,
        "cases": cases,
        "passed": all(case["passed"] for case in cases),
    }


def write_reference_evidence(evidence: dict[str, Any], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination
