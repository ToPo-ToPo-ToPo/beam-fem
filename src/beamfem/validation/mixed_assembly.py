"""Dedicated truss + frame + shell assembly verification case."""

from __future__ import annotations

import numpy as np

from ..forces import recover_forces
from ..material import Material, Section
from ..model import Model, UX, UY, UZ, RX, RY, RZ
from ..shell import recover_shell_forces
from ..solver import solve_static


def run_mixed_assembly_case(tolerance: float = 1e-9) -> dict[str, object]:
    """Verify three formulations in one global solve against closed forms."""
    material = Material(E=200e9, nu=0.3, rho=7850.0)
    frame_section = Section.rectangle(0.1, 0.1)
    truss_section = Section(A=4e-4, Iy=1e-8, Iz=1e-8, J=2e-8)
    model = Model()

    f0, f1 = model.add_node(0, 0), model.add_node(2, 0)
    t0, t1 = model.add_node(0, 1), model.add_node(1.5, 1)
    s0, s1, s2 = model.add_node(0, 2), model.add_node(1, 2), model.add_node(0, 3)
    model.add_element(f0, f1, material, frame_section)
    model.add_truss(t0, t1, material, truss_section)
    model.add_shell(s0, s1, s2, material, 0.02)

    model.fix(f0)
    model.fix(f1, [UY, UZ, RX, RY, RZ])
    model.fix(t0)
    model.fix(t1, [UY, UZ, RX, RY, RZ])
    frame_load, truss_load = 10_000.0, 8_000.0
    model.add_load(f1, UX, frame_load)
    model.add_load(t1, UX, truss_load)

    strain = 2.5e-4
    for node in (s0, s1, s2):
        x, y, _ = model.nodes[node]
        prescribed = (strain * x, -material.nu * strain * (y - 2.0), 0, 0, 0, 0)
        for dof, value in enumerate(prescribed):
            model.constraints[(node, dof)] = float(value)

    static = solve_static(model)
    line_forces = recover_forces(model, static)
    shell = recover_shell_forces(model, static).shells[0]
    expected_frame_u = frame_load * 2.0 / (material.E * frame_section.A)
    expected_truss_u = truss_load * 1.5 / (material.E * truss_section.A)
    metrics = {
        "frame_displacement_relative_error": abs(
            static.node_disp(f1)[UX] / expected_frame_u - 1.0
        ),
        "truss_displacement_relative_error": abs(
            static.node_disp(t1)[UX] / expected_truss_u - 1.0
        ),
        "frame_axial_force_relative_error": abs(
            line_forces[0].ends("N")[0] / frame_load - 1.0
        ),
        "truss_axial_force_relative_error": abs(
            line_forces[1].ends("N")[0] / truss_load - 1.0
        ),
        "shell_sigma_x_relative_error": abs(shell.get("sx") / (material.E * strain) - 1.0),
        "shell_sigma_y_normalized": abs(shell.get("sy")) / (material.E * strain),
        "shell_shear_normalized": abs(shell.get("sxy")) / (material.E * strain),
    }
    return {
        "case_id": "mixed-truss-frame-shell-closed-form-v1",
        "formulations": ["axial_truss", "timoshenko_frame", "cst_dkt_shell"],
        "metrics": metrics,
        "tolerance": tolerance,
        "passed": bool(max(metrics.values()) <= tolerance),
    }
