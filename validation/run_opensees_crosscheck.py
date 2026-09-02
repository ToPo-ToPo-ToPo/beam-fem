"""Cross-check linear and bilinear truss responses against OpenSeesPy."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess

import numpy as np
import beamfem

from beamfem import (
    BilinearIsotropicHardening,
    Material,
    Model,
    Section,
    UX,
    UY,
    UZ,
    solve_nonlinear_truss,
    solve_static,
)


ROOT = Path(__file__).resolve().parents[1]


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _beamfem_model(*, force: float = 0.0) -> tuple[Model, int]:
    model = Model()
    first = model.add_node(0.0, 0.0, 0.0)
    second = model.add_node(1.0, 0.0, 0.0)
    material = Material(200.0e9, rho=7850.0)
    section = Section(A=0.01, Iy=1.0e-6, Iz=1.0e-6, J=1.0e-6)
    model.add_truss(first, second, material, section)
    model.fix(first)
    model.fix(second, (UY, UZ))
    if force:
        model.add_load(second, UX, force)
    return model, second


def _opensees_linear(force: float) -> tuple[float, float]:
    import openseespy.opensees as ops

    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 2)
    ops.node(1, 0.0, 0.0)
    ops.node(2, 1.0, 0.0)
    ops.fix(1, 1, 1)
    ops.fix(2, 0, 1)
    ops.uniaxialMaterial("Elastic", 1, 200.0e9)
    ops.element("truss", 1, 1, 2, 0.01, 1)
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    ops.load(2, force, 0.0)
    ops.system("BandGeneral")
    ops.numberer("Plain")
    ops.constraints("Plain")
    ops.integrator("LoadControl", 1.0)
    ops.algorithm("Linear")
    ops.analysis("Static")
    if ops.analyze(1) != 0:
        raise RuntimeError("OpenSees linear truss analysis failed")
    displacement = float(ops.nodeDisp(2, 1))
    axial_force = float(ops.eleForce(1)[2])
    ops.wipe()
    return displacement, axial_force


def _opensees_bilinear(target_displacement: float) -> tuple[float, float]:
    import openseespy.opensees as ops

    elastic, yield_stress, tangent, area = 200.0e9, 250.0e6, 2.0e9, 0.01
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 2)
    ops.node(1, 0.0, 0.0)
    ops.node(2, 1.0, 0.0)
    ops.fix(1, 1, 1)
    ops.fix(2, 0, 1)
    ops.uniaxialMaterial("Steel01", 1, yield_stress, elastic, tangent / elastic)
    ops.element("truss", 1, 1, 2, area, 1)
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    ops.load(2, 1.0, 0.0)
    ops.constraints("Plain")
    ops.numberer("Plain")
    ops.system("BandGeneral")
    ops.test("NormUnbalance", 1.0e-8, 50)
    ops.algorithm("Newton")
    steps = 30
    ops.integrator("DisplacementControl", 2, 1, target_displacement / steps)
    ops.analysis("Static")
    if ops.analyze(steps) != 0:
        raise RuntimeError("OpenSees bilinear truss analysis failed")
    displacement = float(ops.nodeDisp(2, 1))
    stress = float(ops.eleForce(1)[2]) / area
    ops.wipe()
    return displacement, stress


def generate_evidence() -> dict:
    force = 1.0e6
    linear_model, second = _beamfem_model(force=force)
    linear = solve_static(linear_model)
    beamfem_linear_displacement = float(linear.node_disp(second)[UX])
    opensees_linear_displacement, opensees_linear_force = _opensees_linear(force)

    target = 0.003
    nonlinear_model, second = _beamfem_model()
    nonlinear = solve_nonlinear_truss(
        nonlinear_model,
        BilinearIsotropicHardening(200.0e9, 250.0e6, 2.0e9),
        load_factors=(0.0, 1.0),
        displacement_pattern={(second, UX): target},
        maximum_step=0.05,
    )
    beamfem_nonlinear_displacement = float(nonlinear.node_disp(second)[UX])
    beamfem_nonlinear_stress = float(nonlinear.element_states[0].stress)
    opensees_nonlinear_displacement, opensees_nonlinear_stress = _opensees_bilinear(target)

    tolerances = {"relative": 1.0e-8, "absolute": 1.0e-6}

    def close(first: float, second: float) -> bool:
        return bool(np.isclose(first, second, rtol=tolerances["relative"],
                               atol=tolerances["absolute"]))

    checks = {
        "linear_displacement": close(
            beamfem_linear_displacement, opensees_linear_displacement
        ),
        "linear_axial_force": close(force, opensees_linear_force),
        "bilinear_displacement": close(
            beamfem_nonlinear_displacement, opensees_nonlinear_displacement
        ),
        "bilinear_stress": close(
            beamfem_nonlinear_stress, opensees_nonlinear_stress
        ),
        "beamfem_nonlinear_converged": nonlinear.converged,
    }
    return {
        "evidence_schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "platform": platform.platform(), "python": platform.python_version(),
            "beamfem": beamfem.__version__, "git_commit": _git_commit(),
        },
        "external_solver": {"name": "OpenSeesPy", "version": version("openseespy")},
        "tolerances": tolerances,
        "linear": {
            "beamfem_displacement": beamfem_linear_displacement,
            "opensees_displacement": opensees_linear_displacement,
            "opensees_axial_force": opensees_linear_force,
        },
        "bilinear": {
            "target_displacement": target,
            "beamfem_displacement": beamfem_nonlinear_displacement,
            "opensees_displacement": opensees_nonlinear_displacement,
            "beamfem_stress": beamfem_nonlinear_stress,
            "opensees_stress": opensees_nonlinear_stress,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "validation" / "opensees_crosscheck_evidence.json",
    )
    args = parser.parse_args()
    evidence = generate_evidence()
    args.output.write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if not evidence["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
