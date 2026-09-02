"""Exact-vs-MILP evidence for a statically determinate discrete truss."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import subprocess
from time import perf_counter

import numpy as np

from beamfem import Material, Model, Section, UX, UY, UZ
from beamfem.optimize import (
    DesignState, DiscreteStructuralProblem, LoadCase, SectionCatalog,
    SectionOption, StressLimit,
)
from beamfem.optimize.backends import ExactBackend, MILPBackend
from beamfem.optimize.backends.milp import build_truss_sizing_milp
from beamfem.optimize.topology import GroundStructure


def build_micro_case():
    """Two axial members in series; capacity alone determines both sections."""
    material = Material(E=200e9, nu=0.3, rho=7850.0, name="steel")
    small = Section(A=1.0e-4, Iy=1e-9, Iz=1e-9, J=2e-9, name="S")
    large = Section(A=2.0e-4, Iy=4e-9, Iz=4e-9, J=8e-9, name="L")
    allowable = 60e6
    catalog = SectionCatalog("bar", (
        SectionOption("OFF", None),
        SectionOption("S", small, material, allowable, allowable),
        SectionOption("L", large, material, allowable, allowable),
    ))
    model = Model()
    nodes = [model.add_node(float(index), 0.0, 0.0) for index in range(3)]
    model.add_truss(nodes[0], nodes[1], material, large)
    model.add_truss(nodes[1], nodes[2], material, large)
    model.pin(nodes[0])
    model.fix(nodes[1], [UY, UZ])
    model.fix(nodes[2], [UY, UZ])
    load = 8000.0  # S capacity=6000 N, L capacity=12000 N
    problem = DiscreteStructuralProblem(
        model=model, catalogs=(catalog, catalog),
        load_cases=(LoadCase("tension", {(nodes[2], UX): load}),
                    LoadCase("compression", {(nodes[2], UX): -load})),
        constraints=(StressLimit(),), initial_design=DesignState((2, 2)),
    )
    ground = GroundStructure(
        nodes=np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
        members=[(0, 1), (1, 2)],
        supports={0: [0, 1], 1: [1], 2: [1]},
        load_cases=[{(2, 0): load}, {(2, 0): -load}],
    )
    formulation = build_truss_sizing_milp(
        ground, [0.0, small.A, large.A], material.rho,
        allowable, allowable,
    )
    return problem, formulation


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def collect_evidence() -> dict[str, object]:
    problem, formulation = build_micro_case()
    started = perf_counter(); exact = ExactBackend(max_combinations=100).solve(problem)
    exact_seconds = perf_counter() - started
    started = perf_counter(); milp = MILPBackend(formulation).solve(problem)
    milp_seconds = perf_counter() - started
    exact_choices = list(exact.design.choices)
    milp_choices = list(milp.design.choices)
    checks = {
        "same_design": exact_choices == milp_choices,
        "same_objective": bool(np.isclose(exact.objective, milp.objective, rtol=0.0, atol=1e-12)),
        "exact_feasible": exact.feasible,
        "milp_fem_feasible": milp.feasible,
        "milp_zero_gap": bool(abs(float(milp.solver_metadata["mip_gap"])) <= 1e-12),
        "off_index_preserved": exact_choices == [2, 2] and milp_choices == [2, 2],
    }
    problem_payload = json.dumps({
        "members": [[0, 1], [1, 2]], "areas": [0.0, 1.0e-4, 2.0e-4],
        "loads": [8000.0, -8000.0], "allowable": 60e6,
    }, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "schema_version": 1,
        "environment": {"machine": platform.machine(), "platform": platform.platform(),
                        "python": platform.python_version(), "git_commit": _git_commit()},
        "problem": {"members": 2, "states_per_member": 3, "combinations": 9,
                    "input_sha256": hashlib.sha256(problem_payload).hexdigest(),
                    "section_index_map": {"0": "OFF", "1": "S", "2": "L"},
                    "load_cases": ["tension", "compression"],
                    "constraints": ["axial_stress"],
                    "milp_scope": "truss_equilibrium_and_section_capacity"},
        "exact": {"design": exact_choices, "objective_mass_kg": exact.objective,
                  "feasible": exact.feasible, "evaluations": exact.evaluations,
                  "runtime_seconds": exact_seconds},
        "milp": {"design": milp_choices, "objective_mass_kg": milp.objective,
                 "linear_objective_mass_kg": milp.solver_metadata["linear_objective"],
                 "feasible_after_common_fem": milp.feasible,
                 "mip_gap": milp.solver_metadata["mip_gap"], "runtime_seconds": milp_seconds},
        "checks": checks, "strict_acceptance_passed": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("validation/exact_milp_micro_evidence.json"))
    args = parser.parse_args()
    evidence = collect_evidence()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, ensure_ascii=False,
                                      allow_nan=False) + "\n", encoding="utf-8")
    if not evidence["strict_acceptance_passed"]:
        raise SystemExit("Exact/MILP strict acceptance failed")
    print(args.output)


if __name__ == "__main__":
    main()
