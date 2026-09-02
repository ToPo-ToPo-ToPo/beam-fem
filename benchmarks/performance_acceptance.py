"""Generate machine-readable Phase-4 performance acceptance evidence.

The runner reports observations without converting unmet targets into passes.
Run from the repository root with::

    python -m benchmarks.performance_acceptance \
        --output validation/performance_evidence.json
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
import statistics
import subprocess
import tempfile
from time import perf_counter

import numpy as np

from beamfem import Material, Model, Section, UY
from beamfem.io import build_discrete_problem
from beamfem.optimize.backends import (
    ExactBackend, MILPBackend, SequentialQUBOOptimizer, SimulatedAnnealingBackend,
)
from beamfem.optimize.backends.milp import build_truss_sizing_milp
from beamfem.optimize.qubo import AdaptivePenalty, LocalQUBOBuilder
from beamfem.optimize.topology import GroundStructure
from beamfem.solver import factorize_static, solve_static

from .quantum_truss.generate_cases import generate_case


def _repository_metadata() -> dict[str, object]:
    def git(*args: str) -> str | None:
        try:
            return subprocess.check_output(["git", *args], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    return {
        "machine": platform.machine(), "platform": platform.platform(),
        "processor": platform.processor() or None,
        "python": platform.python_version(), "implementation": platform.python_implementation(),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
    }


def _beam_model(elements: int = 30) -> tuple[Model, int]:
    material = Material(E=200e9, nu=0.3, rho=7850.0)
    section = Section.rectangle(0.05, 0.08)
    model = Model()
    nodes = [model.add_node(i / elements * 4.0, 0.0, 0.0) for i in range(elements + 1)]
    for first, second in zip(nodes, nodes[1:]):
        model.add_element(first, second, material, section)
    model.fix(nodes[0])
    return model, nodes[-1]


def factorization_evidence(repeats: int = 9, load_cases: int = 16) -> dict[str, object]:
    baseline, reused = [], []
    for repeat in range(repeats + 1):
        model, tip = _beam_model()
        forces = [-1000.0 * (1.0 + index / load_cases) for index in range(load_cases)]
        start = perf_counter()
        for force in forces:
            model.nodal_loads = {(tip, UY): force}
            solve_static(model)
        baseline_time = perf_counter() - start
        model.nodal_loads = {}
        start = perf_counter()
        factorization = factorize_static(model)
        for force in forces:
            model.nodal_loads = {(tip, UY): force}
            factorization.solve_model(model)
        reused_time = perf_counter() - start
        if repeat:  # first iteration warms imports and allocator paths
            baseline.append(baseline_time); reused.append(reused_time)
    baseline_median = statistics.median(baseline)
    reused_median = statistics.median(reused)
    speedup = baseline_median / reused_median
    return {
        "repeats": repeats, "load_cases_per_repeat": load_cases,
        "baseline_seconds": baseline, "reuse_seconds": reused,
        "baseline_median_seconds": baseline_median,
        "reuse_median_seconds": reused_median,
        "speedup": speedup, "target_speedup": 3.0,
        "threshold_met": speedup >= 3.0,
        "comparison_scope": "assembly_and_factorization_per_load_vs_one_factorization_per_design",
    }


class _UncachedEvaluationProblem:
    """Thread-safe facade: FEM analyses are independent and cache writes disabled."""
    def __init__(self, problem):
        self.problem = problem
        self.initial_design = problem.initial_design
        self.catalogs = problem.catalogs
        problem.evaluate(problem.initial_design, use_cache=False)  # initialize evaluator before threads

    def evaluate(self, design):
        return self.problem.evaluate(design, use_cache=False)


def _truss_problem(size: str):
    document = generate_case(size)
    for member in document["members"]:
        member["member_type"] = "truss"
    return document, build_discrete_problem(document).problem


def parallel_candidate_evidence(repeats: int = 5, workers: int = 4,
                                target_speedup: float = 3.0) -> dict[str, object]:
    sequential, parallel = [], []
    for _ in range(repeats):
        _, raw = _truss_problem("small")
        problem = _UncachedEvaluationProblem(raw)
        start = perf_counter()
        LocalQUBOBuilder(problem, max_candidates=6,
                         penalty=AdaptivePenalty(value=1e6)).build(problem.initial_design)
        sequential.append(perf_counter() - start)
        _, raw = _truss_problem("small")
        problem = _UncachedEvaluationProblem(raw)
        start = perf_counter()
        LocalQUBOBuilder(problem, max_candidates=6, parallel_workers=workers,
                         parallel_safe=True,
                         penalty=AdaptivePenalty(value=1e6)).build(problem.initial_design)
        parallel.append(perf_counter() - start)
    sequential_median = statistics.median(sequential)
    parallel_median = statistics.median(parallel)
    speedup = sequential_median / parallel_median
    return {
        "repeats": repeats, "workers": workers,
        "sequential_seconds": sequential, "parallel_seconds": parallel,
        "sequential_median_seconds": sequential_median,
        "parallel_median_seconds": parallel_median,
        "speedup": speedup, "target_speedup": target_speedup,
        "threshold_met": speedup >= target_speedup,
        "release_gate_passed": speedup >= target_speedup,
        "note": "Observed local FEM candidate-build wall time; unmet speedup remains a failed gate.",
    }


def _ground_structure(document) -> GroundStructure:
    node_ids = {node["id"]: index for index, node in enumerate(document["nodes"])}
    nodes = np.asarray([node["xyz"] for node in document["nodes"]], dtype=float)
    members = [(node_ids[m["nodes"][0]], node_ids[m["nodes"][1]]) for m in document["members"]]
    dofs = {"UX": 0, "UY": 1, "UZ": 2}
    supports = {node_ids[s["node"]]: [dofs[d] for d in s["dofs"] if d in dofs]
                for s in document["supports"]}
    cases = []
    for loads in document["load_cases"].values():
        case = {}
        for load in loads:
            for dof, value in enumerate(load["force"]):
                if value:
                    key = (node_ids[load["node"]], dof)
                    case[key] = case.get(key, 0.0) + value
        cases.append(case)
    return GroundStructure(nodes, members, supports, cases)


def scale_evidence() -> dict[str, object]:
    evidence = {}
    for size in ("small", "medium", "large"):
        document, problem = _truss_problem(size)
        members = len(document["members"])
        states = 5 ** members
        item: dict[str, object] = {"nodes": len(document["nodes"]), "members": members,
                                  "design_states": states}
        try:
            ExactBackend(max_combinations=200_000).solve(problem)
            item["full_exact_limit_triggered"] = False
        except ValueError as exc:
            item["full_exact_limit_triggered"] = True
            item["full_exact_message"] = str(exc)
        if size == "small":
            builder = LocalQUBOBuilder(problem, max_candidates=6,
                penalty=AdaptivePenalty(value=1e6))
            started = perf_counter(); qubo, decode = builder.build(problem.initial_design)
            build_seconds = perf_counter() - started
            started = perf_counter(); exact = qubo.exact_solution(); exact_seconds = perf_counter() - started
            evaluation = problem.evaluate(type(problem.initial_design)(decode(exact.bits)))
            material = document["materials"]["steel"]
            areas = [0.0] + [entry["area"] for entry in document["section_catalogs"]["round_bar"]]
            formulation = build_truss_sizing_milp(
                _ground_structure(document), areas, material["density"],
                material["tension_allowable"], material["compression_allowable"],
            )
            started = perf_counter(); milp_result = MILPBackend(formulation).solve(problem)
            item.update({
                "local_exact_qubo": {"variables": qubo.n_variables,
                    "fem_evaluations": builder.last_metadata["fem_evaluations"],
                    "build_seconds": build_seconds, "solve_seconds": exact_seconds,
                    "energy": exact.energy, "fem_objective": evaluation.objective,
                    "fem_feasible": evaluation.feasible},
                "equilibrium_capacity_milp": {"runtime_seconds": perf_counter()-started,
                    "mip_gap": milp_result.solver_metadata.get("mip_gap"),
                    "linear_objective": milp_result.solver_metadata.get("linear_objective"),
                    "fem_objective": milp_result.objective, "fem_feasible": milp_result.feasible,
                    "scope": milp_result.solver_metadata.get("formulation_scope")},
            })
        else:
            checkpoint = Path(tempfile.mkdtemp(prefix=f"beamfem-{size}-acceptance-")) / "checkpoint.json"
            builder = LocalQUBOBuilder(problem, max_candidates=4,
                penalty=AdaptivePenalty(value=1e6))
            optimizer = SequentialQUBOOptimizer(
                SimulatedAnnealingBackend(sweeps=50, restarts=2, seed=1), builder,
                max_iterations=1, checkpoint_path=checkpoint,
            )
            started = perf_counter(); result = optimizer.solve(problem); elapsed = perf_counter()-started
            resumed_builder = LocalQUBOBuilder(problem, max_candidates=4,
                penalty=AdaptivePenalty(value=1e6))
            resumed = SequentialQUBOOptimizer(
                SimulatedAnnealingBackend(sweeps=50, restarts=2, seed=1), resumed_builder,
                max_iterations=1, checkpoint_path=checkpoint, resume=True,
            ).solve(problem)
            item["limited_local_run"] = {
                "runtime_seconds": elapsed, "fem_evaluations": result.evaluations,
                "fem_objective": result.objective, "fem_feasible": result.feasible,
                "checkpoint_written": checkpoint.exists(),
                "checkpoint_resumed": resumed.solver_metadata["resumed"],
                "accepted_as_solution": bool(result.feasible),
            }
        evidence[size] = item
    return evidence


def collect_evidence() -> dict[str, object]:
    factorization = factorization_evidence()
    parallel = parallel_candidate_evidence()
    scale = scale_evidence()
    return {
        "schema_version": 1, "environment": _repository_metadata(),
        "factorization_reuse": factorization, "parallel_candidate_evaluation": parallel,
        "scale_cases": scale,
        "required_performance_gates": {
            "factorization_3x_speedup": factorization["threshold_met"],
            "medium_exact_limit_triggered": scale["medium"]["full_exact_limit_triggered"],
            "medium_checkpoint_resume": scale["medium"]["limited_local_run"]["checkpoint_resumed"],
            "large_exact_limit_triggered": scale["large"]["full_exact_limit_triggered"],
            "large_checkpoint_resume": scale["large"]["limited_local_run"]["checkpoint_resumed"],
            "all_required_performance_gates_passed": bool(
                factorization["threshold_met"]
                and scale["medium"]["full_exact_limit_triggered"]
                and scale["medium"]["limited_local_run"]["checkpoint_resumed"]
                and scale["large"]["full_exact_limit_triggered"]
                and scale["large"]["limited_local_run"]["checkpoint_resumed"]
            ),
        },
        "known_performance_limitations": {
            "parallel_3x_speedup": parallel["threshold_met"],
        },
        "solution_quality_gates": {
            "small_milp_fem_feasible": scale["small"]["equilibrium_capacity_milp"]["fem_feasible"],
            "medium_candidate_feasible": scale["medium"]["limited_local_run"]["fem_feasible"],
            "large_candidate_feasible": scale["large"]["limited_local_run"]["fem_feasible"],
            "all_solution_quality_gates_passed": bool(
                scale["small"]["equilibrium_capacity_milp"]["fem_feasible"]
                and scale["medium"]["limited_local_run"]["fem_feasible"]
                and scale["large"]["limited_local_run"]["fem_feasible"]
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("validation/performance_evidence.json"))
    args = parser.parse_args()
    evidence = collect_evidence()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
