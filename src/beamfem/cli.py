"""Command-line entry point for audited discrete structural optimization."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

from .io import (
    RunStatus, build_discrete_problem, create_run_manifest, load_problem_spec,
    load_run_manifest, verify_resume_compatibility, write_design_report,
    write_result_json, write_run_manifest,
)
from .optimize.backends import (
    ExactBackend,
    GreedyBackend,
    QAOABackend,
    QiskitNotInstalledError,
    SequentialQUBOOptimizer,
    SimulatedAnnealingBackend,
    SolverLimits,
)
from .optimize.qubo import AdaptivePenalty, LocalQUBOBuilder, TrustRegion
from .validation import (
    build_audit_metadata, build_dependency_audit, diagnose_problem_spec,
    sha256_file, write_dependency_audit,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run reproducible discrete frame optimization from JSON/YAML."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("exact", "greedy", "sa", "qaoa"), default="greedy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time-limit", type=float)
    parser.add_argument(
        "--memory-limit-mb", type=float,
        help="stop between optimization steps when process peak RSS reaches this limit",
    )
    parser.add_argument("--max-evaluations", type=int)
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--exact-max-combinations", type=int, default=200_000)
    parser.add_argument("--penalty", type=float, default=1.0e6)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument(
        "--parallel-workers", type=int, default=1,
        help="isolated FEM worker processes for local-QUBO candidate evaluation",
    )
    parser.add_argument("--trust-radius", type=int, default=2)
    parser.add_argument("--sa-sweeps", type=int, default=1500)
    parser.add_argument("--sa-restarts", type=int, default=8)
    parser.add_argument("--qaoa-reps", type=int, default=1)
    parser.add_argument("--qaoa-maxiter", type=int, default=100)
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--fallback", choices=("none", "greedy", "sa"), default="greedy")
    parser.add_argument("--manifest", type=Path, help="write an atomic resumable run manifest")
    parser.add_argument("--resume", action="store_true", help="resume/revalidate the specified manifest")
    parser.add_argument("--html-report", type=Path, help="write a preliminary-design HTML report")
    parser.add_argument("--dependency-audit", type=Path, help="write checksums and an SBOM-like inventory")
    parser.add_argument(
        "--optimizer-checkpoint", type=Path,
        help="write/resume the integrity-checked local-QUBO iteration checkpoint",
    )
    return parser


def _solver_settings(args) -> dict:
    excluded = {
        "input", "output", "manifest", "resume", "html_report", "dependency_audit",
        "optimizer_checkpoint",
    }
    return {key: value for key, value in vars(args).items() if key not in excluded}


def _sequential(problem, args, quantum: bool, *, use_checkpoint: bool = True):
    radius = max(1, args.trust_radius)
    workers = max(1, args.parallel_workers)
    builder = LocalQUBOBuilder(
        problem,
        max_candidates=args.candidates,
        trust_region=TrustRegion(radius=radius, minimum=1, maximum=max(8, radius)),
        penalty=AdaptivePenalty(value=args.penalty),
        parallel_workers=workers,
        parallel_backend="process",
        persistent_workers=workers > 1,
    )
    if quantum:
        solver = QAOABackend(
            reps=args.qaoa_reps,
            shots=args.shots,
            seed=args.seed,
            maxiter=args.qaoa_maxiter,
        )
    else:
        solver = SimulatedAnnealingBackend(
            sweeps=args.sa_sweeps, restarts=args.sa_restarts, seed=args.seed
        )
    return SequentialQUBOOptimizer(
        solver, builder, max_iterations=args.max_iterations,
        checkpoint_path=args.optimizer_checkpoint if use_checkpoint else None,
        resume=args.resume if use_checkpoint else False,
    )


def _backend(problem, args):
    if args.backend == "exact":
        return ExactBackend(max_combinations=args.exact_max_combinations)
    if args.backend == "greedy":
        return GreedyBackend(penalty=args.penalty, pairwise=True)
    if args.backend == "sa":
        return _sequential(problem, args, False)
    return _sequential(problem, args, True)


def _fallback(problem, args):
    if args.fallback == "greedy":
        return GreedyBackend(penalty=args.penalty, pairwise=True)
    if args.fallback == "sa":
        # A QAOA checkpoint has a different solver fingerprint and must never
        # be reused by the SA fallback.
        return _sequential(problem, args, False, use_checkpoint=False)
    return None


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if args.resume and args.manifest is None:
        raise ValueError("--resume requires --manifest")
    spec = load_problem_spec(args.input)
    settings = _solver_settings(args)
    manifest = None
    if args.manifest is not None:
        if args.resume:
            manifest = load_run_manifest(args.manifest)
            verify_resume_compatibility(
                manifest, spec.data, solver=args.backend,
                solver_settings=settings, seed=args.seed,
            )
            if manifest.status is RunStatus.COMPLETED and args.output.exists():
                artifact_paths = manifest.checkpoint.get("artifact_paths", {})
                for name, expected_digest in manifest.artifacts.items():
                    stored_path = artifact_paths.get(name)
                    if not stored_path:
                        raise ValueError(
                            f"completed manifest lacks path for artifact {name!r}"
                        )
                    artifact_path = Path(stored_path)
                    if not artifact_path.is_file():
                        raise ValueError(f"completed artifact is missing: {artifact_path}")
                    if sha256_file(artifact_path).digest != expected_digest:
                        raise ValueError(
                            f"completed artifact checksum mismatch: {artifact_path}"
                        )
                print(f"run {manifest.run_id} already completed: output={args.output}")
                return 0
        else:
            manifest = create_run_manifest(
                spec.data, solver=args.backend, solver_settings=settings, seed=args.seed
            )
        manifest = manifest.advance(
            "input_validated", checkpoint={"input": str(args.input.resolve())}
        )
        write_run_manifest(manifest, args.manifest)
    preflight = diagnose_problem_spec(spec.data)
    built = build_discrete_problem(spec)
    limits = SolverLimits(
        max_evaluations=args.max_evaluations,
        max_iterations=args.max_iterations,
        time_limit=args.time_limit,
        memory_limit_mb=args.memory_limit_mb,
    )
    warnings = list(preflight.warnings)
    selected_backend = args.backend

    def solve_and_close(backend):
        try:
            return backend.solve(built.problem, limits=limits)
        finally:
            builder = getattr(backend, "builder", None)
            close = getattr(builder, "close", None)
            if close is not None:
                close()

    try:
        try:
            result = solve_and_close(_backend(built.problem, args))
        except (QiskitNotInstalledError, RuntimeError) as exc:
            fallback = _fallback(built.problem, args) if args.backend == "qaoa" else None
            if fallback is None:
                raise
            warnings.append(f"QAOA failed; used {args.fallback} fallback: {exc}")
            selected_backend = args.fallback
            result = solve_and_close(fallback)
    except Exception as exc:
        if manifest is not None:
            manifest = manifest.advance(
                "solver_failed",
                checkpoint={"error_type": type(exc).__name__, "message": str(exc)},
                status=RunStatus.FAILED,
            )
            write_run_manifest(manifest, args.manifest)
        raise

    audit = build_audit_metadata(
        solver=selected_backend,
        seed=args.seed,
        solver_settings={
            **settings,
            "selected_backend": selected_backend,
        },
        warnings=warnings,
        repository=Path(__file__).resolve().parents[2],
    )
    payload = {
        "problem": spec.data.get("name", args.input.stem),
        "input": str(args.input.resolve()),
        "node_ids": dict(built.node_ids),
        "member_ids": dict(built.member_ids),
        "preflight": [
            {
                "code": item.code,
                "severity": item.severity.value,
                "message": item.message,
                "location": item.location,
            }
            for item in preflight.diagnostics
        ],
        "optimization": result,
    }
    write_result_json(payload, args.output, audit=audit)
    if args.html_report is not None:
        write_design_report(
            payload, args.html_report, audit=audit, manifest=manifest,
            title=f"beamfem preliminary report: {payload['problem']}",
        )
    if args.dependency_audit is not None:
        artifact_paths = [args.output]
        if args.html_report is not None:
            artifact_paths.append(args.html_report)
        packages = ["beamfem", "numpy", "scipy"]
        if args.backend == "qaoa":
            packages.extend(("qiskit", "qiskit-optimization", "qiskit-aer"))
        dependency_audit = build_dependency_audit(
            packages=packages, artifacts=artifact_paths
        )
        write_dependency_audit(dependency_audit, args.dependency_audit)
    if manifest is not None:
        artifacts = {"result": sha256_file(args.output).digest}
        artifact_paths = {"result": str(args.output.resolve())}
        if args.html_report is not None:
            artifacts["html_report"] = sha256_file(args.html_report).digest
            artifact_paths["html_report"] = str(args.html_report.resolve())
        if args.dependency_audit is not None:
            artifacts["dependency_audit"] = sha256_file(args.dependency_audit).digest
            artifact_paths["dependency_audit"] = str(args.dependency_audit.resolve())
        manifest = manifest.advance(
            "optimization_completed",
            checkpoint={
                "backend": selected_backend,
                "objective": result.objective,
                "feasible": result.feasible,
                "evaluations": result.evaluations,
                "artifact_paths": artifact_paths,
            },
            status=RunStatus.COMPLETED,
        )
        manifest = replace(manifest, artifacts=artifacts)
        write_run_manifest(manifest, args.manifest)
    print(
        f"{result.backend}: objective={result.objective:.6g}, "
        f"feasible={result.feasible}, evaluations={result.evaluations}, "
        f"output={args.output}"
    )
    return 0 if result.feasible else 2


if __name__ == "__main__":
    sys.exit(main())
