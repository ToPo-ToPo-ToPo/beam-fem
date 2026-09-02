"""Backend-neutral benchmark runner for discrete truss optimization."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Mapping, Protocol

from beamfem.io import load_problem_spec, validate_problem_spec, write_result_json
from beamfem.validation import build_audit_metadata, diagnose_problem_spec

from .generate_cases import CASE_SIZES, generate_case


class SolverCallable(Protocol):
    def __call__(
        self, problem: Mapping[str, Any], settings: Mapping[str, Any]
    ) -> Any: ...


@dataclass(frozen=True)
class BenchmarkRecord:
    case: str
    solver: str
    seed: int | None
    runtime_seconds: float
    node_count: int
    member_count: int
    design_state_count: int
    result: Any
    diagnostics: tuple[dict[str, Any], ...]


def _load_callable(reference: str) -> SolverCallable:
    module_name, separator, attribute = reference.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("solver factory must use 'package.module:function'")
    candidate = getattr(importlib.import_module(module_name), attribute)
    if not callable(candidate):
        raise TypeError(f"{reference!r} does not resolve to a callable")
    return candidate


def run_benchmark(
    problem: Mapping[str, Any],
    *,
    case: str,
    solver_name: str,
    solver: SolverCallable | None = None,
    seed: int | None = None,
    solver_settings: Mapping[str, Any] | None = None,
    repository: str | Path | None = None,
) -> tuple[BenchmarkRecord, Any]:
    """Validate, diagnose, time, and audit one solver invocation.

    A ``None`` solver performs a useful dry run of the benchmark assets. This
    keeps the runner operational before or without optional solver backends.
    """

    validated = validate_problem_spec(problem)
    report = diagnose_problem_spec(validated.data)
    settings = dict(solver_settings or {})
    if seed is not None:
        settings.setdefault("seed", seed)

    started = perf_counter()
    result = (
        solver(validated.data, settings)
        if solver is not None
        else {"status": "validated", "feasible": None}
    )
    elapsed = perf_counter() - started
    catalog_sizes = {
        name: len(entries)
        for name, entries in validated.data["section_catalogs"].items()
    }
    state_count = sum(
        1 + catalog_sizes[member["catalog"]] for member in validated.data["members"]
    )
    diagnostics = tuple(
        {
            "code": item.code,
            "severity": item.severity.value,
            "message": item.message,
            "location": item.location,
        }
        for item in report.diagnostics
    )
    record = BenchmarkRecord(
        case=case,
        solver=solver_name,
        seed=seed,
        runtime_seconds=elapsed,
        node_count=len(validated.data["nodes"]),
        member_count=len(validated.data["members"]),
        design_state_count=state_count,
        result=result,
        diagnostics=diagnostics,
    )
    audit = build_audit_metadata(
        solver=solver_name,
        seed=seed,
        solver_settings=settings,
        warnings=report.warnings,
        repository=repository,
    )
    return record, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--size", choices=sorted(CASE_SIZES))
    source.add_argument("--input", type=Path)
    parser.add_argument("--solver-factory")
    parser.add_argument("--solver-name", default="dry-run")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.input:
        problem = load_problem_spec(args.input).data
        case = args.input.stem
    else:
        problem = generate_case(args.size)
        case = str(args.size)
    solver = _load_callable(args.solver_factory) if args.solver_factory else None
    repository = Path(__file__).resolve().parents[2]
    record, audit = run_benchmark(
        problem,
        case=case,
        solver_name=args.solver_name,
        solver=solver,
        seed=args.seed,
        repository=repository,
    )
    write_result_json(record, args.output, audit=audit)


if __name__ == "__main__":
    main()
