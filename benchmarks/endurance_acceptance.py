"""Bounded large-case endurance and determinism acceptance benchmark."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
import tracemalloc

from beamfem.optimize.problem import DesignState

from .performance_acceptance import _truss_problem


def _rss_bytes() -> int:
    """Return peak resident bytes using only platform standard libraries."""

    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class _Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]
        counters = _Counters(); counters.cb = ctypes.sizeof(counters)
        process = ctypes.windll.kernel32.GetCurrentProcess()
        if not ctypes.windll.psapi.GetProcessMemoryInfo(
            process, ctypes.byref(counters), counters.cb
        ):
            raise OSError("GetProcessMemoryInfo failed")
        return int(counters.PeakWorkingSetSize)
    import resource

    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_endurance(*, minimum_seconds: float = 30.0, minimum_evaluations: int = 100,
                  memory_limit_mb: float = 1024.0) -> dict:
    if minimum_seconds < 0.0 or minimum_evaluations < 1 or memory_limit_mb <= 0.0:
        raise ValueError("endurance limits must be nonnegative/positive")
    _, problem = _truss_problem("large")
    base = tuple(
        max(index for index, option in enumerate(catalog) if option.active)
        for catalog in problem.catalogs
    )
    rss_start = _rss_bytes()
    tracemalloc.start()
    start = perf_counter()
    evaluations = 0
    finite = True
    signatures: dict[tuple[int, ...], tuple[float, float, bool]] = {}
    deterministic = True
    while evaluations < minimum_evaluations or perf_counter() - start < minimum_seconds:
        values = list(base)
        member = evaluations % len(values)
        active = [index for index, option in enumerate(problem.catalogs[member]) if option.active]
        values[member] = active[(evaluations // len(values)) % len(active)]
        design = DesignState(values)
        result = problem.evaluate(design)
        signature = (float(result.objective), float(result.mass), bool(result.feasible))
        finite = finite and all(math.isfinite(value) for value in signature[:2])
        key = design.choices
        if key in signatures:
            deterministic = deterministic and signatures[key] == signature
        else:
            signatures[key] = signature
        evaluations += 1
    elapsed = perf_counter() - start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_end = _rss_bytes()
    memory_limit = int(memory_limit_mb * 1024 * 1024)
    checks = {
        "minimum_duration_reached": elapsed >= minimum_seconds,
        "minimum_evaluations_reached": evaluations >= minimum_evaluations,
        "finite_results": finite,
        "repeat_results_deterministic": deterministic,
        "python_peak_memory_within_limit": peak <= memory_limit,
        "rss_peak_within_limit": rss_end <= memory_limit,
    }
    return {
        "evidence_schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "platform": platform.platform(), "python": platform.python_version(),
            "git_commit": _git_commit(),
        },
        "case": "large (201 candidate members)",
        "minimum_seconds": minimum_seconds,
        "elapsed_seconds": elapsed,
        "minimum_evaluations": minimum_evaluations,
        "completed_evaluations": evaluations,
        "unique_designs": len(signatures),
        "evaluations_per_second": evaluations / elapsed,
        "memory_limit_mb": memory_limit_mb,
        "tracemalloc_current_bytes": current,
        "tracemalloc_peak_bytes": peak,
        "rss_start_bytes": rss_start,
        "rss_peak_bytes": rss_end,
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--evaluations", type=int, default=100)
    parser.add_argument("--memory-limit-mb", type=float, default=1024.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence = run_endurance(
        minimum_seconds=args.seconds,
        minimum_evaluations=args.evaluations,
        memory_limit_mb=args.memory_limit_mb,
    )
    args.output.write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if not evidence["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
