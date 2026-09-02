"""Combine all machine evidence and external approvals into one release decision."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def collect_release_decision(root: Path = ROOT) -> dict:
    validation = root / "validation"
    evidence_paths = {
        "reference": validation / "reference_evidence.json",
        "exact_milp": validation / "exact_milp_micro_evidence.json",
        "performance": validation / "performance_evidence.json",
        "stochastic": validation / "stochastic_evidence.json",
        "legacy_regression": validation / "legacy_regression_evidence.json",
        "quantum_smoke": validation / "quantum_evidence.json",
        "security": validation / "security_evidence.json",
        "dependency_inventory": validation / "dependency_evidence.json",
    }
    missing = [name for name, path in evidence_paths.items() if not path.is_file()]
    if missing:
        raise ValueError("missing release evidence: " + ", ".join(missing))
    evidence = {name: _load(path) for name, path in evidence_paths.items()}
    performance = evidence["performance"]
    automated_checks = {
        "reference_cases": bool(evidence["reference"].get("passed")),
        "exact_milp_micro": bool(
            evidence["exact_milp"].get("strict_acceptance_passed")
        ),
        "performance": bool(
            performance["required_performance_gates"][
                "all_required_performance_gates_passed"
            ]
        ),
        "solution_quality": bool(
            performance["solution_quality_gates"][
                "all_solution_quality_gates_passed"
            ]
        ),
        "stochastic_ten_seed": bool(evidence["stochastic"].get("passed")),
        "legacy_159_91_88_regression": bool(
            evidence["legacy_regression"].get("passed")
        ),
        "quantum_noisy_smoke": bool(evidence["quantum_smoke"].get("passed")),
        "dependency_security": bool(evidence["security"].get("passed")),
        "dependency_inventory": bool(evidence["dependency_inventory"].get("passed")),
        "dependency_lock_present": (root / "requirements-release.lock").is_file(),
    }
    independent = _load(validation / "independent_review_template.json")
    pilot_files = sorted((validation / "pilots").glob("*.json")) if (
        validation / "pilots"
    ).is_dir() else []
    approved_pilots = [path for path in pilot_files if _load(path).get("status") == "approved"]
    external_checks = {
        "independent_engineer_review": independent.get("status") == "approved",
        "minimum_two_approved_pilots": len(approved_pilots) >= 2,
    }
    automated_passed = all(automated_checks.values())
    external_passed = all(external_checks.values())
    artifacts = {
        name: {
            "path": str(path.relative_to(root)),
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for name, path in evidence_paths.items()
    }
    lock = root / "requirements-release.lock"
    artifacts["dependency_lock"] = {
        "path": str(lock.relative_to(root)),
        "sha256": _sha256(lock),
        "size_bytes": lock.stat().st_size,
    }
    return {
        "evidence_schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "git_commit": _git_commit(),
        },
        "automated_checks": automated_checks,
        "external_checks": external_checks,
        "approved_pilot_files": [str(path.relative_to(root)) for path in approved_pilots],
        "artifacts": artifacts,
        "automated_release_candidate_passed": automated_passed,
        "external_approvals_passed": external_passed,
        "eligible_for_v1_0": automated_passed and external_passed,
        "release_stage": (
            "v1.0-eligible" if automated_passed and external_passed
            else "release-candidate" if automated_passed
            else "blocked"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "validation" / "release_gate_evidence.json"
    )
    args = parser.parse_args()
    decision = collect_release_decision()
    args.output.write_text(
        json.dumps(decision, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    if not decision["automated_release_candidate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
