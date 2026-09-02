"""Combine all machine evidence and external approvals into one release decision."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit(root: Path = ROOT) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _timestamp(value: Any) -> bool:
    if not _nonempty(value):
        return False
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None


def _commit_id(value: Any) -> bool:
    return isinstance(value, str) and len(value) in {40, 64} and all(
        character in "0123456789abcdefABCDEF" for character in value
    )


def _approved_review(record: dict, candidate_commit: str | None) -> tuple[bool, list[str]]:
    """Validate a real independent-review record, not a status-only toggle."""

    errors: list[str] = []
    if record.get("status") != "approved":
        errors.append("status is not approved")
    if not _commit_id(candidate_commit) or record.get("candidate_commit") != candidate_commit:
        errors.append("candidate_commit does not match the release candidate")
    reviewer = record.get("reviewer")
    if not isinstance(reviewer, dict):
        errors.append("reviewer record is missing")
    else:
        for field in ("name", "organization", "qualification"):
            if not _nonempty(reviewer.get(field)):
                errors.append(f"reviewer.{field} is missing")
        if reviewer.get("independent_from_implementation") is not True:
            errors.append("reviewer is not confirmed independent")
    checks = record.get("checks")
    if not isinstance(checks, dict) or not checks or any(
        value != "approved" for value in checks.values()
    ):
        errors.append("every review check must be approved")
    if record.get("blocking_findings") not in ([], ()):
        errors.append("blocking findings remain open")
    if not _timestamp(record.get("reviewed_at")):
        errors.append("reviewed_at must be an ISO-8601 timestamp with timezone")
    if not _nonempty(record.get("signature_or_approval_reference")):
        errors.append("signature_or_approval_reference is missing")
    return not errors, errors


def _safe_artifact(root: Path, relative: Any) -> Path | None:
    if not _nonempty(relative):
        return None
    candidate = (root / str(relative)).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    return candidate


def _approved_pilot(record: dict, root: Path,
                    candidate_commit: str | None) -> tuple[bool, list[str]]:
    """Validate pilot approval, artifact integrity, and comparison results."""

    errors: list[str] = []
    if record.get("status") != "approved":
        errors.append("status is not approved")
    if not _nonempty(record.get("pilot_id")):
        errors.append("pilot_id is missing")
    if not _commit_id(candidate_commit) or record.get("candidate_commit") != candidate_commit:
        errors.append("candidate_commit does not match the release candidate")
    for label in ("input", "output"):
        path = _safe_artifact(root, record.get(f"{label}_path"))
        digest = record.get(f"{label}_checksum")
        if path is None or not path.is_file():
            errors.append(f"{label} artifact is missing or outside the repository")
        elif not _nonempty(digest) or _sha256(path) != digest:
            errors.append(f"{label} checksum does not match")
    if not _nonempty(record.get("independent_reference")):
        errors.append("independent_reference is missing")
    comparisons = record.get("comparisons")
    if not isinstance(comparisons, dict) or not comparisons or any(
        value != "pass" for value in comparisons.values()
    ):
        errors.append("every pilot comparison must pass")
    if not _nonempty(record.get("accepted_by")):
        errors.append("accepted_by is missing")
    if not _timestamp(record.get("accepted_at")):
        errors.append("accepted_at must be an ISO-8601 timestamp with timezone")
    if not _nonempty(record.get("signature_or_approval_reference")):
        errors.append("signature_or_approval_reference is missing")
    return not errors, errors


def _external_record(path: Path | None, root: Path) -> tuple[dict, str | None]:
    """Load an approval record only from outside the source repository."""

    if path is None:
        return {}, "external approval record was not supplied"
    resolved = path.resolve()
    if not resolved.is_file():
        return {}, f"external approval record is missing: {resolved}"
    try:
        resolved.relative_to(root.resolve())
    except ValueError:
        pass
    else:
        return {}, "approval records inside the source repository are not trusted"
    try:
        return _load(resolved), None
    except (OSError, json.JSONDecodeError) as exc:
        return {}, f"external approval record is invalid: {exc}"


def collect_release_decision(
    root: Path = ROOT, *, independent_review_path: Path | None = None,
    pilot_record_paths: tuple[Path, ...] = (),
) -> dict:
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
        "code_check_reference": validation / "code_check_reference_evidence.json",
        "opensees_crosscheck": validation / "opensees_crosscheck_evidence.json",
        "endurance": validation / "endurance_evidence.json",
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
        "code_check_reference": bool(evidence["code_check_reference"].get("passed")),
        "opensees_crosscheck": bool(evidence["opensees_crosscheck"].get("passed")),
        "endurance": bool(evidence["endurance"].get("passed")),
        "dependency_lock_present": (root / "requirements-release.lock").is_file(),
    }
    candidate_commit = _git_commit(root)
    independent, independent_load_error = _external_record(
        independent_review_path, root
    )
    review_approved, review_errors = _approved_review(independent, candidate_commit)
    if independent_load_error is not None:
        review_errors.insert(0, independent_load_error)
    pilot_files = tuple(pilot_record_paths)
    approved_pilots = []
    pilot_errors: dict[str, list[str]] = {}
    pilot_ids: set[str] = set()
    for path in pilot_files:
        record, load_error = _external_record(path, root)
        approved, errors = _approved_pilot(record, root, candidate_commit)
        if load_error is not None:
            errors.insert(0, load_error)
        pilot_id = str(record.get("pilot_id", ""))
        if approved and pilot_id in pilot_ids:
            approved = False
            errors.append("pilot_id duplicates another approved pilot")
        if approved:
            pilot_ids.add(pilot_id)
            approved_pilots.append(path)
        else:
            pilot_errors[str(path.resolve())] = errors
    if len(approved_pilots) < 2:
        pilot_errors["_gate"] = [
            "at least two distinct externally supplied approved pilot records are required; "
            f"received {len(approved_pilots)}"
        ]
    external_checks = {
        "independent_engineer_review": review_approved,
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
    if lock.is_file():
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
            "git_commit": candidate_commit,
        },
        "automated_checks": automated_checks,
        "external_checks": external_checks,
        "external_check_diagnostics": {
            "independent_review": review_errors,
            "pilots": pilot_errors,
        },
        "external_approval_inputs": {
            "independent_review": None if independent_review_path is None else str(independent_review_path.resolve()),
            "pilot_records": [str(path.resolve()) for path in pilot_files],
            "policy": "approval records committed inside the source repository are never trusted",
        },
        "approved_pilot_files": [str(path.resolve()) for path in approved_pilots],
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
    parser.add_argument(
        "--independent-review", type=Path,
        help="signed review JSON outside the source repository",
    )
    parser.add_argument(
        "--pilot-record", type=Path, action="append", default=[],
        help="signed pilot approval JSON outside the source repository; repeat at least twice",
    )
    args = parser.parse_args()
    decision = collect_release_decision(
        independent_review_path=args.independent_review,
        pilot_record_paths=tuple(args.pilot_record),
    )
    args.output.write_text(
        json.dumps(decision, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    if not decision["automated_release_candidate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
