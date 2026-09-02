"""Run pip-audit against the locked RC environment and retain JSON evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def summarize_pip_audit(payload: dict | list) -> tuple[int, int]:
    dependencies = payload.get("dependencies", []) if isinstance(payload, dict) else payload
    vulnerabilities = sum(len(item.get("vulns", [])) for item in dependencies)
    return len(dependencies), vulnerabilities


def collect_evidence(lock: Path) -> dict:
    command = [
        sys.executable, "-m", "pip_audit", "--format=json", "--requirement", str(lock)
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "pip-audit did not return JSON; install pip-audit and verify network access: "
            + completed.stderr.strip()
        ) from exc
    dependencies, vulnerabilities = summarize_pip_audit(payload)
    return {
        "evidence_schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {"platform": platform.platform(), "python": platform.python_version()},
        "lock_file": str(lock.relative_to(ROOT)),
        "command": command[2:],
        "dependency_count": dependencies,
        "vulnerability_count": vulnerabilities,
        "pip_audit_exit_code": completed.returncode,
        "audit": payload,
        "stderr": completed.stderr.strip(),
        "passed": completed.returncode == 0 and vulnerabilities == 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=ROOT / "requirements-release.lock")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "validation" / "security_evidence.json"
    )
    args = parser.parse_args()
    evidence = collect_evidence(args.lock.resolve())
    args.output.write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    if not evidence["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
