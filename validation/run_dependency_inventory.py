"""Generate the locked dependency inventory and validation-artifact checksums."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess

from beamfem.validation import build_dependency_audit


ROOT = Path(__file__).resolve().parents[1]


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _locked_names(lock: Path) -> list[str]:
    names = []
    for line in lock.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^([A-Za-z0-9_.-]+)==", line.strip())
        if match:
            names.append(match.group(1))
    if not names:
        raise ValueError("dependency lock contains no pinned packages")
    return names


def collect_inventory(lock: Path, artifacts: list[Path]) -> dict:
    audit = build_dependency_audit(packages=_locked_names(lock), artifacts=artifacts)
    return {
        "evidence_schema_version": "1.0",
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "git_commit": _git_commit(),
        },
        "lock_file": str(lock.relative_to(ROOT)),
        "lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
        "inventory": asdict(audit),
        "passed": all(item.version != "not-installed" for item in audit.dependencies),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=ROOT / "requirements-release.lock")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "validation" / "dependency_evidence.json"
    )
    args = parser.parse_args()
    artifacts = sorted(
        path for path in (ROOT / "validation").glob("*_evidence.json")
        if path.name not in {"dependency_evidence.json", "release_gate_evidence.json"}
    )
    evidence = collect_inventory(args.lock.resolve(), artifacts)
    args.output.write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    if not evidence["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
