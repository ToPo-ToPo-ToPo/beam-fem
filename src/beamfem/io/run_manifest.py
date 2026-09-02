"""Atomic, resumable run manifests for CLI and API workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
import hashlib
import hmac
import json
from pathlib import Path
import uuid
from typing import Any, Mapping

from .result_writer import to_serializable


MANIFEST_SCHEMA_VERSION = "1.0"


class RunStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


def canonical_checksum(value: Any) -> str:
    payload = json.dumps(
        to_serializable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    created_at_utc: str
    updated_at_utc: str
    problem_checksum: str
    solver: str
    solver_settings: dict[str, Any]
    seed: int | None
    status: RunStatus = RunStatus.CREATED
    completed_steps: tuple[str, ...] = ()
    checkpoint: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    manifest_schema_version: str = MANIFEST_SCHEMA_VERSION
    integrity_checksum: str = ""

    def advance(
        self,
        step: str,
        *,
        checkpoint: Mapping[str, Any] | None = None,
        status: RunStatus = RunStatus.RUNNING,
    ) -> "RunManifest":
        steps = self.completed_steps
        if step not in steps:
            steps += (step,)
        return replace(
            self,
            updated_at_utc=datetime.now(timezone.utc).isoformat(),
            completed_steps=steps,
            checkpoint=dict(checkpoint or self.checkpoint),
            status=status,
        )


def create_run_manifest(
    problem: Any,
    *,
    solver: str,
    solver_settings: Mapping[str, Any] | None = None,
    seed: int | None = None,
    run_id: str | None = None,
) -> RunManifest:
    now = datetime.now(timezone.utc).isoformat()
    return RunManifest(
        run_id=run_id or str(uuid.uuid4()),
        created_at_utc=now,
        updated_at_utc=now,
        problem_checksum=canonical_checksum(problem),
        solver=solver,
        solver_settings=dict(solver_settings or {}),
        seed=seed,
    )


def write_run_manifest(manifest: RunManifest, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = to_serializable(asdict(manifest))
    payload.pop("integrity_checksum", None)
    payload["integrity_checksum"] = canonical_checksum(payload)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def load_run_manifest(path: str | Path) -> RunManifest:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("manifest_schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported run manifest schema")
    supplied_checksum = str(data.pop("integrity_checksum", ""))
    actual_checksum = canonical_checksum(data)
    if not hmac.compare_digest(supplied_checksum, actual_checksum):
        raise ValueError("run manifest integrity checksum mismatch")
    try:
        data["status"] = RunStatus(data["status"])
        data["completed_steps"] = tuple(data.get("completed_steps", ()))
        return RunManifest(**data, integrity_checksum=supplied_checksum)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid run manifest: {exc}") from exc


def verify_resume_compatibility(
    manifest: RunManifest,
    problem: Any,
    *,
    solver: str,
    solver_settings: Mapping[str, Any],
    seed: int | None,
) -> None:
    mismatches = []
    if manifest.problem_checksum != canonical_checksum(problem):
        mismatches.append("problem checksum")
    if manifest.solver != solver:
        mismatches.append("solver")
    if manifest.solver_settings != dict(solver_settings):
        mismatches.append("solver settings")
    if manifest.seed != seed:
        mismatches.append("seed")
    if mismatches:
        raise ValueError("cannot resume: changed " + ", ".join(mismatches))
