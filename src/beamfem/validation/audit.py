"""Reproducibility metadata independent of a particular solver backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping, Sequence


def _package_version() -> str:
    # Prefer the imported source package. Editable development trees may retain
    # stale egg-info from an older build even though the running code is newer.
    try:
        from beamfem import __version__

        return __version__
    except (ImportError, AttributeError):
        try:
            return version("beamfem")
        except PackageNotFoundError:
            return "unknown"


def _git_commit(repository: str | Path | None) -> tuple[str | None, bool | None]:
    if repository is None:
        return None, None
    cwd = Path(repository).resolve()
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=cwd,
                check=True,
                capture_output=True,
                text=True,
                timeout=3,
            ).stdout.strip()
        )
        return commit, dirty
    except (OSError, subprocess.SubprocessError):
        return None, None


@dataclass(frozen=True)
class AuditMetadata:
    """Information required to reproduce and review an optimization run."""

    created_at_utc: str
    beamfem_version: str
    python_version: str
    platform: str
    git_commit: str | None
    git_dirty: bool | None
    seed: int | None
    solver: str
    solver_settings: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()


def build_audit_metadata(
    *,
    solver: str,
    seed: int | None = None,
    solver_settings: Mapping[str, Any] | None = None,
    warnings: Sequence[str] = (),
    repository: str | Path | None = None,
    created_at: datetime | None = None,
) -> AuditMetadata:
    """Capture deterministic solver settings and environment provenance."""

    if not solver.strip():
        raise ValueError("solver must be a non-empty name")
    timestamp = created_at or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    commit, dirty = _git_commit(repository)
    return AuditMetadata(
        created_at_utc=timestamp.astimezone(timezone.utc).isoformat(),
        beamfem_version=_package_version(),
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        git_commit=commit,
        git_dirty=dirty,
        seed=seed,
        solver=solver,
        solver_settings=dict(solver_settings or {}),
        warnings=tuple(str(warning) for warning in warnings),
    )
