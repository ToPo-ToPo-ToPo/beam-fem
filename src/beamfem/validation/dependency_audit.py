"""Checksums and a compact SBOM-like dependency inventory."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
from typing import Iterable, Sequence


@dataclass(frozen=True)
class DependencyRecord:
    name: str
    version: str
    license: str | None
    requires: tuple[str, ...]


@dataclass(frozen=True)
class ArtifactChecksum:
    path: str
    algorithm: str
    digest: str
    size_bytes: int


@dataclass(frozen=True)
class DependencyAudit:
    format: str
    generated_at_utc: str
    python: str
    dependencies: tuple[DependencyRecord, ...]
    artifacts: tuple[ArtifactChecksum, ...]


def sha256_file(path: str | Path) -> ArtifactChecksum:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return ArtifactChecksum(str(source), "sha256", digest.hexdigest(), source.stat().st_size)


def _dependency(name: str) -> DependencyRecord:
    distribution = metadata.distribution(name)
    license_value = distribution.metadata.get("License-Expression") or distribution.metadata.get("License")
    return DependencyRecord(
        name=distribution.metadata.get("Name", name),
        version=distribution.version,
        license=license_value or None,
        requires=tuple(sorted(distribution.requires or ())),
    )


def build_dependency_audit(
    *,
    packages: Sequence[str] = ("beamfem", "numpy", "scipy"),
    artifacts: Iterable[str | Path] = (),
) -> DependencyAudit:
    records = []
    for name in packages:
        try:
            records.append(_dependency(name))
        except metadata.PackageNotFoundError:
            records.append(DependencyRecord(name, "not-installed", None, ()))
    return DependencyAudit(
        format="beamfem-sbom-lite/1.0",
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        python=platform.python_version(),
        dependencies=tuple(records),
        artifacts=tuple(sha256_file(path) for path in artifacts),
    )


def write_dependency_audit(audit: DependencyAudit, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(asdict(audit), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination
