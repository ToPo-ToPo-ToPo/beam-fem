"""Integrity-checked release archives for retention and rollback drills."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Iterable
import zipfile


ARCHIVE_MANIFEST = "release-manifest.json"


def _digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def create_release_archive(
    files: Iterable[str | Path],
    destination: str | Path,
    *,
    retention_days: int = 365,
) -> Path:
    """Create an atomic ZIP with per-file hashes and an expiry date."""

    if retention_days < 1:
        raise ValueError("retention_days must be positive")
    sources = [Path(item).resolve() for item in files]
    if not sources or any(not source.is_file() for source in sources):
        raise ValueError("release archive requires one or more existing files")
    names = [source.name for source in sources]
    if len(set(names)) != len(names):
        raise ValueError("release archive file basenames must be unique")

    now = datetime.now(timezone.utc)
    records = []
    payloads = {}
    for source in sources:
        data = source.read_bytes()
        payloads[source.name] = data
        records.append(
            {"path": source.name, "sha256": _digest_bytes(data), "size_bytes": len(data)}
        )
    manifest = {
        "format": "beamfem-release-archive/1.0",
        "created_at_utc": now.isoformat(),
        "retain_until_utc": (now + timedelta(days=retention_days)).isoformat(),
        "retention_days": retention_days,
        "artifacts": records,
    }
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in payloads.items():
            archive.writestr(name, data)
        archive.writestr(
            ARCHIVE_MANIFEST,
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
    temporary.replace(target)
    verify_release_archive(target)
    return target


def verify_release_archive(path: str | Path) -> dict:
    """Verify membership, sizes, and SHA-256 digests and return the manifest."""

    with zipfile.ZipFile(path, "r") as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or ARCHIVE_MANIFEST not in names:
            raise ValueError("invalid or duplicate release archive members")
        try:
            manifest = json.loads(archive.read(ARCHIVE_MANIFEST))
        except (json.JSONDecodeError, KeyError) as exc:
            raise ValueError("invalid release archive manifest") from exc
        if manifest.get("format") != "beamfem-release-archive/1.0":
            raise ValueError("unsupported release archive format")
        expected = {ARCHIVE_MANIFEST}
        for record in manifest.get("artifacts", []):
            name = str(record.get("path", ""))
            if not name or Path(name).name != name:
                raise ValueError("unsafe release archive path")
            expected.add(name)
            data = archive.read(name)
            if len(data) != int(record["size_bytes"]):
                raise ValueError(f"release archive size mismatch: {name}")
            if _digest_bytes(data) != record["sha256"]:
                raise ValueError(f"release archive checksum mismatch: {name}")
        if set(names) != expected:
            raise ValueError("release archive has undeclared members")
        return manifest


def restore_release_archive(path: str | Path, destination: str | Path) -> tuple[Path, ...]:
    """Restore a verified archive without overwriting an existing file."""

    manifest = verify_release_archive(path)
    target = Path(destination)
    target.mkdir(parents=True, exist_ok=True)
    restored = []
    for record in manifest["artifacts"]:
        output = target / record["path"]
        if output.exists():
            raise FileExistsError(f"rollback target already exists: {output}")
    with zipfile.ZipFile(path, "r") as archive:
        for record in manifest["artifacts"]:
            output = target / record["path"]
            with tempfile.NamedTemporaryFile(dir=target, delete=False) as temporary:
                temporary.write(archive.read(record["path"]))
                temp_path = Path(temporary.name)
            temp_path.replace(output)
            restored.append(output)
    return tuple(restored)
