"""Integrity-checked local persistence for expensive FEM evaluations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pickle
from typing import Any


MAGIC = b"BEAMFEM_EVALUATION_CACHE_V1\n"


def problem_context_checksum(problem: Any) -> str:
    """Hash the serialized problem definition, excluding its process cache."""
    payload = pickle.dumps(problem, protocol=5)
    return hashlib.sha256(payload).hexdigest()


class PersistentEvaluationCache:
    """Atomic cache file bound to exactly one structural problem definition.

    The payload uses Python pickle to preserve sparse matrices and rich result
    objects. It must therefore only be used for cache files created locally by
    beamfem; the checksum detects corruption but is not an authenticity proof.
    """

    def __init__(self, path: str | Path, context_checksum: str):
        self.path = Path(path)
        self.context_checksum = str(context_checksum)

    def load(self) -> dict[Any, Any]:
        if not self.path.exists():
            return {}
        raw = self.path.read_bytes()
        if not raw.startswith(MAGIC):
            raise ValueError("invalid persistent FEM cache header")
        remainder = raw[len(MAGIC):]
        try:
            header_raw, payload = remainder.split(b"\n", 1)
            header = json.loads(header_raw)
        except (ValueError, json.JSONDecodeError) as exc:
            raise ValueError("invalid persistent FEM cache metadata") from exc
        if header.get("context_sha256") != self.context_checksum:
            raise ValueError("persistent FEM cache problem context mismatch")
        if hashlib.sha256(payload).hexdigest() != header.get("payload_sha256"):
            raise ValueError("persistent FEM cache checksum mismatch")
        entries = pickle.loads(payload)
        if not isinstance(entries, dict):
            raise ValueError("persistent FEM cache payload must be a dictionary")
        return entries

    def save(self, entries: dict[Any, Any]) -> None:
        payload = pickle.dumps(entries, protocol=5)
        header = json.dumps({
            "schema_version": 1,
            "context_sha256": self.context_checksum,
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
            "entries": len(entries),
        }, sort_keys=True, separators=(",", ":")).encode("utf-8")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_bytes(MAGIC + header + b"\n" + payload)
        temporary.replace(self.path)
