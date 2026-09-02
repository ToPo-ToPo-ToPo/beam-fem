"""Resolve versioned material and section CSV references at the file boundary."""

from __future__ import annotations

from copy import deepcopy
import csv
import hashlib
from pathlib import Path
import re
from typing import Any, Mapping


_NUMERIC_FIELDS = frozenset({
    "E", "density", "nu", "tension_allowable", "compression_allowable",
    "cost_per_kg", "carbon_per_kg", "area", "I", "J", "ky", "kz", "cy",
    "cz", "slenderness",
})


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _reference(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object")
    unknown = set(value) - {"path", "version", "sha256", "row"}
    if unknown:
        raise ValueError(f"{path} contains unsupported fields: {sorted(unknown)}")
    if not isinstance(value.get("path"), str) or not value["path"]:
        raise ValueError(f"{path}.path must be a non-empty string")
    if not isinstance(value.get("version"), str) or not value["version"]:
        raise ValueError(f"{path}.version must be a non-empty string")
    digest = value.get("sha256")
    if digest is not None and (
        not isinstance(digest, str) or re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None
    ):
        raise ValueError(f"{path}.sha256 must be a 64-character hexadecimal digest")
    return value


def _rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() != ".csv":
        raise ValueError(f"external catalog must be CSV: {path}")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"external catalog has no header: {path}")
        records = []
        for row_number, row in enumerate(reader, start=2):
            cleaned: dict[str, Any] = {}
            for key, raw in row.items():
                if key is None or raw is None or not raw.strip():
                    continue
                value: Any = raw.strip()
                if key in _NUMERIC_FIELDS:
                    try:
                        value = float(value)
                    except ValueError as exc:
                        raise ValueError(
                            f"{path}:{row_number} field {key!r} must be numeric"
                        ) from exc
                cleaned[key] = value
            records.append(cleaned)
    if not records:
        raise ValueError(f"external catalog has no data rows: {path}")
    return records


def resolve_external_catalogs(document: Mapping[str, Any], source: str | Path) -> dict[str, Any]:
    """Merge ``external_catalogs`` CSV references without overriding inline data.

    Material CSV rows require ``id,E,density``. Section CSV rows require
    ``id,area``. Each reference requires an explicit human-managed ``version``;
    an optional SHA-256 pins exact file contents.
    """

    data = deepcopy(dict(document))
    references = data.get("external_catalogs")
    if references is None:
        return data
    if not isinstance(references, Mapping):
        raise ValueError("external_catalogs must be an object")
    unknown = set(references) - {"materials", "sections"}
    if unknown:
        raise ValueError(f"external_catalogs contains unsupported groups: {sorted(unknown)}")
    base = Path(source).resolve().parent
    materials = data.setdefault("materials", {})
    sections = data.setdefault("section_catalogs", {})
    if not isinstance(materials, dict) or not isinstance(sections, dict):
        raise ValueError("materials and section_catalogs must be objects")
    provenance: dict[str, dict[str, Any]] = {"materials": {}, "sections": {}}

    for group, destination, required in (
        ("materials", materials, {"id", "E", "density"}),
        ("sections", sections, {"id", "area"}),
    ):
        group_refs = references.get(group, {})
        if not isinstance(group_refs, Mapping):
            raise ValueError(f"external_catalogs.{group} must be an object")
        for name, raw_reference in group_refs.items():
            ref = _reference(raw_reference, f"external_catalogs.{group}.{name}")
            catalog_path = (base / ref["path"]).resolve()
            if not catalog_path.is_file():
                raise ValueError(f"external catalog does not exist: {ref['path']}")
            digest = _digest(catalog_path)
            expected = ref.get("sha256")
            if expected is not None and digest.lower() != expected.lower():
                raise ValueError(f"external catalog checksum mismatch: {ref['path']}")
            records = _rows(catalog_path)
            if group == "materials":
                row_id = str(ref.get("row", name))
                selected = [record for record in records if str(record.get("id")) == row_id]
                if len(selected) != 1:
                    raise ValueError(
                        f"external material {name!r} requires exactly one row id {row_id!r}"
                    )
                value = dict(selected[0]); value.pop("id", None)
                if not required <= set(selected[0]):
                    raise ValueError(f"external material {name!r} requires columns {sorted(required)}")
            else:
                if any(not required <= set(record) for record in records):
                    raise ValueError(f"external section catalog {name!r} requires columns {sorted(required)}")
                value = records
            if name in destination:
                raise ValueError(f"external catalog {name!r} duplicates an inline definition")
            destination[name] = value
            provenance[group][str(name)] = {
                "path": str(ref["path"]), "version": str(ref["version"]), "sha256": digest,
            }
    data["catalog_sources"] = provenance
    return data
