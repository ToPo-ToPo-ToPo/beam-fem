"""Stable JSON and CSV writers for optimization results."""

from __future__ import annotations

import csv
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from enum import Enum
import json
import math
from pathlib import Path
from typing import Any, Mapping


def to_serializable(value: Any) -> Any:
    """Convert dataclasses, mappings, numpy values, and Protocol-like results."""

    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("result contains NaN or infinity")
        return value
    if isinstance(value, Enum):
        return to_serializable(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    # Domain results may intentionally exclude heavy/non-serializable FEM
    # matrices from their public representation.  Respect that contract before
    # recursively expanding the dataclass with asdict().
    if hasattr(value, "as_dict") and callable(value.as_dict):
        return to_serializable(value.as_dict())
    if is_dataclass(value) and not isinstance(value, type):
        return to_serializable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_serializable(item) for item in value]
    if hasattr(value, "tolist"):
        return to_serializable(value.tolist())
    if hasattr(value, "item"):
        return to_serializable(value.item())
    if hasattr(value, "__dict__"):
        return to_serializable(vars(value))
    raise TypeError(f"unsupported result value: {type(value).__name__}")


def _result_document(result: Any, audit: Any | None) -> dict[str, Any]:
    document: dict[str, Any] = {
        "result_schema_version": "1.0",
        "result": to_serializable(result),
    }
    if audit is not None:
        document["audit"] = to_serializable(audit)
    return document


def write_result_json(
    result: Any,
    path: str | Path,
    *,
    audit: Any | None = None,
    indent: int = 2,
) -> Path:
    """Write a complete, versioned result document atomically."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(
            _result_document(result, audit),
            stream,
            ensure_ascii=False,
            indent=indent,
            sort_keys=True,
        )
        stream.write("\n")
    temporary.replace(destination)
    return destination


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    serial = to_serializable(value)
    if isinstance(serial, Mapping):
        flat: dict[str, Any] = {}
        for key, item in serial.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            flat.update(_flatten(item, child))
        return flat
    if isinstance(serial, list):
        return {prefix: json.dumps(serial, ensure_ascii=False, sort_keys=True)}
    return {prefix: serial}


def write_result_csv(result: Any, path: str | Path) -> Path:
    """Write one flattened summary row suitable for benchmark aggregation."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    row = _flatten(result)
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    return destination
