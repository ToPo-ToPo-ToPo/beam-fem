"""Dependency-free preliminary PDF summary writer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .html_report import DISCLAIMER
from .result_writer import to_serializable


def _ascii(value: Any) -> str:
    return (str(value).encode("ascii", "replace").decode("ascii")
            .replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)"))


def _pdf_bytes(lines: list[str]) -> bytes:
    pages = [lines[index:index + 44] for index in range(0, len(lines), 44)] or [[]]
    page_ids = [4 + 2 * index for index in range(len(pages))]
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        ("<< /Type /Pages /Kids [" + " ".join(f"{item} 0 R" for item in page_ids) +
         f"] /Count {len(page_ids)} >>").encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for index, page in enumerate(pages):
        page_id = page_ids[index]
        content_id = page_id + 1
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>".encode()
        )
        commands = ["BT /F1 10 Tf 45 755 Td"]
        for line_index, line in enumerate(page):
            if line_index:
                commands.append("0 -16 Td")
            commands.append(f"({_ascii(line)}) Tj")
        commands.append("ET")
        stream = "\n".join(commands).encode("ascii")
        objects.append(f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream")
    result = bytearray(b"%PDF-1.4\n%beamfem\n")
    offsets = [0]
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(result))
        result.extend(f"{number} 0 obj\n".encode()); result.extend(obj); result.extend(b"\nendobj\n")
    xref = len(result)
    result.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    for offset in offsets[1:]:
        result.extend(f"{offset:010d} 00000 n \n".encode())
    result.extend(f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    return bytes(result)


def write_design_pdf(
    result: Any, path: str | Path, *,
    title: str = "beamfem preliminary design report", audit: Any | None = None,
) -> Path:
    serial = to_serializable(result)
    summary = serial if isinstance(serial, Mapping) else {"result": serial}
    optimization = summary.get("optimization", summary)
    if not isinstance(optimization, Mapping):
        optimization = summary
    lines = [title, "", "PRELIMINARY / EXTERNAL PROFESSIONAL REVIEW REQUIRED",
             _ascii(DISCLAIMER), ""]
    for key in ("problem", "backend", "status", "objective", "feasible", "runtime",
                "evaluations", "iterations"):
        value = optimization.get(key, summary.get(key))
        if value is not None:
            lines.append(f"{key}: {json.dumps(value, ensure_ascii=True)}")
    constraints = optimization.get("constraints", [])
    if isinstance(constraints, list):
        lines.extend(["", "Governing constraint utilizations:"])
        ranked = sorted(
            (item for item in constraints if isinstance(item, Mapping)
             and isinstance(item.get("utilization"), (int, float))),
            key=lambda item: float(item["utilization"]), reverse=True,
        )[:12]
        for item in ranked:
            lines.append(
                f"- {item.get('constraint_id', item.get('kind', '?'))}: "
                f"{float(item['utilization']):.6g}"
            )
    if audit is not None:
        lines.extend(["", "Audit:", json.dumps(
            to_serializable(audit), ensure_ascii=True, sort_keys=True
        )[:2000]])
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_bytes(_pdf_bytes(lines)); temporary.replace(destination)
    return destination
