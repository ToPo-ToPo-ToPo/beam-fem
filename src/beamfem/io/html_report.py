"""Dependency-free, self-contained preliminary design HTML report."""

from __future__ import annotations

from html import escape
import json
from pathlib import Path
from typing import Any, Mapping

from .result_writer import to_serializable


DISCLAIMER = (
    "検証・予備設計用出力です。法令・AIJ・JIS・ANSI/AISCその他の正式適合、"
    "設計承認または安全性認証を示しません。責任ある構造設計者による外部レビューが必須です。"
)


def _rows(mapping: Mapping[str, Any]) -> str:
    return "".join(
        "<tr><th>" + escape(str(key)) + "</th><td><code>" +
        escape(json.dumps(value, ensure_ascii=False, sort_keys=True)) + "</code></td></tr>"
        for key, value in mapping.items()
    )


def render_design_report(
    result: Any,
    *,
    title: str = "beamfem preliminary design report",
    audit: Any | None = None,
    code_checks: Any | None = None,
    manifest: Any | None = None,
) -> str:
    data = to_serializable(result)
    summary = data if isinstance(data, Mapping) else {"result": data}
    sections = [f"<h2>Result</h2><table>{_rows(summary)}</table>"]
    for heading, value in (
        ("Code-check trace", code_checks),
        ("Run manifest", manifest),
        ("Audit metadata", audit),
    ):
        if value is not None:
            serial = to_serializable(value)
            content = serial if isinstance(serial, Mapping) else {"value": serial}
            sections.append(f"<h2>{escape(heading)}</h2><table>{_rows(content)}</table>")
    return """<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>
body{{font:15px system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#18212b}}
.gate{{border:3px solid #a11;background:#fff2f2;padding:1rem;font-weight:700}}
table{{border-collapse:collapse;width:100%;table-layout:fixed}}th,td{{border:1px solid #ccd;padding:.5rem;text-align:left;vertical-align:top;word-break:break-word}}th{{width:25%;background:#f3f5f7}}code{{white-space:pre-wrap}}
</style></head><body><h1>{title}</h1><div class="gate">{disclaimer}</div>{sections}
<h2>Review gate</h2><p><strong>External professional review: REQUIRED</strong>. Automated pass results cannot close this gate.</p>
</body></html>""".format(
        title=escape(title), disclaimer=escape(DISCLAIMER), sections="".join(sections)
    )


def write_design_report(result: Any, path: str | Path, **kwargs: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(render_design_report(result, **kwargs), encoding="utf-8")
    temporary.replace(destination)
    return destination
