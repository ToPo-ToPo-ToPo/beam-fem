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


def _optimization(data: Mapping[str, Any]) -> Mapping[str, Any]:
    value = data.get("optimization", data)
    return value if isinstance(value, Mapping) else data


def _utilization_svg(constraints: Any) -> str:
    if not isinstance(constraints, list):
        return "<p>No utilization records.</p>"
    records = [item for item in constraints if isinstance(item, Mapping)
               and isinstance(item.get("utilization"), (int, float))]
    if not records:
        return "<p>No utilization records.</p>"
    records = sorted(records, key=lambda item: float(item["utilization"]), reverse=True)[:12]
    width, row_height = 760, 30
    parts = [f'<svg role="img" aria-label="constraint utilization chart" viewBox="0 0 {width} {35 + row_height * len(records)}">']
    for index, item in enumerate(records):
        y = 20 + index * row_height
        utilization = max(0.0, float(item["utilization"]))
        bar = min(500.0, 500.0 * utilization)
        color = "#b42318" if utilization > 1.0 else "#19724c"
        label = escape(str(item.get("constraint_id") or item.get("kind") or index))
        parts.append(f'<text x="0" y="{y + 14}" font-size="12">{label}</text>')
        parts.append(f'<rect x="190" y="{y}" width="{bar:.2f}" height="18" fill="{color}"/>')
        parts.append(f'<text x="700" y="{y + 14}" font-size="12">{utilization:.3g}</text>')
    parts.append('</svg>')
    return "".join(parts)


def _history_svg(history: Any) -> str:
    if not isinstance(history, list) or not history:
        return "<p>No optimization history.</p>"
    values = [float(item["objective"]) for item in history
              if isinstance(item, Mapping) and isinstance(item.get("objective"), (int, float))]
    if not values:
        return "<p>No optimization history.</p>"
    width, height = 760, 220
    low, high = min(values), max(values)
    span = high - low or 1.0
    points = []
    for index, value in enumerate(values):
        x = 30 + index * 700 / max(1, len(values) - 1)
        y = 20 + (high - value) * 160 / span
        points.append(f"{x:.2f},{y:.2f}")
    return (f'<svg role="img" aria-label="optimization objective history" viewBox="0 0 {width} {height}">'
            '<line x1="30" y1="180" x2="730" y2="180" stroke="#667"/>'
            f'<polyline points="{" ".join(points)}" fill="none" stroke="#1769aa" stroke-width="3"/>'
            f'<text x="30" y="205" font-size="12">iterations: {len(values)}</text>'
            f'<text x="520" y="205" font-size="12">objective: {values[-1]:.6g}</text></svg>')


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
    optimization = _optimization(summary)
    sections.append("<h2>Constraint utilization</h2>" +
                    _utilization_svg(optimization.get("constraints")))
    sections.append("<h2>Optimization history</h2>" +
                    _history_svg(optimization.get("history")))
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


def render_comparison_report(
    baseline: Any, candidate: Any, *, title: str = "beamfem run comparison"
) -> str:
    """Render a dedicated, dependency-free comparison of two stored runs."""

    left_raw, right_raw = to_serializable(baseline), to_serializable(candidate)
    left = left_raw.get("result", left_raw) if isinstance(left_raw, Mapping) else {"value": left_raw}
    right = right_raw.get("result", right_raw) if isinstance(right_raw, Mapping) else {"value": right_raw}
    left_opt, right_opt = _optimization(left), _optimization(right)
    rows = []
    for field in ("backend", "status", "feasible", "objective", "runtime", "evaluations", "iterations"):
        a, b = left_opt.get(field), right_opt.get(field)
        delta = ""
        if isinstance(a, (int, float)) and not isinstance(a, bool) and isinstance(b, (int, float)) and not isinstance(b, bool):
            delta = f"{float(b) - float(a):.6g}"
        rows.append("<tr><th>" + escape(field) + "</th><td><code>" +
                    escape(json.dumps(a, ensure_ascii=False)) + "</code></td><td><code>" +
                    escape(json.dumps(b, ensure_ascii=False)) + "</code></td><td>" + escape(delta) + "</td></tr>")
    return """<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{title}</title>
<style>body{{font:15px system-ui;max-width:1100px;margin:2rem auto}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccd;padding:.5rem}}.gate{{border:3px solid #a11;padding:1rem;background:#fff2f2}}</style></head>
<body><h1>{title}</h1><div class="gate">{disclaimer}</div><table><thead><tr><th>Field</th><th>Baseline</th><th>Candidate</th><th>Delta</th></tr></thead><tbody>{rows}</tbody></table>
<h2>Baseline history</h2>{left_history}<h2>Candidate history</h2>{right_history}
<p><strong>External professional review: REQUIRED</strong>.</p></body></html>""".format(
        title=escape(title), disclaimer=escape(DISCLAIMER), rows="".join(rows),
        left_history=_history_svg(left_opt.get("history")), right_history=_history_svg(right_opt.get("history")),
    )


def write_comparison_report(baseline: Any, candidate: Any, path: str | Path, **kwargs: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(render_comparison_report(baseline, candidate, **kwargs), encoding="utf-8")
    temporary.replace(destination)
    return destination
